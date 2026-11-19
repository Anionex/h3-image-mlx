"""Private stage entry point. Each child releases its weights on exit."""
import base64
import ctypes
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from .models import dit_file, model_paths, manifest_for, text_file
from .runtime import tensorops_available_on_macos


def select_backend(requested, device, precision="int8"):
    manifest_for(precision)
    if precision == "bf16":
        if requested == "w8a8":
            raise ValueError("W8A8 requires INT8 weights; use --backend auto or portable with BF16")
        return "portable"
    native = bool(re.search(r"\bApple M[56]\b", device)) and tensorops_available_on_macos()
    if requested == "auto":
        return "w8a8" if native else "portable"
    if requested == "w8a8" and not native:
        raise ValueError(f"W8A8 requires an M5/M6 GPU and macOS 26.2 or later; found {device}")
    return requested


def stage_request():
    return json.load(sys.stdin)


def emit(kind, **data):
    print(f"{kind} " + json.dumps(data), file=sys.stderr, flush=True)


def process_usage():
    if sys.platform != "darwin":
        return dict(peak_bytes=0, disk_read_bytes=0)
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    # rusage_info_v4 has a 16-byte UUID before the uint64 fields. Fields 16
    # and 28 are ri_diskio_bytesread and ri_lifetime_max_phys_footprint.
    data = (ctypes.c_uint64 * 37)()
    if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(data)) != 0:
        return dict(peak_bytes=0, disk_read_bytes=0)
    return dict(peak_bytes=data[30], disk_read_bytes=data[18])


def main(stage, work, request=None):
    began = time.perf_counter()
    import mlx.core as mx
    from . import core
    if request is None:
        request = json.loads((work / "request.json").read_text())
    info = mx.device_info()
    precision = request.get("precision", "int8")
    _, _, vae, tokenizer, vae_config = model_paths(precision)
    backend = select_backend(request["backend"], info["device_name"], precision)
    core.backend.set(backend)
    core.uncached_weights.set(request.get("weight_cache") == "uncached")
    from ._portable import _legacy_metal
    core.legacy_int8_reads.set(precision == "int8" and backend == "portable" and _legacy_metal())
    mx.set_cache_limit(96 << 20)
    # A 96 MB buffer cache makes MLX allocate fresh Metal buffers for most ops at image sizes (2x slower DiT at
    # 2048^2). With H3_GPU_BUDGET_MB set, the DiT stage may cache up to ~0.5 MB per token, within the budget left
    # after the stage's own measured peak.
    models = Path(request["models"])
    refs = [io.BytesIO(base64.b64decode(ref["data"])) if isinstance(ref, dict) else ref
            for ref in request["references"]]
    metrics = {
        "weight_read_bytes": 0, "prefetch_peak_bytes": 0,
        "weight_io": dict(calls=0, logical_bytes=0, aligned_bytes=0),
        "weight_cache": "uncached" if core.uncached_weights.get() or core.flag("H3_NOCACHE") else "system",
    }
    core.stage_weight_metrics.set(metrics)

    def progress(label, step=0, total=0):
        emit("PROGRESS", stage=label, step=step, total=total)
    embedding, latents, result = (str(work / (key + ".safetensors")) for key in ("embedding", "references", "latent"))
    mode = request.get("dit_mode")
    auto_resident = mode == "auto" and backend == "w8a8" and info["memory_size"] >= 32 * 1024**3
    # The exact DiT sequence length becomes available after encoding.
    m5 = backend == "w8a8" and bool(re.search(r"\bApple M5\b", info["device_name"]))
    likely_resident = auto_resident and not refs and request["width"] * request["height"] // 1024 < 8192
    core.m5_resident.set(m5 and (mode == "resident" or likely_resident))
    overlap = request.get("overlap_streaming")
    if overlap is None and request.get("weight_cache") == "uncached" and not any(
        name in os.environ for name in ("H3_PREFETCH", "H3_ASYNC")
    ):
        from .streaming import StreamPolicy

        overlap = StreamPolicy().use_overlap()
    core.stream_overlap.set(overlap if mode == "stream" else None)
    if stage == "text":
        progress("Encoding prompt")
        from .text import encode
        # H3_TEXT / H3_TOKENIZER / H3_VAE / H3_DIT: use another checkpoint (e.g. P-H3-Image) instead of the manifest's.
        x, tags, calls = encode(os.environ.get("H3_TEXT") or text_file(models, precision), os.environ.get("H3_TOKENIZER") or models / tokenizer, request["prompt"], refs)
        mx.save_safetensors(embedding, {"embedding": x, "tags": tags})
        metrics["w8a8_calls"] = calls
    elif stage == "references":
        progress("Encoding references")
        from .ref_vae import encode_reference
        stats = json.loads((models / vae_config).read_text())
        zs = {}
        with core.Weights(models / vae, True) as weights:
            for i, ref in enumerate(refs):
                zs[str(i)] = encode_reference(weights, ref, stats)
                mx.eval(zs[str(i)])
        mx.save_safetensors(latents, zs)
    elif stage == "dit":
        progress("Loading model")
        from .dit import denoise
        inputs = mx.load(embedding)
        images = mx.load(latents) if refs else {}
        zs = [images[str(i)] for i in range(len(refs))]
        width, height = request["width"], request["height"]
        rows = inputs["embedding"].shape[0] + width * height // 1024 + sum(z.shape[1] * z.shape[2] // 4 for z in zs)
        resident = mode == "resident" or (auto_resident and rows <= 8192)
        dit_path = os.environ.get("H3_DIT") or dit_file(models, precision)
        zero_copy = False
        if os.environ.get("H3_ZERO_COPY") == "1":
            with core.Weights(dit_path) as probe:
                zero_copy = probe.mapped is not None
        # Zero copy keeps the entry's mode: resident holds every block's mapped buffer (MLX keeps its buffers resident,
        # so ~20 GB is wired as with a copy, but loading skips the copy); stream drops each block after use.
        core.m5_resident.set(m5 and resident)
        budget = int(os.environ.get("H3_GPU_BUDGET_MB") or 0)
        if budget:
            # Measured DiT peaks without cache (MB): resident ~21300-21800 at 1024^2-2048^2; stream ~2700 / 3100 /
            # 6200 at 1046 / 4118 / 16406 tokens. Plus 500 MB margin.
            own_peak = 22000 if resident and not zero_copy else 3000 + rows * 23 // 100
            cache = min(rows // 2, budget - own_peak)    # MB
            if cache > 96:
                mx.set_cache_limit(cache << 20)
        core.stream_overlap.set(overlap if not resident else None)
        # H3_DIT=<file>: use another DiT checkpoint with the same layout (e.g. one with a LoRA merged in).
        def schedule(sigmas):
            metrics["sample_sigmas"] = sigmas

        def step(done, _elapsed):
            progress("Denoising", done, len(metrics["sample_sigmas"]) - 1)

        z, calls = denoise(
            dit_path,
            inputs["embedding"], width, height, request["seed"], resident, zs,
            inputs["tags"], progress=step, precision=precision, on_schedule=schedule,
        )
        metrics.update(dit_mode="resident" if resident else "stream", w8a8_calls=calls)
        mx.save_safetensors(result, {"latent": z})
    elif stage == "vae":
        progress("Decoding image")
        from .vae import decode
        stats = json.loads((models / vae_config).read_text())
        decode(os.environ.get("H3_VAE") or models / vae, mx.load(result)["latent"], stats).save(work / "image.png")
    else:
        raise ValueError(f"Unknown stage: {stage}")
    mx.synchronize()
    emit("METRICS", backend=backend, elapsed_s=time.perf_counter() - began,
         mlx_peak_bytes=mx.get_peak_memory(), **process_usage(), **metrics)


if __name__ == "__main__":
    work = Path(sys.argv[2])
    main(sys.argv[1], work, None if (work / "request.json").exists() else stage_request())
