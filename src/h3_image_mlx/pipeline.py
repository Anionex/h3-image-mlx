"""Text-to-image and ordered multi-reference editing in isolated MLX stages."""

import base64
import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

from PIL import Image

from ._subprocess import Cancelled, run_worker
from .api import GenerationResult
from .models import check_models, manifest_for, model_paths, text_file
from .runtime import _stage_command


def embedding_cache(directory, prompt, refs, backend, precision="int8"):
    """Opt-in (H3_EMBED_CACHE=dir): reuse the text-stage output for a repeated prompt + references."""
    root = os.environ.get("H3_EMBED_CACHE")
    if not root:
        return None
    encoder = Path(os.environ.get("H3_TEXT") or text_file(directory, precision)).resolve()
    stat = encoder.stat()
    tokenizer = Path(os.environ.get("H3_TOKENIZER") or directory / model_paths(precision)[3])
    key = [prompt, backend, precision, str(encoder), stat.st_size, stat.st_mtime_ns,
           hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
           [hashlib.sha256(p if isinstance(p, bytes) else p.read_bytes()).hexdigest() for p in refs]]
    name = hashlib.sha256(json.dumps(key).encode()).hexdigest()
    Path(root).expanduser().mkdir(parents=True, exist_ok=True)
    return Path(root).expanduser() / f"{name}.safetensors"


def generate_result(
    prompt: str,
    *,
    models: str | Path,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    references: Sequence[str | Path | bytes] = (),
    backend: str = "auto",
    dit_mode: str = "auto",
    precision: str = "int8",
    private: bool = False,
    overlap_streaming: bool | None = None,
    weight_cache: str = "system",
    on_progress=None,
    cancel=None,
) -> GenerationResult:
    if cancel is not None and cancel.is_set():
        raise Cancelled("Generation cancelled")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Provide a nonempty prompt")
    if not all(type(v) is int and 256 <= v <= 4096 and v % 32 == 0 for v in (width, height)):
        raise ValueError("Width and height must be multiples of 32 from 256 through 4096")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if backend not in ("auto", "portable", "w8a8") or dit_mode not in ("auto", "stream", "resident"):
        raise ValueError("Unknown compute backend or DiT mode")
    if weight_cache not in ("system", "uncached"):
        raise ValueError("weight_cache must be system or uncached")
    manifest_for(precision)
    if precision == "bf16" and backend == "w8a8":
        raise ValueError("W8A8 requires INT8 weights; use --backend auto or portable with BF16")
    if isinstance(references, (str, Path, bytes)) or len(references) > 16:
        raise ValueError("Supply an ordered list of up to 16 reference images")
    refs = []
    for ref in references:
        ref = ref if isinstance(ref, bytes) else Path(ref).expanduser().resolve()
        with Image.open(io.BytesIO(ref) if isinstance(ref, bytes) else ref) as im:
            if im.format not in ("PNG", "JPEG", "WEBP"):
                raise ValueError("Unsupported reference image format")
            if max(im.size) / min(im.size) * 1024**2 > 12_845_056:
                raise ValueError("Reference exceeds the vision encoder aspect ratio budget")
        refs.append(ref)
    directory = check_models(models, precision=precision)
    with tempfile.TemporaryDirectory(prefix="h3-image-mlx-") as tmp:
        work = Path(tmp)
        args = dict(prompt=prompt, models=str(directory), width=width, height=height, seed=seed,
                    backend=backend, dit_mode=dit_mode, precision=precision,
                    overlap_streaming=overlap_streaming, weight_cache=weight_cache)
        encoded = [dict(data=base64.b64encode(ref).decode()) if isinstance(ref, bytes) else str(ref)
                   for ref in refs]
        stages = ["text", *(["references"] if refs else []), "dit", "vae"]
        cached = None if private or weight_cache == "uncached" else embedding_cache(directory, prompt, refs, backend, precision)
        if cached and cached.is_file():
            shutil.copy(cached, work / "embedding.safetensors")
            stages.remove("text")
        metrics = []
        for stage in stages:
            # Later stages use the ordered latent tensors and need only the reference count.
            request = {**args, "references": encoded if stage in ("text", "references") else [None] * len(refs)}
            _, metric = run_worker(
                _stage_command(stage, tmp), input_bytes=json.dumps(request).encode(),
                cancel=cancel, on_progress=on_progress, label=f"H3 {stage}",
            )
            metrics.append({"stage": stage, **metric})
            if stage == "text" and cached:
                with tempfile.NamedTemporaryFile(dir=cached.parent) as temp:
                    shutil.copyfile(work / "embedding.safetensors", temp.name)
                    # Link the completed file atomically, tolerating another writer winning.
                    try:
                        os.link(temp.name, cached)
                    except FileExistsError:
                        pass
        if cancel is not None and cancel.is_set():
            raise Cancelled("Generation cancelled")
        with Image.open(work / "image.png") as image:
            return GenerationResult(image.convert("RGB"), dict(
                model="h3", seed=seed, stages=metrics,
                sample_sigmas=next(m.get("sample_sigmas") for m in metrics if m["stage"] == "dit"),
                backend=metrics[-1].get("backend", backend),
                weight_cache=metrics[-1].get("weight_cache", weight_cache),
                dit_mode=next(m.get("dit_mode", dit_mode) for m in metrics if m["stage"] == "dit"),
                worker_peak_bytes=max(m.get("peak_bytes", 0) for m in metrics),
                mlx_peak_bytes=max(m.get("mlx_peak_bytes", 0) for m in metrics),
            ))


def generate(prompt: str, **kwargs) -> Image.Image:
    """Render a PIL RGB image. Refer to ordered references as <Picture 1>, etc."""
    return generate_result(prompt, **kwargs).image
