"""Private stage entry point. Each child releases its weights on exit."""
import json
import re
import sys
from pathlib import Path
from .models import DIT, TEXT, TOKENIZER, VAE, VAE_CONFIG


def select_backend(requested, device):
    native = bool(re.search(r"\bApple M[56]\b", device))
    if requested == "auto":
        return "w8a8" if native else "portable"
    if requested == "w8a8" and not native:
        raise ValueError(f"The W8A8 TensorOps kernel requires an M5/M6 GPU; found {device}")
    return requested


def main(stage, work):
    import mlx.core as mx
    from . import core
    request = json.loads((work / "request.json").read_text())
    info = mx.device_info()
    backend = select_backend(request["backend"], info["device_name"])
    core.backend.set(backend)
    mx.set_cache_limit(96 << 20)
    models = Path(request["models"])
    refs = request["references"]
    embedding, latents, result = (str(work / (key + ".safetensors")) for key in ("embedding", "references", "latent"))
    if stage == "text":
        from .text import encode
        x, tags, _ = encode(models / TEXT, models / TOKENIZER, request["prompt"], refs)
        mx.save_safetensors(embedding, {"embedding": x, "tags": tags})
    elif stage == "references":
        from .ref_vae import encode_reference
        weights = core.Weights(models / VAE, True)
        stats = json.loads((models / VAE_CONFIG).read_text())
        zs = {}
        for i, ref in enumerate(refs):
            zs[str(i)] = encode_reference(weights, ref, stats)
            mx.eval(zs[str(i)])
        mx.save_safetensors(latents, zs)
    elif stage == "dit":
        from .dit import denoise
        inputs = mx.load(embedding)
        images = mx.load(latents) if refs else {}
        zs = [images[str(i)] for i in range(len(refs))]
        width, height = request["width"], request["height"]
        rows = inputs["embedding"].shape[0] + width * height // 1024 + sum(z.shape[1] * z.shape[2] // 4 for z in zs)
        mode = request["dit_mode"]
        resident = mode == "resident" or (mode == "auto" and backend == "w8a8" and
                    info["memory_size"] >= 32 * 1024**3 and rows <= 8192)
        z, _ = denoise(models / DIT, inputs["embedding"], width, height, request["seed"], resident, zs, inputs["tags"])
        mx.save_safetensors(result, {"latent": z})
    elif stage == "vae":
        from .vae import decode
        stats = json.loads((models / VAE_CONFIG).read_text())
        decode(models / VAE, mx.load(result)["latent"], stats).save(work / "image.png")
    else:
        raise ValueError(f"Unknown stage: {stage}")
    mx.synchronize()


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
