"""Text-to-image and ordered multi-reference editing in isolated MLX stages."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from PIL import Image
from .models import TEXT, check_models


def embedding_cache(directory, prompt, refs, backend):
    """Opt-in (H3_EMBED_CACHE=dir): reuse the text-stage output for a repeated prompt + references."""
    root = os.environ.get("H3_EMBED_CACHE")
    if not root:
        return None
    encoder = Path(os.environ.get("H3_TEXT") or directory / TEXT)  # an overridden encoder gets its own entries
    key = [prompt, backend, encoder.stat().st_size, [hashlib.sha256(p.read_bytes()).hexdigest() for p in refs]]
    name = hashlib.sha256(json.dumps(key).encode()).hexdigest()
    Path(root).expanduser().mkdir(parents=True, exist_ok=True)
    return Path(root).expanduser() / f"{name}.safetensors"


def generate(prompt: str, *, models: str | Path, width: int = 1024, height: int = 1024,
             seed: int = 42, references: Sequence[str | Path] = (),
             backend: str = "auto", dit_mode: str = "auto") -> Image.Image:
    """Render a PIL RGB image. Refer to ordered reference files as <Picture 1>, etc."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Provide a nonempty prompt")
    if not all(type(v) is int and 256 <= v <= 4096 and v % 32 == 0 for v in (width, height)):
        raise ValueError("Width and height must be multiples of 32 from 256 through 4096")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if backend not in ("auto", "portable", "w8a8") or dit_mode not in ("auto", "stream", "resident"):
        raise ValueError("Unknown compute backend or DiT mode")
    if isinstance(references, (str, Path)) or len(references) > 16:
        raise ValueError("Supply an ordered list of up to 16 reference files")
    refs = [Path(ref).expanduser().resolve() for ref in references]
    for path in refs:
        with Image.open(path) as im:
            if im.format not in ("PNG", "JPEG", "WEBP"):
                raise ValueError(f"Unsupported reference image format: {path}")
            if max(im.size) / min(im.size) * 1024**2 > 12_845_056:
                raise ValueError(f"Reference exceeds the vision encoder aspect ratio budget: {path}")
    directory = check_models(models)
    with tempfile.TemporaryDirectory(prefix="h3-image-mlx-") as tmp:
        work = Path(tmp)
        args = dict(prompt=prompt, models=str(directory), width=width, height=height, seed=seed,
                    references=[str(p) for p in refs], backend=backend, dit_mode=dit_mode)
        (work / "request.json").write_text(json.dumps(args))
        stages = ["text", *(["references"] if refs else []), "dit", "vae"]
        cached = embedding_cache(directory, prompt, refs, backend)
        if cached and cached.is_file():
            shutil.copy(cached, work / "embedding.safetensors")
            stages.remove("text")
        for stage in stages:
            proc = subprocess.run([sys.executable, "-m", "h3_image_mlx._stage", stage, tmp],
                                  capture_output=True, text=True)
            if proc.returncode:
                raise RuntimeError(f"H3 {stage} stage exited with {proc.returncode}: {proc.stderr.strip()}")
            if stage == "text" and cached:
                shutil.copy(work / "embedding.safetensors", cached.with_suffix(".tmp"))
                cached.with_suffix(".tmp").replace(cached)
        with Image.open(work / "image.png") as image:
            return image.convert("RGB")
