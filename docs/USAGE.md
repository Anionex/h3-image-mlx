# Usage guide

[README](../README.md) · **English** · [简体中文](USAGE.zh-CN.md) · [Performance guide](PERFORMANCE.md)

## Install and download

Use Python 3.11 or newer on an Apple Silicon Mac with Metal. A fresh model download needs 49.30 GB of disk space.

```bash
python -m pip install --upgrade 'h3-image-mlx==0.1.0a2'
h3-image-mlx fetch --models ./models
```

ModelScope is the default source. Use `--source huggingface` with `fetch` to select Hugging Face. Downloads resume from partial files. The model files and VAE configuration are fetched separately from the Python package.

## Generate and edit

```bash
h3-image-mlx generate --models ./models --output teapot.png \
  --prompt 'A yellow enamel teapot and a white cup on a blue wooden table.'

h3-image-mlx generate --models ./models --output edit.png \
  --reference teapot.png --reference bird.png \
  --prompt 'Edit <Picture 1>. Replace the white cup with the bird from <Picture 2>. Keep the teapot and table.'
```

References are ordered: `<Picture 1>` is the first `--reference`, and so on. Supply up to 16 PNG, JPEG or WebP files. Output is always an RGB PNG. The default sampler runs six steps. GitHub `main` also provides [experimental few-step LoRA settings](PERFORMANCE.md#experimental-few-step-lora) through environment variables; these are not included in PyPI `0.1.0a2`.

## CLI parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--models` | Required | Model directory for `fetch`, `verify` and `generate` |
| `--prompt` | Required | Nonempty generation or editing instruction |
| `--output` | Required | Output path ending in `.png` |
| `--width`, `--height` | `1024` each | Multiples of 32 from 256 through 4096 |
| `--seed` | `42` | Nonnegative integer |
| `--reference` | None | Repeat for each image, in prompt order |
| `--backend` | `auto` | `auto`, `w8a8` or `portable` |
| `--dit-mode` | `auto` | `auto`, `stream` or `resident` |
| `--source` (`fetch` only) | `modelscope` | `modelscope` or `huggingface` |

For lower memory use, select `--dit-mode stream`. We recommend trying the [optional streaming acceleration](PERFORMANCE.md#recommended-streaming-acceleration), especially on a 16 GB M5 Mac. Hardware selection and all performance environment variables are documented there.

## Python

`generate` returns a Pillow image. Pass reference paths in a list, in the same order as in the CLI.

```python
from h3_image_mlx import generate

image = generate(
    "Replace the cup in <Picture 1> with the bird from <Picture 2>.",
    models="./models",
    references=["teapot.png", "bird.png"],
    seed=42,
    dit_mode="stream",
)
image.save("edit.png")
```

## Reuse existing weights

Point `--models` at a directory with this layout. Symlinks work.

```text
diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
vae/minimax_h3_image_vae_fp16.safetensors
vae/config.json
tokenizer.json
```

`fetch` checks existing files and downloads missing ones from ModelScope. If you already have the three weight files and tokenizer, it downloads just the 1,826-byte VAE configuration. For Hugging Face, add `--source huggingface`. To check the files against the [SHA-256 manifest](../src/h3_image_mlx/weights.json), run `h3-image-mlx verify --models ./models`.
