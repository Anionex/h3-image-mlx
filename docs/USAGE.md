# Usage guide

[README](../README.md) · **English** · [简体中文](USAGE.zh-CN.md) · [Performance guide](PERFORMANCE.md)

## Install and download

Use Python 3.11 or newer on an Apple Silicon Mac with Metal. A fresh model download needs 49.30 GB of disk space.

```bash
python -m pip install --upgrade 'h3-image-mlx==0.1.0a3'
h3-image-mlx fetch --models ./models
```

ModelScope is the default source. Use `--source huggingface` with `fetch` to select Hugging Face. Downloads resume from partial files. The model files and VAE configuration are fetched separately from the Python package.

## BF16 weights

Select `--precision bf16` to use the unquantized BF16 DiT and text encoder while retaining the existing AdaLN pruning. INT8 remains the default. BF16 uses standard MLX floating-point operations and does not quantize activations to INT8; `--backend w8a8` is only available with INT8 weights. The existing FP16 VAE, tokenizer and VAE configuration are shared.

This option is available in the source checkout; install it with `python -m pip install -e .` until the next package release. A fresh BF16 model directory requires **92.92 GB** of disk space. If the shared components are already present, the two additional BF16 files need **91.74 GB**. Downloads check available space before transferring files.

```bash
h3-image-mlx fetch --models ./models --precision bf16
h3-image-mlx verify --models ./models --precision bf16
h3-image-mlx generate --models ./models --precision bf16 --dit-mode stream \
  --prompt 'A yellow enamel teapot on a blue wooden table.' --output bf16.png
```

ModelScope remains the default download source; add `--source huggingface` to `fetch` to switch. Both profiles can coexist in one directory. `--precision` is explicit on each command, so downloading BF16 does not change the default INT8 selection. Python callers use `generate(..., precision="bf16")`.

In BF16 mode, `--dit-mode auto` selects streaming on every chip; `resident` is still available explicitly for Macs with sufficient memory. Streaming, prefetch/async settings and reference-image ordering work as before. An `H3_DIT` override must also contain floating-point weights when BF16 is selected.

BF16 files, in addition to the shared VAE and tokenizer:

```text
diffusion_models/minimax_h3_image_pruned_bf16.safetensors
text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors
```

The DiT is from [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image); the matching 50-layer text encoder is from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3). Exact revisions, sizes and SHA-256 hashes are pinned in [weights-bf16.json](../src/h3_image_mlx/weights-bf16.json).

## Generate and edit

```bash
h3-image-mlx generate --models ./models --output teapot.png \
  --prompt 'A yellow enamel teapot and a white cup on a blue wooden table.'

h3-image-mlx generate --models ./models --output edit.png \
  --reference teapot.png --reference bird.png \
  --prompt 'Edit <Picture 1>. Replace the white cup with the bird from <Picture 2>. Keep the teapot and table.'
```

References are ordered: `<Picture 1>` is the first `--reference`, and so on. Supply up to 16 PNG, JPEG or WebP files. Output is always an RGB PNG. The default sampler runs six steps. Version `0.1.0a3` also provides [experimental few-step LoRA settings](PERFORMANCE.md#experimental-few-step-lora) through environment variables, off by default.

## CLI parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--models` | Required | Model directory for `fetch`, `verify` and `generate` |
| `--precision` | `int8` | `int8` or `bf16`, for `fetch`, `verify` and `generate` |
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
