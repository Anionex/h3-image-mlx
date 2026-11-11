

![H3 Image MLX in liquid-glass lettering on a blue gradient](assets/hero.jpg)

<p align="center"><em>This banner was generated end to end with MiniMax H3 Image using <a href="https://github.com/Anionex/h3-image-mlx">H3 Image MLX</a>.</em></p>

# H3 Image MLX
**English** · [简体中文](README.zh-CN.md)

[![Follow @anion_ex on X](https://img.shields.io/static/v1?label=&message=%40anion_ex&color=171717&style=flat-square&logo=x&logoColor=white)](https://x.com/anion_ex)
[![PyPI version](https://img.shields.io/pypi/v/h3-image-mlx?include_prereleases&style=flat-square)](https://pypi.org/project/h3-image-mlx/)
[![Requires Python 3.11 or newer](https://img.shields.io/static/v1?label=Python&message=3.11%2B&color=3776AB&style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Package license: Apache-2.0 AND MIT](https://img.shields.io/static/v1?label=License&message=Apache-2.0%20AND%20MIT&color=326790&style=flat-square)](pyproject.toml)





[Quick start](#generate-your-first-image) · [Image editing](#edit-with-reference-images) · [How it works](#how-it-works) · [Benchmarks](#benchmark-results)

Generate and edit images on an Apple Silicon Mac with MiniMax H3 Image. Optimized for fast inference, with 4K generation using as little as 7 GB of peak inference-process memory. Available through a CLI and Python API, ready to use out of the box.

If you find this repository useful, feel free to 🌟 & fork~

## Highlights

- **Native Apple Silicon inference.** Built on MLX to take advantage of your Mac's compute capabilities, H3 Image runs locally with text-to-image generation and multi-image editing.
- **W8A8 quantization and Metal acceleration.** Reduce weight memory usage with a portable MLX inference path, plus W8A8 Metal TensorOps acceleration on supported chips. Compatible with M1–M4.
- **Staged inference with on-demand block loading.** Each stage runs in a separate process to release its memory on completion. Denoiser weights can be read one block at a time, greatly reducing peak memory usage and leaving more room for other apps.

For reference, in our 1024×1024 generation benchmark, streaming reduced the peak inference-process footprint from **21.07 GB to 2.67 GB**, with median runtime increasing from **22.31 s to 30.90 s**. [Benchmark conditions and results](#benchmark-results)

### Benchmark results

| Device / memory | Pipeline | DiT mode | Resolution | Time (s) | Process peak (GB) | Source |
| --- | --- | --- | --- | ---: | ---: | --- |
| <sub>M5 Pro MacBook / 48 GB</sub> | H3 Image MLX | `resident` | 512×512 | 16.50 | 20.56 | [Anionex](benchmarks/raw/teapot-512.json) |
| <sub>M5 Pro MacBook / 48 GB</sub> | H3 Image MLX | `resident` | 1024×1024 | 22.31 | 21.07 | [Anionex](benchmarks/raw/teapot-1024-ab.json) |
| <sub>M5 Pro MacBook / 48 GB</sub> | H3 Image MLX | `stream` | 1024×1024 | **30.90** | **2.67** | [Anionex](benchmarks/raw/teapot-1024-ab.json) |
| <sub>M5 Pro MacBook / 48 GB</sub> | H3 Image MLX | `resident` | 2048×2048 | 65.54 | 22.04 | [Anionex](benchmarks/raw/teapot-2048.json) |
| <sub>M5 Pro MacBook / 48 GB</sub> | H3 Image MLX | `stream` | 4096×4096 | **356.84** | **6.86** | [Anionex](benchmarks/raw/teapot-4096.json) |
| <sub>M5 MacBook Air / 16 GB</sub> | H3 Image MLX | `stream` | 1024×1024 | **70.51** | **2.48** | [arielfikru](benchmarks/raw/teapot-1024-m5-air-16gb.json) |
| <sub>M5 MacBook Air / 16 GB</sub> | H3 Image MLX | `stream` | 2048×2048 | **240.78** | **2.78** | [arielfikru](benchmarks/raw/teapot-2048-m5-air-16gb.json) |
| <sub>M5 Pro Mac mini / 64 GB</sub> | ComfyUI + MPS INT8 | — | 1024×1024 | 89.55 | 31.42 | [pratikgx](https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench) |
| <sub>M5 Pro Mac mini / 64 GB</sub> | ComfyUI + MPS INT8 | — | 2048×2048 | 164.90 | 36.21 | [pratikgx](https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench) |
| <sub>M5 Pro Mac mini / 64 GB</sub> | ComfyUI + MPS INT8 | — | 4096×4096 | 547.31 | 48.78 | [pratikgx](https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench) |

<sub>Time is a three-run median or a single run; memory is peak process physical footprint in decimal GB. ComfyUI uses the ComfyUI-MPS-INT8 extension. See the [full records](BENCHMARKS.md) for run counts, environments and measurement methods.</sub>

### Rainy-night watchmaker

[![A watchmaker lifts a brass gear above an open watch, with a rain-streaked window behind him.](assets/watchmaker-preview.jpg)](assets/watchmaker-4096.png)

[Open the original 4096×4096 PNG (22.7 MB)](assets/watchmaker-4096.png?raw=true)

Generated on the M5 Pro (48 GB) in 6 min 10 s, with a peak inference-process footprint of 6.88 GB.

## Generate your first image

Use Python 3.11 or later on a Mac with Metal. Allow 49.30 GB of disk space for a fresh model download. Check the [MiniMax model license](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE) for regional and usage restrictions.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'h3-image-mlx==0.1.0a1'

h3-image-mlx fetch --models ./models
h3-image-mlx generate --models ./models --output teapot.png \
  --prompt 'A yellow enamel teapot and a white cup on a blue wooden table.'
```

This writes a 1024×1024 RGB PNG. Set `--width` and `--height` to change the size. Each dimension accepts multiples of 32 from 256 through 4096.

## Edit with reference images

To replace the cup in `teapot.png` with a bird from your own photo, save the photo as `bird.png`. Number the pictures in your prompt in the order you pass them on the command line.

```bash
h3-image-mlx generate --models ./models --output edit.png \
  --reference teapot.png --reference bird.png \
  --prompt 'Edit <Picture 1>. Replace the white cup with the bird from <Picture 2>. Keep the teapot and table.'
```

You can pass up to 16 references. On an M5 Pro, we completed one- and two-reference edits. The 16-reference test reached the third of six denoising steps before cancellation. [Validation notes](VALIDATION.md)

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

`fetch` checks existing files and downloads missing ones from ModelScope. If you already have the three weight files and tokenizer, it downloads just the 1,826-byte VAE configuration. For Hugging Face, add `--source huggingface`. To check the files against the [SHA-256 manifest](src/h3_image_mlx/weights.json), run `h3-image-mlx verify --models ./models`.

## How it works

The pipeline reduces memory use at three levels: between model stages, within the denoiser, and during image decoding.

![H3 Image MLX pipeline: encode the prompt, stream denoiser weights, then decode overlapping image tiles. Each stage releases its model memory before the next starts.](assets/how-it-works.jpg)

- **Run one stage at a time.** Text encoding, denoising and VAE decoding run in separate processes. Each saves its intermediate output and exits before the next loads its weights, so the three models do not need to stay in memory together. Editing adds a reference-image VAE encoding stage before denoising.
- **Keep quantized projections compact.** The runtime reads Comfy-Org's INT8 ConvRot weights directly. Activations are rotated and dynamically quantized to INT8 before quantized linear projections. Supported chips use W8A8 Metal TensorOps; the portable backend evaluates the same quantized projections with tiled standard MLX operations. Other operations retain their appropriate floating-point precision.
- **Load denoiser weights as needed.** In `stream` mode, tensors are read from safetensors as each transformer block runs, rather than cached for all six steps. Denoising still operates on the full image's latent sequence. `resident` mode caches weights to reduce repeated reads. The runtime also precomputes the steps' AdaLN modulation values from the compressed curves in Comfy-Org's pruned checkpoint, then releases the setup weights.
- **Decode in overlapping tiles.** The VAE decodes small groups of latent tiles and blends their overlapping edges into the output image, limiting decoder working memory at high resolutions.

Streaming trades additional weight reads for a smaller memory footprint. Peak usage still includes activations, attention buffers and image buffers, so resolution and reference-image count matter. See the [benchmark results](#benchmark-results) for the measured trade-off.

## Hardware and memory

| Chip | Default compute backend | Hardware tests |
| --- | --- | --- |
| M5 family | W8A8 Metal TensorOps | Text-to-image, single- and two-reference edits on M5 Pro (48 GB); text-to-image and a single-reference edit on M5 MacBook Air (16 GB), see [benchmarks](BENCHMARKS.md#community-run-macbook-air-m5-16-gb) |
| M1–M4 | Portable MLX | Portable backend exercised on M5 Pro. M1–M4 tests pending. |
| M6 family | W8A8 Metal TensorOps | Pending |

Text encoding, six-step denoising and image decoding each run in a separate process that releases its weights on exit. Editing adds a VAE encoding stage for the references.

Apple Silicon shares memory between the CPU and GPU. To leave room for other apps, add `--dit-mode stream` to read the denoiser weights one block at a time. With memory to spare, use `--dit-mode resident` to keep weights between steps and cut generation time. The default, `auto`, streams on the portable backend, below 32 GiB of system memory, or above 8,192 packed token positions.

To compare compute backends on an M5 Pro, run the same prompt with `--backend w8a8` and `--backend portable`. See the [validation notes](VALIDATION.md) for settings and pixel comparisons.

## Credits and licenses

MiniMax developed [H3 Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) and supplies the [VAE configuration](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/492fd8c4eca666f1085f57e30738c1e0d86b3439/vae/config.json). [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) provides the quantized weights. The three W8A8 Metal kernel files come from [mlx-serve](https://github.com/ddalcu/mlx-serve/tree/9191fbdf697df333cac19ec7a93eaa1098ce3ad5/src/kernels) and retain their [MIT license](src/h3_image_mlx/kernels/LICENSE). This project’s runtime code uses [Apache-2.0](LICENSE).

## About

I'm [anionex](https://anionex.me/), an AI-native developer who once ranked No. 3 on GitHub's global developer trending list, with more than 18.7k stars across my projects. If you would like to follow my future work, [follow me on X](https://x.com/anion_ex) or [GitHub](https://github.com/Anionex).
