

![H3 Image MLX in liquid-glass lettering on a blue gradient](assets/hero.jpg)

<p align="center"><em>This banner was generated end to end with MiniMax H3 Image using <a href="https://github.com/Anionex/h3-image-mlx">H3 Image MLX</a>.</em></p>
<p align="center">If you find this repository useful, feel free to 🌟 &amp; fork~</p>

# H3 Image MLX
**English** · [简体中文](README.zh-CN.md)

[![Follow @anion_ex on X](https://img.shields.io/static/v1?label=&message=%40anion_ex&color=171717&style=flat-square&logo=x&logoColor=white)](https://x.com/anion_ex)
[![PyPI version](https://img.shields.io/pypi/v/h3-image-mlx?include_prereleases&style=flat-square)](https://pypi.org/project/h3-image-mlx/)
[![Requires Python 3.11 or newer](https://img.shields.io/static/v1?label=Python&message=3.11%2B&color=3776AB&style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Package license: Apache-2.0 AND MIT](https://img.shields.io/static/v1?label=License&message=Apache-2.0%20AND%20MIT&color=326790&style=flat-square)](pyproject.toml)





[H3 quick start](#generate-your-first-h3-image) · [Qwen](#qwen-image-21-turbo) · [Image editing](#edit-with-reference-images) · [How it works](#how-it-works) · [Benchmarks](#benchmark-results) · [Usage](docs/USAGE.md) · [Performance](docs/PERFORMANCE.md)

Generate and edit images locally on Apple Silicon with **MiniMax H3 Image and Qwen Image 2.1 Turbo**. One CLI and Python API, with model-specific sampling, LoRA, and memory policies. H3 uses isolated Python/MLX stages; Qwen adds an optional standalone native runtime. [Qwen setup and examples](docs/QWEN.md) · [Architecture](docs/ARCHITECTURE.md)

## Highlights

- **Native Apple Silicon inference.** Built on MLX to take advantage of your Mac's compute capabilities, H3 Image and Qwen Image 2.1 Turbo run locally with text-to-image generation and ordered multi-image editing.
- **W8A8 quantization and Metal acceleration.** Reduce weight memory usage with a portable MLX inference path, plus W8A8 Metal TensorOps acceleration on supported chips. Compatible with M1–M4.
- **Staged inference with on-demand block loading.** Each stage runs in a separate process to release its memory on completion. Denoiser weights can be read one block at a time, greatly reducing peak memory usage and leaving more room for other apps.
- **Overlap weight reads with GPU computation.** Prefetch the next block while evaluating the current one, keeping streaming memory low and reducing time spent waiting for weights. [Enable prefetch and async](#recommended-streaming-acceleration)

Qwen also exposes the desktop engine’s LoRA stacks, sampling presets, CFG, image-to-image, progress and cancellation through typed options. [Qwen capabilities and usage](docs/QWEN.md)

### Benchmark results

#### Qwen Image 2.1 Turbo: verified uncached weight reads

M5 Pro 48 GB, eight steps, `micro-fast`, three complete generations per size. The benchmark bypasses the macOS weight-file cache throughout text encoding, denoising and decoding.

| Resolution | Median generation time | Highest process peak | Weight-file cache |
| --- | ---: | ---: | ---: |
| 512×512 | **16.94 s** | **0.85 GB** | 0 bytes |
| 1024×1024 | **47.67 s** | **1.41 GB** | 0 bytes |

<sub>All trials included, including the first after cache clearing. Peak is the largest worker physical footprint, not total system RAM. Each generation physically read 64.11 GB; prepared weights already existed. [Conditions and all trials](BENCHMARKS.md#qwen-image-21-turbo-uncached) · [Source JSON](benchmarks/qwen/2026-11-14/summary.json)</sub>

#### Earlier H3 results with system file caching

M5 Pro 48 GB, version `0.1.0a3`, 1024×1024, identical weights, prompt and seed, original six-step sampler, no embedding cache. **These historical H3 runs allowed macOS file caching.**

| Mode | Median generation time | Highest process peak |
| --- | ---: | ---: |
| Resident | 23.63 s | 21.18 GB |
| Streaming | 32.33 s | 2.53 GB |
| **Streaming + prefetch + async** | **23.61 s** | **3.23 GB** |

<sub>Resident and prefetch+async: five measured runs each; plain streaming: three. Warmups excluded. Memory is the highest kernel-recorded process physical footprint across measured runs, in decimal GB. [Full results and timing ranges](BENCHMARKS.md#streaming-prefetch-and-async) · [Source JSON](benchmarks/controlled/2026-11-12-prefetch/summary.json)</sub>

#### Same-machine controlled comparison

This historical H3 comparison allowed system file caching. The same M5 Pro MacBook (48 GB), weight files, teapot prompt, initial noise and six-step sigma grid. Each group has one warmup and three measured generations.

| Resolution | MLX DiT mode | ComfyUI time (s) | MLX time (s) | Speed ratio | ComfyUI peak (GB) | MLX peak (GB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1024×1024 | `resident` | 87.15 | 19.90 | **4.38×** | 31.29 | 21.16 |
| 1024×1024 | `stream` | 87.15 | **28.82** | **3.02×** | 31.29 | **2.53** |
| 2048×2048 | `resident` | 134.17 | 57.43 | **2.34×** | 38.21 | 22.27 |
| 2048×2048 | `stream` | 134.17 | **67.06** | **2.00×** | 38.21 | **3.48** |
| 4096×4096 | `stream` | 570.59 | **317.29** | **1.80×** | 42.31 | **7.18** |

<sub>Time is the median complete generation request. Memory is the kernel-recorded single-process physical-footprint peak, including warmup, in decimal GB. Speed ratio = local ComfyUI time ÷ MLX time.</sub>

This compares complete pipelines: ComfyUI's default text encoder runs on CPU; MLX uses GPU. The main table reproduces the community configuration. The additional native INT8 DiT path, pinned versions and measurement method are in the [full benchmark records](BENCHMARKS.md#controlled-test-setup), with [source JSON](benchmarks/controlled/2026-11-11/summary.json) in this repository.

#### Supplementary community measurements

| Device / memory | Pipeline | DiT mode | Resolution | Time (s) | Process peak (GB) | Source |
| --- | --- | --- | --- | ---: | ---: | --- |
| <sub>M5 MacBook Air / 16 GB</sub> | H3 Image MLX | `stream` | 1024×1024 | **70.51** | **2.48** | [arielfikru](benchmarks/raw/teapot-1024-m5-air-16gb.json) |
| <sub>M5 MacBook Air / 16 GB</sub> | H3 Image MLX | `stream` | 2048×2048 | **240.78** | **2.78** | [arielfikru](benchmarks/raw/teapot-2048-m5-air-16gb.json) |
| <sub>M5 Pro Mac mini / 64 GB</sub> | ComfyUI + MPS INT8 | — | 1024×1024 | 89.55 | 31.42 | [pratikgx](https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench) |
| <sub>M5 Pro Mac mini / 64 GB</sub> | ComfyUI + MPS INT8 | — | 2048×2048 | 164.90 | 36.21 | [pratikgx](https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench) |
| <sub>M5 Pro Mac mini / 64 GB</sub> | ComfyUI + MPS INT8 | — | 4096×4096 | 547.31 | 48.78 | [pratikgx](https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench) |

<sub>Independent measurements on different machines, retaining contributors' original results. Run counts and measurement methods are in the [full records](BENCHMARKS.md#supplementary-earlier-and-community-results), alongside the earlier M5 Pro runs.</sub>

### Generation gallery

[![Six H3 Image MLX examples: a feast, a fox in flight, a celadon tea set, a woodland chapel, a watercolor illustration and a rainy-night watchmaker.](assets/generation-gallery.jpg)](assets/generation-gallery.jpg)

Generated locally with H3 Image MLX: food, wildlife, product photography, architecture, illustration and a 4K watchmaker portrait. Click the collage to view it at full size.

## Qwen Image 2.1 Turbo

Qwen support is currently available from this source branch with the optional native runtime wheel; the published `0.1.0a4` package remains H3-only.

```python
from h3_image_mlx import generate, QwenOptions, Lora

image = generate(
    "A red ceramic teapot on a white tabletop.",
    model="qwen-image-2.1-turbo", models="./qwen-models",
    qwen=QwenOptions(memory_mode="micro-fast", loras=(Lora("style.safetensors", 0.7),)),
)
image.save("teapot.png")
```

Use up to ten ordered references, independently scaled LoRAs, Turbo 3/4/8-step or Lightning/DMD2 presets, CFG and negative prompts, variation and RGBA output. Prompt enhancement is outside this interface. [Installation, downloads, CLI and Python examples](docs/QWEN.md).

## Generate your first H3 image

Version `0.1.0a4` also supports **unquantized BF16 DiT and text-encoder weights**, retaining AdaLN pruning and staged streaming. Select `--precision bf16`; INT8 remains the default. [BF16 download and usage](docs/USAGE.md#bf16-weights)

Use Python 3.11 or later on a Mac with Metal. Allow 49.30 GB of disk space for a fresh model download. Check the [MiniMax model license](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE) for regional and usage restrictions.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'h3-image-mlx==0.1.0a4'

h3-image-mlx fetch --models ./models
h3-image-mlx generate --models ./models --output teapot.png \
  --prompt 'A yellow enamel teapot and a white cup on a blue wooden table.'
```

This writes a 1024×1024 RGB PNG. Set `--width` and `--height` to change the size. Each dimension accepts multiples of 32 from 256 through 4096.

## Recommended streaming acceleration

**Use weight prefetch and asynchronous evaluation together with `--dit-mode stream` to reduce weight-loading waits. Both switches are off by default.** The six-step sampler stays unchanged. Our earlier cache-enabled 1024×1024 H3 test matched resident-mode speed with about 85% less peak process memory. [Same-machine results](#earlier-h3-results-with-system-file-caching) · [Contributor measurements](docs/PERFORMANCE.md#reported-results)

Available from `0.1.0a2`; upgrade and enable both switches for this command:

```bash
python -m pip install --upgrade 'h3-image-mlx==0.1.0a4'
H3_ASYNC=1 H3_PREFETCH=1 h3-image-mlx generate \
  --models ./models --dit-mode stream --output teapot.png \
  --prompt 'A yellow enamel teapot and a white cup on a blue wooden table.'
```

[Usage guide: CLI, Python and existing weights](docs/USAGE.md) · [Performance guide: switches, caching and profiling](docs/PERFORMANCE.md)

**Experimental:** optional two-/three-step LoRA inference and LoRA-merged INT8 checkpoints are available on GitHub `main`, off by default. [Usage and quality trade-offs](docs/PERFORMANCE.md#experimental-few-step-lora)

## Edit with reference images

To replace the cup in `teapot.png` with a bird from your own photo, save the photo as `bird.png`. Number the pictures in your prompt in the order you pass them on the command line.

```bash
h3-image-mlx generate --models ./models --output edit.png \
  --reference teapot.png --reference bird.png \
  --prompt 'Edit <Picture 1>. Replace the white cup with the bird from <Picture 2>. Keep the teapot and table.'
```

You can pass up to 16 references. On an M5 Pro, we completed one- and two-reference edits. The 16-reference test reached the third of six denoising steps before cancellation. [Validation notes](VALIDATION.md)

## How it works

The H3 pipeline reduces memory use at three levels: between model stages, within the denoiser, and during image decoding.

![H3 Image MLX pipeline: encode the prompt, stream denoiser weights, then decode overlapping image tiles. Each stage releases its model memory before the next starts.](assets/how-it-works.jpg)

- **Run one stage at a time.** Text encoding, denoising and VAE decoding run in separate processes. Each saves its intermediate output and exits before the next loads its weights, so the three models do not need to stay in memory together. Editing adds a reference-image VAE encoding stage before denoising.
- **Keep quantized projections compact.** The runtime reads Comfy-Org's INT8 ConvRot weights directly. Activations are rotated and dynamically quantized to INT8 before quantized linear projections. Supported chips use W8A8 Metal TensorOps; the portable backend evaluates the same quantized projections with tiled standard MLX operations. Other operations retain their appropriate floating-point precision.
- **Load denoiser weights as needed.** In `stream` mode, tensors are read from safetensors as each transformer block runs, rather than cached for all six steps. Denoising still operates on the full image's latent sequence. `resident` mode caches weights to reduce repeated reads. The runtime also precomputes the steps' AdaLN modulation values from the compressed curves in Comfy-Org's pruned checkpoint, then releases the setup weights.
- **Decode in overlapping tiles.** The VAE decodes small groups of latent tiles and blends their overlapping edges into the output image, limiting decoder working memory at high resolutions.

Streaming keeps memory low by reading weights on demand. Prefetch and asynchronous evaluation overlap these reads with GPU work to reduce the time cost. Peak usage still includes activations, attention buffers and image buffers, so resolution and reference-image count matter. See the [benchmark results](#benchmark-results) for the measured trade-off.

## Hardware and memory

M5/M6 select W8A8 Metal TensorOps by default; M1–M4 use the portable MLX backend. Use `--dit-mode stream` for lower memory usage or `--dit-mode resident` to retain denoiser weights between steps. [Hardware validation and automatic mode selection](docs/PERFORMANCE.md#hardware-and-memory)

## Documentation

| Document | Contents |
| --- | --- |
| [Qwen guide](docs/QWEN.md) | Qwen setup, LoRA, presets, editing, memory modes and runtime build |
| [Architecture](docs/ARCHITECTURE.md) | Public requests, model adapters and isolated workers |
| [Usage guide](docs/USAGE.md) | CLI parameters, reference editing, Python API and existing weights |
| [Performance guide](docs/PERFORMANCE.md) | Recommended streaming switches, hardware modes, embedding cache and profiling |
| [Benchmarks](BENCHMARKS.md) | Measurements, environments and source JSON |
| [Validation notes](VALIDATION.md) | Completed checks and remaining hardware coverage |

## Credits and licenses

MiniMax developed [H3 Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) and supplies the [VAE configuration](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/492fd8c4eca666f1085f57e30738c1e0d86b3439/vae/config.json). [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) provides the quantized weights. The three W8A8 Metal kernel files come from [mlx-serve](https://github.com/ddalcu/mlx-serve/tree/9191fbdf697df333cac19ec7a93eaa1098ce3ad5/src/kernels) and retain their [MIT license](src/h3_image_mlx/kernels/LICENSE). This project’s runtime code uses [Apache-2.0](LICENSE).

## About

I'm [anionex](https://anionex.me/), an AI-native developer who once ranked No. 3 on GitHub's global developer trending list, with more than 18.7k stars across my projects. If you would like to follow my future work, [follow me on X](https://x.com/anion_ex) or [GitHub](https://github.com/Anionex).
