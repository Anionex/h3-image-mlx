# Benchmarks

Measured on an **Apple M5 Pro with 48 GB of unified memory**, using INT8 ConvRot weights and the W8A8 Metal backend. Our runs use the same teapot prompt, seed **42**, and **six denoising steps**.

## Generation time and memory

| Resolution | DiT loading | Runs | End-to-end time (s) | Peak process memory (GB) | Source |
| --- | --- | ---: | ---: | ---: | --- |
| 512×512 | `resident` | 1 | 16.50 | 20.56 | [JSON](benchmarks/raw/teapot-512.json) |
| 1024×1024 | `resident` | 3 | 22.31 | 21.07 | [JSON](benchmarks/raw/teapot-1024-ab.json) |
| 1024×1024 | `stream` | 3 | 30.90 | 2.67 | [JSON](benchmarks/raw/teapot-1024-ab.json) |
| 2048×2048 | `resident` | 1 | 65.54 | 22.04 | [JSON](benchmarks/raw/teapot-2048.json) |
| 4096×4096 | `stream` | 1 | 356.84 | 6.86 | [JSON](benchmarks/raw/teapot-4096.json) |

`resident` keeps denoiser weights in memory between steps; `stream` reads them block by block.

Time covers text encoding through PNG output. For each 1024×1024 mode, time is the median of three runs and memory is the highest peak across those runs. Other rows are single runs. Peak process memory is the maximum `physical_peak_bytes` among the sequential stage processes, measured with macOS `proc_pid_rusage`. GB = 1,000,000,000 bytes. Values are rounded to two decimal places.

At 1024×1024, streaming reduced peak process memory by **18.40 GB (87.3%)**, adding **8.58 s** to median generation time. All six outputs matched pixel for pixel.

## Community baseline: ComfyUI + MPS INT8

External baseline: **ComfyUI + ComfyUI-MPS-INT8**, measured by **pratikgx** on November 10, 2026. Both pipelines use Comfy-Org's pruned INT8 ConvRot weights, seed 42 and six sampling steps.

| Resolution | Our DiT mode | Community time (s) | Our time (s) | Speed ratio | Community memory (GB) | Our memory (GB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1024×1024 | `resident` | 89.55 | **22.31** | **4.01×** | 31.42 | **21.07** |
| 1024×1024 | `stream` | 89.55 | **30.90** | **2.90×** | 31.42 | **2.67** |
| 2048×2048 | `resident` | 164.90 | **65.54** | **2.52×** | 36.21 | **22.04** |
| 4096×4096 | `stream` | 547.31 | **356.84** | **1.53×** | 48.78 | **6.86** |

Speed ratio = community end-to-end time ÷ our end-to-end time. Memory is peak process footprint; community GiB values are converted to GB using `GiB × 1024³ / 10⁹`. Ratios are calculated before rounding.

At 1024×1024, our resident mode takes **75.1% less time**. Our streamed 4K run takes **34.8% less time** and uses **85.9% less peak process memory** than the community result.

### Test setup

| | Community baseline | H3 Image MLX |
| --- | --- | --- |
| Hardware | M5 Pro Mac mini, 20 GPU cores, 64 GB | M5 Pro MacBook, 16 GPU cores, 48 GB |
| Runtime | ComfyUI + ComfyUI-MPS-INT8, PyTorch MPS | MLX + W8A8 Metal TensorOps |
| Prompt | MiniMax's expanded example prompt | Teapot prompt below |
| Runs | One warmup, three timed runs per size | Three per mode at 1024×1024; one at 2048×2048 and 4096×4096 |
| Timing | Text encoding through VAE decode | Text encoding through PNG output |

These are independent measurements. The community author reported a concurrent inference app and background indexing. Their timings are medians and their memory peaks are sampled every 0.2 seconds; our stage processes report their lifetime peak footprint. Hardware, prompts and background load differ between the two tests.

### Source and attribution

Source: [pratikgx/minimax-h3-image-m5-pro-mac-mini-bench](https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench), revision `ccf4a00ea925953c33e6157f6d0ce35eaf938fd8`, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- [Original summary JSON](https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench/blob/ccf4a00ea925953c33e6157f6d0ce35eaf938fd8/summary.json) · [unchanged local copy](benchmarks/community/pratikgx-summary.json)
- [Environment and versions](https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench/blob/ccf4a00ea925953c33e6157f6d0ce35eaf938fd8/environment.json)
- [Test method and background-load notes](https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench/blob/ccf4a00ea925953c33e6157f6d0ce35eaf938fd8/README.md)

The comparison tables convert units and calculate ratios from the source values. Our measurements are linked in the table above.

## Individual 1024×1024 runs

| Run | DiT loading | End-to-end time (s) | Peak process memory (GB) |
| --- | --- | ---: | ---: |
| 1 | `resident` | 22.29 | 21.05 |
| 1 | `stream` | 30.90 | 2.61 |
| 2 | `resident` | 22.31 | 20.99 |
| 2 | `stream` | 30.35 | 2.67 |
| 3 | `resident` | 24.58 | 21.07 |
| 3 | `stream` | 30.90 | 2.52 |

Source: [paired-run JSON](benchmarks/raw/teapot-1024-ab.json), including full-precision timings, byte counts and output RGB hashes.

## Prompt

```text
A close-up product photograph of one bright yellow enamel teapot with a black arched handle and a round lid, standing on a blue wooden table. One small white ceramic cup sits immediately to its right. The background is plain warm beige. Soft morning window light falls from the left. Realistic enamel, clear silhouettes, no text.
```

The linked JSON files are unchanged copies of the original measurement records. The 1024×1024 paired runs supply the figures in the README chart.
