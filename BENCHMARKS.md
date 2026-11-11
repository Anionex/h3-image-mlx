# Benchmarks

Results from our **M5 Pro MacBook (48 GB)**, [arielfikru](https://github.com/arielfikru)'s **M5 MacBook Air (16 GB)** and [pratikgx](https://huggingface.co/pratikgx)'s **M5 Pro Mac mini (64 GB)**. H3 Image MLX runs use the W8A8 Metal backend; the external ComfyUI results use the ComfyUI-MPS-INT8 extension. All rows use INT8 ConvRot weights and six denoising steps.

## Generation time and memory

| Device / memory | Pipeline | DiT mode | Resolution | Time (s) | Process peak (GB) | Source |
| --- | --- | --- | --- | ---: | ---: | --- |
| M5 Pro MacBook / 48 GB | H3 Image MLX | `resident` | 512×512 | 16.50 | 20.56 | [Anionex](benchmarks/raw/teapot-512.json) |
| M5 Pro MacBook / 48 GB | H3 Image MLX | `resident` | 1024×1024 | 22.31 | 21.07 | [Anionex](benchmarks/raw/teapot-1024-ab.json) |
| M5 Pro MacBook / 48 GB | H3 Image MLX | `stream` | 1024×1024 | 30.90 | 2.67 | [Anionex](benchmarks/raw/teapot-1024-ab.json) |
| M5 Pro MacBook / 48 GB | H3 Image MLX | `resident` | 2048×2048 | 65.54 | 22.04 | [Anionex](benchmarks/raw/teapot-2048.json) |
| M5 Pro MacBook / 48 GB | H3 Image MLX | `stream` | 4096×4096 | 356.84 | 6.86 | [Anionex](benchmarks/raw/teapot-4096.json) |
| M5 MacBook Air / 16 GB | H3 Image MLX | `stream` | 1024×1024 | 70.51 | 2.48 | [arielfikru](benchmarks/raw/teapot-1024-m5-air-16gb.json) |
| M5 MacBook Air / 16 GB | H3 Image MLX | `stream` | 2048×2048 | 240.78 | 2.78 | [arielfikru](benchmarks/raw/teapot-2048-m5-air-16gb.json) |
| M5 Pro Mac mini / 64 GB | ComfyUI + MPS INT8 | — | 1024×1024 | 89.55 | 31.42 | [pratikgx](https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench) |
| M5 Pro Mac mini / 64 GB | ComfyUI + MPS INT8 | — | 2048×2048 | 164.90 | 36.21 | [pratikgx](https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench) |
| M5 Pro Mac mini / 64 GB | ComfyUI + MPS INT8 | — | 4096×4096 | 547.31 | 48.78 | [pratikgx](https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench) |

`resident` keeps denoiser weights in memory between steps; `stream` reads them block by block.

H3 Image MLX times cover text encoding through PNG output. Both devices report three-run medians at 1024×1024; other MLX rows are single runs. ComfyUI reports three-run medians from text encoding through VAE decode. Memory is the highest process peak reported for each row; GB = 1,000,000,000 bytes. Values are rounded to two decimal places.

Our M5 Pro measurements use the maximum `physical_peak_bytes` among the sequential stage processes, measured with macOS `proc_pid_rusage`. The Air contributor uses `/usr/bin/time -l`; the ComfyUI contributor samples process footprint every 0.2 seconds. Their individual setups and raw records are below.

At 1024×1024, streaming reduced peak process memory by **18.40 GB (87.3%)**, adding **8.58 s** to median generation time. All six outputs matched pixel for pixel.

## Community baseline: ComfyUI + MPS INT8

External baseline: **ComfyUI + ComfyUI-MPS-INT8**, measured by **pratikgx** on November 10, 2026. Both pipelines use Comfy-Org's pruned INT8 ConvRot weights, seed 42 and six sampling steps.

The unified table above includes these results. Community memory values are converted from GiB to GB using `GiB × 1024³ / 10⁹`.

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

The unified table converts memory units from the source values. Each row links to its measurement source.

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

## Community run: MacBook Air M5, 16 GB

Contributed by [arielfikru](https://github.com/arielfikru) in [PR #1](https://github.com/Anionex/h3-image-mlx/pull/1), on the base M5 chip with 16 GB. It uses the same teapot prompt, seed 42 and six steps, with default settings. On this machine `auto` selects the `w8a8` backend and `stream` loading.

| Run | Text (s) | DiT (s) | VAE (s) | End-to-end (s) |
| --- | ---: | ---: | ---: | ---: |
| 1024×1024 run 1 | 6.30 | 52.20 | 1.24 | 59.85 |
| 1024×1024 run 2 | 5.83 | 62.24 | 2.30 | 70.51 |
| 1024×1024 run 3 | 7.83 | 73.45 | 1.84 | 83.28 |
| 2048×2048 run 1 | 8.25 | 225.38 | 7.02 | 240.78 |

- **Identical output across chips.** All three 1024×1024 runs have RGB SHA-256 `97d3c28bc44c4a740d9841882b62080ab21e5ed6c898f5894aa25020c3a6f4f1`, the same as every M5 Pro run in the paired-run JSON.
- **Sustained runs slow down.** The three back-to-back 1024×1024 runs took 59.85 s, 70.51 s and 83.28 s; nearly all of the increase is in the DiT stage. The MacBook Air has no fan, so this is likely thermal throttling. Expect longer times in batches than in a single run.
- **Swap was essentially unchanged.** About 1.1 GB was already in use from other apps; the largest start-to-end increase recorded was 0.38 MiB.
- **Reference editing also fits.** A single-reference edit at 2048×1152 (reference image of the same size) completed in 391 s with a peak of 3.59 GB. That run used a different prompt and is not part of the table.

Hardware: MacBook Air (Mac17,3), Apple M5, 10-core CPU, 8-core GPU, 16 GB, macOS 26.6.2, MLX 0.32.4, commit `64df997`. Peak memory is the highest lifetime peak physical footprint among the stage processes, as reported by `/usr/bin/time -l`. Everyday apps were open during the runs.

## Prompt

```text
A close-up product photograph of one bright yellow enamel teapot with a black arched handle and a round lid, standing on a blue wooden table. One small white ceramic cup sits immediately to its right. The background is plain warm beige. Soft morning window light falls from the left. Realistic enamel, clear silhouettes, no text.
```

The linked JSON files are unchanged copies of the original measurement records. The 1024×1024 paired runs supply our M5 Pro figures in the README table.
