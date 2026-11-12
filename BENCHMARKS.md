# Benchmarks

The November 12 test compares resident inference with streaming, prefetch and asynchronous evaluation. The November 11 **same-machine ComfyUI comparison on an M5 Pro MacBook (48 GB)** follows: **11 groups, 33 measured generations and 11 warmups**. Earlier M5 Pro runs and independent community results remain below as supplementary data.

## Streaming prefetch and async

November 12, 2026, M5 Pro 48 GB, PyPI `h3-image-mlx==0.1.0a3`, MLX 0.32.4, W8A8. At **1024×1024**, streaming with `H3_ASYNC=1 H3_PREFETCH=1` matches resident-mode median generation time while reducing the highest measured process peak by **84.74%**.

| Mode | Measured runs | Median generation (s) | Generation range (s) | Median DiT (s) | Process peak range (GB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Resident | 5 | 23.63 | 21.27–29.40 | 18.05 | 21.01–21.18 |
| Streaming | 3 | 32.33 | 30.33–39.43 | 26.79 | 2.51–2.53 |
| Streaming + prefetch + async | 5 | **23.61** | 20.88–29.19 | 18.11 | **2.53–3.23** |

All 16 images (13 measured generations and three warmups) have identical RGB pixels. The optimized streaming median is 0.08% below resident mode and 1.37× as fast as plain streaming.

- Same original weight files, [teapot prompt](#prompt), seed 42 and six-step sampler; no LoRA or embedding cache. All three stages execute for every request.
- One warmup per mode, followed by three interleaved rounds: R→S→P, S→P→R, P→R→S. After observing timing variation, two additional pairs run P→R and R→P. R is resident, S plain streaming, P streaming with prefetch and async. All measured samples are included.
- Generation timing includes text encoding, denoising, VAE decoding and subprocess startup; it excludes saving the returned PIL image to a final PNG. DiT timing includes subprocess startup. These are a separate experiment from the November 11 ComfyUI comparison below.
- Process memory is the kernel lifetime physical-footprint peak, taking the maximum across the three isolated stage processes, in decimal GB. Ranges and headline maxima use measured runs only; warmups are excluded.
- Before this experiment, an idle App model retained about 17.52 GB and a resident trial took 168.21 s under memory pressure. That preliminary batch is excluded. After the App model process exited, all warmups and measurements were restarted; background applications stayed open.

[Configuration](benchmarks/controlled/2026-11-12-prefetch/configuration.json) · [All runs](benchmarks/controlled/2026-11-12-prefetch/runs.json) · [Summary](benchmarks/controlled/2026-11-12-prefetch/summary.json) · [Independent result verification](benchmarks/controlled/2026-11-12-prefetch/verification.json)

## Same-machine controlled comparison

Both paths use the same local Comfy-Org INT8 ConvRot weights, teapot prompt, initial noise and six Euler steps. Each group has one warmup and three timed requests.

| Resolution | MLX DiT mode | ComfyUI time (s) | MLX time (s) | Speed ratio | ComfyUI peak (GB) | MLX peak (GB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1024×1024 | `resident` | 87.15 | 19.90 | **4.38×** | 31.29 | 21.16 |
| 1024×1024 | `stream` | 87.15 | **28.82** | **3.02×** | 31.29 | **2.53** |
| 2048×2048 | `resident` | 134.17 | 57.43 | **2.34×** | 38.21 | 22.27 |
| 2048×2048 | `stream` | 134.17 | **67.06** | **2.00×** | 38.21 | **3.48** |
| 4096×4096 | `stream` | 570.59 | **317.29** | **1.80×** | 42.31 | **7.18** |

Time is the median complete request, including PNG output. Peak memory is the kernel-recorded single-process lifetime physical footprint, including warmup, in decimal GB (10⁹ bytes). Speed ratio = the local stock-gate ComfyUI median ÷ the MLX median, calculated before rounding.

At 1024×1024, streaming reduces the MLX peak from **21.16 GB to 2.53 GB**, with median time increasing from **19.90 s to 28.82 s**. Resident and streaming outputs match pixel for pixel at both 1024×1024 and 2048×2048.

## Controlled test setup

- **Machine:** M5 Pro MacBook, 16 GPU cores, 48 GiB unified memory, macOS 26.6, AC power. GPU runs were serial; background applications were left open and system memory/swap was recorded.
- **MLX:** project commit [`64df997`](https://github.com/Anionex/h3-image-mlx/tree/64df997cd00a698302b1303f51028eee179487fe), MLX 0.32.4, W8A8 Metal TensorOps. Text, DiT and VAE execute in separate processes for every request. 4K uses `stream`, matching `auto` on this machine.
- **ComfyUI:** commit [`0439dd7`](https://github.com/Comfy-Org/ComfyUI/tree/0439dd79e6c06a17f146eafe41542614420f38b8), [ComfyUI-MPS-INT8](https://github.com/QZGao/ComfyUI-MPS-INT8/tree/b23c371505a365e68c32cb12ca5bcde65656abcd) at `b23c371`, PyTorch 2.15.0. Model nodes run directly in an isolated environment without the UI server; model objects persist across warmup and timed requests. Neither path caches text conditioning between requests.
- **Shared inputs:** the [teapot prompt](#prompt), seed 42, guidance 1, the same local DiT, text-encoder and FP16 VAE files listed in the [weight manifest](src/h3_image_mlx/weights.json). A CPU PyTorch float32 noise tensor is saved once per size and injected into both paths. All 69 token IDs and VAE spatial tile positions match.
- **Schedule alignment:** the wrapper replaces ComfyUI's slightly interpolated `simple` schedule with the exact model PDD sigma grid used by MLX. Both arrays are retained in [configuration validation](benchmarks/controlled/2026-11-11/configuration-validation.json).
- **Timing:** one fresh worker per group, one full warmup, then three measured requests. Persistent model-object initialization is recorded separately. Every measured request includes text encoding, denoising, VAE decoding and PNG writing. Wall-clock and monotonic durations were checked for discontinuities.

### INT8 capability gate and text encoding

The pinned ComfyUI version returns `False` for MPS in `supports_int8_compute`; registering the extension alone does not override it. The reproduced community configuration therefore records **zero native INT8 calls**. The additional `comfy-native` wrapper enables only that MPS gate and records **1,200 native INT8 DiT calls per generation**. Its results are kept separate; the main speed ratios use the faster stock-gate path. The community publication does not document a separate local gate override.

ComfyUI's default text encoder runs on **CPU**, while MLX uses **GPU**. These are complete-pipeline timings; the stage breakdown below shows where the time is spent.

### Memory measurement

The README and main table use the kernel's lifetime physical-footprint counter: ComfyUI's persistent worker lifetime versus the maximum across all MLX stage processes, **including warmup on both sides**. The complete table also reports a separate **100 ms sampled process-tree peak during the three measured requests**, including the controller worker and its children. Sampling can miss brief spikes. These are process footprints, not total machine-memory requirements. System-wide swap maxima are preserved in the JSON and include other applications.

## Complete controlled measurements

| Resolution | Pipeline | Median time (s) | Range (s) | Sampled tree peak (GB) | Kernel process peak (GB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1024×1024 | ComfyUI, stock gate | 87.15 | 86.37–87.85 | 31.30 | 31.29 |
| 1024×1024 | ComfyUI, native INT8 DiT | 94.49 | 90.19–97.83 | 27.89 | 27.88 |
| 1024×1024 | MLX `resident` | 19.90 | 19.51–21.02 | 21.05 | 21.16 |
| 1024×1024 | MLX `stream` | 28.82 | 28.67–29.02 | 2.38 | 2.53 |
| 2048×2048 | ComfyUI, stock gate | 134.17 | 133.96–136.57 | 38.22 | 38.21 |
| 2048×2048 | ComfyUI, native INT8 DiT | 233.79 | 233.69–237.19 | 34.75 | 34.74 |
| 2048×2048 | MLX `resident` | 57.43 | 56.88–57.82 | 22.16 | 22.27 |
| 2048×2048 | MLX `stream` | 67.06 | 66.55–67.69 | 3.20 | 3.48 |
| 4096×4096 | ComfyUI, stock gate | 570.59 | 553.50–674.30 | 40.21 | 42.31 |
| 4096×4096 | ComfyUI, native INT8 DiT | 1134.28 | 1130.95–1180.55 | 37.37 | 39.61 |
| 4096×4096 | MLX `stream` | 317.29 | 316.58–317.76 | 7.05 | 7.18 |

### Stage timings

| Resolution | Pipeline | Text (s) | DiT (s) | VAE (s) |
| --- | --- | ---: | ---: | ---: |
| 1024×1024 | ComfyUI, stock gate | 39.34 | 47.07 | 0.67 |
| 1024×1024 | ComfyUI, native INT8 DiT | 41.95 | 51.28 | 0.56 |
| 1024×1024 | MLX `resident` | 4.19 | 14.71 | 0.68 |
| 1024×1024 | MLX `stream` | 4.18 | 23.71 | 0.68 |
| 2048×2048 | ComfyUI, stock gate | 36.10 | 96.30 | 2.17 |
| 2048×2048 | ComfyUI, native INT8 DiT | 38.01 | 193.33 | 2.34 |
| 2048×2048 | MLX `resident` | 4.16 | 50.66 | 2.32 |
| 2048×2048 | MLX `stream` | 4.38 | 60.06 | 2.31 |
| 4096×4096 | ComfyUI, stock gate | 35.25 | 523.39 | 10.02 |
| 4096×4096 | ComfyUI, native INT8 DiT | 41.59 | 1082.40 | 14.28 |
| 4096×4096 | MLX `stream` | 4.93 | 303.87 | 7.96 |

Stage timings include each implementation's stage-local loading. MLX's VAE stage also includes PNG writing; ComfyUI's PNG writing is included in total request time, outside its VAE stage. Speed ratios use complete request time.

### Verification and source records

All **44 output images** passed dimension and RGB-hash checks. MLX repeated outputs are pixel-identical in every group; resident and streaming match at 1024×1024 and 2048×2048. The native ComfyUI INT8 variant has small pixel differences between repeated runs at all three sizes. The reviewed 4K outputs from all paths have similar composition and share a double-spout teapot artifact.

- [Full-precision summary](benchmarks/controlled/2026-11-11/summary.json): medians, ranges, both memory metrics, stage medians, system-swap maxima and output hashes.
- [Individual runs](benchmarks/controlled/2026-11-11/runs.json): all 11 warmups and 33 timed requests, stage counters, native-call counts and input-noise hashes. This is a compact export of the original JSON; numeric fields are unchanged, while local output paths and per-second system telemetry are omitted.
- [Configuration checks](benchmarks/controlled/2026-11-11/configuration-validation.json) · [Result verification](benchmarks/controlled/2026-11-11/verification.json) · [Pinned-source verification](benchmarks/controlled/2026-11-11/source-verification.json).

The summary omits only local image paths. Source files matched the pinned archives, and the MLX working tree was clean at the recorded commit. The interrupted disk-space pilot and the initial INT8-gate diagnostic are excluded from all reported measurements.

## Supplementary: earlier and community results

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

`resident` keeps denoiser weights in memory between steps; `stream` reads them block by block.

H3 Image MLX times cover text encoding through PNG output. Both devices report three-run medians at 1024×1024; other MLX rows are single runs. ComfyUI reports three-run medians from text encoding through VAE decode. Memory is the highest process peak reported for each row; GB = 1,000,000,000 bytes. Values are rounded to two decimal places.

Our M5 Pro measurements use the maximum `physical_peak_bytes` among the sequential stage processes, measured with macOS `proc_pid_rusage`. The Air contributor uses `/usr/bin/time -l`; the ComfyUI contributor samples process footprint every 0.2 seconds. Their individual setups and raw records are below.

In the earlier M5 Pro test at 1024×1024, streaming reduced peak process memory by **18.40 GB (87.3%)**, adding **8.58 s** to median generation time. All six outputs matched pixel for pixel.

## Community baseline: ComfyUI + MPS INT8

External baseline: **ComfyUI + ComfyUI-MPS-INT8**, measured by **pratikgx** on November 10, 2026. Both pipelines use Comfy-Org's pruned INT8 ConvRot weights, seed 42 and six sampling steps.

The supplementary table above includes these results. Community memory values are converted from GiB to GB using `GiB × 1024³ / 10⁹`.

### Earlier cross-machine test setup

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

The supplementary table converts memory units from the source values. Each row links to its measurement source.

## Individual 1024×1024 runs

| Run | DiT loading | End-to-end time (s) | Peak process memory (GB) |
| --- | --- | ---: | ---: |
| 1 | `resident` | 22.29 | 21.05 |
| 1 | `stream` | 30.90 | 2.61 |
| 2 | `resident` | 22.31 | 20.99 |
| 2 | `stream` | 30.35 | 2.67 |
| 3 | `resident` | 24.58 | 21.07 |
| 3 | `stream` | 30.90 | 2.52 |

These are the earlier M5 Pro runs. Source: [paired-run JSON](benchmarks/raw/teapot-1024-ab.json), including full-precision timings, byte counts and output RGB hashes.

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

The earlier `benchmarks/raw/` and `benchmarks/community/` JSON files remain unchanged. The current same-machine comparison uses the controlled exports linked above.
