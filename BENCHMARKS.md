# Benchmarks

Measured on an **Apple M5 Pro with 48 GB of unified memory**, using INT8 ConvRot weights and the W8A8 Metal backend. All rows use the same teapot prompt, seed **42**, and **six denoising steps**.

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
