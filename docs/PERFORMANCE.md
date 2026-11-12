# Performance guide

[README](../README.md) · **English** · [简体中文](PERFORMANCE.zh-CN.md) · [Usage guide](USAGE.md)

## Recommended streaming acceleration

Starting with `0.1.0a2`, we recommend trying **both weight prefetch and asynchronous evaluation** in `stream` mode, especially on a 16 GB M5 Mac. They are **off by default** and must be enabled explicitly.

```bash
H3_ASYNC=1 H3_PREFETCH=1 h3-image-mlx generate \
  --models ./models --dit-mode stream --output teapot.png \
  --prompt 'A yellow enamel teapot and a white cup on a blue wooden table.'
```

The GPU evaluates the current transformer block while the next block's weights are read and prepared. The sampler still runs six steps. With asynchronous evaluation, up to two blocks of weights are kept alive instead of waiting for each block before proceeding.

Remove the two environment variables to use the default path. If you previously exported them in your shell, run `unset H3_ASYNC H3_PREFETCH`, or set both to `0` for a single command. CLI and Python calls inherit the same environment variables.

## Reported results

[PR #2](https://github.com/Anionex/h3-image-mlx/pull/2), contributed by arielfikru: M5 MacBook Air, 16 GB, macOS 26.6.2, MLX 0.32.4, teapot prompt, seed 42, six steps, W8A8 and streaming. These comparisons enable `H3_ASYNC=1 H3_PREFETCH=1` without the embedding cache.

| Measurement | Default | Both switches enabled |
| --- | ---: | ---: |
| 1024×1024, full generation | 59.8 s | 34.8 s rested; 43.6 s warm |
| 1024×1024, DiT only, interleaved A/B | 52.3 s | 31.3–31.7 s |
| 2048×2048, DiT only | 144.7 s | 124.6 s |
| 1024×1024, peak process footprint | 2.5 GB | 2.5 GB |
| 2048×2048, peak process footprint | 2.81 GB | 3.06 GB |

The contributor reports pixel-identical outputs at both resolutions. Full-generation and DiT measurements came from different runs. The 2048 pair ran the optimized path first, with a two-minute pause before the default path. Gains depend on available memory and shared memory bandwidth; sustained GPU work also makes the fanless Air more heat-sensitive. M5 Pro, resident mode, the portable backend, reference editing and 4K were not covered by these optimization measurements.

## Environment variables

All are optional; unset means disabled.

| Variable | Effect |
| --- | --- |
| `H3_ASYNC=1` | Queue DiT blocks with `mx.async_eval` and overlap work between adjacent blocks |
| `H3_PREFETCH=1` | Read the next block's raw weight bytes on a worker thread; MLX calls remain on the main thread |
| `H3_EMBED_CACHE=<directory>` | Reuse text-stage output for repeated prompts and ordered references |
| `H3_PROFILE=<file.json>` | Save DiT timing, weight-read statistics and the MLX allocator peak |

### Repeated-prompt cache

```bash
H3_EMBED_CACHE=./.h3-embedding-cache h3-image-mlx generate \
  --models ./models --output teapot.png \
  --prompt 'A yellow enamel teapot and a white cup on a blue wooden table.'
```

The first request writes the cache; a matching request skips text encoding. Denoising, image decoding and reference VAE encoding still run. Cache files take disk space. Use a separate cache directory for concurrent jobs, and clear or change the directory after replacing text weights or the tokenizer. Remove `H3_EMBED_CACHE` to stop using the cache; delete that cache directory to reclaim its space.

### Profiling

Set `H3_PROFILE=./dit-profile.json` on a generation command. The file records setup and per-step times, read bytes and time, conversion time, prefetch wait time, evaluation time and the MLX allocator peak. Overlapped timings are not additive. The allocator peak is a different measurement from process physical footprint. Use a different filename for each result you want to keep.

## Hardware and memory

| Chip | Default compute backend | Hardware tests |
| --- | --- | --- |
| M5 family | W8A8 Metal TensorOps | Text-to-image, single- and two-reference edits on M5 Pro (48 GB); text-to-image and a single-reference edit on M5 MacBook Air (16 GB), see [benchmarks](../BENCHMARKS.md#community-run-macbook-air-m5-16-gb) |
| M1–M4 | Portable MLX | Portable backend exercised on M5 Pro. M1–M4 tests pending. |
| M6 family | W8A8 Metal TensorOps | Pending |

Text encoding, six-step denoising and image decoding each run in a separate process that releases its weights on exit. Editing adds a VAE encoding stage for the references.

Apple Silicon shares memory between the CPU and GPU. To leave room for other apps, add `--dit-mode stream` to read the denoiser weights one block at a time. With memory to spare, use `--dit-mode resident` to keep weights between steps and cut generation time. The default, `auto`, streams on the portable backend, below 32 GiB of system memory, or above 8,192 packed token positions.

To compare compute backends on an M5 Pro, run the same prompt with `--backend w8a8` and `--backend portable`. See the [validation notes](../VALIDATION.md) for settings and pixel comparisons.
