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

## Experimental few-step LoRA

Available starting with `0.1.0a3` on PyPI. All experimental settings are off by default. Without them, generation and editing retain the original six-step path. Upgrade to try it:

```bash
python -m pip install --upgrade 'h3-image-mlx==0.1.0a3'
```

### Load a LoRA directly

Download `minimax_h3_pdmd_2nfe_comfyui.safetensors` from [akhaliq/minimax-h3-image-pdmd-2step](https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step/tree/main) and save it under `./loras/`. This is the ComfyUI-layout conversion tested in [PR #3](https://github.com/Anionex/h3-image-mlx/pull/3); the upstream adapter uses different tensor names. LoRAs are downloaded separately from `h3-image-mlx fetch`.

```bash
env -u H3_DIT H3_LORA=./loras/minimax_h3_pdmd_2nfe_comfyui.safetensors \
  H3_STEPS=2 H3_SHIFT=12 H3_ASYNC=1 H3_PREFETCH=1 \
  h3-image-mlx generate --models ./models --dit-mode stream \
  --width 1024 --height 1024 --seed 42 --output teapot-2step.png \
  --prompt 'A yellow enamel teapot and a white cup on a blue wooden table.'
```

Set `H3_STEPS=3` and change the output filename to compare three steps. The [upstream pDMD adapter](https://huggingface.co/pdmd2026/pdmd_2NFE_lora) was trained for two steps; three steps are an inference setting tested by the contributor. In their teapot example, two steps left a fine weave on flat areas, while three steps largely removed it. Both can change composition and detail compared with the six-step image, even at the same seed.

| Setting | Effect when explicitly set |
| --- | --- |
| `H3_LORA=<file>` | Apply a single ComfyUI-layout adapter alongside the quantized projections |
| `H3_STEPS=2` or `3` | Use a shifted sampling grid with fewer steps; pair with the matching distilled LoRA or merged checkpoint |
| `H3_SHIFT=12` | Sampling-grid shift; 12 is used when few-step sampling is enabled and no shift is specified |
| `H3_DIT=<file>` | Override the DiT checkpoint with another file using the same layout |

The CLI and Python API read the same environment variables. To return to the original six-step checkpoint, run `unset H3_LORA H3_STEPS H3_SHIFT H3_DIT`. The independent `H3_ASYNC` and `H3_PREFETCH` switches may remain enabled.

### Merge the LoRA into an INT8 checkpoint

Direct loading adds low-rank computation at inference time without changing the existing model file. The optional [merge script](../scripts/merge_lora_int8.py) instead applies the adapter to BF16 weights before ConvRot quantization and writes a separate INT8 checkpoint. Run it from a checkout of this repository or the unpacked source distribution, with the package dependencies installed:

```bash
python scripts/merge_lora_int8.py \
  --base ./models/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors \
  --lora ./loras/minimax_h3_pdmd_2nfe_comfyui.safetensors \
  --out ./models/diffusion_models/minimax_h3_image_pdmd_merged_int8_convrot.safetensors
```

The script uses macOS APFS cloning and reads roughly 40 GB of upstream BF16 tensors over HTTP without keeping a full BF16 download. Allow space for an additional INT8 checkpoint, roughly 21 GB, as the clone is rewritten. `--rev` can pin the upstream Hugging Face revision. Use a distinct output path, never the base model path; reuse an existing output only when resuming the same base, LoRA and upstream revision. Progress is saved beside the output as `.progress.json`. Wait for `DONE` before using the result; `--limit` is a development option that leaves a partial checkpoint.

```bash
env -u H3_LORA \
  H3_DIT=./models/diffusion_models/minimax_h3_image_pdmd_merged_int8_convrot.safetensors \
  H3_STEPS=3 H3_SHIFT=12 H3_ASYNC=1 H3_PREFETCH=1 \
  h3-image-mlx generate --models ./models --dit-mode stream \
  --output teapot-merged.png \
  --prompt 'A yellow enamel teapot and a white cup on a blue wooden table.'
```

Do not also load the same `H3_LORA` with a merged checkpoint: that would apply it twice. Quantization makes the merged and direct-LoRA outputs numerically different. `verify --models` continues to verify the original pinned model files, not an `H3_DIT` override.

### Contributor measurements and coverage

[PR #3](https://github.com/Anionex/h3-image-mlx/pull/3) reports these 1024×1024 teapot runs on an M5 MacBook Air with 16 GB, seed 42, W8A8, streaming and both acceleration switches enabled:

| Mode | Complete generation |
| --- | ---: |
| Original six steps | 34.8–43.6 s |
| Direct LoRA, two steps | 23.4 s |
| Direct LoRA, three steps | 25.7–28.6 s |
| Direct LoRA, three steps, embedding-cache hit | 18.7 s |

The merged checkpoint reduced DiT-only time from 23.9/23.0 s to 20.3/18.3 s in two interleaved rounds on a warm machine using swap. These are separate measurements from the complete-generation table. No comparable peak-memory figure was supplied for the experimental modes.

The contributor tested one prompt at 1024×1024. Other LoRAs, reference editing with the LoRA, other resolutions, resident mode and the portable backend remain unverified. Our regression checks separately cover existing generation and editing with the experimental settings unset; see [validation notes](../VALIDATION.md#experimental-lora-integration).


## Hardware and memory

| Chip | Default compute backend | Hardware tests |
| --- | --- | --- |
| M5 family | W8A8 Metal TensorOps | Text-to-image, single- and two-reference edits on M5 Pro (48 GB); text-to-image and a single-reference edit on M5 MacBook Air (16 GB), see [benchmarks](../BENCHMARKS.md#community-run-macbook-air-m5-16-gb) |
| M1–M4 | Portable MLX | Portable backend exercised on M5 Pro. M1–M4 tests pending. |
| M6 family | W8A8 Metal TensorOps | Pending |

Text encoding, six-step denoising and image decoding each run in a separate process that releases its weights on exit. Editing adds a VAE encoding stage for the references.

Apple Silicon shares memory between the CPU and GPU. To leave room for other apps, add `--dit-mode stream` to read the denoiser weights one block at a time. With memory to spare, use `--dit-mode resident` to keep weights between steps and cut generation time. The default, `auto`, streams on the portable backend, below 32 GiB of system memory, or above 8,192 packed token positions.

To compare compute backends on an M5 Pro, run the same prompt with `--backend w8a8` and `--backend portable`. See the [validation notes](../VALIDATION.md) for settings and pixel comparisons.
