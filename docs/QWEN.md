# Qwen Image 2.1 Turbo

Qwen shares the package's CLI and Python API with H3. Its native worker is a separate optional platform package, so H3 installations keep their existing runtime. The implementation reuses the desktop app's model engine without starting the app or an HTTP server. Prompt enhancement and desktop UI/history are outside this API.

## Install from this source branch

Qwen support in this branch requires the Python package plus the native runtime wheel built below. It is not included in the already-published `0.1.0a4` wheel.

```bash
python -m pip install .
python -m pip install packages/qwen-runtime/dist/h3_image_mlx_qwen_runtime-*.whl
```

The native wheel contains its dynamic libraries and Metal resources. It requires Apple Silicon and macOS 26.2 or later. On M5/M6 the default is W8A8 TensorOps; earlier chips select ordinary quantized MLX operations. The portable path is tested on M5; M1–M4 still need hardware validation.

## Fetch and generate

```bash
h3-image-mlx fetch --model qwen-image-2.1-turbo --models ./qwen-models
h3-image-mlx generate --model qwen-image-2.1-turbo --models ./qwen-models \
  --prompt 'A red ceramic teapot on a white tabletop.' --output teapot.png \
  --memory-mode micro-fast --progress --metrics metrics.json
```

ModelScope is preferred. `--source huggingface` chooses the alternate pinned source; downloaded files are size/hash checked. The original Q8 model pack occupies about 17.73 GB. W8A8 preparation adds about 7.25 GB of persistent DiT files and needs 8 GB free space. Editing may derive an additional vision-only shard. `--no-prepare` reads original weights without preparing a second DiT copy. Preparation is separate from generation metrics. Existing model directories can be reused directly.

Four memory policies are available: `resident`, `prefetch`, `micro-fast`, and `micro`. Resolution accepts multiples of 32, 256–4096; validated resolutions are listed in the benchmark/validation records. Eight Turbo steps remain the default.

## Python, progress and cancellation

```python
from threading import Event
from h3_image_mlx import generate_result, QwenOptions, Lora

cancel = Event()  # another thread can call cancel.set()
result = generate_result(
    "A red ceramic teapot on a white tabletop.",
    model="qwen-image-2.1-turbo", models="./qwen-models",
    width=1024, height=1024, seed=42,
    qwen=QwenOptions(memory_mode="micro-fast"),
    on_progress=print, cancel=cancel,
)
result.image.save("teapot.png")
print(result.metrics)
```

`generate(...)` returns only the PIL image. Existing H3 calls remain compatible. `QwenOptions(runtime="/path/to/runtime")` selects an explicit native bundle. `cache=` selects where derived weights and model views live; `prepared=` reuses an existing prepared pack after checking its transformer identity.

The parent creates no MLX GPU context. It launches workers, drains stdout/stderr, dispatches actual step events, enforces timeout/cancellation and reaps its workers. `QwenOptions(private=True)` sends prompt/reference data through pipes rather than temporary prompt or latent files. Private mode does not automatically save an image or prompt history; caller-requested output files and the OS swap mechanism are separate.

## Editing, variation and transparency

```bash
h3-image-mlx generate --model qwen-image-2.1-turbo --models ./qwen-models \
  --reference teapot.png --reference palette.png \
  --prompt 'Use the teapot from <image1> with the blue glaze from <image2>.' \
  --output edited.png --memory-mode micro-fast
```

Up to ten ordered references use `<image1>`, `<image2>`, etc. PNG/JPEG/WebP inputs retain alpha when converted for the engine. A mask or pose rendering is another reference image and counts toward the same limit. No reference receives a hidden canvas/subject role. `--reference-resolution 256` through `1024` controls conditioning resolution.

`--reference-mode variation --strength 0.5` uses one image for image-to-image denoising. `--transparent` requests RGBA output; ordinary outputs are RGB. Editing, variation, transparency, LoRA and nondefault sampling use the complete-request engine entry point, and their process memory differs from plain staged text-to-image.

## LoRA and sampling presets

Load local adapters with independent scales:

```bash
h3-image-mlx generate --model qwen-image-2.1-turbo --models ./qwen-models \
  --prompt 'A character following the pose in <image1>.' --reference pose.png \
  --lora ./pose-adapter.safetensors --lora-scale 0.7 \
  --output posed.png
```

Repeat `--lora` and `--lora-scale` for a stack of up to eight adapters, including a sampling-preset adapter. If scales are omitted, all default to 1.0. Shapes, supported layer names and paired A/B matrices are checked before inference. Adapters are attached at inference time; source model files are not modified.

The desktop's sampling profiles are exposed directly:

| Profile | Model / adapter | Steps |
|---|---|---:|
| `complete` | Existing Turbo checkpoint | 8; custom grid requires explicit sigmas |
| `turbo4`, `turbo3` | Turbo checkpoint with a shorter sigma grid | 4 / 3 |
| `standard` | Base Qwen Image 2.1 DiT | `--steps` |
| `lightning` | Base DiT + LightX2V adapter | 4 |
| `dmd2` | Base DiT + DMD2 adapter | 2 |
| `turbo8` | Base DiT + official Turbo adapter | 8 |

```bash
h3-image-mlx fetch-profile --profile lightning --directory ./qwen-presets
h3-image-mlx generate --model qwen-image-2.1-turbo --models ./qwen-models \
  --profile lightning --profile-assets ./qwen-presets --auto-strength \
  --prompt 'A red ceramic teapot.' --output lightning.png
```

`--profile-strength` sets a preset adapter's scale. `--auto-strength` applies the desktop Lightning rule and is accepted only for that preset. Additional `--lora` arguments may be combined with presets. Base/adapter downloads are explicit; generation does not silently download them. Some third-party adapters have only a Hugging Face source.

Lightning, DMD2 and Turbo-adapter presets fix CFG to 1 and clear the negative prompt, matching the desktop preset behavior. For the complete or standard profile, `--cfg` and `--negative-prompt` expose real CFG; at CFG 1 the negative prompt does not activate another branch. `--sigmas` supplies a descending sigma for each `--steps` step with the `complete` profile. A preset's fixed grid cannot be overridden accidentally.

## Cache policy and measurement

`--weight-cache uncached` uses page-aligned uncached weight reads through a bounded buffer. `system` retains normal macOS file caching. This is independent of persistent prepared weight files and the explicit prefetch buffers counted in process memory.

The uncached option selects a read policy; the benchmark runner additionally checks file residency and physical reads before accepting a low-memory result. A cold start alone is insufficient. Read [BENCHMARKS](../BENCHMARKS.md) for measured conditions. Historical cached speed numbers must not be read as minimum-machine-memory requirements.

## Build the optional runtime

The source revision and patch digest are in `native/source.json`. Starting from that revision of `ddalcu/mlx-serve`, apply `native/qwen-runtime.patch`, initialize its pinned submodules, and follow that repository's source-build prerequisites (Xcode and pinned Zig/MLX). Build and bundle with:

```bash
python scripts/build_qwen_runtime.py --source /path/to/patched/mlx-serve --build
python -m build --wheel packages/qwen-runtime
```

An already-built source tree can omit `--build`. The bundler checks its source patch against the pinned digest, collects the dylib dependency closure and Metal resource, rewrites local library paths, signs the copies, and includes license notices plus per-file hashes. It does not modify source binaries or installed applications. `native/aligned_nocache.c` supplies the optional uncached reader. The runtime wheel contains no model weights.

Uncached reads currently require W8A8, `micro` / `micro-fast`, and default eight-step plain text-to-image. Editing, LoRA, private pipes, the portable backend and other modes reject this option until their read paths are verified.
