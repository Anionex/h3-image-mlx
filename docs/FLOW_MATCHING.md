# DiT and flow-matching inference on Apple Silicon

The library stages inference so model weights can be released at the end of each stage. Large Transformer stacks can read one block at a time from safetensors. The executor can read the following block on a background thread while Metal evaluates the current block.

| Model | Local weight format | What uses the shared execution code |
| --- | --- | --- |
| MiniMax H3 Image | Existing ConvRot INT8 or BF16 checkpoints | Bounded DiT and text prefetch, aligned reads, MLX evaluation |
| Qwen Image 2.1 Turbo | Existing Turbo Q8 checkpoint | Optional MLX text-to-image denoiser with cached text keys and values. Text encoding and VAE use the existing native runtime. |
| Z-Image Turbo | MFlux sharded safetensors | Text encoder and DiT blocks. The MFlux sampler and VAE generate the image. |
| MOSS-SoundEffect v2.0 | Converted MOSS MLX BF16 or affine-quantized checkpoint | Text, paired CFG denoising, and DAC decoder. The original full-context solver supplies the latents. |
| Ming-3.0-TTS 0.5B | MLX Audio checkpoint | Selective loading and layer-by-layer waveform decoding. The compact DiTAR stays in memory for repeated flow-matching calls. |

## Installation

Use an Apple Silicon Mac and a Python version supported by the selected dependencies. MFlux 0.24.0 and MLX Audio 0.6.1 are the adapter versions used for validation.

```sh
uv pip install --python .venv/bin/python -e '.[mflux,speech,sound-effects]'
uv pip install --python .venv/bin/python \
  'moss-sfx-mlx @ git+https://github.com/xocialize/moss-soundeffect-mlx.git@4e8e3e36d4d9c991d16a3e65a4b7099d406b9ed8'
```

Qwen's text encoder and VAE require a native runtime. Install its local wheel into the checkout's virtual environment or pass the existing binary with `--runtime`. The [runtime build instructions](QWEN.md#build-the-optional-runtime) cover building the native package from source.

Prepare local model directories with the [Z-Image Turbo MFlux Q4 pack](https://huggingface.co/mflux-community/z-image-turbo-mflux-q4), [MOSS MLX weights](https://huggingface.co/mlx-community/MOSS-SoundEffect-v2.0-bf16), or [Ming MLX weights](https://huggingface.co/mlx-community/Ming-3.0-tts-0.5B-8bit). Each checkpoint is subject to its upstream terms.

## Run from the CLI

```sh
h3-image-mlx generate --model h3 --models ./h3 --prompt 'A red teapot' \
  --dit-mode stream --weight-cache uncached --stream-execution overlap --output h3.png

h3-image-mlx generate --model qwen-image-2.1-turbo --models ./qwen --prompt 'A red teapot' \
  --no-prepare --memory-mode micro-fast --qwen-dit-execution stream-overlap \
  --weight-cache uncached --output qwen.png

h3-image-mlx generate --model z-image-turbo --models ./z-image --prompt 'A red teapot' \
  --weight-cache uncached --output z-image.png

h3-image-mlx audio --model moss-soundeffect-v2 --models ./moss --prompt 'Rain on a glass roof' \
  --seconds 3 --weight-cache uncached --output rain.wav

h3-image-mlx audio --model ming-3.0-tts --models ./ming --prompt '你好，欢迎收听。' \
  --language zh --weight-cache uncached --output speech.wav
```

Use `--metrics` to write execution details next to an image or waveform. Python provides the same dispatch through `generate_result(...)` and `generate_audio_result(...)`. Qwen reference editing, LoRA, alternative samplers, and resident sessions are available through the established native path.

## Extend the executor

A model adapter loads the component configurations and supplies block order, a mapping from model parameter names to checkpoint tensor names, and the block forward functions. `TensorStore` indexes sharded checkpoints and reads only the requested tensors. `BlockStream.register` and `BlockStream.run` work with explicit functions. `BlockStream.wrap` binds a compatible MLX module to checkpoint names for the duration of each forward. Model adapters keep their own conditioning and sampling logic. Keep executor calls in Python and apply `mx.compile` to pure tensor functions inside the blocks.

`StreamPolicy` selects synchronous or overlapped execution. The default enables overlap on Metal. The prefetch budget covers pending raw read buffers, including alignment overhead. Overlap allows up to two submitted block evaluations and one pending read in `BlockStream`. Active weights, resident conditioning and model components, activations, and the MLX allocator contribute to overall memory independently.

On macOS, uncached mode uses page-aligned offsets, lengths, and destination buffers with `F_NOCACHE`. Run `benchmarks/h3_streaming.py` for a same-seed H3 comparison. It records process and MLX peaks, system memory and swap, file residency, disk counter deltas, and exact RGB hashes. A strict-cache result requires sustained low weight residency and adequate per-stage and system physical reads.
