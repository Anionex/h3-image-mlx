# H3 Image MLX

Generate and edit images locally with MiniMax H3 Image and the Comfy-Org INT8 ConvRot weights. The runtime accepts ordered reference files and runs the text encoder, optional reference VAE, six-step DiT and image decoder in separate processes.

## Install

Use Python 3.11+ on an Apple Silicon Mac with Metal. In a fresh environment, install the alpha release:

```bash
uv venv --python 3.13
uv pip install 'h3-image-mlx==0.1.0a1'
source .venv/bin/activate
```

## Model files

Supply a directory containing these five files. Their source revisions, sizes and SHA-256 digests are pinned in the package's `weights.json`. You can reuse an existing model directory or link to its files. A fresh download needs 49.30 GB of disk space. The runtime loads all weights locally. Review the [MiniMax model license](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE) for its territorial, use and distribution terms before fetching or using the model.

```text
diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
vae/minimax_h3_image_vae_fp16.safetensors
vae/config.json
tokenizer.json
```

```bash
h3-image-mlx verify --models ./models
# To fetch the five files into an empty directory, select a source:
h3-image-mlx fetch --models ./models --source modelscope
# --source huggingface selects the pinned Hugging Face files instead.
```

## Generate and edit

```bash
h3-image-mlx generate --models ./models --prompt 'A yellow enamel teapot on a blue wooden table' --output teapot.png
h3-image-mlx generate --models ./models --reference teapot.png --reference bird.png \
  --prompt 'Edit <Picture 1>. Replace the cup with the bird from <Picture 2>.' --output edit.png
```

```python
from h3_image_mlx import generate

im = generate("A red fox in fresh snow", models="./models", width=1024, height=1024,
              seed=42, dit_mode="stream")
im.save("fox.png")
```

Images are RGB. Use multiples of 32 from 256 to 4096 for each dimension. Pass up to 16 references in order using repeated `--reference` or the `references` argument. The original desktop pipeline completed single- and two-reference edits on the M5 Pro; its 16-reference run reached denoising step three before cancellation.

## Hardware and memory

`--backend auto` selects the W8A8 TensorOps kernel on M5/M6 and the experimental portable path on earlier Apple GPUs. M5 Pro 48 GB is the physically tested hardware. M1–M4 and M6 need on-device validation. Explicit choices are `w8a8` and `portable`.

`--dit-mode stream` reads DiT weights one layer at a time; `resident` keeps them between denoising steps. The default `auto` streams when using the portable backend, on machines with less than 32 GiB, or above 8192 packed sequence rows. This selection heuristic does not establish minimum supported machine memory.

Existing installations with the four weight and tokenizer files can run `fetch` once to verify them and add the 1,826-byte VAE configuration.

## Attribution

This project’s own runtime code is licensed under Apache-2.0. The model and third-party kernels retain their respective licenses.

MiniMax developed [H3 Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image). [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) provides the pinned quantized weights. The VAE configuration downloads separately from the official [VAE config](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/492fd8c4eca666f1085f57e30738c1e0d86b3439/vae/config.json). The upstream model agreement and attribution are included in the package. The three W8A8 Metal source files come from [mlx-serve](https://github.com/ddalcu/mlx-serve/tree/9191fbdf697df333cac19ec7a93eaa1098ce3ad5/src/kernels) with their MIT license.
