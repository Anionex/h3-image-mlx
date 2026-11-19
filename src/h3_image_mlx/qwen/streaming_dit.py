"""Run the Qwen 2.1 Turbo denoiser with shared disk-backed block execution."""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .._stage import emit, process_usage
from ..streaming import BlockStream, StreamPolicy, TensorStore


def _upstream():
    try:
        from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
        from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer_block import Qwen21TransformerBlock
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"The streamed Qwen DiT needs the MFLUX extra ({exc.name} is missing)") from exc
    return Qwen21Transformer, Qwen21TransformerBlock


def _config(models):
    config = json.loads((models / "transformer/config.json").read_text())
    if not config.get("causal_condition") or config.get("patch_size") != 1:
        raise ValueError("Expected the Qwen 2.1 Turbo single-stream, causal DiT")
    dimensions = (
        "in_channels", "out_channels", "attention_head_dim", "num_attention_heads",
        "context_in_dim", "mlp_ratio", "axes_dims_rope", "eps",
    )
    return {key: config[key] for key in dimensions}, config["num_layers"]


def _load_roots(transformer, store):
    # The packing tool uses a numeric child name for the modulation projection.
    mapping = {
        key: key.replace("modulation.layers.1.", "modulation.1.", 1)
        for key, _ in tree_flatten(transformer.parameters())
    }
    weights = store.arrays(store.read(mapping.values()))
    transformer.load_weights([(key, weights[source]) for key, source in mapping.items()])
    return store.size(mapping.values())


def _denoise(models, store, text, width, height, seed, policy):
    Transformer, Block = _upstream()
    config, n_blocks = _config(models)
    transformer = Transformer(num_layers=0, **config)
    root_weight_bytes = _load_roots(transformer, store)

    block = Block(
        dim=config["attention_head_dim"] * config["num_attention_heads"],
        attention_head_dim=config["attention_head_dim"],
        num_attention_heads=config["num_attention_heads"],
        mlp_ratio=config["mlp_ratio"],
        eps=config["eps"],
    )
    # The ddalcu pack stores affine Q8 layer weights, scales and biases separately.
    nn.quantize(block, group_size=64, bits=8)
    unloaded = tree_flatten(block.parameters())
    if not any(key.endswith(".scales") for key, _ in unloaded):
        raise ValueError("Expected affine Q8 Qwen DiT projections")

    rows, cols = height // 16, width // 16
    text_length = text.shape[1]
    cos, sin = transformer.pos_embed(text_length, rows, cols)
    image_rope = (cos[text_length:], sin[text_length:])
    text_rope = (cos[:text_length], sin[:text_length])
    text_hidden = transformer.txt_in(text)
    mx.eval(text_hidden)
    transformer.txt_in = None
    shape = (1, rows * cols, config["in_channels"])
    image_hidden = mx.random.normal(shape, dtype=mx.float32, key=mx.random.key(seed)).astype(mx.bfloat16)
    grid = json.loads((models / "model_index.json").read_text())["sample_sigmas"]
    if len(grid) != 8 or any(not 0 < value <= 1 for value in grid) or any(a <= b for a, b in zip(grid, grid[1:])):
        raise ValueError("Expected the checkpoint's descending eight-step Turbo sigma schedule")
    sigmas = np.asarray([*grid, 0], dtype=np.float32)

    # These projections share the same weights at all steps. Keep the original
    # two-row (target, text) input per projection, then release the root weights.
    modulations = []
    for sigma in sigmas[:-1]:
        temb = transformer.time_text_embed(mx.array([float(sigma), 0.0], dtype=mx.float32))
        modulation = transformer.modulation(temb)
        scale = transformer.norm_out.linear(nn.silu(temb))[0][None, None, :]
        mx.eval(modulation, scale)
        modulations.append((modulation, scale))
    transformer.time_text_embed = None
    transformer.modulation = None
    transformer.norm_out.linear = None
    mx.clear_cache()

    text_cache = []

    with BlockStream(store, policy) as stream:
        for index in range(n_blocks):
            stream.register(index, {
                key: f"transformer_blocks.{index}.{key}" for key, _ in unloaded
            })

        for step, (sigma, following) in enumerate(zip(sigmas, sigmas[1:])):
            modulation, scale = modulations[step]
            mod1, mod2 = mx.split(modulation, 2, axis=-1)
            text_mod = (mod1[1][None, None, :], mod2[1][None, None, :])
            image_mod = (mod1[0][None, None, :], mod2[0][None, None, :])
            x = transformer.img_in(image_hidden)
            for index in range(n_blocks):
                def forward(weights):
                    block.load_weights(list(weights.items()))
                    try:
                        if step == 0:
                            next_text, key, value = block.text_forward(text_hidden, *text_mod, *text_rope)
                            next_image = block.image_forward(x, key, value, *image_mod, *image_rope)
                            return next_image, next_text, key, value
                        key, value = text_cache[index]
                        return block.image_forward(x, key, value, *image_mod, *image_rope)
                    finally:
                        # Lazy graph inputs retain the bound parameters through GPU evaluation.
                        block.load_weights(unloaded)

                if step == 0:
                    x, text_hidden, key, value = stream.run(index, forward)
                    text_cache.append((key, value))
                else:
                    x = stream.run(index, forward)

            velocity = transformer.proj_out(transformer.norm_out(x, scale))
            dt = float(following - sigma)
            image_hidden = (image_hidden.astype(mx.float32) + velocity.astype(mx.float32) * dt).astype(mx.bfloat16)
            mx.eval(image_hidden)
            text_hidden = None
            emit("PROGRESS", stage="Denoising", step=step + 1, total=len(grid))

    latent = image_hidden.reshape(1, rows, cols, -1).transpose(0, 3, 1, 2)
    return latent, dict(stream.metrics, root_weight_bytes=root_weight_bytes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path)
    parser.add_argument("text", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--uncached", action="store_true")
    args = parser.parse_args(argv)

    mx.set_cache_limit(96 << 20)
    paths = sorted((args.models / "transformer").glob("*.safetensors"))
    text = mx.load(str(args.text))["value"]
    policy = StreamPolicy(overlap=args.overlap, prefetch_bytes=512 << 20)
    with TensorStore(paths, uncached=args.uncached) as store:
        latent, metrics = _denoise(args.models, store, text, args.width, args.height, args.seed, policy)
        if not mx.all(mx.isfinite(latent)).item():
            raise RuntimeError("Qwen DiT produced non-finite latents")
        mx.save_safetensors(str(args.output), {"value": latent})
        weight_io = dict(store.io)
    mx.synchronize()
    emit("METRICS", backend="mlx-q8", mlx_peak_bytes=mx.get_peak_memory(),
         stream=metrics, weight_io=weight_io, **process_usage())
    return 0


if __name__ == "__main__":
    sys.exit(main())
