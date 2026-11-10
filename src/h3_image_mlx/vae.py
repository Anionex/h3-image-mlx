import math

import numpy as np
from .core import Weights, mx, norm, rope, swiglu
from PIL import Image


def tile_positions(size):
    if size <= 16:
        return [0], size
    count = math.ceil((size - 4) / 12)
    extra = 16 * count - 4 * (count - 1) - size
    gaps = [
        12 - extra // (count - 1) - (i < extra % (count - 1)) for i in range(count - 1)
    ]
    return [0, *np.cumsum(gaps).tolist()], 16


def decode_tiles(w, z):
    batch, h, width, channels = z.shape
    z = z.reshape(batch, -1, channels)
    x = z @ w["post_quant_conv.weight"][:, :, 0, 0].T + w["post_quant_conv.bias"]
    x = w.linear("decoder.x_embedder", x)
    x = mx.concatenate(
        [
            x,
            mx.broadcast_to(w["decoder.register_tokens"], (batch, 4, 1024)),
            mx.zeros_like(x[:, :1]),
        ],
        axis=1,
    )
    coords = np.meshgrid(
        (np.arange(h) + 0.5) * 2 / h - 1,
        (np.arange(width) + 0.5) * 2 / width - 1,
        indexing="ij",
    )
    positions = np.stack(coords, axis=-1).reshape(-1, 2)
    positions = np.concatenate([positions, np.zeros((5, 2))])
    ang = (2 * np.pi * positions[:, :, None] * 100 ** (-np.arange(8) / 8)).reshape(
        -1, 16
    )
    angles = mx.array(ang.astype(np.float32))
    for i in range(24):
        b = f"decoder.transformer_blocks.{i}"
        qkv = w.linear(b + ".attn.to_qkv", norm(x, w[b + ".norm1.weight"])).reshape(
            batch, x.shape[1], 16, 192
        )
        q, k, v = mx.split(qkv.transpose(0, 2, 1, 3), 3, axis=-1)
        q, k = rope(norm(q), angles), rope(norm(k), angles)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
        x = (
            x
            + w.linear(b + ".attn.to_out", out.transpose(0, 2, 1, 3).reshape(x.shape))
            * w[b + ".scale1"]
        )
        out = swiglu(w.linear(b + ".ff.w1", norm(x, w[b + ".norm2.weight"])))
        x = x + w.linear(b + ".ff.w2", out) * w[b + ".scale2"]
        mx.eval(x)
    x = mx.fast.layer_norm(
        x[:, : h * width],
        w["decoder.norm_out.weight"],
        w["decoder.norm_out.bias"],
        1e-5,
    )
    out = w.linear("decoder.proj_out", x)
    return (
        out.reshape(batch, h, width, 3, 16, 16)
        .transpose(0, 1, 4, 2, 5, 3)
        .reshape(batch, h * 16, width * 16, 3)
    )


def decode(path, latent, stats):
    w = Weights(path, True)
    z = latent.transpose(1, 2, 0) * mx.array(stats["latents_std"]) + mx.array(
        stats["latents_mean"]
    )
    z = z.astype(mx.float16)
    yy, th = tile_positions(z.shape[0])
    xx, tw = tile_positions(z.shape[1])
    canvas = np.zeros((z.shape[0] * 16, z.shape[1] * 16, 3), np.float32)
    for i, y in enumerate(yy):
        strip = np.zeros((th * 16, canvas.shape[1], 3), np.float32)
        for start in range(0, len(xx), 4):
            xs = xx[start : start + 4]
            group = mx.stack([z[y : y + th, x : x + tw] for x in xs])
            tiles = np.asarray(decode_tiles(w, group).astype(mx.float32))
            for off, (tile, x) in enumerate(zip(tiles, xs)):
                j = start + off
                tile = tile.copy()
                if i:
                    overlap = (yy[i - 1] + th - y) * 16
                    alpha = (
                        np.arange(overlap, dtype=np.float32)[:, None, None] / overlap
                    )
                    old = canvas[16 * y : 16 * y + overlap, 16 * x : 16 * (x + tw)]
                    tile[:overlap] = old + (tile[:overlap] - old) * alpha
                if j:
                    overlap = (xx[j - 1] + tw - x) * 16
                    alpha = (
                        np.arange(overlap, dtype=np.float32)[None, :, None] / overlap
                    )
                    old = strip[:, 16 * x : 16 * x + overlap]
                    tile[:, :overlap] = old + (tile[:, :overlap] - old) * alpha
                strip[:, 16 * x : 16 * (x + tw)] = tile
        canvas[16 * y : 16 * (y + th)] = strip
    rgb = canvas * np.array([0.229, 0.224, 0.225], np.float32) + np.array(
        [0.485, 0.456, 0.406], np.float32
    )
    if not np.isfinite(rgb).all():
        raise ArithmeticError("Non-finite VAE output")
    return Image.fromarray((rgb.clip(0, 1) * 255).round().astype(np.uint8))
