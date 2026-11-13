"""Text-to-image forward on the six-step H3 Image grid."""

import itertools
import json
import os
import time

import numpy as np
from .core import Weights, attention, flag, heads, mx, norm, rope, swiglu


def qkv_attention(w, prefix, x, angles=None):
    q, k, v = [
        heads(t, 128) for t in mx.split(w.linear(prefix + ".qkv_proj", x), 3, -1)
    ]
    q = norm(q, w[prefix + ".q_norm.weight"])
    k = norm(k, w[prefix + ".k_norm.weight"])
    if angles is not None:
        q, k = rope(q, angles), rope(k, angles)
    return w.linear(prefix + ".out_proj", attention(q, k, v))


def mlp(w, prefix, x):
    return w.linear(prefix + ".fc2", swiglu(w.linear(prefix + ".fc1", x)))


def prepare_text(w, embedding):
    x = w.linear("condition_proj", embedding.astype(mx.bfloat16))
    for i in range(2):
        p = f"token_refiner.blocks.{i}"
        x = x + qkv_attention(w, p + ".attn", norm(x, w[p + ".norm1.weight"]))
        x = x + mlp(w, p + ".mlp", norm(x, w[p + ".norm2.weight"]))
    return norm(x, w["token_refiner.final_norm.weight"])


def image_positions(h, width, cursor):
    area = (h * width) ** 0.5

    def axis(n):
        return (np.arange(n) * (n / area / n) + (1 - n / area) / 2) * 32

    yy, xx = np.meshgrid(axis(h), axis(width), indexing="ij")
    return np.stack([np.full(h * width, cursor), yy.flatten(), xx.flatten()], axis=-1)


def angles_for(w, ntext, grids):
    text = np.zeros((ntext, 3), np.float32)
    text[:, 0] = np.arange(ntext)
    positions = [text] + [
        image_positions(h, width, ntext + i) for i, (h, width) in enumerate(grids)
    ]
    pos = mx.array(np.concatenate(positions).astype(np.float32))
    return (pos[:, :, None] * w["rope.inv_freq"]).reshape(pos.shape[0], -1)


def curve_coordinates(w, sigmas):
    t = np.clip(1 - np.asarray(sigmas, dtype=np.float32), 0, 1)
    table = np.asarray(w["adaln_t_table"])
    pos = t * (table.shape[0] - 1)
    idx = np.minimum(pos.astype(np.int32), len(table) - 2)
    return (
        table[idx]
        + (table[idx + 1] - table[idx]) * (pos - idx.astype(np.float32))[:, None]
    )


def precompute(w, prefix, coords, modalities, parts):
    p = prefix + ".adaln_proj.linear"
    # Eight basis coordinates per time. CPU float32 preserves the small dot product.
    weights = np.asarray(w[p + ".weight"].astype(mx.float32))
    bias = np.asarray(w[p + ".bias"].astype(mx.float32))
    mod = coords @ weights.T + bias
    return mx.array(mod.reshape(len(coords), modalities, parts, -1))


def modulate(x, table, segments, scale=False):
    buffers = []
    for start, stop, row in segments:
        value = table[row].astype(x.dtype)
        buffers.append(x[start:stop] * (1 + value if scale else value))
    return mx.concatenate(buffers)


def shifted(x, scale, shift, segments):
    return mx.concatenate(
        [
            x[start:stop] * (1 + scale[row].astype(x.dtype))
            + shift[row].astype(x.dtype)
            for start, stop, row in segments
        ]
    )


def head_mix(grid, sigma, sigma_next):
    """Which PDD heads a step from sigma to sigma_next uses, as (head index, weight) offsets on head 0.

    Mirrors ComfyUI's _pdd_head: a step that spans several intervals of the checkpoint's own
    sigma grid takes the dt-weighted mean of their heads. On the stock grid this is one head.
    """
    n = len(grid) - 1
    start, stop = (min(range(n + 1), key=lambda k: abs(grid[k] - s)) for s in (sigma, sigma_next))
    start = min(start, n - 1)
    stop = max(stop, start + 1)
    dt = [grid[k] - grid[k + 1] for k in range(start, stop)]
    return [(k, dt[k - start] / sum(dt)) for k in range(max(start, 1), stop)]


def patchify(latent):
    c, h, width = latent.shape
    return (
        latent.reshape(c, h // 2, 2, width // 2, 2)
        .transpose(1, 3, 0, 2, 4)
        .reshape(-1, c * 4)
    )


def denoise(
    path, text, width, height, seed, resident=True, refs=(), tags=None, progress=None,
    precision=None,
):
    if min(width, height) < 32 or width % 32 or height % 32:
        raise ValueError("Image dimensions must be positive multiples of 32")
    began = time.perf_counter()
    w = Weights(path, resident)
    if precision == "bf16":
        for key, spec in w.header.items():
            if key.endswith(".weight") and spec["dtype"] not in ("BF16", "F16", "F32"):
                if w.pool:
                    w.pool.shutdown()
                w.file.close()
                raise ValueError(f"BF16 requires floating-point weights; {path}: {key} is {spec['dtype']}. Check H3_DIT.")
    # H3_ASYNC: queue each block on the GPU and read the next block's weights meanwhile.
    pipelined, profile = bool(flag("H3_ASYNC")), dict(steps=[], eval_s=0.0)
    config = json.loads(w.header["__metadata__"]["config"])["transformer"]
    grid = sigmas = config["pdd_sigmas"]
    # Opt-in few-step mode: H3_LORA=<file> beside the quantized weights, H3_STEPS on a shifted uniform grid.
    if os.environ.get("H3_LORA"):
        w.lora = Weights(os.environ["H3_LORA"])
    steps = int(flag("H3_STEPS"))
    if steps:
        shift = flag("H3_SHIFT") or 12.0
        sigmas = [shift * t / (1 + (shift - 1) * t) for t in (1 - k / steps for k in range(steps + 1))]
    coords = curve_coordinates(w, sigmas[:-1])
    if refs:
        coords = np.concatenate([coords, curve_coordinates(w, [0.001])])
    blocks = [f"blocks.{i}" for i in range(50)]
    mods = [precompute(w, b, coords, 3, 6) for b in blocks]
    final_mod = precompute(w, "final_layer", coords, 1, 2)
    text = prepare_text(w, text)
    mx.eval(text, final_mod, *mods)
    # The text refiner and compressed AdaLN projections are used only during setup.
    w.release()
    mx.clear_cache()
    rows, cols = height // 32, width // 32
    ntext = text.shape[0]
    grids = [(z.shape[1] // 2, z.shape[2] // 2) for z in refs]
    angles = angles_for(w, ntext, [*grids, (rows, cols)])
    fixed = [text]
    text_tags = np.asarray(tags) if tags is not None else np.ones(ntext, int)
    splits = np.r_[0, np.flatnonzero(np.diff(text_tags)) + 1, ntext]
    segments = [(a, b, int(text_tags[a])) for a, b in itertools.pairwise(splits)]
    target_start = ntext
    for ref in refs:
        r = patchify(ref)
        aug_noise = np.random.default_rng(seed).standard_normal(
            r.shape, dtype=np.float32
        )
        r = r * 0.999 + mx.array(aug_noise) * 0.001
        fixed.append(w.linear("video_patch_proj", r).astype(text.dtype))
        segments.append((target_start, target_start + r.shape[0], 3))
        target_start += r.shape[0]
    segments.append((target_start, target_start + rows * cols, 0))
    prefix = mx.concatenate(fixed)
    mx.eval(prefix)
    rng = np.random.default_rng(seed)
    latent = mx.array(rng.standard_normal((24, rows * 2, cols * 2), dtype=np.float32))
    patches = (
        latent.reshape(24, rows, 2, cols, 2)
        .transpose(1, 3, 0, 2, 4)
        .reshape(rows * cols, 96)
    )
    profile["setup_s"] = time.perf_counter() - began
    last = len(sigmas) - 2
    for step, (sigma, next_sigma) in enumerate(itertools.pairwise(sigmas)):
        start = time.monotonic()
        previous = None
        x = mx.concatenate(
            [prefix, w.linear("video_patch_proj", patches).astype(text.dtype)]
        )
        for i, b in enumerate(blocks):
            if i < 49 or step < last:
                w.ahead(blocks[(i + 1) % 50], keep=(b,))
            tab = mods[i][step]
            if refs:
                tab = mx.concatenate([tab, mods[i][-1, :1]])
            h = shifted(norm(x, w[b + ".norm1.weight"]), tab[:, 1], tab[:, 0], segments)
            x = x + modulate(
                qkv_attention(w, b + ".attn", h, angles), tab[:, 2], segments
            )
            h = shifted(norm(x, w[b + ".norm2.weight"]), tab[:, 4], tab[:, 3], segments)
            x = x + modulate(mlp(w, b + ".mlp", h), tab[:, 5], segments)
            t = time.perf_counter()
            if pipelined:
                # At most two blocks of weights are alive: one running, one queued.
                mx.async_eval(x)
                if previous is not None:
                    mx.eval(previous)
                previous = x
            else:
                mx.eval(x)
            profile["eval_s"] += time.perf_counter() - t
        shift, scale = final_mod[step, 0]
        x = (
            norm(x[target_start:], w["final_layer.norm.weight"]).astype(mx.float32)
            * (1 + scale)
            + shift
        )
        head = w["final_layer.video_out.weight"].reshape(6, 96, -1)
        bias = w["final_layer.video_out.bias"].reshape(6, 96)
        ww, bb = head[0], bias[0]
        if steps:
            for k, part in head_mix(grid, sigma, next_sigma):
                ww, bb = ww + part * head[k], bb + part * bias[k]
        elif step:
            ww, bb = ww + head[step], bb + bias[step]
        patches = patches + (sigma - next_sigma) * (x @ ww.T + bb)
        mx.eval(patches)
        profile["steps"].append(time.monotonic() - start)
        if progress:
            progress(step + 1, time.monotonic() - start)
    latent = (
        patches.reshape(rows, cols, 24, 2, 2)
        .transpose(2, 0, 3, 1, 4)
        .reshape(24, rows * 2, cols * 2)
    )
    if os.environ.get("H3_PROFILE"):
        profile.update(w.stats, total_s=time.perf_counter() - began, tokens=int(x.shape[0]) + target_start,
                       peak_mlx_gb=mx.get_peak_memory() / 1e9)
        with open(os.environ["H3_PROFILE"], "w") as out:
            json.dump(profile, out)
    return latent, w.int8_calls
