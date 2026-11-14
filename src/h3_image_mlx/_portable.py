"""Evaluate H3's quantized projections on Apple GPUs and the MLX CPU backend."""

from functools import cache
from pathlib import Path

import mlx.core as mx
import numpy as np

# An int8 dot product of length 1024 has an absolute sum <= 2**24, so its
# integer partial sums are exactly representable in float32. H3's longest
# projection has K=25600, safely within int32 for the combined sum.
_DOT_CHUNK = 1024
_TILE = 1024
_CHUNKS_PER_BATCH = 4


@cache
def _simd_gemm():
    return mx.fast.metal_kernel(
        name="h3_portable_simd",
        input_names=["x", "w", "s", "z"],
        output_names=["y"],
        header="#include <metal_simdgroup_matrix>\n",
        source=Path(__file__).with_name("kernels").joinpath("portable_simd.metal").read_text(),
    )


def portable_matmul(xq, weight, xs, scale, output_dtype):
    """Compute row-scaled int8 dot products without the M5 Metal TensorOps kernel."""
    rows, width = xq.shape
    channels = weight.shape[0]
    row_scale = xs.reshape(rows, 1).astype(mx.float32)
    channel_scale = mx.broadcast_to(scale.reshape(-1), (channels,)).astype(mx.float32)

    aligned = width % 64 == 0 and channels % 64 == 0
    if rows <= 512 and aligned and mx.default_device() == mx.Device(mx.gpu):
        bm, bn, bk = (32, 64, 64) if rows <= 128 else (64, 64, 32)
        return _simd_gemm()(
            inputs=[xq, weight, row_scale, channel_scale],
            template=[("T", output_dtype), ("BM", bm), ("BN", bn), ("BK", bk)],
            grid=(channels // bn * 128, (rows + bm - 1) // bm, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(rows, channels)],
            output_dtypes=[output_dtype],
        )[0]

    if 512 < rows <= 1536 and aligned and mx.default_device() == mx.Device(mx.gpu):
        return wide_matmul(xq, weight, row_scale, channel_scale, output_dtype)

    row_tiles = []
    for r in range(0, rows, _TILE):
        columns = []
        row_slice = slice(r, r + _TILE)
        for c in range(0, channels, _TILE):
            col_slice = slice(c, c + _TILE)
            # Batch exact partial dots along K. Limit the number of temporary
            # result planes, even for H3's 25,600-wide text projections.
            a = xq[row_slice]
            b = weight[col_slice]
            full_width = width - width % _DOT_CHUNK
            batch_width = _CHUNKS_PER_BATCH * _DOT_CHUNK
            total = None
            for k in range(0, full_width, batch_width):
                inner = slice(k, min(k + batch_width, full_width))
                av = a[:, inner].reshape(a.shape[0], -1, _DOT_CHUNK).transpose(1, 0, 2)
                bv = b[:, inner].reshape(b.shape[0], -1, _DOT_CHUNK).transpose(1, 0, 2)
                dots = av.astype(mx.float32) @ bv.astype(mx.float32).swapaxes(1, 2)
                partial = dots.astype(mx.int32).sum(axis=0)
                total = partial if total is None else total + partial
            if full_width < width:
                tail = a[:, full_width:].astype(mx.float32) @ b[:, full_width:].astype(mx.float32).T
                partial = tail.astype(mx.int32)
                total = partial if total is None else total + partial
            # Evaluate before advancing to the next tile: no dense floating
            # point copy of an entire large weight is retained across tiles.
            y = total.astype(mx.float32) * row_scale[row_slice] * channel_scale[col_slice]
            y = y.astype(output_dtype)
            mx.eval(y)
            columns.append(y)
        row = mx.concatenate(columns, axis=1) if len(columns) > 1 else columns[0]
        mx.eval(row)
        row_tiles.append(row)
    return mx.concatenate(row_tiles, axis=0) if len(row_tiles) > 1 else row_tiles[0]


def wide_matmul(xq, weight, row_scale, channel_scale, output_dtype):
    rows, width = xq.shape
    channels = weight.shape[0]
    # Group by the actual activation magnitudes. Every sub-dot has an
    # absolute sum <= 2**24 for every possible signed 8-bit weight.
    x = xq.astype(mx.int32)
    group = 1024
    pad = (-width) % group
    abs_x = mx.abs(x)
    if pad:
        abs_x = mx.pad(abs_x, [(0, 0), (0, pad)])
    norms = np.array(abs_x.reshape(rows, -1, group).sum(2).max(0))
    spans = []
    start = total = 0
    for i, norm in enumerate(norms):
        if total + int(norm) > (1 << 17):
            spans.append((start, i * group))
            start, total = i * group, 0
        total += int(norm)
    spans.append((start, width))
    a = xq.astype(mx.float32)
    columns = []
    for c in range(0, channels, 2048):
        b = weight[c:c+2048]
        accum = None
        for begin, end in spans:
            dot = a[:,begin:end] @ b[:,begin:end].astype(mx.float32).T
            if len(spans) == 1:
                accum = dot
            else:
                partial = dot.astype(mx.int32)
                accum = partial if accum is None else accum + partial
        y = (accum.astype(mx.float32) * row_scale * channel_scale[c:c+2048]).astype(output_dtype)
        mx.eval(y)
        columns.append(y)
    return mx.concatenate(columns,1)
