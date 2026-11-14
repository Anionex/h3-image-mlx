"""Evaluate H3's quantized projections with standard MLX operations."""

import mlx.core as mx

# An int8 dot product of length 1024 has an absolute sum <= 2**24, so its
# integer partial sums are exactly representable in float32. H3's longest
# projection has K=25600, safely within int32 for the combined sum.
_DOT_CHUNK = 1024
_TILE = 1024
_CHUNKS_PER_BATCH = 4


def portable_matmul(xq, weight, xs, scale, output_dtype):
    """Compute row-scaled int8 dot products without the M5 Metal TensorOps kernel."""
    rows, width = xq.shape
    channels = weight.shape[0]
    row_scale = xs.reshape(rows, 1).astype(mx.float32)
    channel_scale = mx.broadcast_to(scale.reshape(-1), (channels,)).astype(mx.float32)

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
