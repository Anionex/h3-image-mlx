"""H3 Image inference primitives. Metal GEMM from ddalcu/mlx-serve @ 9191fbdf."""

import fcntl
import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from functools import cache
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent
KERNELS = ROOT / "kernels"
backend = ContextVar("h3_backend", default="portable")


@cache
def tensorops_kernel():
    return mx.fast.metal_kernel(
        name="msv_w8a8_gemm",
        input_names=["xq", "wq", "xs", "ws"],
        output_names=["y"],
        header=(KERNELS / "mpp_tensor_ops.metal").read_text(),
        source=(KERNELS / "w8a8_gemm_body.metal").read_text()
        + "\n"
        + (KERNELS / "w8a8_epi_plain.metal").read_text(),
    )


H4 = np.ones((4, 4), dtype=np.float32)
np.fill_diagonal(H4[:, ::-1], -1)
H256 = mx.array(np.kron(np.kron(H4, H4), np.kron(H4, H4)) / 16)
DTYPES = {"I8": "i1", "U8": "u1", "F16": "<f2", "BF16": "<u2", "F32": "<f4"}


def flag(name):
    """Opt-in tuning knob read from the environment; unset or invalid means off."""
    try:
        return float(os.environ.get(name) or 0)
    except ValueError:
        return 0.0


@mx.compile
def rotate_quantize(x):
    rotated = (x.astype(mx.float32).reshape(-1, 256) @ H256).reshape(x.shape)
    scale = mx.maximum(mx.max(mx.abs(rotated), axis=-1) / 127, 1e-20)
    return mx.clip(mx.round(rotated / scale[:, None]), -127, 127).astype(mx.int8), scale


def w8a8(x, weight, scale):
    rows = x.shape[0]
    n = weight.shape[0]
    xq, xs = rotate_quantize(x)
    if backend.get() == "portable":
        from ._portable import portable_matmul
        return portable_matmul(xq, weight, xs, scale.reshape(-1), x.dtype)
    bm = 128 if rows < 8192 else 64
    bn = 8192 // bm
    (y,) = tensorops_kernel()(
        inputs=[xq, weight, xs, scale.reshape(-1)],
        template=[("BM", bm), ("BN", bn), ("T", x.dtype)],
        grid=((n + bn - 1) // bn * 128, (rows + bm - 1) // bm, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )
    return y


class Weights:
    """Read individual tensors so a streaming stage can release each block."""

    def __init__(self, path, resident=False):
        self.file = open(path, "rb")
        (length,) = struct.unpack("<Q", self.file.read(8))
        self.start = 8 + length
        self.header = json.loads(self.file.read(length))
        self.resident = resident
        self.cache = {}
        self.int8_calls = 0
        self.validated_projections = set()
        # Opt-in knobs. None of them changes a single weight byte or the op graph.
        self.pool = ThreadPoolExecutor(1) if flag("H3_PREFETCH") else None
        self.pending = {}
        self.stats = dict(read_s=0.0, read_bytes=0, convert_s=0.0, wait_s=0.0)
    def release(self):
        self.cache.clear()

    def ahead(self, group, keep=()):
        """Start reading one block's tensors on a worker thread (raw bytes only)."""
        for name in [g for g in self.pending if g != group and g not in keep]:
            del self.pending[name]
        if not self.pool or group in self.pending:
            return
        spans = [
            (key, *spec["data_offsets"])
            for key, spec in self.header.items()
            if key.startswith(group + ".")
            and key not in self.cache
            and "adaln_proj" not in key
            and not key.endswith((".comfy_quant", ".config"))
        ]
        fd, start = self.file.fileno(), self.start

        def job():
            t = time.perf_counter()
            out = {key: os.pread(fd, b - a, start + a) for key, a, b in spans}
            return out, time.perf_counter() - t

        self.pending[group] = self.pool.submit(job)

    def _read(self, key, row=None):
        spec = self.header[key]
        a, b = spec["data_offsets"]
        shape = spec["shape"]
        if row is not None:
            stride = (b - a) // shape[0]
            a += int(row) * stride
            b = a + stride
            shape = shape[1:]
        buf = None
        t = time.perf_counter()
        future = row is None and self.pending.get(".".join(key.split(".")[:2]))
        if future:
            fetched, seconds = future.result()
            buf = fetched.pop(key, None)
            if "counted" not in fetched:
                fetched["counted"] = True
                self.stats["read_s"] += seconds
            self.stats["wait_s"] += time.perf_counter() - t
        if buf is None:
            buf = os.pread(self.file.fileno(), b - a, self.start + a)
            self.stats["read_s"] += time.perf_counter() - t
        self.stats["read_bytes"] += len(buf)
        t = time.perf_counter()
        data = mx.array(np.frombuffer(buf, DTYPES[spec["dtype"]]).reshape(shape))
        if spec["dtype"] == "BF16":
            data = data.view(mx.bfloat16)
        self.stats["convert_s"] += time.perf_counter() - t
        return data

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
        x = self._read(key)
        if self.resident:
            self.cache[key] = x
        return x

    def linear(self, prefix, x):
        w = self[prefix + ".weight"]
        if w.dtype == mx.int8:
            if prefix not in self.validated_projections:
                a, b = self.header[prefix + ".comfy_quant"]["data_offsets"]
                metadata = json.loads(
                    os.pread(self.file.fileno(), b - a, self.start + a)
                )
                if metadata != {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": 256,
                }:
                    raise ValueError(f"Unsupported quantization: {prefix}: {metadata}")
                self.validated_projections.add(prefix)
            self.int8_calls += 1
            y = w8a8(x, w, self[prefix + ".weight_scale"])
        else:
            y = x @ w.astype(x.dtype).T
        if prefix + ".bias" in self.header:
            y = y + self[prefix + ".bias"].astype(y.dtype)
        return y


def norm(x, weight=None, eps=1e-5):
    return mx.fast.rms_norm(x, None if weight is None else weight.astype(x.dtype), eps)


def heads(x, head_dim):
    return x.reshape(1, x.shape[0], -1, head_dim).transpose(0, 2, 1, 3)


def attention(q, k, v, mask=None):
    out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=q.shape[-1] ** -0.5, mask=mask
    )
    return out.transpose(0, 2, 1, 3).reshape(q.shape[2], -1)


def rope(x, angles):
    n = angles.shape[-1]
    c, s = mx.cos(angles).astype(x.dtype), mx.sin(angles).astype(x.dtype)
    a, b = x[..., :n], x[..., n : 2 * n]
    return mx.concatenate([a * c - b * s, a * s + b * c, x[..., 2 * n :]], axis=-1)


def swiglu(x):
    gate, up = mx.split(x, 2, axis=-1)
    return (gate * mx.sigmoid(gate)) * up
