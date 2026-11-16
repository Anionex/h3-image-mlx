"""Portable projection checks; set H3_IMAGE_TEST_WEIGHTS to a local model directory."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from h3_image_mlx import _portable


def first_rows(path, key, count):
    """Read complete dot products from the first output channels of one weight."""
    with path.open("rb") as f:
        header_bytes = int.from_bytes(f.read(8), "little")
        table = json.loads(f.read(header_bytes))
        base = f.tell()
        spec = table[key]
        dtype = {"I8": "i1", "F32": "<f4", "F16": "<f2"}[spec["dtype"]]
        rows, width = spec["shape"]
        assert count <= rows
        f.seek(base + spec["data_offsets"][0])
        return np.frombuffer(f.read(count * width * np.dtype(dtype).itemsize), dtype).reshape(count, width)


class PortableProjectionTests(unittest.TestCase):
    def check_outputs(self, x, w, xs, scale, output_dtype):
        result = _portable.portable_matmul(mx.array(x), mx.array(w), mx.array(xs), mx.array(scale), output_dtype)
        self.assertEqual(result.dtype, output_dtype)
        # int64 provides an independent integer oracle, even across 1024-wide tiles.
        dot = x.astype(np.int64) @ w.astype(np.int64).T
        scaled = dot.astype(np.float32) * xs.reshape(-1, 1) * scale.reshape(-1)
        expected = mx.array(scaled).astype(output_dtype)
        np.testing.assert_array_equal(np.array(result.astype(mx.float32)), np.array(expected.astype(mx.float32)))

    def test_output_tiles_odd_shapes_and_scales(self):
        rng = np.random.default_rng(42)
        x = rng.integers(-127, 128, (9, 2053), dtype=np.int8)
        w = rng.integers(-128, 128, (17, 2053), dtype=np.int8)
        xs = np.linspace(.001, .017, len(x), dtype=np.float32)
        x[0] = 0
        with patch.object(_portable, "_TILE", 5):
            self.check_outputs(x, w, xs, np.array([.004], np.float32), mx.float32)
            self.check_outputs(x, w, xs, np.linspace(.001, .016, len(w), dtype=np.float32)[:, None], mx.bfloat16)

    def test_batch_and_tail_boundaries(self):
        rng = np.random.default_rng(91)
        batch = _portable._DOT_CHUNK * _portable._CHUNKS_PER_BATCH
        for width in (17, batch - 1, batch, batch + 1):
            with self.subTest(width=width):
                x = rng.integers(-128, 128, (3, width), dtype=np.int8)
                w = rng.integers(-128, 128, (7, width), dtype=np.int8)
                self.check_outputs(x, w, np.ones(3, np.float32), np.ones(7, np.float32), mx.float32)

    def test_worst_case_dot_and_cancellation(self):
        # Positive partials exceed the float32 exact-integer range before
        # cancellation leaves a small, odd integer in the first output.
        x = np.full((3, 25600), 127, dtype=np.int8)
        w = np.full((7, 25600), 127, dtype=np.int8)
        w[0,12800:] = -127
        w[0,0] = 126
        w[1,::2] = -128
        self.check_outputs(x, w, np.ones(3, np.float32), np.ones(7, np.float32), mx.float32)

    @unittest.skipUnless(mx.metal.is_available(), "requires Metal")
    def test_simd_gemm_exact_scales_edges_and_cancellation(self):
        rng = np.random.default_rng(19)
        for rows, width in ((33, 1088), (129, 25600)):
            with self.subTest(rows=rows):
                x = rng.integers(-128, 128, (rows, width), dtype=np.int8)
                w = rng.integers(-128, 128, (64, width), dtype=np.int8)
                x[0] = -128
                w[0] = -128
                w[1, :width // 2] = 127
                w[1, width // 2:] = -127
                w[1, 0] = 126
                xs = rng.uniform(.001, .02, rows).astype(np.float32)
                for dtype in (mx.float32, mx.bfloat16):
                    self.check_outputs(x, w, xs, np.array([.013], np.float32), dtype)

    def test_simd_reads_byte_aligned_dlpack_weight(self):
        if not mx.metal.is_available():
            self.skipTest("requires Metal")
        rng = np.random.default_rng(412)
        width, channels = 1088, 64
        buf = rng.integers(-128, 128, channels * width + 1, dtype=np.int8)
        w = buf[1:].reshape(channels, width)
        self.assertEqual(w.ctypes.data % 4, 1)
        weight = mx.from_dlpack(w)
        for rows in (33, 513):
            x = rng.integers(-128, 128, (rows, width), dtype=np.int8)
            expected = (x.astype(np.int32) @ w.astype(np.int32).T).astype(np.float32)
            with patch.object(_portable, "_legacy_metal", return_value=True), patch.object(_portable, "_has_integer_tensor_ops", return_value=False):
                output = _portable.portable_matmul(mx.array(x), weight, mx.ones((rows,)), mx.ones((channels,)), mx.float32)
            np.testing.assert_array_equal(np.array(output), expected)

    def test_direct_tensor_cancellation_and_row_boundary(self):
        if not mx.metal.is_available() or not _portable._has_integer_tensor_ops():
            self.skipTest("requires direct integer TensorOps")
        width = 25600
        x = np.full((513, width), -128, np.int8)
        w = np.full((64, width), 127, np.int8)
        w[0, width // 2:] = -127
        w[0, 0] = 126
        self.check_outputs(x, w, np.ones(len(x), np.float32), np.ones(64, np.float32), mx.float32)

    def test_direct_tensor_pads_partial_output_tiles(self):
        if not mx.metal.is_available() or not _portable._has_integer_tensor_ops():
            self.skipTest("requires direct integer TensorOps")
        # A partially filled tile on the last channel group must be writable
        # even on devices that issue a full-tile store.
        x = mx.ones((1093, 5376), dtype=mx.int8)
        w = mx.ones((5376, 5376), dtype=mx.int8)
        dots = _portable._tensor_dot(x, w)
        self.assertEqual(dots.shape, (1093, 5376))
        self.assertTrue(mx.all(dots == 5376).item())

    def test_full_row_batch_handles_a_partial_dot_and_odd_row_count(self):
        rng = np.random.default_rng(32)
        width = 5376
        x = rng.integers(-128, 128, (1025, width), dtype=np.int8)
        w = rng.integers(-128, 128, (64, width), dtype=np.int8)
        x[0] = -128
        w[0] = -128
        self.check_outputs(x, w, np.ones(len(x), np.float32), np.ones(64, np.float32), mx.float32)

    def test_legacy_large_simd_projection_extremes_and_partial_tiles(self):
        if not mx.metal.is_available():
            self.skipTest("requires Metal")
        rng = np.random.default_rng(502)
        rows, width, channels = 513, 8256, 1088
        a = rng.integers(-128, 128, (rows, width), dtype=np.int8)
        b = rng.integers(-128, 128, (channels, width), dtype=np.int8)
        a[0] = -128
        b[0] = -128
        b[-1, :width // 2] = 127
        b[-1, width // 2:] = -127
        b[-1, 0] = 126
        xs = np.linspace(.125, .5, rows, dtype=np.float32)
        ws = np.linspace(.25, .75, channels, dtype=np.float32)
        indices = [0, 1, 1023, 1024, channels - 1]
        oracle = a.astype(np.int64) @ b[indices].astype(np.int64).T
        reference = oracle.astype(np.float32) * xs[:, None] * ws[indices]
        with patch.object(_portable, "_legacy_metal", return_value=True), patch.object(_portable, "_has_integer_tensor_ops", return_value=False):
            for dtype in (mx.float32, mx.bfloat16):
                y = _portable.portable_matmul(mx.array(a), mx.array(b), mx.array(xs), mx.array(ws), dtype)
                self.assertTrue(mx.array_equal(y[:, indices], mx.array(reference).astype(dtype)).item())

    def test_convrot_projection_without_custom_metal_kernel(self):
        # Integer activations make the normalized Hadamard rotation exact.
        rng = np.random.default_rng(13)
        x = rng.integers(-8, 9, (11, 2048)).astype(np.float32)
        x[0] = 0
        w = rng.integers(-128, 128, (73, 2048), dtype=np.int8)
        scale = np.linspace(.002, .03, len(w), dtype=np.float32)[:, None]
        h = np.ones((4, 4), dtype=np.float32)
        np.fill_diagonal(h[:, ::-1], -1)
        h = np.kron(np.kron(h, h), np.kron(h, h)) / 16
        rot = (x.reshape(-1, 256) @ h).reshape(x.shape)
        xs = np.maximum(np.abs(rot).max(axis=1) / 127, 1e-20)
        q = np.rint(rot / xs[:, None]).clip(-127, 127).astype(np.int64)
        dots = q @ w.astype(np.int64).T
        expected = dots.astype(np.float32) * xs[:, None] * scale.reshape(-1)
        with patch.object(mx.fast, "metal_kernel", side_effect=AssertionError("unexpected custom Metal kernel")):
            from h3_image_mlx import core
            token = core.backend.set("portable")
            try:
                y = core.w8a8(mx.array(x), mx.array(w), mx.array(scale))
                np.testing.assert_array_equal(np.array(y), expected)
            finally:
                core.backend.reset(token)

    @unittest.skipUnless(os.environ.get("H3_IMAGE_TEST_WEIGHTS"), "requires local H3 weights")
    def test_real_dit_and_text_projections(self):
        root = Path(os.environ["H3_IMAGE_TEST_WEIGHTS"])
        rng = np.random.default_rng(20261111)
        cases = [
            ("diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors", "blocks.0.attn.out_proj"),
            ("text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "model.layers.0.mlp.down_proj"),
        ]
        for filename, name in cases:
            with self.subTest(projection=name):
                w = first_rows(root / filename, name + ".weight", 65)
                scale = first_rows(root / filename, name + ".weight_scale", 65).astype(np.float32)
                x = rng.integers(-127, 128, (19, w.shape[1]), dtype=np.int8)
                xs = rng.uniform(.001, .025, len(x)).astype(np.float32)
                self.check_outputs(x, w, xs, scale, mx.float32)
                self.check_outputs(x, w, xs, scale, mx.bfloat16)


if __name__ == "__main__":
    unittest.main()
