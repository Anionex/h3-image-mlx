import unittest

import mlx.core as mx
import numpy as np

from h3_image_mlx import core, dit


class FusedBlockOps(unittest.TestCase):
    """The fused DiT elementwise ops give the same bytes as the composed ops."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.segments = [(0, 5, 1), (5, 9, 2), (9, 21, 0)]
        self.x = mx.array(rng.standard_normal((21, 64)), dtype=mx.bfloat16)
        self.y = mx.array(rng.standard_normal((21, 64)), dtype=mx.bfloat16)
        self.scale = mx.array(rng.standard_normal((3, 64)), dtype=mx.float32)
        self.shift = mx.array(rng.standard_normal((3, 64)), dtype=mx.float32)

    def test_shifted(self):
        seg = dit._segment_rows(self.segments)
        expected = dit.shifted(self.x, self.scale, self.shift, self.segments)
        self.assertTrue(mx.array_equal(dit.shifted_fused(self.x, self.scale, self.shift, seg), expected))

    def test_gated_add(self):
        seg = dit._segment_rows(self.segments)
        expected = self.x + dit.modulate(self.y, self.scale, self.segments)
        self.assertTrue(mx.array_equal(dit.gated_add(self.x, self.y, self.scale, seg), expected))

    def test_swiglu_and_rope(self):
        rng = np.random.default_rng(11)
        h = mx.array(rng.standard_normal((21, 128)), dtype=mx.bfloat16)
        self.assertTrue(mx.array_equal(dit.swiglu_fused(h), core.swiglu(h)))
        q = mx.array(rng.standard_normal((4, 21, 64)), dtype=mx.bfloat16)
        angles = mx.array(rng.standard_normal((21, 32)), dtype=mx.float32)
        self.assertTrue(mx.array_equal(dit.rope_fused(q, angles), core.rope(q, angles)))


if __name__ == "__main__":
    unittest.main()
