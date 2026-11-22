import os
import tempfile
import unittest

import mlx.core as mx
import numpy as np

from h3_image_mlx import core


class NativeReads(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, "preadv"), "requires preadv")
    def test_unaligned_slice_ending_at_eof(self):
        page = os.sysconf("SC_PAGESIZE")
        source = np.random.default_rng(19).bytes(8 * 1024**2 + page + 55)
        with tempfile.TemporaryFile() as weights:
            weights.write(source)
            weights.flush()
            for offset, length in [(0, 0), (page - 1, 2), (7, len(source) - 7)]:
                self.assertEqual(core._read_owned(weights.fileno(), length, offset).tobytes(),
                                 source[offset:offset + length])
            with self.assertRaises(EOFError):
                core._read_owned(weights.fileno(), len(source) + 1, 0)


class NativeProjection(unittest.TestCase):
    def check(self, rng, m, k, n, bm, bn, bk):
        a = rng.integers(-128, 128, (m, k), dtype=np.int8)
        b = rng.integers(-128, 128, (n, k), dtype=np.int8)
        a[0] = -128
        b[0] = 127
        b[1, :k//2], b[1, k//2:] = 127, -127
        b[1, 0] = 126
        xs = rng.uniform(.01, .1, m).astype(np.float32)
        ws = rng.uniform(.01, .1, n).astype(np.float32)
        expected = (a.astype(np.int64) @ b.astype(np.int64).T).astype(np.float32) * xs[:, None] * ws
        y = core.tensorops_matmul(*map(mx.array, [a, b, xs, ws]), mx.bfloat16, bm, bn, bk)
        self.assertEqual(y.shape, (m, n))
        self.assertTrue(mx.array_equal(y, mx.array(expected).astype(mx.bfloat16)).item())

    def setUp(self):
        if not mx.metal.is_available() or "Apple M5" not in mx.device_info()["device_name"]:
            self.skipTest("requires an Apple M5 GPU")

    def test_m5_tiles_and_edges(self):
        rng = np.random.default_rng(389)
        for m, k, n, bm, bn in [(1, 5120, 121, 32, 64), (32, 256, 64, 32, 64),
                                (65, 5376, 129, 64, 128), (33, 25600, 7, 64, 128)]:
            with self.subTest(shape=(m, k, n)):
                self.check(rng, m, k, n, bm, bn, 256)

    def test_m5_tail_tiles(self):
        # M and N that are not multiples of the tile: the last tiles are shifted to end at M and N (or the operand
        # is padded to one tile), and every output element must still match the integer reference. Run with
        # MTL_SHADER_VALIDATION=1 to catch a slice that leaves xq or wq: validation invalidates such a tile.
        rng = np.random.default_rng(1046)
        for bm, bn, bk in [(32, 64, 256), (64, 128, 256), (64, 128, 64), (64, 128, 32)]:
            for m in [1, 63, 65, 130]:
                for n in [127, 129, 200]:
                    with self.subTest(tiles=(bm, bn, bk), m=m, n=n):
                        self.check(rng, m, 512, n, bm, bn, bk)
