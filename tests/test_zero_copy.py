"""Page-aligned copies (align.py) and zero-copy reads (H3_ZERO_COPY=1) return the same tensors as the original file."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from h3_image_mlx import core
from h3_image_mlx.align import align_file, aligned_name, is_aligned


class ZeroCopyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(0)
        self.tensors = {
            "a.weight": mx.array(rng.integers(-127, 128, (64, 300), dtype=np.int8)),
            "a.scale": mx.array(rng.standard_normal(64).astype(np.float32)),
            "b.weight": mx.array(rng.standard_normal((7, 129)).astype(np.float32)).astype(mx.bfloat16),
            "c.bias": mx.array(rng.standard_normal(3).astype(np.float16)),
        }
        self.src = Path(self.tmp.name) / "model.safetensors"
        mx.save_safetensors(str(self.src), self.tensors)

    def tearDown(self):
        self.tmp.cleanup()

    def read_all(self, path, zero_copy):
        with patch.dict(os.environ, {"H3_ZERO_COPY": "1" if zero_copy else ""}):
            weights = core.Weights(path)
            self.assertEqual(weights.mapped is not None, zero_copy)
            out = {key: weights[key] for key in self.tensors}
            mx.eval(list(out.values()))
            weights.file.close()
            return out

    def test_aligned_copy_reads_the_same_tensors(self):
        dst = align_file(self.src, aligned_name(self.src))
        self.assertTrue(is_aligned(dst))
        self.assertFalse(is_aligned(self.src))
        weights = core.Weights(dst)
        header, start = weights.header, weights.start
        weights.file.close()
        for key in self.tensors:
            self.assertEqual((start + header[key]["data_offsets"][0]) % core.PAGE, 0, key)
        reference = self.read_all(self.src, zero_copy=False)
        for zero_copy in (False, True):
            got = self.read_all(dst, zero_copy)
            for key, value in reference.items():
                self.assertEqual(got[key].dtype, value.dtype, key)
                self.assertEqual(got[key].shape, value.shape, key)
                self.assertTrue(mx.array_equal(got[key], value).item(), key)

    def test_unaligned_file_never_maps(self):
        with patch.dict(os.environ, {"H3_ZERO_COPY": "1"}):
            weights = core.Weights(self.src)
            self.assertIsNone(weights.mapped)
            weights.file.close()


if __name__ == "__main__":
    unittest.main()
