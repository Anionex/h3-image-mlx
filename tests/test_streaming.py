"""Checks for disk-backed block lifetimes and execution parity."""

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from h3_image_mlx.streaming import BlockStream, StreamPolicy, TensorStore


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.device = mx.default_device()
        mx.set_default_device(mx.cpu)
        self.addCleanup(mx.set_default_device, self.device)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.first = self.root / "first.safetensors"
        self.second = self.root / "second.safetensors"
        mx.save_safetensors(str(self.first), {"first.weight": mx.array([[1.0, 2.0], [0.0, 1.0]])})
        mx.save_safetensors(str(self.second), {"second.weight": mx.array([[2.0, 3.0], [2.0, 1.0]])})

    def test_streaming_matches_two_linear_layers_across_modes(self):
        value = mx.array([[3.0, 4.0]])
        results = []
        for policy in (StreamPolicy(overlap=False), StreamPolicy(overlap=True, prefetch_bytes=64)):
            with TensorStore((self.first, self.second)) as store, BlockStream(store, policy) as stream:
                first = nn.Linear(2, 2, bias=False)
                second = nn.Linear(2, 2, bias=False)
                apply_first = stream.wrap(first, "first", {"weight": "first.weight"})
                apply_second = stream.wrap(second, "second", {"weight": "second.weight"})
                for _ in range(2):
                    results.append(np.array(apply_second(apply_first(value))))
                    self.assertEqual(first.weight.size, 0)
                    self.assertEqual(second.weight.size, 0)
                self.assertEqual(stream.metrics["blocks"], 4)
                self.assertEqual(stream.metrics["prefetch_hits"], 2 if policy.overlap else 0)
                self.assertLessEqual(stream.metrics["prefetch_peak_bytes"], policy.prefetch_bytes)
        for result in results:
            np.testing.assert_array_equal(result, [[34.0, 26.0]])

    def test_oversized_block_is_read_without_prefetch(self):
        with TensorStore((self.first, self.second)) as store:
            with BlockStream(store, StreamPolicy(overlap=True, prefetch_bytes=1)) as stream:
                for name in ("first", "second"):
                    stream.register(name, {"w": f"{name}.weight"})
                stream.run("first", lambda weights: weights["w"])
                stream.run("second", lambda weights: weights["w"])
                self.assertEqual(stream.metrics["prefetch_peak_bytes"], 0)
                self.assertEqual(stream.metrics["weight_read_bytes"], 32)
                self.assertGreaterEqual(store.io["logical_bytes"], 32)

    def test_duplicate_tensor_across_shards_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate tensor first.weight"):
            TensorStore((self.first, self.first))


if __name__ == "__main__":
    unittest.main()
