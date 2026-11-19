import importlib.util
import unittest

import mlx.core as mx

from h3_image_mlx.adapters.moss_soundeffect import _CachedCrossAttention, _decode_frame_count


@unittest.skipUnless(importlib.util.find_spec("moss_sfx_mlx"), "requires the upstream MOSS MLX runtime")
class MossOptimizationTests(unittest.TestCase):
    def test_dac_prefix_covers_the_requested_samples(self):
        from moss_sfx_mlx.model.dac_vae import Decoder

        rates = [8, 5, 4, 3, 2]
        with mx.stream(mx.cpu):
            decoder = Decoder(8, 32, rates)
            latent = mx.random.normal((1, 30, 8), key=mx.random.key(3))
            reference = decoder(latent)
            for output_samples in (1, 959, 960, 961, 3880, 18500):
                with self.subTest(samples=output_samples):
                    frames = _decode_frame_count(output_samples, rates)
                    cropped = decoder(latent[:, :frames])[:, :output_samples]
                    difference = mx.max(mx.abs(cropped - reference[:, :output_samples]))
                    self.assertLessEqual(difference.item(), 3e-7)

    def test_cross_attention_cache_uses_distinct_cfg_contexts(self):
        from moss_sfx_mlx.model.wan_video_dit import CrossAttention

        with mx.stream(mx.cpu):
            baseline = CrossAttention(64, 4)
            contexts = tuple(mx.random.normal((1, 3, 64), key=key) for key in mx.random.split(mx.random.key(4)))
            optimized = _CachedCrossAttention(baseline, contexts)
            for key in mx.random.split(mx.random.key(5)):
                hidden = mx.random.normal((1, 5, 64), key=key)
                for branch, context in enumerate(contexts):
                    self.assertTrue(mx.array_equal(optimized(hidden, branch), baseline(hidden, context)).item())
