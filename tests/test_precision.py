"""Weight precision selection and floating-point inference regressions."""
import json
import os
import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from h3_image_mlx import core
from h3_image_mlx._stage import select_backend
from h3_image_mlx.models import manifest_for, model_paths, check_models
from h3_image_mlx.pipeline import embedding_cache, generate


class PrecisionTests(unittest.TestCase):
    def test_profiles_keep_shared_components_and_select_both_bf16_models(self):
        int8, bf16 = model_paths('int8'), model_paths('bf16')
        self.assertIn('pruned_bf16', bf16[0])
        self.assertIn('_bf16', bf16[1])
        self.assertNotEqual(int8[:2], bf16[:2])
        self.assertEqual(int8[2:], bf16[2:])
        for precision in ('int8', 'bf16'):
            m = manifest_for(precision)
            self.assertEqual(m['total_bytes'], sum(e['bytes'] for e in m['files']))
            self.assertTrue(all(e['sha256'] and e['modelscope'] and e['huggingface'] for e in m['files']))
        with self.assertRaisesRegex(ValueError, 'precision'):
            manifest_for('fp8')

    def test_bf16_uses_standard_mlx_on_all_chips(self):
        for chip in ('Apple M1', 'Apple M4 Pro', 'Apple M5 Pro'):
            for requested in ('auto', 'portable'):
                self.assertEqual(select_backend(requested, chip, 'bf16'), 'portable')
            with self.assertRaisesRegex(ValueError, 'INT8'):
                select_backend('w8a8', chip, 'bf16')

    def test_bf16_missing_file_guidance_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, r'--precision bf16'):
                check_models(tmp, precision='bf16')
        with self.assertRaisesRegex(ValueError, 'precision'):
            generate('test', models='missing', precision='fp8')
        with self.assertRaisesRegex(ValueError, 'INT8'):
            generate('test', models='missing', precision='bf16', backend='w8a8')

    def test_embedding_cache_separates_precisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for precision in ('int8', 'bf16'):
                text = root / model_paths(precision)[1]
                text.parent.mkdir(parents=True, exist_ok=True)
                text.write_bytes(b'same-size')
            with patch.dict(os.environ, {'H3_EMBED_CACHE':str(root/'cache')}):
                a = embedding_cache(root, 'same prompt', [], 'auto', 'int8')
                b = embedding_cache(root, 'same prompt', [], 'auto', 'bf16')
            self.assertNotEqual(a, b)

    def test_pipeline_passes_precision_to_every_stage(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def run(command, **kwargs):
                from PIL import Image
                stage, folder = command[-2:]
                work = Path(folder)
                request = json.loads((work/'request.json').read_text())
                self.assertEqual(request['precision'], 'bf16')
                calls.append(stage)
                if stage == 'vae':
                    Image.new('RGB', (256, 256), 'blue').save(work/'image.png')
                return subprocess.CompletedProcess(command, 0, '', '')
            with patch.dict(os.environ, {'H3_EMBED_CACHE':''}), patch('h3_image_mlx.pipeline.check_models', return_value=root) as check, patch('h3_image_mlx.pipeline.subprocess.run', side_effect=run):
                generate('test', models=root, width=256, height=256, precision='bf16')
                check.assert_called_once_with(root, precision='bf16')
            self.assertEqual(calls, ['text', 'dit', 'vae'])

    def test_bf16_dit_stage_streams_automatically_and_selects_weight_file(self):
        from h3_image_mlx import _stage
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            request = dict(models=str(work),precision='bf16',backend='auto',references=[],width=256,height=256,seed=42,dit_mode='auto')
            (work/'request.json').write_text(json.dumps(request))
            mx.save_safetensors(str(work/'embedding.safetensors'), {'embedding':mx.zeros((1,5120)), 'tags':mx.ones((1,), dtype=mx.int32)})
            with patch.dict(os.environ, {'H3_DIT':''}), patch.object(mx,'device_info',return_value={'device_name':'Apple M5 Pro','memory_size':48*1024**3}), patch('h3_image_mlx.dit.denoise',return_value=(mx.zeros((24,16,16)),0)) as denoise:
                _stage.main('dit',work)
                args,kwargs=denoise.call_args
                self.assertEqual(args[0],work/model_paths('bf16')[0])
                self.assertFalse(args[5])
                self.assertEqual(kwargs['precision'],'bf16')

    def test_quantized_dit_override_is_rejected_for_bf16(self):
        from h3_image_mlx.dit import denoise
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'quantized.safetensors'
            mx.save_safetensors(str(file),{'blocks.0.attn.qkv_proj.weight':mx.zeros((1,1),dtype=mx.int8)})
            with self.assertRaisesRegex(ValueError,'Check H3_DIT'):
                denoise(file,None,256,256,42,precision='bf16')

    def test_fetch_checks_selected_budget_before_network(self):
        from h3_image_mlx.download import fetch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp, patch('h3_image_mlx.download.shutil.disk_usage',return_value=SimpleNamespace(free=1)), patch('h3_image_mlx.download._download_file') as download:
            with self.assertRaisesRegex(OSError, '92,916,491,959'):
                fetch(Path(tmp),precision='bf16')
            download.assert_not_called()

    def test_bf16_projection_never_quantizes_activations(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'weights.safetensors'
            weight = mx.array([[0.25, -1.5, 0.75], [-0.5, 1.25, 2.0]], dtype=mx.bfloat16)
            bias = mx.array([0.125, -0.25], dtype=mx.bfloat16)
            mx.save_safetensors(str(file), {'p.weight':weight,'p.bias':bias})
            w = core.Weights(file)
            try:
                x = mx.array([[1.0, 0.5, -2.0]], dtype=mx.bfloat16)
                with patch.object(core, 'w8a8', side_effect=AssertionError('BF16 used quantization')):
                    actual = w.linear('p', x)
                    mx.eval(actual)
                expected = np.array(x.astype(mx.float32)) @ np.array(weight.astype(mx.float32)).T + np.array(bias.astype(mx.float32))
                np.testing.assert_array_equal(np.array(actual.astype(mx.float32)), expected)
                self.assertEqual(actual.dtype, mx.bfloat16)
                self.assertEqual(w.int8_calls, 0)
            finally:
                w.file.close()

    @unittest.skipUnless(os.environ.get('H3_IMAGE_TEST_BF16_SAMPLES'), 'requires ranged BF16 samples')
    def test_real_bf16_dit_text_and_vision_projections(self):
        root = Path(os.environ['H3_IMAGE_TEST_BF16_SAMPLES'])
        rng = np.random.default_rng(42)
        for filename in ('component-0.safetensors', 'component-1.safetensors'):
            w = core.Weights(root/filename)
            try:
                for key, spec in w.header.items():
                    if key == '__metadata__':
                        continue
                    a,b = spec['data_offsets']
                    bits = np.frombuffer(os.pread(w.file.fileno(), b-a, w.start+a), dtype='<u2').astype(np.uint32)
                    original = (bits << 16).view(np.float32).reshape(spec['shape'])
                    np.testing.assert_array_equal(np.array(w[key].astype(mx.float32)),original)
                    x = mx.array(rng.normal(size=(4,spec['shape'][1])).astype(np.float32)).astype(mx.bfloat16)
                    with patch.object(core,'w8a8',side_effect=AssertionError('quantized BF16')):
                        actual = np.array(w.linear(key.removesuffix('.weight'),x).astype(mx.float32))
                    reference = np.array(x.astype(mx.float32)).astype(np.float64) @ original.astype(np.float64).T
                    rounded = np.array(mx.array(reference.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32))
                    np.testing.assert_allclose(actual,rounded,rtol=0.008,atol=0.0001,err_msg=key)
                self.assertEqual(w.int8_calls,0)
            finally:
                w.file.close()


if __name__ == '__main__':
    unittest.main()
