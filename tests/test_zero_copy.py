"""Page-aligned weight conversion, verification and mapped reads."""
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from h3_image_mlx import core
from h3_image_mlx.align import align_file, align_models, aligned_name, is_aligned, read_header
from h3_image_mlx.cli import main
from h3_image_mlx.models import check_models


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

    def use_pinned_model_fixture(self, metadata=None):
        root = self.src.parent.resolve()
        header, start = read_header(self.src)
        if metadata is not None:
            header["__metadata__"] = metadata
        payload = self.src.read_bytes()[start:]
        original_header = json.dumps(header, indent=2).encode()
        original_header += b" " * ((-len(original_header)) % 8)
        self.src.write_bytes(struct.pack("<Q", len(original_header)) + original_header + payload)
        text = root / "text.safetensors"
        text.write_bytes(self.src.read_bytes())
        files = []
        for path in (self.src, text):
            data = path.read_bytes()
            files.append({"path": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        manifest = patch("h3_image_mlx.models.MANIFEST", {"files": files})
        manifest.start()
        self.addCleanup(manifest.stop)
        return root

    def corrupt_aligned_tensor(self, path):
        header, start = read_header(path)
        offset = start + header["a.weight"]["data_offsets"][0]
        with path.open("r+b") as file:
            file.seek(offset)
            original_byte = file.read(1)[0]
            file.seek(offset)
            file.write(bytes([original_byte ^ 1]))

    def replace_aligned_header(self, path, header, start):
        payload = path.read_bytes()[start:]
        raw = json.dumps(header, separators=(",", ":")).encode().ljust(start - 8, b" ")
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)

    def test_cli_verifies_aligned_only_and_detects_payload_corruption(self):
        root = self.use_pinned_model_fixture()
        aligned = aligned_name(self.src)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(main(["align", "--models", str(root)]), 0, errors.getvalue())
            self.assertFalse(self.src.exists())
            self.assertEqual(main(["verify", "--models", str(root)]), 0, errors.getvalue())
            self.corrupt_aligned_tensor(aligned)
            self.assertEqual(main(["verify", "--models", str(root)]), 1)
            self.assertEqual(main(["align", "--models", str(root)]), 1)
        self.assertTrue(aligned.exists())

    def test_verify_detects_modified_runtime_metadata(self):
        config = '{"transformer":{"pdd_sigmas":[1.0,0.0]}}'
        root = self.use_pinned_model_fixture({"config": config})
        align_models(root)
        self.assertEqual(check_models(root, hashes=True), root)

        aligned = aligned_name(self.src)
        header, start = read_header(aligned)
        header["__metadata__"]["config"] = config.replace("1.0", "0.5")
        self.replace_aligned_header(aligned, header, start)
        with self.assertRaisesRegex(ValueError, "(?i)metadata|config"):
            check_models(root, hashes=True)

    def test_align_repairs_a_damaged_copy_before_removing_the_original(self):
        root = self.use_pinned_model_fixture()
        align_models(root, keep_original=True)
        aligned = aligned_name(self.src)
        self.corrupt_aligned_tensor(aligned)
        corrupted = aligned.read_bytes()
        original = self.src.read_bytes()
        entries = set(root.iterdir())

        with patch("h3_image_mlx.align.os.replace", side_effect=OSError("cannot publish copy")):
            with self.assertRaisesRegex(OSError, "cannot publish copy"):
                align_models(root)
        self.assertEqual(self.src.read_bytes(), original)
        self.assertEqual(aligned.read_bytes(), corrupted)
        self.assertEqual(set(root.iterdir()), entries)

        align_models(root)
        self.assertFalse(self.src.exists())
        self.assertEqual(check_models(root, hashes=True), root)

    def test_legacy_aligned_copy_uses_original_header_when_available(self):
        root = self.use_pinned_model_fixture()
        align_models(root, keep_original=True)
        aligned = aligned_name(self.src)
        header, start = read_header(aligned)
        header["__metadata__"] = {"h3_aligned": str(core.PAGE), "h3_source": self.src.name}
        self.replace_aligned_header(aligned, header, start)

        self.assertEqual(check_models(root, hashes=True), root)
        original = self.src.read_bytes()
        self.src.unlink()
        self.assertEqual(check_models(root), root)
        with self.assertRaisesRegex(ValueError, "fetch|original|restore"):
            check_models(root, hashes=True)
        with self.assertRaisesRegex(ValueError, "fetch|original|restore"):
            align_models(root)

        self.src.write_bytes(original)
        align_models(root)
        self.assertFalse(self.src.exists())
        self.assertEqual(check_models(root, hashes=True), root)

    def test_truncated_source_fails_without_leaving_a_temporary_copy(self):
        self.src.write_bytes(self.src.read_bytes()[:-1])
        dst = aligned_name(self.src)
        code = "from h3_image_mlx.align import align_file; import sys; align_file(*sys.argv[1:])"
        result = subprocess.run(
            [sys.executable, "-c", code, str(self.src), str(dst)],
            capture_output=True, text=True, timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stderr, r"(?i)truncat|unexpected (end|safetensors data size)|short read")
        self.assertEqual(list(self.src.parent.iterdir()), [self.src])


if __name__ == "__main__":
    unittest.main()
