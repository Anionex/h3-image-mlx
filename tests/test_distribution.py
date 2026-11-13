"""Distribution and dispatch regressions, runnable against an installed wheel."""
import hashlib
import json
import subprocess
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image

from h3_image_mlx import generate
from h3_image_mlx._stage import select_backend
from h3_image_mlx.download import _download_file
from h3_image_mlx.models import check_models


class PackageTests(unittest.TestCase):
    def test_packaged_runtime_resources(self):
        package = files("h3_image_mlx")
        self.assertFalse(package.joinpath("vae_stats.json").is_file())
        for name in ("MODEL_LICENSE.txt", "weights.json", "weights-bf16.json", "NOTICE.txt",
                     "kernels/LICENSE", "kernels/mpp_tensor_ops.metal",
                     "kernels/w8a8_gemm_body.metal", "kernels/w8a8_epi_plain.metal"):
            self.assertTrue(package.joinpath(name).is_file(), name)

    def test_device_selection(self):
        for device in ("M1", "M2 Max", "M3 Ultra", "M4 Pro"):
            self.assertEqual(select_backend("auto", f"Apple {device}"), "portable")
            with self.assertRaises(ValueError):
                select_backend("w8a8", f"Apple {device}")
        self.assertEqual(select_backend("auto", "Apple M5 Pro"), "w8a8")
        self.assertEqual(select_backend("portable", "Apple M5 Pro"), "portable")

    def test_order_and_stage_boundaries(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            paths = [root / f"reference-{i}.png" for i in reversed(range(16))]
            for i, path in enumerate(paths):
                Image.new("RGB", (32, 32), (i, 0, 0)).save(path)
            def run(command, **kwargs):
                stage, folder = command[-2:]
                work = Path(folder)
                calls.append(stage)
                request = json.loads((work / "request.json").read_text())
                self.assertEqual(request["references"], list(map(str, paths)))
                if stage == "vae":
                    Image.new("RGB", (256, 256), "blue").save(work / "image.png")
                return subprocess.CompletedProcess(command, 0, "", "")
            with patch("h3_image_mlx.pipeline.check_models", return_value=root), patch("h3_image_mlx.pipeline.subprocess.run", side_effect=run):
                image = generate("Use <Picture 16>", models=root, references=paths, width=256, height=256)
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))
            self.assertEqual(calls, ["text","references","dit","vae"])
            with self.assertRaisesRegex(ValueError, "16"):
                generate("Edit", models=root, references=paths+[paths[0]])

    def test_invalid_request(self):
        with self.assertRaisesRegex(ValueError, "multiples"):
            generate("Prompt", models="missing", width=999)
        with self.assertRaisesRegex(FileNotFoundError, "Missing model file"):
            generate("Prompt", models="missing")

    def test_verify_and_range_resume(self):
        raw = b"test model data"
        entry = {"path":"model.bin", "bytes":len(raw), "sha256":hashlib.sha256(raw).hexdigest()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            dest = root / entry["path"]
            dest.with_name(dest.name+".part").write_bytes(raw[:4])
            def respond(request):
                self.assertEqual(request.headers["range"], "bytes=4-")
                return httpx.Response(206, content=raw[4:], headers={"Content-Range":f"bytes 4-{len(raw)-1}/{len(raw)}"})
            with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                _download_file(client, "https://example.test/model", dest, entry)
            with patch("h3_image_mlx.models.MANIFEST", {"files":[entry]}):
                self.assertEqual(check_models(root, hashes=True), root)
                dest.write_bytes(b"x"*len(raw))
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    check_models(root, hashes=True)


if __name__ == "__main__":
    unittest.main()
