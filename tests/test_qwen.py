"""Public API, model planning and real subprocess failure/cancellation tests."""

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from h3_image_mlx import QwenOptions, generate
from h3_image_mlx.qwen.assets import identity, validate_prepared
from h3_image_mlx.qwen.lora import inspect_adapter
from h3_image_mlx.qwen.pipeline import encode_references, validate
from h3_image_mlx.qwen.runtime import Cancelled, locate, run_worker
from PIL import Image
from safetensors.numpy import save_file


class Requests(unittest.TestCase):
    def test_native_runtime_enforces_minor_macos_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "qwen-stream-probe"
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
            (root / "runtime.json").write_text(json.dumps({"minimum_macos": "26.2"}))
            with patch(
                "h3_image_mlx.qwen.runtime.platform.mac_ver",
                return_value=("26.1", (), "arm64"),
            ):
                with self.assertRaisesRegex(RuntimeError, "macOS 26.2"):
                    locate(binary)
            with patch(
                "h3_image_mlx.qwen.runtime.platform.mac_ver",
                return_value=("26.6", (), "arm64"),
            ):
                self.assertEqual(locate(binary), binary.resolve())

    def test_model_options_are_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "QwenOptions"):
            generate("x", models="missing", qwen=QwenOptions())
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            generate("x", models="missing", model="qwen-image-2.0")
        with self.assertRaisesRegex(ValueError, "callbacks"):
            generate("x", models="missing", on_progress=print)

    def test_uncached_policy_rejects_unverified_execution_paths(self):
        for backend, options in (
            (
                "portable",
                QwenOptions(weight_cache="uncached", memory_mode="micro-fast"),
            ),
            ("w8a8", QwenOptions(weight_cache="uncached", memory_mode="resident")),
            (
                "w8a8",
                QwenOptions(
                    weight_cache="uncached", memory_mode="micro-fast", private=True
                ),
            ),
        ):
            with (
                patch("h3_image_mlx.qwen.pipeline.assets.check"),
                patch("h3_image_mlx.qwen.pipeline.locate"),
                patch(
                    "h3_image_mlx.qwen.pipeline.subprocess.check_output",
                    return_value="Apple M5 Pro",
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "Uncached reads currently require"
                ):
                    generate(
                        "tea",
                        models="unused",
                        model="qwen-image-2.1-turbo",
                        backend=backend,
                        qwen=options,
                    )

    def test_parameter_and_reference_validation(self):
        for options in (
            QwenOptions(steps=0),
            QwenOptions(steps=4),
            QwenOptions(guidance_scale=float("nan")),
            QwenOptions(auto_strength=True),
            QwenOptions(sample_sigmas=(1, 0.5)),
            QwenOptions(mode="variation"),
        ):
            with self.assertRaises(ValueError):
                validate("tea", 512, 512, 42, [], options)
        with self.assertRaises(ValueError):
            validate("<image2>", 512, 512, 42, ["one"], QwenOptions())
        validate(
            "<image2>",
            512,
            768,
            42,
            ["one", "two"],
            QwenOptions(sample_sigmas=(1, 0.5), steps=2),
        )

    def test_lora_dimensions_pairs_and_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.safetensors"
            weights = {
                "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": np.ones(
                    (2, 4096), np.float32
                ),
                "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": np.ones(
                    (4096, 2), np.float32
                ),
            }
            save_file(weights, path)
            self.assertEqual(inspect_adapter(path), 1)
            weights.pop(next(iter(weights)))
            save_file(weights, path)
            with self.assertRaises(ValueError):
                inspect_adapter(path)

    def test_rgba_and_reference_order_preserved(self):
        import base64
        import io

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, color in enumerate([(255, 0, 0, 128), (0, 255, 0, 255)]):
                p = Path(tmp) / f"{i}.png"
                Image.new("RGBA", (32, 32), color).save(p)
                paths.append(p)
            for encoded, color in zip(
                encode_references(paths), [(255, 0, 0, 128), (0, 255, 0, 255)]
            ):
                with Image.open(io.BytesIO(base64.b64decode(encoded))) as im:
                    self.assertEqual(im.getpixel((0, 0)), color)

    def test_cache_is_bound_to_transformer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "transformer").mkdir()
            w = root / "transformer/x.safetensors"
            w.write_bytes(b"original")
            cache = root / "cache"
            cache.mkdir()
            (cache / "manifest.json").write_text(
                json.dumps({"transformer_identity": identity(root / "transformer")})
            )
            validate_prepared(cache, root)
            w.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "another transformer"):
                validate_prepared(cache, root)


class ProcessTests(unittest.TestCase):
    def script(self, root, source):
        path = root / "worker"
        path.write_text(f"#!{sys.executable}\n" + source)
        path.chmod(0o755)
        return path

    def test_progress_stream_and_binary_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.script(
                Path(tmp),
                """import sys,json,time
sys.stdin.buffer.read()
for i in range(3):
 print('PROGRESS '+json.dumps(dict(stage='Generating',step=i+1,total=3)),file=sys.stderr,flush=True)
 time.sleep(.02)
sys.stdout.buffer.write(b'PNG-fixture')
print('FOOTPRINT 1234 DISK_READ 4321',file=sys.stderr)
print('METRICS {"mlx_peak_bytes":987}',file=sys.stderr)
""",
            )
            seen = []
            out, m = run_worker([path], input_bytes=b"request", on_progress=seen.append)
            self.assertEqual(out, b"PNG-fixture")
            self.assertEqual([x["step"] for x in seen], [1, 2, 3])
            self.assertEqual(m["disk_read_bytes"], 4321)

    def test_cancel_and_timeout_reap_worker(self):
        for cancellation in (False, True):
            with (
                self.subTest(cancellation=cancellation),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                pidfile = root / "pid"
                path = self.script(
                    root,
                    f'import os,time\nopen({str(pidfile)!r},"w").write(str(os.getpid()))\ntime.sleep(30)\n',
                )
                event = threading.Event()
                timer = threading.Timer(1, event.set)
                if cancellation:
                    timer.start()
                try:
                    with self.assertRaises(Cancelled if cancellation else TimeoutError):
                        run_worker(
                            [path], cancel=event, timeout=3 if cancellation else 1
                        )
                finally:
                    timer.cancel()
                pid = int(pidfile.read_text())
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

    def test_nonzero_worker_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.script(
                Path(tmp),
                'import sys\nprint("bad input",file=sys.stderr)\nsys.exit(4)\n',
            )
            with self.assertRaisesRegex(RuntimeError, "bad input"):
                run_worker([path])


if __name__ == "__main__":
    unittest.main()
