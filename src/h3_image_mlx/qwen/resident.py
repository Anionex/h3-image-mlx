"""Reusable Qwen weights in an isolated native process."""

import base64
import io
import json
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import httpx
from PIL import Image

from .._subprocess import Cancelled
from ..api import GenerationResult
from .assets import MANIFEST, identity
from .runtime import locate


class QwenSession:
    """Serial Qwen request session. Call close() or use a context manager to unload."""

    def __init__(self, runtime=None):
        self.runtime = runtime
        self._lock = threading.RLock()
        self._process = self._directory = self._key = self._url = None
        self._weight_loads = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def status(self):
        process = self._process
        return {
            "pid": process.pid if process and process.poll() is None else None,
            "weight_loads": self._weight_loads,
        }

    def close(self):
        with self._lock:
            process = self._process
            self._process = self._key = self._url = None
            try:
                if process and process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=4)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            finally:
                if self._directory:
                    self._directory.cleanup()
                    self._directory = None

    def _drain_log(self, pipe):
        with pipe:
            for line in pipe:
                if "[image] Qwen-Image-2.1 ready (DiT" in line:
                    self._weight_loads += 1

    def _start(self, model, backend, auto_adapter, cancel):
        key = (model, identity(model / "transformer"), auto_adapter)
        self._check(cancel)
        if self._key == key and self.status()["pid"]:
            return
        self.close()

        runtime = locate(self.runtime, resident=True)
        self._directory = tempfile.TemporaryDirectory(prefix="qwen-resident-")
        root = Path(self._directory.name)
        link = root / MANIFEST["repo"]
        link.parent.mkdir(parents=True)
        link.symlink_to(model, target_is_directory=True)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = dict(os.environ, MLX_SERVE_MF_DQ_GEMM="0",
                   QWEN_IMAGE_FUSED_ROPE="1", QWEN_IMAGE_FUSED_SWIGLU="1",
                   SHENGTU_LIGHTNING_AUTO_ADAPTER=str(auto_adapter))
        args = [runtime, "--serve", "--host", "127.0.0.1", "--port", str(port),
                "--model-dir", root, "--no-plugins", "--skip-mem-preflight",
                "--ctx-size", "8192", "--reasoning-budget", "0", "--idle-evict-secs", "0",
                "--log-file", "off", "--prefix-cache-disk", "0",
                "--w8a8" if backend == "w8a8" else "--no-w8a8"]
        try:
            process = self._process = subprocess.Popen(
                args, env=env, start_new_session=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, errors="replace",
            )
            threading.Thread(target=self._drain_log, args=(process.stdout,), daemon=True).start()
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 30
            with httpx.Client(trust_env=False, timeout=1) as client:
                while time.monotonic() < deadline:
                    self._check(cancel)
                    if process.poll() is not None:
                        raise RuntimeError(f"Qwen resident runtime exited ({process.returncode})")
                    try:
                        response = client.get(url + "/v1/models")
                        ids = [entry["id"] for entry in response.json()["data"]]
                        if response.is_success and MANIFEST["repo"] in ids:
                            self._key, self._url = key, url
                            return
                    except (httpx.HTTPError, ValueError, KeyError):
                        pass
                    time.sleep(0.1)
            raise TimeoutError("Qwen resident runtime did not discover its model")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(cancel):
        if cancel is not None and cancel.is_set():
            raise Cancelled("Generation cancelled")

    def _generate(self, model, body, backend, options, on_progress, cancel, schedule, auto_adapter):
        with self._lock:
            started = time.monotonic()
            body = {"stream": True, **body, "model": MANIFEST["repo"],
                    "response_format": "b64_json", "shengtu_private": options.private,
                    "shengtu_w8a8": backend == "w8a8"}
            if on_progress:
                on_progress(dict(stage="Loading model", step=0, total=body["steps"]))
            self._start(model, backend, auto_adapter, cancel)
            if on_progress:
                on_progress(dict(stage="Loading model", step=0, total=body["steps"],
                                 worker_pid=self.status()["pid"]))
            image, streamed_metrics = self._stream(body, options.timeout, on_progress, cancel)
            return GenerationResult(image, {
                **streamed_metrics, "model": "qwen-image-2.1-turbo", "backend": backend,
                "execution": "resident", "memory_mode": options.memory_mode,
                "profile": options.profile, "steps": streamed_metrics.get("steps", body["steps"]),
                "sample_sigmas": schedule, "seed": body["seed"], "weight_read_policy": "system",
                "generation_seconds": time.monotonic() - started, **self.status(),
            })

    def _stream(self, body, timeout, on_progress, cancel):
        done = threading.Event()
        connection = None

        def trace(event, info):
            nonlocal connection
            if event == "connection.connect_tcp.complete":
                connection = info["return_value"].get_extra_info("socket")
                self._check(cancel)

        def interrupt():
            while not done.wait(0.05):
                if cancel.is_set() and connection is not None:
                    try:
                        connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    return

        watcher = None
        if cancel is not None:
            watcher = threading.Thread(target=interrupt, daemon=True)
            watcher.start()
        try:
            self._check(cancel)
            with httpx.Client(trust_env=False, timeout=httpx.Timeout(timeout, connect=5)) as client:
                with client.stream("POST", self._url + "/v1/images/generations",
                                   json=body, extensions={"trace": trace}) as response:
                    if response.is_error:
                        response.read()
                        response.raise_for_status()
                    if "text/event-stream" in response.headers.get("content-type", ""):
                        for line in response.iter_lines():
                            self._check(cancel)
                            if not line.startswith("data: ") or line == "data: [DONE]":
                                continue
                            event = json.loads(line[6:])
                            if event.get("type") == "progress" and on_progress:
                                on_progress({k: v for k, v in event.items() if k != "type"})
                            if event.get("type") == "error":
                                raise RuntimeError(event.get("message") or "Qwen generation failed")
                            if event.get("type") == "complete":
                                return self._decode(event, body["transparent"])
                        raise RuntimeError("Qwen resident stream ended without an image")
                    response.read()
                    self._check(cancel)
                    return self._decode(response.json(), body["transparent"])
        except httpx.HTTPError as error:
            self._check(cancel)
            if isinstance(error, httpx.HTTPStatusError):
                error.response.read()
                try:
                    message = error.response.json()["error"]["message"]
                except (ValueError, KeyError, TypeError):
                    message = f"Qwen runtime returned HTTP {error.response.status_code}"
                raise RuntimeError(message) from error
            raise RuntimeError("Qwen resident runtime disconnected") from error
        finally:
            done.set()
            if watcher:
                watcher.join()

    @staticmethod
    def _decode(event, transparent):
        data = event.get("data")
        if not data:
            raise RuntimeError("Qwen resident runtime returned an empty image")
        with Image.open(io.BytesIO(base64.b64decode(data[0]["b64_json"], validate=True))) as image:
            return image.convert("RGBA" if transparent else "RGB"), event.get("metadata") or {}
