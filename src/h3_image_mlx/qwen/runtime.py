"""Native worker execution, independent of model semantics and desktop state."""

import importlib.resources
import json
import os
import platform
from pathlib import Path

from .._subprocess import Cancelled, run_worker


def locate(explicit=None, *, resident=False, check_os=True):
    name = "mlx-serve" if resident else "qwen-stream-probe"
    if explicit:
        path = Path(explicit).expanduser().resolve()
        path = path / name if path.is_dir() else path
    else:
        try:
            path = Path(
                str(
                    importlib.resources.files("h3_image_mlx_native").joinpath(
                        f"payload/{name}"
                    )
                )
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Qwen requires the optional h3-image-mlx-qwen-runtime package, or QwenOptions(runtime=...). See docs/QWEN.md."
            ) from exc
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"Qwen runtime is not executable: {path}")
    manifest = path.parent / "runtime.json"
    if check_os and manifest.is_file():
        minimum = json.loads(manifest.read_text()).get("minimum_macos")
        if minimum:
            required = tuple(map(int, minimum.split(".")))
            current = tuple(map(int, platform.mac_ver()[0].split(".")[: len(required)]))
            if current < required:
                raise RuntimeError(
                    f"This Qwen runtime requires macOS {minimum} or later"
                )
    return path



def qwen_runtime_path(*, resident=False, check_os=True):
    """Locate the compatible executable from the optional native runtime package."""
    return locate(resident=resident, check_os=check_os)
