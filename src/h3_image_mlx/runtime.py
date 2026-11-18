"""Entry point for hosts that freeze Python into a desktop executable."""

import sys
from pathlib import Path

from .qwen.runtime import qwen_runtime_path


_STAGE_FLAG = "--h3-image-mlx-stage"


def dispatch_frozen_stage() -> bool:
    """Handle a core-owned stage in a frozen host, before running host startup.

    Call from the executable's main block. Return True when the stage was run.
    """
    if sys.argv[1:2] != [_STAGE_FLAG]:
        return False
    if sys.argv[2:] == ["probe"]:
        import mlx.core as mx

        x = mx.array([1.0, 2.0]) + 1
        mx.eval(x)
        if x.tolist() != [2.0, 3.0]:
            raise RuntimeError("MLX Metal runtime probe failed")
        return True
    from ._stage import main, stage_request

    main(sys.argv[2], Path(sys.argv[3]), stage_request())
    return True


def _stage_command(stage, work):
    if getattr(sys, "frozen", False):
        return [sys.executable, _STAGE_FLAG, stage, work]
    return [sys.executable, "-m", "h3_image_mlx._stage", stage, work]




def probe_frozen_runtime():
    """Run a Metal operation through the frozen executable's core-stage entry."""
    import subprocess

    if not getattr(sys, "frozen", False):
        raise RuntimeError("This probe requires a frozen executable")
    subprocess.run([sys.executable, _STAGE_FLAG, "probe"], check=True, timeout=20)
