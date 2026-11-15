import atexit
import json
import os
import sys
import time
from pathlib import Path
from metrics import footprint

start = time.monotonic()
calls = 0
# Only benchmark child processes receive this module via PYTHONPATH.
# Both arms force identical legacy dispatch when screening on a newer host.
if os.environ.get("H3_BENCH_ARM") in ("baseline", "candidate"):
    from h3_image_mlx import _portable
    if os.environ.get("H3_LEGACY_PROXY") == "1":
        _portable._has_integer_tensor_ops = lambda: False
        _portable._legacy_metal = lambda: True
    if os.environ.get("H3_BENCH_ARM") == "candidate":
        original = _portable._large_portable_matmul
        def counted(*args):
            global calls
            calls += 1
            return original(*args)
        _portable._large_portable_matmul = counted

@atexit.register
def record():
    stats = footprint()
    if stats is None:
        return
    path = Path(os.environ["H3_TRIAL_DIR"]) / f"{os.getpid()}.json"
    path.write_text(json.dumps(dict(argv=sys.argv, elapsed=time.monotonic()-start,
                                    candidate_calls=calls, **stats)))
