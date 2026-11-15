"""Compare three old-Metal implementations at the four observed H3 denoiser shapes."""
import importlib.util
import json
import platform
import statistics
import subprocess
import time
from importlib.metadata import distribution
from pathlib import Path

import mlx.core as mx
import numpy as np

root = Path(__file__).resolve().parents[2]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = load("release", distribution("h3-image-mlx").locate_file("h3_image_mlx/_portable.py"))
current = load("current", root / "pr/src/h3_image_mlx/_portable.py")
split = load("split", root / "src/h3_image_mlx/_portable.py")
methods = (base, current, split)
assert platform.mac_ver()[0].startswith("15.") and mx.metal.is_available()
assert not split._has_integer_tensor_ops()
out = dict(cpu=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
           gpu=mx.device_info()["device_name"], os=platform.mac_ver()[0], samples=[])
for k, n in [(5376, 21504), (7168, 5376), (5376, 28672), (14336, 5376)]:
    rows = 1038
    # Magnitudes exercise the certificate recorded for each shape in the teapot render.
    # These are synthetic inputs. No model tensors are sent to the runner.
    rng = np.random.default_rng(k + n)
    x_bound, w_bound = (64, 24) if k < 8192 else (88, 67)
    x = mx.array(rng.integers(-x_bound, x_bound, (rows, k), dtype=np.int8))
    w = mx.array(rng.integers(-w_bound, w_bound, (n, k), dtype=np.int8))
    args = (x, w, mx.ones((rows,)), mx.ones((n,)), mx.float32)
    mx.eval(*args[:-1])
    spans = len(current._safe_spans(x))
    energies = [current._max_row_energy(t) for t in (x, w)]
    mx.eval(*energies)
    full = energies[0].item() * energies[1].item() <= 1 << 48
    assert spans > 1 and full == (k < 8192)
    if k == 14336:
        assert split._certified_halves(x, w) == [(0, k // 2), (k // 2, k)]
    reference = base.portable_matmul(*args)
    mx.eval(reference)
    for method in methods[1:]:
        assert mx.array_equal(reference, method.portable_matmul(*args)).item()
    del reference
    seconds = [[] for _ in methods]
    for trial in range(6):
        for offset in range(3):
            idx = (trial + offset) % 3
            start = time.perf_counter()
            y = methods[idx].portable_matmul(*args)
            mx.eval(y)
            seconds[idx].append(time.perf_counter() - start)
            del y
    ms = [round(1000 * statistics.median(group), 2) for group in seconds]
    print(f"shape M,K,N={rows},{k},{n} median ms release/PR/split {ms}, L1 spans={spans}, full energy={full}", flush=True)
    out["samples"].append(dict(rows=rows,k=k,n=n,spans=spans,full_energy=full,seconds=seconds))
Path("shape-results.json").write_text(json.dumps(out, indent=2))
