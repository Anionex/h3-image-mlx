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
    # Geometry screening only. Validate identical outputs before recording times.
    methods = {
        "release": lambda: base.portable_matmul(*args),
        "pr": lambda: current.portable_matmul(*args),
    }
    def wide(panel, padded):
        a = x.astype(mx.float32)
        if padded:
            a = mx.pad(a, [(0, (-rows) % 64), (0, 0)])
        parts = []
        for lo in range(0, n, panel):
            if full:
                part = (a @ w[lo:lo+panel].astype(mx.float32).T)[:rows]
            else:
                # The two half bounds were independently verified above.
                b = w[lo:lo+panel]
                left = a[:, :k//2] @ b[:, :k//2].astype(mx.float32).T
                right = a[:, k//2:] @ b[:, k//2:].astype(mx.float32).T
                part = (left.astype(mx.int32) + right.astype(mx.int32))[:rows].astype(mx.float32)
            part = part * args[2][:, None] * args[3][lo:lo+panel]
            mx.eval(part)
            parts.append(part)
        return mx.concatenate(parts, 1)

    def simd(bm, bn, bk):
        return current._simd_gemm()(
            inputs=[x, w, args[2], args[3]],
            template=[("T", mx.float32), ("BM", bm), ("BN", bn), ("BK", bk)],
            grid=(n//bn*128, (rows+bm-1)//bm, 1), threadgroup=(128,1,1),
            output_shapes=[(rows,n)], output_dtypes=[mx.float32])[0]

    for panel in (1024, 2048, 4096, 8192):
        for pad in (False, True):
            methods[f"wide{panel}-pad{int(pad)}"] = lambda p=panel, b=pad: wide(p,b)
    methods["simd32"] = lambda: simd(32,64,64)
    methods["simd64"] = lambda: simd(64,64,32)
    ref = methods["release"]()
    mx.eval(ref)
    for name, fn in methods.items():
        assert mx.array_equal(ref, fn()).item(), name
    del ref
    names = list(methods)
    seconds = {name: [] for name in names}
    for trial in range(5):
        for idx in rng.permutation(len(names)):
            name = names[idx]
            start = time.perf_counter()
            y = methods[name]()
            mx.eval(y)
            seconds[name].append(time.perf_counter()-start)
            del y
    ms = {name:round(1000*statistics.median(v),2) for name,v in seconds.items()}
    print(f"shape {rows},{k},{n} median ms {ms}", flush=True)
    out["samples"].append(dict(rows=rows,k=k,n=n,seconds=seconds))
Path("shape-results.json").write_text(json.dumps(out, indent=2))
