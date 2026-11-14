"""Time baseline and candidate portable projections on identical, GPU-resident inputs."""
import argparse
import importlib.util
import statistics
import time

import mlx.core as mx
import numpy as np

from h3_image_mlx._portable import portable_matmul


def compare(baseline, rows, channels, width):
    rng = np.random.default_rng(42)
    x, w = (mx.array(rng.integers(-128, 128, (n, width), dtype=np.int8)) for n in (rows, channels))
    args = (x, w, mx.ones((rows,)), mx.ones((channels,)), mx.float32)
    mx.eval(*args[:-1])
    methods = (baseline, portable_matmul)
    reference, candidate = (method(*args) for method in methods)
    if not mx.array_equal(reference, candidate).item():
        raise ValueError(f"Projection outputs differ for {(rows, channels, width)}")
    del reference, candidate
    times = [[], []]
    for trial in range(8):
        for index in (trial % 2, 1 - trial % 2):
            start = time.perf_counter()
            output = methods[index](*args)
            mx.eval(output)
            times[index].append((time.perf_counter() - start) * 1000)
            del output
    before, after = map(statistics.median, times)
    print(f"{rows} x {channels} x {width}: {before:.2f} -> {after:.2f} ms ({before / after:.2f}x)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Path to the previous revision of _portable.py")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("baseline", args.baseline)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"MLX {mx.__version__}, {mx.device_info()['device_name']}", flush=True)
    for shape in ((32, 25600, 5120), (1024, 21504, 5376), (64, 5120, 25600)):
        compare(module.portable_matmul, *shape)
