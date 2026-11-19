"""Measure H3 streaming with fixed seeds and verified physical weight reads."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import threading
import time
from pathlib import Path

from h3_image_mlx import generate_result
from h3_image_mlx.models import dit_file, model_paths, text_file
from qwen.memory import call, disk_bytes, file_residency, process_tree_rss, system_sample


PROMPT = "A red ceramic teapot on a white tabletop, soft side lighting, no text."
MAX_RESIDENCY = 16 << 20
MIN_DISK_RATIO = 0.95


def trial(args, mode, paths, name):
    before = file_residency(paths)
    if before["resident_bytes"] > MAX_RESIDENCY:
        raise RuntimeError(f"Weight residency before {name} exceeds {MAX_RESIDENCY} bytes")
    vm_before, disk_before = system_sample(), disk_bytes()
    stop = threading.Event()
    cancel = threading.Event()
    samples = []
    errors = []

    def monitor():
        try:
            while not stop.is_set():
                sample = system_sample()
                sample["weight_cache_bytes"] = file_residency(paths)["resident_bytes"]
                sample["process_tree_rss_bytes"] = process_tree_rss(os.getpid())
                samples.append(sample)
                if sample["pressure"] >= 4:
                    errors.append("System memory pressure reached a critical level")
                    cancel.set()
                stop.wait(0.5)
        except Exception as exc:
            errors.append(str(exc))
            cancel.set()

    sampler = threading.Thread(target=monitor)
    deadline = threading.Timer(args.timeout, cancel.set)
    sampler.start()
    deadline.start()
    started = time.monotonic()
    try:
        result = generate_result(
            PROMPT, models=args.models, seed=42, width=args.size, height=args.size,
            backend=args.backend, dit_mode="resident" if mode == "resident" else "stream",
            weight_cache="uncached", overlap_streaming=mode == "overlap", cancel=cancel,
        )
        seconds = time.monotonic() - started
    finally:
        stop.set()
        deadline.cancel()
        sampler.join()
        (args.output / f"{name}-system.json").write_text(json.dumps(samples, indent=2))
    after = file_residency(paths)
    vm_after, physical = system_sample(), disk_bytes() - disk_before
    peak_cache = max(before["resident_bytes"], after["resident_bytes"],
                     *[sample["weight_cache_bytes"] for sample in samples])
    if peak_cache > MAX_RESIDENCY or not samples:
        errors.append("Weight residency check failed")
    total_weights = 0
    for stage in result.metrics["stages"]:
        expected = stage["weight_io"]["aligned_bytes"]
        total_weights += expected
        if not expected or stage["disk_read_bytes"] < MIN_DISK_RATIO * expected:
            errors.append(f"{stage['stage']}: physical disk reads too low")
    if physical < MIN_DISK_RATIO * total_weights:
        errors.append("System physical disk reads too low")
    result.image.save(args.output / f"{name}.png")
    return dict(
        name=name, mode=mode, seconds=seconds, metrics=result.metrics, verified=not errors,
        errors=errors, cache_before=before, cache_after=after, cache_peak_bytes=peak_cache,
        sampled_process_tree_rss_bytes=max((sample["process_tree_rss_bytes"] for sample in samples), default=0),
        system_before=vm_before, system_after=vm_after, disk_read_bytes=physical,
        rgb_sha256=hashlib.sha256(result.image.tobytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--backend", choices=("auto", "portable", "w8a8"), default="auto")
    parser.add_argument("--modes", choices=("sync", "overlap", "resident"), nargs="+", default=("sync", "overlap"))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.trials <= 0 or args.timeout <= 0:
        parser.error("trials and timeout must be positive")
    if not 256 <= args.size <= 4096 or args.size % 32:
        parser.error("size must be a multiple of 32 from 256 to 4096")
    args.models = args.models.resolve()
    paths = [
        Path(os.environ.get("H3_DIT") or dit_file(args.models, "int8")),
        Path(os.environ.get("H3_TEXT") or text_file(args.models, "int8")),
        Path(os.environ.get("H3_VAE") or args.models / model_paths()[2]),
    ]
    if os.environ.get("H3_LORA"):
        paths.append(Path(os.environ["H3_LORA"]))
    args.output.mkdir(parents=True, exist_ok=False)
    configuration = dict(
        platform=platform.platform(), chip=call("sysctl", "-n", "machdep.cpu.brand_string").strip(),
        size=args.size, prompt=PROMPT, seed=42, backend=args.backend,
        modes=args.modes, trials=args.trials, paths=[str(path.resolve()) for path in paths],
        max_weight_residency_bytes=MAX_RESIDENCY, min_disk_read_ratio=MIN_DISK_RATIO,
    )
    (args.output / "configuration.json").write_text(json.dumps(configuration, indent=2))
    rows = []
    for index in range(args.trials):
        modes = args.modes if index % 2 == 0 else args.modes[::-1]
        for mode in modes:
            row = trial(args, mode, paths, f"{mode}-{index}")
            rows.append(row)
            (args.output / "runs.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps({k: row[k] for k in ("name", "seconds", "verified", "errors")}), flush=True)
            if not row["verified"]:
                raise SystemExit("Uncached measurement rejected")
    summary = {
        mode: {
            "median_seconds": statistics.median(r["seconds"] for r in rows if r["mode"] == mode),
            "peak_process_bytes": max(r["metrics"]["worker_peak_bytes"] for r in rows if r["mode"] == mode),
            "sampled_process_tree_rss_bytes": max(r["sampled_process_tree_rss_bytes"] for r in rows if r["mode"] == mode),
        }
        for mode in args.modes
    }
    summary["rgb_equal"] = len({r["rgb_sha256"] for r in rows}) == 1
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    if not summary["rgb_equal"]:
        raise SystemExit("Generated RGB output differs")


if __name__ == "__main__":
    main()
