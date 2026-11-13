"""Serial, verified uncached Qwen benchmarks via the public library API."""

import argparse
import hashlib
import json
import statistics
import subprocess
import threading
from pathlib import Path

from h3_image_mlx import QwenOptions, generate_result
from memory import disk_bytes, file_residency, system_sample

PROMPT = "A small red ceramic teapot on a plain white tabletop, spout on the left and handle on the right, soft daylight. No text, no cups."


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--runtime", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--sizes", type=int, nargs="+", default=[1024])
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["micro-fast", "micro"],
        default=["micro-fast"],
    )
    args = p.parse_args()
    if args.trials < 1:
        p.error("trials must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(
        [*args.models.rglob("*.safetensors"), *args.prepared.glob("*.safetensors")]
    )
    from h3_image_mlx.qwen.runtime import locate

    binary = locate(args.runtime)
    (args.output / "configuration.json").write_text(
        json.dumps(
            dict(
                prompt=PROMPT,
                seed=42,
                steps=8,
                model="qwen-image-2.1-turbo",
                runtime_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                trials=args.trials,
                sizes=args.sizes,
                modes=args.modes,
                weight_cache="uncached",
                max_weight_residency_bytes=16 * 1024**2,
                min_read_ratio=0.95,
                timing="fresh text, DiT and VAE processes including loading plus parent PNG conversion; preparation excluded",
                memory="maximum child lifetime physical footprint, decimal GB; whole-system RAM is separate",
            ),
            indent=2,
        )
    )
    subprocess.run(["sudo", "-n", "/usr/sbin/purge"], check=True, timeout=60)
    rows = []
    for size in args.sizes:
        for mode in args.modes:
            for trial in range(args.trials):
                name = f"{size}-{mode}-{trial}"
                before = file_residency(paths)
                system_before = system_sample()
                disk_before = disk_bytes()
                stop = threading.Event()
                cancel = threading.Event()
                samples = []
                errors = []

                def sample():
                    try:
                        while not stop.is_set():
                            s = system_sample()
                            s["weight_cache_bytes"] = file_residency(paths)[
                                "resident_bytes"
                            ]
                            samples.append(s)
                            if s["pressure"] >= 4:
                                cancel.set()
                            stop.wait(0.5)
                    except Exception as e:
                        errors.append(str(e))
                        cancel.set()

                thread = threading.Thread(target=sample)
                thread.start()
                try:
                    result = generate_result(
                        PROMPT,
                        models=args.models,
                        model="qwen-image-2.1-turbo",
                        width=size,
                        height=size,
                        seed=42,
                        qwen=QwenOptions(
                            runtime=args.runtime,
                            prepared=args.prepared,
                            cache=args.output / "derived",
                            memory_mode=mode,
                            weight_cache="uncached",
                        ),
                        cancel=cancel,
                    )
                finally:
                    stop.set()
                    thread.join(30)
                after = file_residency(paths)
                system_after = system_sample()
                disk_after = disk_bytes()
                violations = list(errors)
                if not samples:
                    violations.append("no monitor samples")
                cache_peak = max(
                    [
                        before["resident_bytes"],
                        after["resident_bytes"],
                        *[s["weight_cache_bytes"] for s in samples],
                    ]
                )
                if cache_peak > 16 * 1024**2:
                    violations.append("weight cache exceeds allowance")
                for stage in result.metrics["stages"]:
                    io = stage.get("weight_io", {})
                    if (
                        io.get("errors", 1)
                        or io.get("aligned_bytes", 0) <= 0
                        or stage.get("disk_read_bytes", 0)
                        < 0.95 * io.get("aligned_bytes", 0)
                    ):
                        violations.append(
                            stage["stage"] + ": missing/failed physical-read proof"
                        )
                row = dict(
                    name=name,
                    size=size,
                    mode=mode,
                    trial=trial,
                    cache_verified=not violations,
                    violations=violations,
                    metrics=result.metrics,
                    weight_cache_peak_bytes=cache_peak,
                    before=before,
                    after=after,
                    system_before=system_before,
                    system_after=system_after,
                    system_disk_read_bytes=disk_after - disk_before,
                    rgb_sha256=hashlib.sha256(result.image.tobytes()).hexdigest(),
                )
                result.image.save(args.output / (name + ".png"))
                (args.output / (name + "-system.json")).write_text(
                    json.dumps(samples, indent=2)
                )
                rows.append(row)
                (args.output / "runs.json").write_text(json.dumps(rows, indent=2))
                print(
                    json.dumps(
                        dict(
                            name=name,
                            verified=row["cache_verified"],
                            seconds=result.metrics["generation_seconds"],
                            peak=result.metrics["worker_peak_bytes"],
                            cache_peak=cache_peak,
                            violations=violations,
                        )
                    ),
                    flush=True,
                )
                if violations:
                    raise RuntimeError("Uncached measurement rejected")
    summary = []
    for size in args.sizes:
        for mode in args.modes:
            group = [r for r in rows if r["size"] == size and r["mode"] == mode]
            summary.append(
                dict(
                    size=size,
                    mode=mode,
                    trials=len(group),
                    median_seconds=statistics.median(
                        r["metrics"]["generation_seconds"] for r in group
                    ),
                    min_seconds=min(r["metrics"]["generation_seconds"] for r in group),
                    max_seconds=max(r["metrics"]["generation_seconds"] for r in group),
                    max_process_GB=max(r["metrics"]["worker_peak_bytes"] for r in group)
                    / 1e9,
                    median_disk_read_GB=statistics.median(
                        r["metrics"]["disk_read_bytes"] for r in group
                    )
                    / 1e9,
                    max_weight_cache_bytes=max(
                        r["weight_cache_peak_bytes"] for r in group
                    ),
                    deterministic=len({r["rgb_sha256"] for r in group}) == 1,
                )
            )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
