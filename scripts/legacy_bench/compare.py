"""Compare the installed H3 release with one or two checkouts on the same Mac."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys


def source_hashes(checkout):
    src = checkout.resolve() / "src/h3_image_mlx"
    files = [src / name for name in ("_portable.py", "core.py", "_stage.py")]
    files += sorted((src / "kernels").glob("portable*.metal"))
    return {str(f.relative_to(src)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def revision(checkout):
    value = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
    return value.stdout.strip() if value.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("candidate", "weights", "results"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--alternative", type=Path, help="Second checkout to compare against the same release")
    parser.add_argument("--pairs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds allowed per complete render")
    parser.add_argument("--proxy", action="store_true", help="Force legacy dispatch on a newer Mac for screening")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")

    checkouts = {"candidate": args.candidate}
    if args.alternative:
        checkouts["alternative"] = args.alternative
        order = ["baseline", "alternative", "candidate", "alternative", "candidate", "baseline",
                 "candidate", "baseline", "alternative"][:3 * args.pairs]
    else:
        order = ["baseline", "candidate", "candidate", "baseline", "baseline", "candidate"][:2 * args.pairs]

    root = args.results.resolve()
    root.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    dylib = root / "audit.dylib"
    subprocess.run(["clang", "-O2", "-dynamiclib", str(here / "io.c"), "-o", str(dylib)], check=True)
    env = dict(os.environ, H3_BENCH_DYLIB=str(dylib))
    cfg = dict(order=order, proxy=args.proxy, size=1024,
               sources={arm: source_hashes(path) for arm, path in checkouts.items()},
               revisions={arm: revision(path) for arm, path in checkouts.items()})
    (root / "configuration.json").write_text(json.dumps(cfg, indent=2))
    rows = []
    for number, arm in enumerate(order, 1):
        dest = root / f"{number}-{arm}"
        cmd = [sys.executable, str(here / "trial.py"), "--arm", "baseline" if arm == "baseline" else "candidate",
               "--candidate", str(checkouts.get(arm, args.candidate)), "--weights", str(args.weights),
               "--result", str(dest), "--timeout", str(args.timeout)]
        if args.proxy:
            cmd.append("--proxy")
        print(f"[{number}/{len(order)}] {arm}", flush=True)
        subprocess.run(cmd, env=env, check=True)
        r = json.loads((dest / "result.json").read_text())
        if not r["accepted"] or (rows and r["sha256"] != rows[0]["sha256"]):
            raise ValueError(f"Output or I/O verification failed: {dest}")
        r["arm"] = arm
        r["peak_bytes"] = max(s["lifetime_peak_bytes"] for s in r["stages"].values())
        (dest / "result.json").write_text(json.dumps(r, indent=2))
        rows.append(r)

    summary = {arm: {key: median(r[key] for r in rows if r["arm"] == arm)
                     for key in ("wall", "peak_bytes")} for arm in dict.fromkeys(order)}
    for arm in checkouts:
        key = "latency_reduction" if arm == "candidate" else "alternative_latency_reduction"
        summary[key] = 1 - summary[arm]["wall"] / summary["baseline"]["wall"]
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
