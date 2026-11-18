"""Bundle an already-built staged Qwen runner and its dylib dependency closure."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


def run(cmd):
    return subprocess.check_output(
        [str(x) for x in cmd], text=True, stderr=subprocess.STDOUT, timeout=120
    )


def prepare(source, destination, *, with_resident=False, deployment_target="26.2"):
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / "native/source.json").read_text())
    source_patch = run(
        [
            "git",
            "-C",
            source,
            "diff",
            provenance["base_commit"],
            "--",
            "src",
            "build.zig",
            "build",
        ]
    ).encode()
    if hashlib.sha256(source_patch).hexdigest() != provenance["patch_sha256"]:
        raise ValueError("Native source differs from the pinned Qwen source patch")
    binary = source / "zig-out/bin/qwen-stream-probe"
    if not binary.is_file():
        raise ValueError("Build the qwen-probe ReleaseFast target first")
    executables = [binary]
    if with_resident:
        resident = source / "zig-out/bin/mlx-serve"
        if not resident.is_file():
            raise FileNotFoundError(resident)
        executables.append(resident)
    executable_names = {p.name for p in executables}
    destination.mkdir(parents=True, exist_ok=True)
    lib = destination / "lib"
    lib.mkdir(exist_ok=True)
    roots = [
        source / "lib/mlx/lib",
        source / "lib/llama/lib",
        Path("/opt/homebrew/opt/webp/lib"),
    ]
    queue = list(executables)
    seen = {}
    deps = {}
    while queue:
        current = queue.pop(0).resolve()
        if current.name in seen:
            continue
        target = (
            destination / current.name
            if current.name in executable_names
            else lib / current.name
        )
        if target.exists():
            target.unlink()
        shutil.copy2(current, target)
        seen[current.name] = target
        refs = [
            line.strip().split(" (")[0]
            for line in run(["otool", "-L", current]).splitlines()[1:]
        ]
        deps[current.name] = []
        for dep in refs:
            if dep.startswith(("/System/", "/usr/lib/")):
                continue
            if dep.startswith("@loader_path/"):
                candidate = current.parent / dep.removeprefix("@loader_path/")
            elif dep.startswith("@rpath/"):
                candidate = next(
                    (
                        p / Path(dep).name
                        for p in roots
                        if (p / Path(dep).name).is_file()
                    ),
                    None,
                )
            else:
                candidate = Path(dep)
            if candidate is None or not candidate.is_file():
                raise ValueError(f"Unresolved native dependency: {dep}")
            resolved = candidate.resolve()
            deps[current.name].append((dep, resolved.name))
            queue.append(resolved)
    for name, target in seen.items():
        for original, dependency in deps[name]:
            replacement = (
                "@executable_path/lib/" if name in executable_names else "@loader_path/"
            ) + dependency
            if original != replacement:
                run(["install_name_tool", "-change", original, replacement, target])
        if target.suffix == ".dylib":
            run(["install_name_tool", "-id", "@loader_path/" + name, target])
        run(["codesign", "--force", "--sign", "-", target])
    shutil.copy2(source / "lib/mlx/lib/mlx.metallib", lib / "mlx.metallib")
    run(
        [
            "clang",
            "-O2",
            "-include",
            "limits.h",
            "-dynamiclib",
            f"-mmacosx-version-min={deployment_target}",
            root / "native/aligned_nocache.c",
            "-o",
            lib / "aligned_nocache.dylib",
        ]
    )
    run(["codesign", "--force", "--sign", "-", lib / "aligned_nocache.dylib"])
    public_base = "4e231ceebe63308cc038117d40b9f9b64bd0372e"
    source_patch = run(
        ["git", "-C", source, "diff", public_base, "--", "src", "build.zig", "build"]
    ).encode()
    manifest = {
        "upstream_base": public_base,
        "source_patch_sha256": hashlib.sha256(source_patch).hexdigest(),
        "runner_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "executables": sorted(executable_names),
        "base_commit": run(["git", "-C", source, "rev-parse", "HEAD"]).strip(),
        "files": {
            str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in destination.rglob("*")
            if p.is_file() and p.name != "runtime.json"
        },
    }
    versions = []
    for file in (*(destination / name for name in executable_names), *lib.glob("*.dylib")):
        build = run(["vtool", "-show-build", file])
        versions.extend(
            tuple(map(int, v.split(".")[:2]))
            for v in re.findall(r"minos (\d+\.\d+)", build)
        )
    minimum = max(versions)
    requested = tuple(map(int, deployment_target.split(".")))
    if minimum > requested:
        raise ValueError(f"Native dependency targets macOS {'.'.join(map(str, minimum))}; rebuild for {deployment_target}")
    manifest["minimum_macos"] = ".".join(map(str, minimum))
    # Python packaging uses a zero minor component for macOS 11 and later.
    minor = 0 if minimum[0] >= 11 else minimum[1]
    manifest["wheel_platform"] = f"macosx_{minimum[0]}_{minor}_arm64"
    licenses = destination / "licenses"
    licenses.mkdir(exist_ok=True)
    for name in ("LICENSE", "LICENSE-APACHE-2.0", "NOTICE"):
        shutil.copy2(source / name, licenses / name)
    manifest["files"] = {
        str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in destination.rglob("*")
        if p.is_file() and p.name != "runtime.json"
    }
    (destination / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Pinned source checkout with native/qwen-runtime.patch applied",
    )
    p.add_argument(
        "--build", action="store_true", help="Compile ReleaseFast before packaging"
    )
    p.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "packages/qwen-runtime/src/h3_image_mlx_native/payload",
    )
    p.add_argument("--with-resident", action="store_true",
                   help="Include the reusable Qwen and PE server")
    p.add_argument("--deployment-target", default="26.2", choices=("26.0", "26.2"))
    args = p.parse_args()
    source = args.source.resolve()
    if args.build:
        command = [str(source / ".zig-toolchain/zig"), "build", "-Doptimize=ReleaseFast",
                   "-Dstrip=true", f"-Dtarget=native-native.{args.deployment_target}"]
        if args.with_resident:
            subprocess.run(command + ["-Dversion=26.11.2"], cwd=source, check=True, timeout=1200)
        subprocess.run(command + ["-Dqwen-probe=true"], cwd=source, check=True, timeout=1200)
    print(json.dumps(prepare(
        source, args.destination.resolve(), with_resident=args.with_resident,
        deployment_target=args.deployment_target,
    ), indent=2))
