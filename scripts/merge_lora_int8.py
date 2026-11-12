"""Build an int8 ConvRot DiT with a LoRA merged in BEFORE quantization.

Rotated int8 weights cannot take a LoRA merge, so h3-image-mlx runs the LoRA as a side path
(about +2 s per step on an M5 Air). This script removes that cost: for every LoRA module it
streams the bf16 weight from Hugging Face (HTTP range, nothing stored), adds
(alpha / rank) * B @ A in float32, applies the 256-group Hadamard rotation, quantizes to int8
with a per-row scale, and overwrites that tensor in a copy of the existing int8 checkpoint.
All other tensors and the header (metadata, sigmas, heads) stay byte-identical.

Usage: python scripts/merge_lora_int8.py --base <int8.safetensors> --lora <lora.safetensors> --out <new.safetensors>
Resumable: finished modules are listed in <out>.progress.json.
"""
import argparse, json, struct, subprocess, sys, time, urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/{rev}/diffusion_models/minimax_h3_image_pruned_bf16.safetensors"
H4 = np.ones((4, 4), np.float32)
np.fill_diagonal(H4[:, ::-1], -1)
H256 = np.kron(np.kron(H4, H4), np.kron(H4, H4)) / 16


def fetch(url, a, b, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
            data = urllib.request.urlopen(req, timeout=180).read()
            if len(data) == b - a + 1:
                return data
        except Exception as error:  # noqa: BLE001 - retry any network failure
            last = error
        time.sleep(2 + 3 * k)
    raise RuntimeError(f"range {a}-{b} failed after {tries} tries: {last if 'last' in dir() else 'short read'}")


def local_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def bf16(raw, shape):
    return (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)


def to_bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")  # round to nearest even


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--lora", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rev", default="main"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    url = REPO.format(rev=a.rev)
    (n,) = struct.unpack("<Q", fetch(url, 0, 7))
    remote, rstart = json.loads(fetch(url, 8, 7 + n)), 8 + n
    base, bstart = local_header(a.base)
    lora, lstart = local_header(a.lora)
    lfile = open(a.lora, "rb")

    def lora_tensor(key):
        spec = lora[key]; x, y = spec["data_offsets"]; lfile.seek(lstart + x); raw = lfile.read(y - x)
        return int(np.frombuffer(raw, "<i8")[0]) if spec["dtype"] == "I64" else bf16(raw, spec["shape"])

    modules = sorted({k[len("diffusion_model."):-len(".lora_A.weight")] for k in lora if k.endswith(".lora_A.weight")})
    plain = set()  # modules the base keeps in bf16 (token refiner): merge and write bf16, no download
    for m in modules:
        w8, wr = base[m + ".weight"], remote[m + ".weight"]
        assert wr["dtype"] == "BF16" and w8["shape"] == wr["shape"], m
        if w8["dtype"] == "BF16":
            plain.add(m)
            continue
        assert w8["dtype"] == "I8", m
        assert base[m + ".weight_scale"]["dtype"] == "F32" and base[m + ".weight_scale"]["shape"] == [w8["shape"][0], 1], m
    out, progress = Path(a.out), Path(a.out + ".progress.json")
    done = set(json.loads(progress.read_text())) if progress.exists() and out.exists() else set()
    if not out.exists():
        subprocess.run(["cp", "-c", a.base, str(out)], check=True)  # APFS clone: instant, copy-on-write
    todo = [m for m in modules if m not in done][: a.limit or None]
    print(f"{len(modules)} LoRA modules, {len(done)} already done, {len(todo)} to build", flush=True)
    t0, got = time.time(), 0
    with ThreadPoolExecutor(2) as pool, open(out, "r+b") as f:
        def read_base(m):
            x, y = base[m + ".weight"]["data_offsets"]
            with open(a.base, "rb") as src:
                src.seek(bstart + x)
                return src.read(y - x)

        def start(m):
            if m in plain:
                return m, pool.submit(read_base, m)
            x, y = remote[m + ".weight"]["data_offsets"]
            return m, pool.submit(fetch, url, rstart + x, rstart + y - 1)
        queue, pending = deque(todo), deque()
        while queue or pending:
            while queue and len(pending) < 2:
                pending.append(start(queue.popleft()))
            m, fut = pending.popleft()
            raw = fut.result(); got += len(raw)
            shape = base[m + ".weight"]["shape"]
            W = bf16(raw, shape).copy(); del raw
            A, B = lora_tensor(f"diffusion_model.{m}.lora_A.weight"), lora_tensor(f"diffusion_model.{m}.lora_B.weight")
            alpha = lora_tensor(f"diffusion_model.{m}.alpha") if f"diffusion_model.{m}.alpha" in lora else A.shape[0]
            delta = (B @ A) * np.float32(alpha / A.shape[0])
            rel = float(np.linalg.norm(delta) / np.linalg.norm(W))
            W += delta; del delta, A, B
            if m in plain:
                x, y = base[m + ".weight"]["data_offsets"]
                f.seek(bstart + x); f.write(to_bf16(W).tobytes()); f.flush()
                done.add(m); progress.write_text(json.dumps(sorted(done)))
                print(f"[{len(done):3d}/{len(modules)}] {m:44s} delta/W {rel*100:5.2f}%  bf16 merged (not quantized in base)", flush=True)
                continue
            W = (W.reshape(shape[0], -1, 256) @ H256).reshape(shape)
            scale = np.maximum(np.abs(W).max(axis=1) / 127, 1e-20).astype(np.float32)
            np.divide(W, scale[:, None], out=W); np.rint(W, out=W); np.clip(W, -127, 127, out=W)
            q = W.astype(np.int8); del W
            x, y = base[m + ".weight"]["data_offsets"]
            f.seek(bstart + x); old = np.frombuffer(f.read(y - x), "i1").reshape(shape)
            changed = float(np.mean(old != q)); del old
            f.seek(bstart + x); f.write(q.tobytes())
            x, y = base[m + ".weight_scale"]["data_offsets"]
            assert y - x == scale.nbytes
            f.seek(bstart + x); f.write(scale.astype("<f4").tobytes())
            f.flush(); done.add(m); progress.write_text(json.dumps(sorted(done)))
            el = time.time() - t0
            print(f"[{len(done):3d}/{len(modules)}] {m:44s} delta/W {rel*100:5.2f}%  int8 changed {changed*100:5.1f}%  "
                  f"{got/1e9:5.1f} GB in {el:5.0f}s ({got/1e6/max(el,1):.0f} MB/s)", flush=True)
    print("DONE" if len(done) == len(modules) else "PARTIAL", str(out), flush=True)


if __name__ == "__main__":
    main()
