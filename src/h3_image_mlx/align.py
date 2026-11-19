"""Page-aligned DiT checkpoint for zero-copy loading (H3_ZERO_COPY=1).

Rewrites a safetensors file so every tensor's data starts on a 16 KiB page: the header is space-padded so the data
section starts on a page, gaps are filled with U8 tensors named "__h3pad.N" (the format allows no holes), and the file
ends on a page. Tensor bytes are copied unchanged and verified. `h3-image-mlx align --models DIR` replaces the pinned
DiT and text encoder by aligned copies; models.py accepts and prefers them.
"""
import hashlib
import json
import os
import struct
from pathlib import Path

PAGE = 16384
SUFFIX = ".aligned16k.safetensors"


def aligned_name(path):
    path = Path(path)
    return path.with_name(path.name.removesuffix(".safetensors") + SUFFIX)


def read_header(path):
    with open(path, "rb") as file:
        (length,) = struct.unpack("<Q", file.read(8))
        return json.loads(file.read(length)), 8 + length


def is_aligned(path):
    header, start = read_header(path)
    return (header.get("__metadata__") or {}).get("h3_aligned") == str(PAGE) and start % PAGE == 0


def align_file(src, dst):
    """Write the page-aligned copy of src to dst (via dst.tmp) and verify every tensor."""
    header, start = read_header(src)
    meta = header.pop("__metadata__", None) or {}
    items = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    out, plan, offset, pads = {}, [], 0, 0

    def pad(size):
        nonlocal offset, pads
        out[f"__h3pad.{pads}"] = {"dtype": "U8", "shape": [size], "data_offsets": [offset, offset + size]}
        plan.append((None, size))
        offset += size
        pads += 1

    for key, spec in items:
        a, b = spec["data_offsets"]
        if offset % PAGE:
            pad(PAGE - offset % PAGE)
        out[key] = dict(spec, data_offsets=[offset, offset + b - a])
        plan.append((start + a, b - a))
        offset += b - a
    if offset % PAGE:
        pad(PAGE - offset % PAGE)
    meta = dict(meta, h3_aligned=str(PAGE), h3_source=Path(src).name)
    text = json.dumps({"__metadata__": meta, **out}, separators=(",", ":"))
    text += " " * ((-(8 + len(text.encode()))) % PAGE)
    tmp = Path(str(dst) + ".tmp")
    with open(src, "rb") as source, open(tmp, "wb") as target:
        target.write(struct.pack("<Q", len(text.encode())))
        target.write(text.encode())
        for position, size in plan:
            if position is None:
                target.write(b"\0" * size)
                continue
            source.seek(position)
            while size:
                chunk = source.read(min(size, 64 << 20))
                target.write(chunk)
                size -= len(chunk)
    new_header, new_start = read_header(tmp)
    with open(src, "rb") as source, open(tmp, "rb") as target:
        for key, spec in items:
            a, b = spec["data_offsets"]
            c, d = new_header[key]["data_offsets"]
            if (new_start + c) % PAGE or d - c != b - a:
                raise RuntimeError(f"alignment failed for {key}")
            source.seek(start + a)
            target.seek(new_start + c)
            left, h1, h2 = b - a, hashlib.blake2b(), hashlib.blake2b()
            while left:
                n = min(left, 64 << 20)
                h1.update(source.read(n))
                h2.update(target.read(n))
                left -= n
            if h1.digest() != h2.digest():
                raise RuntimeError(f"copy mismatch for {key}")
    os.replace(tmp, dst)
    return Path(dst)


def align_models(directory, precision="int8", keep_original=False):
    """Align the pinned DiT and text encoder; remove each original unless keep_original."""
    from .models import ALIGNABLE, model_paths

    done = []
    for index in ALIGNABLE:
        src = Path(directory).expanduser().resolve() / model_paths(precision)[index]
        dst = aligned_name(src)
        if not (dst.is_file() and is_aligned(dst)):
            if not src.is_file():
                raise FileNotFoundError(f"Missing file to align: {src}")
            align_file(src, dst)
        if src.is_file() and not keep_original:
            src.unlink()
        done.append(str(dst))
    return "\n".join(done)
