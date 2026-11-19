"""Write page-aligned safetensors while keeping the original checksum verifiable."""
import fcntl
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

PAGE = 16384
SUFFIX = ".aligned16k.safetensors"
PAD_PREFIX = "__h3pad."
SOURCE_HEADER = "h3_source_header"
MAX_HEADER_BYTES = 100_000_000  # safetensors format limit


def aligned_name(path):
    path = Path(path)
    return path.with_name(path.name.removesuffix(".safetensors") + SUFFIX)


def _read_header(file):
    prefix = file.read(8)
    length = int.from_bytes(prefix, "little")
    if len(prefix) != 8 or not 0 < length <= MAX_HEADER_BYTES:
        raise ValueError(f"Invalid safetensors header: {file.name}")
    raw = file.read(length)
    if len(raw) != length:
        raise ValueError(f"Truncated safetensors header: {file.name}")
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header: {file.name}")
    metadata = header.get("__metadata__")
    if metadata is None:
        header["__metadata__"] = {}
    elif not isinstance(metadata, dict):
        raise ValueError(f"Invalid safetensors metadata: {file.name}")
    return header, raw


def read_header(path):
    with open(path, "rb") as file:
        header, raw = _read_header(file)
    return header, 8 + len(raw)


def is_aligned(path):
    header, start = read_header(path)
    return header.get("__metadata__", {}).get("h3_aligned") == str(PAGE) and start % PAGE == 0


def _tensor_layout(header, path, size=None):
    tensors = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        offsets = spec.get("data_offsets") if isinstance(spec, dict) else None
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)):
            raise ValueError(f"Invalid offsets for {name}: {path}")
        first, last = offsets
        tensors.append((first, last, name))
    tensors.sort()
    end = 0
    for first, last, name in tensors:
        if first != end or last < first:
            raise ValueError(f"Invalid offsets for {name}: {path}")
        end = last
    if size is not None and end > size:
        raise ValueError(f"Truncated safetensors data: {path}")
    if size is not None and end < size:
        raise ValueError(f"Unexpected safetensors data size: {path}")
    return tensors


def _chunks(file, size):
    while size:
        chunk = file.read(min(size, 16 << 20))
        if not chunk:
            raise ValueError(f"Unexpected end of weight file: {file.name}")
        size -= len(chunk)
        yield chunk


def check_aligned(path, source, expected_hash=None, *, allow_legacy=True):
    """Check the layout and, when requested, hash the original header and tensor bytes."""
    with path.open("rb") as file:
        header, raw = _read_header(file)
        start = 8 + len(raw)
        size = os.fstat(file.fileno()).st_size
        meta = header.get("__metadata__", {})
        if (meta.get("h3_aligned") != str(PAGE) or meta.get("h3_source") != source.name
                or start % PAGE or size % PAGE):
            raise ValueError(f"Not a page-aligned copy of {source.name}: {path}")
        layout = _tensor_layout(header, path, size - start)
        if any(first % PAGE and not name.startswith(PAD_PREFIX) for first, _, name in layout):
            raise ValueError(f"Unaligned tensor data: {path}")
        if expected_hash is None:
            return

        original_text = meta.get(SOURCE_HEADER)
        if isinstance(original_text, str):
            original_raw = original_text.encode("utf-8")
            original = json.loads(original_raw)
            if not isinstance(original, dict):
                raise ValueError(f"Invalid original safetensors header: {path}")
        elif allow_legacy and source.is_file():
            with source.open("rb") as original_file:
                original, original_raw = _read_header(original_file)
        else:
            raise ValueError(f"Missing original header for SHA-256 verification: {path}. "
                             "Restore the original with fetch, then run align again.")

        alignment_keys = {"h3_aligned", "h3_source", SOURCE_HEADER}
        original_meta = original.get("__metadata__") or {}
        if (not isinstance(original_meta, dict)
                or {k: v for k, v in meta.items() if k not in alignment_keys}
                != {k: v for k, v in original_meta.items() if k not in alignment_keys}):
            raise ValueError(f"Checkpoint metadata mismatch: {path}")
        if any(name not in original and not name.startswith(PAD_PREFIX) for _, _, name in layout):
            raise ValueError(f"Unexpected tensors in aligned copy: {path}")
        digest = hashlib.sha256(struct.pack("<Q", len(original_raw)) + original_raw)
        for a, b, name in _tensor_layout(original, path):
            spec = header.get(name)
            if spec is None:
                raise ValueError(f"Missing tensor {name} in aligned copy: {path}")
            first, last = spec["data_offsets"]
            if first % PAGE or last - first != b - a or spec != dict(original[name], data_offsets=[first, last]):
                raise ValueError(f"Tensor layout mismatch for {name}: {path}")
            file.seek(start + first)
            for chunk in _chunks(file, b - a):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"SHA-256 mismatch: {path}")


def align_file(src, dst, *, expected_hash=None):
    """Write and verify a new aligned copy before replacing its destination."""
    src, dst = Path(src), Path(dst)
    with src.open("rb") as source:
        header, raw = _read_header(source)
        start = 8 + len(raw)
        source_stat = os.fstat(source.fileno())
        items = _tensor_layout(header, src, source_stat.st_size - start)
        if any(name.startswith(PAD_PREFIX) for _, _, name in items):
            raise ValueError(f"Reserved padding tensor names in original checkpoint: {src}")
        meta = dict(header.get("__metadata__", {}), h3_aligned=str(PAGE), h3_source=src.name)
        meta[SOURCE_HEADER] = raw.decode("utf-8")
        out, plan, offset = {"__metadata__": meta}, [], 0

        def pad():
            nonlocal offset
            size = -offset % PAGE
            if size:
                out[f"{PAD_PREFIX}{len(plan)}"] = {"dtype": "U8", "shape": [size],
                                                   "data_offsets": [offset, offset + size]}
                plan.append((None, size))
                offset += size

        for first, last, name in items:
            pad()
            size = last - first
            out[name] = dict(header[name], data_offsets=[offset, offset + size])
            plan.append((start + first, size))
            offset += size
        pad()
        encoded = json.dumps(out, separators=(",", ":")).encode("utf-8")
        encoded += b" " * (-(8 + len(encoded)) % PAGE)
        if len(encoded) > MAX_HEADER_BYTES:
            raise ValueError(f"Aligned header exceeds the safetensors size limit: {dst}")

        digest = hashlib.sha256(struct.pack("<Q", len(raw)) + raw)
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=dst.parent, prefix=f".{dst.name}.", suffix=".tmp", delete=False) as target:
                temp = Path(target.name)
                target.write(struct.pack("<Q", len(encoded)))
                target.write(encoded)
                for position, size in plan:
                    if position is None:
                        target.write(b"\0" * size)
                    else:
                        source.seek(position)
                        for chunk in _chunks(source, size):
                            digest.update(chunk)
                            target.write(chunk)
                if expected_hash is not None and digest.hexdigest() != expected_hash:
                    raise ValueError(f"SHA-256 mismatch: {src}")
                target.flush()
                os.fchmod(target.fileno(), source_stat.st_mode & 0o666)
                os.fsync(target.fileno())
            check_aligned(temp, src, digest.hexdigest())
            temp.replace(dst)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
    return dst


def align_models(directory, precision="int8", keep_original=False):
    """Verify each aligned checkpoint before removing its original."""
    from .models import ALIGNABLE, manifest_for, sha256

    directory = Path(directory).expanduser().resolve()
    entries = manifest_for(precision)["files"]
    done = []
    with (directory / ".h3-fetch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for index in ALIGNABLE:
            entry = entries[index]
            src = directory / entry["path"]
            dst = aligned_name(src)
            try:
                check_aligned(dst, src, entry["sha256"], allow_legacy=False)
            except (OSError, ValueError):
                if not src.is_file():
                    raise
                align_file(src, dst, expected_hash=entry["sha256"])
            else:
                if src.is_file() and not keep_original and sha256(src) != entry["sha256"]:
                    raise ValueError(f"SHA-256 mismatch: {src}")
            if src.is_file() and not keep_original:
                src.unlink()
            done.append(str(dst))
    return "\n".join(done)
