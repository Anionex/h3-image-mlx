"""Bounded safetensors reads and block execution for MLX denoisers."""

import fcntl
import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten


_PAGE = os.sysconf("SC_PAGESIZE")
_READ_CHUNK = 8 << 20
_DTYPES = {
    "BOOL": "?", "I8": "i1", "U8": "u1", "I16": "<i2", "U16": "<u2",
    "I32": "<i4", "U32": "<u4", "I64": "<i8", "U64": "<u8",
    "F16": "<f2", "BF16": "<u2", "F32": "<f4",
}


def enable_uncached(file):
    if not hasattr(fcntl, "F_NOCACHE") or not hasattr(os, "preadv"):
        raise RuntimeError("Uncached weights require aligned macOS positional reads")
    fcntl.fcntl(file.fileno(), fcntl.F_NOCACHE, 1)
    fcntl.fcntl(file.fileno(), fcntl.F_RDAHEAD, 0)


def _aligned_size(length, offset):
    return ((offset % _PAGE + length + _PAGE - 1) // _PAGE) * _PAGE


def aligned_storage(length, offset):
    return _aligned_size(length, offset) + _PAGE - 1 if length else 0


def read_aligned(fd, length, offset, *, meter=None):
    """Read from page-aligned offsets into owned, page-aligned storage."""
    if not length:
        return np.empty(0, dtype=np.uint8)
    skip = offset % _PAGE
    size = _aligned_size(length, offset)
    storage = np.empty(size + _PAGE - 1, dtype=np.uint8)
    start = -storage.ctypes.data % _PAGE
    data = storage[start:start + size]
    done = 0
    while done < skip + length:
        end = min(done + _READ_CHUNK, size)
        count = os.preadv(fd, [data[done:end]], offset - skip + done)
        done += count
        if count == 0 or (done < skip + length and done % _PAGE != 0):
            raise EOFError("Unexpected end of aligned weight read")
    if meter is not None:
        meter.record(length, done)
    return data[skip:skip + length]


def read_buffered(fd, length, offset, *, meter=None):
    data = os.pread(fd, length, offset)
    if len(data) != length:
        raise EOFError("Unexpected end of weight file")
    if meter is not None:
        meter.record(length, 0)
    return data


def read_header(file, read=read_buffered):
    fd = file.fileno()
    (length,) = struct.unpack("<Q", read(fd, 8, 0))
    if length > 100_000_000 or 8 + length > os.fstat(fd).st_size:
        raise ValueError(f"Invalid safetensors header length in {file.name}")
    header = json.loads(bytes(read(fd, length, 8)))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header in {file.name}")
    return header, 8 + length


class ReadMeter:
    """Count payload and actual aligned read bytes across the calling and I/O threads."""

    def __init__(self):
        self._data = dict(calls=0, logical_bytes=0, aligned_bytes=0)
        self._lock = Lock()

    def record(self, logical, aligned):
        with self._lock:
            self._data["calls"] += 1
            self._data["logical_bytes"] += logical
            self._data["aligned_bytes"] += aligned

    def snapshot(self):
        with self._lock:
            return self._data.copy()


class TensorStore:
    """Index one or more safetensors files without mapping or loading their tensors."""

    def __init__(self, paths, *, uncached=False):
        if isinstance(paths, (str, os.PathLike)):
            paths = (paths,)
        self._files = ExitStack()
        self._location = {}
        self._meter = ReadMeter()
        self.uncached = uncached
        try:
            for path in paths:
                self._index(Path(path))
        except BaseException:
            self.close()
            raise

    def _index(self, path):
        file = self._files.enter_context(path.open("rb", buffering=0))
        if self.uncached:
            enable_uncached(file)
        header, start = read_header(file, self._read)
        fd = file.fileno()
        extent = os.fstat(fd).st_size - start
        for key, spec in header.items():
            if key == "__metadata__":
                continue
            if key in self._location:
                raise ValueError(f"Duplicate tensor {key} in {path}")
            begin, end = spec["data_offsets"]
            if not 0 <= begin <= end <= extent:
                raise ValueError(f"Invalid offsets for {key} in {path}")
            self._location[key] = fd, start + begin, end - begin, spec

    def _read(self, fd, length, offset):
        read = read_aligned if self.uncached else read_buffered
        return read(fd, length, offset, meter=self._meter)

    @property
    def io(self):
        return self._meter.snapshot()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._files.close()
        self._location.clear()

    def __iter__(self):
        return iter(self._location)

    def __contains__(self, key):
        return key in self._location

    def keys(self):
        return self._location.keys()

    def spec(self, key):
        return self._location[key][3]

    def size(self, keys):
        """Logical bytes of the selected tensors."""
        return sum(self._location[key][2] for key in dict.fromkeys(keys))

    def read_cost(self, keys):
        """Upper bound on the storage retained by a raw read."""
        if not self.uncached:
            return self.size(keys)
        sizes = (self._location[key][1:3] for key in dict.fromkeys(keys))
        return sum(aligned_storage(length, offset) for offset, length in sizes)

    def read(self, keys):
        locations = ((key, self._location[key][:3]) for key in dict.fromkeys(keys))
        return {key: self._read(fd, length, offset) for key, (fd, offset, length) in locations}

    def arrays(self, batch):
        arrays = {}
        for key, buffer in batch.items():
            spec = self.spec(key)
            dtype = spec["dtype"]
            if dtype not in _DTYPES:
                raise ValueError(f"Unsupported tensor dtype {dtype} for {key}")
            view = np.frombuffer(buffer, _DTYPES[dtype]).reshape(spec["shape"])
            if isinstance(buffer, np.ndarray) and view.size and view.flags.aligned:
                array = mx.from_dlpack(view)
            else:
                array = mx.array(view)
            arrays[key] = array.view(mx.bfloat16) if dtype == "BF16" else array
        return arrays

    def tensor(self, key):
        return self.arrays(self.read((key,)))[key]


@dataclass(frozen=True)
class StreamPolicy:
    """Preserve block ordering and select how CPU reads overlap device evaluation."""

    overlap: bool | None = None
    prefetch_bytes: int = 512 << 20

    def __post_init__(self):
        if self.prefetch_bytes < 0:
            raise ValueError("prefetch_bytes must be non-negative")

    def use_overlap(self):
        if self.overlap is not None:
            return self.overlap
        return self.prefetch_bytes > 0 and mx.default_device() == mx.gpu and mx.metal.is_available()


class Prefetch:
    """A single I/O worker and a byte budget for pending raw buffers."""

    def __init__(self, budget):
        self.budget = budget
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weight-prefetch")
        self._pending = {}
        self.reserved = 0
        self.peak_reserved = 0

    def submit(self, key, cost, read):
        if key in self._pending or cost > self.budget - self.reserved:
            return False
        self._pending[key] = self._pool.submit(read), cost
        self.reserved += cost
        self.peak_reserved = max(self.peak_reserved, self.reserved)
        return True

    def future(self, key):
        existing = self._pending.get(key)
        return existing[0] if existing else None

    def retain(self, keys):
        for key in tuple(self._pending):
            if key in keys:
                continue
            future, cost = self._pending.pop(key)
            if not future.cancel():
                future.exception()
            self.reserved -= cost

    def take(self, key):
        if key not in self._pending:
            return None
        future, cost = self._pending.pop(key)
        try:
            return future.result()
        finally:
            self.reserved -= cost

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()
        self.reserved = 0


class EvaluationWindow:
    """Evaluate each boundary while retaining at most one earlier async result."""

    def __init__(self, overlap):
        self.overlap = overlap
        self._previous = None

    def submit(self, value):
        if not self.overlap:
            mx.eval(value)
            return
        mx.async_eval(value)
        if self._previous is not None:
            mx.eval(self._previous)
        self._previous = value

    def finish(self):
        if self._previous is not None:
            mx.eval(self._previous)
            self._previous = None


class BlockStream:
    """Schedule named blocks while leaving their forward functions and sampler intact."""

    def __init__(self, store, policy=None, *, wrap=False):
        self.store = store
        self.policy = policy or StreamPolicy()
        overlap = self.policy.use_overlap()
        self._prefetch = Prefetch(self.policy.prefetch_bytes) if overlap and self.policy.prefetch_bytes else None
        self._evaluation = EvaluationWindow(overlap)
        self._block_ids = []
        self._blocks = {}
        self._wrap = wrap
        self._started = False
        self.metrics = {
            "overlap": overlap,
            "uncached": store.uncached,
            "blocks": 0,
            "weight_read_bytes": 0,
            "weight_read_s": 0.0,
            "weight_wait_s": 0.0,
            "prefetch_budget_bytes": self.policy.prefetch_bytes,
            "prefetch_peak_bytes": 0,
            "prefetch_hits": 0,
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self._evaluation.finish()
        finally:
            if self._prefetch is not None:
                self._prefetch.close()

    def register(self, block_id, mapping):
        if self._started or block_id in self._blocks:
            raise ValueError(f"Cannot register block {block_id} at this point")
        mapping = dict(mapping)
        keys = tuple(dict.fromkeys(mapping.values()))
        weight_bytes = self.store.size(keys)
        cost = self.store.read_cost(keys)
        self._blocks[block_id] = len(self._block_ids), mapping, keys, weight_bytes, cost
        self._block_ids.append(block_id)

    def _read(self, keys):
        began = time.perf_counter()
        return self.store.read(keys), time.perf_counter() - began

    def run(self, block_id, forward):
        self._started = True
        index, mapping, keys, weight_bytes, _ = self._blocks[block_id]
        fetched = None
        if self._prefetch is not None:
            now = time.perf_counter()
            fetched = self._prefetch.take(block_id)
            if fetched is not None:
                self.metrics["prefetch_hits"] += 1
                self.metrics["weight_wait_s"] += time.perf_counter() - now
            if index + 1 < len(self._block_ids) or self._wrap:
                next_id = self._block_ids[(index + 1) % len(self._block_ids)]
                _, _, next_keys, _, cost = self._blocks[next_id]
                self._prefetch.retain((next_id,))
                self._prefetch.submit(next_id, cost, lambda: self._read(next_keys))
                self.metrics["prefetch_peak_bytes"] = self._prefetch.peak_reserved
            else:
                self._prefetch.retain(())
        if fetched is None:
            fetched = self._read(keys)
        batch, read_s = fetched
        self.metrics["blocks"] += 1
        self.metrics["weight_read_bytes"] += weight_bytes
        self.metrics["weight_read_s"] += read_s
        arrays = self.store.arrays(batch)
        result = forward({name: arrays[key] for name, key in mapping.items()})
        self._evaluation.submit(result)
        return result

    def wrap(self, block, block_id, mapping):
        """Replace mapped MLX parameters with empty placeholders between forwards."""
        params = dict(tree_flatten(block.parameters()))
        for name, key in mapping.items():
            if tuple(self.store.spec(key)["shape"]) != params[name].shape:
                raise ValueError(f"Tensor {key} does not match the shape of parameter {name}")
        empty = tree_unflatten([(name, mx.zeros((0,), dtype=params[name].dtype)) for name in mapping])
        self.register(block_id, mapping)
        block.update(empty)

        def forward(*args, **kwargs):
            def invoke(weights):
                block.update(tree_unflatten(list(weights.items())))
                try:
                    return block(*args, **kwargs)
                finally:
                    block.update(empty)

            return self.run(block_id, invoke)

        return forward
