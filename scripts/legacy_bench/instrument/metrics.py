"""macOS process accounting shared by both controlled benchmark workers."""
import ctypes
import json
import os
import time
from pathlib import Path

LIB = ctypes.CDLL('/usr/lib/libproc.dylib')

def footprint(pid=None):
    buf = ctypes.create_string_buffer(16 + 35 * 8)
    if LIB.proc_pid_rusage(pid or os.getpid(), 4, ctypes.byref(buf)):
        return None
    get = lambda i: ctypes.c_uint64.from_buffer(buf, 16 + 8*i).value
    return {'footprint_bytes': get(7), 'lifetime_peak_bytes': get(28),
            'resident_bytes': get(6), 'disk_read_bytes': get(16)}

def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def event(directory, kind, **fields):
    row = dict(event=kind, wall=time.time(), monotonic=time.monotonic(), pid=os.getpid(), **fields)
    with (Path(directory)/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)
    return row
