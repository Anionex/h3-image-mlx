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

