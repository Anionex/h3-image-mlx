import ctypes,os,subprocess,plistlib
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.mmap.restype = ctypes.c_void_p
LIBC.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
LIBC.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


def disk_bytes():
    data = plistlib.loads(subprocess.check_output(['ioreg', '-r', '-c', 'IOBlockStorageDriver', '-a'], timeout=20))
    return sum(x.get('Statistics', {}).get('Bytes (Read)', 0) for x in data)


def file_residency(paths):
    """mincore queries mapped file pages without faulting in their contents."""
    page = os.sysconf('SC_PAGESIZE')
    records = []
    for path in paths:
        size = path.stat().st_size
        with path.open('rb') as f:
            addr = LIBC.mmap(None, size, 1, 1, f.fileno(), 0)  # PROT_READ, MAP_SHARED
            if addr == ctypes.c_void_p(-1).value:
                raise OSError(ctypes.get_errno(), 'mmap')
            try:
                n = (size + page - 1) // page
                vec = (ctypes.c_ubyte * n)()
                if LIBC.mincore(addr, size, vec):
                    raise OSError(ctypes.get_errno(), 'mincore')
                resident = sum(bool(v & 1) for v in vec) * page
                records.append(dict(path=str(path), bytes=size, resident_bytes=min(size, resident)))
            finally:
                LIBC.munmap(addr, size)
    return dict(files=records, bytes=sum(r['bytes'] for r in records), resident_bytes=sum(r['resident_bytes'] for r in records))


