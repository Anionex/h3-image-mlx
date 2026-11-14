import subprocess,importlib.util,tempfile,time,statistics
from pathlib import Path
import mlx.core as mx,numpy as np
from h3_image_mlx._portable import portable_matmul
source=subprocess.check_output(["git","show","HEAD^:src/h3_image_mlx/_portable.py"])
base={}
# Use the installed baseline kernel resource for small row counts.
import h3_image_mlx._portable as impl
base["__file__"]=impl.__file__
exec(source,base)
rng=np.random.default_rng(7)
for m,n,k in ((1100,21504,5376),(1100,5376,14336)):
 for std in (16,30,80,1000):
    x=mx.array(np.clip(np.rint(rng.normal(0,std,(m,k))),-128,127).astype('i1'))
    w=mx.array(rng.integers(-128,128,(n,k),dtype=np.int8))
    args=(x,w,mx.ones((m,)),mx.ones((n,)),mx.float32)
    mx.eval(*args[:-1]);ref=base['portable_matmul'](*args);mx.eval(ref); times=[[],[]]
    for i in range(4):
     for j in (i%2,1-i%2):
        start=time.perf_counter();y=(base['portable_matmul'],portable_matmul)[j](*args);mx.eval(y);times[j].append((time.perf_counter()-start)*1000)
        if not mx.array_equal(y,ref).item():raise RuntimeError("Mismatch")
    print(m,n,k,std,[round(statistics.median(t),2) for t in times],flush=True)
