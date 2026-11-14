import subprocess,importlib.util,tempfile,time,statistics
from pathlib import Path
import mlx.core as mx,numpy as np
from h3_image_mlx._portable import portable_matmul
source=subprocess.check_output(["git","show","2e82f83:src/h3_image_mlx/_portable.py"])
base={}
# Use the installed baseline kernel resource for small row counts.
import h3_image_mlx._portable as impl
base["__file__"]=impl.__file__
exec(source,base)
def packed_dot(x,w,s,z,d,chunk,tile):
    m,k=x.shape;n=len(w)
    a=x.astype(mx.float32)
    outs=[]
    for c in range(0,n,tile):
        end=min(c+tile,n)
        total=None
        for i in range(0,k,chunk):
            b=w[c:end,i:i+chunk]
            ones=mx.ones((len(b),b.shape[1]//64))
            packed=b.view(mx.uint32)^mx.array(0x80808080,mx.uint32)
            partial=mx.quantized_matmul(a[:,i:i+chunk],packed,ones,-128*ones,group_size=64,bits=8).astype(mx.int32)
            total=partial if total is None else total+partial
        y=(total.astype(mx.float32)*s[:,None]*z[c:end]).astype(d);mx.eval(y);outs.append(y)
    return mx.concatenate(outs,1)
def planes(x,w,s,z,d,tile):
    m,k=x.shape;n=len(w);chunks=k//1024
    a=x.astype(mx.float32)
    av=a[:,:chunks*1024].reshape(m,chunks,1024).transpose(1,0,2)
    outs=[]
    for c in range(0,n,tile):
        b=w[c:c+tile]; bv=b[:,:chunks*1024].reshape(len(b),chunks,1024).transpose(1,0,2)
        dots=av@bv.astype(mx.float32).swapaxes(1,2)
        total=dots.astype(mx.int32).sum(0)
        if k%1024:total=total+(a[:,chunks*1024:]@b[:,chunks*1024:].astype(mx.float32).T).astype(mx.int32)
        y=(total.astype(mx.float32)*s[:,None]*z[c:c+tile]).astype(d);mx.eval(y);outs.append(y)
    return mx.concatenate(outs,1)
methods={f"packet{tile}":lambda *args,tile=tile:packed_dot(*args,1024,tile) for tile in (1024,4096,8192)}
methods.update({f"planes{tile}":lambda *args,tile=tile:planes(*args,tile) for tile in (512,1024,2048,4096)})
rng=np.random.default_rng(7)
for m,n,k in ((1100,21504,5376),(1100,5376,14336)):
 for std in (16,40):
    x=mx.array(np.clip(np.rint(rng.normal(0,std,(m,k))),-128,127).astype('i1'))
    w=mx.array(rng.integers(-128,128,(n,k),dtype=np.int8))
    args=(x,w,mx.ones((m,)),mx.ones((n,)),mx.float32)
    mx.eval(*args[:-1]);ref=base['portable_matmul'](*args);mx.eval(ref); times=[[],[]]
    for name, candidate in methods.items():
     times=[[],[]]
     for i in range(3):
      for j in (i%2,1-i%2):
        start=time.perf_counter();y=(base['portable_matmul'],candidate)[j](*args);mx.eval(y);times[j].append((time.perf_counter()-start)*1000)
        if not mx.array_equal(y,ref).item():raise RuntimeError("Mismatch")
     print(m,n,k,std,name,[round(statistics.median(t),2) for t in times],flush=True)
