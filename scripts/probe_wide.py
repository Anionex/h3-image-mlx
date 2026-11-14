"""Compare exact wide GEMMs with the split-K baseline on the Metal runner."""
import time
import statistics
import numpy as np
import mlx.core as mx
from h3_image_mlx._portable import portable_matmul

def op(x,w,s,z,d,bits,piece):
    rows,width=x.shape; n=w.shape[0]; cols=[]
    for c in range(0,n,1024):
        ws=w[c:c+1024];t=None
        for k in range(0,width,piece):
            a=x[:,k:k+piece].astype(mx.float32);b=ws[:,k:k+piece]
            if bits:
                packed=b.view(mx.uint32)^mx.array(0x80808080,dtype=mx.uint32)
                ones=mx.ones((len(b),a.shape[-1]//64))
                sub=mx.quantized_matmul(a,packed,ones,-128*ones,group_size=64,bits=8)
            else:sub=a@b.astype(mx.float32).T
            sub=sub.astype(mx.int32)
            t=sub if t is None else t+sub
        y=(t.astype(mx.float32)*s[:,None]*z[c:c+1024]).astype(d)
        mx.eval(y);cols.append(y)
    return mx.concatenate(cols,1)

rng=np.random.default_rng(17)
for rows,n,k in [(320,21504,5376),(1100,21504,5376),(1100,5376,14336)]:
    args=[mx.array(np.clip(np.rint(rng.normal(0,std,(v,k))),-128,127).astype('i1')) for v,std in ((rows,23),(n,28))]+[mx.ones((rows,)), mx.ones((n,)),mx.float32]
    mx.eval(*args[:-1]); ref=portable_matmul(*args);mx.eval(ref)
    for bits,piece in [(0,0),(0,4096),(0,k),(8,4096),(8,k)]:
        ts=[]
        for i in range(4):
            start=time.perf_counter()
            v=op(*args,bits,piece) if piece else portable_matmul(*args)
            mx.eval(v);ts.append((time.perf_counter()-start)*1000)
        print(rows,n,k,bits,piece,round(statistics.median(ts[1:]),2),mx.array_equal(ref,v).item(),flush=True)
