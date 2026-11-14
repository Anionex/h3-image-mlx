import time,statistics,mlx.core as mx,numpy as np
from pathlib import Path
from h3_image_mlx._portable import portable_matmul
src=Path(__file__).with_suffix(".metal").read_text()
def kernel(dtype):
    body=src if dtype=="half" else src.replace("half","float")
    return mx.fast.metal_kernel(name="split_h3_"+dtype,input_names=["x","w"],output_names=["y"],header="#include <metal_simdgroup_matrix>\n",source=body)
kernels={t:kernel(t) for t in ("half","float")}
def op(x,w,s,z,d,bm,bn,bk,ck,t):
    m,k=x.shape;n=len(w);chunks=(k+ck-1)//ck
    # Bound the temporary reduction planes by tiling output channels.
    out=[]
    for c in range(0,n,2048):
        b=w[c:c+2048];width=len(b)
        y=kernels[t](inputs=[x,b],template=[("BM",bm),("BN",bn),("BK",bk),("CHUNK",ck)],grid=(width//bn*128,(m+bm-1)//bm,chunks),threadgroup=(128,1,1),output_shapes=[(chunks,m,width)],output_dtypes=[mx.int32])[0]
        v=(y.sum(0).astype(mx.float32)*s[:,None]*z[c:c+width]).astype(d);mx.eval(v);out.append(v)
    return mx.concatenate(out,1)
rng=np.random.default_rng(42)
for m,n,k in [(320,21504,5376),(1100,21504,5376),(1100,5376,14336)]:
    args=[mx.array(rng.integers(-128,128,(v,k),dtype=np.int8)) for v in (m,n)]+[mx.ones((m,)),mx.ones((n,)),mx.float32]
    mx.eval(*args[:-1]);ref=portable_matmul(*args);mx.eval(ref)
    for tile in [()]+[(bm,64,32,ck,t) for bm in (32,64) for ck in (256,1024) for t in ('half','float')]:
        ts=[]
        for _ in range(3):
            st=time.perf_counter();v=op(*args,*tile) if tile else portable_matmul(*args);mx.eval(v);ts.append((time.perf_counter()-st)*1000)
        print(m,n,k,tile,round(statistics.mean(ts[1:]),2),mx.array_equal(v,ref).item(),flush=True)
