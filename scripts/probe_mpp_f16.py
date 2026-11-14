"""Run FP16-input/FP32-output Metal 4 matmul; all H3 partials remain exact."""
import time, statistics
from pathlib import Path
import numpy as np, mlx.core as mx
from h3_image_mlx.core import KERNELS
from h3_image_mlx._portable import portable_matmul
s=(KERNELS/'w8a8_gemm_body.metal').read_text().replace('int8_t','half').replace('int32_t','float')
a=s.index('for (int k0 = 0;');s=s[:a]+s[a:].replace('for (int k0 = 0; k0 < K;','int first = int(threadgroup_position_in_grid.z) * 1024;\nfor (int k0 = first; k0 < min(first + 1024, K);')
s+='''\nfor(uint16_t i=0;i<cT.get_capacity();++i){
auto ids=cT.get_multidimensional_index(i);
int n=n0+ids[0],m=m0+ids[1];
if(m<M&&n<N)y[(size_t(threadgroup_position_in_grid.z)*M+m)*N+n]=int(cT[i]);}
'''
kern=mx.fast.metal_kernel(name='mpp_half_exact',input_names=['xq','wq'],output_names=['y'],header=(KERNELS/'mpp_tensor_ops.metal').read_text(),source=s)
def op(x,w,s,z,d,bm,bn):
    n=len(w);m,k=x.shape; chunks=(k+1023)//1024; cols=[]
    a=x.astype(mx.float16)
    for c in range(0,n,2048):
        b=w[c:c+2048].astype(mx.float16);width=len(b)
        v=kern(inputs=[a,b],template=[('BM',bm),('BN',bn)],grid=((width+bn-1)//bn*128,(m+bm-1)//bm,chunks),threadgroup=(128,1,1),output_shapes=[(chunks,m,width)],output_dtypes=[mx.int32])[0]
        y=(v.sum(0).astype(mx.float32)*s[:,None]*z[c:c+width]).astype(d);mx.eval(y);cols.append(y)
    return mx.concatenate(cols,1)
rng=np.random.default_rng(42)
print(mx.device_info(),flush=True)
for m,n,k in ((32,25600,5120),(320,21504,5376),(1100,21504,5376),(1100,5376,14336)):
    args=[mx.array(rng.integers(-128,128,(v,k),dtype=np.int8)) for v in (m,n)]+[mx.ones((m,)),mx.ones((n,)),mx.float32]
    mx.eval(*args[:-1]);ref=portable_matmul(*args);mx.eval(ref)
    for tile in [(),(32,32),(64,32),(64,64),(128,64)]:
        ts=[]
        for i in range(3):
            st=time.perf_counter();out=op(*args,*tile) if tile else portable_matmul(*args);mx.eval(out);ts.append((time.perf_counter()-st)*1000)
        assert mx.array_equal(out,ref).item()
        print(m,n,k,tile, round(statistics.mean(ts[1:]),2),flush=True)
