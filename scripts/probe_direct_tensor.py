from pathlib import Path
import sys,time,statistics
import mlx.core as mx,numpy as np
from h3_image_mlx._portable import portable_matmul
from h3_image_mlx.core import KERNELS
kind=sys.argv[1]
src="""
const int M=x_shape[0], N=w_shape[0], K=x_shape[1];
const int m=threadgroup_position_in_grid.y*BM, n=threadgroup_position_in_grid.x*BN;
constexpr auto desc=matmul2d_descriptor(BM,BN,static_cast<int>(dynamic_extent),false,true,false);
matmul2d<desc,execution_simdgroups<4>> op;
tensor<device S,dextents<int,2>,tensor_inline> A((device S*)x,dextents<int,2>(K,M));
tensor<device S,dextents<int,2>,tensor_inline> B((device S*)w,dextents<int,2>(K,N));
tensor<device D,dextents<int,2>,tensor_inline> C((device D*)y,dextents<int,2>(N,M));
auto a=A.template slice<dynamic_extent,BM>(0,m);
auto b=B.template slice<dynamic_extent,BN>(0,n);
auto c=C.template slice<BN,BM>(n,m);
op.run(a,b,c);
"""
kern=mx.fast.metal_kernel(name="probe_direct_"+kind,input_names=["x","w"],output_names=["y"],header=(KERNELS/'mpp_tensor_ops.metal').read_text(),source=src)
def op(x,w,s,z,d,bm,bn):
    m,k=x.shape;n=len(w);v=0
    for i in range(0,k,k if kind=='i8' else 1024):
        q=k if kind=='i8' else 1024
        st,dt=(mx.int8,mx.int32) if kind=='i8' else (mx.float16,mx.float32)
        a=x[:,i:i+q].astype(st);b=w[:,i:i+q].astype(st)
        partial=kern(inputs=[a,b],template=[('BM',bm),('BN',bn),('S',st),('D',dt)],grid=((n+bn-1)//bn*128,(m+bm-1)//bm,1),threadgroup=(128,1,1),output_shapes=[(m,n)],output_dtypes=[dt])[0]
        v=v+partial.astype(mx.int32)
    return (v.astype(mx.float32)*s[:,None]*z).astype(d)
rng=np.random.default_rng(44)
for m,n,k in [(128,128,2048),(1100,21504,5376),(1100,5376,14336)]:
    args=[mx.array(rng.integers(-128,128,(r,k),dtype=np.int8)) for r in (m,n)]+[mx.ones((m,)),mx.ones((n,)),mx.float32]
    mx.eval(*args[:-1]);ref=portable_matmul(*args);mx.eval(ref)
    for tile in [(),(64,32),(64,64),(128,64)]:
        ts=[]
        for j in range(3):
            st=time.perf_counter();v=op(*args,*tile) if tile else portable_matmul(*args);mx.eval(v);ts.append((time.perf_counter()-st)*1000)
        assert mx.array_equal(ref,v).item()
        print("direct",kind,m,n,k,tile,round(statistics.mean(ts[1:]),2),flush=True)
