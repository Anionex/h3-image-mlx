import sys
from pathlib import Path
import mlx.core as mx
from h3_image_mlx.core import tensorops_kernel
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bench_portable as bench

def mpp(x,w,s,z,d):
    rows,width=x.shape;n=w.shape[0]
    return tensorops_kernel()(inputs=[x,w,s,z],template=[("BM",128),("BN",64),("T",d)],grid=((n+63)//64*128,(rows+127)//128,1),threadgroup=(128,1,1),output_shapes=[(rows,n)],output_dtypes=[d])[0]
print(mx.device_info(),flush=True)
base=bench.portable_matmul
bench.portable_matmul=mpp
for shape in ((32,25600,5120),(320,21504,5376),(1024,21504,5376),(64,5120,25600)):
    bench.compare(base,*shape)
