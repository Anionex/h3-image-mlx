from cert_dot import checked_wide
import statistics,time
import mlx.core as mx,numpy as np
from h3_image_mlx._portable import portable_matmul
rng=np.random.default_rng(1234)
for k,n in ((5376,21504),(7168,5376),(14336,5376)):
    m=1100
    x,w=[mx.array(np.clip(np.rint(rng.normal(0,s,(r,k))),-128,127).astype('i1')) for r,s in [(m,20),(n,28)]]
    scales=mx.ones((m,1)); z=mx.ones((n,)); mx.eval(x,w,scales,z)
    args=(x,w,scales,z,mx.float32);ref=portable_matmul(*args);mx.eval(ref)
    for packed in (True,False):
     for tile in (1024,4096,8192):
      ts=[[],[]]
      for i in range(3):
       for j in (i%2,1-i%2):
        start=time.perf_counter();v=checked_wide(*args,tile=tile,packed=packed) if j else portable_matmul(*args)
        mx.eval(v);ts[j].append((time.perf_counter()-start)*1000)
        assert mx.array_equal(v,ref).item()
      print("cert",k,n,packed,tile,[round(statistics.median(t),2) for t in ts],flush=True)
