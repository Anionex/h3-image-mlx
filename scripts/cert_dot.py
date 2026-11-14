import math
from functools import cache
import mlx.core as mx
@cache
def _norm_kernel():
    return mx.fast.metal_kernel(name="h3_group_sqnorm", input_names=["a"],output_names=["norm"],source="""
      uint lane=thread_position_in_threadgroup.x;
      uint start=threadgroup_position_in_grid.x*1024, row=threadgroup_position_in_grid.y;
      uint k=a_shape[1], batch=(k+1023)/1024;
      int sum=0;
      for (uint i=start+lane;i<min(k,start+1024u);i+=32) {
        int val=a[row*k+i];sum+=val*val;
      }
      int n=simd_sum(sum);
      if(lane==0) norm[row*batch+start/1024]=n;
    """)
def group_norm(v):
    r,k=v.shape;groups=(k+1023)//1024
    out=_norm_kernel()(inputs=[v],grid=(groups*32,r,1),threadgroup=(32,1,1),output_shapes=[(r,groups)],output_dtypes=[mx.int32])[0]
    return out.max(0)
def checked_wide(x,w,s,z,d):
    m,k=x.shape;n=len(w)
    av=group_norm(x);bv=group_norm(w);mx.eval(av,bv)
    # For every prefix or subset of each chunk, bound sum(abs(x*w)) by
    # sqrt(norm2(x)*norm2(w)); use an upper integer square root.
    spans=[];start=total=0
    for i,(a,b) in enumerate(zip(av.tolist(),bv.tolist())):
        prod=a*b
        bound=min(2**24, math.isqrt(prod-1)+1 if prod else 0)
        if total+bound>2**24:
            spans.append((start,i*1024))
            start,total=i*1024,0
        total+=bound
    spans.append((start,k))
    cols=[]
    for c in range(0,n,2048):
        b=w[c:c+2048]
        t=None
        for start,end in spans:
            part=x[:,start:end].astype(mx.float32)@b[:,start:end].astype(mx.float32).T
            if len(spans)==1:t=part
            else:
                part=part.astype(mx.int32)
                t=part if t is None else t+part
        out=(t.astype(mx.float32)*s*z[c:c+2048]).astype(d);mx.eval(out);cols.append(out)
    return mx.concatenate(cols,1)
