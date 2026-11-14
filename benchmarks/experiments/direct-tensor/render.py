import argparse, hashlib, json, os, subprocess, sys, threading, time
from pathlib import Path
root=Path.cwd()
exp=root/"benchmarks/experiments/direct-tensor"
import audit as a
p=argparse.ArgumentParser()
p.add_argument('tree');p.add_argument('trial');p.add_argument('size',type=int)
args=p.parse_args()
dest=root/'results'/args.trial;dest.mkdir(parents=True)
weights=Path(os.environ['RUNNER_TEMP'])/'h3-models';files=sorted(weights.rglob('*.safetensors'))
def residency(): return a.file_residency(files)['resident_bytes']
samples=[residency()]
if samples[0]>16*2**20: raise RuntimeError(f'file cache already warm: {samples}')
done=threading.Event()
def monitor():
    while not done.wait(3):samples.append(residency())
t=threading.Thread(target=monitor);t.start()
env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}
env.update(PYTHONPATH=os.pathsep.join(map(str,[root/args.tree/'src',exp/'instrument'])),IMAGE_MLX_UNCACHED_ROOTS=str(weights),DYLD_INSERT_LIBRARIES=str(exp/'libstrict.dylib'),H3_TRIAL_DIR=str(dest))
cmd=[sys.executable,'-m','h3_image_mlx.cli','generate','--prompt','A photographic study of a blue porcelain teapot on a wooden table.','--models',str(weights),'--backend','portable','--dit-mode','stream','--width',str(args.size),'--height',str(args.size),'--seed','42','--output',str(dest/'image.png')]
try:
    disk=a.disk_bytes();start=time.perf_counter()
    with (dest/'log').open('w') as log: subprocess.run(cmd,env=env,stdout=log,stderr=log,timeout=900,check=True)
    wall=time.perf_counter()-start;disk=a.disk_bytes()-disk
finally: done.set();t.join()
samples.append(residency())
stages={}
for f in dest.glob('*.io'):
    calls,logical,aligned,errors=map(int,f.read_text().split())
    meta=json.loads(f.with_suffix('.json').read_text());stage=meta['argv'][1]
    stages[stage]=dict(calls=calls,logical=logical,aligned=aligned,errors=errors,**meta)
ok=set(stages)=={'dit','text','vae'} and max(samples)<=16*2**20 and all(s['errors']==0 and s['disk_read_bytes']>=.95*s['aligned'] for s in stages.values()) and disk>=.95*sum(s['aligned'] for s in stages.values())
report=dict(tree=args.tree,size=args.size,wall=wall,physical=disk,residency=samples,stages=stages,accepted=ok,sha256=hashlib.sha256((dest/'image.png').read_bytes()).hexdigest())
(dest/'result.json').write_text(json.dumps(report,indent=2))
print(wall,ok,[(k,round(s['elapsed'],2),s['disk_read_bytes']/s['aligned']) for k,s in stages.items()],max(samples),flush=True)
if not ok:sys.exit(2)
