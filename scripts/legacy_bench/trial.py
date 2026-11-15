import argparse, hashlib, json, os, subprocess, sys, threading, time
from pathlib import Path
import platform
from importlib.metadata import version
import cache_stats as a
import mlx.core as mx
here = Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--weights',type=Path,required=True)
p.add_argument('--result',type=Path,required=True)
p.add_argument('--arm',choices=['baseline','candidate'],required=True)
p.add_argument('--proxy',action='store_true')
args=p.parse_args()
arm=args.arm
weights=args.weights.resolve(strict=True)
files=sorted(weights.rglob('*.safetensors'))
assert files and mx.metal.is_available() and version('h3-image-mlx')=='0.1.0a8'
cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
system=platform.mac_ver()[0]
if not args.proxy:
    assert 'Apple M1' in cpu and tuple(map(int,system.split('.')[:2]))<(26,4),(cpu,system)
dest=args.result.resolve()
dest.mkdir(parents=True)
sources=args.candidate.resolve(strict=True)/'src'
assert (sources/'h3_image_mlx/kernels/portable_reduce.metal').is_file()
def residency(): return a.file_residency(files)['resident_bytes']
monitor_errors=[]
samples=[residency()]
sample_times=[time.monotonic()]
if samples[0]>16*2**20: raise RuntimeError(f'file cache already warm: {samples}')
done=threading.Event()
def monitor():
    try:
        while not done.wait(3):
            samples.append(residency())
            sample_times.append(time.monotonic())
    except Exception as exc: monitor_errors.append(repr(exc))
t=threading.Thread(target=monitor);t.start()
env={k:v for k,v in os.environ.items() if not k.startswith('H3_')}
paths=[str(here/'instrument')]
if arm=='candidate':paths.append(str(sources))
env.update(PYTHONPATH=os.pathsep.join(paths),IMAGE_MLX_UNCACHED_ROOTS=str(weights),
           DYLD_INSERT_LIBRARIES=str(here/'audit.dylib'),H3_TRIAL_DIR=str(dest),
           H3_BENCH_ARM=arm,H3_LEGACY_PROXY=str(int(args.proxy)))
cmd=[sys.executable,'-m','h3_image_mlx.cli','generate','--prompt','A photographic study of a blue porcelain teapot on a wooden table.','--models',str(weights),'--backend','portable','--dit-mode','stream','--width','1024','--height','1024','--seed','42','--output',str(dest/'image.png')]
try:
    disk=a.disk_bytes();start=time.perf_counter()
    with (dest/'log').open('w') as log: subprocess.run(cmd,env=env,stdout=log,stderr=log,timeout=900,check=True)
    wall=time.perf_counter()-start;disk=a.disk_bytes()-disk
finally: done.set();t.join()
samples.append(residency())
sample_times.append(time.monotonic())
stages={}
for f in dest.glob('*.io'):
    calls,logical,aligned,read_errors=map(int,f.read_text().split())
    meta=json.loads(f.with_suffix('.json').read_text());stage=meta['argv'][1]
    stages[stage]=dict(calls=calls,logical=logical,aligned=aligned,errors=read_errors,**meta)
ok=not monitor_errors and set(stages)=={'dit','text','vae'} and max(samples)<=16*2**20 and all(s['errors']==0 and s['calls']>0 and s['aligned']>=s['logical']>0 and s['disk_read_bytes']>=.95*s['aligned'] for s in stages.values()) and disk>=.95*sum(s['aligned'] for s in stages.values())
ok=ok and ((stages['dit']['candidate_calls']>0) == (arm=='candidate'))
for stage in stages.values(): stage.pop('argv')
report=dict(cpu=cpu,gpu=mx.device_info()['device_name'],mlx=version('mlx'),macos=system,proxy=args.proxy,
            candidate_calls=stages['dit']['candidate_calls'],arm=arm,size=1024,wall=wall,physical=disk,residency=samples,monitor_errors=monitor_errors,
            sample_times_s=[v-sample_times[0] for v in sample_times],stages=stages,accepted=ok,sha256=hashlib.sha256((dest/'image.png').read_bytes()).hexdigest())
(dest/'result.json').write_text(json.dumps(report,indent=2))
print(wall,ok,[(k,round(s['elapsed'],2),s['disk_read_bytes']/s['aligned']) for k,s in stages.items()],max(samples),flush=True)
if not ok:sys.exit(2)
