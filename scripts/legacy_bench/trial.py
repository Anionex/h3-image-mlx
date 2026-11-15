import argparse, hashlib, json, os, subprocess, sys, threading, time
from pathlib import Path
import platform
import re
import signal
from importlib.metadata import distribution, version
import cache_stats as a
import mlx.core as mx
here = Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--weights',type=Path,required=True)
p.add_argument('--result',type=Path,required=True)
p.add_argument('--arm',choices=['baseline','candidate'],required=True)
p.add_argument('--proxy',action='store_true')
p.add_argument('--timeout',type=int,default=1800)
args=p.parse_args()
arm=args.arm
weights=args.weights.resolve(strict=True)
manifest_path = distribution('h3-image-mlx').locate_file('h3_image_mlx/weights.json')
manifest = json.loads(manifest_path.read_text())
files = []
for entry in manifest['files']:
    if entry['path'].endswith('.safetensors'):
        path = weights / entry['path']
        if path.stat().st_size != entry['bytes']:
            raise ValueError(f"Weight file size differs from the release manifest: {path}")
        files.append(path.resolve(strict=True))
if not mx.metal.is_available():
    raise RuntimeError("Provide a local H3 weights directory and a working Metal device")
if version('h3-image-mlx') != '0.1.0a8' or version('mlx') != '0.32.4':
    raise RuntimeError("The controlled comparison requires h3-image-mlx 0.1.0a8 and MLX 0.32.4")
cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
system=platform.mac_ver()[0]
if not args.proxy:
    if not re.match(r'^Apple M[1-4]\b', cpu) or tuple(map(int,system.split('.')[:2])) >= (26,4):
        raise RuntimeError(f"Expected an M1–M4 Mac below macOS 26.4. Detected {cpu}, macOS {system}")
if a.disk_bytes() <= 0:
    raise RuntimeError("macOS physical disk-read counters are unavailable on this host")
dest=args.result.resolve()
dest.mkdir(parents=True)
sources=args.candidate.resolve(strict=True)/'src'
if not (sources/'h3_image_mlx/_portable.py').is_file():
    raise ValueError(f"Missing H3 portable backend in {sources}")
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
env.update(PYTHONPATH=os.pathsep.join(paths),IMAGE_MLX_UNCACHED_ROOTS='\n'.join([str(weights), *(str(f) for f in files if not f.is_relative_to(weights))]),
           DYLD_INSERT_LIBRARIES=os.environ.get('H3_BENCH_DYLIB',str(here/'audit.dylib')),H3_TRIAL_DIR=str(dest),
           H3_BENCH_ARM=arm,H3_LEGACY_PROXY=str(int(args.proxy)))
cmd=[sys.executable,'-m','h3_image_mlx.cli','generate','--prompt','A photographic study of a blue porcelain teapot on a wooden table.','--models',str(weights),'--backend','portable','--dit-mode','stream','--width','1024','--height','1024','--seed','42','--output',str(dest/'image.png')]
try:
    disk=a.disk_bytes();start=time.perf_counter()
    with (dest/'log').open('w') as log:
        child = subprocess.Popen(cmd, env=env, stdout=log, stderr=log, start_new_session=True)
        try:
            if child.wait(timeout=args.timeout):
                raise RuntimeError(f"Generation failed. See {dest / 'log'}")
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            raise
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
ram=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))
report=dict(cpu=cpu,unified_memory_bytes=ram,gpu=mx.device_info()['device_name'],mlx=version('mlx'),macos=system,proxy=args.proxy,
            candidate_calls=stages['dit']['candidate_calls'],arm=arm,size=1024,wall=wall,physical=disk,residency=samples,monitor_errors=monitor_errors,
            sample_times_s=[v-sample_times[0] for v in sample_times],stages=stages,accepted=ok,sha256=hashlib.sha256((dest/'image.png').read_bytes()).hexdigest())
(dest/'result.json').write_text(json.dumps(report,indent=2))
print(wall,ok,[(k,round(s['elapsed'],2),s['disk_read_bytes']/s['aligned']) for k,s in stages.items()],max(samples),flush=True)
if not ok:sys.exit(2)
