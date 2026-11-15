"""Fetch exactly the a8 manifest inside a disposable GitHub-hosted macOS VM."""
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import time

manifest = json.loads(Path('src/h3_image_mlx/weights.json').read_text())
destination = Path(sys.argv[1]).resolve()
temp = Path(os.environ['RUNNER_TEMP']).resolve()
assert destination.parent == temp and os.environ['RUNNER_ENVIRONMENT'] == 'github-hosted'
required = sum(f['bytes'] for f in manifest['files']) + 2 * 1024**3
# Only release other Xcode installations in this disposable VM if space is short.
active = Path(subprocess.check_output(['xcode-select','-p'],text=True).strip()).resolve()
for install in sorted(Path('/Applications').glob('Xcode_*.app')):
    if shutil.disk_usage(temp).free >= required:
        break
    if install.is_symlink() or active.is_relative_to(install.resolve()):
        continue
    print('Freeing an unused runner toolchain:', install, flush=True)
    subprocess.run(['sudo','-n','rm','-rf','--',str(install)],check=True)
free = shutil.disk_usage(temp).free
print('Free/required bytes:', free, required, flush=True)
assert free >= required
largest = max(manifest['files'], key=lambda f: f['bytes'])
# Check that mirrors honor ranges and can finish within the job's download window.
rates = {}
for source in ('modelscope','huggingface'):
    started = time.monotonic()
    result = subprocess.run(['curl','-sSfL','--max-time','15','--range','0-8388607',
                             '--max-filesize','8388608','-o','/dev/null','-w','%{http_code} %{size_download}',
                             largest[source]],capture_output=True,text=True)
    if result.returncode == 0 and result.stdout.split()==['206','8388608']:
        rates[source] = 8388608 / (time.monotonic() - started)
print('Sampled mirror bytes/sec:', rates, flush=True)
ordered = sorted(rates,key=rates.get,reverse=True)
if rates.get('modelscope',0)>=30_000_000:
    ordered=['modelscope']+[s for s in ordered if s!='modelscope']
for source in ordered:
    try:
        subprocess.run(['h3-image-mlx','fetch','--source',source,'--models',str(destination)],
                       timeout=1350, check=True)
        break
    except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as err:
        print(source,type(err).__name__,flush=True)
else:
    raise SystemExit('Neither mirror completed a verified download')
