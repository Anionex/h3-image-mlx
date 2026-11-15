"""Paired legacy projection trials, using the installed a8 as the control."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('candidate', 'weights', 'results'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--pairs',type=int,choices=[1,2,3],default=3)
    parser.add_argument('--proxy',action='store_true')
    args = parser.parse_args()
    root = args.results.resolve()
    root.mkdir(parents=True)
    src = args.candidate.resolve() / 'src/h3_image_mlx'
    order = ['baseline','candidate','candidate','baseline','baseline','candidate'][:args.pairs*2]
    config = dict(order=order, weights=str(args.weights), proxy=args.proxy, size=1024,
                  source_sha256={str(f.relative_to(src)):hashlib.sha256(f.read_bytes()).hexdigest()
                                 for f in (src/'_portable.py', *sorted((src/'kernels').glob('portable*.metal')))})
    (root/'configuration.json').write_text(json.dumps(config,indent=2))
    rows=[]
    for number,arm in enumerate(order,1):
        dest=root/f'{number}-{arm}'
        cmd=[sys.executable,str(Path(__file__).with_name('trial.py')),'--arm',arm,
             '--candidate',str(args.candidate),'--weights',str(args.weights),'--result',str(dest)]
        if args.proxy: cmd.append('--proxy')
        subprocess.run(cmd,check=True)
        row=json.loads((dest/'result.json').read_text())
        row['peak_bytes']=max(s['lifetime_peak_bytes'] for s in row['stages'].values())
        rows.append(row)
        if not row['accepted'] or row['sha256']!=rows[0]['sha256']:
            raise ValueError(f'Trial failed verification: {dest}')
    summary={arm:{key:median(r[key] for r in rows if r['arm']==arm) for key in ('wall','peak_bytes')}
             for arm in set(order)}
    summary['latency_reduction']=1-summary['candidate']['wall']/summary['baseline']['wall']
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
