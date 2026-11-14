import atexit, json, os, sys, time
from pathlib import Path
started=time.perf_counter()
def record():
    from metrics import footprint
    p=Path(os.environ['H3_TRIAL_DIR'])/f'{os.getpid()}.json'
    p.write_text(json.dumps(dict(argv=sys.argv,elapsed=time.perf_counter()-started,**footprint())))
atexit.register(record)
