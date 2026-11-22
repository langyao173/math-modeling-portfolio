"""Low-CPU queue: missingness inference starts only after TabDPT releases CPU."""
from pathlib import Path
import json
import os
import subprocess
import time

HERE=Path(__file__).resolve().parent
OUT=HERE/'missingness'
OUT.mkdir(exist_ok=True)


def status(state,**extra):
    value=dict(state=state,timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra)
    path=OUT/'queue_status.json';temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value,indent=2));temp.replace(path)
    print(json.dumps(value),flush=True)


status('waiting_for_tabdpt_complete')
start=time.monotonic()
while True:
    source=HERE/'tabdpt/queue_status.json'
    if source.exists():
        state=json.loads(source.read_text())['state']
        if state=='all_done':break
        if state.startswith(('failed','blocked')):
            status('blocked_upstream_error',upstream_state=state)
            raise RuntimeError('Inspect the upstream benchmark before starting missingness computation.')
    if time.monotonic()-start>12*3600:
        status('blocked_wait_timeout')
        raise TimeoutError('Upstream modern benchmarks did not finish in twelve hours.')
    time.sleep(20)
env=os.environ.copy();env['BEER_MODERN_THREADS']='6'
status('running_numerical_probe_and_frozen_missingness')
with (OUT/'run.log').open('w') as stream:
    result=subprocess.run([str(HERE/'venv/bin/python'),str(HERE/'run_missingness_tabicl.py')],
                          env=env,stdout=stream,stderr=subprocess.STDOUT)
if result.returncode:
    status('failed_missingness',returncode=result.returncode)
    raise RuntimeError('Missingness benchmark failed; inspect log and resume checkpoints.')
final=json.loads((OUT/'status.json').read_text())
assert final['complete'] and final['completed_folds']==5
status('all_done',metrics=final['metrics'])
