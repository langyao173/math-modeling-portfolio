"""Resume clean policy-disjoint work after the documented namespace recovery.

No cross-namespace signals. The verified old dispatcher is already terminated.
This process owns both new policy workers in one PID namespace. The affected
or interrupted directories were archived and are never aggregated.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import json
import subprocess
import time
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'perspective'


def status(state,**extra):
    rec=dict(state=state,timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             recovery_record='handoff_recovery.json',**extra)
    p=OUT/'parallel_queue_status.json';t=p.with_suffix('.tmp.json')
    t.write_text(json.dumps(rec,indent=2));t.replace(p)
    print(json.dumps(rec),flush=True)


def aggregate():
    rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('*/beer_*/DONE.json'))]
    temp=OUT/'comparison.tmp.csv'
    pd.DataFrame(rows).to_csv(temp,index=False);temp.replace(OUT/'comparison.csv')
    return rows


for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        a=[v.decode(errors='replace') for v in p.joinpath('cmdline').read_bytes().split(b'\0') if v]
    except OSError:continue
    if len(a)>1 and Path(a[1]).name=='perspective_solver.py':
        raise RuntimeError('Old dispatcher still exists; refuse to launch another worker.')
    if len(a)>1 and Path(a[1]).name=='perspective_policy_worker.py' and '--policy' in a:
        raise RuntimeError('Another policy worker exists; refuse duplicate writes.')

assert (OUT/'handoff_recovery.json').exists()
assert not (OUT/'exceedance/beer_81').exists()
assert not (OUT/'exceedance/beer_218').exists()
workers={};streams=[]
for policy in ('strict','exceedance'):
    log=(OUT/f'{policy}_recovered_run.log').open('w');streams.append(log)
    workers[policy]=subprocess.Popen([str(HERE/'venv/bin/python'),str(HERE/'perspective_policy_worker.py'),
                                     '--policy',policy,'--seconds','180'],
                                    stdout=log,stderr=subprocess.STDOUT)
started=time.monotonic()
while True:
    rows=aggregate()
    count={policy:len(list((OUT/policy).glob('beer_*/DONE.json'))) for policy in workers}
    codes={policy:worker.poll() for policy,worker in workers.items()}
    status('running_clean_disjoint_workers_after_recovery',completed_cases=len(rows),
           cases_by_policy=count,worker_returncodes=codes)
    if any(code not in (None,0) for code in codes.values()):
        raise RuntimeError('Recovered policy worker failed; inspect preserved log.')
    if all(count[policy]==11 for policy in count) and all(code==0 for code in codes.values()):
        break
    if time.monotonic()-started>6*3600:
        raise TimeoutError('Clean policy workers did not complete within six hours.')
    time.sleep(15)
for log in streams:log.close()
assert len(rows)==22
done=dict(cases_completed=22,expanded=True,seconds_per_case=180,
          execution='Completed clean serial cases plus disjoint one-thread policy workers; overlap-affected and interrupted cases excluded and rerun from fresh original warm starts.',
          recovery_record='handoff_recovery.json',excluded_archive='excluded_overlap_recovery')
(OUT/'DONE.json').write_text(json.dumps(done,indent=2))
status('all_done',completed_cases=22)
