"""Safe resource handoff after the two original SCIP searches finish.

1. One free CPU runs a bounded SDP construction and a one-thread resource probe.
2. The old perspective dispatcher is stopped only after a case writes DONE.
3. Two policy-disjoint workers resume remaining cases, one CPU each.

Only this task's original perspective_solver.py process is ever signaled.
All completed cases are reused; any next-case initialization is archived.
"""
from pathlib import Path
import json
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import shutil
import signal
import subprocess
import time
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/'perspective'
MOD=HERE.parent/'modern_prediction'
PYTHON=str(HERE/'venv/bin/python')


def status(state,**extra):
    value=dict(state=state,timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra)
    path=OUT/'parallel_queue_status.json';temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value,indent=2));temp.replace(path)
    print(json.dumps(value),flush=True)


def dispatcher_pid():
    matches=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            argv=p.joinpath('cmdline').read_bytes().split(b'\0')
            if len(argv)<3:continue
            name=argv[1].decode(errors='replace')
            if Path(name).name!='perspective_solver.py':continue
            try:
                resolved=(Path('/proc')/p.name/'cwd').resolve().joinpath(name).resolve()
            except (OSError,PermissionError) as exc:
                raise RuntimeError('A perspective dispatcher exists but its namespace/cwd cannot be verified; abort handoff without archiving or starting workers.') from exc
            if resolved!=HERE/'perspective_solver.py':continue
            if b'--expand-if-better' in argv:matches.append(int(p.name))
        except (OSError,PermissionError):continue
    if len(matches)>1:raise RuntimeError('Ambiguous perspective dispatcher; no process will be signaled.')
    if matches:
        try:os.kill(matches[0],0)
        except ProcessLookupError as exc:
            raise RuntimeError('Dispatcher is visible in /proc but cannot be signaled from this PID namespace; abort handoff.') from exc
    return matches[0] if matches else None


def completion_paths():
    return set(OUT.glob('*/beer_*/DONE.json'))


def aggregate():
    rows=[json.loads(path.read_text()) for path in sorted(completion_paths())]
    temp=OUT/'comparison.tmp.csv'
    pd.DataFrame(rows).to_csv(temp,index=False);temp.replace(OUT/'comparison.csv')
    return rows


status('waiting_for_original_22_cases')
start=time.monotonic()
while any(len(list((HERE/policy).glob('beer_*/DONE.json')))<11 for policy in ('strict','exceedance')):
    if time.monotonic()-start>6*3600:raise TimeoutError('Original SCIP cases did not finish within six hours.')
    time.sleep(10)

sdp=HERE/'perspective_sdp';sdp.mkdir(exist_ok=True)
status('running_120s_sdp_on_one_free_cpu')
with (sdp/'diagonal_sdp.log').open('w') as stream:
    sdp_result=subprocess.run([PYTHON,str(HERE/'diagonal_sdp_admm.py'),'--seconds','120'],stdout=stream,stderr=subprocess.STDOUT)
status('running_separate_one_thread_tabdpt_timing',sdp_returncode=sdp_result.returncode)
with (MOD/'tabdpt/resource_probe_one_thread.log').open('w') as stream:
    timer_result=subprocess.run([str(MOD/'venv_tabdpt/bin/python'),str(MOD/'tabdpt/resource_probe_one_thread.py')],
                                stdout=stream,stderr=subprocess.STDOUT)

pid=dispatcher_pid()
boundary=dict(dispatcher_pid=pid,sdp_returncode=sdp_result.returncode,resource_probe_returncode=timer_result.returncode)
if pid is not None:
    known=completion_paths()
    status('waiting_for_perspective_case_done_boundary',dispatcher_pid=pid,already_completed=len(known))
    while True:
        completed=completion_paths()
        active=[d for d in OUT.glob('*/beer_*') if d.is_dir() and (d/'protocol.json').exists() and not (d/'DONE.json').exists()]
        if completed-known or not active:
            boundary['newly_completed_at_stop']=[str(p.relative_to(OUT)) for p in completed-known]
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
            break
        if not Path(f'/proc/{pid}').exists():break
        time.sleep(.25)
    # A process that already wrote DONE has no unsaved result needed by workers.
    for _ in range(100):
        proc=Path(f'/proc/{pid}')
        if not proc.exists():break
        try:
            if proc.joinpath('stat').read_text().split()[2]=='Z':break
        except OSError:break
        time.sleep(.1)
    else:raise RuntimeError('Dispatcher did not terminate; do not start conflicting workers.')

archived=[]
for d in list(OUT.glob('*/beer_*')):
    if d.is_dir() and not (d/'DONE.json').exists():
        archive=OUT/'interrupted_initialization'
        archive.mkdir(exist_ok=True)
        target=archive/f'{d.parent.name}_{d.name}_{int(time.time())}'
        shutil.move(str(d),str(target));archived.append(str(target.relative_to(OUT)))
boundary['archived_unfinished_initialization']=archived
(OUT/'parallel_handoff.json').write_text(json.dumps(boundary,indent=2))
status('running_two_policy_disjoint_workers',**boundary)
workers=[];streams=[]
for policy in ('strict','exceedance'):
    stream=(OUT/f'{policy}_parallel_run.log').open('w');streams.append(stream)
    worker=subprocess.Popen([PYTHON,str(HERE/'perspective_policy_worker.py'),'--policy',policy,'--seconds','180'],
                             stdout=stream,stderr=subprocess.STDOUT)
    workers.append((policy,worker))
while any(worker.poll() is None for _,worker in workers):
    rows=aggregate()
    status('running_two_policy_disjoint_workers',completed_cases=len(rows),
           workers={policy:worker.poll() for policy,worker in workers})
    time.sleep(15)
for stream in streams:stream.close()
codes={policy:worker.returncode for policy,worker in workers}
if any(codes.values()):
    status('failed_policy_worker',returncodes=codes)
    raise RuntimeError('A disjoint worker failed; inspect its log and completed cases.')
rows=aggregate()
assert len(rows)==22
(OUT/'DONE.json').write_text(json.dumps(dict(cases_completed=22,expanded=True,
    execution='Initial serial cases followed by two policy-disjoint one-thread workers',
    seconds_per_case=180,parallel_handoff=boundary),indent=2))
status('all_done',completed_cases=22,returncodes=codes)
