"""Run the SDP MIQCP pilot only when a policy worker releases one CPU."""
from pathlib import Path
import json
import subprocess
import time

HERE=Path(__file__).resolve().parent
BASE=HERE/'perspective'
OUT=HERE/'perspective_sdp'
OUT.mkdir(exist_ok=True)


def status(state,**extra):
    rec=dict(state=state,timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra)
    temp=OUT/'queue_status.tmp.json';temp.write_text(json.dumps(rec,indent=2));temp.replace(OUT/'queue_status.json')
    print(json.dumps(rec),flush=True)


status('waiting_for_sdp_candidate_and_one_policy_cpu')
start=time.monotonic()
while True:
    candidate=(OUT/'diagonal_sdp.npz').exists()
    free_slot=any((BASE/f'DONE_{p}.json').exists() for p in ('strict','exceedance'))
    if (BASE/'DONE.json').exists():
        free_slot=free_slot or json.loads((BASE/'DONE.json').read_text()).get('cases_completed',0)==22
    if candidate and free_slot:break
    if time.monotonic()-start>8*3600:raise TimeoutError('No free policy CPU and saved SDP in eight hours.')
    time.sleep(15)
status('running_180s_sdp_pilot_and_conditional_expansion')
with (OUT/'run.log').open('w') as stream:
    result=subprocess.run([str(HERE/'venv/bin/python'),str(HERE/'perspective_sdp_solver.py'),
                           '--seconds','180','--expand-if-better'],stdout=stream,stderr=subprocess.STDOUT)
if result.returncode:
    status('failed',returncode=result.returncode)
    raise RuntimeError('SDP perspective solver failed; inspect log, preserve all checkpoints.')
done=json.loads((OUT/'DONE.json').read_text())
status('all_done',**done)
