"""Audit the completed registered SDP pilot or full expansion on one CPU."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import hashlib
import json
import subprocess
import time

HERE=Path(__file__).resolve().parent
OUT=HERE/'perspective_sdp'
OUT.mkdir(exist_ok=True)
state=OUT/'integrity_queue_status.json'
state.write_text(json.dumps(dict(state='waiting_for_registered_sdp_cases'),indent=2))
start=time.monotonic()
while not (OUT/'DONE.json').exists():
    if time.monotonic()-start>8*3600:raise TimeoutError('SDP comparison did not complete.')
    time.sleep(15)
done=json.loads((OUT/'DONE.json').read_text())
expected=22 if done['expanded'] else 1
assert done['cases_completed']==expected
state.write_text(json.dumps(dict(state='running_independent_sdp_and_case_audit',expected=expected),indent=2))
with (OUT/'independent_audit.log').open('w') as log:
    result=subprocess.run([str(HERE/'venv/bin/python'),str(HERE/'audit_perspective_results.py'),
                           '--directory',str(OUT)],stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
    state.write_text(json.dumps(dict(state='audit_failed',returncode=result.returncode),indent=2))
    raise RuntimeError('SDP independent audit failed; no integrity gate produced.')
a=json.loads((OUT/'independent_audit.json').read_text())
assert a['completed_cases']==expected and a['all_full_witnesses_pass']
matrix=json.loads((OUT/'independent_sdp_certificate.json').read_text())
assert matrix['passed']
files={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
       for p in OUT.glob('*/beer_*/*') if p.is_file()}
for name in ('diagonal_sdp.npz','diagonal_sdp_audit.json','independent_sdp_certificate.json',
             'independent_audit.json','pilot_decision.json'):
    files[name]=hashlib.sha256((OUT/name).read_bytes()).hexdigest()
gate=dict(state='complete_independently_verified',completed_cases=expected,
          all_full_witnesses_pass=True,sdpp_primal_dual_pass=True,
          expanded=done['expanded'],result_file_sha256=files,
          timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
tmp=OUT/'INTEGRITY_DONE.tmp.json'
tmp.write_text(json.dumps(gate,indent=2));tmp.replace(OUT/'INTEGRITY_DONE.json')
state.write_text(json.dumps(dict(state='all_done',completed_cases=expected),indent=2))
print(json.dumps(dict(state='all_done',completed_cases=expected)),flush=True)
