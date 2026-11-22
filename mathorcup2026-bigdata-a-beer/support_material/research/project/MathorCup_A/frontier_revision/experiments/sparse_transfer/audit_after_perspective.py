"""One-thread final integrity gate after both disjoint workers finish."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import hashlib
import json
import subprocess
import time

HERE=Path(__file__).resolve().parent
OUT=HERE/'perspective'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state(value,**extra):
    data=dict(state=value,timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra)
    (OUT/'integrity_queue_status.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(data),flush=True)


state('waiting_for_both_clean_policy_workers')
begin=time.monotonic()
while not (OUT/'DONE.json').exists():
    if time.monotonic()-begin>6*3600:raise TimeoutError('Missing completed perspective run.')
    time.sleep(15)
done=json.loads((OUT/'DONE.json').read_text())
assert done['cases_completed']==22
assert all((OUT/f'DONE_{policy}.json').exists() for policy in ('strict','exceedance'))
state('running_complete_independent_audit',threads=1)
with (OUT/'independent_audit.log').open('w') as log:
    result=subprocess.run([str(HERE/'venv/bin/python'),str(HERE/'audit_perspective_results.py'),
                           '--directory',str(OUT)],stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
    state('audit_failed',returncode=result.returncode)
    raise RuntimeError('Perspective final audit failed; inspect retained log. No integrity gate produced.')
audit=json.loads((OUT/'independent_audit.json').read_text())
assert audit['completed_cases']==22 and audit['all_full_witnesses_pass']
records=audit['records']
assert len({(r['policy'],r['beer']) for r in records})==22
for policy in ('strict','exceedance'):
    assert sum(r['policy']==policy for r in records)==11
files={}
names=('DONE.json','protocol.json','solver_parameters.json','solver_bound.json','scip.log',
       'raw_pool.npz','raw_checks.csv','raw_profiles.csv','rank_candidates.npz',
       'profiles.csv','checks.csv','warm_start.json','warm_start.npz','diagonal_split.npz')
for path in sorted(OUT.glob('*/beer_*/DONE.json')):
    for name in names:
        source=path.parent/name
        assert source.exists()
        files[str(source.relative_to(OUT))]=sha(source)
certificate=dict(state='complete_independently_verified',completed_cases=22,
    all_full_witnesses_pass=True,threads=1,
    matched_budget_seconds=180,policies={'strict':11,'exceedance':11},
    audit_sha256=sha(OUT/'independent_audit.json'),
    audit_code_sha256=sha(HERE/'audit_perspective_results.py'),
    recovery_record_sha256=sha(OUT/'handoff_recovery.json'),
    excluded_directories=['excluded_overlap_recovery','interrupted_initialization'],
    checks='Original warm-source and constraint hashes, declared margins and 180-second budget, PSD and algebraic identities, every stored raw pool point preserved, all accepted decoded witnesses independently re-evaluated.',
    limitation='SCIP floating-point global bounds and double-precision matrix checks; no exact-arithmetic certificate.',
    timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    result_file_sha256=files)
temp=OUT/'INTEGRITY_DONE.tmp.json'
temp.write_text(json.dumps(certificate,indent=2));temp.replace(OUT/'INTEGRITY_DONE.json')
state('all_done',completed_cases=22,gate='INTEGRITY_DONE.json')
