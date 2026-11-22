"""Use one released bootstrap-training CPU for the fixed generation challenge."""
from pathlib import Path
import datetime
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
gate = HERE.parent/'transfer_robustness/PREFIT_COMPLETE.json'
status = HERE/'penalized_em_queue.json'
start = time.monotonic()
status.write_text(json.dumps(dict(stage='waiting_for_training_cache',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), gate=str(gate)), indent=2))
while not gate.exists():
    if time.monotonic()-start > 6*3600:
        raise TimeoutError('Training cache did not finish within six hours')
    time.sleep(15)
data = json.loads(gate.read_text())
assert data['completed'] and int(data['prefitted']) == 200
status.write_text(json.dumps(dict(stage='running',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2))
with (HERE/'penalized_em.log').open('w') as log:
    result = subprocess.run([sys.executable, str(HERE/'run_penalized_em.py')],
                            stdout=log, stderr=subprocess.STDOUT)
status.write_text(json.dumps(dict(stage='complete' if result.returncode == 0 else 'failed',
    returncode=result.returncode, utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2))
raise SystemExit(result.returncode)
