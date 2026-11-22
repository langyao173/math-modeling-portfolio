"""Serialize the authorized TabDPT pilot and benchmark after all TabICL folds.

This process only checks files while waiting; it starts no extra inference thread.
"""
from pathlib import Path
import json
import os
import subprocess
import time

HERE = Path(__file__).resolve().parent
MODERN = HERE.parent
ENVIRONMENT = os.environ.copy()
ENVIRONMENT['BEER_TABDPT_THREADS'] = '6'
PYTHON = str(MODERN / 'venv_tabdpt' / 'bin' / 'python')
SCRIPT = str(HERE / 'run_tabdpt_benchmark.py')


def status(state, **extra):
    record = {'state': state, 'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra}
    temporary = HERE / 'queue_status.tmp.json'
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    temporary.replace(HERE / 'queue_status.json')
    print(json.dumps(record), flush=True)


status('waiting_for_tabicl_complete')
started = time.monotonic()
while True:
    log = (MODERN / 'tabicl_run.log').read_text()
    files = [MODERN / 'checkpoints' / f'{view}_fold{fold}_complete.npz'
             for view in ('rank_pca64_y16', 'raw231_y16', 'raw231_y50') for fold in range(5)]
    if all(path.exists() for path in files) and '\nCOMPLETE ' in log:
        break
    if 'Traceback (most recent call last):' in log:
        status('blocked_tabicl_error')
        raise RuntimeError('TabICL log contains a traceback; inspect before starting a new workload.')
    if time.monotonic() - started > 6 * 3600:
        status('blocked_wait_timeout')
        raise TimeoutError('TabICL did not finish within six hours.')
    time.sleep(20)

status('running_resource_only_pilot')
with (HERE / 'pilot.log').open('w') as stream:
    completed = subprocess.run([PYTHON, SCRIPT, '--pilot'], env=ENVIRONMENT, stdout=stream, stderr=subprocess.STDOUT)
if completed.returncode:
    status('failed_pilot', returncode=completed.returncode)
    raise RuntimeError('TabDPT pilot failed; inspect pilot.log.')
pilot = json.loads((HERE / 'pilot_timing.json').read_text())
status('pilot_complete_starting_frozen_benchmark', pilot_seconds=pilot['total_s'],
       estimated_250_target_minutes=pilot['total_s'] * 250 / 60,
       estimate_scope='Linear extrapolation from one target; actual per-target and fold timings are recorded.')
with (HERE / 'run.log').open('w') as stream:
    completed = subprocess.run([PYTHON, SCRIPT], env=ENVIRONMENT, stdout=stream, stderr=subprocess.STDOUT)
if completed.returncode:
    status('failed_benchmark', returncode=completed.returncode)
    raise RuntimeError('TabDPT benchmark failed; checkpoint resume remains available.')
result = json.loads((HERE / 'status.json').read_text())
assert result['completed_folds'] == 5
status('all_done', metrics=result['metrics'])
