"""Portable serial reproduction of registered perspective experiments and audit.

The historical shared-resource queue scripts are retained as execution records.
This entry point needs no original PID namespace, queue, or local venv path.
Completed per-case checkpoints are reused only as provided by the core runner.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import json
import subprocess
import sys
import hashlib
from types import SimpleNamespace
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    if args.variant == 'lambda':
        import perspective_solver as solver
        from control_problem import FixedProblem
        folder = HERE / 'perspective'
        sources = sorted(FixedProblem('strict', style_epsilon=1e-5).contexts)
        results = [solver.run_case(policy, beer, 180.)
                   for policy in ('strict', 'exceedance') for beer in sources]
        assert len(results) == 22
        solver.write_json(folder / 'DONE.json', dict(cases_completed=22, expanded=True,
                          seconds_per_case=180., execution='Portable sequential reproduction of all 22 registered cases.'))
    else:
        import perspective_sdp_solver as solver
        solver.main(SimpleNamespace(seconds=180., expand_if_better=True))
        folder = HERE / 'perspective_sdp'
    subprocess.run([sys.executable, str(HERE / 'audit_perspective_results.py'),
                    '--directory', str(folder)], check=True)
    audit = json.loads((folder / 'independent_audit.json').read_text())
    done = json.loads((folder / 'DONE.json').read_text())
    assert audit['completed_cases'] == done['cases_completed']
    assert audit['all_full_witnesses_pass']
    if args.variant == 'sdp':
        assert json.loads((folder / 'independent_sdp_certificate.json').read_text())['passed']
    files = {str(path.relative_to(folder)): sha(path)
             for path in folder.glob('*/beer_*/*') if path.is_file()}
    record = dict(state='complete_independently_verified',
                  completed_cases=done['cases_completed'], all_full_witnesses_pass=True,
                  audit_sha256=sha(folder / 'independent_audit.json'),
                  audit_code_sha256=sha(HERE / 'audit_perspective_results.py'),
                  execution='Portable serial runner; original logged wall times remain with saved checkpoints.',
                  result_file_sha256=files)
    temp = folder / 'INTEGRITY_DONE.tmp.json'
    temp.write_text(json.dumps(record, indent=2))
    temp.replace(folder / 'INTEGRITY_DONE.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['lambda', 'sdp'], required=True)
    with threadpool_limits(limits=1):
        main(parser.parse_args())
