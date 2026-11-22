# -*- coding: utf-8 -*-
"""Run the whole pipeline in order and record environment and timing.

    python run_all.py                      # tables -> results -> figures
    python run_all.py --procs 8            # worker processes for the Problem-3 replay
    python run_all.py --from p3_duration   # resume from a step
    python run_all.py --only p1a_audit p1b_state_model
    python run_all.py --list

Protocol order: every model is fitted on the training environments, selected on the validation
environments and frozen (results/frozen_*.json) before the step that touches the test environments.
All temporary files and caches (numba, matplotlib, byte-code) are redirected into the project folder
(_tmp/, _cache/), so that a run writes nothing outside it.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)

STEPS = [
    ('s02_build_dataset', 's02_build_dataset.py', []),
    ('p1a_audit', 'p1a_audit.py', []),
    ('p1b_state_model', 'p1b_state_model.py', []),
    ('p1c_quality_model', 'p1c_quality_model.py', []),
    ('p1d_factors', 'p1d_factors.py', []),
    ('p2a_masking', 'p2a_masking.py', []),
    ('p2b_policy', 'p2b_policy.py', []),
    ('p2c_natural', 'p2c_natural.py', []),
    ('p3_duration', 'p3_duration.py', []),
    ('p3_tune', 'p3_tune.py', []),
    ('p3_run_validation1', 'p3_run.py', ['validation1', '--procs', '{procs}']),
    ('p3_pooltransfer', 'p3_pooltransfer.py', ['--procs', '{procs}']),
    ('p3_run_validation2', 'p3_run.py', ['validation2', '--procs', '{procs}']),
    ('p3_run_test', 'p3_run.py', ['test', '--procs', '{procs}']),
    ('p3_analysis', 'p3_analysis.py', []),
    ('fig_mechanism', 'fig_mechanism.py', []),
    ('fig_p1', 'fig_p1.py', []),
    ('fig_results', 'fig_results.py', []),
    ('fig_p3', 'fig_p3.py', []),
    ('make_tables', 'make_tables.py', []),
]


def environment():
    info = dict(python=sys.version, executable=os.path.basename(sys.executable), platform=platform.platform(),
                machine=platform.machine(), processor=platform.processor(), cpu_count=os.cpu_count())
    for m in ('numpy', 'pandas', 'scipy', 'sklearn', 'statsmodels', 'numba', 'pyarrow', 'matplotlib'):
        try:
            info[m] = __import__(m).__version__
        except Exception as e:                                   # pragma: no cover
            info[m] = 'not available (%s)' % type(e).__name__
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=max(2, min(8, (os.cpu_count() or 2) // 2)))
    ap.add_argument('--from', dest='start', default=None)
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--skip', nargs='*', default=[])
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    names = [s[0] for s in STEPS]
    if a.list:
        print('\n'.join(names)); return
    todo = STEPS
    if a.start:
        todo = STEPS[names.index(a.start):]
    if a.only:
        todo = [s for s in STEPS if s[0] in a.only]
    todo = [s for s in todo if s[0] not in a.skip and os.path.isfile(os.path.join(HERE, s[1]))]

    env = dict(os.environ)
    tmp = os.path.join(PROJ, '_tmp'); cache = os.path.join(PROJ, '_cache')
    for d in (tmp, os.path.join(cache, 'numba'), os.path.join(cache, 'mpl'), os.path.join(cache, 'pyc'), os.path.join(PROJ, 'logs'),
              os.path.join(PROJ, 'results')):
        os.makedirs(d, exist_ok=True)
    env.update(TMP=tmp, TEMP=tmp, TMPDIR=tmp, NUMBA_CACHE_DIR=os.path.join(cache, 'numba'), MPLCONFIGDIR=os.path.join(cache, 'mpl'),
               PYTHONPYCACHEPREFIX=os.path.join(cache, 'pyc'), PYTHONIOENCODING='utf-8', PYTHONHASHSEED='0', MPLBACKEND='Agg')

    full = len(todo) == len(STEPS)                 # partial runs do not overwrite the records of the full run
    if full:
        with open(os.path.join(PROJ, 'results', 'environment.json'), 'w', encoding='utf-8') as fh:
            json.dump(environment(), fh, indent=1)
    timing = []
    t_all = time.time()
    for name, script, args in todo:
        cmd = [sys.executable, os.path.join(HERE, script)] + [x.format(procs=a.procs) for x in args]
        log = os.path.join(PROJ, 'logs', 'run_%s.log' % name)
        print('[%s] %-20s -> logs/run_%s.log' % (time.strftime('%H:%M:%S'), name, name), flush=True)
        t0 = time.time()
        with open(log, 'w', encoding='utf-8') as fh:
            rc = subprocess.call(cmd, cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        timing.append(dict(step=name, seconds=round(dt, 1), returncode=rc))
        with open(os.path.join(PROJ, 'logs', 'run_all_timing.json' if full else 'run_partial_timing.json'), 'w', encoding='utf-8') as fh:
            json.dump(dict(procs=a.procs, steps=timing, total_seconds=round(time.time() - t_all, 1)), fh, indent=1)
        print('    %s in %.0f s' % ('ok' if rc == 0 else 'FAILED (%d)' % rc, dt), flush=True)
        if rc != 0:
            sys.exit(rc)
    print('all steps finished in %.1f min' % ((time.time() - t_all) / 60.0), flush=True)


if __name__ == '__main__':
    main()
