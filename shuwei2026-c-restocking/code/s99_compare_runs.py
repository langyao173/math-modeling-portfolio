# -*- coding: utf-8 -*-
"""Compare the result files of two runs of the pipeline (reproduction check).

    python s99_compare_runs.py <reference_results_dir> <new_results_dir> [--out report.json]

Every CSV / JSON file present in both directories is compared cell by cell.  Numeric cells are compared
through  |a - b| / max(|a|, |b|, floor)  (floor 1e-6, so that values around zero are judged on an absolute
scale); non-numeric cells must be identical.  Files whose content is machine dependent by construction
(run times, environment) are listed but not judged.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

MACHINE_DEPENDENT = ('p3_runtime_', 'environment', 'p2_policy_meta.json', 'repro_report')
FLOOR = 1e-6


def _flat(o, pre=''):
    if isinstance(o, dict):
        for k, v in o.items():
            for x in _flat(v, pre + '/' + str(k)):
                yield x
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            for x in _flat(v, pre + '/' + str(i)):
                yield x
    else:
        yield pre, o


def _num(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def cmp_values(a, b):
    """-> (n_numeric, n_text, max_rel, n_text_diff, n_nan_mismatch)"""
    if _num(a) and _num(b):
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            return 1, 0, 0.0, 0, int(math.isnan(fa) != math.isnan(fb))
        if math.isinf(fa) or math.isinf(fb):
            return 1, 0, 0.0 if fa == fb else float('inf'), 0, 0
        return 1, 0, abs(fa - fb) / max(abs(fa), abs(fb), FLOOR), 0, 0
    return 0, 1, 0.0, int(str(a) != str(b)), 0


def cmp_csv(pa, pb):
    A = pd.read_csv(pa); B = pd.read_csv(pb)
    if list(A.columns) != list(B.columns) or len(A) != len(B):
        return dict(status='shape differs', shape_ref=list(A.shape), shape_new=list(B.shape))
    nn = nt = ntd = nnan = 0; mx = 0.0; worst = None
    for c in A.columns:
        a, b = A[c], B[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(a):
            x = a.to_numpy(dtype=float); y = b.to_numpy(dtype=float)
            nnan += int((np.isnan(x) != np.isnan(y)).sum())
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.any():
                with np.errstate(invalid='ignore'):
                    r = np.abs(x[ok] - y[ok]) / np.maximum(np.maximum(np.abs(x[ok]), np.abs(y[ok])), FLOOR)
                r = np.where(np.isnan(r), 0.0, r)
                if r.max() > mx:
                    mx = float(r.max()); worst = c
            nn += int(ok.sum())
        else:
            x = np.array(['' if pd.isna(v) else str(v) for v in a], dtype=object)
            y = np.array(['' if pd.isna(v) else str(v) for v in b], dtype=object)
            nt += len(a); ntd += int((x != y).sum())
    return dict(status='compared', numeric_cells=nn, text_cells=nt, max_rel_diff=mx, worst_column=worst, text_diffs=ntd, nan_mismatch=nnan)


def cmp_json(pa, pb):
    A = dict(_flat(json.load(open(pa, encoding='utf-8')))); B = dict(_flat(json.load(open(pb, encoding='utf-8'))))
    if set(A) != set(B):
        return dict(status='keys differ', only_ref=len(set(A) - set(B)), only_new=len(set(B) - set(A)))
    nn = nt = ntd = nnan = 0; mx = 0.0; worst = None
    for k in A:
        a, b, r, td, nm = cmp_values(A[k], B[k])
        nn += a; nt += b; ntd += td; nnan += nm
        if r > mx:
            mx = r; worst = k
    return dict(status='compared', numeric_cells=nn, text_cells=nt, max_rel_diff=mx, worst_column=worst, text_diffs=ntd, nan_mismatch=nnan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref'); ap.add_argument('new'); ap.add_argument('--out', default=None); ap.add_argument('--tol', type=float, default=1e-6)
    a = ap.parse_args()
    fa = set(f for f in os.listdir(a.ref) if f.endswith(('.csv', '.json')))
    fb = set(f for f in os.listdir(a.new) if f.endswith(('.csv', '.json')))
    rep = dict(reference=os.path.basename(os.path.abspath(a.ref)), new=os.path.basename(os.path.abspath(a.new)), tolerance=a.tol,
               only_in_reference=sorted(fa - fb), only_in_new=sorted(fb - fa), files={})
    for f in sorted(fa & fb):
        try:
            r = cmp_csv(os.path.join(a.ref, f), os.path.join(a.new, f)) if f.endswith('.csv') else cmp_json(os.path.join(a.ref, f), os.path.join(a.new, f))
        except Exception as e:
            r = dict(status='error: %s' % e)
        r['judged'] = not f.startswith(MACHINE_DEPENDENT) and f not in MACHINE_DEPENDENT
        if r['status'] == 'compared':
            r['identical_within_tol'] = bool(r['max_rel_diff'] <= a.tol and r['text_diffs'] == 0 and r['nan_mismatch'] == 0)
        rep['files'][f] = r
    judged = {f: r for f, r in rep['files'].items() if r['judged']}
    ok = [f for f, r in judged.items() if r.get('identical_within_tol')]
    bad = {f: r for f, r in judged.items() if not r.get('identical_within_tol')}
    rep['summary'] = dict(files_compared=len(judged), files_within_tolerance=len(ok), files_outside_tolerance=len(bad),
                          numeric_cells=int(sum(r.get('numeric_cells', 0) for r in judged.values())),
                          max_rel_diff=float(max([r.get('max_rel_diff', 0.0) for r in judged.values()] or [0.0])))
    print(json.dumps(rep['summary'], indent=1))
    for f, r in sorted(bad.items(), key=lambda kv: -kv[1].get('max_rel_diff', 9e9) if kv[1]['status'] == 'compared' else -9e9):
        print('  DIFF %-42s %s' % (f, {k: (round(v, 10) if isinstance(v, float) else v) for k, v in r.items() if k not in ('judged',)}))
    if rep['only_in_reference'] or rep['only_in_new']:
        print('  only in reference:', rep['only_in_reference'], '\n  only in new:', rep['only_in_new'])
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(rep, fh, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
