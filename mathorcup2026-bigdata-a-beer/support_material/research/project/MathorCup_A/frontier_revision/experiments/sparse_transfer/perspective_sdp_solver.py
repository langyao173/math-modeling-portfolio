"""Inject a saved safely feasible SDP diagonal into the validated MIQCP core.

The shared core file and running lambda-min workers are not modified. Only
this Python process replaces the diagonal-split provider and output root.
The constraints, original accepted warm start, seed and 180-second budget
remain identical. Every injected dependency and source hash is recorded.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import perspective_solver as core
from control_problem import FixedProblem,file_sha

HERE=Path(__file__).resolve().parent
OUT=HERE/'perspective_sdp'
LAMBDA_OUT=HERE/'perspective'
core.OUT=OUT
CORE_SOLVE=core.solve_perspective


def saved_sdp_split(P):
    a=np.load(OUT/'diagonal_sdp.npz')
    audit=json.loads((OUT/'diagonal_sdp_audit.json').read_text())
    symmetric=(P+P.T)/2
    if np.max(abs(a['P']-symmetric))>1e-10:
        raise AssertionError('Saved SDP belongs to another frozen precision matrix.')
    if np.linalg.eigvalsh(a['R']).min()<=0 or a['diagonal'].min()<0:
        raise AssertionError('Saved SDP split failed PSD validation.')
    if np.max(abs(a['P']-a['R']-np.diag(a['diagonal'])))>1e-10:
        raise AssertionError('Saved SDP no longer satisfies P=R+D.')
    return a['P'],a['diagonal'],a['R'],a['factor'],audit


def solve_with_sdp_metadata(problem,beer,time_limit,warm,directory):
    protocol=json.loads((directory/'protocol.json').read_text())
    protocol.update(diagonal_split='Safely scaled approximate max-trace SDP from standard ADMM',
                    diagonal_sdp_sha256=file_sha(OUT/'diagonal_sdp.npz'),
                    diagonal_sdp_audit_sha256=file_sha(OUT/'diagonal_sdp_audit.json'),
                    shared_constraint_core_sha256=file_sha(HERE/'perspective_solver.py'),
                    variant_wrapper_sha256=file_sha(__file__),
                    warm_comparison='Same original fully checked minimum-raw-field witness as lambda-min case; no warm advantage from later solutions')
    core.write_json(directory/'protocol.json',protocol)
    pool,record=CORE_SOLVE(problem,beer,time_limit,warm,directory)
    record['method']='perspective_sdp_max_trace'
    record['diagonal_sdp_sha256']=protocol['diagonal_sdp_sha256']
    return pool,record


core.diagonal_split=saved_sdp_split
core.solve_perspective=solve_with_sdp_metadata


def case(policy,beer,seconds):
    result=core.run_case(policy,beer,seconds)
    reference=LAMBDA_OUT/policy/f'beer_{beer}'/'DONE.json'
    if reference.exists():
        previous=json.loads(reference.read_text())
        lb=max(result['old_integer_lower_bound'],previous['new_integer_lower_bound'])
        ub=min(result['old_full_upper_bound'],previous['new_full_upper_bound'])
        result.update(lambda_min_dual_bound=previous['new_dual_bound'],
                      lambda_min_integer_lower_bound=previous['new_integer_lower_bound'],
                      lambda_min_full_upper_bound=previous['new_full_upper_bound'],
                      best_prior_integer_lower_bound=lb,best_prior_full_upper_bound=ub,
                      clear_improvement_vs_best_prior=bool(result['new_integer_lower_bound']>lb or result['new_full_upper_bound']<ub))
    else:
        result.update(best_prior_integer_lower_bound=result['old_integer_lower_bound'],
                      best_prior_full_upper_bound=result['old_full_upper_bound'],
                      clear_improvement_vs_best_prior=result['clear_improvement'])
    result['method']='perspective_sdp_max_trace'
    core.write_json(OUT/policy/f'beer_{beer}'/'DONE.json',result)
    print('SDP_CASE_DONE '+json.dumps(result),flush=True)
    return result


def main(args):
    if not (OUT/'diagonal_sdp.npz').exists():raise FileNotFoundError('Run diagonal_sdp_admm.py first.')
    pilot=case('strict',124,args.seconds)
    results=[pilot]
    expand=bool(args.expand_if_better and pilot['clear_improvement_vs_best_prior'])
    core.write_json(OUT/'pilot_decision.json',dict(expand=expand,
        rule='At least one tighter integer lower-bound unit or one smaller full-checked raw-field upper bound than both old and lambda-min references',pilot=pilot))
    if expand:
        problem=FixedProblem('strict',style_epsilon=1e-5)
        for policy in ('strict','exceedance'):
            for beer in problem.contexts:
                if (policy,beer)==('strict',124):continue
                # Wait for a complete lambda-min case to make every paired
                # comparison explicit. Its worker can continue in the other CPU.
                started=time.monotonic()
                while not (LAMBDA_OUT/policy/f'beer_{beer}'/'DONE.json').exists():
                    if time.monotonic()-started>6*3600:raise TimeoutError('Missing lambda-min comparison case.')
                    time.sleep(15)
                results.append(case(policy,beer,args.seconds))
                pd.DataFrame(results).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(results).to_csv(OUT/'comparison.csv',index=False)
    core.write_json(OUT/'DONE.json',dict(cases_completed=len(results),expanded=expand,seconds_per_case=args.seconds,
        source='Established diagonal SDP and perspective construction; applied to frozen beer-transfer constraints'))
    print('COMPLETE '+json.dumps(dict(cases_completed=len(results),expanded=expand)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seconds',type=float,default=180.)
    parser.add_argument('--expand-if-better',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):main(args)
