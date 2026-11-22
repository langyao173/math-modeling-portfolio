"""Perspective MIQCP, independent of the concurrently running original solver.

The diagonal split and perspective construction are established methods of
Frangioni & Gentile (2006, 2007) and Gunluk & Linderoth (2010). This file applies
them to the frozen beer intervention constraints; it does not claim a new
general optimization method. See perspective_method.md for proofs and sources.

Every stored SCIP pool point is retained before any fixed-support polishing.
A lower bound concerns the relaxation; an upper bound is reported only after
raw-field decoding and the frozen six-model sensory check.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'

import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
from pyscipopt import Model, quicksum
from threadpoolctl import threadpool_limits
from control_problem import FixedProblem, AW, file_sha

HERE = Path(__file__).resolve().parent
OUT = HERE / 'perspective'
RHO = .1
MARGIN = 0.
STYLE_EPSILON = 1e-5
PSD_GUARD = 1e-8


def write_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temp.replace(path)


def diagonal_split(P):
    """Conservative scalar diagonal split, with explicit numerical audit."""
    P0 = np.asarray(P, float)
    P = (P0 + P0.T) / 2
    asymmetry = float(np.max(np.abs(P0-P0.T)))
    eig = np.linalg.eigvalsh(P)
    if eig[0] <= 0:
        raise ValueError('Frozen precision matrix is not positive definite.')
    diagonal = np.full(len(P), (1-PSD_GUARD)*eig[0])
    R = P - np.diag(diagonal)
    eigen_r = np.linalg.eigvalsh(R)
    if eigen_r[0] <= 0:
        raise ValueError('Conservative diagonal split failed its PSD check.')
    L = np.linalg.cholesky(R).T
    audit = dict(method='lambda_min_identity', guard=PSD_GUARD,
                 original_asymmetry_max=asymmetry,
                 P_eigen_min=float(eig[0]), P_eigen_max=float(eig[-1]),
                 diagonal_min=float(diagonal.min()), diagonal_max=float(diagonal.max()),
                 R_eigen_min=float(eigen_r[0]), R_eigen_max=float(eigen_r[-1]),
                 split_identity_max=float(np.max(np.abs(P-R-np.diag(diagonal)))),
                 factor_identity_max=float(np.max(np.abs(R-L.T@L))),
                 symmetry_is_quadratically_equivalent=True)
    if audit['factor_identity_max'] > 1e-9*max(1., np.max(np.abs(P))):
        raise ValueError('Factorization residual exceeds numerical guard.')
    return P, diagonal, R, L, audit


def accepted_warm(problem, beer):
    """Read minimum raw-field accepted old witness; do not alter or polish it."""
    base = HERE / problem.policy / f'beer_{beer}'
    if not (base/'DONE.json').exists():
        return None, None
    checks = pd.read_csv(base/'checks.csv')
    profiles = pd.read_csv(base/'profiles.csv')
    accepted = checks['accepted'].astype(str).str.lower().eq('true')
    eligible = checks[accepted & (checks.target_distance_reduction >= RHO-1e-8)]
    eligible = eligible.sort_values(['raw_fields_changed', 'quadratic_objective'])
    ctx = problem.contexts[beer]
    for _, oldcheck in eligible.iterrows():
        cid = int(oldcheck.candidate_id)
        selected = profiles.loc[profiles.candidate_id.eq(cid)]
        if len(selected) != 1:
            raise ValueError('Ambiguous old candidate_id.')
        row = selected.iloc[0].to_dict()
        row.update(required_rho=RHO, required_proxy_margin=MARGIN)
        check, _ = problem.check_rows([row])
        if not bool(check.iloc[0].accepted):
            continue
        x = np.asarray([row[k] for k in problem.features], float)
        delta = problem.control.map_.transform(x[None])[0] - ctx['z0']
        same = (x == ctx['x0']) | (np.isnan(x) & np.isnan(ctx['x0']))
        support = np.union1d(np.flatnonzero(~same), ctx['mandatory']).astype(int)
        outside = np.ones(problem.p, bool)
        outside[support] = False
        if np.max(np.abs(delta[outside]), initial=0.) > 1e-10:
            raise ValueError('Raw support disagrees with transformed warm point.')
        delta[outside] = 0.
        info = dict(source=str(base/'profiles.csv'), candidate_id=cid,
                    source_profiles_sha256=file_sha(base/'profiles.csv'),
                    raw_fields_changed=int(check.iloc[0].raw_fields_changed),
                    support_size=int(len(support)),
                    target_distance_reduction=float(check.iloc[0].target_distance_reduction),
                    full_check=check.iloc[0].to_dict())
        return dict(delta=delta, support=support, row=row), info
    return None, None


def solve_perspective(problem, beer, time_limit, warm, directory):
    c = problem.contexts[beer]
    p = problem.p
    P, diag, R, L, audit = diagonal_split(problem.P)
    np.savez_compressed(directory/'diagonal_split.npz', P=P, diagonal=diag, R=R, factor=L)
    write_json(directory/'diagonal_split_audit.json', audit)
    m = Model(f'perspective_sparse_{beer}_{problem.policy}')
    m.setRealParam('limits/time', float(time_limit))
    m.setRealParam('limits/gap', 0.)
    m.setRealParam('numerics/feastol', 1e-8)
    m.setIntParam('randomization/randomseedshift', 20261024)
    m.setIntParam('parallel/maxnthreads', 1)
    m.setIntParam('display/verblevel', 4)
    # delta^2-t*b has an indefinite Hessian. Its feasible set is a rotated
    # second-order cone; allow SCIP's SOC handler to recognize it correctly.
    m.setBoolParam('constraints/nonlinear/assumeconvex', False)
    m.setBoolParam('nlhdlr/soc/enabled', True)
    m.setBoolParam('nlhdlr/perspective/enabled', True)
    m.setLogfile(str(directory/'scip.log'))
    params = m.getParams()
    write_json(directory/'solver_parameters.json', {key: params[key] for key in (
        'limits/time', 'limits/gap', 'limits/maxsol', 'numerics/feastol',
        'randomization/randomseedshift', 'parallel/maxnthreads',
        'constraints/nonlinear/assumeconvex', 'nlhdlr/soc/enabled',
        'nlhdlr/soc/compeigenvalues', 'nlhdlr/perspective/enabled')})
    mandatory = set(c['mandatory'])
    d = [m.addVar(f'd{j}', lb=min(0., c['lo'][j]), ub=max(0., c['hi'][j])) for j in range(p)]
    b = [m.addVar(f'b{j}', vtype='B', lb=1 if j in mandatory else 0) for j in range(p)]
    maxabs = np.maximum(np.abs(c['lo']), np.abs(c['hi']))
    t = [m.addVar(f't{j}', lb=0., ub=float(maxabs[j]**2)) for j in range(p)]
    for j in range(p):
        lo, hi = float(c['lo'][j]), float(c['hi'][j])
        m.addCons(d[j] <= hi*b[j], name=f'upper_link_{j}')
        m.addCons(d[j] >= lo*b[j], name=f'lower_link_{j}')
        m.addCons(d[j]*d[j] <= t[j]*b[j], name=f'rotated_soc_{j}')
        m.addCons(t[j] <= float(maxabs[j]**2)*b[j], name=f't_link_{j}')
        m.addCons(t[j] <= (lo+hi)*d[j]-lo*hi*b[j], name=f'interval_secant_{j}')
    A, rhs = problem.linear_constraints(beer, margin=MARGIN)
    for k, (row, rr) in enumerate(zip(A, rhs)):
        m.addCons(quicksum(float(row[j])*d[j] for j in range(p)) <= float(rr), name=f'guard_{k}')
    v = [m.addVar(f'rfactor{j}', lb=None, ub=None) for j in range(p)]
    for i in range(p):
        m.addCons(v[i] == quicksum(float(L[i,j])*d[j] for j in range(p) if L[i,j] != 0.))
    offset = c['z0']-problem.mu
    # g and c MUST retain the full P, not R, after diagonal splitting.
    gradient = P@offset
    constant = float(offset@P@offset)
    radius2 = (1-RHO)**2 * constant
    m.addCons(quicksum(x*x for x in v) + quicksum(float(diag[j])*t[j] for j in range(p))
              + quicksum(float(2*gradient[j])*d[j] for j in range(p)) <= radius2-constant,
              name='perspective_target_ball')
    m.setObjective(quicksum(b), 'minimize')
    warm_added = False
    if warm is not None:
        dd = warm['delta']
        supp = set(warm['support'].tolist())
        sol = m.createSol()
        vv = L@dd
        for j in range(p):
            m.setSolVal(sol, d[j], float(dd[j]))
            m.setSolVal(sol, b[j], float(j in supp))
            m.setSolVal(sol, t[j], float(dd[j]**2))
            m.setSolVal(sol, v[j], float(vv[j]))
        warm_added = bool(m.addSol(sol))
    started = time.perf_counter()
    m.optimize()
    elapsed = time.perf_counter()-started
    pool = []
    for i, sol in enumerate(m.getSols()):
        dd = np.array([m.getSolVal(sol, x) for x in d])
        bb = np.array([m.getSolVal(sol, x) for x in b])
        tt = np.array([m.getSolVal(sol, x) for x in t])
        ss = np.flatnonzero(bb > .5)
        pool.append(dict(delta=dd, binary=bb, epigraph=tt, support=ss,
                         pool_id=i, solver_objective=float(m.getSolObjVal(sol))))
    record = dict(beer=int(beer), policy=problem.policy, rho=RHO, margin=MARGIN,
                  style_epsilon=STYLE_EPSILON, time_limit=float(time_limit),
                  status=str(m.getStatus()), dual_bound=float(m.getDualbound()),
                  primal_bound=float(m.getPrimalbound()), gap=float(m.getGap()),
                  nodes=int(m.getNNodes()), seconds=elapsed, solutions=len(pool),
                  warm_added=warm_added, solver='SCIP 10.0.2 / PySCIPOpt 6.3.0',
                  method='perspective_lambda_min', diagonal_split=audit,
                  lower_bound_scope='Frozen linear guards + rho=.1 geometry; full sensory constraint relaxed',
                  primal_bound_scope='Relaxation incumbent, not automatically a full-problem upper bound')
    m.freeProb()
    return pool, record


def run_case(policy, beer, time_limit=180.):
    directory = OUT / policy / f'beer_{beer}'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'DONE.json').exists():
        return json.loads((directory/'DONE.json').read_text())
    started = time.perf_counter()
    p = FixedProblem(policy, style_epsilon=STYLE_EPSILON)
    if beer not in p.contexts:
        raise ValueError(f'No source beer {beer}.')
    old_base = HERE / policy / f'beer_{beer}'
    if not (old_base/'DONE.json').exists():
        raise FileNotFoundError(f'Original matched case is not complete: {old_base}')
    warm, warm_info = accepted_warm(p, beer)
    if warm is None:
        raise RuntimeError('No original accepted rho>=.1 witness for a matched warm start.')
    write_json(directory/'warm_start.json', warm_info)
    np.savez_compressed(directory/'warm_start.npz', delta=warm['delta'], support=warm['support'])
    protocol = dict(policy=policy, beer=int(beer), rho=RHO, proxy_margin=MARGIN,
                    style_epsilon=STYLE_EPSILON, seconds=float(time_limit), threads=1,
                    diagonal_split='lambda_min(P)*(1-1e-8)*I', interval_secants=True,
                    nonlinear_assumeconvex=False, full_decode_and_sensory_checks=True,
                    retain_all_stored_raw_pool_points=True,
                    code_sha256=file_sha(__file__), control_code_sha256=file_sha(HERE/'control_problem.py'),
                    predictor_sha256=file_sha(AW/'q1/sensory_ensemble.joblib'),
                    control_sha256=file_sha(AW/'q3/transfer_and_augmentation_models.joblib'),
                    classifier_sha256=file_sha(AW/'competition_revision/style_feasibility/frozen_style_classifier.joblib'))
    write_json(directory/'protocol.json', protocol)
    pool, bound = solve_perspective(p, beer, time_limit, warm, directory)
    write_json(directory/'solver_bound.json', bound)
    np.savez_compressed(directory/'raw_pool.npz',
        delta=np.stack([c['delta'] for c in pool]) if pool else np.empty((0,p.p)),
        binary=np.stack([c['binary'] for c in pool]) if pool else np.empty((0,p.p)),
        epigraph=np.stack([c['epigraph'] for c in pool]) if pool else np.empty((0,p.p)),
        support=np.stack([np.isin(np.arange(p.p),c['support']) for c in pool]) if pool else np.empty((0,p.p),bool),
        solver_objective=np.array([c['solver_objective'] for c in pool]))
    # Raw points are saved and independently decoded before any polishing.
    rows = [dict(warm['row'], candidate_id=0, method='original_accepted_warm', pool_id=-1,
                 required_rho=RHO, required_proxy_margin=MARGIN)]
    candidates = [dict(delta=warm['delta'], support=warm['support'])]
    for c in pool:
        row, _ = p.decode(beer, c['delta'], c['support'])
        row.update(candidate_id=len(rows), method='perspective_raw_pool', pool_id=c['pool_id'],
                   required_rho=RHO, required_proxy_margin=MARGIN)
        rows.append(row)
        candidates.append(c)
    raw_checks, raw_pred = p.check_rows(rows)
    raw_meta = pd.DataFrame(rows)[['candidate_id','method','pool_id']]
    pd.concat([raw_meta, raw_checks], axis=1).to_csv(directory/'raw_checks.csv', index=False)
    pd.DataFrame(rows).to_csv(directory/'raw_profiles.csv', index=False)
    raw_pred.to_csv(directory/'raw_predicted_sensory.csv', index=False)
    polish_records = []
    for c in pool:
        q, delta, violation = p.polish(beer, c['support'], margin=MARGIN, initial=c['delta'], rho=RHO)
        row, _ = p.decode(beer, delta, c['support'])
        row.update(candidate_id=len(rows), method='perspective_fixed_support_polish', pool_id=c['pool_id'],
                   required_rho=RHO, required_proxy_margin=MARGIN, qp_violation=violation)
        rows.append(row)
        candidates.append(dict(delta=delta, support=c['support']))
        polish_records.append(dict(pool_id=c['pool_id'], success=bool(q.success), message=str(q.message),
                                   iterations=int(q.nit), linear_box_violation=violation,
                                   geometry_violation=float((p.contexts[beer]['z0']+delta-p.mu)@p.P@
                                       (p.contexts[beer]['z0']+delta-p.mu)-(1-RHO)**2*2*p.contexts[beer]['F0'])))
    checks, pred = p.check_rows(rows)
    checks = pd.concat([pd.DataFrame(rows)[['candidate_id','method','pool_id']], checks], axis=1)
    checks.to_csv(directory/'checks.csv', index=False)
    pd.DataFrame(rows).to_csv(directory/'profiles.csv', index=False)
    pred.to_csv(directory/'predicted_sensory.csv', index=False)
    pd.DataFrame(polish_records).to_csv(directory/'polish_records.csv', index=False)
    np.savez_compressed(directory/'rank_candidates.npz', delta=np.stack([c['delta'] for c in candidates]),
                       support=np.stack([np.isin(np.arange(p.p), c['support']) for c in candidates]))
    old_bound = json.loads((old_base/'solver_bound.json').read_text())
    if abs(old_bound['rho']-RHO)>1e-12 or abs(old_bound['margin']-MARGIN)>1e-12:
        raise ValueError('Original bound is not for the matched constraints.')
    valid = checks[checks.accepted]
    full_ub = int(valid.raw_fields_changed.min()) if len(valid) else None
    old_full_ub = int(warm_info['raw_fields_changed'])
    # Numerical tolerance is explicit; raw solver bounds are retained alongside ceil.
    new_lb_int = int(math.ceil(bound['dual_bound']-1e-7))
    old_lb_int = int(math.ceil(old_bound['dual_bound']-1e-7))
    result = dict(beer=int(beer), policy=policy, rho=RHO, margin=MARGIN, style_epsilon=STYLE_EPSILON,
                  seconds=time.perf_counter()-started, solver_seconds=bound['seconds'],
                  raw_pool_points=len(pool), full_accepted=int(checks.accepted.sum()),
                  new_dual_bound=bound['dual_bound'], old_dual_bound=old_bound['dual_bound'],
                  new_integer_lower_bound=new_lb_int, old_integer_lower_bound=old_lb_int,
                  new_relaxation_primal_bound=bound['primal_bound'], old_relaxation_primal_bound=old_bound['primal_bound'],
                  new_full_upper_bound=full_ub, old_full_upper_bound=old_full_ub,
                  matched_old_solver_seconds=old_bound['seconds'],
                  clear_improvement=bool(new_lb_int>old_lb_int or (full_ub is not None and full_ub<old_full_ub)),
                  clear_improvement_rule='At least one unit tighter integer lower bound or smaller full-checked raw-field upper bound')
    write_json(directory/'DONE.json', result)
    print('CASE_DONE '+json.dumps(result), flush=True)
    return result


def run_pilot_and_expand(time_limit, expand):
    OUT.mkdir(parents=True, exist_ok=True)
    pilot = run_case('strict', 124, time_limit)
    results = [pilot]
    if expand and pilot['clear_improvement']:
        p = FixedProblem('strict', style_epsilon=STYLE_EPSILON)
        cases = [(policy, beer) for policy in ('strict','exceedance') for beer in p.contexts]
        pending = [(policy,beer) for policy,beer in cases if (policy,beer)!=('strict',124)]
        wait_start = time.monotonic()
        # Parent's original search may still be producing later cases. Never
        # read partial originals or change them; ready cases are processed first.
        while pending:
            ready = [(policy,beer) for policy,beer in pending if (HERE/policy/f'beer_{beer}'/'DONE.json').exists()]
            if not ready:
                write_json(OUT/'waiting.json', dict(pending=pending, reason='Waiting for matched original completed cases'))
                if time.monotonic()-wait_start > 6*3600:
                    raise TimeoutError('Original matched cases not complete after six hours.')
                time.sleep(20)
                continue
            for policy, beer in ready:
                results.append(run_case(policy, beer, time_limit))
                pending.remove((policy,beer))
                pd.DataFrame(results).to_csv(OUT/'comparison.csv', index=False)
    pd.DataFrame(results).to_csv(OUT/'comparison.csv', index=False)
    write_json(OUT/'DONE.json', dict(cases_completed=len(results), expanded=bool(expand and pilot['clear_improvement']), pilot=pilot))
    print('COMPLETE '+json.dumps(dict(cases_completed=len(results))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=180.)
    parser.add_argument('--expand-if-better', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run_pilot_and_expand(args.seconds, args.expand_if_better)
