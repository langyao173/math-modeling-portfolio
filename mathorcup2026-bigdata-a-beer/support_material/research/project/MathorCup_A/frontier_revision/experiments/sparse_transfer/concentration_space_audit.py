"""Can quantified concentration fields alone reach the target region?

This supplementary feasibility challenge keeps every unanchored or non-
concentration field fixed. It is not a minimum-edit search. Convex quadratic
duals bound the best possible target distance on the entire allowed subspace;
decoded witnesses also face the complete frozen sensory checks.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
import osqp
from scipy import sparse
from scipy.optimize import linprog
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits
from control_problem import FixedProblem, AW, file_sha

HERE = Path(__file__).resolve().parent
OUT = HERE / 'concentration_space'
MARGINS = [0., .025, .05, .10, .15, .25]


def dump(path, value):
    temp = Path(path).with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def solve(problem, beer, support, margin, folder):
    c = problem.contexts[beer]
    a, rhs = problem.linear_constraints(beer, margin=margin)
    lo, hi = c['lo'][support], c['hi'][support]
    G = np.vstack([a[:, support], np.eye(len(support)), -np.eye(len(support))])
    h = np.r_[rhs, hi, -lo]
    P = (problem.P[np.ix_(support, support)] + problem.P[np.ix_(support, support)].T) / 2
    g = c['gradient'][support]
    case = dict(beer=int(beer), policy=problem.policy, linear_margin=margin,
                allowed_concentration_fields=len(support), required_rho=.1)
    feasibility = linprog(np.zeros(len(support)), A_ub=G, b_ub=h,
                          bounds=[(None, None)] * len(support), method='highs')
    arrays = dict(P=P, g=g, G=G, h=h, support=support, lo=lo, hi=hi,
                  F0=np.array([c['F0']]))
    row = None
    if not feasibility.success:
        if feasibility.status != 2:
            raise RuntimeError('LP feasibility did not resolve: ' + feasibility.message)
        # A normalized Farkas direction; residual robustness uses the known
        # finite box instead of assuming G.T @ multiplier is exactly zero.
        alternative = linprog(h, A_eq=np.vstack([G.T, np.ones(len(h))]),
                              b_eq=np.r_[np.zeros(len(support)), 1.],
                              bounds=[(0., None)] * len(h), method='highs')
        if not alternative.success:
            raise RuntimeError('Infeasibility returned without a saved separating direction.')
        nu = np.maximum(alternative.x, 0.)
        v = G.T @ nu
        minimum_box_value = float(np.sum(np.where(v >= 0, v * lo, v * hi)))
        upper_value = float(nu @ h)
        separation = minimum_box_value - upper_value
        if separation <= 1e-8:
            raise RuntimeError('Numerical Farkas separation is too small to report.')
        arrays.update(farkas_multiplier=nu)
        case.update(linear_feasible=False, geometry_impossible=True,
                    farkas_separation=separation, farkas_stationarity=float(np.max(abs(v))),
                    full_model_accepted=False)
    else:
        qp = osqp.OSQP()
        qp.setup(P=sparse.csc_matrix(np.triu(P)), q=g, A=sparse.csc_matrix(G),
                 l=np.full(len(h), -np.inf), u=h, verbose=False, eps_abs=1e-10,
                 eps_rel=1e-10, max_iter=200000, polishing=True)
        result = qp.solve()
        if result.info.status_val not in (1, 2):
            raise RuntimeError('Distance QP did not solve: ' + result.info.status)
        d = result.x
        nu = np.maximum(result.y, 0.)
        residual = g + G.T @ nu
        dual = float(c['F0'] - nu @ h - .5 * residual @ cho_solve(cho_factor(P, lower=True), residual))
        primal = float(c['F0'] + g @ d + .5 * d @ P @ d)
        violation = float(np.max(G @ d - h))
        if violation > 2e-7 or primal - dual < -1e-7:
            raise RuntimeError('Distance certificate failed primal/dual numerical checks.')
        delta = np.zeros(problem.p)
        delta[support] = d
        row, _ = problem.decode(beer, delta, support)
        row.update(candidate_id=int(round(margin * 1000)), required_rho=.1,
                   required_proxy_margin=margin, method='concentration_subspace_distance_minimum')
        checked, sensory = problem.check_rows([row])
        check = checked.iloc[0]
        arrays.update(primal=d, dual_multiplier=nu, full_delta=delta)
        case.update(linear_feasible=True, qp_status=result.info.status,
                    distance_half_square_primal=primal, distance_half_square_dual=dual,
                    primal_dual_gap=primal - dual, primal_violation=violation,
                    best_distance_reduction_upper_bound=1 - float(np.sqrt(max(dual, 0.) / c['F0'])),
                    geometry_impossible=bool(dual > .9**2 * c['F0'] + 1e-7),
                    full_model_accepted=bool(check.accepted),
                    actual_fields_changed=int(check.raw_fields_changed),
                    actual_distance_reduction=float(check.target_distance_reduction),
                    full_ensemble_stale_delta=float(check.ensemble_stale_delta),
                    minimum_style_margin=float(check.min_style_margin))
        pd.DataFrame([row]).to_csv(folder / 'profile.csv', index=False)
        checked.to_csv(folder / 'full_check.csv', index=False)
        sensory.to_csv(folder / 'sensory.csv', index=False)
        if check.accepted:
            original = c['src']
            frozen = [name for j, name in enumerate(problem.features) if j not in set(support)]
            assert all(row[name] == original[name] or np.isnan(row[name]) and np.isnan(original[name]) for name in frozen)
    np.savez_compressed(folder / 'certificate.npz', **arrays)
    dump(folder / 'result.json', case)
    return case


def main(args):
    OUT.mkdir(exist_ok=True)
    gate = HERE.parent / 'generation_moments/penalized_em/COMPLETE.json'
    if args.wait_for_mcem:
        began = time.monotonic()
        while not gate.exists():
            if time.monotonic() - began > 8 * 3600:
                raise TimeoutError('MCEM slot did not become free.')
            time.sleep(15)
    evidence = AW / 'nature_upgrade/data_claim_audit/paired_70_field_encodings.csv'
    units = pd.read_csv(evidence)
    allowed_names = set(units.loc[units.unit.isin(['mg/L', 'ug/L', 'g/L', 'v/v']), 'field'])
    first = FixedProblem('strict', style_epsilon=1e-5)
    support = np.array([j for j, name in enumerate(first.features) if name in allowed_names], int)
    selected = [first.features[j] for j in support]
    assert all(set(c['mandatory']) <= set(support) for c in first.contexts.values())
    units[units.field.isin(selected)].to_csv(OUT / 'allowed_fields.csv', index=False)
    protocol = dict(trigger='Field-level export showed that the primary sparse designs use both calibrated and unanchored input scales. This post-diagnostic challenge checks reachability when only explicitly quantified concentration fields may change.',
                    concentration_units=['mg/L', 'ug/L', 'g/L', 'v/v'], allowed_fields=selected,
                    excluded='Non-concentration indices, arbitrary units, missing units, unmatched fields, derived fields and technical standards.',
                    policies=['strict', 'exceedance'], linear_stale_margins=MARGINS,
                    required_rho=.1, style_epsilon=1e-5,
                    objective='Minimum pure target Mahalanobis squared distance on the full allowed concentration subspace.',
                    scope='Reachability and complete witness challenge; neither minimum-edit optimum nor physical process feasibility.',
                    primary_candidate_pool_unchanged=True, evidence_sha256=file_sha(evidence),
                    code_sha256=file_sha(__file__))
    dump(OUT / 'protocol.json', protocol)
    records = []
    for policy in ('strict', 'exceedance'):
        problem = first if policy == 'strict' else FixedProblem(policy, style_epsilon=1e-5)
        for beer in sorted(problem.contexts):
            for margin in MARGINS:
                folder = OUT / policy / f'beer_{beer}_margin_{int(round(margin * 1000))}'
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / 'result.json'
                record = json.loads(path.read_text()) if path.exists() else solve(problem, beer, support, margin, folder)
                records.append(record)
                print(json.dumps(record), flush=True)
    table = pd.DataFrame(records)
    table.to_csv(OUT / 'all_cases.csv', index=False)
    summary = []
    for (policy, beer), group in table.groupby(['policy', 'beer']):
        zero = group[group.linear_margin.eq(0)].iloc[0]
        accepted = group[group.full_model_accepted]
        best = accepted.sort_values('actual_distance_reduction', ascending=False).iloc[0] if len(accepted) else None
        summary.append(dict(policy=policy, beer=int(beer), allowed_fields=len(support),
                            zero_margin_linear_feasible=bool(zero.linear_feasible),
                            zero_margin_geometry_impossible=bool(zero.geometry_impossible),
                            any_complete_witness=bool(len(accepted)), complete_witness_count=len(accepted),
                            best_full_distance_reduction=None if best is None else float(best.actual_distance_reduction),
                            best_full_actual_edits=None if best is None else int(best.actual_fields_changed)))
    pd.DataFrame(summary).to_csv(OUT / 'summary.csv', index=False)
    dump(OUT / 'COMPLETE.json', dict(complete=True, source_policy_cases=22,
                  margin_cases=len(table), allowed_fields=len(support),
                  primary_candidate_pool_unchanged=True,
                  full_feasible_cases=int(sum(row['any_complete_witness'] for row in summary))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-for-mcem', action='store_true')
    with threadpool_limits(limits=1):
        main(parser.parse_args())
