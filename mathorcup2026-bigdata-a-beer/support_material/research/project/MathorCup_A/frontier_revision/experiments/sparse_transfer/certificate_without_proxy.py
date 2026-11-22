"""Lower bounds for the broader model without the extra linear stale guard.

The frozen nonlinear stale constraint is relaxed by the solver. The returned
dual bound therefore applies to the broader box + classification + distance +
nonlinear-stale problem. It is deliberately kept separate from all guarded
model lower bounds. Existing fully verified guarded witnesses remain valid
upper bounds for either problem. No candidate here enters primary selection.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'

from pathlib import Path
import argparse
import json
import math
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import perspective_solver as core
from control_problem import FixedProblem, AW, file_sha

HERE = Path(__file__).resolve().parent
OUT = HERE / 'certificate_without_proxy'
SDP = HERE / 'perspective_sdp'


class WithoutLinearGuard(FixedProblem):
    def linear_constraints(self, beer, margin=0., style_epsilon=None):
        a, rhs = super().linear_constraints(beer, margin, style_epsilon)
        if not np.array_equal(a[0], self.a) or rhs[0] != -margin:
            raise AssertionError('The omitted row must be exactly the linear stale guard.')
        return a[1:], rhs[1:]


def saved_sdp_split(precision):
    archive = np.load(SDP / 'diagonal_sdp.npz')
    audit = json.loads((SDP / 'diagonal_sdp_audit.json').read_text())
    if np.max(abs(archive['P'] - (precision + precision.T) / 2)) > 1e-10:
        raise AssertionError('SDP split does not match the frozen precision.')
    if np.linalg.eigvalsh(archive['R']).min() <= 0 or archive['diagonal'].min() < 0:
        raise AssertionError('Saved diagonal split is not safely positive semidefinite.')
    if np.max(abs(archive['P'] - archive['R'] - np.diag(archive['diagonal']))) > 1e-10:
        raise AssertionError('Diagonal splitting identity failed.')
    return archive['P'], archive['diagonal'], archive['R'], archive['factor'], audit


def solve_case(problem, beer, seconds):
    folder = OUT / problem.policy / f'beer_{beer}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'DONE.json').exists():
        prior = json.loads((folder / 'DONE.json').read_text())
        if prior['seconds_limit'] != seconds:
            raise ValueError('Existing result has another time budget.')
        return prior
    warm, warm_info = core.accepted_warm(problem, beer)
    if warm is None:
        raise RuntimeError('A fully checked original guarded witness is required.')
    core.write_json(folder / 'warm_start.json', warm_info)
    np.savez_compressed(folder / 'warm_start.npz', delta=warm['delta'], support=warm['support'])
    a, rhs = problem.linear_constraints(beer)
    np.savez_compressed(folder / 'retained_linear_constraints.npz', A=a, rhs=rhs,
                        omitted_guard=problem.a, omitted_rhs=np.array([0.]))
    protocol = dict(beer=int(beer), policy=problem.policy, rho=.1, style_epsilon=1e-5,
                    seconds_limit=seconds, threads=1, omit_only_linear_stale_guard=True,
                    nonlinear_stale_relaxed_for_lower_bound=True,
                    complete_upper_checks_keep_both_stale_conditions=True,
                    lower_bound_scope='Broader model: box, chemical policy, classification, distance, and complete nonlinear frozen stale constraint; no linear stale guard.',
                    candidate_pool_enters_primary_selection=False,
                    wrapper_sha256=file_sha(__file__),
                    solver_core_sha256=file_sha(HERE / 'perspective_solver.py'),
                    control_sha256=file_sha(HERE / 'control_problem.py'),
                    diagonal_split_sha256=file_sha(SDP / 'diagonal_sdp.npz'),
                    predictor_sha256=file_sha(AW / 'q1/sensory_ensemble.joblib'),
                    model_sha256=file_sha(AW / 'q3/transfer_and_augmentation_models.joblib'),
                    classifier_sha256=file_sha(AW / 'competition_revision/style_feasibility/frozen_style_classifier.joblib'))
    core.write_json(folder / 'protocol.json', protocol)
    pool, bound = core.solve_perspective(problem, beer, seconds, warm, folder)
    bound.update(method='perspective_sdp_without_linear_stale_guard',
                 lower_bound_scope=protocol['lower_bound_scope'])
    core.write_json(folder / 'solver_bound.json', bound)
    np.savez_compressed(folder / 'raw_pool.npz',
                        delta=np.stack([x['delta'] for x in pool]) if pool else np.empty((0, problem.p)),
                        binary=np.stack([x['binary'] for x in pool]) if pool else np.empty((0, problem.p)),
                        epigraph=np.stack([x['epigraph'] for x in pool]) if pool else np.empty((0, problem.p)),
                        objective=np.array([x['solver_objective'] for x in pool]))
    rows = [dict(warm['row'], candidate_id=0, method='guarded_original_warm',
                 required_rho=.1, required_proxy_margin=0.)]
    for candidate in pool:
        row, _ = problem.decode(beer, candidate['delta'], candidate['support'])
        row.update(candidate_id=len(rows), method='unguarded_relaxation_pool',
                   required_rho=.1, required_proxy_margin=0.)
        rows.append(row)
    # Inherited check_rows explicitly evaluates a @ delta and f_star, so all
    # accepted rows satisfy the stronger primary model as well.
    checks, sensory = problem.check_rows(rows)
    pd.DataFrame(rows).to_csv(folder / 'raw_profiles.csv', index=False)
    checks.to_csv(folder / 'full_checks.csv', index=False)
    sensory.to_csv(folder / 'predicted_sensory.csv', index=False)
    valid = checks[checks.accepted]
    assert len(valid) >= 1
    upper = int(valid.raw_fields_changed.min())
    lower = max(len(problem.contexts[beer]['mandatory']), int(math.ceil(bound['dual_bound'] - 1e-6)))
    if lower > upper:
        raise AssertionError('Broader model lower bound exceeds a fully checked guarded upper bound.')
    result = dict(beer=int(beer), policy=problem.policy, rho=.1,
                  broader_dual_bound=bound['dual_bound'], broader_integer_lower_bound=lower,
                  stronger_guarded_full_upper_bound=upper,
                  relaxation_primal=bound['primal_bound'], seconds_limit=seconds,
                  solver_seconds=bound['seconds'], status=bound['status'],
                  stored_pool_points=len(pool), fully_guarded_accepted=len(valid),
                  omitted_constraints=1, retained_classification_rows=len(rhs),
                  bound_scope=protocol['lower_bound_scope'],
                  source_files={name: file_sha(folder / name) for name in
                                ('protocol.json', 'solver_bound.json', 'raw_pool.npz',
                                 'raw_profiles.csv', 'full_checks.csv')})
    core.write_json(folder / 'DONE.json', result)
    print('BROADER_CERTIFICATE ' + json.dumps(result), flush=True)
    return result


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    gate = HERE / 'perspective/INTEGRITY_DONE.json'
    if args.wait_for_integrity:
        start = time.monotonic()
        while not gate.exists():
            if time.monotonic() - start > 8 * 3600:
                raise TimeoutError('Clean perspective audit did not finish within eight hours.')
            time.sleep(15)
    if not gate.exists():
        raise RuntimeError('Run only after the clean main perspective integrity audit.')
    core.diagonal_split = saved_sdp_split
    results = []
    for policy in ('strict', 'exceedance'):
        problem = WithoutLinearGuard(policy, style_epsilon=1e-5)
        for beer in sorted(problem.contexts):
            results.append(solve_case(problem, beer, args.seconds))
            pd.DataFrame(results).to_csv(OUT / 'comparison.csv', index=False)
    assert len(results) == 22
    core.write_json(OUT / 'COMPLETE.json', dict(complete=True, cases=22,
                    seconds_per_case=args.seconds, primary_candidate_pool_unchanged=True,
                    constraint_scope='No extra linear stale guard; nonlinear stale relaxed only for the lower bound.',
                    comparison_sha256=file_sha(OUT / 'comparison.csv')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--wait-for-integrity', action='store_true')
    with threadpool_limits(limits=1):
        main(parser.parse_args())
