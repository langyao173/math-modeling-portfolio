"""Minimize physical cost while retaining the rolling carbon and peak limits.

The eight V15 fixed-prefix domains and their certified minimum ENS are reused.
Three declared constraints arms isolate carbon, peak, and their joint effect.
AI-assisted: OpenAI Codex, 2026-10-19.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix, vstack

from src import v15_dispatch_bound_v2 as parent
from src.io_utils import verify_review

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/v16/joint_dispatch'
PARENT = ROOT / 'outputs/v15/dispatch_bound_v2'
PARENT_SHA = '9b26b42ad3f649e317f21cf23b704763274ee2ae82b944ec1c81380e129cf52b'
ARMS = ('carbon_cap', 'peak_cap', 'joint_caps')
GRID = tuple({**condition, 'arm': arm,
              'case_id': condition['condition_id'] + '_' + arm}
             for condition in parent.CONDITIONS for arm in ARMS)


def load_json(path):
    return json.loads(path.read_text('utf-8'))


def carbon_objective(case, hours):
    """Combustion Mt CO2 per hourly GWh; direct service is a fixed constant."""
    indices = parent.layout(hours)
    c = np.zeros(24 * hours)
    for fuel in ('coal', 'gas'):
        c[indices['gen'][:, parent.h.GENERATORS.index(fuel)]] = (
            parent.h.FUEL_EMISSION_MT_PER_MTCE[fuel] /
            (parent.h.GWH_PER_MTCE * case['fuel_efficiency'][fuel]))
    return c


def load_domain(condition):
    cid = condition['condition_id']
    folder = PARENT / (cid + '_offline')
    rolling = pd.read_csv(PARENT / (cid + '_rolling/actions_states.csv'), float_precision='round_trip')
    parameters = load_json(folder / 'parameters_initial_state.json')
    case = parent.make_case(parameters['configuration'])
    with np.load(folder / 'submitted_problem.npz', allow_pickle=False) as stored:
        problem = {name: stored[name] for name in stored.files}
    certificate = load_json(folder / 'result.json')['offline_certificate']
    return case, rolling, problem, certificate


def constrained_domain(problem, emissions, baseline_peak, ens_bound, arm):
    """Only append the predeclared limits to the unchanged original domain."""
    assert arm in ARMS
    matrix = csr_matrix((problem['A_ub_data'], problem['A_ub_indices'], problem['A_ub_indptr']),
                        shape=tuple(problem['A_ub_shape']))
    equality = csr_matrix((problem['A_eq_data'], problem['A_eq_indices'], problem['A_eq_indptr']),
                          shape=tuple(problem['A_eq_shape']))
    rows = [matrix, csr_matrix(problem['ens_objective'].reshape(1, -1))]
    rhs = [problem['b_ub'], np.array([ens_bound + parent.ENS_CAP_SLACK])]
    if arm in ('carbon_cap', 'joint_caps'):
        rows.append(csr_matrix(emissions.reshape(1, -1)))
        rhs.append(np.array([float(emissions @ problem['baseline_x'])]))
    bounds = [(None if np.isneginf(lo) else float(lo), None if np.isposinf(hi) else float(hi))
              for lo, hi in zip(*problem['bounds'])]
    if arm in ('peak_cap', 'joint_caps'):
        indices = np.flatnonzero(problem['ens_objective'])
        for index in indices:
            lower, upper = bounds[index]
            bounds[index] = (lower, baseline_peak if upper is None else min(upper, baseline_peak))
    return vstack(rows, format='csr'), np.concatenate(rhs), equality, problem['b_eq'], bounds


def replay(case, rolling, solution, calls):
    """Feed each original action into the original physical state transition."""
    old_plan = parent.h._plan
    indices = parent.layout(case['hours'] - parent.BRANCH_HOUR)

    def plan(actual, hour, state, strategy):
        assert actual is case and strategy == 'coordinated'
        if hour < parent.BRANCH_HOUR:
            action = parent.action_from_row(rolling.iloc[hour])
        else:
            step = hour - parent.BRANCH_HOUR
            action = {name: solution[index[step]].copy() for name, index in indices.items()}
        calls.append({'hour': hour, 'pre_state': deepcopy(state), 'action': deepcopy(action)})
        return action

    parent.h._plan = plan
    try:
        return parent.h.simulate(case, 'coordinated')
    finally:
        parent.h._plan = old_plan


def prepare():
    assert not (OUT / 'protocol.json').exists(), 'Preserve the frozen protocol'
    assert parent.digest(PARENT / 'analysis.json') == PARENT_SHA
    old = load_json(PARENT / 'analysis.json')
    assert old['status'] == 'COMPLETED_SIXTEEN_VALIDATED_FIXED_PREFIX_PATHS' and not old['source_drift']
    for review in ('code_review.json', 'execution_review.json'):
        verify_review(PARENT / review)
    sources = dict(old['protocol']['source_sha256'])
    for relative, expected in old['output_sha256'].items():
        path = PARENT / relative
        assert parent.digest(path) == expected, relative
        sources[path.relative_to(ROOT).as_posix()] = expected
    for name in ('analysis.json', 'protocol.json', 'code_review.json', 'execution_review.json'):
        path = PARENT / name
        sources[path.relative_to(ROOT).as_posix()] = parent.digest(path)
    for name in ('src/v16_joint_dispatch.py', 'tests/test_v16_joint_dispatch.py'):
        sources[name] = parent.digest(ROOT / name)
    for name, expected in sources.items():
        assert parent.digest(ROOT / name) == expected, name
    protocol = {
        'status': 'PREPARED_NOT_RUN', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_class': 'FIXED_PREFIX_OFFLINE_COST_UNDER_BASELINE_CARBON_AND_PEAK_CAPS',
        'parent_analysis_sha256': PARENT_SHA, 'conditions': parent.CONDITIONS, 'arms': ARMS,
        'case_grid': GRID, 'paths': 24, 'hours_per_path': 360, 'expected_LP_calls': 24,
        'expected_actual_physical_hours': 8640, 'actual_scientific_LP_calls_at_freeze': 0,
        'domain': 'Reuse the exact eight V15 submitted original CSR/RHS/bounds and fixed actual first 48h. Preserve direct fuel shortage=0, all daily debt deadlines and free/full end SOC. Do not rerun or change the original rolling baselines or ENS minimizations.',
        'objective': 'Actual physical resource cost at the certified original minimum ENS plus the unchanged 1e-6 GWh cap. Retain all operating, fuel, cycling, shift, net SOC draw and holding constants. No carbon-charge or shortage-penalty term.',
        'carbon_cap': 'Primary h48..359 actual combustion CO2 no greater than the paired rolling path. Since direct fuel service is fixed, constrain the exact coal+gas generation emissions vector against that vector evaluated on the saved rolling baseline. No additional carbon allowance.',
        'peak_cap': 'Every primary-hour electricity ENS <= paired rolling primary peak in GW. Tighten all unserved upper bounds; no additional peak allowance.',
        'comparisons': 'All eight conditions times carbon-only, peak-only, joint limits, compared with the same rolling baseline and V15 unconstrained resource minimum. Keep every signed metric, all terminal fuel/SOC/thermal/debt states and any failure. No post-result arm, phase, or threshold selection.',
        'information': 'All new paths still receive the complete true remaining 312h once at h48. This evaluates achievable tradeoffs in the matched model domain; it does not isolate forecast value or establish an implementable rolling-controller improvement.',
        'verification': 'Verify each saved baseline inclusion before its one LP; save submitted matrices, exact solution and all marginals; validate primal/dual/KKT; replay all 360 actions through the original physical transition, original and independent audits; match first 48h and full h48 branch state; reconcile actual resource, CO2, ENS and peak against the submitted vectors.',
        'tolerances': {'feasibility': parent.FEAS_TOL, 'certificate_cost': parent.COST_TOL,
                       'ens_cap_slack_gwh': parent.ENS_CAP_SLACK},
        'source_sha256': sources,
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__},
        'new_native_runs': 0, 'commands': ['python -B -X utf8 -m src.v16_joint_dispatch --prepare',
                                         'python -B -X utf8 -m src.v16_joint_dispatch --run']}
    OUT.mkdir(parents=True, exist_ok=True)
    parent.event.write_json(OUT / 'protocol.json', protocol)
    pd.DataFrame(GRID).to_csv(OUT / 'case_grid.csv', index=False)
    print(json.dumps({'status': protocol['status'], 'paths': 24, 'scientific_LP_calls': 0}))


def run_case(row):
    target = OUT / row['case_id']
    target.mkdir(exist_ok=False)
    record = {**row, 'status': 'RUNNING', 'complete': False, 'actual_LP_calls': 0,
              'successful_LP_calls': 0, 'actual_physical_hours': 0, 'validated_physical_hours': 0}
    calls, logs = [], []
    started = time.perf_counter()
    try:
        case, rolling, problem, inherited = load_domain(row)
        n = case['hours'] - parent.BRANCH_HOUR
        emissions = carbon_objective(case, n)
        peak = float(rolling.iloc[48:].unserved_gw.max())
        A, b, E, d, bounds = constrained_domain(problem, emissions, peak, inherited['ens_primal_gwh'], row['arm'])
        inclusion = parent.primal_residual(problem['baseline_x'], A, b, E, d, bounds)
        assert max(inclusion.values()) <= parent.FEAS_TOL, inclusion
        parent.event.write_json(target / 'parameters_initial_state.json', {
            'configuration': parent.design.case_configuration(row),
            'initial_state_and_controls': parent.design.initial_state_and_controls(case),
            'arm': row['arm'], 'terminal_condition': row['terminal_condition']})
        parent.design.exogenous_table(case).to_csv(target / 'exogenous_inputs.csv', index=False, float_format='%.17g')
        np.savez_compressed(target / 'submitted_problem.npz', A_ub_data=A.data, A_ub_indices=A.indices,
            A_ub_indptr=A.indptr, A_ub_shape=A.shape, A_eq_data=E.data, A_eq_indices=E.indices,
            A_eq_indptr=E.indptr, A_eq_shape=E.shape, b_ub=b, b_eq=d,
            bounds=np.array(parent.finite_bounds(bounds)), objective=problem['resource_objective'],
            resource_constant=problem['resource_constant'], ens_objective=problem['ens_objective'],
            carbon_objective=emissions, baseline_x=problem['baseline_x'])
        log = {'condition_id': row['condition_id'], 'arm': row['arm'], 'method': 'highs',
               'baseline_inclusion': inclusion, 'A_ub_sha256': parent.terminal.matrix_hash(A),
               'b_ub_sha256': parent.terminal.array_hash(b), 'A_eq_sha256': parent.terminal.matrix_hash(E),
               'b_eq_sha256': parent.terminal.array_hash(d), 'bounds_sha256': parent.terminal.bounds_hash(bounds),
               'objective_sha256': parent.terminal.array_hash(problem['resource_objective'])}
        logs.append(log)
        record['actual_LP_calls'] += 1
        solution = parent.h.linprog(problem['resource_objective'], A_ub=A, b_ub=b, A_eq=E, b_eq=d,
                                    bounds=bounds, method='highs')
        log.update(parent.solution_certificate(solution, problem['resource_objective'], A, b, E, d, bounds))
        record['successful_LP_calls'] = int(solution.success)
        assert solution.success, log
        np.savez_compressed(target / 'solution.npz', x=solution.x,
            equality_marginals=solution.eqlin.marginals, inequality_marginals=solution.ineqlin.marginals,
            lower_marginals=solution.lower.marginals, upper_marginals=solution.upper.marginals)
        assert max(log['primal_residual'].values()) <= parent.FEAS_TOL, log
        assert max(log['dual_stationarity_residual'], log['dual_sign_residual']) <= parent.FEAS_TOL, log
        assert abs(log['primal_minus_dual']) <= parent.COST_TOL, log
        result = replay(case, rolling, solution.x, calls)
        table = result['timeseries']
        table.to_csv(target / 'actions_states.csv', index=False, float_format='%.17g')
        record['actual_physical_hours'] = len(table)
        original = parent.h.audit(case, table)
        residuals, independent = parent.event.independent_audit(case, table)
        residuals.to_csv(target / 'independent_physical_audit.csv', index=False, float_format='%.17g')
        terminal = parent.terminal.terminal_audit(case, table, row['terminal_condition'] == 'full')
        prefix = parent.compare_columns(table.iloc[:48], rolling.iloc[:48])
        old_calls = [json.loads(line) for line in (PARENT / (row['condition_id'] + '_rolling') /
                     'actual_pre_states_actions.jsonl').read_text('utf-8').splitlines()]
        branch = parent.reference_field_comparison(parent.event.json_value(calls[48]['pre_state']), old_calls[48]['pre_state'])
        assert original['passed'] and independent['passed'] and terminal['passed'] and prefix['passed'] and branch['passed']
        vector_error = float(np.abs(parent.vector_from_table(table.iloc[48:]) - solution.x).max())
        assert vector_error <= parent.FEAS_TOL
        metrics = parent.terminal.summarize(case, table, 'coordinated', row['terminal_condition'] == 'full')
        metrics.update(information_contract='Fixed actual h0..47; full true h48..359 once; baseline carbon/peak limits',
                       evidence_class='OFFLINE_FIXED_PREFIX_JOINT_GOAL_TRADEOFF', forecast_hours=312)
        key = 'primary_h48_359_'
        physical = metrics[key + parent.terminal.PHYSICAL_COST]
        assert abs(physical - float(problem['resource_objective'] @ solution.x + problem['resource_constant'])) <= parent.COST_TOL
        direct = sum(case['direct_fuel_demand_mtce'][48:, j].sum() * parent.h.FUEL_EMISSION_MT_PER_MTCE[fuel]
                     for j, fuel in enumerate(parent.h.FUELS))
        assert abs(metrics[key + 'total_co2_mt'] - float(emissions @ solution.x + direct)) <= parent.FEAS_TOL
        assert metrics[key + 'unserved_gwh'] <= inherited['ens_primal_gwh'] + parent.ENS_CAP_SLACK + parent.FEAS_TOL
        baseline = load_json(PARENT / (row['condition_id'] + '_rolling/summary.json'))
        if row['arm'] in ('carbon_cap', 'joint_caps'):
            assert metrics[key + 'total_co2_mt'] <= baseline[key + 'total_co2_mt'] + parent.FEAS_TOL
        if row['arm'] in ('peak_cap', 'joint_caps'):
            assert metrics[key + 'peak_unserved_gw'] <= peak + parent.FEAS_TOL
        parent.event.write_json(target / 'summary.json', metrics)
        recovery = parent.recovery_report(table, parent.design.case_configuration(row))
        pd.DataFrame(recovery).to_csv(target / 'recovery.csv', index=False)
        record.update(status='COMPLETED_AND_VALIDATED', complete=True, validated_physical_hours=len(table),
            metrics=metrics, original_physical_audit=original, independent_physical_audit=independent,
            terminal_soc_audit=terminal, fixed_prefix_comparison=prefix, branch_state_comparison=branch,
            solution_physical_vector_error=vector_error, recovery=recovery)
    except Exception as failure:
        record.update(status='FAILED', error=str(failure), error_type=type(failure).__name__, traceback=traceback.format_exc())
    finally:
        record.update(wall_seconds=time.perf_counter()-started, attempted_physical_hours=len(calls))
        for name, rows in (('solver_logs.jsonl', logs), ('actual_pre_states_actions.jsonl', calls)):
            with (target / name).open('w', encoding='utf-8') as stream:
                for item in rows:
                    stream.write(json.dumps(parent.event.json_value(item), ensure_ascii=False, allow_nan=False)+'\n')
        parent.event.write_json(target / 'result.json', record)
    return record


def run():
    assert not (OUT / 'analysis.json').exists(), 'Preserve the original complete experiment identity'
    protocol = load_json(OUT / 'protocol.json')
    assert protocol['status'] == 'PREPARED_NOT_RUN'
    verify_review(OUT / 'code_review.json')
    for name, expected in protocol['source_sha256'].items():
        assert parent.digest(ROOT / name) == expected, name
    results = []
    for row in GRID:
        record = run_case(row)
        results.append(record)
        parent.event.write_json(OUT / 'progress.json', {'cases': results})
        print(json.dumps({key: record[key] for key in ('case_id', 'status', 'actual_LP_calls', 'actual_physical_hours')}) , flush=True)
    pd.DataFrame([{key: row[key] for key in ('condition_id', 'arm', 'case_id', 'status', 'complete')} |
                  row.get('metrics', {}) for row in results]).to_csv(
                      OUT / 'all_24_case_metrics.csv', index=False, float_format='%.17g')
    drift = [name for name, expected in protocol['source_sha256'].items() if parent.digest(ROOT / name) != expected]
    paired = []
    for record in results:
        item = {key: record[key] for key in ('condition_id', 'arm', 'case_id', 'status', 'complete')}
        if record['complete']:
            for reference in ('rolling', 'offline'):
                baseline = load_json(PARENT / (record['condition_id'] + '_' + reference) / 'summary.json')
                for metric, value in record['metrics'].items():
                    if isinstance(value, (float, int)) and isinstance(baseline.get(metric), (float, int)):
                        item['minus_' + reference + '_' + metric] = value - baseline[metric]
        paired.append(item)
    pd.DataFrame(paired).to_csv(OUT / 'all_24_paired_metrics.csv', index=False, float_format='%.17g')
    complete = len(results) == 24 and all(row['complete'] for row in results) and not drift
    audit = {'status': 'COMPLETED_ALL_24_FIXED_CONSTRAINT_PATHS' if complete else 'FAILED_OR_INCOMPLETE_FIXED_COHORT',
             'protocol': protocol, 'cases': results, 'source_drift': drift,
             'actual_LP_calls': sum(r['actual_LP_calls'] for r in results),
             'successful_LP_calls': sum(r['successful_LP_calls'] for r in results),
             'actual_physical_hours': sum(r['actual_physical_hours'] for r in results),
             'validated_physical_hours': sum(r['validated_physical_hours'] for r in results),
             'output_sha256': {p.relative_to(OUT).as_posix(): parent.digest(p) for p in OUT.rglob('*')
                               if p.is_file() and p.name not in ('analysis.json', 'progress.json', 'code_review.json')}}
    parent.event.write_json(OUT / 'analysis.json', audit)
    print(json.dumps({key: audit[key] for key in ('status', 'actual_LP_calls', 'successful_LP_calls', 'actual_physical_hours', 'validated_physical_hours')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        run()
    else:
        parser.print_help()
