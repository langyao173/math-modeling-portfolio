"""Measure remaining dispatch space with complete historical recovery metadata.

Eight fixed conditions compare the existing 24-hour controller with an offline
312-hour lexicographic LP. The original physical transition is reused throughout.
AI-assisted: OpenAI Codex, 2026-10-19.
"""
from __future__ import annotations

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

from src import v3_hourly as h
from src import v13_event_design as design
from src import v13_event_information as event
from src import v14_terminal_reserve as terminal
from src.v10_prepare_inputs import make_case
from src.v11_recovery_sensitivity import digest, recovery_windows
from src.v13_storage_duration import reference_field_comparison

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/v15/dispatch_bound_v2'
FAILED_PREDECESSOR = ROOT / 'outputs/v15/dispatch_bound'
FAILED_ANALYSIS_SHA = '47acaa292c6ec15e9d90c035407ba107ca26d48fb7ea0f37029cd124af0102ff'
BRANCH_HOUR = 48
ENS_CAP_SLACK = 1e-6
FEAS_TOL = 1e-6
BOUND_TOL = 5e-6
COST_TOL = 1e-5
CONDITIONS = tuple(
    {'condition_id': f'phase{phase:02d}_p{int(power*100)}_{end}',
     'phase_hours': phase, 'duration_hours': 48, 'strategy': 'coordinated',
     'storage_power_multiplier': power, 'terminal_condition': end}
    for phase in (0, 21) for power in (1., 1.5) for end in ('free', 'full'))
GRID = tuple({**condition, 'case_id': condition['condition_id'] + '_' + method,
              'method': method} for method in ('rolling', 'offline')
             for condition in CONDITIONS)
WIDTHS = (('gen', 7), ('ch', 2), ('dis', 2), ('soc', 2), ('fuel', 3),
          ('direct_short', 3), ('out', 1), ('repay', 1), ('debt', 1),
          ('unserved', 1), ('spill', 1))


def layout(nstep):
    """The frozen original planner's 24-variable hourly block layout."""
    result, offset = {}, 0
    for name, width in WIDTHS:
        indices = np.arange(offset, offset + nstep * width)
        result[name] = indices.reshape(nstep, width) if width > 1 else indices
        offset += nstep * width
    assert offset == 24 * nstep
    return result


def action_from_row(row):
    return {'gen': np.array([row[g + '_gw'] for g in h.GENERATORS]),
            'ch': np.array([row[s + '_charge_gw'] for s in h.STORES]),
            'dis': np.array([row[s + '_discharge_gw'] for s in h.STORES]),
            'direct_short': np.array([row[f + '_direct_unserved_mtce'] for f in h.FUELS]),
            'out': float(row.shift_out_gw), 'repay': float(row.shift_in_gw),
            'unserved': float(row.unserved_gw), 'spill': float(row.spill_gw)}


def vector_from_table(frame):
    n = len(frame)
    ix = layout(n)
    values = np.zeros(24 * n)
    columns = {
        'gen': [g + '_gw' for g in h.GENERATORS],
        'ch': [s + '_charge_gw' for s in h.STORES],
        'dis': [s + '_discharge_gw' for s in h.STORES],
        'soc': [s + '_soc_gwh' for s in h.STORES],
        'fuel': [f + '_stock_mtce' for f in h.FUELS],
        'direct_short': [f + '_direct_unserved_mtce' for f in h.FUELS],
        'out': ['shift_out_gw'], 'repay': ['shift_in_gw'], 'debt': ['shift_debt_gwh'],
        'unserved': ['unserved_gw'], 'spill': ['spill_gw']}
    for name, fields in columns.items():
        values[ix[name].ravel()] = frame[fields].to_numpy(float).ravel()
    return values


def finite_bounds(bounds):
    lower = np.array([-np.inf if lo is None else lo for lo, _ in bounds])
    upper = np.array([np.inf if hi is None else hi for _, hi in bounds])
    return lower, upper


def primal_residual(x, A_ub, b_ub, A_eq, b_eq, bounds):
    lower, upper = finite_bounds(bounds)
    return {'equality': float(np.abs(A_eq @ x - b_eq).max(initial=0)),
            'inequality': float(np.maximum(A_ub @ x - b_ub, 0).max(initial=0)),
            'lower_bound': float(np.maximum(lower - x, 0).max(initial=0)),
            'upper_bound': float(np.maximum(x - upper, 0).max(initial=0))}


def solution_certificate(solution, c, A_ub, b_ub, A_eq, b_eq, bounds):
    """Check the actual HiGHS primal and dual certificates in common units."""
    receipt = {'status': int(solution.status), 'success': bool(solution.success),
               'message': str(solution.message), 'iterations': int(solution.nit)}
    if not solution.success:
        return receipt
    x = solution.x
    y_eq, y_ub = solution.eqlin.marginals, solution.ineqlin.marginals
    y_lo, y_hi = solution.lower.marginals, solution.upper.marginals
    lower, upper = finite_bounds(bounds)
    dual_value = float(b_eq @ y_eq + b_ub @ y_ub)
    dual_value += float(lower[np.isfinite(lower)] @ y_lo[np.isfinite(lower)])
    dual_value += float(upper[np.isfinite(upper)] @ y_hi[np.isfinite(upper)])
    stationarity = c - A_eq.T @ y_eq - A_ub.T @ y_ub - y_lo - y_hi
    primal_value = float(c @ x)
    receipt.update(primal_residual=primal_residual(x, A_ub, b_ub, A_eq, b_eq, bounds),
                   primal_objective=primal_value, dual_objective=dual_value,
                   primal_minus_dual=primal_value - dual_value,
                   dual_stationarity_residual=float(np.abs(stationarity).max(initial=0)),
                   dual_sign_residual=float(max(np.maximum(y_ub, 0).max(initial=0),
                                                np.maximum(-y_lo, 0).max(initial=0),
                                                np.maximum(y_hi, 0).max(initial=0))))
    return receipt


def resource_objective(case, start, state, nstep):
    """Actual fuel use, operating/cycling/shift cost and net SOC draw."""
    ix, cfg = layout(nstep), case['config']
    c = np.zeros(24 * nstep)
    for j, gen in enumerate(h.GENERATORS):
        cost = (cfg['fuel_value_million_cny_mtce'][gen] /
                (h.GWH_PER_MTCE * case['fuel_efficiency'][gen])
                if gen in ('coal', 'gas') else cfg['generation_cost_cny_kwh'][gen])
        c[ix['gen'][:, j]] = cost
    c[ix['ch']] = c[ix['dis']] = cfg['storage_cycle_cost_cny_kwh']
    c[ix['out']] = c[ix['repay']] = cfg['shift_cost_cny_kwh']
    c[ix['soc'][-1]] = -cfg['stored_electricity_value_million_cny_gwh']
    constant = sum(case['direct_fuel_demand_mtce'][start:start+nstep, j].sum() *
                   cfg['fuel_value_million_cny_mtce'][fuel] for j, fuel in enumerate(h.FUELS))
    constant += sum(state['soc'].values()) * cfg['stored_electricity_value_million_cny_gwh']
    initial_fuel_value = sum(case['initial_fuel_stocks_mtce'][j] *
                             cfg['fuel_value_million_cny_mtce'][fuel]
                             for j, fuel in enumerate(h.FUELS))
    constant += initial_fuel_value * cfg['stock_holding_rate_per_year'] * nstep / 8760
    return c, float(constant)


def offline_simulation(case, baseline, require_full, target, calls, solver_logs):
    """Replay the fixed prefix, solve once at h48, then apply each real action."""
    original_plan, original_solver = h._plan, h.linprog
    original_horizon = case['config']['forecast_hours']
    context, planned = {}, {}
    case['config']['forecast_hours'] = case['hours'] - BRANCH_HOUR

    def solve(original_c, *, A_ub, b_ub, A_eq, b_eq, bounds, method):
        assert context['hour'] == BRANCH_HOUR
        n = case['hours'] - BRANCH_HOUR
        ix = layout(n)
        assert len(original_c) == len(bounds) == 24 * n
        submitted = list(bounds)
        for index in ix['direct_short'].ravel():
            submitted[index] = (0., 0.)
        if require_full:
            for j, store in enumerate(h.STORES):
                energy = case['storage'][store]['energy_gwh']
                assert bounds[ix['soc'][-1, j]] == (0, energy)
                submitted[ix['soc'][-1, j]] = (energy, energy)
        baseline_x = vector_from_table(baseline.iloc[BRANCH_HOUR:])
        feasible = primal_residual(baseline_x, A_ub, b_ub, A_eq, b_eq, submitted)
        assert max(feasible.values()) <= FEAS_TOL, ('Baseline inclusion failed', feasible)
        c1 = np.zeros(len(original_c))
        c1[ix['unserved']] = 1.
        c2, constant = resource_objective(case, BRANCH_HOUR, context['state'], n)
        problem = dict(A_ub_data=A_ub.data, A_ub_indices=A_ub.indices,
                       A_ub_indptr=A_ub.indptr, A_ub_shape=np.array(A_ub.shape),
                       A_eq_data=A_eq.data, A_eq_indices=A_eq.indices,
                       A_eq_indptr=A_eq.indptr, A_eq_shape=np.array(A_eq.shape),
                       b_ub=b_ub, b_eq=b_eq, bounds=np.array(finite_bounds(submitted)),
                       original_objective=original_c, ens_objective=c1,
                       resource_objective=c2, resource_constant=np.array(constant), baseline_x=baseline_x)
        np.savez_compressed(target / 'submitted_problem.npz', **problem)
        for stage, c in (('minimum_ens', c1), ('minimum_resource_at_ens_bound', c2)):
            if stage == 'minimum_ens':
                actual_A, actual_b = A_ub, b_ub
            else:
                actual_A = vstack((A_ub, csr_matrix(c1.reshape(1, -1))), format='csr')
                actual_b = np.r_[b_ub, planned['ens_primal_gwh'] + ENS_CAP_SLACK]
            log = {'stage': stage, 'branch_hour': BRANCH_HOUR, 'hours': n,
                   'baseline_feasibility': feasible, 'method': method,
                   'objective_sha256': terminal.array_hash(c),
                   'A_ub_sha256': terminal.matrix_hash(actual_A),
                   'b_ub_sha256': terminal.array_hash(actual_b),
                   'A_eq_sha256': terminal.matrix_hash(A_eq),
                   'b_eq_sha256': terminal.array_hash(b_eq),
                   'bounds_sha256': terminal.bounds_hash(submitted)}
            solver_logs.append(log)
            solution = original_solver(c, A_ub=actual_A, b_ub=actual_b,
                                       A_eq=A_eq, b_eq=b_eq, bounds=submitted, method=method)
            log.update(solution_certificate(solution, c, actual_A, actual_b, A_eq, b_eq, submitted))
            assert solution.success, (stage, solution.status, solution.message)
            assert max(log['primal_residual'].values()) <= FEAS_TOL, log
            assert log['dual_stationarity_residual'] <= FEAS_TOL and log['dual_sign_residual'] <= FEAS_TOL, log
            assert abs(log['primal_minus_dual']) <= COST_TOL, log
            np.savez_compressed(target / (stage + '_solution.npz'),
                                x=solution.x, equality_marginals=solution.eqlin.marginals,
                                inequality_marginals=solution.ineqlin.marginals,
                                lower_marginals=solution.lower.marginals,
                                upper_marginals=solution.upper.marginals)
            if stage == 'minimum_ens':
                planned.update(ens_primal_gwh=float(c1 @ solution.x),
                               ens_dual_gwh=log['dual_objective'],
                               ens_primal_minus_dual_gwh=log['primal_minus_dual'])
                assert max(planned['ens_primal_gwh'], planned['ens_dual_gwh']) <= float(c1 @ baseline_x) + BOUND_TOL
        planned.update(x=solution.x, resource_constant=constant,
                       resource_objective_value=float(c2 @ solution.x + constant),
                       second_stage_ens=float(c1 @ solution.x), indices=ix)
        assert abs(planned['second_stage_ens'] - planned['ens_primal_gwh']) <= ENS_CAP_SLACK + FEAS_TOL
        return solution

    def plan(actual, start, state, strategy):
        assert actual is case and strategy == 'coordinated'
        record = {'hour': start, 'pre_state': deepcopy(state)}
        calls.append(record)
        try:
            if start < BRANCH_HOUR:
                action = action_from_row(baseline.iloc[start])
            else:
                if start == BRANCH_HOUR:
                    context.update(hour=start, state=deepcopy(state))
                    original_plan(actual, start, state, strategy)
                    context.clear()
                i = start - BRANCH_HOUR
                action = {name: planned['x'][indices[i]].copy()
                          for name, indices in planned['indices'].items()}
            record['action'] = deepcopy(action)
            return action
        except Exception as failure:
            record.update(error=str(failure), error_type=type(failure).__name__)
            raise

    h._plan, h.linprog = plan, solve
    try:
        result = h.simulate(case, 'coordinated')
        result['offline_certificate'] = {key: value for key, value in planned.items()
                                         if key not in ('x', 'indices')}
        return result
    finally:
        h._plan, h.linprog = original_plan, original_solver
        case['config']['forecast_hours'] = original_horizon


def reference_folder(row):
    arm = 'base' if row['storage_power_multiplier'] == 1. else 'power'
    if row['phase_hours'] == 0:
        end = 'full_soc' if row['terminal_condition'] == 'full' else 'unconstrained'
        return ROOT / 'outputs/v14/terminal_reserve' / f'{end}_{arm}'
    if row['terminal_condition'] == 'free':
        return ROOT / 'outputs/v13/event_design' / f'phase21_duration48_{arm}'
    return None


def compare_columns(actual, reference):
    assert list(actual.columns) == list(reference.columns) and len(actual) == len(reference)
    differences = {name: float(np.abs(actual[name].to_numpy() - reference[name].to_numpy()).max(initial=0))
                   for name in reference.columns}
    return {'passed': all(value <= FEAS_TOL for value in differences.values()),
            'all_column_max_absolute_difference': differences}


def prepare():
    if (OUT / 'protocol.json').exists():
        raise FileExistsError('Keep the existing protocol unchanged')
    checks, _, sources = terminal.input_preflight()
    assert digest(FAILED_PREDECESSOR / 'analysis.json') == FAILED_ANALYSIS_SHA
    failed = json.loads((FAILED_PREDECESSOR / 'analysis.json').read_text('utf-8'))
    assert failed['status'] == 'FAILED_OR_INCOMPLETE_FIXED_COHORT'
    assert failed['actual_solver_calls'] == failed['successful_solver_calls'] == 2880
    assert sum(r['status'] == 'NOT_RUN_AFTER_BASELINE_GATE_FAILURE' for r in failed['cases']) == 8
    for name, expected in failed['output_sha256'].items():
        assert digest(FAILED_PREDECESSOR / name) == expected, name
    for name in ('src/v15_dispatch_bound.py', 'tests/test_v15_dispatch_bound.py',
                 'outputs/v15/dispatch_bound/protocol.json', 'outputs/v15/dispatch_bound/analysis.json',
                 'outputs/v15/dispatch_bound/code_review.json',
                 'outputs/v15/dispatch_bound_v2/predecessor_failure_review.json'):
        sources[name] = digest(ROOT / name)
    sources['src/v15_dispatch_bound_v2.py'] = digest(Path(__file__))
    sources['tests/test_v15_dispatch_bound_v2.py'] = digest(ROOT / 'tests/test_v15_dispatch_bound_v2.py')
    for study in ('outputs/v14/terminal_reserve', 'outputs/v13/event_design'):
        folder = ROOT / study
        receipt = json.loads((folder / 'analysis.json').read_text('utf-8'))
        assert receipt['status'].startswith('COMPLETED_') and not receipt['source_drift']
        for relative, expected in receipt['output_sha256'].items():
            assert digest(folder / relative) == expected, relative
        for name in ('analysis.json', 'protocol.json', 'execution_review.json'):
            path = folder / name
            sources[path.relative_to(ROOT).as_posix()] = digest(path)
    for row in GRID[:8]:
        folder = reference_folder(row)
        if folder is not None:
            for name in ('actions_states.csv', 'summary.json', 'recovery.csv'):
                path = folder / name
                sources[path.relative_to(ROOT).as_posix()] = digest(path)
    protocol = {
        'status': 'PREPARED_NOT_RUN', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_class': 'FIXED_PREFIX_OFFLINE_LEXICOGRAPHIC_PHYSICAL_LOWER_BOUND',
        'predecessor': {'analysis_sha256': FAILED_ANALYSIS_SHA,
                        'status': failed['status'], 'actual_solver_calls': 2880,
                        'completed_physical_trajectory_rows': 2880,
                        'validated_physical_rows': failed['physical_replay_hours'],
                        'offline_solver_calls': 0},
        'correction': 'Add the historical recovery metadata post_shock_observation_hours = observation_end_hour - shock_end_hour before complete field comparison; count physical replay rows when the physical simulation returns and validated rows only after every gate passes. No scientific cohort, objective, input, constraint, threshold or stopping-rule change.',
        'scientific_design_changed': False,
        'conditions': CONDITIONS, 'case_grid': GRID, 'paths': 16, 'hours_per_path': 360,
        'branch_hour': BRANCH_HOUR, 'actual_scientific_solver_calls_at_freeze': 0,
        'expected_new_solver_calls': 8 * 360 + 8 * 2, 'expected_physical_replay_hours': 16 * 360,
        'execution_order': 'All eight rolling baselines first: six complete frozen-field bridges and two new phase21/full baselines. Release eight offline cases only if every baseline passes. After release attempt all eight even on failure; retain all outcomes.',
        'prefix_contract': 'Each offline path replays exactly the paired actual rolling h0..47 actions through original h.simulate. All h48 pre-state fields, including previous_available, thermal, fuel, SOC and debt, continue unchanged. No reset or new initial state.',
        'physical_domain': 'Original h._plan constraints on absolute h48..359; original ramp, fuel balances, availability, storage energy/power/efficiencies and every absolute 24h debt deadline. Direct fuel shortages fixed at zero, with full baseline vector inclusion checked before either solve. Only the two global-end SOC bounds become full in full cases; free cases retain original bounds.',
        'first_objective': 'Unweighted sum of electricity unserved GWh over h48..359; all other objective entries exactly zero. Save optimal primal/dual certificates and submitted matrices/bounds.',
        'second_objective': 'Keep ENS <= first optimum + 1e-6 GWh. Minimize actual remaining-window physical resource cost: fuel burn, nonfossil operation, storage cycling, shifting and net SOC draw, with original direct fuel service and prorated initial-fuel holding as constants. No shortage penalty or carbon-charge term. Check objective plus constants against exported physical account.',
        'information': 'Offline LP receives the actual complete remaining 312h input arrays once at h48. It is a model-scoped lower bound conditional on the fixed prefix, rather than an implementable forecast controller. Any gap may reflect information, receding decisions and differing objectives together.',
        'reporting': 'All eight conditions, baseline ENS, first-stage lower bound, actual second-stage ENS, signed gap/ratio, primal/dual residuals, cost/CO2/peak/full terminal fuel/SOC/thermal/debt and original recovery windows. No phase selection or new population-success estimate.',
        'tolerances': {'primal_and_dual': FEAS_TOL, 'ens_cap_slack_gwh': ENS_CAP_SLACK,
                       'baseline_lower_bound_gwh': BOUND_TOL, 'cost_identity_million_cny': COST_TOL},
        'source_sha256': sources, 'input_checks': checks,
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__},
        'new_native_runs': 0, 'paid_services_used': False,
        'commands': ['python -m src.v15_dispatch_bound_v2 --prepare', 'python -m src.v15_dispatch_bound_v2 --run']}
    OUT.mkdir(parents=True, exist_ok=True)
    event.write_json(OUT / 'protocol.json', protocol)
    pd.DataFrame(GRID).to_csv(OUT / 'case_grid.csv', index=False)
    print(json.dumps({'status': protocol['status'], 'paths': 16, 'scientific_solver_calls': 0}))
    return protocol


def recovery_report(table, config):
    """Preserve the original recovery rule and its observation-duration field."""
    rows = recovery_windows(table, design.recovery_configuration(config))
    for row in rows:
        row['post_shock_observation_hours'] = row['observation_end_hour'] - row['shock_end_hour']
    return rows


def run_case(row):
    target = OUT / row['case_id']
    target.mkdir(parents=True, exist_ok=False)
    config = design.case_configuration(row)
    case = make_case(config)
    full = row['terminal_condition'] == 'full'
    record = {**row, 'status': 'RUNNING', 'complete': False,
              'validated_physical_replay_hours': 0}
    calls, solver_logs = [], []
    event.write_json(target / 'parameters_initial_state.json', {
        'configuration': config, 'terminal_condition': row['terminal_condition'],
        'initial_state_and_controls': design.initial_state_and_controls(case)})
    design.exogenous_table(case).to_csv(target / 'exogenous_inputs.csv', index=False, float_format='%.17g')
    started = time.perf_counter()
    try:
        if row['method'] == 'rolling':
            result = terminal.simulate_terminal(case, 'coordinated', 'original', full, calls, solver_logs)
        else:
            baseline = pd.read_csv(OUT / (row['condition_id'] + '_rolling/actions_states.csv'), float_precision='round_trip')
            result = offline_simulation(case, baseline, full, target, calls, solver_logs)
        result['timeseries'].to_csv(target / 'actions_states.csv', index=False, float_format='%.17g')
        table = pd.read_csv(target / 'actions_states.csv', float_precision='round_trip')
        record['physical_replay_hours'] = len(table)
        original = h.audit(case, table)
        residuals, independent = event.independent_audit(case, table)
        residuals.to_csv(target / 'independent_physical_audit.csv', index=False, float_format='%.17g')
        end_audit = terminal.terminal_audit(case, table, full)
        record.update(original_physical_audit=original, independent_physical_audit=independent,
                      terminal_soc_audit=end_audit)
        assert original['passed'] and independent['passed'] and end_audit['passed']
        assert np.abs(table[[f + '_direct_unserved_mtce' for f in h.FUELS]].to_numpy()).max() <= FEAS_TOL
        summary = terminal.summarize(case, table, 'coordinated', full)
        if row['method'] == 'offline':
            summary.update(forecast_hours=case['hours'] - BRANCH_HOUR,
                           information_contract='Fixed paired actual h0..47 actions; complete true h48..359 inputs once at h48',
                           evidence_class='OFFLINE_FIXED_PREFIX_PHYSICAL_LOWER_BOUND')
        recovery = recovery_report(table, config)
        event.write_json(target / 'summary.json', summary)
        pd.DataFrame(recovery).to_csv(target / 'recovery.csv', index=False)
        if row['method'] == 'rolling':
            folder = reference_folder(row)
            if folder is not None:
                reference = pd.read_csv(folder / 'actions_states.csv', float_precision='round_trip')
                bridge = compare_columns(table, reference)
                bridge['summary'] = reference_field_comparison(summary, json.loads((folder / 'summary.json').read_text('utf-8')))
                receipt = json.loads((folder.parent / 'analysis.json').read_text('utf-8'))
                old_case = next(item for item in receipt['cases'] if item['case_id'] == folder.name)
                bridge['recovery'] = reference_field_comparison(recovery, old_case['recovery'])
                bridge['passed'] = bridge['passed'] and bridge['summary']['passed'] and bridge['recovery']['passed']
                event.write_json(target / 'frozen_anchor_comparison.json', bridge)
                assert bridge['passed'], bridge
            else:
                free_id = row['condition_id'].replace('_full', '_free') + '_rolling'
                reference = pd.read_csv(OUT / free_id / 'actions_states.csv', float_precision='round_trip')
                prefix = compare_columns(table.iloc[:336], reference.iloc[:336])
                event.write_json(target / 'before_final_window_comparison.json', prefix)
                assert prefix['passed'], prefix
            assert len(solver_logs) == 360
        else:
            prefix = compare_columns(table.iloc[:BRANCH_HOUR], baseline.iloc[:BRANCH_HOUR])
            assert prefix['passed'], prefix
            certificate = result['offline_certificate']
            assert abs(table.loc[table.hour.ge(BRANCH_HOUR), 'unserved_gw'].sum() - certificate['second_stage_ens']) <= FEAS_TOL
            cost = summary['primary_h48_359_' + terminal.PHYSICAL_COST]
            assert abs(cost - certificate['resource_objective_value']) <= COST_TOL
            record.update(offline_certificate=certificate, fixed_prefix_comparison=prefix)
            assert len(solver_logs) == 2
        record.update(status='COMPLETED_AND_VALIDATED', complete=True, metrics=summary, recovery=recovery,
                      validated_physical_replay_hours=len(table))
    except Exception as failure:
        record.update(status='FAILED', error=str(failure), error_type=type(failure).__name__, traceback=traceback.format_exc())
        if not (target / 'actions_states.csv').exists():
            try:
                prefix = event.prefix_from_calls(case, calls)
                prefix.to_csv(target / 'observed_prefix_actions_states.csv', index=False, float_format='%.17g')
                record['observed_prefix_hours'] = len(prefix)
            except Exception as evidence_failure:
                record['prefix_evidence_error'] = {'type': type(evidence_failure).__name__, 'error': str(evidence_failure)}
    finally:
        record.update(wall_seconds=time.perf_counter() - started,
                      actual_solver_calls=len(solver_logs),
                      successful_solver_calls=sum(log.get('success') is True for log in solver_logs),
                      attempted_physical_hours=len(calls))
        with (target / 'solver_logs.jsonl').open('w', encoding='utf-8') as stream:
            for log in solver_logs:
                stream.write(json.dumps(event.json_value(log), ensure_ascii=False, allow_nan=False) + '\n')
        with (target / 'actual_pre_states_actions.jsonl').open('w', encoding='utf-8') as stream:
            for call in calls:
                compact = {key: value for key, value in call.items() if key in ('hour', 'pre_state', 'action', 'error', 'error_type')}
                stream.write(json.dumps(event.json_value(compact), ensure_ascii=False, allow_nan=False) + '\n')
        event.write_json(target / 'result.json', record)
    return record


def run():
    from src.io_utils import verify_review
    protocol = json.loads((OUT / 'protocol.json').read_text('utf-8'))
    assert protocol['status'] == 'PREPARED_NOT_RUN' and protocol['case_grid'] == list(GRID)
    assert not (OUT / 'analysis.json').exists(), 'Preserve the actual experiment'
    verify_review(OUT / 'code_review.json')
    for relative, expected in protocol['source_sha256'].items():
        assert digest(ROOT / relative) == expected, relative
    results, released = [], False
    for i, row in enumerate(GRID):
        if i == 8:
            released = all(result['complete'] for result in results)
        if i >= 8 and not released:
            record = {**row, 'complete': False, 'status': 'NOT_RUN_AFTER_BASELINE_GATE_FAILURE'}
        else:
            record = run_case(row)
        results.append(record)
        print(row['case_id'], record['status'], flush=True)
    drift = [path for path, expected in protocol['source_sha256'].items() if digest(ROOT / path) != expected]
    rows = []
    for condition in CONDITIONS:
        baseline = next(r for r in results if r['case_id'] == condition['condition_id'] + '_rolling')
        offline = next(r for r in results if r['case_id'] == condition['condition_id'] + '_offline')
        valid = baseline['complete'] and offline['complete'] and not drift
        item = {**condition, 'pair_status': 'COMPLETE' if valid else 'INCOMPLETE',
                'baseline_status': baseline['status'], 'offline_status': offline['status']}
        if valid:
            b = baseline['metrics']['primary_h48_359_unserved_gwh']
            certificate = offline['offline_certificate']
            lower = certificate['ens_dual_gwh']
            assert max(lower, certificate['ens_primal_gwh']) <= b + BOUND_TOL
            item.update(baseline_ens_gwh=b, offline_ens_primal_gwh=certificate['ens_primal_gwh'],
                        offline_ens_dual_gwh=lower, offline_ens_primal_minus_dual_gwh=certificate['ens_primal_minus_dual_gwh'],
                        actual_second_stage_ens_gwh=offline['offline_certificate']['second_stage_ens'],
                        baseline_minus_offline_dual_bound_gwh=b - lower,
                        baseline_minus_offline_feasible_ens_gwh=b - certificate['second_stage_ens'],
                        offline_dual_to_baseline_fraction=lower / b if b else None)
            for metric in terminal.COMPARE_METRICS:
                item['offline_minus_baseline_' + metric] = offline['metrics'][metric] - baseline['metrics'][metric]
        rows.append(item)
    pd.DataFrame(rows).to_csv(OUT / 'all_eight_paired_bounds.csv', index=False, float_format='%.17g')
    pd.DataFrame([{key: value for key, value in r.items() if not isinstance(value, (dict, list))} for r in results]).to_csv(OUT / 'all_case_status.csv', index=False)
    successful = sum(r.get('successful_solver_calls', 0) for r in results)
    complete = all(r['complete'] for r in results) and not drift
    if complete:
        assert successful == 2896 and sum(r['physical_replay_hours'] for r in results) == 5760
    report = {'status': 'COMPLETED_SIXTEEN_VALIDATED_FIXED_PREFIX_PATHS' if complete else 'FAILED_OR_INCOMPLETE_FIXED_COHORT',
              'protocol': protocol, 'cases': results, 'source_drift': drift,
              'actual_solver_calls': sum(r.get('actual_solver_calls', 0) for r in results),
              'successful_solver_calls': successful,
              'physical_replay_hours': sum(r.get('physical_replay_hours', r.get('observed_prefix_hours', 0)) for r in results),
              'validated_physical_replay_hours': sum(r.get('validated_physical_replay_hours', 0) for r in results),
              'new_native_runs': 0,
              'output_sha256': {p.relative_to(OUT).as_posix(): digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'analysis.json'}}
    event.write_json(OUT / 'analysis.json', report)
    print(report['status'], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()


if __name__ == '__main__':
    main()
