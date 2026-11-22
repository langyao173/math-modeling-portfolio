"""Test peak protection while matching each reference optimum's full end state.

AI-assisted: OpenAI Codex, 2026-10-19. A separate fixed 16-path follow-up;
neither the V15 lower bound nor the V16 three-arm experiment is modified.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack

from src import v16_joint_dispatch as joint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/v16/endpoint_peak'
JOINT_SHA = '051f999d86ebebcfa1d3efba178fa5d6965c1fcd98d623888a277a8638f5fe6a'
FAMILIES = ('unconstrained_resource_minimum', 'carbon_capped_resource_minimum')
GRID = tuple({**condition, 'reference_family': family,
              'case_id': condition['condition_id'] + '_' + family}
             for condition in joint.parent.CONDITIONS for family in FAMILIES)


def endpoint_indices(hours):
    ix = joint.parent.layout(hours)
    return np.r_[ix['fuel'][-1], ix['soc'][-1],
                 ix['gen'][-1, [joint.parent.h.GENERATORS.index(fuel) for fuel in ('coal', 'gas')]]]


def match_endpoint(E, rhs, reference_x, hours):
    """Append seven equalities without deleting an original bound or equation."""
    indices = endpoint_indices(hours)
    rows = csr_matrix((np.ones(7), (np.arange(7), indices)), shape=(7, len(reference_x)))
    return vstack((E, rows), format='csr'), np.r_[rhs, reference_x[indices]]


def reference_folder(row):
    if row['reference_family'] == FAMILIES[0]:
        return joint.PARENT / (row['condition_id'] + '_offline'), 'minimum_resource_at_ens_bound_solution.npz'
    return joint.OUT / (row['condition_id'] + '_carbon_cap'), 'solution.npz'


def prepare():
    assert not (OUT / 'protocol.json').exists(), 'Preserve the original protocol'
    assert joint.parent.digest(joint.OUT / 'analysis.json') == JOINT_SHA
    previous = joint.load_json(joint.OUT / 'analysis.json')
    assert previous['status'] == 'COMPLETED_ALL_24_FIXED_CONSTRAINT_PATHS' and not previous['source_drift']
    joint.verify_review(joint.OUT / 'execution_review.json')
    sources = dict(previous['protocol']['source_sha256'])
    for name, expected in previous['output_sha256'].items():
        path = joint.OUT / name
        assert joint.parent.digest(path) == expected, name
        sources[path.relative_to(ROOT).as_posix()] = expected
    for name in ('analysis.json', 'code_review.json', 'execution_review.json'):
        path = joint.OUT / name
        sources[path.relative_to(ROOT).as_posix()] = joint.parent.digest(path)
    for name in ('src/v16_endpoint_peak.py', 'tests/test_v16_endpoint_peak.py'):
        sources[name] = joint.parent.digest(ROOT / name)
    protocol = {
        'status': 'PREPARED_NOT_RUN', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_class': 'POST_RESULT_FIXED_ENDPOINT_PEAK_PROTECTION_FOLLOWUP',
        'reason': 'The completed 24-arm experiment preserved minimum resource cost while capping peak, but four peak-capped paths had different final coal generation. Test full state matching directly instead of interpreting that endpoint change as an identical continuation state.',
        'parent_analysis_sha256': JOINT_SHA, 'case_grid': GRID, 'paths': 16,
        'expected_new_LP_calls': 16, 'expected_physical_hours_if_all_feasible': 5760,
        'actual_scientific_LP_calls_at_freeze': 0,
        'reference_families': {
            FAMILIES[0]: 'Each matched V15 no-additional-carbon minimum-resource solution, at its own saved minimum ENS plus 1e-6 GWh.',
            FAMILIES[1]: 'Each matched V16 carbon_cap minimum-resource solution, retaining its original rolling CO2 cap.'},
        'endpoint_contract': 'Append seven equations fixing final three fuel stocks, two storage SOC and two thermal outputs to the respective reference optimum saved LP vector. Preserve every original equation/bound and the global free/full SOC contract. Original final debt=0; final previous_available is the same fixed exogenous input. Verify all seven actual physical endpoints and the complete reference final state separately. These are the reference optima endpoints, not rolling-controller endpoints.',
        'peak_contract': 'Every h48..359 electricity unserved value <= paired rolling-controller primary peak. Retain the original certified ENS cap and all full direct fuel services.',
        'objective': 'Exactly the same physical resource coefficient vector and constants as the respective certified reference domain. The reference optimum is a cost lower bound. Report every signed new-minus-reference difference and primal/dual certificates.',
        'failure_policy': 'Attempt all 16 predeclared cases once, including after any failed or infeasible case; retain the original solver status and submitted problem. A reference optimum need not satisfy the added peak cap, so feasibility is tested, never assumed. No fallback, relaxing an equality, or changing a threshold.',
        'information': 'Same fixed actual first 48h and complete true remaining 312h once at h48, with original physical transitions. Fixed-model endpoint-matched offline schedule comparison.',
        'source_sha256': sources,
        'tolerances': {'feasibility': joint.parent.FEAS_TOL, 'cost_certificate_million_cny': joint.parent.COST_TOL,
                       'ens_cap_slack_gwh': joint.parent.ENS_CAP_SLACK},
        'environment': previous['protocol']['environment'], 'new_native_runs': 0,
        'commands': ['python -B -X utf8 -m src.v16_endpoint_peak --prepare',
                     'python -B -X utf8 -m src.v16_endpoint_peak --run']}
    OUT.mkdir(parents=True, exist_ok=True)
    joint.parent.event.write_json(OUT / 'protocol.json', protocol)
    pd.DataFrame(GRID).to_csv(OUT / 'case_grid.csv', index=False)
    print(json.dumps({'status': protocol['status'], 'paths': 16, 'scientific_LP_calls': 0}))


def run_case(row):
    target = OUT / row['case_id']; target.mkdir(exist_ok=False)
    record = {**row, 'status': 'RUNNING', 'complete': False, 'actual_LP_calls': 0,
              'successful_LP_calls': 0, 'actual_physical_hours': 0, 'validated_physical_hours': 0}
    logs, calls = [], []; started = time.perf_counter()
    try:
        case, rolling, problem, inherited = joint.load_domain(row)
        n = case['hours'] - joint.parent.BRANCH_HOUR
        folder, solution_name = reference_folder(row)
        with np.load(folder / solution_name, allow_pickle=False) as stored:
            reference_x = stored['x'].copy()
        arm = 'peak_cap' if row['reference_family'] == FAMILIES[0] else 'joint_caps'
        emissions = joint.carbon_objective(case, n)
        peak = float(rolling.iloc[48:].unserved_gw.max())
        A, b, original_E, original_d, bounds = joint.constrained_domain(problem, emissions, peak, inherited['ens_primal_gwh'], arm)
        E, d = match_endpoint(original_E, original_d, reference_x, n)
        indices = endpoint_indices(n)
        np.savez_compressed(target / 'submitted_problem.npz', A_ub_data=A.data, A_ub_indices=A.indices,
            A_ub_indptr=A.indptr, A_ub_shape=A.shape, A_eq_data=E.data, A_eq_indices=E.indices,
            A_eq_indptr=E.indptr, A_eq_shape=E.shape, b_ub=b, b_eq=d,
            bounds=np.array(joint.parent.finite_bounds(bounds)), objective=problem['resource_objective'],
            resource_constant=problem['resource_constant'], reference_x=reference_x, endpoint_indices=indices)
        log = {'method': 'highs', 'A_ub_sha256': joint.parent.terminal.matrix_hash(A),
               'A_eq_sha256': joint.parent.terminal.matrix_hash(E), 'b_ub_sha256': joint.parent.terminal.array_hash(b),
               'b_eq_sha256': joint.parent.terminal.array_hash(d), 'bounds_sha256': joint.parent.terminal.bounds_hash(bounds),
               'objective_sha256': joint.parent.terminal.array_hash(problem['resource_objective'])}
        logs.append(log); record['actual_LP_calls'] = 1
        solution = joint.parent.h.linprog(problem['resource_objective'], A_ub=A, b_ub=b, A_eq=E, b_eq=d,
                                          bounds=bounds, method='highs')
        log.update(joint.parent.solution_certificate(solution, problem['resource_objective'], A, b, E, d, bounds))
        record['successful_LP_calls'] = int(solution.success)
        if not solution.success:
            record.update(status='INFEASIBLE' if solution.status == 2 else 'SOLVER_FAILED',
                          solver_status=int(solution.status), solver_message=str(solution.message))
            return record
        np.savez_compressed(target / 'solution.npz', x=solution.x,
            equality_marginals=solution.eqlin.marginals, inequality_marginals=solution.ineqlin.marginals,
            lower_marginals=solution.lower.marginals, upper_marginals=solution.upper.marginals)
        assert max(log['primal_residual'].values()) <= joint.parent.FEAS_TOL, log
        assert max(log['dual_stationarity_residual'], log['dual_sign_residual']) <= joint.parent.FEAS_TOL, log
        assert abs(log['primal_minus_dual']) <= joint.parent.COST_TOL, log
        result = joint.replay(case, rolling, solution.x, calls)
        table = result['timeseries']; record['actual_physical_hours'] = len(table)
        table.to_csv(target / 'actions_states.csv', index=False, float_format='%.17g')
        residuals, independent = joint.parent.event.independent_audit(case, table)
        residuals.to_csv(target / 'independent_physical_audit.csv', index=False, float_format='%.17g')
        end = joint.parent.terminal.terminal_audit(case, table, row['terminal_condition'] == 'full')
        prefix = joint.parent.compare_columns(table.iloc[:48], rolling.iloc[:48])
        actual_vector = joint.parent.vector_from_table(table.iloc[48:])
        endpoint_error = float(np.abs(actual_vector[indices] - reference_x[indices]).max())
        reference_table = pd.read_csv(folder / 'actions_states.csv', float_precision='round_trip')
        state_columns = [fuel+'_stock_mtce' for fuel in joint.parent.h.FUELS] + [store+'_soc_gwh' for store in joint.parent.h.STORES] + ['coal_gw', 'gas_gw', 'shift_debt_gwh']
        state_differences = {name: float(table[name].iloc[-1]-reference_table[name].iloc[-1]) for name in state_columns}
        assert result['audit']['passed'] and independent['passed'] and end['passed'] and prefix['passed']
        assert max(endpoint_error, max(abs(v) for v in state_differences.values())) <= joint.parent.FEAS_TOL
        assert float(np.abs(actual_vector-solution.x).max()) <= joint.parent.FEAS_TOL
        metrics = joint.parent.terminal.summarize(case, table, 'coordinated', row['terminal_condition'] == 'full')
        key = 'primary_h48_359_'; cost = metrics[key+joint.parent.terminal.PHYSICAL_COST]
        reference_cost = float(problem['resource_objective'] @ reference_x + problem['resource_constant'])
        reference_summary = joint.load_json(folder / 'summary.json')
        assert abs(reference_cost-reference_summary[key+joint.parent.terminal.PHYSICAL_COST]) <= joint.parent.COST_TOL
        assert abs(cost-float(problem['resource_objective']@solution.x+problem['resource_constant'])) <= joint.parent.COST_TOL
        assert cost >= reference_cost-joint.parent.COST_TOL
        assert abs(metrics[key+'total_co2_mt']-reference_summary[key+'total_co2_mt']) <= joint.parent.FEAS_TOL
        assert metrics[key+'peak_unserved_gw'] <= peak+joint.parent.FEAS_TOL
        assert metrics[key+'unserved_gwh'] <= inherited['ens_primal_gwh']+joint.parent.ENS_CAP_SLACK+joint.parent.FEAS_TOL
        metrics.update(forecast_hours=312, information_contract='Fixed actual 48h; true remaining 312h; full reference endpoint matching',
                       evidence_class='OFFLINE_FIXED_REFERENCE_ENDPOINT_PEAK_PROTECTION')
        joint.parent.event.write_json(target / 'summary.json', metrics)
        pd.DataFrame(joint.parent.recovery_report(table, joint.parent.design.case_configuration(row))).to_csv(target / 'recovery.csv',index=False)
        record.update(status='COMPLETED_AND_VALIDATED', complete=True, validated_physical_hours=len(table), metrics=metrics,
            original_physical_audit=result['audit'], independent_physical_audit=independent, terminal_soc_audit=end,
            fixed_prefix_comparison=prefix, endpoint_error=endpoint_error, final_state_differences=state_differences,
            resource_cost_minus_reference_million_cny=cost-reference_cost,
            total_co2_minus_reference_mt=metrics[key+'total_co2_mt']-reference_summary[key+'total_co2_mt'],
            peak_minus_rolling_gw=metrics[key+'peak_unserved_gw']-peak)
    except Exception as failure:
        record.update(status='FAILED', error=str(failure), error_type=type(failure).__name__, traceback=traceback.format_exc())
    finally:
        record.update(wall_seconds=time.perf_counter()-started, attempted_physical_hours=len(calls))
        for name, values in (('solver_logs.jsonl',logs),('actual_pre_states_actions.jsonl',calls)):
            with (target/name).open('w',encoding='utf-8') as stream:
                for item in values: stream.write(json.dumps(joint.parent.event.json_value(item),ensure_ascii=False,allow_nan=False)+'\n')
        joint.parent.event.write_json(target/'result.json',record)
    return record


def run():
    assert not (OUT/'analysis.json').exists(), 'Preserve the original complete cohort identity'
    protocol = joint.load_json(OUT/'protocol.json')
    assert protocol['status'] == 'PREPARED_NOT_RUN'
    joint.verify_review(OUT/'code_review.json')
    for name, expected in protocol['source_sha256'].items(): assert joint.parent.digest(ROOT/name)==expected,name
    results=[]
    for row in GRID:
        result=run_case(row); results.append(result)
        joint.parent.event.write_json(OUT/'progress.json',{'cases':results})
        print(json.dumps({key:result[key] for key in ('case_id','status','actual_LP_calls','actual_physical_hours')}),flush=True)
    drift=[name for name,expected in protocol['source_sha256'].items() if joint.parent.digest(ROOT/name)!=expected]
    pd.DataFrame([{key:row[key] for key in ('condition_id','reference_family','case_id','status','complete')} |
                  {key:row.get(key) for key in ('resource_cost_minus_reference_million_cny','total_co2_minus_reference_mt','peak_minus_rolling_gw','endpoint_error')}
                  for row in results]).to_csv(OUT/'all_16_paired_metrics.csv',index=False,float_format='%.17g')
    status='COMPLETED_ALL_16_ENDPOINT_MATCHED_PATHS' if all(row['complete'] for row in results) and not drift else 'FAILED_OR_INCOMPLETE_FIXED_COHORT'
    audit={'status':status,'protocol':protocol,'cases':results,'source_drift':drift,
        'attempted_all_16_cases':len(results)==16,
        'infeasible_cases':sum(row['status']=='INFEASIBLE' for row in results),
        **{key:sum(row[key] for row in results) for key in ('actual_LP_calls','successful_LP_calls','actual_physical_hours','validated_physical_hours')},
        'output_sha256':{p.relative_to(OUT).as_posix():joint.parent.digest(p) for p in OUT.rglob('*')
                         if p.is_file() and p.name not in ('analysis.json','progress.json','code_review.json')}}
    joint.parent.event.write_json(OUT/'analysis.json',audit)
    print(json.dumps({key:audit[key] for key in ('status','actual_LP_calls','successful_LP_calls','actual_physical_hours','validated_physical_hours')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.prepare: prepare()
    elif args.run: run()
    else: parser.print_help()
