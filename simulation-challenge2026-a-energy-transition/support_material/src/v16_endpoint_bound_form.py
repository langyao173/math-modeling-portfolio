"""Replace seven unit endpoint equations by exactly equivalent fixed bounds.

AI-assisted: OpenAI Codex, 2026-10-19. Preserve the original NPZ bytes and
acceptance tolerances; separately save the equivalent seven-fixed-bound form.
"""
import argparse
from datetime import datetime, timezone
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src import v16_endpoint_peak as endpoint

joint = endpoint.joint
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/v16/endpoint_bound_form'
PARENT = ROOT / 'outputs/v16/endpoint_peak'
DIAGNOSTIC = ROOT / 'outputs/v16/endpoint_numerics'
DIAGNOSTIC_SHA = '1b58310c968d4a8e23ebb15f9f1797f14e5267c6d86d029039f3d67e2e3aa3fc'
PARENT_SHA = '747b77e8530593d76109cb7342655c409a01d66169d357879ccb158d774f2c1f'
METHODS = ('highs-ds', 'highs-ipm')
GRID = tuple({**row, 'original_case_id': row['case_id'], 'method': method,
              'case_id': row['case_id'] + '_bounds_' + method.replace('-', '_')}
             for row in endpoint.GRID
             if row['phase_hours'] == 0 and row['reference_family'] == endpoint.FAMILIES[1]
             for method in METHODS)


def load_problem(path):
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def equivalent_bound_domain(problem):
    """A unit equality x_i=t and an existing bound intersection equal [t,t]."""
    A = csr_matrix((problem['A_ub_data'], problem['A_ub_indices'], problem['A_ub_indptr']),
                   shape=tuple(problem['A_ub_shape']))
    E = csr_matrix((problem['A_eq_data'], problem['A_eq_indices'], problem['A_eq_indptr']),
                   shape=tuple(problem['A_eq_shape']))
    bounds = [(None if np.isneginf(lo) else float(lo), None if np.isposinf(hi) else float(hi))
              for lo, hi in zip(*problem['bounds'])]
    indices = problem['endpoint_indices']
    targets = problem['reference_x'][indices]
    assert len(indices) == len(set(indices)) == 7
    selected = E[-7:].tocoo()
    assert selected.nnz == 7
    np.testing.assert_array_equal(selected.row, np.arange(7))
    np.testing.assert_array_equal(selected.col, indices)
    np.testing.assert_array_equal(selected.data, np.ones(7))
    np.testing.assert_array_equal(problem['b_eq'][-7:], targets)
    # Exact membership matters: no clipping, tolerance or domain enlargement.
    assert np.all(targets >= problem['bounds'][0, indices])
    assert np.all(targets <= problem['bounds'][1, indices])
    fixed = list(bounds)
    for index, value in zip(indices, targets):
        fixed[int(index)] = (float(value), float(value))
    original = (A, problem['b_ub'], E, problem['b_eq'], bounds)
    reduced = (A, problem['b_ub'], E[:-7].tocsr(), problem['b_eq'][:-7], fixed)
    return original, reduced


def map_to_original(solution, problem):
    """Transfer fixed-bound multipliers to the original seven free equality duals."""
    mapped = deepcopy(solution)
    if solution.success:
        indices = problem['endpoint_indices']
        mu = solution.lower.marginals[indices] + solution.upper.marginals[indices]
        mapped.eqlin.marginals = np.r_[solution.eqlin.marginals, mu]
        mapped.lower.marginals[indices] = 0.
        mapped.upper.marginals[indices] = 0.
    return mapped


def solve_saved_problem(problem, method):
    assert method in METHODS
    original, reduced = equivalent_bound_domain(problem)
    A, b, E, d, bounds = reduced
    raw = joint.parent.h.linprog(problem['objective'], A_ub=A, b_ub=b,
        A_eq=E, b_eq=d, bounds=bounds, method=method, options={'presolve': False})
    return map_to_original(raw, problem), original, raw, reduced


def prepare():
    assert not (OUT / 'protocol.json').exists(), 'Preserve the original diagnostic protocol'
    assert joint.parent.digest(PARENT / 'analysis.json') == PARENT_SHA
    original = joint.load_json(PARENT / 'analysis.json')
    assert original['actual_LP_calls'] == 16 and original['successful_LP_calls'] == 12
    failed = {row['case_id'] for row in original['cases'] if row['status'] == 'INFEASIBLE'}
    assert failed == {row['original_case_id'] for row in GRID} and len(failed) == 4
    assert not original['source_drift']
    joint.verify_review(PARENT / 'execution_review.json')
    assert joint.parent.digest(DIAGNOSTIC / 'analysis.json') == DIAGNOSTIC_SHA
    diagnosis = joint.load_json(DIAGNOSTIC / 'analysis.json')
    assert diagnosis['actual_LP_calls'] == 8 and diagnosis['successful_LP_calls'] == 2
    assert diagnosis['actual_physical_hours'] == diagnosis['validated_physical_hours'] == 720
    joint.verify_review(DIAGNOSTIC / 'execution_review.json')
    sources = dict(diagnosis['protocol']['source_sha256'])
    for name, expected in original['output_sha256'].items():
        path = PARENT / name
        assert joint.parent.digest(path) == expected, name
        sources[path.relative_to(ROOT).as_posix()] = expected
    for name in ('analysis.json', 'code_review.json', 'execution_review.json'):
        path = PARENT / name
        sources[path.relative_to(ROOT).as_posix()] = joint.parent.digest(path)
    for name, expected in diagnosis['output_sha256'].items():
        path = DIAGNOSTIC / name
        assert joint.parent.digest(path) == expected, name
        sources[path.relative_to(ROOT).as_posix()] = expected
    for name in ('analysis.json', 'code_review.json', 'execution_review.json'):
        path = DIAGNOSTIC / name
        sources[path.relative_to(ROOT).as_posix()] = joint.parent.digest(path)
    for name in ('src/v16_endpoint_bound_form.py', 'tests/test_v16_endpoint_bound_form.py'):
        sources[name] = joint.parent.digest(ROOT / name)
    protocol = {
        'status': 'PREPARED_NOT_RUN', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_class': 'FOUR_ORIGINAL_STATUS2_PROBLEMS_TWO_METHODS_EQUIVALENT_ENDPOINT_BOUND_FORM',
        'parent_analysis_sha256': PARENT_SHA, 'case_grid': GRID, 'paths': 8,
        'preserved_first_diagnostic_analysis_sha256': DIAGNOSTIC_SHA,
        'expected_new_LP_calls': 8, 'expected_physical_hours_if_all_valid': 2880,
        'actual_scientific_LP_calls_at_freeze': 0,
        'matrix_contract': 'Preserve the exact original NPZ bytes. The only reformulation removes the final seven unit endpoint equalities and fixes those seven variables to the identical targets using bounds. Assert targets are exactly within the original bounds, without tolerance or clipping. All other CSR/RHS/bounds/objective/constants remain unchanged; save the actual reformulated problem separately. The physical feasible set is identical.',
        'methods': METHODS, 'options': {'presolve': False},
        'ordering': 'Four original status-2 cases, each dual simplex then interior point, all once even on any failure. Both methods are reported; the first is the predeclared primary diagnostic, the second a separate cross-check, never a fallback.',
        'interpretation': 'Retain all sixteen original attempts and all eight presolve-disabled attempts, including every status 2. For each successful bound-form solve, move its seven fixed-bound multiplier sums to the original seven equality multipliers and set those original bound multipliers to zero. Check both the actual reformulated KKT and the original saved-domain KKT, then unchanged physical replay, prefix, ENS/CO2/peak and full endpoints. This is an equivalent representation, not a physical assumption or allowance change.',
        'tolerances': original['protocol']['tolerances'],
        'source_sha256': sources, 'environment': original['protocol']['environment'],
        'official_method_documentation': [
            'https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.linprog-highs.html',
            'https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.linprog-highs-ds.html',
            'https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.linprog-highs-ipm.html'],
        'new_native_runs': 0,
        'commands': ['python -B -X utf8 -m src.v16_endpoint_bound_form --prepare',
                     'python -B -X utf8 -m src.v16_endpoint_bound_form --run']}
    OUT.mkdir(parents=True, exist_ok=True)
    joint.parent.event.write_json(OUT / 'protocol.json', protocol)
    pd.DataFrame(GRID).to_csv(OUT / 'case_grid.csv', index=False)
    print(json.dumps({'status': protocol['status'], 'paths': 8, 'scientific_LP_calls': 0}))


def run_case(row):
    target = OUT / row['case_id']; target.mkdir(exist_ok=False)
    record = {**row, 'status': 'RUNNING', 'complete': False, 'actual_LP_calls': 0,
              'successful_LP_calls': 0, 'actual_physical_hours': 0, 'validated_physical_hours': 0}
    calls, logs = [], []; started = time.perf_counter()
    try:
        original = PARENT / row['original_case_id'] / 'submitted_problem.npz'
        copied = target / 'submitted_problem.npz'; shutil.copyfile(original, copied)
        assert joint.parent.digest(copied) == joint.parent.digest(original)
        problem = load_problem(copied)
        record['actual_LP_calls'] = 1
        solution, domain, raw, reduced = solve_saved_problem(problem, row['method'])
        A, b, E, d, bounds = domain
        actual_A, actual_b, actual_E, actual_d, actual_bounds = reduced
        np.savez_compressed(target / 'bound_form_problem.npz',
            **{name+'_'+field: value for name,matrix in (('A_ub',actual_A),('A_eq',actual_E))
               for field,value in (('data',matrix.data),('indices',matrix.indices),('indptr',matrix.indptr),('shape',np.array(matrix.shape)))},
            b_ub=actual_b,b_eq=actual_d,bounds=np.array(joint.parent.finite_bounds(actual_bounds)),
            objective=problem['objective'],resource_constant=problem['resource_constant'])
        log = {'method': row['method'], 'options': {'presolve': False},
               'submitted_problem_sha256': joint.parent.digest(copied),
               'A_ub_sha256': joint.parent.terminal.matrix_hash(A), 'A_eq_sha256': joint.parent.terminal.matrix_hash(E),
               'b_ub_sha256': joint.parent.terminal.array_hash(b), 'b_eq_sha256': joint.parent.terminal.array_hash(d),
               'bounds_sha256': joint.parent.terminal.bounds_hash(bounds),
               'objective_sha256': joint.parent.terminal.array_hash(problem['objective']),
               'crossover_iterations': int(solution.crossover_nit),
               'actual_bound_form_problem_sha256': joint.parent.digest(target/'bound_form_problem.npz'),
               'actual_bound_form_certificate': joint.parent.solution_certificate(raw,problem['objective'],*reduced)}
        log.update(joint.parent.solution_certificate(solution, problem['objective'], A, b, E, d, bounds))
        logs.append(log); record['successful_LP_calls'] = int(solution.success)
        if not solution.success:
            record.update(status='INFEASIBLE' if solution.status == 2 else 'SOLVER_FAILED',
                          solver_status=int(solution.status), solver_message=str(solution.message))
            return record
        np.savez_compressed(target / 'bound_form_solution.npz', x=raw.x,
            equality_marginals=raw.eqlin.marginals, inequality_marginals=raw.ineqlin.marginals,
            lower_marginals=raw.lower.marginals, upper_marginals=raw.upper.marginals)
        actual_certificate = log['actual_bound_form_certificate']
        assert max(actual_certificate['primal_residual'].values()) <= joint.parent.FEAS_TOL
        assert max(actual_certificate['dual_stationarity_residual'], actual_certificate['dual_sign_residual']) <= joint.parent.FEAS_TOL
        assert abs(actual_certificate['primal_minus_dual']) <= joint.parent.COST_TOL
        np.savez_compressed(target / 'solution.npz', x=solution.x,
            equality_marginals=solution.eqlin.marginals, inequality_marginals=solution.ineqlin.marginals,
            lower_marginals=solution.lower.marginals, upper_marginals=solution.upper.marginals)
        assert max(log['primal_residual'].values()) <= joint.parent.FEAS_TOL, log
        assert max(log['dual_stationarity_residual'], log['dual_sign_residual']) <= joint.parent.FEAS_TOL, log
        assert abs(log['primal_minus_dual']) <= joint.parent.COST_TOL, log
        case, rolling, _, inherited = joint.load_domain(row)
        result = joint.replay(case, rolling, solution.x, calls)
        table = result['timeseries']; record['actual_physical_hours'] = len(table)
        table.to_csv(target / 'actions_states.csv', index=False, float_format='%.17g')
        residuals, independent = joint.parent.event.independent_audit(case, table)
        residuals.to_csv(target / 'independent_physical_audit.csv', index=False, float_format='%.17g')
        end = joint.parent.terminal.terminal_audit(case, table, row['terminal_condition'] == 'full')
        prefix = joint.parent.compare_columns(table.iloc[:48], rolling.iloc[:48])
        actual = joint.parent.vector_from_table(table.iloc[48:])
        indices = problem['endpoint_indices']
        endpoint_error = float(np.abs(actual[indices] - problem['reference_x'][indices]).max())
        assert result['audit']['passed'] and independent['passed'] and end['passed'] and prefix['passed']
        assert max(endpoint_error, float(np.abs(actual-solution.x).max())) <= joint.parent.FEAS_TOL
        metrics = joint.parent.terminal.summarize(case, table, 'coordinated', row['terminal_condition'] == 'full')
        key = 'primary_h48_359_'; fee = key+joint.parent.terminal.PHYSICAL_COST
        cost = metrics[fee]
        reference_cost = float(problem['objective'] @ problem['reference_x'] + problem['resource_constant'])
        reference, _ = endpoint.reference_folder(row)
        old = joint.load_json(reference / 'summary.json')
        reference_table = pd.read_csv(reference / 'actions_states.csv', float_precision='round_trip')
        fields = [f+'_stock_mtce' for f in joint.parent.h.FUELS] + [s+'_soc_gwh' for s in joint.parent.h.STORES] + ['coal_gw','gas_gw','shift_debt_gwh']
        differences = {name: float(table[name].iloc[-1]-reference_table[name].iloc[-1]) for name in fields}
        assert max(abs(v) for v in differences.values()) <= joint.parent.FEAS_TOL
        assert abs(cost-float(problem['objective']@solution.x+problem['resource_constant'])) <= joint.parent.COST_TOL
        assert abs(reference_cost-old[fee]) <= joint.parent.COST_TOL and cost >= reference_cost-joint.parent.COST_TOL
        assert abs(metrics[key+'total_co2_mt']-old[key+'total_co2_mt']) <= joint.parent.FEAS_TOL
        assert metrics[key+'peak_unserved_gw'] <= float(rolling.iloc[48:].unserved_gw.max())+joint.parent.FEAS_TOL
        assert metrics[key+'unserved_gwh'] <= inherited['ens_primal_gwh']+joint.parent.ENS_CAP_SLACK+joint.parent.FEAS_TOL
        metrics.update(forecast_hours=312, information_contract='Exact saved endpoint-matched matrix; fixed 48h prefix; true 312h',
                       evidence_class='EQUIVALENT_ENDPOINT_FIXED_BOUND_FORM_ORIGINAL_KKT_VALIDATED')
        joint.parent.event.write_json(target / 'summary.json', metrics)
        pd.DataFrame(joint.parent.recovery_report(table,joint.parent.design.case_configuration(row))).to_csv(target / 'recovery.csv',index=False)
        record.update(status='COMPLETED_AND_VALIDATED',complete=True,validated_physical_hours=len(table),metrics=metrics,
            original_physical_audit=result['audit'],independent_physical_audit=independent,terminal_soc_audit=end,
            fixed_prefix_comparison=prefix,endpoint_error=endpoint_error,final_state_differences=differences,
            resource_cost_minus_reference_million_cny=cost-reference_cost,
            total_co2_minus_reference_mt=metrics[key+'total_co2_mt']-old[key+'total_co2_mt'])
    except Exception as failure:
        record.update(status='FAILED',error=str(failure),error_type=type(failure).__name__,traceback=traceback.format_exc())
    finally:
        record.update(wall_seconds=time.perf_counter()-started,attempted_physical_hours=len(calls))
        for name,values in (('solver_logs.jsonl',logs),('actual_pre_states_actions.jsonl',calls)):
            with (target/name).open('w',encoding='utf-8') as stream:
                for item in values:stream.write(json.dumps(joint.parent.event.json_value(item),ensure_ascii=False,allow_nan=False)+'\n')
        joint.parent.event.write_json(target/'result.json',record)
    return record


def run():
    assert not (OUT/'analysis.json').exists(), 'Preserve the original diagnostic identity'
    protocol=joint.load_json(OUT/'protocol.json');assert protocol['status']=='PREPARED_NOT_RUN'
    joint.verify_review(OUT/'code_review.json')
    for name,expected in protocol['source_sha256'].items():assert joint.parent.digest(ROOT/name)==expected,name
    results=[]
    for row in GRID:
        result=run_case(row);results.append(result);joint.parent.event.write_json(OUT/'progress.json',{'cases':results})
        print(json.dumps({key:result[key] for key in ('case_id','status','actual_LP_calls','actual_physical_hours')}),flush=True)
    drift=[name for name,expected in protocol['source_sha256'].items() if joint.parent.digest(ROOT/name)!=expected]
    pd.DataFrame([{key:r[key] for key in ('original_case_id','method','case_id','status','complete')} |
        {key:r.get(key) for key in ('resource_cost_minus_reference_million_cny','total_co2_minus_reference_mt','endpoint_error')}
        for r in results]).to_csv(OUT/'all_eight_diagnostic_results.csv',index=False,float_format='%.17g')
    audit={'status':'COMPLETED_ALL_EIGHT_EQUIVALENT_BOUND_FORM_DIAGNOSTICS' if all(r['complete'] for r in results) and not drift else 'FAILED_OR_INCOMPLETE_FIXED_DIAGNOSTIC_SET',
        'protocol':protocol,'cases':results,'source_drift':drift,
        **{key:sum(r[key] for r in results) for key in ('actual_LP_calls','successful_LP_calls','actual_physical_hours','validated_physical_hours')},
        'output_sha256':{p.relative_to(OUT).as_posix():joint.parent.digest(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('analysis.json','progress.json','code_review.json')}}
    joint.parent.event.write_json(OUT/'analysis.json',audit)
    print(json.dumps({key:audit[key] for key in ('status','actual_LP_calls','successful_LP_calls','actual_physical_hours','validated_physical_hours')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    elif args.run:run()
    else:parser.print_help()
