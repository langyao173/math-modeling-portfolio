"""Common final storage reserve challenge of four frozen dispatch controllers.

Only the two SOC bounds at the true global endpoint change. The original LP,
local storage rule, real plant and all other coefficients remain frozen.
AI-assisted: OpenAI Codex / GPT-6, 2026-10-19.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd
import scipy

from src import v3_hourly as h
from src import v13_event_information as event
from src import v13_storage_baseline as storage
from src import v13_event_design as design
from src.v13_storage_duration import reference_field_comparison
from src.v10_prepare_inputs import make_case
from src.v10_validate_dispatch import _prefix_audit
from src.v11_recovery_sensitivity import digest, recovery_windows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/v14/terminal_reserve'
STORAGE_OUT = ROOT/'outputs/v13/storage_baseline'
STORAGE_ANALYSIS_SHA256 = '226681246a962cfed47914ee01cae30d2cb8f8c72c544790be6b3b2939e02750'
ARMS = tuple(row for row in storage.GRID if row['case_id'] != 'decentralized')
CONDITIONS = ('unconstrained','full_soc')
GRID = tuple({'case_id':condition+'_'+arm['case_id'], 'arm':arm['case_id'],
              'terminal_condition':condition,'strategy':arm['strategy'],
              'storage_power_multiplier':arm['storage_power_multiplier'],
              'local_rule':arm['local_rule'],'anchor':condition=='unconstrained'}
             for condition in CONDITIONS for arm in ARMS)
WITHIN_CONDITION_PAIRS = (
    {'left_arm':'reachable_local','right_arm':'base','comparison_type':'POLICY_AT_FIXED_POWER'},
    {'left_arm':'reachable_local_power','right_arm':'power','comparison_type':'POLICY_AT_FIXED_POWER'},
    {'left_arm':'base','right_arm':'power','comparison_type':'POWER_INTERVENTION_ENERGY_FIXED'},
    {'left_arm':'reachable_local','right_arm':'reachable_local_power','comparison_type':'POWER_INTERVENTION_ENERGY_FIXED'},
)
PHYSICAL_COST = 'physical_cost_excluding_shortage_carbon_including_holding_million_cny'
WINDOWS = (('whole_h0_359',0,360),('warmup_h0_47',0,48),('primary_h48_359',48,360))
COMPARE_METRICS = design.COMPARE_METRICS + tuple(
    label+'_'+metric for label,_,_ in WINDOWS for metric in (PHYSICAL_COST,'total_co2_mt')) + (
    'coal_terminal_thermal_gw','gas_terminal_thermal_gw')


def array_hash(values):
    values = np.asarray(values)
    receipt = hashlib.sha256()
    receipt.update(values.dtype.str.encode('ascii'))
    receipt.update(str(values.shape).encode('ascii'))
    receipt.update(values.tobytes(order='C'))
    return receipt.hexdigest()


def matrix_hash(matrix):
    assert matrix.format == 'csr'
    return {'shape':list(matrix.shape),'data':array_hash(matrix.data),
            'indices':array_hash(matrix.indices),'indptr':array_hash(matrix.indptr)}


def bounds_hash(bounds):
    return hashlib.sha256(json.dumps(event.json_value(bounds),separators=(',',':'),allow_nan=False).encode('ascii')).hexdigest()


def simulate_terminal(case, strategy, local_rule, require_full, calls=None, solver_logs=None):
    """Adapt only final SOC bounds at the original serial linprog boundary."""
    if calls is None:
        calls = []
    if solver_logs is None:
        solver_logs = []
    original_plan, original_solver = h._plan, h.linprog
    context = {}

    def plan(actual,start,state,actual_strategy):
        assert actual is case and actual_strategy == strategy
        assert not context, 'Serial planner context is already active'
        context.update(hour=start,nstep=min(int(case['config']['forecast_hours']),case['hours']-start))
        try:
            return original_plan(actual,start,state,actual_strategy)
        finally:
            context.clear()

    def solve(c, *, A_ub, b_ub, A_eq, b_eq, bounds, method):
        start, nstep = context['hour'],context['nstep']
        assert len(c) == len(bounds) == 24*nstep, 'Frozen original LP variable layout changed'
        stop = start+nstep
        # Original layout: gen(7n), ch(2n), dis(2n), soc(2n), then fuel/debt/service.
        soc_offset = nstep*(len(h.GENERATORS)+2*len(h.STORES))
        final_soc_indices = [soc_offset+(nstep-1)*len(h.STORES)+j for j in range(len(h.STORES))]
        modified = bounds
        changes = []
        if require_full and stop == case['hours']:
            modified = list(bounds)
            for store,index in zip(h.STORES,final_soc_indices):
                energy = case['storage'][store]['energy_gwh']
                assert bounds[index] == (0,energy), 'Original SOC bound changed'
                modified[index] = (energy,energy)
                changes.append({'store':store,'index':index,'old':bounds[index],'new':modified[index]})
        changed_indices = [j for j,(old,new) in enumerate(zip(bounds,modified)) if old != new]
        expected_changes = final_soc_indices if require_full and stop == case['hours'] else []
        assert changed_indices == expected_changes
        coefficients = {'objective':array_hash(c),'A_ub':matrix_hash(A_ub),'b_ub':array_hash(b_ub),
                        'A_eq':matrix_hash(A_eq),'b_eq':array_hash(b_eq)}
        log = {'hour':start,'forecast_stop_hour_exclusive':stop,'global_endpoint_hour_exclusive':case['hours'],
               'nstep':nstep,'terminal_soc_enforced':bool(require_full and stop==case['hours']),
               'final_soc_variable_indices':final_soc_indices,'changed_bounds':changes,
               'objective_and_constraints_sha256':coefficients,
               'original_bounds_sha256':bounds_hash(bounds),'submitted_bounds_sha256':bounds_hash(modified),
               'unchanged_other_bounds_sha256':bounds_hash([bound for j,bound in enumerate(bounds) if j not in changed_indices]),
               'method':method,'variables':len(c)}
        assert log['unchanged_other_bounds_sha256'] == bounds_hash([bound for j,bound in enumerate(modified) if j not in changed_indices])
        solver_logs.append(log)
        try:
            solution = original_solver(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=modified,method=method)
            log.update(status=int(solution.status),success=bool(solution.success),message=str(solution.message),
                       iterations=int(solution.nit),objective_value=None if solution.fun is None else float(solution.fun))
            return solution
        except Exception as failure:
            log.update(error_type=type(failure).__name__,error=str(failure))
            raise

    h._plan, h.linprog = plan,solve
    try:
        return storage.simulate_storage(case,strategy,local_rule,calls)
    finally:
        h._plan, h.linprog = original_plan,original_solver


def window_metrics(case, table):
    """Allocate actual consumption, SOC draw and common holding by absolute hour."""
    fuel = sum((table[f+'_power_burn_mtce']+table[f+'_direct_served_mtce'])*case['config']['fuel_value_million_cny_mtce'][f] for f in h.FUELS)
    nonfossil = sum(table[g+'_gw']*case['config']['generation_cost_cny_kwh'][g] for g in h.GENERATORS if g not in ('coal','gas'))
    cycling = sum((table[s+'_charge_gw']+table[s+'_discharge_gw'])*case['config']['storage_cycle_cost_cny_kwh'] for s in h.STORES)
    shift = (table.shift_out_gw+table.shift_in_gw)*case['config']['shift_cost_cny_kwh']
    soc_draw = sum((table[s+'_discharge_gw']/case['storage'][s]['eta_discharge']-
                    table[s+'_charge_gw']*case['storage'][s]['eta_charge'])*case['config']['stored_electricity_value_million_cny_gwh'] for s in h.STORES)
    holding = sum(case['initial_fuel_stocks_mtce'][j]*case['config']['fuel_value_million_cny_mtce'][f] for j,f in enumerate(h.FUELS))*case['config']['stock_holding_rate_per_year']/8760
    physical = fuel+nonfossil+cycling+shift+soc_draw+holding
    co2 = sum((table[f+'_power_burn_mtce']+table[f+'_direct_served_mtce'])*h.FUEL_EMISSION_MT_PER_MTCE[f] for f in h.FUELS)
    result = {}
    for label,start,stop in WINDOWS:
        mask = (table.hour>=start)&(table.hour<stop)
        result[label+'_'+PHYSICAL_COST] = float(physical[mask].sum())
        result[label+'_total_co2_mt'] = float(co2[mask].sum())
    return result


def summarize(case, table, strategy, required=False):
    summary = design.summarize(case,table,strategy)
    summary.update(window_metrics(case,table))
    summary['terminal_soc_condition'] = 'full_soc' if required else 'unconstrained'
    if required:
        summary['boundary'] = case['boundary'].replace(
            'storage terminal state is reported, not forced cyclic.',
            'battery and pumped SOC at the actual global endpoint must equal full energy capacity; fuel terminal inventory remains free and valued at the original prices.')
    assert abs(summary['whole_h0_359_'+PHYSICAL_COST]-summary[PHYSICAL_COST]) <= event.TOLERANCE
    assert abs(summary['whole_h0_359_total_co2_mt']-summary['total_co2_mt']) <= event.TOLERANCE
    for fuel in ('coal','gas'):
        summary[fuel+'_initial_thermal_gw'] = case['initial_thermal_gw'][fuel]
        summary[fuel+'_terminal_thermal_gw'] = float(table[fuel+'_gw'].iloc[-1])
    return summary


def terminal_audit(case, table, required):
    residuals = {store:float(table[store+'_soc_gwh'].iloc[-1]-case['storage'][store]['energy_gwh']) for store in h.STORES}
    complete = len(table)==case['hours'] and int(table.hour.iloc[-1])==case['hours']-1
    return {'passed':complete and (not required or max(abs(v) for v in residuals.values())<=event.TOLERANCE),
            'required':required,'global_terminal_hour':case['hours']-1,
            'target_soc_gwh':{store:case['storage'][store]['energy_gwh'] for store in h.STORES},
            'actual_soc_gwh':{store:float(table[store+'_soc_gwh'].iloc[-1]) for store in h.STORES},
            'actual_minus_full_soc_gwh':residuals,'tolerance':event.TOLERANCE}


def input_preflight():
    checks,sources = storage.input_preflight()
    receipt_path = STORAGE_OUT/'analysis.json'
    assert digest(receipt_path)==STORAGE_ANALYSIS_SHA256
    receipt = json.loads(receipt_path.read_text('utf-8'))
    assert receipt['status']=='COMPLETED_FIVE_VALIDATED_360H_STORAGE_BASELINE_PATHS'
    assert receipt['actual_successful_lp_hours']==1800 and not receipt['source_drift']
    for relative,expected in receipt['protocol']['source_sha256'].items():
        assert digest(ROOT/relative)==expected, f'Frozen storage source drift: {relative}'
    paths = [Path(__file__),ROOT/'tests/test_v14_terminal_reserve.py',ROOT/'src/v13_event_design.py',
             ROOT/'src/v13_storage_duration.py',receipt_path,STORAGE_OUT/'protocol.json']
    inputs = {}
    for arm in ARMS:
        case = make_case(storage.case_configuration(arm))
        assert case['hours']==360 and case['config']['forecast_hours']==24
        assert case['v10_config']['event_starts']==[48,192]
        assert case['v10_config']['event_duration_hours']==48 and case['v10_config']['shock_severity']==1.
        assert case['v10_config']['supply_policy']=='inherited'
        inputs[arm['case_id']] = design.initial_state_and_controls(case)
        for name in ('actions_states.csv','result.json','summary.json','recovery.csv'):
            path = STORAGE_OUT/arm['case_id']/name
            assert digest(path)==receipt['output_sha256'][path.relative_to(STORAGE_OUT).as_posix()]
            paths.append(path)
    assert {item['storage']['battery']['energy_gwh'] for item in inputs.values()}=={245.9468560688265}
    assert {item['storage']['pumped']['energy_gwh'] for item in inputs.values()}=={574.6914469722752}
    for path in paths:
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    return checks,inputs,sources


def prepare():
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Frozen protocol exists; do not overwrite it')
    checks,inputs,sources = input_preflight()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol = {
        'status':'PREPARED_NOT_RUN','frozen_utc':datetime.now(timezone.utc).isoformat(),
        'analysis_class':'POST_HOC_ENGINEERING_COMMON_FINAL_STORAGE_RESERVE_CHALLENGE',
        'grid':GRID,'case_count':8,'hours_per_case':360,'expected_total_lp_hours':2880,
        'executed_scientific_lp_hours_at_freeze':0,'verification_lp_calls_per_full_test_suite':3,
        'configurations':{row['case_id']:storage.case_configuration(row) for row in GRID},
        'initial_state_and_controls':inputs,
        'terminal_change':'In four new cases only, require battery and pumped SOC at end of actual hour359 to equal each original energy capacity. The requirement enters a rolling LP only when its absolute forecast window includes hour359. Both powers retain identical E and initial SOC. Fuel terminal states remain free and are valued by the original common engineering prices.',
        'solver_boundary':'Keep original h._plan and h.simulate. A serial, finally-restored h._plan context provides actual absolute start/nstep to a short h.linprog adapter. Original variable blocks put final SOC at n*(7+2+2)+2*(n-1)+j; assert total24n, original bounds(0,E), and change only these two to(E,E) at start+n==360. Other objective/matrices/RHS are passed as original objects; retain byte hashes, all other bounds hash, two changes and raw solver status. No copied solver, new objective, softened endpoint or local action override.',
        'unchanged':'Original knee2060 snapshot; 360h; phase0; event starts48/192, duration48, severity1; inherited fuel; common24h perfect information; first action and real feedback; P1/P1.5 energy fixed; initial fuel/SOC/thermal/debt; original objectives/ramp/daily deadlines/finalzero debt/physical plant; four original coordinated or reachable-all-source local controllers. Annual controls/capital remain frozen.',
        'execution_order':'Four unconstrained full-path anchors first, then four common100%SOC cases in the same arm order.',
        'anchor_release_gate':'Every original action/state column, original summary field and recovery field must match its frozen V13 storage_baseline reference within1e-6; both physical audits pass. If any of the four anchors fails, retain all four actual outcomes and four explicit NOT_RUN new cases.',
        'failure_policy':'After anchor release attempt every new case even if a fixed local plan conflicts with the endpoint or its LP is infeasible. Preserve original failed hour/state/input/solver status and successful physical prefix. Secondary prefix evidence failures are separate and cannot erase the original exception or cancel later cases. Never relax a bound, erase debt, retune local control or substitute a coordinated action.',
        'comparison_release':'No numerical comparison before all eight cases are attempted and source hashes remain unchanged. Four within-arm terminal-condition contrasts and four same-condition contrasts per condition are predeclared. Every failed/incomplete pair has explicit statuses and null numerical differences; no full-domain benefit/range on failed cases.',
        'within_condition_pairs':WITHIN_CONDITION_PAIRS,
        'across_condition_pairs':'Each same frozen arm: full_soc minus unconstrained; comparison type TERMINAL_SOC_REQUIREMENT_INTERVENTION.',
        'comparison_metrics':COMPARE_METRICS,
        'primary_window':'absolute48..359; ENS, original-demand fraction, peak, actual physical resource/holding cost and CO2',
        'secondary_windows':'whole0..359 and warmup0..47; same costs/CO2; complete fuel/SOC/thermal/debt, shortage-charge overlap, original recovery/rebreach',
        'window_cost_formula':'Actual direct fuel delivery plus generator fuel burn at frozen fuel unit values, nonfossil generation cost, storage cycling, shifting, net SOC draw at0.32millionCNY/GWh and prorated original initial-fuel holding. Exclude shortage penalties and carbon charges; full sum must reproduce original resource_accounted-shortage-carbon+holding within1e-6. Fuel and SOC inventories retain the original price convention; do not add capital or equal-fuel terminal constraints.',
        'recovery_rule':{'stock_ratio':.95,'service_limit':.01,'window_hours':24,'event_boundaries':[192,360],'normalization':'each actual pre-event end state at start-1'},
        'physical_gate':'Original h.audit plus reused independent per-hour conservation and physical bounds, and actual true-end100%SOC residual within1e-6 when required.',
        'evidence_storage':'Full actual action/state CSV, full exogenous CSV, initial controls and hashes. Compact actual pre-state/action logs preserve whole-input hash and absolute forecast indices verified against all original forecast slices. LP logs retain objective/CSR matrix/RHS/bounds hashes, changed indices/old/new and status; future local plans replay from frozen code and actual states.',
        'interpretation':'New post-hoc engineered endpoint challenge of these four fixed controllers; not an observed fleet estimate, independent event sample, annual capital re-optimization or algorithm-independent optimality result.',
        'input_checks':checks,'source_sha256':sources,
        'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'solver':'scipy.optimize.linprog(method=highs)'},
        'commands':{'test':'python -m unittest discover -s tests -p test_v14_terminal_reserve.py -v','prepare':'python -m src.v14_terminal_reserve --prepare','run_after_independent_review':'python -m src.v14_terminal_reserve --run'},
        'working_directory':str(ROOT),'new_native_runs':0,'paid_services_used':False,
    }
    event.write_json(OUT/'protocol.json',protocol)
    pd.DataFrame(GRID).to_csv(OUT/'case_grid.csv',index=False)
    print(json.dumps({'status':protocol['status'],'cases':8,'scientific_lp_hours':0}))
    return protocol


def anchor_comparison(row,table,summary,recovery):
    folder = STORAGE_OUT/row['arm']
    old = pd.read_csv(folder/'actions_states.csv',float_precision='round_trip')
    assert list(old.columns)==list(table.columns) and len(old)==len(table)==360
    differences = {key:float(np.abs(table[key].to_numpy()-old[key].to_numpy()).max()) for key in old.columns}
    old_summary = json.loads((folder/'summary.json').read_text('utf-8'))
    receipt = json.loads((STORAGE_OUT/'analysis.json').read_text('utf-8'))
    old_case = next(item for item in receipt['cases'] if item['case_id']==row['arm'])
    summary_check = reference_field_comparison(summary,old_summary)
    recovery_check = reference_field_comparison(recovery,old_case['recovery'])
    return {'passed':all(np.isfinite(v) and v<=event.TOLERANCE for v in differences.values()) and summary_check['passed'] and recovery_check['passed'],
            'reference_analysis_sha256':STORAGE_ANALYSIS_SHA256,'all_column_max_absolute_differences':differences,
            'all_original_summary_fields':summary_check,'all_original_recovery_fields':recovery_check,
            'reference_backend':'Frozen V13 storage_baseline SciPy, not a new native run.'}


def run_case(row,config,target):
    target.mkdir(parents=True,exist_ok=False)
    case = make_case(config)
    required = row['terminal_condition']=='full_soc'
    event.write_json(target/'parameters_initial_state.json',{'configuration':config,'terminal_condition':row['terminal_condition'],
        'local_rule':row['local_rule'],'annual_capital_reestimated':False,'initial_state_and_controls':design.initial_state_and_controls(case)})
    input_path = target/'exogenous_inputs.csv'
    design.exogenous_table(case).to_csv(input_path,index=False,float_format='%.17g')
    input_hash = digest(input_path)
    inputs = design.forecast_arrays(pd.read_csv(input_path,float_precision='round_trip'))
    calls,solver_logs = [],[]
    record = dict(row,status='RUNNING',complete=False,exogenous_sha256=input_hash)
    started = time.perf_counter()
    try:
        result = simulate_terminal(case,config['strategy'],row['local_rule'],required,calls,solver_logs)
        result['timeseries'].to_csv(target/'actions_states.csv',index=False,float_format='%.17g')
        table = pd.read_csv(target/'actions_states.csv',float_precision='round_trip')
        original = h.audit(case,table)
        residual,independent = event.independent_audit(case,table)
        residual.to_csv(target/'independent_physical_audit.csv',index=False,float_format='%.17g')
        terminal = terminal_audit(case,table,required)
        record.update(executed_hours=len(table),original_physical_audit=original,independent_physical_audit=independent,terminal_soc_audit=terminal)
        assert original['passed'] and independent['passed'] and terminal['passed'], 'Physical or actual terminal audit failed'
        assert len(calls)==len(solver_logs)==360, 'Planner/solver evidence length mismatch'
        assert sum(log['terminal_soc_enforced'] for log in solver_logs)==(24 if required else 0)
        summary = summarize(case,table,config['strategy'],required)
        recovery = recovery_windows(table,design.recovery_configuration(config))
        event.write_json(target/'summary.json',summary)
        pd.DataFrame(recovery).to_csv(target/'recovery.csv',index=False)
        charge = table.battery_charge_gw+table.pumped_charge_gw
        shortage = (table.unserved_gw>1e-5)|(table[[f+'_direct_unserved_mtce' for f in h.FUELS]].sum(axis=1)>1e-8)
        table.loc[(charge>1e-5)&shortage].to_csv(target/'shortage_while_charging.csv',index=False,float_format='%.17g')
        record.update(metrics=summary,recovery=recovery)
        if row['anchor']:
            comparison = anchor_comparison(row,table,summary,recovery)
            event.write_json(target/'anchor_comparison.json',comparison)
            record['anchor_comparison'] = comparison
            assert comparison['passed'], 'Original four-arm unconstrained anchor gate failed'
        record.update(status='COMPLETED_AND_VALIDATED',complete=True)
    except Exception as failure:
        planner_failure = bool(calls and 'error' in calls[-1])
        record.update(status='FAILED',error=str(failure),error_type=type(failure).__name__,traceback=traceback.format_exc(),
                      failure_hour=calls[-1]['hour'] if planner_failure else None,last_attempted_hour=calls[-1]['hour'] if calls else None,
                      failure_phase='PLANNER' if planner_failure else 'SIMULATION_OR_VALIDATION')
        if not (target/'actions_states.csv').exists():
            try:
                prefix = event.prefix_from_calls(case,calls)
                record['observed_prefix_hours'] = len(prefix)
                if len(prefix):
                    prefix.to_csv(target/'observed_prefix_actions_states.csv',index=False,float_format='%.17g')
                    residual,independent = event.independent_audit(case,prefix,complete=False)
                    residual.to_csv(target/'prefix_independent_physical_audit.csv',index=False,float_format='%.17g')
                    record['prefix_original_physical_audit'] = _prefix_audit(case,prefix)
                    record['prefix_independent_physical_audit'] = independent
            except Exception as evidence_failure:
                record['prefix_evidence_error'] = {'type':type(evidence_failure).__name__,'error':str(evidence_failure),'traceback':traceback.format_exc()}
    finally:
        record.update(successful_lp_hours=sum('action' in call for call in calls),attempted_lp_hours=len(calls),
                      actual_solver_calls=len(solver_logs),wall_seconds=time.perf_counter()-started)
        with (target/'solver_logs.jsonl').open('w',encoding='utf-8') as stream:
            for log in solver_logs:
                stream.write(json.dumps(event.json_value(log),ensure_ascii=False,allow_nan=False)+'\n')
        try:
            compact,checks = design.compact_planner_logs(calls,inputs,input_hash)
            with (target/'planner_actual_states_actions.jsonl').open('w',encoding='utf-8') as stream:
                for call in compact:
                    stream.write(json.dumps(event.json_value(call),ensure_ascii=False,allow_nan=False)+'\n')
            record['forecast_slice_verification'] = checks
        except Exception as evidence_failure:
            record.update(status='FAILED',complete=False,evidence_log_error={'type':type(evidence_failure).__name__,'error':str(evidence_failure),'traceback':traceback.format_exc()})
            with (target/'planner_actual_states_actions_unverified_forecast.jsonl').open('w',encoding='utf-8') as stream:
                for call in calls:
                    observed = {key:value for key,value in call.items() if key not in ('forecast_window','local_storage_forecast')}
                    observed['forecast_input'] = {'file':'exogenous_inputs.csv','sha256':input_hash,'start_hour':call['hour'],
                        'stop_hour_exclusive':call['forecast_window_stop'],'saved_slice_exact_match':False}
                    stream.write(json.dumps(event.json_value(observed),ensure_ascii=False,allow_nan=False)+'\n')
        event.write_json(target/'result.json',record)
    print(json.dumps({key:record.get(key) for key in ('case_id','status','executed_hours','successful_lp_hours','error')},ensure_ascii=False),flush=True)
    return record


def paired_comparisons(outcomes):
    by_key = {(row['terminal_condition'],row['arm']):row for row in outcomes}
    definitions = [{'comparison_id':'terminal_'+arm['case_id'],'comparison_type':'TERMINAL_SOC_REQUIREMENT_INTERVENTION',
                    'left':('full_soc',arm['case_id']),'right':('unconstrained',arm['case_id'])} for arm in ARMS]
    for condition in CONDITIONS:
        for pair in WITHIN_CONDITION_PAIRS:
            definitions.append({'comparison_id':condition+'_'+pair['left_arm']+'_minus_'+pair['right_arm'],
                'comparison_type':pair['comparison_type'],'left':(condition,pair['left_arm']),'right':(condition,pair['right_arm'])})
    rows = []
    for definition in definitions:
        left,right = by_key[definition['left']],by_key[definition['right']]
        valid = left['complete'] and right['complete']
        row = {key:value for key,value in definition.items() if key not in ('left','right')}
        row.update(left_case_id=left['case_id'],right_case_id=right['case_id'],
            pair_status='COMPLETE' if valid else 'FAILED_OR_INCOMPLETE_PAIR',left_status=left['status'],right_status=right['status'],
            left_executed_hours=left.get('executed_hours',left.get('observed_prefix_hours',0)),
            right_executed_hours=right.get('executed_hours',right.get('observed_prefix_hours',0)))
        if valid:
            assert left['exogenous_sha256']==right['exogenous_sha256'], 'Paired external inputs differ'
        for metric in COMPARE_METRICS:
            row['left_minus_right_'+metric] = left['metrics'][metric]-right['metrics'][metric] if valid else None
        row['left_recovery'] = left.get('recovery') if valid else None
        row['right_recovery'] = right.get('recovery') if valid else None
        rows.append(row)
    return rows


def run():
    protocol_path = OUT/'protocol.json'
    protocol = json.loads(protocol_path.read_text('utf-8'))
    assert protocol['status']=='PREPARED_NOT_RUN'
    assert not (OUT/'analysis.json').exists(), 'Do not overwrite an actual experiment'
    for relative,expected in protocol['source_sha256'].items():
        assert digest(ROOT/relative)==expected, f'Frozen source/input drift: {relative}'
    assert pd.read_csv(OUT/'case_grid.csv').to_dict('records')==list(GRID)
    outcomes = []
    started = time.perf_counter()
    for row in GRID:
        target = OUT/row['case_id']
        config = protocol['configurations'][row['case_id']]
        if not row['anchor'] and not all(item['complete'] for item in outcomes if item['anchor']):
            target.mkdir(exist_ok=False)
            event.write_json(target/'parameters.json',{'configuration':config,'terminal_condition':row['terminal_condition']})
            record = dict(row,status='NOT_RUN',reason='ORIGINAL_FOUR_ANCHOR_GATE_FAILED',complete=False,
                          executed_hours=0,successful_lp_hours=0,attempted_lp_hours=0,actual_solver_calls=0)
            event.write_json(target/'result.json',record)
        else:
            assert not target.exists(), 'Do not overwrite an executed case'
            record = run_case(row,config,target)
        outcomes.append(record)
    drift = {relative:digest(ROOT/relative) for relative,expected in protocol['source_sha256'].items() if digest(ROOT/relative)!=expected}
    anchor_passed = all(row['complete'] for row in outcomes if row['anchor'])
    all_attempted = len(outcomes)==8 and all(row['status']!='NOT_RUN' for row in outcomes)
    complete = all(row['complete'] for row in outcomes) and not drift
    comparisons = paired_comparisons(outcomes) if all_attempted and not drift else []
    if comparisons:
        event.write_json(OUT/'paired_comparisons.json',comparisons)
        flat = [{**{key:value for key,value in row.items() if key not in ('left_recovery','right_recovery')},
                 'left_recovery_json':json.dumps(event.json_value(row['left_recovery']),allow_nan=False),
                 'right_recovery_json':json.dumps(event.json_value(row['right_recovery']),allow_nan=False)} for row in comparisons]
        pd.DataFrame(flat).to_csv(OUT/'paired_comparisons.csv',index=False,float_format='%.17g')
    pd.DataFrame([{**{key:row[key] for key in ('case_id','arm','terminal_condition','status','complete','successful_lp_hours')},
                   'executed_hours':row.get('executed_hours',row.get('observed_prefix_hours',0)),**row.get('metrics',{})} for row in outcomes]).to_csv(OUT/'all_case_metrics.csv',index=False,float_format='%.17g')
    if complete:
        assert sum(row['successful_lp_hours'] for row in outcomes)==2880
    status = 'COMPLETED_EIGHT_VALIDATED_360H_TERMINAL_RESERVE_PATHS' if complete else 'FAILED_FIXED_TERMINAL_RESERVE_COHORT'
    if not anchor_passed:
        status = 'FAILED_ORIGINAL_FOUR_ANCHOR_GATE'
    report = {'status':status,'analysis_class':protocol.get('analysis_class'),'case_count':len(outcomes),
        'original_four_anchor_gate_passed':anchor_passed,'all_eight_cases_attempted':all_attempted,
        'not_run_cases':sum(row['status']=='NOT_RUN' for row in outcomes),'valid_cases':sum(row['complete'] for row in outcomes),
        'complete_pairs':sum(row['pair_status']=='COMPLETE' for row in comparisons),'pair_denominator':12,
        'global_numeric_domain_status':'AVAILABLE_ALL_EIGHT_VALID' if complete else 'NOT_ESTABLISHED_FOR_FULL_DOMAIN',
        'full_domain_ranges':{},'expected_total_lp_hours':2880,
        'actual_successful_lp_hours':sum(row['successful_lp_hours'] for row in outcomes),
        'actual_attempted_lp_hours':sum(row['attempted_lp_hours'] for row in outcomes),
        'actual_solver_calls':sum(row['actual_solver_calls'] for row in outcomes),
        'actual_complete_trajectory_hours':sum(row.get('executed_hours',0) for row in outcomes),
        'wall_seconds':time.perf_counter()-started,'source_drift':drift,'protocol_sha256':digest(protocol_path),
        'protocol':protocol,'cases':outcomes,'paired_comparisons':comparisons,'new_native_runs':0,'paid_services_used':False,
        'output_sha256':{path.relative_to(OUT).as_posix():digest(path) for path in OUT.rglob('*') if path.is_file() and path.name!='analysis.json'}}
    event.write_json(OUT/'analysis.json',report)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--run',action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
