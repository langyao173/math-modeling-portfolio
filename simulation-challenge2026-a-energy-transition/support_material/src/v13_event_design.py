"""Frozen phase/duration sensitivity of five existing V13 controllers.

This is a development experiment on designed events, not a probability sample.
Only event phase and duration change; all five controllers remain frozen.
AI-assisted: OpenAI Codex / GPT-6, 2026-10-19.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime,timezone
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
from src.v10_prepare_inputs import configuration,make_case
from src.v10_validate_dispatch import _prefix_audit
from src.v11_recovery_sensitivity import digest,recovery_windows

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/v13/event_design'
STORAGE_OUT=ROOT/'outputs/v13/storage_baseline'
PHASES=(0,3,6,9,12,15,18,21)
DURATIONS=(12,24,48,72)
CELLS=((0,48),)+tuple((phase,duration) for phase in PHASES for duration in DURATIONS if (phase,duration)!=(0,48))
GRID=tuple({'case_id':f'phase{phase:02d}_duration{duration:02d}_{arm["case_id"]}',
            'arm':arm['case_id'],'phase_hours':phase,'duration_hours':duration,
            'strategy':arm['strategy'],'storage_power_multiplier':arm['storage_power_multiplier'],
            'local_rule':arm['local_rule'],'anchor':(phase,duration)==(0,48)}
           for phase,duration in CELLS for arm in storage.GRID)
PAIRS=(
    {'left_arm':'decentralized','right_arm':'base','comparison_type':'POLICY_AT_FIXED_POWER'},
    {'left_arm':'reachable_local','right_arm':'base','comparison_type':'POLICY_AT_FIXED_POWER'},
    {'left_arm':'reachable_local_power','right_arm':'power','comparison_type':'POLICY_AT_FIXED_POWER'},
    {'left_arm':'base','right_arm':'power','comparison_type':'POWER_INTERVENTION_ENERGY_FIXED'},
    {'left_arm':'reachable_local','right_arm':'reachable_local_power','comparison_type':'POWER_INTERVENTION_ENERGY_FIXED'},
)
COMPARE_METRICS=(
    'primary_h48_359_unserved_gwh','primary_unserved_fraction_of_original_demand',
    'primary_h48_359_peak_unserved_gw','unserved_gwh','whole_unserved_fraction_of_original_demand',
    'warmup_h0_47_unserved_gwh','direct_fuel_unserved_mtce',
    'physical_cost_excluding_shortage_carbon_including_holding_million_cny','total_co2_mt',
    'coal_terminal_stock_mtce','oil_terminal_stock_mtce','gas_terminal_stock_mtce',
    'battery_terminal_soc_gwh','pumped_terminal_soc_gwh','terminal_shift_debt_gwh',
    'shift_out_gwh','shift_in_gwh','electricity_shortage_while_charging_hours',
    'charge_during_electricity_shortage_gwh','unserved_during_charging_gwh',
    'direct_fuel_shortage_while_charging_hours',
)


def case_configuration(row):
    """Pass actual phase/duration into V10 make_case; never reuse demo defaults."""
    return configuration(event_starts=[48+row['phase_hours'],192+row['phase_hours']],
                         event_duration_hours=row['duration_hours'],strategy=row['strategy'],
                         storage_power_multiplier=row['storage_power_multiplier'])


def recovery_configuration(config):
    return pd.Series({**config,'event1_start_hour':config['event_starts'][0],
                      'event2_start_hour':config['event_starts'][1]})


def exogenous_table(case):
    columns={'hour':np.arange(case['hours']),'demand_gw':case['demand_gw'],
             'stress_active':case['stress_mask'].astype(int)}
    columns.update({gen+'_available_gw':case['available_gw'][:,j] for j,gen in enumerate(h.GENERATORS)})
    for j,fuel in enumerate(h.FUELS):
        for label,key in (('domestic_arrival','fuel_domestic_arrivals_mtce'),
                          ('import_arrival','fuel_import_arrivals_mtce'),('emergency_arrival','emergency_arrivals_mtce'),
                          ('arrival','fuel_arrivals_mtce'),('direct_demand','direct_fuel_demand_mtce')):
            columns[fuel+'_'+label+'_mtce']=case[key][:,j]
    return pd.DataFrame(columns)


def forecast_arrays(frame):
    return {'available_gw':frame[[gen+'_available_gw' for gen in h.GENERATORS]].to_numpy(float),
            'demand_gw':frame.demand_gw.to_numpy(float),
            'direct_fuel_demand_mtce':frame[[fuel+'_direct_demand_mtce' for fuel in h.FUELS]].to_numpy(float),
            'fuel_arrivals_mtce':frame[[fuel+'_arrival_mtce' for fuel in h.FUELS]].to_numpy(float)}


def compact_planner_logs(calls,inputs,input_hash):
    """Check every perfect-information slice, retain actual states/actions once.

    Full future local plans are not saved. Their frozen implementation, actual
    pre-state, input file and absolute window uniquely specify their replay.
    """
    records=[]
    checks=0
    for call in calls:
        start,stop=call['hour'],call['forecast_window_stop']
        for key in event.FORECAST_KEYS:
            np.testing.assert_array_equal(call['forecast_window'][key],inputs[key][start:stop],
                                          err_msg=f'saved external input slice {key} h{start}')
            checks+=1
        record={key:deepcopy(value) for key,value in call.items()
                if key not in ('forecast_window','local_storage_forecast')}
        record['forecast_input']={'file':'exogenous_inputs.csv','sha256':input_hash,
                                  'start_hour':start,'stop_hour_exclusive':stop,
                                  'array_keys':list(event.FORECAST_KEYS),'saved_slice_exact_match':True}
        if 'local_storage_forecast' in call:
            local=call['local_storage_forecast']
            record['current_local_storage_forecast']={
                'reachable_thermal_gw':local['reachable_thermal_gw'][0].copy(),
                'net_before_storage_gw':float(local['net_before_storage_gw'][0]),
                'storage_first_actions':{store:{'charge_gw':float(local['local_storage_plan'][store][0][0]),
                                                'discharge_gw':float(local['local_storage_plan'][store][1][0])} for store in h.STORES}}
        records.append(record)
    return records,{'matching_forecast_windows':len(calls),'matching_forecast_key_windows':checks,
                    'future_local_plans_saved':False,
                    'future_local_plan_reconstruction':'Frozen v13_storage_baseline.all_source_local_storage + actual pre_state + saved exogenous inputs and absolute window.'}


def summarize(case,table,strategy):
    result=storage.summarize(case,table,strategy)
    primary_demand=float(case['demand_gw'][48:360].sum())
    whole_demand=float(case['demand_gw'].sum())
    result.update(primary_original_demand_gwh=primary_demand,
                  primary_unserved_fraction_of_original_demand=result['primary_h48_359_unserved_gwh']/primary_demand,
                  whole_original_demand_gwh=whole_demand,
                  whole_unserved_fraction_of_original_demand=result['unserved_gwh']/whole_demand)
    return result


def initial_state_and_controls(case):
    return {'fuel_mtce':case['initial_fuel_stocks_mtce'],'thermal_gw':case['initial_thermal_gw'],
            'debt_gwh':0.,'storage':case['storage'],'capacity_gw':case['capacity_gw'],
            'controller_config':case['config'],'fuel_efficiency':case['fuel_efficiency'],
            'previous_available_gw':{fuel:case['available_gw'][0,h.GENERATORS.index(fuel)] for fuel in ('coal','gas')}}


def anchor_comparison(arm,case,table,recovery):
    reference=pd.read_csv(STORAGE_OUT/f'{arm}/actions_states.csv',float_precision='round_trip')
    assert set(reference.columns)<=set(table.columns)
    differences={name:float(np.abs(table[name].to_numpy()-reference[name].to_numpy()).max()) for name in reference.columns}
    old,new=summarize(case,reference,case['v10_config']['strategy']),summarize(case,table,case['v10_config']['strategy'])
    metric_diffs={key:new[key]-old[key] for key in COMPARE_METRICS}
    original_recovery=recovery_windows(reference,recovery_configuration(case['v10_config']))
    result={'frozen_storage_path_max_absolute_column_differences':differences,
            'all_reference_columns_match':all(value<=event.TOLERANCE for value in differences.values()),
            'numeric_metric_differences':metric_diffs,
            'recovery_decisions_match':event._recovery_decisions(recovery)==event._recovery_decisions(original_recovery),
            'new_recovery':recovery,'frozen_reference_recovery':original_recovery,
            'reference_backend':'Frozen completed V13 storage-baseline SciPy path; no new native execution.'}
    result['passed']=(result['all_reference_columns_match'] and result['recovery_decisions_match']
                      and all(abs(value)<=event.TOLERANCE for value in metric_diffs.values()))
    return result


def input_preflight():
    checks,sources=storage.input_preflight()
    receipt=json.loads((STORAGE_OUT/'analysis.json').read_text('utf-8'))
    assert receipt['status']=='COMPLETED_FIVE_VALIDATED_360H_STORAGE_BASELINE_PATHS'
    assert receipt['actual_successful_lp_hours']==1800 and not receipt['source_drift']
    for relative,expected in receipt['protocol']['source_sha256'].items():
        assert digest(ROOT/relative)==expected,f'Completed storage-study source drift: {relative}'
    paths=[Path(__file__),ROOT/'tests/test_v13_event_design.py',STORAGE_OUT/'analysis.json',STORAGE_OUT/'protocol.json']
    for arm in storage.GRID:
        for filename in ('actions_states.csv','result.json','summary.json','recovery.csv'):
            path=STORAGE_OUT/arm['case_id']/filename
            assert digest(path)==receipt['output_sha256'][path.relative_to(STORAGE_OUT).as_posix()]
            paths.append(path)
    cells=[]
    reference=make_case(case_configuration(GRID[0]))
    for phase,duration in CELLS:
        row=next(r for r in GRID if r['phase_hours']==phase and r['duration_hours']==duration)
        config=case_configuration(row)
        case=make_case(config)
        assert int(case['stress_mask'].sum())==2*duration
        assert case['hours']==360 and case['config']['forecast_hours']==24
        np.testing.assert_array_equal(case['initial_fuel_stocks_mtce'],reference['initial_fuel_stocks_mtce'])
        assert case['initial_thermal_gw']==reference['initial_thermal_gw']
        start1,start2=config['event_starts']
        first_gap=start2-start1-duration
        final_gap=360-start2-duration
        assert first_gap>=24 and final_gap>=24
        cells.append({'phase_hours':phase,'duration_hours':duration,'event_starts':config['event_starts'],
                      'event1_post_shock_observation_hours':first_gap,'event2_post_shock_observation_hours':final_gap})
    for path in paths:
        sources[path.relative_to(ROOT).as_posix()]=digest(path)
    return checks,cells,sources


def prepare():
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Frozen protocol already exists; it will not be overwritten')
    checks,cells,sources=input_preflight()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={
        'status':'PREPARED_NOT_RUN','frozen_utc':datetime.now(timezone.utc).isoformat(),
        'analysis_class':'DEVELOPMENT_EVENT_DESIGN_SENSITIVITY_OF_FIVE_FROZEN_CONTROLLERS',
        'grid':GRID,'case_count':160,'design_cells':32,'arms_per_cell':5,'hours_per_case':360,
        'expected_total_lp_hours':57600,'executed_scientific_lp_hours_at_freeze':0,
        'verification_lp_calls_per_full_test_suite':2,'phases_hours':PHASES,'durations_hours':DURATIONS,
        'configurations':{row['case_id']:case_configuration(row) for row in GRID},
        'execution_order':'Five phase0/duration48 original anchors first, then remaining cells in phase/duration order and original five-arm order.',
        'anchor_release_gate':'All five complete original anchors must pass frozen storage-study full trajectory, metrics, recovery and both physical audits before releasing the remaining 155 paths. Anchor failure preserves five actual outcomes and 155 explicit NOT_RUN records.',
        'failure_policy':'After anchor release, attempt every predeclared case even if an individual LP or audit fails. Preserve original exception, failed state/packet, all successful actions and observed physical prefixes. Never tune or repair the controller to make a case pass.',
        'comparison_release':'No numerical comparison before all 160 cases have been attempted. At completion retain all 32 cells x five predeclared contrasts: three POLICY_AT_FIXED_POWER and two POWER_INTERVENTION_ENERGY_FIXED, with both statuses and executed hours. A failed/incomplete pair has null numerical differences. Report complete-case counts per arm /32 and complete pairs /32. Never delete failed cells or average only surviving cells.',
        'domain_claim_gate':'Full-domain min/max and benefit intervals require all 160 valid cases. Any failed case leaves full-domain numerical comparison unestablished; display the complete grid including failure marks and do not select a favorable subdomain.',
        'unchanged':'Snapshot plan_02349_2060; 360h; severity 1; inherited supply; common 24h perfect information; true state feedback; original objectives, physical constraints, initial fuel/SOC/thermal/debt, absolute daily debt deadlines, storage energy, and frozen five controllers. Only event_starts=[48+phase,192+phase] and event_duration_hours vary.',
        'primary_metrics':{'hours':'absolute 48..359 inclusive','metrics':['ENS_GWh','ENS_fraction_of_original_demand','peak_unserved_GW'],
                           'denominator':'Actual exogenous demand before flexible-load shifting, summed over the same absolute primary window. All five arms in a cell use the same demand.'},
        'secondary_metrics':['all360_and_fixed_warmup_h0_47_ENS','direct_fuel_shortage','physical_cost_excluding_shortage_carbon_including_holding','CO2','initial_and_terminal_fuel_SOC_debt','shift_out_in','recovery_and_rebreach','shortage_while_charging'],
        'physical_cost_formula':'resource_accounted - shortage_penalty - carbon_cost + stock_holding; million CNY engineering values.',
        'recovery_rule':{'stock_ratio':.95,'service_limit':.01,'window_hours':24,'normalization':'actual pre-event state at each actual event start-1','event1_observation_end':'actual second-event start','event2_observation_end':360},
        'cell_observation_windows':cells,
        'interpretation_limits':'Designed development cells are not observed event probabilities or independent random draws. Phase changes load/weather and absolute repayment-deadline alignment together. Duration changes shock dose and recovery observation length. Fixed h48..359 includes preparation hours before phase-shifted event onset. Same-power strategy comparisons are within a cell only; this cohort does not test shorter event-information horizons.',
        'pairs':PAIRS,'comparison_metrics':COMPARE_METRICS,
        'evidence_storage':'Save full actual action/state trace, full case exogenous CSV, initial state/controls, config and source hashes. Per-hour logs retain actual pre_state/action, absolute forecast indices and exogenous file hash; verify all four perfect forecast slices exactly against round-trip-read saved inputs. Store only current reachable thermal/net/local first actions; full future local plans are NOT saved and are replayable from the frozen rule, pre_state and saved inputs.',
        'physical_gate':'Original h.audit plus reused independent per-hour conservation and actual bounds at 1e-6; failed prefixes retain actual absolute debt deadlines.',
        'input_checks':checks,'source_sha256':sources,
        'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'solver':'scipy.optimize.linprog(method=highs)'},
        'commands':{'test':'python -m unittest discover -s tests -p test_v13_event_design.py -v','prepare':'python -m src.v13_event_design --prepare','run_after_independent_review':'python -m src.v13_event_design --run'},
        'working_directory':str(ROOT),'new_native_runs':0,'paid_services_used':False,
    }
    event.write_json(OUT/'protocol.json',protocol)
    pd.DataFrame(GRID).to_csv(OUT/'case_grid.csv',index=False)
    print(json.dumps({'status':protocol['status'],'cases':160,'scientific_lp_hours':0}))
    return protocol


def run_case(row,config,target):
    target.mkdir(parents=True,exist_ok=True)
    case=make_case(config)
    event.write_json(target/'parameters_initial_state.json',{'configuration':config,'local_rule':row['local_rule'],'initial_state_and_controls':initial_state_and_controls(case)})
    input_path=target/'exogenous_inputs.csv'
    exogenous_table(case).to_csv(input_path,index=False,float_format='%.17g')
    input_hash=digest(input_path)
    inputs=forecast_arrays(pd.read_csv(input_path,float_precision='round_trip'))
    calls=[]
    record=dict(row,status='RUNNING',complete=False,exogenous_sha256=input_hash)
    before=time.perf_counter()
    try:
        result=storage.simulate_storage(case,config['strategy'],row['local_rule'],calls)
        result['timeseries'].to_csv(target/'actions_states.csv',index=False,float_format='%.17g')
        table=pd.read_csv(target/'actions_states.csv',float_precision='round_trip')
        original_audit=h.audit(case,table)
        residual,independent=event.independent_audit(case,table)
        residual.to_csv(target/'independent_physical_audit.csv',index=False,float_format='%.17g')
        record.update(executed_hours=len(table),original_physical_audit=original_audit,independent_physical_audit=independent)
        assert original_audit['passed'] and independent['passed'],'Physical audit failed'
        summary=summarize(case,table,config['strategy'])
        recovery=recovery_windows(table,recovery_configuration(config))
        for result_row in recovery:
            result_row['post_shock_observation_hours']=result_row['observation_end_hour']-result_row['shock_end_hour']
        event.write_json(target/'summary.json',summary)
        pd.DataFrame(recovery).to_csv(target/'recovery.csv',index=False)
        charge=table.battery_charge_gw+table.pumped_charge_gw
        shortage=(table.unserved_gw>1e-5)|(table[[fuel+'_direct_unserved_mtce' for fuel in h.FUELS]].sum(axis=1)>1e-8)
        table.loc[(charge>1e-5)&shortage].to_csv(target/'shortage_while_charging.csv',index=False,float_format='%.17g')
        record.update(metrics=summary,recovery=recovery)
        if row['anchor']:
            comparison=anchor_comparison(row['arm'],case,table,recovery)
            event.write_json(target/'anchor_comparison.json',comparison)
            record['anchor_comparison']=comparison
            assert comparison['passed'],'Original five-arm anchor gate failed'
        record.update(status='COMPLETED_AND_VALIDATED',complete=True)
    except Exception as failure:
        planner_failure=bool(calls and 'error' in calls[-1])
        record.update(status='FAILED',error=str(failure),error_type=type(failure).__name__,traceback=traceback.format_exc(),
                      failure_hour=calls[-1]['hour'] if planner_failure else None,
                      last_attempted_hour=calls[-1]['hour'] if calls else None,
                      failure_phase='PLANNER' if planner_failure else 'SIMULATION_OR_VALIDATION')
        if not (target/'actions_states.csv').exists():
            prefix=event.prefix_from_calls(case,calls)
            record['observed_prefix_hours']=len(prefix)
            if len(prefix):
                prefix.to_csv(target/'observed_prefix_actions_states.csv',index=False,float_format='%.17g')
                try:
                    residual,independent=event.independent_audit(case,prefix,complete=False)
                    residual.to_csv(target/'prefix_independent_physical_audit.csv',index=False,float_format='%.17g')
                    record['prefix_original_physical_audit']=_prefix_audit(case,prefix)
                    record['prefix_independent_physical_audit']=independent
                except Exception as audit_failure:
                    record['prefix_audit_error']={'type':type(audit_failure).__name__,'error':str(audit_failure),'traceback':traceback.format_exc()}
    finally:
        record.update(successful_lp_hours=sum('action' in call for call in calls),attempted_lp_hours=len(calls),wall_seconds=time.perf_counter()-before)
        try:
            compact,checks=compact_planner_logs(calls,inputs,input_hash)
            with (target/'planner_actual_states_actions.jsonl').open('w',encoding='utf-8') as stream:
                for call in compact:
                    stream.write(json.dumps(event.json_value(call),ensure_ascii=False,allow_nan=False)+'\n')
            record['forecast_slice_verification']=checks
        except Exception as log_failure:
            record.update(status='FAILED',complete=False,
                          evidence_log_error={'type':type(log_failure).__name__,'error':str(log_failure),'traceback':traceback.format_exc()})
            # A rejected forecast audit cannot erase the observed state/action evidence.
            with (target/'planner_actual_states_actions_unverified_forecast.jsonl').open('w',encoding='utf-8') as stream:
                for call in calls:
                    observed={key:value for key,value in call.items() if key not in ('forecast_window','local_storage_forecast')}
                    observed['forecast_input']={'file':'exogenous_inputs.csv','sha256':input_hash,
                                                'start_hour':call['hour'],'stop_hour_exclusive':call['forecast_window_stop'],
                                                'saved_slice_exact_match':False}
                    stream.write(json.dumps(event.json_value(observed),ensure_ascii=False,allow_nan=False)+'\n')
        event.write_json(target/'result.json',record)
    print(json.dumps({key:record.get(key) for key in ('case_id','status','executed_hours','successful_lp_hours','error')},ensure_ascii=False),flush=True)
    return record


def cell_comparisons(outcomes):
    by_key={(row['phase_hours'],row['duration_hours'],row['arm']):row for row in outcomes}
    rows=[]
    for phase in PHASES:
        for duration in DURATIONS:
            for pair in PAIRS:
                left_arm,right_arm=pair['left_arm'],pair['right_arm']
                left,right=by_key[(phase,duration,left_arm)],by_key[(phase,duration,right_arm)]
                valid=left['complete'] and right['complete']
                record={'phase_hours':phase,'duration_hours':duration,**pair,
                        'pair_status':'COMPLETE' if valid else 'FAILED_OR_INCOMPLETE_PAIR',
                        'left_status':left['status'],'right_status':right['status'],
                        'left_executed_hours':left.get('executed_hours',left.get('observed_prefix_hours',0)),
                        'right_executed_hours':right.get('executed_hours',right.get('observed_prefix_hours',0))}
                if valid:
                    assert left['exogenous_sha256']==right['exogenous_sha256'],'Within-cell actual external inputs differ'
                for metric in COMPARE_METRICS:
                    record['left_minus_right_'+metric]=(left['metrics'][metric]-right['metrics'][metric]) if valid else None
                # Preserve recovery states on both sides, including failure/null.
                record['left_recovery']=left.get('recovery') if valid else None
                record['right_recovery']=right.get('recovery') if valid else None
                rows.append(record)
    return rows


def run():
    protocol_path=OUT/'protocol.json'
    protocol=json.loads(protocol_path.read_text('utf-8'))
    assert protocol['status']=='PREPARED_NOT_RUN'
    assert not (OUT/'analysis.json').exists(),'Do not overwrite an actual experiment'
    for relative,expected in protocol['source_sha256'].items():
        assert digest(ROOT/relative)==expected,f'Frozen source/input drift: {relative}'
    assert pd.read_csv(OUT/'case_grid.csv').to_dict('records')==list(GRID)
    outcomes=[]
    started=time.perf_counter()
    for row in GRID:
        target=OUT/row['case_id']
        config=protocol['configurations'][row['case_id']]
        if not row['anchor'] and not all(item['complete'] for item in outcomes if item['anchor']):
            target.mkdir(exist_ok=False)
            event.write_json(target/'parameters.json',config)
            record=dict(row,status='NOT_RUN',reason='ORIGINAL_FIVE_ANCHOR_GATE_FAILED',complete=False,
                        executed_hours=0,successful_lp_hours=0,attempted_lp_hours=0)
            event.write_json(target/'result.json',record)
        else:
            assert not target.exists(),'Do not overwrite an executed case'
            record=run_case(row,config,target)
        outcomes.append(record)
    drift={relative:digest(ROOT/relative) for relative,expected in protocol['source_sha256'].items() if digest(ROOT/relative)!=expected}
    complete=all(row['complete'] for row in outcomes) and not drift
    anchor_passed=all(row['complete'] for row in outcomes if row['anchor'])
    all_attempted=all(row['status']!='NOT_RUN' for row in outcomes) and len(outcomes)==160
    comparisons=cell_comparisons(outcomes) if all_attempted and not drift else []
    if comparisons:
        event.write_json(OUT/'within_cell_comparisons.json',comparisons)
        flat=[{**{key:value for key,value in row.items() if key not in ('left_recovery','right_recovery')},
               'left_recovery_json':json.dumps(event.json_value(row['left_recovery']),allow_nan=False),
               'right_recovery_json':json.dumps(event.json_value(row['right_recovery']),allow_nan=False)} for row in comparisons]
        pd.DataFrame(flat).to_csv(OUT/'within_cell_comparisons.csv',index=False,float_format='%.17g')
    pd.DataFrame([{**{key:row[key] for key in ('case_id','arm','phase_hours','duration_hours','status','complete','successful_lp_hours')},
                   'executed_hours':row.get('executed_hours',row.get('observed_prefix_hours',0)),**row.get('metrics',{})} for row in outcomes]).to_csv(OUT/'all_case_metrics.csv',index=False,float_format='%.17g')
    counts={arm['case_id']:{'valid_cells':sum(row['complete'] and row['arm']==arm['case_id'] for row in outcomes),'denominator':32} for arm in storage.GRID}
    pair_counts={pair['left_arm']+' minus '+pair['right_arm']:{'comparison_type':pair['comparison_type'],
        'complete_pairs':sum(row['pair_status']=='COMPLETE' and row['left_arm']==pair['left_arm'] and row['right_arm']==pair['right_arm'] for row in comparisons),
        'denominator':32} for pair in PAIRS}
    ranges={}
    if complete:
        assert sum(row['successful_lp_hours'] for row in outcomes)==57600
        for pair in PAIRS:
            selected=[row for row in comparisons if row['left_arm']==pair['left_arm'] and row['right_arm']==pair['right_arm']]
            ranges[pair['left_arm']+' minus '+pair['right_arm']]={'comparison_type':pair['comparison_type'],
                'metrics':{metric:{'min':min(row['left_minus_right_'+metric] for row in selected),
                                   'max':max(row['left_minus_right_'+metric] for row in selected)} for metric in COMPARE_METRICS}}
    status='COMPLETED_160_VALIDATED_360H_EVENT_DESIGN_PATHS' if complete else 'FAILED_FIXED_EVENT_DESIGN_COHORT'
    if not anchor_passed:
        status='FAILED_ORIGINAL_FIVE_ANCHOR_GATE'
    report={'status':status,'analysis_class':protocol.get('analysis_class'),'case_count':len(outcomes),'design_cells':32,
            'original_five_anchor_gate_passed':anchor_passed,'all_160_cases_attempted':all_attempted,
            'not_run_cases':sum(row['status']=='NOT_RUN' for row in outcomes),'valid_cases':sum(row['complete'] for row in outcomes),
            'arm_valid_cell_counts':counts,'complete_pair_counts':pair_counts,
            'global_numeric_domain_status':'AVAILABLE_ALL_160_VALID' if complete else 'NOT_ESTABLISHED_FOR_FULL_DOMAIN',
            'full_domain_ranges':ranges,'expected_total_lp_hours':57600,
            'actual_successful_lp_hours':sum(row['successful_lp_hours'] for row in outcomes),
            'actual_attempted_lp_hours':sum(row['attempted_lp_hours'] for row in outcomes),
            'actual_complete_trajectory_hours':sum(row.get('executed_hours',0) for row in outcomes),
            'wall_seconds':time.perf_counter()-started,'source_drift':drift,'protocol_sha256':digest(protocol_path),
            'protocol':protocol,'cases':outcomes,'within_cell_comparisons':comparisons,
            'new_native_runs':0,'paid_services_used':False,
            'output_sha256':{path.relative_to(OUT).as_posix():digest(path) for path in OUT.rglob('*') if path.is_file() and path.name!='analysis.json'}}
    event.write_json(OUT/'analysis.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--run',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else run()
