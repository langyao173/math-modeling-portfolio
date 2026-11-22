"""Battery energy-configuration challenge of five frozen storage controllers.

The reported 2024 national multi-technology duration challenges the engineering
4h energy assumption; it does not calibrate a 2060 battery fleet or its capital.
AI-assisted: OpenAI Codex / GPT-6, 2026-10-19.
"""
from __future__ import annotations

import argparse
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

from src import v3_hourly as h
from src import v13_event_information as event
from src import v13_storage_baseline as storage
from src import v13_event_design as design
from src.v10_prepare_inputs import make_case
from src.v10_validate_dispatch import _prefix_audit
from src.v11_recovery_sensitivity import digest, recovery_windows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/v13/storage_duration'
STORAGE_OUT = ROOT/'outputs/v13/storage_baseline'
SOURCE_DIR = ROOT/'data/v13/parameter_sources'
SOURCE_PDF = SOURCE_DIR/'NEA_China_New_Energy_Storage_Report_2025.pdf'
SOURCE_PDF_SHA256 = '319066e4840bc923122a16ee6d0169c540e8d798831866382d68df4cb33c6044'
STORAGE_ANALYSIS_SHA256 = '226681246a962cfed47914ee01cae30d2cb8f8c72c544790be6b3b2939e02750'
DURATIONS = (4., 2.3)
GRID = tuple({'case_id':f'battery_{str(duration).replace(".","p")}_{arm["case_id"]}',
              'arm':arm['case_id'], 'battery_duration_hours':duration,
              'strategy':arm['strategy'], 'storage_power_multiplier':arm['storage_power_multiplier'],
              'local_rule':arm['local_rule'], 'anchor':duration == 4.}
             for duration in DURATIONS for arm in storage.GRID)
WITHIN_DURATION_PAIRS = design.PAIRS
COMPARE_METRICS = design.COMPARE_METRICS


def case_configuration(row):
    return storage.case_configuration(row)


def make_duration_case(row, config=None):
    """Set E using the unmultiplied snapshot P; both power arms share that E."""
    case = make_case(case_configuration(row) if config is None else config)
    baseline_power = case['capacity_gw']['battery_power']
    energy = baseline_power * row['battery_duration_hours']
    case['capacity_gw']['battery_energy'] = energy
    case['storage']['battery']['energy_gwh'] = energy
    case['storage']['battery']['initial_soc_gwh'] = .5 * energy
    return case


def source_evidence():
    index = json.loads((SOURCE_DIR/'index.json').read_text('utf-8'))
    assert digest(SOURCE_PDF) == SOURCE_PDF_SHA256 == index['sha256']
    assert SOURCE_PDF.stat().st_size == index['bytes']
    expected = {'evidence_pdf_page_1based':21, 'evidence_printed_page':11,
                'reported_2024_new_storage_power_GW':73.76,
                'reported_2024_new_storage_energy_GWh':168.,
                'reported_average_duration_hours':2.3}
    for key, value in expected.items():
        assert index[key] == value, f'Parameter source identity changed: {key}'
    return {**index, 'statistical_year':2024,
            'application':'Use reported rounded 2.3h as a battery-energy engineering challenge to the existing 4h case; retain all dispatch/control and annual investment decisions.',
            'scope_limit':'The report describes the national multi-technology new-type storage fleet in 2024. It is not pure lithium-only evidence or an observation/forecast for 2030 or 2060.',
            'capital_scope':'Annual investment actions and capital metrics are not re-estimated. Hourly resource/holding cost is an operational accounting result, not a full investment-economic comparison.'}


def input_preflight():
    checks, sources = storage.input_preflight()
    receipt_path = STORAGE_OUT/'analysis.json'
    assert digest(receipt_path) == STORAGE_ANALYSIS_SHA256
    receipt = json.loads(receipt_path.read_text('utf-8'))
    assert receipt['status'] == 'COMPLETED_FIVE_VALIDATED_360H_STORAGE_BASELINE_PATHS'
    assert receipt['actual_successful_lp_hours'] == 1800 and not receipt['source_drift']
    for relative, expected in receipt['protocol']['source_sha256'].items():
        assert digest(ROOT/relative) == expected, f'Frozen storage source drift: {relative}'
    paths = [Path(__file__), ROOT/'tests/test_v13_storage_duration.py',
             ROOT/'src/v13_event_design.py', SOURCE_PDF, SOURCE_DIR/'index.json',
             receipt_path, STORAGE_OUT/'protocol.json']
    for arm in storage.GRID:
        for name in ('actions_states.csv','result.json','summary.json','recovery.csv'):
            path = STORAGE_OUT/arm['case_id']/name
            assert digest(path) == receipt['output_sha256'][path.relative_to(STORAGE_OUT).as_posix()]
            paths.append(path)
    inputs = {}
    for row in GRID:
        original = make_case(case_configuration(row))
        case = make_duration_case(row)
        for key in original:
            if key not in ('storage','capacity_gw'):
                if isinstance(original[key],np.ndarray):
                    np.testing.assert_array_equal(case[key],original[key])
                else:
                    assert case[key] == original[key], f'Unexpected change to {key}'
        assert case['storage']['pumped'] == original['storage']['pumped']
        for key in ('power_gw','eta_charge','eta_discharge'):
            assert case['storage']['battery'][key] == original['storage']['battery'][key]
        for key in original['capacity_gw']:
            if key != 'battery_energy':
                assert case['capacity_gw'][key] == original['capacity_gw'][key]
        energy = case['capacity_gw']['battery_power']*row['battery_duration_hours']
        assert case['storage']['battery']['energy_gwh'] == energy
        assert case['storage']['battery']['initial_soc_gwh'] == .5*energy
        if row['anchor']:
            assert case['storage'] == original['storage']
            assert case['capacity_gw'] == original['capacity_gw']
        inputs[row['case_id']] = design.initial_state_and_controls(case)
    for duration in DURATIONS:
        group = [inputs[row['case_id']] for row in GRID if row['battery_duration_hours'] == duration]
        assert len({item['storage']['battery']['energy_gwh'] for item in group}) == 1
        assert len({item['storage']['battery']['initial_soc_gwh'] for item in group}) == 1
        assert len({item['storage']['pumped']['energy_gwh'] for item in group}) == 1
    for path in paths:
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    return checks, inputs, source_evidence(), sources


def prepare():
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Frozen protocol exists; do not overwrite it')
    checks, inputs, evidence, sources = input_preflight()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'status':'PREPARED_NOT_RUN', 'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'analysis_class':'DEVELOPMENT_BATTERY_ENERGY_CONFIGURATION_CHALLENGE',
        'grid':GRID, 'case_count':10, 'hours_per_case':360, 'expected_total_lp_hours':3600,
        'executed_scientific_lp_hours_at_freeze':0, 'verification_lp_calls_per_full_test_suite':2,
        'battery_duration_hours':DURATIONS, 'parameter_source':evidence,
        'configurations':{row['case_id']:case_configuration(row) for row in GRID},
        'initial_state_and_controls':inputs,
        'energy_change':'Battery energy = unmultiplied snapshot battery power * duration; initial SOC = 0.5 * this energy. Both P1 and P1.5 receive identical energy at each duration. Pumped energy remains the original six-hour configuration, with original initial SOC.',
        'unchanged':'plan_02349_2060 snapshot; 360h; event starts [48,192], each 48h; severity 1; inherited fuel; common 24h perfect information; original five frozen controllers; same external arrays, fuel/thermal/debt initialization, efficiencies, original LP objective/constraints, physical state transitions and absolute daily debt deadlines. Annual controls and investment capital remain frozen.',
        'execution_order':'Five 4h original anchors first in storage.GRID order, then five 2.3h cases in the same order.',
        'anchor_release_gate':'All five original 4h paths must reproduce every frozen storage-baseline action/state column, reference summary field and recovery field within 1e-6, and pass both physical audits. Anchor failure retains all five outcomes and five explicit NOT_RUN new cases.',
        'failure_policy':'After anchor release attempt all five new cases. Preserve original exceptions, failure hour/pre-state, full input/hash/window index, actual actions and successful physical prefixes. No LP relaxation, debt erasure, fallback or action substitution.',
        'comparison_release':'Numerical comparisons are released only after all ten cases have been attempted and source hashes remain unchanged. Preserve all five within-arm duration pairs and all five predeclared contrasts within each duration. Incomplete/failed pairs keep statuses and null numerical differences. No full-domain range or overall benefit claim if any case fails.',
        'primary_metrics':{'interval':'absolute hours 48..359 inclusive', 'metrics':['ENS_GWh','ENS_fraction_of_original_demand','peak_unserved_GW'],
                           'denominator':'Actual exogenous demand before flexible-load shifting, summed over the same absolute window.'},
        'secondary_metrics':['whole360_and_warmup_h0_47_ENS','direct_fuel_shortage','physical_cost_excluding_shortage_carbon_including_holding','CO2','initial_and_terminal_fuel_SOC_debt','shift_out_in','original_event_recovery_and_rebreach','shortage_while_charging'],
        'physical_cost_formula':'resource_accounted - shortage_penalty - carbon_cost + stock_holding; million CNY engineering values. This excludes new storage capital because annual investment capital is not recalculated.',
        'recovery_rule':{'stock_ratio':.95,'service_limit':.01,'window_hours':24,'event_boundaries':[192,360],'normalization':'actual pre-event end state at start-1'},
        'physical_gate':'Original h.audit and reused independent per-hour action/state conservation and actual bounds at 1e-6.',
        'within_arm_duration_pairs':[{ 'arm':arm['case_id'],'left_duration_hours':2.3,'right_duration_hours':4.,'comparison_type':'BATTERY_ENERGY_INTERVENTION_FIXED_POWER_AND_POLICY'} for arm in storage.GRID],
        'within_duration_pairs':WITHIN_DURATION_PAIRS, 'comparison_metrics':COMPARE_METRICS,
        'evidence_storage':'Full actual action/state CSV, complete exogenous CSV, initial controls, source hashes. Each hourly log retains actual pre-state/action and absolute window/hash; all four saved perfect-information slices are verified. Current local telemetry is saved; future local plans are reconstructed from frozen code, state and inputs.',
        'input_checks':checks, 'source_sha256':sources,
        'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'solver':'scipy.optimize.linprog(method=highs)'},
        'commands':{'test':'python -m unittest discover -s tests -p test_v13_storage_duration.py -v',
                    'prepare':'python -m src.v13_storage_duration --prepare',
                    'run_after_independent_review':'python -m src.v13_storage_duration --run'},
        'working_directory':str(ROOT), 'new_native_runs':0, 'paid_services_used':False,
    }
    event.write_json(OUT/'protocol.json',protocol)
    pd.DataFrame(GRID).to_csv(OUT/'case_grid.csv',index=False)
    print(json.dumps({'status':protocol['status'],'cases':10,'scientific_lp_hours':0}))
    return protocol


def reference_field_comparison(new, reference):
    """Compare every original JSON field; retain all numeric differences."""
    numeric, different = {}, {}
    def compare(actual, expected, key):
        if isinstance(expected,dict):
            for name,value in expected.items():
                if name not in actual:
                    different[key+'/'+name] = {'missing':True}
                else:
                    compare(actual[name],value,key+'/'+name)
        elif isinstance(expected,list):
            if len(actual) != len(expected):
                different[key] = {'expected_length':len(expected),'actual_length':len(actual)}
            else:
                for index,value in enumerate(expected):
                    compare(actual[index],value,key+'/'+str(index))
        elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
            numeric[key] = float(actual)-float(expected)
        elif actual != expected:
            different[key] = {'new':actual,'reference':expected}
    compare(new,reference,'')
    return {'passed':not different and all(np.isfinite(v) and abs(v)<=event.TOLERANCE for v in numeric.values()),
            'numeric_differences':numeric,'other_differences':different}


def anchor_comparison(row, case, table, summary, recovery):
    folder = STORAGE_OUT/row['arm']
    original = pd.read_csv(folder/'actions_states.csv',float_precision='round_trip')
    assert list(original.columns) == list(table.columns)
    assert len(original) == len(table) == 360
    differences = {key:float(np.abs(table[key].to_numpy()-original[key].to_numpy()).max()) for key in original.columns}
    old_summary = json.loads((folder/'summary.json').read_text('utf-8'))
    receipt = json.loads((STORAGE_OUT/'analysis.json').read_text('utf-8'))
    old_case = next(item for item in receipt['cases'] if item['case_id'] == row['arm'])
    summary_check = reference_field_comparison(summary,old_summary)
    recovery_check = reference_field_comparison(recovery,old_case['recovery'])
    result = {'reference_analysis_sha256':STORAGE_ANALYSIS_SHA256,
              'frozen_storage_path_max_absolute_column_differences':differences,
              'all_reference_columns_match':all(np.isfinite(v) and v<=event.TOLERANCE for v in differences.values()),
              'all_reference_summary_fields':summary_check,'all_reference_recovery_fields':recovery_check,
              'reference_backend':'Completed V13 storage-baseline SciPy; no new native execution.'}
    result['passed'] = result['all_reference_columns_match'] and summary_check['passed'] and recovery_check['passed']
    return result


def run_case(row, config, target):
    target.mkdir(parents=True,exist_ok=False)
    case = make_duration_case(row,config)
    event.write_json(target/'parameters_initial_state.json',{
        'configuration':config,'battery_duration_hours':row['battery_duration_hours'],'local_rule':row['local_rule'],
        'battery_energy_formula':'unmultiplied_snapshot_power_GW * duration_h',
        'annual_capital_reestimated':False,'initial_state_and_controls':design.initial_state_and_controls(case)})
    input_path = target/'exogenous_inputs.csv'
    design.exogenous_table(case).to_csv(input_path,index=False,float_format='%.17g')
    input_hash = digest(input_path)
    inputs = design.forecast_arrays(pd.read_csv(input_path,float_precision='round_trip'))
    calls = []
    record = dict(row,status='RUNNING',complete=False,exogenous_sha256=input_hash)
    started = time.perf_counter()
    try:
        result = storage.simulate_storage(case,config['strategy'],row['local_rule'],calls)
        result['timeseries'].to_csv(target/'actions_states.csv',index=False,float_format='%.17g')
        table = pd.read_csv(target/'actions_states.csv',float_precision='round_trip')
        original_audit = h.audit(case,table)
        residual, independent = event.independent_audit(case,table)
        residual.to_csv(target/'independent_physical_audit.csv',index=False,float_format='%.17g')
        record.update(executed_hours=len(table),original_physical_audit=original_audit,independent_physical_audit=independent)
        assert original_audit['passed'] and independent['passed'], 'Physical audit failed'
        summary = design.summarize(case,table,config['strategy'])
        recovery = recovery_windows(table,design.recovery_configuration(config))
        event.write_json(target/'summary.json',summary)
        pd.DataFrame(recovery).to_csv(target/'recovery.csv',index=False)
        charge = table.battery_charge_gw+table.pumped_charge_gw
        shortage = (table.unserved_gw>1e-5)|(table[[f+'_direct_unserved_mtce' for f in h.FUELS]].sum(axis=1)>1e-8)
        table.loc[(charge>1e-5)&shortage].to_csv(target/'shortage_while_charging.csv',index=False,float_format='%.17g')
        record.update(metrics=summary,recovery=recovery)
        if row['anchor']:
            comparison = anchor_comparison(row,case,table,summary,recovery)
            event.write_json(target/'anchor_comparison.json',comparison)
            record['anchor_comparison'] = comparison
            assert comparison['passed'], 'Original five-arm 4h anchor gate failed'
        record.update(status='COMPLETED_AND_VALIDATED',complete=True)
    except Exception as failure:
        planner_failure = bool(calls and 'error' in calls[-1])
        record.update(status='FAILED',error=str(failure),error_type=type(failure).__name__,traceback=traceback.format_exc(),
                      failure_hour=calls[-1]['hour'] if planner_failure else None,
                      last_attempted_hour=calls[-1]['hour'] if calls else None,
                      failure_phase='PLANNER' if planner_failure else 'SIMULATION_OR_VALIDATION')
        if not (target/'actions_states.csv').exists():
            try:
                prefix = event.prefix_from_calls(case,calls)
                record['observed_prefix_hours'] = len(prefix)
                if len(prefix):
                    prefix.to_csv(target/'observed_prefix_actions_states.csv',index=False,float_format='%.17g')
                    residual, independent = event.independent_audit(case,prefix,complete=False)
                    residual.to_csv(target/'prefix_independent_physical_audit.csv',index=False,float_format='%.17g')
                    record['prefix_original_physical_audit'] = _prefix_audit(case,prefix)
                    record['prefix_independent_physical_audit'] = independent
            except Exception as evidence_failure:
                record['prefix_evidence_error'] = {'type':type(evidence_failure).__name__,'error':str(evidence_failure),'traceback':traceback.format_exc()}
    finally:
        record.update(successful_lp_hours=sum('action' in call for call in calls),attempted_lp_hours=len(calls),wall_seconds=time.perf_counter()-started)
        try:
            compact, checks = design.compact_planner_logs(calls,inputs,input_hash)
            with (target/'planner_actual_states_actions.jsonl').open('w',encoding='utf-8') as stream:
                for call in compact:
                    stream.write(json.dumps(event.json_value(call),ensure_ascii=False,allow_nan=False)+'\n')
            record['forecast_slice_verification'] = checks
        except Exception as log_failure:
            record.update(status='FAILED',complete=False,
                          evidence_log_error={'type':type(log_failure).__name__,'error':str(log_failure),'traceback':traceback.format_exc()})
            with (target/'planner_actual_states_actions_unverified_forecast.jsonl').open('w',encoding='utf-8') as stream:
                for call in calls:
                    observed = {key:value for key,value in call.items() if key not in ('forecast_window','local_storage_forecast')}
                    observed['forecast_input'] = {'file':'exogenous_inputs.csv','sha256':input_hash,
                                                  'start_hour':call['hour'],'stop_hour_exclusive':call['forecast_window_stop'],
                                                  'saved_slice_exact_match':False}
                    stream.write(json.dumps(event.json_value(observed),ensure_ascii=False,allow_nan=False)+'\n')
        event.write_json(target/'result.json',record)
    print(json.dumps({key:record.get(key) for key in ('case_id','status','executed_hours','successful_lp_hours','error')},ensure_ascii=False),flush=True)
    return record


def pair_comparisons(outcomes):
    by_key = {(row['battery_duration_hours'],row['arm']):row for row in outcomes}
    definitions = []
    for arm in storage.GRID:
        definitions.append({'comparison_id':'duration_'+arm['case_id'],
            'comparison_type':'BATTERY_ENERGY_INTERVENTION_FIXED_POWER_AND_POLICY',
            'left':(2.3,arm['case_id']),'right':(4.,arm['case_id'])})
    for duration in DURATIONS:
        for pair in WITHIN_DURATION_PAIRS:
            definitions.append({'comparison_id':f'{duration}h_'+pair['left_arm']+'_minus_'+pair['right_arm'],
                'comparison_type':pair['comparison_type'],
                'left':(duration,pair['left_arm']),'right':(duration,pair['right_arm'])})
    rows = []
    for definition in definitions:
        left, right = by_key[definition['left']],by_key[definition['right']]
        valid = left['complete'] and right['complete']
        row = {key:value for key,value in definition.items() if key not in ('left','right')}
        row.update(left_case_id=left['case_id'],right_case_id=right['case_id'],
                   pair_status='COMPLETE' if valid else 'FAILED_OR_INCOMPLETE_PAIR',
                   left_status=left['status'],right_status=right['status'],
                   left_executed_hours=left.get('executed_hours',left.get('observed_prefix_hours',0)),
                   right_executed_hours=right.get('executed_hours',right.get('observed_prefix_hours',0)))
        if valid:
            assert left['exogenous_sha256'] == right['exogenous_sha256'], 'Paired external inputs differ'
        for metric in COMPARE_METRICS:
            row['left_minus_right_'+metric] = left['metrics'][metric]-right['metrics'][metric] if valid else None
        row['left_recovery'] = left.get('recovery') if valid else None
        row['right_recovery'] = right.get('recovery') if valid else None
        rows.append(row)
    return rows


def run():
    protocol_path = OUT/'protocol.json'
    protocol = json.loads(protocol_path.read_text('utf-8'))
    assert protocol['status'] == 'PREPARED_NOT_RUN'
    assert not (OUT/'analysis.json').exists(), 'Do not overwrite an actual experiment'
    for relative, expected in protocol['source_sha256'].items():
        assert digest(ROOT/relative) == expected, f'Frozen source/input drift: {relative}'
    assert pd.read_csv(OUT/'case_grid.csv').to_dict('records') == list(GRID)
    outcomes = []
    started = time.perf_counter()
    for row in GRID:
        target = OUT/row['case_id']
        config = protocol['configurations'][row['case_id']]
        if not row['anchor'] and not all(item['complete'] for item in outcomes if item['anchor']):
            target.mkdir(exist_ok=False)
            event.write_json(target/'parameters.json',{'configuration':config,'battery_duration_hours':row['battery_duration_hours']})
            record = dict(row,status='NOT_RUN',reason='ORIGINAL_FIVE_4H_ANCHOR_GATE_FAILED',complete=False,
                          executed_hours=0,successful_lp_hours=0,attempted_lp_hours=0)
            event.write_json(target/'result.json',record)
        else:
            assert not target.exists(), 'Do not overwrite an executed case'
            record = run_case(row,config,target)
        outcomes.append(record)
    drift = {relative:digest(ROOT/relative) for relative,expected in protocol['source_sha256'].items() if digest(ROOT/relative)!=expected}
    anchor_passed = all(row['complete'] for row in outcomes if row['anchor'])
    all_attempted = len(outcomes) == 10 and all(row['status']!='NOT_RUN' for row in outcomes)
    complete = all(row['complete'] for row in outcomes) and not drift
    comparisons = pair_comparisons(outcomes) if all_attempted and not drift else []
    if comparisons:
        event.write_json(OUT/'paired_comparisons.json',comparisons)
        flat = [{**{key:value for key,value in row.items() if key not in ('left_recovery','right_recovery')},
                 'left_recovery_json':json.dumps(event.json_value(row['left_recovery']),allow_nan=False),
                 'right_recovery_json':json.dumps(event.json_value(row['right_recovery']),allow_nan=False)} for row in comparisons]
        pd.DataFrame(flat).to_csv(OUT/'paired_comparisons.csv',index=False,float_format='%.17g')
    pd.DataFrame([{**{key:row[key] for key in ('case_id','arm','battery_duration_hours','status','complete','successful_lp_hours')},
                   'executed_hours':row.get('executed_hours',row.get('observed_prefix_hours',0)),**row.get('metrics',{})} for row in outcomes]).to_csv(OUT/'all_case_metrics.csv',index=False,float_format='%.17g')
    if complete:
        assert sum(row['successful_lp_hours'] for row in outcomes) == 3600
    status = 'COMPLETED_TEN_VALIDATED_360H_STORAGE_DURATION_PATHS' if complete else 'FAILED_FIXED_STORAGE_DURATION_COHORT'
    if not anchor_passed:
        status = 'FAILED_ORIGINAL_FIVE_4H_ANCHOR_GATE'
    report = {'status':status,'analysis_class':protocol.get('analysis_class'),'case_count':len(outcomes),
              'original_five_4h_anchor_gate_passed':anchor_passed,'all_ten_cases_attempted':all_attempted,
              'not_run_cases':sum(row['status']=='NOT_RUN' for row in outcomes),'valid_cases':sum(row['complete'] for row in outcomes),
              'arm_valid_case_counts':{arm['case_id']:{'valid_cases':sum(row['complete'] and row['arm']==arm['case_id'] for row in outcomes),'denominator':2} for arm in storage.GRID},
              'complete_pairs':sum(row['pair_status']=='COMPLETE' for row in comparisons),'pair_denominator':15,
              'global_numeric_domain_status':'AVAILABLE_ALL_TEN_VALID' if complete else 'NOT_ESTABLISHED_FOR_FULL_DOMAIN',
              'full_domain_ranges':{}, 'expected_total_lp_hours':3600,
              'actual_successful_lp_hours':sum(row['successful_lp_hours'] for row in outcomes),
              'actual_attempted_lp_hours':sum(row['attempted_lp_hours'] for row in outcomes),
              'actual_complete_trajectory_hours':sum(row.get('executed_hours',0) for row in outcomes),
              'wall_seconds':time.perf_counter()-started,'source_drift':drift,'protocol_sha256':digest(protocol_path),
              'protocol':protocol,'cases':outcomes,'paired_comparisons':comparisons,
              'new_native_runs':0,'paid_services_used':False,
              'output_sha256':{path.relative_to(OUT).as_posix():digest(path) for path in OUT.rglob('*') if path.is_file() and path.name!='analysis.json'}}
    event.write_json(OUT/'analysis.json',report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--run',action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
