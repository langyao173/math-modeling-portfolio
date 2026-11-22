"""Matched no-event paths and three explicitly separate inventory references.

Preparation runs no LP. The normal path keeps each frozen annual snapshot and
hourly controller; only the original designed events are absent. Old recovery
decisions are read, never changed or assigned to the normal counterfactual.
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
from src import v13_repair_operations as repair
from src.v13_event_information import independent_audit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'outputs/v13/repair_operations'
OUT = 'outputs/v13/inventory_reference'
HOURS = 360
WINDOWS = ((1,48,168,192),(2,192,336,360))


def case_grid():
    return [dict(case_id=row['case_id']+'_normal',stress_case_id=row['case_id'],
                 snapshot_id=row['snapshot_id'],plan=row['plan'],year=row['year'],
                 strategy=row['strategy'],hours=HOURS)
            for row in repair.case_grid()]


def normal_case(snapshot):
    case = h.make_case(snapshot,'normal',HOURS)
    # The generic normal factory exposes a generic timing mask even when no
    # shock changes its arrays. This experiment has no events, including labels.
    case['stress_mask'][:] = False
    case.update(evidence_class='MATCHED_NO_EVENT_COUNTERFACTUAL_DESIGN_WINDOW',
                initialization_contract='Same frozen annual end-state fuel stocks/capacity/realized flows; engineering SOC50%, thermal0, debt0',
                event_starts=[],counterfactual='Original event availability/demand/import perturbations absent; no new control')
    return case


def ratio(numerator,denominator):
    if denominator is None:
        return None,'REFERENCE_PATH_UNAVAILABLE'
    if not np.isfinite(denominator) or not np.isfinite(numerator):
        raise ValueError('Nonfinite stock cannot support an inventory ratio')
    if denominator <= 0:
        return None,'NONPOSITIVE_DENOMINATOR'
    return float(numerator/denominator),'VALID'


def load_stress(root,entry):
    target = Path(root)/SOURCE/'cases'/entry['stress_case_id']
    result = json.loads((target/'result.json').read_text('utf-8'))
    if not result['complete'] or result['status'] != 'COMPLETED_AND_VALIDATED':
        raise AssertionError('A complete original stress pair is required: '+entry['stress_case_id'])
    table = pd.read_csv(target/'actions_states.csv')
    recovery = pd.read_csv(target/'recovery.csv')
    if len(table) != HOURS or table.hour.tolist() != list(range(HOURS)) or len(recovery) != 2:
        raise AssertionError('Incomplete original stress path or recovery rows')
    return table,recovery


def paired_diagnostics(entry,snapshot,stress,normal,original_recovery):
    """All 360x3 hourly and 2x3 window cells remain, including failed pairs."""
    identity = {key:entry[key] for key in ('case_id','stress_case_id','snapshot_id','plan','year','strategy')}
    initial = snapshot['fuel_initial_stocks_mtce']
    hourly,windows = [],[]
    complete_normal = normal is not None
    if complete_normal and (len(normal) != HOURS or normal.hour.tolist() != list(range(HOURS))):
        raise AssertionError('Only a validated complete normal path may form a matched pair')
    for hour in range(HOURS):
        event = 2 if hour >= 192 else 1 if hour >= 48 else None
        for fuel in h.FUELS:
            actual = float(stress.loc[hour,fuel+'_stock_mtce'])
            baseline = float(normal.loc[hour,fuel+'_stock_mtce']) if complete_normal else None
            origin = float(initial[fuel])
            reference_hour = (191 if event == 2 else 47) if event else None
            pre = float(stress.loc[reference_hour,fuel+'_stock_mtce']) if event else None
            a,a_status = ratio(actual,pre) if event else (None,'NOT_APPLICABLE_BEFORE_ORIGINAL_EVENT')
            b,b_status = ratio(actual,origin)
            c,c_status = ratio(actual,baseline)
            n,n_status = ratio(baseline,origin) if baseline is not None else (None,'REFERENCE_PATH_UNAVAILABLE')
            difference = actual-baseline if baseline is not None else None
            hourly.append(dict(identity,hour=hour,fuel=fuel,event_reference=event,
                stress_stock_mtce=actual,normal_stock_mtce=baseline,t0_stock_mtce=origin,
                original_event_prestock_mtce=pre,
                A_stress_over_preevent=a,A_reference_status=a_status,
                B_stress_over_t0=b,B_reference_status=b_status,
                C_stress_over_matched_normal=c,C_reference_status=c_status,
                normal_over_t0=n,normal_t0_reference_status=n_status,
                stress_minus_normal_stock_mtce=difference,
                matched_extra_inventory_draw_normal_minus_stress_mtce=-difference if difference is not None else None,
                stress_gap_to_95pct_t0_mtce=max(0.,.95*origin-actual),
                stress_gap_to_100pct_t0_mtce=max(0.,origin-actual),
                normal_gap_to_95pct_t0_mtce=max(0.,.95*origin-baseline) if baseline is not None else None,
                normal_gap_to_100pct_t0_mtce=max(0.,origin-baseline) if baseline is not None else None,
                pair_status='COMPLETE_MATCHED_PATHS' if complete_normal else 'NORMAL_PATH_UNAVAILABLE'))
    for event,start,first,stop in WINDOWS:
        original = original_recovery.loc[original_recovery.event == event].iloc[0]
        def original_hour(name):
            return None if pd.isna(original[name]) else int(original[name])
        for fuel in h.FUELS:
            part = [row for row in hourly if row['fuel']==fuel and first <= row['hour'] < stop]
            end = part[-1]
            actual = np.array([row['stress_stock_mtce'] for row in part])
            baseline = np.array([row['normal_stock_mtce'] for row in part],float) if complete_normal else None
            origin = float(initial[fuel])
            def minimum(key):
                values = [row[key] for row in part]
                return float(min(values)) if all(value is not None for value in values) else None
            delta = actual-baseline if complete_normal else None
            record = dict(identity,event=event,original_event_start_hour=start,
                window_first_hour=first,window_end_exclusive=stop,hours=24,fuel=fuel,
                pair_status='COMPLETE_MATCHED_PATHS' if complete_normal else 'NORMAL_PATH_UNAVAILABLE',
                t0_stock_mtce=origin,original_event_prestock_mtce=end['original_event_prestock_mtce'],
                stress_window_min_stock_mtce=float(actual.min()),stress_window_end_stock_mtce=float(actual[-1]),
                normal_window_min_stock_mtce=float(baseline.min()) if complete_normal else None,
                normal_window_end_stock_mtce=float(baseline[-1]) if complete_normal else None,
                A_min_stress_over_preevent=minimum('A_stress_over_preevent'),
                A_end_stress_over_preevent=end['A_stress_over_preevent'],
                B_min_stress_over_t0=minimum('B_stress_over_t0'),B_end_stress_over_t0=end['B_stress_over_t0'],
                C_min_stress_over_matched_normal=minimum('C_stress_over_matched_normal'),
                C_end_stress_over_matched_normal=end['C_stress_over_matched_normal'],
                C_reference_status=('REFERENCE_PATH_UNAVAILABLE' if not complete_normal else
                                    'NONPOSITIVE_DENOMINATOR_IN_WINDOW' if (baseline<=0).any() else 'VALID'),
                C_nonpositive_reference_hours=int((baseline<=0).sum()) if complete_normal else None,
                normal_min_over_t0=minimum('normal_over_t0'),normal_end_over_t0=end['normal_over_t0'],
                stress_minus_normal_window_min_mtce=float(delta.min()) if complete_normal else None,
                stress_minus_normal_window_max_mtce=float(delta.max()) if complete_normal else None,
                stress_minus_normal_window_end_mtce=float(delta[-1]) if complete_normal else None,
                matched_extra_draw_window_min_mtce=float((-delta).min()) if complete_normal else None,
                matched_extra_draw_window_max_mtce=float((-delta).max()) if complete_normal else None,
                matched_extra_draw_window_end_mtce=float(-delta[-1]) if complete_normal else None,
                stress_gap_to_95pct_t0_at_window_min_mtce=float(max(0.,.95*origin-actual.min())),
                stress_gap_to_100pct_t0_at_window_min_mtce=float(max(0.,origin-actual.min())),
                stress_gap_to_95pct_t0_at_window_end_mtce=end['stress_gap_to_95pct_t0_mtce'],
                stress_gap_to_100pct_t0_at_window_end_mtce=end['stress_gap_to_100pct_t0_mtce'],
                normal_gap_to_95pct_t0_at_window_min_mtce=float(max(0.,.95*origin-baseline.min())) if complete_normal else None,
                normal_gap_to_100pct_t0_at_window_min_mtce=float(max(0.,origin-baseline.min())) if complete_normal else None,
                original_final_window_service_ready=bool(original.final_window_service_ready),
                original_final_window_stock_ready=bool(original.final_window_stock_ready),
                original_final_window_joint_ready=bool(original.final_window_joint_ready),
                original_service_confirmation_hour=original_hour('service_confirmation_hour'),
                original_joint_confirmation_hour=original_hour('joint_confirmation_hour'),
                original_service_right_censored=bool(original.service_right_censored),
                original_joint_right_censored=bool(original.joint_right_censored),
                original_stock_rebreach_hour_after_joint_confirmation=original_hour('stock_rebreach_hour_after_joint_confirmation'))
            windows.append(record)
    return hourly,windows


def source_names():
    names = ['src/v13_inventory_reference.py','tests/test_v13_inventory_reference.py',
             'src/v3_hourly.py','src/v4_hourly.py','src/v13_repair_operations.py',
             'src/v13_event_information.py',SOURCE+'/analysis.json',SOURCE+'/protocol.json',
             SOURCE+'/snapshots.json',SOURCE+'/annual_source_audit.json']
    names += [SOURCE+'/inputs/'+snapshot_id+'/initial_state.json'
              for snapshot_id in sorted({row['snapshot_id'] for row in case_grid()})]
    names += [SOURCE+'/cases/'+row['stress_case_id']+'/'+name for row in case_grid()
              for name in ('actions_states.csv','result.json','recovery.csv')]
    return names


def prepare(root=ROOT):
    root = Path(root).resolve(); out = root/OUT
    if (out/'analysis.json').exists() or any((out/'cases').glob('*')):
        raise FileExistsError('Preserve actual normal execution; no preparation overwrite')
    out.mkdir(parents=True,exist_ok=True)
    names = source_names()
    before = {name:repair.digest(root/name) for name in names}
    source_receipt = json.loads((root/SOURCE/'analysis.json').read_text('utf-8'))
    if source_receipt['status'] != 'COMPLETED_SIXTEEN_VALIDATED_360H_PATHS' or source_receipt['actual_successful_lp_hours'] != 5760:
        raise AssertionError('Complete original sixteen-path cohort required')
    snapshots = json.loads((root/SOURCE/'snapshots.json').read_text('utf-8'))
    initial_checks = {}
    for snapshot_id,snapshot in snapshots.items():
        case = normal_case(snapshot)
        expected = json.loads((root/SOURCE/'inputs'/snapshot_id/'initial_state.json').read_text('utf-8'))
        actual = repair.json_value(repair.physical.initial_state(case))
        if actual != expected:
            raise AssertionError('Normal initial state differs from stress: '+snapshot_id)
        target = out/'inputs'/snapshot_id; target.mkdir(parents=True,exist_ok=True)
        repair.write_json(target/'snapshot.json',snapshot)
        repair.write_json(target/'initial_state.json',actual)
        repair.write_json(target/'initial_and_parameters.json',{key:value for key,value in case.items() if not isinstance(value,np.ndarray)})
        repair.external_frame(case).to_csv(target/'external_inputs.csv',index=False,float_format='%.17g')
        initial_checks[snapshot_id] = dict(passed=True,exact_initial_state_equal=True,
                                          event_active_hours=int(case['stress_mask'].sum()),hours=HOURS)
    for entry in case_grid():
        stress,recovery = load_stress(root,entry)
        for name in ('actions_states.csv','result.json','recovery.csv'):
            key = 'cases/'+entry['stress_case_id']+'/'+name
            if repair.digest(root/SOURCE/key) != source_receipt['output_sha256'][key]:
                raise AssertionError('Original stress evidence changed: '+key)
        # No new solve: verify the complete old trajectory against its own inputs.
        actual_case = repair.make_case_from_snapshot(snapshots[entry['snapshot_id']])
        if not h.audit(actual_case,stress)['passed']:
            raise AssertionError('Original stress pair failed read-only physical check')
    repair.write_json(out/'snapshots.json',snapshots)
    pd.DataFrame(case_grid()).to_csv(out/'case_grid.csv',index=False)
    after = {name:repair.digest(root/name) for name in names}
    if before != after:
        raise AssertionError('Frozen sources/inputs changed during preparation')
    artifacts = {path.relative_to(out).as_posix():repair.digest(path) for path in out.rglob('*')
                 if path.is_file() and path.name not in ('protocol.json','preparation_audit.json','code_review.json')}
    protocol = dict(status='PREPARED_NOT_RUN',prepared_utc=datetime.now(timezone.utc).isoformat(),
        analysis_class='EXPLORATORY_MATCHED_NO_EVENT_COUNTERFACTUAL_AND_SEPARATE_INVENTORY_REFERENCE_DIAGNOSTIC',
        cases=case_grid(),case_count=16,hours_per_case=HOURS,planned_lp_hours=5760,
        source_sha256=before,prepared_artifact_sha256=artifacts,
        command='python -B -X utf8 -m src.v13_inventory_reference --run',
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
        design='Same frozen four controls x2030/2060 x coordinated/decentralized,360h,original24h deterministic engineering forecasts. Remove only event wind/solar/hydro availability, demand and import perturbations. Inherited supply; no contracts or parameter/asset/stock changes.',
        initialization='Exact original annual end-year fuel/capacity/realized domestic and import flows, SOC50%,thermal0,debt0. Compare complete initial state to frozen stress initial state; execute original h.simulate, no debt/stock reset.',
        reference_A='Original stress stock divided by its own event h47/h191 pre-stock; keep original saved95%24h service+stock recovery and rebreach decisions unchanged. Normal path has no event recovery.',
        reference_B='Stress stock/t0 initial stock and normal stock/t0, explicitly separate from original event-normalized definition.',
        reference_C='Stress_stock(h)/matched_normal_stock(h) at identical absolute hours. Nonpositive normal denominator => null+reason, never epsilon. Whole-window minimum is null if any hourly ratio is undefined; keep all cells.',
        windows=[dict(event=e,original_event_start_hour=s,first_hour=lo,end_exclusive=hi,hours=24) for e,s,lo,hi in WINDOWS],
        paired_outputs='17280 hourly fuel cells (16x360x3),96 fixed window-fuel cells (16x2x3),48 fuel terminal rows. Record stress/normal levels, A/B/C, normal/t0, actual differences and t0 95%/100% quantity gaps.',
        interpretation='Normal-minus-stress inventory is the additional draw in this same-model designed counterfactual, not measured national causal impact. Negative differences remain signed. Own demand paths differ by frozen annual plan; no cross-plan causal attribution.',
        costs='Report original physical operating cost and signed normal/stress differences separately. Refill-to95%/100% gaps are Mtce quantities only, never a second loss valuation; physical cost already accounts for net inventory draw.',
        failure_rule='Each failure preserves actual attempted hour/pre-state/actions/error and completed physical prefix. Continue every subsequent fixed case. Failed normal pairs retain null normal-dependent diagnostics in every fixed hourly/window/terminal cell; no omissions or successful full-path claim.',
        prohibited_changes='No new search, selection, contract control,45day inventory allocation, relaxed state/debt constraints, old95%gate changes, source/result/PASS renaming or new native execution.')
    repair.write_json(out/'protocol.json',protocol)
    audit = dict(status='PASS_PREPARED_NOT_RUN',new_hourly_lp_calls=0,planned_cases=16,planned_lp_hours=5760,
                 initial_state_checks=initial_checks,existing_stress_paths_checked=16,
                 source_hashes_unchanged=True,protocol_sha256=repair.digest(out/'protocol.json'))
    repair.write_json(out/'preparation_audit.json',audit)
    print(json.dumps({key:audit[key] for key in ('status','new_hourly_lp_calls','planned_cases','protocol_sha256')},ensure_ascii=False),flush=True)
    return audit


def run_case(root,out,entry,snapshot):
    target = Path(out)/'cases'/entry['case_id']; target.mkdir(parents=True)
    case = normal_case(snapshot); calls = []; started = time.perf_counter()
    record = dict(entry,status='RUNNING',complete=False,started_utc=datetime.now(timezone.utc).isoformat())
    try:
        with (target/'actual_pre_states_and_actions.jsonl').open('w',encoding='utf-8') as stream:
            result = repair.simulate_recorded(case,entry['strategy'],calls,stream)
        result['timeseries'].to_csv(target/'actions_states.csv',index=False,float_format='%.17g')
        table = pd.read_csv(target/'actions_states.csv')
        replay = repair.prefix_from_calls(case,calls)
        np.testing.assert_allclose(table.to_numpy(float),replay[table.columns].to_numpy(float),rtol=0,atol=1e-9)
        ordinary = h.audit(case,table)
        residual,independent = independent_audit(case,table)
        residual.to_csv(target/'independent_physical_audit.csv',index=False,float_format='%.17g')
        repair.write_json(target/'physical_audit.json',ordinary)
        repair.write_json(target/'independent_physical_audit.json',independent)
        if not ordinary['passed'] or not independent['passed']:
            raise AssertionError('Normal exported actual path failed physical checks')
        summary = repair.metrics(case,table,entry['strategy'])
        summary.update(counterfactual='MATCHED_NO_EVENT_PATH',no_event_recovery_defined=True)
        repair.write_json(target/'summary.json',summary)
        record.update(status='COMPLETED_AND_VALIDATED',complete=True,executed_hours=len(table),
                      metrics=summary,physical_audit=ordinary,independent_physical_audit=independent)
    except Exception as failure:
        record.update(status='FAILED',error=str(failure),error_type=type(failure).__name__,traceback=traceback.format_exc(),
                      failure_hour=(calls[-1]['hour'] if calls and 'error' in calls[-1] else None),
                      last_attempted_hour=(calls[-1]['hour'] if calls else None))
        if not (target/'actions_states.csv').exists():
            try:
                prefix = repair.prefix_from_calls(case,calls)
                record['executed_prefix_hours'] = len(prefix)
                if len(prefix):
                    prefix.to_csv(target/'observed_prefix_actions_states.csv',index=False,float_format='%.17g')
                    residual,independent = independent_audit(case,prefix,complete=False)
                    residual.to_csv(target/'prefix_independent_physical_audit.csv',index=False,float_format='%.17g')
                    record['prefix_physical_audit'] = repair.physical.prefix_audit(case,prefix)
                    record['prefix_independent_physical_audit'] = independent
            except Exception as audit_failure:
                record['prefix_evidence_error'] = dict(error=str(audit_failure),error_type=type(audit_failure).__name__,
                                                      traceback=traceback.format_exc())
    finally:
        record.update(successful_lp_hours=sum('action' in call for call in calls),attempted_lp_hours=len(calls),
                      wall_seconds=time.perf_counter()-started,completed_utc=datetime.now(timezone.utc).isoformat())
        repair.write_json(target/'result.json',record)
    return record


def terminal_rows(entry,snapshot,stress,normal):
    result = []
    for fuel in h.FUELS:
        actual = float(stress[fuel+'_stock_mtce'].iloc[-1])
        baseline = float(normal[fuel+'_stock_mtce'].iloc[-1]) if normal is not None else None
        origin = float(snapshot['fuel_initial_stocks_mtce'][fuel])
        c,status = ratio(actual,baseline)
        result.append(dict(case_id=entry['case_id'],stress_case_id=entry['stress_case_id'],snapshot_id=entry['snapshot_id'],
            plan=entry['plan'],year=entry['year'],strategy=entry['strategy'],fuel=fuel,hour=359,
            t0_stock_mtce=origin,stress_terminal_stock_mtce=actual,normal_terminal_stock_mtce=baseline,
            stress_minus_normal_terminal_stock_mtce=actual-baseline if baseline is not None else None,
            matched_extra_draw_normal_minus_stress_mtce=baseline-actual if baseline is not None else None,
            B_stress_over_t0=ratio(actual,origin)[0],normal_over_t0=ratio(baseline,origin)[0] if baseline is not None else None,
            C_stress_over_matched_normal=c,C_reference_status=status,
            stress_gap_to_95pct_t0_mtce=max(0.,.95*origin-actual),stress_gap_to_100pct_t0_mtce=max(0.,origin-actual),
            normal_gap_to_95pct_t0_mtce=max(0.,.95*origin-baseline) if baseline is not None else None,
            normal_gap_to_100pct_t0_mtce=max(0.,origin-baseline) if baseline is not None else None))
    return result


def run(root=ROOT):
    root = Path(root).resolve(); out = root/OUT
    protocol_path = out/'protocol.json'; protocol = json.loads(protocol_path.read_text('utf-8'))
    if (out/'analysis.json').exists() or any((out/'cases').glob('*')):
        raise FileExistsError('Preserve actual normal execution; no overwrite or implicit resume')
    for name,expected in protocol['source_sha256'].items():
        if repair.digest(root/name) != expected:
            raise AssertionError('Frozen source/input changed: '+name)
    for name,expected in protocol['prepared_artifact_sha256'].items():
        if repair.digest(out/name) != expected:
            raise AssertionError('Prepared input changed: '+name)
    snapshots = json.loads((out/'snapshots.json').read_text('utf-8'))
    outcomes = []; hourly = []; windows = []; terminals = []; metrics = []
    started = time.perf_counter()
    for entry in protocol['cases']:
        snapshot = snapshots[entry['snapshot_id']]
        record = run_case(root,out,entry,snapshot)
        normal = pd.read_csv(out/'cases'/entry['case_id']/'actions_states.csv') if record['complete'] else None
        stress,original = load_stress(root,entry)
        pair_hourly,pair_windows = paired_diagnostics(entry,snapshot,stress,normal,original)
        hourly.extend(pair_hourly); windows.extend(pair_windows)
        terminals.extend(terminal_rows(entry,snapshot,stress,normal))
        input_prefix = 'inputs/'+entry['snapshot_id']+'/'
        record['prepared_input_sha256'] = {name:value for name,value in protocol['prepared_artifact_sha256'].items() if name.startswith(input_prefix)}
        stress_result = json.loads((root/SOURCE/'cases'/entry['stress_case_id']/'result.json').read_text('utf-8'))
        difference = {key:float(stress_result['metrics'][key]-record['metrics'][key]) if record['complete'] else None
                      for key in ('full_h0_359_unserved_gwh','primary_h48_359_unserved_gwh',
                                  'physical_operating_cost_million_cny','total_co2_mt')}
        record['stress_minus_normal_metric_differences'] = difference
        if (out/'cases'/entry['case_id']).exists():
            repair.write_json(out/'cases'/entry['case_id']/'result.json',record)
        outcomes.append(record)
        metrics.append({**{key:record[key] for key in ('case_id','stress_case_id','snapshot_id','plan','year','strategy','status','complete')},
                        **record.get('metrics',{}),**{'stress_minus_normal_'+key:value for key,value in difference.items()}})
        print(json.dumps({key:record.get(key) for key in ('case_id','status','executed_hours','successful_lp_hours','error')},ensure_ascii=False),flush=True)
    assert len(hourly)==17280 and len(windows)==96 and len(terminals)==48 and len(outcomes)==16
    pd.DataFrame(hourly).to_csv(out/'paired_hourly_stock_references.csv',index=False,float_format='%.17g')
    pd.DataFrame(windows).to_csv(out/'paired_window_stock_references.csv',index=False,float_format='%.17g')
    pd.DataFrame(terminals).to_csv(out/'paired_terminal_stocks.csv',index=False,float_format='%.17g')
    pd.DataFrame(metrics).to_csv(out/'normal_case_metrics.csv',index=False,float_format='%.17g')
    # JSON retains explicit nulls and reasons, including undefined failed pairs.
    repair.write_json(out/'paired_window_stock_references.json',windows)
    repair.write_json(out/'paired_terminal_stocks.json',terminals)
    drift = {name:repair.digest(root/name) for name,expected in protocol['source_sha256'].items() if repair.digest(root/name) != expected}
    complete = all(row['complete'] for row in outcomes)
    successful = sum(row['successful_lp_hours'] for row in outcomes)
    if complete:
        assert successful == 5760
    report = dict(status='COMPLETED_SIXTEEN_VALIDATED_MATCHED_NORMAL_PATHS' if complete and not drift else 'FAILED_FIXED_SIXTEEN_NORMAL_PATH_COHORT',
        protocol_sha256=repair.digest(protocol_path),protocol=protocol,cases=outcomes,case_count=16,
        validated_complete_cases=sum(row['complete'] for row in outcomes),planned_lp_hours=5760,
        actual_successful_lp_hours=successful,actual_attempted_lp_hours=sum(row['attempted_lp_hours'] for row in outcomes),
        actual_complete_trajectory_hours=sum(row.get('executed_hours',0) for row in outcomes),
        hourly_fuel_cells=len(hourly),window_fuel_cells=len(windows),terminal_fuel_cells=len(terminals),
        complete_matched_case_count=sum(row['complete'] for row in outcomes),source_drift=drift,
        no_normal_event_recovery_defined=True,old_recovery_decisions_unchanged=True,new_native_runs=0,
        wall_seconds=time.perf_counter()-started,argv=sys.argv,
        output_sha256={path.relative_to(out).as_posix():repair.digest(path) for path in out.rglob('*')
                       if path.is_file() and path.name != 'analysis.json'})
    repair.write_json(out/'analysis.json',report)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true'); mode.add_argument('--run',action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
