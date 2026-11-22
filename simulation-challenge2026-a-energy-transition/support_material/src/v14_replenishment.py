"""Frozen annual assets x original paid V10 replenishment, without retuning.

The original calendar-based emergency delivery design is copied explicitly.
Normal inherited-policy paths are resource targets, not same-contract-policy
counterfactuals. Preparation runs no hourly LP or native simulation.
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
from src.v10_prepare_inputs import configuration
from src.v11_recovery_sensitivity import recovery_windows
from src.v13_event_information import independent_audit, _enrich_native
from src.v13_inventory_reference import ratio

ROOT = Path(__file__).resolve().parents[1]
OUT = 'outputs/v14/replenishment'
REPAIR = 'outputs/v13/repair_operations'
NORMAL = 'outputs/v13/inventory_reference'
PLANS = ('knee','repair_005','repair_006','repair_007')
HOURS = 360


def case_grid():
    return [dict(case_id=f'{plan}_2060_{policy}_coordinated',snapshot_id=f'{plan}_2060',
                 inherited_case_id=f'{plan}_2060_coordinated',
                 inherited_normal_case_id=f'{plan}_2060_coordinated_normal',
                 plan=plan,year=2060,strategy='coordinated',supply_policy=policy,hours=HOURS,
                 requires_bridge=(policy=='inherited' or plan=='knee'))
            for policy in ('inherited','contract') for plan in PLANS]


def make_case(snapshot,policy):
    if policy not in ('inherited','contract'):
        raise ValueError('Only the two original supply policies are frozen')
    case = repair.make_case_from_snapshot(snapshot)
    config = configuration(supply_policy=policy)
    cap = np.array([snapshot['fuel_import_capacity_mtce_per_year'][fuel]/8760 for fuel in h.FUELS])
    emergency = np.zeros((HOURS,3)); schedule = []
    for event,(first,last) in enumerate(((108,192),(252,360)),1):
        channel = cap*.5*config['contract_multiplier']
        allocation = case['initial_fuel_stocks_mtce']*.5*config['contract_multiplier']/(last-first)
        rate = np.minimum(channel,allocation) if policy=='contract' else np.zeros(3)
        emergency[first:last] += rate
        for j,fuel in enumerate(h.FUELS):
            schedule.append(dict(event=event,fuel=fuel,delivery_first_hour=first,delivery_end_exclusive=last,
                delivery_hours=last-first,contract_multiplier=1.,contract_lead_hours=12,
                channel_surge_limit_mtce_per_hour=float(channel[j]),
                initial_stock_allocation_limit_mtce_per_hour=float(allocation[j]),
                actual_fixed_contract_rate_mtce_per_hour=float(rate[j]),
                planned_total_delivery_mtce=float(rate[j]*(last-first)),
                active_rate_limit=('NO_CONTRACT' if policy=='inherited' else
                                   'NOMINAL_CHANNEL_50PCT_SURGE' if channel[j]<=allocation[j] else 'INITIAL_STOCK_50PCT_ALLOCATION')))
    case['emergency_arrivals_mtce'] = emergency
    case['fuel_arrivals_mtce'] = case['fuel_domestic_arrivals_mtce']+case['fuel_import_arrivals_mtce']+emergency
    case.update(v10_config=config,supply_policy=policy,contract_schedule=schedule,
        nominal_import_channel_mtce_per_hour=dict(zip(h.FUELS,cap.tolist())),
        evidence_class='EXPLORATORY_FIXED_ANNUAL_ASSET_AND_ORIGINAL_PAID_REPLENISHMENT_COMBINATION',
        contract_information='Externally predetermined scenario calendar; order/delivery windows are not triggered by realized ENS. Dispatch still sees the original common24h forecasts.',
        contract_supply_assumption='Additional external emergency deliveries, up to50% of nominal annual import-channel capacity per hour, on top of inherited imports. No new transport CAPEX or total arrival-capacity constraint is modeled.',
        contract_cash_assumption='Explicit operating procurement liability at original unit values plus50% premium. No financing/budget decision or deduction from annual asset-investment budget.',
        contract_premium_fraction=.5)
    return case


def flow_ledger(case):
    rows = []
    for hour in range(HOURS):
        for j,fuel in enumerate(h.FUELS):
            price = case['config']['fuel_value_million_cny_mtce'][fuel]
            extra = float(case['emergency_arrivals_mtce'][hour,j])
            base = extra*price
            rows.append(dict(hour=hour,fuel=fuel,supply_policy=case['supply_policy'],
                domestic_arrival_mtce=float(case['fuel_domestic_arrivals_mtce'][hour,j]),
                inherited_import_arrival_mtce=float(case['fuel_import_arrivals_mtce'][hour,j]),
                contract_emergency_arrival_mtce=extra,
                total_import_arrival_mtce=float(case['fuel_import_arrivals_mtce'][hour,j]+extra),
                total_fuel_arrival_mtce=float(case['fuel_arrivals_mtce'][hour,j]),
                nominal_import_channel_mtce_per_hour=case['nominal_import_channel_mtce_per_hour'][fuel],
                engineering_unit_value_million_cny_per_mtce=price,
                contract_base_value_million_cny=base,contract_premium_million_cny=.5*base,
                contract_cash_million_cny=1.5*base))
    return pd.DataFrame(rows)


def metrics(case,table,policy):
    result = repair.metrics(case,table,'coordinated')
    base = float(case['emergency_arrivals_mtce'].sum(axis=0) @ np.array([
        case['config']['fuel_value_million_cny_mtce'][fuel] for fuel in h.FUELS]))
    premium = .5*base
    resource = result['resource_accounted_cost_million_cny']-result['shortage_penalty_million_cny']-result['carbon_cost_million_cny']
    result.update(supply_policy=policy,contract_incremental_base_procurement_value_million_cny=base,
        contract_premium_million_cny=premium,contract_incremental_procurement_cash_million_cny=base+premium,
        inherited_fuel_arrival_procurement_value_million_cny=result['fuel_arrival_purchase_value_million_cny']-base,
        total_fuel_procurement_cash_including_premium_million_cny=result['fuel_arrival_purchase_value_million_cny']+premium,
        resource_cost_excluding_shortage_carbon_and_holding_million_cny=resource,
        physical_operating_cost_with_contract_premium_million_cny=result['physical_operating_cost_million_cny']+premium,
        penalty_inclusive_cost_with_contract_premium_million_cny=result['physical_operating_cost_million_cny']+premium+result['shortage_penalty_million_cny']+result['carbon_cost_million_cny'],
        python_legacy_resource_accounted_cost_scope='Original Python field includes shortage penalty and carbon charge; use explicit resource_cost_excluding_shortage_carbon_and_holding for comparison with Julia resource cost.',
        procurement_cash_scope='Fuel purchase cash is not resource consumption: terminal fuel assets reconcile procurement. Do not add incremental contract base cash again to physical cost.',
        contract_information=case['contract_information'],contract_supply_assumption=case['contract_supply_assumption'],
        contract_cash_assumption=case['contract_cash_assumption'])
    for j,fuel in enumerate(h.FUELS):
        result[fuel+'_contract_extra_delivery_mtce'] = float(case['emergency_arrivals_mtce'][:,j].sum())
        result[fuel+'_contract_peak_rate_mtce_per_hour'] = float(case['emergency_arrivals_mtce'][:,j].max())
        result[fuel+'_peak_total_import_rate_mtce_per_hour'] = float((case['fuel_import_arrivals_mtce'][:,j]+case['emergency_arrivals_mtce'][:,j]).max())
        result[fuel+'_peak_total_import_over_nominal_channel_hourly_rate'] = float((case['fuel_import_arrivals_mtce'][:,j]+case['emergency_arrivals_mtce'][:,j]).max()/case['nominal_import_channel_mtce_per_hour'][fuel])
    return result


def reference_bridge(root,plan,policy,case,table):
    root = Path(root)
    if policy=='inherited':
        source = root/REPAIR/'cases'/f'{plan}_2060_coordinated'/'actions_states.csv'
    elif plan=='knee':
        source = root/'outputs/v11/solver_paths/python_contract.csv'
    else:
        raise ValueError('Only four inherited paths and the existing knee contract have old reference paths')
    old = pd.read_csv(source)
    columns = old.select_dtypes(include=np.number).columns.tolist()
    if len(table)!=HOURS or table.hour.tolist()!=list(range(HOURS)):
        return dict(passed=False,reason='Complete ordered360h path required')
    errors = {name:float(np.max(np.abs(table[name].to_numpy()-old[name].to_numpy()))) for name in columns}
    same = all(np.allclose(table[name],old[name],rtol=1e-11,atol=1e-7) for name in columns)
    report = dict(passed=bool(same),reference=source.relative_to(root).as_posix(),
                  complete_numeric_path_gate=dict(passed=bool(same),atol=1e-7,rtol=1e-11,absolute_max_errors=errors))
    if policy=='contract':
        native = _enrich_native(case,pd.read_csv(root/'outputs/v10/anymath/cloud_final/contract_actions_states.csv'))
        local_summary,native_summary = metrics(case,table,policy),metrics(case,native,policy)
        keys = ('full_h0_359_unserved_gwh','primary_h48_359_unserved_gwh','peak_unserved_gw','direct_fuel_unserved_mtce')
        differences = {key:local_summary[key]-native_summary[key] for key in keys}
        recovery = repair.recovery_difference(recovery_windows(table,repair.recovery_config()),recovery_windows(native,repair.recovery_config()))
        native_original = pd.read_csv(root/'outputs/v10/anymath/cloud_final/contract_summary.csv').iloc[0]
        native_cost_error = native_summary['physical_operating_cost_with_contract_premium_million_cny']-native_original.physical_operating_cost_with_contract_premium_million_cny
        native_pass = all(abs(value)<=1e-6 for value in differences.values()) and not recovery and abs(native_cost_error)<=1e-6
        report.update(passed=bool(same and native_pass),native_business_gate=dict(passed=native_pass,
            service_differences=differences,recovery_mismatches=recovery,
            recomputed_native_physical_cost_minus_original_summary_million_cny=float(native_cost_error)),
            local_minus_native_observed_state_cost_differences={key:float(local_summary[key]-native_summary[key])
                for key in ('coal_terminal_stock_mtce','oil_terminal_stock_mtce','gas_terminal_stock_mtce',
                            'physical_operating_cost_with_contract_premium_million_cny','total_co2_mt')},
            difference_scope='Report observed backend state/cost differences; do not infer their cause or assert identical native states.')
    return report


def inventory_diagnostics(entry,snapshot,table,normal):
    """C is the inherited-policy no-event target, not a contract counterfactual."""
    identity = {key:entry[key] for key in ('case_id','snapshot_id','plan','year','strategy','supply_policy')}
    hourly,windows = [],[]
    for hour in range(HOURS):
        event = 2 if hour>=192 else 1 if hour>=48 else None
        for fuel in h.FUELS:
            origin = float(snapshot['fuel_initial_stocks_mtce'][fuel])
            actual = float(table.loc[hour,fuel+'_stock_mtce']) if table is not None else None
            target = float(normal.loc[hour,fuel+'_stock_mtce'])
            pre = float(table.loc[191 if event==2 else 47,fuel+'_stock_mtce']) if event and table is not None else None
            c,status = ratio(actual,target) if actual is not None else (None,'TRAJECTORY_UNAVAILABLE')
            hourly.append(dict(identity,hour=hour,fuel=fuel,event_reference=event,t0_stock_mtce=origin,
                stock_mtce=actual,original_event_prestock_mtce=pre,
                A_stock_over_own_preevent=ratio(actual,pre)[0] if pre is not None else None,
                B_stock_over_t0=ratio(actual,origin)[0] if actual is not None else None,
                inherited_normal_resource_target_stock_mtce=target,
                C_stock_over_inherited_normal_resource_target=c,C_reference_status=status,
                C_reference_policy='inherited',C_reference_role='ORIGINAL_NO_EVENT_RESOURCE_TARGET',
                C_is_same_policy_counterfactual=(entry['supply_policy']=='inherited'),
                stock_minus_inherited_normal_target_mtce=actual-target if actual is not None else None,
                gap_to_95pct_t0_mtce=max(0.,.95*origin-actual) if actual is not None else None,
                gap_to_100pct_t0_mtce=max(0.,origin-actual) if actual is not None else None))
    for event,first,last in ((1,168,192),(2,336,360)):
        for fuel in h.FUELS:
            part = [row for row in hourly if row['fuel']==fuel and first<=row['hour']<last]
            def minimum(key):
                values = [row[key] for row in part]
                return float(min(values)) if all(value is not None for value in values) else None
            end = part[-1]
            windows.append(dict(identity,event=event,fuel=fuel,window_first_hour=first,window_end_exclusive=last,
                A_min_stock_over_own_preevent=minimum('A_stock_over_own_preevent'),
                B_min_stock_over_t0=minimum('B_stock_over_t0'),
                C_min_stock_over_inherited_normal_resource_target=minimum('C_stock_over_inherited_normal_resource_target'),
                C_end_stock_over_inherited_normal_resource_target=end['C_stock_over_inherited_normal_resource_target'],
                C_nonpositive_target_hours=sum(row['inherited_normal_resource_target_stock_mtce']<=0 for row in part),
                C_reference_status=('TRAJECTORY_UNAVAILABLE' if table is None else
                    'NONPOSITIVE_TARGET_IN_WINDOW' if any(row['inherited_normal_resource_target_stock_mtce']<=0 for row in part) else 'VALID'),
                C_reference_policy='inherited',C_reference_role='ORIGINAL_NO_EVENT_RESOURCE_TARGET',
                C_is_same_policy_counterfactual=(entry['supply_policy']=='inherited'),
                min_stock_minus_inherited_normal_target_mtce=minimum('stock_minus_inherited_normal_target_mtce'),
                end_stock_minus_inherited_normal_target_mtce=end['stock_minus_inherited_normal_target_mtce'],
                min_stock_mtce=minimum('stock_mtce'),end_stock_mtce=end['stock_mtce'],
                t0_stock_mtce=end['t0_stock_mtce'],end_gap_to_95pct_t0_mtce=end['gap_to_95pct_t0_mtce'],
                end_gap_to_100pct_t0_mtce=end['gap_to_100pct_t0_mtce']))
    return hourly,windows


def source_names():
    names = ['src/v14_replenishment.py','tests/test_v14_replenishment.py','src/v3_hourly.py','src/v4_hourly.py',
             'src/v10_prepare_inputs.py','src/v11_recovery_sensitivity.py','src/v13_repair_operations.py',
             'src/v13_inventory_reference.py','src/v13_event_information.py',
             REPAIR+'/analysis.json',REPAIR+'/protocol.json',REPAIR+'/snapshots.json',REPAIR+'/annual_source_audit.json',
             NORMAL+'/analysis.json','outputs/v11/solver_paths/analysis.json','outputs/v11/solver_paths/python_contract.csv',
             'outputs/v10/anymath/execution_receipt.json']
    for entry in case_grid()[:4]:
        names += [REPAIR+'/cases/'+entry['inherited_case_id']+'/'+name for name in ('actions_states.csv','result.json','recovery.csv')]
        names += [NORMAL+'/cases/'+entry['inherited_normal_case_id']+'/'+name for name in ('actions_states.csv','result.json')]
        names += [REPAIR+'/inputs/'+entry['snapshot_id']+'/initial_state.json']
    names += ['outputs/v10/anymath/cloud_final/contract_'+suffix+'.csv' for suffix in ('actions_states','summary','recovery','config')]
    return names


def prepare(root=ROOT):
    root = Path(root).resolve(); out = root/OUT
    if (out/'analysis.json').exists() or any((out/'cases').glob('*')):
        raise FileExistsError('Preserve actual V14 execution')
    out.mkdir(parents=True,exist_ok=True)
    before = {name:repair.digest(root/name) for name in source_names()}
    snapshots = json.loads((root/REPAIR/'snapshots.json').read_text('utf-8'))
    receipts = {name:json.loads((root/name/'analysis.json').read_text('utf-8')) for name in (REPAIR,NORMAL)}
    if receipts[REPAIR]['status']!='COMPLETED_SIXTEEN_VALIDATED_360H_PATHS' or receipts[NORMAL]['status']!='COMPLETED_SIXTEEN_VALIDATED_MATCHED_NORMAL_PATHS':
        raise AssertionError('Complete frozen V13 inherited and no-event paths required')
    native_receipt = json.loads((root/'outputs/v10/anymath/execution_receipt.json').read_text('utf-8'))
    local_receipt = json.loads((root/'outputs/v11/solver_paths/analysis.json').read_text('utf-8'))
    for suffix in ('actions_states','summary','recovery','config'):
        key = 'outputs/v10/anymath/cloud_final/contract_'+suffix+'.csv'
        if repair.digest(root/key)!=native_receipt['artifact_sha256'][key]:
            raise AssertionError('Original native contract hash changed')
    if repair.digest(root/'outputs/v11/solver_paths/python_contract.csv')!=local_receipt['output_sha256']['python_contract.csv']:
        raise AssertionError('Original V11 contract reference changed')
    bridges = {}; commitments = []
    for entry in case_grid():
        snapshot = snapshots[entry['snapshot_id']]
        case = make_case(snapshot,entry['supply_policy'])
        expected_initial = json.loads((root/REPAIR/'inputs'/entry['snapshot_id']/'initial_state.json').read_text('utf-8'))
        actual_initial = repair.json_value(repair.physical.initial_state(case))
        if actual_initial!=expected_initial:
            raise AssertionError('Asset/state initialization changed: '+entry['case_id'])
        target = out/'inputs'/entry['case_id']; target.mkdir(parents=True,exist_ok=True)
        repair.write_json(target/'snapshot.json',snapshot)
        repair.write_json(target/'initial_state.json',actual_initial)
        repair.write_json(target/'initial_and_parameters.json',{key:value for key,value in case.items() if not isinstance(value,np.ndarray)})
        external = repair.external_frame(case)
        for j,fuel in enumerate(h.FUELS):
            external[fuel+'_contract_emergency_arrival_mtce'] = case['emergency_arrivals_mtce'][:,j]
            external[fuel+'_total_import_arrival_mtce'] = case['fuel_import_arrivals_mtce'][:,j]+case['emergency_arrivals_mtce'][:,j]
        external.to_csv(target/'external_inputs.csv',index=False,float_format='%.17g')
        ledger = flow_ledger(case); ledger.to_csv(target/'planned_flow_and_cash.csv',index=False,float_format='%.17g')
        pd.DataFrame(case['contract_schedule']).to_csv(target/'contract_schedule.csv',index=False,float_format='%.17g')
        commitments.append(dict(entry,planned_extra_mtce={fuel:float(case['emergency_arrivals_mtce'][:,j].sum()) for j,fuel in enumerate(h.FUELS)},
            planned_base_value_million_cny=float(ledger.contract_base_value_million_cny.sum()),
            planned_premium_million_cny=float(ledger.contract_premium_million_cny.sum()),
            planned_incremental_cash_million_cny=float(ledger.contract_cash_million_cny.sum())))
        if entry['requires_bridge']:
            path = (root/REPAIR/'cases'/entry['inherited_case_id']/'actions_states.csv' if entry['supply_policy']=='inherited'
                    else root/'outputs/v11/solver_paths/python_contract.csv')
            old = pd.read_csv(path); audit = h.audit(case,old)
            _,independent = independent_audit(case,old)
            comparison = reference_bridge(root,entry['plan'],entry['supply_policy'],case,old)
            if not audit['passed'] or not independent['passed'] or not comparison['passed']:
                raise AssertionError('Old observed path does not match prepared inputs: '+entry['case_id'])
            bridges[entry['case_id']] = dict(physical_audit=audit,independent_physical_audit=independent,reference=comparison)
    for entry in case_grid()[:4]:
        for folder,case_id,names in ((REPAIR,entry['inherited_case_id'],('actions_states.csv','result.json','recovery.csv')),
                                     (NORMAL,entry['inherited_normal_case_id'],('actions_states.csv','result.json'))):
            for name in names:
                key = 'cases/'+case_id+'/'+name
                if repair.digest(root/folder/key)!=receipts[folder]['output_sha256'][key]:
                    raise AssertionError('Frozen V13 reference changed: '+folder+'/'+key)
    repair.write_json(out/'snapshots.json',{key:value for key,value in snapshots.items() if key.endswith('_2060')})
    pd.DataFrame(case_grid()).to_csv(out/'case_grid.csv',index=False)
    repair.write_json(out/'planned_contract_commitments.json',commitments)
    after = {name:repair.digest(root/name) for name in before}
    if before!=after:
        raise AssertionError('Frozen source/input drift during preparation')
    artifacts = {path.relative_to(out).as_posix():repair.digest(path) for path in out.rglob('*')
                 if path.is_file() and path.name not in ('protocol.json','preparation_audit.json','code_review.json')}
    protocol = dict(status='PREPARED_NOT_RUN',prepared_utc=datetime.now(timezone.utc).isoformat(),
        analysis_class='EXPLORATORY_FIXED_FOUR_ANNUAL_ASSETS_X_ORIGINAL_TWO_REPLENISHMENT_POLICIES',
        cases=case_grid(),case_count=8,hours_per_case=HOURS,planned_lp_hours=2880,
        source_sha256=before,prepared_artifact_sha256=artifacts,
        command='python -B -X utf8 -m src.v14_replenishment --run',
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
        fixed_design='Four frozen2060 snapshots; original360h events[48,96),[192,240),severity1; coordinated24h true forecast,storage power1,SOC50%,thermal0,debt0; no new search/tuning/independent48 access.',
        replenishment_rule='Original V10: contract_multiplier1,lead12,premium_fraction0.5. Predetermined calendar deliveries[108,192)84h and[252,360)108h; rate=min(channel_annual/8760*0.5,initial_stock*0.5/window_hours), separately per fuel/window. No realized ENS trigger.',
        information='Contract delivery design uses the externally supplied full event/recovery calendar. Dispatch still optimizes only common24h true engineering forecasts. This is not a policy proven feasible with only contemporaneous event observations.',
        supply_origin='Declared additional external emergency fuel deliveries. Total delivery can exceed nominal channel/hour baseline:coal150% for knee/005/006,coal123.0927% for007,andgas150% for allfour. No new supplier,transport construction or financing has been executed;50% premium represents the existing engineering contract assumption, not evidence that logistics capacity was built.',
        cash_boundary='Operating procurement liability at original engineering fuel unit values: base cash+50% premium. No financing constraint and no subtraction from annual asset-investment budget. Keep base/premium/cash and terminal-asset-adjusted resource cost distinct; do not add base cash twice to physical loss.',
        annual_capital='Frozen2025..2060 annual capital,discount4%/origin2024,paid capital and2060spending remain separate from360h procurement/physical operating cost. No new transport CAPEX is estimated.',
        reference_target_C='Read original V13 inherited-policy no-event path at same asset/year/hour only as ORIGINAL_NO_EVENT_RESOURCE_TARGET. A contract path over that target is not a same-contract-policy counterfactual. No four additional normal-policy cases are added.',
        inventory='Retain own-event-prestock A,initialt0 B,inherited-normal resource-target C,actual signed stock differences and95/100%t0 quantity gaps. Nonpositive reference denominator=>null+reason,never epsilon. Original95%24h service+stock recovery definition remains unchanged.',
        bridge_gate='First4 inherited full numeric paths must reproduce frozen V13; fifth knee contract must reproduce frozen V11 Python full numeric path and original native ENS/peak/direct shortage/recovery, plus independent native cost-accounting identity. Only after all5 pass may final3contract cases run.',
        failure_rule='Keep all8 fixed rows. First5 are attempted even if a prior bridge fails. If any bridge/solve/audit fails,final3 remain NOT_RUN_BRIDGE_FAILED with explicit reason. After bridge success,all3newcontract cases are attempted even if one fails. Preserve original error plus separate prefix-evidence error; never relax/relabel failures.',
        claim='A fixed designed combination experiment. Successful delivery/accounting is not measured national supply, financing readiness or new native execution.')
    repair.write_json(out/'protocol.json',protocol)
    audit = dict(status='PASS_PREPARED_NOT_RUN',new_hourly_lp_calls=0,planned_cases=8,planned_lp_hours=2880,
        exact_initial_states_preserved=True,readonly_old_path_checks=bridges,
        protocol_sha256=repair.digest(out/'protocol.json'),source_hashes_unchanged=True)
    repair.write_json(out/'preparation_audit.json',audit)
    print(json.dumps({key:audit[key] for key in ('status','new_hourly_lp_calls','planned_cases','protocol_sha256')},ensure_ascii=False),flush=True)
    return audit


def run_case(root,out,entry,snapshot,annual_evidence):
    target = Path(out)/'cases'/entry['case_id']; target.mkdir(parents=True)
    case = make_case(snapshot,entry['supply_policy']); calls=[]; started=time.perf_counter()
    record = dict(entry,status='RUNNING',complete=False,started_utc=datetime.now(timezone.utc).isoformat())
    try:
        with (target/'actual_pre_states_and_actions.jsonl').open('w',encoding='utf-8') as stream:
            result = repair.simulate_recorded(case,'coordinated',calls,stream)
        result['timeseries'].to_csv(target/'actions_states.csv',index=False,float_format='%.17g')
        table = pd.read_csv(target/'actions_states.csv'); replay=repair.prefix_from_calls(case,calls)
        np.testing.assert_allclose(table.to_numpy(float),replay[table.columns].to_numpy(float),rtol=0,atol=1e-9)
        ordinary=h.audit(case,table); residual,independent=independent_audit(case,table)
        residual.to_csv(target/'independent_physical_audit.csv',index=False,float_format='%.17g')
        repair.write_json(target/'physical_audit.json',ordinary)
        repair.write_json(target/'independent_physical_audit.json',independent)
        if not ordinary['passed'] or not independent['passed']:
            raise AssertionError('Actual exported path failed physical accounting')
        summary=metrics(case,table,entry['supply_policy'])
        summary.update(annual_evidence[entry['plan']]['capital_by_snapshot_year']['2060'])
        summary['annual2060_nominal_net_emissions_mt']=annual_evidence[entry['plan']]['nominal_summary']['net_emissions_2060_mt']
        recovery=recovery_windows(table,repair.recovery_config())
        repair.write_json(target/'summary.json',summary)
        pd.DataFrame(recovery).to_csv(target/'recovery.csv',index=False)
        record.update(metrics=summary,recovery=recovery,executed_hours=len(table),physical_audit=ordinary,
                      independent_physical_audit=independent)
        if entry['requires_bridge']:
            bridge=reference_bridge(root,entry['plan'],entry['supply_policy'],case,table)
            repair.write_json(target/'reference_bridge.json',bridge); record['reference_bridge']=bridge
            if not bridge['passed']:
                raise AssertionError('Actual full numeric/reference business bridge failed')
        record.update(status='COMPLETED_AND_VALIDATED',complete=True)
    except Exception as failure:
        record.update(status='FAILED',error=str(failure),error_type=type(failure).__name__,traceback=traceback.format_exc(),
            failure_hour=calls[-1]['hour'] if calls and 'error' in calls[-1] else None,
            last_attempted_hour=calls[-1]['hour'] if calls else None)
        if not (target/'actions_states.csv').exists():
            try:
                prefix=repair.prefix_from_calls(case,calls); record['executed_prefix_hours']=len(prefix)
                if len(prefix):
                    prefix.to_csv(target/'observed_prefix_actions_states.csv',index=False,float_format='%.17g')
                    residual,independent=independent_audit(case,prefix,complete=False)
                    residual.to_csv(target/'prefix_independent_physical_audit.csv',index=False,float_format='%.17g')
                    record['prefix_physical_audit']=repair.physical.prefix_audit(case,prefix)
                    record['prefix_independent_physical_audit']=independent
            except Exception as evidence_failure:
                record['prefix_evidence_error']=dict(error=str(evidence_failure),error_type=type(evidence_failure).__name__,traceback=traceback.format_exc())
    finally:
        record.update(successful_lp_hours=sum('action' in call for call in calls),attempted_lp_hours=len(calls),
                      wall_seconds=time.perf_counter()-started,completed_utc=datetime.now(timezone.utc).isoformat())
        repair.write_json(target/'result.json',record)
    return record


def run(root=ROOT):
    root=Path(root).resolve(); out=root/OUT
    protocol_path=out/'protocol.json'; protocol=json.loads(protocol_path.read_text('utf-8'))
    if (out/'analysis.json').exists() or any((out/'cases').glob('*')):
        raise FileExistsError('Preserve actual V14 execution; no implicit overwrite or resume')
    for name,expected in protocol['source_sha256'].items():
        if repair.digest(root/name)!=expected:
            raise AssertionError('Frozen source/input changed: '+name)
    for name,expected in protocol['prepared_artifact_sha256'].items():
        if repair.digest(out/name)!=expected:
            raise AssertionError('Prepared artifact changed: '+name)
    snapshots=json.loads((out/'snapshots.json').read_text('utf-8'))
    annual=json.loads((root/REPAIR/'annual_source_audit.json').read_text('utf-8'))
    outcomes=[]; hourly=[]; windows=[]; started=time.perf_counter()
    for index,entry in enumerate(protocol['cases']):
        if index>=5 and not all(row['complete'] and row.get('reference_bridge',{}).get('passed',False) for row in outcomes[:5]):
            record=dict(entry,status='NOT_RUN_BRIDGE_FAILED',complete=False,successful_lp_hours=0,attempted_lp_hours=0,
                        reason='One or more of the five mandatory fresh bridges did not pass')
        else:
            record=run_case(root,out,entry,snapshots[entry['snapshot_id']],annual)
        table=pd.read_csv(out/'cases'/entry['case_id']/'actions_states.csv') if record['complete'] else None
        normal=pd.read_csv(root/NORMAL/'cases'/entry['inherited_normal_case_id']/'actions_states.csv')
        cells,window_cells=inventory_diagnostics(entry,snapshots[entry['snapshot_id']],table,normal)
        hourly.extend(cells); windows.extend(window_cells)
        record['prepared_input_sha256']={name:value for name,value in protocol['prepared_artifact_sha256'].items()
                                       if name.startswith('inputs/'+entry['case_id']+'/')}
        if (out/'cases'/entry['case_id']).exists():
            repair.write_json(out/'cases'/entry['case_id']/'result.json',record)
        outcomes.append(record)
        print(json.dumps({key:record.get(key) for key in ('case_id','status','executed_hours','successful_lp_hours','error')},ensure_ascii=False),flush=True)
    assert len(outcomes)==8 and len(hourly)==8640 and len(windows)==48
    rows=[{**{key:row[key] for key in ('case_id','snapshot_id','plan','year','supply_policy','status','complete')},**row.get('metrics',{})} for row in outcomes]
    pd.DataFrame(rows).to_csv(out/'all_case_metrics.csv',index=False,float_format='%.17g')
    pd.DataFrame(hourly).to_csv(out/'hourly_inventory_reference_targets.csv',index=False,float_format='%.17g')
    pd.DataFrame(windows).to_csv(out/'window_inventory_reference_targets.csv',index=False,float_format='%.17g')
    repair.write_json(out/'window_inventory_reference_targets.json',windows)
    complete=all(row['complete'] for row in outcomes); successful=sum(row['successful_lp_hours'] for row in outcomes)
    if complete:
        assert successful==2880
    contrasts=[]
    by_id={row['case_id']:row for row in outcomes}
    for plan in PLANS:
        a=by_id[f'{plan}_2060_inherited_coordinated']; b=by_id[f'{plan}_2060_contract_coordinated']
        contrasts.append(dict(plan=plan,contrast='contract minus inherited',complete=a['complete'] and b['complete'],
            differences={key:float(b['metrics'][key]-a['metrics'][key]) if a['complete'] and b['complete'] else None
                for key in ('primary_h48_359_unserved_gwh','full_h0_359_unserved_gwh','primary_h48_359_peak_unserved_gw',
                            'physical_operating_cost_with_contract_premium_million_cny','total_co2_mt',
                            'coal_terminal_stock_mtce','oil_terminal_stock_mtce','gas_terminal_stock_mtce')}))
    repair.write_json(out/'within_asset_contract_contrasts.json',contrasts)
    drift={name:repair.digest(root/name) for name,expected in protocol['source_sha256'].items() if repair.digest(root/name)!=expected}
    report=dict(status='COMPLETED_EIGHT_VALIDATED_ASSET_REPLENISHMENT_PATHS' if complete and not drift else 'FAILED_FIXED_EIGHT_REPLENISHMENT_COHORT',
        protocol_sha256=repair.digest(protocol_path),protocol=protocol,cases=outcomes,case_count=8,
        validated_complete_cases=sum(row['complete'] for row in outcomes),planned_lp_hours=2880,
        actual_successful_lp_hours=successful,actual_attempted_lp_hours=sum(row['attempted_lp_hours'] for row in outcomes),
        actual_complete_trajectory_hours=sum(row.get('executed_hours',0) for row in outcomes),
        five_reference_bridges_passed=all(row['complete'] and row.get('reference_bridge',{}).get('passed',False) for row in outcomes[:5]),
        hourly_fuel_cells=len(hourly),window_fuel_cells=len(windows),within_asset_contract_contrasts=contrasts,
        source_drift=drift,new_native_runs=0,new_normal_policy_runs=0,wall_seconds=time.perf_counter()-started,argv=sys.argv,
        output_sha256={path.relative_to(out).as_posix():repair.digest(path) for path in out.rglob('*') if path.is_file() and path.name!='analysis.json'})
    repair.write_json(out/'analysis.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(); mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true'); mode.add_argument('--run',action='store_true')
    args=parser.parse_args(); prepare() if args.prepare else run()
