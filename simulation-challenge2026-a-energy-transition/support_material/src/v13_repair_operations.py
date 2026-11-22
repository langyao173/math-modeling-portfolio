"""Fixed annual repair controls transferred to identical hourly design windows.

The old simulator and first-action physics stay unchanged. Preparation contains
annual nominal reconstructions and old-result checks, but no hourly LP solves.
AI-assisted: OpenAI Codex / GPT-6, 2026-10-19.
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

from src import v3_hourly as h
from src import v4_hourly as physical
from src import v6_annual as annual
from src.v5_planning import decode, digest, summarize
from src.v11_recovery_sensitivity import recovery_windows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/v13/repair_operations'
PLANS = ('knee', 'repair_005', 'repair_006', 'repair_007')
YEARS = (2030, 2060)
STRATEGIES = ('coordinated', 'decentralized')
HOURS = 360
SOURCE_NAMES = (
    'src/v13_repair_operations.py', 'tests/test_v13_repair_operations.py',
    'src/v3_hourly.py', 'src/v4_hourly.py', 'src/v6_annual.py',
    'src/v5_planning.py', 'src/v5_annual.py', 'src/v3_annual.py', 'src/v3_annual_calibrated.py',
    'src/v3_calibration_candidate.py', 'src/v11_recovery_sensitivity.py',
    'config/v3_transition.json', 'config/v6_annual_planning.json',
    'data/annual_china.csv', 'data/ember_power_china.csv',
    'data/supplement_carriers_2024.csv', 'data/energy_production_china.csv',
    'data/fuel_supply_trade.csv', 'data/v6_fuel_import_channels.csv',
    'outputs/v6/resilience/native_inputs.json',
    'outputs/v6/planning/planning_audit.json',
    'outputs/v6/planning/trajectory_plan_02349.csv',
    'outputs/v6/planning/actions_plan_02349.csv',
    'outputs/v11/robust_repair/candidate_scores.csv',
    'outputs/v11/robust_repair/selected_before_independent_test.json',
    'outputs/v12/native/annual_native_audit.json',
    'outputs/v12/native/native_trace.csv', 'outputs/v12/native/native_actions.csv',
    'outputs/v11/solver_paths/analysis.json',
    'outputs/v11/solver_paths/python_base.csv',
    'outputs/v11/solver_paths/python_decentralized.csv',
    'outputs/v10/anymath/execution_receipt.json',
) + tuple(f'outputs/v10/anymath/cloud_final/{arm}_{suffix}.csv'
          for arm in ('base', 'decentralized')
          for suffix in ('actions_states', 'config', 'summary', 'recovery'))


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding='utf-8')


def case_grid():
    # Both old-snapshot strategy bridges precede the fourteen new design paths.
    pairs = [('knee', 2060)] + [(p, y) for p in PLANS for y in YEARS if (p, y) != ('knee', 2060)]
    return [dict(case_id=f'{plan}_{year}_{strategy}', snapshot_id=f'{plan}_{year}',
                 plan=plan, year=year, strategy=strategy, hours=HOURS,
                 bridge_required=(plan == 'knee' and year == 2060))
            for plan, year in pairs for strategy in STRATEGIES]


def compare_nominal(actual, frozen):
    result = {}
    for name, value in actual.items():
        expected = float(frozen['nominal_'+name])
        result[name] = dict(actual=float(value), expected=expected,
                            absolute_error=abs(float(value)-expected),
                            passed=bool(np.isclose(value, expected, rtol=1e-11, atol=1e-7)))
    if not all(row['passed'] for row in result.values()):
        raise AssertionError('Frozen nominal summary changed: '+str(result))
    return result


def build_snapshots(root=ROOT):
    """Read original/native states; reconstruct only two fixed nominal paths."""
    root = Path(root)
    original = json.loads((root/'outputs/v6/resilience/native_inputs.json').read_text('utf-8'))['snapshots']
    scores = pd.read_csv(root/'outputs/v11/robust_repair/candidate_scores.csv').set_index('candidate_id')
    planning = json.loads((root/'config/v6_annual_planning.json').read_text('utf-8'))
    selected = json.loads((root/'outputs/v11/robust_repair/selected_before_independent_test.json').read_text('utf-8'))
    if selected['candidate']['candidate_id'] != 'repair_005':
        raise AssertionError('Original independent-test selection identity changed')
    receipt = json.loads((root/'outputs/v12/native/annual_native_audit.json').read_text('utf-8'))
    if receipt['status'] != 'PASS_ACTUAL_NATIVE_REPAIR_005_217_STATE_AND_PHYSICAL_AUDIT':
        raise AssertionError('Actual repair_005 annual native receipt required')
    for name, key in (('native_trace.csv', 'native_trace_csv'), ('native_actions.csv', 'native_actions_csv')):
        if digest(root/'outputs/v12/native'/name) != receipt['output_hashes'][key]:
            raise AssertionError('Native annual output hash changed: '+name)
    data = annual.load_data(root)
    params = annual.calibrate(data, 2020)
    snapshots = {f'knee_{y}': deepcopy(original[f'plan_02349_{y}']) for y in YEARS}
    old_planning = json.loads((root/'outputs/v6/planning/planning_audit.json').read_text('utf-8'))
    for name in ('trajectory_plan_02349.csv','actions_plan_02349.csv'):
        if digest(root/'outputs/v6/planning'/name) != old_planning['outputs_sha256'][name]:
            raise AssertionError('Original knee annual source hash changed: '+name)
    knee_frame = pd.read_csv(root/'outputs/v6/planning/trajectory_plan_02349.csv')
    knee_actions = pd.read_csv(root/'outputs/v6/planning/actions_plan_02349.csv')
    knee_nominal = summarize(knee_frame,knee_actions,data,planning)
    evidence = {'knee': dict(source_kind='FROZEN_ORIGINAL_SNAPSHOT_JSON',
                            source='outputs/v6/resilience/native_inputs.json',
                            actions_source='outputs/v6/planning/actions_plan_02349.csv',
                            frozen_controls=scores.loc['repair_000', planning['decision_names']].to_dict(),
                            nominal_summary=knee_nominal,
                            nominal_summary_comparison=compare_nominal(knee_nominal,scores.loc['repair_000']),
                            capital_by_snapshot_year=capital_by_year(knee_actions,planning),
                            passed=True)}
    for candidate in PLANS[1:]:
        frozen = scores.loc[candidate].to_dict()
        if candidate == 'repair_005':
            frame = pd.read_csv(root/'outputs/v12/native/native_trace.csv')
            actions = pd.read_csv(root/'outputs/v12/native/native_actions.csv')
            kind = 'ACTUAL_MATLAB_NATIVE_ANNUAL_TRACE'
            source = 'outputs/v12/native/native_trace.csv'
            for name in planning['decision_names']:
                if not np.isclose(frozen[name], selected['candidate'][name], rtol=0, atol=1e-14):
                    raise AssertionError('Frozen repair_005 control differs from actual native selection')
        else:
            vector = np.array([frozen[name] for name in planning['decision_names']], float)
            bounds = np.asarray(planning['bounds'], float)
            if (vector < bounds[:, 0]).any() or (vector > bounds[:, 1]).any():
                raise AssertionError('Frozen repair control outside original bounds')
            frame, actions = annual.simulate(data, params, decode(vector, candidate), investor_ledger=False)
            kind = 'FROZEN_CONTROLS_NOMINAL_RECONSTRUCTION'
            source = f'outputs/v13/repair_operations/annual/{candidate}_trace.csv'
        if len(frame) != 36 or len(actions) != 432 or frame.year.tolist() != list(range(2025,2061)):
            raise AssertionError('Incomplete annual nominal source: '+candidate)
        nominal = summarize(frame, actions, data, planning)
        comparison = compare_nominal(nominal, frozen)
        for year in YEARS:
            row = frame.loc[frame.year == year].iloc[0]
            snapshots[f'{candidate}_{year}'] = annual.snapshot_from_row(row, candidate)
        evidence[candidate] = dict(source_kind=kind, source=source,
                                  actions_source=('outputs/v12/native/native_actions.csv' if candidate == 'repair_005' else f'outputs/v13/repair_operations/annual/{candidate}_actions.csv'),
                                  frozen_controls={name:float(frozen[name]) for name in planning['decision_names']},
                                  nominal_summary=nominal, nominal_summary_comparison=comparison,
                                  capital_by_snapshot_year=capital_by_year(actions,planning),
                                  passed=True, frame=frame, actions=actions)
    return snapshots, evidence


def capital_by_year(actions, planning):
    if planning['discount_origin_year'] != 2024 or planning['discount_rate'] != .04:
        raise AssertionError('Preserve original 2024-origin 4% discount convention')
    result = {}
    for year in YEARS:
        part = actions.loc[actions.year <= year]
        discount = 1.04**(part.year-2024)
        result[str(year)] = dict(annual_nominal_discount_origin_year=2024,
            annual_nominal_discount_rate=.04,
            annual_nominal_discounted_capital_2025_to_snapshot_year_trillion_cny=float((part.capital_spend_million_cny/discount).sum()/1e6),
            annual_nominal_paid_capital_2025_to_snapshot_year_trillion_cny=float(part.capital_spend_million_cny.sum()/1e6),
            annual_nominal_snapshot_year_capital_trillion_cny=float(actions.loc[actions.year == year,'capital_spend_million_cny'].sum()/1e6))
    return result


def make_case_from_snapshot(snapshot):
    """Apply the exact V10 360h event design to this annual end-state."""
    case = h.make_case(snapshot, 'normal', HOURS)
    event = np.zeros(HOURS, dtype=bool)
    event[48:96] = event[192:240] = True
    case['stress_mask'] = event
    # Keep the arithmetic expression used by V10, including its float values.
    case['fuel_import_arrivals_mtce'][event] *= 1-1.*(1-np.array([.50, .05, .05]))
    for tech, retention in (('wind', .25), ('solar', .55), ('hydro', .70)):
        case['available_gw'][event, h.GENERATORS.index(tech)] *= 1-1.*(1-retention)
    case['demand_gw'][event] *= 1.08
    case['fuel_arrivals_mtce'] = case['fuel_domestic_arrivals_mtce']+case['fuel_import_arrivals_mtce']
    case.update(shock='v10_two_events', event_starts=[48,192],
                emergency_arrivals_mtce=np.zeros((HOURS,3)),
                evidence_class='EXPLORATORY_FROZEN_ANNUAL_STATE_HOURLY_DESIGN_WINDOW',
                initialization_contract='Annual end-year stocks/capacity/realized flows; SOC50%, thermal0, debt0 engineering initialization')
    return case


def external_frame(case):
    columns = {'hour':np.arange(HOURS), 'stress_active':case['stress_mask'].astype(int),
               'demand_gw':case['demand_gw']}
    columns.update({g+'_available_gw':case['available_gw'][:,j] for j,g in enumerate(h.GENERATORS)})
    for name, array in (('domestic_arrival',case['fuel_domestic_arrivals_mtce']),
                        ('import_arrival',case['fuel_import_arrivals_mtce']),
                        ('arrival',case['fuel_arrivals_mtce']),
                        ('direct_demand',case['direct_fuel_demand_mtce'])):
        columns.update({f+'_'+name+'_mtce':array[:,j] for j,f in enumerate(h.FUELS)})
    return pd.DataFrame(columns)


def recovery_config():
    return pd.Series(dict(event1_start_hour=48, event2_start_hour=192,
                          event_duration_hours=48, hours=HOURS))


def metrics(case, table, strategy):
    result = physical.physical_summary(case, table, strategy)
    result.update(electric_shortfall_fraction_denominator='Original unshifted demand_gw sum in the stated hour window',
                  supply_policy='inherited',contract_premium_million_cny=0.,
                  information_available_hours=24,used_planning_hours=24)
    for label, part in (('full_h0_359',table), ('warmup_h0_47',table.iloc[:48]),
                        ('primary_h48_359',table.iloc[48:])):
        result[label+'_unserved_gwh'] = float(part.unserved_gw.sum())
        result[label+'_unserved_fraction_of_demand'] = float(part.unserved_gw.sum()/part.demand_gw.sum())
        result[label+'_peak_unserved_gw'] = float(part.unserved_gw.max())
        result[label+'_loss_of_load_hours'] = int((part.unserved_gw > 1e-5).sum())
        for fuel in h.FUELS:
            result[label+'_'+fuel+'_direct_unserved_mtce'] = float(part[fuel+'_direct_unserved_mtce'].sum())
            requested = float(part[fuel+'_direct_demand_mtce'].sum())
            result[label+'_'+fuel+'_direct_demand_mtce'] = requested
            result[label+'_'+fuel+'_direct_unserved_fraction'] = float(part[fuel+'_direct_unserved_mtce'].sum()/requested) if requested else 0.
    return result


def recovery_difference(actual, reference):
    fields = ('event','service_confirmation_hour','joint_confirmation_hour',
              'service_right_censored','joint_right_censored',
              'stock_rebreach_hour_after_joint_confirmation','final_window_joint_ready')
    mismatches = []
    for a,b in zip(actual, reference):
        for field in fields:
            if a[field] != b[field]:
                mismatches.append(dict(event=a['event'],field=field,actual=a[field],expected=b[field]))
    return mismatches


def compare_with_references(root, strategy, case, table):
    root = Path(root)
    arm = 'base' if strategy == 'coordinated' else 'decentralized'
    old = pd.read_csv(root/f'outputs/v11/solver_paths/python_{arm}.csv')
    native = pd.read_csv(root/f'outputs/v10/anymath/cloud_final/{arm}_actions_states.csv')
    # Native exports contain raw actions/states. Derive the two audit-only
    # columns required by the original Python accounting, using the same units.
    for j,fuel in enumerate(h.FUELS):
        native[fuel+'_direct_served_mtce'] = case['direct_fuel_demand_mtce'][:,j]-native[fuel+'_direct_unserved_mtce']
        native[fuel+'_power_burn_mtce'] = native[fuel+'_gw']/(h.GWH_PER_MTCE*case['fuel_efficiency'][fuel]) if fuel in ('coal','gas') else 0.
    columns = old.select_dtypes(include=[np.number]).columns.tolist()
    if len(table) != HOURS or table.hour.tolist() != list(range(HOURS)):
        return dict(passed=False, reason='Complete ordered 360h trace required')
    old_errors = {col:float(np.max(np.abs(table[col].to_numpy()-old[col].to_numpy()))) for col in columns}
    same_path = all(np.allclose(table[col], old[col], rtol=1e-11, atol=1e-7) for col in columns)
    local_metrics, native_metrics = metrics(case,table,strategy), metrics(case,native,strategy)
    service_keys = ('full_h0_359_unserved_gwh','primary_h48_359_unserved_gwh',
                    'primary_h48_359_peak_unserved_gw','peak_unserved_gw','direct_fuel_unserved_mtce')
    differences = {key:local_metrics[key]-native_metrics[key] for key in service_keys}
    recovery = recovery_windows(table,recovery_config())
    native_recovery = recovery_windows(native,recovery_config())
    mismatches = recovery_difference(recovery,native_recovery)
    native_pass = all(abs(value) <= 1e-6 for value in differences.values()) and not mismatches
    numeric_differences = {key:float(value-native_metrics[key]) for key,value in local_metrics.items()
                           if isinstance(value,(int,float,np.number)) and key in native_metrics
                           and isinstance(native_metrics[key],(int,float,np.number))}
    reported_native = pd.read_csv(root/f'outputs/v10/anymath/cloud_final/{arm}_summary.csv').iloc[0]
    native_cost_error = float(native_metrics['physical_operating_cost_million_cny']-reported_native['physical_operating_cost_with_contract_premium_million_cny'])
    return dict(passed=bool(same_path and native_pass and abs(native_cost_error)<=1e-6),
                old_python_complete_numeric_path_gate=dict(passed=same_path,absolute_max_errors=old_errors,
                                                          atol=1e-7,rtol=1e-11),
                native_service_and_recovery_gate=dict(passed=native_pass,metric_differences=differences,
                                                      tolerance=1e-6,recovery_mismatches=mismatches),
                local_minus_native_observed_summary_differences=numeric_differences,
                recomputed_native_physical_cost_minus_original_summary_million_cny=native_cost_error,
                native_physical_cost_reference='Recomputed with v4 physical_summary on native actions; Julia resource_accounted_cost excludes shortage/carbon while Python includes them',
                state_cost_difference_scope='Observed backend differences are reported; service parity does not establish identical states/costs or their cause')


def simulate_recorded(case, strategy, calls, stream=None):
    """Use the unchanged simulator, recording actual states before each solve."""
    original = h._plan
    def recorded(actual, hour, state, selected):
        record = dict(hour=hour, pre_state=deepcopy(state))
        calls.append(record)
        try:
            action = original(actual,hour,state,selected)
            record['action'] = deepcopy(action)
            return action
        except Exception as failure:
            record.update(error=str(failure),error_type=type(failure).__name__)
            raise
        finally:
            if stream is not None:
                stream.write(json.dumps(json_value(record),ensure_ascii=False,allow_nan=False)+'\n')
                stream.flush()
    h._plan = recorded
    try:
        return h.simulate(case,strategy)
    finally:
        h._plan = original


def prefix_from_calls(case, calls):
    state = physical.initial_state(case)
    rows = []
    for call in calls:
        for key in ('fuel','debt'):
            np.testing.assert_allclose(state[key],call['pre_state'][key],rtol=0,atol=1e-9)
        for key in ('thermal','soc','previous_available'):
            for name,value in state[key].items():
                np.testing.assert_allclose(value,call['pre_state'][key][name],rtol=0,atol=1e-9)
        if 'action' not in call:
            break
        rows.append(physical.execute_action(case,call['hour'],state,call['action']))
    return pd.DataFrame(rows)


def prepare(root=ROOT):
    root = Path(root).resolve()
    out = root/'outputs/v13/repair_operations'
    if (out/'analysis.json').exists() or any((out/'cases').glob('*')):
        raise FileExistsError('Scientific execution outputs exist; preserve them')
    out.mkdir(parents=True,exist_ok=True)
    before = {name:digest(root/name) for name in SOURCE_NAMES}
    snapshots,evidence = build_snapshots(root)
    (out/'annual').mkdir(exist_ok=True)
    for candidate in PLANS[1:]:
        item = evidence[candidate]
        frame,actions = item.pop('frame'),item.pop('actions')
        if candidate != 'repair_005':
            frame.to_csv(out/'annual'/f'{candidate}_trace.csv',index=False,float_format='%.17g')
            actions.to_csv(out/'annual'/f'{candidate}_actions.csv',index=False,float_format='%.17g')
    write_json(out/'snapshots.json',snapshots)
    write_json(out/'annual_source_audit.json',evidence)
    pd.DataFrame(case_grid()).to_csv(out/'case_grid.csv',index=False)
    readonly_reference = {}
    for snapshot_id,snap in snapshots.items():
        target = out/'inputs'/snapshot_id
        target.mkdir(parents=True,exist_ok=True)
        case = make_case_from_snapshot(snap)
        write_json(target/'snapshot.json',snap)
        write_json(target/'initial_and_parameters.json',
                   {key:value for key,value in case.items() if not isinstance(value,np.ndarray)})
        write_json(target/'initial_state.json',physical.initial_state(case))
        external_frame(case).to_csv(target/'external_inputs.csv',index=False,float_format='%.17g')
        if snapshot_id == 'knee_2060':
            for strategy in STRATEGIES:
                arm = 'base' if strategy == 'coordinated' else 'decentralized'
                frame = pd.read_csv(root/f'outputs/v11/solver_paths/python_{arm}.csv')
                audit = h.audit(case,frame)
                bridge = compare_with_references(root,strategy,case,frame)
                if not audit['passed'] or not bridge['passed']:
                    raise AssertionError('Existing knee2060 reference does not match prepared case')
                readonly_reference[strategy] = dict(physical_audit=audit,bridge=bridge)
    receipt = json.loads((root/'outputs/v10/anymath/execution_receipt.json').read_text('utf-8'))
    for arm in ('base','decentralized'):
        for suffix in ('actions_states','config','summary','recovery'):
            key = f'outputs/v10/anymath/cloud_final/{arm}_{suffix}.csv'
            if digest(root/key) != receipt['artifact_sha256'][key]:
                raise AssertionError('Frozen native reference hash changed: '+key)
    old_receipt = json.loads((root/'outputs/v11/solver_paths/analysis.json').read_text('utf-8'))
    for arm in ('base','decentralized'):
        if digest(root/f'outputs/v11/solver_paths/python_{arm}.csv') != old_receipt['output_sha256'][f'python_{arm}.csv']:
            raise AssertionError('Frozen V11 Python reference hash changed')
    after = {name:digest(root/name) for name in SOURCE_NAMES}
    if before != after:
        raise AssertionError('Source/input drift during preparation')
    artifacts = {path.relative_to(out).as_posix():digest(path) for path in out.rglob('*')
                 if path.is_file() and path.name not in ('protocol.json','preparation_audit.json','code_review.json')}
    protocol = dict(status='PREPARED_NOT_RUN',prepared_utc=datetime.now(timezone.utc).isoformat(),
        analysis_class='EXPLORATORY_FIXED_FOUR_PLAN_TWO_YEAR_TWO_STRATEGY_COHORT',
        cases=case_grid(),case_count=16,hours_per_case=HOURS,planned_lp_hours=5760,
        source_sha256=before,prepared_artifact_sha256=artifacts,
        command='python -B -X utf8 -m src.v13_repair_operations --run',
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
        design=dict(event_starts=[48,192],event_duration_hours=48,shock_severity=1.,supply_policy='inherited',
                    storage_power_multiplier=1.,forecast_hours=24,information_contract='COMMON_24H_PERFECT_FORECAST_ROLLING_FIRST_ACTION'),
        initialization='Own annual end-state assets/stocks/realized domestic and import flows. Engineering SOC50%, thermal0, flexible debt0. No inventory resets during a path; both strategies share exact snapshot and initial state.',
        interpretation='A designed 360h window at each annual end-state, not the realized chronology inside that year; new hourly results use local SciPy, not a new native execution.',
        primary_metric='ENS GWh at h48..359; report full360/warmup separately, demand-normalized fraction, peak, direct coal/oil/gas shortages, terminal stocks/SOC/debt, CO2, physical operating cost and both-event recovery/censor/rebreach.',
        annual_capital='Actual nominal actions give cumulative 2025..snapshot-year discounted capital, undiscounted paid capital and snapshot-year spending; discount origin2024/rate4%. Preserve full2025..2060 discounted capital for path identity. All are separate from 360h physical operating cost.',
        bridge_gate='First knee2060 two strategy paths must match full frozen V11 Python numeric paths atol1e-7 rtol1e-11 and original native ENS/peak/direct shortage within1e-6 plus exact recovery decisions. Native states/cost deviations are reported without asserting identity/cause.',
        failure_rule='Any case solve/physical/reference failure stops all later cases. Save actual attempted hour/state/actions/prefix/error; every remaining fixed-cohort row is NOT_RUN. Never relax debt, fuel, ramp or service constraints.',
        selection='Four controls fixed before this experiment; no search, selection, independent48 retest or condition tuning.',
        raw_artifacts='Each case: original initial state, prepared external arrays, all actual pre-solve states/actions JSONL, full actions/states CSV or actual executed prefix, physical audits, summary and two recovery rows.')
    write_json(out/'protocol.json',protocol)
    audit = dict(status='PASS_PREPARED_NOT_RUN',new_hourly_lp_calls=0,planned_cases=16,planned_lp_hours=5760,
                 source_hashes_unchanged=True,annual_source_audit_sha256=digest(out/'annual_source_audit.json'),
                 snapshots_sha256=digest(out/'snapshots.json'),protocol_sha256=digest(out/'protocol.json'),
                 readonly_existing_path_checks=readonly_reference)
    write_json(out/'preparation_audit.json',audit)
    print(json.dumps({k:audit[k] for k in ('status','new_hourly_lp_calls','planned_cases','protocol_sha256')},ensure_ascii=False),flush=True)
    return audit


def run(root=ROOT):
    root = Path(root).resolve()
    out = root/'outputs/v13/repair_operations'
    protocol_path = out/'protocol.json'
    protocol = json.loads(protocol_path.read_text('utf-8'))
    if (out/'analysis.json').exists() or any((out/'cases').glob('*')):
        raise FileExistsError('Preserve actual execution; this command cannot overwrite or resume it')
    for name,expected in protocol['source_sha256'].items():
        if digest(root/name) != expected:
            raise AssertionError('Source/input changed after freeze: '+name)
    for name,expected in protocol['prepared_artifact_sha256'].items():
        if digest(out/name) != expected:
            raise AssertionError('Prepared input changed after freeze: '+name)
    snapshots = json.loads((out/'snapshots.json').read_text('utf-8'))
    annual_evidence = json.loads((out/'annual_source_audit.json').read_text('utf-8'))
    outcomes = []
    failed = None
    started = time.perf_counter()
    for entry in protocol['cases']:
        input_prefix = 'inputs/'+entry['snapshot_id']+'/'
        record = dict(entry,status='NOT_RUN',complete=False,successful_lp_hours=0,attempted_lp_hours=0,
                      prepared_input_sha256={name:value for name,value in protocol['prepared_artifact_sha256'].items()
                                             if name.startswith(input_prefix)})
        if failed is not None:
            record['reason'] = 'Stopped after '+failed
            outcomes.append(record)
            continue
        target = out/'cases'/entry['case_id']
        target.mkdir(parents=True)
        case = make_case_from_snapshot(snapshots[entry['snapshot_id']])
        calls = []
        record.update(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat())
        before = time.perf_counter()
        try:
            with (target/'actual_pre_states_and_actions.jsonl').open('w',encoding='utf-8') as stream:
                result = simulate_recorded(case,entry['strategy'],calls,stream)
            result['timeseries'].to_csv(target/'actions_states.csv',index=False,float_format='%.17g')
            table = pd.read_csv(target/'actions_states.csv')
            replay = prefix_from_calls(case,calls)
            np.testing.assert_allclose(table.to_numpy(float),replay[table.columns].to_numpy(float),rtol=0,atol=1e-9)
            check = h.audit(case,table)
            write_json(target/'physical_audit.json',check)
            if not check['passed']:
                raise AssertionError('Exported actual trajectory failed physical audit')
            summary = metrics(case,table,entry['strategy'])
            summary['annual_nominal_2025_2060_discounted_capital_trillion_cny'] = annual_evidence[entry['plan']]['nominal_summary']['discounted_capital_trillion_cny']
            summary.update(annual_evidence[entry['plan']]['capital_by_snapshot_year'][str(entry['year'])])
            recovery = recovery_windows(table,recovery_config())
            write_json(target/'summary.json',summary)
            pd.DataFrame(recovery).to_csv(target/'recovery.csv',index=False)
            record.update(metrics=summary,recovery=recovery,executed_hours=len(table),physical_audit=check)
            if entry['bridge_required']:
                comparison = compare_with_references(root,entry['strategy'],case,table)
                write_json(target/'reference_comparison.json',comparison)
                record['reference_comparison'] = comparison
                if not comparison['passed']:
                    raise AssertionError('New knee2060 path failed frozen Python/native bridge gate')
            record.update(status='COMPLETED_AND_VALIDATED',complete=True)
        except Exception as failure:
            failed = entry['case_id']
            record.update(status='FAILED',error=str(failure),error_type=type(failure).__name__,traceback=traceback.format_exc(),
                          failure_hour=(calls[-1]['hour'] if calls and 'error' in calls[-1] else None),
                          last_attempted_hour=(calls[-1]['hour'] if calls else None))
            if not (target/'actions_states.csv').exists():
                prefix = prefix_from_calls(case,calls)
                record['executed_prefix_hours'] = len(prefix)
                if len(prefix):
                    prefix.to_csv(target/'observed_prefix_actions_states.csv',index=False,float_format='%.17g')
                    record['prefix_physical_audit'] = physical.prefix_audit(case,prefix)
                    write_json(target/'prefix_physical_audit.json',record['prefix_physical_audit'])
        finally:
            record.update(successful_lp_hours=sum('action' in call for call in calls),attempted_lp_hours=len(calls),
                          wall_seconds=time.perf_counter()-before,completed_utc=datetime.now(timezone.utc).isoformat())
            write_json(target/'result.json',record)
        outcomes.append(record)
        print(json.dumps({key:record.get(key) for key in ('case_id','status','executed_hours','successful_lp_hours','error')},ensure_ascii=False),flush=True)
    completed = all(row['complete'] for row in outcomes)
    rows = [{**{key:row[key] for key in ('case_id','snapshot_id','plan','year','strategy','status','complete','successful_lp_hours')},
             **row.get('metrics',{})} for row in outcomes]
    pd.DataFrame(rows).to_csv(out/'all_case_metrics.csv',index=False,float_format='%.17g')
    drift = {name:digest(root/name) for name,expected in protocol['source_sha256'].items() if digest(root/name) != expected}
    contrasts = []
    if completed:
        assert sum(row['successful_lp_hours'] for row in outcomes) == 5760
        by_id = {row['case_id']:row for row in outcomes}
        for plan in PLANS:
            for year in YEARS:
                a,b = [by_id[f'{plan}_{year}_{strategy}']['metrics'] for strategy in ('decentralized','coordinated')]
                contrasts.append(dict(plan=plan,year=year,comparison='decentralized minus coordinated',
                                      primary_ENS_difference_gwh=a['primary_h48_359_unserved_gwh']-b['primary_h48_359_unserved_gwh'],
                                      primary_peak_difference_gw=a['primary_h48_359_peak_unserved_gw']-b['primary_h48_359_peak_unserved_gw'],
                                      physical_operating_cost_difference_million_cny=a['physical_operating_cost_million_cny']-b['physical_operating_cost_million_cny']))
        pd.DataFrame(contrasts).to_csv(out/'within_snapshot_strategy_contrasts.csv',index=False,float_format='%.17g')
    report = dict(status='COMPLETED_SIXTEEN_VALIDATED_360H_PATHS' if completed and not drift else 'FAILED_FIXED_SIXTEEN_PATH_COHORT',
                  protocol_sha256=digest(protocol_path),protocol=protocol,cases=outcomes,
                  planned_cases=16,validated_complete_cases=sum(row['complete'] for row in outcomes),
                  planned_lp_hours=5760,actual_successful_lp_hours=sum(row['successful_lp_hours'] for row in outcomes),
                  actual_attempted_lp_hours=sum(row['attempted_lp_hours'] for row in outcomes),
                  actual_complete_trajectory_hours=sum(row.get('executed_hours',0) for row in outcomes),
                  new_native_runs=0,source_drift=drift,within_snapshot_strategy_contrasts=contrasts,
                  wall_seconds=time.perf_counter()-started,argv=sys.argv,
                  output_sha256={path.relative_to(out).as_posix():digest(path) for path in out.rglob('*')
                                 if path.is_file() and path.name != 'analysis.json'})
    write_json(out/'analysis.json',report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--run',action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
