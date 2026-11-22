"""Fixed-controller event-information ablation on the existing V10 experiment.

Only the event information supplied to the existing 24-hour planner changes.
The original hourly simulator advances the actual states and audits its output.
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
from src.v10_prepare_inputs import configuration, make_case, snapshot
from src.v10_validate_dispatch import config_from_csv, _prefix_audit
from src.v11_recovery_sensitivity import digest, recovery_windows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/v13/event_information'
CLOUD = ROOT / 'outputs/v10/anymath/cloud_final'
LOCAL = ROOT / 'outputs/v11/solver_paths'
ARMS = ('base', 'power', 'decentralized')
INFORMATION_HOURS = (1, 6, 24)
EXECUTION_ORDER = (24, 1, 6)
FORECAST_KEYS = ('available_gw', 'demand_gw', 'direct_fuel_demand_mtce', 'fuel_arrivals_mtce')
GRID = tuple({'case_id': f'{arm}_L{length}', 'arm': arm, 'event_information_hours': length}
             for length in EXECUTION_ORDER for arm in ARMS)
TOLERANCE = 1e-6


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')


def normal_case(config):
    """Return only known normal engineering profiles and the fixed controls.

    Event starts/duration and the true future arrays are absent from this object.
    The storage multiplier changes power only, exactly as frozen V10 make_case.
    """
    if config['supply_policy'] != 'inherited':
        raise ValueError('Event information experiment requires inherited supply; contracts are excluded')
    case = h.make_case(snapshot(), 'normal', config['hours'])
    case['stress_mask'][:] = False
    for store in h.STORES:
        case['storage'][store]['power_gw'] *= config['storage_power_multiplier']
    return case


def forecast_case(normal, revealed_event_prefix, start, severity):
    """Use a revealed event prefix, then persist its last revealed event state.

    The absolute 360-hour axis and past are retained. Daily debt deadlines in
    _plan consequently retain their original meaning. Normal wind/solar/demand
    cycles are known for the entire experiment; L limits event information only.
    """
    known = np.asarray(revealed_event_prefix)
    if known.dtype != np.bool_ or known.ndim != 1 or not start < len(known) <= normal['hours']:
        raise ValueError('Reveal a boolean event prefix including the current hour')
    if not 0 <= severity <= 1:
        raise ValueError('Event severity must lie in [0,1]')
    event = np.empty(normal['hours'], dtype=bool)
    event[:len(known)] = known
    event[len(known):] = known[-1]
    az = severity * event.astype(float)
    case = dict(normal)
    case['stress_mask'] = event
    case['available_gw'] = normal['available_gw'].copy()
    for tech, loss in (('wind', .75), ('solar', .45), ('hydro', .30)):
        case['available_gw'][:, h.GENERATORS.index(tech)] *= 1 - loss * az
    case['demand_gw'] = normal['demand_gw'] * (1 + .08 * az)
    case['fuel_import_arrivals_mtce'] = normal['fuel_import_arrivals_mtce'] * (1 - az[:, None] * np.array([.5, .95, .95]))
    case['fuel_arrivals_mtce'] = normal['fuel_domestic_arrivals_mtce'] + case['fuel_import_arrivals_mtce']
    case['direct_fuel_demand_mtce'] = normal['direct_fuel_demand_mtce'].copy()
    return case


def simulate_information(real, normal, strategy, information_hours, calls=None):
    """Temporarily replace only the planner input in this serial local process.

    Every plan receives the actual current state. The existing physical simulate
    loop always receives the true case. On failure, calls retain pre-solve state,
    packets, successful actions and the original exception; finally restores _plan.
    """
    if information_hours not in INFORMATION_HOURS:
        raise ValueError('Event information hours must be one of 1, 6, 24')
    if real['v10_config']['supply_policy'] != 'inherited':
        raise ValueError('Event information experiment requires inherited supply')
    if calls is None:
        calls = []
    original = h._plan

    def plan(actual, start, state, actual_strategy):
        assert actual is real and actual_strategy == strategy
        stop = min(start + information_hours, real['hours'])
        # The forecast function receives no complete mask, starts or true arrays.
        revealed = real['stress_mask'][:stop].copy()
        packet = forecast_case(normal, revealed, start, real['v10_config']['shock_severity'])
        horizon_stop = min(start + real['config']['forecast_hours'], real['hours'])
        record = {'hour': start, 'revealed_prefix_hours': stop,
                  'last_revealed_event_active': bool(revealed[-1]),
                  'pre_state': deepcopy(state), 'forecast_window_stop': horizon_stop,
                  'forecast_window': {key: packet[key][start:horizon_stop].copy() for key in FORECAST_KEYS}}
        calls.append(record)
        try:
            for key in FORECAST_KEYS:
                np.testing.assert_array_equal(packet[key][start], real[key][start], err_msg=f'current truth {key} h{start}')
                if information_hours == 24:
                    np.testing.assert_array_equal(packet[key][start:horizon_stop], real[key][start:horizon_stop], err_msg=f'L24 truth {key} h{start}')
            action = original(packet, start, state, actual_strategy)
        except Exception as failure:
            record.update(error=str(failure), error_type=type(failure).__name__)
            raise
        record['action'] = deepcopy(action)
        return action

    h._plan = plan
    try:
        return h.simulate(real, strategy)
    finally:
        h._plan = original


def independent_audit(case, table, complete=True):
    """Reconstruct per-hour conservation and actual constraints from CSV exports."""
    count = len(table)
    if not count:
        return pd.DataFrame(), {'passed': False, 'reason': 'no completed hours', 'hours': 0}
    if not np.array_equal(table.hour, np.arange(count)) or (complete and count != case['hours']):
        raise AssertionError('Independent audit hour coverage mismatch')
    residual = pd.DataFrame({'hour': np.arange(count)})
    positive = lambda x: np.maximum(0., x)
    gen = table[[g+'_gw' for g in h.GENERATORS]].to_numpy(float)
    availability = case['available_gw'][:count]
    residual['actual_generation_bound_gw'] = np.maximum(positive(-gen).max(axis=1), positive(gen-availability).max(axis=1))
    residual['actual_availability_export_gw'] = np.abs(table[[g+'_available_gw' for g in h.GENERATORS]].to_numpy()-availability).max(axis=1)
    residual['actual_demand_export_gw'] = np.abs(table.demand_gw-case['demand_gw'][:count])
    residual['actual_event_export'] = np.abs(table.stress_active.to_numpy()-case['stress_mask'][:count])
    balance = gen.sum(axis=1) + table.unserved_gw - table.spill_gw + table.shift_out_gw - table.shift_in_gw - case['demand_gw'][:count]
    for j, fuel in enumerate(h.FUELS):
        stock = np.r_[case['initial_fuel_stocks_mtce'][j], table[fuel+'_stock_mtce'].to_numpy()]
        short = table[fuel+'_direct_unserved_mtce'].to_numpy()
        direct = case['direct_fuel_demand_mtce'][:count, j]
        arrival = case['fuel_arrivals_mtce'][:count, j]
        burn = gen[:, h.GENERATORS.index(fuel)]/(h.GWH_PER_MTCE*case['fuel_efficiency'][fuel]) if fuel in ('coal', 'gas') else np.zeros(count)
        residual[fuel+'_stock_balance_mtce'] = np.abs(np.diff(stock)-arrival+direct-short+burn)
        residual[fuel+'_stock_nonnegative_mtce'] = positive(-stock[1:])
        residual[fuel+'_direct_service_bound_mtce'] = np.maximum(positive(-short), positive(short-direct))
        residual[fuel+'_actual_supply_export_mtce'] = np.abs(table[fuel+'_arrival_mtce']-arrival)
        residual[fuel+'_direct_demand_export_mtce'] = np.abs(table[fuel+'_direct_demand_mtce']-direct)
        residual[fuel+'_direct_served_export_mtce'] = np.abs(table[fuel+'_direct_served_mtce']-(direct-short))
        residual[fuel+'_burn_export_mtce'] = np.abs(table[fuel+'_power_burn_mtce']-burn)
    for store in h.STORES:
        st = case['storage'][store]
        charge, discharge = (table[store+'_'+name+'_gw'].to_numpy() for name in ('charge', 'discharge'))
        soc = np.r_[st['initial_soc_gwh'], table[store+'_soc_gwh'].to_numpy()]
        residual[store+'_soc_balance_gwh'] = np.abs(np.diff(soc)-st['eta_charge']*charge+discharge/st['eta_discharge'])
        residual[store+'_soc_bounds_gwh'] = np.maximum(positive(-soc[1:]), positive(soc[1:]-st['energy_gwh']))
        residual[store+'_power_bounds_gw'] = np.maximum.reduce([positive(-charge), positive(-discharge), positive(charge+discharge-st['power_gw'])])
        balance += discharge-charge
    residual['power_balance_gw'] = np.abs(balance)
    out, repay = table.shift_out_gw.to_numpy(), table.shift_in_gw.to_numpy()
    debt = np.r_[0., table.shift_debt_gwh.to_numpy()]
    demand = case['demand_gw'][:count]
    cfg = case['config']
    residual['debt_balance_gwh'] = np.abs(np.diff(debt)-out+repay)
    residual['debt_nonnegative_gwh'] = positive(-debt[1:])
    residual['flex_bound_gw'] = np.maximum.reduce([positive(-out), positive(-repay), positive(out-cfg['flex_fraction']*demand), positive(repay-cfg['repay_fraction']*demand)])
    deadline = (np.arange(count)+1) % cfg['deadline_hours'] == 0
    if count == case['hours']:
        deadline[-1] = True
    residual['debt_deadline_gwh'] = np.where(deadline, np.abs(debt[1:]), 0.)
    residual['slack_bounds_gw'] = np.maximum.reduce([positive(-table.unserved_gw.to_numpy()), positive(-table.spill_gw.to_numpy()), positive(table.unserved_gw.to_numpy()+out-demand)])
    residual['served_power_export_gw'] = np.abs(table.served_power_gw-(demand-out+repay-table.unserved_gw))
    for fuel in ('coal', 'gas'):
        generation = np.r_[case['initial_thermal_gw'][fuel], table[fuel+'_gw'].to_numpy()]
        available = np.r_[case['available_gw'][0, h.GENERATORS.index(fuel)], case['available_gw'][:count, h.GENERATORS.index(fuel)]]
        ramp = case['capacity_gw'][fuel]*cfg['ramp_fraction'][fuel]
        residual[fuel+'_ramp_gw'] = np.maximum(positive(np.diff(generation)-ramp), positive(-np.diff(generation)-np.maximum(ramp, positive(-np.diff(available)))))
    biomass_limit = case['biomass_generation_budget_twh_per_year']*1000*count/8760
    biomass_violation = max(0., float(table.biomass_gw.sum()-biomass_limit))
    values = residual.drop(columns='hour').to_numpy(float)
    maxima = {name: float(residual[name].max()) for name in residual if name != 'hour'}
    finite = bool(np.isfinite(values).all() and np.isfinite(table.select_dtypes(include=np.number)).all().all())
    maximum = max(maxima.values())
    return residual, {'passed': finite and maximum <= TOLERANCE and biomass_violation <= TOLERANCE,
                      'hours': count, 'complete': count == case['hours'], 'all_finite': finite,
                      'maximum_residual': maximum, 'tolerance': TOLERANCE,
                      'biomass_budget_violation_gwh': biomass_violation, 'maxima': maxima}


def prefix_from_calls(case, calls):
    """Export completed actions using states observed at the following solve.

    This reconstructs an interrupted observation only; it never executes another
    transition or treats a planned LP state as an actual plant state.
    """
    records = []
    for previous, following in zip(calls, calls[1:]):
        if 'action' not in previous:
            break
        t, action, state = previous['hour'], previous['action'], following['pre_state']
        row = {'hour': t, 'demand_gw': case['demand_gw'][t], 'stress_active': int(case['stress_mask'][t]),
               'shift_out_gw': float(action['out']), 'shift_in_gw': float(action['repay']),
               'unserved_gw': float(action['unserved']), 'spill_gw': float(action['spill']),
               'shift_debt_gwh': state['debt']}
        for j, gen in enumerate(h.GENERATORS):
            row[gen+'_gw'] = action['gen'][j]
            row[gen+'_available_gw'] = case['available_gw'][t,j]
        for j, fuel in enumerate(h.FUELS):
            row[fuel+'_stock_mtce'] = state['fuel'][j]
            row[fuel+'_arrival_mtce'] = case['fuel_arrivals_mtce'][t,j]
            row[fuel+'_direct_demand_mtce'] = case['direct_fuel_demand_mtce'][t,j]
            row[fuel+'_direct_unserved_mtce'] = action['direct_short'][j]
            row[fuel+'_direct_served_mtce'] = case['direct_fuel_demand_mtce'][t,j]-action['direct_short'][j]
            row[fuel+'_power_burn_mtce'] = row[fuel+'_gw']/(h.GWH_PER_MTCE*case['fuel_efficiency'][fuel]) if fuel in ('coal','gas') else 0.
        for j, store in enumerate(h.STORES):
            row[store+'_charge_gw'], row[store+'_discharge_gw'] = action['ch'][j], action['dis'][j]
            row[store+'_soc_gwh'] = state['soc'][store]
        row['served_power_gw'] = row['demand_gw']-row['shift_out_gw']+row['shift_in_gw']-row['unserved_gw']
        records.append(row)
    return pd.DataFrame(records)


def _enrich_native(case, frame):
    table = frame.copy()
    for j, fuel in enumerate(h.FUELS):
        table[fuel+'_direct_served_mtce'] = case['direct_fuel_demand_mtce'][:,j]-table[fuel+'_direct_unserved_mtce']
        table[fuel+'_power_burn_mtce'] = table[fuel+'_gw']/(h.GWH_PER_MTCE*case['fuel_efficiency'][fuel]) if fuel in ('coal','gas') else 0.
    return table


def _expected_config(arm):
    changes = {'power': {'storage_power_multiplier': 1.5}, 'decentralized': {'strategy': 'decentralized'}}
    return configuration(**changes.get(arm, {}))


def input_preflight():
    """Read/audit frozen references and packet shapes without any LP solves."""
    native_receipt = json.loads((ROOT/'outputs/v10/anymath/execution_receipt.json').read_text('utf-8'))
    local_receipt = json.loads((LOCAL/'analysis.json').read_text('utf-8'))
    assert native_receipt['total_hours'] == 1800 and local_receipt['local_new_lp_hours'] == 1800
    sources = [Path(__file__), ROOT/'tests/test_v13_event_information.py',
               ROOT/'src/v3_hourly.py', ROOT/'src/v10_prepare_inputs.py',
               ROOT/'src/v10_validate_dispatch.py', ROOT/'src/v11_solver_paths.py',
               ROOT/'src/v11_recovery_sensitivity.py', ROOT/'outputs/v6/resilience/native_inputs.json',
               ROOT/'outputs/v10/anymath/execution_receipt.json', LOCAL/'analysis.json']
    configs, checks = {}, {}
    for arm in ARMS:
        config_path, native_path = CLOUD/f'{arm}_config.csv', CLOUD/f'{arm}_actions_states.csv'
        local_path = LOCAL/f'python_{arm}.csv'
        for path in (config_path, native_path):
            assert native_receipt['artifact_sha256'][path.relative_to(ROOT).as_posix()] == digest(path), str(path)
        assert local_receipt['output_sha256'][local_path.name] == digest(local_path)
        cfg = config_from_csv(config_path)
        assert cfg == _expected_config(arm), (arm, cfg)
        configs[arm] = cfg
        real, normal = make_case(cfg), normal_case(cfg)
        native = _enrich_native(real, pd.read_csv(native_path))
        local = pd.read_csv(local_path)
        native_audit, local_audit = h.audit(real, native), h.audit(real, local)
        assert native_audit['passed'] and local_audit['passed']
        for store in h.STORES:
            assert normal['storage'][store] == real['storage'][store]
        for length in INFORMATION_HOURS:
            for start in range(360):
                packet = forecast_case(normal, real['stress_mask'][:min(start+length,360)], start, cfg['shock_severity'])
                for key in FORECAST_KEYS:
                    np.testing.assert_array_equal(packet[key][start], real[key][start])
                    if length == 24:
                        stop = min(start+24,360)
                        np.testing.assert_array_equal(packet[key][start:stop], real[key][start:stop])
        checks[arm] = {'native_physical_audit_passed': True, 'v11_local_physical_audit_passed': True,
                       'packet_current_truth_checks': 1080, 'L24_whole_window_checks': 360,
                       'initial_fuel_stocks_mtce': real['initial_fuel_stocks_mtce'],
                       'initial_thermal_gw': real['initial_thermal_gw'], 'storage': real['storage'],
                       'controller_config': real['config']}
        sources.extend((config_path, native_path, local_path))
    return configs, checks, {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}


def prepare():
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Frozen protocol already exists; it will not be overwritten')
    configs, checks, sources = input_preflight()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'status': 'PREPARED_NOT_RUN', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_class': 'EXPLORATORY_FIXED_CONTROLLER_EVENT_INFORMATION_ABLATION',
        'grid': GRID, 'case_count': 9, 'hours_per_case': 360, 'expected_total_lp_hours': 3240,
        'executed_lp_hours_at_freeze': 0, 'snapshot': 'plan_02349_2060', 'configurations': configs,
        'optimization_window_hours': 24, 'information_hours': INFORMATION_HOURS,
        'execution_order': EXECUTION_ORDER,
        'L24_release_gate': 'Run all three L24 arms first. Release the six L1/L6 paths only after all three complete physical audits and every L24 reference gate passes. If any L24 arm fails, retain all three L24 outcomes and record all six later paths NOT_RUN; no information contrasts or partial performance conclusion.',
        'verification_lp_calls_per_full_test_suite': 2,
        'event_information_rule': 'At absolute hour t, observe true mask prefix [:min(t+L,T)]; persist its last event state after that prefix. No event schedule or unrevealed actual arrays enters forecast_case.',
        'known_background': 'Full normal engineering wind, solar, hydro, demand, domestic flow and direct fuel demand cycles are known. L describes event information, not all exogenous prediction.',
        'execution': 'Original v3_hourly._plan objective/constraints and v3_hourly.simulate physical loop; same absolute 360h timeline, initial stocks/SOC/thermal/debt, power multiplier and first-action execution. Serial process; temporary _plan patch restored in finally.',
        'excluded': 'No contracts, altered objective, controller retuning, debt reset, hard-constraint relaxation, perfect-information fallback, additional cloud/native runs or paid services.',
        'primary_metrics': {'interval': 'absolute hours 48..359 inclusive', 'metrics': ['ENS_GWh', 'peak_unserved_GW']},
        'secondary_metrics': ['whole_360h_and_warmup_ENS', 'direct_fuel_shortage', 'resource_accounted_cost', 'CO2', 'terminal_fuel_SOC_debt', 'recovery_and_rebreach'],
        'recovery_rule': {'stock_ratio': .95, 'service_limit': .01, 'window_hours': 24, 'event_boundaries': [192,360], 'pre_event_stock_reference': 'actual end-state at start-1'},
        'comparisons': 'Within each same L: decentralized minus base and base minus power. Across L: information effects, not isolated strategy superiority.',
        'L24_gates': {
            'V11_local': {'all_reference_trajectory_columns_present': True, 'all_numeric_column_absolute_tolerance': 1e-6,
                          'ENS_GWh_absolute_tolerance': 1e-6, 'peak_GW_absolute_tolerance': 1e-6, 'recovery_decision_fields_equal': True},
            'native': {'hourly_ENS_GW_absolute_tolerance': 1e-6, 'ENS_GWh_absolute_tolerance': 1e-6,
                       'peak_GW_absolute_tolerance': 1e-6, 'recovery_decision_fields_equal': True,
                       'inventory_SOC_and_cost': 'Report actual backend state/action/metric differences without attributing their cause.'},
            'identity': 'New SciPy, frozen V11 SciPy and original native AnyMath/Julia paths remain separate; no action rewriting or added tie break.'},
        'failure_rule': 'Keep original exception, failed hour/state/packet and completed observed prefix; retain infeasible paths in all-nine result set. No partial-cohort performance claim.',
        'physical_gate': 'Every completed CSV passes original h.audit and independent per-hour conservation/actual supply, generation, stock, SOC, debt, deadlines and ramp checks at 1e-6.',
        'input_checks': checks, 'source_sha256': sources,
        'environment': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'solver': 'scipy.optimize.linprog(method=highs)'},
        'commands': {'test': 'python -m unittest discover -s tests -p test_v13_event_information.py -v', 'prepare': 'python -m src.v13_event_information --prepare', 'run_after_independent_review': 'python -m src.v13_event_information --run'},
        'working_directory': str(ROOT), 'new_native_runs': 0,
    }
    write_json(OUT/'protocol.json', protocol)
    pd.DataFrame(GRID).to_csv(OUT/'case_grid.csv', index=False)
    print(json.dumps({'status': protocol['status'], 'cases': 9, 'scientific_lp_hours': 0}))
    return protocol


def metrics(case, table, strategy):
    result = h.summarize(case, table, strategy)
    post = table.loc[table.hour >= 48]
    result.update(primary_h48_359_unserved_gwh=float(post.unserved_gw.sum()),
                  primary_h48_359_peak_unserved_gw=float(post.unserved_gw.max()),
                  warmup_h0_47_unserved_gwh=float(table.loc[table.hour < 48, 'unserved_gw'].sum()),
                  resource_excluding_shortage_and_carbon_million_cny=(result['resource_accounted_cost_million_cny']-result['shortage_penalty_million_cny']-result['carbon_cost_million_cny']))
    return result


def _recovery_decisions(rows):
    fields = ('event', 'service_confirmation_hour', 'joint_confirmation_hour', 'service_right_censored',
              'joint_right_censored', 'stock_rebreach_hour_after_joint_confirmation',
              'final_window_service_ready', 'final_window_stock_ready', 'final_window_joint_ready')
    return [{key: row[key] for key in fields} for row in rows]


def reference_comparison(arm, table, config):
    local = pd.read_csv(LOCAL/f'python_{arm}.csv')
    native = pd.read_csv(CLOUD/f'{arm}_actions_states.csv')
    cfg_row = pd.read_csv(CLOUD/f'{arm}_config.csv').iloc[0]
    assert set(local.columns) <= set(table.columns), 'A V11 reference observable is missing'
    differences = {name: float(np.abs(table[name].to_numpy()-local[name].to_numpy()).max()) for name in local.columns}
    rec = recovery_windows(table, cfg_row)
    local_rec, native_rec = recovery_windows(local,cfg_row), recovery_windows(native,cfg_row)
    native_hourly_ens = float(np.abs(table.unserved_gw-native.unserved_gw).max())
    native_differences = {name: float(np.abs(table[name].to_numpy()-native[name].to_numpy()).max())
                          for name in local.columns if name in native.columns}
    real = make_case(config)
    native_summary = metrics(real,_enrich_native(real,native),config['strategy'])
    new_summary = metrics(real,table,config['strategy'])
    metric_names = ('unserved_gwh', 'peak_unserved_gw', 'primary_h48_359_unserved_gwh',
                    'primary_h48_359_peak_unserved_gw', 'direct_fuel_unserved_mtce',
                    'resource_accounted_cost_million_cny', 'resource_excluding_shortage_and_carbon_million_cny',
                    'total_co2_mt', 'terminal_shift_debt_gwh')
    metric_names += tuple(fuel+'_terminal_stock_mtce' for fuel in h.FUELS)
    metric_names += tuple(store+'_terminal_soc_gwh' for store in h.STORES)
    native_metric_differences = {name: new_summary[name]-native_summary[name] for name in metric_names}
    result = {'V11_local_max_absolute_column_differences': differences,
              'V11_local_all_common_columns_match': all(value <= TOLERANCE for value in differences.values()),
              'V11_local_ENS_difference_gwh': float(table.unserved_gw.sum()-local.unserved_gw.sum()),
              'V11_local_peak_difference_gw': float(table.unserved_gw.max()-local.unserved_gw.max()),
              'V11_local_recovery_decisions_match': _recovery_decisions(rec) == _recovery_decisions(local_rec),
              'native_max_hourly_ENS_difference_gw': native_hourly_ens,
              'native_total_ENS_difference_gwh': float(table.unserved_gw.sum()-native.unserved_gw.sum()),
              'native_peak_difference_gw': float(table.unserved_gw.max()-native.unserved_gw.max()),
              'native_recovery_decisions_match': _recovery_decisions(rec) == _recovery_decisions(native_rec),
              'native_max_absolute_column_differences_report_only': native_differences,
              'native_metric_differences_report_only': native_metric_differences,
              'native_independently_reconstructed_metrics': {name:native_summary[name] for name in metric_names},
              'new_recovery': rec, 'V11_local_recovery': local_rec, 'native_recovery': native_rec,
              'identity': 'New SciPy path, V11 local SciPy reference and original AnyMath/Julia path remain separate; actual backend state/action/metric differences are reported without attributing their cause.'}
    result['passed'] = (result['V11_local_all_common_columns_match'] and result['V11_local_recovery_decisions_match']
                        and abs(result['V11_local_ENS_difference_gwh']) <= TOLERANCE and abs(result['V11_local_peak_difference_gw']) <= TOLERANCE
                        and native_hourly_ens <= TOLERANCE and abs(result['native_total_ENS_difference_gwh']) <= TOLERANCE
                        and abs(result['native_peak_difference_gw']) <= TOLERANCE
                        and result['native_recovery_decisions_match'])
    return result


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
    for entry in GRID:
        case_id, arm, length = entry['case_id'], entry['arm'], entry['event_information_hours']
        target = OUT/case_id
        target.mkdir(exist_ok=False)
        cfg = protocol['configurations'][arm]
        real, normal, calls = make_case(cfg), normal_case(cfg), []
        real['information_contract'] = f'COMMON_{length}H_EVENT_INFORMATION_WITH_KNOWN_NORMAL_PROFILES_24H_OPTIMIZATION'
        write_json(target/'parameters_initial_state.json', {'configuration': cfg, 'event_information_hours': length,
                   'initial_state_and_controls': protocol['input_checks'][arm]})
        record = dict(entry, status='RUNNING', complete=False)
        if length != 24 and not all(row['complete'] for row in outcomes if row['event_information_hours'] == 24):
            record.update(status='NOT_RUN', reason='L24_REFERENCE_BRIDGE_GATE_FAILED',
                          executed_hours=0, successful_lp_hours=0, attempted_lp_hours=0, wall_seconds=0.)
            write_json(target/'result.json',record)
            outcomes.append(record)
            print(json.dumps({key:record[key] for key in ('case_id','status','reason')},ensure_ascii=False),flush=True)
            continue
        case_started = time.perf_counter()
        try:
            result = simulate_information(real, normal, cfg['strategy'], length, calls)
            table = result['timeseries']
            table.to_csv(target/'actions_states.csv', index=False, float_format='%.17g')
            table = pd.read_csv(target/'actions_states.csv')
            original_audit = h.audit(real, table)
            residual, independent = independent_audit(real,table)
            residual.to_csv(target/'independent_physical_audit.csv',index=False,float_format='%.17g')
            record.update(executed_hours=len(table), original_physical_audit=original_audit, independent_physical_audit=independent)
            assert original_audit['passed'] and independent['passed'], 'Physical audit failed'
            summary = metrics(real, table, cfg['strategy'])
            cfg_row = pd.read_csv(CLOUD/f'{arm}_config.csv').iloc[0]
            recovery = recovery_windows(table,cfg_row)
            write_json(target/'summary.json',summary)
            pd.DataFrame(recovery).to_csv(target/'recovery.csv',index=False)
            record.update(metrics=summary, recovery=recovery)
            if length == 24:
                comparison = reference_comparison(arm,table,cfg)
                write_json(target/'L24_reference_comparison.json',comparison)
                record['L24_reference_comparison'] = comparison
                assert comparison['passed'], 'L24 reference gate failed; retain actual path and discrepancies'
            record.update(status='COMPLETED_AND_VALIDATED',complete=True)
        except Exception as failure:
            planner_failure = bool(calls and 'error' in calls[-1])
            record.update(status='FAILED',error=str(failure),error_type=type(failure).__name__,traceback=traceback.format_exc(),
                          failure_hour=(calls[-1]['hour'] if planner_failure else None),
                          last_attempted_hour=(calls[-1]['hour'] if calls else None),
                          failure_phase=('PLANNER' if planner_failure else 'SIMULATION_OR_VALIDATION'))
            if not (target/'actions_states.csv').exists():
                prefix = prefix_from_calls(real,calls)
                record['observed_prefix_hours'] = len(prefix)
                if len(prefix):
                    prefix.to_csv(target/'observed_prefix_actions_states.csv',index=False,float_format='%.17g')
                    try:
                        residual, independent = independent_audit(real,prefix,complete=False)
                        residual.to_csv(target/'prefix_independent_physical_audit.csv',index=False,float_format='%.17g')
                        record['prefix_original_physical_audit'] = _prefix_audit(real,prefix)
                        record['prefix_independent_physical_audit'] = independent
                    except Exception as audit_failure:
                        record['prefix_audit_error'] = {'type':type(audit_failure).__name__, 'error':str(audit_failure), 'traceback':traceback.format_exc()}
        finally:
            record.update(successful_lp_hours=sum('action' in call for call in calls), attempted_lp_hours=len(calls),
                          wall_seconds=time.perf_counter()-case_started)
            with (target/'planner_state_packets_actions.jsonl').open('w',encoding='utf-8') as stream:
                for call in calls:
                    stream.write(json.dumps(json_value(call),ensure_ascii=False,allow_nan=False)+'\n')
            write_json(target/'result.json',record)
        outcomes.append(record)
        print(json.dumps({key:record.get(key) for key in ('case_id','status','executed_hours','successful_lp_hours','error')},ensure_ascii=False),flush=True)
    complete = all(row['complete'] for row in outcomes)
    total_success = sum(row['successful_lp_hours'] for row in outcomes)
    rows, contrasts = [], []
    for row in outcomes:
        rows.append({**{key:row[key] for key in ('case_id','arm','event_information_hours','status','complete','successful_lp_hours')}, **row.get('metrics',{})})
    pd.DataFrame(rows).to_csv(OUT/'all_case_metrics.csv',index=False,float_format='%.17g')
    if complete:
        assert total_success == 3240
        by_id = {row['case_id']: row for row in outcomes}
        for length in INFORMATION_HOURS:
            for left,right in (('decentralized','base'),('base','power')):
                a,b = by_id[f'{left}_L{length}']['metrics'], by_id[f'{right}_L{length}']['metrics']
                contrasts.append({'event_information_hours': length, 'comparison': left+' minus '+right,
                    'primary_ENS_difference_gwh': a['primary_h48_359_unserved_gwh']-b['primary_h48_359_unserved_gwh'],
                    'primary_peak_difference_gw': a['primary_h48_359_peak_unserved_gw']-b['primary_h48_359_peak_unserved_gw']})
        pd.DataFrame(contrasts).to_csv(OUT/'within_information_contrasts.csv',index=False,float_format='%.17g')
    drift = {relative:digest(ROOT/relative) for relative,expected in protocol['source_sha256'].items() if digest(ROOT/relative) != expected}
    bridge_passed = all(row['complete'] for row in outcomes if row['event_information_hours'] == 24)
    status = 'COMPLETED_NINE_VALIDATED_360H_PATHS' if complete and not drift else 'FAILED_FIXED_NINE_PATH_COHORT'
    if not bridge_passed:
        status = 'FAILED_L24_REFERENCE_BRIDGE_GATE'
    report = {'status': status,
              'case_count':len(outcomes),'validated_complete_cases':sum(row['complete'] for row in outcomes),
              'L24_bridge_gate_passed':bridge_passed,
              'not_run_cases_due_to_L24_gate':sum(row['status'] == 'NOT_RUN' for row in outcomes),
              'expected_total_lp_hours':3240,'actual_successful_lp_hours':total_success,
              'actual_attempted_lp_hours':sum(row['attempted_lp_hours'] for row in outcomes),
              'actual_complete_trajectory_hours':sum(row.get('executed_hours',0) for row in outcomes),
              'wall_seconds':time.perf_counter()-started,'source_drift':drift,'protocol_sha256':digest(protocol_path),
              'protocol':protocol,'cases':outcomes,'within_information_contrasts':contrasts,
              'new_native_runs':0,'paid_services_used':False,
              'output_sha256':{p.relative_to(OUT).as_posix():digest(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'analysis.json'}}
    write_json(OUT/'analysis.json',report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--run',action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
