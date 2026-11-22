"""Post-hoc developer challenge to the nonfossil-only local storage baseline.

This adds a reachable-all-source surplus rule, not an economic optimizer. The
original market LP and actual physical simulator remain unchanged. Existing V13
event-information results and their original strategy comparison stay frozen.
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
from src import v13_event_information as event
from src.v10_prepare_inputs import configuration, make_case
from src.v10_validate_dispatch import _prefix_audit
from src.v11_recovery_sensitivity import digest, recovery_windows

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/v13/storage_baseline'
EVENT_OUT=ROOT/'outputs/v13/event_information'
GRID=(
    {'case_id':'base','source_arm':'base','strategy':'coordinated','storage_power_multiplier':1.,'local_rule':'original','anchor':True},
    {'case_id':'power','source_arm':'power','strategy':'coordinated','storage_power_multiplier':1.5,'local_rule':'original','anchor':True},
    {'case_id':'decentralized','source_arm':'decentralized','strategy':'decentralized','storage_power_multiplier':1.,'local_rule':'original','anchor':True},
    {'case_id':'reachable_local','source_arm':'decentralized','strategy':'decentralized','storage_power_multiplier':1.,'local_rule':'reachable_all_source','anchor':False},
    {'case_id':'reachable_local_power','source_arm':'power','strategy':'decentralized','storage_power_multiplier':1.5,'local_rule':'reachable_all_source','anchor':False},
)


def case_configuration(row):
    return configuration(strategy=row['strategy'],storage_power_multiplier=row['storage_power_multiplier'])


def reachable_thermal(case,start,count,state):
    """Forecast a power upper bound from the actual pre-action thermal output.

    This bound does not reserve fuel, direct fuel service or debt repayment. Those
    remain original LP constraints; a forced local plan may therefore fail or
    coincide with shortage. It is never silently repaired.
    """
    available=case['available_gw'][start:start+count]
    result=np.empty((count,2))
    for j,fuel in enumerate(('coal','gas')):
        previous=float(state['thermal'][fuel])
        ramp=case['capacity_gw'][fuel]*case['config']['ramp_fraction'][fuel]
        for k in range(count):
            previous=min(float(available[k,h.GENERATORS.index(fuel)]),previous+ramp)
            result[k,j]=previous
    return result


def all_source_local_storage(case,start,count,state,telemetry=None):
    """Keep the original greedy store order/bounds; change its surplus estimate."""
    available=case['available_gw'][start:start+count]
    reachable=reachable_thermal(case,start,count,state)
    net=case['demand_gw'][start:start+count]-available[:,2:].sum(axis=1)-reachable.sum(axis=1)
    if telemetry is not None:
        telemetry.update(reachable_thermal_gw=reachable.copy(),net_before_storage_gw=net.copy())
    result={}
    for store in h.STORES:
        st,soc=case['storage'][store],state['soc'][store]
        charge,discharge=np.zeros(count),np.zeros(count)
        for k in range(count):
            if net[k]<0:
                charge[k]=min(-net[k],st['power_gw'],max(0,st['energy_gwh']-soc)/st['eta_charge'])
                soc+=charge[k]*st['eta_charge']
                net[k]+=charge[k]
            else:
                discharge[k]=min(net[k],st['power_gw'],max(0,soc)*st['eta_discharge'])
                soc-=discharge[k]/st['eta_discharge']
                net[k]-=discharge[k]
        result[store]=(charge,discharge)
    if telemetry is not None:
        telemetry['local_storage_plan']=deepcopy(result)
    return result


def simulate_storage(case,strategy,local_rule,calls=None):
    """Run the true simulator with a serial, finally-restored local-rule patch."""
    if local_rule not in ('original','reachable_all_source'):
        raise ValueError('Unknown local storage rule')
    if local_rule=='reachable_all_source' and strategy!='decentralized':
        raise ValueError('The reachable-all-source rule is a decentralized challenger')
    if calls is None:
        calls=[]
    original_plan,original_local=h._plan,h._local_storage

    def local(actual,start,count,state):
        telemetry={}
        actions=all_source_local_storage(actual,start,count,state,telemetry)
        assert calls and calls[-1]['hour']==start
        calls[-1]['local_storage_forecast']=telemetry
        return actions

    def plan(actual,start,state,actual_strategy):
        assert actual is case and actual_strategy==strategy
        stop=min(start+case['config']['forecast_hours'],case['hours'])
        record={'hour':start,'pre_state':deepcopy(state),'forecast_window_stop':stop,
                'local_rule':local_rule,
                'forecast_window':{key:case[key][start:stop].copy() for key in event.FORECAST_KEYS}}
        calls.append(record)
        try:
            actions=original_plan(actual,start,state,actual_strategy)
        except Exception as failure:
            record.update(error_type=type(failure).__name__,error=str(failure))
            raise
        record['action']=deepcopy(actions)
        return actions

    h._plan=plan
    if local_rule=='reachable_all_source':
        h._local_storage=local
    try:
        return h.simulate(case,strategy)
    finally:
        h._plan=original_plan
        h._local_storage=original_local


def input_preflight():
    """Check frozen reference identity and actual input shapes without solving."""
    configs,checks,sources=event.input_preflight()
    receipt=json.loads((EVENT_OUT/'analysis.json').read_text('utf-8'))
    assert receipt['status']=='COMPLETED_NINE_VALIDATED_360H_PATHS'
    assert receipt['actual_successful_lp_hours']==3240 and not receipt['source_drift']
    for relative,expected in receipt['protocol']['source_sha256'].items():
        assert digest(ROOT/relative)==expected, f'Completed event-study source drift: {relative}'
    paths=[Path(__file__),ROOT/'tests/test_v13_storage_baseline.py',EVENT_OUT/'analysis.json',EVENT_OUT/'protocol.json']
    for arm in ('base','power','decentralized'):
        trace=EVENT_OUT/f'{arm}_L24/actions_states.csv'
        result_path=EVENT_OUT/f'{arm}_L24/result.json'
        expected=receipt['output_sha256'][trace.relative_to(EVENT_OUT).as_posix()]
        assert digest(trace)==expected
        result=json.loads(result_path.read_text('utf-8'))
        assert result['complete'] and result['successful_lp_hours']==360
        assert result['original_physical_audit']['passed'] and result['independent_physical_audit']['passed']
        assert configs[arm]==event._expected_config(arm)
        paths.extend((trace,result_path))
    for path in paths:
        sources[path.relative_to(ROOT).as_posix()]=digest(path)
    return checks,sources


def prepare():
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Frozen protocol already exists; it will not be overwritten')
    checks,sources=input_preflight()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={
        'status':'PREPARED_NOT_RUN','frozen_utc':datetime.now(timezone.utc).isoformat(),
        'analysis_class':'POST_HOC_DEVELOPMENT_CHALLENGE_TO_NONFOSSIL_ONLY_LOCAL_STORAGE_BASELINE',
        'motivation':'The completed fixed event-information cohort showed that the original local storage rule had no eligible nonfossil charging surplus during its two shocks. This is a new comparator challenge, not a revision of that completed result.',
        'grid':GRID,'case_count':5,'hours_per_case':360,'expected_total_lp_hours':1800,
        'executed_scientific_lp_hours_at_freeze':0,'verification_lp_calls_per_full_test_suite':2,
        'configurations':{row['case_id']:case_configuration(row) for row in GRID},
        'snapshot':'plan_02349_2060','information':'Common 24-hour perfect forecasts, first action only, same actual state feedback.',
        'anchor_release_gate':'Run all three original 360h anchors first. Require their complete paths to match frozen V13 L24 and V11 trajectories, original native ENS/peak/recovery gates, and both physical audits. Only then run both new local-rule paths. If any anchor fails, retain all anchor outcomes and mark both challengers NOT_RUN.',
        'new_local_rule':{
            'thermal_upper_bound':'For coal/gas separately, initialize previous with the actual current pre-action thermal output. For each absolute forecast hour k, reachable[k]=min(actual forecast available[k],previous+capacity*ramp_fraction), then previous=reachable[k].',
            'net':'demand - nonfossil available sum - coal/gas reachable upper bounds',
            'storage':'Original battery-then-pumped sequence, greedy net surplus charging / deficit discharging, identical efficiencies, SOC bounds and charge+discharge power limit.',
            'implementation':'Replace only h._local_storage for the two new decentralized paths, in one serial process; always restore original _local_storage and _plan in finally. Original anchors keep the original local function.',
            'boundary':'Reachability is a power upper bound, not a fuel-feasible or economically optimal schedule. The rule ignores direct-fuel needs and flexible-load repayment when estimating surplus. Forced local charging can coincide with electricity/direct-fuel shortage or make the unchanged market LP infeasible; retain these outcomes without fallback or tuning.'},
        'unchanged':'Original h._plan objective/constraints, true h.simulate transitions, inherited fuel, event starts [48,192], 48h duration, severity 1, 360h, normal engineering profiles, initial fuel/SOC/thermal/debt, daily debt deadlines and original storage energy.',
        'primary_metrics':{'interval':'absolute hours 48..359 inclusive','metrics':['ENS_GWh','peak_unserved_GW']},
        'secondary_metrics':['whole_360h_and_warmup_ENS','direct_fuel_shortage','physical_cost_excluding_shortage_and_carbon_including_holding','CO2','initial_and_terminal_fuel_SOC_debt','shift_out_and_in','recovery_and_rebreach','shortage_while_charging_overlap'],
        'physical_cost_formula':'resource_accounted_cost - shortage_penalty - carbon_cost + stock_holding_cost; million CNY, declared engineering values.',
        'recovery_rule':{'stock_ratio':.95,'service_limit':.01,'window_hours':24,'event_boundaries':[192,360],'normalization':'each actual pre-event end state at start-1'},
        'physical_gate':'Every completed CSV passes original h.audit and reused independent per-hour action/state conservation and actual bounds at 1e-6.',
        'comparison_scope':'Compare original and reachable local controls at power 1; compare each local control against coordinated at the same power; compare reachable-local power 1 vs 1.5. No general optimality or algorithm-independent benefit claim.',
        'failure_rule':'Keep all five cases, original exceptions and failed hour/state/packet, all successful actions and observed completed prefixes. Do not clear debt, relax constraints, skip failed paths or substitute coordinated actions. No complete-cohort comparison if any case fails.',
        'input_checks':checks,'source_sha256':sources,
        'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'solver':'scipy.optimize.linprog(method=highs)'},
        'commands':{'test':'python -m unittest discover -s tests -p test_v13_storage_baseline.py -v','prepare':'python -m src.v13_storage_baseline --prepare','run_after_independent_review':'python -m src.v13_storage_baseline --run'},
        'working_directory':str(ROOT),'new_native_runs':0,'paid_services_used':False,
    }
    event.write_json(OUT/'protocol.json',protocol)
    pd.DataFrame(GRID).to_csv(OUT/'case_grid.csv',index=False)
    print(json.dumps({'status':protocol['status'],'cases':5,'scientific_lp_hours':0}))
    return protocol


def summarize(case,table,strategy):
    summary=event.metrics(case,table,strategy)
    summary['physical_cost_excluding_shortage_carbon_including_holding_million_cny']=(summary['resource_accounted_cost_million_cny']-summary['shortage_penalty_million_cny']-summary['carbon_cost_million_cny']+summary['stock_holding_cost_million_cny'])
    charge=table.battery_charge_gw+table.pumped_charge_gw
    electric=(table.unserved_gw>1e-5)&(charge>1e-5)
    direct=(table[[fuel+'_direct_unserved_mtce' for fuel in h.FUELS]].sum(axis=1)>1e-8)&(charge>1e-5)
    summary.update(electricity_shortage_while_charging_hours=int(electric.sum()),
                   charge_during_electricity_shortage_gwh=float(charge[electric].sum()),
                   unserved_during_charging_gwh=float(table.loc[electric,'unserved_gw'].sum()),
                   direct_fuel_shortage_while_charging_hours=int(direct.sum()))
    return summary


def anchor_comparison(arm,table,config):
    original=pd.read_csv(EVENT_OUT/f'{arm}_L24/actions_states.csv')
    assert set(original.columns)<=set(table.columns)
    differences={name:float(np.abs(table[name].to_numpy()-original[name].to_numpy()).max()) for name in original.columns}
    old_summary=summarize(make_case(config),original,config['strategy'])
    new_summary=summarize(make_case(config),table,config['strategy'])
    metric_diff={name:new_summary[name]-old_summary[name] for name in old_summary
                 if isinstance(old_summary[name],(int,float)) and not isinstance(old_summary[name],bool)
                 and old_summary[name] is not None}
    result={'frozen_V13_L24_path_max_absolute_column_differences':differences,
            'frozen_V13_L24_all_columns_match':all(value<=event.TOLERANCE for value in differences.values()),
            'frozen_V13_L24_numeric_metric_differences':metric_diff,
            'V11_and_native_reference':event.reference_comparison(arm,table,config)}
    result['passed']=result['frozen_V13_L24_all_columns_match'] and result['V11_and_native_reference']['passed']
    return result


def run():
    protocol_path=OUT/'protocol.json'
    protocol=json.loads(protocol_path.read_text('utf-8'))
    assert protocol['status']=='PREPARED_NOT_RUN'
    assert not (OUT/'analysis.json').exists(),'Do not overwrite an actual experiment'
    for relative,expected in protocol['source_sha256'].items():
        assert digest(ROOT/relative)==expected, f'Frozen source/input drift: {relative}'
    assert pd.read_csv(OUT/'case_grid.csv').to_dict('records')==list(GRID)
    outcomes=[]
    started=time.perf_counter()
    for row in GRID:
        target=OUT/row['case_id']
        target.mkdir(exist_ok=False)
        config=protocol['configurations'][row['case_id']]
        event.write_json(target/'parameters_initial_state.json',{'configuration':config,'local_rule':row['local_rule'],'initial_state_and_controls':protocol['input_checks'][row['source_arm']]})
        record=dict(row,status='RUNNING',complete=False)
        if not row['anchor'] and not all(item['complete'] for item in outcomes if item['anchor']):
            record.update(status='NOT_RUN',reason='ORIGINAL_ANCHOR_GATE_FAILED',executed_hours=0,
                          successful_lp_hours=0,attempted_lp_hours=0,wall_seconds=0.)
            event.write_json(target/'result.json',record)
            outcomes.append(record)
            continue
        case=make_case(config)
        calls=[]
        case_started=time.perf_counter()
        try:
            result=simulate_storage(case,config['strategy'],row['local_rule'],calls)
            result['timeseries'].to_csv(target/'actions_states.csv',index=False,float_format='%.17g')
            table=pd.read_csv(target/'actions_states.csv')
            original_audit=h.audit(case,table)
            residual,independent=event.independent_audit(case,table)
            residual.to_csv(target/'independent_physical_audit.csv',index=False,float_format='%.17g')
            record.update(executed_hours=len(table),original_physical_audit=original_audit,independent_physical_audit=independent)
            assert original_audit['passed'] and independent['passed'],'Physical audit failed'
            summary=summarize(case,table,config['strategy'])
            cfg_row=pd.Series({**config,'event1_start_hour':config['event_starts'][0],'event2_start_hour':config['event_starts'][1]})
            recovery=recovery_windows(table,cfg_row)
            event.write_json(target/'summary.json',summary)
            pd.DataFrame(recovery).to_csv(target/'recovery.csv',index=False)
            charge=table.battery_charge_gw+table.pumped_charge_gw
            short=(table.unserved_gw>1e-5)|(table[[fuel+'_direct_unserved_mtce' for fuel in h.FUELS]].sum(axis=1)>1e-8)
            table.loc[(charge>1e-5)&short].to_csv(target/'shortage_while_charging.csv',index=False,float_format='%.17g')
            record.update(metrics=summary,recovery=recovery)
            if row['anchor']:
                comparison=anchor_comparison(row['source_arm'],table,config)
                event.write_json(target/'anchor_comparison.json',comparison)
                record['anchor_comparison']=comparison
                assert comparison['passed'],'Original anchor reference gate failed'
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
            record.update(successful_lp_hours=sum('action' in call for call in calls),attempted_lp_hours=len(calls),wall_seconds=time.perf_counter()-case_started)
            with (target/'planner_state_packets_actions.jsonl').open('w',encoding='utf-8') as stream:
                for call in calls:
                    stream.write(json.dumps(event.json_value(call),ensure_ascii=False,allow_nan=False)+'\n')
            event.write_json(target/'result.json',record)
        outcomes.append(record)
        print(json.dumps({key:record.get(key) for key in ('case_id','status','executed_hours','successful_lp_hours','error')},ensure_ascii=False),flush=True)
    drift={relative:digest(ROOT/relative) for relative,expected in protocol['source_sha256'].items() if digest(ROOT/relative)!=expected}
    complete=all(row['complete'] for row in outcomes) and not drift
    anchor_passed=all(row['complete'] for row in outcomes if row['anchor'])
    comparisons=[]
    if complete:
        assert sum(row['successful_lp_hours'] for row in outcomes)==1800
        by_id={row['case_id']:row['metrics'] for row in outcomes}
        for left,right in (('decentralized','base'),('reachable_local','base'),('reachable_local_power','power'),('decentralized','reachable_local'),('reachable_local','reachable_local_power')):
            a,b=by_id[left],by_id[right]
            fields=('primary_h48_359_unserved_gwh','primary_h48_359_peak_unserved_gw','physical_cost_excluding_shortage_carbon_including_holding_million_cny','total_co2_mt')
            comparisons.append({'comparison':left+' minus '+right,**{field:a[field]-b[field] for field in fields}})
        pd.DataFrame(comparisons).to_csv(OUT/'within_cohort_comparisons.csv',index=False,float_format='%.17g')
    pd.DataFrame([{**{key:row[key] for key in ('case_id','local_rule','storage_power_multiplier','status','complete','successful_lp_hours')},**row.get('metrics',{})} for row in outcomes]).to_csv(OUT/'all_case_metrics.csv',index=False,float_format='%.17g')
    status='COMPLETED_FIVE_VALIDATED_360H_STORAGE_BASELINE_PATHS' if complete else 'FAILED_FIXED_STORAGE_BASELINE_COHORT'
    if not anchor_passed:
        status='FAILED_ORIGINAL_ANCHOR_GATE'
    report={'status':status,'analysis_class':protocol.get('analysis_class'),'case_count':len(outcomes),
            'original_anchor_gate_passed':anchor_passed,'not_run_challengers':sum(row['status']=='NOT_RUN' for row in outcomes),
            'validated_complete_cases':sum(row['complete'] for row in outcomes),'expected_total_lp_hours':1800,
            'actual_successful_lp_hours':sum(row['successful_lp_hours'] for row in outcomes),
            'actual_attempted_lp_hours':sum(row['attempted_lp_hours'] for row in outcomes),
            'actual_complete_trajectory_hours':sum(row.get('executed_hours',0) for row in outcomes),
            'wall_seconds':time.perf_counter()-started,'source_drift':drift,'protocol_sha256':digest(protocol_path),
            'protocol':protocol,'cases':outcomes,'within_cohort_comparisons':comparisons,
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
