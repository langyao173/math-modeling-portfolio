"""V4 explicit-constraint operation experiments; V3 is frozen.

The copied LP assembly derives from src/v3_hourly.py. Physical equations and
independent audits are unchanged. New hard limits are never relaxed. Full-window
frontiers are offline optimistic bounds, not online algorithm improvements.
AI-assisted: OpenAI Codex, 2026-10-17.
"""
from __future__ import annotations
from copy import deepcopy
import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from src import v3_hourly as v3
from src.v3_hourly import (make_case, audit, summarize, GENERATORS, FUELS, STORES,
                          GWH_PER_MTCE, FUEL_EMISSION_MT_PER_MTCE, _local_storage)

class InfeasiblePlan(RuntimeError):
    def __init__(self, hour, status, message):
        super().__init__(f"V4 LP hour {hour}, status {status}: {message}")
        self.hour, self.status, self.solver_message = hour, status, message

def _solve(case, start, state, strategy, nstep, limits=None, objective="legacy", all_steps=False):
    cfg = case["config"]
    names = [("gen", nstep*7), ("ch", nstep*2), ("dis", nstep*2),
             ("soc", nstep*2), ("fuel", nstep*3), ("direct_short", nstep*3),
             ("out", nstep), ("repay", nstep), ("debt", nstep),
             ("unserved", nstep), ("spill", nstep)]
    ix, offset = {}, 0
    for name, count in names:
        ix[name] = np.arange(offset, offset+count)
        offset += count
    for name, width in (("gen", 7), ("ch", 2), ("dis", 2), ("soc", 2), ("fuel", 3), ("direct_short", 3)):
        ix[name] = ix[name].reshape(nstep, width)
    c = np.zeros(offset)
    bounds = [(0, None)]*offset
    available = case["available_gw"][start:start+nstep]
    demand = case["demand_gw"][start:start+nstep]
    direct = case["direct_fuel_demand_mtce"][start:start+nstep]
    arrivals = case["fuel_arrivals_mtce"][start:start+nstep]
    local = _local_storage(case, start, nstep, state) if strategy == "fixed_rule" else None
    for k in range(nstep):
        urgency = 1+1e-5*(nstep-k)/nstep  # deterministic first-service tie break only
        for j, g in enumerate(GENERATORS):
            bounds[ix["gen"][k, j]] = (0, float(available[k, j]))
            carbon = (FUEL_EMISSION_MT_PER_MTCE[g]/(GWH_PER_MTCE*case["fuel_efficiency"][g])
                      *cfg["carbon_price_cny_t"] if g in {"coal", "gas"} else 0)
            c[ix["gen"][k, j]] = cfg["generation_cost_cny_kwh"][g]+carbon
        for j, f in enumerate(FUELS):
            bounds[ix["direct_short"][k, j]] = (0, float(direct[k, j]))
            c[ix["direct_short"][k, j]] = (cfg["direct_shortage_cost_cny_kwh_th"]*GWH_PER_MTCE*urgency
                -cfg["fuel_value_million_cny_mtce"][f]
                -FUEL_EMISSION_MT_PER_MTCE[f]*cfg["carbon_price_cny_t"])
        for j, s in enumerate(STORES):
            st = case["storage"][s]
            bounds[ix["soc"][k, j]] = (0, st["energy_gwh"])
            if k == nstep-1:
                c[ix["soc"][k, j]] = -cfg["stored_electricity_value_million_cny_gwh"]
            for kind in ("ch", "dis"):
                bounds[ix[kind][k, j]] = (0, st["power_gw"])
                c[ix[kind][k, j]] = cfg["storage_cycle_cost_cny_kwh"]
            if local is not None:
                for kind, vals in zip(("ch", "dis"), local[s]):
                    value = float(vals[k])
                    bounds[ix[kind][k, j]] = (value, value)
        bounds[ix["out"][k]] = (0, 0 if strategy in {"fixed_rule", "myopic_1h", "noflex_24h"} else float(cfg["flex_fraction"]*demand[k]))
        bounds[ix["repay"][k]] = (0, float(cfg["repay_fraction"]*demand[k]))
        c[ix["out"][k]] = c[ix["repay"][k]] = cfg["shift_cost_cny_kwh"]
        c[ix["unserved"][k]] = cfg["unserved_cost_cny_kwh"]*urgency
        # Debt is a real obligation: no shedding or resetting it at the deadline.
        if (start+k+1) % cfg["deadline_hours"] == 0 or start+k == case["hours"]-1:
            bounds[ix["debt"][k]] = (0, 0)
    if objective in {"physical_cost", "emissions"}:
        for k in range(nstep):
            c[ix["unserved"][k]] = 0
            for j, f in enumerate(FUELS):
                c[ix["direct_short"][k, j]] = (-cfg["fuel_value_million_cny_mtce"][f]
                    if objective == "physical_cost" else -FUEL_EMISSION_MT_PER_MTCE[f])
            for j, g in enumerate(GENERATORS):
                c[ix["gen"][k, j]] = (cfg["generation_cost_cny_kwh"][g]
                    if objective == "physical_cost" else
                    FUEL_EMISSION_MT_PER_MTCE[g]/(GWH_PER_MTCE*case["fuel_efficiency"][g])
                    if g in {"coal", "gas"} else 0)
            if objective == "emissions":
                for kind in ("ch", "dis", "soc", "out", "repay"):
                    c[ix[kind][k]] = 0
    # Fixed rule uses the same deterministic declared fuel-price merit order;
    # no carbon repricing, intertemporal storage choice or new flexible debt.
    if strategy == "fixed_rule":
        for k in range(nstep):
            for j, g in enumerate(GENERATORS):
                c[ix["gen"][k, j]] = cfg["generation_cost_cny_kwh"][g]
    eq_rows, eq_rhs, ub_rows, ub_rhs = [], [], [], []
    def eq(row, rhs):
        eq_rows.append(row); eq_rhs.append(rhs)
    def ub(row, rhs):
        ub_rows.append(row); ub_rhs.append(rhs)
    for k in range(nstep):
        row = {int(i): 1 for i in ix["gen"][k]}
        for j in range(2):
            row[int(ix["dis"][k, j])] = 1
            row[int(ix["ch"][k, j])] = -1
        row.update({int(ix["out"][k]): 1, int(ix["repay"][k]): -1,
                    int(ix["unserved"][k]): 1, int(ix["spill"][k]): -1})
        eq(row, demand[k])
        ub({int(ix["unserved"][k]): 1, int(ix["out"][k]): 1}, demand[k])
        for j, f in enumerate(FUELS):
            row = {int(ix["fuel"][k, j]): 1, int(ix["direct_short"][k, j]): -1}
            rhs = arrivals[k, j]-direct[k, j]
            if k == 0:
                rhs += state["fuel"][j]
            else:
                row[int(ix["fuel"][k-1, j])] = -1
            if f in {"coal", "gas"}:
                row[int(ix["gen"][k, GENERATORS.index(f)])] = 1/(GWH_PER_MTCE*case["fuel_efficiency"][f])
            eq(row, rhs)
        for j, s in enumerate(STORES):
            st = case["storage"][s]
            row = {int(ix["soc"][k, j]): 1, int(ix["ch"][k, j]): -st["eta_charge"],
                   int(ix["dis"][k, j]): 1/st["eta_discharge"]}
            rhs = state["soc"][s] if k == 0 else 0
            if k:
                row[int(ix["soc"][k-1, j])] = -1
            eq(row, rhs)
            ub({int(ix["ch"][k, j]): 1, int(ix["dis"][k, j]): 1}, st["power_gw"])
        row = {int(ix["debt"][k]): 1, int(ix["out"][k]): -1, int(ix["repay"][k]): 1}
        rhs = state["debt"] if k == 0 else 0
        if k:
            row[int(ix["debt"][k-1])] = -1
        eq(row, rhs)
        for f in ("coal", "gas"):
            j = GENERATORS.index(f)
            ramp = case["capacity_gw"][f]*cfg["ramp_fraction"][f]
            if k == 0:
                prev = state["thermal"][f]
                drop = max(0, state["previous_available"][f]-available[k, j])
                ub({int(ix["gen"][k, j]): 1}, prev+ramp)
                ub({int(ix["gen"][k, j]): -1}, max(ramp, drop)-prev)
            else:
                drop = max(0, available[k-1, j]-available[k, j])
                ub({int(ix["gen"][k, j]): 1, int(ix["gen"][k-1, j]): -1}, ramp)
                ub({int(ix["gen"][k, j]): -1, int(ix["gen"][k-1, j]): 1}, max(ramp, drop))
    if limits is not None:
        if "carbon_cap_mt" in limits:
            carbon = {}
            direct_constant = 0.0
            for k in range(nstep):
                for j, f in enumerate(FUELS):
                    carbon[int(ix["direct_short"][k, j])] = -FUEL_EMISSION_MT_PER_MTCE[f]
                    direct_constant += direct[k, j]*FUEL_EMISSION_MT_PER_MTCE[f]
                for f in ("coal", "gas"):
                    carbon[int(ix["gen"][k, GENERATORS.index(f)])] = FUEL_EMISSION_MT_PER_MTCE[f]/(GWH_PER_MTCE*case["fuel_efficiency"][f])
            ub(carbon, limits["carbon_cap_mt"]-direct_constant)
        if "power_unserved_cap_gwh" in limits:
            ub({int(i):1.0 for i in ix["unserved"]}, limits["power_unserved_cap_gwh"])
        if "direct_unserved_cap_mtce" in limits:
            ub({int(i):1.0 for i in ix["direct_short"].ravel()}, limits["direct_unserved_cap_mtce"])
    def matrix(rows):
        mat = lil_matrix((len(rows), offset))
        for i, row in enumerate(rows):
            for j, value in row.items():
                mat[i, j] = value
        return mat.tocsr()
    solution = linprog(c, A_ub=matrix(ub_rows), b_ub=np.array(ub_rhs),
                       A_eq=matrix(eq_rows), b_eq=np.array(eq_rhs), bounds=bounds,
                       method="highs")
    if not solution.success:
        raise InfeasiblePlan(start, solution.status, solution.message)
    return {name: solution.x[indices] if all_steps else solution.x[indices[0]] for name, indices in ix.items()}



def initial_state(case):
    return {"fuel": np.array(case["initial_fuel_stocks_mtce"], float),
            "thermal": deepcopy(case["initial_thermal_gw"]),
            "soc": {s: case["storage"][s]["initial_soc_gwh"] for s in STORES},
            "debt": 0.0, "previous_available": {f: case["available_gw"][0, GENERATORS.index(f)] for f in ("coal", "gas")}}


def execute_action(case, t, state, action):
    """Update the true plant, independent of the optimizer's predicted state."""
    row = {"hour": t, "demand_gw": case["demand_gw"][t],
           "stress_active": int(case["stress_mask"][t]),
           "shift_out_gw": float(action["out"]), "shift_in_gw": float(action["repay"]),
           "unserved_gw": float(action["unserved"]), "spill_gw": float(action["spill"])}
    for j, g in enumerate(GENERATORS):
        row[g+"_gw"], row[g+"_available_gw"] = action["gen"][j], case["available_gw"][t,j]
    for j, f in enumerate(FUELS):
        burn = row[f+"_gw"]/(GWH_PER_MTCE*case["fuel_efficiency"][f]) if f in {"coal", "gas"} else 0
        short = action["direct_short"][j]
        delivered = case["direct_fuel_demand_mtce"][t,j]-short
        state["fuel"][j] += case["fuel_arrivals_mtce"][t,j]-delivered-burn
        row.update({f+"_stock_mtce":state["fuel"][j], f+"_arrival_mtce":case["fuel_arrivals_mtce"][t,j],
                    f+"_direct_demand_mtce":case["direct_fuel_demand_mtce"][t,j],
                    f+"_direct_served_mtce":delivered, f+"_direct_unserved_mtce":short,
                    f+"_power_burn_mtce":burn})
    for j,s in enumerate(STORES):
        st=case["storage"][s]
        ch,dis=action["ch"][j],action["dis"][j]
        state["soc"][s] += st["eta_charge"]*ch-dis/st["eta_discharge"]
        row[s+"_charge_gw"],row[s+"_discharge_gw"],row[s+"_soc_gwh"]=ch,dis,state["soc"][s]
    state["debt"] += row["shift_out_gw"]-row["shift_in_gw"]
    row["shift_debt_gwh"]=state["debt"]
    row["served_power_gw"]=row["demand_gw"]-row["shift_out_gw"]+row["shift_in_gw"]-row["unserved_gw"]
    for f in ("coal","gas"):
        state["thermal"][f]=row[f+"_gw"]
        state["previous_available"][f]=case["available_gw"][t,GENERATORS.index(f)]
    return row


def physical_summary(case, table, strategy):
    result=summarize(case,table,strategy)
    result["physical_operating_cost_million_cny"]=(
        result["fuel_consumption_and_nonfossil_cost_million_cny"]+
        result["storage_cost_million_cny"]+result["shift_cost_million_cny"]+
        result["net_soc_draw_value_million_cny"]+result["stock_holding_cost_million_cny"])
    result["physical_cost_excludes"]="unserved-service penalty, carbon-price transfer and sunk installed-plant capital"
    result["information_available_hours"]=24
    result["used_planning_hours"] = 1 if strategy in {"fixed_rule","myopic_1h"} else (case["hours"] if strategy.startswith("offline") else 24)
    return result


def commitment_audit(summary, limits, tolerance=1e-5):
    mapping={"carbon_cap_mt":"total_co2_mt", "power_unserved_cap_gwh":"unserved_gwh",
             "direct_unserved_cap_mtce":"direct_fuel_unserved_mtce"}
    residuals={key:float(max(0,summary[mapping[key]]-value)) for key,value in (limits or {}).items()}
    return {"passed":all(x<=tolerance for x in residuals.values()),"limits":limits or {},
            "violations":residuals,"tolerance":tolerance}


def prefix_audit(case, table):
    """Audit executed physics without inventing a not-yet-due final deadline."""
    if table.empty:
        return {"passed":True,"scope":"EMPTY_EXECUTED_PREFIX","completed_hours":0}
    count=len(table)
    clipped=deepcopy(case)
    clipped["hours"]=count
    for key,value in clipped.items():
        if isinstance(value,np.ndarray) and value.ndim>0 and len(value)==case["hours"]:
            clipped[key]=value[:count]
    checks=audit(clipped,table)
    due=((table.hour.to_numpy()+1)%case["config"]["deadline_hours"])==0
    checks["debt_deadline_gwh"]=float(np.abs(table.loc[due,"shift_debt_gwh"]).max()) if due.any() else 0.0
    check_values=[v for k,v in checks.items() if k not in {"passed","all_finite","tolerance","interpretation","storage_simultaneous_operation_gw"}]
    checks["passed"]=bool(checks["all_finite"] and all(v<=checks["tolerance"] for v in check_values))
    checks.update(scope="EXECUTED_PREFIX_ONLY_NOT_FULL_HORIZON",completed_hours=count)
    return checks


def simulate(case, strategy="coordinated_24h", limits=None):
    """Same 24h information interface, actual first-action state transitions.

    Budget pacing is explicitly proportional and can be infeasible. A failure
    stops execution and returns its prefix; it never masquerades as 720h success.
    Legacy aliases delegate directly to the frozen V3 implementation.
    """
    if strategy in {"coordinated","decentralized"}:
        if limits is not None:
            raise ValueError("Legacy parity aliases do not accept V4 limits")
        result=v3.simulate(case,strategy)
        result["summary"]=physical_summary(case,result["timeseries"],strategy)
        result["status"]="COMPLETED"
        result["commitments"]={"passed":True,"limits":{},"violations":{}}
        return result
    if strategy not in {"fixed_rule","myopic_1h","coordinated_24h","capped_24h","noflex_24h","carbon_budget_remaining_total"}:
        raise ValueError(f"Unknown V4 strategy {strategy}")
    if (strategy in {"capped_24h","carbon_budget_remaining_total"}) != (limits is not None):
        raise ValueError("Carbon-constrained strategies require explicit limits")
    if limits is not None and (set(limits)-{"carbon_cap_mt","power_unserved_cap_gwh","direct_unserved_cap_mtce"} or
        any(not np.isfinite(x) or x<0 for x in limits.values())):
        raise ValueError("Limits must be known finite nonnegative cumulative quantities")
    if case["config"]["forecast_hours"]!=24:
        raise ValueError("V4 online experiment fixes the common information interface at 24h")
    state,records=initial_state(case),[]
    spent={k:0.0 for k in (limits or {})}
    start=time.perf_counter()
    for t in range(case["hours"]):
        n=min(1 if strategy in {"fixed_rule","myopic_1h"} else 24,case["hours"]-t)
        window_limits={key:max(0,value-spent[key])*(n/(case["hours"]-t) if key=="carbon_cap_mt" and strategy=="capped_24h" else 1.0) for key,value in (limits or {}).items()}
        try:
            if strategy=="coordinated_24h":
                action=v3._plan(case,t,state,"coordinated")
            else:
                action=_solve(case,t,state,strategy,n,window_limits or None)
        except InfeasiblePlan as exc:
            table=pd.DataFrame(records)
            return {"status":"INFEASIBLE_STOPPED", "timeseries":table,
                    "summary":{"strategy":strategy,"completed_hours":len(records),"hours":case["hours"],
                               "failed_hour":t,"runtime_seconds":time.perf_counter()-start},
                    "audit":prefix_audit(case,table),
                    "commitments":{"passed":False,"limits":limits,"spent_at_failure":spent},
                    "failure":{"hour":t,"solver_status":exc.status,"message":exc.solver_message,
                               "window_limits":window_limits}}
        row=execute_action(case,t,state,action)
        records.append(row)
        if limits:
            values={"carbon_cap_mt":sum((row[f+"_direct_served_mtce"]+row[f+"_power_burn_mtce"])*FUEL_EMISSION_MT_PER_MTCE[f] for f in FUELS),
                    "power_unserved_cap_gwh":row["unserved_gw"],
                    "direct_unserved_cap_mtce":sum(row[f+"_direct_unserved_mtce"] for f in FUELS)}
            for key in spent:
                spent[key]+=values[key]
    table=pd.DataFrame(records)
    checks=audit(case,table)
    if not checks["passed"]:
        raise AssertionError(f"V4 physical state audit failed: {checks}")
    summary=physical_summary(case,table,strategy)
    summary["runtime_seconds"]=time.perf_counter()-start
    return {"status":"COMPLETED","timeseries":table,"summary":summary,"audit":checks,
            "commitments":commitment_audit(summary,limits)}


def solve_offline(case, limits, objective="physical_cost"):
    """One full-information LP: explicitly an unattainable-information bound."""
    start=time.perf_counter()
    state=initial_state(case)
    strategy="offline_"+objective
    try:
        plan=_solve(case,0,state,strategy,case["hours"],limits,objective,all_steps=True)
    except InfeasiblePlan as exc:
        return {"status":"INFEASIBLE","timeseries":pd.DataFrame(),
                "summary":{"strategy":strategy,"hours":case["hours"]},"audit":{"passed":False},
                "commitments":{"passed":False,"limits":limits},
                "failure":{"solver_status":exc.status,"message":exc.solver_message}}
    records=[]
    for t in range(case["hours"]):
        records.append(execute_action(case,t,state,{key:val[t] for key,val in plan.items()}))
    table=pd.DataFrame(records)
    checks=audit(case,table)
    if not checks["passed"]:
        raise AssertionError(f"Offline exported plan failed independent state audit: {checks}")
    summary=physical_summary(case,table,strategy)
    summary.update(information_available_hours=case["hours"], information_contract="OFFLINE_FULL_HORIZON_PERFECT_INFORMATION_OPTIMISTIC_BOUND",
                   runtime_seconds=time.perf_counter()-start)
    return {"status":"COMPLETED","timeseries":table,"summary":summary,"audit":checks,
            "commitments":commitment_audit(summary,limits)}
