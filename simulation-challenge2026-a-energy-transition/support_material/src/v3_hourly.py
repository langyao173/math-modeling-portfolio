"""Fuel-conserving, state-feedback energy-security experiments.

Annual observations identify the initial scale, not the hourly chronology. The
hourly demand/weather/arrival profiles are declared engineering experiments.
Both policies see the same 24-hour forecast and execute only its first action.
The decentralized policy commits storage locally before clearing the remaining
market; the coordinated policy clears storage, fuel use and flexible load jointly.
AI-assisted implementation: OpenAI Codex, 2026-10-16.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

GENERATORS = ("coal", "gas", "nuclear", "hydro", "wind", "solar", "biomass")
FUELS = ("coal", "oil", "gas")
STORES = ("battery", "pumped")
GWH_PER_MTCE = 8141.0
FUEL_EMISSION_MT_PER_MTCE = {
    fuel: carbon_content_kg_gj*29.3076/1000
    for fuel, carbon_content_kg_gj in zip(FUELS, (94.6, 73.3, 56.1))
}


def make_case(snapshot, shock="normal", hours=720):
    """Create common raw exogenous inputs, never old dispatch decisions.

    GW/GWh, Mtce fuel, million CNY. All fuel arrays use FUELS order. Stocks,
    SOC and flexible-load debt in exported traces are end-of-hour states.
    """
    if shock not in {"normal", "import_cut", "renewable_drought", "compound"}:
        raise ValueError(f"Unknown shock: {shock}")
    if not isinstance(hours, (int, np.integer)) or hours < 2:
        raise ValueError("hours must be an integer >= 2")
    required = {"year", "electricity_demand_twh", "capacity_gw",
                "fuel_domestic_supply_mtce_per_year", "fuel_import_supply_mtce_per_year",
                "fuel_initial_stocks_mtce", "direct_fuel_demand_mtce_per_year",
                "fuel_efficiency", "carbon_price_cny_t"}
    if required - set(snapshot):
        raise ValueError(f"Missing snapshot fields: {sorted(required-set(snapshot))}")
    capacity_names = set(GENERATORS) | {f"{s}_{k}" for s in STORES for k in ("power", "energy")}
    cap = {g: float(snapshot["capacity_gw"].get(g, 0)) for g in capacity_names}
    if any(not np.isfinite(x) or x < 0 for x in cap.values()):
        raise ValueError("Capacities must be finite and nonnegative")
    for key in ("fuel_domestic_supply_mtce_per_year", "fuel_import_supply_mtce_per_year",
                "fuel_initial_stocks_mtce", "direct_fuel_demand_mtce_per_year"):
        if set(snapshot[key]) != set(FUELS) or any(
                not np.isfinite(v) or v < 0 for v in snapshot[key].values()):
            raise ValueError(f"{key} must contain three finite nonnegative fuel quantities")
    if set(snapshot["fuel_efficiency"]) != {"coal", "gas"} or any(
            not np.isfinite(v) or not 0 < v <= 1 for v in snapshot["fuel_efficiency"].values()):
        raise ValueError("Coal/gas efficiencies must lie in (0,1]")
    if not np.isfinite(snapshot["electricity_demand_twh"]) or snapshot["electricity_demand_twh"] <= 0:
        raise ValueError("Electricity demand must be positive and finite")
    if not np.isfinite(snapshot["carbon_price_cny_t"]) or snapshot["carbon_price_cny_t"] < 0:
        raise ValueError("Carbon price must be finite and nonnegative")
    t = np.arange(hours)
    daily = 1 + .12*np.cos(2*np.pi*(t % 24-15)/24) + .03*np.cos(4*np.pi*(t % 24-19)/24)
    demand = float(snapshot["electricity_demand_twh"])*1000/8760*daily
    wind_cf = np.clip(.29*(1+.35*np.sin(2*np.pi*t/72)+.12*np.sin(2*np.pi*t/17)), 0, .85)
    solar_shape = np.maximum(0, np.sin(np.pi*(t % 24-6)/12))
    solar_cf = solar_shape*.17/(np.maximum(0, np.sin(np.pi*(np.arange(24)-6)/12)).mean())
    cf = {"coal": .9, "gas": .92, "nuclear": .88, "hydro": .35,
          "wind": wind_cf, "solar": solar_cf, "biomass": .65}
    available = np.column_stack([np.broadcast_to(cap[g]*cf[g], hours) for g in GENERATORS])
    biomass_budget = float(snapshot.get("biomass_generation_budget_twh_per_year", 0.0))
    if cap["biomass"] > 0 and "biomass_generation_budget_twh_per_year" not in snapshot:
        raise ValueError("Nonzero biomass capacity requires annual feedstock generation budget")
    if not np.isfinite(biomass_budget) or biomass_budget < 0:
        raise ValueError("Biomass generation budget must be finite and nonnegative")
    # Explicit uniform feedstock allocation: an annual energy bound becomes a
    # common hourly upper bound; no claim of measured biomass logistics.
    available[:, GENERATORS.index("biomass")] = np.minimum(
        available[:, GENERATORS.index("biomass")], biomass_budget*1000/8760)
    domestic = np.tile([snapshot["fuel_domestic_supply_mtce_per_year"][f]/8760 for f in FUELS], (hours, 1))
    imports = np.tile([snapshot["fuel_import_supply_mtce_per_year"][f]/8760 for f in FUELS], (hours, 1))
    direct = np.tile([snapshot["direct_fuel_demand_mtce_per_year"][f]/8760 for f in FUELS], (hours, 1))
    # First five days normal; twenty days disruption; recovery in the last five.
    start, stop = (min(120, hours//4), min(600, max(hours//4+1, 5*hours//6)))
    mask = (t >= start) & (t < stop)
    if shock in {"import_cut", "compound"}:
        imports[mask] *= np.array([.50, .05, .05])
    if shock in {"renewable_drought", "compound"}:
        available[mask, GENERATORS.index("wind")] *= .25
        available[mask, GENERATORS.index("solar")] *= .55
        available[mask, GENERATORS.index("hydro")] *= .70
    if shock == "compound":
        demand[mask] *= 1.08
    stores = {}
    for s, rte in (("battery", .90), ("pumped", .78)):
        p, e = cap[s+"_power"], cap[s+"_energy"]
        if (p == 0) != (e == 0):
            raise ValueError(f"{s} requires both power and energy, or both zero")
        stores[s] = {"power_gw": p, "energy_gwh": e, "initial_soc_gwh": .5*e,
                     "eta_charge": float(np.sqrt(rte)), "eta_discharge": float(np.sqrt(rte))}
    return {"year": int(snapshot["year"]), "scenario": snapshot.get("scenario", "unspecified"),
            "hours": hours, "dt_h": 1.0, "shock": shock,
            "demand_gw": demand, "available_gw": available,
            "fuel_domestic_arrivals_mtce": domestic, "fuel_import_arrivals_mtce": imports,
            "fuel_arrivals_mtce": domestic+imports, "direct_fuel_demand_mtce": direct,
            "initial_fuel_stocks_mtce": np.array([snapshot["fuel_initial_stocks_mtce"][f] for f in FUELS]),
            "initial_thermal_gw": {"coal": 0.0, "gas": 0.0},
            "fuel_efficiency": deepcopy(snapshot["fuel_efficiency"]), "capacity_gw": cap,
            "storage": stores, "stress_mask": mask,
            "biomass_generation_budget_twh_per_year": biomass_budget,
            "config": {"forecast_hours": 24, "deadline_hours": 24,
                       "ramp_fraction": {"coal": .15, "gas": .70},
                       "flex_fraction": .08, "repay_fraction": .20,
                       "generation_cost_cny_kwh": {"coal": .32, "gas": .65, "nuclear": .09,
                           "hydro": .025, "wind": 0.0, "solar": 0.0, "biomass": .35},
                       "carbon_price_cny_t": float(snapshot["carbon_price_cny_t"]),
                       "unserved_cost_cny_kwh": 30.0, "direct_shortage_cost_cny_kwh_th": 30.0,
                       "storage_cycle_cost_cny_kwh": .005, "shift_cost_cny_kwh": .003,
                       "fuel_value_million_cny_mtce": {
                           "coal": .32*GWH_PER_MTCE*snapshot["fuel_efficiency"]["coal"],
                           "oil": 3500.0,
                           "gas": .65*GWH_PER_MTCE*snapshot["fuel_efficiency"]["gas"]},
                       "stored_electricity_value_million_cny_gwh": .32,
                       "stock_holding_rate_per_year": .05},
            "evidence_class": "DESIGNED_HOURLY_DYNAMIC_EXPERIMENT",
            "information_contract": "COMMON_24H_PERFECT_FORECAST_ROLLING_FIRST_ACTION",
            "boundary": "Single aggregate bus, continuous ramp-constrained plants; no unit commitment, spatial network or empirical hourly calibration. Hydro is an exogenous available-flow bound. Arrivals are domestic/import flows, not unconstrained balancing fuel. Daily flexible-load deadlines and final zero debt; storage terminal state is reported, not forced cyclic."}


def _local_storage(case, start, count, state):
    """Independent storage agents use present local surplus/deficit priority.

    They do not price future fuel scarcity. Plans respect actual SOC, efficiency
    and shared power limits, and charge only from nonfossil available surplus.
    The remaining common market LP still clears generation, fuels and flexible load.
    """
    available = case["available_gw"][start:start+count]
    net = case["demand_gw"][start:start+count]-available[:, 2:].sum(axis=1)
    result = {}
    for s in STORES:
        store, soc = case["storage"][s], state["soc"][s]
        ch, dis = np.zeros(count), np.zeros(count)
        for k in range(count):
            if net[k] < 0:
                ch[k] = min(-net[k], store["power_gw"], max(0, store["energy_gwh"]-soc)/store["eta_charge"])
                soc += ch[k]*store["eta_charge"]
                net[k] += ch[k]
            else:
                dis[k] = min(net[k], store["power_gw"], max(0, soc)*store["eta_discharge"])
                soc -= dis[k]/store["eta_discharge"]
                net[k] -= dis[k]
        result[s] = (ch, dis)
    return result


def _plan(case, start, state, strategy):
    nstep = min(int(case["config"]["forecast_hours"]), case["hours"]-start)
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
    local = _local_storage(case, start, nstep, state) if strategy == "decentralized" else None
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
        bounds[ix["out"][k]] = (0, float(cfg["flex_fraction"]*demand[k]))
        bounds[ix["repay"][k]] = (0, float(cfg["repay_fraction"]*demand[k]))
        c[ix["out"][k]] = c[ix["repay"][k]] = cfg["shift_cost_cny_kwh"]
        c[ix["unserved"][k]] = cfg["unserved_cost_cny_kwh"]*urgency
        # Debt is a real obligation: no shedding or resetting it at the deadline.
        if (start+k+1) % cfg["deadline_hours"] == 0 or start+k == case["hours"]-1:
            bounds[ix["debt"][k]] = (0, 0)
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
        raise RuntimeError(f"v3 {strategy} hour {start} LP status {solution.status}: {solution.message}")
    return {name: solution.x[indices[0]] for name, indices in ix.items()}


def simulate(case, strategy="coordinated"):
    """Roll forward actual fuel/SOC/output/debt states, not a full-window plan."""
    if strategy not in {"coordinated", "decentralized"}:
        raise ValueError(f"Unknown strategy: {strategy}")
    state = {"fuel": np.array(case["initial_fuel_stocks_mtce"], float),
             "thermal": deepcopy(case["initial_thermal_gw"]),
             "soc": {s: case["storage"][s]["initial_soc_gwh"] for s in STORES}, "debt": 0.0,
             "previous_available": {f: case["available_gw"][0, GENERATORS.index(f)] for f in ("coal", "gas")}}
    records = []
    started = time.perf_counter()
    for t in range(case["hours"]):
        action = _plan(case, t, state, strategy)
        row = {"hour": t, "demand_gw": case["demand_gw"][t],
               "stress_active": int(case["stress_mask"][t]),
               "shift_out_gw": float(action["out"]), "shift_in_gw": float(action["repay"]),
               "unserved_gw": float(action["unserved"]), "spill_gw": float(action["spill"])}
        for j, g in enumerate(GENERATORS):
            row[g+"_gw"] = action["gen"][j]
            row[g+"_available_gw"] = case["available_gw"][t, j]
        # State transitions below are physical equations, separate from LP states.
        for j, f in enumerate(FUELS):
            burn = row[f+"_gw"]/(GWH_PER_MTCE*case["fuel_efficiency"][f]) if f in {"coal", "gas"} else 0
            direct_short = action["direct_short"][j]
            delivered = case["direct_fuel_demand_mtce"][t, j]-direct_short
            state["fuel"][j] += case["fuel_arrivals_mtce"][t, j]-delivered-burn
            row[f+"_stock_mtce"] = state["fuel"][j]
            row[f+"_arrival_mtce"] = case["fuel_arrivals_mtce"][t, j]
            row[f+"_direct_demand_mtce"] = case["direct_fuel_demand_mtce"][t, j]
            row[f+"_direct_served_mtce"] = delivered
            row[f+"_direct_unserved_mtce"] = direct_short
            row[f+"_power_burn_mtce"] = burn
        for j, s in enumerate(STORES):
            st = case["storage"][s]
            ch, dis = action["ch"][j], action["dis"][j]
            state["soc"][s] += st["eta_charge"]*ch-dis/st["eta_discharge"]
            row[s+"_charge_gw"], row[s+"_discharge_gw"] = ch, dis
            row[s+"_soc_gwh"] = state["soc"][s]
        state["debt"] += row["shift_out_gw"]-row["shift_in_gw"]
        row["shift_debt_gwh"] = state["debt"]
        row["served_power_gw"] = row["demand_gw"]-row["shift_out_gw"]+row["shift_in_gw"]-row["unserved_gw"]
        for f in ("coal", "gas"):
            state["thermal"][f] = row[f+"_gw"]
            state["previous_available"][f] = case["available_gw"][t, GENERATORS.index(f)]
        records.append(row)
    table = pd.DataFrame(records)
    checks = audit(case, table)
    if not checks["passed"]:
        raise RuntimeError(f"v3 independent physical audit failed: {checks}")
    summary = summarize(case, table, strategy)
    summary["runtime_seconds"] = time.perf_counter()-started
    return {"timeseries": table, "summary": summary, "audit": checks}


def audit(case, table, tolerance=1e-5):
    """Reconstruct all balances from exported actions; never inspect LP matrices."""
    residuals = {}
    h = case["hours"]
    if len(table) != h or not np.array_equal(table.hour.to_numpy(), np.arange(h)):
        return {"passed": False, "reason": "hour index mismatch"}
    numeric = table.select_dtypes(include=[np.number]).to_numpy()
    finite = bool(np.isfinite(numeric).all())
    gen = table[[g+"_gw" for g in GENERATORS]].to_numpy()
    residuals["generation_bound_gw"] = float(max(0, -gen.min(), (gen-case["available_gw"]).max()))
    residuals["biomass_feedstock_budget_violation_gwh"] = float(max(0,
        gen[:, GENERATORS.index("biomass")].sum()-case["biomass_generation_budget_twh_per_year"]*1000*h/8760))
    balance = gen.sum(axis=1)+table.unserved_gw.to_numpy()-table.spill_gw.to_numpy()-case["demand_gw"]+table.shift_out_gw.to_numpy()-table.shift_in_gw.to_numpy()
    for j, f in enumerate(FUELS):
        stocks = np.r_[case["initial_fuel_stocks_mtce"][j], table[f+"_stock_mtce"].to_numpy()]
        short = table[f+"_direct_unserved_mtce"].to_numpy()
        served = case["direct_fuel_demand_mtce"][:, j]-short
        burn = gen[:, GENERATORS.index(f)]/(GWH_PER_MTCE*case["fuel_efficiency"][f]) if f in {"coal", "gas"} else np.zeros(h)
        residuals[f+"_fuel_balance_mtce"] = float(np.abs(np.diff(stocks)-case["fuel_arrivals_mtce"][:, j]+served+burn).max())
        residuals[f+"_fuel_bound_mtce"] = float(max(0, -stocks.min(), -short.min(), (short-case["direct_fuel_demand_mtce"][:, j]).max()))
        # Exported derived quantities cannot be silently altered either.
        for column, expected in ((f+"_direct_served_mtce", served), (f+"_power_burn_mtce", burn),
                                 (f+"_arrival_mtce", case["fuel_arrivals_mtce"][:, j]),
                                 (f+"_direct_demand_mtce", case["direct_fuel_demand_mtce"][:, j])):
            residuals[column+"_export_error"] = float(np.abs(table[column].to_numpy()-expected).max())
    for s in STORES:
        st = case["storage"][s]
        ch, dis = (table[s+"_"+k+"_gw"].to_numpy() for k in ("charge", "discharge"))
        soc = np.r_[st["initial_soc_gwh"], table[s+"_soc_gwh"].to_numpy()]
        residuals[s+"_state_residual_gwh"] = float(np.abs(np.diff(soc)-st["eta_charge"]*ch+dis/st["eta_discharge"]).max())
        residuals[s+"_bounds"] = float(max(0, -soc.min(), (soc-st["energy_gwh"]).max(), -ch.min(), -dis.min(), (ch+dis-st["power_gw"]).max()))
        balance += dis-ch
    residuals["power_balance_gw"] = float(np.abs(balance).max())
    residuals["demand_export_error_gw"] = float(np.abs(table.demand_gw.to_numpy()-case["demand_gw"]).max())
    debt = np.r_[0, table.shift_debt_gwh.to_numpy()]
    out, repay = table.shift_out_gw.to_numpy(), table.shift_in_gw.to_numpy()
    residuals["debt_state_residual_gwh"] = float(np.abs(np.diff(debt)-out+repay).max())
    deadline = (np.arange(h)+1) % case["config"]["deadline_hours"] == 0
    deadline[-1] = True
    residuals["debt_deadline_gwh"] = float(np.abs(debt[1:][deadline]).max())
    residuals["flex_bounds"] = float(max(0, -debt.min(), -out.min(), -repay.min(),
        (out-case["config"]["flex_fraction"]*case["demand_gw"]).max(),
        (repay-case["config"]["repay_fraction"]*case["demand_gw"]).max()))
    residuals["slack_bounds"] = float(max(0, -table.unserved_gw.min(), -table.spill_gw.min(),
        (table.unserved_gw.to_numpy()+out-case["demand_gw"]).max()))
    for f in ("coal", "gas"):
        j = GENERATORS.index(f)
        g = np.r_[case["initial_thermal_gw"][f], table[f+"_gw"].to_numpy()]
        ramp = case["capacity_gw"][f]*case["config"]["ramp_fraction"][f]
        avail = np.r_[case["available_gw"][0, j], case["available_gw"][:, j]]
        forced = np.maximum(0, -np.diff(avail))
        residuals[f+"_ramp_violation_gw"] = float(max(0, (np.diff(g)-ramp).max(), (-np.diff(g)-np.maximum(ramp, forced)).max()))
    residuals["served_power_export_error_gw"] = float(np.abs(table.served_power_gw.to_numpy()-(case["demand_gw"]-out+repay-table.unserved_gw.to_numpy())).max())
    return {"passed": finite and all(v <= tolerance for v in residuals.values()),
            "all_finite": finite, "tolerance": tolerance, **residuals,
            "storage_simultaneous_operation_gw": float(max(np.minimum(table[s+"_charge_gw"], table[s+"_discharge_gw"]).max() for s in STORES)),
            "interpretation": "Actions and end states are independently reconstructed; state-accounting verification is not empirical hourly validation."}


def summarize(case, table, strategy):
    cfg = case["config"]
    electric_co2 = sum(table[f+"_power_burn_mtce"].sum()*FUEL_EMISSION_MT_PER_MTCE[f] for f in ("coal", "gas"))
    direct_co2 = sum(table[f+"_direct_served_mtce"].sum()*FUEL_EMISSION_MT_PER_MTCE[f] for f in FUELS)
    # Arrivals plus inventory draw equal consumption, so initial stocks are not
    # free fuel and terminal fuel assets are not charged twice.
    fuel_arrival_value, fuel_inventory_draw_value = 0.0, 0.0
    for j, f in enumerate(FUELS):
        price = cfg["fuel_value_million_cny_mtce"][f]
        fuel_arrival_value += case["fuel_arrivals_mtce"][:, j].sum()*price
        fuel_inventory_draw_value += (case["initial_fuel_stocks_mtce"][j]-table[f+"_stock_mtce"].iloc[-1])*price
    nonfuel_generation_cost = sum(table[g+"_gw"].sum()*cfg["generation_cost_cny_kwh"][g] for g in GENERATORS if g not in {"coal", "gas"})
    fuel_cost = fuel_arrival_value+fuel_inventory_draw_value+nonfuel_generation_cost
    soc_draw_value = sum((case["storage"][s]["initial_soc_gwh"]-table[s+"_soc_gwh"].iloc[-1])
                        *cfg["stored_electricity_value_million_cny_gwh"] for s in STORES)
    store_cost = sum((table[s+"_charge_gw"].sum()+table[s+"_discharge_gw"].sum())*cfg["storage_cycle_cost_cny_kwh"] for s in STORES)
    shift_cost = (table.shift_out_gw.sum()+table.shift_in_gw.sum())*cfg["shift_cost_cny_kwh"]
    direct_short = sum(table[f+"_direct_unserved_mtce"].sum() for f in FUELS)
    shortage_cost = table.unserved_gw.sum()*cfg["unserved_cost_cny_kwh"]+direct_short*GWH_PER_MTCE*cfg["direct_shortage_cost_cny_kwh_th"]
    carbon_cost = (electric_co2+direct_co2)*cfg["carbon_price_cny_t"]
    initial_inventory_value = sum(case["initial_fuel_stocks_mtce"][j]*cfg["fuel_value_million_cny_mtce"][f] for j, f in enumerate(FUELS))
    holding_cost = initial_inventory_value*cfg["stock_holding_rate_per_year"]*case["hours"]/8760
    row = {"scenario": case["scenario"], "year": case["year"], "shock": case["shock"], "strategy": strategy,
           "hours": case["hours"], "forecast_hours": cfg["forecast_hours"],
           "information_contract": case["information_contract"], "evidence_class": case["evidence_class"],
           "unserved_gwh": float(table.unserved_gw.sum()), "peak_unserved_gw": float(table.unserved_gw.max()),
           "loss_of_load_hours": int((table.unserved_gw > 1e-5).sum()),
           "direct_fuel_unserved_mtce": float(direct_short), "electricity_co2_mt": float(electric_co2),
           "direct_fuel_co2_mt": float(direct_co2), "total_co2_mt": float(electric_co2+direct_co2),
           "shift_out_gwh": float(table.shift_out_gw.sum()), "shift_in_gwh": float(table.shift_in_gw.sum()),
           "terminal_shift_debt_gwh": float(table.shift_debt_gwh.iloc[-1]),
           "fuel_consumption_and_nonfossil_cost_million_cny": float(fuel_cost), "storage_cost_million_cny": float(store_cost),
           "shift_cost_million_cny": float(shift_cost), "shortage_penalty_million_cny": float(shortage_cost),
           "carbon_cost_million_cny": float(carbon_cost),
           "total_objective_components_million_cny": float(fuel_cost+store_cost+shift_cost+shortage_cost+carbon_cost),
           "fuel_arrival_purchase_value_million_cny": float(fuel_arrival_value),
           "net_fuel_inventory_draw_value_million_cny": float(fuel_inventory_draw_value),
           "net_soc_draw_value_million_cny": float(soc_draw_value),
           "resource_accounted_cost_million_cny": float(fuel_cost+store_cost+shift_cost+shortage_cost+carbon_cost+soc_draw_value),
           "initial_fuel_inventory_value_million_cny": float(initial_inventory_value),
           "terminal_fuel_inventory_value_million_cny": float(initial_inventory_value-fuel_inventory_draw_value),
           "stock_holding_cost_million_cny": float(holding_cost),
           "resource_plus_holding_cost_million_cny": float(fuel_cost+store_cost+shift_cost+shortage_cost+carbon_cost+soc_draw_value+holding_cost),
           "stock_holding_assumption": "5 percent/year on committed initial fuel inventory value, prorated by modeled hours; engineering capital charge, not observed storage tariff. Reported separately and not a dispatch decision cost.",
           "valuation_scope": "Declared constant engineering unit values. Fuel arrivals plus initial-minus-terminal stock equal consumption; coal/gas unit values reproduce generation fuel costs. Nonfossil operating costs are separate. SOC draw is charged at a common 0.32 million CNY/GWh. Not observed market prices or national welfare.",
           "boundary": case["boundary"]}
    for f in FUELS:
        row[f+"_direct_unserved_mtce"] = float(table[f+"_direct_unserved_mtce"].sum())
        row[f+"_initial_stock_mtce"] = float(case["initial_fuel_stocks_mtce"][FUELS.index(f)])
        row[f+"_terminal_stock_mtce"] = float(table[f+"_stock_mtce"].iloc[-1])
        empty = table.loc[table[f+"_stock_mtce"] < 1e-7, "hour"]
        row[f+"_first_empty_hour"] = int(empty.iloc[0]) if len(empty) else None
    for s in STORES:
        row[s+"_initial_soc_gwh"] = case["storage"][s]["initial_soc_gwh"]
        row[s+"_terminal_soc_gwh"] = float(table[s+"_soc_gwh"].iloc[-1])
    return row


def run_horizon_diagnostic(snapshots_path, protocol_path, output):
    """Post-hoc explanation of two predeclared failures, not strategy selection.

    Reuse the four already completed 24h inventory cases, then execute all four
    72h cases. Within each horizon both policies receive identical information.
    Across horizons the information itself is stronger; this is not an isolated
    algorithm comparison and cannot establish that a longer horizon always wins.
    """
    snapshots_path, protocol_path, output = map(Path, (snapshots_path, protocol_path, output))
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    snapshot_hash = hashlib.sha256(snapshots_path.read_bytes()).hexdigest()
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    snapshots = json.loads(snapshots_path.read_text(encoding="utf-8"))
    design = json.loads(protocol_path.read_text(encoding="utf-8"))["posthoc_forecast_diagnostic"]
    completed = json.loads((output/"hourly_audit.json").read_text(encoding="utf-8"))
    if (not completed["passed"] or completed["source_sha256"] != source_hash
            or completed["snapshot_sha256"] != snapshot_hash or completed["protocol_sha256"] != protocol_hash):
        raise ValueError("Complete the current frozen 64+12 cohort before reusing its 24h traces")
    original = {r["case_id"]: r for r in completed["inventory_audits"]}
    rows, checks = [], []
    for condition in design["cases"]:
        year, days = condition["year"], condition["initial_stock_days"]
        snap = snapshots[f"{design['scenario']}_{year}"]
        for horizon in design["forecast_hours"]:
            case = make_case(snap, design["shock"], design["hours"])
            case["initial_fuel_stocks_mtce"] = np.array([
                (snap["fuel_domestic_supply_mtce_per_year"][f]+snap["fuel_import_supply_mtce_per_year"][f])*days/365
                for f in FUELS])
            case["config"]["forecast_hours"] = horizon
            case["information_contract"] = f"COMMON_{horizon}H_PERFECT_FORECAST_ROLLING_FIRST_ACTION"
            for strategy in design["strategies"]:
                case_id = f"{design['scenario']}_{year}_stock{days}_h{horizon}_{strategy}"
                if horizon == 24:
                    original_id = f"inventory_{design['scenario']}_{year}_stock{days}_{design['shock']}_{strategy}"
                    path = output/("hourly_"+original_id+".csv")
                    if hashlib.sha256(path.read_bytes()).hexdigest() != original[original_id]["trace_sha256"]:
                        raise AssertionError("Cannot reuse a changed 24h trace")
                    table = pd.read_csv(path)
                    summary = summarize(case, table, strategy)
                    reused_from = str(path).replace("\\", "/")
                else:
                    result = simulate(case, strategy)
                    table, summary = result["timeseries"], result["summary"]
                    path = output/("hourly_horizon_"+case_id+".csv")
                    table.to_csv(path, index=False)
                    table = pd.read_csv(path)
                    reused_from = None
                physical = audit(case, table)
                if not physical["passed"]:
                    raise AssertionError(f"Forecast diagnostic CSV failed independent audit: {case_id}")
                rows.append({**summary, "initial_stock_days_of_normal_arrivals": days,
                             "diagnostic_status": "POST_HOC_MECHANISM_DIAGNOSTIC",
                             "reused_from": reused_from})
                checks.append({"case_id": case_id, "forecast_hours": horizon,
                               "trace": str(path).replace("\\", "/"),
                               "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "reused_existing_24h_trace": horizon == 24, **physical})
                print("POSTHOC", case_id, "PASS", flush=True)
    if (hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source_hash
            or hashlib.sha256(snapshots_path.read_bytes()).hexdigest() != snapshot_hash
            or hashlib.sha256(protocol_path.read_bytes()).hexdigest() != protocol_hash):
        raise AssertionError("Diagnostic inputs/source changed during execution")
    pd.DataFrame(rows).to_csv(output/"hourly_horizon_diagnostic_summary.csv", index=False)
    report = {"status": "POST_HOC_MECHANISM_DIAGNOSTIC", "passed": all(r["passed"] for r in checks),
              "cases": len(checks), "reused_24h_cases": sum(r["reused_existing_24h_trace"] for r in checks),
              "snapshot_sha256": snapshot_hash, "source_sha256": source_hash, "protocol_sha256": protocol_hash,
              "warning": "72h provides more future information than 24h. Differences across horizons are not algorithm-only gains; no result-dependent horizon selection.",
              "audits": checks, "created_utc": datetime.now(timezone.utc).isoformat()}
    (output/"hourly_horizon_diagnostic_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=Path("outputs/v3/snapshots.json"))
    parser.add_argument("--hours", type=int, default=720)
    parser.add_argument("--output", type=Path, default=Path("outputs/v3"))
    parser.add_argument("--protocol", type=Path, default=Path("config/v3_experiments.json"))
    parser.add_argument("--horizon-diagnostic-only", action="store_true")
    args = parser.parse_args()
    if args.horizon_diagnostic_only:
        run_horizon_diagnostic(args.snapshots, args.protocol, args.output)
        return
    input_hash = hashlib.sha256(args.snapshots.read_bytes()).hexdigest()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol_hash = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    snapshots = json.loads(args.snapshots.read_text(encoding="utf-8"))
    design = protocol["hourly"]
    if args.hours != design["hours"]:
        raise ValueError("CLI fixed-cohort hours must match protocol; use make_case API for exploratory fixtures")
    args.output.mkdir(parents=True, exist_ok=True)
    results, audits = [], []
    def export(case, strategy, case_id):
        result = simulate(case, strategy)
        trace_path = args.output/("hourly_"+case_id+".csv")
        result["timeseries"].to_csv(trace_path, index=False)
        roundtrip_audit = audit(case, pd.read_csv(trace_path))
        if not roundtrip_audit["passed"]:
            raise AssertionError(f"Exported trace failed independent audit: {case_id}")
        evidence = {"case_id": case_id, "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                    "csv_roundtrip_passed": True, **roundtrip_audit}
        print(case_id, "PASS", flush=True)
        return result["summary"], evidence
    for scenario in design["scenarios"]:
        for year in design["years"]:
            snap = snapshots[f"{scenario}_{year}"]
            for shock in design["shocks"]:
                case = make_case(snap, shock, args.hours)
                for strategy in design["strategies"]:
                    result, evidence = export(case, strategy, f"{scenario}_{year}_{shock}_{strategy}")
                    results.append(result); audits.append(evidence)
    pd.DataFrame(results).to_csv(args.output/"hourly_summary.csv", index=False)
    reserve_results, reserve_audits = [], []
    reserve_design = protocol["inventory_policy_experiment"]
    for year in reserve_design["years"]:
        snap = snapshots[f"{reserve_design['scenario']}_{year}"]
        for days in reserve_design["initial_stock_days_of_normal_arrivals"]:
            case = make_case(snap, reserve_design["shock"], reserve_design["hours"])
            case["initial_fuel_stocks_mtce"] = np.array([
                (snap["fuel_domestic_supply_mtce_per_year"][f]+snap["fuel_import_supply_mtce_per_year"][f])*days/365
                for f in FUELS])
            for strategy in reserve_design["strategies"]:
                case_id = f"inventory_{reserve_design['scenario']}_{year}_stock{days}_{reserve_design['shock']}_{strategy}"
                result, evidence = export(case, strategy, case_id)
                result["initial_stock_days_of_normal_arrivals"] = days
                reserve_results.append(result); reserve_audits.append(evidence)
    pd.DataFrame(reserve_results).to_csv(args.output/"hourly_inventory_summary.csv", index=False)
    if (hashlib.sha256(args.snapshots.read_bytes()).hexdigest() != input_hash
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source_hash
            or hashlib.sha256(args.protocol.read_bytes()).hexdigest() != protocol_hash):
        raise AssertionError("Snapshots, protocol or hourly source changed during the fixed cohort")
    (args.output/"hourly_audit.json").write_text(json.dumps({"passed": all(a["passed"] for a in audits),
        "cases": len(audits), "hours_per_case": args.hours, "snapshot_sha256": input_hash,
        "source_sha256": source_hash, "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_hash, "inventory_policy_cases": len(reserve_audits),
        "inventory_audits": reserve_audits, "audits": audits}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
