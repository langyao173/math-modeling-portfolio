"""V6 continuous stress using separately estimated gross import capacity.

The native MATLAB case builder implements the same fixed protocol. The unchanged
V3 LP reoptimizes from the true plant state each hour; phase changes never reset
stocks, SOC, thermal ramp states or demand debt. AI-assisted, 2026-10-18.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from src import v3_hourly as h
from src.v4_hourly import physical_summary

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "config/v6_resilience.json"
OUT = ROOT / "outputs/v6/resilience"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    return json.loads(PROTOCOL.read_text("utf-8"))


def make_case(snapshot, supply_policy="inherited", hours=None, event_mode="repeated"):
    p = protocol()
    if supply_policy not in p["supply_policies"] or event_mode not in {"single", "repeated"}:
        raise ValueError("Unknown supply policy or event mode")
    hours = p["hours"] if hours is None else hours
    if not isinstance(hours, (int, np.integer)) or not 2 <= hours <= p["hours"]:
        raise ValueError("Duration crops the fixed 2160-hour design; integer 2--2160 required")
    c = h.make_case(snapshot, "normal", hours)
    t = np.arange(hours)
    phase = np.full(hours, "recovery_2", dtype="<U16")
    phase[t < p["events"][1][0]] = "recovery_1"
    phase[t < p["warmup_hours"]] = "warmup"
    c["stress_mask"][:] = False
    emergency = np.zeros((hours, 3))
    events = p["events"] if event_mode == "repeated" else p["events"][:1]
    # Contract rates/volumes use the full predeclared design, never cropped run
    # length. Thus changing the displayed duration cannot change earlier inputs.
    import_capacities = np.array([snapshot["fuel_import_capacity_mtce_per_year"][f] for f in h.FUELS]) / 8760
    stocks = c["initial_fuel_stocks_mtce"]
    for index, (lo, hi) in enumerate(events):
        mask = (t >= lo) & (t < hi)
        c["stress_mask"] |= mask
        phase[mask] = f"shock_{index + 1}"
        c["fuel_import_arrivals_mtce"][mask] *= np.array(p["import_retention"])
        for g, retention in p["renewable_retention"].items():
            c["available_gw"][mask, h.GENERATORS.index(g)] *= retention
        c["demand_gw"][mask] *= p["event_demand_multiplier"]
        if supply_policy == "contract":
            delivery_start = hi + p["contract_lead_hours"]
            delivery_end = p["events"][index + 1][0] if index + 1 < len(p["events"]) else p["hours"]
            count = delivery_end - delivery_start
            rate = np.minimum(import_capacities * p["contract_max_fraction_normal_import_rate"],
                              stocks * p["contract_max_fraction_initial_stock_per_event"] / count)
            emergency[(t >= delivery_start) & (t < delivery_end)] += rate
    if event_mode == "single":
        phase[t >= p["events"][0][1]] = "recovery_1"
    c["fuel_arrivals_mtce"] = c["fuel_domestic_arrivals_mtce"] + c["fuel_import_arrivals_mtce"] + emergency
    prices = np.array([c["config"]["fuel_value_million_cny_mtce"][f] for f in h.FUELS])
    base_value = float(emergency.sum(axis=0) @ prices)
    c.update(phase_label=phase, emergency_arrivals_mtce=emergency,
             supply_policy=supply_policy, event_mode=event_mode, event_protocol=p,
             contract_base_value_million_cny=base_value,
             contract_premium_million_cny=base_value * p["contract_premium_fraction"],
             contract_purchase_value_million_cny=base_value * (1 + p["contract_premium_fraction"]),
             shock="two_event" if event_mode == "repeated" else "single_event",
             information_contract="COMMON_24H_PERFECT_FORECAST_ROLLING_FIRST_ACTION",
             boundary=p["scope"])
    return c


def stage_metrics(c, frame):
    p = c["event_protocol"]
    result = []
    for name in dict.fromkeys(c["phase_label"]):
        mask = c["phase_label"] == name
        part = frame.loc[mask]
        row = {"phase": str(name), "start_hour": int(part.hour.iloc[0]),
               "stop_hour_exclusive": int(part.hour.iloc[-1]) + 1,
               "hours": len(part), "unserved_gwh": float(part.unserved_gw.sum()),
               "demand_gwh": float(part.demand_gw.sum())}
        row["power_shortage_fraction"] = row["unserved_gwh"] / row["demand_gwh"]
        for f in h.FUELS:
            requested = float(part[f + "_direct_demand_mtce"].sum())
            row[f + "_direct_unserved_mtce"] = float(part[f + "_direct_unserved_mtce"].sum())
            row[f + "_direct_shortage_fraction"] = row[f + "_direct_unserved_mtce"] / requested if requested else 0.
            row[f + "_minimum_stock_mtce"] = float(part[f + "_stock_mtce"].min())
            row[f + "_terminal_stock_mtce"] = float(part[f + "_stock_mtce"].iloc[-1])
        row["service_passed"] = bool(row["power_shortage_fraction"] <= p["service_shortage_threshold_fraction"]
                                     and all(row[f + "_direct_shortage_fraction"] <= p["service_shortage_threshold_fraction"] for f in h.FUELS))
        result.append(row)
    return result


def recovery_metrics(c, frame):
    p = c["event_protocol"]
    results = []
    events = p["events"] if c["event_mode"] == "repeated" else p["events"][:1]
    for index, (lo, hi) in enumerate(events):
        if lo >= c["hours"]:
            continue
        end = min(p["events"][index + 1][0] if index + 1 < len(p["events"]) else p["hours"], c["hours"])
        target = frame.loc[lo - 1, [f + "_stock_mtce" for f in h.FUELS]].to_numpy(float)
        row = {"event": index + 1, "recovery_start_hour": hi,
               "assessment_end_hour_exclusive": end,
               "service_recovery_hours": None, "joint_inventory_recovery_hours": None}
        for f, value in zip(h.FUELS, target):
            row[f + "_preevent_stock_mtce"] = float(value)
        window = p["recovery_sustained_hours"]
        # End-of-hour stock recovery and delivered-service recovery are distinct.
        # The first successful interval must persist for 24 actual hours.
        for start in range(hi, end - window + 1):
            part = frame.iloc[start:start + window]
            service_ok = part.unserved_gw.sum() <= p["service_shortage_threshold_fraction"] * part.demand_gw.sum() + 1e-8
            for f in h.FUELS:
                service_ok &= part[f + "_direct_unserved_mtce"].sum() <= p["service_shortage_threshold_fraction"] * part[f + "_direct_demand_mtce"].sum() + 1e-8
            stocks_ok = bool(np.all(part[[f + "_stock_mtce" for f in h.FUELS]].to_numpy() >= p["recovery_stock_fraction_of_preevent"] * target - 1e-8))
            if service_ok and row["service_recovery_hours"] is None:
                row["service_recovery_hours"] = start - hi
            if service_ok and stocks_ok and row["joint_inventory_recovery_hours"] is None:
                row["joint_inventory_recovery_hours"] = start - hi
        for f, value in zip(h.FUELS, target):
            row[f + "_retained_fraction_at_next_boundary"] = float(frame[f + "_stock_mtce"].iloc[end - 1] / value) if value > 1e-12 and end > hi else None
        results.append(row)
    return results


def simulate(case, strategy="coordinated"):
    if strategy not in protocol()["strategies"]:
        raise ValueError("This fixed comparison supports coordinated/decentralized only")
    result = h.simulate(case, strategy)
    frame = result["timeseries"]
    frame["phase"] = case["phase_label"]
    for j, f in enumerate(h.FUELS):
        frame[f + "_emergency_arrival_mtce"] = case["emergency_arrivals_mtce"][:, j]
    summary = physical_summary(case, frame, strategy)
    warmup = frame.hour < case["event_protocol"]["warmup_hours"]
    summary.update(supply_policy=case["supply_policy"], event_mode=case["event_mode"],
                   warmup_unserved_gwh=float(frame.loc[warmup, "unserved_gw"].sum()),
                   post_warmup_unserved_gwh=float(frame.loc[~warmup, "unserved_gw"].sum()),
                   contract_base_value_million_cny=case["contract_base_value_million_cny"],
                   contract_premium_million_cny=case["contract_premium_million_cny"],
                   contract_purchase_value_million_cny=case["contract_purchase_value_million_cny"],
                   resource_cost_with_contract_premium_million_cny=summary["physical_operating_cost_million_cny"] + case["contract_premium_million_cny"],
                   cash_fuel_purchases_including_contract_premium_million_cny=summary["fuel_arrival_purchase_value_million_cny"] + case["contract_premium_million_cny"])
    # Fuel is paid on arrival. The resource-cost account separately credits the
    # terminal stock asset; it does not charge the base contract fuel twice.
    result.update(summary=summary, stages=stage_metrics(case, frame), recovery=recovery_metrics(case, frame), status="COMPLETED")
    return result


def _run_one(task):
    case_id, snapshot, strategy, supply, mode = task
    timer = time.perf_counter()
    case = make_case(snapshot, supply, event_mode=mode)
    result = simulate(case, strategy)
    file = OUT / (case_id + ".csv")
    result["timeseries"].to_csv(file, index=False)
    # Independently re-read the delivered table before accepting its physics.
    physical = h.audit(case, pd.read_csv(file))
    if not physical["passed"]:
        raise AssertionError(f"Saved continuous trace failed: {case_id}")
    record = {"case_id": case_id, "trace": str(file.relative_to(ROOT)).replace("\\", "/"),
              "trace_sha256": sha(file), "audit": physical, "summary": result["summary"],
              "stages": result["stages"], "recovery": result["recovery"],
              "status": "COMPLETED", "runtime_seconds": time.perf_counter() - timer}
    (OUT / (case_id + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2), "utf-8")
    print(case_id, "2160h physics PASS", round(record["runtime_seconds"], 2), "s", flush=True)
    return record


def main(workers=3):
    p = protocol()
    selected_path = ROOT / "outputs/v6/planning/selected_cases.csv"
    snapshot_path = ROOT / "outputs/v6/planning/selected_snapshots.json"
    chosen = pd.read_csv(selected_path)
    snapshots = json.loads(snapshot_path.read_text("utf-8"))
    tasks, plan_roles = [], {}
    for role in p["selected_roles"]:
        matches = chosen.loc[chosen.roles.fillna("").str.split("|").map(lambda parts: role in parts), "case_id"].tolist()
        if len(matches) != 1:
            raise AssertionError(f"Exactly one predeclared plan for {role} required")
        plan_roles.setdefault(matches[0], []).append(role)
    for plan in plan_roles:
        for year in p["years"]:
            for policy in p["supply_policies"]:
                for strategy in p["strategies"]:
                    key = f"{plan}_{year}"
                    tasks.append((f"{key}_{policy}_{strategy}_repeated", snapshots[key], strategy, policy, "repeated"))
    knee = next(plan for plan, roles in plan_roles.items() if "frontier_knee" in roles)
    for policy in p["supply_policies"]:
        key = f"{knee}_2060"
        tasks.append((f"{key}_{policy}_coordinated_single", snapshots[key], "coordinated", policy, "single"))
    for policy in p["supply_policies"]:
        key = f"{knee}_2060"
        zero_coal = deepcopy(snapshots[key])
        zero_coal["fuel_import_supply_mtce_per_year"]["coal"] = 0.
        zero_coal["fuel_import_capacity_mtce_per_year"]["coal"] = 0.
        tasks.append((f"{key}_{policy}_coordinated_zero_coal_channel", zero_coal, "coordinated", policy, "repeated"))
    sources = [Path(__file__), PROTOCOL, selected_path, snapshot_path, ROOT / "src/v3_hourly.py", ROOT / "src/v4_hourly.py", ROOT / "data/v6_fuel_import_channels.csv"]
    before = {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in sources}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frozen_cohort.json").write_text(json.dumps({"protocol": p, "plan_roles": plan_roles,
        "case_ids": [t[0] for t in tasks], "source_sha256": before}, ensure_ascii=False, indent=2), "utf-8")
    native_inputs = {"default_snapshot_key": f"{knee}_2060", "snapshots": snapshots,
                     "protocol": p, "source_sha256": before}
    (OUT / "native_inputs.json").write_text(json.dumps(native_inputs, ensure_ascii=False, indent=2), "utf-8")
    timer = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(_run_one, tasks))
    if before != {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in sources}:
        raise AssertionError("Result-affecting source changed during fixed cohort")
    pd.DataFrame([{"case_id": r["case_id"], **r["summary"]} for r in records]).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame([{"case_id": r["case_id"], **v} for r in records for v in r["stages"]]).to_csv(OUT / "stages.csv", index=False)
    pd.DataFrame([{"case_id": r["case_id"], **v} for r in records for v in r["recovery"]]).to_csv(OUT / "recovery.csv", index=False)
    receipt = {"status": "PASS_COMPLETE_PREDECLARED_CONTINUOUS_STATE_COHORT", "cases": len(records),
               "hours_per_case": p["hours"], "source_sha256": before, "records": records,
               "runtime_seconds": time.perf_counter() - timer, "created_utc": datetime.now(timezone.utc).isoformat(),
               "boundary": p["scope"]}
    (OUT / "audit.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
    print(receipt["status"], len(records), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    main(args.workers)
