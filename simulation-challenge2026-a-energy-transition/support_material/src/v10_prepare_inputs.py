"""Prepare exogenous inputs and an independent Python LP reference for AnyMath.

AI tool: OpenAI Codex / GPT-6, OpenAI; used 2026-10-18; release date unverified.
No dispatch action or terminal state is supplied to the Julia optimizer.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src import v3_hourly as h

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "simulation/anymath_v10"
OUT = ROOT / "outputs/v10/dispatch"


def snapshot():
    source = json.loads((ROOT / "outputs/v6/resilience/native_inputs.json").read_text("utf-8"))
    return source["snapshots"]["plan_02349_2060"]


def configuration(design="demo", **changes):
    c = dict(hours=360 if design == "demo" else 2160,
             event_starts=[48, 192] if design == "demo" else [120, 1200],
             event_duration_hours=48 if design == "demo" else 480,
             shock_severity=1.0, supply_policy="inherited", contract_multiplier=1.0,
             contract_lead_hours=12 if design == "demo" else 48,
             storage_power_multiplier=1.0, strategy="coordinated", design=design)
    c.update(changes)
    return c


def make_case(config):
    s = snapshot()
    hours, starts, duration = config["hours"], config["event_starts"], config["event_duration_hours"]
    assert 2 <= hours <= 2160 and len(starts) == 2
    assert 0 <= starts[0] < starts[0] + duration < starts[1] < starts[1] + duration < hours
    assert 0 <= config["shock_severity"] <= 1
    assert config["supply_policy"] in {"inherited", "contract"}
    assert config["contract_multiplier"] >= 0 and config["storage_power_multiplier"] > 0
    c = h.make_case(s, "normal", hours)
    c["stress_mask"][:] = False
    severity = config["shock_severity"]
    emergency = np.zeros((hours, 3))
    t = np.arange(hours)
    normal_cap = np.array([s["fuel_import_capacity_mtce_per_year"][f] / 8760 for f in h.FUELS])
    for i, start in enumerate(starts):
        stop = start + duration
        mask = (t >= start) & (t < stop)
        c["stress_mask"] |= mask
        c["fuel_import_arrivals_mtce"][mask] *= 1 - severity * (1 - np.array([.50, .05, .05]))
        for tech, retention in (("wind", .25), ("solar", .55), ("hydro", .70)):
            c["available_gw"][mask, h.GENERATORS.index(tech)] *= 1 - severity * (1 - retention)
        c["demand_gw"][mask] *= 1 + .08 * severity
        if config["supply_policy"] == "contract":
            lo = stop + config["contract_lead_hours"]
            hi = starts[i+1] if i+1 < len(starts) else hours
            assert lo < hi, "Contract delivery window must remain positive"
            rate = np.minimum(normal_cap * .5 * config["contract_multiplier"],
                              c["initial_fuel_stocks_mtce"] * .5 * config["contract_multiplier"] / (hi-lo))
            emergency[(t >= lo) & (t < hi)] += rate
    c["fuel_arrivals_mtce"] = c["fuel_domestic_arrivals_mtce"] + c["fuel_import_arrivals_mtce"] + emergency
    for store in h.STORES:
        c["storage"][store]["power_gw"] *= config["storage_power_multiplier"]
    c.update(shock="v10_two_events", emergency_arrivals_mtce=emergency,
             v10_config=deepcopy(config), event_starts=starts,
             evidence_class="DESIGNED_REOPTIMIZATION_EXPERIMENT")
    return c


def prepare():
    MODEL.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    s = snapshot()
    c = h.make_case(s, "normal", 2160)
    columns = {"hour": np.arange(2160), "demand_gw": c["demand_gw"]}
    columns.update({g+"_available_gw": c["available_gw"][:, j] for j, g in enumerate(h.GENERATORS)})
    for kind, values in (("domestic", c["fuel_domestic_arrivals_mtce"]),
                         ("import", c["fuel_import_arrivals_mtce"]),
                         ("direct_demand", c["direct_fuel_demand_mtce"])):
        columns.update({f+"_"+kind+"_mtce": values[:, j] for j, f in enumerate(h.FUELS)})
    target = MODEL / "v10_base_inputs.csv"
    pd.DataFrame(columns).to_csv(target, index=False, float_format="%.17g")
    def vals(d, keys):
        return "[" + ", ".join(repr(float(d[k])) for k in keys) + "]"
    cfg = c["config"]
    text = f'''# AI tool: OpenAI Codex / GPT-6, OpenAI; used 2026-10-18; release date unverified.
# Frozen annual knee state, 2060; hourly profiles are engineering designs.
const V10_GENERATORS = ["coal", "gas", "nuclear", "hydro", "wind", "solar", "biomass"]
const V10_FUELS = ["coal", "oil", "gas"]
const V10_STORES = ["battery", "pumped"]
const V10_INITIAL_FUEL = {vals(s["fuel_initial_stocks_mtce"], h.FUELS)}
const V10_INITIAL_SOC = {[c["storage"][x]["initial_soc_gwh"] for x in h.STORES]}
const V10_STORAGE_POWER = {[c["storage"][x]["power_gw"] for x in h.STORES]}
const V10_STORAGE_ENERGY = {[c["storage"][x]["energy_gwh"] for x in h.STORES]}
const V10_ETA = {[c["storage"][x]["eta_charge"] for x in h.STORES]}
const V10_EFFICIENCY = {vals(s["fuel_efficiency"], ("coal", "gas"))}
const V10_THERMAL_CAPACITY = {vals(s["capacity_gw"], ("coal", "gas"))}
const V10_IMPORT_CAPACITY_PER_H = {[s["fuel_import_capacity_mtce_per_year"][f]/8760 for f in h.FUELS]}
const V10_GENERATION_COST = {vals(cfg["generation_cost_cny_kwh"], h.GENERATORS)}
const V10_FUEL_VALUE = {vals(cfg["fuel_value_million_cny_mtce"], h.FUELS)}
const V10_CARBON_PRICE = {repr(s["carbon_price_cny_t"])}
const V10_FUEL_EMISSION = {[h.FUEL_EMISSION_MT_PER_MTCE[f] for f in h.FUELS]}

"""A parameter change starts a fresh case; it never relabels an old trajectory.
The 360h demo is a separate, shorter engineering design, not the V6 2160h cohort.
"""
function v10_config(; design="demo", hours=(design=="demo" ? 360 : 2160),
    event_starts=(design=="demo" ? [48,192] : [120,1200]),
    event_duration_hours=(design=="demo" ? 48 : 480), shock_severity=1.0,
    supply_policy="inherited", contract_multiplier=1.0,
    contract_lead_hours=(design=="demo" ? 12 : 48), storage_power_multiplier=1.0,
    strategy="coordinated")
    return (; design, hours, event_starts=Int.(event_starts), event_duration_hours,
        shock_severity, supply_policy, contract_multiplier, contract_lead_hours,
        storage_power_multiplier, strategy)
end
'''
    (MODEL / "v10_config.jl").write_text(text, "utf-8")
    manifest = dict(status="EXOGENOUS_INPUTS_ONLY", hours=2160,
                    columns=list(columns), no_actions_or_end_states=True,
                    input_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    snapshot_source="outputs/v6/resilience/native_inputs.json",
                    snapshot_key="plan_02349_2060", demo_design=configuration(),
                    long_design=configuration("long"))
    (OUT / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")


def baseline(design="demo", name="base", **changes):
    cfg = configuration(design, **changes)
    c = make_case(cfg)
    before = time.perf_counter()
    result = h.simulate(c, cfg["strategy"])
    frame = result["timeseries"]
    frame.to_csv(OUT / f"python_{name}.csv", index=False, float_format="%.17g")
    receipt = dict(config=cfg, audit=result["audit"], summary=result["summary"],
                   runtime_seconds=time.perf_counter()-before,
                   source="Independent existing SciPy/HiGHS rolling LP; no cloud execution claim")
    (OUT / f"python_{name}.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
    print(name, frame.unserved_gw.sum(), "GWh", receipt["runtime_seconds"], "s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--long", action="store_true")
    args = parser.parse_args()
    prepare()
    if args.baseline:
        design = "long" if args.long else "demo"
        baseline(design, "long" if args.long else "base")
        if not args.long:
            baseline(design, "mild", shock_severity=.5)
            baseline(design, "contract", supply_policy="contract")
            baseline(design, "power", storage_power_multiplier=1.5)
            baseline(design, "decentralized", strategy="decentralized")
