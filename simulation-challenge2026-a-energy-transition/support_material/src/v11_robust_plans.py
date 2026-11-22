"""Re-evaluate six fixed V6 plans on the original 48 engineering designs.

No controls are selected or optimized. The historical model, original results,
and original design remain unchanged. AI-assisted: OpenAI Codex, 2026-10-19.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy

from src import v6_annual as annual
from src.v5_planning import BASE_VECTOR, decode, digest, reliability_feasible, summarize

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_FIELDS = ("technology_cost_multiplier", "cost_decline_per_year",
                      "annual_budget_multiplier", "construction_delay_multiplier",
                      "supply_multiplier", "import_multiplier")


def perturbation(data, params, vector, factors, name):
    """Use the original joint-design mapping, preserving fixed plan controls."""
    scenario = deepcopy(data["config"]["scenarios"]["baseline"])
    scenario.update(decode(vector, name))
    scenario["gdp_growth"] = [v*factors["gdp_growth_multiplier"] for v in scenario["gdp_growth"]]
    scenario["carbon_price_drift_cny_t"] *= factors["carbon_price_drift_multiplier"]
    for field in ENVIRONMENT_FIELDS:
        scenario[field] = factors[field]
    engineering = data["config"]["engineering"]
    overrides = {"cost_response": engineering["cost_response"]*factors["consumer_response_multiplier"],
                 "shortage_response": engineering["shortage_response"]*factors["consumer_response_multiplier"],
                 "investment_feedback": engineering["investment_feedback"]*factors["investor_response_multiplier"]}
    fitted = deepcopy(params)
    fitted["expectations"]["utilisation_response"] *= factors["investor_response_multiplier"]
    return scenario, fitted, overrides


def yearly_reliability(frame, data, protocol):
    """Keep each assessed year/carrier; direct-fuel failures cannot average out."""
    future = frame.loc[frame.year >= protocol["reliability"]["first_year"]]
    efficiency = data["config"]["engineering"]["direct_use_efficiency"]
    service = future.electricity_generation_twh.to_numpy()*efficiency[3]
    rows = []
    def append(carrier, requested, delivered, threshold, unit):
        requested, delivered = np.asarray(requested), np.asarray(delivered)
        fraction = np.divide(requested-delivered, requested, out=np.zeros(len(future)), where=requested > 1e-12)
        fraction = np.maximum(fraction, 0.)
        for year, request, delivery, loss in zip(future.year, requested, delivered, fraction):
            rows.append({"year": int(year), "carrier": carrier, "requested": float(request),
                         "delivered": float(delivery), "unit": unit,
                         "shortfall_fraction": float(loss), "threshold_fraction": threshold,
                         "violates_reliability": bool(loss > threshold+1e-9)})
    limits = protocol["reliability"]
    append("electricity", future.electricity_demand_twh, future.electricity_generation_twh,
           limits["max_power_shortfall_fraction"], "TWh")
    for j, fuel in enumerate(annual.FUELS):
        requested, delivered = future[fuel+"_direct_demand_mtce"], future[fuel+"_direct_delivered_mtce"]
        append(fuel, requested, delivered, limits["max_direct_fuel_shortfall_fraction"], "Mtce")
        service += delivered.to_numpy()*annual.TWH_PER_MTCE*efficiency[j]
    append("useful_service", future.useful_service_demand_twh, service,
           limits["max_useful_service_shortfall_fraction"], "TWh_useful")
    return pd.DataFrame(rows)


def assert_metrics(actual, expected, context):
    errors = {}
    for field, value in actual.items():
        error = abs(value-float(expected[field]))
        if not np.isclose(value, float(expected[field]), rtol=1e-11, atol=1e-7):
            raise AssertionError(f"{context}: {field} differs by {error}")
        errors[field] = error
    return max(errors.values(), default=0.)


def run(root=ROOT):
    root = Path(root).resolve()
    out = root/"outputs/v11/robust_plans"
    out.mkdir(parents=True, exist_ok=True)
    sources = ["src/v11_robust_plans.py", "tests/test_v11_robust_plans.py", "src/v6_annual.py",
               "src/v6_annual_experiments.py", "src/v6_planning.py", "src/v5_planning.py",
               "src/v5_annual.py", "src/v3_annual.py", "src/v3_annual_calibrated.py",
               "src/v3_calibration_candidate.py", "config/v3_transition.json",
               "config/v6_annual_experiments.json", "config/v6_annual_planning.json",
               "data/annual_china.csv", "data/ember_power_china.csv", "data/energy_production_china.csv",
               "data/fuel_supply_trade.csv", "data/supplement_carriers_2024.csv",
               "data/v6_fuel_import_channels.csv", "outputs/v6/planning/selected_cases.csv",
               "outputs/v6/planning/planning_audit.json", "outputs/v6/annual/joint_design.csv",
               "outputs/v6/annual/joint_endpoints.csv", "outputs/v6/annual/calibration.json"]
    before = {name: digest(root/name) for name in sources}
    plan_protocol = json.loads((root/"config/v6_annual_planning.json").read_text("utf-8"))
    experiment_protocol = json.loads((root/"config/v6_annual_experiments.json").read_text("utf-8"))
    plans = pd.read_csv(root/"outputs/v6/planning/selected_cases.csv")
    designs = pd.read_csv(root/"outputs/v6/annual/joint_design.csv")
    original_endpoints = pd.read_csv(root/"outputs/v6/annual/joint_endpoints.csv")
    if len(plans) != 6 or len(designs) != 48 or plans.case_id.duplicated().any() or designs["case"].duplicated().any():
        raise AssertionError("Exactly six unique fixed plans and 48 original design cases are required")
    protocol = {"status": "FIXED_PLAN_DEVELOPMENT_STRESS_REEVALUATION",
                "fixed_before_new_evaluations_utc": datetime.now(timezone.utc).isoformat(),
                "plan_ids": plans.case_id.tolist(), "roles": dict(zip(plans.case_id, plans.roles)),
                "design_ids": designs["case"].tolist(), "design_source": "outputs/v6/annual/joint_design.csv",
                "expected_new_plan_design_runs": 288, "initial_year": 2024, "final_year": 2060,
                "calibration_end": 2020, "planning_protocol": plan_protocol,
                "joint_design_protocol": experiment_protocol["joint_design"],
                "control_composition": "Fixed period budgets multiply the external budget factor; fixed carbon drift multiplies the external drift factor. All priorities remain fixed. Other factors follow the original V6 joint-design mapping.",
                "unchanged_structure": "Original step scarcity price, construction queue, initial resources, retirement, sink and profit-response rule. Investor response factor scales operating feedback and fitted utilisation response, as in the original 48 designs; it does not scale the 0.25 profit rule.",
                "interpretation": "The 48 engineering designs have no probability weights. Counts are design coverage, not reliability probabilities. No resampling, optimization, reselection, or native execution is performed."}
    (out/"protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), "utf-8")
    receipt = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
               "command": "python -B -X utf8 -m src.v11_robust_plans", "argv": sys.argv,
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "pandas": pd.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
               "source_input_hashes_before": before, "protocol_sha256": digest(out/"protocol.json")}
    receipt_path = out/"analysis.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
    started = time.perf_counter()
    data = annual.load_data(root)
    params = annual.calibrate(data, 2020)
    (out/"calibration.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), "utf-8")
    nominal = []
    for plan in plans.to_dict("records"):
        vector = [plan[name] for name in plan_protocol["decision_names"]]
        frame, actions = annual.simulate(data, params, decode(vector, plan["case_id"]), investor_ledger=False)
        metrics = summarize(frame, actions, data, plan_protocol)
        error = assert_metrics(metrics, plan, plan["case_id"]+" nominal")
        if reliability_feasible(metrics, plan_protocol) != bool(plan["annual_reliability_feasible"]):
            raise AssertionError("Nominal feasibility differs")
        nominal.append({"plan_id": plan["case_id"], "roles": plan["roles"], "max_absolute_metric_error": error,
                        **metrics, "annual_reliability_feasible": reliability_feasible(metrics, plan_protocol)})
    pd.DataFrame(nominal).to_csv(out/"nominal_reconstruction.csv", index=False)
    replay = []
    for design in designs.to_dict("records"):
        factors = {name: design[name] for name in experiment_protocol["joint_design"]["parameters"]}
        scenario, fitted, overrides = perturbation(data, params, BASE_VECTOR, factors, "baseline")
        frame, _ = annual.simulate(data, fitted, scenario, engineering_overrides=overrides, investor_ledger=False)
        expected = original_endpoints.loc[original_endpoints["case"] == design["case"]]
        for year in experiment_protocol["endpoint_years"]:
            actual = frame.loc[frame.year == year].iloc[0]
            target = expected.loc[expected.year == year].iloc[0]
            numeric = {name: float(actual[name]) for name in frame.select_dtypes(include=np.number).columns}
            replay.append({"design_id": design["case"], "year": year,
                           "compared_numeric_fields": len(numeric),
                           "max_absolute_metric_error": assert_metrics(numeric, target, design["case"]+str(year))})
    pd.DataFrame(replay).to_csv(out/"original_design_reconstruction.csv", index=False)
    rows, annual_frames, action_frames, checks = [], [], [], []
    for plan in plans.to_dict("records"):
        vector = [plan[name] for name in plan_protocol["decision_names"]]
        for design in designs.to_dict("records"):
            factors = {name: design[name] for name in experiment_protocol["joint_design"]["parameters"]}
            scenario, fitted, overrides = perturbation(data, params, vector, factors, plan["case_id"]+"__"+design["case"])
            frame, actions = annual.simulate(data, fitted, scenario, engineering_overrides=overrides, investor_ledger=False)
            if len(frame) != 36 or len(actions) != 432 or frame.year.tolist() != list(range(2025, 2061)):
                raise AssertionError("Incomplete annual trajectory")
            metrics = summarize(frame, actions, data, plan_protocol)
            identity = {"plan_id": plan["case_id"], "roles": plan["roles"], "design_id": design["case"]}
            yearly = yearly_reliability(frame, data, plan_protocol).assign(**identity)
            feasible = reliability_feasible(metrics, plan_protocol)
            if feasible != (not yearly.violates_reliability.any()):
                raise AssertionError("Year/carrier checks disagree with original feasibility definition")
            worst = yearly.loc[yearly.shortfall_fraction.idxmax()]
            rows.append({**identity, **metrics, "annual_reliability_feasible": feasible,
                         "violating_year_carrier_count": int(yearly.violates_reliability.sum()),
                         "violating_year_count": int(yearly.loc[yearly.violates_reliability, "year"].nunique()),
                         "worst_year": int(worst.year), "worst_carrier": worst.carrier,
                         "worst_shortfall_fraction": float(worst.shortfall_fraction), **factors})
            annual_frames.append(frame.assign(**identity))
            action_frames.append(actions.assign(**identity))
            checks.append(yearly)
        print(plan["case_id"], "completed 48 fixed designs", flush=True)
    results = pd.DataFrame(rows)
    all_yearly = pd.concat(checks, ignore_index=True)
    pd.concat(annual_frames, ignore_index=True).to_csv(out/"all_trajectories.csv", index=False)
    pd.concat(action_frames, ignore_index=True).to_csv(out/"all_actions.csv", index=False)
    results.to_csv(out/"plan_design_metrics.csv", index=False)
    all_yearly.to_csv(out/"annual_reliability.csv", index=False)
    violations = all_yearly.loc[all_yearly.violates_reliability]
    violations.to_csv(out/"violating_year_carriers.csv", index=False)
    plan_summary = []
    for plan_id, group in results.groupby("plan_id", sort=False):
        worst = group.loc[group.worst_shortfall_fraction.idxmax()]
        item = {"case_id": plan_id, "roles": group.roles.iloc[0], "design_count": len(group),
                "reliable_design_count": int(group.annual_reliability_feasible.sum()),
                "worst_design_id": worst.design_id, "worst_year": int(worst.worst_year),
                "worst_carrier": worst.worst_carrier, "worst_shortfall_fraction": worst.worst_shortfall_fraction}
        for metric in ("net_emissions_2060_mt", "discounted_capital_trillion_cny", "max_power_shortfall_fraction",
                       "max_direct_fuel_shortfall_fraction", "max_useful_service_shortfall_fraction"):
            item.update({metric+"_min": float(group[metric].min()), metric+"_median": float(group[metric].median()),
                         metric+"_max": float(group[metric].max())})
        plan_summary.append(item)
    pd.DataFrame(plan_summary).to_csv(out/"plan_summary.csv", index=False)
    pairs = []
    for design_id, group in results.groupby("design_id", sort=False):
        for left, right in combinations(group.to_dict("records"), 2):
            pairs.append({"design_id": design_id, "left_plan": left["plan_id"], "right_plan": right["plan_id"],
                          "left_reliable": left["annual_reliability_feasible"], "right_reliable": right["annual_reliability_feasible"],
                          "left_minus_right_capital_trillion_cny": left["discounted_capital_trillion_cny"]-right["discounted_capital_trillion_cny"],
                          "left_minus_right_net_emissions_mt": left["net_emissions_2060_mt"]-right["net_emissions_2060_mt"],
                          "left_minus_right_worst_shortfall_fraction": left["worst_shortfall_fraction"]-right["worst_shortfall_fraction"]})
    pd.DataFrame(pairs).to_csv(out/"paired_fixed_plan_comparisons.csv", index=False)
    after = {name: digest(root/name) for name in sources}
    if before != after:
        raise AssertionError("Source/input changed during fixed-plan reevaluation")
    if len(results) != 288 or len(all_yearly) != 288*31*5:
        raise AssertionError("Incomplete fixed cohort")
    receipt.update(status="PASS_6_NOMINAL_48_DESIGN_REPLAYS_AND_288_FIXED_PLAN_RUNS",
                   completed_utc=datetime.now(timezone.utc).isoformat(), runtime_seconds=time.perf_counter()-started,
                   nominal_reconstructions=len(nominal), original_design_replay_runs=48,
                   original_design_endpoint_checks=len(replay), fixed_plan_design_runs=len(results),
                   actual_annual_rows=288*36, actual_action_rows=288*432,
                   assessed_year_carrier_rows=len(all_yearly), violating_year_carrier_rows=len(violations),
                   reliable_plan_design_count=int(results.annual_reliability_feasible.sum()),
                   source_input_hashes_after=after, plan_summaries=plan_summary,
                   artifact_sha256={p.name: digest(p) for p in out.iterdir() if p.is_file() and p != receipt_path})
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
    print(pd.DataFrame(plan_summary)[["case_id", "roles", "reliable_design_count", "worst_shortfall_fraction"]].to_string(index=False))
    print(receipt["status"], flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root)
