"""Execute the predeclared V4 mechanism cohort and joint engineering design.

No parameter selection or 2025 validation tuning is performed here. All cases,
including null and adverse results, are saved before comparisons are computed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import pandas as pd
import scipy
from scipy.stats import qmc

from src import v3_annual as frozen
from src import v4_annual as model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(root):
    files = ["src/v3_annual.py", "src/v3_annual_calibrated.py", "src/v3_calibration_candidate.py",
             "src/v4_annual.py", "src/v4_mechanisms.py", "config/v3_transition.json",
             "config/v4_mechanisms.json", "tests/test_v4_annual.py", "docs/V4机制与实验契约.md",
             "data/annual_china.csv", "data/ember_power_china.csv", "data/supplement_carriers_2024.csv",
             "data/energy_production_china.csv", "data/fuel_supply_trade.csv"]
    return {name: sha(root / name) for name in files}


def enrich(frame, actions, data):
    """Derive service losses independently from realised deliveries, not labels."""
    result = frame.copy()
    efficiency = data["config"]["engineering"]["direct_use_efficiency"]
    service = sum(result[f + "_direct_delivered_mtce"] * model.TWH_PER_MTCE * efficiency[j]
                  for j, f in enumerate(model.FUELS))
    service += result.electricity_generation_twh * efficiency[3]
    result["useful_service_delivered_twh"] = service
    result["useful_service_unserved_twh"] = np.maximum(result.useful_service_demand_twh - service, 0.)
    result["useful_service_unserved_fraction"] = result.useful_service_unserved_twh / result.useful_service_demand_twh
    result["direct_fuel_unserved_mtce"] = sum(result[f + "_direct_demand_mtce"] - result[f + "_direct_delivered_mtce"] for f in model.FUELS)
    result["cumulative_investment_million_cny"] = result.investment_spend_million_cny.cumsum()
    result["cumulative_energy_co2_mt"] = result.energy_co2_mt.cumsum()
    result["cumulative_useful_service_unserved_twh"] = result.useful_service_unserved_twh.cumsum()
    result["cumulative_electricity_unserved_twh"] = result.electricity_unserved_twh.cumsum()
    result["cumulative_direct_fuel_unserved_mtce"] = result.direct_fuel_unserved_mtce.cumsum()
    nonfossil_caps = sum(result[t + "_capacity_gw"] for t in model.TECHS[2:])
    result["nonfossil_electric_capacity_gw"] = nonfossil_caps
    result["nonfossil_electric_capacity_share"] = nonfossil_caps / sum(result[t + "_capacity_gw"] for t in model.TECHS)
    # Investments in annual fuel-production capacity carry Mtce/year units and
    # must never be added to GW orders. Monetary spending is comparable.
    nonfossil_spend = actions.loc[actions.agent.isin(model.TECHS[2:])].groupby("year").capital_spend_million_cny.sum()
    result["nonfossil_investment_million_cny"] = result.year.map(nonfossil_spend).fillna(0.)
    result["cumulative_nonfossil_investment_million_cny"] = result.nonfossil_investment_million_cny.cumsum()
    return result


def audit_saved(frame, actions, gate):
    """Recompute physical and action invariants from persisted CSV fields."""
    fuel_residual = max(float((frame[f + "_domestic_supply_mtce"] + frame[f + "_imports_mtce"]
                               - frame[f + "_stock_change_mtce"] - frame[f + "_direct_delivered_mtce"]
                               - frame[f + "_power_use_mtce"]).abs().max()) for f in model.FUELS)
    power_residual = float((frame.electricity_generation_twh + frame.electricity_unserved_twh - frame.electricity_demand_twh).abs().max())
    minimum_stock = min(float(frame[f + "_stock_mtce"].min()) for f in model.FUELS)
    budget_excess = float(np.maximum(frame.investment_spend_million_cny - frame.investment_budget_million_cny, 0.).max())
    spend_by_year = actions.groupby("year").capital_spend_million_cny.sum().reindex(frame.year).to_numpy()
    action_error = float(np.abs(spend_by_year - frame.investment_spend_million_cny.to_numpy()).max())
    desired_rows = gate["expected_rows_per_run"]
    if len(frame) != desired_rows or len(actions) != desired_rows * len(model.ASSETS):
        raise AssertionError("Incomplete annual cohort member")
    for values in (frame.select_dtypes(include=np.number), actions.drop(columns=["simulated_profit_million_cny", "simulated_revenue_million_cny", "simulated_operating_cost_million_cny", "annual_capital_charge_million_cny"]).select_dtypes(include=np.number)):
        if not np.isfinite(values.to_numpy()).all(): raise AssertionError("Nonfinite persisted annual result")
    if fuel_residual > gate["fuel_balance_atol_mtce"] or power_residual > gate["power_balance_atol_twh"]:
        raise AssertionError("Saved annual physical ledger fails")
    if minimum_stock < -gate["negative_stock_atol_mtce"] or budget_excess > gate["budget_atol_million_cny"] or action_error > gate["budget_atol_million_cny"]:
        raise AssertionError("Saved annual stock/capital/action invariant fails")
    return {"status": "PASS", "years": len(frame), "actions": len(actions), "fuel_balance_max_mtce": fuel_residual,
            "power_balance_max_twh": power_residual, "minimum_fuel_stock_mtce": minimum_stock,
            "budget_excess_million_cny": budget_excess, "action_spend_error_million_cny": action_error}


def recent_expectations(data, params, expectation_end=None):
    result = deepcopy(params)
    end = params["train_end"] if expectation_end is None else expectation_end
    a, _, _, service, _, _ = frozen._observations(data, end)
    for key, values in (("service_fit", service), ("electricity_fit", a.electricity_demand_twh.to_numpy(float))):
        result["expectations"][key] = {"mode": "increment3", "increment": float(np.diff(values)[-3:].mean())}
    result["v4_expectation_fit_end"] = int(end)
    return result


def run(root):
    root = Path(root).resolve()
    out = root / "outputs/v4/annual"
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((root / "config/v4_mechanisms.json").read_text("utf-8"))
    before = source_hashes(root)
    receipt = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
               "command": "python -X utf8 -m src.v4_mechanisms --root <project-root>", "argv": sys.argv,
               "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
               "source_input_hashes_before": before, "protocol": protocol, "cases": []}
    receipt_path = out / "run_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
    try:
        data = model.load_data(root)
        params = model.calibrate(data, protocol["calibration_end"])
        (out / "calibration.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), "utf-8")
        (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), "utf-8")
        frames, action_frames, endpoints = [], [], []

        def execute(case_id, kind, scenario, case_params, mechanisms, overrides=None, design_values=None):
            frame, actions = model.simulate(data, case_params, scenario, start_year=protocol["initial_year"],
                                             end_year=protocol["final_year"], mechanisms=mechanisms,
                                             engineering_overrides=overrides)
            frame = enrich(frame, actions, data)
            frame.insert(0, "case", case_id); frame.insert(1, "experiment_kind", kind)
            actions.insert(0, "case", case_id); actions.insert(1, "experiment_kind", kind)
            traces = out / "trajectories"; traces.mkdir(exist_ok=True)
            frame_path = traces / (case_id + ".csv"); action_path = traces / (case_id + "_actions.csv")
            frame.to_csv(frame_path, index=False); actions.to_csv(action_path, index=False)
            saved_frame = pd.read_csv(frame_path); saved_actions = pd.read_csv(action_path)
            audit = audit_saved(saved_frame, saved_actions, protocol["execution_gate"])
            receipt["cases"].append({"case": case_id, "kind": kind, "scenario": scenario if isinstance(scenario, str) else scenario,
                                      "mechanisms": model.resolve_mechanisms(mechanisms), "engineering_overrides": overrides or {},
                                      "expectation_fit_end": case_params.get("v4_expectation_fit_end", case_params["train_end"]),
                                      "expectations": case_params["expectations"],
                                      "design_values": design_values or {}, "audit": audit,
                                      "trajectory": str(frame_path.relative_to(root)).replace("\\", "/"), "trajectory_sha256": sha(frame_path),
                                      "actions": str(action_path.relative_to(root)).replace("\\", "/"), "actions_sha256": sha(action_path)})
            frames.append(frame); action_frames.append(actions)
            endpoint = frame.loc[frame.year.isin(protocol["endpoint_years"])].copy()
            if design_values:
                for key, value in design_values.items(): endpoint["design_" + key] = value
            endpoints.append(endpoint)
            # Keep failure evidence recoverable even if a later case raises.
            receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")

        for scenario in protocol["scenarios"]:
            for case in protocol["mechanism_cases"]:
                fit_end = protocol["initial_year"] if case.get("expectations") == "initial_year_recent_increment3" else params["train_end"]
                case_params = recent_expectations(data, params, fit_end) if case.get("expectations") else params
                execute(scenario + "__" + case["case"], "mechanism", scenario, case_params, case["mechanisms"])

        design = protocol["joint_design"]
        keys = list(design["parameters"])
        bounds = np.array([design["parameters"][key] for key in keys])
        unit_design = qmc.LatinHypercube(d=len(keys), scramble=True, seed=design["seed"]).random(n=design["n_parameter_vectors"])
        values = qmc.scale(unit_design, bounds[:, 0], bounds[:, 1])
        design_frame = pd.DataFrame(values, columns=keys)
        design_frame.insert(0, "case", ["joint_%03d" % j for j in range(len(values))])
        design_frame.to_csv(out / "joint_design.csv", index=False)
        engineering = data["config"]["engineering"]
        for _, vector in design_frame.iterrows():
            factors = {key: float(vector[key]) for key in keys}
            scenario = deepcopy(data["config"]["scenarios"][design["scenario"]])
            scenario["gdp_growth"] = [v * factors["gdp_growth_multiplier"] for v in scenario["gdp_growth"]]
            scenario["carbon_price_drift_cny_t"] *= factors["carbon_price_drift_multiplier"]
            for key in ("technology_cost_multiplier", "cost_decline_per_year", "annual_budget_multiplier", "construction_delay_multiplier", "supply_multiplier", "import_multiplier"):
                scenario[key] = factors[key]
            overrides = {"cost_response": engineering["cost_response"] * factors["consumer_response_multiplier"],
                         "shortage_response": engineering["shortage_response"] * factors["consumer_response_multiplier"],
                         "investment_feedback": engineering["investment_feedback"] * factors["investor_response_multiplier"]}
            case_params = deepcopy(params)
            case_params["expectations"]["utilisation_response"] *= factors["investor_response_multiplier"]
            execute(str(vector["case"]), "joint_design", scenario, case_params, {}, overrides, factors)

        trajectories = pd.concat(frames, ignore_index=True)
        all_actions = pd.concat(action_frames, ignore_index=True)
        all_endpoints = pd.concat(endpoints, ignore_index=True)
        trajectories.to_csv(out / "all_trajectories.csv", index=False)
        all_actions.to_csv(out / "all_actions.csv", index=False)
        mechanism_endpoints = all_endpoints.loc[all_endpoints.experiment_kind == "mechanism"].copy()
        mechanism_endpoints.to_csv(out / "mechanism_endpoints.csv", index=False)
        joint_endpoints = all_endpoints.loc[all_endpoints.experiment_kind == "joint_design"].copy()
        joint_endpoints.to_csv(out / "joint_endpoints.csv", index=False)
        deltas = []
        for scenario in protocol["scenarios"]:
            baseline = mechanism_endpoints.loc[mechanism_endpoints["case"] == scenario + "__full"].set_index("year")
            for _, row in mechanism_endpoints.loc[mechanism_endpoints.scenario == scenario].iterrows():
                record = {"case": row["case"], "scenario": scenario, "year": int(row.year)}
                for metric in protocol["comparisons"]["metrics"]:
                    record[metric] = float(row[metric]); record["full_" + metric] = float(baseline.loc[row.year, metric])
                    record["delta_" + metric] = float(row[metric] - baseline.loc[row.year, metric])
                record["delta_nonfossil_percentage_points"] = record["delta_nonfossil_share"] * 100
                deltas.append(record)
        pd.DataFrame(deltas).to_csv(out / "mechanism_contributions.csv", index=False)
        range_rows, associations = [], []
        for year, group in joint_endpoints.groupby("year"):
            for metric in protocol["comparisons"]["metrics"]:
                range_rows.append({"year": int(year), "metric": metric, "n_design_cases": len(group),
                                   "minimum": float(group[metric].min()), "median": float(group[metric].median()),
                                   "maximum": float(group[metric].max()), "interpretation": "range over declared engineering design; not probability interval"})
                for key in keys:
                    # Pearson correlation of ranks is Spearman's descriptive
                    # association, used here without significance/causal claims.
                    association = group["design_" + key].rank().corr(group[metric].rank())
                    associations.append({"year": int(year), "metric": metric, "parameter": key,
                                         "spearman_rank_association": float(association) if np.isfinite(association) else None,
                                         "interpretation": "descriptive within this joint design; not causal variance allocation"})
        pd.DataFrame(range_rows).to_csv(out / "joint_design_ranges.csv", index=False)
        pd.DataFrame(associations).to_csv(out / "joint_design_rank_associations.csv", index=False)
        counts = pd.Series([case["kind"] for case in receipt["cases"]]).value_counts()
        gate = protocol["execution_gate"]
        if int(counts.get("mechanism", 0)) != gate["expected_mechanism_runs"] or int(counts.get("joint_design", 0)) != gate["expected_joint_runs"]:
            raise AssertionError("Incomplete fixed cohort; no partial comparison accepted")
        receipt["source_input_hashes_after"] = source_hashes(root)
        if receipt["source_input_hashes_after"] != before: raise AssertionError("Source/input changed during fixed cohort")
        receipt.update(status="PASS_COMPLETE_FIXED_COHORT", completed_utc=datetime.now(timezone.utc).isoformat(),
                       mechanism_runs=int(counts["mechanism"]), joint_design_runs=int(counts["joint_design"]),
                       annual_rows=len(trajectories), investment_action_rows=len(all_actions),
                       worst_fuel_balance_mtce=max(c["audit"]["fuel_balance_max_mtce"] for c in receipt["cases"]),
                       worst_power_balance_twh=max(c["audit"]["power_balance_max_twh"] for c in receipt["cases"]))
        receipt["output_sha256"] = {str(path.relative_to(out)).replace("\\", "/"): sha(path)
                                    for path in sorted(out.rglob("*")) if path.is_file() and path != receipt_path}
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
        print(json.dumps({key: receipt[key] for key in ("status", "mechanism_runs", "joint_design_runs", "annual_rows", "investment_action_rows", "worst_fuel_balance_mtce", "worst_power_balance_twh")}, ensure_ascii=False, indent=2))
    except Exception:
        receipt.update(status="FAILED_PARTIAL_NOT_FOR_SELECTION", failed_utc=datetime.now(timezone.utc).isoformat(), error=traceback.format_exc())
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    run(parser.parse_args().root)
