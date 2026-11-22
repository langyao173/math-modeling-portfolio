"""Bounded development repair, followed by one frozen independent-design test.

Original V10 controls/results and V11 fixed-plan evidence remain unchanged.
AI-assisted: OpenAI Codex, 2026-10-19.
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
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from src import v6_annual as annual
from src.v5_planning import decode, digest, summarize, reliability_feasible
from src.v11_robust_plans import perturbation, yearly_reliability

ROOT = Path(__file__).resolve().parents[1]
REPAIR_INDICES = [0, 1, 2, 8, 4]
SHORTFALLS = ["max_power_shortfall_fraction", "max_direct_fuel_shortfall_fraction", "max_useful_service_shortfall_fraction"]


def explicit_candidates(base):
    def changed(updates):
        vector = np.array(base, float).copy()
        for index, value in updates.items():
            vector[index] = value
        return vector
    combined = {0:3., 1:3., 2:3., 8:2., 4:.1}
    return [("original_knee", changed({})), ("budget_ceiling", changed({0:3., 1:3., 2:3.})),
            ("fuel_priority_ceiling", changed({8:2.})), ("gas_generation_priority_floor", changed({4:.1})),
            ("budget_fuel_and_gas", changed(combined)),
            ("coal_substitution", changed({**combined, 3:2.})),
            ("firm_nonfossil_substitution", changed({**combined, 5:4.})),
            ("mixed_supply_generation", changed({**combined, 3:1.5, 5:4., 6:2., 7:2., 9:3.}))]


def choose_candidate(rows):
    feasible = [r for r in rows if r["worst_shortfall_fraction"] <= .01+1e-9]
    tail = lambda r: (r["nominal_net_emissions_2060_mt"], r["nominal_discounted_capital_trillion_cny"], r["candidate_id"])
    return min(feasible, key=tail) if feasible else min(rows, key=lambda r: (r["worst_shortfall_fraction"], *tail(r)))


def run(root=ROOT):
    root = Path(root).resolve()
    out = root/"outputs/v11/robust_repair"
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out/"analysis.json"
    if (out/"selected_before_independent_test.json").exists():
        raise FileExistsError("Repair selection already frozen; do not repeat or tune against the independent test")
    planning = json.loads((root/"config/v6_annual_planning.json").read_text("utf-8"))
    original = json.loads((root/"config/v6_annual_experiments.json").read_text("utf-8"))
    plans = pd.read_csv(root/"outputs/v6/planning/selected_cases.csv")
    knee = plans.loc[plans.roles.str.contains("frontier_knee")].iloc[0]
    base = knee[planning["decision_names"]].to_numpy(float)
    bounds = np.asarray(planning["bounds"], float)
    explicit = explicit_candidates(base)
    sources = ["src/v11_robust_repair.py", "tests/test_v11_robust_repair.py", "src/v11_robust_plans.py",
               "src/v6_annual.py", "src/v5_planning.py", "src/v3_annual.py", "src/v3_annual_calibrated.py",
               "src/v3_calibration_candidate.py", "config/v3_transition.json", "config/v6_annual_planning.json",
               "config/v6_annual_experiments.json", "data/annual_china.csv", "data/ember_power_china.csv",
               "data/energy_production_china.csv", "data/fuel_supply_trade.csv", "data/supplement_carriers_2024.csv",
               "data/v6_fuel_import_channels.csv", "outputs/v6/planning/selected_cases.csv",
               "outputs/v6/annual/joint_design.csv", "outputs/v11/robust_plans/analysis.json",
               "outputs/v11/robust_plans/plan_design_metrics.csv"]
    before = {name: digest(root/name) for name in sources}
    protocol = {"status": "EXPLORATORY_REPAIR_ON_ALREADY_SEEN_48_DESIGNS",
                "frozen_before_evaluation_utc": datetime.now(timezone.utc).isoformat(),
                "development_design": "Original 48 V6 engineering designs, already seen; no fresh-validation claim",
                "independent_design": {"method": "scrambled Latin hypercube", "seed": 20261019, "count": 48,
                                       "ranges": original["joint_design"]["parameters"],
                                       "access_rule": "Generated before development, consumed only after the final vector is written and frozen; exactly one final-vector assessment, no subsequent tuning"},
                "explicit_candidates": [{"label": name, "controls": dict(zip(planning["decision_names"], vector.tolist()))} for name, vector in explicit],
                "search": {"condition": "Run only if none of the eight explicit candidates satisfies all 48 designs at 1%",
                           "method": "one differential_evolution minimax search", "seed": 202610190,
                           "indices": REPAIR_INDICES, "names": [planning["decision_names"][i] for i in REPAIR_INDICES],
                           "bounds": bounds[REPAIR_INDICES].tolist(), "fixed_other_controls": base.tolist(),
                           "popsize": 4, "maxiter": 5, "polish": False, "tol": 0., "atol": 0.,
                           "max_requests": 120, "max_all_requests": 128, "workers": 1, "updating": "immediate"},
                "selection": "Minimize maximum shortfall across 48 designs, all 2030-2060 years and electricity/direct coal/oil/gas/useful service. Among candidates feasible at <=1%, minimize actual nominal 2060 net CO2 then actual nominal discounted capital; otherwise minimize worst shortfall then those nominal quantities.",
                "physical_contract": "Original ten-dimensional control bounds, resources, initial construction queue, construction/retirement, yearly build caps, price and fuel rules stay fixed. Environmental factors follow V6 mapping. No free supply, faster construction or relaxed reliability gate.",
                "claim": "Independent engineering design evaluation is not an empirical holdout or a probability estimate. A minimax improvement above 1% is not robust feasibility.",
                "source_input_hashes": before}
    # These files are written before any candidate simulation or independent score.
    names = list(original["joint_design"]["parameters"])
    limits = np.asarray([original["joint_design"]["parameters"][n] for n in names])
    points = qmc.LatinHypercube(d=len(names), scramble=True, seed=20261019).random(n=48)
    independent = pd.DataFrame(qmc.scale(points, limits[:,0], limits[:,1]), columns=names)
    independent.insert(0, "case", [f"independent_{i:03d}" for i in range(48)])
    independent.to_csv(out/"independent_design_frozen.csv", index=False)
    protocol["independent_design"]["sha256"] = digest(out/"independent_design_frozen.csv")
    (out/"protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), "utf-8")
    receipt = {"status": "RUNNING_DEVELOPMENT", "started_utc": datetime.now(timezone.utc).isoformat(),
               "command": "python -B -X utf8 -m src.v11_robust_repair", "argv": sys.argv,
               "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
               "protocol_sha256": digest(out/"protocol.json"), "source_input_hashes_before": before}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
    print("FROZEN: 8 explicit, at most 120 DE requests; independent seed 20261019 / 48 designs", flush=True)
    started = time.perf_counter()
    data = annual.load_data(root)
    params = annual.calibrate(data, 2020)
    development = pd.read_csv(root/"outputs/v6/annual/joint_design.csv").to_dict("records")
    rows, design_rows, by_vector = [], [], {}
    requests = 0
    stage = "explicit"
    def evaluate(vector, label="minimax"):
        nonlocal requests
        requests += 1
        if requests > 128:
            raise AssertionError("Fixed repair evaluation budget exceeded")
        vector = np.asarray(vector, float)
        if vector.shape != (10,) or not np.isfinite(vector).all() or (vector<bounds[:,0]).any() or (vector>bounds[:,1]).any():
            raise ValueError("Repair must remain within original ten control bounds")
        key = tuple(vector.tolist())
        if key in by_vector:
            return rows[by_vector[key]]
        candidate_id = f"repair_{len(rows):03d}"
        nominal_frame, nominal_actions = annual.simulate(data, params, decode(vector, candidate_id), investor_ledger=False)
        nominal = summarize(nominal_frame, nominal_actions, data, planning)
        local = []
        for design in development:
            scenario, fitted, overrides = perturbation(data, params, vector, design, candidate_id+"__"+design["case"])
            frame, actions = annual.simulate(data, fitted, scenario, engineering_overrides=overrides, investor_ledger=False)
            metrics = summarize(frame, actions, data, planning)
            local.append({"candidate_id": candidate_id, "design_id": design["case"], **metrics,
                          "worst_shortfall_fraction": max(metrics[n] for n in SHORTFALLS),
                          "annual_reliability_feasible": reliability_feasible(metrics, planning)})
        table = pd.DataFrame(local)
        worst = table.loc[table.worst_shortfall_fraction.idxmax()]
        row = {"candidate_id": candidate_id, "label": label, "stage": stage,
               "worst_shortfall_fraction": float(worst.worst_shortfall_fraction), "worst_design_id": worst.design_id,
               "reliable_design_count": int(table.annual_reliability_feasible.sum()),
               **{"nominal_"+n: value for n, value in nominal.items()},
               **dict(zip(planning["decision_names"], vector.tolist()))}
        by_vector[key] = len(rows)
        rows.append(row)
        design_rows.extend(local)
        # Each completed candidate is retained even if a later solve fails.
        pd.DataFrame(rows).to_csv(out/"candidate_scores.csv", index=False)
        pd.DataFrame(design_rows).to_csv(out/"candidate_design_metrics.csv", index=False)
        if stage == "explicit" or len(rows)%20 == 0:
            print(candidate_id, label, "worst", row["worst_shortfall_fraction"], "reliable", row["reliable_design_count"], flush=True)
        return row
    for label, vector in explicit:
        evaluate(vector, label)
    search_result = None
    if not any(r["worst_shortfall_fraction"] <= .01+1e-9 for r in rows):
        stage = "minimax_development"
        def objective(part):
            vector = base.copy()
            vector[REPAIR_INDICES] = part
            return evaluate(vector)["worst_shortfall_fraction"]
        result = differential_evolution(objective, bounds[REPAIR_INDICES], seed=202610190,
                    popsize=4, maxiter=5, polish=False, tol=0., atol=0., workers=1, updating="immediate")
        search_result = {"success": bool(result.success), "message": str(result.message),
                         "iterations": int(result.nit), "function_requests": int(result.nfev),
                         "best_search_worst_shortfall_fraction": float(result.fun)}
    selected = choose_candidate(rows)
    vector = np.array([selected[name] for name in planning["decision_names"]])
    selection = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "candidate": selected,
                 "unique_candidates": len(rows), "requests": requests, "search": search_result,
                 "development_scores_sha256": digest(out/"candidate_scores.csv"),
                 "development_design_metrics_sha256": digest(out/"candidate_design_metrics.csv"),
                 "independent_design_sha256": digest(out/"independent_design_frozen.csv")}
    (out/"selected_before_independent_test.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), "utf-8")
    print("SELECTED BEFORE INDEPENDENT TEST", selected["candidate_id"], selected["worst_shortfall_fraction"], flush=True)
    receipt["status"] = "SELECTION_FROZEN_INDEPENDENT_TEST_RUNNING"
    receipt["selection_sha256"] = digest(out/"selected_before_independent_test.json")
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
    final_metrics, final_annual, final_actions, final_yearly = [], [], [], []
    # The independent factor rows are consumed for the first time only here.
    independent_records = pd.read_csv(out/"independent_design_frozen.csv").to_dict("records")
    for cohort, designs in (("development_reconstruction", development), ("independent_once", independent_records)):
        for design in designs:
            scenario, fitted, overrides = perturbation(data, params, vector, design, selected["candidate_id"]+"__"+design["case"])
            frame, actions = annual.simulate(data, fitted, scenario, engineering_overrides=overrides, investor_ledger=False)
            metrics = summarize(frame, actions, data, planning)
            identity = {"candidate_id": selected["candidate_id"], "cohort": cohort, "design_id": design["case"]}
            yearly = yearly_reliability(frame, data, planning).assign(**identity)
            worst = yearly.loc[yearly.shortfall_fraction.idxmax()]
            row = {**identity, **metrics, "annual_reliability_feasible": reliability_feasible(metrics, planning),
                   "worst_shortfall_fraction": float(worst.shortfall_fraction), "worst_year": int(worst.year), "worst_carrier": worst.carrier}
            if len(frame)!=36 or len(actions)!=432 or row["annual_reliability_feasible"] != (not yearly.violates_reliability.any()):
                raise AssertionError("Incomplete final trajectory or mismatched year/carrier assessment")
            if cohort == "development_reconstruction":
                old = next(r for r in design_rows if r["candidate_id"]==selected["candidate_id"] and r["design_id"]==design["case"])
                for field in [*SHORTFALLS, "net_emissions_2060_mt", "discounted_capital_trillion_cny"]:
                    if not np.isclose(row[field], old[field], rtol=1e-11, atol=1e-7):
                        raise AssertionError("Final development reconstruction changed")
            final_metrics.append(row)
            final_annual.append(frame.assign(**identity))
            final_actions.append(actions.assign(**identity))
            final_yearly.append(yearly)
    final = pd.DataFrame(final_metrics)
    final.to_csv(out/"final_design_metrics.csv", index=False)
    pd.concat(final_annual, ignore_index=True).to_csv(out/"final_all_trajectories.csv", index=False)
    pd.concat(final_actions, ignore_index=True).to_csv(out/"final_all_actions.csv", index=False)
    yearly = pd.concat(final_yearly, ignore_index=True)
    yearly.to_csv(out/"final_annual_reliability.csv", index=False)
    yearly.loc[yearly.violates_reliability].to_csv(out/"final_violating_year_carriers.csv", index=False)
    summaries = []
    for cohort, group in final.groupby("cohort", sort=False):
        worst = group.loc[group.worst_shortfall_fraction.idxmax()]
        summaries.append({"cohort": cohort, "design_count": len(group),
                          "reliable_design_count": int(group.annual_reliability_feasible.sum()),
                          "worst_shortfall_fraction": float(worst.worst_shortfall_fraction),
                          "worst_design_id": worst.design_id, "worst_year": int(worst.worst_year), "worst_carrier": worst.worst_carrier,
                          "net_emissions_2060_mt_max": float(group.net_emissions_2060_mt.max()),
                          "discounted_capital_trillion_cny_max": float(group.discounted_capital_trillion_cny.max())})
    pd.DataFrame(summaries).to_csv(out/"final_summary.csv", index=False)
    after = {name: digest(root/name) for name in sources}
    if before != after or len(final)!=96 or len(independent_records)!=48:
        raise AssertionError("Changed sources or incomplete final cohort")
    receipt.update(status="PASS_BOUNDED_DEVELOPMENT_AND_ONE_INDEPENDENT_DESIGN_TEST",
                   completed_utc=datetime.now(timezone.utc).isoformat(), runtime_seconds=time.perf_counter()-started,
                   unique_candidates=len(rows), requests=requests, development_design_simulations=len(design_rows),
                   final_design_simulations=len(final), final_annual_rows=len(final)*36, final_action_rows=len(final)*432,
                   selected_candidate=selected, search=search_result, final_summaries=summaries,
                   source_input_hashes_after=after,
                   improvement_from_original_knee_worst_fraction=rows[0]["worst_shortfall_fraction"]-selected["worst_shortfall_fraction"],
                   artifact_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file() and p!=receipt_path})
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        run(args.root)
    except Exception:
        receipt_path = args.root/"outputs/v11/robust_repair/analysis.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text("utf-8"))
            receipt.update(status="FAILED_RETAINED_RESULTS_NOT_FINAL", error=traceback.format_exc())
            receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
        raise
