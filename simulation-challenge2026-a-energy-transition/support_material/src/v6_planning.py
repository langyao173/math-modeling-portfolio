"""V6 fixed-budget investment search using repaired independent import channels.

The controls, seeds, bounds and five optimizer budgets match V5. Unchanged
accounting and Pareto helpers are reused; every V6 candidate is freshly solved.
AI-assisted: OpenAI Codex/GPT-6, OpenAI, 2026-10-18.
"""
from __future__ import annotations
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from src import v6_annual as annual
from src.v5_planning import BASE_VECTOR, digest, decode, summarize, reliability_feasible, nondominated_mask
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v6/planning"


class Evaluation:
    def __init__(self, data, params, protocol):
        self.data, self.params, self.protocol = data, params, protocol
        self.rows, self.controls, self.by_vector = [], [], {}
        self.requests = 0
        self.stage = "explicit"

    def evaluate(self, vector):
        # Fitness and constraint/selection reads often request the identical
        # expensive trajectory. Reuse only exact controls, never rounded keys.
        self.requests += 1
        key = tuple(float(v) for v in vector)
        if key in self.by_vector:
            return self.rows[self.by_vector[key]]
        case_id = f"plan_{len(self.rows):05d}"
        frame, actions = annual.simulate(self.data, self.params, decode(vector, case_id), investor_ledger=False)
        row = {"case_id": case_id, "search_stage": self.stage, **summarize(frame, actions, self.data, self.protocol)}
        row.update(dict(zip(self.protocol["decision_names"], key)))
        row["annual_reliability_feasible"] = reliability_feasible(row, self.protocol)
        if self.rows:
            assert abs(row["industrial_co2_2060_mt"]-self.rows[0]["industrial_co2_2060_mt"]) < 1e-6
        self.by_vector[key] = len(self.rows)
        self.rows.append(row)
        self.controls.append(np.array(vector, float).copy())
        return row

    def score(self, vector, objective, target=None):
        row = self.evaluate(vector)
        reliability_penalty = sum(max(0, row[name]/value-1) for name, value in self.protocol["reliability"].items() if name != "first_year")
        carbon_penalty = max(0, (row["net_emissions_2060_mt"]-target)/1000) if target is not None else 0
        base = row["net_emissions_2060_mt"]/1000 if objective == "emissions" else row["discounted_capital_trillion_cny"]
        return base+1000*(reliability_penalty+carbon_penalty)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT/"config/v6_annual_planning.json").read_text("utf-8"))
    source_names = ["src/v6_planning.py", "src/v6_annual.py", "src/v3_annual.py", "src/v3_annual_calibrated.py",
                    "src/v3_calibration_candidate.py", "config/v3_transition.json", "config/v6_annual_planning.json",
                    "data/annual_china.csv", "data/ember_power_china.csv", "data/energy_production_china.csv",
                    "data/fuel_supply_trade.csv", "data/supplement_carriers_2024.csv", "data/v6_fuel_import_channels.csv", "src/v5_annual.py", "src/v5_planning.py"]
    before = {name: digest(ROOT/name) for name in source_names}
    started = time.perf_counter()
    data = annual.load_data(ROOT)
    evaluator = Evaluation(data, annual.calibrate(data, 2020), protocol)
    bounds = np.asarray(protocol["bounds"], float)
    baseline = evaluator.evaluate(BASE_VECTOR)
    for budget, clean, thermal, firm, store, fuel, carbon in [(1.5,2,.5,2,2,1,9), (2,3,.2,3,3,1.5,15), (3,4,.1,4,4,2,15)]:
        evaluator.evaluate([budget,budget,budget,thermal,thermal,firm,clean,store,fuel,carbon])
    evaluator.stage = "sobol_design"
    points = qmc.Sobol(len(bounds), scramble=True, seed=protocol["seed"]).random_base2(protocol["sobol_power"])
    for point in qmc.scale(points, bounds[:,0], bounds[:,1]):
        evaluator.evaluate(point)
    searches = []
    tasks = [("minimum_net", "emissions", None), ("minimum_capital", "cost", None)]
    tasks += [(f"target_{target:g}", "cost", target) for target in protocol["net_emission_targets_mt"] if target >= baseline["structural_net_floor_mt"]]
    for index, (name, objective, target) in enumerate(tasks):
        evaluator.stage = name
        result = differential_evolution(lambda x: evaluator.score(x, objective, target), bounds,
                    seed=protocol["seed"]+index, **protocol["search"], workers=1, updating="immediate")
        row = evaluator.evaluate(result.x)
        searches.append({"task": name, "target_mt": target, "optimizer_converged": bool(result.success),
                         "message": str(result.message), "iterations": int(result.nit),
                         "optimizer_case_id": row["case_id"], "objective_including_search_penalty": float(result.fun)})
        print(name, "evaluated", len(evaluator.rows), "reliability", row["annual_reliability_feasible"], flush=True)
    table = pd.DataFrame(evaluator.rows)
    table["pareto_unrestricted"] = nondominated_mask(table[["discounted_capital_trillion_cny", "net_emissions_2060_mt", "max_useful_service_shortfall_fraction"]])
    table["pareto_annual_reliable"] = False
    feasible = table.loc[table.annual_reliability_feasible]
    if len(feasible):
        table.loc[feasible.index, "pareto_annual_reliable"] = nondominated_mask(feasible[["discounted_capital_trillion_cny", "net_emissions_2060_mt"]])
    selected = {baseline["case_id"]: {"baseline"}}
    target_results = []
    for target in protocol["net_emission_targets_mt"]:
        eligible = feasible.loc[feasible.net_emissions_2060_mt <= target+1e-7]
        if target < baseline["structural_net_floor_mt"]-1e-7:
            status, winner = "STRUCTURALLY_INFEASIBLE_IN_DECLARED_DOMAIN", None
        elif eligible.empty:
            status, winner = "NO_FEASIBLE_CANDIDATE_FOUND_WITHIN_FIXED_SEARCH", None
        else:
            status = "FEASIBLE_CANDIDATE_FOUND_NOT_GLOBAL_MINIMUM"
            winner = eligible.loc[eligible.discounted_capital_trillion_cny.idxmin()]
            selected.setdefault(winner.case_id, set()).add(f"target_{target:g}")
        target_results.append({"target_net_mt": target, "status": status,
                               "best_case_id": None if winner is None else winner.case_id,
                               "best_observed_discounted_capital_trillion_cny": None if winner is None else float(winner.discounted_capital_trillion_cny),
                               "eligible_candidates": len(eligible), "structural_floor_mt": baseline["structural_net_floor_mt"]})
    if len(feasible):
        minimum_net = feasible.loc[feasible.net_emissions_2060_mt.idxmin()]
        minimum_cost = feasible.loc[feasible.discounted_capital_trillion_cny.idxmin()]
        front = table.loc[table.pareto_annual_reliable].copy()
        cols = ["discounted_capital_trillion_cny", "net_emissions_2060_mt"]
        span = np.maximum(front[cols].max()-front[cols].min(), 1e-12)
        knee = front.loc[(((front[cols]-front[cols].min())/span)**2).sum(axis=1).idxmin()]
        for role, row in (("minimum_net_emissions", minimum_net), ("minimum_capital", minimum_cost), ("frontier_knee", knee)):
            selected.setdefault(row.case_id, set()).add(role)
    selected_rows, snapshots = [], {}
    for case_id, roles in selected.items():
        index = int(case_id.split("_")[1])
        frame, actions = annual.simulate(data, evaluator.params, decode(evaluator.controls[index], case_id))
        again = summarize(frame, actions, data, protocol)
        for key, value in again.items():
            assert np.isclose(value, evaluator.rows[index][key], rtol=1e-12, atol=1e-8)
        frame.to_csv(OUT/f"trajectory_{case_id}.csv", index=False)
        actions.to_csv(OUT/f"actions_{case_id}.csv", index=False)
        selected_rows.append({**evaluator.rows[index], "roles": "|".join(sorted(roles)),
                              "annual_trace_sha256": digest(OUT/f"trajectory_{case_id}.csv"),
                              "action_trace_sha256": digest(OUT/f"actions_{case_id}.csv")})
        for year in protocol["hourly_screening"]["years"]:
            row = frame.loc[frame.year == year].iloc[0]
            snapshots[f"{case_id}_{year}"] = annual.snapshot_from_row(row)
    table.to_csv(OUT/"candidate_evaluations.csv", index=False)
    table.loc[table.pareto_annual_reliable].sort_values("discounted_capital_trillion_cny").to_csv(OUT/"annual_reliable_frontier.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(OUT/"selected_cases.csv", index=False)
    pd.DataFrame(target_results).to_csv(OUT/"target_search_results.csv", index=False)
    (OUT/"selected_snapshots.json").write_text(json.dumps(snapshots, ensure_ascii=False, indent=2), "utf-8")
    after = {name: digest(ROOT/name) for name in source_names}
    assert before == after, "Planning source/input changed during search"
    report = {"status": "PASS_COMPLETE_FIXED_SEARCH_AND_SELECTED_TRACE_RECONSTRUCTION",
              "candidate_evaluations": len(table), "evaluation_requests_including_exact_reuse": evaluator.requests,
              "annual_reliable_candidates": len(feasible), "annual_reliable_frontier_candidates": int(table.pareto_annual_reliable.sum()),
              "selected_cases": len(selected_rows), "searches": searches, "targets": target_results,
              "runtime_seconds": time.perf_counter()-started, "source_hashes": before,
              "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
              "command": "python -X utf8 -m src.v6_planning", "claim_boundary": protocol["claim_boundary"],
              "structural_bound": protocol["structural_bound"],
              "outputs_sha256": {p.name: digest(p) for p in OUT.iterdir() if p.is_file() and p.name != "planning_audit.json"}}
    (OUT/"planning_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({key: report[key] for key in ("status", "candidate_evaluations", "annual_reliable_candidates", "selected_cases", "runtime_seconds")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
