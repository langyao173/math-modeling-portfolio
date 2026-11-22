"""Supplemental 2^3 model-conditional mechanism decomposition, no retuning."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from itertools import product
import json
from math import factorial
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from src import v4_annual as model
from src.v4_mechanisms import audit_saved, enrich, sha


PLAYERS = ("government", "consumer", "investor")
CELLS = tuple("".join(map(str, bits)) for bits in product((0, 1), repeat=3))


def decompose(values):
    """Three-player Shapley values plus all Mobius interaction coefficients."""
    if set(values) != set(CELLS) or not np.isfinite(list(values.values())).all():
        raise ValueError("A finite value for every one of the eight cells is required")
    interactions = {}
    for bits in sorted(CELLS, key=lambda value: (value.count("1"), value)):
        if bits == "000": continue
        subsets = [sub for sub in interactions if all(int(a) <= int(b) for a, b in zip(sub, bits))]
        interactions[bits] = values[bits] - values["000"] - sum(interactions[sub] for sub in subsets)
    shapley = {}
    for j, player in enumerate(PLAYERS):
        marginal = 0.
        for bits in CELLS:
            if bits[j] == "1": continue
            others = bits.count("1")
            added = bits[:j] + "1" + bits[j+1:]
            weight = factorial(others) * factorial(2 - others) / factorial(3)
            marginal += weight * (values[added] - values[bits])
        shapley[player] = marginal
    total = values["111"] - values["000"]
    return {"shapley": shapley, "interactions": interactions, "total_effect": total,
            "shapley_sum_error": sum(shapley.values()) - total,
            "reconstruction_error": sum(interactions.values()) - total}


def run(root):
    root = Path(root).resolve(); out = root / "outputs/v4/annual/factorial"; out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((root / "config/v4_factorial.json").read_text("utf-8"))
    original_path = root / protocol["source_cohort_receipt"]
    original = json.loads(original_path.read_text("utf-8"))
    if original["status"] != "PASS_COMPLETE_FIXED_COHORT": raise AssertionError("Complete source cohort required")
    source_names = list(original["source_input_hashes_before"]) + ["src/v4_factorial.py", "config/v4_factorial.json", "tests/test_v4_factorial.py", protocol["source_cohort_receipt"]]
    before = {name: sha(root / name) for name in source_names}
    if any(before[name] != digest for name, digest in original["source_input_hashes_before"].items()):
        raise AssertionError("Frozen source cohort has drifted")
    receipt = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
               "command": "python -X utf8 -m src.v4_factorial", "source_input_hashes_before": before,
               "protocol": protocol, "cases": []}
    receipt_path = out / "run_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
    try:
        data = model.load_data(root); params = model.calibrate(data, protocol["calibration_end"])
        gate = original["protocol"]["execution_gate"]
        values_rows, shapley_rows, interaction_rows = [], [], []
        for scenario in protocol["scenarios"]:
            scenario_values = {}
            for bits in CELLS:
                if bits in protocol["reused_cells"]:
                    old_case = scenario + "__" + protocol["reused_cells"][bits]
                    record = next(case for case in original["cases"] if case["case"] == old_case)
                    trace_path = root / record["trajectory"]; actions_path = root / record["actions"]
                    if sha(trace_path) != record["trajectory_sha256"] or sha(actions_path) != record["actions_sha256"]:
                        raise AssertionError("Reused cell hash mismatch")
                    frame = pd.read_csv(trace_path); actions = pd.read_csv(actions_path); origin = "reused_original_88_case_cohort"
                else:
                    rules = {"carbon_feedback": bits[0] == "1", "consumer_choice": "responsive" if bits[1] == "1" else "frozen_initial",
                             "investor_profit_response": bits[2] == "1", "investor_utilisation_response": bits[2] == "1"}
                    frame, actions = model.simulate(data, params, scenario, start_year=protocol["initial_year"],
                                                     end_year=protocol["final_year"], mechanisms=rules)
                    frame = enrich(frame, actions, data)
                    case_id = scenario + "__factorial_" + bits
                    frame.insert(0, "case", case_id); actions.insert(0, "case", case_id)
                    trace_path = out / (case_id + ".csv"); actions_path = out / (case_id + "_actions.csv")
                    frame.to_csv(trace_path, index=False); actions.to_csv(actions_path, index=False)
                    frame = pd.read_csv(trace_path); actions = pd.read_csv(actions_path); origin = "new_supplementary_run"
                audit = audit_saved(frame, actions, gate)
                receipt["cases"].append({"scenario": scenario, "bits": bits, "origin": origin, "audit": audit,
                                          "trajectory": str(trace_path.relative_to(root)).replace("\\", "/"), "trajectory_sha256": sha(trace_path),
                                          "actions": str(actions_path.relative_to(root)).replace("\\", "/"), "actions_sha256": sha(actions_path)})
                for year in protocol["endpoint_years"]:
                    row = frame.loc[frame.year == year].iloc[0]
                    values_rows.append({"scenario": scenario, "bits": bits, "year": year, "origin": origin,
                                        **{metric: float(row[metric]) for metric in protocol["metrics"]}})
                    scenario_values[bits, year] = row
                receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
            for year in protocol["endpoint_years"]:
                for metric in protocol["metrics"]:
                    cell_values = {bits: float(scenario_values[bits, year][metric]) for bits in CELLS}
                    result = decompose(cell_values)
                    if abs(result["shapley_sum_error"]) > 1e-7 or abs(result["reconstruction_error"]) > 1e-7:
                        raise AssertionError("Attribution does not reconstruct the actual model contrast")
                    for player, value in result["shapley"].items():
                        shapley_rows.append({"scenario": scenario, "year": year, "metric": metric, "player": player,
                                             "contribution": value, "all_disabled_value": cell_values["000"],
                                             "full_value": cell_values["111"], "full_minus_disabled": result["total_effect"],
                                             "sum_error": result["shapley_sum_error"], "interpretation": "conditional model decomposition; not empirical causal identification"})
                    for interaction, value in result["interactions"].items():
                        interaction_rows.append({"scenario": scenario, "year": year, "metric": metric, "mechanisms": "+".join(PLAYERS[j] for j, bit in enumerate(interaction) if bit == "1"),
                                                 "order": interaction.count("1"), "interaction": value,
                                                 "full_minus_disabled": result["total_effect"], "reconstruction_error": result["reconstruction_error"]})
        pd.DataFrame(values_rows).to_csv(out / "factorial_values.csv", index=False)
        pd.DataFrame(shapley_rows).to_csv(out / "shapley_contributions.csv", index=False)
        pd.DataFrame(interaction_rows).to_csv(out / "nonadditive_interactions.csv", index=False)
        reused = sum(case["origin"] == "reused_original_88_case_cohort" for case in receipt["cases"])
        new = len(receipt["cases"]) - reused
        if (reused, new, len(receipt["cases"])) != (protocol["expected_reused_runs"], protocol["expected_new_runs"], protocol["expected_cells"]):
            raise AssertionError("Incomplete declared supplemental cohort")
        after = {name: sha(root / name) for name in source_names}
        if before != after: raise AssertionError("Source/input drift during supplemental execution")
        receipt.update(status="PASS_12_NEW_20_REUSED_FACTORIAL", completed_utc=datetime.now(timezone.utc).isoformat(),
                       new_runs=new, reused_runs=reused, factorial_cells=len(receipt["cases"]), source_input_hashes_after=after)
        receipt["output_sha256"] = {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file() and path != receipt_path}
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
        print(json.dumps({key: receipt[key] for key in ("status", "new_runs", "reused_runs", "factorial_cells")}, indent=2))
    except Exception:
        receipt.update(status="FAILED_PARTIAL_NOT_FOR_SELECTION", error=traceback.format_exc())
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), "utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    run(parser.parse_args().root)
