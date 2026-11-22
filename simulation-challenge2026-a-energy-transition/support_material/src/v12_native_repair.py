"""Prepare frozen repair controls and separate audit-only Python references.

This command does not launch MATLAB or change any V6/V11 result or model.
AI-assisted: OpenAI Codex, 2026-10-19.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat, savemat

from src import v6_annual as annual
from src.v5_planning import decode, digest, summarize

ROOT = Path(__file__).resolve().parents[1]


def controls_from_selection(selection, planning):
    chosen = selection["candidate"]
    if chosen["candidate_id"] != "repair_005":
        raise ValueError("V12 implements only the already frozen repair_005")
    vector = np.array([chosen[name] for name in planning["decision_names"]], float)
    bounds = np.asarray(planning["bounds"], float)
    if vector.shape != (10,) or not np.isfinite(vector).all() or (vector<bounds[:,0]).any() or (vector>bounds[:,1]).any():
        raise ValueError("Frozen repair controls must satisfy the original ten-dimensional bounds")
    return decode(vector, "repair_005"), vector


def state_reference(data, params, factors):
    scenario = annual._decision_scenario(data, factors)
    rules = annual.resolve_mechanisms()
    state, runtime = annual.initialize_state(data, params, 2024, mechanisms=rules)
    state[212] = scenario.get("carbon_price_start_cny_t", 100.)
    if scenario.get("clear_initial_pipeline", False):
        state[20:200] = 0.
    initial = state.copy()
    states, metrics = [], []
    gdp = runtime["base_gdp"]
    for year in range(2025, 2061):
        next_gdp = runtime["gdp_history"].get(str(year), gdp*(1+scenario["gdp_growth"][0 if year<=2030 else 1 if year<=2040 else 2]))
        state, row = annual.step_numeric(state, runtime, scenario, year, next_gdp/gdp, rules)
        states.append(state.copy())
        metrics.append(row.copy())
        gdp = next_gdp
    return initial, np.array(states), np.array(metrics), runtime, scenario


def prepare(root=ROOT):
    root = Path(root).resolve()
    out = root/"outputs/v12/native"
    if (out/"annual_native_audit.json").exists():
        raise FileExistsError("A native execution receipt exists; keep its prepared inputs and references unchanged")
    out.mkdir(parents=True, exist_ok=True)
    selection_path = root/"outputs/v11/robust_repair/selected_before_independent_test.json"
    selection = json.loads(selection_path.read_text("utf-8"))
    planning = json.loads((root/"config/v6_annual_planning.json").read_text("utf-8"))
    factors, vector = controls_from_selection(selection, planning)
    source_names = ["src/v12_native_repair.py", "tests/test_v12_native_repair.py",
                    "simulation/v12/v12_annual_repair_audit.m", "src/v6_annual.py", "src/v5_planning.py",
                    "src/v5_annual.py", "src/v3_annual.py", "src/v3_annual_calibrated.py", "src/v3_calibration_candidate.py",
                    "config/v3_transition.json", "config/v6_annual_planning.json", "data/annual_china.csv",
                    "data/ember_power_china.csv", "data/supplement_carriers_2024.csv", "data/energy_production_china.csv",
                    "data/fuel_supply_trade.csv", "data/v6_fuel_import_channels.csv",
                    "simulation/v6/v6_annual_run.m", "simulation/v6/v6_annual_step.m", "simulation/v6/v6_build_annual_model.m",
                    "simulation/v6/V6_Energy_Transition_ABM.slx", "simulation/v3/v3_annual_expectations.m",
                    "simulation/v3/v3_annual_forecast_level.m", "simulation/v3/v3_execute.m", "simulation/v3/v3_live_monitor.m",
                    "outputs/v6/native/annual_native_inputs.mat", "outputs/v11/robust_repair/selected_before_independent_test.json",
                    "outputs/v11/robust_repair/analysis.json"]
    before = {name: digest(root/name) for name in source_names}
    data = annual.load_data(root)
    params = annual.calibrate(data, 2020)
    initial, states, metrics, runtime, scenario = state_reference(data, params, factors)
    native_initial = loadmat(root/"outputs/v6/native/annual_native_inputs.mat", simplify_cells=True)["initial_state"]
    np.testing.assert_allclose(initial, native_initial, rtol=0., atol=1e-10)
    if states.shape != (36,217) or metrics.shape != (36,132) or not np.isfinite(states).all() or not np.isfinite(metrics).all():
        raise AssertionError("Full finite 36-year state/metric references required")
    frame, actions = annual.simulate(data, params, factors)
    summary = summarize(frame, actions, data, planning)
    if len(frame)!=36 or len(actions)!=432:
        raise AssertionError("Incomplete nominal reference")
    errors = {}
    for name, value in summary.items():
        expected = float(selection["candidate"]["nominal_"+name])
        errors[name] = abs(value-expected)
        if not np.isclose(value, expected, rtol=1e-11, atol=1e-7):
            raise AssertionError("Frozen nominal result changed: "+name)
    # The only new native input contains time-constant policies, with no future values.
    savemat(out/"native_controls.mat", {"factors": factors}, long_field_names=True)
    # This file is loaded by the MATLAB audit only AFTER native simulation returns.
    savemat(out/"python_reference.mat", {"initial_state": initial, "state": states, "raw_metrics": metrics}, long_field_names=True)
    pd.DataFrame(states).to_csv(out/"python_states.csv", index=False, header=False)
    pd.DataFrame(metrics).to_csv(out/"python_raw_metrics.csv", index=False, header=False)
    frame.to_csv(out/"python_trace.csv", index=False)
    actions.to_csv(out/"python_actions.csv", index=False)
    (out/"python_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    after = {name: digest(root/name) for name in source_names}
    if before != after:
        raise AssertionError("Frozen source/input changed during Python preparation")
    output_names = ["native_controls.mat", "python_reference.mat", "python_states.csv", "python_raw_metrics.csv",
                    "python_trace.csv", "python_actions.csv", "python_summary.json"]
    manifest = {"status": "PASS_PYTHON_REFERENCE_PREPARED_NATIVE_EXECUTION_PENDING",
                "prepared_utc": datetime.now(timezone.utc).isoformat(), "command": "python -B -X utf8 -m src.v12_native_repair",
                "argv": sys.argv, "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                                    "pandas": pd.__version__, "scipy": scipy.__version__},
                "candidate_id": "repair_005", "decision_names": planning["decision_names"], "decision_vector": vector.tolist(),
                "factors": factors, "scenario": scenario, "initial_year": 2024, "final_year": 2060, "calibration_end": 2020,
                "annual_steps": 36, "native_state_width": 217, "raw_metric_width": 132, "actions": 432,
                "frozen_nominal_max_absolute_error": max(errors.values()),
                "runtime_retirement_fraction": runtime["engineering"]["retirement_fraction"],
                "source_input_hashes_before": [{"path": name, "sha256": value} for name,value in before.items()],
                "source_input_hashes_after": [{"path": name, "sha256": value} for name,value in after.items()],
                "prepared_artifacts": [{"path": "outputs/v12/native/"+name, "sha256": digest(out/name)} for name in output_names],
                "native_input_boundary": "V6 original observed initial state and fixed calibration plus native_controls.mat factors only; no future Python states/actions enter native simulation",
                "reference_boundary": "python_reference.mat and reference CSV/summary are audit-only; MATLAB entry loads them after v6_annual_run has completed",
                "matlab_entry": "addpath(fullfile(pwd,'simulation','v12')); report=v12_annual_repair_audit();",
                "claim": "Preparation is not MATLAB execution. Native parity and physical success require the separate actual execution receipt."}
    (out/"preparation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({"status": manifest["status"], "state_shape": states.shape, "metric_shape": metrics.shape,
                      "actions": len(actions), "net_2060_mt": summary["net_emissions_2060_mt"],
                      "discounted_capital_trillion_cny": summary["discounted_capital_trillion_cny"]}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    prepare(parser.parse_args().root)
