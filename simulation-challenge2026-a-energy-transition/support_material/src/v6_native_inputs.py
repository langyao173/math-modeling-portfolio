"""Export fixed V6 decisions/initial state; keep future Python audit data separate.

annual_native_inputs.mat contains no future capacity, generation or order path.
annual_python_reference.mat is loaded by the audit only, never by the model.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from src import v6_annual as model
from src.v6_annual_experiments import sha
from src.v6_planning import decode


ROOT = Path(__file__).resolve().parents[1]


def run(root=ROOT):
    root = Path(root).resolve(); out = root / "outputs/v6/native"; out.mkdir(parents=True, exist_ok=True)
    data = model.load_data(root); params = model.calibrate(data, 2020)
    params.pop("training_cv_records")
    initial, runtime = model.initialize_state(data, params, 2024)
    cases = {name: {"scenario": deepcopy(scenario), "mechanisms": model.resolve_mechanisms()}
             for name, scenario in data["config"]["scenarios"].items()}
    for name, rules in (("consumer_frozen", {"consumer_choice": "frozen_initial"}),
                        ("consumer_response_off", {"consumer_choice": "exogenous_trend"}),
                        ("zero_coal_channel", {"coal_import_channel": "zero_channel"}),
                        ("zero_matched_supply", {"coal_import_channel": "zero_matched_supply"}),
                        ("smooth_price", {"market_price_rule": "smooth"}),
                        ("government_feedback_off", {"carbon_feedback": False}),
                        ("minimum_one_year", {"construction_years": "minimum_one_year"}),
                        ("investor_profit_off", {"investor_profit_response": False}),
                        ("legacy_v4", {"investor_profit_rule": "legacy_v4"})):
        cases[name] = {"scenario": deepcopy(data["config"]["scenarios"]["baseline"]),
                       "mechanisms": model.resolve_mechanisms(rules)}
    selected_path = root / "outputs/v6/planning/selected_cases.csv"
    selected = pd.read_csv(selected_path)
    planning = json.loads((root / "config/v6_annual_planning.json").read_text("utf-8"))
    planning_receipt = json.loads((root / "outputs/v6/planning/planning_audit.json").read_text("utf-8"))
    if not planning_receipt["status"].startswith("PASS_COMPLETE_FIXED_SEARCH"):
        raise AssertionError("Complete planning search required before native selected-case checks")
    for _, row in selected.iterrows():
        scenario = model._decision_scenario(data, decode([float(row[name]) for name in planning["decision_names"]], str(row.case_id)))
        cases[str(row.case_id)] = {"scenario": scenario, "mechanisms": model.resolve_mechanisms()}
    role_ids = {}
    for role in ("frontier_knee", "minimum_net_emissions"):
        matched = selected.loc[selected.roles.fillna("").str.split("|", regex=False).map(lambda values: role in values)]
        if len(matched) != 1: raise AssertionError("One selected candidate required for role " + role)
        role_ids[role] = str(matched.iloc[0].case_id)
    audit_names = list(dict.fromkeys(["baseline", "policy", "risk", "technology", "consumer_frozen", "consumer_response_off", "zero_coal_channel", "zero_matched_supply", "smooth_price", "government_feedback_off", "minimum_one_year", "investor_profit_off", "legacy_v4", *role_ids.values()]))
    native_path = out / "annual_native_inputs.mat"
    # Only common observations/calibration plus time-constant decisions enter the
    # native model. The separate reference file is not a Simulink input.
    savemat(native_path, {"initial_state": initial, "parameters": runtime, "cases": cases,
                          "audit_case_names": np.array(audit_names, dtype=object)}, long_field_names=True)
    (out / "annual_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), "utf-8")
    references = {}; case_records = []
    for name, definition in cases.items():
        scenario = definition["scenario"]; rules = definition["mechanisms"]
        frame, actions = model.simulate(data, params, scenario, mechanisms=rules)
        frame_path = out / ("annual_python_" + name + ".csv"); actions_path = out / ("annual_python_" + name + "_actions.csv")
        frame.to_csv(frame_path, index=False); actions.to_csv(actions_path, index=False)
        state, case_runtime = model.initialize_state(data, params, 2024, mechanisms=rules)
        state[212] = scenario.get("carbon_price_start_cny_t", 100.)
        if scenario.get("clear_initial_pipeline", False): state[20:model.QEND] = 0.
        state_rows, metric_rows = [], []; gdp = case_runtime["base_gdp"]
        for year in range(2025, 2061):
            next_gdp = case_runtime["gdp_history"].get(str(year), gdp * (1 + scenario["gdp_growth"][0 if year <= 2030 else 1 if year <= 2040 else 2]))
            state, metrics = model.step_numeric(state, case_runtime, scenario, year, next_gdp/gdp, rules)
            gdp = next_gdp; state_rows.append(state.copy()); metric_rows.append(metrics.copy())
        references[name] = {"state": np.array(state_rows), "metrics": np.array(metric_rows)}
        case_records.append({"case": name, "scenario": scenario, "mechanisms": rules,
                             "trace": str(frame_path.relative_to(root)).replace("\\", "/"), "trace_sha256": sha(frame_path),
                             "actions": str(actions_path.relative_to(root)).replace("\\", "/"), "actions_sha256": sha(actions_path)})
        if name.startswith("plan_"):
            planning_trace = root / "outputs/v6/planning" / ("trajectory_" + name + ".csv")
            planning_actions = root / "outputs/v6/planning" / ("actions_" + name + ".csv")
            expected = pd.read_csv(planning_trace); expected_actions = pd.read_csv(planning_actions)
            for actual, saved in ((frame, expected), (actions, expected_actions)):
                for column in actual.select_dtypes(include=np.number):
                    np.testing.assert_allclose(actual[column], saved[column], rtol=1e-10, atol=1e-7, equal_nan=True)
    reference_path = out / "annual_python_reference.mat"
    savemat(reference_path, {"references": references}, long_field_names=True)
    sources = ["src/v6_native_inputs.py", "src/v6_annual.py", "src/v6_planning.py", "src/v3_annual.py", "src/v3_annual_calibrated.py", "src/v3_calibration_candidate.py",
               "config/v3_transition.json", "config/v6_annual_planning.json", "data/annual_china.csv", "data/ember_power_china.csv", "data/supplement_carriers_2024.csv", "data/energy_production_china.csv", "data/fuel_supply_trade.csv", "data/v6_fuel_import_channels.csv", "src/v5_annual.py",
               "outputs/v6/planning/planning_audit.json", "outputs/v6/planning/selected_cases.csv"]
    manifest = {"status": "FIXED_INPUTS_AND_SEPARATE_AUDIT_REFERENCES_EXPORTED", "generated_utc": datetime.now(timezone.utc).isoformat(),
                "native_input_scope": "Observed initial state, fixed calibration, mechanism definitions and constant decision policies only. No future Python controls/states are passed to Simulink.",
                "audit_reference_scope": "Separate audit-only file contains Python reference states and metrics; native runner does not load it.",
                "native_inputs_sha256": sha(native_path), "python_reference_sha256": sha(reference_path),
                "case_definitions_sha256": sha(out / "annual_cases.json"),
                "role_ids": role_ids, "audit_case_names": audit_names, "cases": case_records,
                "source_input_sha256": {name: sha(root / name) for name in sources}}
    (out / "annual_inputs_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({"native_cases_exported": len(cases), "audit_case_names": audit_names, "role_ids": role_ids,
                      "native_inputs_sha256": manifest["native_inputs_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
