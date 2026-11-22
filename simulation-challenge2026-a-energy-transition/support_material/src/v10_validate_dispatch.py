"""Audit downloaded Julia/native actions from independently reconstructed states.

AI tool: OpenAI Codex / GPT-6, OpenAI; used 2026-10-18; release date unverified.
Every Julia rolling-LP objective is compared against a fresh SciPy LP at the
same downloaded pre-hour state. Tied optimal actions need not be identical.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import tomllib

import numpy as np
import pandas as pd
import scipy

from src import v3_hourly as h
from src.v10_prepare_inputs import configuration, make_case, OUT, ROOT


def config_from_csv(path):
    row = pd.read_csv(path).iloc[0].to_dict()
    starts = [int(row.pop("event1_start_hour")), int(row.pop("event2_start_hour"))]
    for name in ("hours", "event_duration_hours", "contract_lead_hours"):
        row[name] = int(row[name])
    for name in ("shock_severity", "contract_multiplier", "storage_power_multiplier"):
        row[name] = float(row[name])
    row["event_starts"] = starts
    return row


def _prefix_audit(case, table):
    """Retain the experiment's actual debt deadlines at an interrupted boundary."""
    count = len(table)
    cropped = deepcopy(case)
    cropped["hours"] = count
    for key, value in case.items():
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == case["hours"]:
            cropped[key] = value[:count]
    checks = h.audit(cropped, table)
    deadlines = (table.hour.to_numpy()+1) % case["config"]["deadline_hours"] == 0
    checks["debt_deadline_gwh"] = float(np.abs(table.shift_debt_gwh.to_numpy()[deadlines]).max()) if deadlines.any() else 0.0
    metadata = {"passed", "all_finite", "tolerance", "storage_simultaneous_operation_gw", "interpretation"}
    checks["passed"] = checks["all_finite"] and all(value <= checks["tolerance"]
                                                  for key, value in checks.items() if key not in metadata)
    checks["prefix_boundary"] = "Interrupted observation is not the experiment end; debt and all eight states carry forward."
    return checks


def validate(directory, name, *, allow_prefix=False):
    directory = Path(directory)
    config_path = directory / f"{name}_config.csv"
    trace_path = directory / f"{name}_actions_states.csv"
    config = config_from_csv(config_path)
    case = make_case(config)
    table = pd.read_csv(trace_path)
    assert 0 < len(table) <= config["hours"]
    if not allow_prefix:
        assert len(table) == config["hours"], "A complete validation requires every declared hour"
    assert (table.lp_status == "OPTIMAL").all()
    assert table.forecast_hours.tolist() == [min(24, config["hours"]-i) for i in range(len(table))]
    audit_table = table.copy()
    for j, f in enumerate(h.FUELS):
        audit_table[f+"_direct_served_mtce"] = case["direct_fuel_demand_mtce"][:len(table), j] - table[f+"_direct_unserved_mtce"]
        audit_table[f+"_power_burn_mtce"] = table[f+"_gw"] / (8141 * case["fuel_efficiency"][f]) if f in {"coal", "gas"} else 0.0
    physical = _prefix_audit(case, audit_table) if len(table) < config["hours"] else h.audit(case, audit_table)
    assert physical["passed"], physical
    # Objective equality is invariant to solver tie choices, unlike action equality.
    original_linprog = h.linprog
    captured = {}
    def solve_and_capture(*args, **kwargs):
        result = original_linprog(*args, **kwargs)
        captured["objective"] = float(result.fun) if result.success else None
        return result
    state = {"fuel": np.array(case["initial_fuel_stocks_mtce"], float),
             "thermal": deepcopy(case["initial_thermal_gw"]),
             "soc": {s: case["storage"][s]["initial_soc_gwh"] for s in h.STORES},
             "debt": 0.0,
             "previous_available": {f: case["available_gw"][0, h.GENERATORS.index(f)] for f in ("coal", "gas")}}
    objective_errors = []
    action_errors = []
    h.linprog = solve_and_capture
    try:
        for hour, row in table.iterrows():
            action = h._plan(case, hour, state, config["strategy"])
            objective_errors.append(abs(captured["objective"] - row.lp_objective_million_cny))
            comparisons = [(action["gen"], row[[g+"_gw" for g in h.GENERATORS]].to_numpy(float)),
                           (action["ch"], row[[s+"_charge_gw" for s in h.STORES]].to_numpy(float)),
                           (action["dis"], row[[s+"_discharge_gw" for s in h.STORES]].to_numpy(float)),
                           (action["direct_short"], row[[f+"_direct_unserved_mtce" for f in h.FUELS]].to_numpy(float)),
                           (np.array([action[k] for k in ("out", "repay", "unserved", "spill")], float),
                            row[["shift_out_gw", "shift_in_gw", "unserved_gw", "spill_gw"]].to_numpy(float))]
            action_errors.append(max(float(np.max(np.abs(a-b))) for a, b in comparisons))
            state["fuel"] = row[[f+"_stock_mtce" for f in h.FUELS]].to_numpy(float)
            state["soc"] = {s: float(row[s+"_soc_gwh"]) for s in h.STORES}
            state["thermal"] = {f: float(row[f+"_gw"]) for f in ("coal", "gas")}
            state["previous_available"] = {f: case["available_gw"][hour, h.GENERATORS.index(f)] for f in ("coal", "gas")}
            state["debt"] = float(row.shift_debt_gwh)
    finally:
        h.linprog = original_linprog
    errors = np.array(objective_errors)
    scale = np.maximum(1, np.abs(table.lp_objective_million_cny.to_numpy()))
    assert np.all(errors <= 1e-5 + 1e-8*scale), f"Rolling LP objective mismatch: {errors.max()}"
    record = {"status": ("PASS_INDEPENDENT_VALID_PREFIX_PHYSICS_AND_ALL_SAME_STATE_ROLLING_LP_OBJECTIVES"
                         if len(table) < config["hours"] else "PASS_INDEPENDENT_PHYSICS_AND_ALL_SAME_STATE_ROLLING_LP_OBJECTIVES"),
              "case": name, "config": config, "hours": len(table),
              "declared_hours": config["hours"], "complete": len(table) == config["hours"],
              "plant_backends": table.plant_backend.unique().tolist(), "physical_audit": physical,
              "max_same_state_objective_error_million_cny": float(errors.max()),
              "max_same_state_objective_relative_error": float((errors/scale).max()),
              "max_tied_first_action_difference": max(action_errors),
              "independent_environment": {"python": sys.version, "scipy": scipy.__version__,
                                          "numpy": np.__version__, "pandas": pd.__version__,
                                          "solver_interface": "scipy.optimize.linprog(method=highs)"},
              "interpretation": "Objective equivalence and feasible actual states establish LP equivalence; alternate tied first actions are allowed.",
              "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
              "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"):
                  hashlib.sha256(path.read_bytes()).hexdigest() for path in
                  [Path(__file__), ROOT/"src/v10_prepare_inputs.py", ROOT/"src/v3_hourly.py",
                   ROOT/"outputs/v6/resilience/native_inputs.json"]}}
    if allow_prefix and len(table) < config["hours"]:
        expected_recovery = _recovery_reference(case, table)
        if expected_recovery:
            recovery = pd.read_csv(directory/f"{name}_recovery.csv")
            assert len(recovery) == len(expected_recovery)
            for index, expected_row in enumerate(expected_recovery):
                for field, expected in expected_row.items():
                    _assert_value(recovery.iloc[index][field], expected,
                                  f"{name}:observed event{index+1}:{field}")
        record["observed_recovery_windows"] = expected_recovery
    target = OUT / f"independent_{name}.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2), "utf-8")
    print(name, record["status"], errors.max(), flush=True)
    return record


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _recovery_reference(case, table):
    """Recompute actual completed windows without using Julia recovery output."""
    config = case["v10_config"]
    starts = config["event_starts"]
    rows = []
    for event, start in enumerate(starts):
        stop = start + config["event_duration_hours"]
        designed_end = starts[event+1] if event+1 < len(starts) else config["hours"]
        end = min(designed_end, len(table))
        if len(table) < stop:
            continue
        pre = (case["initial_fuel_stocks_mtce"] if start == 0 else
               table.loc[start-1, [f+"_stock_mtce" for f in h.FUELS]].to_numpy(float))
        target = .95 * pre
        stocks = table[[f+"_stock_mtce" for f in h.FUELS]].to_numpy()
        def ready(first):
            last = first+24
            part = table.iloc[first:last]
            service = part.unserved_gw.sum() <= .01*part.demand_gw.sum()+1e-8
            service &= all(part[f+"_direct_unserved_mtce"].sum() <=
                           .01*case["direct_fuel_demand_mtce"][first:last, j].sum()+1e-8
                           for j, f in enumerate(h.FUELS))
            return service, bool(np.all(stocks[first:last] >= target-1e-8))
        service_start = joint_start = None
        for first in range(stop, end-23):
            service, stock = ready(first)
            if service and service_start is None:
                service_start = first
            if service and stock and joint_start is None:
                joint_start = first
        rebreach = None
        if joint_start is not None:
            for hour in range(joint_start+24, end):
                if np.any(stocks[hour] < target-1e-8):
                    rebreach = hour
                    break
        service, stock = ready(end-24) if end >= stop+24 else (False, False)
        row = dict(event=event+1, start_hour=start, shock_end_hour=stop,
                   observation_end_hour=end, first_service_24h_start=service_start,
                   first_joint_24h_start=joint_start,
                   service_confirmation_hour=None if service_start is None else service_start+23,
                   joint_confirmation_hour=None if joint_start is None else joint_start+23,
                   service_wait_hours=None if service_start is None else service_start-stop,
                   joint_wait_hours=None if joint_start is None else joint_start-stop,
                   service_right_censored=service_start is None, joint_right_censored=joint_start is None,
                   observation_complete=end == designed_end, final_24h_joint_ready=bool(service and stock),
                   stock_rebreach_hour_after_joint_confirmation=rebreach)
        for j, fuel in enumerate(h.FUELS):
            row["target_"+fuel+"_mtce"] = target[j]
            row[fuel+"_boundary_ratio"] = stocks[end-1, j]/pre[j] if pre[j] > 1e-12 else None
        rows.append(row)
    return rows


def _assert_value(actual, expected, context):
    if expected is None:
        assert pd.isna(actual), (context, actual, expected)
    elif isinstance(expected, (bool, np.bool_)):
        assert actual == expected, (context, actual, expected)
    elif isinstance(expected, (int, float, np.number)):
        assert np.isclose(actual, expected, atol=1e-6, rtol=1e-9), (context, actual, expected)
    else:
        assert actual == expected, (context, actual, expected)


def _verify_segmented_resume(directory, batch, expected_sources, source_hashes):
    """Verify actual restart evidence rather than treating it as one old batch."""
    assert batch["status"] == "COMPLETED_EXPLICITLY_SEGMENTED_NATIVE_FEEDBACK_COHORT"
    assert batch["kernel_segments"] == 2 and batch["total_unique_successful_hours"] == 1800
    assert batch["execution_mode"] == "ORIGINAL_KERNEL_INTERRUPTED_THEN_RESTORED_FROM_VALIDATED_NATIVE_PREFIX"
    resume_sources = batch["resume_source_sha256"]
    assert set(resume_sources) == {"resume_v10_remaining.jl"}
    for name, digest in resume_sources.items():
        local = ROOT / "simulation/anymath_v10" / name
        assert _sha(local) == _sha(directory/name) == digest
        source_hashes[_relative(local)] = digest
    prior_directory = ROOT / "outputs/v10/anymath/cloud_interrupted"
    archive = ROOT / "outputs/v10/anymath/V10_interrupted_export.tar"
    assert _sha(archive) == batch["interrupted_archive_sha256"]
    prior = batch["prior_evidence_sha256"]
    required = {"batch_failure.toml", "power_failed_prefix_actions_states.csv",
                "power_failed_prefix_physical_audit.csv", "power_failed_prefix_config.csv",
                "power_failed_prefix_config.txt", "power_failed_prefix_summary.csv",
                "power_failed_prefix_recovery.csv"}
    required.update(f"{name}_execution.toml" for name in ("base", "mild", "contract"))
    assert required <= set(prior)
    for name, digest in prior.items():
        assert Path(name).name == name
        assert _sha(prior_directory/name) == _sha(directory/name) == digest, "Prior evidence changed: "+name
    failure = tomllib.loads((directory/"batch_failure.toml").read_text("utf-8"))
    assert failure["status"] == "FAILED" and failure["case_name"] == "power" and failure["completed_case_count"] == 3
    assert failure["source_sha256"] == expected_sources and failure["package_versions"] == batch["package_versions"]
    assert all(token in failure["error"] for token in ("stream is closed or unusable", "RemoteTerminal", "println"))
    prefix_report = validate(directory, "power_failed_prefix", allow_prefix=True)
    prefix = pd.read_csv(directory/"power_failed_prefix_actions_states.csv", float_precision="round_trip")
    final = pd.read_csv(directory/"power_actions_states.csv", float_precision="round_trip")
    count = len(prefix)
    assert count == 174 and prefix_report["hours"] == count and not prefix_report["complete"]
    assert config_from_csv(directory/"power_failed_prefix_config.csv") == configuration(storage_power_multiplier=1.5)
    pd.testing.assert_frame_equal(final.iloc[:count].reset_index(drop=True), prefix,
                                  check_dtype=False, check_exact=True)
    power = tomllib.loads((directory/"power_execution.toml").read_text("utf-8"))
    dec = tomllib.loads((directory/"decentralized_execution.toml").read_text("utf-8"))
    assert power["execution_mode"] == "RESUMED_AFTER_KERNEL_RESTART"
    assert dec["execution_mode"] == "NEW_CASE_AFTER_KERNEL_RESTART"
    assert power["resumed_native_prefix_hours"] == count and power["new_native_transition_calls"] == 360-count
    assert dec["resumed_native_prefix_hours"] == 0 and dec["new_native_transition_calls"] == 360
    last = prefix.iloc[-1]
    expected_state = dict(fuel=[last[f+"_stock_mtce"] for f in h.FUELS],
                          soc=[last[s+"_soc_gwh"] for s in h.STORES],
                          thermal=[last.coal_gw, last.gas_gw], debt=last.shift_debt_gwh)
    for key, expected in expected_state.items():
        assert np.array_equal(np.asarray(power["resume_initial_state"][key]), np.asarray(expected)), "Resume state mismatch: "+key
    for metadata, start in ((power, count), (dec, 0)):
        assert metadata["current_segment_first_hour"] == start and metadata["current_segment_last_hour"] == 359
        assert metadata["prefix_fieldwise_identity"] is True
        assert metadata["prior_evidence_sha256"] == prior and metadata["resume_source_sha256"] == resume_sources
        assert metadata["interrupted_archive_sha256"] == batch["interrupted_archive_sha256"]
        assert metadata["environment_versions"] == batch["environment_versions"]
        assert metadata["native_power_residual_scope"] == "current successful resume segment; prefix native exact maximum was not persisted"
        assert metadata["current_segment_wall_seconds"]+1e-6 >= metadata["current_segment_recorded_step_seconds"] >= 0
        summary = pd.read_csv(directory/f"{metadata['case_name']}_summary.csv").iloc[0]
        _assert_value(summary.elapsed_seconds, metadata["prefix_recorded_step_seconds"]+
                      metadata["current_segment_recorded_step_seconds"], "Recorded step time excludes idle gap")
    prior_seconds = pd.read_csv(directory/"power_failed_prefix_summary.csv").elapsed_seconds.iloc[0]
    _assert_value(power["prefix_recorded_step_seconds"], prior_seconds, "Original recorded prefix seconds")
    prior_power_residual = pd.read_csv(directory/"power_failed_prefix_physical_audit.csv").power_residual_gw.max()
    _assert_value(power["prefix_reconstructed_power_residual_gw"], prior_power_residual,
                  "Prefix reconstructed residual is separate from new native signal maximum")
    assert batch["prior_successful_unique_hours"] == 1080+count
    assert batch["current_restart_new_native_transition_calls"] == 720-count
    assert batch["prior_successful_unique_hours"]+batch["current_restart_new_native_transition_calls"] == 1800
    report = dict(status="PASS_EXPLICIT_RESTART_PREFIX_IDENTITY_AND_8_STATE_CONTINUATION",
                  original_completed_cases=3, successful_power_prefix_hours=count,
                  new_power_native_calls=360-count, new_decentralized_native_calls=360,
                  current_restart_new_native_calls=720-count, unique_successful_hours=1800,
                  resume_initial_state=expected_state, prefix_fieldwise_identity=True,
                  original_failure_path=_relative(directory/"batch_failure.toml"),
                  original_failure_sha256=_sha(directory/"batch_failure.toml"),
                  interrupted_archive_sha256=_sha(archive), resume_source_sha256=resume_sources,
                  native_residual_scope="Prior native exact maximum unavailable; prior reconstructed physics and new native signal maximum are separate.",
                  timing_scope="Prior recorded step groups plus current recorded step groups; restart wall separate; no idle gap imputed.")
    report_path = OUT / "independent_segmented_resume.json"
    report_path.write_text(json.dumps(_json_value(report), ensure_ascii=False, indent=2), "utf-8")
    return dict(segmented=True, kernel_segments=2, prior_successful_hours=1080+count,
                current_restart_new_native_calls=720-count,
                report_paths=[OUT/"independent_power_failed_prefix.json", report_path],
                artifact_paths=[archive, ROOT/"outputs/v10/anymath/interruption_receipt.json"])


def _validated_receipt(directory, artifacts):
    """Accept only the complete five-case native cohort, never partial results."""
    directory = Path(directory).resolve()
    target = ROOT / "outputs/v10/anymath/execution_receipt.json"
    manifest_path = ROOT / "outputs/v10/anymath/local_source_hashes.json"
    expected_sources = json.loads(manifest_path.read_text("utf-8"))
    source_names = {"v10_dispatch.jl", "v10_config.jl", "v10_native_plant.jl", "v10_gui.jl",
                    "v10_app.jl", "v10_dashboard.html", "start_v10.jl", "v10_run_cases.jl",
                    "v10_base_inputs.csv"}
    assert set(expected_sources) == source_names
    source_hashes = {}
    downloaded_source_files = []
    for name, digest in expected_sources.items():
        source = ROOT / "simulation/anymath_v10" / name
        assert _sha(source) == digest.lower(), f"Current source drift: {name}"
        source_hashes[_relative(source)] = digest.lower()
        downloaded = directory / name
        if downloaded.is_file():
            assert _sha(downloaded) == digest.lower(), f"Downloaded source drift: {name}"
            downloaded_source_files.append(_relative(downloaded))
    segmented = (directory / "segmented_execution.toml").is_file()
    batch_path = directory / ("segmented_execution.toml" if segmented else "batch_execution.toml")
    batch = tomllib.loads(batch_path.read_text("utf-8"))
    if segmented:
        assert not (directory / "batch_execution.toml").exists(), "Do not fabricate the missing original completion record"
        resume = _verify_segmented_resume(directory, batch, expected_sources, source_hashes)
    else:
        assert batch["status"] == "COMPLETED_FIVE_SEQUENTIAL_NATIVE_FEEDBACK_CASES"
        resume = dict(segmented=False, kernel_segments=1, report_paths=[], artifact_paths=[])
    assert batch["case_count"] == 5 and batch["single_native_model_sequential"] is True
    assert batch["source_sha256"] == expected_sources
    assert batch["native_transition_calls_total"] == 1800
    expected_configs = {
        "base": configuration(), "mild": configuration(shock_severity=.5),
        "contract": configuration(supply_policy="contract"),
        "power": configuration(storage_power_multiplier=1.5),
        "decentralized": configuration(strategy="decentralized")}
    assert [record["case_name"] for record in batch["cases"]] == list(expected_configs)
    versions = batch["package_versions"]
    assert all(isinstance(versions.get(name), str) and versions[name]
               for name in ("Julia", "JuMP", "HiGHS"))
    actual_environment = tomllib.loads((directory / "environment_versions.toml").read_text("utf-8"))
    assert all(actual_environment[name] == value for name, value in versions.items())
    assert isinstance(actual_environment.get("HiGHS_solver"), str) and actual_environment["HiGHS_solver"]
    if segmented:
        assert batch["environment_versions"] == actual_environment
    records = []
    summary_frames, recovery_frames = [], []
    validation_files = list(resume["report_paths"])
    for name, expected_config in expected_configs.items():
        metadata = tomllib.loads((directory / f"{name}_execution.toml").read_text("utf-8"))
        assert metadata == next(row for row in batch["cases"] if row["case_name"] == name)
        assert metadata["status"] == "COMPLETED"
        assert metadata["parameters"] == expected_config
        assert metadata["source_sha256"] == expected_sources
        assert metadata["package_versions"] == versions
        assert metadata["plant_backend"] == "ANYMATH_NATIVE_BLOCK_FEEDBACK"
        assert metadata["information_contract"] == "COMMON_24H_PERFECT_FORECAST_ROLLING_FIRST_ACTION"
        assert metadata["executed_hours"] == metadata["optimal_lp_hours"] == metadata["native_transition_calls"] == 360
        assert metadata["resumed_native_prefix_hours"] + metadata["new_native_transition_calls"] == 360
        assert metadata["maximum_physical_residual"] < 1e-6
        assert metadata["native_maximum_power_residual_gw"] < 1e-6
        assert config_from_csv(directory / f"{name}_config.csv") == expected_config
        report = validate(directory, name)
        assert report["plant_backends"] == ["ANYMATH_NATIVE_BLOCK_FEEDBACK"]
        report_path = OUT / f"independent_{name}.json"
        validation_files.append(report_path)
        case = make_case(expected_config)
        table = pd.read_csv(directory / f"{name}_actions_states.csv")
        for j, fuel in enumerate(h.FUELS):
            assert np.allclose(table[fuel+"_emergency_mtce"], case["emergency_arrivals_mtce"][:, j],
                               atol=1e-12, rtol=1e-12), name+":emergency arrival export "+fuel
        audit_table = table.copy()
        for j, fuel in enumerate(h.FUELS):
            audit_table[fuel+"_direct_served_mtce"] = case["direct_fuel_demand_mtce"][:, j]-table[fuel+"_direct_unserved_mtce"]
            audit_table[fuel+"_power_burn_mtce"] = table[fuel+"_gw"]/(8141*case["fuel_efficiency"][fuel]) if fuel in {"coal", "gas"} else 0.0
        summary = pd.read_csv(directory / f"{name}_summary.csv").iloc[0].to_dict()
        assert summary["complete"] == True and summary["executed_hours"] == 360
        assert summary["plant_backend"] == "ANYMATH_NATIVE_BLOCK_FEEDBACK"
        assert summary["optimal_lp_hours"] == 360
        for field in ("design", "strategy", "supply_policy", "shock_severity", "event_duration_hours",
                      "storage_power_multiplier", "contract_multiplier"):
            _assert_value(summary[field], expected_config[field], name+":parameter "+field)
        independently = h.summarize(case, audit_table, expected_config["strategy"])
        for field in ("unserved_gwh", "peak_unserved_gw", "loss_of_load_hours", "direct_fuel_unserved_mtce",
                      "electricity_co2_mt", "direct_fuel_co2_mt", "total_co2_mt", "terminal_shift_debt_gwh",
                      "shortage_penalty_million_cny", "carbon_cost_million_cny", "stock_holding_cost_million_cny",
                      "fuel_arrival_purchase_value_million_cny"):
            _assert_value(summary[field], independently[field], name+":"+field)
        resource = (independently["resource_accounted_cost_million_cny"]-
                    independently["shortage_penalty_million_cny"]-independently["carbon_cost_million_cny"])
        emergency_value = float(case["emergency_arrivals_mtce"].sum(axis=0) @
                                np.array([case["config"]["fuel_value_million_cny_mtce"][f] for f in h.FUELS]))
        physical_cost = resource+independently["stock_holding_cost_million_cny"]+.5*emergency_value
        _assert_value(summary["contract_base_value_million_cny"], emergency_value, name+":contract base value")
        _assert_value(summary["contract_premium_million_cny"], .5*emergency_value, name+":contract premium")
        warmup = table.hour < expected_config["event_starts"][0]
        _assert_value(summary["warmup_unserved_gwh"], table.loc[warmup, "unserved_gw"].sum(), name+":warmup ENS")
        _assert_value(summary["post_warmup_unserved_gwh"], table.loc[~warmup, "unserved_gw"].sum(), name+":post-warmup ENS")
        _assert_value(summary["resource_accounted_cost_million_cny"], resource, name+":resource cost")
        _assert_value(summary["physical_operating_cost_with_contract_premium_million_cny"], physical_cost, name+":physical cost")
        _assert_value(summary["penalty_inclusive_cost_million_cny"], physical_cost+
                      independently["shortage_penalty_million_cny"]+independently["carbon_cost_million_cny"], name+":penalty-inclusive cost")
        for fuel in h.FUELS:
            _assert_value(summary[fuel+"_terminal_stock_mtce"], table[fuel+"_stock_mtce"].iloc[-1], name+":terminal "+fuel)
        physical_csv = pd.read_csv(directory / f"{name}_physical_audit.csv")
        assert physical_csv.hour.tolist() == list(range(360))
        residuals = physical_csv.drop(columns="hour").to_numpy(float)
        assert np.isfinite(residuals).all() and residuals.min() >= 0 and residuals.max() < 1e-6
        _assert_value(metadata["maximum_physical_residual"], residuals.max(), name+":reported residual")
        _assert_value(summary["maximum_physical_residual"], residuals.max(), name+":summary residual")
        recovery = pd.read_csv(directory / f"{name}_recovery.csv")
        expected_recovery = _recovery_reference(case, table)
        assert len(recovery) == 2
        for index, expected_row in enumerate(expected_recovery):
            for field, expected in expected_row.items():
                _assert_value(recovery.iloc[index][field], expected, f"{name}:event{index+1}:{field}")
        records.append(dict(name=name, parameters=expected_config, hours=360,
                            nativecalls=metadata["native_transition_calls"],
                            new_nativecalls=metadata["new_native_transition_calls"],
                            resumed_native_prefix_hours=metadata["resumed_native_prefix_hours"],
                            metrics=summary, recovery=expected_recovery,
                            validation_report=_relative(report_path), validation_report_sha256=_sha(report_path),
                            max_same_state_objective_error_million_cny=report["max_same_state_objective_error_million_cny"]))
        summary_frames.append(pd.read_csv(directory / f"{name}_summary.csv").assign(case_name=name))
        recovery_frames.append(recovery.assign(case_name=name))
    assert batch["new_native_transition_calls_total"] == sum(row["new_nativecalls"] for row in records)
    assert batch["resumed_native_prefix_hours_total"] == sum(row["resumed_native_prefix_hours"] for row in records)
    summary_name = "segmented_summary.csv" if segmented else "batch_summary.csv"
    recovery_name = "segmented_recovery.csv" if segmented else "batch_recovery.csv"
    for file, frames in ((summary_name, summary_frames), (recovery_name, recovery_frames)):
        expected = pd.concat(frames, ignore_index=True).sort_index(axis=1)
        actual = pd.read_csv(directory / file).sort_index(axis=1)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-9, atol=1e-6)
    for dependency in (Path(__file__), ROOT/"src/v10_prepare_inputs.py", ROOT/"src/v3_hourly.py",
                       ROOT/"outputs/v6/resilience/native_inputs.json"):
        source_hashes[_relative(dependency)] = _sha(dependency)
    raw_files = {path for path in directory.rglob("*") if path.is_file() and
                 path.suffix.lower() in {".csv", ".toml", ".engee", ".jl", ".html", ".txt", ".log",
                                         ".svg", ".png", ".jpg", ".jpeg", ".pdf"}}
    for artifact in [*artifacts, *resume["artifact_paths"]]:
        path = Path(artifact).resolve()
        if path.is_dir():
            raw_files.update(item for item in path.rglob("*") if item.is_file())
        else:
            raw_files.add(path)
    raw_files.update(validation_files)
    raw_files.add(manifest_path)
    raw_files.discard(target.resolve())
    receipt = dict(status="PASS_FIVE_ANYMATH_NATIVE_CLOSED_LOOP_REOPTIMIZATION_CASES_AND_INDEPENDENT_1800H_VALIDATION",
                   software="AnyMath native discrete resource network + Julia JuMP/HiGHS rolling MPC",
                   case_count=5, total_hours=1800, total_native_transition_calls=1800,
                   new_native_transition_calls=batch["new_native_transition_calls_total"],
                   resumed_native_prefix_hours=batch["resumed_native_prefix_hours_total"],
                   segmented=segmented, kernel_segments=resume["kernel_segments"],
                   current_restart_new_native_calls=resume.get("current_restart_new_native_calls"),
                   aggregate_summary_path=_relative(directory/summary_name),
                   aggregate_recovery_path=_relative(directory/recovery_name),
                   package_versions=actual_environment, source_sha256=source_hashes,
                   independent_environment={"python": sys.version, "scipy": scipy.__version__,
                                            "numpy": np.__version__, "pandas": pd.__version__,
                                            "solver_interface": "scipy.optimize.linprog(method=highs)"},
                   downloaded_source_files_checked=downloaded_source_files,
                   artifact_sha256={_relative(path): _sha(path) for path in sorted(raw_files)},
                   cases=records, validation_reports=[dict(path=_relative(path), sha256=_sha(path)) for path in validation_files],
                   interpretation=("Five designed 360h experiments, explicitly completed across a preserved kernel interruption and state continuation. " if segmented else "Five designed 360h experiments. ")+
                                  "Every actual native end-state feeds the next fresh 24h LP; recovery windows use completed observations only. Independent SciPy re-solves confirm every same-state window objective.",
                   batch_execution_path=_relative(batch_path), batch_execution_sha256=_sha(batch_path))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_json_value(receipt), ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
    print(receipt["status"], target, flush=True)
    return receipt


def execution_receipt(directory, artifacts=()):
    """Write a failed gate on any actual validation error, then preserve failure."""
    try:
        return _validated_receipt(directory, artifacts)
    except Exception as failure:
        target = ROOT / "outputs/v10/anymath/execution_receipt.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(dict(status="FAILED_ANYMATH_EXECUTION_VALIDATION",
                                         directory=str(Path(directory)),
                                         error=str(failure),
                                         validator_sha256=_sha(__file__)),
                                     ensure_ascii=False, indent=2), "utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--receipt", action="store_true")
    parser.add_argument("--prefix", action="store_true")
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.receipt:
        execution_receipt(args.directory, args.artifact)
    else:
        assert args.names, "Specify case names, or --receipt for the complete cohort"
        for name in args.names:
            validate(args.directory, name, allow_prefix=args.prefix)
