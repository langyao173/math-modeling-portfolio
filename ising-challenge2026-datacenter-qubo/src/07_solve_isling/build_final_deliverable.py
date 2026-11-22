"""Build the final TAOLIN-DC-SIM solution deliverable package."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
DELIVERABLE_DIR = ROOT / "deliverables" / "final_solution_package"
ZIP_PATH = ROOT / "deliverables" / "TAOLIN_DC_SIM_final_solution_package.zip"
INDUSTRIAL_ARTIFACTS_DIR = ROOT / "deliverables" / "industrial_ising_artifacts"

BEST_RESULT_JSON = REPORTS_DIR / "qubo_solver_results_full_cloudseed_peakguard.json"
BEST_RISK_CSV = REPORTS_DIR / "qubo_risk_accepted_actions_full_cloudseed_peakguard.csv"
BEST_ACCEPTED_CSV = REPORTS_DIR / "qubo_accepted_actions_full_rec_cvar0p10_cut3_truepolish5_ms_local.csv"
PROMOTION_REPORT = REPORTS_DIR / "qubo_solver_report_full_cloudseed_peakguard.md"
LOCAL_SUMMARY = REPORTS_DIR / "local_recourse_multistart_summary.md"
RECOURSE_SUMMARY = REPORTS_DIR / "recourse_qubo_summary.md"
CLOUD_SUMMARY = REPORTS_DIR / "cloud_scale_summary.md"
SOLVER_SCRIPT = ROOT / "src" / "07_solve_isling" / "solve_qubo_platform.py"
SOLVER_README = ROOT / "src" / "07_solve_isling" / "README.md"
BUILDER_SCRIPT = ROOT / "src" / "07_solve_isling" / "build_final_deliverable.py"
PROMOTION_SCRIPT = ROOT / "src" / "07_solve_isling" / "promote_cloud_candidate.py"
VERIFIER_SCRIPT = ROOT / "src" / "07_solve_isling" / "verify_submission.py"
FINAL_REPORT = DELIVERABLE_DIR / "FINAL_REPORT.md"
FINAL_METRICS = DELIVERABLE_DIR / "FINAL_METRICS.json"
FINAL_SOLUTION = DELIVERABLE_DIR / "FINAL_SOLUTION.csv"
FULL_SELECTED_ACTIONS = DELIVERABLE_DIR / "FULL_SELECTED_ACTIONS.csv"
QUBO_ACCEPTED_ACTIONS = DELIVERABLE_DIR / "QUBO_ACCEPTED_ACTIONS.csv"
VALIDATION_JSON = DELIVERABLE_DIR / "VALIDATION.json"
RECOMPUTED_VALIDATION_JSON = DELIVERABLE_DIR / "VALIDATION_RECOMPUTED.json"
MANIFEST_JSON = DELIVERABLE_DIR / "MANIFEST.json"
VALIDATION_ACTIONS = DELIVERABLE_DIR / "validation_data" / "actions_for_validation.csv"
VALIDATION_SITE_TS = DELIVERABLE_DIR / "validation_data" / "site_timeseries_for_validation.csv"
VALIDATOR_SCRIPT = DELIVERABLE_DIR / "validate_final_solution.py"
ACTION_VALIDATION_COLUMNS = [
    "action_id",
    "job_id",
    "site",
    "start_slot",
    "end_slot",
    "mode",
    "gpu_req",
    "power_mw",
    "energy_mwh",
    "energy_cost_usd",
    "sla_cost_usd",
    "extra_mode_cost_usd",
    "action_cost",
    "is_dummy",
    "gpu_vector",
    "power_vector_mw",
    "conflict_keys",
    "source",
]
FINAL_SOLUTION_COLUMNS = [
    "job_id",
    "action_id",
    "site",
    "start_slot",
    "end_slot",
    "mode",
    "gpu_req",
    "power_mw",
    "energy_mwh",
    "action_cost",
    "is_dummy",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_link_like(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", lambda: False)
    if path.is_symlink() or is_junction():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def validate_path_components(path: Path, *, label: str) -> None:
    root_absolute = ROOT.absolute()
    try:
        parts = path.absolute().relative_to(root_absolute).parts
    except ValueError as exc:
        raise RuntimeError(f"{label} is outside the repository: {path}") from exc
    cursor = ROOT
    for part in parts:
        cursor = cursor / part
        if is_link_like(cursor):
            raise RuntimeError(f"{label} traverses a link, junction, or reparse point: {cursor}")


def validate_source_file(path: Path, *, label: str) -> Path:
    validate_path_components(path, label=label)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{label} escapes the repository: {path}") from exc
    if not path.is_file():
        raise RuntimeError(f"{label} is not a regular file: {path}")
    return resolved


def validate_source_tree(path: Path, *, label: str) -> None:
    validate_path_components(path, label=label)
    if is_link_like(path) or not path.is_dir():
        raise RuntimeError(f"{label} is missing, non-directory, or linked: {path}")
    validate_source_file_marker = ROOT.resolve(strict=True)
    try:
        path.resolve(strict=True).relative_to(validate_source_file_marker)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{label} escapes the repository: {path}") from exc
    for entry in path.rglob("*"):
        if is_link_like(entry):
            raise RuntimeError(f"{label} contains a link or junction: {entry}")
        try:
            entry.resolve(strict=True).relative_to(validate_source_file_marker)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"{label} contains an escaping path: {entry}") from exc
        if not entry.is_file() and not entry.is_dir():
            raise RuntimeError(f"{label} contains a non-regular entry: {entry}")


def safe_remove_tree(path: Path, *, parent: Path, expected_name: str) -> None:
    if is_link_like(path):
        raise RuntimeError(f"Refusing to remove a linked output path: {path}")
    resolved_parent = parent.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    if resolved_path.parent != resolved_parent or resolved_path.name != expected_name:
        raise RuntimeError(f"Refusing to remove an unexpected output path: {resolved_path}")
    shutil.rmtree(resolved_path)


def normalized_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def strict_bool_value(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def action_rows_match(selected: pd.Series, canonical: pd.Series) -> bool:
    for column in ("job_id", "site", "mode", "source"):
        if str(selected[column]) != str(canonical[column]):
            return False
    for column in ("start_slot", "end_slot", "gpu_req"):
        if int(selected[column]) != int(canonical[column]):
            return False
    try:
        if strict_bool_value(selected["is_dummy"]) != strict_bool_value(canonical["is_dummy"]):
            return False
    except ValueError:
        return False
    for column in (
        "power_mw",
        "energy_mwh",
        "energy_cost_usd",
        "sla_cost_usd",
        "extra_mode_cost_usd",
        "action_cost",
    ):
        if not abs(float(selected[column]) - float(canonical[column])) <= 1e-12:
            return False
    for column in ("gpu_vector", "power_vector_mw", "conflict_keys"):
        if normalized_json_value(selected[column]) != normalized_json_value(canonical[column]):
            return False
    return True


def sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        root_text = str(ROOT).replace("\\", "/")
        reports_text = str(REPORTS_DIR).replace("\\", "/")
        if normalized == root_text:
            return "."
        if normalized == reports_text:
            return "reports"
        if normalized.startswith(root_text + "/"):
            return normalized.replace(root_text + "/", "", 1)
        if normalized.startswith(reports_text + "/"):
            return "reports/" + normalized.replace(reports_text + "/", "", 1)
        if re.match(r"^[A-Za-z]:/", normalized):
            marker = "/TAOLIN-DC-SIM/"
            if marker in normalized:
                return normalized.split(marker, 1)[1]
            return Path(normalized).name
        return value
    return value


def write_sanitized_json(source: Path, target: Path) -> None:
    payload = sanitize_value(load_json(source))
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_solution(result: dict[str, Any], solution: pd.DataFrame, actions: pd.DataFrame) -> dict[str, Any]:
    expected_jobs = int(result["selected_job_count"])
    expected_actions = int(result["selected_action_count"])
    risk_metrics = result["risk_accepted_solution"]["metrics"]

    action_ids = set(actions["action_id"].astype(str).tolist())
    solution_action_ids = solution["action_id"].astype(str).tolist()
    duplicate_jobs = solution.loc[solution["job_id"].duplicated(keep=False), "job_id"].astype(str).tolist()
    missing_action_ids = sorted(set(solution_action_ids) - action_ids)
    duplicate_action_ids = sorted(
        actions.loc[actions["action_id"].astype(str).duplicated(keep=False), "action_id"]
        .astype(str)
        .unique()
        .tolist()
    )
    canonical_by_id = actions.assign(action_id=actions["action_id"].astype(str)).set_index("action_id")
    binding_mismatches: list[str] = []
    if not duplicate_action_ids:
        for _, selected in solution.iterrows():
            action_id = str(selected["action_id"])
            if action_id not in canonical_by_id.index:
                continue
            if not action_rows_match(selected, canonical_by_id.loc[action_id]):
                binding_mismatches.append(action_id)
    parsed_dummy_values: list[bool] = []
    invalid_dummy_values: list[str] = []
    for value in solution["is_dummy"].tolist():
        try:
            parsed_dummy_values.append(strict_bool_value(value))
        except ValueError:
            invalid_dummy_values.append(str(value))
    dummy_count = int(sum(parsed_dummy_values))
    one_hot_total = int(risk_metrics["one_hot"]["total"])

    checks = {
        "row_count_matches_jobs": len(solution) == expected_jobs,
        "unique_jobs_matches_jobs": int(solution["job_id"].nunique()) == expected_jobs,
        "selected_action_count_matches_instance": expected_actions == len(actions),
        "no_duplicate_jobs": len(duplicate_jobs) == 0,
        "all_action_ids_exist": len(missing_action_ids) == 0,
        "canonical_action_ids_unique": len(duplicate_action_ids) == 0,
        "selected_rows_match_canonical_actions": len(binding_mismatches) == 0,
        "is_dummy_values_valid": len(invalid_dummy_values) == 0,
        "dummy_count_matches_metrics": dummy_count == int(risk_metrics["dummy_actions"]),
        "one_hot_violations_zero": one_hot_total == 0,
        "source_matches_risk_accepted": set(solution["selection_source"].astype(str)) == {
            str(result["risk_accepted_solution"]["source"])
        },
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "row_count": len(solution),
        "unique_jobs": int(solution["job_id"].nunique()),
        "expected_jobs": expected_jobs,
        "selected_action_count": expected_actions,
        "dummy_actions": dummy_count,
        "one_hot_violations": one_hot_total,
        "duplicate_jobs": duplicate_jobs,
        "missing_action_ids": missing_action_ids,
        "duplicate_action_ids": duplicate_action_ids,
        "canonical_binding_mismatches": sorted(set(binding_mismatches))[:25],
        "invalid_is_dummy_values": sorted(set(invalid_dummy_values))[:25],
    }


def write_validation_data(actions: pd.DataFrame) -> None:
    VALIDATION_ACTIONS.parent.mkdir(parents=True, exist_ok=True)
    actions.loc[:, ACTION_VALIDATION_COLUMNS].to_csv(
        VALIDATION_ACTIONS,
        index=False,
        encoding="utf-8",
    )
    site_ts = pd.read_parquet(ROOT / "data" / "interim" / "site_timeseries.parquet")
    site_columns = [
        "slot",
        "lmp_usd_mwh",
        "carbon_kgco2_mwh",
        "penalty_short_usd_mwh",
        "flex_call_prob",
        "renewable_mw",
    ]
    site_ts.loc[:, site_columns].to_csv(VALIDATION_SITE_TS, index=False, encoding="utf-8")


VALIDATOR_CODE = r'''"""Self-contained validator for the TAOLIN-DC-SIM final solution package.

Run from the extracted final_solution_package directory:

    python validate_final_solution.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FINAL_SOLUTION_FIELDS = [
    "job_id", "action_id", "site", "start_slot", "end_slot", "mode",
    "gpu_req", "power_mw", "energy_mwh", "action_cost", "is_dummy",
]
FULL_SOLUTION_FIELDS = [
    "selection_source", "action_id", "job_id", "site", "start_slot", "end_slot",
    "mode", "gpu_req", "power_mw", "energy_mwh", "energy_cost_usd",
    "sla_cost_usd", "extra_mode_cost_usd", "action_cost", "is_dummy",
    "gpu_vector", "power_vector_mw", "conflict_keys", "source",
]
ACTION_FIELDS = [
    "action_id", "job_id", "site", "start_slot", "end_slot", "mode", "gpu_req",
    "power_mw", "energy_mwh", "energy_cost_usd", "sla_cost_usd",
    "extra_mode_cost_usd", "action_cost", "is_dummy", "gpu_vector",
    "power_vector_mw", "conflict_keys", "source",
]
SITE_FIELDS = [
    "slot", "lmp_usd_mwh", "carbon_kgco2_mwh", "penalty_short_usd_mwh",
    "flex_call_prob", "renewable_mw",
]


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def parse_vector(value: str) -> list[float]:
    return [float(v) for v in json.loads(value)]


def action_rows_match(selected: dict[str, str], canonical: dict[str, str]) -> bool:
    for column in ("job_id", "site", "mode", "source"):
        if selected.get(column) != canonical.get(column):
            return False
    try:
        for column in ("start_slot", "end_slot", "gpu_req"):
            if int(selected.get(column, "")) != int(canonical.get(column, "")):
                return False
        if parse_bool(selected.get("is_dummy", "")) != parse_bool(canonical.get("is_dummy", "")):
            return False
        for column in (
            "power_mw",
            "energy_mwh",
            "energy_cost_usd",
            "sla_cost_usd",
            "extra_mode_cost_usd",
            "action_cost",
        ):
            if not math.isclose(
                float(selected.get(column, "")),
                float(canonical.get(column, "")),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                return False
        for column in ("gpu_vector", "power_vector_mw", "conflict_keys"):
            if json.loads(selected.get(column, "")) != json.loads(canonical.get(column, "")):
                return False
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def compact_rows_match(compact: dict[str, str], full: dict[str, str]) -> bool:
    for column in ("job_id", "action_id", "site", "mode"):
        if compact.get(column) != full.get(column):
            return False
    try:
        for column in ("start_slot", "end_slot", "gpu_req"):
            if int(compact.get(column, "")) != int(full.get(column, "")):
                return False
        if parse_bool(compact.get("is_dummy", "")) != parse_bool(full.get("is_dummy", "")):
            return False
        for column in ("power_mw", "energy_mwh", "action_cost"):
            if not math.isclose(
                float(compact.get(column, "")),
                float(full.get(column, "")),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                return False
    except (TypeError, ValueError):
        return False
    return True


def read_csv(path: Path, expected_fields: list[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields or len(set(reader.fieldnames or [])) != len(expected_fields):
            raise ValueError(f"Unexpected or duplicate CSV header in {path.name}")
        return list(reader)


def add_vectors(rows: list[dict[str, str]], column: str, horizon: int) -> list[float]:
    load = [0.0] * horizon
    for row in rows:
        vector = parse_vector(row[column])
        if len(vector) != horizon:
            raise ValueError(f"Vector length mismatch in {column}: {len(vector)} != {horizon}")
        for idx, value in enumerate(vector):
            load[idx] += value
    return load


def baseline_solution(actions: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in actions:
        groups.setdefault(row["job_id"], []).append(row)
    selected = []
    for job_id in sorted(groups):
        group = groups[job_id]
        real = [row for row in group if not parse_bool(row["is_dummy"])]
        candidates = real if real else group
        selected.append(min(candidates, key=lambda row: float(row["action_cost"])))
    return selected


def recourse_scenarios(horizon: int) -> list[dict[str, object]]:
    evening = [1.0] * horizon
    if horizon >= 18:
        for idx in range(16, min(21, horizon)):
            evening[idx] = 0.55
    solar_short = [1.0] * horizon
    if horizon >= 14:
        for idx in range(9, min(15, horizon)):
            solar_short[idx] = 0.60
    return [
        {"name": "nominal", "probability": 0.50, "renewable_factor": [1.0] * horizon, "price_factor": [1.0] * horizon, "flex_factor": 0.50},
        {"name": "green_short_peak_price", "probability": 0.20, "renewable_factor": [0.55] * horizon, "price_factor": [1.35] * horizon, "flex_factor": 0.85},
        {"name": "high_green_low_price", "probability": 0.15, "renewable_factor": [1.25] * horizon, "price_factor": [0.75] * horizon, "flex_factor": 0.25},
        {"name": "evening_stress", "probability": 0.10, "renewable_factor": evening, "price_factor": [1.60 if v < 1.0 else 1.05 for v in evening], "flex_factor": 0.90},
        {"name": "solar_midday_shortfall", "probability": 0.05, "renewable_factor": solar_short, "price_factor": [1.25 if v < 1.0 else 1.00 for v in solar_short], "flex_factor": 0.70},
    ]


def weighted_cvar(costs: list[float], probabilities: list[float], alpha: float) -> dict[str, float]:
    pairs = sorted(zip(costs, probabilities), key=lambda item: item[0])
    cumulative = 0.0
    var = pairs[-1][0]
    for cost, prob in pairs:
        cumulative += prob
        if cumulative >= alpha:
            var = cost
            break
    tail_prob = max(1.0 - alpha, 1e-12)
    above_mass = sum(prob for cost, prob in pairs if cost > var)
    above_cost = sum(cost * prob for cost, prob in pairs if cost > var)
    var_mass = max(tail_prob - above_mass, 0.0)
    return {"alpha": alpha, "var": float(var), "cvar": float((above_cost + var * var_mass) / tail_prob)}


def evaluate_recourse(load_mw: list[float], reference_load_mw: list[float], site_rows: list[dict[str, str]], alpha: float = 0.95) -> dict[str, object]:
    horizon = len(site_rows)
    if len(load_mw) != horizon or len(reference_load_mw) != horizon:
        raise ValueError("Schedule load and site horizon lengths differ.")
    lmp = [float(row["lmp_usd_mwh"]) for row in site_rows]
    carbon = [float(row["carbon_kgco2_mwh"]) for row in site_rows]
    penalty_short = [float(row["penalty_short_usd_mwh"]) for row in site_rows]
    flex_call_prob = [float(row["flex_call_prob"]) for row in site_rows]
    renewable_raw = [float(row["renewable_mw"]) for row in site_rows]
    renewable_max = max(max(renewable_raw), 1e-9)
    renewable_cf = [value / renewable_max for value in renewable_raw]
    green_capacity_mw = max(max(reference_load_mw) * 0.85, 1e-9)
    base_green_mw = [cf * green_capacity_mw for cf in renewable_cf]
    flex_target_base = [0.30 * prob * max(ref, 0.0) for prob, ref in zip(flex_call_prob, reference_load_mw)]
    flex_delivery_potential = [max(ref - load, 0.0) for ref, load in zip(reference_load_mw, load_mw)]

    scenario_rows = []
    for scenario in recourse_scenarios(horizon):
        renewable_factor = scenario["renewable_factor"]
        price_factor = scenario["price_factor"]
        flex_factor = float(scenario["flex_factor"])
        renewable_avail = [base * factor for base, factor in zip(base_green_mw, renewable_factor)]
        price = [base * factor for base, factor in zip(lmp, price_factor)]
        flex_target = [base * flex_factor for base in flex_target_base]
        renewable_used = [min(load, avail) for load, avail in zip(load_mw, renewable_avail)]
        grid_import = [max(load - avail, 0.0) for load, avail in zip(load_mw, renewable_avail)]
        curtailment = [max(avail - load, 0.0) for load, avail in zip(load_mw, renewable_avail)]
        flex_delivered = [min(target, potential) for target, potential in zip(flex_target, flex_delivery_potential)]
        flex_shortfall = [max(target - delivered, 0.0) for target, delivered in zip(flex_target, flex_delivered)]
        grid_cost = sum(imported * price_value for imported, price_value in zip(grid_import, price))
        curtailment_cost = sum(curtailment) * 5.0
        flex_shortfall_cost = sum(shortfall * penalty for shortfall, penalty in zip(flex_shortfall, penalty_short))
        total_cost = grid_cost + curtailment_cost + flex_shortfall_cost
        scenario_rows.append({
            "name": scenario["name"],
            "probability": float(scenario["probability"]),
            "total_recourse_cost_usd": total_cost,
            "grid_cost_usd": grid_cost,
            "curtailment_cost_usd": curtailment_cost,
            "flex_shortfall_cost_usd": flex_shortfall_cost,
            "grid_import_mwh": sum(grid_import),
            "renewable_used_mwh": sum(renewable_used),
            "curtailment_mwh": sum(curtailment),
            "flex_target_mwh": sum(flex_target),
            "flex_delivered_mwh": sum(flex_delivered),
            "flex_shortfall_mwh": sum(flex_shortfall),
            "carbon_kgco2": sum(imported * carbon_value for imported, carbon_value in zip(grid_import, carbon)),
        })
    costs = [row["total_recourse_cost_usd"] for row in scenario_rows]
    probabilities = [row["probability"] for row in scenario_rows]
    expected_cost = sum(cost * prob for cost, prob in zip(costs, probabilities))
    cvar = weighted_cvar(costs, probabilities, alpha)
    worst = max(scenario_rows, key=lambda row: row["total_recourse_cost_usd"])
    return {
        "expected_recourse_cost_usd": expected_cost,
        "var_recourse_cost_usd": cvar["var"],
        "cvar_recourse_cost_usd": cvar["cvar"],
        "worst_scenario": worst["name"],
        "worst_recourse_cost_usd": worst["total_recourse_cost_usd"],
        "scenarios": scenario_rows,
    }


def main() -> int:
    final_solution = read_csv(ROOT / "FINAL_SOLUTION.csv", FINAL_SOLUTION_FIELDS)
    full_solution = read_csv(ROOT / "FULL_SELECTED_ACTIONS.csv", FULL_SOLUTION_FIELDS)
    actions = read_csv(ROOT / "validation_data" / "actions_for_validation.csv", ACTION_FIELDS)
    site_rows = read_csv(ROOT / "validation_data" / "site_timeseries_for_validation.csv", SITE_FIELDS)
    metrics = json.loads((ROOT / "FINAL_METRICS.json").read_text(encoding="utf-8"))
    horizon = len(site_rows)
    baseline_rows = baseline_solution(actions)
    reference_load = add_vectors(baseline_rows, "power_vector_mw", horizon)
    solution_load = add_vectors(full_solution, "power_vector_mw", horizon)
    recourse = evaluate_recourse(solution_load, reference_load, site_rows)

    final_schema_ok = bool(final_solution) and all(
        list(row) == FINAL_SOLUTION_FIELDS for row in final_solution
    )
    full_schema_ok = bool(full_solution) and all(
        list(row) == FULL_SOLUTION_FIELDS for row in full_solution
    )
    final_pairs = [(row.get("job_id"), row.get("action_id")) for row in final_solution]
    full_pairs = [(row.get("job_id"), row.get("action_id")) for row in full_solution]
    full_by_pair = {pair: row for pair, row in zip(full_pairs, full_solution)}
    compact_binding_mismatches = sorted(
        str(row.get("action_id"))
        for pair, row in zip(final_pairs, final_solution)
        if pair not in full_by_pair or not compact_rows_match(row, full_by_pair[pair])
    )
    action_ids = [row["action_id"] for row in actions]
    canonical_by_id = {row["action_id"]: row for row in actions}
    binding_mismatches = sorted(
        row["action_id"]
        for row in full_solution
        if row["action_id"] not in canonical_by_id
        or not action_rows_match(row, canonical_by_id[row["action_id"]])
    )
    invalid_boolean_rows = []
    final_dummy_values = []
    full_dummy_values = []
    for label, rows, parsed in (
        ("FINAL_SOLUTION", final_solution, final_dummy_values),
        ("FULL_SELECTED_ACTIONS", full_solution, full_dummy_values),
    ):
        for index, row in enumerate(rows, start=2):
            try:
                parsed.append(parse_bool(row.get("is_dummy", "")))
            except ValueError:
                invalid_boolean_rows.append(f"{label}:{index}")
    checks = {
        "final_solution_exists": (ROOT / "FINAL_SOLUTION.csv").exists(),
        "final_solution_schema_exact": final_schema_ok,
        "full_selected_actions_schema_exact": full_schema_ok,
        "final_solution_matches_full_selected_actions": (
            len(final_pairs) == len(set(final_pairs))
            and len(full_pairs) == len(set(full_pairs))
            and set(final_pairs) == set(full_pairs)
            and not compact_binding_mismatches
        ),
        "row_count_588": len(final_solution) == 588,
        "unique_jobs_588": len({row["job_id"] for row in final_solution}) == 588,
        "canonical_action_ids_unique": len(action_ids) == len(set(action_ids)),
        "selected_rows_match_canonical_actions": not binding_mismatches,
        "selection_source_matches_metrics": {
            row.get("selection_source") for row in full_solution
        } == {str(metrics["source"])},
        "boolean_values_valid": not invalid_boolean_rows,
        "dummy_actions_match_metrics": (
            sum(final_dummy_values) == int(metrics["dummy_actions"])
            and sum(full_dummy_values) == int(metrics["dummy_actions"])
        ),
        "expected_recourse_matches_metrics": math.isclose(recourse["expected_recourse_cost_usd"], float(metrics["expected_recourse_cost_usd"]), abs_tol=5e-7),
        "cvar95_matches_metrics": math.isclose(recourse["cvar_recourse_cost_usd"], float(metrics["cvar95_recourse_cost_usd"]), abs_tol=5e-7),
        "metrics_json_expected_matches": math.isclose(float(metrics["expected_recourse_cost_usd"]), recourse["expected_recourse_cost_usd"], abs_tol=1e-9),
        "metrics_json_cvar_matches": math.isclose(float(metrics["cvar95_recourse_cost_usd"]), recourse["cvar_recourse_cost_usd"], abs_tol=1e-9),
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "row_count": len(final_solution),
        "unique_jobs": len({row["job_id"] for row in final_solution}),
        "dummy_actions": sum(final_dummy_values),
        "invalid_boolean_rows": invalid_boolean_rows[:25],
        "canonical_binding_mismatches": binding_mismatches[:25],
        "compact_binding_mismatches": compact_binding_mismatches[:25],
        "expected_recourse_cost_usd": recourse["expected_recourse_cost_usd"],
        "cvar95_recourse_cost_usd": recourse["cvar_recourse_cost_usd"],
        "scenario_costs": recourse["scenarios"],
    }
    (ROOT / "VALIDATION_RECOMPUTED.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def write_validator_script() -> None:
    VALIDATOR_SCRIPT.write_text(VALIDATOR_CODE, encoding="utf-8")


def write_final_solution(solution: pd.DataFrame) -> None:
    solution.loc[:, FINAL_SOLUTION_COLUMNS].to_csv(
        FINAL_SOLUTION,
        index=False,
        encoding="utf-8-sig",
    )
    solution.to_csv(FULL_SELECTED_ACTIONS, index=False, encoding="utf-8-sig")


def metric_block(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["risk_accepted_solution"]["metrics"]
    recourse = metrics["recourse"]
    return {
        "source": result["risk_accepted_solution"]["source"],
        "selected_jobs": result["selected_job_count"],
        "selected_action_variables": result["selected_action_count"],
        "true_action_cost_usd": metrics["true_action_cost_usd"],
        "qubo_energy": metrics["qubo_energy"],
        "expected_recourse_cost_usd": recourse["expected_recourse_cost_usd"],
        "var95_recourse_cost_usd": recourse["var_recourse_cost_usd"],
        "cvar95_recourse_cost_usd": recourse["cvar_recourse_cost_usd"],
        "worst_scenario": recourse["worst_scenario"],
        "worst_recourse_cost_usd": recourse["worst_recourse_cost_usd"],
        "peak_it_mw": metrics["peak_it_mw"],
        "mean_it_mw": metrics["mean_it_mw"],
        "energy_mwh": metrics["energy_mwh"],
        "dummy_actions": metrics["dummy_actions"],
        "one_hot_violations": metrics["one_hot"]["total"],
        "mode_counts": metrics["mode_counts"],
    }


def write_final_report(result: dict[str, Any], validation: dict[str, Any]) -> None:
    metrics = metric_block(result)
    cloud_task_id = result.get("promotion", {}).get("cloud_task_id")
    cloud_replay_id = str(cloud_task_id) if cloud_task_id else "<recorded-task-id>"
    cloud_task_text = (
        f"`{cloud_task_id}`" if cloud_task_id else "the recorded CloudSolver task"
    )
    if metrics["dummy_actions"]:
        dummy_caveat = f"- {metrics['dummy_actions']} dummy actions remain; further gains require upstream real-action generation for those jobs."
    else:
        dummy_caveat = "- The former dummy-only horizon-edge jobs are covered by `horizon_rescue` candidate actions; replacing that engineering proxy with real cross-site/migration actions remains future work."
    industrial_lines: list[str] = []
    industrial_summary_path = INDUSTRIAL_ARTIFACTS_DIR / "industrial_ising_evidence_summary.json"
    if industrial_summary_path.exists():
        summary = load_json(industrial_summary_path)
        family_parts = [
            f"{name} ({payload['spins']} spins)"
            for name, payload in summary.get("families", {}).items()
        ]
        cloud_environment = summary.get("cloud_environment", {})
        cloud_status = (
            "CloudSolver rows were not executed in this rebuild because `PLECTRUM_API_KEY` was absent."
            if not cloud_environment.get("plectrum_api_key_present") and summary.get("cloud_rows", 0) == 0
            else "CloudSolver rows are included in the same-QUBO oracle replacement table."
        )
        industrial_lines = [
            "## Industrial Ising Evidence",
            "",
            "- Positioning: Ising-assisted bounded large-neighborhood optimization for binary action selection; exact recourse replay remains the business objective.",
            f"- LNS portfolio: {', '.join(family_parts)}.",
            f"- Same-QUBO oracle replacement rows: {summary.get('oracle_rows', 0)} total, {summary.get('cloud_rows', 0)} CloudSolver rows, {summary.get('failed_rows', 0)} failed rows.",
            f"- Cloud status: {cloud_status}",
            "- Required evidence tables are packaged under `artifacts/solver_runs/`, `artifacts/tables/`, and `artifacts/validation/`.",
            "- QUBO energy is retained only as solver/proxy evidence; final performance is recomputed from exact recourse and CVaR replay.",
            "",
        ]
    lines = [
        "# TAOLIN-DC-SIM Final Solution Package",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "## Final Solution",
        "",
        "- Selected solution: `FINAL_SOLUTION.csv`",
        "- Full selected action table: `FULL_SELECTED_ACTIONS.csv`",
        f"- Risk-accepted source: `{metrics['source']}`",
        f"- The final 588-job schedule descends from the verified 700-spin CloudSolver neighborhood task {cloud_task_text}; the frozen evidence retains its task ID, returned length, and energy summary, but not the raw spin vector.",
        "- The retained post-polish schedule is independently replayable, but the original cloud assignment cannot be reproduced bit by bit or causally separated from the subsequent contextual polish.",
        "- Promotion was gated on zero one-hot violations, zero dummy actions, the action-cost guard, and strict improvement over the prior final in expected recourse, CVaR95, and peak IT load.",
        "",
        "## Metrics",
        "",
        f"- Jobs selected: {metrics['selected_jobs']}",
        f"- QUBO variables / spins: {metrics['selected_action_variables']}",
        f"- One-hot violations: {metrics['one_hot_violations']}",
        f"- Dummy actions: {metrics['dummy_actions']}",
        f"- True action cost: {metrics['true_action_cost_usd']:.6f}",
        f"- QUBO energy (proxy only, not final business objective): {metrics['qubo_energy']:.6f}",
        f"- Expected recourse cost: {metrics['expected_recourse_cost_usd']:.6f}",
        f"- CVaR95 recourse cost: {metrics['cvar95_recourse_cost_usd']:.6f}",
        f"- Peak IT MW: {metrics['peak_it_mw']:.6f}",
        f"- Mode counts: `{metrics['mode_counts']}`",
        "",
        "## Validation",
        "",
        f"- Status: `{validation['status']}`",
        f"- Rows: {validation['row_count']}",
        f"- Unique jobs: {validation['unique_jobs']}",
        f"- Expected jobs: {validation['expected_jobs']}",
        f"- Missing action ids: {len(validation['missing_action_ids'])}",
        f"- Duplicate jobs: {len(validation['duplicate_jobs'])}",
        "",
        *industrial_lines,
        "## Reproduce",
        "",
        "```powershell",
        "python src/07_solve_isling/build_industrial_ising_evidence.py --target-spins 700 --use-cloud `",
        f"  --cloud-families peak_shave_lns --reuse-cloud-task-id {cloud_replay_id}",
        "python src/07_solve_isling/promote_cloud_candidate.py",
        "python src/07_solve_isling/build_final_deliverable.py",
        "```",
        "",
        "Inside the extracted package, recompute validation and recourse metrics with:",
        "",
        "```powershell",
        "python validate_final_solution.py",
        "```",
        "",
        "## Caveats",
        "",
        "- CAISO price/demand data remains synthetic in the current cleaned dataset.",
        "- Recourse uses fixed stress scenarios and normalized renewable availability.",
        "- Storage, unit commitment, DC power flow, and integer redispatch are not included in this MVP recourse evaluator.",
        dummy_caveat,
        "- The Ising solver is claimed only for bounded binary neighborhoods, not for global solution of the full compute-power recourse problem.",
    ]
    FINAL_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_evidence_files() -> list[Path]:
    evidence_dir = DELIVERABLE_DIR / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in [BEST_RESULT_JSON]:
        validate_source_file(source, label="sanitized evidence source")
        target = evidence_dir / source.name
        write_sanitized_json(source, target)
        copied.append(target)
    for source in [
        BEST_RISK_CSV,
        BEST_ACCEPTED_CSV,
        PROMOTION_REPORT,
        LOCAL_SUMMARY,
        RECOURSE_SUMMARY,
        CLOUD_SUMMARY,
        SOLVER_SCRIPT,
        PROMOTION_SCRIPT,
        BUILDER_SCRIPT,
        VERIFIER_SCRIPT,
        SOLVER_README,
    ]:
        validate_source_file(source, label="evidence source")
        target = evidence_dir / source.name
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def copy_industrial_artifacts() -> None:
    if not INDUSTRIAL_ARTIFACTS_DIR.exists():
        return
    validate_source_tree(INDUSTRIAL_ARTIFACTS_DIR, label="industrial artifact tree")
    target = DELIVERABLE_DIR / "artifacts"
    if target.exists():
        safe_remove_tree(target, parent=DELIVERABLE_DIR, expected_name="artifacts")
    shutil.copytree(INDUSTRIAL_ARTIFACTS_DIR, target)


def write_manifest() -> None:
    records = []
    for path in sorted(DELIVERABLE_DIR.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            records.append(
                {
                    "path": path.relative_to(DELIVERABLE_DIR).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = {
        "generated_at": utc_now(),
        "package_root": ".",
        "file_count": len(records),
        "files": records,
    }
    MANIFEST_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_embedded_validator() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(VALIDATOR_SCRIPT.name)],
        cwd=DELIVERABLE_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(RECOMPUTED_VALIDATION_JSON.read_text(encoding="utf-8"))
    if payload["status"] != "pass":
        raise RuntimeError(f"Embedded validation failed: {completed.stdout}")
    return payload


def build_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(DELIVERABLE_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=f"final_solution_package/{path.relative_to(DELIVERABLE_DIR).as_posix()}")
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"Corrupt zip member: {bad}")


def run_post_archive_verifier() -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            str(VERIFIER_SCRIPT),
            "--root",
            str(ROOT),
            "--package-dir",
            str(DELIVERABLE_DIR),
            "--zip-path",
            str(ZIP_PATH),
        ],
        cwd=ROOT,
        check=True,
    )
    report_path = ROOT / "deliverables" / "SUBMISSION_VERIFICATION.json"
    payload = load_json(report_path)
    if payload.get("overall_status") != "ready":
        raise RuntimeError(
            f"Post-archive verification is not ready: {payload.get('overall_status')}"
        )
    return payload


def main() -> int:
    source_files = [
        BEST_RESULT_JSON,
        BEST_RISK_CSV,
        BEST_ACCEPTED_CSV,
        PROMOTION_REPORT,
        LOCAL_SUMMARY,
        RECOURSE_SUMMARY,
        CLOUD_SUMMARY,
        SOLVER_SCRIPT,
        PROMOTION_SCRIPT,
        BUILDER_SCRIPT,
        VERIFIER_SCRIPT,
        SOLVER_README,
        ROOT / "data" / "processed" / "actions.parquet",
        ROOT / "data" / "interim" / "site_timeseries.parquet",
    ]
    for source in source_files:
        validate_source_file(source, label="final-package input")
    validate_path_components(INDUSTRIAL_ARTIFACTS_DIR, label="industrial artifact tree")
    if INDUSTRIAL_ARTIFACTS_DIR.exists():
        validate_source_tree(INDUSTRIAL_ARTIFACTS_DIR, label="industrial artifact tree")

    result = load_json(BEST_RESULT_JSON)
    actions = pd.read_parquet(ROOT / "data" / "processed" / "actions.parquet")
    risk_solution = pd.read_csv(BEST_RISK_CSV)
    accepted_solution = pd.read_csv(BEST_ACCEPTED_CSV)

    validate_path_components(DELIVERABLE_DIR, label="final deliverable output")
    validate_path_components(ZIP_PATH, label="final ZIP output")
    if DELIVERABLE_DIR.exists():
        safe_remove_tree(
            DELIVERABLE_DIR,
            parent=ROOT / "deliverables",
            expected_name="final_solution_package",
        )
    DELIVERABLE_DIR.mkdir(parents=True, exist_ok=True)

    validation = validate_solution(result, risk_solution, actions)
    if validation["status"] != "pass":
        raise RuntimeError(f"Final solution validation failed: {validation}")

    write_final_solution(risk_solution)
    accepted_solution.to_csv(QUBO_ACCEPTED_ACTIONS, index=False, encoding="utf-8-sig")
    FINAL_METRICS.write_text(json.dumps(metric_block(result), ensure_ascii=False, indent=2), encoding="utf-8")
    VALIDATION_JSON.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    write_final_report(result, validation)
    write_validation_data(actions)
    write_validator_script()
    copy_evidence_files()
    copy_industrial_artifacts()
    recomputed = run_embedded_validator()
    write_manifest()
    build_zip()
    post_archive = run_post_archive_verifier()

    print(f"Wrote {DELIVERABLE_DIR}")
    print(f"Wrote {ZIP_PATH}")
    print(f"Validation: {validation['status']}")
    print(f"Embedded validation: {recomputed['status']}")
    print(f"Post-archive verification: {post_archive['overall_status']}")
    print(f"Risk CVaR95: {metric_block(result)['cvar95_recourse_cost_usd']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
