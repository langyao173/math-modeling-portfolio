"""Self-contained validator for the TAOLIN-DC-SIM final solution package.

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
