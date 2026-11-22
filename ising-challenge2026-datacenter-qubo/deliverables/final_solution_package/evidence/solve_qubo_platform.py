"""Build and solve a compact task-action QUBO for the TAOLIN-DC-SIM MVP.

The QUBO keeps the discrete scheduling core on the Ising/QUBO side:
one binary variable per candidate action, one-hot penalties per job, and
time-risk/peak-load terms over the selected action power vectors. Continuous
power recourse is evaluated after decoding through scenario costs and CVaR.
Optional recourse-linear terms can also embed first-order expected/CVaR
dispatch prices directly into the QUBO coefficients.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"


GEAR_BY_NAME = {
    "fast": "GEAR_FAST",
    "balanced": "GEAR_BALANCED",
    "precise": "GEAR_PRECISE",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_vector(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.tolist()]
    return [float(v) for v in json.loads(value)]


def load_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jobs = pd.read_parquet(root / "data" / "interim" / "jobs_manifest.parquet")
    actions = pd.read_parquet(root / "data" / "processed" / "actions.parquet")
    site_ts = pd.read_parquet(root / "data" / "interim" / "site_timeseries.parquet")
    return jobs, actions, site_ts


def slot_risk_weights(site_ts: pd.DataFrame) -> np.ndarray:
    """Return dimensionless risk weights, one per time slot."""
    ordered = site_ts.sort_values("slot")

    def normalize(series: pd.Series) -> np.ndarray:
        values = series.astype(float).to_numpy()
        lo = float(np.min(values))
        hi = float(np.max(values))
        if math.isclose(lo, hi):
            return np.zeros_like(values)
        return (values - lo) / (hi - lo)

    price = normalize(ordered["lmp_usd_mwh"])
    carbon = normalize(ordered["carbon_kgco2_mwh"])
    flex = normalize(ordered["flex_call_prob"])
    renewable_short = normalize(ordered["demand_forecast_mw"] - ordered["renewable_mw"])
    return 0.40 * price + 0.25 * carbon + 0.20 * flex + 0.15 * renewable_short


def choose_jobs(
    actions: pd.DataFrame,
    site_ts: pd.DataFrame,
    max_jobs: int,
    seed: int,
) -> list[str]:
    """Choose a deterministic high-leverage neighborhood, or all jobs."""
    all_jobs = sorted(actions["job_id"].unique().tolist())
    if max_jobs <= 0 or max_jobs >= len(all_jobs):
        return all_jobs

    risks = slot_risk_weights(site_ts)
    scored: list[tuple[float, str]] = []
    for job_id, group in actions.groupby("job_id", sort=True):
        if bool(group["is_dummy"].all()):
            score = -1.0
        else:
            best = 0.0
            for _, row in group.loc[~group["is_dummy"]].iterrows():
                power = np.array(parse_vector(row["power_vector_mw"]), dtype=float)
                best = max(best, float(power @ (1.0 + risks)))
            mode_bonus = float(group["mode"].nunique()) * 1e-6
            score = best + mode_bonus
        scored.append((score, job_id))

    rng = np.random.default_rng(seed)
    # Stable tie jitter avoids always selecting lexicographic neighbors when scores match.
    scored = [(score + float(rng.uniform(0.0, 1e-9)), job_id) for score, job_id in scored]
    scored.sort(reverse=True)
    selected = sorted(job_id for _, job_id in scored[:max_jobs])
    return selected


def cost_scale(actions: pd.DataFrame) -> float:
    non_dummy = actions.loc[~actions["is_dummy"], "action_cost"].astype(float)
    positive = non_dummy.loc[non_dummy > 0]
    if positive.empty:
        return 1.0
    return max(float(np.percentile(positive, 90)), 1e-6)


def build_qubo(
    actions: pd.DataFrame,
    site_ts: pd.DataFrame,
    one_hot_penalty: float,
    peak_weight: float,
    risk_weight: float,
    recourse_weight: float,
    recourse_cvar_weight: float,
    dummy_cost: float,
    dummy_only_cost: float,
    cost_cap: float,
    reference_load_mw: np.ndarray | None = None,
    recourse_alpha: float = 0.95,
    recourse_finite_delta_mw: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(actions)
    q = np.zeros((n, n), dtype=np.float64)
    risks = slot_risk_weights(site_ts)
    power_vectors = np.array([parse_vector(v) for v in actions["power_vector_mw"]], dtype=np.float64)
    nonzero_power = power_vectors[power_vectors > 0]
    power_scale = float(np.median(nonzero_power)) if nonzero_power.size else 1.0
    power_scale = max(power_scale, 1e-9)

    scale = cost_scale(actions)
    has_real_action = actions.groupby("job_id")["is_dummy"].transform(lambda values: bool((~values).any()))
    recourse_meta: dict[str, Any] = {
        "enabled": bool(abs(recourse_weight) > 0.0 or abs(recourse_cvar_weight) > 0.0),
        "weight": recourse_weight,
        "cvar_weight": recourse_cvar_weight,
        "alpha": recourse_alpha,
        "finite_delta_mw": recourse_finite_delta_mw,
    }
    expected_recourse_terms = np.zeros(n, dtype=np.float64)
    cvar_recourse_terms = np.zeros(n, dtype=np.float64)
    if recourse_meta["enabled"]:
        if reference_load_mw is None:
            raise ValueError("reference_load_mw is required when recourse QUBO terms are enabled.")
        marginal = recourse_marginal_slot_prices(
            reference_load_mw=reference_load_mw,
            site_ts=site_ts,
            alpha=recourse_alpha,
            finite_delta_mw=recourse_finite_delta_mw,
        )
        expected_action_cost = power_vectors @ marginal["expected_slot_price_usd_mwh"]
        cvar_action_cost = power_vectors @ marginal["cvar_slot_price_usd_mwh"]
        scale_candidates = np.concatenate(
            [
                np.abs(expected_action_cost[~actions["is_dummy"].to_numpy()]),
                np.abs(cvar_action_cost[~actions["is_dummy"].to_numpy()]),
            ]
        )
        positive_scale_candidates = scale_candidates[scale_candidates > 1e-12]
        recourse_scale = (
            float(np.median(positive_scale_candidates)) if positive_scale_candidates.size else 1.0
        )
        recourse_scale = max(recourse_scale, 1e-12)
        expected_recourse_terms = expected_action_cost / recourse_scale
        cvar_recourse_terms = cvar_action_cost / recourse_scale
        recourse_meta.update(
            {
                "action_term_scale_usd": recourse_scale,
                "expected_slot_price_usd_mwh": [
                    float(v) for v in marginal["expected_slot_price_usd_mwh"].tolist()
                ],
                "cvar_slot_price_usd_mwh": [
                    float(v) for v in marginal["cvar_slot_price_usd_mwh"].tolist()
                ],
                "reference_scenario_costs": marginal["reference_scenario_costs"],
                "cvar_scenario_weights": marginal["cvar_scenario_weights"],
                "expected_action_term_min": float(np.min(expected_recourse_terms)),
                "expected_action_term_max": float(np.max(expected_recourse_terms)),
                "cvar_action_term_min": float(np.min(cvar_recourse_terms)),
                "cvar_action_term_max": float(np.max(cvar_recourse_terms)),
            }
        )

    linear_terms: list[float] = []
    for idx, row in actions.reset_index(drop=True).iterrows():
        if bool(row["is_dummy"]):
            h = dummy_cost if bool(has_real_action.iloc[idx]) else dummy_only_cost
        else:
            scaled_cost = min(float(row["action_cost"]) / scale, cost_cap)
            risk_term = float(power_vectors[idx] @ risks) / power_scale
            recourse_term = (
                recourse_weight * expected_recourse_terms[idx]
                + recourse_cvar_weight * cvar_recourse_terms[idx]
            )
            h = scaled_cost + risk_weight * risk_term + recourse_term
        linear_terms.append(float(h))
        q[idx, idx] += float(h)

    for _, idxs in actions.groupby("job_id", sort=False).indices.items():
        group = list(idxs)
        for i in group:
            q[i, i] -= one_hot_penalty
        for left, i in enumerate(group):
            for j in group[left + 1 :]:
                q[i, j] += one_hot_penalty
                q[j, i] += one_hot_penalty

    if peak_weight > 0:
        scaled_power = power_vectors / power_scale
        q += peak_weight * (scaled_power @ scaled_power.T)

    meta = {
        "n_variables": n,
        "n_jobs": int(actions["job_id"].nunique()),
        "one_hot_penalty": one_hot_penalty,
        "peak_weight": peak_weight,
        "risk_weight": risk_weight,
        "recourse": recourse_meta,
        "dummy_cost": dummy_cost,
        "dummy_only_cost": dummy_only_cost,
        "cost_cap": cost_cap,
        "action_cost_scale_p90": scale,
        "power_scale_mw_median_nonzero": power_scale,
        "q_min": float(np.min(q)),
        "q_max": float(np.max(q)),
        "q_abs_max": float(np.max(np.abs(q))),
        "q_nonzero": int(np.count_nonzero(np.abs(q) > 1e-12)),
        "slot_risk_weights": [float(v) for v in risks.tolist()],
        "linear_term_min": float(np.min(linear_terms)),
        "linear_term_max": float(np.max(linear_terms)),
    }
    return q, meta


def recourse_terms_enabled(args: argparse.Namespace) -> bool:
    return bool(abs(args.recourse_weight) > 0.0 or abs(args.recourse_cvar_weight) > 0.0)


def build_qubo_from_args(
    actions: pd.DataFrame,
    site_ts: pd.DataFrame,
    args: argparse.Namespace,
    reference_load_mw: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    return build_qubo(
        actions,
        site_ts,
        one_hot_penalty=args.one_hot_penalty,
        peak_weight=args.peak_weight,
        risk_weight=args.risk_weight,
        recourse_weight=args.recourse_weight,
        recourse_cvar_weight=args.recourse_cvar_weight,
        dummy_cost=args.dummy_cost,
        dummy_only_cost=args.dummy_only_cost,
        cost_cap=args.cost_cap,
        reference_load_mw=reference_load_mw,
        recourse_alpha=args.recourse_alpha,
        recourse_finite_delta_mw=args.recourse_finite_delta_mw,
    )


def run_recourse_cut_iterations(
    actions: pd.DataFrame,
    site_ts: pd.DataFrame,
    args: argparse.Namespace,
    baseline_x: np.ndarray,
    baseline_load_mw: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    """Update the recourse marginalization point using deterministic candidates."""
    reference_load = baseline_load_mw.copy().astype(np.float64)
    start_x = baseline_x.copy().astype(int)
    records: list[dict[str, Any]] = []
    previous_cvar: float | None = None

    for iteration in range(args.recourse_cut_iterations):
        q_iter, meta_iter = build_qubo_from_args(actions, site_ts, args, reference_load)
        x_iter, polish_report = greedy_one_hot_search(
            actions,
            q_iter,
            start_x,
            max_passes=args.post_polish_passes,
        )
        metrics_iter = schedule_metrics(
            actions,
            site_ts,
            q_iter,
            x_iter,
            reference_load_mw=baseline_load_mw,
        )
        _, candidate_load, _ = selected_vectors(actions, x_iter, len(site_ts))
        load_delta = candidate_load - reference_load
        cvar_value = float(metrics_iter["recourse"]["cvar_recourse_cost_usd"])
        record = {
            "iteration": iteration + 1,
            "reference_load_linf_delta_mw": float(np.max(np.abs(load_delta))) if load_delta.size else 0.0,
            "reference_load_l1_delta_mwh": float(np.sum(np.abs(load_delta))),
            "greedy_total_improvements": int(polish_report["total_improvements"]),
            "qubo_energy": float(metrics_iter["qubo_energy"]),
            "true_action_cost_usd": float(metrics_iter["true_action_cost_usd"]),
            "expected_recourse_cost_usd": float(metrics_iter["recourse"]["expected_recourse_cost_usd"]),
            "cvar_recourse_cost_usd": cvar_value,
            "peak_it_mw": float(metrics_iter["peak_it_mw"]),
            "dummy_actions": int(metrics_iter["dummy_actions"]),
            "one_hot_violations": int(metrics_iter["one_hot"]["total"]),
            "recourse_action_scale_usd": meta_iter["recourse"].get("action_term_scale_usd"),
        }
        records.append(record)

        damping = min(max(float(args.recourse_cut_damping), 0.0), 1.0)
        reference_load = (1.0 - damping) * reference_load + damping * candidate_load
        start_x = x_iter.astype(int)

        cvar_change = None if previous_cvar is None else abs(previous_cvar - cvar_value)
        previous_cvar = cvar_value
        if (
            cvar_change is not None
            and cvar_change <= args.recourse_cut_stop_cvar_tol
            and record["reference_load_linf_delta_mw"] <= args.recourse_cut_stop_load_tol_mw
        ):
            record["stop_reason"] = "cvar_and_load_tolerance"
            break

    return reference_load, records, start_x


def recourse_polish_objective(
    recourse: dict[str, Any],
    true_action_cost_usd: float,
    peak_it_mw: float,
    expected_weight: float,
    cvar_weight: float,
    peak_weight: float,
    action_cost_weight: float,
) -> float:
    return float(
        expected_weight * recourse["expected_recourse_cost_usd"]
        + cvar_weight * recourse["cvar_recourse_cost_usd"]
        + peak_weight * peak_it_mw
        + action_cost_weight * true_action_cost_usd
    )


def recourse_polish_search(
    actions: pd.DataFrame,
    site_ts: pd.DataFrame,
    start_x: np.ndarray,
    reference_load_mw: np.ndarray,
    max_passes: int,
    max_action_cost_usd: float,
    max_dummy_actions: int,
    expected_weight: float,
    cvar_weight: float,
    peak_weight: float,
    action_cost_weight: float,
    min_gain: float,
    job_order: str = "forward",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Coordinate descent using exact recourse/CVaR evaluation as the objective."""
    x = start_x.copy().astype(int)
    horizon = len(site_ts)
    power_vectors = np.array([parse_vector(v) for v in actions["power_vector_mw"]], dtype=np.float64)
    action_costs = actions["action_cost"].astype(float).to_numpy()
    is_dummy = actions["is_dummy"].astype(bool).to_numpy()
    groups = [
        [actions.index.get_loc(idx) for idx in group.index]
        for _, group in actions.groupby("job_id", sort=False)
    ]
    if job_order == "reverse":
        groups = list(reversed(groups))
    elif job_order in {"high_load_first", "low_load_first"}:
        groups = sorted(
            groups,
            key=lambda positions: max(float(power_vectors[pos].sum()) for pos in positions),
            reverse=job_order == "high_load_first",
        )
    elif job_order != "forward":
        raise ValueError(f"Unsupported recourse polish job order: {job_order}")

    selected_positions = np.flatnonzero(x == 1)
    current_load = power_vectors[selected_positions].sum(axis=0) if len(selected_positions) else np.zeros(horizon)
    current_cost = float(action_costs[selected_positions].sum()) if len(selected_positions) else 0.0
    current_dummy = int(is_dummy[selected_positions].sum()) if len(selected_positions) else 0
    current_recourse = evaluate_recourse(current_load, reference_load_mw, site_ts)
    current_peak = float(current_load.max()) if current_load.size else 0.0
    current_objective = recourse_polish_objective(
        current_recourse,
        current_cost,
        current_peak,
        expected_weight=expected_weight,
        cvar_weight=cvar_weight,
        peak_weight=peak_weight,
        action_cost_weight=action_cost_weight,
    )

    pass_reports: list[dict[str, Any]] = []
    total_improvements = 0
    for pass_idx in range(max_passes):
        improvements = 0
        objective_gain = 0.0
        for positions in groups:
            selected = [pos for pos in positions if x[pos] == 1]
            if len(selected) != 1:
                continue
            current_pos = selected[0]
            best_pos = current_pos
            best_load = current_load
            best_cost = current_cost
            best_dummy = current_dummy
            best_recourse = current_recourse
            best_peak = current_peak
            best_objective = current_objective
            for candidate in positions:
                if candidate == current_pos:
                    continue
                candidate_cost = current_cost - float(action_costs[current_pos]) + float(action_costs[candidate])
                candidate_dummy = current_dummy - int(is_dummy[current_pos]) + int(is_dummy[candidate])
                if candidate_cost > max_action_cost_usd + 1e-9 or candidate_dummy > max_dummy_actions:
                    continue
                candidate_load = current_load - power_vectors[current_pos] + power_vectors[candidate]
                candidate_recourse = evaluate_recourse(candidate_load, reference_load_mw, site_ts)
                candidate_peak = float(candidate_load.max()) if candidate_load.size else 0.0
                candidate_objective = recourse_polish_objective(
                    candidate_recourse,
                    candidate_cost,
                    candidate_peak,
                    expected_weight=expected_weight,
                    cvar_weight=cvar_weight,
                    peak_weight=peak_weight,
                    action_cost_weight=action_cost_weight,
                )
                if candidate_objective < best_objective - min_gain:
                    best_pos = candidate
                    best_load = candidate_load
                    best_cost = candidate_cost
                    best_dummy = candidate_dummy
                    best_recourse = candidate_recourse
                    best_peak = candidate_peak
                    best_objective = candidate_objective
            if best_pos != current_pos:
                x[current_pos] = 0
                x[best_pos] = 1
                objective_gain += current_objective - best_objective
                current_load = best_load
                current_cost = best_cost
                current_dummy = best_dummy
                current_recourse = best_recourse
                current_peak = best_peak
                current_objective = best_objective
                improvements += 1
                total_improvements += 1
        pass_reports.append(
            {
                "pass": pass_idx + 1,
                "improvements": improvements,
                "objective_gain": float(objective_gain),
                "objective": float(current_objective),
                "true_action_cost_usd": float(current_cost),
                "expected_recourse_cost_usd": float(current_recourse["expected_recourse_cost_usd"]),
                "cvar_recourse_cost_usd": float(current_recourse["cvar_recourse_cost_usd"]),
                "peak_it_mw": float(current_peak),
                "dummy_actions": int(current_dummy),
            }
        )
        if improvements == 0:
            break

    return x, {
        "passes": pass_reports,
        "total_improvements": total_improvements,
        "objective_weights": {
            "expected": expected_weight,
            "cvar": cvar_weight,
            "peak": peak_weight,
            "action_cost": action_cost_weight,
        },
        "max_action_cost_usd": max_action_cost_usd,
        "max_dummy_actions": max_dummy_actions,
        "min_gain": min_gain,
        "job_order": job_order,
    }


def parse_csv_tokens(value: str) -> list[str]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    return tokens or []


def load_external_start_vector(actions: pd.DataFrame, path: Path) -> np.ndarray:
    """Load a one-action-per-job start assignment from a CSV file."""
    frame = pd.read_csv(path)
    required = {"job_id", "action_id"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"External start {path} is missing columns: {', '.join(missing_columns)}"
        )
    frame = frame.loc[:, ["job_id", "action_id"]].copy()
    frame["job_id"] = frame["job_id"].astype(str)
    frame["action_id"] = frame["action_id"].astype(str)
    if frame["job_id"].duplicated().any():
        raise ValueError(f"External start {path} contains duplicate job_id values.")
    expected_jobs = set(actions["job_id"].astype(str))
    provided_jobs = set(frame["job_id"])
    if provided_jobs != expected_jobs:
        missing_jobs = len(expected_jobs - provided_jobs)
        extra_jobs = len(provided_jobs - expected_jobs)
        raise ValueError(
            f"External start job set mismatch: missing={missing_jobs}, extra={extra_jobs}."
        )
    valid_pairs = set(
        zip(actions["job_id"].astype(str), actions["action_id"].astype(str), strict=False)
    )
    provided_pairs = set(zip(frame["job_id"], frame["action_id"], strict=False))
    invalid_pairs = sorted(provided_pairs - valid_pairs)
    if invalid_pairs:
        raise ValueError(
            f"External start contains {len(invalid_pairs)} unknown job/action pairs."
        )
    x = np.array(
        [
            int((str(job_id), str(action_id)) in provided_pairs)
            for job_id, action_id in zip(
                actions["job_id"], actions["action_id"], strict=False
            )
        ],
        dtype=int,
    )
    violations = one_hot_violations(actions, x)
    if int(violations["total"]) != 0:
        raise ValueError(
            f"External start is not one-hot feasible: {violations['total']} violations."
        )
    return x


def run_recourse_polish_multistart(
    actions: pd.DataFrame,
    site_ts: pd.DataFrame,
    q: np.ndarray,
    starts: dict[str, np.ndarray],
    args: argparse.Namespace,
    baseline_metrics: dict[str, Any],
    baseline_load_mw: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    max_action_cost_usd = (
        float(baseline_metrics["true_action_cost_usd"]) + float(args.risk_max_action_cost_increase)
    )
    max_dummy_actions = int(baseline_metrics["dummy_actions"])
    start_tokens = parse_csv_tokens(args.recourse_polish_starts)
    order_tokens = parse_csv_tokens(args.recourse_polish_orders)
    if "all" in start_tokens:
        start_tokens = ["baseline", "greedy", "cut"]
        if "external" in starts:
            start_tokens.append("external")
    if "all" in order_tokens:
        order_tokens = ["forward", "reverse", "high_load_first", "low_load_first"]
    start_tokens = start_tokens or ["baseline"]
    order_tokens = order_tokens or ["forward"]

    unique_starts: dict[str, np.ndarray] = {}
    seen: set[bytes] = set()
    for start_name in start_tokens:
        if start_name not in starts:
            continue
        candidate = starts[start_name].astype(int)
        key = candidate.tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique_starts[start_name] = candidate
    if not unique_starts:
        unique_starts["baseline"] = starts["baseline"].astype(int)

    best_x: np.ndarray | None = None
    best_report: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_objective = math.inf
    best_feasible = False
    attempts: list[dict[str, Any]] = []

    for start_name, start_x in unique_starts.items():
        for job_order in order_tokens:
            polished_x, polish_report = recourse_polish_search(
                actions=actions,
                site_ts=site_ts,
                start_x=start_x,
                reference_load_mw=baseline_load_mw,
                max_passes=args.recourse_polish_passes,
                max_action_cost_usd=max_action_cost_usd,
                max_dummy_actions=max_dummy_actions,
                expected_weight=args.recourse_polish_expected_weight,
                cvar_weight=args.recourse_polish_cvar_weight,
                peak_weight=args.recourse_polish_peak_weight,
                action_cost_weight=args.recourse_polish_action_cost_weight,
                min_gain=args.recourse_polish_min_gain,
                job_order=job_order,
            )
            metrics = schedule_metrics(actions, site_ts, q, polished_x, reference_load_mw=baseline_load_mw)
            objective = recourse_polish_objective(
                metrics["recourse"],
                float(metrics["true_action_cost_usd"]),
                float(metrics["peak_it_mw"]),
                expected_weight=args.recourse_polish_expected_weight,
                cvar_weight=args.recourse_polish_cvar_weight,
                peak_weight=args.recourse_polish_peak_weight,
                action_cost_weight=args.recourse_polish_action_cost_weight,
            )
            feasible = bool(
                metrics["one_hot"]["total"] == 0
                and int(metrics["dummy_actions"]) <= max_dummy_actions
                and float(metrics["true_action_cost_usd"]) <= max_action_cost_usd + 1e-9
            )
            attempts.append(
                {
                    "start": start_name,
                    "job_order": job_order,
                    "feasible": feasible,
                    "objective": float(objective),
                    "qubo_energy": float(metrics["qubo_energy"]),
                    "true_action_cost_usd": float(metrics["true_action_cost_usd"]),
                    "expected_recourse_cost_usd": float(metrics["recourse"]["expected_recourse_cost_usd"]),
                    "cvar_recourse_cost_usd": float(metrics["recourse"]["cvar_recourse_cost_usd"]),
                    "peak_it_mw": float(metrics["peak_it_mw"]),
                    "dummy_actions": int(metrics["dummy_actions"]),
                    "one_hot_violations": int(metrics["one_hot"]["total"]),
                    "total_improvements": int(polish_report["total_improvements"]),
                }
            )
            better = False
            if feasible and not best_feasible:
                better = True
            elif feasible == best_feasible and objective < best_objective - 1e-9:
                better = True
            if better:
                best_feasible = feasible
                best_objective = objective
                best_x = polished_x
                best_report = polish_report
                best_metrics = metrics

    if best_x is None or best_report is None or best_metrics is None:
        raise RuntimeError("Recourse polish did not produce any candidate.")

    best_report = dict(best_report)
    best_report.update(
        {
            "multi_start": True,
            "requested_starts": start_tokens,
            "requested_orders": order_tokens,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "best_objective": float(best_objective),
            "best_feasible": best_feasible,
            "max_action_cost_usd": max_action_cost_usd,
            "max_dummy_actions": max_dummy_actions,
        }
    )
    return best_x, best_report, best_metrics


def qubo_energy(q: np.ndarray, x: np.ndarray) -> float:
    return float(x @ q @ x)


def baseline_solution(actions: pd.DataFrame) -> np.ndarray:
    x = np.zeros(len(actions), dtype=int)
    for _, group in actions.groupby("job_id", sort=False):
        non_dummy = group.loc[~group["is_dummy"]]
        candidates = non_dummy if not non_dummy.empty else group
        idx = int(candidates["action_cost"].astype(float).idxmin())
        x[actions.index.get_loc(idx)] = 1
    return x


def greedy_one_hot_search(
    actions: pd.DataFrame,
    q: np.ndarray,
    start_x: np.ndarray,
    max_passes: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Coordinate-descent search over one job's action choice at a time."""
    x = start_x.copy().astype(int)
    qx = q @ x
    total_improvements = 0
    pass_reports: list[dict[str, Any]] = []

    groups = [
        [actions.index.get_loc(idx) for idx in group.index]
        for _, group in actions.groupby("job_id", sort=False)
    ]
    for pass_idx in range(max_passes):
        improvements = 0
        energy_gain = 0.0
        for positions in groups:
            selected = [pos for pos in positions if x[pos] == 1]
            current = selected[0] if len(selected) == 1 else None
            best_delta = 0.0
            best_vector: np.ndarray | None = None
            for candidate in positions:
                delta = np.zeros(len(x), dtype=np.float64)
                if current is not None:
                    delta[current] -= 1.0
                else:
                    for pos in selected:
                        delta[pos] -= 1.0
                delta[candidate] += 1.0
                if not np.any(delta):
                    continue
                change = float(2.0 * (delta @ qx) + delta @ q @ delta)
                if change < best_delta - 1e-9:
                    best_delta = change
                    best_vector = delta
            if best_vector is not None:
                x = (x + best_vector).astype(int)
                qx = qx + q @ best_vector
                improvements += 1
                total_improvements += 1
                energy_gain += -best_delta
        pass_reports.append(
            {
                "pass": pass_idx + 1,
                "improvements": improvements,
                "energy_gain": energy_gain,
            }
        )
        if improvements == 0:
            break

    return x, {"passes": pass_reports, "total_improvements": total_improvements}


def repair_one_hot(actions: pd.DataFrame, q: np.ndarray, raw_x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    repaired = raw_x.copy().astype(int)
    group_reports: list[dict[str, Any]] = []
    for job_id, group in actions.groupby("job_id", sort=False):
        positions = [actions.index.get_loc(idx) for idx in group.index]
        selected = [pos for pos in positions if repaired[pos] == 1]
        if len(selected) == 1:
            continue
        # Choose the action with lowest marginal QUBO contribution under the current vector.
        best_pos = min(
            positions,
            key=lambda pos: float(q[pos, pos] + 2.0 * (q[pos, :] @ repaired - q[pos, pos] * repaired[pos])),
        )
        for pos in positions:
            repaired[pos] = 1 if pos == best_pos else 0
        group_reports.append(
            {
                "job_id": job_id,
                "raw_selected_count": len(selected),
                "chosen_action_id": str(actions.iloc[best_pos]["action_id"]),
            }
        )
    return repaired, {"one_hot_repairs": group_reports, "repair_count": len(group_reports)}


def one_hot_violations(actions: pd.DataFrame, x: np.ndarray) -> dict[str, int]:
    zero = 0
    multi = 0
    for _, group in actions.groupby("job_id", sort=False):
        positions = [actions.index.get_loc(idx) for idx in group.index]
        count = int(x[positions].sum())
        if count == 0:
            zero += 1
        elif count > 1:
            multi += 1
    return {"zero_selected_jobs": zero, "multi_selected_jobs": multi, "total": zero + multi}


def selected_vectors(actions: pd.DataFrame, x: np.ndarray, horizon: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    selected = actions.loc[x.astype(bool)].copy()
    power_vectors = np.array([parse_vector(v) for v in selected["power_vector_mw"]], dtype=np.float64)
    gpu_vectors = np.array([parse_vector(v) for v in selected["gpu_vector"]], dtype=np.float64)
    if len(selected):
        load = power_vectors.sum(axis=0)
        gpu_load = gpu_vectors.sum(axis=0)
    else:
        load = np.zeros(horizon)
        gpu_load = np.zeros(horizon)
    return selected, load, gpu_load


def recourse_scenarios(horizon: int) -> list[dict[str, Any]]:
    evening = np.ones(horizon)
    if horizon >= 18:
        evening[16:21] = 0.55
    solar_short = np.ones(horizon)
    if horizon >= 14:
        solar_short[9:15] = 0.60
    return [
        {
            "name": "nominal",
            "probability": 0.50,
            "renewable_factor": np.ones(horizon),
            "price_factor": np.ones(horizon),
            "flex_factor": 0.50,
        },
        {
            "name": "green_short_peak_price",
            "probability": 0.20,
            "renewable_factor": 0.55 * np.ones(horizon),
            "price_factor": 1.35 * np.ones(horizon),
            "flex_factor": 0.85,
        },
        {
            "name": "high_green_low_price",
            "probability": 0.15,
            "renewable_factor": 1.25 * np.ones(horizon),
            "price_factor": 0.75 * np.ones(horizon),
            "flex_factor": 0.25,
        },
        {
            "name": "evening_stress",
            "probability": 0.10,
            "renewable_factor": evening,
            "price_factor": np.where(evening < 1.0, 1.60, 1.05),
            "flex_factor": 0.90,
        },
        {
            "name": "solar_midday_shortfall",
            "probability": 0.05,
            "renewable_factor": solar_short,
            "price_factor": np.where(solar_short < 1.0, 1.25, 1.00),
            "flex_factor": 0.70,
        },
    ]


def recourse_inputs(site_ts: pd.DataFrame, reference_load_mw: np.ndarray) -> dict[str, Any]:
    ordered_site = site_ts.sort_values("slot")
    renewable_raw = ordered_site["renewable_mw"].astype(float).to_numpy()
    renewable_cf = renewable_raw / max(float(renewable_raw.max()), 1e-9)

    green_capacity_mw = max(float(reference_load_mw.max()) * 0.85, 1e-9)
    return {
        "lmp": ordered_site["lmp_usd_mwh"].astype(float).to_numpy(),
        "carbon": ordered_site["carbon_kgco2_mwh"].astype(float).to_numpy(),
        "penalty_short": ordered_site["penalty_short_usd_mwh"].astype(float).to_numpy(),
        "base_green_mw": renewable_cf * green_capacity_mw,
        "flex_target_base": 0.30
        * ordered_site["flex_call_prob"].astype(float).to_numpy()
        * np.maximum(reference_load_mw, 0.0),
        "green_capacity_mw": green_capacity_mw,
    }


def scenario_recourse_row(
    load_mw: np.ndarray,
    reference_load_mw: np.ndarray,
    inputs: dict[str, Any],
    scenario: dict[str, Any],
) -> dict[str, Any]:
    renewable_avail = inputs["base_green_mw"] * np.asarray(scenario["renewable_factor"], dtype=float)
    price = inputs["lmp"] * np.asarray(scenario["price_factor"], dtype=float)
    flex_target = inputs["flex_target_base"] * float(scenario["flex_factor"])
    flex_delivery_potential = np.maximum(reference_load_mw - load_mw, 0.0)

    renewable_used = np.minimum(load_mw, renewable_avail)
    grid_import = np.maximum(load_mw - renewable_avail, 0.0)
    curtailment = np.maximum(renewable_avail - load_mw, 0.0)
    flex_delivered = np.minimum(flex_target, flex_delivery_potential)
    flex_shortfall = np.maximum(flex_target - flex_delivered, 0.0)

    grid_cost = float(grid_import @ price)
    curtailment_cost = float(curtailment.sum() * 5.0)
    flex_shortfall_cost = float(flex_shortfall @ inputs["penalty_short"])
    total_cost = grid_cost + curtailment_cost + flex_shortfall_cost
    return {
        "name": scenario["name"],
        "probability": float(scenario["probability"]),
        "total_recourse_cost_usd": total_cost,
        "grid_cost_usd": grid_cost,
        "curtailment_cost_usd": curtailment_cost,
        "flex_shortfall_cost_usd": flex_shortfall_cost,
        "grid_import_mwh": float(grid_import.sum()),
        "renewable_used_mwh": float(renewable_used.sum()),
        "curtailment_mwh": float(curtailment.sum()),
        "flex_target_mwh": float(flex_target.sum()),
        "flex_delivered_mwh": float(flex_delivered.sum()),
        "flex_shortfall_mwh": float(flex_shortfall.sum()),
        "carbon_kgco2": float(grid_import @ inputs["carbon"]),
    }


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
    return {
        "alpha": alpha,
        "var": float(var),
        "cvar": float((above_cost + var * var_mass) / tail_prob),
    }


def cvar_scenario_weights(costs: list[float], probabilities: list[float], alpha: float) -> list[float]:
    """Return d(CVaR_alpha)/d(cost_scenario) weights for fixed scenario ordering."""
    indexed = sorted(enumerate(zip(costs, probabilities)), key=lambda item: item[1][0])
    cumulative = 0.0
    var = indexed[-1][1][0]
    for _, (cost, prob) in indexed:
        cumulative += prob
        if cumulative >= alpha:
            var = cost
            break

    tail_prob = max(1.0 - alpha, 1e-12)
    weights = np.zeros(len(costs), dtype=np.float64)
    above_mass = 0.0
    var_indices: list[int] = []
    var_prob = 0.0
    for idx, (cost, prob) in indexed:
        if cost > var + 1e-12:
            weights[idx] = prob / tail_prob
            above_mass += prob
        elif math.isclose(cost, var, rel_tol=1e-12, abs_tol=1e-12):
            var_indices.append(idx)
            var_prob += prob

    var_mass = max(tail_prob - above_mass, 0.0)
    if var_indices and var_mass > 0.0 and var_prob > 0.0:
        for idx in var_indices:
            weights[idx] += (var_mass * probabilities[idx] / var_prob) / tail_prob
    return [float(v) for v in weights.tolist()]


def recourse_marginal_slot_prices(
    reference_load_mw: np.ndarray,
    site_ts: pd.DataFrame,
    alpha: float = 0.95,
    finite_delta_mw: float = 1e-6,
) -> dict[str, Any]:
    """Compute finite-difference marginal recourse prices at a reference schedule."""
    horizon = len(site_ts)
    if len(reference_load_mw) != horizon:
        raise ValueError("Reference load and site horizon lengths differ.")

    inputs = recourse_inputs(site_ts, reference_load_mw)
    scenarios = recourse_scenarios(horizon)
    reference_costs: list[float] = []
    probabilities: list[float] = []
    scenario_marginals: list[np.ndarray] = []
    delta = max(float(finite_delta_mw), 1e-12)

    for scenario in scenarios:
        base_row = scenario_recourse_row(reference_load_mw, reference_load_mw, inputs, scenario)
        reference_costs.append(float(base_row["total_recourse_cost_usd"]))
        probabilities.append(float(scenario["probability"]))
        marginal = np.zeros(horizon, dtype=np.float64)
        for slot in range(horizon):
            up = reference_load_mw.copy()
            up[slot] += delta
            up_cost = scenario_recourse_row(up, reference_load_mw, inputs, scenario)[
                "total_recourse_cost_usd"
            ]
            down_delta = min(delta, max(float(reference_load_mw[slot]), 0.0))
            if down_delta > 0.0:
                down = reference_load_mw.copy()
                down[slot] -= down_delta
                down_cost = scenario_recourse_row(down, reference_load_mw, inputs, scenario)[
                    "total_recourse_cost_usd"
                ]
                marginal[slot] = (float(up_cost) - float(down_cost)) / (delta + down_delta)
            else:
                marginal[slot] = (float(up_cost) - float(base_row["total_recourse_cost_usd"])) / delta
        scenario_marginals.append(marginal)

    scenario_marginal_matrix = np.vstack(scenario_marginals)
    probability_vector = np.array(probabilities, dtype=np.float64)
    cvar_weights = np.array(cvar_scenario_weights(reference_costs, probabilities, alpha), dtype=np.float64)
    expected_prices = probability_vector @ scenario_marginal_matrix
    cvar_prices = cvar_weights @ scenario_marginal_matrix
    return {
        "expected_slot_price_usd_mwh": expected_prices,
        "cvar_slot_price_usd_mwh": cvar_prices,
        "reference_scenario_costs": [
            {
                "name": str(scenario["name"]),
                "probability": float(prob),
                "cost_usd": float(cost),
                "cvar_weight": float(weight),
            }
            for scenario, prob, cost, weight in zip(
                scenarios,
                probabilities,
                reference_costs,
                cvar_weights.tolist(),
            )
        ],
        "cvar_scenario_weights": {
            str(scenario["name"]): float(weight)
            for scenario, weight in zip(scenarios, cvar_weights.tolist())
        },
    }


def evaluate_recourse(
    load_mw: np.ndarray,
    reference_load_mw: np.ndarray,
    site_ts: pd.DataFrame,
    alpha: float = 0.95,
) -> dict[str, Any]:
    """Evaluate exact per-slot renewable/grid/flex recourse for a fixed schedule."""
    ordered_site = site_ts.sort_values("slot")
    horizon = len(ordered_site)
    if len(load_mw) != horizon or len(reference_load_mw) != horizon:
        raise ValueError("Schedule load and site horizon lengths differ.")

    inputs = recourse_inputs(site_ts, reference_load_mw)

    scenario_rows: list[dict[str, Any]] = []
    for scenario in recourse_scenarios(horizon):
        scenario_rows.append(scenario_recourse_row(load_mw, reference_load_mw, inputs, scenario))

    costs = [row["total_recourse_cost_usd"] for row in scenario_rows]
    probabilities = [row["probability"] for row in scenario_rows]
    expected_cost = float(sum(cost * prob for cost, prob in zip(costs, probabilities)))
    cvar = weighted_cvar(costs, probabilities, alpha=alpha)
    worst = max(scenario_rows, key=lambda row: row["total_recourse_cost_usd"])
    return {
        "model": "closed_form_per_slot_grid_renewable_flex_shortfall",
        "alpha": alpha,
        "green_capacity_mw": inputs["green_capacity_mw"],
        "expected_recourse_cost_usd": expected_cost,
        "var_recourse_cost_usd": cvar["var"],
        "cvar_recourse_cost_usd": cvar["cvar"],
        "worst_scenario": worst["name"],
        "worst_recourse_cost_usd": worst["total_recourse_cost_usd"],
        "scenario_count": len(scenario_rows),
        "scenarios": scenario_rows,
    }


def schedule_metrics(
    actions: pd.DataFrame,
    site_ts: pd.DataFrame,
    q: np.ndarray,
    x: np.ndarray,
    reference_load_mw: np.ndarray | None = None,
) -> dict[str, Any]:
    selected, load, gpu_load = selected_vectors(actions, x, len(site_ts))
    reference = load if reference_load_mw is None else reference_load_mw

    ordered_site = site_ts.sort_values("slot")
    lmp = ordered_site["lmp_usd_mwh"].astype(float).to_numpy()
    carbon = ordered_site["carbon_kgco2_mwh"].astype(float).to_numpy()
    renewable = ordered_site["renewable_mw"].astype(float).to_numpy()

    imported_mwh = np.maximum(load - renewable, 0.0)
    renewable_used = np.minimum(load, renewable)
    mode_counts = Counter(selected["mode"].astype(str).tolist())

    return {
        "selected_actions": int(x.sum()),
        "jobs": int(actions["job_id"].nunique()),
        "one_hot": one_hot_violations(actions, x),
        "dummy_actions": int(selected["is_dummy"].sum()),
        "mode_counts": dict(sorted(mode_counts.items())),
        "qubo_energy": qubo_energy(q, x),
        "true_action_cost_usd": float(selected["action_cost"].astype(float).sum()),
        "energy_mwh": float(selected["energy_mwh"].astype(float).sum()),
        "energy_cost_usd": float(selected["energy_cost_usd"].astype(float).sum()),
        "sla_cost_usd": float(selected["sla_cost_usd"].astype(float).sum()),
        "extra_mode_cost_usd": float(selected["extra_mode_cost_usd"].astype(float).sum()),
        "peak_it_mw": float(load.max()) if load.size else 0.0,
        "mean_it_mw": float(load.mean()) if load.size else 0.0,
        "peak_gpu_req": float(gpu_load.max()) if gpu_load.size else 0.0,
        "price_weighted_energy_cost_usd": float(load @ lmp),
        "carbon_kgco2": float(load @ carbon),
        "renewable_used_mwh": float(renewable_used.sum()),
        "grid_import_mwh_proxy": float(imported_mwh.sum()),
        "recourse": evaluate_recourse(load, reference, site_ts),
        "slot_load_mw": [float(v) for v in load.tolist()],
        "slot_gpu_req": [float(v) for v in gpu_load.tolist()],
    }


def import_plectrum() -> dict[str, Any]:
    from plectrum import (  # type: ignore
        CloudSolver,
        GEAR_BALANCED,
        GEAR_FAST,
        GEAR_PRECISE,
        LocalSolver,
        PlectrumError,
        QuboTask,
    )

    return {
        "CloudSolver": CloudSolver,
        "GEAR_BALANCED": GEAR_BALANCED,
        "GEAR_FAST": GEAR_FAST,
        "GEAR_PRECISE": GEAR_PRECISE,
        "LocalSolver": LocalSolver,
        "PlectrumError": PlectrumError,
        "QuboTask": QuboTask,
    }


def solve_with_plectrum(
    q: np.ndarray,
    solver_name: str,
    gear_name: str,
    task_name: str,
    shot_count: int,
    use_credit: bool,
    use_coupon: bool,
) -> dict[str, Any]:
    plectrum = import_plectrum()
    gear = plectrum[GEAR_BY_NAME[gear_name]]
    task = plectrum["QuboTask"](
        name=task_name,
        data=q,
        shot_count=shot_count,
        use_credit=use_credit,
        use_coupon=use_coupon,
    )
    if solver_name == "local":
        solver = plectrum["LocalSolver"](gear=gear)
    elif solver_name == "cloud":
        api_key = os.getenv("PLECTRUM_API_KEY")
        if not api_key:
            raise RuntimeError("PLECTRUM_API_KEY is not set for cloud solving.")
        solver = plectrum["CloudSolver"](api_key=api_key, gear=gear)
    else:
        raise ValueError(f"Unsupported solver: {solver_name}")

    result = task.solve(solver=solver)
    spin_config = result.get("result", {}).get("spin_config", [])
    x = np.array(spin_config, dtype=int)
    if len(x) != len(q):
        raise RuntimeError(f"Solver returned {len(x)} values for {len(q)} variables.")
    # Cloud QUBO returns 0/1. Keep this guard for possible local/SDK variants.
    if set(np.unique(x).tolist()).issubset({-1, 1}):
        x = ((x + 1) // 2).astype(int)
    return {
        "solver": solver_name,
        "gear": gear_name,
        "raw_result": result,
        "x": x,
    }


def write_outputs(
    args: argparse.Namespace,
    selected_jobs: list[str],
    selected_actions: pd.DataFrame,
    baseline_x: np.ndarray,
    qubo_meta: dict[str, Any],
    baseline_metrics: dict[str, Any],
    solver_records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = REPORTS_DIR / args.output_json
    report_path = REPORTS_DIR / args.output_md
    solution_path = REPORTS_DIR / args.output_solution_csv
    risk_solution_path = REPORTS_DIR / args.output_risk_solution_csv

    serializable_records = []
    best_source = "baseline"
    best_metrics = baseline_metrics
    best_x = baseline_x.astype(int)
    risk_source = "baseline"
    risk_metrics = baseline_metrics
    risk_x = baseline_x.astype(int)
    max_risk_dummy_actions = int(baseline_metrics["dummy_actions"])
    max_risk_action_cost = (
        float(baseline_metrics["true_action_cost_usd"]) + float(args.risk_max_action_cost_increase)
    )
    for record in solver_records:
        if record.get("status") == "failed":
            serializable_records.append(
                {
                    "solver": record["solver"],
                    "gear": record.get("gear"),
                    "status": "failed",
                    "error_type": record["error_type"],
                    "error_message": record["error_message"],
                }
            )
            continue
        result = record["raw_result"]
        compact_result = {
            "task_id": result.get("task_id"),
            "result": {
                "energy": result.get("result", {}).get("energy"),
                "time": result.get("result", {}).get("time"),
                "ok": result.get("result", {}).get("ok"),
                "spin_config_length": len(result.get("result", {}).get("spin_config", [])),
            },
        }
        serializable_records.append(
            {
                "solver": record["solver"],
                "gear": record["gear"],
                "status": "success",
                "raw_solver_result": compact_result,
                "raw_metrics": record["raw_metrics"],
                "repaired_metrics": record["repaired_metrics"],
                "repair": record["repair"],
            }
        )
        metrics = record["repaired_metrics"]
        if metrics["qubo_energy"] < best_metrics["qubo_energy"] - 1e-9:
            best_source = str(record["solver"])
            best_metrics = metrics
            best_x = record["repaired_x"].astype(int)
        if (
            metrics["one_hot"]["total"] == 0
            and int(metrics["dummy_actions"]) <= max_risk_dummy_actions
            and float(metrics["true_action_cost_usd"]) <= max_risk_action_cost
            and metrics["recourse"]["cvar_recourse_cost_usd"]
            < risk_metrics["recourse"]["cvar_recourse_cost_usd"] - 1e-9
        ):
            risk_source = str(record["solver"])
            risk_metrics = metrics
            risk_x = record["repaired_x"].astype(int)

    accepted_actions = selected_actions.loc[best_x.astype(bool)].copy()
    accepted_actions.insert(0, "selection_source", best_source)
    accepted_actions.to_csv(solution_path, index=False, encoding="utf-8-sig")
    risk_actions = selected_actions.loc[risk_x.astype(bool)].copy()
    risk_actions.insert(0, "selection_source", risk_source)
    risk_actions.to_csv(risk_solution_path, index=False, encoding="utf-8-sig")

    safe_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    payload = {
        "generated_at": utc_now(),
        "root": str(ROOT),
        "args": safe_args,
        "selected_job_count": len(selected_jobs),
        "selected_action_count": int(len(selected_actions)),
        "selected_jobs": selected_jobs,
        "qubo": qubo_meta,
        "baseline": baseline_metrics,
        "accepted_solution": {
            "source": best_source,
            "metrics": best_metrics,
            "actions_csv": str(solution_path),
        },
        "risk_accepted_solution": {
            "source": risk_source,
            "metrics": risk_metrics,
            "actions_csv": str(risk_solution_path),
            "selection_rule": "minimum_CVaR95_among_one_hot_candidates_with_dummy_count_no_greater_than_baseline_and_action_cost_within_limit",
            "max_action_cost_usd": max_risk_action_cost,
        },
        "solvers": serializable_records,
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# TAOLIN-DC-SIM QUBO Solver Report",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "## Instance",
        "",
        f"- Jobs in QUBO: {len(selected_jobs)}",
        f"- Candidate action variables / spins: {len(selected_actions)}",
        f"- Encoding efficiency R_job = {len(selected_actions) / max(len(selected_jobs), 1):.4f}",
        f"- QUBO nonzeros: {qubo_meta['q_nonzero']}",
        f"- QUBO value range: [{qubo_meta['q_min']:.6g}, {qubo_meta['q_max']:.6g}]",
        f"- One-hot penalty: {qubo_meta['one_hot_penalty']}",
        f"- Peak weight: {qubo_meta['peak_weight']}",
        f"- Risk weight: {qubo_meta['risk_weight']}",
        f"- Recourse expected weight: {qubo_meta['recourse']['weight']}",
        f"- Recourse CVaR weight: {qubo_meta['recourse']['cvar_weight']}",
        f"- Recourse CVaR alpha: {qubo_meta['recourse']['alpha']}",
        f"- Recourse action scale USD: "
        f"{qubo_meta['recourse'].get('action_term_scale_usd', 'disabled')}",
        f"- Recourse cut iterations: "
        f"{qubo_meta['recourse'].get('cut', {}).get('completed_iterations', 0)}/"
        f"{qubo_meta['recourse'].get('cut', {}).get('requested_iterations', 0)}",
        f"- Dummy cost: {qubo_meta['dummy_cost']}",
        f"- Dummy-only cost: {qubo_meta['dummy_only_cost']}",
        f"- Risk-accepted action cost limit: {max_risk_action_cost:.6f}",
        "",
        "## Baseline",
        "",
        f"- True action cost: {baseline_metrics['true_action_cost_usd']:.6f}",
        f"- QUBO energy: {baseline_metrics['qubo_energy']:.6f}",
        f"- Peak IT MW: {baseline_metrics['peak_it_mw']:.6f}",
        f"- Expected recourse cost: {baseline_metrics['recourse']['expected_recourse_cost_usd']:.6f}",
        f"- CVaR95 recourse cost: {baseline_metrics['recourse']['cvar_recourse_cost_usd']:.6f}",
        f"- Dummy actions: {baseline_metrics['dummy_actions']}",
        f"- Mode counts: `{baseline_metrics['mode_counts']}`",
        "",
        "## Accepted Solution",
        "",
        f"- Source: {best_source}",
        f"- QUBO energy: {best_metrics['qubo_energy']:.6f}",
        f"- True action cost: {best_metrics['true_action_cost_usd']:.6f}",
        f"- Peak IT MW: {best_metrics['peak_it_mw']:.6f}",
        f"- Expected recourse cost: {best_metrics['recourse']['expected_recourse_cost_usd']:.6f}",
        f"- CVaR95 recourse cost: {best_metrics['recourse']['cvar_recourse_cost_usd']:.6f}",
        f"- Worst recourse scenario: `{best_metrics['recourse']['worst_scenario']}`",
        f"- Dummy actions: {best_metrics['dummy_actions']}",
        f"- Mode counts: `{best_metrics['mode_counts']}`",
        f"- Selected actions CSV: `{solution_path.relative_to(ROOT).as_posix()}`",
        "",
        "## Risk-Accepted Solution",
        "",
        f"- Source: {risk_source}",
        f"- QUBO energy: {risk_metrics['qubo_energy']:.6f}",
        f"- True action cost: {risk_metrics['true_action_cost_usd']:.6f}",
        f"- Peak IT MW: {risk_metrics['peak_it_mw']:.6f}",
        f"- Expected recourse cost: {risk_metrics['recourse']['expected_recourse_cost_usd']:.6f}",
        f"- CVaR95 recourse cost: {risk_metrics['recourse']['cvar_recourse_cost_usd']:.6f}",
        f"- Dummy actions: {risk_metrics['dummy_actions']}",
        f"- Mode counts: `{risk_metrics['mode_counts']}`",
        f"- Selected actions CSV: `{risk_solution_path.relative_to(ROOT).as_posix()}`",
        "",
        "## Solver Results",
        "",
    ]
    solver_header = lines[-2:]
    lines = lines[:-2]
    recourse_cut = qubo_meta["recourse"].get("cut", {})
    if recourse_cut.get("enabled"):
        lines.extend(
            [
                "## Recourse Cut Iterations",
                "",
                f"- Completed iterations: {recourse_cut['completed_iterations']}",
                f"- Damping: {recourse_cut['damping']}",
                f"- Final reference delta from baseline MW: "
                f"{recourse_cut['final_reference_linf_delta_from_baseline_mw']:.6f}",
                "",
                "| Iteration | QUBO | True cost | Expected recourse | CVaR95 | Peak MW | Dummy | Ref delta MW | Improvements |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for cut_record in recourse_cut["iterations"]:
            lines.append(
                f"| {cut_record['iteration']} "
                f"| {cut_record['qubo_energy']:.6f} "
                f"| {cut_record['true_action_cost_usd']:.6f} "
                f"| {cut_record['expected_recourse_cost_usd']:.6f} "
                f"| {cut_record['cvar_recourse_cost_usd']:.6f} "
                f"| {cut_record['peak_it_mw']:.6f} "
                f"| {cut_record['dummy_actions']} "
                f"| {cut_record['reference_load_linf_delta_mw']:.6f} "
                f"| {cut_record['greedy_total_improvements']} |"
            )
        lines.append("")
    lines.extend(solver_header)
    for record in serializable_records:
        if record.get("status") == "failed":
            lines.extend(
                [
                    f"### {record['solver']} ({record.get('gear')})",
                    "",
                    f"- Status: failed",
                    f"- Error type: `{record['error_type']}`",
                    f"- Error: {record['error_message']}",
                    "",
                ]
            )
            continue
        raw = record["raw_metrics"]
        repaired = record["repaired_metrics"]
        task_id = record["raw_solver_result"].get("task_id")
        task_line = f", task_id={task_id}" if task_id else ""
        lines.extend(
            [
                f"### {record['solver']} ({record['gear']}{task_line})",
                "",
                f"- Raw one-hot violations: {raw['one_hot']['total']} "
                f"(zero={raw['one_hot']['zero_selected_jobs']}, multi={raw['one_hot']['multi_selected_jobs']})",
                f"- Repaired one-hot violations: {repaired['one_hot']['total']}",
                f"- Repair count: {record['repair']['repair_count']}",
                f"- Repaired QUBO energy: {repaired['qubo_energy']:.6f}",
                f"- Repaired true action cost: {repaired['true_action_cost_usd']:.6f}",
                f"- Repaired peak IT MW: {repaired['peak_it_mw']:.6f}",
                f"- Repaired expected recourse cost: {repaired['recourse']['expected_recourse_cost_usd']:.6f}",
                f"- Repaired CVaR95 recourse cost: {repaired['recourse']['cvar_recourse_cost_usd']:.6f}",
                f"- Repaired dummy actions: {repaired['dummy_actions']}",
                f"- Repaired mode counts: `{repaired['mode_counts']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Caveats",
            "",
            "- CAISO price/demand data is still synthetic in the current cleaned dataset; this run is valid as a structural QUBO/platform test, not as a market-facing claim.",
            "- Recourse is evaluated after decoding using fixed stress scenarios and a normalized data-center renewable budget because the workload trace and site renewable data are at different scales.",
            "- When recourse QUBO weights are enabled, expected/CVaR recourse enters as a first-order marginal-price linearization around the baseline schedule, not as a full binary recourse formulation.",
            "- When recourse polish is enabled, one-job candidate moves are evaluated with the exact closed-form recourse/CVaR evaluator under the configured action-cost and dummy-action guards.",
            "- Storage, unit commitment, DC power flow, and integer redispatch are not yet included in the recourse model.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--solver", choices=("local", "cloud", "both"), default="local")
    parser.add_argument("--gear", choices=("fast", "balanced", "precise"), default="balanced")
    parser.add_argument("--max-jobs", type=int, default=120, help="0 means use all jobs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shot-count", type=int, default=10)
    parser.add_argument("--post-polish-passes", type=int, default=5)
    parser.add_argument("--one-hot-penalty", type=float, default=80.0)
    parser.add_argument("--peak-weight", type=float, default=0.002)
    parser.add_argument("--risk-weight", type=float, default=0.10)
    parser.add_argument("--recourse-weight", type=float, default=0.0)
    parser.add_argument("--recourse-cvar-weight", type=float, default=0.0)
    parser.add_argument("--recourse-alpha", type=float, default=0.95)
    parser.add_argument("--recourse-finite-delta-mw", type=float, default=1e-6)
    parser.add_argument("--recourse-cut-iterations", type=int, default=0)
    parser.add_argument("--recourse-cut-damping", type=float, default=1.0)
    parser.add_argument("--recourse-cut-stop-cvar-tol", type=float, default=1e-6)
    parser.add_argument("--recourse-cut-stop-load-tol-mw", type=float, default=1e-9)
    parser.add_argument("--recourse-polish-passes", type=int, default=0)
    parser.add_argument("--recourse-polish-expected-weight", type=float, default=0.0)
    parser.add_argument("--recourse-polish-cvar-weight", type=float, default=1.0)
    parser.add_argument("--recourse-polish-peak-weight", type=float, default=0.0)
    parser.add_argument("--recourse-polish-action-cost-weight", type=float, default=0.0)
    parser.add_argument("--recourse-polish-min-gain", type=float, default=1e-6)
    parser.add_argument("--recourse-polish-starts", default="baseline")
    parser.add_argument("--recourse-polish-orders", default="forward")
    parser.add_argument(
        "--recourse-polish-start-csv",
        type=Path,
        default=None,
        help=(
            "Optional one-action-per-job CSV added as the 'external' recourse-polish start. "
            "Use --recourse-polish-starts external or all to evaluate it."
        ),
    )
    parser.add_argument("--dummy-cost", type=float, default=120.0)
    parser.add_argument("--dummy-only-cost", type=float, default=25.0)
    parser.add_argument("--cost-cap", type=float, default=20.0)
    parser.add_argument("--risk-max-action-cost-increase", type=float, default=1.0)
    parser.add_argument("--use-credit", action="store_true")
    parser.add_argument("--use-coupon", action="store_true")
    parser.add_argument("--output-json", default="qubo_solver_results.json")
    parser.add_argument("--output-md", default="qubo_solver_report.md")
    parser.add_argument("--output-solution-csv", default="qubo_accepted_actions.csv")
    parser.add_argument("--output-risk-solution-csv", default="qubo_risk_accepted_actions.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    jobs, actions, site_ts = load_tables(root)
    selected_jobs = choose_jobs(actions, site_ts, max_jobs=args.max_jobs, seed=args.seed)
    selected_actions = actions.loc[actions["job_id"].isin(selected_jobs)].copy().reset_index(drop=True)
    selected_actions["job_id"] = selected_actions["job_id"].astype(str)
    baseline_x = baseline_solution(selected_actions)
    _, baseline_load, _ = selected_vectors(selected_actions, baseline_x, len(site_ts))
    recourse_reference_load = baseline_load
    recourse_cut_records: list[dict[str, Any]] = []
    recourse_cut_start_x = baseline_x
    if recourse_terms_enabled(args) and args.recourse_cut_iterations > 0:
        recourse_reference_load, recourse_cut_records, recourse_cut_start_x = run_recourse_cut_iterations(
            actions=selected_actions,
            site_ts=site_ts,
            args=args,
            baseline_x=baseline_x,
            baseline_load_mw=baseline_load,
        )

    q, qubo_meta = build_qubo_from_args(
        selected_actions,
        site_ts,
        args,
        reference_load_mw=recourse_reference_load,
    )
    if recourse_cut_records:
        _, final_reference_selected_load, _ = selected_vectors(selected_actions, recourse_cut_start_x, len(site_ts))
        qubo_meta["recourse"]["cut"] = {
            "enabled": True,
            "requested_iterations": int(args.recourse_cut_iterations),
            "completed_iterations": len(recourse_cut_records),
            "damping": float(args.recourse_cut_damping),
            "stop_cvar_tol": float(args.recourse_cut_stop_cvar_tol),
            "stop_load_tol_mw": float(args.recourse_cut_stop_load_tol_mw),
            "final_reference_linf_delta_from_baseline_mw": float(
                np.max(np.abs(recourse_reference_load - baseline_load))
            )
            if len(baseline_load)
            else 0.0,
            "final_reference_linf_delta_from_last_candidate_mw": float(
                np.max(np.abs(recourse_reference_load - final_reference_selected_load))
            )
            if len(baseline_load)
            else 0.0,
            "iterations": recourse_cut_records,
        }
    else:
        qubo_meta["recourse"]["cut"] = {
            "enabled": False,
            "requested_iterations": int(args.recourse_cut_iterations),
            "completed_iterations": 0,
        }

    baseline_metrics = schedule_metrics(selected_actions, site_ts, q, baseline_x, reference_load_mw=baseline_load)
    polished_x, polish_report = greedy_one_hot_search(
        selected_actions,
        q,
        baseline_x,
        max_passes=5,
    )
    polish_metrics = schedule_metrics(selected_actions, site_ts, q, polished_x, reference_load_mw=baseline_load)

    solvers = ["local", "cloud"] if args.solver == "both" else [args.solver]
    solver_records: list[dict[str, Any]] = [
        {
            "solver": "greedy_polish",
            "gear": "deterministic",
            "raw_result": {
                "task_id": None,
                "result": {
                    "energy": polish_metrics["qubo_energy"],
                    "time": None,
                    "ok": True,
                    "spin_config": polished_x.astype(int).tolist(),
                },
            },
            "raw_metrics": polish_metrics,
            "repaired_metrics": polish_metrics,
            "repaired_x": polished_x,
            "repair": {
                "repair_count": 0,
                "one_hot_repairs": [],
                "polish": polish_report,
            },
        }
    ]
    if recourse_cut_records:
        cut_polished_x, cut_polish_report = greedy_one_hot_search(
            selected_actions,
            q,
            recourse_cut_start_x,
            max_passes=args.post_polish_passes,
        )
        cut_polish_metrics = schedule_metrics(
            selected_actions,
            site_ts,
            q,
            cut_polished_x,
            reference_load_mw=baseline_load,
        )
        solver_records.append(
            {
                "solver": "recourse_cut_greedy",
                "gear": "deterministic",
                "raw_result": {
                    "task_id": None,
                    "result": {
                        "energy": cut_polish_metrics["qubo_energy"],
                        "time": None,
                        "ok": True,
                        "spin_config": cut_polished_x.astype(int).tolist(),
                    },
                },
                "raw_metrics": cut_polish_metrics,
                "repaired_metrics": cut_polish_metrics,
                "repaired_x": cut_polished_x,
                "repair": {
                    "repair_count": 0,
                    "one_hot_repairs": [],
                    "polish": cut_polish_report,
                    "recourse_cut_iterations": recourse_cut_records,
                },
            }
        )
    if args.recourse_polish_passes > 0:
        recourse_starts = {
            "baseline": baseline_x,
            "greedy": polished_x,
        }
        if recourse_cut_records:
            recourse_starts["cut"] = cut_polished_x
        if args.recourse_polish_start_csv is not None:
            recourse_starts["external"] = load_external_start_vector(
                selected_actions,
                args.recourse_polish_start_csv.resolve(),
            )
        recourse_polished_x, recourse_polish_report, recourse_polish_metrics = run_recourse_polish_multistart(
            actions=selected_actions,
            site_ts=site_ts,
            q=q,
            starts=recourse_starts,
            args=args,
            baseline_metrics=baseline_metrics,
            baseline_load_mw=baseline_load,
        )
        solver_records.append(
            {
                "solver": "recourse_true_polish",
                "gear": "exact_recourse_coordinate_descent",
                "raw_result": {
                    "task_id": None,
                    "result": {
                        "energy": recourse_polish_metrics["qubo_energy"],
                        "time": None,
                        "ok": True,
                        "spin_config": recourse_polished_x.astype(int).tolist(),
                    },
                },
                "raw_metrics": recourse_polish_metrics,
                "repaired_metrics": recourse_polish_metrics,
                "repaired_x": recourse_polished_x,
                "repair": {
                    "repair_count": 0,
                    "one_hot_repairs": [],
                    "recourse_polish": recourse_polish_report,
                },
            }
        )
    task_name = f"TAOLIN-DC-SIM-QUBO-{len(selected_jobs)}jobs-{len(selected_actions)}vars"
    for solver_name in solvers:
        try:
            solved = solve_with_plectrum(
                q=q,
                solver_name=solver_name,
                gear_name=args.gear,
                task_name=task_name,
                shot_count=args.shot_count,
                use_credit=args.use_credit,
                use_coupon=args.use_coupon,
            )
            raw_x = solved["x"]
            repaired_x, repair = repair_one_hot(selected_actions, q, raw_x)
            repaired_metrics = schedule_metrics(
                selected_actions,
                site_ts,
                q,
                repaired_x,
                reference_load_mw=baseline_load,
            )
            solver_records.append(
                {
                    "solver": solved["solver"],
                    "gear": solved["gear"],
                    "raw_result": solved["raw_result"],
                    "raw_metrics": schedule_metrics(selected_actions, site_ts, q, raw_x, reference_load_mw=baseline_load),
                    "repaired_metrics": repaired_metrics,
                    "repaired_x": repaired_x,
                    "repair": repair,
                }
            )
            if args.post_polish_passes > 0:
                polished_from_solver_x, solver_polish_report = greedy_one_hot_search(
                    selected_actions,
                    q,
                    repaired_x,
                    max_passes=args.post_polish_passes,
                )
                polished_from_solver_metrics = schedule_metrics(
                    selected_actions,
                    site_ts,
                    q,
                    polished_from_solver_x,
                    reference_load_mw=baseline_load,
                )
                if (
                    solver_polish_report["total_improvements"] > 0
                    or polished_from_solver_metrics["qubo_energy"] < repaired_metrics["qubo_energy"] - 1e-9
                ):
                    solver_records.append(
                        {
                            "solver": f"{solved['solver']}_polished",
                            "gear": f"{solved['gear']}+greedy",
                            "raw_result": {
                                "task_id": solved["raw_result"].get("task_id"),
                                "result": {
                                    "energy": repaired_metrics["qubo_energy"],
                                    "time": solved["raw_result"].get("result", {}).get("time"),
                                    "ok": solved["raw_result"].get("result", {}).get("ok"),
                                    "spin_config": repaired_x.astype(int).tolist(),
                                },
                            },
                            "raw_metrics": repaired_metrics,
                            "repaired_metrics": polished_from_solver_metrics,
                            "repaired_x": polished_from_solver_x,
                            "repair": {
                                "repair_count": repair["repair_count"],
                                "one_hot_repairs": repair["one_hot_repairs"],
                                "post_polish": solver_polish_report,
                            },
                        }
                    )
        except Exception as exc:
            solver_records.append(
                {
                    "solver": solver_name,
                    "gear": args.gear,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    result_path, report_path = write_outputs(
        args=args,
        selected_jobs=selected_jobs,
        selected_actions=selected_actions,
        baseline_x=baseline_x,
        qubo_meta=qubo_meta,
        baseline_metrics=baseline_metrics,
        solver_records=solver_records,
    )
    print(f"Wrote {result_path}")
    print(f"Wrote {report_path}")
    for record in solver_records:
        if record.get("status") == "failed":
            print(f"{record['solver']}: failed ({record['error_type']}: {record['error_message']})")
            continue
        metrics = record["repaired_metrics"]
        print(
            f"{record['solver']}: energy={metrics['qubo_energy']:.6f}, "
            f"cost={metrics['true_action_cost_usd']:.6f}, "
            f"peak_mw={metrics['peak_it_mw']:.6f}, "
            f"repairs={record['repair']['repair_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
