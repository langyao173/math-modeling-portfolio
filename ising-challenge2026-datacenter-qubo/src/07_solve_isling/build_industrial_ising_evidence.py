"""Build industrial Ising evidence artifacts for TAOLIN-DC-SIM.

This script adds an auditable evidence layer on top of the existing final
solution. It does not overwrite FINAL_SOLUTION.csv. Instead it constructs a
bounded LNS portfolio, runs same-QUBO oracle replacement experiments, replays
each decoded solution with the exact recourse evaluator, persists every
successful polished assignment, and writes compact tables for the final report
and submission package.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
DELIVERABLES_DIR = ROOT / "deliverables"
OUT_DIR = DELIVERABLES_DIR / "industrial_ising_artifacts"
PACKAGE_ARTIFACTS_DIR = DELIVERABLES_DIR / "final_solution_package" / "artifacts"
FINAL_PACKAGE_ZIP = DELIVERABLES_DIR / "TAOLIN_DC_SIM_final_solution_package.zip"

FINAL_RESULT = REPORTS_DIR / "qubo_solver_results_full_rec_cvar0p10_cut3_truepolish5_ms_local.json"
FINAL_RISK_CSV = REPORTS_DIR / "qubo_risk_accepted_actions_full_rec_cvar0p10_cut3_truepolish5_ms_local.csv"
STRUCTURAL_RESULT = REPORTS_DIR / "qubo_solver_results_full_d120_do25_cloudpolish.json"
TRUE_RECOURSE_CLOUD_RESULT = REPORTS_DIR / "qubo_solver_results_full_rec_cvar0p10_cut3_truepolish3_cloud.json"


FAMILY_ORDER = [
    "base_lns",
    "aligned_recourse_lns",
    "tail_cvar_lns",
    "cloud_sparse_tail_lns",
    "dummy_rescue_lns",
    "peak_shave_lns",
    "renewable_absorb_lns",
]


def load_solver_module() -> Any:
    solver_path = ROOT / "src" / "07_solve_isling" / "solve_qubo_platform.py"
    spec = importlib.util.spec_from_file_location("solve_qubo_platform", solver_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load solver module from {solver_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOLVER = load_solver_module()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_matches_instance(result: dict[str, Any], total_jobs: int, total_spins: int) -> bool:
    return (
        int(result.get("selected_job_count", -1)) == int(total_jobs)
        and int(result.get("selected_action_count", -1)) == int(total_spins)
    )


def ensure_dirs() -> None:
    for subdir in [
        OUT_DIR / "qubo_matrices",
        OUT_DIR / "solver_runs",
        OUT_DIR / "solver_runs" / "candidate_solutions",
        OUT_DIR / "validation",
        OUT_DIR / "tables",
    ]:
        subdir.mkdir(parents=True, exist_ok=True)


def final_solution_vector(actions: pd.DataFrame, final_solution: pd.DataFrame) -> np.ndarray:
    selected_ids = set(final_solution["action_id"].astype(str).tolist())
    return actions["action_id"].astype(str).isin(selected_ids).astype(int).to_numpy()


def subset_incumbent_vector(actions: pd.DataFrame, final_solution: pd.DataFrame) -> np.ndarray:
    selected_by_job = dict(
        zip(final_solution["job_id"].astype(str), final_solution["action_id"].astype(str), strict=False)
    )
    x = np.zeros(len(actions), dtype=int)
    for job_id, group in actions.groupby("job_id", sort=False):
        action_id = selected_by_job.get(str(job_id))
        if action_id is None:
            positions = [actions.index.get_loc(idx) for idx in group.index]
            x[positions[0]] = 1
            continue
        mask = group["action_id"].astype(str) == action_id
        if bool(mask.any()):
            pos = actions.index.get_loc(group.loc[mask].index[0])
        else:
            pos = actions.index.get_loc(group.index[0])
        x[pos] = 1
    return x


def action_power_matrix(actions: pd.DataFrame) -> np.ndarray:
    return np.array([SOLVER.parse_vector(v) for v in actions["power_vector_mw"]], dtype=np.float64)


def family_scores(
    actions: pd.DataFrame,
    final_solution: pd.DataFrame,
    site_ts: pd.DataFrame,
    final_load: np.ndarray,
) -> dict[str, dict[str, float]]:
    risks = SOLVER.slot_risk_weights(site_ts)
    marginal = SOLVER.recourse_marginal_slot_prices(final_load, site_ts, alpha=0.95)
    cvar_price = np.asarray(marginal["cvar_slot_price_usd_mwh"], dtype=float)
    ordered_site = site_ts.sort_values("slot")
    renewable = ordered_site["renewable_mw"].astype(float).to_numpy()
    price = ordered_site["lmp_usd_mwh"].astype(float).to_numpy()
    renewable_norm = renewable / max(float(renewable.max()), 1e-9)
    price_norm = (price.max() - price) / max(float(price.max() - price.min()), 1e-9)
    renewable_absorb_weight = 0.65 * renewable_norm + 0.35 * price_norm
    peak_slots = np.argsort(final_load)[-3:]

    final_by_job = final_solution.set_index(final_solution["job_id"].astype(str))
    scores = {family: {} for family in FAMILY_ORDER}
    for job_id, group in actions.groupby("job_id", sort=True):
        job = str(job_id)
        power = action_power_matrix(group)
        selected = final_by_job.loc[job] if job in final_by_job.index else None
        selected_power = np.zeros(power.shape[1], dtype=float)
        selected_is_dummy = False
        if selected is not None:
            selected_power = np.asarray(SOLVER.parse_vector(selected["power_vector_mw"]), dtype=float)
            selected_is_dummy = bool(selected["is_dummy"])

        has_real = bool((~group["is_dummy"].astype(bool)).any())
        dummy_only = bool(group["is_dummy"].astype(bool).all())
        max_risk_power = float(np.max(power @ (1.0 + risks)))
        tail_score = float(max(float(selected_power @ cvar_price), float(np.max(power @ cvar_price))))
        peak_score = float(max(float(selected_power[peak_slots].sum()), float(np.max(power[:, peak_slots].sum(axis=1)))))
        absorb_score = float(max(0.0, float(np.max(power @ renewable_absorb_weight) - selected_power @ renewable_absorb_weight)))
        dummy_score = 1_000_000.0 if selected_is_dummy or dummy_only else 0.0
        if not has_real:
            dummy_score += 100_000.0
        dummy_score += max(0.0, 10.0 - float(len(group)))

        scores["base_lns"][job] = max_risk_power
        scores["aligned_recourse_lns"][job] = max(tail_score, peak_score) + max_risk_power * 1e-2
        scores["tail_cvar_lns"][job] = tail_score
        scores["cloud_sparse_tail_lns"][job] = 0.70 * tail_score + 0.25 * peak_score + 0.05 * max_risk_power
        scores["dummy_rescue_lns"][job] = dummy_score + max_risk_power * 1e-3
        scores["peak_shave_lns"][job] = peak_score
        scores["renewable_absorb_lns"][job] = absorb_score
    return scores


def select_bounded_jobs(
    actions: pd.DataFrame,
    scores: dict[str, float],
    target_spins: int,
    max_spins: int,
    must_include: set[str] | None = None,
) -> list[str]:
    must_include = must_include or set()
    group_sizes = {str(job): len(group) for job, group in actions.groupby("job_id", sort=True)}
    ordered = sorted(group_sizes, key=lambda job: (job not in must_include, -scores.get(job, 0.0), job))
    selected: list[str] = []
    total = 0
    for job in ordered:
        size = int(group_sizes[job])
        if selected and total + size > max_spins:
            continue
        selected.append(job)
        total += size
        if total >= target_spins and must_include.issubset(set(selected)):
            break
    return sorted(selected)


def family_spin_limits(family: str, target_spins: int, max_spins: int) -> tuple[int, int]:
    if family == "cloud_sparse_tail_lns":
        return min(target_spins, 520), min(max_spins, 720)
    if family == "aligned_recourse_lns":
        return min(target_spins, 700), min(max_spins, 960)
    return target_spins, max_spins


def family_qubo_params(family: str) -> dict[str, float]:
    params = {
        "one_hot_penalty": 80.0,
        "peak_weight": 0.002,
        "risk_weight": 0.10,
        "recourse_weight": 0.0,
        "recourse_cvar_weight": 0.10,
        "dummy_cost": 120.0,
        "dummy_only_cost": 25.0,
        "cost_cap": 20.0,
    }
    if family == "aligned_recourse_lns":
        params.update({"peak_weight": 0.001, "risk_weight": 0.02, "recourse_weight": 0.0, "recourse_cvar_weight": 0.10, "dummy_cost": 160.0, "cost_cap": 16.0})
    elif family == "tail_cvar_lns":
        params.update({"peak_weight": 0.001, "risk_weight": 0.08, "recourse_cvar_weight": 0.40})
    elif family == "cloud_sparse_tail_lns":
        params.update({"one_hot_penalty": 52.0, "peak_weight": 0.001, "risk_weight": 0.02, "recourse_weight": 0.0, "recourse_cvar_weight": 0.10, "dummy_cost": 150.0, "cost_cap": 14.0})
    elif family == "dummy_rescue_lns":
        params.update({"dummy_cost": 180.0, "dummy_only_cost": 25.0, "recourse_cvar_weight": 0.15})
    elif family == "peak_shave_lns":
        params.update({"peak_weight": 0.020, "risk_weight": 0.08, "recourse_cvar_weight": 0.10})
    elif family == "renewable_absorb_lns":
        params.update({"peak_weight": 0.001, "risk_weight": 0.04, "recourse_weight": 0.05, "recourse_cvar_weight": 0.08})
    return params


def apply_family_adjustment(
    family: str,
    q: np.ndarray,
    actions: pd.DataFrame,
    site_ts: pd.DataFrame,
) -> dict[str, Any]:
    if family != "renewable_absorb_lns":
        return {"family_adjustment": "none"}
    ordered_site = site_ts.sort_values("slot")
    renewable = ordered_site["renewable_mw"].astype(float).to_numpy()
    price = ordered_site["lmp_usd_mwh"].astype(float).to_numpy()
    renewable_norm = renewable / max(float(renewable.max()), 1e-9)
    price_norm = (price.max() - price) / max(float(price.max() - price.min()), 1e-9)
    reward_weight = 0.65 * renewable_norm + 0.35 * price_norm
    power = action_power_matrix(actions)
    reward = power @ reward_weight
    positive = reward[reward > 1e-12]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, 1e-12)
    q[np.diag_indices_from(q)] -= 0.08 * reward / scale
    return {
        "family_adjustment": "renewable_absorb_linear_reward",
        "reward_scale": scale,
        "reward_min": float(reward.min()) if reward.size else 0.0,
        "reward_max": float(reward.max()) if reward.size else 0.0,
    }


def qubo_for_family(
    family: str,
    actions: pd.DataFrame,
    site_ts: pd.DataFrame,
    final_load: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    params = family_qubo_params(family)
    q, meta = SOLVER.build_qubo(
        actions,
        site_ts,
        reference_load_mw=final_load,
        recourse_alpha=0.95,
        recourse_finite_delta_mw=1e-6,
        **params,
    )
    meta.update({"family": family, "params": params})
    meta.update(apply_family_adjustment(family, q, actions, site_ts))
    meta.update(
        {
            "q_min": float(np.min(q)),
            "q_max": float(np.max(q)),
            "q_abs_max": float(np.max(np.abs(q))),
            "q_nonzero": int(np.count_nonzero(np.abs(q) > 1e-12)),
        }
    )
    return q, meta


def context_from_full_solution(
    full_actions: pd.DataFrame,
    family_actions: pd.DataFrame,
    family_incumbent_x: np.ndarray,
    full_final_x: np.ndarray,
    baseline_x: np.ndarray,
    site_ts: pd.DataFrame,
) -> dict[str, Any]:
    _, full_final_load, _ = SOLVER.selected_vectors(full_actions, full_final_x, len(site_ts))
    _, baseline_load, _ = SOLVER.selected_vectors(full_actions, baseline_x, len(site_ts))
    selected_family, family_load, _ = SOLVER.selected_vectors(family_actions, family_incumbent_x, len(site_ts))
    selected_full = full_actions.loc[full_final_x.astype(bool)].copy()
    final_cost = float(selected_full["action_cost"].astype(float).sum())
    final_dummy = int(selected_full["is_dummy"].astype(bool).sum())
    return {
        "site_ts": site_ts,
        "baseline_load": baseline_load,
        "full_final_load": full_final_load,
        "outside_load": full_final_load - family_load,
        "outside_cost": float(final_cost - selected_family["action_cost"].astype(float).sum()),
        "outside_dummy": int(final_dummy - selected_family["is_dummy"].astype(bool).sum()),
        "final_cost_limit": final_cost + 1.0,
        "final_dummy_limit": final_dummy,
        "full_jobs": int(full_actions["job_id"].nunique()),
    }


def contextual_metrics(
    actions: pd.DataFrame,
    q: np.ndarray,
    x: np.ndarray,
    context: dict[str, Any],
) -> dict[str, Any]:
    selected, load, gpu_load = SOLVER.selected_vectors(actions, x, len(context["site_ts"]))
    full_load = context["outside_load"] + load
    ordered_site = context["site_ts"].sort_values("slot")
    renewable = ordered_site["renewable_mw"].astype(float).to_numpy()
    carbon = ordered_site["carbon_kgco2_mwh"].astype(float).to_numpy()
    recourse = SOLVER.evaluate_recourse(full_load, context["baseline_load"], context["site_ts"])
    selected_cost = float(selected["action_cost"].astype(float).sum()) if len(selected) else 0.0
    full_cost = float(context["outside_cost"] + selected_cost)
    dummy = int(context["outside_dummy"] + (selected["is_dummy"].astype(bool).sum() if len(selected) else 0))
    imported = np.maximum(full_load - renewable, 0.0)
    return {
        "family_jobs": int(actions["job_id"].nunique()),
        "full_jobs": int(context["full_jobs"]),
        "selected_actions": int(x.sum()),
        "one_hot": SOLVER.one_hot_violations(actions, x),
        "dummy_actions": dummy,
        "qubo_energy": SOLVER.qubo_energy(q, x),
        "true_action_cost_usd": full_cost,
        "peak_it_mw": float(full_load.max()) if full_load.size else 0.0,
        "peak_gpu_req": float(gpu_load.max()) if gpu_load.size else 0.0,
        "grid_import_peak_mw": float(imported.max()) if imported.size else 0.0,
        "grid_import_mwh_proxy": float(imported.sum()),
        "carbon_kgco2": float(imported @ carbon),
        "recourse": recourse,
        "slot_load_mw": [float(v) for v in full_load.tolist()],
    }


def true_objective(metrics: dict[str, Any]) -> float:
    recourse = metrics["recourse"]
    return float(recourse["expected_recourse_cost_usd"] + recourse["cvar_recourse_cost_usd"])


def guarded_true_objective(metrics: dict[str, Any], context: dict[str, Any]) -> float:
    objective = true_objective(metrics)
    one_hot_total = int(metrics["one_hot"]["total"])
    dummy_over = max(0, int(metrics["dummy_actions"]) - int(context["final_dummy_limit"]))
    cost_over = max(0.0, float(metrics["true_action_cost_usd"]) - float(context["final_cost_limit"]))
    return float(objective + 10_000.0 * one_hot_total + 1_000.0 * dummy_over + 0.01 * cost_over)


def apply_true_objective_alignment(
    family: str,
    q: np.ndarray,
    actions: pd.DataFrame,
    context: dict[str, Any],
    incumbent_x: np.ndarray,
) -> dict[str, Any]:
    weights = {
        "aligned_recourse_lns": 18.0,
        "cloud_sparse_tail_lns": 12.0,
    }
    weight = weights.get(family)
    if weight is None:
        return {"true_objective_alignment": "disabled"}

    base_metrics = contextual_metrics(actions, q, incumbent_x, context)
    base_objective = guarded_true_objective(base_metrics, context)
    deltas = np.zeros(len(actions), dtype=np.float64)
    evaluations = 0

    groups = [
        [actions.index.get_loc(idx) for idx in group.index]
        for _, group in actions.groupby("job_id", sort=False)
    ]
    for positions in groups:
        selected = [pos for pos in positions if incumbent_x[pos] == 1]
        if len(selected) != 1:
            continue
        current_pos = selected[0]
        for candidate_pos in positions:
            if candidate_pos == current_pos:
                continue
            candidate_x = incumbent_x.copy()
            candidate_x[current_pos] = 0
            candidate_x[candidate_pos] = 1
            metrics = contextual_metrics(actions, q, candidate_x, context)
            objective = guarded_true_objective(metrics, context)
            deltas[candidate_pos] = objective - base_objective
            evaluations += 1

    centered = deltas.copy()
    for positions in groups:
        centered[positions] -= float(np.mean(centered[positions]))
    usable = np.abs(centered[np.abs(centered) > 1e-12])
    scale = float(np.median(usable)) if usable.size else 1.0
    scale = max(scale, 1e-12)
    normalized = np.clip(centered / scale, -3.0, 3.0)
    q[np.diag_indices_from(q)] += weight * normalized
    return {
        "true_objective_alignment": "one_swap_exact_recourse_delta",
        "alignment_weight": float(weight),
        "alignment_evaluations": int(evaluations),
        "alignment_base_objective": float(base_objective),
        "alignment_delta_scale": float(scale),
        "alignment_delta_min": float(deltas.min()) if deltas.size else 0.0,
        "alignment_delta_max": float(deltas.max()) if deltas.size else 0.0,
        "alignment_normalized_min": float(normalized.min()) if normalized.size else 0.0,
        "alignment_normalized_max": float(normalized.max()) if normalized.size else 0.0,
    }


def apply_cloud_friendly_scaling(family: str, q: np.ndarray) -> dict[str, Any]:
    if family != "cloud_sparse_tail_lns":
        return {"cloud_friendly_scaling": "disabled"}
    before_abs_max = float(np.max(np.abs(q))) if q.size else 0.0
    before_nonzero = int(np.count_nonzero(np.abs(q) > 1e-12))
    if before_abs_max <= 0.0:
        return {
            "cloud_friendly_scaling": "noop_zero_matrix",
            "cloud_abs_max_before": before_abs_max,
            "cloud_nonzero_before": before_nonzero,
        }

    zero_threshold = max(1e-9, before_abs_max * 2e-4)
    q[np.abs(q) < zero_threshold] = 0.0
    after_sparse_abs_max = float(np.max(np.abs(q))) if q.size else 0.0
    scale = min(1.0, 45.0 / max(after_sparse_abs_max, 1e-12))
    q *= scale
    q[:] = np.round(q, 8)
    return {
        "cloud_friendly_scaling": "sparse_dynamic_range_capped",
        "cloud_abs_max_before": before_abs_max,
        "cloud_nonzero_before": before_nonzero,
        "cloud_zero_threshold": float(zero_threshold),
        "cloud_scale": float(scale),
        "cloud_abs_max_after": float(np.max(np.abs(q))) if q.size else 0.0,
        "cloud_nonzero_after": int(np.count_nonzero(np.abs(q) > 1e-12)),
    }


def apply_submission_cost_guard(
    family: str,
    q: np.ndarray,
    actions: pd.DataFrame,
    incumbent_x: np.ndarray,
) -> dict[str, Any]:
    weights = {
        "aligned_recourse_lns": 24.0,
        "cloud_sparse_tail_lns": 18.0,
    }
    weight = weights.get(family)
    if weight is None:
        return {"submission_cost_guard": "disabled"}

    selected_cost_by_job: dict[str, float] = {}
    for job_id, group in actions.groupby("job_id", sort=False):
        positions = [actions.index.get_loc(idx) for idx in group.index]
        selected = [pos for pos in positions if incumbent_x[pos] == 1]
        if selected:
            selected_cost_by_job[str(job_id)] = float(actions.iloc[selected[0]]["action_cost"])

    cost_delta = np.zeros(len(actions), dtype=np.float64)
    for idx, row in actions.iterrows():
        baseline = selected_cost_by_job.get(str(row["job_id"]), float(row["action_cost"]))
        cost_delta[int(idx)] = max(0.0, float(row["action_cost"]) - baseline)

    positive = cost_delta[cost_delta > 1e-12]
    scale = float(np.percentile(positive, 25)) if positive.size else 1.0
    scale = max(scale, 1e-12)
    normalized = np.clip(cost_delta / scale, 0.0, 6.0)
    q[np.diag_indices_from(q)] += weight * normalized
    return {
        "submission_cost_guard": "positive_action_cost_delta_from_incumbent",
        "submission_cost_guard_weight": float(weight),
        "submission_cost_delta_scale": float(scale),
        "submission_cost_delta_max": float(cost_delta.max()) if cost_delta.size else 0.0,
        "submission_cost_normalized_max": float(normalized.max()) if normalized.size else 0.0,
    }


def refresh_q_stats(meta: dict[str, Any], q: np.ndarray) -> None:
    meta.update(
        {
            "q_min": float(np.min(q)) if q.size else 0.0,
            "q_max": float(np.max(q)) if q.size else 0.0,
            "q_abs_max": float(np.max(np.abs(q))) if q.size else 0.0,
            "q_nonzero": int(np.count_nonzero(np.abs(q) > 1e-12)),
        }
    )


def random_restart(actions: pd.DataFrame, q: np.ndarray, restarts: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = [
        [actions.index.get_loc(idx) for idx in group.index]
        for _, group in actions.groupby("job_id", sort=False)
    ]
    best_x: np.ndarray | None = None
    best_energy = math.inf
    for _ in range(restarts):
        x = np.zeros(len(actions), dtype=int)
        for positions in groups:
            x[int(rng.choice(positions))] = 1
        energy = SOLVER.qubo_energy(q, x)
        if energy < best_energy:
            best_energy = energy
            best_x = x
    if best_x is None:
        raise RuntimeError("Random restart did not generate a solution.")
    return best_x


def contextual_recourse_polish(
    actions: pd.DataFrame,
    q: np.ndarray,
    start_x: np.ndarray,
    context: dict[str, Any],
    max_passes: int,
    max_dummy_actions: int,
    max_action_cost_usd: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = start_x.copy().astype(int)
    groups = [
        [actions.index.get_loc(idx) for idx in group.index]
        for _, group in actions.groupby("job_id", sort=False)
    ]
    current = contextual_metrics(actions, q, x, context)
    current_objective = true_objective(current)
    pass_rows: list[dict[str, Any]] = []
    total_improvements = 0
    for pass_idx in range(max_passes):
        improvements = 0
        for positions in groups:
            selected = [pos for pos in positions if x[pos] == 1]
            if len(selected) != 1:
                continue
            current_pos = selected[0]
            best_pos = current_pos
            best_metrics = current
            best_objective = current_objective
            for candidate in positions:
                if candidate == current_pos:
                    continue
                candidate_x = x.copy()
                candidate_x[current_pos] = 0
                candidate_x[candidate] = 1
                metrics = contextual_metrics(actions, q, candidate_x, context)
                if metrics["dummy_actions"] > max_dummy_actions:
                    continue
                if metrics["true_action_cost_usd"] > max_action_cost_usd + 1e-9:
                    continue
                objective = true_objective(metrics)
                if objective < best_objective - 1e-6:
                    best_pos = candidate
                    best_metrics = metrics
                    best_objective = objective
            if best_pos != current_pos:
                x[current_pos] = 0
                x[best_pos] = 1
                current = best_metrics
                current_objective = best_objective
                improvements += 1
                total_improvements += 1
        pass_rows.append(
            {
                "pass": pass_idx + 1,
                "improvements": improvements,
                "objective": current_objective,
                "expected_recourse_cost_usd": current["recourse"]["expected_recourse_cost_usd"],
                "cvar95_recourse_cost_usd": current["recourse"]["cvar_recourse_cost_usd"],
            }
        )
        if improvements == 0:
            break
    return x, {"passes": pass_rows, "total_improvements": total_improvements}


def solver_x(
    oracle: str,
    family: str,
    q: np.ndarray,
    actions: pd.DataFrame,
    incumbent_x: np.ndarray,
    seed: int,
    use_credit: bool,
    use_coupon: bool,
    reuse_cloud_task_id: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    if oracle == "incumbent":
        x = incumbent_x.copy()
        raw_result = {"source": "restricted_final_incumbent"}
    elif oracle == "greedy_one_hot":
        x, raw_result = SOLVER.greedy_one_hot_search(actions, q, incumbent_x, max_passes=5)
    elif oracle == "random_restart_20":
        x = random_restart(actions, q, restarts=20, seed=seed)
        raw_result = {"restarts": 20}
    elif oracle.startswith("local_"):
        gear = oracle.replace("local_", "")
        result = SOLVER.solve_with_plectrum(
            q,
            solver_name="local",
            gear_name=gear,
            task_name=f"{family}_{oracle}",
            shot_count=100,
            use_credit=False,
            use_coupon=False,
        )
        x = result["x"]
        raw_result = result["raw_result"]
    elif oracle == "cloud_balanced":
        if reuse_cloud_task_id:
            api_key = os.getenv("PLECTRUM_API_KEY")
            if not api_key:
                raise RuntimeError("PLECTRUM_API_KEY is required to retrieve a cloud task.")
            plectrum = SOLVER.import_plectrum()
            cloud_solver = plectrum["CloudSolver"](
                api_key=api_key,
                gear=plectrum["GEAR_BALANCED"],
            )
            task = cloud_solver.get_task(str(reuse_cloud_task_id))
            task_result = task.get("result") or {}
            spin_config = task_result.get("spin_config") or []
            x = np.asarray(spin_config, dtype=int)
            if len(x) != len(q):
                raise RuntimeError(
                    f"Retrieved cloud task has {len(x)} spins; expected {len(q)} for {family}."
                )
            if set(np.unique(x).tolist()).issubset({-1, 1}):
                x = ((x + 1) // 2).astype(int)
            if not set(np.unique(x).tolist()).issubset({0, 1}):
                raise RuntimeError("Retrieved cloud task contains non-binary spin values.")
            reported_energy = float(task_result["energy"])
            recomputed_energy = float(SOLVER.qubo_energy(q, x))
            if not math.isclose(reported_energy, recomputed_energy, rel_tol=0.0, abs_tol=1e-6):
                raise RuntimeError(
                    "Retrieved cloud task does not match the reconstructed QUBO: "
                    f"reported_energy={reported_energy:.12f}, "
                    f"recomputed_energy={recomputed_energy:.12f}."
                )
            result = {
                "x": x,
                "raw_result": {
                    "task_id": str(task.get("id", reuse_cloud_task_id)),
                    "result": {
                        "energy": reported_energy,
                        "time": task.get("durations"),
                        "ok": int(task.get("status", -1)) == 2,
                        "spin_config": x.astype(int).tolist(),
                    },
                },
            }
        else:
            result = SOLVER.solve_with_plectrum(
                q,
                solver_name="cloud",
                gear_name="balanced",
                task_name=f"{family}_cloud_balanced",
                shot_count=100,
                use_credit=use_credit,
                use_coupon=use_coupon,
            )
        x = result["x"]
        raw_result = {
            "task_id": result["raw_result"].get("task_id"),
            "result": {
                "energy": result["raw_result"].get("result", {}).get("energy"),
                "time": result["raw_result"].get("result", {}).get("time"),
                "ok": result["raw_result"].get("result", {}).get("ok"),
                "spin_config_length": len(result["raw_result"].get("result", {}).get("spin_config", [])),
            },
        }
    else:
        raise ValueError(f"Unsupported oracle: {oracle}")
    return x.astype(int), {"raw_result": raw_result, "solve_seconds": time.perf_counter() - started}


def persist_candidate_solution(
    family: str,
    oracle: str,
    full_actions: pd.DataFrame,
    family_actions: pd.DataFrame,
    full_final_x: np.ndarray,
    polished_x: np.ndarray,
    row: dict[str, Any],
) -> None:
    """Persist the full assignment behind an evidence row for replay/promotion."""
    family_mask = full_actions["job_id"].astype(str).isin(
        set(family_actions["job_id"].astype(str))
    ).to_numpy()
    if int(family_mask.sum()) != len(polished_x):
        raise RuntimeError(
            f"Candidate shape mismatch for {family}/{oracle}: "
            f"mask={int(family_mask.sum())}, vector={len(polished_x)}."
        )
    full_candidate_x = full_final_x.copy().astype(int)
    full_candidate_x[family_mask] = polished_x.astype(int)
    selected = full_actions.loc[full_candidate_x.astype(bool)].copy()
    if len(selected) != int(full_actions["job_id"].nunique()):
        raise RuntimeError(
            f"Candidate {family}/{oracle} selects {len(selected)} rows instead of one per job."
        )
    selected["selection_source"] = f"{family}:{oracle}:contextual_recourse_polish"
    stem = f"{family}__{oracle}"
    out_dir = OUT_DIR / "solver_runs" / "candidate_solutions"
    selected.to_csv(out_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "family": family,
        "oracle": oracle,
        "status": row.get("status", "success"),
        "jobs": int(selected["job_id"].nunique()),
        "selected_actions": int(len(selected)),
        "accepted_under_submission_guard": int(row["accepted_under_submission_guard"]),
        "true_action_cost_usd": float(row["polished_true_action_cost_usd"]),
        "expected_recourse_cost_usd": float(row["polished_expected_recourse"]),
        "cvar95_recourse_cost_usd": float(row["polished_cvar95"]),
        "true_objective": float(row["polished_true_objective"]),
        "source_result": json.loads(str(row["raw_result"])),
    }
    (out_dir / f"{stem}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def oracle_row(
    family: str,
    oracle: str,
    actions: pd.DataFrame,
    q: np.ndarray,
    context: dict[str, Any],
    raw_x: np.ndarray,
    raw_info: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    raw_metrics = contextual_metrics(actions, q, raw_x, context)
    repaired_x, repair = SOLVER.repair_one_hot(actions, q, raw_x)
    repaired_metrics = contextual_metrics(actions, q, repaired_x, context)
    polished_x, polish_report = contextual_recourse_polish(
        actions,
        q,
        repaired_x,
        context,
        max_passes=2,
        max_dummy_actions=int(context["final_dummy_limit"]),
        max_action_cost_usd=float(context["final_cost_limit"]),
    )
    polished_metrics = contextual_metrics(actions, q, polished_x, context)
    accepted = int(
        polished_metrics["one_hot"]["total"] == 0
        and polished_metrics["dummy_actions"] <= int(context["final_dummy_limit"])
        and polished_metrics["true_action_cost_usd"] <= float(context["final_cost_limit"]) + 1e-9
    )
    row = {
        "family": family,
        "oracle": oracle,
        "jobs": int(actions["job_id"].nunique()),
        "spins": int(len(actions)),
        "raw_qubo_energy": raw_metrics["qubo_energy"],
        "raw_feasible": int(raw_metrics["one_hot"]["total"] == 0),
        "raw_one_hot_violations": raw_metrics["one_hot"]["total"],
        "repair_count": int(repair["repair_count"]),
        "repaired_qubo_energy": repaired_metrics["qubo_energy"],
        "repaired_true_action_cost_usd": repaired_metrics["true_action_cost_usd"],
        "repaired_expected_recourse": repaired_metrics["recourse"]["expected_recourse_cost_usd"],
        "repaired_cvar95": repaired_metrics["recourse"]["cvar_recourse_cost_usd"],
        "repaired_dummy": repaired_metrics["dummy_actions"],
        "repaired_true_objective": true_objective(repaired_metrics),
        "repaired_guarded_objective": guarded_true_objective(repaired_metrics, context),
        "polished_true_action_cost_usd": polished_metrics["true_action_cost_usd"],
        "polished_expected_recourse": polished_metrics["recourse"]["expected_recourse_cost_usd"],
        "polished_cvar95": polished_metrics["recourse"]["cvar_recourse_cost_usd"],
        "polished_dummy": polished_metrics["dummy_actions"],
        "polished_true_objective": true_objective(polished_metrics),
        "polished_guarded_objective": guarded_true_objective(polished_metrics, context),
        "polish_improvements": int(polish_report["total_improvements"]),
        "accepted_under_submission_guard": accepted,
        "end_to_end_seconds": float(raw_info["solve_seconds"]),
        "raw_result": json.dumps(raw_info["raw_result"], ensure_ascii=False, sort_keys=True),
    }
    return row, polished_x


def industrial_metrics(name: str, metrics: dict[str, Any], total_jobs: int) -> dict[str, Any]:
    recourse = metrics["recourse"]
    scenarios = recourse["scenarios"]
    costs = [float(row["total_recourse_cost_usd"]) for row in scenarios]
    probs = [float(row["probability"]) for row in scenarios]
    cvar_weights = SOLVER.cvar_scenario_weights(costs, probs, alpha=0.95)
    flex_target = sum(float(row["probability"]) * float(row["flex_target_mwh"]) for row in scenarios)
    flex_short = sum(float(row["probability"]) * float(row["flex_shortfall_mwh"]) for row in scenarios)
    tail_short = sum(float(weight) * float(row["flex_shortfall_mwh"]) for weight, row in zip(cvar_weights, scenarios, strict=False))
    nominal = next((row for row in scenarios if row["name"] == "nominal"), scenarios[0])
    mode_counts = metrics.get("mode_counts", {})
    delay_count = int(mode_counts.get("delay", 0))
    dummy = int(metrics["dummy_actions"])
    return {
        "solution": name,
        "FlexDeliveryRate": 1.0 - flex_short / max(flex_target, 1e-12),
        "TailShortfall95_mwh": tail_short,
        "SLAViolationRate_delay_or_dummy": (delay_count + dummy) / max(total_jobs, 1),
        "DummyExplainability_dummy_only_share": 1.0 if dummy else 0.0,
        "GridImportPeak_mw": metrics.get("grid_import_peak_mw", metrics.get("peak_it_mw", 0.0)),
        "RenewableCurtailment_mwh": float(nominal["curtailment_mwh"]),
        "ReboundLoad_mwh_last6": float(sum(metrics.get("slot_load_mw", [])[-6:])),
        "expected_recourse_cost_usd": recourse["expected_recourse_cost_usd"],
        "cvar95_recourse_cost_usd": recourse["cvar_recourse_cost_usd"],
        "dominant_tail_scenario": recourse["worst_scenario"],
    }


def write_qubo_artifacts(family_data: dict[str, dict[str, Any]]) -> None:
    rows = []
    for family, data in family_data.items():
        q_path = OUT_DIR / "qubo_matrices" / f"Q_{family}.npy"
        np.save(q_path, data["q"])
        actions = data["actions"].reset_index(drop=True)
        index = actions[
            ["job_id", "action_id", "site", "start_slot", "end_slot", "mode", "is_dummy"]
        ].copy()
        index.insert(0, "family", family)
        index.insert(1, "var_idx", np.arange(len(index), dtype=int))
        index.to_csv(OUT_DIR / "qubo_matrices" / f"index_{family}.csv", index=False, encoding="utf-8-sig")
        rows.extend(index.to_dict("records"))
        meta_path = OUT_DIR / "qubo_matrices" / f"meta_{family}.json"
        meta_path.write_text(json.dumps(data["meta"], ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT_DIR / "qubo_matrices" / "qubo_index.csv", index=False, encoding="utf-8-sig")


def package_scan(package_dir: Path) -> tuple[dict[str, int], list[dict[str, str]]]:
    patterns = {
        "absolute_path": re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\"),
        "token_or_secret": re.compile(
            r"(08c81dd3ab|(?:API[_-]?KEY|TOKEN|SECRET)\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{12,})",
            re.IGNORECASE,
        ),
        "cache_dependency": re.compile(r"(^|/|\\)(__pycache__|\.cache|site-packages|\.venv|venv)(/|\\|$)", re.IGNORECASE),
    }
    counts = {key: 0 for key in patterns}
    hits: list[dict[str, str]] = []
    if not package_dir.exists():
        return counts, hits
    for path in package_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(package_dir).as_posix()
        text = path.read_bytes().decode("utf-8", errors="ignore")
        for key, pattern in patterns.items():
            if pattern.search(rel) or pattern.search(text):
                counts[key] += 1
                hits.append({"file": rel, "pattern": key})
    return counts, hits


def clean_room_validation() -> dict[str, Any]:
    package_dir = DELIVERABLES_DIR / "final_solution_package"
    validation = read_json(package_dir / "VALIDATION.json")
    recomputed = read_json(package_dir / "VALIDATION_RECOMPUTED.json")
    bad_member = None
    if FINAL_PACKAGE_ZIP.exists():
        with zipfile.ZipFile(FINAL_PACKAGE_ZIP, "r") as zf:
            bad_member = zf.testzip()
    scan_counts, hits = package_scan(package_dir)
    payload = {
        "zip_bad_member": bad_member,
        "validation_status": validation["status"],
        "recomputed_status": recomputed["status"],
        "jobs": validation["unique_jobs"],
        "one_hot_violations": validation["one_hot_violations"],
        "dummy_actions": validation["dummy_actions"],
        "expected_recourse_cost_usd": recomputed["expected_recourse_cost_usd"],
        "cvar95_recourse_cost_usd": recomputed["cvar95_recourse_cost_usd"],
        "scan_counts": scan_counts,
        "scan_hits": hits,
    }
    return payload


def penalty_sweep(
    family_actions: pd.DataFrame,
    incumbent_x: np.ndarray,
    site_ts: pd.DataFrame,
    final_load: np.ndarray,
    context: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for factor in [0.5, 1.0, 2.0, 4.0]:
        params = family_qubo_params("base_lns")
        params["one_hot_penalty"] = 80.0 * factor
        q, meta = SOLVER.build_qubo(
            family_actions,
            site_ts,
            reference_load_mw=final_load,
            recourse_alpha=0.95,
            recourse_finite_delta_mw=1e-6,
            **params,
        )
        x, report = SOLVER.greedy_one_hot_search(family_actions, q, incumbent_x, max_passes=5)
        metrics = contextual_metrics(family_actions, q, x, context)
        rows.append(
            {
                "one_hot_penalty_factor": factor,
                "one_hot_penalty": params["one_hot_penalty"],
                "raw_feasible": int(metrics["one_hot"]["total"] == 0),
                "repair_cost_proxy": 0,
                "true_objective_expected_plus_cvar": true_objective(metrics),
                "CVaR95": metrics["recourse"]["cvar_recourse_cost_usd"],
                "dummy_count": metrics["dummy_actions"],
                "qubo_energy": metrics["qubo_energy"],
                "coefficient_abs_range": meta["q_abs_max"],
                "greedy_improvements": int(report["total_improvements"]),
            }
        )
    return pd.DataFrame(rows)


def correlation_table(oracle_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in oracle_df.groupby("family", sort=False):
        guarded_col = "polished_guarded_objective" if "polished_guarded_objective" in group.columns else "polished_true_objective"
        if len(group) >= 2 and group["repaired_qubo_energy"].nunique() > 1 and group[guarded_col].nunique() > 1:
            corr = float(np.corrcoef(group["repaired_qubo_energy"], group[guarded_col])[0, 1])
            rank_corr = float(group["repaired_qubo_energy"].rank(method="average").corr(group[guarded_col].rank(method="average")))
        else:
            corr = float("nan")
            rank_corr = float("nan")
        if len(group) >= 2 and group["repaired_qubo_energy"].nunique() > 1 and group["polished_true_objective"].nunique() > 1:
            recourse_corr = float(np.corrcoef(group["repaired_qubo_energy"], group["polished_true_objective"])[0, 1])
        else:
            recourse_corr = float("nan")
        best_qubo_oracle = str(group.loc[group["repaired_qubo_energy"].idxmin(), "oracle"])
        best_true_oracle = str(group.loc[group[guarded_col].idxmin(), "oracle"])
        best_recourse_oracle = str(group.loc[group["polished_true_objective"].idxmin(), "oracle"])
        rows.append(
            {
                "family": family,
                "samples": len(group),
                "qubo_energy_vs_true_objective_corr": corr,
                "qubo_energy_vs_true_objective_rank_corr": rank_corr,
                "qubo_energy_vs_recourse_only_corr": recourse_corr,
                "best_qubo_oracle": best_qubo_oracle,
                "best_true_objective_oracle": best_true_oracle,
                "best_recourse_only_oracle": best_recourse_oracle,
                "best_oracle_match": int(best_qubo_oracle == best_true_oracle),
            }
        )
    return pd.DataFrame(rows)


def cloud_contribution_table(oracle_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "family",
        "cloud_objective",
        "best_non_cloud_objective",
        "cloud_contribution_positive_if_gt0",
        "cloud_cvar95",
        "best_non_cloud_cvar95",
        "best_non_cloud_oracle",
    ]
    rows = []
    for family, group in oracle_df.groupby("family", sort=False):
        accepted = group.loc[group["accepted_under_submission_guard"].astype(int) == 1]
        cloud = accepted.loc[accepted["oracle"] == "cloud_balanced"]
        non_cloud = accepted.loc[accepted["oracle"] != "cloud_balanced"]
        if cloud.empty or non_cloud.empty:
            continue
        best_cloud = cloud.loc[cloud["polished_true_objective"].idxmin()]
        best_local = non_cloud.loc[non_cloud["polished_true_objective"].idxmin()]
        rows.append(
            {
                "family": family,
                "cloud_objective": float(best_cloud["polished_true_objective"]),
                "best_non_cloud_objective": float(best_local["polished_true_objective"]),
                "cloud_contribution_positive_if_gt0": float(best_local["polished_true_objective"] - best_cloud["polished_true_objective"]),
                "cloud_cvar95": float(best_cloud["polished_cvar95"]),
                "best_non_cloud_cvar95": float(best_local["polished_cvar95"]),
                "best_non_cloud_oracle": str(best_local["oracle"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def copy_artifacts_to_package() -> None:
    if PACKAGE_ARTIFACTS_DIR.exists():
        shutil.rmtree(PACKAGE_ARTIFACTS_DIR)
    shutil.copytree(OUT_DIR, PACKAGE_ARTIFACTS_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-spins",
        type=int,
        default=700,
        help="Nominal bounded-neighborhood size; 700 reproduces the submitted evidence portfolio.",
    )
    parser.add_argument("--max-spins", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--use-cloud", action="store_true")
    parser.add_argument("--cloud-families", default=",".join(FAMILY_ORDER))
    parser.add_argument(
        "--reuse-cloud-task-id",
        default=None,
        help=(
            "Retrieve and replay one existing CloudSolver task instead of creating a new task. "
            "Requires exactly one --cloud-families entry and PLECTRUM_API_KEY."
        ),
    )
    parser.add_argument("--use-credit", action="store_true")
    parser.add_argument("--use-coupon", action="store_true")
    parser.add_argument("--include-local-solver", action="store_true")
    parser.add_argument("--local-solver-families", default="base_lns")
    parser.add_argument("--copy-to-package", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    jobs, actions, site_ts = SOLVER.load_tables(ROOT)
    actions = actions.copy().reset_index(drop=True)
    actions["job_id"] = actions["job_id"].astype(str)
    final_solution = pd.read_csv(FINAL_RISK_CSV)
    final_solution["job_id"] = final_solution["job_id"].astype(str)
    full_final_x = final_solution_vector(actions, final_solution)
    baseline_x = SOLVER.baseline_solution(actions)
    _, final_load, _ = SOLVER.selected_vectors(actions, full_final_x, len(site_ts))
    final_context = context_from_full_solution(actions, actions, full_final_x, full_final_x, baseline_x, site_ts)
    final_result = read_json(FINAL_RESULT)

    scores = family_scores(actions, final_solution, site_ts, final_load)
    dummy_jobs = set(final_solution.loc[final_solution["is_dummy"].astype(bool), "job_id"].astype(str).tolist())
    cloud_families = {token.strip() for token in args.cloud_families.split(",") if token.strip()}
    if args.reuse_cloud_task_id and (not args.use_cloud or len(cloud_families) != 1):
        raise ValueError(
            "--reuse-cloud-task-id requires --use-cloud and exactly one --cloud-families entry."
        )
    local_solver_families = {token.strip() for token in args.local_solver_families.split(",") if token.strip()}
    family_data: dict[str, dict[str, Any]] = {}
    oracle_rows: list[dict[str, Any]] = []

    for family in FAMILY_ORDER:
        must_include = dummy_jobs if family == "dummy_rescue_lns" else set()
        family_target_spins, family_max_spins = family_spin_limits(family, args.target_spins, args.max_spins)
        selected_jobs = select_bounded_jobs(
            actions,
            scores[family],
            target_spins=family_target_spins,
            max_spins=family_max_spins,
            must_include=must_include,
        )
        family_actions = actions.loc[actions["job_id"].isin(selected_jobs)].copy().reset_index(drop=True)
        incumbent_x = subset_incumbent_vector(family_actions, final_solution)
        q, meta = qubo_for_family(family, family_actions, site_ts, final_load)
        context = context_from_full_solution(actions, family_actions, incumbent_x, full_final_x, baseline_x, site_ts)
        meta.update(
            {
                "target_spins_requested": int(args.target_spins),
                "max_spins_requested": int(args.max_spins),
                "family_target_spins": int(family_target_spins),
                "family_max_spins": int(family_max_spins),
            }
        )
        meta.update(apply_true_objective_alignment(family, q, family_actions, context, incumbent_x))
        meta.update(apply_submission_cost_guard(family, q, family_actions, incumbent_x))
        meta.update(apply_cloud_friendly_scaling(family, q))
        refresh_q_stats(meta, q)
        family_data[family] = {
            "actions": family_actions,
            "q": q,
            "meta": meta,
            "incumbent_x": incumbent_x,
            "context": context,
        }

        oracles = ["incumbent", "greedy_one_hot", "random_restart_20"]
        if args.include_local_solver and family in local_solver_families:
            oracles.extend(["local_fast", "local_balanced"])
        if args.use_cloud and family in cloud_families:
            oracles.append("cloud_balanced")
        for oracle in oracles:
            try:
                raw_x, raw_info = solver_x(
                    oracle,
                    family,
                    q,
                    family_actions,
                    incumbent_x,
                    seed=args.seed,
                    use_credit=args.use_credit,
                    use_coupon=args.use_coupon,
                    reuse_cloud_task_id=(
                        args.reuse_cloud_task_id if oracle == "cloud_balanced" else None
                    ),
                )
                row, polished_x = oracle_row(
                    family, oracle, family_actions, q, context, raw_x, raw_info
                )
                row["status"] = "success"
                persist_candidate_solution(
                    family=family,
                    oracle=oracle,
                    full_actions=actions,
                    family_actions=family_actions,
                    full_final_x=full_final_x,
                    polished_x=polished_x,
                    row=row,
                )
            except Exception as exc:  # Keep evidence generation robust.
                row = {
                    "family": family,
                    "oracle": oracle,
                    "jobs": int(family_actions["job_id"].nunique()),
                    "spins": int(len(family_actions)),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            oracle_rows.append(row)

    write_qubo_artifacts(family_data)
    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(OUT_DIR / "solver_runs" / "oracle_replacement_summary.csv", index=False, encoding="utf-8-sig")
    oracle_df.loc[oracle_df["oracle"].astype(str).str.startswith("cloud")].to_csv(
        OUT_DIR / "solver_runs" / "cloud_raw_runs.csv", index=False, encoding="utf-8-sig"
    )
    oracle_df.loc[~oracle_df["oracle"].astype(str).str.startswith("cloud")].to_csv(
        OUT_DIR / "solver_runs" / "local_solver_runs.csv", index=False, encoding="utf-8-sig"
    )
    correlation = correlation_table(oracle_df.loc[oracle_df["status"] == "success"].copy())
    correlation.to_csv(OUT_DIR / "validation" / "qubo_energy_true_objective_check.csv", index=False, encoding="utf-8-sig")
    cloud_contribution = cloud_contribution_table(oracle_df.loc[oracle_df["status"] == "success"].copy())
    cloud_contribution.to_csv(OUT_DIR / "tables" / "cloud_contribution.csv", index=False, encoding="utf-8-sig")

    base = family_data["base_lns"]
    penalty_sweep(base["actions"], base["incumbent_x"], site_ts, final_load, base["context"]).to_csv(
        OUT_DIR / "tables" / "penalty_sweep.csv", index=False, encoding="utf-8-sig"
    )

    total_jobs = int(actions["job_id"].nunique())
    total_spins = int(len(actions))
    final_metrics = final_result["risk_accepted_solution"]["metrics"]
    final_industrial = industrial_metrics("final_full_solution", final_metrics, total_jobs=total_jobs)
    pd.DataFrame([final_industrial]).to_csv(OUT_DIR / "tables" / "industrial_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "jobs": total_jobs,
        "spins": total_spins,
        "one_hot_violations": 0,
        "dummy_actions": int(final_metrics["dummy_actions"]),
        "expected_recourse_cost_usd": final_metrics["recourse"]["expected_recourse_cost_usd"],
        "cvar95_recourse_cost_usd": final_metrics["recourse"]["cvar_recourse_cost_usd"],
        "dominant_tail_scenario": final_metrics["recourse"]["worst_scenario"],
    }]).to_csv(OUT_DIR / "tables" / "final_metrics.csv", index=False, encoding="utf-8-sig")
    final_solution.loc[final_solution["is_dummy"].astype(bool), [
        "job_id",
        "action_id",
        "mode",
        "is_dummy",
    ]].to_csv(OUT_DIR / "tables" / "dummy_job_explanation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(final_metrics["recourse"]["scenarios"]).to_csv(
        OUT_DIR / "tables" / "tail_scenario_breakdown.csv", index=False, encoding="utf-8-sig"
    )

    ablation_rows = []
    if STRUCTURAL_RESULT.exists():
        structural_result = read_json(STRUCTURAL_RESULT)
        if result_matches_instance(structural_result, total_jobs, total_spins):
            structural = structural_result["risk_accepted_solution"]["metrics"]
            ablation_rows.append({
                "layer": "model_structural_proxy",
                "source": "structural_qubo_proxy",
                "status": "current",
                "expected_recourse_cost_usd": structural["recourse"]["expected_recourse_cost_usd"],
                "cvar95_recourse_cost_usd": structural["recourse"]["cvar_recourse_cost_usd"],
                "dummy_actions": structural["dummy_actions"],
            })
        else:
            ablation_rows.append({
                "layer": "model_structural_proxy",
                "source": STRUCTURAL_RESULT.name,
                "status": "skipped_stale_instance",
                "expected_recourse_cost_usd": "",
                "cvar95_recourse_cost_usd": "",
                "dummy_actions": "",
            })
    if TRUE_RECOURSE_CLOUD_RESULT.exists():
        true_recourse_result = read_json(TRUE_RECOURSE_CLOUD_RESULT)
        if result_matches_instance(true_recourse_result, total_jobs, total_spins):
            true_recourse = true_recourse_result["risk_accepted_solution"]["metrics"]
            ablation_rows.append({
                "layer": "model_true_recourse",
                "source": "cloud_integrated_true_recourse",
                "status": "current",
                "expected_recourse_cost_usd": true_recourse["recourse"]["expected_recourse_cost_usd"],
                "cvar95_recourse_cost_usd": true_recourse["recourse"]["cvar_recourse_cost_usd"],
                "dummy_actions": true_recourse["dummy_actions"],
            })
        else:
            ablation_rows.append({
                "layer": "model_true_recourse",
                "source": TRUE_RECOURSE_CLOUD_RESULT.name,
                "status": "skipped_stale_instance",
                "expected_recourse_cost_usd": "",
                "cvar95_recourse_cost_usd": "",
                "dummy_actions": "",
            })
    ablation_rows.append({
        "layer": "final_local_polish",
        "source": "pass5_true_recourse_polish",
        "status": "current",
        "expected_recourse_cost_usd": final_metrics["recourse"]["expected_recourse_cost_usd"],
        "cvar95_recourse_cost_usd": final_metrics["recourse"]["cvar_recourse_cost_usd"],
        "dummy_actions": final_metrics["dummy_actions"],
    })
    pd.DataFrame(ablation_rows).to_csv(
        OUT_DIR / "tables" / "ablation_model_solver_hardware.csv", index=False, encoding="utf-8-sig"
    )

    clean_room = clean_room_validation()
    (OUT_DIR / "validation" / "clean_room_validation.json").write_text(
        json.dumps(clean_room, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    scan_lines = [
        f"absolute_path_hits={clean_room['scan_counts']['absolute_path']}",
        f"token_or_secret_hits={clean_room['scan_counts']['token_or_secret']}",
        f"cache_dependency_hits={clean_room['scan_counts']['cache_dependency']}",
    ]
    (OUT_DIR / "validation" / "package_secret_scan.txt").write_text("\n".join(scan_lines) + "\n", encoding="utf-8")

    cloud_environment = {
        "use_cloud_requested": bool(args.use_cloud),
        "plectrum_api_key_present": bool(os.getenv("PLECTRUM_API_KEY")),
        "cloud_families_requested": sorted(cloud_families),
        "cloud_rows": int((oracle_df["oracle"] == "cloud_balanced").sum()),
    }
    (OUT_DIR / "validation" / "cloud_availability.json").write_text(
        json.dumps(cloud_environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "families": {
            family: {
                "jobs": int(data["actions"]["job_id"].nunique()),
                "spins": int(len(data["actions"])),
                "q_abs_max": data["meta"]["q_abs_max"],
            }
            for family, data in family_data.items()
        },
        "oracle_rows": int(len(oracle_df)),
        "cloud_rows": int((oracle_df["oracle"] == "cloud_balanced").sum()),
        "failed_rows": int((oracle_df["status"] == "failed").sum()) if "status" in oracle_df else 0,
        "cloud_environment": cloud_environment,
        "output_dir": "deliverables/industrial_ising_artifacts",
    }
    (OUT_DIR / "industrial_ising_evidence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.copy_to_package:
        copy_artifacts_to_package()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
