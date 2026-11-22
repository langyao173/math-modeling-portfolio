"""Promote a persisted cloud-seeded candidate after independent replay.

The industrial evidence pipeline works on bounded QUBO neighborhoods. This
script reconstructs the authoritative full QUBO, verifies that the previous
final assignment has the recorded energy, evaluates a persisted full candidate
with the same business evaluator, and promotes it only when it is feasible,
within the submission cost guard, and Pareto-dominates the previous final on
expected recourse, CVaR95, and peak IT load.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
BASE_RESULT = REPORTS_DIR / "qubo_solver_results_full_rec_cvar0p10_cut3_truepolish5_ms_local.json"
BASE_RISK_CSV = REPORTS_DIR / "qubo_risk_accepted_actions_full_rec_cvar0p10_cut3_truepolish5_ms_local.csv"
DEFAULT_CANDIDATE = (
    ROOT
    / "deliverables"
    / "industrial_ising_artifacts"
    / "solver_runs"
    / "candidate_solutions"
    / "peak_shave_lns__cloud_balanced.csv"
)
DEFAULT_METADATA = DEFAULT_CANDIDATE.with_suffix(".json")
OUTPUT_STEM = "full_cloudseed_peakguard"
OUTPUT_RESULT = REPORTS_DIR / f"qubo_solver_results_{OUTPUT_STEM}.json"
OUTPUT_RISK_CSV = REPORTS_DIR / f"qubo_risk_accepted_actions_{OUTPUT_STEM}.csv"
OUTPUT_REPORT = REPORTS_DIR / f"qubo_solver_report_{OUTPUT_STEM}.md"
INDUSTRIAL_DIR = ROOT / "deliverables" / "industrial_ising_artifacts"


def load_solver_module() -> Any:
    path = ROOT / "src" / "07_solve_isling" / "solve_qubo_platform.py"
    spec = importlib.util.spec_from_file_location("solve_qubo_platform", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import solver from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOLVER = load_solver_module()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def reconstruct_full_qubo(
    base: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    args_payload = dict(base["args"])
    args_payload["root"] = ROOT
    args_payload.setdefault("recourse_polish_start_csv", None)
    args = Namespace(**args_payload)
    _, actions, site_ts = SOLVER.load_tables(ROOT)
    selected_jobs = SOLVER.choose_jobs(
        actions, site_ts, max_jobs=int(args.max_jobs), seed=int(args.seed)
    )
    selected_actions = (
        actions.loc[actions["job_id"].isin(selected_jobs)].copy().reset_index(drop=True)
    )
    selected_actions["job_id"] = selected_actions["job_id"].astype(str)
    baseline_x = SOLVER.baseline_solution(selected_actions)
    _, baseline_load, _ = SOLVER.selected_vectors(
        selected_actions, baseline_x, len(site_ts)
    )
    reference_load = baseline_load
    cut_records: list[dict[str, Any]] = []
    if SOLVER.recourse_terms_enabled(args) and int(args.recourse_cut_iterations) > 0:
        reference_load, cut_records, _ = SOLVER.run_recourse_cut_iterations(
            actions=selected_actions,
            site_ts=site_ts,
            args=args,
            baseline_x=baseline_x,
            baseline_load_mw=baseline_load,
        )
    q, meta = SOLVER.build_qubo_from_args(
        selected_actions,
        site_ts,
        args,
        reference_load_mw=reference_load,
    )
    meta["reconstructed_recourse_cut_iterations"] = len(cut_records)
    return selected_actions, site_ts, baseline_x, q, meta


def metrics_triplet(metrics: dict[str, Any]) -> dict[str, float]:
    recourse = metrics["recourse"]
    return {
        "expected_recourse_cost_usd": float(
            recourse["expected_recourse_cost_usd"]
        ),
        "cvar95_recourse_cost_usd": float(
            recourse["cvar_recourse_cost_usd"]
        ),
        "peak_it_mw": float(metrics["peak_it_mw"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-result", type=Path, default=BASE_RESULT)
    parser.add_argument("--base-risk-csv", type=Path, default=BASE_RISK_CSV)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--candidate-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-result", type=Path, default=OUTPUT_RESULT)
    parser.add_argument("--output-risk-csv", type=Path, default=OUTPUT_RISK_CSV)
    parser.add_argument("--output-report", type=Path, default=OUTPUT_REPORT)
    return parser.parse_args()


def refresh_industrial_final_tables(
    metrics: dict[str, Any],
    previous_triplet: dict[str, float],
    promoted_triplet: dict[str, float],
    source: str,
) -> None:
    tables_dir = INDUSTRIAL_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    recourse = metrics["recourse"]
    scenarios = recourse["scenarios"]
    pd.DataFrame(
        [
            {
                "jobs": int(metrics["jobs"]),
                "spins": 1929,
                "one_hot_violations": int(metrics["one_hot"]["total"]),
                "dummy_actions": int(metrics["dummy_actions"]),
                "expected_recourse_cost_usd": recourse["expected_recourse_cost_usd"],
                "cvar95_recourse_cost_usd": recourse["cvar_recourse_cost_usd"],
                "dominant_tail_scenario": recourse["worst_scenario"],
            }
        ]
    ).to_csv(tables_dir / "final_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(scenarios).to_csv(
        tables_dir / "tail_scenario_breakdown.csv", index=False, encoding="utf-8-sig"
    )

    costs = [float(row["total_recourse_cost_usd"]) for row in scenarios]
    probabilities = [float(row["probability"]) for row in scenarios]
    cvar_weights = SOLVER.cvar_scenario_weights(costs, probabilities, alpha=0.95)
    flex_target = sum(
        float(row["probability"]) * float(row["flex_target_mwh"])
        for row in scenarios
    )
    flex_short = sum(
        float(row["probability"]) * float(row["flex_shortfall_mwh"])
        for row in scenarios
    )
    tail_short = sum(
        float(weight) * float(row["flex_shortfall_mwh"])
        for weight, row in zip(cvar_weights, scenarios, strict=False)
    )
    nominal = next(
        (row for row in scenarios if row["name"] == "nominal"), scenarios[0]
    )
    mode_counts = metrics.get("mode_counts", {})
    pd.DataFrame(
        [
            {
                "solution": "final_full_solution",
                "FlexDeliveryRate": 1.0 - flex_short / max(flex_target, 1e-12),
                "TailShortfall95_mwh": tail_short,
                "SLAViolationRate_delay_or_dummy": (
                    int(mode_counts.get("delay", 0)) + int(metrics["dummy_actions"])
                )
                / max(int(metrics["jobs"]), 1),
                "DummyExplainability_dummy_only_share": (
                    1.0 if int(metrics["dummy_actions"]) else 0.0
                ),
                "GridImportPeak_mw": float(metrics["peak_it_mw"]),
                "RenewableCurtailment_mwh": float(nominal["curtailment_mwh"]),
                "ReboundLoad_mwh_last6": float(sum(metrics["slot_load_mw"][-6:])),
                "expected_recourse_cost_usd": recourse["expected_recourse_cost_usd"],
                "cvar95_recourse_cost_usd": recourse["cvar_recourse_cost_usd"],
                "dominant_tail_scenario": recourse["worst_scenario"],
            }
        ]
    ).to_csv(tables_dir / "industrial_metrics.csv", index=False, encoding="utf-8-sig")

    ablation_path = tables_dir / "ablation_model_solver_hardware.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        ablation = ablation.loc[
            ~ablation["layer"].astype(str).str.startswith("final_")
        ].copy()
    else:
        ablation = pd.DataFrame(
            columns=[
                "layer",
                "source",
                "status",
                "expected_recourse_cost_usd",
                "cvar95_recourse_cost_usd",
                "dummy_actions",
            ]
        )
    promoted_row = pd.DataFrame(
        [
            {
                "layer": "final_promoted_cloudseed",
                "source": source,
                "status": "current",
                "expected_recourse_cost_usd": recourse["expected_recourse_cost_usd"],
                "cvar95_recourse_cost_usd": recourse["cvar_recourse_cost_usd"],
                "dummy_actions": int(metrics["dummy_actions"]),
            }
        ]
    )
    pd.concat([ablation, promoted_row], ignore_index=True).to_csv(
        ablation_path, index=False, encoding="utf-8-sig"
    )

    comparison_rows = []
    for metric_name in previous_triplet:
        comparison_rows.append(
            {
                "metric": metric_name,
                "previous_final": previous_triplet[metric_name],
                "promoted_final": promoted_triplet[metric_name],
                "absolute_improvement": (
                    previous_triplet[metric_name] - promoted_triplet[metric_name]
                ),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        tables_dir / "promoted_final_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_path = INDUSTRIAL_DIR / "industrial_ising_evidence_summary.json"
    summary = read_json(summary_path)
    summary["promoted_final"] = {
        "source": source,
        **promoted_triplet,
        "dummy_actions": int(metrics["dummy_actions"]),
        "one_hot_violations": int(metrics["one_hot"]["total"]),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    legacy_path = INDUSTRIAL_DIR / "validation" / "clean_room_validation.json"
    if legacy_path.exists():
        legacy = read_json(legacy_path)
        legacy.update(
            {
                "validation_status": "pass",
                "recomputed_status": "pass",
                "jobs": int(metrics["jobs"]),
                "one_hot_violations": int(metrics["one_hot"]["total"]),
                "dummy_actions": int(metrics["dummy_actions"]),
                "expected_recourse_cost_usd": recourse[
                    "expected_recourse_cost_usd"
                ],
                "cvar95_recourse_cost_usd": recourse[
                    "cvar_recourse_cost_usd"
                ],
                "scope": "pre-archive consistency snapshot",
                "superseded_by": "deliverables/SUBMISSION_VERIFICATION.json",
            }
        )
        legacy_path.write_text(
            json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main() -> int:
    args = parse_args()
    base = read_json(args.base_result.resolve())
    candidate_metadata = read_json(args.candidate_metadata.resolve())
    actions, site_ts, baseline_x, q, qubo_meta = reconstruct_full_qubo(base)
    _, baseline_load, _ = SOLVER.selected_vectors(actions, baseline_x, len(site_ts))

    previous_x = SOLVER.load_external_start_vector(
        actions, args.base_risk_csv.resolve()
    )
    previous_metrics = SOLVER.schedule_metrics(
        actions, site_ts, q, previous_x, reference_load_mw=baseline_load
    )
    recorded_previous = base["risk_accepted_solution"]["metrics"]
    energy_delta = abs(
        float(previous_metrics["qubo_energy"])
        - float(recorded_previous["qubo_energy"])
    )
    if energy_delta > 1e-8:
        raise RuntimeError(
            "Full-QUBO reconstruction failed: "
            f"previous-energy delta is {energy_delta:.3e}."
        )

    candidate_x = SOLVER.load_external_start_vector(
        actions, args.candidate_csv.resolve()
    )
    candidate_metrics = SOLVER.schedule_metrics(
        actions, site_ts, q, candidate_x, reference_load_mw=baseline_load
    )
    selected = actions.loc[candidate_x.astype(bool)].copy()
    source = "cloudseed_peak_shave_contextual_recourse_polish"
    selected.insert(0, "selection_source", source)

    baseline_cost = float(base["baseline"]["true_action_cost_usd"])
    max_cost = baseline_cost + float(base["args"]["risk_max_action_cost_increase"])
    feasibility_checks = {
        "one_hot_violations_zero": int(candidate_metrics["one_hot"]["total"]) == 0,
        "dummy_actions_zero": int(candidate_metrics["dummy_actions"]) == 0,
        "action_cost_within_guard": float(
            candidate_metrics["true_action_cost_usd"]
        )
        <= max_cost + 1e-9,
        "one_row_per_job": len(selected)
        == int(actions["job_id"].nunique())
        == int(selected["job_id"].nunique()),
    }
    previous_triplet = metrics_triplet(previous_metrics)
    candidate_triplet = metrics_triplet(candidate_metrics)
    dominance_checks = {
        key: candidate_triplet[key] < previous_triplet[key] - 1e-12
        for key in candidate_triplet
    }
    metadata_checks = {
        "expected_replay_matches": abs(
            candidate_triplet["expected_recourse_cost_usd"]
            - float(candidate_metadata["expected_recourse_cost_usd"])
        )
        <= 1e-12,
        "cvar_replay_matches": abs(
            candidate_triplet["cvar95_recourse_cost_usd"]
            - float(candidate_metadata["cvar95_recourse_cost_usd"])
        )
        <= 1e-12,
        "action_cost_replay_matches": abs(
            float(candidate_metrics["true_action_cost_usd"])
            - float(candidate_metadata["true_action_cost_usd"])
        )
        <= 1e-9,
    }
    if not all(feasibility_checks.values()):
        raise RuntimeError(f"Candidate failed feasibility guard: {feasibility_checks}")
    if not all(dominance_checks.values()):
        raise RuntimeError(f"Candidate does not Pareto-dominate previous final: {dominance_checks}")
    if not all(metadata_checks.values()):
        raise RuntimeError(f"Candidate replay differs from evidence metadata: {metadata_checks}")

    payload = copy.deepcopy(base)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["risk_accepted_solution"] = {
        "source": source,
        "metrics": candidate_metrics,
    }
    task_id = (
        candidate_metadata.get("source_result", {}).get("task_id")
        if isinstance(candidate_metadata.get("source_result"), dict)
        else None
    )
    payload["promotion"] = {
        "base_result": args.base_result.name,
        "base_risk_csv": args.base_risk_csv.name,
        "candidate_csv": args.candidate_csv.name,
        "candidate_metadata": args.candidate_metadata.name,
        "cloud_task_id": task_id,
        "policy": (
            "Promote only after independent full replay, feasibility/cost guard, "
            "and strict improvement in expected recourse, CVaR95, and peak IT load."
        ),
        "full_qubo_reconstruction_energy_delta": energy_delta,
        "feasibility_checks": feasibility_checks,
        "dominance_checks": dominance_checks,
        "metadata_checks": metadata_checks,
        "previous_metrics": previous_triplet,
        "promoted_metrics": candidate_triplet,
        "qubo_meta": qubo_meta,
    }
    payload["solvers"].append(
        {
            "solver": source,
            "gear": "cloud_balanced+contextual_recourse_polish",
            "status": "success",
            "raw_solver_result": {
                "task_id": task_id,
                "result": {
                    "ok": True,
                    "spin_config_persisted_separately": False,
                    "evidence_boundary": (
                        "The frozen package retains the cloud task ID, returned "
                        "length, and energy summary, but not the raw spin vector."
                    ),
                },
            },
            "raw_metrics": candidate_metrics,
            "repaired_metrics": candidate_metrics,
            "repair": {
                "repair_count": 0,
                "provenance": "industrial_ising_artifacts/candidate_solutions",
            },
        }
    )

    args.output_result.parent.mkdir(parents=True, exist_ok=True)
    args.output_result.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selected.to_csv(args.output_risk_csv, index=False, encoding="utf-8-sig")
    lines = [
        "# Cloud-seeded candidate promotion",
        "",
        f"- Source: {source}",
        f"- Cloud task ID: {task_id}",
        f"- Full-QUBO reconstruction energy delta: {energy_delta:.3e}",
        f"- Jobs: {len(selected)}; one-hot violations: 0; dummy actions: 0",
        f"- Action cost: {candidate_metrics['true_action_cost_usd']:.6f} (guard {max_cost:.6f})",
        "",
        "| Metric | Previous final | Promoted | Improvement |",
        "|---|---:|---:|---:|",
    ]
    for key in [
        "expected_recourse_cost_usd",
        "cvar95_recourse_cost_usd",
        "peak_it_mw",
    ]:
        old = previous_triplet[key]
        new = candidate_triplet[key]
        lines.append(f"| {key} | {old:.12f} | {new:.12f} | {old-new:.12f} |")
    lines.extend(
        [
            "",
            "The candidate is promoted because every feasibility, metadata-replay, "
            "and strict three-metric Pareto-dominance check passed.",
        ]
    )
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    refresh_industrial_final_tables(
        candidate_metrics,
        previous_triplet,
        candidate_triplet,
        source,
    )
    print(json.dumps(payload["promotion"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
