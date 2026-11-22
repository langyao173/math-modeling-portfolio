"""Build the competition report and experiment tables from existing artifacts."""

from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
OUT_DIR = ROOT / "deliverables" / "competition_report"
TABLE_DIR = OUT_DIR / "tables"
INDUSTRIAL_DIR = ROOT / "deliverables" / "industrial_ising_artifacts"
FINAL_RESULT_NAME = "qubo_solver_results_full_cloudseed_peakguard.json"
FINAL_RISK_NAME = "qubo_risk_accepted_actions_full_cloudseed_peakguard.csv"


def load_solver_module() -> Any:
    solver_path = ROOT / "src" / "07_solve_isling" / "solve_qubo_platform.py"
    spec = importlib.util.spec_from_file_location("solve_qubo_platform", solver_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load solver module from {solver_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOLVER = load_solver_module()


def read_json(name: str) -> dict[str, Any]:
    return json.loads((REPORTS_DIR / name).read_text(encoding="utf-8"))


def metric_row(run: str, view: str, source: str, metrics: dict[str, Any], notes: str = "") -> dict[str, Any]:
    recourse = metrics["recourse"]
    return {
        "run": run,
        "view": view,
        "source": source,
        "jobs": metrics["jobs"],
        "selected_actions": metrics["selected_actions"],
        "one_hot_violations": metrics["one_hot"]["total"],
        "dummy_actions": metrics["dummy_actions"],
        "qubo_energy": metrics["qubo_energy"],
        "true_action_cost_usd": metrics["true_action_cost_usd"],
        "expected_recourse_cost_usd": recourse["expected_recourse_cost_usd"],
        "cvar95_recourse_cost_usd": recourse["cvar_recourse_cost_usd"],
        "worst_scenario": recourse["worst_scenario"],
        "peak_it_mw": metrics["peak_it_mw"],
        "mode_counts": json.dumps(metrics["mode_counts"], ensure_ascii=False, sort_keys=True),
        "notes": notes,
    }


def solver_rows(result: dict[str, Any], run: str) -> list[dict[str, Any]]:
    rows = [metric_row(run, "baseline", "baseline", result["baseline"])]
    for record in result["solvers"]:
        if record.get("status") != "success":
            continue
        rows.append(metric_row(run, "solver", record["solver"], record["repaired_metrics"]))
    rows.append(
        metric_row(
            run,
            "accepted",
            result["accepted_solution"]["source"],
            result["accepted_solution"]["metrics"],
        )
    )
    if "risk_accepted_solution" in result:
        rows.append(
            metric_row(
                run,
                "risk_accepted",
                result["risk_accepted_solution"]["source"],
                result["risk_accepted_solution"]["metrics"],
            )
        )
    return rows


def find_solver_record(result: dict[str, Any], solver_name: str) -> dict[str, Any]:
    for record in result["solvers"]:
        if record.get("solver") == solver_name and record.get("status") == "success":
            return record
    raise KeyError(f"Missing successful solver record: {solver_name}")


def required_final_metrics_table(result: dict[str, Any]) -> pd.DataFrame:
    metrics = result["risk_accepted_solution"]["metrics"]
    recourse = metrics["recourse"]
    return pd.DataFrame(
        [
            {
                "jobs": result["selected_job_count"],
                "spins": result["selected_action_count"],
                "dummy_actions": metrics["dummy_actions"],
                "one_hot_violations": metrics["one_hot"]["total"],
                "expected_recourse_cost_usd": recourse["expected_recourse_cost_usd"],
                "cvar95_recourse_cost_usd": recourse["cvar_recourse_cost_usd"],
                "source": result["risk_accepted_solution"]["source"],
            }
        ]
    )


def process_row(
    stage: str,
    source: str,
    metrics: dict[str, Any],
    spins: int,
    repair_count: int | None,
    notes: str,
) -> dict[str, Any]:
    recourse = metrics["recourse"]
    return {
        "stage": stage,
        "source": source,
        "jobs": metrics["jobs"],
        "spins": spins,
        "dummy_actions": metrics["dummy_actions"],
        "one_hot_violations": metrics["one_hot"]["total"],
        "repair_count": "" if repair_count is None else repair_count,
        "expected_recourse_cost_usd": recourse["expected_recourse_cost_usd"],
        "cvar95_recourse_cost_usd": recourse["cvar_recourse_cost_usd"],
        "peak_it_mw": metrics["peak_it_mw"],
        "notes": notes,
    }


def required_optimization_process_table(cloud_result: dict[str, Any], final_result: dict[str, Any]) -> pd.DataFrame:
    cloud = find_solver_record(cloud_result, "cloud")
    cloud_polished = find_solver_record(cloud_result, "cloud_polished")
    recourse_polished = find_solver_record(cloud_result, "recourse_true_polish")
    final_metrics = final_result["risk_accepted_solution"]["metrics"]
    spins = final_result["selected_action_count"]
    return pd.DataFrame(
        [
            process_row(
                "cloud_raw",
                "CloudSolver",
                cloud["raw_metrics"],
                spins,
                None,
                "raw platform solution before local repair/polish",
            ),
            process_row(
                "repair",
                "cloud_polished",
                cloud_polished["repaired_metrics"],
                spins,
                int(cloud_polished["repair"].get("repair_count", 0)),
                "one-hot repair plus QUBO post-polish",
            ),
            process_row(
                "recourse_polish",
                "recourse_true_polish",
                recourse_polished["repaired_metrics"],
                spins,
                int(recourse_polished["repair"].get("repair_count", 0)),
                "exact true recourse/CVaR coordinate polish",
            ),
            process_row(
                "local_polish",
                final_result["risk_accepted_solution"]["source"],
                final_metrics,
                spins,
                0,
                "full local pass5 multi-start true recourse polish",
            ),
        ]
    )


def required_baseline_comparison_table(
    final_result: dict[str, Any],
    structural_result: dict[str, Any],
    recourse_result: dict[str, Any],
) -> pd.DataFrame:
    rows = [
        metric_row(
            "initial_solution",
            "baseline",
            "baseline",
            final_result["baseline"],
            "initial deterministic per-job baseline",
        ),
        metric_row(
            "structural_qubo_proxy",
            "risk_accepted",
            structural_result["risk_accepted_solution"]["source"],
            structural_result["risk_accepted_solution"]["metrics"],
            "structural QUBO proxy without true recourse objective",
        ),
        metric_row(
            "true_recourse",
            "risk_accepted",
            recourse_result["risk_accepted_solution"]["source"],
            recourse_result["risk_accepted_solution"]["metrics"],
            "true recourse/CVaR-aware route",
        ),
        metric_row(
            "final_solution",
            "risk_accepted",
            final_result["risk_accepted_solution"]["source"],
            final_result["risk_accepted_solution"]["metrics"],
            "current final submitted solution",
        ),
    ]
    return pd.DataFrame(rows)


def required_cleanroom_audit_table() -> pd.DataFrame:
    package_dir = ROOT / "deliverables" / "final_solution_package"
    zip_path = ROOT / "deliverables" / "TAOLIN_DC_SIM_final_solution_package.zip"
    final_csv = package_dir / "FINAL_SOLUTION.csv"
    validation_path = package_dir / "VALIDATION.json"
    recomputed_path = package_dir / "VALIDATION_RECOMPUTED.json"
    metrics_path = package_dir / "FINAL_METRICS.json"

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    recomputed = json.loads(recomputed_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    final_frame = pd.read_csv(final_csv)

    expected_diff = abs(
        float(metrics["expected_recourse_cost_usd"])
        - float(recomputed["expected_recourse_cost_usd"])
    )
    cvar_diff = abs(
        float(metrics["cvar95_recourse_cost_usd"])
        - float(recomputed["cvar95_recourse_cost_usd"])
    )
    metrics_consistent = math.isclose(
        float(metrics["expected_recourse_cost_usd"]),
        float(recomputed["expected_recourse_cost_usd"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) and math.isclose(
        float(metrics["cvar95_recourse_cost_usd"]),
        float(recomputed["cvar95_recourse_cost_usd"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    expected_columns = [
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

    scan_patterns = {
        "absolute_path": re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\"),
        "token_or_secret": re.compile(
            r"(08c81dd3ab|(?:API[_-]?KEY|TOKEN|SECRET)\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{12,})",
            re.IGNORECASE,
        ),
        "cache_dependency": re.compile(r"(^|/|\\)(__pycache__|\.cache|site-packages|\.venv|venv)(/|\\|$)", re.IGNORECASE),
    }
    scan_counts = {key: 0 for key in scan_patterns}
    if package_dir.exists():
        for path in package_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(package_dir).as_posix()
            text = path.read_bytes().decode("utf-8", errors="ignore")
            for key, pattern in scan_patterns.items():
                if pattern.search(rel) or pattern.search(text):
                    scan_counts[key] += 1

    rows = [
        {
            "audit_item": "zip_and_final_csv",
            "status": "pass" if zip_path.exists() and final_csv.exists() else "fail",
            "evidence": f"zip_exists={zip_path.exists()}; final_csv_exists={final_csv.exists()}",
        },
        {
            "audit_item": "validation_matches_csv",
            "status": "pass"
            if validation["status"] == "pass"
            and validation["row_count"] == len(final_frame)
            and validation["unique_jobs"] == final_frame["job_id"].nunique()
            else "fail",
            "evidence": f"rows={len(final_frame)}; unique_jobs={final_frame['job_id'].nunique()}; validation_status={validation['status']}",
        },
        {
            "audit_item": "cleanroom_recompute",
            "status": "pass" if recomputed["status"] == "pass" else "fail",
            "evidence": f"expected={recomputed['expected_recourse_cost_usd']:.6f}; cvar95={recomputed['cvar95_recourse_cost_usd']:.6f}",
        },
        {
            "audit_item": "metrics_consistency",
            "status": "pass" if metrics_consistent else "fail",
            "evidence": (
                "FINAL_METRICS.json agrees with VALIDATION_RECOMPUTED.json "
                f"within 1e-12 (expected_diff={expected_diff:.3e}; "
                f"cvar95_diff={cvar_diff:.3e})"
            ),
        },
        {
            "audit_item": "csv_field_consistency",
            "status": "pass" if list(final_frame.columns) == expected_columns else "fail",
            "evidence": ",".join(final_frame.columns),
        },
        {
            "audit_item": "one_hot_violation_zero",
            "status": "pass" if validation["one_hot_violations"] == 0 else "fail",
            "evidence": f"one_hot_violations={validation['one_hot_violations']}",
        },
        {
            "audit_item": "absolute_path_scan",
            "status": "pass" if scan_counts["absolute_path"] == 0 else "fail",
            "evidence": f"hits={scan_counts['absolute_path']}",
        },
        {
            "audit_item": "token_scan",
            "status": "pass" if scan_counts["token_or_secret"] == 0 else "fail",
            "evidence": f"hits={scan_counts['token_or_secret']}",
        },
        {
            "audit_item": "cache_dependency_scan",
            "status": "pass" if scan_counts["cache_dependency"] == 0 else "fail",
            "evidence": f"hits={scan_counts['cache_dependency']}",
        },
    ]
    return pd.DataFrame(rows)


def build_random_baseline_table() -> pd.DataFrame:
    jobs, actions, site_ts = SOLVER.load_tables(ROOT)
    selected_jobs = SOLVER.choose_jobs(actions, site_ts, max_jobs=0, seed=42)
    selected_actions = actions.loc[actions["job_id"].isin(selected_jobs)].copy().reset_index(drop=True)
    selected_actions["job_id"] = selected_actions["job_id"].astype(str)
    baseline_x = SOLVER.baseline_solution(selected_actions)
    _, baseline_load, _ = SOLVER.selected_vectors(selected_actions, baseline_x, len(site_ts))
    q, _ = SOLVER.build_qubo(
        selected_actions,
        site_ts,
        one_hot_penalty=80.0,
        peak_weight=0.002,
        risk_weight=0.10,
        recourse_weight=0.0,
        recourse_cvar_weight=0.10,
        dummy_cost=120.0,
        dummy_only_cost=25.0,
        cost_cap=20.0,
        reference_load_mw=baseline_load,
    )

    groups = [
        [selected_actions.index.get_loc(idx) for idx in group.index]
        for _, group in selected_actions.groupby("job_id", sort=False)
    ]
    rng = np.random.default_rng(20260630)
    random_rows: list[dict[str, Any]] = []
    for sample_idx in range(200):
        x = np.zeros(len(selected_actions), dtype=int)
        for positions in groups:
            # Keep dummy-only jobs feasible; otherwise sample from real actions.
            real_positions = [
                pos for pos in positions if not bool(selected_actions.iloc[pos]["is_dummy"])
            ]
            candidates = real_positions if real_positions else positions
            x[int(rng.choice(candidates))] = 1
        metrics = SOLVER.schedule_metrics(selected_actions, site_ts, q, x, reference_load_mw=baseline_load)
        random_rows.append(
            {
                "sample": sample_idx,
                "true_action_cost_usd": metrics["true_action_cost_usd"],
                "expected_recourse_cost_usd": metrics["recourse"]["expected_recourse_cost_usd"],
                "cvar95_recourse_cost_usd": metrics["recourse"]["cvar_recourse_cost_usd"],
                "peak_it_mw": metrics["peak_it_mw"],
                "dummy_actions": metrics["dummy_actions"],
            }
        )
    frame = pd.DataFrame(random_rows)
    summary = frame.agg(
        {
            "true_action_cost_usd": ["min", "mean", "median", "max"],
            "expected_recourse_cost_usd": ["min", "mean", "median", "max"],
            "cvar95_recourse_cost_usd": ["min", "mean", "median", "max"],
            "peak_it_mw": ["min", "mean", "median", "max"],
            "dummy_actions": ["min", "mean", "median", "max"],
        }
    )
    rows = []
    for statistic in summary.index:
        row = {
            "run": "random_feasible_200",
            "view": statistic,
            "source": "random_feasible",
            "jobs": int(selected_actions["job_id"].nunique()),
            "selected_actions": int(selected_actions["job_id"].nunique()),
            "one_hot_violations": 0,
        }
        row.update({column: float(summary.loc[statistic, column]) for column in summary.columns})
        rows.append(row)
    return pd.DataFrame(rows)


def build_dummy_table() -> pd.DataFrame:
    actions = pd.read_parquet(ROOT / "data" / "processed" / "actions.parquet")
    final_solution = pd.read_csv(REPORTS_DIR / FINAL_RISK_NAME)
    dummy_jobs = final_solution.loc[final_solution["is_dummy"].astype(bool), "job_id"].astype(str).tolist()
    rows = []
    for job_id in dummy_jobs:
        group = actions.loc[actions["job_id"].astype(str) == job_id].copy()
        real = group.loc[~group["is_dummy"].astype(bool)]
        rows.append(
            {
                "job_id": job_id,
                "candidate_actions": len(group),
                "real_candidate_actions": len(real),
                "dummy_candidate_actions": int(group["is_dummy"].astype(bool).sum()),
                "selected_action_id": str(final_solution.loc[final_solution["job_id"].astype(str) == job_id, "action_id"].iloc[0]),
                "reason": "dummy_only_job_no_real_feasible_action"
                if real.empty
                else "dummy_selected_despite_real_actions",
                "min_real_action_cost_usd": None if real.empty else float(real["action_cost"].min()),
            }
        )
    columns = [
        "job_id",
        "candidate_actions",
        "real_candidate_actions",
        "dummy_candidate_actions",
        "selected_action_id",
        "reason",
        "min_real_action_cost_usd",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("job_id")


def build_tail_scenario_table(result: dict[str, Any]) -> pd.DataFrame:
    scenarios = result["risk_accepted_solution"]["metrics"]["recourse"]["scenarios"]
    frame = pd.DataFrame(scenarios)
    frame["tail_rank"] = frame["total_recourse_cost_usd"].rank(method="first", ascending=False).astype(int)
    return frame.sort_values("total_recourse_cost_usd", ascending=False)


def build_cloud_staged_check_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for jobs_count in [120, 240, 360]:
        data = safe_result(f"qubo_solver_results_{jobs_count}_rec_cvar0p10_cut3_truepolish5_cloud_check.json")
        if data is None:
            continue
        rows.append(metric_row(f"cloud_check_{jobs_count}", "baseline", "baseline", data["baseline"]))
        for record in data["solvers"]:
            if record.get("status") != "success":
                continue
            if record["solver"] in {"cloud", "cloud_polished", "recourse_true_polish"}:
                rows.append(
                    metric_row(
                        f"cloud_check_{jobs_count}",
                        "solver",
                        record["solver"],
                        record["repaired_metrics"],
                    )
                )
        rows.append(
            metric_row(
                f"cloud_check_{jobs_count}",
                "risk_accepted",
                data["risk_accepted_solution"]["source"],
                data["risk_accepted_solution"]["metrics"],
                "selected_for_true_recourse_not_for_full_cloud_escalation",
            )
        )
    return pd.DataFrame(rows)


def read_artifact_csv(relative_path: str) -> pd.DataFrame:
    path = INDUSTRIAL_DIR / relative_path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_industrial_lns_portfolio_table() -> pd.DataFrame:
    summary_path = INDUSTRIAL_DIR / "industrial_ising_evidence_summary.json"
    if not summary_path.exists():
        return pd.DataFrame()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    purpose = {
        "base_lns": "general bounded action-selection improvement",
        "aligned_recourse_lns": "one-swap exact recourse/CVaR diagonal alignment",
        "tail_cvar_lns": "CVaR95 tail-risk directed coefficients",
        "cloud_sparse_tail_lns": "sparse capped-range CloudSolver-friendly tail family",
        "dummy_rescue_lns": "dummy and near-dummy action-space audit",
        "peak_shave_lns": "grid/import peak reduction pressure",
        "renewable_absorb_lns": "shift load toward renewable/low-price slots",
    }
    for family, data in summary.get("families", {}).items():
        rows.append(
            {
                "family": family,
                "purpose": purpose.get(family, ""),
                "jobs": data.get("jobs"),
                "spins": data.get("spins"),
                "q_abs_max": data.get("q_abs_max"),
            }
        )
    return pd.DataFrame(rows)


def safe_result(name: str) -> dict[str, Any] | None:
    path = REPORTS_DIR / name
    if not path.exists():
        return None
    return read_json(name)


def result_matches_instance(result: dict[str, Any], reference: dict[str, Any]) -> bool:
    return (
        int(result.get("selected_job_count", -1)) == int(reference.get("selected_job_count", -2))
        and int(result.get("selected_action_count", -1)) == int(reference.get("selected_action_count", -2))
    )


def current_result(name: str, reference: dict[str, Any]) -> dict[str, Any] | None:
    result = safe_result(name)
    if result is None or not result_matches_instance(result, reference):
        return None
    return result


def write_tables() -> dict[str, pd.DataFrame]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    final = read_json(FINAL_RESULT_NAME)
    cloud_final = current_result("qubo_solver_results_full_rec_cvar0p10_cut3_truepolish3_cloud.json", final)
    structural = current_result("qubo_solver_results_full_d120_do25_cloudpolish.json", final)
    rec_linear = current_result("qubo_solver_results_full_rec_cvar0p10_cloud.json", final)

    tables: dict[str, pd.DataFrame] = {}
    dummy_jobs_table = build_dummy_table()
    tables["required_01_final_metrics"] = required_final_metrics_table(final)
    if cloud_final is not None:
        tables["required_02_optimization_process"] = required_optimization_process_table(cloud_final, final)
    else:
        tables["required_02_optimization_process"] = pd.DataFrame(
            [
                process_row(
                    "local_polish",
                    final["risk_accepted_solution"]["source"],
                    final["risk_accepted_solution"]["metrics"],
                    int(final["selected_action_count"]),
                    0,
                    "full-instance Cloud result not regenerated for the 1929-action table; same-QUBO LNS Cloud evidence is reported in Section 7",
                )
            ]
        )
    baseline_rows = [
        metric_row("initial_solution", "baseline", "baseline", final["baseline"], "initial deterministic per-job baseline")
    ]
    if structural is not None:
        baseline_rows.append(
            metric_row(
                "structural_qubo_proxy",
                "risk_accepted",
                structural["risk_accepted_solution"]["source"],
                structural["risk_accepted_solution"]["metrics"],
                "structural QUBO proxy without true recourse objective",
            )
        )
    if cloud_final is not None:
        baseline_rows.append(
            metric_row(
                "true_recourse",
                "risk_accepted",
                cloud_final["risk_accepted_solution"]["source"],
                cloud_final["risk_accepted_solution"]["metrics"],
                "current true recourse/CVaR-aware cloud route",
            )
        )
    baseline_rows.append(
        metric_row(
            "final_solution",
            "risk_accepted",
            final["risk_accepted_solution"]["source"],
            final["risk_accepted_solution"]["metrics"],
            "current final submitted solution",
        )
    )
    tables["required_03_baseline_comparison"] = pd.DataFrame(baseline_rows)
    tables["required_04_dummy_jobs"] = dummy_jobs_table
    tables["required_05_cleanroom_audit"] = required_cleanroom_audit_table()
    tables["01_final_metrics"] = pd.DataFrame(
        [
            metric_row(
                f"full_{final['selected_action_count']}_spin",
                "risk_accepted",
                final["risk_accepted_solution"]["source"],
                final["risk_accepted_solution"]["metrics"],
                "current cloud-seeded, independently replayed deliverable",
            )
        ]
    )
    if cloud_final is not None:
        tables["02_cloud_repair_polish_progression"] = pd.DataFrame(
            [
                row
                for row in solver_rows(cloud_final, "full_cloud_cut_truepolish")
                if row["source"] in {"baseline", "cloud", "cloud_polished", "recourse_true_polish", "recourse_cut_greedy"}
            ]
        )
    else:
        tables["02_cloud_repair_polish_progression"] = pd.DataFrame()
    structural_rows = []
    if structural is not None:
        structural_rows.append(metric_row("structural_qubo_proxy", "risk_accepted", structural["risk_accepted_solution"]["source"], structural["risk_accepted_solution"]["metrics"]))
    if rec_linear is not None:
        structural_rows.append(metric_row("recourse_linear_cloud", "risk_accepted", rec_linear["risk_accepted_solution"]["source"], rec_linear["risk_accepted_solution"]["metrics"]))
    if cloud_final is not None:
        structural_rows.append(metric_row("cut_true_polish_cloud_integrated", "risk_accepted", cloud_final["risk_accepted_solution"]["source"], cloud_final["risk_accepted_solution"]["metrics"]))
    structural_rows.append(
        metric_row(
            "cloudseed_peakguard_promoted",
            "risk_accepted",
            final["risk_accepted_solution"]["source"],
            final["risk_accepted_solution"]["metrics"],
        )
    )
    tables["03_structural_vs_true_recourse"] = pd.DataFrame(structural_rows)

    solver_baseline_rows = []
    solver_baseline_rows.extend(
        [
            row
            for row in solver_rows(final, "full_promoted")
            if row["source"] in {
                "baseline",
                "local",
                "local_polished",
                "greedy_polish",
                "recourse_true_polish",
                "cloudseed_peak_shave_contextual_recourse_polish",
            }
        ]
    )
    if cloud_final is not None:
        solver_baseline_rows.extend(
            [
                row
                for row in solver_rows(cloud_final, "full_cloud_integrated")
                if row["source"] in {"cloud", "cloud_polished", "recourse_true_polish"}
            ]
        )
    solver_baseline_rows.extend(build_random_baseline_table().to_dict("records"))
    tables["04_solver_baselines"] = pd.DataFrame(solver_baseline_rows)

    sensitivity_rows: list[dict[str, Any]] = []
    for run, name in [
        ("dummy25_cloud", "qubo_solver_results_full_credit.json"),
        ("dummy120_no_split_cloud", "qubo_solver_results_full_d120_credit.json"),
        ("dummy120_dummyonly25_cloud", "qubo_solver_results_full_d120_do25_cloudpolish.json"),
        ("rec_cvar0p10_cloud", "qubo_solver_results_full_rec_cvar0p10_cloud.json"),
        ("rec_cvar0p10_cut3_cloud", "qubo_solver_results_full_rec_cvar0p10_cut3_truepolish3_cloud.json"),
        ("rec_cvar0p10_cut3_multistart_local_pass3", "qubo_solver_results_full_rec_cvar0p10_cut3_truepolish3_ms_local.json"),
        ("rec_cvar0p10_cut3_multistart_local_pass5", "qubo_solver_results_full_rec_cvar0p10_cut3_truepolish5_ms_local.json"),
        ("cloudseed_peakguard_promoted", FINAL_RESULT_NAME),
    ]:
        data = safe_result(name)
        if data is None or not result_matches_instance(data, final):
            continue
        key = "risk_accepted_solution" if "risk_accepted_solution" in data else "accepted_solution"
        sensitivity_rows.append(metric_row(run, key.replace("_solution", ""), data[key]["source"], data[key]["metrics"]))
    for jobs_count in ["120", "240", "360", "full"]:
        data = safe_result(f"qubo_solver_results_{jobs_count}_rec_cvar0p10_cut3_truepolish3_ms_local.json")
        if data is None or (jobs_count == "full" and not result_matches_instance(data, final)):
            continue
        sensitivity_rows.append(
            metric_row(
                f"neighborhood_{jobs_count}",
                "risk_accepted",
                data["risk_accepted_solution"]["source"],
                data["risk_accepted_solution"]["metrics"],
                "local_multistart_neighborhood_sensitivity",
            )
    )
    tables["05_sensitivity"] = pd.DataFrame(sensitivity_rows)
    tables["06_dummy_jobs"] = dummy_jobs_table
    tables["07_tail_scenarios"] = build_tail_scenario_table(final)
    tables["08_cloud_staged_check"] = build_cloud_staged_check_table()
    tables["09_industrial_lns_portfolio"] = build_industrial_lns_portfolio_table()
    tables["10_same_qubo_oracle_replacement"] = read_artifact_csv("solver_runs/oracle_replacement_summary.csv")
    tables["11_industrial_metrics"] = read_artifact_csv("tables/industrial_metrics.csv")
    tables["12_cloud_contribution"] = read_artifact_csv("tables/cloud_contribution.csv")
    tables["13_qubo_energy_true_objective"] = read_artifact_csv("validation/qubo_energy_true_objective_check.csv")
    tables["14_penalty_sweep"] = read_artifact_csv("tables/penalty_sweep.csv")
    exact_baseline_path = TABLE_DIR / "15_exact_controlled_baseline.csv"
    if not exact_baseline_path.is_file():
        raise FileNotFoundError(
            "Controlled exact baseline is missing; run build_exact_controlled_baseline.py first."
        )
    tables["15_exact_controlled_baseline"] = pd.read_csv(exact_baseline_path)

    for name, frame in tables.items():
        if name == "15_exact_controlled_baseline":
            # This table and its companion JSON are a hash-frozen solver
            # snapshot (including real wall times), not a derived report table.
            # Keep the exact-baseline builder as the sole writer.
            continue
        frame.to_csv(TABLE_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    return tables


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if frame.empty:
        return "_Table not generated._"
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        return f"_Table missing columns: {', '.join(missing)}._"
    subset = frame.loc[:, columns].copy()
    if max_rows is not None:
        subset = subset.head(max_rows)
    for column in subset.columns:
        if pd.api.types.is_float_dtype(subset[column]):
            subset[column] = subset[column].map(lambda value: f"{value:.6f}")
    return subset.to_markdown(index=False)


def write_report(tables: dict[str, pd.DataFrame]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final = tables["01_final_metrics"].iloc[0]
    required_final = tables["required_01_final_metrics"].iloc[0]
    jobs_count = int(required_final["jobs"])
    spins_count = int(required_final["spins"])
    dummy_count = int(required_final["dummy_actions"])
    worst = tables["07_tail_scenarios"].iloc[0]
    if dummy_count:
        dummy_summary = f"保留 {dummy_count} 个 dummy 作业，原因见 dummy jobs 解释表。"
        dummy_detail = f"最终共有 {dummy_count} 个 dummy jobs。下表区分 dummy-only 与有真实候选但仍被选择 dummy 的情况。"
        limitation_dummy = "- 仍保留的 dummy actions 需要继续回到上游候选动作生成阶段扩充可行真实动作。"
    else:
        dummy_summary = "上游 action table 已补入 horizon-rescue 真实候选，最终 dummy actions 为 0。"
        dummy_detail = "最终无 dummy jobs；此前 horizon 边界任务已由上游 `horizon_rescue` 候选动作吸收。"
        limitation_dummy = "- dummy-only 边界已通过 horizon-rescue 候选动作处理；后续仍需用真实迁移/跨站动作替换该工程化 rescue proxy。"

    lines = [
        "# TAOLIN-DC-SIM Competition Report",
        "",
        "## 摘要",
        "",
        f"本文将 GPU 任务调度、绿电利用、追索风险和 QUBO/Ising 求解结合为一个离散优化流程。最终实例包含 {jobs_count} 个作业、{spins_count} 个候选动作自旋变量。最终提交解覆盖 {jobs_count}/{jobs_count} 个作业，one-hot violation 为 0，{dummy_summary} 经 700-spin 云端候选、独立全量回放与三指标 Pareto guard 晋升后的 expected recourse 为 {final['expected_recourse_cost_usd']:.6f}，CVaR95 为 {final['cvar95_recourse_cost_usd']:.6f}。",
        "",
        "## 1. 问题定义",
        "",
        "给定 GPU 作业集合、候选调度动作和站点级绿电/价格/碳排时间序列，目标是在每个作业恰选一个动作的前提下，降低调度成本、峰值负载和尾部追索风险。追索风险来自绿电短缺、高价、flex 调用和可再生出力压力场景。离散选择部分编码为 QUBO，便于提交到 QUBO/Ising 求解器；连续 grid/renewable/flex-shortfall recourse 通过闭式逐时 evaluator 做后验和局部精修。",
        "",
        "## 2. 数据与任务构造",
        "",
        f"- 作业数：{jobs_count}",
        f"- QUBO 变量/候选动作数：{spins_count}",
        "- 候选动作类型：normal、delay、degrade、horizon_rescue、dummy",
        "- dummy action 含义：无真实可行动作时的占位选择；horizon_rescue 用于处理 24h 边界内可部分执行但完整 duration 溢出的任务",
        "- 时间粒度：24 个逐时 slot",
        "",
        "## 3. 模型",
        "",
        "QUBO 包含 one-hot 约束惩罚、动作成本、结构化时段风险项、峰值负载二次项和可选 recourse marginal 线性项。recourse/CVaR evaluator 对固定调度解计算各压力场景下的 grid import、renewable curtailment、flex shortfall 与 CVaR95。后续加入 iterative recourse cut，用候选解更新 marginal reference；最终再用 exact recourse polish 在 one-job 邻域直接优化真实 CVaR。",
        "",
        "## 4. 求解流程",
        "",
        "流程为：构造候选动作表 -> 生成 QUBO -> LocalSolver/CloudSolver bounded oracle -> one-hot repair -> exact recourse replay -> local polish -> 三指标 submission guard -> 成果包验收。历史 staged 记录覆盖 120/240/360 作业及一次旧版 1922-spin 全量调用；当前终版的云端贡献严格限定为一个 700-spin `peak_shave_lns` 同构邻域种子。完整 1929 变量调度未被声明为云端全局求解，终版只在独立 588 作业回放同时改善 expected recourse、CVaR 与峰值负荷后晋升。",
        "",
        "## 5. 必备结果表",
        "",
        "### 5.1 最终指标表",
        "",
        markdown_table(
            tables["required_01_final_metrics"],
            [
                "jobs",
                "spins",
                "dummy_actions",
                "one_hot_violations",
                "expected_recourse_cost_usd",
                "cvar95_recourse_cost_usd",
                "source",
            ],
        ),
        "",
        "### 5.2 优化过程表",
        "",
        markdown_table(
            tables["required_02_optimization_process"],
            [
                "stage",
                "source",
                "jobs",
                "spins",
                "dummy_actions",
                "one_hot_violations",
                "repair_count",
                "expected_recourse_cost_usd",
                "cvar95_recourse_cost_usd",
                "notes",
            ],
        ),
        "",
        "### 5.3 Baseline 对比表",
        "",
        markdown_table(
            tables["required_03_baseline_comparison"],
            [
                "run",
                "source",
                "dummy_actions",
                "one_hot_violations",
                "expected_recourse_cost_usd",
                "cvar95_recourse_cost_usd",
                "peak_it_mw",
                "notes",
            ],
        ),
        "",
        "### 5.4 Dummy jobs 解释表",
        "",
        dummy_detail,
        "",
        markdown_table(
            tables["required_04_dummy_jobs"],
            [
                "job_id",
                "candidate_actions",
                "real_candidate_actions",
                "dummy_candidate_actions",
                "selected_action_id",
                "reason",
            ],
        ),
        "",
        "### 5.5 Clean-room 审计表",
        "",
        markdown_table(
            tables["required_05_cleanroom_audit"],
            [
                "audit_item",
                "status",
                "evidence",
            ],
        ),
        "",
        "## 6. 支撑实验表",
        "",
        "### 6.1 Cloud raw / repair / polish / exact recourse polish",
        "",
        markdown_table(
            tables["02_cloud_repair_polish_progression"],
            [
                "source",
                "one_hot_violations",
                "dummy_actions",
                "true_action_cost_usd",
                "expected_recourse_cost_usd",
                "cvar95_recourse_cost_usd",
                "peak_it_mw",
            ],
        ),
        "",
        "### 6.2 Structural QUBO proxy vs true recourse/CVaR",
        "",
        markdown_table(
            tables["03_structural_vs_true_recourse"],
            [
                "run",
                "source",
                "dummy_actions",
                "expected_recourse_cost_usd",
                "cvar95_recourse_cost_usd",
                "peak_it_mw",
            ],
        ),
        "",
        "### 6.3 LocalSolver / CloudSolver / random baseline",
        "",
        markdown_table(
            tables["04_solver_baselines"],
            [
                "run",
                "view",
                "source",
                "dummy_actions",
                "expected_recourse_cost_usd",
                "cvar95_recourse_cost_usd",
                "peak_it_mw",
            ],
            max_rows=16,
        ),
        "",
        "### 6.4 Penalty / neighborhood / recourse cut sensitivity",
        "",
        markdown_table(
            tables["05_sensitivity"],
            [
                "run",
                "source",
                "dummy_actions",
                "expected_recourse_cost_usd",
                "cvar95_recourse_cost_usd",
                "peak_it_mw",
                "notes",
            ],
        ),
        "",
        "### 6.5 Dummy jobs explanation",
        "",
        dummy_detail,
        "",
        markdown_table(
            tables["06_dummy_jobs"],
            [
                "job_id",
                "candidate_actions",
                "real_candidate_actions",
                "dummy_candidate_actions",
                "selected_action_id",
                "reason",
            ],
        ),
        "",
        "### 6.6 Tail scenario analysis",
        "",
        f"最终 CVaR 由最坏场景 `{worst['name']}` 主导，该场景总 recourse cost 为 {worst['total_recourse_cost_usd']:.6f}。",
        "",
        markdown_table(
            tables["07_tail_scenarios"],
            [
                "name",
                "probability",
                "total_recourse_cost_usd",
                "grid_cost_usd",
                "curtailment_cost_usd",
                "flex_shortfall_cost_usd",
                "grid_import_mwh",
                "tail_rank",
            ],
        ),
        "",
        "### 6.7 Staged cloud check after SDK repair",
        "",
        markdown_table(
            tables["08_cloud_staged_check"],
            [
                "run",
                "view",
                "source",
                "one_hot_violations",
                "dummy_actions",
                "expected_recourse_cost_usd",
                "cvar95_recourse_cost_usd",
                "peak_it_mw",
            ],
        ),
        "",
        "结论：历史 120/240/360 job staged 云端复核均成功且 one-hot violations 为 0；当前终版的可复核 CloudSolver 贡献以第 7 节 700-spin same-QUBO oracle replacement 与持久化候选为准。",
        "",
        "## 7. 工业级 Ising 证据链",
        "",
        "本轮增强将表述从“QUBO 能跑出一个本地 polish 解”升级为“Ising-assisted bounded large-neighborhood optimizer”。CloudSolver 用作 bounded binary action-selection neighborhood 的 oracle；所有 raw output 均经过相同 decode、one-hot repair、exact recourse replay 与 submission guard 审计。最终业务指标仍以 exact recourse/CVaR replay 为准，不用 QUBO energy 充当业务目标。",
        "",
        "### 7.1 多 QUBO 邻域组合",
        "",
        markdown_table(
            tables["09_industrial_lns_portfolio"],
            ["family", "purpose", "jobs", "spins", "q_abs_max"],
        ),
        "",
        "### 7.2 Same-QUBO oracle replacement",
        "",
        markdown_table(
            tables["10_same_qubo_oracle_replacement"],
            [
                "family",
                "oracle",
                "status",
                "spins",
                "raw_feasible",
                "repair_count",
                "accepted_under_submission_guard",
                "polished_cvar95",
                "polished_true_objective",
            ],
            max_rows=30,
        ),
        "",
        "### 7.3 工业指标",
        "",
        markdown_table(
            tables["11_industrial_metrics"],
            [
                "solution",
                "FlexDeliveryRate",
                "TailShortfall95_mwh",
                "SLAViolationRate_delay_or_dummy",
                "GridImportPeak_mw",
                "RenewableCurtailment_mwh",
                "expected_recourse_cost_usd",
                "cvar95_recourse_cost_usd",
                "dominant_tail_scenario",
            ],
        ),
        "",
        "### 7.4 Cloud contribution and proxy audit",
        "",
        markdown_table(
            tables["12_cloud_contribution"],
            [
                "family",
                "cloud_objective",
                "best_non_cloud_objective",
                "cloud_contribution_positive_if_gt0",
                "cloud_cvar95",
                "best_non_cloud_cvar95",
                "best_non_cloud_oracle",
            ],
        ),
        "",
        markdown_table(
            tables["13_qubo_energy_true_objective"],
            [
                "family",
                "samples",
                "qubo_energy_vs_true_objective_corr",
                "qubo_energy_vs_true_objective_rank_corr",
                "qubo_energy_vs_recourse_only_corr",
                "best_qubo_oracle",
                "best_true_objective_oracle",
                "best_recourse_only_oracle",
                "best_oracle_match",
            ],
        ),
        "",
        "解释：oracle replacement 表把 CloudSolver、incumbent、greedy 和 random restart 放在同一个 QUBO、同一个 decode/repair/replay 规则下比较。`true_objective` audit 使用 exact recourse/CVaR 加 submission guard penalty；`recourse_only` 单独报告纯 expected+CVaR 相关性，避免不可提交的低 recourse 随机解被误判为业务可接受改进。`aligned_recourse_lns` 和 `cloud_sparse_tail_lns` 在 QUBO diagonal 中注入 one-swap exact guarded recourse/CVaR delta，并对 Cloud-friendly family 做稀疏化和动态范围封顶。",
        "",
        "### 7.5 Penalty calibration sweep",
        "",
        markdown_table(
            tables["14_penalty_sweep"],
            [
                "one_hot_penalty_factor",
                "one_hot_penalty",
                "raw_feasible",
                "true_objective_expected_plus_cvar",
                "CVaR95",
                "dummy_count",
                "coefficient_abs_range",
                "greedy_improvements",
            ],
        ),
        "",
        "### 7.6 受控规模精确 MILP 基线",
        "",
        "对 30/60/100 个嵌套作业集合开放全部真实动作，其余作业固定为终版；"
        "HiGHS 最小化与统一评估器一致的期望追索加五场景 CVaR95，并在主目标容差内二级最小化峰值。"
        "该实验只给出受控邻域证书，不是 588 作业完整问题的全局最优证书。",
        "",
        markdown_table(
            tables["15_exact_controlled_baseline"],
            [
                "controlled_jobs",
                "solution",
                "solver_primary_status",
                "audit_objective_usd",
                "audit_gap_to_exact_pct",
                "peak_it_mw",
                "changed_jobs",
            ],
        ),
        "",
        "## 8. 局限与边界",
        "",
        "- 当前市场价格数据仍为 synthetic fallback，不能作为真实市场收益结论。",
        limitation_dummy,
        "- recourse evaluator 尚未纳入 storage、unit commitment、DC power flow 与整数 redispatch。",
        "- 云端额度应受控使用；报告只声明 bounded binary neighborhood oracle，不声明伊辛机全局求解完整调度问题。",
        "",
        "## 9. 提交材料位置",
        "",
        "- `deliverables/final_solution_package/FINAL_SOLUTION.csv`",
        "- `deliverables/final_solution_package/VALIDATION.json`",
        "- `deliverables/final_solution_package/VALIDATION_RECOMPUTED.json`",
        "- `deliverables/final_solution_package/artifacts/`",
        "- `deliverables/TAOLIN_DC_SIM_final_solution_package.zip`",
    ]
    (OUT_DIR / "COMPETITION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    tables = write_tables()
    write_report(tables)
    print(f"Wrote {OUT_DIR / 'COMPETITION_REPORT.md'}")
    print(f"Wrote tables to {TABLE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
