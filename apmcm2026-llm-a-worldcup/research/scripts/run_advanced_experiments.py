"""Checkpointed execution of the preregistered advanced study."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from worldcup_forecasting.application.runtime import write_csv, write_json
from worldcup_forecasting.experiments._nested_selection_types import frame_content_hash
from worldcup_forecasting.experiments.ablations import run_measured_ablations
from worldcup_forecasting.experiments.audit_baselines import fit_fixed_audit_baselines
from worldcup_forecasting.experiments.audit_execution import (
    evaluate_persisted_blind_audit,
    freeze_audit_models,
    load_batch_blind_2014_features,
    load_isolated_2014_truth,
    persist_audit_result,
    persist_blind_audit_predictions,
    prepare_frozen_audit_predictions,
)
from worldcup_forecasting.experiments.baseline_suite import run_baseline_suite
from worldcup_forecasting.experiments.development_snapshot import (
    load_development_snapshot,
)
from worldcup_forecasting.experiments.evidence import summarize_oof_evidence
from worldcup_forecasting.experiments.format_transport import run_format_transport_audits
from worldcup_forecasting.experiments.nested_calibration import apply_nested_calibration
from worldcup_forecasting.experiments.nested_decision import apply_nested_decision
from worldcup_forecasting.experiments.nested_selection import (
    fit_final_selected_model,
    run_nested_selection,
    select_final_development_spec,
)
from worldcup_forecasting.experiments.parameter_uncertainty import (
    TournamentBootstrapConfig,
    run_tournament_cluster_refits,
)
from worldcup_forecasting.experiments.sarmanov_dependence import (
    apply_nested_sarmanov_exploration,
)
from worldcup_forecasting.experiments.statistical_closure import (
    build_statistical_closure,
)
from worldcup_forecasting.experiments.stress import run_measured_stress_tests
from worldcup_forecasting.models.advanced_count import (
    EquivariantCountModel,
    joint_batch_from_rates,
)
from worldcup_forecasting.simulation.adaptive import (
    FROZEN_ADAPTIVE_CONFIG,
    FROZEN_ADAPTIVE_SEED,
    FROZEN_PROTOCOL_DEVIATION_SHA256,
    adaptive_contract_sha256,
    simulate_tournament_adaptive,
)
from worldcup_forecasting.simulation.format_policy import (
    FIFA_2026_ANNEX_C_CSV_SHA256,
    FIFA_2026_REGULATIONS_SHA256,
)
from worldcup_forecasting.simulation.scenario import build_format_scenario_provider
from worldcup_forecasting.simulation.tournament import (
    TournamentFormat,
    simulate_tournament,
)
from worldcup_forecasting.uncertainty.decomposition import (
    hierarchical_variance_decomposition,
)
from worldcup_forecasting.validation._promotion_artifacts import checkpoint_status
from worldcup_forecasting.validation.artifact_integrity import (
    build_stage_manifest,
    payload_sha256,
    read_verified_artifact_bytes,
    verify_stage_manifest,
)
from worldcup_forecasting.validation.audit_orchestration import run_one_time_audit
from worldcup_forecasting.validation.pre2014_freeze import git_output
from worldcup_forecasting.validation.promotion import (
    issue_promotion_certificate,
    verify_promotion_certificate,
)
from worldcup_forecasting.validation.runtime_reproducibility import (
    RUNTIME_REFERENCE_PROTOCOL,
    RUNTIME_REFERENCE_PROTOCOL_SHA256,
    seal_ablation_runtime_evidence,
    seal_baseline_runtime_evidence,
    seal_nested_runtime_evidence,
)
from worldcup_forecasting.validation.stage_orchestration import (
    run_all_pipeline,
    run_preaudit_pipeline,
)
from worldcup_forecasting.validation.transport import (
    kl_edition_robust_risk,
    paired_tournament_cluster_interval,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "temp" / "advanced_evidence"
DEVELOPMENT_SNAPSHOT = ROOT / "data" / "development_snapshot"
FORMAT_STAGES = (
    "group_stage",
    "round_of_32",
    "round_of_16",
    "quarter_final",
    "semi_final",
    "third_place",
    "final",
)


def _protocol_lock() -> dict[str, Any]:
    lock_path = ROOT / "plan" / "protocol_lock.json"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    preregistration = ROOT / str(payload["protocol_path"])
    observed = hashlib.sha256(preregistration.read_bytes()).hexdigest()
    if observed != payload["protocol_sha256"]:
        raise RuntimeError("Preregistration content no longer matches its frozen protocol lock")
    return payload


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _dump_joblib(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _persist_deterministic_stage_summary(
    summary: dict[str, Any], path: Path, *, started: float
) -> dict[str, Any]:
    """Seal deterministic science fields while returning operational wall time."""
    if "runtime_seconds" in summary:
        raise ValueError("Runtime must not enter a manifest-bound stage summary")
    write_json(summary, path)
    return {**summary, "runtime_seconds": time.perf_counter() - started}


def _artifact_manifest(
    directory: Path,
    *,
    relative_paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    if relative_paths is None:
        relative_paths = [
            path.relative_to(directory).as_posix()
            for path in sorted(item for item in directory.rglob("*") if item.is_file())
            if not path.name.endswith("artifact_manifest.json")
        ]
    return build_stage_manifest(directory, relative_paths)


def _load_locked_nested(output: Path) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    checkpoint = checkpoint_status(output, "nested")
    if checkpoint["status"] != "complete":
        raise RuntimeError("Nested artifacts must be manifest-verified before deserialization")
    binding = checkpoint["manifest"]
    nested_dir = output / "nested"
    nested = joblib.load(
        io.BytesIO(read_verified_artifact_bytes(nested_dir, "nested_result.joblib", binding))
    )
    identity = json.loads(
        read_verified_artifact_bytes(nested_dir, "frozen_model_identity.json", binding).decode(
            "utf-8"
        )
    )
    snapshot = load_development_snapshot(DEVELOPMENT_SNAPSHOT)
    development = snapshot.features.copy()
    years = pd.to_numeric(development["Year"], errors="raise")
    if int(years.min()) != 1930 or int(years.max()) != 2010:
        raise RuntimeError("Evaluation may use development editions 1930-2010 only")
    if bool(years.gt(2010).any()):
        raise RuntimeError("Audit-edition rows are forbidden during evaluation")
    observed_frame_hash = frame_content_hash(development)
    snapshot_manifest_hash = snapshot.manifest_sha256
    if (
        observed_frame_hash != identity["development_frame_hash"]
        or snapshot.development_frame_hash != identity["development_frame_hash"]
        or snapshot.feature_schema_hash != identity["feature_schema_hash"]
        or snapshot_manifest_hash != identity["development_snapshot_manifest_sha256"]
    ):
        raise RuntimeError("Development feature content no longer matches frozen identity")
    protocol = _protocol_lock()
    if identity["protocol_hash"] != protocol["protocol_sha256"]:
        raise RuntimeError("Frozen model identity does not match the protocol lock")
    return nested, development, identity


def _load_locked_nested_joblib(output: Path, relative_path: str) -> Any:
    """Deserialize the exact nested bytes covered by a fresh checkpoint verification."""
    checkpoint = checkpoint_status(output, "nested")
    if checkpoint["status"] != "complete":
        raise RuntimeError("Nested artifacts must be manifest-verified before deserialization")
    payload = read_verified_artifact_bytes(output / "nested", relative_path, checkpoint["manifest"])
    return joblib.load(io.BytesIO(payload))


def _combined_fold_table(
    baseline: Any,
    raw_evidence: Any,
    calibrated_evidence: Any,
) -> pd.DataFrame:
    baseline_outcome = baseline.outcome_fold_metrics.copy()
    baseline_joint = baseline.joint_fold_metrics.copy()
    baseline_rows = baseline_outcome.merge(
        baseline_joint,
        on=["outer_fold_id", "test_year", "model", "runtime_seconds"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_joint"),
    )
    candidate_rows: list[pd.DataFrame] = []
    for name, evidence in (
        ("advanced_nested_raw", raw_evidence),
        ("advanced_nested_calibrated", calibrated_evidence),
    ):
        frame = evidence.fold_metrics.copy()
        frame["model"] = name
        candidate_rows.append(frame)
    combined = pd.concat([baseline_rows, *candidate_rows], ignore_index=True, sort=False)
    return combined.sort_values(["test_year", "model"]).reset_index(drop=True)


def _comparison_certificates(folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = "advanced_nested_calibrated"
    baseline = "legacy_mirrored_dixon_coles"
    common_metrics = (
        "log_loss",
        "multiclass_brier",
        "scoreline_log_loss",
        "total_goal_crps",
    )
    interval_rows: list[dict[str, Any]] = []
    robust_rows: list[dict[str, Any]] = []
    for metric in common_metrics:
        if metric not in folds or folds.loc[folds["model"].eq(baseline), metric].isna().any():
            raise RuntimeError(f"Required paired proper score is unavailable: {metric}")
        interval_rows.append(
            paired_tournament_cluster_interval(
                folds,
                candidate=candidate,
                baseline=baseline,
                metric=metric,
                replicates=2_000,
                seed=17_290 + common_metrics.index(metric),
            )
        )
        for model in (candidate, baseline):
            model_folds = folds.loc[folds["model"].eq(model)].sort_values("test_year")
            losses = model_folds[metric].to_numpy(float)
            years = model_folds["test_year"].astype(int).tolist()
            if len(losses) != 7 or not np.isfinite(losses).all():
                raise RuntimeError("Robust-risk sensitivity requires seven finite edition losses")
            for epsilon in (0.0, 0.05, 0.2):
                result = kl_edition_robust_risk(losses, epsilon=epsilon)
                robust_rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "epsilon": epsilon,
                        "nominal_risk": result["nominal_risk"],
                        "robust_risk": result["robust_risk"],
                        "realized_kl": result["realized_kl"],
                        "dual_eta": result["dual_eta"],
                        "edition_years_json": json.dumps(years),
                        "edition_weights_json": json.dumps(result["weights"]),
                        "interpretation": result["interpretation"],
                        "radius_role": "fixed_sensitivity_grid_not_selected_hyperparameter",
                    }
                )
    return pd.DataFrame(interval_rows), pd.DataFrame(robust_rows)


def _future_stress_probe(development: pd.DataFrame, *, rows: int = 8) -> pd.DataFrame:
    latest = development.loc[development["Year"].eq(2010)].head(rows).copy()
    if len(latest) < 2:
        raise RuntimeError("A measured stress probe requires at least two 2010 matches")
    latest["Year"] = 2026
    latest["half_time_home_goals"] = 0
    latest["half_time_away_goals"] = 0
    return latest.reset_index(drop=True)


def _stress_refitter(fitted: Any) -> Any:
    if bool(fitted.residual_applied):
        raise RuntimeError(
            "The selected residual model requires its OOF residual lineage during stress refits"
        )
    base_config = fitted.model_config

    def refit(frame: pd.DataFrame, overrides: dict[str, Any], seed: int) -> Any:
        del seed
        supported: dict[str, Any] = {}
        if "time_decay_half_life_years" in overrides:
            supported["time_decay_half_life_years"] = float(overrides["time_decay_half_life_years"])
        config = replace(base_config, **supported)
        return EquivariantCountModel(config).fit(frame)

    return refit


def _halftime_increment_checker(fitted: Any) -> Any:
    config = fitted.model_config

    def checker(frame: pd.DataFrame, probabilities: np.ndarray, seed: int) -> np.ndarray:
        del probabilities, seed
        rates = 0.5 * np.asarray(fitted.predict_rates(frame), dtype=float)
        increment = joint_batch_from_rates(
            rates,
            shared_fraction=float(fitted.shared_fraction_),
            max_score=config.max_score,
            min_retained_mass=config.min_retained_mass,
            max_support=config.max_support,
        )
        home = pd.to_numeric(frame["half_time_home_goals"], errors="raise").to_numpy(int)
        away = pd.to_numeric(frame["half_time_away_goals"], errors="raise").to_numpy(int)
        support = increment.probability.shape[1]
        result = np.zeros(
            (len(frame), int(home.max()) + support, int(away.max()) + support),
            dtype=float,
        )
        for index, (home_minimum, away_minimum) in enumerate(zip(home, away, strict=True)):
            result[
                index,
                home_minimum : home_minimum + support,
                away_minimum : away_minimum + support,
            ] = increment.probability[index]
        return result

    return checker


def _extreme_support_boundary_audit(
    fitted: Any, development: pd.DataFrame, probe: pd.DataFrame
) -> dict[str, Any]:
    fragments = ("history", "goal", "elo", "attack", "defen")
    model_features = [*fitted.odd_columns_, *fitted.even_columns_]
    columns = [
        name
        for name in model_features
        if any(fragment in name.casefold() for fragment in fragments)
    ]
    stressed = probe.copy(deep=True)
    assigned: dict[str, float] = {}
    for position, column in enumerate(columns):
        values = pd.to_numeric(development[column], errors="coerce").dropna().to_numpy(float)
        span = max(float(np.ptp(values)), float(np.std(values)), 1.0)
        assigned[column] = float((values.max() + 10.0 * span) * (-1) ** position)
        stressed[column] = assigned[column]
    rates = np.asarray(fitted.predict_rates(stressed), dtype=float)
    refused = False
    error_type: str | None = None
    error_message: str | None = None
    try:
        fitted.predict_joint(stressed)
    except RuntimeError as error:
        refused = True
        error_type = type(error).__name__
        error_message = str(error)
    return {
        "scenario": "ten_training_spans_beyond_maximum",
        "feature_values": assigned,
        "minimum_rate": float(rates.min()),
        "maximum_rate": float(rates.max()),
        "declared_maximum_score_support": int(fitted.model_config.max_support),
        "safe_refusal_observed": refused,
        "error_type": error_type,
        "error_message": error_message,
        "interpretation": ("explicit_out_of_support_refusal_not_a_valid_probability_forecast"),
    }


def _summarize_ablations(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["outer_fold_id", "test_year", "seed"]
    a0 = matrix.loc[matrix["variant_id"].eq("A0"), keys + ["log_loss", "multiclass_brier"]].rename(
        columns={
            "log_loss": "a0_log_loss",
            "multiclass_brier": "a0_multiclass_brier",
        }
    )
    paired = matrix.merge(a0, on=keys, how="left", validate="many_to_one")
    paired["delta_log_loss_vs_a0"] = paired["log_loss"] - paired["a0_log_loss"]
    paired["delta_brier_vs_a0"] = paired["multiclass_brier"] - paired["a0_multiclass_brier"]
    rows: list[dict[str, Any]] = []
    for variant, block in paired.groupby("variant_id", sort=True):
        rows.append(
            {
                "variant_id": variant,
                "removed_component": str(block["removed_component"].iloc[0]),
                "paired_evaluations": int(len(block)),
                "mean_log_loss": float(block["log_loss"].mean()),
                "mean_multiclass_brier": float(block["multiclass_brier"].mean()),
                "mean_delta_log_loss_vs_a0": float(block["delta_log_loss_vs_a0"].mean()),
                "mean_delta_brier_vs_a0": float(block["delta_brier_vs_a0"].mean()),
                "log_loss_improvement_count_vs_a0": int(block["delta_log_loss_vs_a0"].lt(0).sum()),
                "brier_improvement_count_vs_a0": int(block["delta_brier_vs_a0"].lt(0).sum()),
                "component_active_in_a0": bool(block["component_active_in_a0"].all()),
                "zero_effect_by_design": bool(block["zero_effect_by_design"].all()),
                "structurally_eligible": bool(block["structurally_eligible"].all()),
                "maximum_symmetry_error": float(block["symmetry_error"].max()),
            }
        )
    return paired, pd.DataFrame(rows)


def _summarize_parameter_draws(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for match_id, block in predictions.groupby("match_id", sort=True):
        row: dict[str, Any] = {
            "match_id": match_id,
            "successful_parameter_draws": int(block["bootstrap_draw_id"].nunique()),
        }
        for outcome in ("home_win", "draw", "away_win"):
            values = block[outcome].to_numpy(float)
            row.update(
                {
                    f"{outcome}_mean": float(values.mean()),
                    f"{outcome}_sd": float(values.std(ddof=1)),
                    f"{outcome}_q05": float(np.quantile(values, 0.05)),
                    f"{outcome}_q50": float(np.quantile(values, 0.50)),
                    f"{outcome}_q95": float(np.quantile(values, 0.95)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _simulation_stage_summary(match_log: pd.DataFrame) -> pd.DataFrame:
    grouped = match_log.groupby("stage", sort=False)
    rows: list[dict[str, Any]] = []
    for stage, block in grouped:
        home = block["home_goals"].to_numpy(int)
        away = block["away_goals"].to_numpy(int)
        rows.append(
            {
                "stage": stage,
                "simulated_matches": int(len(block)),
                "mean_total_goals": float(np.mean(home + away)),
                "mean_home_goals": float(np.mean(home)),
                "mean_away_goals": float(np.mean(away)),
                "regulation_draw_rate": float(np.mean(home == away)),
                "zero_goal_rate": float(np.mean((home + away) == 0)),
            }
        )
    return pd.DataFrame(rows)


def _regime_tilted_provider(provider: Any, log_total_goal_tilt: float) -> Any:
    cache: dict[tuple[str, str, str], np.ndarray] = {}

    def tilted(home: str, away: str, stage: str) -> np.ndarray:
        key = (home, away, stage)
        if key in cache:
            return cache[key]
        base = np.asarray(provider.score_provider(home, away, stage), dtype=float)
        coordinates = np.add.outer(np.arange(base.shape[0]), np.arange(base.shape[1]))
        log_weight = float(log_total_goal_tilt) * coordinates
        log_weight -= float(log_weight.max())
        result = base * np.exp(log_weight)
        result /= result.sum()
        result.setflags(write=False)
        cache[key] = result
        return result

    return tilted


def _rebuild_bootstrap_sample(development: pd.DataFrame, sampled_editions: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for occurrence, token in enumerate(str(sampled_editions).split("|")):
        year = int(token)
        block = development.loc[development["Year"].eq(year)].copy()
        if block.empty:
            raise RuntimeError(f"Bootstrap ledger references unavailable edition {year}")
        block["bootstrap_cluster_occurrence"] = occurrence
        parts.append(block)
    return pd.concat(parts, ignore_index=True)


def run_nested_stage(output: Path) -> dict[str, Any]:
    """Run the 36-candidate nested study and freeze the final serving identity."""
    protocol = _protocol_lock()
    started = time.perf_counter()
    snapshot = load_development_snapshot(DEVELOPMENT_SNAPSHOT)
    development = snapshot.features.copy()
    snapshot_manifest_hash = snapshot.manifest_sha256
    if int(development["Year"].max()) != 2010 or int(development["Year"].min()) != 1930:
        raise RuntimeError("Development feature boundary is not exactly 1930-2010")
    nested = run_nested_selection(
        development,
        feature_schema_hash=snapshot.feature_schema_hash,
    )
    final_selection = select_final_development_spec(
        development,
        feature_schema_hash=snapshot.feature_schema_hash,
    )
    fitted = fit_final_selected_model(development, final_selection)
    nested, final_selection, fitted = seal_nested_runtime_evidence(
        nested,
        final_selection,
        fitted,
    )

    nested_dir = output / "nested"
    for name, frame in (
        ("selected_specs", nested.selected_specs),
        ("candidate_metrics", nested.candidate_metrics),
        ("inner_fold_metrics", nested.inner_fold_metrics),
        ("selected_inner_oof_ledger", nested.selected_inner_oof_ledger),
        ("selected_inner_joint_grid", nested.selected_inner_joint_grid),
        ("outer_oof_ledger", nested.oof_ledger),
        ("outer_rate_ledger", nested.rate_ledger),
        ("outer_joint_grid", nested.joint_grid),
        ("final_selected_summary", final_selection.selected_summary),
        ("final_candidate_metrics", final_selection.candidate_metrics),
        ("final_inner_fold_metrics", final_selection.inner_fold_metrics),
        ("final_inner_oof_ledger", final_selection.selected_inner_oof_ledger),
        ("final_inner_joint_grid", final_selection.selected_inner_joint_grid),
    ):
        write_csv(frame, nested_dir / f"{name}.csv")
    write_json(nested.metadata, nested_dir / "nested_metadata.json")
    write_json(final_selection.metadata, nested_dir / "final_selection_metadata.json")
    write_json(
        {
            **RUNTIME_REFERENCE_PROTOCOL,
            "runtime_reference_protocol_sha256": RUNTIME_REFERENCE_PROTOCOL_SHA256,
            "candidate_runtime_reference_hash": nested.metadata["runtime_reference_hash"],
            "final_fit_runtime_reference_seconds": fitted.fit_runtime_seconds,
        },
        nested_dir / "runtime_reproducibility.json",
    )
    _dump_joblib(nested, nested_dir / "nested_result.joblib")
    _dump_joblib(final_selection, nested_dir / "final_selection.joblib")
    _dump_joblib(fitted, nested_dir / "final_fitted_model.joblib")
    _dump_joblib(development, nested_dir / "development_features.joblib")

    selection_procedure = {
        "protocol_hash": protocol["protocol_sha256"],
        "feature_schema_hash": snapshot.feature_schema_hash,
        "development_snapshot_manifest_sha256": snapshot_manifest_hash,
        "candidate_budget_manifest_hash": nested.metadata["candidate_budget_manifest_hash"],
        "runtime_reference_hash": nested.metadata["runtime_reference_hash"],
        "selection_objective": nested.metadata["selection_objective"],
        "latest_inner_folds": nested.metadata["latest_inner_folds"],
        "outer_test_years": nested.metadata["outer_test_years"],
        "calibration_fit_source": nested.metadata["calibration_fit_source"],
    }
    identity = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_hash": protocol["protocol_sha256"],
        "feature_schema_hash": snapshot.feature_schema_hash,
        "development_snapshot_manifest_sha256": snapshot_manifest_hash,
        "selection_procedure_hash": _sha256_json(selection_procedure),
        "selection_procedure": selection_procedure,
        "selected_candidate_hash": final_selection.selected_candidate_hash,
        "selected_config_hash": final_selection.selected_config_hash,
        "final_fit_hash": fitted.final_fit_hash,
        "development_frame_hash": fitted.development_frame_hash,
        "fit_end_year": fitted.fit_end_year,
        "audit_year": 2014,
        "audit_outcomes_used_for_selection": False,
        "human_blinding_disclosure": protocol["human_blinding_disclosure"],
        "formal_preaudit_reads_combined_raw": False,
    }
    write_json(identity, nested_dir / "frozen_model_identity.json")
    manifest = _artifact_manifest(nested_dir)
    write_json(manifest, nested_dir / "artifact_manifest.json")
    summary = {
        "status": "nested_stage_complete_and_locked",
        "development_rows": int(len(development)),
        "outer_oof_rows": int(len(nested.oof_ledger)),
        "eligible_candidate_rows": int(nested.candidate_metrics["eligible"].sum()),
        "selected_outer_candidate_ids": nested.selected_specs["candidate_id"].tolist(),
        "final_candidate_id": final_selection.selected_candidate_hash,
        "final_fit_hash": fitted.final_fit_hash,
        "identity_path": str(nested_dir / "frozen_model_identity.json"),
        "artifact_count": int(len(manifest)),
        "development_snapshot_manifest_sha256": snapshot_manifest_hash,
        "formal_preaudit_reads_combined_raw": False,
    }
    return _persist_deterministic_stage_summary(
        summary, output / "nested_stage_summary.json", started=started
    )


def run_evaluation_stage(output: Path) -> dict[str, Any]:
    """Evaluate the frozen nested procedure without opening audit-edition rows."""
    started = time.perf_counter()
    nested, development, identity = _load_locked_nested(output)
    baseline = seal_baseline_runtime_evidence(run_baseline_suite(development))
    raw_evidence = summarize_oof_evidence(
        nested.oof_ledger,
        nested.joint_grid,
        pit_seed=31_415,
    )
    calibration = apply_nested_calibration(
        nested.selected_inner_oof_ledger,
        nested.selected_inner_joint_grid,
        nested.oof_ledger,
        nested.joint_grid,
    )
    calibrated_evidence = summarize_oof_evidence(
        calibration.oof_ledger,
        calibration.joint_grid,
        pit_seed=31_415,
    )
    sarmanov = apply_nested_sarmanov_exploration(
        nested.selected_inner_oof_ledger,
        nested.selected_inner_joint_grid,
        nested.oof_ledger,
        nested.joint_grid,
    )
    sarmanov_evidence = summarize_oof_evidence(
        sarmanov.oof_ledger,
        sarmanov.joint_grid,
        pit_seed=31_415,
    )
    decision = apply_nested_decision(
        calibration.decision_training_ledger,
        calibration.oof_ledger,
    )
    fold_table = _combined_fold_table(baseline, raw_evidence, calibrated_evidence)
    paired, robust = _comparison_certificates(fold_table)
    statistical_closure = build_statistical_closure(
        fold_table,
        paired,
        nested.selected_specs,
        raw_evidence.classwise_calibration,
        calibrated_evidence.classwise_calibration,
    )

    evaluation_dir = output / "evaluation"
    tables = {
        "baseline_outcome_fold_metrics": baseline.outcome_fold_metrics,
        "baseline_joint_fold_metrics": baseline.joint_fold_metrics,
        "baseline_prediction_ledger": baseline.prediction_ledger,
        "baseline_joint_grid": baseline.joint_grid,
        "raw_fold_metrics": raw_evidence.fold_metrics,
        "raw_match_losses": raw_evidence.match_losses,
        "raw_classwise_calibration": raw_evidence.classwise_calibration,
        "calibrated_oof_ledger": calibration.oof_ledger,
        "calibrated_joint_grid": calibration.joint_grid,
        "calibration_selection": calibration.selection_table,
        "calibration_decision_training_ledger": calibration.decision_training_ledger,
        "calibrated_fold_metrics": calibrated_evidence.fold_metrics,
        "calibrated_match_losses": calibrated_evidence.match_losses,
        "calibrated_classwise_calibration": calibrated_evidence.classwise_calibration,
        "sarmanov_oof_ledger": sarmanov.oof_ledger,
        "sarmanov_joint_grid": sarmanov.joint_grid,
        "sarmanov_selection": sarmanov.selection_table,
        "sarmanov_fold_metrics": sarmanov_evidence.fold_metrics,
        "sarmanov_match_losses": sarmanov_evidence.match_losses,
        "sarmanov_classwise_calibration": sarmanov_evidence.classwise_calibration,
        "decision_prediction_ledger": decision.prediction_ledger,
        "decision_fold_metrics": decision.fold_metrics,
        "combined_fold_metrics": fold_table,
        "paired_tournament_intervals": paired,
        "kl_edition_robust_sensitivity": robust,
        "statistical_fold_comparison": statistical_closure.fold_comparison,
        "statistical_model_ranking": statistical_closure.model_ranking,
        "statistical_classwise_calibration": statistical_closure.classwise_calibration,
    }
    for name, frame in tables.items():
        write_csv(frame, evaluation_dir / f"{name}.csv")
    write_json(baseline.metadata, evaluation_dir / "baseline_metadata.json")
    write_json(raw_evidence.metadata, evaluation_dir / "raw_evidence_metadata.json")
    write_json(calibration.metadata, evaluation_dir / "calibration_metadata.json")
    write_json(calibration.states, evaluation_dir / "calibration_states.json")
    write_json(
        calibrated_evidence.metadata,
        evaluation_dir / "calibrated_evidence_metadata.json",
    )
    write_json(decision.metadata, evaluation_dir / "decision_metadata.json")
    write_json(decision.states, evaluation_dir / "decision_states.json")
    write_json(sarmanov.metadata, evaluation_dir / "sarmanov_metadata.json")
    write_json(sarmanov.states, evaluation_dir / "sarmanov_states.json")
    write_json(
        sarmanov_evidence.metadata,
        evaluation_dir / "sarmanov_evidence_metadata.json",
    )
    write_json(statistical_closure.summary, evaluation_dir / "statistical_closure.json")
    _dump_joblib(baseline, evaluation_dir / "baseline_result.joblib")
    _dump_joblib(calibration, evaluation_dir / "calibration_result.joblib")
    _dump_joblib(decision, evaluation_dir / "decision_result.joblib")
    _dump_joblib(sarmanov, evaluation_dir / "sarmanov_result.joblib")

    lineage = {
        "schema_version": 1,
        "selection_procedure_hash": identity["selection_procedure_hash"],
        "development_frame_hash": identity["development_frame_hash"],
        "feature_schema_hash": identity["feature_schema_hash"],
        "protocol_hash": identity["protocol_hash"],
        "maximum_input_year": int(development["Year"].max()),
        "audit_year_loaded": False,
        "calibration_fit_scope": calibration.metadata["calibration_fit_scope"],
        "decision_layer": "point_action_only_probabilities_unchanged",
        "sarmanov_role": "exploratory_contradiction_benchmark_not_promotion_eligible",
        "sarmanov_fit_scope": sarmanov.metadata["fit_scope"],
        "sarmanov_audit_year_used": False,
        "cluster_bootstrap_replicates": 2_000,
        "cluster_bootstrap_unit": "tournament_edition",
        "kl_radii": [0.0, 0.05, 0.2],
        "kl_radius_role": "fixed_sensitivity_grid_not_selected_hyperparameter",
    }
    write_json(lineage, evaluation_dir / "evaluation_lineage.json")
    manifest = _artifact_manifest(evaluation_dir)
    write_json(manifest, evaluation_dir / "artifact_manifest.json")
    summary = {
        "status": "evaluation_stage_complete_without_audit_data",
        "development_rows": int(len(development)),
        "outer_editions": int(nested.oof_ledger["prediction_year"].nunique()),
        "calibration_methods": calibration.selection_table["chosen_method"].tolist(),
        "decision_draw_positive_folds": decision.metadata["draw_positive_fold_count"],
        "paired_certificate_count": int(len(paired)),
        "robust_sensitivity_rows": int(len(robust)),
        "sarmanov_theta_by_outer_fold": {
            str(row.selection_outer_fold_id): float(row.theta)
            for row in sarmanov.selection_table.itertuples(index=False)
        },
        "sarmanov_positive_theta_folds": int(sarmanov.selection_table["theta"].gt(0).sum()),
        "sarmanov_role": "exploratory_only_not_promotion_eligible",
        "artifact_count": int(len(manifest)),
        "statistical_closure_passed": statistical_closure.summary["passed"],
        "audit_year_loaded": False,
    }
    return _persist_deterministic_stage_summary(
        summary, output / "evaluation_stage_summary.json", started=started
    )


def run_stress_stage(output: Path) -> dict[str, Any]:
    """Run structural, deletion, recency, and historical format stresses."""
    started = time.perf_counter()
    _, development, identity = _load_locked_nested(output)
    fitted_path = output / "nested" / "final_fitted_model.joblib"
    if not fitted_path.is_file():
        raise FileNotFoundError("Frozen final fitted model is missing")
    fitted = _load_locked_nested_joblib(output, "final_fitted_model.joblib")
    if fitted.final_fit_hash != identity["final_fit_hash"]:
        raise RuntimeError("Stress model does not match the frozen fit identity")
    probe = _future_stress_probe(development)
    support_boundary = _extreme_support_boundary_audit(fitted, development, probe)
    if not support_boundary["safe_refusal_observed"]:
        raise RuntimeError("Ten-span support-boundary audit unexpectedly produced a PMF")
    stress = run_measured_stress_tests(
        fitted_model=fitted,
        training_frame=development,
        probe_frame=probe,
        model_id=identity["final_fit_hash"],
        seeds=(17, 29),
        refit_callback=_stress_refitter(fitted),
        half_time_checker=_halftime_increment_checker(fitted),
        maximum_training_year=2010,
    )
    snapshot = load_development_snapshot(DEVELOPMENT_SNAPSHOT)
    if snapshot.development_frame_hash != identity["development_frame_hash"]:
        raise RuntimeError("Stress snapshot does not match the frozen development identity")
    transport = run_format_transport_audits(snapshot.tournaments)

    stress_dir = output / "stress"
    write_csv(probe, stress_dir / "future_probe.csv")
    write_csv(stress, stress_dir / "structural_stress_matrix.csv")
    write_csv(transport.audits, stress_dir / "format_transport_audits.csv")
    write_json(support_boundary, stress_dir / "extreme_support_boundary.json")
    write_json(transport.summary, stress_dir / "format_transport_summary.json")
    structural_pass = bool(stress["structural_tests_passed"].all())
    lineage = {
        "schema_version": 1,
        "final_fit_hash": identity["final_fit_hash"],
        "selection_procedure_hash": identity["selection_procedure_hash"],
        "protocol_hash": identity["protocol_hash"],
        "training_maximum_year": int(development["Year"].max()),
        "probe_declared_year": int(probe["Year"].max()),
        "frozen_seeds": [17, 29],
        "halftime_kernel": (
            "shifted_bivariate_poisson_second_half_sensitivity_at_half_structural_rates"
        ),
        "halftime_kernel_claim_scope": "support_monotonicity_test_not_calibrated_ht_forecast",
        "extreme_invariant_shift": "five_training_spans_beyond_maximum",
        "extreme_boundary_refusal_shift": "ten_training_spans_beyond_maximum",
        "format_transport_claim_scope": transport.summary["claim_scope"],
        "audit_outcomes_used": False,
    }
    write_json(lineage, stress_dir / "stress_lineage.json")
    manifest = _artifact_manifest(stress_dir)
    write_json(manifest, stress_dir / "artifact_manifest.json")
    summary = {
        "status": "stress_stage_complete",
        "stress_rows": int(len(stress)),
        "scenario_count": int(stress["scenario_id"].nunique()),
        "seed_count": int(stress["seed"].nunique()),
        "all_structural_checks_passed": structural_pass,
        "maximum_symmetry_error": float(stress["symmetry_error"].max()),
        "maximum_simplex_error": float(stress["simplex_error"].max()),
        "maximum_halftime_violation": float(stress["halftime_violation"].max()),
        "ten_span_safe_refusal_observed": bool(support_boundary["safe_refusal_observed"]),
        "format_transport": transport.summary,
        "audit_outcomes_used": False,
    }
    return _persist_deterministic_stage_summary(
        summary, output / "stress_stage_summary.json", started=started
    )


def run_ablation_stage(output: Path) -> dict[str, Any]:
    """Run the complete paired A0--A8 matrix on frozen development folds."""
    started = time.perf_counter()
    nested, development, identity = _load_locked_nested(output)
    result = seal_ablation_runtime_evidence(
        run_measured_ablations(
            development,
            nested.selected_specs,
            feature_schema_hash=identity["feature_schema_hash"],
        )
    )
    paired, summary_table = _summarize_ablations(result.matrix)
    ablation_dir = output / "ablation"
    write_csv(result.matrix, ablation_dir / "ablation_matrix.csv")
    write_csv(result.predictions, ablation_dir / "ablation_predictions.csv")
    write_csv(paired, ablation_dir / "ablation_paired_deltas.csv")
    write_csv(summary_table, ablation_dir / "ablation_summary.csv")
    write_json(result.metadata, ablation_dir / "ablation_metadata.json")
    lineage = {
        "schema_version": 1,
        "selection_procedure_hash": identity["selection_procedure_hash"],
        "development_frame_hash": identity["development_frame_hash"],
        "feature_schema_hash": identity["feature_schema_hash"],
        "protocol_hash": identity["protocol_hash"],
        "training_maximum_year": int(development["Year"].max()),
        "seeds": [17, 29],
        "audit_outcomes_used": False,
        "a7_scope": (
            "exact_zero_team_block_l2_with_full_rank_sum_to_zero_identifiability_"
            "and_context_penalty_held_fixed"
        ),
        "a8_scope": "non_equivariant_diagnostic_structurally_ineligible",
    }
    write_json(lineage, ablation_dir / "ablation_lineage.json")
    manifest = _artifact_manifest(ablation_dir)
    write_json(manifest, ablation_dir / "artifact_manifest.json")
    summary = {
        "status": "ablation_stage_complete_without_audit_data",
        "matrix_rows": int(len(result.matrix)),
        "prediction_rows": int(len(result.predictions)),
        "variants": result.matrix["variant_id"].drop_duplicates().tolist(),
        "outer_editions": int(result.matrix["test_year"].nunique()),
        "seeds": result.matrix["seed"].drop_duplicates().astype(int).tolist(),
        "maximum_a0_a7_symmetry_error": float(
            result.matrix.loc[result.matrix["variant_id"].ne("A8"), "symmetry_error"].max()
        ),
        "a8_maximum_empirical_symmetry_error": float(
            result.matrix.loc[result.matrix["variant_id"].eq("A8"), "symmetry_error"].max()
        ),
        "audit_outcomes_used": False,
    }
    return _persist_deterministic_stage_summary(
        summary, output / "ablation_stage_summary.json", started=started
    )


def run_parameter_uncertainty_stage(output: Path) -> dict[str, Any]:
    """Estimate whole-tournament resampling sensitivity of frozen predictions."""
    started = time.perf_counter()
    _, development, identity = _load_locked_nested(output)
    fitted = _load_locked_nested_joblib(output, "final_fitted_model.joblib")
    if fitted.final_fit_hash != identity["final_fit_hash"]:
        raise RuntimeError("Parameter-sensitivity model identity mismatch")
    if bool(fitted.residual_applied):
        raise RuntimeError(
            "Residual-selected bootstrap refits require cross-fitted residual reconstruction"
        )
    probe = _future_stress_probe(development, rows=8)
    model_config = fitted.model_config

    def refit(frame: pd.DataFrame, seed: int) -> Any:
        del seed
        return EquivariantCountModel(model_config).fit(frame)

    config = TournamentBootstrapConfig(replicates=200, seed=1_729)
    result = run_tournament_cluster_refits(
        development,
        probe,
        refit,
        config=config,
    )
    draw_summary = _summarize_parameter_draws(result.predictions)
    uncertainty_dir = output / "uncertainty"
    write_csv(probe, uncertainty_dir / "parameter_probe.csv")
    write_csv(result.predictions, uncertainty_dir / "parameter_probability_draws.csv")
    write_csv(result.draw_ledger, uncertainty_dir / "parameter_draw_ledger.csv")
    write_csv(result.failures, uncertainty_dir / "parameter_failures.csv")
    write_csv(draw_summary, uncertainty_dir / "parameter_probability_summary.csv")
    metadata = {
        **result.metadata,
        "final_fit_hash": identity["final_fit_hash"],
        "selection_procedure_hash": identity["selection_procedure_hash"],
        "development_frame_hash": identity["development_frame_hash"],
        "feature_schema_hash": identity["feature_schema_hash"],
        "protocol_hash": identity["protocol_hash"],
        "probe_declared_year": 2026,
        "probe_profile_source": "first_eight_2010_profiles_relabelled_as_scenario_only",
        "audit_outcomes_used": False,
    }
    write_json(metadata, uncertainty_dir / "parameter_metadata.json")
    parameter_paths = sorted(
        path.name
        for path in uncertainty_dir.glob("parameter_*")
        if path.is_file() and not path.name.endswith("artifact_manifest.json")
    )
    manifest = _artifact_manifest(uncertainty_dir, relative_paths=parameter_paths)
    write_json(manifest, uncertainty_dir / "parameter_artifact_manifest.json")
    summary = {
        "status": "parameter_uncertainty_stage_complete",
        "replicates_requested": result.metadata["replicates_requested"],
        "replicates_succeeded": result.metadata["replicates_succeeded"],
        "replicates_failed": result.metadata["replicates_failed"],
        "probe_matches": int(result.predictions["match_id"].nunique()),
        "maximum_probability_sd": float(draw_summary.filter(regex=r"_sd$").to_numpy(float).max()),
        "audit_outcomes_used": False,
    }
    return _persist_deterministic_stage_summary(
        summary, output / "parameter_uncertainty_stage_summary.json", started=started
    )


def run_simulation_stage(output: Path) -> dict[str, Any]:
    """Run the official-format scenario under the adaptive MCSE contract."""
    started = time.perf_counter()
    _, development, identity = _load_locked_nested(output)
    fitted = _load_locked_nested_joblib(output, "final_fitted_model.joblib")
    if fitted.final_fit_hash != identity["final_fit_hash"]:
        raise RuntimeError("Simulation model identity mismatch")
    provider = build_format_scenario_provider(
        fitted,
        development,
        profile_cutoff_year=2010,
        precompute_stages=FORMAT_STAGES,
    )
    tournament_format = TournamentFormat.world_cup_2026()
    observed_deviation_hash = _file_sha256(ROOT / "plan" / "PROTOCOL_DEVIATIONS.md")
    if observed_deviation_hash != FROZEN_PROTOCOL_DEVIATION_SHA256:
        raise RuntimeError("Adaptive contract no longer matches the frozen deviation record")
    adaptive_config = FROZEN_ADAPTIVE_CONFIG
    result = simulate_tournament_adaptive(
        teams=provider.teams,
        tournament_format=tournament_format,
        score_provider=provider.score_provider,
        seed=FROZEN_ADAPTIVE_SEED,
        config=adaptive_config,
    )
    simulation_dir = output / "simulation"
    write_csv(result.match_log, simulation_dir / "scenario_match_log.csv.gz")
    write_csv(result.team_summary, simulation_dir / "scenario_team_summary.csv")
    write_csv(provider.profile_table, simulation_dir / "scenario_team_profiles.csv")
    write_csv(result.stage_summary, simulation_dir / "scenario_stage_summary.csv")
    write_csv(result.stopping_ledger, simulation_dir / "adaptive_stopping_ledger.csv")
    write_csv(
        result.batch_event_counts,
        simulation_dir / "adaptive_batch_event_counts.csv.gz",
    )
    write_csv(
        result.continuous_sufficient_statistics,
        simulation_dir / "continuous_sufficient_statistics.csv",
    )
    metadata = {
        **result.metadata,
        **provider.metadata,
        "final_fit_hash": identity["final_fit_hash"],
        "selection_procedure_hash": identity["selection_procedure_hash"],
        "feature_schema_hash": identity["feature_schema_hash"],
        "protocol_hash": identity["protocol_hash"],
        "audit_outcomes_used": False,
        "factual_2026_entrants_used": False,
        "claim_scope": "format_mechanics_and_model_transport_sensitivity_only",
    }
    write_json(metadata, simulation_dir / "scenario_metadata.json")
    manifest = _artifact_manifest(simulation_dir)
    write_json(manifest, simulation_dir / "artifact_manifest.json")
    summary = {
        "status": "illustrative_2026_format_scenario_complete",
        "simulations": int(result.metadata["n_simulations"]),
        "minimum_simulations": adaptive_config.minimum_simulations,
        "maximum_simulations": adaptive_config.maximum_simulations,
        "batch_size": adaptive_config.batch_size,
        "advancement_mcse_threshold": adaptive_config.advancement_mcse_threshold,
        "champion_mcse_threshold": adaptive_config.champion_mcse_threshold,
        "maximum_advancement_mcse": result.metadata["maximum_advancement_mcse"],
        "maximum_champion_mcse": result.metadata["maximum_champion_mcse"],
        "mcse_thresholds_satisfied": result.metadata["mcse_thresholds_satisfied"],
        "stopping_reason": result.metadata["stopping_reason"],
        "matches_per_simulation": tournament_format.matches_per_simulation,
        "simulated_match_rows": int(
            result.metadata["n_simulations"] * tournament_format.matches_per_simulation
        ),
        "audit_log_simulations": adaptive_config.audit_log_simulations,
        "confidence_level": adaptive_config.confidence_level,
        "seed": FROZEN_ADAPTIVE_SEED,
        "adaptive_contract_sha256": adaptive_contract_sha256(),
        "protocol_deviation_sha256": FROZEN_PROTOCOL_DEVIATION_SHA256,
        "stopping_ledger_rows": int(len(result.stopping_ledger)),
        "batch_event_count_rows": int(len(result.batch_event_counts)),
        "continuous_sufficient_statistic_rows": int(len(result.continuous_sufficient_statistics)),
        "match_log_rows": int(len(result.match_log)),
        "champion_probability_sum": float(result.team_summary["champion_probability"].sum()),
        "mean_tournament_goals": result.metadata["mean_tournament_goals"],
        "mean_tournament_goals_mcse": result.metadata["mean_tournament_goals_mcse"],
        "scenario_only_not_factual_forecast": True,
    }
    return _persist_deterministic_stage_summary(
        summary, output / "simulation_stage_summary.json", started=started
    )


def run_variance_decomposition_stage(output: Path) -> dict[str, Any]:
    """Measure parameter/regime/path variance for total tournament goals."""
    started = time.perf_counter()
    _, development, identity = _load_locked_nested(output)
    fitted = _load_locked_nested_joblib(output, "final_fitted_model.joblib")
    if bool(fitted.residual_applied):
        raise RuntimeError("Residual-selected variance refits require OOF reconstruction")
    parameter_checkpoint = checkpoint_status(output, "parameter")
    if parameter_checkpoint["status"] != "complete":
        raise RuntimeError("Parameter checkpoint must be verified before decomposition")
    simulation_checkpoint = checkpoint_status(output, "simulate")
    if simulation_checkpoint["status"] != "complete":
        raise RuntimeError("Replacement simulation must be verified before decomposition")
    simulation_manifest_sha256 = _file_sha256(output / "simulation" / "artifact_manifest.json")
    simulation_metadata = json.loads(
        read_verified_artifact_bytes(
            output / "simulation",
            "scenario_metadata.json",
            simulation_checkpoint["manifest"],
        ).decode("utf-8")
    )
    if (
        simulation_metadata.get("adaptive_contract_sha256") != adaptive_contract_sha256()
        or simulation_metadata.get("protocol_deviation_sha256")
        != FROZEN_PROTOCOL_DEVIATION_SHA256
        or simulation_metadata.get("format_source_sha256")
        != FIFA_2026_REGULATIONS_SHA256
        or simulation_metadata.get("annex_c_csv_sha256")
        != FIFA_2026_ANNEX_C_CSV_SHA256
    ):
        raise RuntimeError("Replacement simulation metadata violates the frozen lineage")
    expected_decomposition_lineage = {
        "adaptive_contract_sha256": adaptive_contract_sha256(),
        "protocol_deviation_sha256": FROZEN_PROTOCOL_DEVIATION_SHA256,
        "format_source_sha256": simulation_metadata["format_source_sha256"],
        "annex_c_csv_sha256": simulation_metadata["annex_c_csv_sha256"],
        "replacement_simulation_manifest_sha256": simulation_manifest_sha256,
    }
    ledger_path = output / "uncertainty" / "parameter_draw_ledger.csv"
    if not ledger_path.is_file():
        raise FileNotFoundError("Run the parameter uncertainty stage first")
    uncertainty_dir = output / "uncertainty"
    cube_path = uncertainty_dir / "variance_draw_cube.joblib"
    decomposition_path = uncertainty_dir / "variance_decomposition.json"
    if cube_path.is_file() and decomposition_path.is_file():
        verified = verify_stage_manifest(
            uncertainty_dir / "decomposition_artifact_manifest.json",
            required_paths=(
                "variance_draw_cube.joblib",
                "variance_cell_summary.csv",
                "variance_decomposition.json",
            ),
        )
        checkpoint_draws = np.asarray(
            joblib.load(
                io.BytesIO(
                    read_verified_artifact_bytes(
                        uncertainty_dir, "variance_draw_cube.joblib", verified
                    )
                )
            ),
            dtype=float,
        )
        checkpoint_payload = json.loads(
            read_verified_artifact_bytes(
                uncertainty_dir, "variance_decomposition.json", verified
            ).decode("utf-8")
        )
        checkpoint_identity = checkpoint_payload.get("metadata", {}).get("final_fit_hash")
        if checkpoint_draws.shape != (20, 3, 100):
            raise RuntimeError("Variance checkpoint has an invalid draw shape")
        if checkpoint_identity != identity["final_fit_hash"]:
            raise RuntimeError("Variance checkpoint belongs to a different frozen model")
        if any(
            checkpoint_payload.get("metadata", {}).get(key) != value
            for key, value in expected_decomposition_lineage.items()
        ):
            raise RuntimeError("Variance checkpoint belongs to a stale simulation contract")
        recomputed = hierarchical_variance_decomposition(checkpoint_draws)
        for name, value in recomputed.components.items():
            if not np.isclose(
                float(checkpoint_payload["components"][name]), value, rtol=0, atol=1e-10
            ):
                raise RuntimeError("Variance checkpoint failed component verification")
        summary = {
            "status": "variance_decomposition_stage_complete_checkpoint_verified",
            "draw_shape": list(checkpoint_draws.shape),
            "components": recomputed.components,
            "shares": recomputed.shares,
            "checkpoint_reused": True,
            "audit_outcomes_used": False,
        }
        return _persist_deterministic_stage_summary(
            summary,
            output / "variance_decomposition_stage_summary.json",
            started=started,
        )
    ledger = (
        pd.read_csv(
            io.BytesIO(
                read_verified_artifact_bytes(
                    uncertainty_dir,
                    "parameter_draw_ledger.csv",
                    parameter_checkpoint["manifest"],
                )
            )
        )
        .sort_values("bootstrap_draw_id")
        .head(20)
    )
    if len(ledger) != 20:
        raise RuntimeError("Variance decomposition requires 20 successful parameter draws")
    format_2026 = TournamentFormat.world_cup_2026()
    tilts = (-0.10, 0.0, 0.10)
    draws = np.empty((20, len(tilts), 100), dtype=float)
    parameter_rows: list[dict[str, Any]] = []
    for parameter_index, row in enumerate(ledger.itertuples(index=False)):
        sample = _rebuild_bootstrap_sample(development, row.sampled_editions)
        model = EquivariantCountModel(fitted.model_config).fit(sample)
        provider = build_format_scenario_provider(
            model,
            development,
            profile_cutoff_year=2010,
            precompute_stages=FORMAT_STAGES,
        )
        for regime_index, tilt in enumerate(tilts):
            simulation = simulate_tournament(
                teams=provider.teams,
                tournament_format=format_2026,
                score_provider=_regime_tilted_provider(provider, tilt),
                n_simulations=100,
                seed=91_000 + parameter_index * 100 + regime_index,
            )
            totals = (
                simulation.match_log.assign(
                    total_goals=lambda frame: frame["home_goals"] + frame["away_goals"]
                )
                .groupby("simulation_id")["total_goals"]
                .sum()
                .sort_index()
                .to_numpy(float)
            )
            draws[parameter_index, regime_index] = totals
            parameter_rows.append(
                {
                    "parameter_draw_index": parameter_index,
                    "bootstrap_draw_id": int(row.bootstrap_draw_id),
                    "sample_hash": str(row.sample_hash),
                    "regime_index": regime_index,
                    "log_total_goal_tilt": tilt,
                    "path_draws": int(len(totals)),
                    "mean_total_goals": float(totals.mean()),
                    "sd_total_goals": float(totals.std(ddof=1)),
                }
            )
    decomposition = hierarchical_variance_decomposition(draws)
    _dump_joblib(draws, uncertainty_dir / "variance_draw_cube.joblib")
    write_csv(
        pd.DataFrame(parameter_rows),
        uncertainty_dir / "variance_cell_summary.csv",
    )
    payload = {
        "components": decomposition.components,
        "shares": decomposition.shares,
        "metadata": {
            **decomposition.metadata,
            "functional": "total_goals_in_104_match_illustrative_tournament",
            "parameter_draws": 20,
            "regime_tilts": list(tilts),
            "paths_per_cell": 100,
            "final_fit_hash": identity["final_fit_hash"],
            "selection_procedure_hash": identity["selection_procedure_hash"],
            **expected_decomposition_lineage,
            "audit_outcomes_used": False,
            "claim_scope": "balanced_sensitivity_decomposition_not_future_coverage",
        },
    }
    write_json(payload, uncertainty_dir / "variance_decomposition.json")
    decomposition_paths = sorted(
        path.name
        for path in uncertainty_dir.glob("variance_*")
        if path.is_file() and not path.name.endswith("artifact_manifest.json")
    )
    manifest = _artifact_manifest(uncertainty_dir, relative_paths=decomposition_paths)
    write_json(manifest, uncertainty_dir / "decomposition_artifact_manifest.json")
    summary = {
        "status": "variance_decomposition_stage_complete",
        "draw_shape": list(draws.shape),
        "components": decomposition.components,
        "shares": decomposition.shares,
        "checkpoint_reused": False,
        "audit_outcomes_used": False,
    }
    return _persist_deterministic_stage_summary(
        summary,
        output / "variance_decomposition_stage_summary.json",
        started=started,
    )


def freeze_audit_baseline_stage(output: Path) -> dict[str, Any]:
    """Fit and seal the three mandatory through-2010 audit comparators."""
    baseline_dir = output / "promotion" / "baselines"
    manifest_path = baseline_dir / "baseline_freeze_manifest.json"
    required = (
        "official_outcome_logit.joblib",
        "independent_mirrored_poisson.joblib",
        "legacy_mirrored_dixon_coles.joblib",
        "baseline_freeze_metadata.json",
    )
    if manifest_path.is_file():
        verified = verify_stage_manifest(manifest_path, required_paths=required)
        metadata = json.loads(
            read_verified_artifact_bytes(
                baseline_dir, "baseline_freeze_metadata.json", verified
            ).decode("utf-8")
        )
        return {
            "status": "audit_baselines_checkpoint_verified",
            "artifact_count": verified["artifact_count"],
            "baseline_bundle_hash": metadata["baseline_bundle_hash"],
        }
    if baseline_dir.exists() and any(baseline_dir.iterdir()):
        raise RuntimeError("Audit baseline freeze is partial; refusing to overwrite it")
    _, development, identity = _load_locked_nested(output)
    frozen = fit_fixed_audit_baselines(
        development,
        development_frame_hash=identity["development_frame_hash"],
        feature_schema_hash=identity["feature_schema_hash"],
    )
    artifacts = {
        "official_outcome_logit": baseline_dir / "official_outcome_logit.joblib",
        "independent_mirrored_poisson": baseline_dir / "independent_mirrored_poisson.joblib",
        "legacy_mirrored_dixon_coles": baseline_dir / "legacy_mirrored_dixon_coles.joblib",
    }
    for name, model in (
        ("official_outcome_logit", frozen.official_outcome_logit),
        ("independent_mirrored_poisson", frozen.independent_mirrored_poisson),
        ("legacy_mirrored_dixon_coles", frozen.legacy_mirrored_dixon_coles),
    ):
        _dump_joblib(model, artifacts[name])
    metadata = json.loads(json.dumps(frozen.metadata))
    metadata["development_snapshot_manifest_sha256"] = identity[
        "development_snapshot_manifest_sha256"
    ]
    metadata["formal_preaudit_reads_combined_raw"] = False
    for name, path in artifacts.items():
        metadata["models"][name]["artifact_path"] = path.relative_to(output).as_posix()
        metadata["models"][name]["artifact_sha256"] = _file_sha256(path)
    metadata["artifact_binding_hash"] = payload_sha256(
        {
            name: {
                "artifact_path": metadata["models"][name]["artifact_path"],
                "artifact_sha256": metadata["models"][name]["artifact_sha256"],
            }
            for name in sorted(artifacts)
        }
    )
    write_json(metadata, baseline_dir / "baseline_freeze_metadata.json")
    manifest = _artifact_manifest(baseline_dir, relative_paths=list(required))
    write_json(manifest, manifest_path)
    return {
        "status": "audit_baselines_frozen_without_audit_data",
        "artifact_count": len(manifest),
        "baseline_bundle_hash": metadata["baseline_bundle_hash"],
    }


def run_promotion_stage(
    output: Path,
    *,
    authentication_key: bytes | None = None,
    key_id: str | None = None,
) -> dict[str, Any]:
    """Seal fixed baselines, evaluate all gates, and create the unopened sentinel."""
    baseline = freeze_audit_baseline_stage(output)
    certificate = issue_promotion_certificate(
        output, authentication_key=authentication_key, key_id=key_id
    )
    payload = certificate["payload"]
    summary = {
        "status": "promotion_complete_audit_authorized",
        "system_promoted": bool(payload["gate"]["system_promoted"]),
        "certificate_sha256": _file_sha256(output / "promotion" / "promotion_certificate.json"),
        "audit_protocol_identity": payload["audit_protocol_identity"],
        "baseline_freeze": baseline,
        "audit_executed": False,
    }
    write_json(summary, output / "promotion_stage_summary.json")
    return certificate


def run_audit_stage(
    output: Path,
    *,
    authentication_key: bytes | None = None,
) -> dict[str, Any]:
    """Execute one authorized audit or verify its cached artifacts without target I/O."""
    if git_output(ROOT, "status", "--porcelain"):
        raise RuntimeError("Formal audit requires a clean Git working tree")
    certificate = verify_promotion_certificate(
        output / "promotion" / "promotion_certificate.json",
        authentication_key=authentication_key,
        revalidate_artifacts=True,
    )
    protocol = certificate["audit_protocol_identity"]
    feature_schema_hash = certificate["model_identity"]["feature_schema_hash"]
    return run_one_time_audit(
        output,
        freeze=lambda: freeze_audit_models(output, audit_protocol_identity=protocol),
        open_blind_features=lambda: load_batch_blind_2014_features(
            ROOT, feature_schema_hash=feature_schema_hash
        ),
        prepare_blind=prepare_frozen_audit_predictions,
        persist_blind=persist_blind_audit_predictions,
        open_audit_truth=lambda: load_isolated_2014_truth(ROOT),
        evaluate_persisted_blind=evaluate_persisted_blind_audit,
        persist=persist_audit_result,
        authentication_key=authentication_key,
    )


def run_preaudit_stage(
    output: Path,
    *,
    authentication_key: bytes | None = None,
    key_id: str | None = None,
) -> dict[str, Any]:
    """Run or verify the seven ordered stages, then promote."""
    runners = {
        "nested": run_nested_stage,
        "evaluate": run_evaluation_stage,
        "stress": run_stress_stage,
        "ablation": run_ablation_stage,
        "parameter": run_parameter_uncertainty_stage,
        "simulate": run_simulation_stage,
        "decompose": run_variance_decomposition_stage,
    }
    return run_preaudit_pipeline(
        output,
        stage_runners=runners,
        promote=lambda root: run_promotion_stage(
            root, authentication_key=authentication_key, key_id=key_id
        ),
        verify_certificate=lambda path: verify_promotion_certificate(
            path,
            authentication_key=authentication_key,
            revalidate_artifacts=True,
        ),
    )


def run_all_stage(
    output: Path,
    *,
    authentication_key: bytes | None = None,
    key_id: str | None = None,
) -> dict[str, Any]:
    """Run pre-audit and then the one-time audit interface."""
    return run_all_pipeline(
        output,
        preaudit=lambda root: run_preaudit_stage(
            root, authentication_key=authentication_key, key_id=key_id
        ),
        audit=lambda root: run_audit_stage(root, authentication_key=authentication_key),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "nested",
            "evaluate",
            "stress",
            "ablation",
            "parameter",
            "simulate",
            "decompose",
            "promote",
            "audit",
            "preaudit",
            "all",
        ),
        default="nested",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--authentication-key-file",
        type=Path,
        default=None,
        help="Optional external HMAC key; SHA-256 integrity remains the default.",
    )
    parser.add_argument("--authentication-key-id", default=None)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    authentication_key = None
    if args.authentication_key_file is not None:
        key_path = args.authentication_key_file.expanduser().resolve()
        authentication_key = key_path.read_bytes()
        if len(authentication_key) < 16:
            raise ValueError("External HMAC key file must contain at least 16 bytes")
    elif args.authentication_key_id is not None:
        raise ValueError("--authentication-key-id requires --authentication-key-file")
    if args.stage == "nested":
        summary = run_nested_stage(output)
    elif args.stage == "evaluate":
        summary = run_evaluation_stage(output)
    elif args.stage == "stress":
        summary = run_stress_stage(output)
    elif args.stage == "ablation":
        summary = run_ablation_stage(output)
    elif args.stage == "parameter":
        summary = run_parameter_uncertainty_stage(output)
    elif args.stage == "simulate":
        summary = run_simulation_stage(output)
    elif args.stage == "decompose":
        summary = run_variance_decomposition_stage(output)
    elif args.stage == "promote":
        summary = run_promotion_stage(
            output,
            authentication_key=authentication_key,
            key_id=args.authentication_key_id,
        )
    elif args.stage == "audit":
        summary = run_audit_stage(output, authentication_key=authentication_key)
    elif args.stage == "preaudit":
        summary = run_preaudit_stage(
            output,
            authentication_key=authentication_key,
            key_id=args.authentication_key_id,
        )
    else:
        summary = run_all_stage(
            output,
            authentication_key=authentication_key,
            key_id=args.authentication_key_id,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
