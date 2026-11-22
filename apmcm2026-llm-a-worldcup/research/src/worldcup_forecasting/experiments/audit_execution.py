"""Freeze all models, predict target-free 2014 features, then score once."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from worldcup_forecasting.application.runtime import write_csv, write_json
from worldcup_forecasting.data import CanonicalDataBundle, build_canonical_data
from worldcup_forecasting.models.score_calibration import outcome_margins
from worldcup_forecasting.validation.artifact_integrity import (
    file_sha256,
    payload_sha256,
    read_verified_artifact_bytes,
    verify_stage_manifest,
)
from worldcup_forecasting.validation.audit_orchestration import (
    FrozenAuditPreparation,
    PersistedAudit,
)
from worldcup_forecasting.validation.freeze_contracts import write_parquet_atomic
from worldcup_forecasting.validation.joint_metrics import joint_forecast_metrics
from worldcup_forecasting.validation.metrics import classification_metrics
from worldcup_forecasting.validation.pre2014_freeze import git_output

from ._nested_selection_types import frame_content_hash
from .audit_baselines import validate_fixed_audit_baseline_models
from .final_audit import (
    _freeze_layers,
    _validate_identity,
    _validate_inner,
    _validate_inner_lineage,
)
from .research_features import build_research_feature_bundle

_OUTCOMES = np.asarray(("home_win", "draw", "away_win"), dtype=object)
_TARGET_COLUMNS = ("target_outcome", "target_home_goals", "target_away_goals")


@dataclass(frozen=True)
class FrozenAuditModels:
    """All six frozen prediction/decision components."""

    advanced: Any
    calibration: Any
    decision: Any
    official: Any
    independent: Any
    dixon_coles: Any
    identity: dict[str, str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AuditComparisonResult:
    """Identical-match predictions, joint grids, metrics, and provenance."""

    predictions: pd.DataFrame
    joint_grid: pd.DataFrame
    metrics: pd.DataFrame
    truth: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BlindAuditPredictions:
    """Immutable target-free predictions persisted before truth access."""

    predictions: pd.DataFrame
    joint_grid: pd.DataFrame
    probability_models: dict[str, np.ndarray]
    joint_models: dict[str, np.ndarray]
    decision: np.ndarray
    metadata: dict[str, Any]


def freeze_audit_models(
    output_root: Path,
    *,
    audit_protocol_identity: dict[str, Any],
) -> FrozenAuditPreparation[FrozenAuditModels]:
    """Load only verified pre-audit artifacts and freeze every fitted layer."""
    root = output_root.resolve()
    nested = root / "nested"
    baseline = root / "promotion" / "baselines"
    nested_binding = verify_stage_manifest(
        nested / "artifact_manifest.json",
        required_paths=(
            "final_fitted_model.joblib",
            "final_inner_oof_ledger.csv",
            "final_inner_joint_grid.csv",
            "frozen_model_identity.json",
        ),
    )
    baseline_binding = verify_stage_manifest(
        baseline / "baseline_freeze_manifest.json",
        required_paths=(
            "official_outcome_logit.joblib",
            "independent_mirrored_poisson.joblib",
            "legacy_mirrored_dixon_coles.joblib",
            "baseline_freeze_metadata.json",
        ),
    )
    identity_record = json.loads(
        read_verified_artifact_bytes(
            nested, "frozen_model_identity.json", nested_binding
        ).decode("utf-8")
    )
    audit_identity = {
        "final_fit_hash": identity_record["final_fit_hash"],
        "selected_config_hash": identity_record["selected_config_hash"],
        "feature_schema_hash": identity_record["feature_schema_hash"],
        "protocol_hash": audit_protocol_identity["frozen_audit_protocol_hash"],
    }
    predictor = joblib.load(
        io.BytesIO(
            read_verified_artifact_bytes(
                nested, "final_fitted_model.joblib", nested_binding
            )
        )
    )
    inner = _validate_inner(
        pd.read_csv(
            io.BytesIO(
                read_verified_artifact_bytes(
                    nested, "final_inner_oof_ledger.csv", nested_binding
                )
            )
        )
    )
    grid = pd.read_csv(
        io.BytesIO(
            read_verified_artifact_bytes(
                nested, "final_inner_joint_grid.csv", nested_binding
            )
        )
    )
    predictor_metadata = _validate_identity(
        predictor, audit_identity, audit_identity["protocol_hash"]
    )
    inner_lineage_hash = _validate_inner_lineage(inner, grid, predictor_metadata, audit_identity)
    selector, decision = _freeze_layers(inner, grid, None)
    official = joblib.load(
        io.BytesIO(
            read_verified_artifact_bytes(
                baseline, "official_outcome_logit.joblib", baseline_binding
            )
        )
    )
    independent = joblib.load(
        io.BytesIO(
            read_verified_artifact_bytes(
                baseline, "independent_mirrored_poisson.joblib", baseline_binding
            )
        )
    )
    dixon_coles = joblib.load(
        io.BytesIO(
            read_verified_artifact_bytes(
                baseline, "legacy_mirrored_dixon_coles.joblib", baseline_binding
            )
        )
    )
    validate_fixed_audit_baseline_models(official, independent, dixon_coles)
    baseline_metadata = json.loads(
        read_verified_artifact_bytes(
            baseline, "baseline_freeze_metadata.json", baseline_binding
        ).decode("utf-8")
    )
    project_root = Path(__file__).resolve().parents[3]
    freeze_manifest_path = project_root / "artifacts" / "freeze" / "FREEZE_MANIFEST.json"
    freeze_manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    formal_source_commit = git_output(project_root, "rev-parse", "HEAD")
    metadata = {
        "audit_protocol_identity": audit_protocol_identity,
        "audit_identity": audit_identity,
        "predictor_metadata": predictor_metadata,
        "inner_lineage_hash": inner_lineage_hash,
        "calibration_state": selector.state_dict(),
        "decision_state": dict(decision.metadata_),
        "baseline_bundle_hash": baseline_metadata["baseline_bundle_hash"],
        "formal_source_commit": formal_source_commit,
        "pre2014_freeze_manifest_sha256": file_sha256(freeze_manifest_path),
        "pre2014_freeze_source_commit": freeze_manifest["source_commit"],
        "pre2014_freeze_generated_at": freeze_manifest["generated_at"],
        "target_rows_loaded": False,
    }
    state_hash = payload_sha256(metadata)
    models = FrozenAuditModels(
        predictor,
        selector,
        decision,
        official,
        independent,
        dixon_coles,
        audit_identity,
        metadata,
    )
    components = {
        "advanced_predictor": True,
        "calibration": True,
        "decision": True,
        "official_outcome_logit": True,
        "independent_mirrored_poisson": True,
        "legacy_mirrored_dixon_coles": True,
    }
    return FrozenAuditPreparation(models, state_hash, components, metadata)


def load_batch_blind_2014_features(
    project_root: Path,
    *,
    feature_schema_hash: str,
) -> pd.DataFrame:
    """Build target-free 2014 features with all dynamic state frozen before 2014.

    The realized fixture identities and schedule define the retrospective audit
    estimand.  No 2014 result can update a feature: score fields are neutralized
    before feature construction, player-event history is restricted to matches
    before 2014, and dynamic/personnel state updates stop before 2014.
    """
    canonical = build_canonical_data(project_root / "data" / "raw")
    through_audit = _through_2014(canonical)
    matches = through_audit.matches.copy()
    audit_mask = pd.to_numeric(matches["Year"], errors="raise").eq(2014)
    matches.loc[audit_mask, ["Home Team Goals", "Away Team Goals"]] = 0
    preaudit_ids = set(matches.loc[~audit_mask, "MatchID"])
    sanitized = CanonicalDataBundle(
        tournaments=through_audit.tournaments.copy(),
        matches=matches,
        players=through_audit.players.loc[
            through_audit.players["MatchID"].isin(preaudit_ids)
        ].copy(),
        reconciliation=through_audit.reconciliation.copy(),
        raw_hashes=dict(through_audit.raw_hashes),
        schema_report=dict(through_audit.schema_report),
        duplicate_report=through_audit.duplicate_report.copy(),
        team_name_registry=through_audit.team_name_registry.copy(),
        stage_registry=through_audit.stage_registry.copy(),
        data_lineage=dict(through_audit.data_lineage),
    )
    bundle = build_research_feature_bundle(
        sanitized,
        state_update_before_year=2014,
    )
    if bundle.schema_hash != feature_schema_hash:
        raise RuntimeError("2014 feature schema differs from the frozen development schema")
    audit = bundle.pre_match.loc[bundle.pre_match["Year"].eq(2014)].copy()
    audit = audit.drop(columns=list(_TARGET_COLUMNS), errors="raise")
    audit = _validate_audit_features(audit)
    if not bool((audit["history_cutoff_year"] < audit["Year"]).all()):
        raise RuntimeError("2014 static history crosses the prediction cutoff")
    prediction = pd.to_datetime(audit["Datetime"], errors="raise")
    for side in ("home", "away"):
        cutoff = pd.to_datetime(audit[f"{side}_dynamic_cutoff_time"], errors="coerce")
        if not bool((cutoff.isna() | cutoff.lt(prediction)).all()):
            raise RuntimeError("2014 dynamic state crosses the match prediction time")
        cutoff_year = cutoff.dropna().dt.year
        if not cutoff_year.empty and int(cutoff_year.max()) >= 2014:
            raise RuntimeError("Batch-blind dynamic state contains a 2014 outcome update")
    return audit


def load_isolated_2014_truth(project_root: Path) -> pd.DataFrame:
    """Open only the 64 audit labels after the blind manifest is sealed."""
    canonical = build_canonical_data(project_root / "data" / "raw")
    matches = canonical.matches.loc[canonical.matches["Year"].eq(2014)].copy()
    home_goals = pd.to_numeric(matches["Home Team Goals"], errors="raise").astype(int)
    away_goals = pd.to_numeric(matches["Away Team Goals"], errors="raise").astype(int)
    truth = pd.DataFrame(
        {
            "Year": matches["Year"].astype(int),
            "MatchID": matches["MatchID"],
            "Datetime": pd.to_datetime(matches["Datetime"], errors="raise"),
            "target_outcome": np.where(
                home_goals.gt(away_goals),
                "home_win",
                np.where(home_goals.eq(away_goals), "draw", "away_win"),
            ),
            "target_home_goals": home_goals,
            "target_away_goals": away_goals,
        }
    )
    return _validate_audit_truth(truth)


def evaluate_frozen_audit(
    frozen: FrozenAuditModels,
    audit_frame: pd.DataFrame,
    *,
    pit_seed: int = 2014,
) -> AuditComparisonResult:
    """Development-only compatibility wrapper; formal audit never calls it."""
    features = audit_frame.drop(columns=list(_TARGET_COLUMNS), errors="raise")
    truth = audit_frame[["Year", "MatchID", "Datetime", *_TARGET_COLUMNS]].copy()
    blind = prepare_frozen_audit_predictions(frozen, features)
    return evaluate_blind_audit(blind, truth, pit_seed=pit_seed)


def prepare_frozen_audit_predictions(
    frozen: FrozenAuditModels,
    audit_frame: pd.DataFrame,
) -> BlindAuditPredictions:
    """Produce every frozen prediction without reading target-column values."""
    features = _validate_audit_features(audit_frame)
    match_ids = features["MatchID"].to_numpy()
    prediction_times = pd.to_datetime(features["Datetime"], errors="raise")

    advanced_raw_batch = frozen.advanced.predict_joint(features)
    advanced_raw_joint = np.asarray(advanced_raw_batch.probability, dtype=float)
    advanced_joint = frozen.calibration.transform(advanced_raw_joint)
    advanced_raw = outcome_margins(advanced_raw_joint)
    advanced = outcome_margins(advanced_joint)
    advanced_decision = frozen.decision.predict(advanced)
    official = np.asarray(frozen.official.predict_proba(features), dtype=float)
    independent_batch = frozen.independent.predict_joint(features)
    dc_batch = frozen.dixon_coles.predict_joint(features)
    independent_joint = np.asarray(independent_batch.probability, dtype=float)
    dc_joint = np.asarray(dc_batch.probability, dtype=float)
    independent = independent_batch.outcome_probabilities()
    dixon_coles = dc_batch.outcome_probabilities()

    probability_models = {
        "advanced_raw": advanced_raw,
        "advanced_calibrated": advanced,
        "official_outcome_logit": official,
        "independent_mirrored_poisson": independent,
        "legacy_mirrored_dixon_coles": dixon_coles,
    }
    joint_models = {
        "advanced_raw": advanced_raw_joint,
        "advanced_calibrated": advanced_joint,
        "independent_mirrored_poisson": independent_joint,
        "legacy_mirrored_dixon_coles": dc_joint,
    }
    predictions = _blind_prediction_ledger(
        match_ids,
        prediction_times,
        probability_models,
        advanced_decision,
    )
    joint_grid = _joint_grid(match_ids, joint_models)
    metadata = {
        "evaluation_year": 2014,
        "evaluation_role": "one_time_audit_only_not_for_selection",
        "match_count": 64,
        "identical_match_ids": True,
        "prediction_models": list(probability_models),
        "joint_models": list(joint_models),
        "advanced_identity": frozen.identity,
        "frozen_state_hash": payload_sha256(frozen.metadata),
        "target_columns_exposed_to_predictors": False,
        "all_predictions_completed_before_target_access": True,
        "truth_columns_accessed": False,
        "blind_prediction_frame_sha256": frame_content_hash(predictions),
        "blind_joint_grid_sha256": frame_content_hash(joint_grid),
        "information_set": "batch_blind_realized_fixture_pre2014_state",
        "dynamic_state_update_before_year": 2014,
        "realized_fixture_identity_used": True,
        "selection_reopened": False,
        "audit_target_used_for_model_choice": False,
        "audit_feature_frame_hash": frame_content_hash(features),
        "formal_source_commit": frozen.metadata["formal_source_commit"],
        "pre2014_freeze_manifest_sha256": frozen.metadata[
            "pre2014_freeze_manifest_sha256"
        ],
        "pre2014_freeze_source_commit": frozen.metadata[
            "pre2014_freeze_source_commit"
        ],
        "predictions_generated_at_utc": datetime.now(UTC).isoformat(),
        "generation_timestamp_trust": "host_system_clock_utc_not_external_authority",
    }
    return BlindAuditPredictions(
        predictions,
        joint_grid,
        probability_models,
        joint_models,
        advanced_decision,
        metadata,
    )


def persist_blind_audit_predictions(
    blind: BlindAuditPredictions,
    audit_dir: Path,
) -> PersistedAudit:
    """Seal target-free predictions before the scoring layer may read truth."""
    write_parquet_atomic(
        blind.predictions,
        audit_dir / "2014_predictions_blind.parquet",
    )
    write_parquet_atomic(
        blind.joint_grid,
        audit_dir / "2014_joint_predictions_blind.parquet",
    )
    metadata = {
        **blind.metadata,
        "status": "blind_predictions_persisted_before_truth_release",
        "prediction_rows": int(len(blind.predictions)),
        "truth_columns_present_in_artifact": False,
    }
    write_json(metadata, audit_dir / "2014_predictions_blind_metadata.json")
    return PersistedAudit(
        (
            "2014_predictions_blind.parquet",
            "2014_joint_predictions_blind.parquet",
            "2014_predictions_blind_metadata.json",
        ),
        metadata,
    )


def load_persisted_blind_audit_predictions(
    audit_dir: Path,
    *,
    expected_manifest_sha256: str,
) -> BlindAuditPredictions:
    """Reconstruct predictions only from bytes bound by the blind manifest."""
    required = (
        "2014_predictions_blind.parquet",
        "2014_joint_predictions_blind.parquet",
        "2014_predictions_blind_metadata.json",
    )
    manifest = verify_stage_manifest(
        audit_dir / "blind_prediction_manifest.json",
        required_paths=required,
    )
    if manifest["manifest_sha256"] != expected_manifest_sha256:
        raise RuntimeError("Blind prediction manifest identity changed before scoring")
    prediction_bytes = read_verified_artifact_bytes(audit_dir, required[0], manifest)
    joint_bytes = read_verified_artifact_bytes(audit_dir, required[1], manifest)
    metadata_bytes = read_verified_artifact_bytes(audit_dir, required[2], manifest)
    predictions = pd.read_parquet(io.BytesIO(prediction_bytes))
    joint_grid = pd.read_parquet(io.BytesIO(joint_bytes))
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    if not isinstance(metadata, dict):
        raise RuntimeError("Blind prediction metadata must be an object")
    if any(column in predictions for column in _TARGET_COLUMNS):
        raise RuntimeError("Persisted blind predictions contain audit truth")
    if frame_content_hash(predictions) != metadata.get("blind_prediction_frame_sha256"):
        raise RuntimeError("Persisted blind outcome predictions changed after sealing")
    if frame_content_hash(joint_grid) != metadata.get("blind_joint_grid_sha256"):
        raise RuntimeError("Persisted blind joint predictions changed after sealing")

    expected_probability_models = tuple(str(item) for item in metadata["prediction_models"])
    expected_joint_models = tuple(str(item) for item in metadata["joint_models"])
    probability_models, match_ids, decision = _probabilities_from_blind_ledger(
        predictions,
        expected_probability_models,
    )
    joint_models = _joints_from_blind_ledger(
        joint_grid,
        expected_joint_models,
        match_ids,
    )
    if int(metadata.get("match_count", -1)) != len(match_ids):
        raise RuntimeError("Blind metadata match count disagrees with persisted forecasts")
    return BlindAuditPredictions(
        predictions=predictions,
        joint_grid=joint_grid,
        probability_models=probability_models,
        joint_models=joint_models,
        decision=decision,
        metadata=metadata,
    )


def evaluate_blind_audit(
    blind: BlindAuditPredictions,
    audit_frame: pd.DataFrame,
    *,
    pit_seed: int = 2014,
) -> AuditComparisonResult:
    """Release truth once and score only the already-persisted prediction payload."""
    audit = _validate_audit_truth(audit_frame)
    match_ids = audit["MatchID"].to_numpy()
    blind_match_ids = blind.predictions["match_id"].drop_duplicates().to_numpy()
    if not np.array_equal(blind_match_ids, match_ids):
        raise RuntimeError("Blind prediction identities changed before truth release")

    # This is the only target-value access in the formal two-phase audit.
    labels, goals = _targets(audit)
    metrics = _metrics(
        labels,
        goals,
        blind.probability_models,
        blind.joint_models,
        blind.decision,
        pit_seed,
    )
    predictions = _scored_prediction_ledger(blind.predictions, match_ids, labels)
    truth = pd.DataFrame(
        {
            "match_id": match_ids,
            "prediction_time": pd.to_datetime(audit["Datetime"], errors="raise"),
            "target_outcome": labels,
            "target_home_goals": goals[:, 0],
            "target_away_goals": goals[:, 1],
            "release_state": "released_once_after_blind_manifest",
        }
    )
    metadata = {
        **blind.metadata,
        "truth_columns_accessed": True,
        "truth_released_after_blind_persistence": True,
        "truth_frame_sha256": frame_content_hash(truth),
        "selection_reopened": False,
        "audit_target_used_for_model_choice": False,
    }
    return AuditComparisonResult(
        predictions=predictions,
        joint_grid=blind.joint_grid,
        metrics=metrics,
        truth=truth,
        metadata=metadata,
    )


def evaluate_persisted_blind_audit(
    audit_dir: Path,
    truth_frame: pd.DataFrame,
    blind_manifest_sha256: str,
    *,
    pit_seed: int = 2014,
) -> AuditComparisonResult:
    """Reload manifest-bound blind outcome and joint forecasts, then score once."""
    blind = load_persisted_blind_audit_predictions(
        audit_dir,
        expected_manifest_sha256=blind_manifest_sha256,
    )
    result = evaluate_blind_audit(blind, truth_frame, pit_seed=pit_seed)
    metadata = {
        **result.metadata,
        "blind_prediction_manifest_sha256": blind_manifest_sha256,
        "scoring_source": "manifest_bound_persisted_blind_predictions",
    }
    return AuditComparisonResult(
        predictions=result.predictions,
        joint_grid=result.joint_grid,
        metrics=result.metrics,
        truth=result.truth,
        metadata=metadata,
    )


def persist_audit_result(result: AuditComparisonResult, audit_dir: Path) -> PersistedAudit:
    """Persist the complete first-run result before the state is marked complete."""
    write_csv(result.predictions, audit_dir / "audit_predictions.csv")
    write_csv(result.joint_grid, audit_dir / "audit_joint_grid.csv.gz")
    write_csv(result.metrics, audit_dir / "audit_metrics.csv")
    write_parquet_atomic(result.truth, audit_dir / "2014_truth_released.parquet")
    write_json(result.metadata, audit_dir / "audit_metadata.json")
    external_audit = {
        "schema_version": 1,
        "status": "one_shot_2014_external_audit_complete",
        "blind_prediction_frame_sha256": result.metadata["blind_prediction_frame_sha256"],
        "truth_frame_sha256": result.metadata["truth_frame_sha256"],
        "truth_released_after_blind_persistence": True,
        "selection_reopened": False,
        "metrics": json.loads(result.metrics.to_json(orient="records")),
        "claim_boundary": "external_audit_only_not_model_selection",
    }
    write_json(external_audit, audit_dir / "2014_external_audit.json")
    summary = {
        "status": "one_time_2014_audit_complete",
        "match_count": 64,
        "models_compared": result.metrics["model"].astype(str).tolist(),
        "target_reads": 1,
        "selection_reopened": False,
        "cached_repeats_only": True,
    }
    write_json(summary, audit_dir / "audit_stage_summary.json")
    return PersistedAudit(
        (
            "audit_predictions.csv",
            "audit_joint_grid.csv.gz",
            "audit_metrics.csv",
            "2014_truth_released.parquet",
            "2014_external_audit.json",
            "audit_metadata.json",
            "audit_stage_summary.json",
        ),
        summary,
    )


def _metrics(
    labels: np.ndarray,
    goals: np.ndarray,
    probabilities: dict[str, np.ndarray],
    joints: dict[str, np.ndarray],
    decision: np.ndarray,
    pit_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, probability in probabilities.items():
        classification, _ = classification_metrics(labels, probability)
        row: dict[str, Any] = {
            "model": name,
            "evaluation_role": "audit_only_not_for_selection",
            **classification,
        }
        if name in joints:
            row.update(joint_forecast_metrics(goals, joints[name], pit_seed=pit_seed))
        rows.append(row)
    truth = np.asarray([int(np.flatnonzero(_OUTCOMES == label)[0]) for label in labels])
    predicted = np.asarray([int(np.flatnonzero(_OUTCOMES == label)[0]) for label in decision])
    rows.append(
        {
            "model": "advanced_cost_sensitive_decision",
            "evaluation_role": "audit_only_not_for_selection",
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, average="macro")),
            "recall_draw": float(
                recall_score(truth, predicted, labels=[1], average="macro", zero_division=0)
            ),
            "decision_draw_predictions": int(np.sum(predicted == 1)),
            "proper_score_applicable": False,
        }
    )
    return pd.DataFrame(rows)


def _blind_prediction_ledger(
    match_ids: np.ndarray,
    prediction_times: pd.Series,
    probabilities: dict[str, np.ndarray],
    decision: np.ndarray,
) -> pd.DataFrame:
    parts = []
    for name, probability in probabilities.items():
        block = pd.DataFrame(
            {
                "match_id": match_ids,
                "prediction_time": prediction_times.to_numpy(),
                "model": name,
                "home_win": probability[:, 0],
                "draw": probability[:, 1],
                "away_win": probability[:, 2],
                "argmax_label": _OUTCOMES[probability.argmax(axis=1)],
            }
        )
        if name == "advanced_calibrated":
            block["decision_label"] = decision
        parts.append(block)
    return pd.concat(parts, ignore_index=True)


def _scored_prediction_ledger(
    blind: pd.DataFrame,
    match_ids: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    truth = pd.DataFrame({"match_id": match_ids, "target_outcome": labels})
    return blind.merge(truth, on="match_id", how="left", validate="many_to_one")


def _joint_grid(match_ids: np.ndarray, models: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, batch in models.items():
        for position, match_id in enumerate(match_ids):
            home, away = np.indices(batch[position].shape)
            rows.extend(
                {
                    "model": name,
                    "match_id": match_id,
                    "home_goals": int(h),
                    "away_goals": int(a),
                    "probability": float(p),
                }
                for h, a, p in zip(home.ravel(), away.ravel(), batch[position].ravel(), strict=True)
            )
    return pd.DataFrame(rows)


def _probabilities_from_blind_ledger(
    predictions: pd.DataFrame,
    expected_models: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    required = {
        "match_id",
        "prediction_time",
        "model",
        "home_win",
        "draw",
        "away_win",
        "argmax_label",
    }
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise RuntimeError("Persisted blind outcome ledger is empty")
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise RuntimeError(f"Persisted blind outcome ledger is missing {missing}")
    if set(predictions["model"].astype(str)) != set(expected_models):
        raise RuntimeError("Persisted blind outcome model set changed")
    probability_models: dict[str, np.ndarray] = {}
    canonical_ids: np.ndarray | None = None
    decision: np.ndarray | None = None
    for model in expected_models:
        block = predictions.loc[predictions["model"].astype(str).eq(model)].sort_values(
            "match_id"
        )
        if len(block) != 64 or block["match_id"].duplicated().any():
            raise RuntimeError(f"Blind outcome model {model} does not contain 64 matches")
        match_ids = block["match_id"].to_numpy()
        if canonical_ids is None:
            canonical_ids = match_ids
        elif not np.array_equal(canonical_ids, match_ids):
            raise RuntimeError("Blind outcome model MatchIDs are not identical")
        probability = block[["home_win", "draw", "away_win"]].to_numpy(float)
        if (
            not np.isfinite(probability).all()
            or np.any(probability < 0.0)
            or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-10, rtol=0.0)
        ):
            raise RuntimeError(f"Blind outcome probabilities are invalid for {model}")
        probability_models[model] = probability
        if model == "advanced_calibrated":
            if "decision_label" not in block or block["decision_label"].isna().any():
                raise RuntimeError("Blind calibrated decisions are missing")
            decision = block["decision_label"].astype(str).to_numpy()
    if canonical_ids is None or decision is None:
        raise RuntimeError("Blind outcome ledger omitted required calibrated predictions")
    return probability_models, canonical_ids, decision


def _joints_from_blind_ledger(
    joint_grid: pd.DataFrame,
    expected_models: tuple[str, ...],
    match_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    required = {"model", "match_id", "home_goals", "away_goals", "probability"}
    if not isinstance(joint_grid, pd.DataFrame) or joint_grid.empty:
        raise RuntimeError("Persisted blind joint ledger is empty")
    missing = sorted(required.difference(joint_grid.columns))
    if missing:
        raise RuntimeError(f"Persisted blind joint ledger is missing {missing}")
    if set(joint_grid["model"].astype(str)) != set(expected_models):
        raise RuntimeError("Persisted blind joint model set changed")
    joints: dict[str, np.ndarray] = {}
    for model in expected_models:
        model_rows = joint_grid.loc[joint_grid["model"].astype(str).eq(model)].copy()
        home = pd.to_numeric(model_rows["home_goals"], errors="raise").astype(int)
        away = pd.to_numeric(model_rows["away_goals"], errors="raise").astype(int)
        if (home < 0).any() or (away < 0).any():
            raise RuntimeError(f"Blind joint support is invalid for {model}")
        home_size = int(home.max()) + 1
        away_size = int(away.max()) + 1
        batch = np.zeros((len(match_ids), home_size, away_size), dtype=float)
        for position, match_id in enumerate(match_ids):
            block = model_rows.loc[model_rows["match_id"].eq(match_id)].copy()
            if len(block) != home_size * away_size:
                raise RuntimeError(f"Blind joint grid is incomplete for {model}/{match_id}")
            if block.duplicated(["home_goals", "away_goals"]).any():
                raise RuntimeError(f"Blind joint grid has duplicate cells for {model}/{match_id}")
            for row in block.itertuples(index=False):
                batch[position, int(row.home_goals), int(row.away_goals)] = float(
                    row.probability
                )
        if (
            not np.isfinite(batch).all()
            or np.any(batch < 0.0)
            or not np.allclose(batch.sum(axis=(1, 2)), 1.0, atol=1e-10, rtol=0.0)
        ):
            raise RuntimeError(f"Blind joint probabilities are invalid for {model}")
        joints[model] = batch
    return joints


def _validate_audit_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"Year", "MatchID", "Datetime"}
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Audit feature frame must be a nonempty DataFrame")
    exposed = sorted(set(_TARGET_COLUMNS).intersection(frame.columns))
    if exposed:
        raise ValueError(f"Blind feature frame exposes target columns: {exposed}")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Audit feature frame is missing columns: {missing}")
    years = pd.to_numeric(frame["Year"], errors="raise")
    if set(years.astype(int)) != {2014} or not np.equal(years, np.floor(years)).all():
        raise ValueError("Audit feature frame must contain exactly the 2014 edition")
    if len(frame) != 64 or frame["MatchID"].isna().any() or frame["MatchID"].duplicated().any():
        raise ValueError("Audit feature frame must contain exactly 64 unique MatchIDs")
    return frame.sort_values("MatchID").reset_index(drop=True)


def _validate_audit_truth(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"Year", "MatchID", "Datetime", *_TARGET_COLUMNS}
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Audit truth frame must be a nonempty DataFrame")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Audit truth frame is missing columns: {missing}")
    unexpected = sorted(set(frame.columns).difference(required))
    if unexpected:
        raise ValueError(f"Audit truth capability exposes non-truth columns: {unexpected}")
    years = pd.to_numeric(frame["Year"], errors="raise")
    if set(years.astype(int)) != {2014} or not np.equal(years, np.floor(years)).all():
        raise ValueError("Audit truth frame must contain exactly the 2014 edition")
    if len(frame) != 64 or frame["MatchID"].isna().any() or frame["MatchID"].duplicated().any():
        raise ValueError("Audit truth frame must contain exactly 64 unique MatchIDs")
    validated = frame.sort_values("MatchID").reset_index(drop=True)
    _targets(validated)
    return validated


def _targets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    labels = frame["target_outcome"].astype(str).to_numpy()
    goals = (
        frame[["target_home_goals", "target_away_goals"]]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(float)
    )
    if not set(labels).issubset(set(_OUTCOMES)) or not np.isfinite(goals).all():
        raise ValueError("Audit targets are invalid")
    if np.any(goals < 0) or not np.equal(goals, np.floor(goals)).all():
        raise ValueError("Audit goals must be nonnegative integers")
    derived = np.where(
        goals[:, 0] > goals[:, 1],
        "home_win",
        np.where(goals[:, 0] == goals[:, 1], "draw", "away_win"),
    )
    if not np.array_equal(labels, derived):
        raise ValueError("Audit outcomes disagree with score targets")
    return labels, goals.astype(int)


def _through_2014(bundle: CanonicalDataBundle) -> CanonicalDataBundle:
    matches = bundle.matches.loc[bundle.matches["Year"].le(2014)].copy()
    match_ids = set(matches["MatchID"])
    return CanonicalDataBundle(
        tournaments=bundle.tournaments.loc[bundle.tournaments["Year"].le(2014)].copy(),
        matches=matches,
        players=bundle.players.loc[bundle.players["MatchID"].isin(match_ids)].copy(),
        reconciliation=bundle.reconciliation.copy(),
        raw_hashes=dict(bundle.raw_hashes),
        schema_report=dict(bundle.schema_report),
        duplicate_report=bundle.duplicate_report.copy(),
        team_name_registry=bundle.team_name_registry.copy(),
        stage_registry=bundle.stage_registry.copy(),
        data_lineage=dict(bundle.data_lineage),
    )


__all__ = [
    "AuditComparisonResult",
    "BlindAuditPredictions",
    "evaluate_blind_audit",
    "evaluate_persisted_blind_audit",
    "FrozenAuditModels",
    "evaluate_frozen_audit",
    "freeze_audit_models",
    "load_batch_blind_2014_features",
    "load_isolated_2014_truth",
    "load_persisted_blind_audit_predictions",
    "persist_blind_audit_predictions",
    "persist_audit_result",
    "prepare_frozen_audit_predictions",
]
