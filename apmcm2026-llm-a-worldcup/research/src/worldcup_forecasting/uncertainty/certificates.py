"""Schema-validated, model-traceable prediction certificates."""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from worldcup_forecasting.reporting.model_state import stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROBABILITY_KEYS = ("home_win", "draw", "away_win")
_CALIBRATION_REQUIRES_ARTIFACT = {
    "none": False,
    "none_official_ols_point_prediction": False,
    "none_official_baseline": False,
    "none_structural_joint_distribution": False,
    "leave_one_tournament_out_absolute_residual_interval": True,
    "strictly_earlier_rolling_absolute_residual_interval": True,
}


def build_prediction_certificate(
    *,
    information_set: str,
    data_cutoff: str,
    raw_data_hashes: dict[str, str],
    feature_schema: list[str],
    model_hash: str,
    model_artifact: str,
    config_hash: str,
    random_seed: int,
    software_versions: dict[str, str],
    calibration_method: str,
    calibration_artifact: str | None,
    prediction: dict[str, Any],
    goal_interval_90: tuple[int | float, int | float] | None,
    out_of_distribution_score: float,
    invariant_status: dict[str, bool],
    scenario_label: str,
    generated_at_utc: str | None = None,
    uncertainty_status: str = "empirical",
    warning: str | None = None,
    ood_components: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Create a complete certificate linked to a persisted fitted-state artifact."""
    certificate: dict[str, Any] = {
        "schema_version": 1,
        "scenario_label": scenario_label,
        "information_set": information_set,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "data_cutoff": data_cutoff,
        "raw_data_hashes": dict(raw_data_hashes),
        "feature_schema": list(feature_schema),
        "feature_schema_hash": stable_hash(feature_schema),
        "model_hash": model_hash,
        "model_artifact": model_artifact,
        "config_hash": config_hash,
        "random_seed": int(random_seed),
        "software_versions": dict(software_versions),
        "calibration_method": calibration_method,
        "calibration_artifact": calibration_artifact,
        "prediction": prediction,
        "goal_interval_90": list(goal_interval_90) if goal_interval_90 is not None else None,
        "uncertainty_status": uncertainty_status,
        "out_of_distribution_score": float(out_of_distribution_score),
        "out_of_distribution_components": dict(ood_components or {}),
        "invariant_status": invariant_status,
        "warning": warning,
    }
    validate_prediction_certificate(certificate)
    return certificate


def prediction_payload(certificate: dict[str, Any]) -> dict[str, Any]:
    """Return the deterministic prediction-facing subset of a full certificate."""
    validate_prediction_certificate(certificate)
    return {
        key: value
        for key, value in certificate.items()
        if key != "generated_at_utc"
    }


def validate_prediction_certificate(certificate: dict[str, Any]) -> None:
    """Reject incomplete, malformed, or internally inconsistent certificates."""
    required = {
        "schema_version",
        "scenario_label",
        "information_set",
        "generated_at_utc",
        "data_cutoff",
        "raw_data_hashes",
        "feature_schema",
        "feature_schema_hash",
        "model_hash",
        "model_artifact",
        "config_hash",
        "random_seed",
        "software_versions",
        "calibration_method",
        "calibration_artifact",
        "prediction",
        "goal_interval_90",
        "uncertainty_status",
        "out_of_distribution_score",
        "out_of_distribution_components",
        "invariant_status",
        "warning",
    }
    missing = sorted(required.difference(certificate))
    if missing:
        raise ValueError(f"Prediction certificate is missing fields: {missing}")
    if certificate["schema_version"] != 1:
        raise ValueError("Unsupported prediction-certificate schema_version")
    if certificate["information_set"] not in {"pre_match", "kickoff", "half_time"}:
        raise ValueError(f"Unsupported information set: {certificate['information_set']}")
    for field in ("scenario_label", "generated_at_utc", "data_cutoff", "calibration_method"):
        if not isinstance(certificate[field], str) or not certificate[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    _validate_timestamp(certificate["generated_at_utc"])
    _validate_data_cutoff(certificate["data_cutoff"])
    for field in ("feature_schema_hash", "model_hash", "config_hash"):
        _validate_hash(certificate[field], field)
    hashes = certificate["raw_data_hashes"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("raw_data_hashes must be nonempty")
    for name, value in hashes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("raw_data_hashes names must be nonempty")
        _validate_hash(value, f"raw_data_hashes.{name}")
    features = certificate["feature_schema"]
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(name, str) or not name.strip() for name in features)
        or len(features) != len(set(features))
    ):
        raise ValueError("feature_schema must contain unique nonempty strings")
    if certificate["feature_schema_hash"] != stable_hash(features):
        raise ValueError("feature_schema_hash does not match feature_schema")
    _validate_relative_artifact(certificate["model_artifact"], "model_artifact")
    calibration_artifact = certificate["calibration_artifact"]
    if calibration_artifact is not None:
        _validate_relative_artifact(calibration_artifact, "calibration_artifact")
    calibration_method = certificate["calibration_method"]
    if calibration_method not in _CALIBRATION_REQUIRES_ARTIFACT:
        raise ValueError(f"Unsupported calibration_method: {calibration_method}")
    artifact_required = _CALIBRATION_REQUIRES_ARTIFACT[calibration_method]
    if artifact_required != (calibration_artifact is not None):
        requirement = "requires" if artifact_required else "forbids"
        raise ValueError(
            f"calibration_method {calibration_method!r} {requirement} "
            "a calibration_artifact"
        )
    versions = certificate["software_versions"]
    if not isinstance(versions, dict) or not versions or any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(value, str)
        or not value.strip()
        for key, value in versions.items()
    ):
        raise ValueError("software_versions must be a nonempty string mapping")
    prediction = certificate["prediction"]
    if not isinstance(prediction, dict) or not prediction:
        raise ValueError("prediction must be a nonempty mapping")
    _validate_finite_tree(prediction, "prediction")
    if all(key in prediction for key in _PROBABILITY_KEYS):
        probabilities = np.asarray([prediction[key] for key in _PROBABILITY_KEYS], dtype=float)
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-10):
            raise ValueError("Outcome probabilities must be nonnegative and sum to one")
    interval = certificate["goal_interval_90"]
    interval_bounds: np.ndarray | None = None
    if interval is not None:
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("goal_interval_90 must contain lower and upper bounds")
        interval_bounds = np.asarray(interval, dtype=float)
        if (
            not np.isfinite(interval_bounds).all()
            or interval_bounds[0] < 0
            or interval_bounds[0] > interval_bounds[1]
        ):
            raise ValueError("goal_interval_90 bounds must be finite, ordered, and nonnegative")
    _validate_prediction_interval_contract(
        certificate["information_set"], prediction, interval_bounds
    )
    ood_score = certificate["out_of_distribution_score"]
    if not isinstance(ood_score, (int, float)) or not np.isfinite(ood_score) or ood_score < 0:
        raise ValueError("OOD score must be finite and nonnegative")
    components = certificate["out_of_distribution_components"]
    if not isinstance(components, dict):
        raise ValueError("out_of_distribution_components must be a mapping")
    _validate_finite_tree(components, "out_of_distribution_components")
    invariants = certificate["invariant_status"]
    if not isinstance(invariants, dict) or not invariants or any(
        not isinstance(key, str) or not isinstance(value, bool)
        for key, value in invariants.items()
    ):
        raise ValueError("invariant_status must be a nonempty boolean mapping")
    failed_invariants = sorted(key for key, value in invariants.items() if not value)
    if failed_invariants:
        raise ValueError(f"Prediction certificate has failed invariants: {failed_invariants}")


def write_certificate(certificate: dict[str, Any], path: Path) -> Path:
    """Persist one full, validated certificate as readable JSON."""
    validate_prediction_certificate(certificate)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(certificate, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _validate_hash(value: Any, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("generated_at_utc must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("generated_at_utc must include a UTC offset")


def _validate_data_cutoff(value: str) -> None:
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("data_cutoff must be an ISO-8601 calendar date") from error


def _validate_prediction_interval_contract(
    information_set: str,
    prediction: dict[str, Any],
    interval: np.ndarray | None,
) -> None:
    predicted_total: float | None = None
    for key in (
        "full_time_total_goals",
        "predicted_full_time_total_goals",
        "predicted_total_goals",
    ):
        if key in prediction:
            predicted_total = float(prediction[key])
            break

    if information_set == "half_time" and "half_time_total_goals" in prediction:
        half_time_total = float(prediction["half_time_total_goals"])
        if predicted_total is not None and predicted_total < half_time_total:
            raise ValueError("Predicted full-time goals cannot be below observed half-time goals")
        if interval is not None and interval[0] < half_time_total:
            raise ValueError("Goal interval cannot extend below observed half-time goals")

    if interval is not None and predicted_total is not None and not (
        interval[0] <= predicted_total <= interval[1]
    ):
        raise ValueError("Goal interval must contain the point prediction")


def _validate_relative_artifact(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"{field} must be a safe POSIX-relative path")


def _validate_finite_tree(value: Any, field: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} keys must be strings")
            _validate_finite_tree(item, f"{field}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_tree(item, f"{field}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not np.isfinite(value):
            raise ValueError(f"{field} must contain only finite numeric values")
    elif value is not None and not isinstance(value, (str, bool)):
        raise ValueError(f"{field} contains unsupported value type {type(value).__name__}")
