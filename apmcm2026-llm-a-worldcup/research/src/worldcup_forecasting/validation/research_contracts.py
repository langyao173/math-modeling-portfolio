"""Fail-closed validators for advanced research evidence artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OUTCOME_LABELS = ("home_win", "draw", "away_win")
ABLATION_COMPONENTS = {
    "A0": "none_full_model",
    "A1": "dynamic_ratings",
    "A2": "player_features",
    "A3": "era_effects",
    "A4": "correlated_count_model",
    "A5": "machine_learning_residual",
    "A6": "probability_calibration",
    "A7": "hierarchical_shrinkage",
    "A8": "symmetry_constraint",
}
STRESS_SCENARIOS = (
    "new_or_rare_team",
    "unknown_team_name",
    "player_absence",
    "extreme_scoreline",
    "high_halftime_score",
    "host_to_neutral",
    "group_to_knockout",
    "team_swap",
    "drop_10pct_history",
    "time_decay_perturbation",
    "random_seed_perturbation",
    "format_2026_out_of_range",
)

_OOF_REQUIRED_COLUMNS = (
    "match_id",
    "prediction_year",
    "prediction_time",
    "train_end_year",
    "outer_fold_id",
    "inner_fold_id",
    "candidate_id",
    "model_id",
    "seed",
    "information_set",
    "fit_index_hash",
    "feature_schema_hash",
    "home_win",
    "draw",
    "away_win",
    "target_outcome",
    "observed_home_goals",
    "observed_away_goals",
    "is_oof",
)


def _require_nonempty_frame(frame: pd.DataFrame, *, artifact: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{artifact} must be a pandas DataFrame")
    if frame.empty:
        raise ValueError(f"{artifact} must be nonempty")
    if not frame.columns.is_unique:
        raise ValueError(f"{artifact} column names must be unique")


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, artifact: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{artifact} is missing required columns: {missing}")


def _require_nonempty_text(frame: pd.DataFrame, columns: Sequence[str], *, artifact: str) -> None:
    for column in columns:
        values = frame[column]
        invalid = values.isna() | values.astype(str).str.strip().eq("")
        if bool(invalid.any()):
            raise ValueError(f"{artifact} column {column!r} contains an empty identifier or hash")


def _numeric_frame(frame: pd.DataFrame, columns: Sequence[str], *, artifact: str) -> pd.DataFrame:
    converted = pd.DataFrame(index=frame.index)
    for column in columns:
        try:
            converted[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{artifact} column {column!r} must be numeric") from exc
    if not np.isfinite(converted.to_numpy(dtype=float)).all():
        raise ValueError(f"{artifact} numeric values must all be finite")
    return converted


def _require_all_stored_numeric_values_finite(frame: pd.DataFrame, *, artifact: str) -> None:
    """Reject non-finite optional diagnostics as well as required metrics."""
    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        return
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{artifact} contains a non-finite stored numeric value")


def _boolean_series(series: pd.Series, *, column: str, artifact: str) -> pd.Series:
    converted: list[bool] = []
    for value in series.tolist():
        if pd.isna(value):
            raise ValueError(f"{artifact} column {column!r} contains a missing boolean")
        if isinstance(value, (bool, np.bool_)):
            converted.append(bool(value))
            continue
        if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            converted.append(bool(value))
            continue
        if isinstance(value, str) and value.casefold() in {"true", "false", "1", "0"}:
            converted.append(value.casefold() in {"true", "1"})
            continue
        raise ValueError(f"{artifact} column {column!r} must contain booleans")
    return pd.Series(converted, index=series.index, dtype=bool)


def _schema_hash(frame: pd.DataFrame) -> str:
    """Hash the ordered public schema rather than storage-specific CSV dtypes."""
    payload = {
        "contract": "worldcup_oof_v1",
        "columns": list(frame.columns),
        "required_columns": list(_OOF_REQUIRED_COLUMNS),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_oof_frame(
    frame: pd.DataFrame,
    *,
    probability_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Validate one auditable, match-level out-of-fold prediction ledger."""
    artifact = "OOF prediction frame"
    _require_nonempty_frame(frame, artifact=artifact)
    _require_columns(frame, _OOF_REQUIRED_COLUMNS, artifact=artifact)
    _require_all_stored_numeric_values_finite(frame, artifact=artifact)
    if not np.isfinite(probability_tolerance) or probability_tolerance <= 0:
        raise ValueError("probability_tolerance must be finite and positive")

    identifier_columns = (
        "outer_fold_id",
        "inner_fold_id",
        "candidate_id",
        "model_id",
        "information_set",
        "fit_index_hash",
        "feature_schema_hash",
    )
    _require_nonempty_text(frame, identifier_columns, artifact=artifact)
    if bool(frame["match_id"].isna().any()):
        raise ValueError("OOF prediction frame match_id values must be nonmissing")

    key_columns = [
        "match_id",
        "outer_fold_id",
        "inner_fold_id",
        "candidate_id",
        "model_id",
        "seed",
        "information_set",
    ]
    if bool(frame.duplicated(key_columns, keep=False).any()):
        raise ValueError("OOF prediction rows have a duplicate scientific key; keys must be unique")

    numeric = _numeric_frame(
        frame,
        (
            "prediction_year",
            "train_end_year",
            "seed",
            "home_win",
            "draw",
            "away_win",
            "observed_home_goals",
            "observed_away_goals",
        ),
        artifact=artifact,
    )
    integer_columns = (
        "prediction_year",
        "train_end_year",
        "seed",
        "observed_home_goals",
        "observed_away_goals",
    )
    for column in integer_columns:
        values = numeric[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"OOF prediction column {column!r} must contain integers")
    if bool((numeric[["observed_home_goals", "observed_away_goals"]] < 0).any().any()):
        raise ValueError("OOF observed goals must be nonnegative")

    probabilities = numeric[list(OUTCOME_LABELS)].to_numpy(dtype=float)
    if bool((probabilities < 0.0).any()) or bool((probabilities > 1.0).any()):
        raise ValueError("OOF probabilities must lie in [0, 1]")
    simplex_error = np.abs(probabilities.sum(axis=1) - 1.0)
    if float(simplex_error.max()) > probability_tolerance:
        raise ValueError("OOF probabilities must sum to one on every row")

    is_oof = _boolean_series(frame["is_oof"], column="is_oof", artifact=artifact)
    if not bool(is_oof.all()):
        raise ValueError("OOF prediction frame contains a non-OOF (in-sample) row")
    train_year = numeric["train_end_year"].to_numpy(dtype=int)
    prediction_year = numeric["prediction_year"].to_numpy(dtype=int)
    if bool((train_year >= prediction_year).any()):
        raise ValueError(
            "train_end_year must be strictly earlier than prediction_year for every OOF row"
        )

    prediction_time = pd.to_datetime(frame["prediction_time"], errors="coerce", utc=True)
    if bool(prediction_time.isna().any()):
        raise ValueError("OOF prediction_time values must be valid timestamps")
    if not np.array_equal(prediction_time.dt.year.to_numpy(dtype=int), prediction_year):
        raise ValueError("OOF prediction_time year must equal prediction_year")

    targets = frame["target_outcome"].astype(str)
    unknown_targets = sorted(set(targets).difference(OUTCOME_LABELS))
    if unknown_targets:
        raise ValueError(f"OOF target_outcome contains unknown labels: {unknown_targets}")
    home_goals = numeric["observed_home_goals"].to_numpy(dtype=int)
    away_goals = numeric["observed_away_goals"].to_numpy(dtype=int)
    derived_targets = np.where(
        home_goals > away_goals,
        "home_win",
        np.where(home_goals < away_goals, "away_win", "draw"),
    )
    if not np.array_equal(targets.to_numpy(dtype=str), derived_targets):
        raise ValueError("OOF target_outcome is inconsistent with the observed score")

    fold_year_counts = (
        frame.assign(_prediction_year=prediction_year)
        .groupby("outer_fold_id", dropna=False)["_prediction_year"]
        .nunique()
    )
    if bool((fold_year_counts != 1).any()):
        raise ValueError("Each outer_fold_id must identify exactly one prediction year")

    fit_group = [
        "outer_fold_id",
        "inner_fold_id",
        "candidate_id",
        "model_id",
        "seed",
        "information_set",
    ]
    fit_hash_counts = frame.groupby(fit_group, dropna=False)["fit_index_hash"].nunique()
    if bool((fit_hash_counts != 1).any()):
        raise ValueError("fit_index_hash must be identical within each fitted model group")
    feature_group = ["candidate_id", "model_id", "information_set"]
    schema_counts = frame.groupby(feature_group, dropna=False)["feature_schema_hash"].nunique()
    if bool((schema_counts != 1).any()):
        raise ValueError("feature_schema_hash must be stable for each candidate model schema")

    return {
        "row_count": int(len(frame)),
        "outer_fold_count": int(frame["outer_fold_id"].nunique()),
        "candidate_count": int(frame["candidate_id"].nunique()),
        "minimum_prediction_year": int(prediction_year.min()),
        "maximum_prediction_year": int(prediction_year.max()),
        "minimum_training_gap_years": int((prediction_year - train_year).min()),
        "maximum_probability_simplex_error": float(simplex_error.max()),
        "all_rows_out_of_fold": True,
        "schema_hash": _schema_hash(frame),
    }


def write_oof_predictions(frame: pd.DataFrame, destination: str | Path) -> Path:
    """Atomically persist a validated OOF ledger as index-free UTF-8 CSV."""
    validate_oof_frame(frame)
    path = Path(destination)
    if path.exists() and path.is_dir():
        raise ValueError(f"OOF destination is a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            frame.to_csv(handle, index=False, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        restored = pd.read_csv(temporary_name)
        if list(restored.columns) != list(frame.columns) or len(restored) != len(frame):
            raise RuntimeError("Persisted OOF CSV failed its schema/row-count round-trip")
        validate_oof_frame(restored)
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def validate_ablation_matrix(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate the complete paired A0--A8 component-removal experiment."""
    artifact = "A0-A8 ablation matrix"
    required = (
        "variant_id",
        "removed_component",
        "outer_fold_id",
        "test_year",
        "seed",
        "evaluation_index_hash",
        "log_loss",
        "multiclass_brier",
        "runtime_seconds",
        "structural_tests_passed",
    )
    _require_nonempty_frame(frame, artifact=artifact)
    _require_columns(frame, required, artifact=artifact)
    _require_all_stored_numeric_values_finite(frame, artifact=artifact)
    _require_nonempty_text(
        frame,
        ("variant_id", "removed_component", "outer_fold_id", "evaluation_index_hash"),
        artifact=artifact,
    )

    variants = set(frame["variant_id"].astype(str))
    expected_variants = set(ABLATION_COMPONENTS)
    if variants != expected_variants:
        missing = sorted(expected_variants.difference(variants))
        extra = sorted(variants.difference(expected_variants))
        raise ValueError(f"A0-A8 ablation matrix is not complete; missing={missing}, extra={extra}")
    observed_mapping = (
        frame[["variant_id", "removed_component"]]
        .drop_duplicates()
        .groupby("variant_id", dropna=False)["removed_component"]
        .agg(list)
    )
    for variant, expected_component in ABLATION_COMPONENTS.items():
        components = observed_mapping.get(variant, [])
        if components != [expected_component]:
            raise ValueError(
                f"Ablation {variant} must remove exactly {expected_component!r}; "
                f"observed {components}"
            )

    numeric = _numeric_frame(
        frame,
        ("test_year", "seed", "log_loss", "multiclass_brier", "runtime_seconds"),
        artifact=artifact,
    )
    for column in ("test_year", "seed"):
        values = numeric[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Ablation column {column!r} must contain integers")
    if bool((numeric[["log_loss", "multiclass_brier", "runtime_seconds"]] < 0).any().any()):
        raise ValueError("Ablation losses and runtimes must be nonnegative")
    structural = _boolean_series(
        frame["structural_tests_passed"],
        column="structural_tests_passed",
        artifact=artifact,
    )
    if not bool(structural.all()):
        raise ValueError("Ablation structural invariants did not all pass")

    normalized = frame.assign(
        _test_year=numeric["test_year"].astype(int),
        _seed=numeric["seed"].astype(int),
        _variant=frame["variant_id"].astype(str),
    )
    unit_columns = ["outer_fold_id", "_test_year", "_seed"]
    key_columns = [*unit_columns, "_variant"]
    if bool(normalized.duplicated(key_columns, keep=False).any()):
        raise ValueError("Ablation matrix contains duplicate variant/evaluation keys")
    for unit, group in normalized.groupby(unit_columns, dropna=False, sort=False):
        unit_variants = set(group["_variant"])
        if unit_variants != expected_variants or len(group) != len(expected_variants):
            raise ValueError(f"Ablation Cartesian matrix is incomplete for evaluation unit {unit}")
        if group["evaluation_index_hash"].nunique() != 1:
            raise ValueError(
                "Ablation variants are not paired: evaluation_index_hash must be identical"
            )

    fold_years = normalized.groupby("outer_fold_id", dropna=False)["_test_year"].nunique()
    if bool((fold_years != 1).any()):
        raise ValueError("Each ablation outer_fold_id must identify one test_year")
    fold_hashes = normalized.groupby(["outer_fold_id", "_test_year"], dropna=False)[
        "evaluation_index_hash"
    ].nunique()
    if bool((fold_hashes != 1).any()):
        raise ValueError(
            "Paired ablation seeds must use an identical evaluation_index_hash per outer fold"
        )

    return {
        "row_count": int(len(frame)),
        "evaluation_unit_count": int(normalized.groupby(unit_columns).ngroups),
        "variants": [f"A{index}" for index in range(9)],
        "complete_cartesian_matrix": True,
        "paired_evaluation_indices": True,
        "all_structural_tests_passed": True,
    }


def validate_stress_matrix(
    frame: pd.DataFrame,
    *,
    probability_tolerance: float = 1e-10,
    symmetry_tolerance: float = 1e-10,
    halftime_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Validate the preregistered 12-scenario structural stress matrix."""
    artifact = "research stress matrix"
    required = (
        "scenario_id",
        "model_id",
        "seed",
        "probability_simplex_error",
        "symmetry_error",
        "halftime_violation",
        "finite_outputs",
        "structural_tests_passed",
        "metric_delta",
    )
    _require_nonempty_frame(frame, artifact=artifact)
    _require_columns(frame, required, artifact=artifact)
    _require_all_stored_numeric_values_finite(frame, artifact=artifact)
    _require_nonempty_text(frame, ("scenario_id", "model_id"), artifact=artifact)
    tolerances = (probability_tolerance, symmetry_tolerance, halftime_tolerance)
    if any(not np.isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("Stress-matrix tolerances must be finite and nonnegative")

    observed_scenarios = set(frame["scenario_id"].astype(str))
    expected_scenarios = set(STRESS_SCENARIOS)
    if observed_scenarios != expected_scenarios:
        missing = sorted(expected_scenarios.difference(observed_scenarios))
        extra = sorted(observed_scenarios.difference(expected_scenarios))
        raise ValueError(f"Stress scenario matrix is incomplete; missing={missing}, extra={extra}")

    numeric = _numeric_frame(
        frame,
        (
            "seed",
            "probability_simplex_error",
            "symmetry_error",
            "halftime_violation",
            "metric_delta",
        ),
        artifact=artifact,
    )
    seeds = numeric["seed"].to_numpy(dtype=float)
    if not np.equal(seeds, np.floor(seeds)).all():
        raise ValueError("Stress-matrix seeds must be integers")
    nonnegative_error_columns = (
        "probability_simplex_error",
        "symmetry_error",
        "halftime_violation",
    )
    if bool((numeric[list(nonnegative_error_columns)] < 0).any().any()):
        raise ValueError("Stress-matrix invariant errors must be nonnegative")

    finite_outputs = _boolean_series(
        frame["finite_outputs"], column="finite_outputs", artifact=artifact
    )
    structural = _boolean_series(
        frame["structural_tests_passed"],
        column="structural_tests_passed",
        artifact=artifact,
    )
    if not bool(finite_outputs.all()):
        raise ValueError("Stress matrix contains non-finite model outputs")
    if not bool(structural.all()):
        raise ValueError("Stress matrix contains a failed structural invariant")
    if float(numeric["probability_simplex_error"].max()) > probability_tolerance:
        raise ValueError("Stress matrix violates the probability-simplex invariant")
    if float(numeric["symmetry_error"].max()) > symmetry_tolerance:
        raise ValueError("Stress matrix violates the team-swap symmetry invariant")
    if float(numeric["halftime_violation"].max()) > halftime_tolerance:
        raise ValueError("Stress matrix violates the full-time/half-time invariant")

    normalized = frame.assign(_seed=numeric["seed"].astype(int))
    key_columns = ["model_id", "scenario_id", "_seed"]
    if bool(normalized.duplicated(key_columns, keep=False).any()):
        raise ValueError("Stress matrix contains duplicate model/scenario/seed rows")
    all_seeds = tuple(sorted(int(seed) for seed in normalized["_seed"].unique()))
    if len(all_seeds) < 2:
        raise ValueError("Stress matrix requires at least two random seeds")
    expected_pairs = {(scenario, seed) for scenario in STRESS_SCENARIOS for seed in all_seeds}
    for model_id, group in normalized.groupby("model_id", dropna=False, sort=False):
        observed_pairs = set(zip(group["scenario_id"], group["_seed"], strict=False))
        if observed_pairs != expected_pairs:
            raise ValueError(
                f"Stress scenario/seed Cartesian matrix is incomplete for model {model_id!r}"
            )

    return {
        "row_count": int(len(frame)),
        "model_count": int(frame["model_id"].nunique()),
        "scenario_count": len(STRESS_SCENARIOS),
        "seed_count": len(all_seeds),
        "complete_seed_matrix": True,
        "all_finite_outputs": True,
        "all_structural_invariants_passed": True,
        "maximum_probability_simplex_error": float(numeric["probability_simplex_error"].max()),
        "maximum_symmetry_error": float(numeric["symmetry_error"].max()),
        "maximum_halftime_violation": float(numeric["halftime_violation"].max()),
    }


__all__ = [
    "ABLATION_COMPONENTS",
    "OUTCOME_LABELS",
    "STRESS_SCENARIOS",
    "validate_ablation_matrix",
    "validate_oof_frame",
    "validate_stress_matrix",
    "write_oof_predictions",
]
