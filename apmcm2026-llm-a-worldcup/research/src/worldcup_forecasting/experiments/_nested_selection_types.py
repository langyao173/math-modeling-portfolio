"""Shared immutable types and provenance helpers for nested selection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from worldcup_forecasting.models.advanced_count import EquivariantCountConfig
from worldcup_forecasting.models.residuals import EquivariantResidualConfig

CountConfigFactory = Callable[..., EquivariantCountConfig]
Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One member of the immutable 36-configuration selection family."""

    alpha: float
    half_life_years: float | None
    shared_score: bool
    residual: bool
    candidate_id: str


@dataclass(frozen=True, slots=True)
class NestedSelectionConfig:
    """Controls that do not alter the preregistered candidate family."""

    outer_test_years: tuple[int, ...] = (1986, 1990, 1994, 1998, 2002, 2006, 2010)
    year_column: str = "Year"
    latest_inner_folds: int = 3
    maximum_selection_year: int = 2010
    forbidden_audit_years: tuple[int, ...] = (2014, 2018)
    seed: int = 31415
    information_set: str = "pre_match"


@dataclass(frozen=True)
class NestedSelectionResult:
    """Tabular artifacts from a complete nested selection run."""

    selected_specs: pd.DataFrame
    candidate_metrics: pd.DataFrame
    inner_fold_metrics: pd.DataFrame
    selected_inner_oof_ledger: pd.DataFrame
    selected_inner_joint_grid: pd.DataFrame
    oof_ledger: pd.DataFrame
    rate_ledger: pd.DataFrame
    joint_grid: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FinalDevelopmentSelectionResult:
    """Frozen 2014-development choice made without loading audit-edition rows."""

    selection_config: NestedSelectionConfig
    selected_spec: CandidateSpec
    selected_model_config: EquivariantCountConfig
    selected_candidate_hash: str
    selected_config_hash: str
    selected_summary: pd.DataFrame
    candidate_metrics: pd.DataFrame
    inner_fold_metrics: pd.DataFrame
    selected_inner_oof_ledger: pd.DataFrame
    selected_inner_joint_grid: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StructuralPrediction:
    """One structural-model prediction from an immutable inner split."""

    validation_year: int
    train_end_year: int
    validation_index: tuple[int, ...]
    rates: np.ndarray
    shared_fraction: float
    runtime_seconds: float
    fit_index_hash: str


def sha256_payload(payload: Any) -> str:
    """Hash a canonical JSON representation of a provenance object."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frame_content_hash(frame: pd.DataFrame) -> str:
    """Hash ordered values, columns, and dtypes for exact fit-data identity."""
    hashed_rows = pd.util.hash_pandas_object(
        frame, index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    return sha256_payload(
        {
            "columns": list(frame.columns),
            "dtypes": frame.dtypes.astype(str).to_dict(),
            "row_hashes": [int(value) for value in hashed_rows],
        }
    )


def residual_config(maximum_selection_year: int) -> EquivariantResidualConfig:
    """Return the fixed bounded residual configuration."""
    return EquivariantResidualConfig(
        enabled=True,
        missing_policy="impute",
        maximum_fit_prediction_year=maximum_selection_year,
    )


def full_config_hash(
    model_config: EquivariantCountConfig,
    spec: CandidateSpec,
    cfg: NestedSelectionConfig,
) -> str:
    """Hash both structural and fixed residual configuration state."""
    residual = (
        asdict(residual_config(cfg.maximum_selection_year))
        if spec.residual
        else {"enabled": False}
    )
    return sha256_payload({"structural": asdict(model_config), "residual": residual})


def inner_fit_hash(
    spec: CandidateSpec,
    model_config: EquivariantCountConfig,
    cfg: NestedSelectionConfig,
    prediction: StructuralPrediction,
    prior: Sequence[StructuralPrediction],
) -> str:
    """Hash one structural/residual cross-fit lineage."""
    return sha256_payload(
        {
            "candidate_id": spec.candidate_id,
            "config_hash": full_config_hash(model_config, spec, cfg),
            "structural_fit": prediction.fit_index_hash,
            "residual_training_fits": (
                [item.fit_index_hash for item in prior] if spec.residual else []
            ),
        }
    )


def candidate_order(spec: CandidateSpec) -> int:
    """Return the frozen product-grid position for deterministic output."""
    half_order = {8.0: 0, 16.0: 1, None: 2}
    return (
        (0.1, 1.0, 10.0).index(spec.alpha) * 12
        + half_order[spec.half_life_years] * 4
        + int(spec.shared_score) * 2
        + int(spec.residual)
    )


def elapsed_seconds(clock: Clock, start: float) -> float:
    """Read a finite nondecreasing injected runtime clock."""
    elapsed = float(clock() - start)
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Runtime clock must produce finite nondecreasing values")
    return elapsed


def validate_candidate_config(
    spec: CandidateSpec, model_config: EquivariantCountConfig
) -> None:
    """Ensure a factory cannot relabel a different model as a frozen candidate."""
    alpha = model_config.alpha
    half_life = model_config.time_decay_half_life_years
    half_matches = (
        half_life is None and spec.half_life_years is None
    ) or (
        half_life is not None
        and not isinstance(half_life, (bool, np.bool_))
        and np.isfinite(float(half_life))
        and spec.half_life_years is not None
        and float(half_life) == float(spec.half_life_years)
    )
    matches = (
        not isinstance(alpha, (bool, np.bool_))
        and np.isfinite(float(alpha))
        and float(alpha) == spec.alpha
        and half_matches
        and isinstance(model_config.shared_score, (bool, np.bool_))
        and bool(model_config.shared_score) == spec.shared_score
    )
    if not matches:
        raise ValueError(
            "Count-config factory output does not match the frozen candidate spec"
        )


def match_ids(frame: pd.DataFrame) -> np.ndarray:
    """Return explicit immutable match identities; never synthesize row numbers."""
    for name in ("match_id", "MatchID"):
        if name in frame:
            return frame[name].to_numpy()
    raise ValueError("Nested selection requires an explicit match_id or MatchID column")


def observed_goals(frame: pd.DataFrame) -> np.ndarray:
    """Resolve complete nonnegative integer score targets."""
    alternatives = (
        ("target_home_goals", "target_away_goals"),
        ("Home Team Goals", "Away Team Goals"),
        ("home_goals", "away_goals"),
    )
    for home, away in alternatives:
        if home in frame and away in frame:
            values = frame[[home, away]].apply(pd.to_numeric, errors="raise").to_numpy(float)
            valid = (
                np.isfinite(values).all()
                and np.all(values >= 0)
                and np.allclose(values, np.rint(values), atol=1e-12)
            )
            if valid:
                return values
    raise ValueError("Could not resolve finite nonnegative observed goal columns")


def prediction_times(frame: pd.DataFrame, year_column: str) -> pd.Series:
    """Resolve prediction timestamps without using future information."""
    for name in ("Datetime", "prediction_time", "date"):
        if name in frame:
            values = pd.to_datetime(frame[name], errors="coerce")
            if values.notna().all():
                return values.reset_index(drop=True)
    years = pd.to_numeric(frame[year_column], errors="raise").astype(int)
    return pd.to_datetime(years.astype(str) + "-01-01").reset_index(drop=True)


__all__ = [
    "CandidateSpec",
    "Clock",
    "CountConfigFactory",
    "FinalDevelopmentSelectionResult",
    "NestedSelectionConfig",
    "NestedSelectionResult",
    "StructuralPrediction",
    "candidate_order",
    "elapsed_seconds",
    "frame_content_hash",
    "full_config_hash",
    "inner_fit_hash",
    "match_ids",
    "observed_goals",
    "prediction_times",
    "residual_config",
    "sha256_payload",
    "validate_candidate_config",
]
