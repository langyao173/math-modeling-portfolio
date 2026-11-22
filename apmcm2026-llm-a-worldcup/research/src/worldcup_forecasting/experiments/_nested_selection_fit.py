"""Candidate generation, chronological fitting, and frozen selection metrics."""

from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from worldcup_forecasting.models.advanced_count import (
    EquivariantCountConfig,
    EquivariantCountModel,
)
from worldcup_forecasting.validation.nested import CandidateBudget, InnerTournamentFold

from ._nested_selection_residual import (
    fit_residual,
    prediction_metrics,
    residual_design,
)
from ._nested_selection_runtime import reference_runtime_seconds
from ._nested_selection_types import (
    CandidateSpec,
    Clock,
    NestedSelectionConfig,
    StructuralPrediction,
    candidate_order,
    elapsed_seconds,
    full_config_hash,
    inner_fit_hash,
    match_ids,
    sha256_payload,
)
from .research_features import config_hash


def frozen_candidate_specs() -> tuple[tuple[CandidateSpec, ...], str]:
    """Return the exact 36 candidates and frozen budget-manifest hash."""
    budget = CandidateBudget(
        max_configurations=36,
        maximum_selection_year=2010,
        forbidden_audit_years=(2014, 2018),
    )
    specs: list[CandidateSpec] = []
    for alpha, half_life, shared, residual in product(
        (0.1, 1.0, 10.0), (8.0, 16.0, None), (False, True), (False, True)
    ):
        identity = {
            "alpha": alpha,
            "half_life_years": half_life,
            "shared_score": shared,
            "residual": residual,
        }
        specs.append(
            CandidateSpec(
                alpha=float(alpha),
                half_life_years=half_life,
                shared_score=bool(shared),
                residual=bool(residual),
                candidate_id=budget.register(identity),
            )
        )
    if len(specs) != 36 or len({spec.candidate_id for spec in specs}) != 36:
        raise RuntimeError("Frozen candidate family is not exactly 36 unique specifications")
    return tuple(specs), budget.freeze()


def validate_inputs(frame: pd.DataFrame, cfg: NestedSelectionConfig) -> None:
    """Reject malformed boundaries before any candidate can be fitted."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Nested selection requires a nonempty feature frame")
    invalid_maximum = (
        isinstance(cfg.maximum_selection_year, bool)
        or not isinstance(cfg.maximum_selection_year, (int, np.integer))
        or cfg.maximum_selection_year <= 0
        or cfg.maximum_selection_year > 2010
    )
    if invalid_maximum:
        raise ValueError("maximum selection year cannot exceed the frozen 2010 boundary")
    if any(
        isinstance(year, bool) or not isinstance(year, (int, np.integer))
        for year in cfg.outer_test_years
    ):
        raise ValueError("outer_test_years must contain finite integer editions")
    if not cfg.outer_test_years or max(cfg.outer_test_years) > cfg.maximum_selection_year:
        raise ValueError("2014/2018 audit years cannot enter candidate selection")
    if any(year in cfg.forbidden_audit_years for year in cfg.outer_test_years):
        raise ValueError("A forbidden audit year was requested for selection")
    invalid_inner_count = (
        isinstance(cfg.latest_inner_folds, bool)
        or not isinstance(cfg.latest_inner_folds, (int, np.integer))
        or cfg.latest_inner_folds not in (1, 2, 3)
    )
    if invalid_inner_count:
        raise ValueError("latest_inner_folds must be 1, 2, or 3")
    if isinstance(cfg.seed, bool) or not isinstance(cfg.seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if not isinstance(cfg.information_set, str) or not cfg.information_set.strip():
        raise ValueError("information_set must be nonempty")
    identities = match_ids(frame)
    if pd.isna(identities).any() or pd.Series(identities).duplicated().any():
        raise ValueError("match identifiers must be complete and unique")


def validate_development_boundary(
    frame: pd.DataFrame, cfg: NestedSelectionConfig
) -> None:
    """Reject audit/future editions before any development-frame processing."""
    if cfg.year_column not in frame:
        raise ValueError(f"Missing year column: {cfg.year_column}")
    values = pd.to_numeric(frame[cfg.year_column], errors="raise").to_numpy(float)
    valid = np.isfinite(values).all() and np.allclose(values, np.rint(values))
    if not valid:
        raise ValueError("Development edition years must be finite integers")
    years = values.astype(int)
    forbidden = sorted(set(years).intersection(cfg.forbidden_audit_years))
    if forbidden or (len(years) and int(years.max()) > cfg.maximum_selection_year):
        raise ValueError(
            "Final development selection cannot load 2014/2018 or post-2010 rows"
        )


def development_inner_folds(
    frame: pd.DataFrame, cfg: NestedSelectionConfig
) -> tuple[InnerTournamentFold, ...]:
    """Build all expanding-origin folds through the frozen 2010 cutoff."""
    validate_development_boundary(frame, cfg)
    years = pd.to_numeric(frame[cfg.year_column], errors="raise").to_numpy(int)
    editions = tuple(sorted(int(year) for year in np.unique(years)))
    if len(editions) <= cfg.latest_inner_folds:
        raise ValueError(
            "Final development selection lacks enough editions for inner validation"
        )
    folds: list[InnerTournamentFold] = []
    for validation_year in editions[1:]:
        train = np.flatnonzero(years < validation_year)
        valid = np.flatnonzero(years == validation_year)
        if not len(train) or not len(valid) or int(years[train].max()) >= validation_year:
            raise RuntimeError("Invalid final-development expanding-origin chronology")
        folds.append(
            InnerTournamentFold(
                fold_id=f"final_development__inner_{validation_year}",
                validation_year=validation_year,
                train_index=tuple(int(index) for index in train),
                validation_index=tuple(int(index) for index in valid),
            )
        )
    return tuple(folds)


def ordered_frame(frame: pd.DataFrame, cfg: NestedSelectionConfig) -> pd.DataFrame:
    """Canonicalize row order so hashes and floating-point fitting are repeatable."""
    sort_columns = [cfg.year_column]
    sort_columns.extend(
        name for name in ("Datetime", "prediction_time", "date") if name in frame
    )
    sort_columns.extend(name for name in ("match_id", "MatchID") if name in frame)
    return frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def frame_schema_hash(frame: pd.DataFrame) -> str:
    """Hash ordered names and storage dtypes for feature-schema provenance."""
    return sha256_payload(
        {"columns": list(frame.columns), "dtypes": frame.dtypes.astype(str).to_dict()}
    )


def structural_predictions(
    frame: pd.DataFrame,
    folds: Sequence[InnerTournamentFold],
    model_config: EquivariantCountConfig,
    cfg: NestedSelectionConfig,
    cache: dict[tuple[str, int], StructuralPrediction],
    clock: Clock,
) -> tuple[StructuralPrediction, ...]:
    """Fit/cache strictly chronological structural predictions for inner editions."""
    model_hash = config_hash(model_config)
    predictions: list[StructuralPrediction] = []
    years = pd.to_numeric(frame[cfg.year_column], errors="raise").to_numpy(int)
    identities = match_ids(frame)
    for fold in folds:
        key = (model_hash, fold.validation_year)
        if key not in cache:
            train = np.asarray(fold.train_index, dtype=int)
            valid = np.asarray(fold.validation_index, dtype=int)
            train_end = int(years[train].max())
            if train_end >= fold.validation_year or fold.validation_year > 2010:
                raise RuntimeError("Structural OOF chronology crossed the frozen boundary")
            start = clock()
            model = EquivariantCountModel(model_config).fit(frame.iloc[train])
            rates = model.predict_rates(frame.iloc[valid])
            cache[key] = StructuralPrediction(
                validation_year=fold.validation_year,
                train_end_year=train_end,
                validation_index=fold.validation_index,
                rates=rates,
                shared_fraction=float(model.shared_fraction_),
                runtime_seconds=elapsed_seconds(clock, start),
                fit_index_hash=sha256_payload(
                    {
                        "config_hash": model_hash,
                        "match_ids": identities[train].tolist(),
                        "train_end_year": train_end,
                    }
                ),
            )
        predictions.append(cache[key])
    return tuple(predictions)


def evaluate_spec(
    frame: pd.DataFrame,
    outer_fold_id: str,
    spec: CandidateSpec,
    model_config: EquivariantCountConfig,
    structural: Sequence[StructuralPrediction],
    evaluation_folds: Sequence[InnerTournamentFold],
    cfg: NestedSelectionConfig,
    clock: Clock,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cross-fit one candidate and compute the frozen equal-edition objective."""
    by_year = {prediction.validation_year: prediction for prediction in structural}
    design, even, odd = residual_design(frame, model_config)
    rows: list[dict[str, Any]] = []
    residual_runtime = 0.0
    for fold in evaluation_folds:
        prediction = by_year[fold.validation_year]
        rates = prediction.rates
        prior = [item for item in structural if item.validation_year < fold.validation_year]
        residual_rows, residual_max_year = 0, np.nan
        if spec.residual and prior:
            corrector, elapsed = fit_residual(frame, design, prior, even, odd, cfg, clock)
            start = clock()
            rates = corrector.transform_rates(
                design.iloc[list(prediction.validation_index)], prediction.rates
            )
            residual_runtime += elapsed + elapsed_seconds(clock, start)
            residual_rows = sum(len(item.validation_index) for item in prior)
            residual_max_year = float(max(item.validation_year for item in prior))
            if residual_max_year >= fold.validation_year:
                raise RuntimeError("Residual cross-fit included its evaluation edition")
        rows.append(
            {
                "outer_fold_id": outer_fold_id,
                "outer_test_year": int(outer_fold_id.rsplit("_", 1)[-1]),
                "candidate_id": spec.candidate_id,
                "candidate_order": candidate_order(spec),
                "config_hash": full_config_hash(model_config, spec, cfg),
                "inner_test_year": fold.validation_year,
                "inner_train_end_year": prediction.train_end_year,
                "residual": spec.residual,
                "residual_training_rows": residual_rows,
                "residual_fit_max_year": residual_max_year,
                "fit_index_hash": inner_fit_hash(spec, model_config, cfg, prediction, prior),
                "eligible": True,
                "failure_reason": "",
                **prediction_metrics(
                    frame.iloc[list(prediction.validation_index)],
                    rates,
                    prediction.shared_fraction,
                    model_config,
                ),
            }
        )
    return rows, _summarize_candidate(
        outer_fold_id, spec, model_config, structural, evaluation_folds, rows, residual_runtime, cfg
    )


def _summarize_candidate(
    outer_fold_id: str,
    spec: CandidateSpec,
    model_config: EquivariantCountConfig,
    structural: Sequence[StructuralPrediction],
    folds: Sequence[InnerTournamentFold],
    rows: list[dict[str, Any]],
    residual_runtime: float,
    cfg: NestedSelectionConfig,
) -> dict[str, Any]:
    by_year = {item.validation_year: item for item in structural}
    losses = np.asarray([row["log_loss"] for row in rows], dtype=float)
    measured_runtime = (
        sum(item.runtime_seconds for item in structural)
        if spec.residual
        else sum(by_year[fold.validation_year].runtime_seconds for fold in folds)
    ) + residual_runtime
    runtime = reference_runtime_seconds(spec.candidate_id)
    complexity = float(
        spec.shared_score + 2 * spec.residual + 0.5 * (spec.half_life_years is not None)
    )
    mean_ll = float(losses.mean())
    mean_brier = float(np.mean([row["multiclass_brier"] for row in rows]))
    mean_ece = float(np.mean([row["ece"] for row in rows]))
    sd_ll = float(losses.std(ddof=0))
    objective = (
        mean_ll + 0.25 * mean_brier + 0.05 * mean_ece + 0.05 * sd_ll
        + 0.001 * np.log1p(runtime) + 0.0005 * complexity
    )
    if not np.isfinite(
        (mean_ll, mean_brier, mean_ece, sd_ll, measured_runtime, runtime, objective)
    ).all():
        raise ValueError("Candidate selection metrics/objective must all be finite")
    return {
        "outer_fold_id": outer_fold_id,
        "outer_test_year": int(outer_fold_id.rsplit("_", 1)[-1]),
        "candidate_id": spec.candidate_id,
        "candidate_order": candidate_order(spec),
        "config_hash": full_config_hash(model_config, spec, cfg),
        "alpha": spec.alpha,
        "half_life_years": spec.half_life_years,
        "half_life_label": "none" if spec.half_life_years is None else str(spec.half_life_years),
        "shared_score": spec.shared_score,
        "residual": spec.residual,
        "inner_fold_count": len(rows),
        "inner_test_years": json.dumps([row["inner_test_year"] for row in rows]),
        "mean_log_loss": mean_ll,
        "mean_multiclass_brier": mean_brier,
        "mean_ece": mean_ece,
        "fold_sd_log_loss": sd_ll,
        "measured_runtime_seconds": float(measured_runtime),
        "runtime_seconds": float(runtime),
        "complexity_units": complexity,
        "selection_objective": float(objective),
        "eligible": True,
        "failure_reason": "",
    }


def failed_candidate(
    outer_fold_id: str,
    spec: CandidateSpec,
    model_config: EquivariantCountConfig | None,
    folds: Sequence[InnerTournamentFold],
    cfg: NestedSelectionConfig,
    error: Exception,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Represent numerical candidate failure without making it selectable."""
    reason = f"{type(error).__name__}: {error}"
    common = {
        "outer_fold_id": outer_fold_id,
        "outer_test_year": int(outer_fold_id.rsplit("_", 1)[-1]),
        "candidate_id": spec.candidate_id,
        "candidate_order": candidate_order(spec),
        "config_hash": (
            full_config_hash(model_config, spec, cfg)
            if model_config is not None
            else sha256_payload(
                {"candidate_id": spec.candidate_id, "configuration_available": False}
            )
        ),
    }
    rows = [
        {
            **common,
            "inner_test_year": fold.validation_year,
            "inner_train_end_year": np.nan,
            "residual": spec.residual,
            "residual_training_rows": 0,
            "residual_fit_max_year": np.nan,
            "fit_index_hash": "",
            "match_count": 0,
            "log_loss": np.nan,
            "multiclass_brier": np.nan,
            "ece": np.nan,
            "scoreline_nll": np.nan,
            "eligible": False,
            "failure_reason": reason,
        }
        for fold in folds
    ]
    summary = {
        **common,
        "alpha": spec.alpha,
        "half_life_years": spec.half_life_years,
        "half_life_label": "none" if spec.half_life_years is None else str(spec.half_life_years),
        "shared_score": spec.shared_score,
        "residual": spec.residual,
        "inner_fold_count": len(folds),
        "inner_test_years": json.dumps([fold.validation_year for fold in folds]),
        **{name: np.nan for name in (
            "mean_log_loss", "mean_multiclass_brier", "mean_ece",
            "fold_sd_log_loss", "measured_runtime_seconds", "runtime_seconds",
            "selection_objective",
        )},
        "complexity_units": float(
            spec.shared_score + 2 * spec.residual + 0.5 * (spec.half_life_years is not None)
        ),
        "eligible": False,
        "failure_reason": reason,
    }
    return rows, summary

__all__ = [
    "development_inner_folds", "evaluate_spec", "failed_candidate",
    "frame_schema_hash", "frozen_candidate_specs", "ordered_frame",
    "structural_predictions", "validate_development_boundary", "validate_inputs",
]
