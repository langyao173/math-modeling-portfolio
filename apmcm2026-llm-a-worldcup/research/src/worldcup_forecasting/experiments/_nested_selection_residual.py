"""Residual cross-fitting and proper-score helpers for nested selection."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from worldcup_forecasting.models._advanced_count_utils import host_sign
from worldcup_forecasting.models.advanced_count import (
    EquivariantCountConfig,
    joint_batch_from_rates,
)
from worldcup_forecasting.models.residuals import EquivariantResidualCorrector
from worldcup_forecasting.validation.metrics import (
    expected_calibration_error,
    multiclass_brier,
)

from ._nested_selection_types import (
    Clock,
    NestedSelectionConfig,
    StructuralPrediction,
    elapsed_seconds,
    match_ids,
    observed_goals,
    residual_config,
)


def residual_design(
    frame: pd.DataFrame, config: EquivariantCountConfig
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Create explicit even/odd residual features from the structural schema."""
    design = pd.DataFrame(index=frame.index)
    even: list[str] = []
    odd: list[str] = []
    for name in (*config.common_pace_features, *config.even_numeric_features):
        if name in frame and name not in even:
            design[name] = pd.to_numeric(frame[name], errors="coerce")
            even.append(name)
    for name in config.odd_numeric_features:
        if name in frame and name not in odd:
            design[name] = pd.to_numeric(frame[name], errors="coerce")
            odd.append(name)
    for position, (home, away) in enumerate(config.paired_numeric_features):
        home_value = pd.to_numeric(frame[home], errors="coerce")
        away_value = pd.to_numeric(frame[away], errors="coerce")
        even_name, odd_name = f"pair_even_{position}", f"pair_odd_{position}"
        design[even_name] = home_value + away_value
        design[odd_name] = home_value - away_value
        even.append(even_name)
        odd.append(odd_name)
    if config.host_side_column and config.host_side_column in frame:
        design["host_side_signed"] = frame[config.host_side_column].map(host_sign)
        odd.append("host_side_signed")
    if not even:
        design["even_origin"] = 0.0
        even.append("even_origin")
    if not odd:
        design["odd_origin"] = 0.0
        odd.append("odd_origin")
    return design, tuple(even), tuple(odd)


def fit_residual(
    frame: pd.DataFrame,
    design: pd.DataFrame,
    predictions: Sequence[StructuralPrediction],
    even: tuple[str, ...],
    odd: tuple[str, ...],
    cfg: NestedSelectionConfig,
    clock: Clock,
) -> tuple[EquivariantResidualCorrector, float]:
    """Fit the bounded corrector from structural OOF rows only."""
    indices = np.concatenate([np.asarray(item.validation_index) for item in predictions])
    rates = np.vstack([item.rates for item in predictions])
    provenance = []
    identities = match_ids(frame)
    for item in predictions:
        index = np.asarray(item.validation_index)
        provenance.append(
            pd.DataFrame(
                {
                    "match_id": identities[index],
                    "is_oof": True,
                    "prediction_role": "structural_oof",
                    "train_end_year": item.train_end_year,
                    "prediction_year": item.validation_year,
                }
            )
        )
    training_design = design.iloc[indices].reset_index(drop=True)
    usable_even = tuple(name for name in even if not training_design[name].isna().all())
    if not usable_even:
        training_design["even_fit_origin"] = 0.0
        design["even_fit_origin"] = 0.0
        usable_even = ("even_fit_origin",)
    corrector = EquivariantResidualCorrector(
        residual_config(cfg.maximum_selection_year)
    )
    start = clock()
    corrector.fit_oof(
        training_design,
        rates,
        observed_goals(frame.iloc[indices]),
        pd.concat(provenance, ignore_index=True),
        even_features=usable_even,
        odd_features=odd,
    )
    return corrector, elapsed_seconds(clock, start)


def prediction_metrics(
    frame: pd.DataFrame,
    rates: np.ndarray,
    shared_fraction: float,
    model_config: EquivariantCountConfig,
) -> dict[str, float]:
    """Compute proper outcome and observed-score log scores."""
    batch = joint_batch_from_rates(
        rates,
        shared_fraction=shared_fraction,
        max_score=model_config.max_score,
        min_retained_mass=model_config.min_retained_mass,
        max_support=model_config.max_support,
    )
    probabilities = batch.outcome_probabilities()
    goals = observed_goals(frame).astype(int)
    labels = np.where(
        goals[:, 0] > goals[:, 1],
        0,
        np.where(goals[:, 0] == goals[:, 1], 1, 2),
    )
    outcome_mass = probabilities[np.arange(len(labels)), labels]
    support = batch.probability.shape[1]
    score_mass = np.asarray(
        [
            batch.probability[row, home, away]
            if home < support and away < support
            else 0.0
            for row, (home, away) in enumerate(goals)
        ]
    )
    return {
        "match_count": int(len(frame)),
        "log_loss": float(-np.mean(np.log(np.clip(outcome_mass, 1e-15, 1.0)))),
        "multiclass_brier": multiclass_brier(labels, probabilities),
        "ece": expected_calibration_error(labels, probabilities),
        "scoreline_nll": float(-np.mean(np.log(np.clip(score_mass, 1e-15, 1.0)))),
    }


__all__ = ["fit_residual", "prediction_metrics", "residual_design"]
