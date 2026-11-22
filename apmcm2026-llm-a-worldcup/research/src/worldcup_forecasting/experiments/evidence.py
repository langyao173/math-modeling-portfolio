"""Convert selected chronological OOF predictions into auditable evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from worldcup_forecasting.validation.joint_metrics import (
    classwise_calibration_diagnostics,
    joint_forecast_metrics,
    total_goal_crps,
)
from worldcup_forecasting.validation.metrics import classification_metrics

_KEYS = ("outer_fold_id", "match_id")
_GRID_COLUMNS = (*_KEYS, "home_goals", "away_goals", "probability")
_OUTCOME_ORDER = ("home_win", "draw", "away_win")


@dataclass(frozen=True)
class ResearchEvidence:
    """Paper-facing tables derived from one selected model's OOF ledger."""

    fold_metrics: pd.DataFrame
    match_losses: pd.DataFrame
    classwise_calibration: pd.DataFrame
    metadata: dict[str, Any]


def reconstruct_joint_batch(
    joint_grid: pd.DataFrame,
    order: pd.DataFrame,
) -> np.ndarray:
    """Reconstruct normalized score matrices in an explicit match-key order."""
    _require_columns(joint_grid, _GRID_COLUMNS, "joint_grid")
    _require_columns(order, _KEYS, "order")
    if joint_grid.empty or order.empty:
        raise ValueError("joint_grid and order must be nonempty")
    if order.duplicated(list(_KEYS)).any():
        raise ValueError("order contains a duplicate match key")
    if joint_grid.duplicated(list(_GRID_COLUMNS[:-1])).any():
        raise ValueError("joint_grid contains a duplicate score cell")
    numeric = joint_grid.loc[:, ["home_goals", "away_goals", "probability"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("joint_grid score coordinates and probabilities must be finite")
    coordinates = numeric[["home_goals", "away_goals"]].to_numpy(float)
    if (
        np.any(coordinates < 0)
        or not np.equal(coordinates, np.floor(coordinates)).all()
        or np.any(numeric["probability"].to_numpy(float) < 0)
    ):
        raise ValueError("joint_grid coordinates must be nonnegative integers and mass nonnegative")

    matrices: list[np.ndarray] = []
    maximum_support = 0
    indexed = (
        joint_grid.assign(
            home_goals=numeric["home_goals"].astype(int),
            away_goals=numeric["away_goals"].astype(int),
            probability=numeric["probability"].astype(float),
        )
        .set_index(list(_KEYS), drop=False)
        .sort_index()
    )
    for key in order.loc[:, list(_KEYS)].itertuples(index=False, name=None):
        if key not in indexed.index:
            raise ValueError(f"joint_grid is missing requested match key {key!r}")
        group = indexed.loc[key]
        if isinstance(group, pd.Series):
            group = group.to_frame().T
        home_values = sorted(group["home_goals"].astype(int).unique().tolist())
        away_values = sorted(group["away_goals"].astype(int).unique().tolist())
        expected_home = list(range(max(home_values) + 1))
        expected_away = list(range(max(away_values) + 1))
        if home_values != expected_home or away_values != expected_away:
            raise ValueError("joint_grid support must be contiguous from zero")
        if len(home_values) != len(away_values) or len(group) != len(home_values) ** 2:
            raise ValueError("joint_grid must contain a complete square rectangular support")
        matrix = np.zeros((len(home_values), len(away_values)), dtype=float)
        matrix[
            group["home_goals"].to_numpy(int),
            group["away_goals"].to_numpy(int),
        ] = group["probability"].to_numpy(float)
        if not np.isclose(matrix.sum(), 1.0, atol=1e-10):
            raise ValueError("Every reconstructed joint matrix must be normalized")
        matrices.append(matrix)
        maximum_support = max(maximum_support, matrix.shape[0])
    result = np.zeros((len(matrices), maximum_support, maximum_support), dtype=float)
    for index, matrix in enumerate(matrices):
        result[index, : matrix.shape[0], : matrix.shape[1]] = matrix
    return result


def summarize_oof_evidence(
    oof_ledger: pd.DataFrame,
    joint_grid: pd.DataFrame,
    *,
    pit_seed: int,
) -> ResearchEvidence:
    """Compute identical-fold outcome, score, interval, and calibration evidence."""
    required = (
        *_KEYS,
        "prediction_year",
        *_OUTCOME_ORDER,
        "target_outcome",
        "observed_home_goals",
        "observed_away_goals",
        "model_id",
        "candidate_id",
        "seed",
    )
    _require_columns(oof_ledger, required, "oof_ledger")
    if oof_ledger.empty or oof_ledger.duplicated(list(_KEYS)).any():
        raise ValueError("oof_ledger must be nonempty with unique outer-fold match keys")
    probability = oof_ledger.loc[:, list(_OUTCOME_ORDER)].to_numpy(float)
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("OOF outcome probabilities must be finite and nonnegative")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("OOF outcome probabilities must sum to one")
    goals = oof_ledger.loc[
        :, ["observed_home_goals", "observed_away_goals"]
    ].to_numpy(int)
    joint = reconstruct_joint_batch(joint_grid, oof_ledger.loc[:, list(_KEYS)])

    fold_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for fold_id, positions in oof_ledger.groupby("outer_fold_id", sort=True).indices.items():
        index = np.asarray(positions, dtype=int)
        block = oof_ledger.iloc[index]
        class_metrics, _ = classification_metrics(
            block["target_outcome"].to_numpy(), probability[index]
        )
        joint_metrics = joint_forecast_metrics(
            goals[index],
            joint[index],
            pit_seed=int(pit_seed) + int(block["prediction_year"].iloc[0]),
        )
        fold_rows.append(
            {
                "outer_fold_id": str(fold_id),
                "test_year": int(block["prediction_year"].iloc[0]),
                "model_id": _one_value(block["model_id"], "model_id"),
                "candidate_id": _one_value(block["candidate_id"], "candidate_id"),
                "seed": int(_one_value(block["seed"], "seed")),
                **class_metrics,
                **joint_metrics,
            }
        )

    label_index = np.asarray(
        [_OUTCOME_ORDER.index(str(value)) for value in oof_ledger["target_outcome"]],
        dtype=int,
    )
    observed_one_hot = np.eye(3)[label_index]
    outcome_loss = -np.log(np.clip(probability[np.arange(len(probability)), label_index], 1e-15, 1))
    brier = np.sum((probability - observed_one_hot) ** 2, axis=1)
    for position, ledger_row in enumerate(oof_ledger.itertuples(index=False)):
        row_joint = joint[position : position + 1]
        row_goals = goals[position : position + 1]
        match_rows.append(
            {
                "outer_fold_id": ledger_row.outer_fold_id,
                "match_id": ledger_row.match_id,
                "test_year": int(ledger_row.prediction_year),
                "target_outcome": ledger_row.target_outcome,
                "outcome_log_loss": float(outcome_loss[position]),
                "multiclass_brier": float(brier[position]),
                "scoreline_log_loss": _single_score_loss(row_goals[0], row_joint[0]),
                "total_goal_crps": total_goal_crps(row_goals, row_joint),
            }
        )
    calibration = pd.DataFrame(
        classwise_calibration_diagnostics(label_index, probability)
    )
    metadata = {
        "row_count": int(len(oof_ledger)),
        "outer_fold_count": int(oof_ledger["outer_fold_id"].nunique()),
        "minimum_prediction_year": int(oof_ledger["prediction_year"].min()),
        "maximum_prediction_year": int(oof_ledger["prediction_year"].max()),
        "pit_seed": int(pit_seed),
        "score_support": int(joint.shape[1]),
    }
    return ResearchEvidence(
        fold_metrics=pd.DataFrame(fold_rows).sort_values("test_year").reset_index(drop=True),
        match_losses=pd.DataFrame(match_rows).sort_values(
            ["test_year", "match_id"]
        ).reset_index(drop=True),
        classwise_calibration=calibration,
        metadata=metadata,
    )


def _single_score_loss(observed: np.ndarray, joint: np.ndarray) -> float:
    home, away = (int(observed[0]), int(observed[1]))
    probability = joint[home, away] if home < len(joint) and away < len(joint) else 0.0
    return float(-np.log(max(float(probability), 1e-15)))


def _one_value(series: pd.Series, label: str) -> Any:
    values = series.drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"Each outer fold must have exactly one {label}")
    return values[0]


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame")
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


__all__ = ["ResearchEvidence", "reconstruct_joint_batch", "summarize_oof_evidence"]
