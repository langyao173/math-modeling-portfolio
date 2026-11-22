"""Tests for applying calibration without crossing an outer-fold boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup_forecasting.experiments.nested_calibration import apply_nested_calibration


def _joint_from_outcome(outcome: np.ndarray) -> np.ndarray:
    matrix = np.full((4, 4), 1e-5)
    masks = [
        np.tril(np.ones((4, 4), dtype=bool), -1),
        np.eye(4, dtype=bool),
        np.triu(np.ones((4, 4), dtype=bool), 1),
    ]
    for target, mask in zip(outcome, masks, strict=True):
        matrix[mask] *= target / matrix[mask].sum()
    return matrix / matrix.sum()


def _append_grid(
    rows: list[dict[str, object]], fold: str, match_id: int, matrix: np.ndarray
) -> None:
    for home in range(matrix.shape[0]):
        for away in range(matrix.shape[1]):
            rows.append(
                {
                    "outer_fold_id": fold,
                    "match_id": match_id,
                    "home_goals": home,
                    "away_goals": away,
                    "probability": matrix[home, away],
                }
            )


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inner_rows: list[dict[str, object]] = []
    inner_grid: list[dict[str, object]] = []
    for position, (year, outcome, score) in enumerate(
        [
            (1986, [0.55, 0.25, 0.20], (1, 0)),
            (1986, [0.25, 0.50, 0.25], (0, 0)),
            (1990, [0.20, 0.25, 0.55], (0, 1)),
            (1990, [0.60, 0.20, 0.20], (2, 0)),
            (1994, [0.30, 0.45, 0.25], (1, 1)),
            (1994, [0.20, 0.20, 0.60], (0, 2)),
        ]
    ):
        match_id = 100 + position
        matrix = _joint_from_outcome(np.asarray(outcome))
        inner_rows.append(
            {
                "outer_fold_id": "outer_1998",
                "inner_fold_id": f"inner_{year}",
                "match_id": match_id,
                "prediction_year": year,
                "train_end_year": year - 4,
                "is_oof": True,
                "observed_home_goals": score[0],
                "observed_away_goals": score[1],
            }
        )
        _append_grid(inner_grid, "outer_1998", match_id, matrix)

    outer_rows: list[dict[str, object]] = []
    outer_grid: list[dict[str, object]] = []
    for position, (outcome, score) in enumerate(
        [([0.45, 0.30, 0.25], (1, 0)), ([0.25, 0.30, 0.45], (0, 1))]
    ):
        match_id = 200 + position
        matrix = _joint_from_outcome(np.asarray(outcome))
        outer_rows.append(
            {
                "outer_fold_id": "outer_1998",
                "inner_fold_id": "selected_refit",
                "match_id": match_id,
                "prediction_year": 1998,
                "train_end_year": 1994,
                "is_oof": True,
                "model_id": "advanced",
                "home_win": outcome[0],
                "draw": outcome[1],
                "away_win": outcome[2],
                "observed_home_goals": score[0],
                "observed_away_goals": score[1],
            }
        )
        _append_grid(outer_grid, "outer_1998", match_id, matrix)
    return (
        pd.DataFrame(inner_rows),
        pd.DataFrame(inner_grid),
        pd.DataFrame(outer_rows),
        pd.DataFrame(outer_grid),
    )


def test_nested_calibration_uses_inner_oof_only_and_preserves_coherence() -> None:
    inner, inner_grid, outer, outer_grid = _inputs()

    # A real multi-fold ledger retains nonconsecutive source indices after filtering.
    # Chronology resolution must be positional, never pandas-index aligned.
    inner.index = np.arange(100, 100 + 3 * len(inner), 3)

    result = apply_nested_calibration(inner, inner_grid, outer, outer_grid)

    assert len(result.oof_ledger) == len(outer)
    np.testing.assert_allclose(
        result.oof_ledger[["home_win", "draw", "away_win"]].sum(axis=1),
        1.0,
        atol=1e-12,
    )
    assert result.selection_table.loc[0, "inner_max_prediction_year"] == 1994
    assert result.selection_table.loc[0, "outer_prediction_year"] == 1998
    assert result.selection_table.loc[0, "inner_oof_only"]
    assert not result.decision_training_ledger.empty
    assert (
        result.decision_training_ledger["train_end_year"]
        < result.decision_training_ledger["prediction_year"]
    ).all()
    margins = []
    for _, group in result.joint_grid.groupby(["outer_fold_id", "match_id"], sort=False):
        matrix = group.pivot(
            index="home_goals", columns="away_goals", values="probability"
        ).to_numpy()
        margins.append([np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()])
    np.testing.assert_allclose(
        np.asarray(margins),
        result.oof_ledger[["home_win", "draw", "away_win"]],
        atol=1e-10,
    )


def test_nested_calibration_rejects_outer_or_inner_leakage() -> None:
    inner, inner_grid, outer, outer_grid = _inputs()
    contaminated = inner.copy()
    contaminated.loc[0, "prediction_year"] = 1998
    with pytest.raises(ValueError, match="inner|outer|earlier"):
        apply_nested_calibration(contaminated, inner_grid, outer, outer_grid)

    audit = outer.assign(prediction_year=2014, train_end_year=2010)
    with pytest.raises(ValueError, match="2014|selection|audit"):
        apply_nested_calibration(inner, inner_grid, audit, outer_grid)
