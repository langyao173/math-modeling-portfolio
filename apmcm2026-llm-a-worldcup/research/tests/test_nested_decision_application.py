"""Tests for separating draw-aware actions from probability calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup_forecasting.experiments.nested_decision import apply_nested_decision


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    training_rows: list[dict[str, object]] = []
    outer_rows: list[dict[str, object]] = []
    for fold_index, outer_year in enumerate((1994, 1998, 2002, 2006)):
        fold = f"outer_{outer_year}"
        for position in range(18):
            label = position % 3
            probability = (
                [0.38, 0.34, 0.28]
                if label == 0
                else [0.36, 0.33, 0.31]
                if label == 1
                else [0.28, 0.34, 0.38]
            )
            training_rows.append(
                {
                    "selection_outer_fold_id": fold,
                    "match_id": fold_index * 1000 + position,
                    "prediction_year": outer_year - 4,
                    "train_end_year": outer_year - 8,
                    "is_oof": True,
                    "home_win": probability[0],
                    "draw": probability[1],
                    "away_win": probability[2],
                    "target_label": label,
                }
            )
        for position, label in enumerate((0, 1, 2, 1, 0, 2)):
            probability = (
                [0.42, 0.33, 0.25]
                if label == 0
                else [0.36, 0.34, 0.30]
                if label == 1
                else [0.25, 0.33, 0.42]
            )
            outer_rows.append(
                {
                    "outer_fold_id": fold,
                    "match_id": 100_000 + fold_index * 100 + position,
                    "prediction_year": outer_year,
                    "home_win": probability[0],
                    "draw": probability[1],
                    "away_win": probability[2],
                    "target_outcome": ("home_win", "draw", "away_win")[label],
                }
            )
    return pd.DataFrame(training_rows), pd.DataFrame(outer_rows)


def test_nested_decision_is_outer_safe_and_never_changes_probabilities() -> None:
    training, outer = _inputs()

    result = apply_nested_decision(training, outer)

    assert len(result.prediction_ledger) == len(outer)
    assert len(result.fold_metrics) == 4
    np.testing.assert_allclose(
        result.prediction_ledger[["home_win", "draw", "away_win"]],
        outer[["home_win", "draw", "away_win"]],
        atol=0.0,
        rtol=0.0,
    )
    assert result.fold_metrics["decision_draw_predictions"].gt(0).all()
    assert result.fold_metrics["probability_layer_changed"].eq(False).all()
    assert set(result.prediction_ledger["decision_label"]) <= {
        "home_win",
        "draw",
        "away_win",
    }
    assert result.metadata["outer_fold_count"] == 4


def test_nested_decision_rejects_non_oof_and_2014_training() -> None:
    training, outer = _inputs()
    contaminated = training.copy()
    contaminated.loc[0, "train_end_year"] = contaminated.loc[0, "prediction_year"]
    with pytest.raises(ValueError, match="OOF|chronolog|earlier"):
        apply_nested_decision(contaminated, outer)

    with pytest.raises(ValueError, match="2014|audit|selection"):
        apply_nested_decision(training, outer.assign(prediction_year=2014))

