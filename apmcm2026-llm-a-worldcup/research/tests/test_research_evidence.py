"""Tests for turning selected OOF ledgers into paper-grade evidence tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup_forecasting.experiments.evidence import (
    reconstruct_joint_batch,
    summarize_oof_evidence,
)


def _ledgers() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    grids: list[dict[str, object]] = []
    cases = [
        ("outer_1994", 1994, 11, 1, 0, [0.7, 0.2, 0.1]),
        ("outer_1994", 1994, 12, 0, 0, [0.2, 0.6, 0.2]),
        ("outer_1998", 1998, 13, 0, 1, [0.1, 0.2, 0.7]),
        ("outer_1998", 1998, 14, 2, 1, [0.6, 0.2, 0.2]),
    ]
    for fold, year, match_id, home, away, outcome in cases:
        label = "home_win" if home > away else "away_win" if home < away else "draw"
        rows.append(
            {
                "outer_fold_id": fold,
                "match_id": match_id,
                "prediction_year": year,
                "home_win": outcome[0],
                "draw": outcome[1],
                "away_win": outcome[2],
                "target_outcome": label,
                "observed_home_goals": home,
                "observed_away_goals": away,
                "model_id": "advanced",
                "candidate_id": "c" * 64,
                "seed": 17,
            }
        )
        matrix = np.full((4, 4), 1e-4)
        matrix[home, away] = 1.0
        matrix /= matrix.sum()
        for score_home in range(4):
            for score_away in range(4):
                grids.append(
                    {
                        "outer_fold_id": fold,
                        "match_id": match_id,
                        "home_goals": score_home,
                        "away_goals": score_away,
                        "probability": matrix[score_home, score_away],
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(grids)


def test_reconstruct_joint_batch_preserves_ledger_order_and_mass() -> None:
    oof, grid = _ledgers()
    order = oof.iloc[[2, 0]][["outer_fold_id", "match_id"]]

    joint = reconstruct_joint_batch(grid, order)

    assert joint.shape == (2, 4, 4)
    np.testing.assert_allclose(joint.sum(axis=(1, 2)), 1.0, atol=1e-12)
    assert np.unravel_index(np.argmax(joint[0]), joint[0].shape) == (0, 1)
    assert np.unravel_index(np.argmax(joint[1]), joint[1].shape) == (1, 0)


def test_summarize_oof_evidence_returns_fold_match_and_calibration_tables() -> None:
    oof, grid = _ledgers()

    evidence = summarize_oof_evidence(oof, grid, pit_seed=29)

    assert set(evidence.fold_metrics["test_year"]) == {1994, 1998}
    assert len(evidence.fold_metrics) == 2
    assert len(evidence.match_losses) == 4
    assert len(evidence.classwise_calibration) == 3
    assert evidence.fold_metrics["scoreline_log_loss"].max() < 0.01
    assert evidence.match_losses["outcome_log_loss"].ge(0).all()
    assert evidence.metadata["outer_fold_count"] == 2


def test_reconstruct_joint_batch_fails_closed_on_incomplete_or_duplicate_cells() -> None:
    oof, grid = _ledgers()
    order = oof.iloc[:1][["outer_fold_id", "match_id"]]
    missing = grid.drop(index=grid.index[0])
    duplicate = pd.concat([grid, grid.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="rectangular|complete"):
        reconstruct_joint_batch(missing, order)
    with pytest.raises(ValueError, match="duplicate"):
        reconstruct_joint_batch(duplicate, order)

