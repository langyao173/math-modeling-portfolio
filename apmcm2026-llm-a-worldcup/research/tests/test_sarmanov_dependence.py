"""Structural and chronology tests for the exploratory Sarmanov benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from worldcup_forecasting.experiments.evidence import reconstruct_joint_batch
from worldcup_forecasting.experiments.sarmanov_dependence import (
    apply_nested_sarmanov_exploration,
    fit_sarmanov_theta,
    sarmanov_joint_from_marginals,
    transform_joint_batch_sarmanov,
)


def _marginal(rate: float, support: int = 8) -> np.ndarray:
    values = poisson.pmf(np.arange(support), rate)
    return values / values.sum()


def _product_batch(rows: int) -> np.ndarray:
    home = _marginal(1.6)
    away = _marginal(1.1)
    return np.repeat(np.outer(home, away)[None, :, :], rows, axis=0)


def test_sarmanov_preserves_marginals_simplex_and_swap_equivariance() -> None:
    home = _marginal(2.1)
    away = _marginal(0.8)
    joint, diagnostics = sarmanov_joint_from_marginals(home, away, theta=-0.73)
    swapped, swapped_diagnostics = sarmanov_joint_from_marginals(
        away, home, theta=-0.73
    )

    assert joint.min() >= 0.0
    assert joint.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(joint.sum(axis=1), home, atol=2e-12)
    np.testing.assert_allclose(joint.sum(axis=0), away, atol=2e-12)
    np.testing.assert_allclose(swapped, joint.T, atol=2e-12)
    assert diagnostics["score_covariance"] < 0.0
    assert swapped_diagnostics["score_covariance"] == pytest.approx(
        diagnostics["score_covariance"], abs=2e-12
    )


def test_sarmanov_positive_parameter_has_positive_covariance() -> None:
    base = _product_batch(3)
    positive, diagnostics = transform_joint_batch_sarmanov(base, theta=0.6)
    zero, zero_diagnostics = transform_joint_batch_sarmanov(base, theta=0.0)

    assert (diagnostics["score_covariance"] > 0.0).all()
    assert np.abs(zero_diagnostics["score_covariance"]).max() < 1e-12
    np.testing.assert_allclose(positive.sum(axis=2), base.sum(axis=2), atol=2e-12)
    np.testing.assert_allclose(positive.sum(axis=1), base.sum(axis=1), atol=2e-12)


def test_inner_oof_fit_can_select_negative_dependence() -> None:
    base = _product_batch(60)
    observed = np.asarray([(0, 2), (2, 0)] * 30, dtype=int)
    state = fit_sarmanov_theta(base, observed)

    assert state["theta"] < 0.0
    assert state["negative_dependence_selected"] is True
    assert state["inner_nll_gain_vs_independence"] > 0.0
    assert state["covariance_max"] < 0.0


def test_nested_sarmanov_is_inner_only_chronological_and_coherent() -> None:
    inner_rows: list[dict[str, object]] = []
    outer_rows: list[dict[str, object]] = []
    inner_grids: list[dict[str, object]] = []
    outer_grids: list[dict[str, object]] = []
    base = _product_batch(1)[0]
    for outer_year in (1998, 2002):
        parent = f"outer_{outer_year}"
        inner_year = outer_year - 4
        inner_fold = f"{parent}__inner_{inner_year}"
        for position in range(8):
            match_id = outer_year * 10_000 + position
            home, away = ((0, 2) if position % 2 == 0 else (2, 0))
            inner_rows.append(
                {
                    "selection_outer_fold_id": parent,
                    "outer_fold_id": inner_fold,
                    "match_id": match_id,
                    "prediction_year": inner_year,
                    "train_end_year": inner_year - 4,
                    "is_oof": True,
                    "home_win": 0.4,
                    "draw": 0.3,
                    "away_win": 0.3,
                    "observed_home_goals": home,
                    "observed_away_goals": away,
                }
            )
            inner_grids.extend(_grid_rows(inner_fold, match_id, base))
        for position in range(3):
            match_id = outer_year * 100 + position
            outer_rows.append(
                {
                    "outer_fold_id": parent,
                    "match_id": match_id,
                    "prediction_year": outer_year,
                    "train_end_year": outer_year - 4,
                    "is_oof": True,
                    "home_win": 0.4,
                    "draw": 0.3,
                    "away_win": 0.3,
                    "observed_home_goals": 1,
                    "observed_away_goals": 0,
                    "target_outcome": "home_win",
                    "model_id": "frozen_count",
                    "candidate_id": "candidate",
                    "seed": 17,
                }
            )
            outer_grids.extend(_grid_rows(parent, match_id, base))

    result = apply_nested_sarmanov_exploration(
        pd.DataFrame(inner_rows),
        pd.DataFrame(inner_grids),
        pd.DataFrame(outer_rows),
        pd.DataFrame(outer_grids),
    )
    order = result.oof_ledger[["outer_fold_id", "match_id"]]
    transformed = reconstruct_joint_batch(result.joint_grid, order)

    assert len(result.selection_table) == 2
    assert (result.selection_table["inner_max_prediction_year"] < result.selection_table[
        "outer_prediction_year"
    ]).all()
    assert (result.selection_table["theta"] < 0.0).all()
    assert result.metadata["confirmatory_status"].startswith("exploratory_only")
    assert result.metadata["audit_year_used"] is False
    assert result.oof_ledger["model_id"].str.endswith("__exploratory_sarmanov").all()
    expected_home = np.repeat(base.sum(axis=1)[None, :], len(transformed), axis=0)
    expected_away = np.repeat(base.sum(axis=0)[None, :], len(transformed), axis=0)
    np.testing.assert_allclose(transformed.sum(axis=2), expected_home, atol=2e-12)
    np.testing.assert_allclose(transformed.sum(axis=1), expected_away, atol=2e-12)


def test_nested_sarmanov_refuses_audit_year() -> None:
    inner = pd.DataFrame(
        [
            {
                "selection_outer_fold_id": "outer_2014",
                "outer_fold_id": "inner_2010",
                "match_id": 1,
                "prediction_year": 2010,
                "train_end_year": 2006,
                "is_oof": True,
                "home_win": 0.4,
                "draw": 0.3,
                "away_win": 0.3,
                "observed_home_goals": 1,
                "observed_away_goals": 1,
            }
        ]
    )
    outer = inner.drop(columns=["selection_outer_fold_id"]).assign(
        outer_fold_id="outer_2014", prediction_year=2014
    )
    grid = pd.DataFrame(_grid_rows("inner_2010", 1, _product_batch(1)[0]))
    outer_grid = pd.DataFrame(_grid_rows("outer_2014", 1, _product_batch(1)[0]))
    with pytest.raises(ValueError, match="2014|audit"):
        apply_nested_sarmanov_exploration(inner, grid, outer, outer_grid)


def _grid_rows(fold: str, match_id: int, matrix: np.ndarray) -> list[dict[str, object]]:
    return [
        {
            "outer_fold_id": fold,
            "match_id": match_id,
            "home_goals": home,
            "away_goals": away,
            "probability": float(matrix[home, away]),
        }
        for home in range(matrix.shape[0])
        for away in range(matrix.shape[1])
    ]
