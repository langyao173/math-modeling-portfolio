"""Tests for exact, identical-fold baseline evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup_forecasting.experiments.baseline_suite import (
    BaselineEvaluationConfig,
    run_baseline_suite,
)


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(91)
    rows: list[dict[str, object]] = []
    for edition, year in enumerate(range(1974, 2003, 4)):
        for match in range(18):
            strength = float(((match + edition) % 7) - 3) / 2.0
            home = int(rng.poisson(np.exp(0.25 + 0.18 * strength)))
            away = int(rng.poisson(np.exp(0.25 - 0.18 * strength)))
            # Guarantee all classes repeatedly without changing the feature chronology.
            if match % 6 == 0:
                home, away = 1, 0
            elif match % 6 == 1:
                home, away = 0, 0
            elif match % 6 == 2:
                home, away = 0, 1
            rows.append(
                {
                    "MatchID": year * 100 + match,
                    "Datetime": pd.Timestamp(year=year, month=6, day=(match % 20) + 1),
                    "Year": year,
                    "historical_appearances_diff": strength,
                    "historical_best_finish_diff": 0.5 * strength,
                    "historical_mean_goals_for_diff": 0.2 * strength,
                    "historical_mean_goals_conceded_diff": -0.1 * strength,
                    "pre_match_elo_diff": 0.3 * strength,
                    "attack_state_diff": 0.2 * strength,
                    "defense_state_diff": -0.15 * strength,
                    "target_home_goals": home,
                    "target_away_goals": away,
                    "target_outcome": (
                        "home_win" if home > away else "away_win" if home < away else "draw"
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_baseline_suite_is_complete_chronological_and_deterministic() -> None:
    config = BaselineEvaluationConfig(outer_test_years=(1994, 1998, 2002), seed=17)
    first = run_baseline_suite(_frame(), config=config, runtime_clock=lambda: 0.0)
    second = run_baseline_suite(_frame(), config=config, runtime_clock=lambda: 0.0)

    assert set(first.outcome_fold_metrics["model"]) == {
        "official_outcome_logit",
        "independent_mirrored_poisson",
        "legacy_mirrored_dixon_coles",
    }
    assert len(first.outcome_fold_metrics) == 9
    assert len(first.joint_fold_metrics) == 6
    assert first.prediction_ledger.groupby(["model", "test_year"]).size().eq(18).all()
    assert first.joint_grid.groupby(["model", "outer_fold_id", "match_id"])[
        "probability"
    ].sum().sub(1.0).abs().max() < 1e-10
    assert (first.prediction_ledger["train_end_year"] < first.prediction_ledger["test_year"]).all()
    pd.testing.assert_frame_equal(first.outcome_fold_metrics, second.outcome_fold_metrics)
    pd.testing.assert_frame_equal(first.prediction_ledger, second.prediction_ledger)
    assert first.metadata["maximum_prediction_year"] == 2002


def test_baseline_suite_refuses_audit_year_during_selection() -> None:
    with pytest.raises(ValueError, match="2014|audit|selection"):
        run_baseline_suite(
            _frame(), config=BaselineEvaluationConfig(outer_test_years=(2014,))
        )

