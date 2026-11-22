"""Contract tests for proper joint-score and uncertainty diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from worldcup_forecasting.validation.joint_metrics import (
    classwise_calibration_diagnostics,
    joint_forecast_metrics,
    randomized_discrete_pit,
    scoreline_negative_log_likelihood,
    total_goal_crps,
)


def _joint(rows: list[tuple[int, int]]) -> np.ndarray:
    probability = np.full((len(rows), 4, 4), 1e-6, dtype=float)
    for index, (home, away) in enumerate(rows):
        probability[index, home, away] = 1.0
        probability[index] /= probability[index].sum()
    return probability


def test_joint_metrics_reward_nearly_perfect_score_distributions() -> None:
    observed = np.asarray([[1, 0], [0, 0], [2, 1], [0, 2]], dtype=int)
    probability = _joint([tuple(row) for row in observed.tolist()])

    summary = joint_forecast_metrics(observed, probability, pit_seed=17)

    assert summary["scoreline_log_loss"] < 1e-3
    assert summary["total_goal_crps"] < 1e-6
    assert summary["total_goal_mae"] < 1e-4
    assert summary["interval_90_coverage"] == pytest.approx(1.0)
    assert summary["interval_90_mean_width"] == pytest.approx(0.0)
    assert summary["support_overflow_rows"] == 0


def test_joint_metrics_report_support_overflow_without_silent_clipping() -> None:
    probability = np.full((2, 3, 3), 1.0 / 9.0)
    observed = np.asarray([[8, 0], [1, 1]], dtype=int)

    summary = joint_forecast_metrics(observed, probability, pit_seed=29)

    assert summary["support_overflow_rows"] == 1
    assert np.isfinite(summary["scoreline_log_loss"])
    assert summary["scoreline_log_loss"] > 10.0


def test_randomized_pit_is_deterministic_and_bounded() -> None:
    probability = np.full((3, 3, 3), 1.0 / 9.0)
    observed = np.asarray([[0, 0], [1, 1], [2, 0]], dtype=int)

    first = randomized_discrete_pit(observed, probability, seed=123)
    second = randomized_discrete_pit(observed, probability, seed=123)

    np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_low_level_metrics_validate_and_match_manual_values() -> None:
    probability = np.zeros((1, 2, 2), dtype=float)
    probability[0] = [[0.2, 0.1], [0.3, 0.4]]
    observed = np.asarray([[1, 0]], dtype=int)

    assert scoreline_negative_log_likelihood(observed, probability) == pytest.approx(
        -np.log(0.3)
    )
    # Total-goal PMF is [0.2, 0.4, 0.4] and the observed total is one.
    assert total_goal_crps(observed, probability) == pytest.approx(0.2**2 + 0.4**2)

    with pytest.raises(ValueError, match="normalized"):
        scoreline_negative_log_likelihood(observed, probability * 0.5)


def test_classwise_calibration_reports_all_classes_and_degenerate_support() -> None:
    probabilities = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.20, 0.60, 0.20],
            [0.10, 0.20, 0.70],
            [0.55, 0.30, 0.15],
            [0.15, 0.55, 0.30],
            [0.30, 0.15, 0.55],
        ]
    )
    labels = np.asarray([0, 1, 2, 0, 1, 2])

    report = classwise_calibration_diagnostics(labels, probabilities)

    assert {row["class_index"] for row in report} == {0, 1, 2}
    assert all(row["status"] == "estimated" for row in report)
    assert all(np.isfinite(row["slope"]) for row in report)

    degenerate = classwise_calibration_diagnostics(
        np.zeros(4, dtype=int), np.tile([0.6, 0.2, 0.2], (4, 1))
    )
    assert degenerate[0]["status"] == "insufficient_class_variation"

