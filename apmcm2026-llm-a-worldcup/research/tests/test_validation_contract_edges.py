"""Fast unit tests for validation, intervals, and resampling contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup_forecasting.uncertainty.intervals import (
    central_discrete_interval,
    empirical_absolute_interval,
    total_goal_distribution,
)
from worldcup_forecasting.validation.bootstrap import (
    clustered_mean_interval,
    paired_fold_comparison,
)
from worldcup_forecasting.validation.leakage import (
    assert_point_in_time,
    assert_probability_simplex,
    symmetry_error,
)
from worldcup_forecasting.validation.metrics import (
    classification_metrics,
    expected_calibration_error,
    multiclass_brier,
    ranked_probability_score,
    regression_metrics,
)
from worldcup_forecasting.validation.rolling import expanding_origin_folds, official_2014_split


def test_discrete_intervals_preserve_probability_and_nonnegative_support() -> None:
    score = np.array([[0.2, 0.3], [0.1, 0.4]])
    total = total_goal_distribution(score)
    np.testing.assert_allclose(total, [0.2, 0.4, 0.4])
    assert central_discrete_interval(total, coverage=0.8) == (0, 2)
    assert empirical_absolute_interval(0.5, np.array([-1.0, 0.2, 0.4]), 0.8) == (0.0, 1.5)


@pytest.mark.parametrize("matrix", [np.ones(3), np.ones((2, 3))])
def test_total_goal_distribution_requires_square_matrix(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError, match="square"):
        total_goal_distribution(matrix)


@pytest.mark.parametrize(
    "mass",
    [
        np.ones((2, 2)),
        np.array([0.5, -0.1, 0.6]),
        np.array([0.2, 0.2]),
    ],
)
def test_central_interval_requires_normalized_nonnegative_mass(mass: np.ndarray) -> None:
    with pytest.raises(ValueError, match="normalized"):
        central_discrete_interval(mass)


@pytest.mark.parametrize("coverage", [0.0, 1.0, -0.1, 1.1])
def test_central_interval_rejects_noncentral_coverage(coverage: float) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        central_discrete_interval(np.array([0.5, 0.5]), coverage)


@pytest.mark.parametrize("residuals", [np.array([]), np.array([np.nan]), np.array([np.inf])])
def test_empirical_interval_requires_finite_calibration_residuals(
    residuals: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="Finite calibration residuals"):
        empirical_absolute_interval(1.0, residuals)


def test_regression_and_classification_metrics_have_expected_values() -> None:
    regression = regression_metrics(np.array([0, 1, 2]), np.array([-1.0, 1.0, 2.0]))
    assert regression["mae"] == pytest.approx(1 / 3)
    assert np.isfinite(regression["poisson_deviance"])

    truth = np.array(["home_win", "draw", "away_win"])
    probabilities = np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    metrics, matrix = classification_metrics(truth, probabilities)
    assert metrics["accuracy"] == 1.0
    assert metrics["n"] == 3
    np.testing.assert_array_equal(matrix, np.eye(3, dtype=int))
    assert multiclass_brier(np.array([0, 1, 2]), probabilities) == pytest.approx(
        metrics["multiclass_brier"]
    )
    assert ranked_probability_score(np.array([0, 1, 2]), probabilities) >= 0
    assert expected_calibration_error(np.array([0, 1, 2]), probabilities, n_bins=4) > 0


@pytest.mark.parametrize(
    "truth,prediction",
    [
        (np.array([]), np.array([])),
        (np.array([1.0]), np.array([1.0, 2.0])),
        (np.ones((1, 1)), np.ones((1, 1))),
        (np.array([np.nan]), np.array([1.0])),
        (np.array([1.0]), np.array([np.inf])),
    ],
)
def test_regression_metrics_reject_invalid_pairs(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="Metric inputs"):
        regression_metrics(truth, prediction)


@pytest.mark.parametrize(
    "probabilities,error",
    [
        (np.ones(3), "two-dimensional"),
        (np.empty((0, 3)), "nonempty"),
        (np.ones((2, 1)), "two-dimensional"),
        (np.array([[np.nan, 0.5, 0.5]]), "finite"),
        (np.array([[-0.1, 0.5, 0.6]]), "nonnegative"),
        (np.array([[0.2, 0.2, 0.2]]), "sum to one"),
    ],
)
def test_probability_metrics_fail_closed(
    probabilities: np.ndarray,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        classification_metrics(np.array(["home_win"]), probabilities)


def test_probability_metric_label_contracts_are_enforced() -> None:
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    with pytest.raises(ValueError, match="equal row counts"):
        multiclass_brier(np.array([0]), probabilities)
    with pytest.raises(ValueError, match="equal row counts"):
        classification_metrics(np.array(["home_win"]), probabilities)
    with pytest.raises(ValueError, match="Unknown outcome"):
        classification_metrics(np.array(["home_win", "not_a_result"]), probabilities)


def test_point_in_time_and_probability_assertions_fail_closed() -> None:
    valid = pd.DataFrame({"Year": [2010, 2014], "history_cutoff_year": [2006, 2010]})
    assert_point_in_time(valid)
    with pytest.raises(ValueError, match="Missing point-in-time"):
        assert_point_in_time(pd.DataFrame({"Year": [2014]}))
    with pytest.raises(AssertionError, match="Feature leakage"):
        assert_point_in_time(
            pd.DataFrame({"Year": [2014], "history_cutoff_year": [2014]})
        )

    assert_probability_simplex(np.array([[0.2, 0.3, 0.5]]))
    for invalid in (
        np.ones(3),
        np.array([[np.nan, 0.0, 1.0]]),
        np.array([[-0.1, 0.5, 0.6]]),
    ):
        with pytest.raises(AssertionError, match="Invalid probability"):
            assert_probability_simplex(invalid)
    with pytest.raises(AssertionError, match="do not sum"):
        assert_probability_simplex(np.array([[0.2, 0.3, 0.4]]))


def test_symmetry_error_handles_valid_and_invalid_shapes() -> None:
    original = np.array([[0.6, 0.3, 0.1]])
    swapped = np.array([[0.1, 0.3, 0.6]])
    assert symmetry_error(original, swapped) == 0.0
    with pytest.raises(ValueError, match="equally shaped"):
        symmetry_error(original, np.array([[0.5, 0.5]]))
    with pytest.raises(ValueError, match="three-class"):
        symmetry_error(np.ones((1, 2)), np.ones((1, 2)))


def test_expanding_origin_folds_enforce_strict_chronology() -> None:
    frame = pd.DataFrame({"edition": [2002, 2006, 2010, 2014, 2014]})
    folds = expanding_origin_folds(frame, [2010, 2014], year_column="edition")
    assert [fold.test_year for fold in folds] == [2010, 2014]
    np.testing.assert_array_equal(folds[0].train_index, [0, 1])
    np.testing.assert_array_equal(folds[1].test_index, [3, 4])

    with pytest.raises(ValueError, match="Missing year"):
        expanding_origin_folds(frame, [2014], year_column="Year")
    with pytest.raises(ValueError, match="Invalid expanding-origin"):
        expanding_origin_folds(frame, [2002], year_column="edition")
    with pytest.raises(ValueError, match="Invalid expanding-origin"):
        expanding_origin_folds(frame, [2018], year_column="edition")


def test_official_split_requires_exact_deduplicated_counts() -> None:
    valid = pd.DataFrame({"Year": np.concatenate([np.full(772, 2010), np.full(64, 2014)])})
    train, test = official_2014_split(valid)
    assert train.size == 772
    assert test.size == 64

    with pytest.raises(AssertionError, match="772/64"):
        official_2014_split(valid.iloc[:-1])


def test_clustered_bootstrap_is_reproducible_and_cluster_aware() -> None:
    values = np.array([1.0, 2.0, 10.0, 12.0])
    clusters = np.array([2006, 2006, 2010, 2010])
    first = clustered_mean_interval(values, clusters, replicates=50, seed=7)
    second = clustered_mean_interval(values, clusters, replicates=50, seed=7)

    assert first == second
    assert first["mean"] == pytest.approx(6.25)
    assert first["clusters"] == 2
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]

    with pytest.raises(ValueError, match="nonempty and aligned"):
        clustered_mean_interval(np.array([]), np.array([]), replicates=2)
    with pytest.raises(ValueError, match="nonempty and aligned"):
        clustered_mean_interval(np.array([1.0]), np.array([1, 2]), replicates=2)


def _comparison_table(candidate: list[float], baseline: list[float]) -> pd.DataFrame:
    rows = []
    for year, candidate_value, baseline_value in zip(
        [2006, 2010, 2014], candidate, baseline, strict=True
    ):
        rows.extend(
            [
                {"test_year": year, "model": "candidate", "score": candidate_value},
                {"test_year": year, "model": "baseline", "score": baseline_value},
            ]
        )
    return pd.DataFrame(rows)


def test_paired_fold_comparison_covers_metric_orientation_and_zero_differences() -> None:
    lower = paired_fold_comparison(
        _comparison_table([0.8, 0.9, 0.7], [1.0, 1.0, 1.0]),
        "candidate",
        "baseline",
        "score",
    )
    assert lower["folds_improved"] == 3
    assert lower["rank_biserial_directional"] == 1.0

    higher = paired_fold_comparison(
        _comparison_table([0.8, 0.9, 0.7], [0.5, 0.6, 0.4]),
        "candidate",
        "baseline",
        "score",
        lower_is_better=False,
    )
    assert higher["folds_improved"] == 3

    with pytest.warns(RuntimeWarning, match="invalid value"):
        tied = paired_fold_comparison(
            _comparison_table([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
            "candidate",
            "baseline",
            "score",
        )
    assert tied["p_value_one_sided"] == 1.0
    assert tied["rank_biserial_directional"] == 0.0

    with pytest.raises(ValueError, match="At least three paired"):
        paired_fold_comparison(
            _comparison_table([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]).query(
                "test_year != 2014"
            ),
            "candidate",
            "baseline",
            "score",
        )
