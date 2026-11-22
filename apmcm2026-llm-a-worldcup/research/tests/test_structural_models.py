import numpy as np
import pandas as pd

from worldcup_forecasting.models.aggregate import AggregateOLS, AggregateOLSConfig
from worldcup_forecasting.models.calibration import TemperatureScaler
from worldcup_forecasting.models.count import outcome_probabilities, score_matrix
from worldcup_forecasting.models.half_time import (
    HalfTimeIncrementModel,
    enforce_half_time_monotonicity,
)
from worldcup_forecasting.models.match import MatchGoalOLS


def test_official_aggregate_formula_is_exactly_three_predictors() -> None:
    assert AggregateOLSConfig().feature_columns == (
        "QualifiedTeams",
        "MatchesPlayed",
        "Attendance",
    )


def test_joint_score_distribution_has_valid_mass() -> None:
    matrix = score_matrix(lambda_home=1.7, lambda_away=1.1, rho=-0.05, max_score=12)
    assert matrix.shape == (13, 13)
    assert np.all(matrix >= 0)
    assert np.isclose(matrix.sum(), 1.0, atol=1e-12)

    probabilities = outcome_probabilities(matrix)
    assert set(probabilities) == {"home_win", "draw", "away_win"}
    assert np.isclose(sum(probabilities.values()), 1.0, atol=1e-12)


def test_swapping_goal_rates_swaps_outcomes() -> None:
    original = outcome_probabilities(score_matrix(1.8, 0.9, rho=-0.04, max_score=12))
    swapped = outcome_probabilities(score_matrix(0.9, 1.8, rho=-0.04, max_score=12))
    assert np.isclose(original["home_win"], swapped["away_win"], atol=1e-12)
    assert np.isclose(original["away_win"], swapped["home_win"], atol=1e-12)
    assert np.isclose(original["draw"], swapped["draw"], atol=1e-12)


def test_temperature_calibration_preserves_simplex() -> None:
    raw = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]])
    labels = np.array([0, 1, 2])
    scaler = TemperatureScaler().fit(raw, labels)
    calibrated = scaler.transform(raw)
    assert np.all(calibrated > 0)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-12)


def test_half_time_prediction_cannot_go_backwards() -> None:
    predicted = np.array([0.5, 2.2, 4.1])
    half_time = np.array([1.0, 3.0, 2.0])
    constrained = enforce_half_time_monotonicity(predicted, half_time)
    np.testing.assert_allclose(constrained, np.array([1.0, 3.0, 4.1]))


def test_attendance_forecast_models_per_match_exposure() -> None:
    years = np.array([1998, 2002, 2006, 2010, 2014])
    matches = np.array([64, 64, 64, 64, 64])
    per_match = np.exp(8.0 + 0.01 * (years - 1998))
    frame = pd.DataFrame(
        {
            "Year": years,
            "MatchesPlayed": matches,
            "Attendance": per_match * matches,
        }
    )

    forecast = AggregateOLS().forecast_attendance(
        frame,
        target_year=2018,
        target_matches_played=64,
    )

    expected = np.exp(8.0 + 0.01 * (2018 - 1998)) * 64
    assert np.isclose(forecast, expected, rtol=1e-10)


def test_match_ols_uses_training_only_attendance_imputation() -> None:
    frame = pd.DataFrame(
        {
            "Year": [2002, 2002, 2006, 2006, 2010, 2010, 2014],
            "canonical_stage": ["group_stage"] * 7,
            "half_time_total_goals": [0, 1, 1, 2, 2, 0, 1],
            "Attendance": [30_000, 40_000, 50_000, 60_000, 70_000, 80_000, np.nan],
            "target_total_goals": [1, 2, 2, 3, 3, 1, 2],
        }
    )

    model = MatchGoalOLS()
    report = model.fit_temporal(frame, train_end_year=2010, test_year=2014)

    assert report["test"]["n"] == 1
    assert model.attendance_median_ == 55_000
    assert np.isfinite(model.predict(frame.tail(1))).all()


def test_match_ols_exact_design_and_prediction_contract() -> None:
    frame = pd.DataFrame(
        {
            "canonical_stage": [
                "group_stage",
                "round_of_16",
                "quarter_final",
                "semi_final",
                "third_place",
                "final",
            ]
            * 2,
            "half_time_total_goals": [0, 1, 2, 1, 3, 0] * 2,
            "Attendance": [35_000, 40_000, 45_000, 50_000, 55_000, 60_000] * 2,
            "target_total_goals": [1, 2, 3, 2, 4, 1] * 2,
        }
    )
    model = MatchGoalOLS().fit(frame)

    assert "half_time_total_goals" in model.design_columns_
    assert "half_time_home_goals" not in model.design_columns_
    assert "half_time_away_goals" not in model.design_columns_
    prediction = model.predict_match(
        stage="group_stage",
        half_time_total_goals=1,
        attendance=45_000,
    )
    assert np.isfinite(prediction)


def test_match_ols_rejects_unknown_stage_and_negative_inputs() -> None:
    frame = pd.DataFrame(
        {
            "canonical_stage": ["group_stage"] * 6,
            "half_time_total_goals": [0, 1, 0, 2, 1, 0],
            "Attendance": [30_000, 40_000, 50_000, 60_000, 70_000, 80_000],
            "target_total_goals": [1, 2, 2, 3, 3, 1],
        }
    )
    model = MatchGoalOLS().fit(frame)

    with np.testing.assert_raises_regex(ValueError, "unseen canonical stage"):
        model.predict_match(
            stage="Group",
            half_time_total_goals=1,
            attendance=45_000,
        )
    with np.testing.assert_raises_regex(ValueError, "nonnegative"):
        model.predict_match(
            stage="group_stage",
            half_time_total_goals=-1,
            attendance=45_000,
        )
    with np.testing.assert_raises_regex(ValueError, "nonnegative"):
        model.predict_match(
            stage="group_stage",
            half_time_total_goals=1,
            attendance=-1,
        )


def test_half_time_increment_uses_canonical_stage_and_total_score() -> None:
    frame = pd.DataFrame(
        {
            "canonical_stage": ["group_stage"] * 8,
            "half_time_total_goals": [0, 1, 0, 2, 1, 0, 2, 1],
            "Attendance": [30_000, 40_000, 50_000, 60_000] * 2,
            "target_total_goals": [1, 2, 1, 3, 2, 1, 4, 2],
        }
    )
    model = HalfTimeIncrementModel().fit(frame)
    result = model.predict_match(
        stage="group_stage",
        half_time_home_goals=1,
        half_time_away_goals=1,
        attendance=45_000,
    )

    assert result["half_time_total_goals"] == 2
    assert result["full_time_total_goals"] >= 2
    components = model.out_of_distribution_components(
        pd.DataFrame(
            {
                "canonical_stage": ["group_stage"],
                "half_time_total_goals": [2],
                "Attendance": [45_000],
            }
        )
    )
    assert set(components) == {"standardized_distance", "range_excess", "unseen_stage"}
    assert components["range_excess"] == 0


def test_half_time_increment_rejects_raw_stage_label() -> None:
    frame = pd.DataFrame(
        {
            "canonical_stage": ["group_stage"] * 6,
            "half_time_total_goals": [0, 1, 0, 2, 1, 0],
            "Attendance": [30_000, 40_000, 50_000, 60_000, 70_000, 80_000],
            "target_total_goals": [1, 2, 1, 3, 2, 1],
        }
    )
    model = HalfTimeIncrementModel().fit(frame)

    with np.testing.assert_raises_regex(ValueError, "unseen canonical stage"):
        model.predict_match(
            stage="Group",
            half_time_home_goals=1,
            half_time_away_goals=0,
            attendance=45_000,
        )
