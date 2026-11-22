import numpy as np
import pandas as pd

from worldcup_forecasting.application.official_protocol import (
    high_correlation_summary,
    match_scenario_ood,
    rolling_half_time_calibration,
)
from worldcup_forecasting.data import build_canonical_data


def test_match_scenario_ood_reports_format_shift_and_numeric_support() -> None:
    matches = pd.DataFrame(
        {
            "half_time_total_goals": [0, 1, 2, 1],
            "Attendance": [30_000, 40_000, 50_000, 60_000],
        }
    )

    score, components = match_scenario_ood(
        matches,
        half_time_total_goals=1,
        attendance=45_000,
        target_teams=48,
        target_matches=104,
        historical_teams=32,
        historical_matches=64,
    )

    assert score > 0
    assert components["numeric_range_excess"] == 0
    assert components["relative_team_count_expansion"] == 0.5
    assert components["relative_match_count_expansion"] == 0.625


def test_high_correlation_summary_is_ranked_and_cautious() -> None:
    correlations = pd.DataFrame(
        {"outcome_home_win": [0.2, -0.4], "outcome_draw": [0.05, 0.15]},
        index=["feature_a", "feature_b"],
    )

    records, markdown = high_correlation_summary(correlations, threshold=0.1)

    assert [round(float(row["absolute_correlation"]), 2) for row in records] == [0.4, 0.2, 0.15]
    assert "not causal effects" in markdown
    assert "feature_b vs home_win: r=-0.400" in markdown


def test_empirical_interval_lower_bound_is_observed_half_time() -> None:
    from worldcup_forecasting.uncertainty.intervals import empirical_absolute_interval

    lower, upper = empirical_absolute_interval(
        prediction=1.5,
        calibration_residuals=np.array([0.1, 0.2, 0.3]),
        coverage=0.9,
        lower_bound=2.0,
    )
    assert lower == 2.0
    assert upper >= lower


def test_rolling_half_time_width_uses_deployed_lower_bound(raw_dir) -> None:
    matches = build_canonical_data(raw_dir).matches
    calibration = rolling_half_time_calibration(
        matches,
        [1986, 1990, 1994, 1998, 2002, 2006, 2010],
        coverage=0.9,
    )
    bounded = calibration.predictions.loc[
        calibration.predictions["prior_calibration_available"]
    ]

    assert len(bounded) == 360
    assert (
        bounded["bounded_interval_lower"]
        >= bounded["observed_half_time_total"]
    ).all()
    assert np.isclose(
        calibration.report["cross_calibrated_mean_interval_width"],
        bounded["bounded_interval_width"].mean(),
    )
    assert np.isclose(
        calibration.report["cross_calibrated_mean_interval_width"],
        3.7643389132,
        atol=1e-10,
    )
