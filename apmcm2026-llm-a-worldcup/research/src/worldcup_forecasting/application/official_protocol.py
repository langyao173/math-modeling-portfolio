"""Small protocol helpers for official temporal calibration and reporting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from worldcup_forecasting.models import HalfTimeIncrementModel
from worldcup_forecasting.uncertainty.intervals import empirical_absolute_interval


@dataclass(frozen=True)
class HalfTimeCalibrationResult:
    """Strictly earlier rolling residuals and cross-calibration diagnostics."""

    residuals: np.ndarray
    predictions: pd.DataFrame
    report: dict[str, float | int | str | None]


def rolling_half_time_calibration(
    matches: pd.DataFrame,
    test_years: list[int],
    *,
    coverage: float = 0.9,
) -> HalfTimeCalibrationResult:
    """Generate half-time residuals with every fit restricted to earlier editions."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie strictly between zero and one")
    complete = matches.dropna(subset=["Attendance"]).copy()
    all_residuals: list[np.ndarray] = []
    rows: list[dict[str, float | int | bool]] = []
    cross_covered: list[np.ndarray] = []
    cross_widths: list[float] = []
    for test_year in sorted(int(year) for year in test_years):
        train = complete.loc[complete["Year"].lt(test_year)]
        test = complete.loc[complete["Year"].eq(test_year)]
        if train.empty or test.empty:
            raise ValueError(f"Invalid half-time rolling fold for {test_year}")
        if int(train["Year"].max()) >= test_year:
            raise AssertionError("Half-time calibration training is not strictly earlier")
        model = HalfTimeIncrementModel().fit(train)
        actual = test["full_time_total_goals"].to_numpy(float)
        prediction = model.predict(test)
        residual = actual - prediction
        half_time = test["half_time_total_goals"].to_numpy(float)
        prior = np.concatenate(all_residuals) if all_residuals else np.asarray([], dtype=float)
        quantile = (
            float(np.quantile(np.abs(prior), coverage, method="higher"))
            if prior.size
            else None
        )
        interval_lower = np.full(residual.size, np.nan, dtype=float)
        interval_upper = np.full(residual.size, np.nan, dtype=float)
        interval_width = np.full(residual.size, np.nan, dtype=float)
        if quantile is not None:
            bounded_intervals = [
                empirical_absolute_interval(
                    float(estimate),
                    prior,
                    coverage=coverage,
                    lower_bound=float(observed_half_time),
                )
                for estimate, observed_half_time in zip(
                    prediction, half_time, strict=True
                )
            ]
            interval_lower = np.asarray(
                [bounds[0] for bounds in bounded_intervals], dtype=float
            )
            interval_upper = np.asarray(
                [bounds[1] for bounds in bounded_intervals], dtype=float
            )
            interval_width = interval_upper - interval_lower
            cross_covered.append(
                (actual >= interval_lower) & (actual <= interval_upper)
            )
            cross_widths.extend(interval_width.tolist())
        for (
            match_id,
            observed_half_time,
            observed,
            estimate,
            error,
            lower,
            upper,
            width,
        ) in zip(
            test["MatchID"],
            half_time,
            actual,
            prediction,
            residual,
            interval_lower,
            interval_upper,
            interval_width,
            strict=True,
        ):
            rows.append(
                {
                    "test_year": test_year,
                    "match_id": int(match_id),
                    "observed_half_time_total": float(observed_half_time),
                    "actual_full_time_total": float(observed),
                    "predicted_full_time_total": float(estimate),
                    "residual": float(error),
                    "prior_calibration_available": quantile is not None,
                    "prior_absolute_residual_quantile": quantile
                    if quantile is not None
                    else float("nan"),
                    "bounded_interval_lower": float(lower),
                    "bounded_interval_upper": float(upper),
                    "bounded_interval_width": float(width),
                }
            )
        all_residuals.append(residual)
    residuals = np.concatenate(all_residuals)
    final_quantile = float(np.quantile(np.abs(residuals), coverage, method="higher"))
    covered = np.concatenate(cross_covered) if cross_covered else np.asarray([], dtype=bool)
    report: dict[str, float | int | str | None] = {
        "method": "strictly_earlier_expanding_origin_absolute_residuals",
        "coverage_target": coverage,
        "selection_year_max": max(test_years),
        "oof_residual_count": int(residuals.size),
        "final_absolute_residual_quantile": final_quantile,
        "cross_calibrated_count": int(covered.size),
        "cross_calibrated_empirical_coverage": float(covered.mean())
        if covered.size
        else None,
        "cross_calibrated_mean_interval_width": float(np.mean(cross_widths))
        if cross_widths
        else None,
        "exchangeability_claim": "none_time_ordered_empirical_diagnostic",
    }
    return HalfTimeCalibrationResult(residuals, pd.DataFrame(rows), report)


def match_scenario_ood(
    training_matches: pd.DataFrame,
    *,
    half_time_total_goals: float,
    attendance: float,
    target_teams: int,
    target_matches: int,
    historical_teams: int,
    historical_matches: int,
) -> tuple[float, dict[str, float]]:
    """Quantify numeric support and known tournament-format shift transparently."""
    numeric = training_matches[["half_time_total_goals", "Attendance"]].dropna().astype(float)
    scenario = np.asarray([half_time_total_goals, attendance], dtype=float)
    if not np.isfinite(scenario).all() or np.any(scenario < 0):
        raise ValueError("Scenario half-time goals and attendance must be finite and nonnegative")
    location = numeric.mean().to_numpy(float)
    scale = numeric.std(ddof=0).replace(0, 1).to_numpy(float)
    lower = numeric.min().to_numpy(float)
    upper = numeric.max().to_numpy(float)
    standardized = float(np.linalg.norm((scenario - location) / scale))
    range_excess = float(
        np.sum(np.maximum(lower - scenario, 0.0) / scale)
        + np.sum(np.maximum(scenario - upper, 0.0) / scale)
    )
    team_expansion = max(float(target_teams / historical_teams - 1.0), 0.0)
    match_expansion = max(float(target_matches / historical_matches - 1.0), 0.0)
    components = {
        "numeric_standardized_distance": standardized,
        "numeric_range_excess": range_excess,
        "unseen_stage": 0.0,
        "relative_team_count_expansion": team_expansion,
        "relative_match_count_expansion": match_expansion,
    }
    score = float(np.sqrt(sum(value * value for value in components.values())))
    return score, components


def high_correlation_summary(
    correlations: pd.DataFrame,
    *,
    threshold: float = 0.1,
) -> tuple[list[dict[str, float | str]], str]:
    """Create a deterministic ranked correlation table and cautious narrative."""
    records: list[dict[str, float | str]] = []
    for feature in correlations.index:
        for outcome in correlations.columns:
            value = float(correlations.loc[feature, outcome])
            if np.isfinite(value) and abs(value) >= threshold:
                records.append(
                    {
                        "feature": str(feature),
                        "outcome": str(outcome).removeprefix("outcome_"),
                        "pearson_correlation": value,
                        "absolute_correlation": abs(value),
                    }
                )
    records.sort(
        key=lambda row: (
            -float(row["absolute_correlation"]),
            str(row["feature"]),
            str(row["outcome"]),
        )
    )
    lines = [
        "# Problem 3 high-correlation feature summary",
        "",
        "These are univariate Pearson associations computed from leakage-safe pre-match history. "
        "They are descriptive and are not causal effects.",
        "",
    ]
    if not records:
        lines.append(
            f"No absolute correlation reached the preregistered threshold {threshold:.2f}."
        )
    else:
        for record in records:
            lines.append(
                f"- {record['feature']} vs {record['outcome']}: "
                f"r={float(record['pearson_correlation']):.3f}."
            )
    return records, "\n".join(lines) + "\n"
