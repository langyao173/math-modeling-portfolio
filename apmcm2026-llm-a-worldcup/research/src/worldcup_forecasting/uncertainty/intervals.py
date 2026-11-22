"""Intervals derived from discrete predictive distributions."""

from __future__ import annotations

import numpy as np


def total_goal_distribution(score_probabilities: np.ndarray) -> np.ndarray:
    """Collapse a score matrix to a total-goal probability mass function."""
    matrix = np.asarray(score_probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A square score probability matrix is required")
    if matrix.size == 0 or not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("Score probabilities must be finite and nonnegative")
    if float(matrix.sum()) <= 0:
        raise ValueError("Score probabilities must have positive mass")
    support = matrix.shape[0]
    totals = np.zeros(2 * support - 1, dtype=float)
    for home_goals in range(support):
        for away_goals in range(support):
            totals[home_goals + away_goals] += matrix[home_goals, away_goals]
    totals /= totals.sum()
    return totals


def central_discrete_interval(probabilities: np.ndarray, coverage: float = 0.9) -> tuple[int, int]:
    """Return an equal-tail central interval for a discrete PMF."""
    mass = np.asarray(probabilities, dtype=float)
    if mass.ndim != 1 or np.any(mass < 0) or not np.isclose(mass.sum(), 1.0, atol=1e-8):
        raise ValueError("A normalized one-dimensional probability mass is required")
    if not 0 < coverage < 1:
        raise ValueError("Coverage must lie strictly between zero and one")
    cumulative = np.cumsum(mass)
    tail = (1.0 - coverage) / 2.0
    lower = int(np.searchsorted(cumulative, tail, side="left"))
    upper = int(np.searchsorted(cumulative, 1.0 - tail, side="left"))
    return lower, min(upper, mass.size - 1)


def empirical_absolute_interval(
    prediction: float,
    calibration_residuals: np.ndarray,
    coverage: float = 0.9,
    lower_bound: float = 0.0,
) -> tuple[float, float]:
    """Return a bounded empirical residual interval without exchangeability claims."""
    residuals = np.abs(np.asarray(calibration_residuals, dtype=float))
    if residuals.size == 0 or not np.isfinite(residuals).all():
        raise ValueError("Finite calibration residuals are required")
    if not np.isfinite(prediction) or not np.isfinite(lower_bound) or lower_bound < 0:
        raise ValueError("Prediction and nonnegative lower bound must be finite")
    if not 0 < coverage < 1:
        raise ValueError("Coverage must lie strictly between zero and one")
    quantile = float(np.quantile(residuals, coverage, method="higher"))
    lower = max(float(lower_bound), float(prediction) - quantile)
    upper = max(lower, float(prediction) + quantile)
    return lower, upper
