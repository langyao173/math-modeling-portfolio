"""Proper scores and uncertainty diagnostics for joint goal distributions.

The functions in this module never clip an observed score into the represented
support. An out-of-support observation is instead recorded explicitly and
receives the declared probability floor for logarithmic scoring.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import kstest
from sklearn.linear_model import LogisticRegression


def scoreline_negative_log_likelihood(
    observed: np.ndarray,
    joint: np.ndarray,
    *,
    probability_floor: float = 1e-15,
) -> float:
    """Return mean negative log probability of the observed exact scoreline."""
    goals, probability = _validate_joint_inputs(observed, joint)
    if not 0 < probability_floor < 1:
        raise ValueError("probability_floor must lie in (0, 1)")
    support = probability.shape[1]
    inside = (
        (goals[:, 0] >= 0) & (goals[:, 1] >= 0) & (goals[:, 0] < support) & (goals[:, 1] < support)
    )
    assigned = np.full(len(goals), float(probability_floor), dtype=float)
    row = np.flatnonzero(inside)
    assigned[row] = probability[row, goals[row, 0], goals[row, 1]]
    return float(-np.log(np.maximum(assigned, probability_floor)).mean())


def total_goal_crps(observed: np.ndarray, joint: np.ndarray) -> float:
    """Return the mean discrete CRPS for total goals."""
    goals, probability = _validate_joint_inputs(observed, joint)
    total_pmf = _total_goal_pmf(probability)
    cumulative = np.cumsum(total_pmf, axis=1)
    support = np.arange(total_pmf.shape[1], dtype=int)
    observed_total = goals.sum(axis=1)
    target_cdf = (support[None, :] >= observed_total[:, None]).astype(float)
    return float(np.mean(np.sum((cumulative - target_cdf) ** 2, axis=1)))


def randomized_discrete_pit(
    observed: np.ndarray,
    joint: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Return randomized probability-integral-transform values for total goals."""
    goals, probability = _validate_joint_inputs(observed, joint)
    total_pmf = _total_goal_pmf(probability)
    cumulative = np.cumsum(total_pmf, axis=1)
    observed_total = goals.sum(axis=1)
    lower = np.zeros(len(goals), dtype=float)
    mass = np.zeros(len(goals), dtype=float)
    support = total_pmf.shape[1]
    previous = (observed_total > 0) & (observed_total < support)
    row = np.flatnonzero(previous)
    lower[row] = cumulative[row, observed_total[row] - 1]
    above = observed_total >= support
    lower[above] = 1.0
    row = np.flatnonzero((observed_total >= 0) & (observed_total < support))
    mass[row] = total_pmf[row, observed_total[row]]
    rng = np.random.default_rng(int(seed))
    return np.clip(lower + rng.random(len(goals)) * mass, 0.0, 1.0)


def joint_forecast_metrics(
    observed: np.ndarray,
    joint: np.ndarray,
    *,
    pit_seed: int,
    coverages: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> dict[str, float | int]:
    """Summarize exact-score, total-goal, PIT, and interval performance."""
    goals, probability = _validate_joint_inputs(observed, joint)
    if not coverages or any(not 0 < value < 1 for value in coverages):
        raise ValueError("coverages must contain probabilities strictly between zero and one")
    total_pmf = _total_goal_pmf(probability)
    total_support = np.arange(total_pmf.shape[1], dtype=float)
    expected_total = total_pmf @ total_support
    observed_total = goals.sum(axis=1).astype(float)
    residual = expected_total - observed_total
    pit = randomized_discrete_pit(goals, probability, seed=pit_seed)
    ks = kstest(pit, "uniform")
    support = probability.shape[1]
    overflow = (goals[:, 0] >= support) | (goals[:, 1] >= support)
    result: dict[str, float | int] = {
        "scoreline_log_loss": scoreline_negative_log_likelihood(goals, probability),
        "total_goal_crps": total_goal_crps(goals, probability),
        "total_goal_mae": float(np.mean(np.abs(residual))),
        "total_goal_rmse": float(np.sqrt(np.mean(residual**2))),
        "pit_mean": float(np.mean(pit)),
        "pit_variance": float(np.var(pit)),
        "pit_ks_statistic": float(ks.statistic),
        "pit_ks_p_value_diagnostic": float(ks.pvalue),
        "support_overflow_rows": int(overflow.sum()),
        "n": int(len(goals)),
    }
    for coverage in coverages:
        lower, upper = _central_intervals(total_pmf, float(coverage))
        covered = (observed_total >= lower) & (observed_total <= upper)
        label = int(round(coverage * 100))
        result[f"interval_{label}_coverage"] = float(covered.mean())
        result[f"interval_{label}_mean_width"] = float(np.mean(upper - lower))
    return result


def classwise_calibration_diagnostics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    """Estimate descriptive one-vs-rest calibration intercepts and slopes."""
    target = np.asarray(labels, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    if probability.ndim != 2 or target.ndim != 1 or len(target) != len(probability):
        raise ValueError("labels and probabilities must be aligned one/two-dimensional arrays")
    if len(target) == 0 or not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("probabilities must be nonempty, finite, and nonnegative")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Every probability row must sum to one")
    if np.any(target < 0) or np.any(target >= probability.shape[1]):
        raise ValueError("labels are outside the probability class range")
    rows: list[dict[str, Any]] = []
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    for class_index in range(probability.shape[1]):
        binary = (target == class_index).astype(int)
        prevalence = float(binary.mean())
        logit = np.log(clipped[:, class_index] / (1.0 - clipped[:, class_index]))
        row: dict[str, Any] = {
            "class_index": int(class_index),
            "n": int(len(target)),
            "prevalence": prevalence,
        }
        if np.unique(binary).size < 2 or np.ptp(logit) <= 1e-12:
            row.update(
                {
                    "status": "insufficient_class_variation",
                    "intercept": None,
                    "slope": None,
                }
            )
        else:
            estimator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2_000)
            estimator.fit(logit[:, None], binary)
            row.update(
                {
                    "status": "estimated",
                    "intercept": float(estimator.intercept_[0]),
                    "slope": float(estimator.coef_[0, 0]),
                }
            )
        rows.append(row)
    return rows


def _validate_joint_inputs(
    observed: np.ndarray,
    joint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    goals = np.asarray(observed)
    probability = np.asarray(joint, dtype=float)
    if goals.ndim != 2 or goals.shape[1] != 2:
        raise ValueError("observed must have shape (n, 2)")
    if probability.ndim != 3 or probability.shape[1] != probability.shape[2]:
        raise ValueError("joint must have shape (n, support, support)")
    if len(goals) == 0 or len(goals) != len(probability):
        raise ValueError("observed and joint must be nonempty and aligned")
    numeric = goals.astype(float)
    if not np.isfinite(numeric).all() or np.any(numeric < 0) or not np.allclose(numeric, goals):
        raise ValueError("observed goals must be finite nonnegative integers")
    goals = numeric.astype(int)
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("joint probabilities must be finite and nonnegative")
    if not np.allclose(probability.sum(axis=(1, 2)), 1.0, atol=1e-8):
        raise ValueError("Every joint probability matrix must be normalized")
    return goals, probability


def _total_goal_pmf(probability: np.ndarray) -> np.ndarray:
    support = probability.shape[1]
    total = np.zeros((len(probability), 2 * support - 1), dtype=float)
    for home in range(support):
        for away in range(support):
            total[:, home + away] += probability[:, home, away]
    return total


def _central_intervals(
    total_pmf: np.ndarray,
    coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    cumulative = np.cumsum(total_pmf, axis=1)
    tail = (1.0 - coverage) / 2.0
    lower = np.argmax(cumulative >= tail, axis=1).astype(float)
    upper = np.argmax(cumulative >= 1.0 - tail, axis=1).astype(float)
    return lower, upper


__all__ = [
    "classwise_calibration_diagnostics",
    "joint_forecast_metrics",
    "randomized_discrete_pit",
    "scoreline_negative_log_likelihood",
    "total_goal_crps",
]
