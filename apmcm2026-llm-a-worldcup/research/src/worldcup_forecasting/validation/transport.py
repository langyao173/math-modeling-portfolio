"""Edition-level distributional robustness and paired cluster uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def _validated_reference(count: int, reference: np.ndarray | None) -> np.ndarray:
    if reference is None:
        return np.full(count, 1.0 / count, dtype=float)
    probability = np.asarray(reference, dtype=float)
    if probability.shape != (count,):
        raise ValueError("Reference edition weights must align with losses")
    if not np.isfinite(probability).all() or np.any(probability <= 0):
        raise ValueError("Reference edition weights must be finite and strictly positive")
    return probability / probability.sum()


def _tilted_weights(losses: np.ndarray, reference: np.ndarray, eta: float) -> np.ndarray:
    logits = np.log(reference) + losses / eta
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def _relative_entropy(weights: np.ndarray, reference: np.ndarray) -> float:
    positive = weights > 0
    return float(np.sum(weights[positive] * np.log(weights[positive] / reference[positive])))


def kl_edition_robust_risk(
    losses: np.ndarray,
    *,
    epsilon: float,
    reference: np.ndarray | None = None,
) -> dict[str, object]:
    """Maximize historical edition-weighted loss inside a KL ambiguity set.

    The result is a sensitivity certificate over observed editions. It is not a coverage or
    future-distribution guarantee.
    """
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Edition losses must be a nonempty finite one-dimensional array")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    nominal = _validated_reference(len(values), reference)
    if epsilon == 0 or np.allclose(values, values[0]):
        return {
            "robust_risk": float(nominal @ values),
            "nominal_risk": float(nominal @ values),
            "epsilon": float(epsilon),
            "realized_kl": 0.0,
            "dual_eta": float("inf"),
            "weights": nominal.tolist(),
            "interpretation": "historical_edition_weight_stress_not_future_bound",
        }

    maximizers = np.isclose(values, values.max(), rtol=0, atol=1e-14)
    limiting_mass = float(nominal[maximizers].sum())
    limiting_kl = float(-np.log(limiting_mass))
    if epsilon >= limiting_kl - 1e-12:
        worst = np.zeros_like(nominal)
        worst[maximizers] = nominal[maximizers] / limiting_mass
        return {
            "robust_risk": float(values.max()),
            "nominal_risk": float(nominal @ values),
            "epsilon": float(epsilon),
            "realized_kl": _relative_entropy(worst, nominal),
            "dual_eta": 0.0,
            "weights": worst.tolist(),
            "interpretation": "historical_edition_weight_stress_not_future_bound",
        }

    def radius_error(log_eta: float) -> float:
        eta = float(np.exp(log_eta))
        weights = _tilted_weights(values, nominal, eta)
        return _relative_entropy(weights, nominal) - epsilon

    lower, upper = -30.0, 30.0
    if radius_error(lower) < 0 or radius_error(upper) > 0:
        raise RuntimeError("Could not bracket the KL robust-risk dual solution")
    log_eta = float(brentq(radius_error, lower, upper, xtol=1e-13, rtol=1e-13))
    eta = float(np.exp(log_eta))
    weights = _tilted_weights(values, nominal, eta)
    realized = _relative_entropy(weights, nominal)
    return {
        "robust_risk": float(weights @ values),
        "nominal_risk": float(nominal @ values),
        "epsilon": float(epsilon),
        "realized_kl": realized,
        "dual_eta": eta,
        "weights": weights.tolist(),
        "interpretation": "historical_edition_weight_stress_not_future_bound",
    }


def paired_tournament_cluster_interval(
    table: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    metric: str,
    year_column: str = "test_year",
    model_column: str = "model",
    replicates: int = 2_000,
    seed: int = 1_729,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Bootstrap paired edition-level differences without resampling matches."""
    required = {year_column, model_column, metric}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Paired cluster table is missing columns: {missing}")
    subset = table.loc[table[model_column].isin([candidate, baseline]), list(required)]
    if subset.duplicated([year_column, model_column]).any():
        raise ValueError("Paired comparison requires one metric row per model and edition")
    pivot = subset.pivot(index=year_column, columns=model_column, values=metric).dropna()
    if candidate not in pivot or baseline not in pivot or len(pivot) < 3:
        raise ValueError("At least three paired tournament editions are required")
    difference = (
        pd.to_numeric(pivot[candidate], errors="raise")
        - pd.to_numeric(pivot[baseline], errors="raise")
    ).to_numpy(float)
    if not np.isfinite(difference).all():
        raise ValueError("Paired edition metric differences must be finite")
    if int(replicates) < 100:
        raise ValueError("At least 100 cluster bootstrap replicates are required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(difference), size=(int(replicates), len(difference)))
    sampled = difference[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "clusters": int(len(difference)),
        "replicates": int(replicates),
        "seed": int(seed),
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "ci_low": float(np.quantile(sampled, alpha)),
        "ci_high": float(np.quantile(sampled, 1.0 - alpha)),
        "sampling_unit": "tournament_edition",
        "paired": True,
    }


__all__ = ["kl_edition_robust_risk", "paired_tournament_cluster_interval"]
