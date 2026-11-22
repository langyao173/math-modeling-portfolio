"""Small convex ensemble for already leakage-safe outcome probabilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class ConvexOutcomeEnsemble:
    """Learn nonnegative simplex weights using multiclass log loss."""

    l2_shrinkage: float = 1e-3

    def __post_init__(self) -> None:
        if self.l2_shrinkage < 0:
            raise ValueError("l2_shrinkage must be nonnegative")
        self.weights_: np.ndarray | None = None

    def fit(
        self,
        component_probabilities: Sequence[np.ndarray] | np.ndarray,
        labels: np.ndarray,
    ) -> ConvexOutcomeEnsemble:
        """Fit weights on OOF/training predictions only."""
        stack = _probability_stack(component_probabilities)
        target = np.asarray(labels, dtype=int)
        if target.ndim != 1 or target.shape[0] != stack.shape[1]:
            raise ValueError("Ensemble labels must have one value per prediction row")
        if np.any(target < 0) or np.any(target >= stack.shape[2]):
            raise ValueError("Ensemble labels are outside the class range")
        count = stack.shape[0]
        equal = np.full(count, 1.0 / count)

        def objective(weights: np.ndarray) -> float:
            blended = np.tensordot(weights, stack, axes=(0, 0))
            selected = np.clip(blended[np.arange(target.size), target], 1e-15, 1.0)
            shrinkage = self.l2_shrinkage * float(np.sum((weights - equal) ** 2))
            return float(-np.mean(np.log(selected)) + shrinkage)

        result = minimize(
            objective,
            equal,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * count,
            constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Convex ensemble optimization failed: {result.message}")
        weights = np.maximum(np.asarray(result.x, dtype=float), 0.0)
        self.weights_ = weights / weights.sum()
        return self

    def predict_proba(
        self,
        component_probabilities: Sequence[np.ndarray] | np.ndarray,
    ) -> np.ndarray:
        """Blend component probabilities using the fitted simplex weights."""
        if self.weights_ is None:
            raise RuntimeError("ConvexOutcomeEnsemble must be fitted before prediction")
        stack = _probability_stack(component_probabilities)
        if stack.shape[0] != self.weights_.size:
            raise ValueError("Prediction component count differs from fitted ensemble")
        blended = np.tensordot(self.weights_, stack, axes=(0, 0))
        return blended / blended.sum(axis=1, keepdims=True)


def _probability_stack(values: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    stack = np.asarray(values, dtype=float)
    if stack.ndim != 3 or stack.shape[0] < 1 or stack.shape[1] < 1 or stack.shape[2] < 2:
        raise ValueError("Component probabilities must have shape (models, rows, classes)")
    if not np.isfinite(stack).all() or np.any(stack < 0):
        raise ValueError("Component probabilities must be finite and nonnegative")
    if not np.allclose(stack.sum(axis=2), 1.0, atol=1e-8):
        raise ValueError("Every component probability row must sum to one")
    return stack
