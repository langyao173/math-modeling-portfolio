"""Training-only temperature calibration for multiclass probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from ._utils import stable_softmax


@dataclass
class TemperatureScaler:
    """Fit one positive temperature by minimizing multiclass log loss."""

    temperature: float = 1.0
    min_temperature: float = 0.05
    max_temperature: float = 20.0

    def __post_init__(self) -> None:
        if not 0 < self.min_temperature < self.max_temperature:
            raise ValueError("Temperature bounds must satisfy 0 < min < max")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Initial temperature must be finite and positive")
        self.temperature_: float = float(self.temperature)
        self.is_fitted_: bool = False

    def fit(self, values: np.ndarray, labels: np.ndarray) -> TemperatureScaler:
        """Fit temperature from OOF/training predictions and integer labels."""
        logits = _to_logits(values)
        target = np.asarray(labels, dtype=int)
        if target.ndim != 1 or target.shape[0] != logits.shape[0]:
            raise ValueError("Calibration labels must have one value per probability row")
        if np.any(target < 0) or np.any(target >= logits.shape[1]):
            raise ValueError("Calibration labels are outside the class range")

        def objective(log_temperature: float) -> float:
            probability = stable_softmax(logits / np.exp(log_temperature))
            selected = np.clip(probability[np.arange(target.size), target], 1e-15, 1.0)
            return float(-np.mean(np.log(selected)))

        result = minimize_scalar(
            objective,
            bounds=(np.log(self.min_temperature), np.log(self.max_temperature)),
            method="bounded",
            options={"xatol": 1e-10},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("Temperature optimization failed")
        self.temperature_ = float(np.exp(result.x))
        self.temperature = self.temperature_
        self.is_fitted_ = True
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply the fitted (or explicitly initialized) temperature."""
        logits = _to_logits(values)
        return stable_softmax(logits / self.temperature_)

    def fit_transform(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit and apply temperature calibration."""
        return self.fit(values, labels).transform(values)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        """Alias for estimator-style use."""
        return self.transform(values)


def _to_logits(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 2:
        raise ValueError("Calibration input must be a nonempty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("Calibration input must be finite")
    looks_like_probability = np.all(array >= 0) and np.allclose(array.sum(axis=1), 1.0, atol=1e-8)
    if looks_like_probability:
        return np.log(np.clip(array, 1e-15, 1.0))
    return array
