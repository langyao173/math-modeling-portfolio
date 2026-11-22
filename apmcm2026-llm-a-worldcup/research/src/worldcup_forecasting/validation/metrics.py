"""Metrics for deterministic and probabilistic World Cup forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    r2_score,
    recall_score,
)

CLASS_ORDER = ("home_win", "draw", "away_win")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the official and research regression metrics."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    _validate_pair(truth, prediction)
    positive_prediction = np.maximum(prediction, 1e-9)
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "mse": float(mean_squared_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
        "poisson_deviance": float(mean_poisson_deviance(truth, positive_prediction)),
    }


def multiclass_brier(y_true_index: np.ndarray, probabilities: np.ndarray) -> float:
    """Return the mean multiclass Brier score (sum across classes)."""
    labels = np.asarray(y_true_index, dtype=int)
    probs = _validate_probabilities(probabilities)
    if labels.shape[0] != probs.shape[0]:
        raise ValueError("Labels and probabilities must have equal row counts")
    observed = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - observed) ** 2, axis=1)))


def ranked_probability_score(y_true_index: np.ndarray, probabilities: np.ndarray) -> float:
    """Return the three-class ranked probability score."""
    labels = np.asarray(y_true_index, dtype=int)
    probs = _validate_probabilities(probabilities)
    if labels.ndim != 1 or labels.shape[0] != probs.shape[0]:
        raise ValueError("Labels and probabilities must have equal row counts")
    if np.any(labels < 0) or np.any(labels >= probs.shape[1]):
        raise ValueError("Labels are outside the probability class range")
    observed = np.eye(probs.shape[1])[labels]
    cumulative_error = np.cumsum(probs, axis=1)[:, :-1] - np.cumsum(observed, axis=1)[:, :-1]
    return float(np.mean(np.sum(cumulative_error**2, axis=1) / (probs.shape[1] - 1)))


def expected_calibration_error(
    y_true_index: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Return top-label expected calibration error."""
    labels = np.asarray(y_true_index, dtype=int)
    probs = _validate_probabilities(probabilities)
    if labels.ndim != 1 or labels.shape[0] != probs.shape[0]:
        raise ValueError("Labels and probabilities must have equal row counts")
    if np.any(labels < 0) or np.any(labels >= probs.shape[1]):
        raise ValueError("Labels are outside the probability class range")
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = predicted == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        included = (confidence > lower) & (confidence <= upper)
        if lower == 0.0:
            included |= confidence == 0.0
        if included.any():
            error += included.mean() * abs(correct[included].mean() - confidence[included].mean())
    return float(error)


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_order: tuple[str, ...] = CLASS_ORDER,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return mandatory and proper-score classification metrics."""
    truth = np.asarray(y_true)
    probs = _validate_probabilities(probabilities)
    if truth.shape[0] != probs.shape[0]:
        raise ValueError("Labels and probabilities must have equal row counts")
    mapping = {label: index for index, label in enumerate(class_order)}
    try:
        truth_index = np.array([mapping[str(label)] for label in truth], dtype=int)
    except KeyError as error:
        raise ValueError(f"Unknown outcome label: {error.args[0]}") from error
    predicted_index = probs.argmax(axis=1)
    matrix = confusion_matrix(truth_index, predicted_index, labels=np.arange(len(class_order)))
    recalls = recall_score(
        truth_index,
        predicted_index,
        labels=np.arange(len(class_order)),
        average=None,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(truth_index, predicted_index)),
        "macro_f1": float(f1_score(truth_index, predicted_index, average="macro")),
        "log_loss": float(log_loss(truth_index, probs, labels=np.arange(len(class_order)))),
        "multiclass_brier": multiclass_brier(truth_index, probs),
        "ranked_probability_score": ranked_probability_score(truth_index, probs),
        "calibration_error": expected_calibration_error(truth_index, probs),
        "n": int(truth.shape[0]),
    }
    for label, recall in zip(class_order, recalls, strict=True):
        metrics[f"recall_{label}"] = float(recall)
    return metrics, matrix


def _validate_pair(first: np.ndarray, second: np.ndarray) -> None:
    if first.ndim != 1 or second.ndim != 1 or first.shape[0] != second.shape[0]:
        raise ValueError("Metric inputs must be equally sized one-dimensional arrays")
    if first.size == 0 or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Metric inputs must be nonempty and finite")


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] < 2:
        raise ValueError("Probabilities must be a nonempty two-dimensional array")
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("Probabilities must be finite and nonnegative")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Every probability row must sum to one")
    return probs
