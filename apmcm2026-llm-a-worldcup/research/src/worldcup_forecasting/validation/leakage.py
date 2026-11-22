"""Fail-closed assertions for temporal features and predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assert_point_in_time(frame: pd.DataFrame) -> None:
    """Require every feature row to use an earlier historical cutoff."""
    required = {"Year", "history_cutoff_year"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing point-in-time columns: {sorted(missing)}")
    invalid = frame.loc[frame["history_cutoff_year"] >= frame["Year"]]
    if not invalid.empty:
        sample = invalid[["Year", "history_cutoff_year"]].head().to_dict("records")
        raise AssertionError(f"Feature leakage detected: {sample}")


def assert_probability_simplex(probabilities: np.ndarray, atol: float = 1e-8) -> None:
    """Require finite nonnegative rows that sum to one."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or not np.isfinite(probs).all() or np.any(probs < -atol):
        raise AssertionError("Invalid probability values")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=atol):
        raise AssertionError("Probability rows do not sum to one")


def symmetry_error(original: np.ndarray, swapped: np.ndarray) -> float:
    """Return maximum outcome error after reversing home/away order."""
    first = np.asarray(original, dtype=float)
    second = np.asarray(swapped, dtype=float)
    if first.shape != second.shape or first.shape[-1] != 3:
        raise ValueError("Symmetry check expects equally shaped three-class probabilities")
    restored = second[..., [2, 1, 0]]
    return float(np.max(np.abs(first - restored)))
