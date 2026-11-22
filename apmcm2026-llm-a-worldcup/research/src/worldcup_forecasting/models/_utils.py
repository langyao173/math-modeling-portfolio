"""Internal validation and configuration helpers for model components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd


def config_values(config: object | None) -> dict[str, Any]:
    """Convert a dataclass or mapping-like configuration to a plain dictionary."""
    if config is None:
        return {}
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    if isinstance(config, Mapping):
        values = dict(config)
    else:
        try:
            values = dict(config)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise TypeError("Model configuration must be a dataclass or mapping") from error
    nested = values.get("model")
    if isinstance(nested, Mapping):
        return {**values, **dict(nested)}
    return values


def dataclass_config(config_type: type, config: object | None) -> Any:
    """Construct a frozen configuration dataclass while ignoring routing keys."""
    if isinstance(config, config_type):
        return config
    values = config_values(config)
    valid_names = config_type.__dataclass_fields__.keys()
    filtered = {name: value for name, value in values.items() if name in valid_names}
    return config_type(**filtered)


def normalized_name(value: str) -> str:
    """Return a punctuation-insensitive column key."""
    return "".join(character for character in value.casefold() if character.isalnum())


def resolve_column(
    frame: pd.DataFrame,
    preferred: str,
    aliases: tuple[str, ...] = (),
) -> str:
    """Resolve an exact or punctuation-insensitive column name without guessing."""
    candidates = (preferred, *aliases)
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    normalized: dict[str, list[str]] = {}
    for column in frame.columns:
        normalized.setdefault(normalized_name(str(column)), []).append(str(column))
    for candidate in candidates:
        matches = normalized.get(normalized_name(candidate), [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous column alias {candidate!r}: {matches}")
    raise ValueError(f"Missing required column {preferred!r}; accepted aliases are {list(aliases)}")


def numeric_column(
    frame: pd.DataFrame,
    preferred: str,
    aliases: tuple[str, ...] = (),
) -> np.ndarray:
    """Extract a numeric column as a one-dimensional float array."""
    column = resolve_column(frame, preferred, aliases)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def require_finite(values: np.ndarray, name: str, *, ndim: int | None = None) -> np.ndarray:
    """Validate a nonempty finite numerical array."""
    array = np.asarray(values, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains missing or non-finite values")
    return array


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the compact deterministic regression metric set."""
    truth = require_finite(y_true, "y_true", ndim=1)
    prediction = require_finite(y_pred, "y_pred", ndim=1)
    if truth.shape != prediction.shape:
        raise ValueError("Regression metric inputs must have equal shapes")
    residual = truth - prediction
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    r_squared = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else 0.0
    return {
        "mae": float(np.mean(np.abs(residual))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r_squared,
        "n": int(truth.size),
    }


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute row-wise softmax with strict finite checks."""
    values = require_finite(logits, "logits", ndim=2)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)
