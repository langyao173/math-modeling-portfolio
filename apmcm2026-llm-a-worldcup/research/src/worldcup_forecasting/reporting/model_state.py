"""Deterministic fitted-state artifacts for traceable prediction certificates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelStateArtifact:
    """A canonical fitted-model state file and its content hash."""

    path: Path
    sha256: str
    feature_schema_hash: str

    def path_sha256(self) -> str:
        """Re-hash the persisted bytes for an independent integrity check."""
        return sha256(self.path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    """Hash a normalized JSON value with a single canonical representation."""
    return sha256(_canonical_bytes(_normalize(value))).hexdigest()


def write_model_state(
    path: Path,
    *,
    model_type: str,
    feature_schema: list[str] | tuple[str, ...],
    fitted_state: dict[str, Any],
    training_contract: dict[str, Any],
    random_seed: int,
) -> ModelStateArtifact:
    """Write enough fitted state to reproduce/fingerprint one model instance."""
    features = [str(name) for name in feature_schema]
    if not model_type.strip():
        raise ValueError("model_type must be nonempty")
    if not features or len(features) != len(set(features)):
        raise ValueError("feature_schema must contain unique nonempty names")
    if any(not name.strip() for name in features):
        raise ValueError("feature_schema must contain unique nonempty names")
    if not fitted_state:
        raise ValueError("fitted_state must be nonempty")
    if not training_contract:
        raise ValueError("training_contract must be nonempty")
    payload = _normalize(
        {
            "schema_version": 1,
            "model_type": model_type,
            "feature_schema": features,
            "feature_schema_hash": stable_hash(features),
            "training_contract": training_contract,
            "random_seed": int(random_seed),
            "fitted_state": fitted_state,
        }
    )
    encoded = _canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return ModelStateArtifact(
        path=path,
        sha256=sha256(encoded).hexdigest(),
        feature_schema_hash=str(payload["feature_schema_hash"]),
    )


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_normalize(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _normalize(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Model state cannot contain non-finite floating-point values")
        return float(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Model state cannot serialize {type(value).__name__}")
