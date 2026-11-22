"""Honest hierarchical sensitivity decomposition for tournament forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class VarianceDecomposition:
    """Additive law-of-total-variance components and their shares."""

    components: dict[str, float]
    shares: dict[str, float]
    metadata: dict[str, Any]


def hierarchical_variance_decomposition(draws: np.ndarray) -> VarianceDecomposition:
    """Decompose balanced parameter, regime, and tournament-path draws.

    Axis zero indexes parameter refits, axis one indexes regime/transport
    scenarios conditional on each refit, and axis two indexes tournament paths
    conditional on both.  The last component necessarily combines bracket and
    match aleatoric variation; separating those without a stronger structural
    intervention would be an unsupported identification claim.
    """
    values = np.asarray(draws, dtype=float)
    if values.ndim != 3:
        raise ValueError("Uncertainty draws must be three-dimensional")
    if any(size < 2 for size in values.shape):
        raise ValueError("Every uncertainty axis requires at least two draws")
    if not np.isfinite(values).all():
        raise ValueError("Uncertainty draws must be finite")

    total = float(np.var(values))
    parameter = float(np.var(values.mean(axis=(1, 2))))
    regime = float(np.mean(np.var(values.mean(axis=2), axis=1)))
    tournament = float(np.mean(np.var(values, axis=2)))
    reconstructed = parameter + regime + tournament
    reconstruction_error = float(abs(total - reconstructed))
    scale = max(1.0, abs(total))
    if reconstruction_error > 1e-12 * scale:
        raise RuntimeError("Hierarchical variance components failed exact reconstruction")

    named = {
        "parameter": parameter,
        "regime_given_parameter": regime,
        "tournament_path_and_match": tournament,
    }
    shares = (
        {name: value / total for name, value in named.items()}
        if total > 0
        else {name: 0.0 for name in named}
    )
    components = {
        **named,
        "total": total,
        "reconstruction_error": reconstruction_error,
    }
    metadata: dict[str, Any] = {
        "draw_shape": [int(size) for size in values.shape],
        "functional_mean": float(np.mean(values)),
        "parameter_axis": 0,
        "regime_axis": 1,
        "tournament_path_axis": 2,
        "tournament_component_scope": "bracket_path_plus_match_aleatoric_not_identified_apart",
        "interpretation": (
            "law_of_total_variance_sensitivity_not_future_coverage_guarantee"
        ),
    }
    return VarianceDecomposition(components, shares, metadata)


__all__ = ["VarianceDecomposition", "hierarchical_variance_decomposition"]
