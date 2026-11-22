"""Contracts for edition-level robustness and paired tournament uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup_forecasting.validation.transport import (
    kl_edition_robust_risk,
    paired_tournament_cluster_interval,
)


def test_kl_robust_risk_is_monotone_bounded_and_auditable() -> None:
    losses = np.array([0.8, 1.0, 1.4, 0.9])
    nominal = kl_edition_robust_risk(losses, epsilon=0.0)
    small = kl_edition_robust_risk(losses, epsilon=0.05)
    large = kl_edition_robust_risk(losses, epsilon=0.5)
    assert nominal["robust_risk"] == pytest.approx(losses.mean(), abs=1e-12)
    assert nominal["weights"] == pytest.approx([0.25] * 4, abs=1e-12)
    assert nominal["robust_risk"] <= small["robust_risk"] <= large["robust_risk"]
    assert large["robust_risk"] <= losses.max() + 1e-10
    for result, radius in ((small, 0.05), (large, 0.5)):
        weights = np.asarray(result["weights"], dtype=float)
        assert np.isfinite(weights).all() and np.all(weights >= 0)
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)
        assert result["realized_kl"] <= radius + 1e-7
        assert int(np.argmax(weights)) == int(np.argmax(losses))
        assert result["interpretation"] == "historical_edition_weight_stress_not_future_bound"


def test_paired_cluster_interval_resamples_editions_not_matches() -> None:
    rows = []
    for year, candidate, baseline in ((1998, 0.9, 1.0), (2002, 1.1, 1.0), (2006, 0.8, 1.0)):
        for model, value in (("candidate", candidate), ("baseline", baseline)):
            rows.append({"test_year": year, "model": model, "log_loss": value})
    table = pd.DataFrame(rows)
    first = paired_tournament_cluster_interval(
        table,
        candidate="candidate",
        baseline="baseline",
        metric="log_loss",
        replicates=500,
        seed=31,
    )
    repeated = paired_tournament_cluster_interval(
        table,
        candidate="candidate",
        baseline="baseline",
        metric="log_loss",
        replicates=500,
        seed=31,
    )
    assert first == repeated
    assert first["clusters"] == 3
    assert first["mean_difference"] == pytest.approx((-0.1 + 0.1 - 0.2) / 3)
    assert first["sampling_unit"] == "tournament_edition"
    assert first["paired"] is True
    assert first["ci_low"] <= first["mean_difference"] <= first["ci_high"]


def test_transport_robustness_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="nonempty|finite"):
        kl_edition_robust_risk(np.array([]), epsilon=0.1)
    with pytest.raises(ValueError, match="epsilon"):
        kl_edition_robust_risk(np.array([1.0, 2.0]), epsilon=-0.1)
    table = pd.DataFrame(
        {"test_year": [2006, 2006], "model": ["candidate", "baseline"], "x": [1.0, 1.0]}
    )
    with pytest.raises(ValueError, match="three|paired"):
        paired_tournament_cluster_interval(
            table,
            candidate="candidate",
            baseline="baseline",
            metric="x",
        )
