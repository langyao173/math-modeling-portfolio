"""Tests for the frozen seven-fold statistical closure contract."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from worldcup_forecasting.experiments.statistical_closure import (
    BASELINE,
    CANDIDATE,
    EXPECTED_YEARS,
    build_statistical_closure,
)
from worldcup_forecasting.validation._promotion_checks import (
    _statistical_closure_checks,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log_improvement = (0.03, 0.02, 0.01, 0.04, 0.02, -0.005, -0.004)
    brier_improvement = (0.02, 0.01, 0.005, 0.02, 0.01, -0.003, -0.002)
    ece_change = (0.01, 0.02, -0.01, 0.03, -0.005, 0.015, 0.01)
    rows: list[dict[str, object]] = []
    for index, year in enumerate(EXPECTED_YEARS):
        n = 50 + index
        baseline_log = 1.05 + index / 100
        baseline_brier = 0.65 + index / 100
        baseline_ece = 0.08
        rows.extend(
            [
                {
                    "test_year": year,
                    "model": BASELINE,
                    "n": n,
                    "log_loss": baseline_log,
                    "multiclass_brier": baseline_brier,
                    "calibration_error": baseline_ece,
                },
                {
                    "test_year": year,
                    "model": CANDIDATE,
                    "n": n,
                    "log_loss": baseline_log - log_improvement[index],
                    "multiclass_brier": baseline_brier - brier_improvement[index],
                    "calibration_error": baseline_ece + ece_change[index],
                },
            ]
        )
    intervals = pd.DataFrame(
        {
            "candidate": [CANDIDATE, CANDIDATE],
            "baseline": [BASELINE, BASELINE],
            "metric": ["log_loss", "multiclass_brier"],
            "replicates": [2_000, 2_000],
            "ci_low": [-0.04, -0.02],
            "ci_high": [-0.001, -0.0005],
        }
    )
    selected = pd.DataFrame(
        {
            "outer_test_year": EXPECTED_YEARS,
            "candidate_id": ["f12"] * 5 + ["f11", "f10"],
        }
    )
    raw = pd.DataFrame(
        {
            "class_index": [0, 1, 2],
            "n": [371, 371, 371],
            "prevalence": [0.45, 0.25, 0.30],
            "intercept": [0.2, -0.1, 0.15],
            "slope": [0.7, 1.4, 0.8],
        }
    )
    calibrated = raw.assign(
        intercept=[0.1, -0.05, 0.08],
        slope=[0.9, 1.2, 0.95],
    )
    return pd.DataFrame(rows), intervals, selected, raw, calibrated


def test_statistical_closure_keeps_all_estimands_and_passes_frozen_rules() -> None:
    result = build_statistical_closure(*_inputs())

    assert result.summary["passed"] is True
    assert result.summary["outer_fold_count"] == 7
    assert result.summary["metric_summary"]["log_loss"]["fold_wins"] == 5
    assert result.summary["metric_summary"]["multiclass_brier"]["fold_wins"] == 5
    assert result.summary["selection_stability"]["modal_selected_fold_count"] == 5
    assert result.summary["calibration_change"]["worsened_folds"] == 5
    assert "mixed secondary result" in result.summary["calibration_change"]["disclosure"]
    assert len(result.fold_comparison) == 7
    assert len(result.classwise_calibration) == 3


def test_statistical_closure_fails_closed_on_wrong_fold_set_or_unbounded_ece() -> None:
    fold_metrics, intervals, selected, raw, calibrated = _inputs()
    with pytest.raises(ValueError, match="seven exact paired editions"):
        build_statistical_closure(
            fold_metrics.loc[fold_metrics["test_year"].ne(1986)],
            intervals,
            selected,
            raw,
            calibrated,
        )

    fold_metrics.loc[fold_metrics["model"].eq(CANDIDATE), "calibration_error"] += 0.10
    result = build_statistical_closure(fold_metrics, intervals, selected, raw, calibrated)
    assert result.summary["passed"] is False
    assert (
        result.summary["internal_promotion_checks"]["calibration_degradation_bounded_and_disclosed"]
        is False
    )


def test_statistical_closure_rejects_fractional_years_and_sample_counts() -> None:
    fold_metrics, intervals, selected, raw, calibrated = _inputs()
    fold_metrics["test_year"] = fold_metrics["test_year"].astype(float)
    fold_metrics.loc[fold_metrics.index[0], "test_year"] = 1986.5
    with pytest.raises(ValueError, match="exact integers"):
        build_statistical_closure(fold_metrics, intervals, selected, raw, calibrated)

    fold_metrics, intervals, selected, raw, calibrated = _inputs()
    fold_metrics["n"] = fold_metrics["n"].astype(float)
    fold_metrics.loc[fold_metrics.index[0], "n"] = 50.5
    with pytest.raises(ValueError, match="exact integers"):
        build_statistical_closure(fold_metrics, intervals, selected, raw, calibrated)


def test_statistical_closure_rejects_missing_selection_and_malformed_intervals() -> None:
    fold_metrics, intervals, selected, raw, calibrated = _inputs()
    selected.loc[0, "candidate_id"] = np.nan
    with pytest.raises(ValueError, match="cannot be missing"):
        build_statistical_closure(fold_metrics, intervals, selected, raw, calibrated)

    fold_metrics, intervals, selected, raw, calibrated = _inputs()
    intervals["replicates"] = intervals["replicates"].astype(float)
    intervals.loc[0, "replicates"] = 2_000.5
    with pytest.raises(ValueError, match="exact integers"):
        build_statistical_closure(fold_metrics, intervals, selected, raw, calibrated)

    fold_metrics, intervals, selected, raw, calibrated = _inputs()
    intervals.loc[0, ["ci_low", "ci_high"]] = [0.1, -0.1]
    with pytest.raises(ValueError, match="cannot exceed"):
        build_statistical_closure(fold_metrics, intervals, selected, raw, calibrated)


def test_statistical_closure_rejects_malformed_classwise_evidence() -> None:
    fold_metrics, intervals, selected, raw, calibrated = _inputs()
    raw.loc[2, "class_index"] = 3
    with pytest.raises(ValueError, match="exact classes"):
        build_statistical_closure(fold_metrics, intervals, selected, raw, calibrated)

    fold_metrics, intervals, selected, raw, calibrated = _inputs()
    raw.loc[0, "n"] = 0
    with pytest.raises(ValueError, match="positive integers"):
        build_statistical_closure(fold_metrics, intervals, selected, raw, calibrated)

    fold_metrics, intervals, selected, raw, calibrated = _inputs()
    calibrated.loc[0, "prevalence"] = 0.9
    with pytest.raises(ValueError, match="summing to one"):
        build_statistical_closure(fold_metrics, intervals, selected, raw, calibrated)


def test_promotion_recomputes_statistical_closure_instead_of_trusting_json(
    tmp_path,
) -> None:
    fold_metrics, intervals, selected, raw, calibrated = _inputs()
    result = build_statistical_closure(fold_metrics, intervals, selected, raw, calibrated)
    evaluation = tmp_path / "evaluation"
    nested = tmp_path / "nested"
    evaluation.mkdir()
    nested.mkdir()
    intervals.to_csv(evaluation / "paired_tournament_intervals.csv", index=False)
    selected.to_csv(nested / "selected_specs.csv", index=False)
    raw.to_csv(evaluation / "raw_classwise_calibration.csv", index=False)
    calibrated.to_csv(evaluation / "calibrated_classwise_calibration.csv", index=False)
    result.fold_comparison.to_csv(evaluation / "statistical_fold_comparison.csv", index=False)
    result.model_ranking.to_csv(evaluation / "statistical_model_ranking.csv", index=False)
    result.classwise_calibration.to_csv(
        evaluation / "statistical_classwise_calibration.csv", index=False
    )
    summary_path = evaluation / "statistical_closure.json"
    summary_path.write_text(json.dumps(result.summary), encoding="utf-8")

    assert _statistical_closure_checks(tmp_path, fold_metrics)["passed"] is True
    tampered = dict(result.summary)
    tampered["passed"] = False
    summary_path.write_text(json.dumps(tampered), encoding="utf-8")
    checked = _statistical_closure_checks(tmp_path, fold_metrics)
    assert checked["recomputed"]["passed"] is True
    assert checked["declared_summary_matches"] is False
    assert checked["passed"] is False
