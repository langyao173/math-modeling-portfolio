"""Contracts for chronological, coherent calibration-method selection."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from worldcup_forecasting.experiments.calibration_selection import (
    CalibrationSelectionConfig,
    ChronologicalCalibrationSelector,
)
from worldcup_forecasting.models.count import score_matrix
from worldcup_forecasting.models.score_calibration import outcome_margins


def _development_oof() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    years = np.repeat(np.array([1994, 1998, 2002, 2006, 2010]), 15)
    labels = np.tile(np.array([0, 1, 2]), len(years) // 3)
    matrices = []
    for index, label in enumerate(labels):
        perturbation = 0.04 * ((index % 5) - 2)
        rates = ((1.35, 0.95), (1.12, 1.08), (0.92, 1.40))[int(label)]
        matrices.append(
            score_matrix(
                rates[0] + perturbation,
                rates[1] - perturbation,
                max_score=7,
                adaptive=False,
            )
        )
    provenance = pd.DataFrame(
        {
            "match_id": np.arange(10_000, 10_000 + len(years)),
            "prediction_year": years,
            "train_end_year": years - 4,
            "outer_fold_id": [f"outer_{year}" for year in years],
            "is_oof": True,
        }
    )
    return np.stack(matrices), labels, provenance


def test_selection_is_chronological_deterministic_and_json_serializable() -> None:
    joint, labels, provenance = _development_oof()
    first = ChronologicalCalibrationSelector().fit_oof(
        joint, labels, provenance=provenance
    )
    second = ChronologicalCalibrationSelector().fit_oof(
        joint, labels, provenance=provenance
    )

    assert first.selection_report_ == second.selection_report_
    assert first.state_dict() == second.state_dict()
    json.dumps(first.selection_report_, sort_keys=True)
    json.dumps(first.state_dict(), sort_keys=True)
    pd.testing.assert_frame_equal(first.crossfit_predictions_, second.crossfit_predictions_)

    assert set(first.selection_report_["methods"]) == {
        "identity",
        "temperature",
        "log_probability_logit",
        "classwise_isotonic",
    }
    for report in first.selection_report_["methods"].values():
        for scored_year, train_max_year in report["train_max_year_by_scored_year"].items():
            if train_max_year is not None:
                assert int(train_max_year) < int(scored_year)
    metric_rows = first.selection_report_["ranking"]
    assert metric_rows
    assert len({tuple(row["scored_years"]) for row in metric_rows}) == 1
    assert [row["selection_objective"] for row in metric_rows] == sorted(
        row["selection_objective"] for row in metric_rows
    )
    for row in metric_rows:
        expected = (
            row["log_loss"]
            + 0.25 * row["multiclass_brier"]
            + 0.05 * row["ece"]
            + 0.05 * row["sd_edition_log_loss"]
        )
        assert row["selection_objective"] == pytest.approx(expected, abs=1e-15)
    assert first.selection_report_["runtime_charge"] == 0.0
    assert first.selection_report_["complexity_charge"] == 0.0


def test_transform_projects_selected_outcomes_back_to_exact_score_margins() -> None:
    joint, labels, provenance = _development_oof()
    selector = ChronologicalCalibrationSelector().fit_oof(
        joint, labels, provenance=provenance
    )
    new_joint = joint[:6]
    calibrated_outcomes = selector.transform_outcomes(outcome_margins(new_joint))
    calibrated_joint = selector.transform(new_joint)

    assert calibrated_joint.shape == new_joint.shape
    assert np.isfinite(calibrated_joint).all() and np.all(calibrated_joint >= 0)
    np.testing.assert_allclose(calibrated_joint.sum(axis=(1, 2)), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        outcome_margins(calibrated_joint), calibrated_outcomes, atol=1e-12
    )
    swapped = selector.transform(new_joint[[0]].transpose(0, 2, 1))
    np.testing.assert_allclose(swapped[0], calibrated_joint[0].T, atol=1e-12)


def test_identity_is_explicit_fallback_when_no_calibrator_has_prior_support() -> None:
    joint, labels, provenance = _development_oof()
    selector = ChronologicalCalibrationSelector(
        CalibrationSelectionConfig(minimum_fit_rows=10_000)
    ).fit_oof(joint, labels, provenance=provenance)

    assert selector.chosen_method_ == "identity"
    assert selector.selection_report_["fallback_reason"] == "no_eligible_calibrated_method"
    np.testing.assert_array_equal(selector.transform(joint[:3]), joint[:3])
    for name in ("temperature", "log_probability_logit", "classwise_isotonic"):
        assert selector.selection_report_["methods"][name]["eligible_years"] == []


def test_same_edition_labels_never_enter_that_editions_calibrator_fit() -> None:
    joint, labels, provenance = _development_oof()
    baseline = ChronologicalCalibrationSelector().fit_oof(
        joint, labels, provenance=provenance
    )
    mutated = labels.copy()
    selected = provenance["prediction_year"].eq(2006).to_numpy()
    mutated[selected] = (mutated[selected] + 1) % 3
    counterfactual = ChronologicalCalibrationSelector().fit_oof(
        joint, mutated, provenance=provenance
    )

    columns = ["method", "row_position", "home_win", "draw", "away_win"]
    original_rows = baseline.crossfit_predictions_.loc[
        baseline.crossfit_predictions_["prediction_year"].eq(2006), columns
    ].reset_index(drop=True)
    counterfactual_rows = counterfactual.crossfit_predictions_.loc[
        counterfactual.crossfit_predictions_["prediction_year"].eq(2006), columns
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(original_rows, counterfactual_rows)


@pytest.mark.parametrize("failure", ["non_oof", "same_year", "post_2010"])
def test_fit_rejects_invalid_oof_or_development_boundaries(failure: str) -> None:
    joint, labels, provenance = _development_oof()
    broken = provenance.copy()
    if failure == "non_oof":
        broken.loc[0, "is_oof"] = False
        match = "OOF|out-of-fold"
    elif failure == "same_year":
        broken.loc[0, "train_end_year"] = broken.loc[0, "prediction_year"]
        match = "earlier|chronolog|train_end"
    else:
        broken.loc[0, "prediction_year"] = 2014
        match = "2010|development|audit"
    with pytest.raises(ValueError, match=match):
        ChronologicalCalibrationSelector().fit_oof(joint, labels, provenance=broken)
