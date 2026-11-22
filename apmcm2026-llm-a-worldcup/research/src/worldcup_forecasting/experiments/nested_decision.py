"""Nested draw-aware point decisions kept separate from forecast probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from worldcup_forecasting.models.decision import OOFCostSensitiveDecisionRule

_OUTCOMES = np.asarray(("home_win", "draw", "away_win"), dtype=object)


@dataclass(frozen=True)
class NestedDecisionResult:
    """Outer decisions, fold metrics, fitted states, and bounded metadata."""

    prediction_ledger: pd.DataFrame
    fold_metrics: pd.DataFrame
    states: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


def apply_nested_decision(
    decision_training_ledger: pd.DataFrame,
    calibrated_outer_oof_ledger: pd.DataFrame,
) -> NestedDecisionResult:
    """Fit one action rule per outer fold using earlier cross-fitted probabilities."""
    training = decision_training_ledger.copy()
    outer = calibrated_outer_oof_ledger.copy()
    _validate_inputs(training, outer)
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    states: dict[str, dict[str, Any]] = {}
    for fold_id, test in outer.groupby("outer_fold_id", sort=True):
        fit = training.loc[training["selection_outer_fold_id"].eq(fold_id)].copy()
        if fit.empty:
            raise ValueError(f"No decision-training OOF rows exist for outer fold {fold_id!r}")
        outer_year = _one_year(test["prediction_year"], "outer")
        if int(fit["prediction_year"].max()) >= outer_year:
            raise ValueError("Decision training predictions must be earlier than the outer fold")
        fit_probability = fit[["home_win", "draw", "away_win"]].to_numpy(float)
        test_probability = test[["home_win", "draw", "away_win"]].to_numpy(float)
        rule = OOFCostSensitiveDecisionRule().fit_oof(
            fit_probability,
            fit["target_label"].to_numpy(int),
            provenance=fit[["is_oof", "prediction_year", "train_end_year"]],
        )
        unchanged = rule.transform_probabilities(test_probability)
        if not np.array_equal(unchanged, test_probability):
            raise RuntimeError("Point-decision layer modified calibrated probabilities")
        decision_label = rule.predict(test_probability)
        baseline_index = test_probability.argmax(axis=1)
        decision_index = np.asarray(
            [int(np.flatnonzero(_OUTCOMES == label)[0]) for label in decision_label],
            dtype=int,
        )
        target_index = np.asarray(
            [int(np.flatnonzero(_OUTCOMES == label)[0]) for label in test["target_outcome"]],
            dtype=int,
        )
        prediction = test.copy()
        prediction["argmax_label"] = _OUTCOMES[baseline_index]
        prediction["decision_label"] = decision_label
        prediction["draw_multiplier"] = float(rule.draw_multiplier_)
        prediction["probability_layer_changed"] = False
        prediction_parts.append(prediction)
        baseline_macro_f1 = float(f1_score(target_index, baseline_index, average="macro"))
        decision_macro_f1 = float(f1_score(target_index, decision_index, average="macro"))
        baseline_draw_recall = float(
            recall_score(target_index, baseline_index, labels=[1], average="macro", zero_division=0)
        )
        decision_draw_recall = float(
            recall_score(target_index, decision_index, labels=[1], average="macro", zero_division=0)
        )
        metric_rows.append(
            {
                "outer_fold_id": str(fold_id),
                "test_year": outer_year,
                "draw_multiplier": float(rule.draw_multiplier_),
                "argmax_accuracy": float(accuracy_score(target_index, baseline_index)),
                "decision_accuracy": float(accuracy_score(target_index, decision_index)),
                "argmax_macro_f1": baseline_macro_f1,
                "decision_macro_f1": decision_macro_f1,
                "macro_f1_gain": decision_macro_f1 - baseline_macro_f1,
                "argmax_draw_recall": baseline_draw_recall,
                "decision_draw_recall": decision_draw_recall,
                "draw_recall_gain": decision_draw_recall - baseline_draw_recall,
                "argmax_draw_predictions": int(np.sum(baseline_index == 1)),
                "decision_draw_predictions": int(np.sum(decision_index == 1)),
                "test_match_count": int(len(test)),
                "probability_layer_changed": False,
                "training_oof_rows": int(len(fit)),
                "training_max_prediction_year": int(fit["prediction_year"].max()),
            }
        )
        states[str(fold_id)] = dict(rule.metadata_)
    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        .sort_values(["prediction_year", "match_id"])
        .reset_index(drop=True)
    )
    metrics = pd.DataFrame(metric_rows).sort_values("test_year").reset_index(drop=True)
    metadata = {
        "outer_fold_count": int(metrics["outer_fold_id"].nunique()),
        "draw_positive_fold_count": int(metrics["decision_draw_predictions"].gt(0).sum()),
        "macro_f1_improved_fold_count": int(metrics["macro_f1_gain"].gt(0).sum()),
        "draw_recall_improved_fold_count": int(metrics["draw_recall_gain"].gt(0).sum()),
        "layer_type": "decision_only",
        "proper_probability_scores_valid_for_decision_layer": False,
        "probabilities_modified": False,
        "audit_year_used": False,
    }
    return NestedDecisionResult(predictions, metrics, states, metadata)


def _validate_inputs(training: pd.DataFrame, outer: pd.DataFrame) -> None:
    training_required = {
        "selection_outer_fold_id",
        "match_id",
        "prediction_year",
        "train_end_year",
        "is_oof",
        "home_win",
        "draw",
        "away_win",
        "target_label",
    }
    outer_required = {
        "outer_fold_id",
        "match_id",
        "prediction_year",
        "home_win",
        "draw",
        "away_win",
        "target_outcome",
    }
    for name, frame, required in (
        ("decision training", training, training_required),
        ("outer decision", outer, outer_required),
    ):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{name} ledger must be a nonempty DataFrame")
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} ledger is missing columns: {missing}")
        probability = frame[["home_win", "draw", "away_win"]].to_numpy(float)
        if not np.isfinite(probability).all() or np.any(probability < 0):
            raise ValueError(f"{name} probabilities must be finite and nonnegative")
        if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError(f"{name} probability rows must sum to one")
    if not training["is_oof"].astype(bool).all():
        raise ValueError("Decision training requires OOF rows only")
    train_end = pd.to_numeric(training["train_end_year"], errors="coerce")
    prediction_year = pd.to_numeric(training["prediction_year"], errors="coerce")
    if (
        train_end.isna().any()
        or prediction_year.isna().any()
        or not train_end.lt(prediction_year).all()
    ):
        raise ValueError("Decision training OOF chronology must be strictly earlier")
    if int(pd.to_numeric(outer["prediction_year"], errors="raise").max()) > 2010:
        raise ValueError("2014/audit predictions are forbidden during decision selection")


def _one_year(series: pd.Series, label: str) -> int:
    values = pd.to_numeric(series, errors="raise").unique()
    if len(values) != 1:
        raise ValueError(f"Each {label} fold must contain exactly one prediction year")
    return int(values[0])


__all__ = ["NestedDecisionResult", "apply_nested_decision"]
