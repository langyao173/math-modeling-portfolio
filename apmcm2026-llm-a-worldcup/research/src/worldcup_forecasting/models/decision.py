"""OOF-fitted point-decision policies that never modify probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .score_calibration import _outcome_labels, _validate_oof_provenance

OUTCOME_ORDER = np.asarray(("home_win", "draw", "away_win"), dtype=object)


@dataclass
class OOFCostSensitiveDecisionRule:
    """Select a symmetric draw utility multiplier from OOF predictions.

    Home and away retain equal utility, so exchanging the teams exchanges their
    decisions.  The fitted rule is a point-action layer: ``transform_probabilities``
    returns the submitted probability array unchanged, and proper probability
    scores are intentionally excluded from its valid evaluation metrics.
    """

    draw_multipliers: tuple[float, ...] = (
        0.75,
        0.9,
        1.0,
        1.05,
        1.1,
        1.2,
        1.35,
        1.5,
        1.75,
        2.0,
        2.5,
        3.0,
    )
    minimum_macro_f1_gain: float = 0.0

    def __post_init__(self) -> None:
        candidates = np.asarray(self.draw_multipliers, dtype=float)
        if candidates.ndim != 1 or candidates.size == 0:
            raise ValueError("draw_multipliers must be a nonempty one-dimensional grid")
        if not np.isfinite(candidates).all() or np.any(candidates <= 0):
            raise ValueError("draw_multipliers must be finite and positive")
        if not np.any(np.isclose(candidates, 1.0)):
            raise ValueError("draw_multipliers must include the unmodified decision rule 1.0")
        if self.minimum_macro_f1_gain < 0:
            raise ValueError("minimum_macro_f1_gain must be nonnegative")
        self.draw_multiplier_: float = 1.0
        self.is_fitted_: bool = False
        self.metadata_: dict[str, object] = {}

    def fit_oof(
        self,
        probabilities: np.ndarray,
        observed: np.ndarray,
        *,
        provenance: pd.DataFrame,
    ) -> OOFCostSensitiveDecisionRule:
        """Fit the point-action utility using chronological OOF rows only."""
        probability = _validate_probabilities(probabilities)
        _validate_oof_provenance(provenance, len(probability))
        labels = _outcome_labels(observed, len(probability))
        baseline_prediction = self._indices(probability, 1.0)
        baseline_f1 = _macro_f1(labels, baseline_prediction)
        baseline_draw_recall = _class_recall(labels, baseline_prediction, 1)
        scored: list[tuple[float, float, float, float]] = []
        for multiplier in sorted(set(float(value) for value in self.draw_multipliers)):
            prediction = self._indices(probability, multiplier)
            scored.append(
                (
                    _macro_f1(labels, prediction),
                    _class_recall(labels, prediction, 1),
                    -abs(np.log(multiplier)),
                    multiplier,
                )
            )
        best_f1, best_draw_recall, _, best_multiplier = max(scored)
        selected = best_f1 > baseline_f1 + self.minimum_macro_f1_gain + 1e-12
        self.draw_multiplier_ = best_multiplier if selected else 1.0
        final_prediction = self._indices(probability, self.draw_multiplier_)
        self.is_fitted_ = True
        self.metadata_ = {
            "layer_type": "decision_only",
            "probability_calibration": False,
            "fit_scope": "training_oof_only",
            "team_swap_symmetric": True,
            "draw_multiplier": float(self.draw_multiplier_),
            "selected": bool(selected),
            "baseline_macro_f1": float(baseline_f1),
            "selected_macro_f1": float(_macro_f1(labels, final_prediction)),
            "baseline_draw_recall": float(baseline_draw_recall),
            "selected_draw_recall": float(_class_recall(labels, final_prediction, 1)),
            "candidate_draw_recall": float(best_draw_recall),
            "valid_evaluation_metrics": (
                "macro_f1",
                "class_recall",
                "balanced_accuracy",
                "expected_decision_cost",
            ),
        }
        return self

    def fit(
        self,
        probabilities: np.ndarray,
        observed: np.ndarray,
        *,
        provenance: pd.DataFrame | None = None,
        is_oof: bool = False,
    ) -> OOFCostSensitiveDecisionRule:
        """Compatibility fit that still requires explicit OOF evidence."""
        if provenance is not None:
            return self.fit_oof(probabilities, observed, provenance=provenance)
        probability = _validate_probabilities(probabilities)
        if not is_oof:
            raise ValueError("Decision-rule fitting requires out-of-fold (OOF) predictions")
        synthetic = pd.DataFrame(
            {
                "is_oof": True,
                "prediction_year": np.ones(len(probability)),
                "train_end_year": np.zeros(len(probability)),
            }
        )
        return self.fit_oof(probability, observed, provenance=synthetic)

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Return cost-sensitive outcome labels without recalibrating probabilities."""
        if not self.is_fitted_:
            raise RuntimeError("OOFCostSensitiveDecisionRule must be fitted before prediction")
        probability = _validate_probabilities(probabilities)
        return OUTCOME_ORDER[self._indices(probability, self.draw_multiplier_)]

    def transform_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """Return an exact copy: this layer has no probabilistic transform."""
        probability = _validate_probabilities(probabilities)
        return probability.copy()

    @staticmethod
    def _indices(probability: np.ndarray, draw_multiplier: float) -> np.ndarray:
        utility = probability.copy()
        utility[:, 1] *= draw_multiplier
        decision = np.argmax(utility, axis=1)
        symmetric_tie = np.isclose(utility[:, 0], utility[:, 2], atol=1e-15, rtol=0) & (
            utility[:, 0] >= utility[:, 1]
        )
        decision[symmetric_tie] = 1
        return decision


CostSensitiveDecisionRule = OOFCostSensitiveDecisionRule


def _validate_probabilities(value: np.ndarray) -> np.ndarray:
    probability = np.asarray(value, dtype=float)
    if probability.ndim != 2 or probability.shape[0] == 0 or probability.shape[1] != 3:
        raise ValueError("Outcome probabilities must have shape (n, 3)")
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("Outcome probabilities must be finite and nonnegative")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Every outcome probability row must sum to one")
    return probability


def _class_recall(truth: np.ndarray, prediction: np.ndarray, label: int) -> float:
    selected = truth == label
    return float(np.mean(prediction[selected] == label)) if selected.any() else 0.0


def _macro_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    scores: list[float] = []
    for label in range(3):
        true_positive = int(np.sum((truth == label) & (prediction == label)))
        false_positive = int(np.sum((truth != label) & (prediction == label)))
        false_negative = int(np.sum((truth == label) & (prediction != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


__all__ = ["CostSensitiveDecisionRule", "OOFCostSensitiveDecisionRule"]
