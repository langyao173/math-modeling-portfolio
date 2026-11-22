"""Problem-3 model registry: one official multiclass logistic regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 2026

# The problem statement names exactly these four historical team indicators.
# A match is represented by home-minus-away differences, so the model remains
# compact and team-order symmetry can be enforced without extra features.
P3_FEATURE_NAMES = [
    "TournamentExperienceDiff",
    "BestStageDiff",
    "HistAvgGFDiff",
    "HistAvgGADiff",
]
P3_FEATURE_SETS = {"题面四项历史特征": P3_FEATURE_NAMES}

# Selected only from pre-2014 rolling development folds.  The one-standard-
# error rule retained the strongest regularization that still predicted all
# three classes, preventing the 2014 test set from entering model selection.
P3_FINAL_CONFIG = {
    "model_name": "题面四特征多分类逻辑回归",
    "feature_names": P3_FEATURE_NAMES,
    "C": 0.001,
    "class_weight": "balanced",
    "augment_swapped": True,
    "form_decay": 0.8,
    "entity_policy": "conservative",
    "selection_window": "1982-2010 rolling development folds; held-out test excluded",
}
P3_DECISION_CONFIG = P3_FINAL_CONFIG


def swap_prematch_feature_rows(
    frame: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return the same match with home and away teams exchanged."""
    source = frame[feature_names].copy()
    swapped = pd.DataFrame(index=source.index)
    for name in feature_names:
        if name.startswith("Home") and f"Away{name[4:]}" in source:
            swapped[name] = source[f"Away{name[4:]}"]
        elif name.startswith("Away") and f"Home{name[4:]}" in source:
            swapped[name] = source[f"Home{name[4:]}"]
        elif name.endswith("Diff"):
            swapped[name] = -source[name]
        else:
            swapped[name] = source[name]
    return swapped[feature_names]


@dataclass
class MulticlassLogisticModel:
    """StandardScaler plus one three-class LogisticRegression estimator."""

    config: dict
    scaler: StandardScaler
    model: LogisticRegression

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        config: dict | None = None,
    ) -> "MulticlassLogisticModel":
        selected = dict(P3_FINAL_CONFIG if config is None else config)
        names = list(selected["feature_names"])
        missing = [name for name in names if name not in features]
        if missing:
            raise ValueError(f"逻辑回归缺少特征：{', '.join(missing)}")
        target = features["Outcome"].map(
            {"Home Win": 0, "Draw": 1, "Away Win": 2}
        )
        if target.isna().any() or target.nunique() != 3:
            raise ValueError("训练标签必须同时包含主胜、平局和客胜。")

        matrix = features[names].copy()
        labels = target.to_numpy(int)
        if bool(selected.get("augment_swapped", True)):
            swapped = swap_prematch_feature_rows(matrix, names)
            matrix = pd.concat([matrix, swapped], ignore_index=True)
            labels = np.concatenate(
                [labels, np.where(labels == 0, 2, np.where(labels == 2, 0, 1))]
            )

        scaler = StandardScaler().fit(matrix)
        model = LogisticRegression(
            C=float(selected["C"]),
            class_weight=selected.get("class_weight"),
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ).fit(scaler.transform(matrix), labels)
        return cls(config=selected, scaler=scaler, model=model)

    def _ordered_probability(self, frame: pd.DataFrame) -> np.ndarray:
        names = list(self.config["feature_names"])
        raw = self.model.predict_proba(self.scaler.transform(frame[names]))
        result = np.zeros((len(frame), 3), dtype=float)
        for index, label in enumerate(self.model.classes_):
            result[:, int(label)] = raw[:, index]
        return result

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        names = list(self.config["feature_names"])
        original = self._ordered_probability(frame)
        swapped = self._ordered_probability(
            swap_prematch_feature_rows(frame, names)
        )
        probabilities = 0.5 * (original + swapped[:, [2, 1, 0]])
        probabilities = np.clip(probabilities, 1e-12, 1.0)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(frame).argmax(axis=1).astype(int)


__all__ = [
    "MulticlassLogisticModel",
    "P3_DECISION_CONFIG",
    "P3_FEATURE_NAMES",
    "P3_FEATURE_SETS",
    "P3_FINAL_CONFIG",
    "RANDOM_STATE",
    "swap_prematch_feature_rows",
]
