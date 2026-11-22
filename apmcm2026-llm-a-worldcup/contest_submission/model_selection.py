"""Compact time-validation utilities for the official logistic model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

try:
    from features import CLASS_TO_INT
    from models import MulticlassLogisticModel, P3_FINAL_CONFIG
except ImportError:
    from .features import CLASS_TO_INT
    from .models import MulticlassLogisticModel, P3_FINAL_CONFIG


def multiclass_brier_score(truth: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(3, dtype=float)[truth]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    truth: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = prediction == truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        selected = (confidence > edges[index]) & (
            confidence <= edges[index + 1]
        )
        if selected.any():
            error += float(selected.mean()) * abs(
                float(correct[selected].mean())
                - float(confidence[selected].mean())
            )
    return float(error)


def metric_row(
    model_name: str,
    year: int,
    truth: np.ndarray,
    prediction: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    draw_mask = truth == 1
    return {
        "Model": model_name,
        "ValidationYear": int(year),
        "Accuracy": float(accuracy_score(truth, prediction)),
        "MacroF1": float(
            f1_score(truth, prediction, average="macro", zero_division=0)
        ),
        "LogLoss": float(log_loss(truth, probabilities, labels=[0, 1, 2])),
        "BrierScore": multiclass_brier_score(truth, probabilities),
        "ECE": expected_calibration_error(truth, probabilities),
        "DrawRecall": (
            float(np.mean(prediction[draw_mask] == 1))
            if draw_mask.any()
            else np.nan
        ),
        "CorrectMatches": int(np.sum(truth == prediction)),
        "MatchCount": int(len(truth)),
        "PredictedClassCount": int(len(np.unique(prediction))),
        "ValidationProtocol": "train years < validation year; whole Cup frozen",
    }


def candidate_configs() -> list[dict]:
    """Small grid fixed before the 2014 test is evaluated."""
    result = []
    for c_value in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        for class_weight in (None, "balanced"):
            config = dict(P3_FINAL_CONFIG)
            config.update({"C": c_value, "class_weight": class_weight})
            result.append(config)
    return result


def evaluate_logistic_fold(
    features: pd.DataFrame,
    year: int,
    config: dict,
) -> tuple[dict, np.ndarray]:
    train = features[features["Year"] < year]
    valid = features[features["Year"] == year]
    if len(train) < 100 or valid.empty:
        raise ValueError(f"{year} 折没有足够的训练或验证样本。")
    model = MulticlassLogisticModel.fit(train, config)
    probabilities = model.predict_proba(valid)
    prediction = probabilities.argmax(axis=1)
    truth = valid["Outcome"].map(CLASS_TO_INT).to_numpy(int)
    return (
        metric_row(config["model_name"], year, truth, prediction, probabilities),
        probabilities,
    )


def rolling_validation(
    features: pd.DataFrame,
    config: dict,
    years: list[int],
) -> pd.DataFrame:
    rows = []
    for year in years:
        if (features["Year"] == year).any():
            row, _ = evaluate_logistic_fold(features, year, config)
            rows.append(row)
    return pd.DataFrame(rows)


def select_config_on_years(
    features: pd.DataFrame,
    validation_years: list[int],
) -> tuple[dict, pd.DataFrame]:
    """One-SE selection using only years before the official 2014 test."""
    configs = candidate_configs()
    rows = []
    for index, config in enumerate(configs):
        detail = rolling_validation(features, config, validation_years)
        rows.append(
            {
                "ConfigIndex": index,
                "C": config["C"],
                "ClassWeight": str(config["class_weight"]),
                "MeanAccuracy": detail["Accuracy"].mean(),
                "StdAccuracy": detail["Accuracy"].std(ddof=1),
                "WorstAccuracy": detail["Accuracy"].min(),
                "MeanDrawRecall": detail["DrawRecall"].mean(),
                "ThreeClassFolds": int(
                    (detail["PredictedClassCount"] == 3).sum()
                ),
                "ValidationFolds": len(detail),
            }
        )
    table = pd.DataFrame(rows)
    table["StandardError"] = table["StdAccuracy"].fillna(0.0) / np.sqrt(
        table["ValidationFolds"]
    )
    best = table.sort_values(
        ["MeanAccuracy", "WorstAccuracy"],
        ascending=[False, False],
        kind="mergesort",
    ).iloc[0]
    threshold = float(best["MeanAccuracy"] - best["StandardError"])
    table["OneSEThreshold"] = threshold
    table["WithinOneSE"] = table["MeanAccuracy"] >= threshold - 1e-12
    eligible = table[
        table["WithinOneSE"]
        & (table["MeanDrawRecall"] > 0)
        & (table["ThreeClassFolds"] > 0)
    ]
    if eligible.empty:
        eligible = table[table["WithinOneSE"]]
    selected = eligible.sort_values(
        ["WorstAccuracy", "C", "MeanAccuracy", "ConfigIndex"],
        ascending=[False, True, False, True],
        kind="mergesort",
    ).iloc[0]
    table["Selected"] = table["ConfigIndex"].eq(int(selected["ConfigIndex"]))
    return configs[int(selected["ConfigIndex"])], table


__all__ = [
    "candidate_configs",
    "evaluate_logistic_fold",
    "expected_calibration_error",
    "metric_row",
    "multiclass_brier_score",
    "rolling_validation",
    "select_config_on_years",
]
