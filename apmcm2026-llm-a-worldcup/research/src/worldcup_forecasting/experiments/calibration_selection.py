"""Chronological selection among predeclared coherent outcome calibrators."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..models.calibration import TemperatureScaler
from ..models.score_calibration import (
    _outcome_labels,
    _validate_oof_provenance,
    minimal_kl_partition_projection,
    outcome_margins,
)
from ..validation.metrics import expected_calibration_error

METHODS = ("identity", "temperature", "log_probability_logit", "classwise_isotonic")
SELECTION_OBJECTIVE = (
    "mean_log_loss + 0.25*mean_multiclass_brier + 0.05*mean_ece "
    "+ 0.05*sd_edition_log_loss"
)


@dataclass(frozen=True)
class CalibrationSelectionConfig:
    """Frozen budget and support rules for calibration selection."""

    maximum_development_year: int = 2010
    minimum_fit_rows: int = 12
    isotonic_minimum_class_count: int = 3
    logistic_c: float = 1.0
    max_iter: int = 2_000
    random_state: int = 31_415

    def __post_init__(self) -> None:
        if self.maximum_development_year < 1930:
            raise ValueError("maximum_development_year is implausibly early")
        if self.minimum_fit_rows < 1 or self.isotonic_minimum_class_count < 1:
            raise ValueError("Calibration support thresholds must be positive")
        if self.logistic_c <= 0 or self.max_iter <= 0:
            raise ValueError("logistic_c and max_iter must be positive")


class ChronologicalCalibrationSelector:
    """Cross-fit, rank, refit, and coherently apply outcome calibration."""

    def __init__(self, config: CalibrationSelectionConfig | None = None) -> None:
        self.config = config or CalibrationSelectionConfig()
        self.chosen_method_: str | None = None
        self.final_calibrator_: Any | None = None
        self.crossfit_predictions_: pd.DataFrame = pd.DataFrame()
        self.selection_report_: dict[str, Any] = {}
        self.state_: dict[str, Any] = {}

    def fit_oof(
        self,
        joint: np.ndarray,
        observed: np.ndarray,
        *,
        provenance: pd.DataFrame,
    ) -> ChronologicalCalibrationSelector:
        """Select using edition-blocked cross-fitting, then refit on development OOF."""
        base = outcome_margins(joint)
        _validate_oof_provenance(provenance, len(base))
        years = _development_years(provenance, self.config.maximum_development_year)
        labels = _outcome_labels(observed, len(base))
        method_report = {
            name: {
                "eligible_years": [],
                "skipped_years": [],
                "train_max_year_by_scored_year": {},
                "crossfit_metrics": None,
            }
            for name in METHODS
        }
        records: list[dict[str, Any]] = []
        unique_years = sorted(np.unique(years).astype(int).tolist())
        for method in METHODS:
            for scored_year in unique_years:
                score_index = np.flatnonzero(years == scored_year)
                train_index = np.flatnonzero(years < scored_year)
                if method == "identity":
                    calibrator: Any = _IdentityCalibrator()
                    train_max: int | None = None
                else:
                    try:
                        calibrator = _fit_method(
                            method,
                            base[train_index],
                            labels[train_index],
                            self.config,
                        )
                    except _Ineligible as error:
                        method_report[method]["skipped_years"].append(
                            {"year": scored_year, "reason": str(error)}
                        )
                        continue
                    train_max = int(years[train_index].max())
                    if train_max >= scored_year:
                        raise AssertionError("Calibration training is not strictly earlier")
                calibrated = _symmetrized_transform(calibrator, base[score_index])
                method_report[method]["eligible_years"].append(scored_year)
                method_report[method]["train_max_year_by_scored_year"][str(scored_year)] = train_max
                for position, probability in zip(score_index, calibrated, strict=True):
                    records.append(
                        {
                            "method": method,
                            "row_position": int(position),
                            "prediction_year": scored_year,
                            "train_max_year": train_max,
                            "home_win": float(probability[0]),
                            "draw": float(probability[1]),
                            "away_win": float(probability[2]),
                            "target_label": int(labels[position]),
                        }
                    )
        self.crossfit_predictions_ = pd.DataFrame.from_records(records).sort_values(
            ["method", "prediction_year", "row_position"], kind="stable"
        ).reset_index(drop=True)
        active = [name for name in METHODS if method_report[name]["eligible_years"]]
        for name in METHODS:
            method_report[name]["overall_eligible"] = bool(method_report[name]["eligible_years"])
        calibrated_active = [name for name in active if name != "identity"]
        fallback_reason: str | None = None
        if not calibrated_active:
            comparison_methods = ["identity"]
            comparison_years = unique_years
            fallback_reason = "no_eligible_calibrated_method"
        else:
            comparison_methods = ["identity", *calibrated_active]
            year_sets = [set(method_report[name]["eligible_years"]) for name in comparison_methods]
            comparison_years = sorted(set.intersection(*year_sets))
            if not comparison_years:
                comparison_methods = ["identity"]
                comparison_years = unique_years
                fallback_reason = "no_common_scored_edition"
        ranking = _rank_methods(
            self.crossfit_predictions_, comparison_methods, comparison_years
        )
        for row in ranking:
            method_report[row["method"]]["crossfit_metrics"] = {
                key: row[key]
                for key in (
                    "log_loss",
                    "multiclass_brier",
                    "ece",
                    "sd_edition_log_loss",
                    "selection_objective",
                    "edition_count",
                )
            }
        chosen = ranking[0]["method"] if ranking else "identity"
        if fallback_reason is None and chosen == "identity":
            fallback_reason = "identity_ranked_best"
        final = _fit_method(chosen, base, labels, self.config)
        self.chosen_method_ = chosen
        self.final_calibrator_ = final
        final_max_year = int(years.max())
        self.selection_report_ = {
            "fit_scope": "development_oof_through_2010",
            "selection_scheme": "strictly_earlier_edition_cross_fitting",
            "selection_objective": SELECTION_OBJECTIVE,
            "runtime_charge": 0.0,
            "complexity_charge": 0.0,
            "charge_scope": (
                "structural_candidate_fixed_before_calibration; no calibration-specific "
                "runtime or complexity charge was defined in the frozen protocol"
            ),
            "equal_edition_weighting": True,
            "methods": method_report,
            "comparison_years": comparison_years,
            "ranking": ranking,
            "chosen_method": chosen,
            "fallback_reason": fallback_reason,
            "final_refit_row_count": int(len(base)),
            "final_refit_max_year": final_max_year,
        }
        self.state_ = {
            "schema_version": 2,
            "chosen_method": chosen,
            "config": asdict(self.config),
            "final_fit": {
                "row_count": int(len(base)),
                "maximum_prediction_year": final_max_year,
                "oof_only": True,
            },
            "calibrator": final.state_dict(),
        }
        return self

    def transform_outcomes(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply the selected outcome calibrator with exact team-swap symmetrization."""
        if self.final_calibrator_ is None:
            raise RuntimeError("ChronologicalCalibrationSelector must be fitted first")
        values = _validate_probabilities(probabilities)
        return _symmetrized_transform(self.final_calibrator_, values)

    def transform(self, joint: np.ndarray) -> np.ndarray:
        """Project calibrated margins back onto the original joint-score shapes."""
        if self.chosen_method_ == "identity":
            outcome_margins(joint)  # validates the joint-score contract
            return np.asarray(joint, dtype=float).copy()
        target = self.transform_outcomes(outcome_margins(joint))
        return minimal_kl_partition_projection(joint, target, zero_mass_policy="uniform")

    def state_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible copy of the fitted state."""
        if not self.state_:
            raise RuntimeError("ChronologicalCalibrationSelector must be fitted first")
        return json.loads(json.dumps(self.state_, sort_keys=True))


class _IdentityCalibrator:
    def transform(self, probability: np.ndarray) -> np.ndarray:
        return np.asarray(probability, dtype=float).copy()

    def state_dict(self) -> dict[str, Any]:
        return {"method": "identity"}


class _TemperatureCalibrator:
    def __init__(self, probability: np.ndarray, labels: np.ndarray) -> None:
        self.scaler = TemperatureScaler().fit(probability, labels)

    def transform(self, probability: np.ndarray) -> np.ndarray:
        return self.scaler.transform(probability)

    def state_dict(self) -> dict[str, Any]:
        return {"method": "temperature", "temperature": float(self.scaler.temperature_)}


class _LogProbabilityLogit:
    def __init__(
        self,
        probability: np.ndarray,
        labels: np.ndarray,
        config: CalibrationSelectionConfig,
    ) -> None:
        self.model = LogisticRegression(
            C=config.logistic_c,
            max_iter=config.max_iter,
            random_state=config.random_state,
        ).fit(_log_features(probability), labels)

    def transform(self, probability: np.ndarray) -> np.ndarray:
        raw = self.model.predict_proba(_log_features(probability))
        order = [list(self.model.classes_).index(label) for label in range(3)]
        return raw[:, order]

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": "log_probability_logit",
            "classes": self.model.classes_.astype(int).tolist(),
            "coefficient": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
        }


class _ClasswiseIsotonic:
    def __init__(
        self,
        probability: np.ndarray,
        labels: np.ndarray,
        minimum_class_count: int,
    ) -> None:
        self.models: list[IsotonicRegression] = []
        for label in range(3):
            target = (labels == label).astype(float)
            positive = int(target.sum())
            negative = int(len(target) - positive)
            if min(positive, negative) < minimum_class_count:
                raise _Ineligible(f"class {label} has insufficient positive/negative support")
            if np.unique(probability[:, label]).size < 2:
                raise _Ineligible(f"class {label} probabilities have no isotonic support")
            self.models.append(
                IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
                    probability[:, label], target
                )
            )

    def transform(self, probability: np.ndarray) -> np.ndarray:
        raw = np.column_stack(
            [model.predict(probability[:, label]) for label, model in enumerate(self.models)]
        )
        total = raw.sum(axis=1, keepdims=True)
        missing = total[:, 0] <= 0
        raw[missing] = probability[missing]
        return raw / raw.sum(axis=1, keepdims=True)

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": "classwise_isotonic",
            "models": [
                {
                    "x_thresholds": model.X_thresholds_.tolist(),
                    "y_thresholds": model.y_thresholds_.tolist(),
                }
                for model in self.models
            ],
        }


class _Ineligible(ValueError):
    pass


def _fit_method(
    method: str,
    probability: np.ndarray,
    labels: np.ndarray,
    config: CalibrationSelectionConfig,
) -> Any:
    if method == "identity":
        return _IdentityCalibrator()
    if len(probability) < config.minimum_fit_rows:
        raise _Ineligible(f"only {len(probability)} earlier rows; need {config.minimum_fit_rows}")
    if method == "temperature":
        return _TemperatureCalibrator(probability, labels)
    if method == "log_probability_logit":
        if set(labels.tolist()) != {0, 1, 2}:
            raise _Ineligible("multinomial logit requires all three outcome classes")
        return _LogProbabilityLogit(probability, labels, config)
    if method == "classwise_isotonic":
        return _ClasswiseIsotonic(
            probability, labels, config.isotonic_minimum_class_count
        )
    raise ValueError(f"Unknown calibration method: {method}")


def _symmetrized_transform(calibrator: Any, probability: np.ndarray) -> np.ndarray:
    direct = np.asarray(calibrator.transform(probability), dtype=float)
    reverse = np.asarray(calibrator.transform(probability[:, [2, 1, 0]]), dtype=float)[
        :, [2, 1, 0]
    ]
    average = 0.5 * (direct + reverse)
    return average / average.sum(axis=1, keepdims=True)


def _rank_methods(
    predictions: pd.DataFrame,
    methods: list[str],
    years: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        edition_scores = []
        for year in years:
            selected = predictions.loc[
                predictions["method"].eq(method)
                & predictions["prediction_year"].eq(year)
            ].sort_values("row_position")
            probability = selected[["home_win", "draw", "away_win"]].to_numpy(float)
            labels = selected["target_label"].to_numpy(int)
            edition_scores.append(_selection_scores(probability, labels))
        score_array = np.asarray(edition_scores, dtype=float)
        mean_log_loss = float(score_array[:, 0].mean())
        mean_brier = float(score_array[:, 1].mean())
        mean_ece = float(score_array[:, 2].mean())
        sd_log_loss = float(score_array[:, 0].std(ddof=0))
        objective = float(
            mean_log_loss
            + 0.25 * mean_brier
            + 0.05 * mean_ece
            + 0.05 * sd_log_loss
        )
        rows.append(
            {
                "method": method,
                "scored_years": years,
                "edition_count": len(years),
                "log_loss": mean_log_loss,
                "multiclass_brier": mean_brier,
                "ece": mean_ece,
                "sd_edition_log_loss": sd_log_loss,
                "selection_objective": objective,
            }
        )
    for metric, rank_name in (("log_loss", "log_loss_rank"), ("multiclass_brier", "brier_rank")):
        order = sorted(range(len(rows)), key=lambda index: (rows[index][metric], index))
        for rank, index in enumerate(order, start=1):
            rows[index][rank_name] = rank
    for row in rows:
        row["rank_sum"] = row["log_loss_rank"] + row["brier_rank"]
    method_order = {name: index for index, name in enumerate(METHODS)}
    return sorted(
        rows,
        key=lambda row: (
            row["selection_objective"],
            row["log_loss"],
            row["multiclass_brier"],
            method_order[row["method"]],
        ),
    )


def _selection_scores(
    probability: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float]:
    selected = np.clip(probability[np.arange(len(labels)), labels], 1e-15, 1.0)
    observed = np.eye(3)[labels]
    return (
        float(-np.mean(np.log(selected))),
        float(np.mean(np.sum((probability - observed) ** 2, axis=1))),
        expected_calibration_error(labels, probability),
    )


def _development_years(provenance: pd.DataFrame, maximum: int) -> np.ndarray:
    values = pd.to_numeric(provenance["prediction_year"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or not np.allclose(values, np.round(values)):
        raise ValueError("prediction_year must contain finite integer editions")
    if np.any(values > maximum):
        raise ValueError(
            f"Calibration development fitting is capped at {maximum}; audit year found"
        )
    return values.astype(int)


def _validate_probabilities(value: np.ndarray) -> np.ndarray:
    probability = np.asarray(value, dtype=float)
    if probability.ndim != 2 or probability.shape[0] == 0 or probability.shape[1] != 3:
        raise ValueError("Outcome probabilities must have shape (n, 3)")
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("Outcome probabilities must be finite and nonnegative")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Every outcome probability row must sum to one")
    return probability


def _log_features(probability: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probability, 1e-12, 1.0))


__all__ = [
    "CalibrationSelectionConfig",
    "ChronologicalCalibrationSelector",
    "METHODS",
    "SELECTION_OBJECTIVE",
]
