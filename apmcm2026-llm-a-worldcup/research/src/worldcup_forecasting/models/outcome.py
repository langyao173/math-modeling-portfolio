"""Leakage-safe multinomial logistic baseline with exact swap symmetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ._utils import dataclass_config, resolve_column
from .registry import register_model

OUTCOME_ORDER = ("home_win", "draw", "away_win")
MANDATORY_OUTCOME_FEATURES = (
    "historical_appearances_diff",
    "historical_best_finish_diff",
    "historical_mean_goals_for_diff",
    "historical_mean_goals_conceded_diff",
)


@dataclass(frozen=True)
class OutcomeLogitConfig:
    """Configuration for the official four-feature outcome baseline."""

    feature_columns: tuple[str, ...] = MANDATORY_OUTCOME_FEATURES
    target_column: str = "target_outcome"
    year_column: str = "Year"
    regularization_c: float = 1.0
    max_iter: int = 2_000
    random_state: int = 1729
    swap_augmentation: bool = True
    symmetrize_prediction: bool = True


@register_model(
    "outcome_logit",
    "outcome_classification",
    ("standardized", "multinomial", "swap_augmentation", "symmetrized_probability"),
)
class OutcomeLogit:
    """Standardized multinomial logit on exactly four antisymmetric features."""

    def __init__(self, config: OutcomeLogitConfig | object | None = None) -> None:
        self.config: OutcomeLogitConfig = dataclass_config(OutcomeLogitConfig, config)
        if len(self.config.feature_columns) != 4:
            raise ValueError("OutcomeLogit requires exactly four difference features")
        self.scaler_: StandardScaler | None = None
        self.classifier_: LogisticRegression | None = None
        self.temporal_metrics_: dict[str, Any] | None = None

    def fit(
        self,
        frame: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series | None = None,
    ) -> OutcomeLogit:
        """Fit with mandatory home/away mirrored augmentation."""
        x = self._features(frame)
        labels = self._labels(frame, y)
        if x.shape[0] != labels.shape[0] or x.shape[0] == 0:
            raise ValueError("Outcome features and labels must have equal nonzero row counts")
        if not np.isfinite(x).all():
            raise ValueError("Outcome difference features must be finite")
        if set(labels) != set(OUTCOME_ORDER):
            raise ValueError("OutcomeLogit training data must contain all three outcome classes")
        if self.config.swap_augmentation:
            x_fit = np.vstack([x, -x])
            labels_fit = np.concatenate([labels, _swap_labels(labels)])
        else:
            x_fit = x
            labels_fit = labels
        self.scaler_ = StandardScaler().fit(x_fit)
        self.classifier_ = LogisticRegression(
            C=float(self.config.regularization_c),
            solver="lbfgs",
            max_iter=int(self.config.max_iter),
            random_state=int(self.config.random_state),
        ).fit(self.scaler_.transform(x_fit), labels_fit)
        return self

    def predict_proba(self, frame: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return probabilities in [home_win, draw, away_win] order."""
        self._require_fitted()
        x = self._features(frame)
        if not np.isfinite(x).all():
            raise ValueError("Outcome difference features must be finite")
        direct = self._ordered_raw_probability(x)
        if not self.config.symmetrize_prediction:
            return direct
        swapped = self._ordered_raw_probability(-x)[:, [2, 1, 0]]
        probability = 0.5 * (direct + swapped)
        return probability / probability.sum(axis=1, keepdims=True)

    def predict(self, frame: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return maximum-probability outcome labels."""
        indices = np.argmax(self.predict_proba(frame), axis=1)
        return np.asarray(OUTCOME_ORDER, dtype=object)[indices]

    def coefficients(self) -> pd.DataFrame:
        """Return standardized multinomial coefficients in canonical class order."""
        self._require_fitted()
        assert self.classifier_ is not None
        order = [list(self.classifier_.classes_).index(label) for label in OUTCOME_ORDER]
        return pd.DataFrame(
            self.classifier_.coef_[order],
            index=OUTCOME_ORDER,
            columns=self.config.feature_columns,
        )

    def fit_temporal(
        self,
        frame: pd.DataFrame,
        *,
        train_end_year: int = 2010,
        test_year: int = 2014,
    ) -> dict[str, Any]:
        """Fit through a cutoff and evaluate the held-out edition."""
        year_name = resolve_column(frame, self.config.year_column, ("year",))
        years = pd.to_numeric(frame[year_name], errors="coerce")
        train = frame.loc[years <= train_end_year].copy()
        test = frame.loc[years == test_year].copy()
        if train.empty or test.empty:
            raise ValueError("Temporal outcome evaluation needs nonempty train and test rows")
        self.fit(train)
        metrics = {
            "train_end_year": int(train_end_year),
            "test_year": int(test_year),
            "train": _classification_summary(self._labels(train, None), self.predict_proba(train)),
            "test": _classification_summary(self._labels(test, None), self.predict_proba(test)),
        }
        self.temporal_metrics_ = metrics
        return metrics

    def _features(self, frame: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(frame, pd.DataFrame):
            columns = [resolve_column(frame, name) for name in self.config.feature_columns]
            return frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        values = np.asarray(frame, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.config.feature_columns):
            raise ValueError("Outcome feature array must have shape (n_matches, 4)")
        return values

    def _labels(
        self,
        frame: pd.DataFrame | np.ndarray,
        y: np.ndarray | pd.Series | None,
    ) -> np.ndarray:
        if y is None:
            if not isinstance(frame, pd.DataFrame):
                raise ValueError("y is required when OutcomeLogit receives an array")
            target = resolve_column(
                frame, self.config.target_column, ("outcome", "scoreline_result")
            )
            source = frame[target].to_numpy()
        else:
            source = np.asarray(y)
        if source.ndim != 1:
            raise ValueError("Outcome labels must be one-dimensional")
        return np.asarray([_normalize_label(value) for value in source], dtype=object)

    def _ordered_raw_probability(self, x: np.ndarray) -> np.ndarray:
        assert self.scaler_ is not None and self.classifier_ is not None
        raw = self.classifier_.predict_proba(self.scaler_.transform(x))
        indices = [list(self.classifier_.classes_).index(label) for label in OUTCOME_ORDER]
        return raw[:, indices]

    def _require_fitted(self) -> None:
        if self.scaler_ is None or self.classifier_ is None:
            raise RuntimeError("OutcomeLogit must be fitted before prediction")


def _normalize_label(value: object) -> str:
    if isinstance(value, (int, np.integer)):
        if int(value) < 0 or int(value) >= len(OUTCOME_ORDER):
            raise ValueError(f"Unknown outcome code: {value}")
        try:
            return OUTCOME_ORDER[int(value)]
        except IndexError as error:
            raise ValueError(f"Unknown outcome code: {value}") from error
    text = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "home": "home_win",
        "homewin": "home_win",
        "h": "home_win",
        "1": "home_win",
        "draw": "draw",
        "d": "draw",
        "0": "draw",
        "away": "away_win",
        "awaywin": "away_win",
        "a": "away_win",
        "2": "away_win",
    }
    normalized = aliases.get(text, text)
    if normalized not in OUTCOME_ORDER:
        raise ValueError(f"Unknown outcome label: {value!r}")
    return normalized


def _swap_labels(labels: np.ndarray) -> np.ndarray:
    mapping = {"home_win": "away_win", "draw": "draw", "away_win": "home_win"}
    return np.asarray([mapping[str(label)] for label in labels], dtype=object)


def _classification_summary(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    truth = np.asarray([OUTCOME_ORDER.index(str(label)) for label in labels], dtype=int)
    predicted = probability.argmax(axis=1)
    observed = np.eye(3)[truth]
    clipped = np.clip(probability, 1e-15, 1.0)
    return {
        "accuracy": float(np.mean(predicted == truth)),
        "log_loss": float(-np.mean(np.log(clipped[np.arange(truth.size), truth]))),
        "multiclass_brier": float(np.mean(np.sum((probability - observed) ** 2, axis=1))),
        "n": int(truth.size),
    }
