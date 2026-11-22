"""Explicitly ordered, non-equivariant A8 diagnostic model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from worldcup_forecasting.models.advanced_count import EquivariantCountConfig


class AsymmetricSidePoisson:
    """Fit home and away goals independently in the recorded team order."""

    def __init__(self, config: EquivariantCountConfig) -> None:
        self.config = config
        self.home_model_: Pipeline | None = None
        self.away_model_: Pipeline | None = None
        self.numeric_: tuple[str, ...] = ()
        self.categorical_: tuple[str, ...] = ()
        self.shared_fraction_: float = 0.0

    def fit(self, frame: pd.DataFrame) -> AsymmetricSidePoisson:
        self.numeric_, self.categorical_ = _features(frame, self.config)
        home_target, away_target = _targets(frame, self.config).T
        self.home_model_ = self._pipeline()
        self.away_model_ = self._pipeline()
        weights = _time_weights(frame, self.config)
        self.home_model_.fit(frame, home_target, poisson__sample_weight=weights)
        self.away_model_.fit(frame, away_target, poisson__sample_weight=weights)
        rates = self.predict_rates(frame)
        if self.config.shared_score:
            covariance = np.mean((home_target - rates[:, 0]) * (away_target - rates[:, 1]))
            scale = float(np.mean(np.minimum(rates[:, 0], rates[:, 1])))
            if scale > 0:
                self.shared_fraction_ = float(
                    np.clip(covariance / scale, 0.0, self.config.max_shared_fraction)
                )
        return self

    def predict_rates(self, frame: pd.DataFrame) -> np.ndarray:
        if self.home_model_ is None or self.away_model_ is None:
            raise RuntimeError("AsymmetricSidePoisson must be fitted before prediction")
        rates = np.column_stack(
            [self.home_model_.predict(frame), self.away_model_.predict(frame)]
        ).astype(float)
        if not np.isfinite(rates).all() or np.any(rates <= 0):
            raise RuntimeError("A8 produced invalid rates")
        return rates

    def _pipeline(self) -> Pipeline:
        transformer = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    list(self.numeric_),
                ),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    list(self.categorical_),
                ),
            ],
            remainder="drop",
        )
        return Pipeline(
            [
                ("design", transformer),
                (
                    "poisson",
                    PoissonRegressor(
                        alpha=max(float(self.config.alpha), 1e-9),
                        max_iter=int(self.config.max_iter),
                        tol=float(self.config.tolerance),
                    ),
                ),
            ]
        )


def _features(
    frame: pd.DataFrame, config: EquivariantCountConfig
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    numeric = [
        *config.odd_numeric_features,
        *config.even_numeric_features,
        *config.common_pace_features,
        *[name for pair in config.paired_numeric_features for name in pair],
    ]
    numeric = list(dict.fromkeys(name for name in numeric if name in frame))
    categorical = [
        config.home_team_column,
        config.away_team_column,
        *config.even_categorical_features,
    ]
    if config.host_side_column:
        categorical.append(config.host_side_column)
    categorical = list(dict.fromkeys(name for name in categorical if name in frame))
    if not numeric:
        raise ValueError("A8 requires at least one ordered numeric feature")
    return tuple(numeric), tuple(categorical)


def _targets(frame: pd.DataFrame, config: EquivariantCountConfig) -> np.ndarray:
    home = config.home_goals_column or config.target_home_column
    away = config.away_goals_column or config.target_away_column
    if home not in frame or away not in frame:
        raise ValueError("A8 target columns are missing")
    values = frame[[home, away]].apply(pd.to_numeric, errors="raise").to_numpy(float)
    valid = (
        np.isfinite(values).all()
        and np.all(values >= 0)
        and np.allclose(values, np.rint(values), atol=1e-12)
    )
    if not valid:
        raise ValueError("A8 target goals must be finite nonnegative integer counts")
    return values


def _time_weights(frame: pd.DataFrame, config: EquivariantCountConfig) -> np.ndarray:
    half_life = config.time_decay_half_life_years
    if half_life is None or config.year_column not in frame:
        return np.ones(len(frame), dtype=float)
    years = pd.to_numeric(frame[config.year_column], errors="raise").to_numpy(float)
    if not np.isfinite(years).all() or not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("A8 time-decay inputs must be finite and positive")
    return np.power(0.5, (float(years.max()) - years) / float(half_life))


__all__ = ["AsymmetricSidePoisson"]
