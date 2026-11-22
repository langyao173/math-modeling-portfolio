"""Nonnegative second-half increment model and monotonicity invariant."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ._utils import dataclass_config, resolve_column
from .match import CANONICAL_STAGES
from .registry import register_model


def enforce_half_time_monotonicity(
    predicted_full_time: np.ndarray | float,
    observed_half_time: np.ndarray | float,
) -> np.ndarray:
    """Project full-time forecasts onto the element-wise half-time lower bound."""
    prediction = np.asarray(predicted_full_time, dtype=float)
    half_time = np.asarray(observed_half_time, dtype=float)
    try:
        prediction, half_time = np.broadcast_arrays(prediction, half_time)
    except ValueError as error:
        raise ValueError("Full-time and half-time values must be broadcast-compatible") from error
    if not np.isfinite(prediction).all() or not np.isfinite(half_time).all():
        raise ValueError("Full-time and half-time values must be finite")
    if np.any(half_time < 0):
        raise ValueError("Observed half-time goals cannot be negative")
    return np.maximum(prediction, half_time)


@dataclass(frozen=True)
class HalfTimeIncrementConfig:
    """Configuration for second-half total-goal Poisson regression."""

    target_column: str = "target_total_goals"
    half_time_total_column: str = "half_time_total_goals"
    attendance_column: str = "Attendance"
    stage_column: str = "canonical_stage"
    full_time_home_column: str = "Home Team Goals"
    full_time_away_column: str = "Away Team Goals"
    alpha: float = 0.5
    max_iter: int = 2_000
    tolerance: float = 1e-8


@register_model(
    "half_time_increment",
    "half_time_increment",
    ("nonnegative_increment", "full_time_monotonicity", "stage_one_hot"),
)
class HalfTimeIncrementModel:
    """Predict nonnegative second-half goals, then add the observed half-time total."""

    def __init__(self, config: HalfTimeIncrementConfig | object | None = None) -> None:
        self.config: HalfTimeIncrementConfig = dataclass_config(HalfTimeIncrementConfig, config)
        self.pipeline_: Pipeline | None = None
        self.stage_levels_: tuple[str, ...] = ()
        self.numeric_location_: dict[str, float] = {}
        self.numeric_scale_: dict[str, float] = {}
        self.numeric_min_: dict[str, float] = {}
        self.numeric_max_: dict[str, float] = {}

    def fit(
        self,
        frame: pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
    ) -> HalfTimeIncrementModel:
        """Fit Poisson regression to the observed second-half increment."""
        design = self._design(frame)
        self.stage_levels_ = tuple(sorted(design["stage"].unique()))
        numeric_columns = ["half_time_total_goals", "attendance"]
        self.numeric_location_ = {
            column: float(design[column].mean()) for column in numeric_columns
        }
        self.numeric_scale_ = {
            column: max(float(design[column].std(ddof=0)), 1.0) for column in numeric_columns
        }
        self.numeric_min_ = {column: float(design[column].min()) for column in numeric_columns}
        self.numeric_max_ = {column: float(design[column].max()) for column in numeric_columns}
        full_time = self._target(frame, y)
        half_time = self._half_time_total(frame)
        increment = full_time - half_time
        if not np.isfinite(increment).all():
            raise ValueError("Half-time training targets must be finite")
        if np.any(increment < -1e-9):
            raise ValueError("Full-time goals cannot be below observed half-time goals")
        increment = np.maximum(increment, 0.0)
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    StandardScaler(),
                    ["half_time_total_goals", "attendance"],
                ),
                (
                    "stage",
                    OneHotEncoder(handle_unknown="error", sparse_output=False),
                    ["stage"],
                ),
            ],
            remainder="drop",
        )
        self.pipeline_ = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "poisson",
                    PoissonRegressor(
                        alpha=float(self.config.alpha),
                        max_iter=int(self.config.max_iter),
                        tol=float(self.config.tolerance),
                    ),
                ),
            ]
        )
        self.pipeline_.fit(design, increment)
        return self

    def predict_increment(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a finite nonnegative second-half total-goal forecast."""
        self._require_fitted()
        assert self.pipeline_ is not None
        prediction = np.asarray(self.pipeline_.predict(self._design(frame)), dtype=float)
        if not np.isfinite(prediction).all():
            raise RuntimeError("Half-time increment model produced non-finite predictions")
        return np.maximum(prediction, 0.0)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return monotone full-time total-goal forecasts."""
        half_time = self._half_time_total(frame)
        full_time = half_time + self.predict_increment(frame)
        return enforce_half_time_monotonicity(full_time, half_time)

    def predict_match(
        self,
        *,
        stage: str,
        half_time_home_goals: float,
        half_time_away_goals: float,
        attendance: float,
    ) -> dict[str, float]:
        """Return a one-match increment/full-time prediction record."""
        row = pd.DataFrame(
            {
                self.config.stage_column: [stage],
                self.config.half_time_total_column: [
                    half_time_home_goals + half_time_away_goals
                ],
                self.config.attendance_column: [attendance],
            }
        )
        increment = float(self.predict_increment(row)[0])
        return {
            "half_time_total_goals": float(half_time_home_goals + half_time_away_goals),
            "second_half_increment": increment,
            "full_time_total_goals": float(self.predict(row)[0]),
        }

    def out_of_distribution_components(self, frame: pd.DataFrame) -> dict[str, float]:
        """Measure numeric distance/range shift against the fitted training support."""
        self._require_fitted()
        design = self._design(frame)
        z_squared = np.zeros(len(design), dtype=float)
        range_excess = np.zeros(len(design), dtype=float)
        for column in ("half_time_total_goals", "attendance"):
            scale = self.numeric_scale_[column]
            values = design[column].to_numpy(float)
            z_squared += ((values - self.numeric_location_[column]) / scale) ** 2
            below = np.maximum(self.numeric_min_[column] - values, 0.0) / scale
            above = np.maximum(values - self.numeric_max_[column], 0.0) / scale
            range_excess += below + above
        return {
            "standardized_distance": float(np.sqrt(z_squared).mean()),
            "range_excess": float(range_excess.mean()),
            "unseen_stage": 0.0,
        }

    def _design(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("HalfTimeIncrementModel requires a nonempty DataFrame")
        half_time_total = resolve_column(
            frame,
            self.config.half_time_total_column,
            ("halftime_total_goals",),
        )
        attendance = resolve_column(frame, self.config.attendance_column, ("attendance",))
        stage = resolve_column(frame, self.config.stage_column, ("stage",))
        design = pd.DataFrame(
            {
                "half_time_total_goals": pd.to_numeric(
                    frame[half_time_total], errors="coerce"
                ),
                "attendance": pd.to_numeric(frame[attendance], errors="coerce"),
                "stage": frame[stage].astype("string"),
            },
            index=frame.index,
        )
        numeric = design[["half_time_total_goals", "attendance"]]
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(float)).all():
            raise ValueError("Half-time model features must be complete and finite numeric values")
        if (numeric < 0).any().any():
            raise ValueError("Half-time model features must be nonnegative")
        if design["stage"].isna().any():
            raise ValueError("Half-time model requires a canonical stage")
        unsupported = sorted(set(design["stage"].astype(str)).difference(CANONICAL_STAGES))
        if unsupported:
            raise ValueError(
                f"Half-time model contains unseen canonical stage(s): {unsupported}"
            )
        if self.stage_levels_:
            unseen = sorted(set(design["stage"].astype(str)).difference(self.stage_levels_))
            if unseen:
                raise ValueError(
                    f"Half-time model contains unseen canonical stage(s): {unseen}"
                )
        return design

    def _half_time_total(self, frame: pd.DataFrame) -> np.ndarray:
        design = self._design(frame)
        return design["half_time_total_goals"].to_numpy(float)

    def _target(
        self,
        frame: pd.DataFrame,
        y: np.ndarray | pd.Series | None,
    ) -> np.ndarray:
        if y is not None:
            values = np.asarray(y, dtype=float)
            if values.ndim != 1 or values.shape[0] != frame.shape[0]:
                raise ValueError("y must have one full-time target per match")
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Full-time targets must be finite and nonnegative")
            return values
        try:
            target = resolve_column(
                frame, self.config.target_column, ("total_goals", "target_total_goals")
            )
            values = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
        except ValueError:
            home = resolve_column(
                frame,
                self.config.full_time_home_column,
                ("target_home_goals", "home_goals"),
            )
            away = resolve_column(
                frame,
                self.config.full_time_away_column,
                ("target_away_goals", "away_goals"),
            )
            values = (
                pd.to_numeric(frame[home], errors="coerce")
                + pd.to_numeric(frame[away], errors="coerce")
            ).to_numpy(float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Full-time targets must be finite and nonnegative")
        return values

    def _require_fitted(self) -> None:
        if self.pipeline_ is None:
            raise RuntimeError("HalfTimeIncrementModel must be fitted before prediction")
