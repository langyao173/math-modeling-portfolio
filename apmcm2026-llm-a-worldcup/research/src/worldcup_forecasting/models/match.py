"""Match-level official OLS baseline for full-time total goals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ._utils import dataclass_config, regression_metrics, resolve_column
from .registry import register_model


@dataclass(frozen=True)
class MatchGoalOLSConfig:
    """Fixed Q2 design: stage, half-time score, and attendance."""

    target_column: str = "target_total_goals"
    year_column: str = "Year"
    stage_column: str = "canonical_stage"
    half_time_total_column: str = "half_time_total_goals"
    attendance_column: str = "Attendance"
    full_time_home_column: str = "Home Team Goals"
    full_time_away_column: str = "Away Team Goals"
    add_intercept: bool = True
    clip_nonnegative: bool = True


CANONICAL_STAGES: tuple[str, ...] = (
    "group_stage",
    "round_of_16",
    "quarter_final",
    "semi_final",
    "third_place",
    "final",
)


@register_model(
    "match_ols",
    "match_regression",
    ("stage_one_hot", "half_time", "temporal_2014_test", "reusable_prediction"),
)
class MatchGoalOLS:
    """OLS match-goal baseline with a train-fitted stage encoding."""

    def __init__(self, config: MatchGoalOLSConfig | object | None = None) -> None:
        self.config: MatchGoalOLSConfig = dataclass_config(MatchGoalOLSConfig, config)
        self.result_: Any | None = None
        self.stage_levels_: tuple[str, ...] = ()
        self.design_columns_: tuple[str, ...] = ()
        self.temporal_metrics_: dict[str, Any] | None = None
        self.attendance_median_: float | None = None

    def fit(self, frame: pd.DataFrame, y: np.ndarray | None = None) -> MatchGoalOLS:
        """Fit HC3 OLS after learning stage levels on the supplied training rows."""
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("MatchGoalOLS.fit requires a nonempty pandas DataFrame")
        predictors = self._extract_predictors(frame)
        self.attendance_median_ = float(predictors["attendance"].median(skipna=True))
        if not np.isfinite(self.attendance_median_):
            raise ValueError("Match OLS training attendance has no finite value")
        predictors["attendance"] = (
            predictors["attendance"].astype(float).fillna(self.attendance_median_)
        )
        target = self._extract_target(frame, y)
        complete = predictors.notna().all(axis=1) & np.isfinite(target)
        if int(complete.sum()) < 5:
            raise ValueError("Match OLS requires at least five complete rows")
        stages = predictors.loc[complete, "stage"].astype(str)
        self.stage_levels_ = tuple(sorted(stages.unique()))
        self.design_columns_ = ()
        design = self._encode(predictors.loc[complete])
        self.design_columns_ = tuple(design.columns)
        self.result_ = sm.OLS(target[complete], design.to_numpy(dtype=float)).fit(cov_type="HC3")
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict full-time total goals with the frozen training design."""
        self._require_fitted()
        predictors = self._extract_predictors(frame)
        assert self.attendance_median_ is not None
        predictors["attendance"] = (
            predictors["attendance"].astype(float).fillna(self.attendance_median_)
        )
        if predictors.isna().any().any():
            raise ValueError("Match prediction features contain missing/non-numeric values")
        design = self._encode(predictors)
        prediction = np.asarray(self.result_.predict(design.to_numpy(dtype=float)), dtype=float)
        if self.config.clip_nonnegative:
            prediction = np.maximum(prediction, 0.0)
        return prediction

    def predict_match(
        self,
        *,
        stage: str,
        half_time_total_goals: float,
        attendance: float,
    ) -> float:
        """Make a reusable one-match prediction without exposing design details."""
        row = pd.DataFrame(
            {
                self.config.stage_column: [stage],
                self.config.half_time_total_column: [half_time_total_goals],
                self.config.attendance_column: [attendance],
            }
        )
        return float(self.predict(row)[0])

    def fit_temporal(
        self,
        frame: pd.DataFrame,
        *,
        train_end_year: int = 2010,
        test_year: int = 2014,
    ) -> dict[str, Any]:
        """Fit through 2010 and report isolated train/2014 metrics."""
        year_name = resolve_column(frame, self.config.year_column, ("year",))
        years = pd.to_numeric(frame[year_name], errors="coerce")
        train = frame.loc[years <= train_end_year].copy()
        test = frame.loc[years == test_year].copy()
        if train.empty or test.empty:
            raise ValueError("Temporal evaluation requires nonempty train and test periods")
        self.fit(train)
        train_truth = self._extract_target(train, None)
        test_truth = self._extract_target(test, None)
        train_prediction = self.predict(train)
        test_prediction = self.predict(test)
        metrics: dict[str, Any] = {
            "train_end_year": int(train_end_year),
            "test_year": int(test_year),
            "train": regression_metrics(train_truth, train_prediction),
            "test": regression_metrics(test_truth, test_prediction),
        }
        self.temporal_metrics_ = metrics
        return metrics

    def coefficients(self) -> dict[str, float]:
        """Return named fitted coefficients."""
        self._require_fitted()
        return {
            name: float(value)
            for name, value in zip(self.design_columns_, self.result_.params, strict=True)
        }

    def _extract_predictors(self, frame: pd.DataFrame) -> pd.DataFrame:
        stage = resolve_column(frame, self.config.stage_column, ("stage",))
        half_time_total = resolve_column(
            frame,
            self.config.half_time_total_column,
            ("halftime_total_goals",),
        )
        attendance = resolve_column(frame, self.config.attendance_column, ("attendance",))
        predictors = pd.DataFrame(
            {
                "stage": frame[stage].astype("string"),
                "half_time_total_goals": pd.to_numeric(
                    frame[half_time_total], errors="coerce"
                ),
                "attendance": pd.to_numeric(frame[attendance], errors="coerce"),
            },
            index=frame.index,
        )
        if predictors["stage"].isna().any():
            raise ValueError("Match features require a canonical stage")
        stages = set(predictors["stage"].astype(str))
        unsupported = sorted(stages.difference(CANONICAL_STAGES))
        if unsupported:
            raise ValueError(f"Match features contain unseen canonical stage(s): {unsupported}")
        numeric = predictors[["half_time_total_goals", "attendance"]]
        finite = numeric.apply(lambda column: np.isfinite(column) | column.isna())
        if not finite.all().all():
            raise ValueError("Match prediction features must be finite")
        if (numeric < 0).any().any():
            raise ValueError("Match prediction features must be nonnegative")
        return predictors

    def _extract_target(self, frame: pd.DataFrame, y: np.ndarray | None) -> np.ndarray:
        if y is not None:
            target = np.asarray(y, dtype=float)
            if target.ndim != 1 or target.shape[0] != frame.shape[0]:
                raise ValueError("y must have one value per match row")
            return target
        try:
            target_name = resolve_column(
                frame, self.config.target_column, ("total_goals", "target_total_goals")
            )
            return pd.to_numeric(frame[target_name], errors="coerce").to_numpy(dtype=float)
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
            return (
                pd.to_numeric(frame[home], errors="coerce")
                + pd.to_numeric(frame[away], errors="coerce")
            ).to_numpy(dtype=float)

    def _encode(self, predictors: pd.DataFrame) -> pd.DataFrame:
        if self.stage_levels_:
            unseen = sorted(set(predictors["stage"].astype(str)).difference(self.stage_levels_))
            if unseen:
                raise ValueError(f"Match features contain unseen canonical stage(s): {unseen}")
        encoded = predictors.loc[:, ["half_time_total_goals", "attendance"]].astype(float)
        for level in self.stage_levels_[1:]:
            encoded[f"stage_{level}"] = (predictors["stage"].astype(str) == level).astype(float)
        if self.config.add_intercept:
            encoded = sm.add_constant(encoded, has_constant="add")
        if self.design_columns_:
            encoded = encoded.reindex(columns=self.design_columns_, fill_value=0.0)
        return encoded

    def _require_fitted(self) -> None:
        if self.result_ is None or not self.design_columns_ or self.attendance_median_ is None:
            raise RuntimeError("MatchGoalOLS must be fitted before prediction")
