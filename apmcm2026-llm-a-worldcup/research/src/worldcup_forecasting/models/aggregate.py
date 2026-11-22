"""Tournament-level OLS baseline with robust inference and frozen 2018 audit logic."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ._utils import dataclass_config, regression_metrics, resolve_column
from .registry import register_model


@dataclass(frozen=True)
class AggregateOLSConfig:
    """Configuration for the official tournament aggregate regression."""

    target_column: str = "GoalsScored"
    feature_columns: tuple[str, ...] = (
        "QualifiedTeams",
        "MatchesPlayed",
        "Attendance",
    )
    year_column: str = "Year"
    attendance_column: str = "Attendance"
    add_intercept: bool = True


@dataclass(frozen=True)
class AggregateOLSReport:
    """Auditable statistics from a fitted aggregate baseline."""

    coefficients: dict[str, float]
    p_values_hc3: dict[str, float]
    standard_errors_hc3: dict[str, float]
    r_squared: float
    adjusted_r_squared: float
    vif: dict[str, float]
    loocv: dict[str, float]
    n_observations: int
    design_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return asdict(self)


@register_model(
    "aggregate_ols",
    "aggregate_regression",
    ("hc3", "vif", "loocv", "pre_2018_attendance_forecast"),
)
class AggregateOLS:
    """Statsmodels OLS baseline with HC3 inference and leave-one-edition-out error."""

    def __init__(self, config: AggregateOLSConfig | object | None = None) -> None:
        self.config: AggregateOLSConfig = dataclass_config(AggregateOLSConfig, config)
        self.result_: Any | None = None
        self.robust_result_: Any | None = None
        self.report_: AggregateOLSReport | None = None
        self.loocv_predictions_: np.ndarray | None = None
        self.attendance_result_: Any | None = None
        self._training_frame: pd.DataFrame | None = None
        self._resolved_features: tuple[str, ...] = ()
        self._resolved_target: str | None = None

    def fit(self, frame: pd.DataFrame) -> AggregateOLS:
        """Fit the fixed aggregate formula on complete tournament rows."""
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("AggregateOLS.fit requires a nonempty pandas DataFrame")
        target = resolve_column(frame, self.config.target_column, ("total_goals",))
        resolved = tuple(resolve_column(frame, name) for name in self.config.feature_columns)
        table = frame.loc[:, [target, *resolved]].apply(pd.to_numeric, errors="coerce")
        complete = table.dropna()
        if complete.shape[0] < 3:
            raise ValueError("Aggregate OLS requires at least three complete tournaments")
        x = self._add_constant(complete.loc[:, resolved])
        y = complete[target].to_numpy(dtype=float)
        result = sm.OLS(y, x.to_numpy(dtype=float)).fit()
        robust = result.get_robustcov_results(cov_type="HC3")
        self.result_ = result
        self.robust_result_ = robust
        self._training_frame = frame.copy()
        self._resolved_features = resolved
        self._resolved_target = target
        names = tuple(x.columns)
        self.loocv_predictions_ = self._loocv(x.to_numpy(dtype=float), y)
        loocv = regression_metrics(y, self.loocv_predictions_)
        self.report_ = AggregateOLSReport(
            coefficients=_named_values(names, robust.params),
            p_values_hc3=_named_values(names, robust.pvalues),
            standard_errors_hc3=_named_values(names, robust.bse),
            r_squared=float(result.rsquared),
            adjusted_r_squared=float(result.rsquared_adj),
            vif=self._vif(x),
            loocv=loocv,
            n_observations=int(complete.shape[0]),
            design_columns=names,
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict total tournament goals using the fitted fixed design."""
        self._require_fitted()
        columns = tuple(resolve_column(frame, name) for name in self.config.feature_columns)
        values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any():
            raise ValueError("Aggregate prediction features contain missing/non-numeric values")
        values.columns = self._resolved_features
        design = self._add_constant(values)
        return np.asarray(self.result_.predict(design.to_numpy(dtype=float)), dtype=float)

    def statistics(self) -> dict[str, Any]:
        """Return coefficients, robust inference, fit, VIF, and LOOCV diagnostics."""
        self._require_fitted()
        assert self.report_ is not None
        return self.report_.to_dict()

    def forecast_attendance(
        self,
        frame: pd.DataFrame | int | None = None,
        target_year: int = 2018,
        target_matches_played: int = 64,
    ) -> float:
        """Forecast total attendance from pre-cutoff attendance per match.

        Passing an integer as the first argument is accepted as a convenient year-only
        call after ``fit``. Dividing by the historical match count prevents format
        expansion from being mistaken for a change in per-match attendance; the
        forecast is converted back to a tournament total using the known target format.
        """
        if isinstance(frame, (int, np.integer)):
            target_year = int(frame)
            frame = None
        source = frame if frame is not None else self._training_frame
        if source is None:
            raise RuntimeError("Provide tournament data or fit AggregateOLS first")
        year_name = resolve_column(source, self.config.year_column)
        attendance_name = resolve_column(source, self.config.attendance_column)
        matches_name = resolve_column(source, "MatchesPlayed", ("matches_played",))
        values = source.loc[:, [year_name, attendance_name, matches_name]].apply(
            pd.to_numeric, errors="coerce"
        )
        values = values.loc[
            (values[year_name] < target_year)
            & (values[attendance_name] > 0)
            & (values[matches_name] > 0)
        ].dropna()
        if values.shape[0] < 3:
            raise ValueError("Log-linear attendance forecast needs at least three prior editions")
        design = sm.add_constant(values[[year_name]], has_constant="add")
        response = np.log(
            values[attendance_name].to_numpy(dtype=float)
            / values[matches_name].to_numpy(dtype=float)
        )
        self.attendance_result_ = sm.OLS(response, design.to_numpy(dtype=float)).fit()
        future = np.array([[1.0, float(target_year)]], dtype=float)
        log_per_match_forecast = float(self.attendance_result_.predict(future)[0])
        if target_matches_played <= 0:
            raise ValueError("target_matches_played must be positive")
        return float(np.exp(log_per_match_forecast) * target_matches_played)

    def forecast_total_goals(
        self,
        *,
        year: int = 2018,
        qualified_teams: int = 32,
        matches_played: int = 64,
        attendance: float | None = None,
    ) -> float:
        """Produce the official one-row tournament forecast."""
        self._require_fitted()
        if attendance is None:
            attendance = self.forecast_attendance(
                target_year=year,
                target_matches_played=matches_played,
            )
        values = {
            "Year": float(year),
            "QualifiedTeams": float(qualified_teams),
            "MatchesPlayed": float(matches_played),
            "Attendance": float(attendance),
        }
        missing = [name for name in self.config.feature_columns if name not in values]
        if missing:
            raise ValueError(
                "forecast_total_goals supports the official feature formula only; "
                f"unresolved features: {missing}"
            )
        row = pd.DataFrame([{name: values[name] for name in self.config.feature_columns}])
        return float(self.predict(row)[0])

    def _add_constant(self, values: pd.DataFrame) -> pd.DataFrame:
        numeric = values.astype(float)
        if not self.config.add_intercept:
            return numeric
        return sm.add_constant(numeric, has_constant="add")

    @staticmethod
    def _loocv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        predictions = np.empty_like(y, dtype=float)
        for index in range(y.size):
            train = np.arange(y.size) != index
            fitted = sm.OLS(y[train], x[train]).fit()
            predictions[index] = float(fitted.predict(x[index : index + 1])[0])
        return predictions

    @staticmethod
    def _vif(design: pd.DataFrame) -> dict[str, float]:
        start = 1 if design.columns[0] == "const" else 0
        values = design.to_numpy(dtype=float)
        diagnostics: dict[str, float] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for index in range(start, values.shape[1]):
                try:
                    score = float(variance_inflation_factor(values, index))
                except (ValueError, np.linalg.LinAlgError):
                    score = float("inf")
                diagnostics[str(design.columns[index])] = score
        return diagnostics

    def _require_fitted(self) -> None:
        if self.result_ is None or self.report_ is None:
            raise RuntimeError("AggregateOLS must be fitted before this operation")


def _named_values(names: tuple[str, ...], values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, values, strict=True)}
