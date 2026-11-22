"""Bounded residual learning that preserves exact team-exchange equivariance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ._utils import dataclass_config


@dataclass(frozen=True, slots=True)
class EquivariantResidualConfig:
    """Frozen controls for the structural-model residual correction layer."""

    enabled: bool = True
    alpha_even: float = 4.0
    alpha_odd: float = 4.0
    max_abs_log_correction: float = 0.30
    residual_pseudocount: float = 0.50
    missing_policy: str = "error"
    maximum_fit_prediction_year: int = 2010


class EquivariantResidualCorrector:
    r"""Correct paired rates without breaking their exchange symmetry.

    If ``c(e)`` is learned from exchange-even inputs and ``g(d)`` from signed
    exchange-odd inputs, the deployed contrast is explicitly antisymmetrized:

    ``a(d) = (g(d) - g(-d)) / 2``.

    Home and away log-rate corrections are then ``c(e)+a(d)`` and
    ``c(e)-a(d)``.  Swapping rates and negating odd inputs therefore exchanges
    the corrected rates exactly.  The layer is fitted only from auditable,
    chronological structural-model OOF predictions.
    """

    def __init__(self, config: EquivariantResidualConfig | object | None = None) -> None:
        self.config: EquivariantResidualConfig = dataclass_config(
            EquivariantResidualConfig, config
        )
        self._validate_config()
        self.even_model_: Pipeline | None = None
        self.odd_model_: Pipeline | None = None
        self.even_features_: tuple[str, ...] = ()
        self.odd_features_: tuple[str, ...] = ()
        self.even_imputation_values_: dict[str, float] = {}
        self.is_fitted_: bool = False
        self.metadata_: dict[str, Any] = {
            "enabled": bool(self.config.enabled),
            "status": "unfitted" if self.config.enabled else "disabled_identity",
            "deterministic": True,
            "max_abs_log_correction": float(self.config.max_abs_log_correction),
            "missing_policy": self.config.missing_policy,
        }

    def fit_oof(
        self,
        frame: pd.DataFrame,
        base_rates: np.ndarray,
        observed_goals: np.ndarray,
        provenance: pd.DataFrame,
        even_features: Sequence[str],
        odd_features: Sequence[str],
    ) -> EquivariantResidualCorrector:
        """Fit common and contrast residuals from structural OOF predictions."""
        _validate_frame(frame)
        rates = _pair_matrix(base_rates, "base_rates", len(frame), positive=True)
        goals = _pair_matrix(observed_goals, "observed_goals", len(frame), positive=False)
        if np.any(goals < 0) or not np.allclose(goals, np.rint(goals), atol=1e-12):
            raise ValueError("observed_goals must contain nonnegative integer counts")
        self._validate_provenance(provenance, len(frame))
        self.even_features_, self.odd_features_ = _validate_feature_names(
            even_features, odd_features
        )
        if not self.config.enabled:
            self.is_fitted_ = True
            self.metadata_["status"] = "disabled_identity"
            return self

        even = self._feature_matrix(frame, self.even_features_, parity="even", fitting=True)
        odd = self._feature_matrix(frame, self.odd_features_, parity="odd", fitting=True)
        pseudocount = float(self.config.residual_pseudocount)
        log_residual = np.log((goals + pseudocount) / (rates + pseudocount))
        common_target = 0.5 * (log_residual[:, 0] + log_residual[:, 1])
        contrast_target = 0.5 * (log_residual[:, 0] - log_residual[:, 1])

        self.even_model_ = _ridge_pipeline(float(self.config.alpha_even))
        self.even_model_.fit(even, common_target)
        # Mirrored augmentation makes the learned contrast balanced around the
        # exchange origin before deployment-time antisymmetrization is applied.
        self.odd_model_ = _ridge_pipeline(float(self.config.alpha_odd))
        self.odd_model_.fit(
            np.vstack([odd, -odd]),
            np.concatenate([contrast_target, -contrast_target]),
        )
        self.is_fitted_ = True
        self.metadata_ = {
            "enabled": True,
            "status": "fitted",
            "fit_scope": "chronological_structural_oof_only",
            "training_rows": len(frame),
            "training_identity_hash": _provenance_hash(provenance),
            "maximum_fit_prediction_year": int(
                pd.to_numeric(provenance["prediction_year"]).max()
            ),
            "even_features": list(self.even_features_),
            "odd_features": list(self.odd_features_),
            "alpha_even": float(self.config.alpha_even),
            "alpha_odd": float(self.config.alpha_odd),
            "max_abs_log_correction": float(self.config.max_abs_log_correction),
            "residual_pseudocount": pseudocount,
            "missing_policy": self.config.missing_policy,
            "even_imputation_values": dict(self.even_imputation_values_),
            "odd_missing_imputation": (
                "symmetry_neutral_zero"
                if self.config.missing_policy == "impute"
                else "not_applicable"
            ),
            "contrast_enforcement": "(g(d)-g(-d))/2",
            "application": "(common+contrast, common-contrast)",
            "deterministic": True,
        }
        return self

    def transform_rates(self, frame: pd.DataFrame, base_rates: np.ndarray) -> np.ndarray:
        """Apply bounded log corrections to paired positive structural rates."""
        _validate_frame(frame)
        rates = _pair_matrix(base_rates, "base_rates", len(frame), positive=True)
        if not self.config.enabled:
            return rates.copy()
        if not self.is_fitted_ or self.even_model_ is None or self.odd_model_ is None:
            raise RuntimeError("EquivariantResidualCorrector must be fitted before use")
        even = self._feature_matrix(frame, self.even_features_, parity="even", fitting=False)
        odd = self._feature_matrix(frame, self.odd_features_, parity="odd", fitting=False)
        common = np.asarray(self.even_model_.predict(even), dtype=float)
        forward = np.asarray(self.odd_model_.predict(odd), dtype=float)
        reverse = np.asarray(self.odd_model_.predict(-odd), dtype=float)
        contrast = 0.5 * (forward - reverse)
        corrections = np.column_stack([common + contrast, common - contrast])
        bound = float(self.config.max_abs_log_correction)
        corrections = np.clip(corrections, -bound, bound)
        corrected = rates * np.exp(corrections)
        if not np.isfinite(corrected).all() or np.any(corrected <= 0):
            raise RuntimeError("Residual correction produced invalid goal rates")
        return corrected

    def _feature_matrix(
        self,
        frame: pd.DataFrame,
        features: tuple[str, ...],
        *,
        parity: str,
        fitting: bool,
    ) -> np.ndarray:
        missing_columns = [name for name in features if name not in frame]
        if missing_columns:
            raise ValueError(f"Missing required {parity} feature columns: {missing_columns}")
        numeric = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
        for name in features:
            invalid_text = frame[name].notna() & numeric[name].isna()
            if invalid_text.any():
                raise ValueError(f"Feature {name!r} contains nonnumeric values")
        values = numeric.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"{parity} features contain infinite values")
        if self.config.missing_policy == "error":
            if np.isnan(values).any():
                raise ValueError(f"{parity} features contain missing/non-finite values")
            return values
        if parity == "odd":
            return np.nan_to_num(values, nan=0.0)
        if fitting:
            for position, name in enumerate(features):
                column = values[:, position]
                if np.isnan(column).all():
                    raise ValueError(f"Even feature {name!r} is entirely missing")
                self.even_imputation_values_[name] = float(np.nanmedian(column))
        for position, name in enumerate(features):
            fill = self.even_imputation_values_.get(name)
            if fill is None:
                raise RuntimeError(f"No fitted imputation value for even feature {name!r}")
            values[np.isnan(values[:, position]), position] = fill
        return values

    def _validate_config(self) -> None:
        if not isinstance(self.config.enabled, bool):
            raise TypeError("enabled must be boolean")
        for name, value in (
            ("alpha_even", self.config.alpha_even),
            ("alpha_odd", self.config.alpha_odd),
            ("max_abs_log_correction", self.config.max_abs_log_correction),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.config.residual_pseudocount) or (
            self.config.residual_pseudocount <= 0
        ):
            raise ValueError("residual_pseudocount must be finite and positive")
        if self.config.missing_policy not in {"error", "impute"}:
            raise ValueError("missing_policy must be 'error' or 'impute'")
        if self.config.maximum_fit_prediction_year <= 0:
            raise ValueError("maximum_fit_prediction_year must be positive")

    def _validate_provenance(self, provenance: pd.DataFrame, rows: int) -> None:
        if not isinstance(provenance, pd.DataFrame) or len(provenance) != rows:
            raise ValueError("OOF provenance must have one row per structural prediction")
        required = {
            "match_id",
            "is_oof",
            "prediction_role",
            "train_end_year",
            "prediction_year",
        }
        missing = sorted(required - set(provenance.columns))
        if missing:
            raise ValueError(f"OOF provenance is missing required columns: {missing}")
        oof = provenance["is_oof"]
        valid_boolean = oof.map(lambda value: isinstance(value, (bool, np.bool_)))
        if not valid_boolean.all() or not oof.astype(bool).all():
            raise ValueError("Residual learning requires genuine out-of-fold (OOF) rows")
        if not provenance["prediction_role"].eq("structural_oof").all():
            raise ValueError("Residual learning accepts structural OOF predictions only")
        train_end = pd.to_numeric(provenance["train_end_year"], errors="coerce")
        prediction = pd.to_numeric(provenance["prediction_year"], errors="coerce")
        if train_end.isna().any() or prediction.isna().any():
            raise ValueError("OOF chronology years must be finite")
        if not train_end.lt(prediction).all():
            raise ValueError("train_end_year must be strictly earlier than prediction_year")
        if prediction.max() > self.config.maximum_fit_prediction_year:
            raise ValueError(
                "Residual OOF fit exceeds the frozen maximum prediction year "
                f"{self.config.maximum_fit_prediction_year}"
            )


def _ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, solver="svd", fit_intercept=True)),
        ]
    )


def _validate_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Residual correction requires a nonempty feature DataFrame")


def _pair_matrix(
    values: np.ndarray,
    name: str,
    rows: int,
    *,
    positive: bool,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (rows, 2):
        raise ValueError(f"{name} must have shape ({rows}, 2), with two team columns")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must contain strictly positive rates")
    return array


def _validate_feature_names(
    even_features: Sequence[str], odd_features: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    even = tuple(even_features)
    odd = tuple(odd_features)
    if not even or not odd:
        raise ValueError("At least one even and one odd residual feature are required")
    if any(not isinstance(name, str) or not name for name in (*even, *odd)):
        raise ValueError("Residual feature names must be nonempty strings")
    if len(set(even)) != len(even) or len(set(odd)) != len(odd):
        raise ValueError("Residual feature names must be unique")
    overlap = sorted(set(even) & set(odd))
    if overlap:
        raise ValueError(f"Features cannot be both exchange-even and exchange-odd: {overlap}")
    return even, odd


def _provenance_hash(provenance: pd.DataFrame) -> str:
    records = provenance.loc[
        :, ["match_id", "train_end_year", "prediction_year", "prediction_role"]
    ].to_dict("records")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["EquivariantResidualConfig", "EquivariantResidualCorrector"]
