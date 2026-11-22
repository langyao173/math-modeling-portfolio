"""Unified Dixon-Coles score kernel and mirrored joint Poisson model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ._utils import dataclass_config, resolve_column
from .outcome import MANDATORY_OUTCOME_FEATURES
from .registry import register_model

OPTIONAL_DYNAMIC_DIFF_FEATURES = (
    "pre_match_elo_diff",
    "attack_state_diff",
    "defense_state_diff",
)


def score_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float = 0.0,
    max_score: int | None = 12,
    *,
    min_retained_mass: float = 0.999,
    max_support: int = 100,
    adaptive: bool = True,
) -> np.ndarray:
    """Return a normalized joint score matrix with Dixon-Coles correction.

    ``max_score`` is the initial truncation. With ``adaptive=True`` the support is
    enlarged only when the corrected retained mass is below ``min_retained_mass``.
    """
    matrix, _ = _adaptive_score_matrix(
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        rho=rho,
        initial_max_score=max_score,
        min_retained_mass=min_retained_mass,
        max_support=max_support,
        adaptive=adaptive,
    )
    return matrix


def outcome_probabilities(matrix: np.ndarray) -> dict[str, float]:
    """Derive home/draw/away probabilities from one score distribution."""
    probability = np.asarray(matrix, dtype=float)
    if probability.ndim != 2 or probability.shape[0] != probability.shape[1]:
        raise ValueError("A score matrix must be a square two-dimensional array")
    if probability.size == 0 or not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("A score matrix must be nonempty, finite, and nonnegative")
    total = float(probability.sum())
    if total <= 0:
        raise ValueError("A score matrix must have positive mass")
    probability = probability / total
    result = {
        "home_win": float(np.tril(probability, k=-1).sum()),
        "draw": float(np.trace(probability)),
        "away_win": float(np.triu(probability, k=1).sum()),
    }
    normalization = sum(result.values())
    return {name: value / normalization for name, value in result.items()}


@dataclass(frozen=True)
class JointGoalBatch:
    """Single source of probability truth for a batch of matches."""

    probability: np.ndarray
    support_max: int
    retained_mass: np.ndarray
    expected_home_goals: np.ndarray
    expected_away_goals: np.ndarray
    model_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = np.asarray(self.probability, dtype=float)
        if values.ndim != 3 or values.shape[1] != values.shape[2]:
            raise ValueError("JointGoalBatch probability must have shape (n, support, support)")
        if self.support_max != values.shape[1] - 1:
            raise ValueError("support_max does not match probability shape")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Joint goal probabilities must be finite and nonnegative")
        if not np.allclose(values.sum(axis=(1, 2)), 1.0, atol=1e-10):
            raise ValueError("Every joint goal matrix must sum to one")
        count = values.shape[0]
        for name, vector in (
            ("retained_mass", self.retained_mass),
            ("expected_home_goals", self.expected_home_goals),
            ("expected_away_goals", self.expected_away_goals),
        ):
            array = np.asarray(vector, dtype=float)
            if array.shape != (count,) or not np.isfinite(array).all():
                raise ValueError(f"{name} must contain one finite value per match")

    def outcome_probabilities(self) -> np.ndarray:
        """Return [home_win, draw, away_win] probabilities for every match."""
        rows = []
        for matrix in self.probability:
            derived = outcome_probabilities(matrix)
            rows.append([derived["home_win"], derived["draw"], derived["away_win"]])
        return np.asarray(rows, dtype=float)

    def total_goal_distribution(self) -> np.ndarray:
        """Return the total-goal PMF on support 0..2*support_max."""
        totals = np.zeros((self.probability.shape[0], 2 * self.support_max + 1), dtype=float)
        for home in range(self.support_max + 1):
            for away in range(self.support_max + 1):
                totals[:, home + away] += self.probability[:, home, away]
        return totals


@dataclass(frozen=True)
class JointPoissonConfig:
    """Configuration for the mirrored team-perspective Poisson model."""

    mandatory_diff_features: tuple[str, ...] = MANDATORY_OUTCOME_FEATURES
    optional_diff_features: tuple[str, ...] = OPTIONAL_DYNAMIC_DIFF_FEATURES
    target_home_column: str = "target_home_goals"
    target_away_column: str = "target_away_goals"
    year_column: str = "Year"
    alpha: float = 0.5
    max_iter: int = 2_000
    tolerance: float = 1e-8
    time_decay_half_life_years: float | None = 12.0
    rho: float | None = None
    max_score: int = 12
    min_retained_mass: float = 0.999
    max_support: int = 100


@register_model(
    "joint_poisson",
    "goal_distribution",
    ("mirrored_team_perspective", "dixon_coles", "adaptive_support", "joint_kernel"),
)
class JointPoissonModel:
    """Penalized Poisson intensity model fit on mirrored team perspectives.

    For every match the training design contains ``x -> home goals`` and
    ``-x -> away goals``. The same fitted model therefore guarantees that swapping
    teams swaps the two intensities exactly.
    """

    def __init__(self, config: JointPoissonConfig | object | None = None) -> None:
        self.config: JointPoissonConfig = dataclass_config(JointPoissonConfig, config)
        self.pipeline_: Pipeline | None = None
        self.feature_columns_: tuple[str, ...] = ()
        self.rho_: float = 0.0

    def fit(self, frame: pd.DataFrame) -> JointPoissonModel:
        """Fit shared attack intensity coefficients on both team perspectives."""
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("JointPoissonModel.fit requires a nonempty DataFrame")
        self.feature_columns_ = self._select_features(frame)
        differences = self._feature_matrix(frame)
        home_goals, away_goals = self._targets(frame)
        if np.any(home_goals < 0) or np.any(away_goals < 0):
            raise ValueError("Goal counts must be nonnegative")
        design = np.vstack([differences, -differences])
        response = np.concatenate([home_goals, away_goals])
        weights = self._time_weights(frame)
        sample_weight = np.concatenate([weights, weights])
        self.pipeline_ = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
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
        self.pipeline_.fit(design, response, poisson__sample_weight=sample_weight)
        rates = self.predict_rates(frame)
        configured_rho = self.config.rho
        self.rho_ = (
            float(configured_rho)
            if configured_rho is not None
            else _estimate_dixon_coles_rho(home_goals, away_goals, rates[:, 0], rates[:, 1])
        )
        _validate_dixon_coles(self.rho_, rates[:, 0], rates[:, 1])
        return self

    def predict_rates(self, frame: pd.DataFrame) -> np.ndarray:
        """Return paired home/away intensities with exact swap equivariance."""
        self._require_fitted()
        differences = self._feature_matrix(frame)
        assert self.pipeline_ is not None
        home = np.asarray(self.pipeline_.predict(differences), dtype=float)
        away = np.asarray(self.pipeline_.predict(-differences), dtype=float)
        rates = np.column_stack([home, away])
        if not np.isfinite(rates).all() or np.any(rates <= 0):
            raise RuntimeError("Poisson model produced invalid intensities")
        return rates

    def predict_joint(self, frame: pd.DataFrame) -> JointGoalBatch:
        """Return adaptively truncated, normalized joint score distributions."""
        rates = self.predict_rates(frame)
        matrices: list[np.ndarray] = []
        retained: list[float] = []
        for home_rate, away_rate in rates:
            matrix, mass = _adaptive_score_matrix(
                lambda_home=float(home_rate),
                lambda_away=float(away_rate),
                rho=self.rho_,
                initial_max_score=int(self.config.max_score),
                min_retained_mass=float(self.config.min_retained_mass),
                max_support=int(self.config.max_support),
                adaptive=True,
            )
            matrices.append(matrix)
            retained.append(mass)
        support_max = max(matrix.shape[0] for matrix in matrices) - 1
        probability = np.zeros((len(matrices), support_max + 1, support_max + 1), dtype=float)
        for index, matrix in enumerate(matrices):
            width = matrix.shape[0]
            probability[index, :width, :width] = matrix
        score = np.arange(support_max + 1, dtype=float)
        expected_home = np.sum(probability * score[None, :, None], axis=(1, 2))
        expected_away = np.sum(probability * score[None, None, :], axis=(1, 2))
        return JointGoalBatch(
            probability=probability,
            support_max=support_max,
            retained_mass=np.asarray(retained, dtype=float),
            expected_home_goals=expected_home,
            expected_away_goals=expected_away,
            model_metadata={
                "name": "joint_poisson",
                "rho": self.rho_,
                "features": list(self.feature_columns_),
                "alpha": float(self.config.alpha),
                "min_retained_mass": float(self.config.min_retained_mass),
            },
        )

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Derive outcome probabilities from the unified joint distribution."""
        return self.predict_joint(frame).outcome_probabilities()

    def _select_features(self, frame: pd.DataFrame) -> tuple[str, ...]:
        mandatory = tuple(
            resolve_column(frame, name) for name in self.config.mandatory_diff_features
        )
        optional: list[str] = []
        for name in self.config.optional_diff_features:
            try:
                optional.append(resolve_column(frame, name))
            except ValueError:
                continue
        selected = (*mandatory, *optional)
        if not selected:
            raise ValueError("JointPoissonModel needs at least one signed difference feature")
        return selected

    def _feature_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.feature_columns_:
            raise RuntimeError("Feature columns are not frozen; fit the model first")
        resolved = [resolve_column(frame, name) for name in self.feature_columns_]
        return frame.loc[:, resolved].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    def _targets(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        home = resolve_column(
            frame,
            self.config.target_home_column,
            ("Home Team Goals", "home_goals"),
        )
        away = resolve_column(
            frame,
            self.config.target_away_column,
            ("Away Team Goals", "away_goals"),
        )
        home_values = pd.to_numeric(frame[home], errors="coerce").to_numpy(float)
        away_values = pd.to_numeric(frame[away], errors="coerce").to_numpy(float)
        if not np.isfinite(home_values).all() or not np.isfinite(away_values).all():
            raise ValueError("Training goal targets must be complete and finite")
        return home_values, away_values

    def _time_weights(self, frame: pd.DataFrame) -> np.ndarray:
        half_life = self.config.time_decay_half_life_years
        if half_life is None:
            return np.ones(frame.shape[0], dtype=float)
        if half_life <= 0:
            raise ValueError("time_decay_half_life_years must be positive or None")
        try:
            year = resolve_column(frame, self.config.year_column, ("year",))
        except ValueError:
            return np.ones(frame.shape[0], dtype=float)
        years = pd.to_numeric(frame[year], errors="coerce").to_numpy(float)
        if not np.isfinite(years).all():
            raise ValueError("Year values must be finite when time decay is enabled")
        return np.power(0.5, (float(years.max()) - years) / float(half_life))

    def _require_fitted(self) -> None:
        if self.pipeline_ is None or not self.feature_columns_:
            raise RuntimeError("JointPoissonModel must be fitted before prediction")


def _adaptive_score_matrix(
    *,
    lambda_home: float,
    lambda_away: float,
    rho: float,
    initial_max_score: int | None,
    min_retained_mass: float,
    max_support: int,
    adaptive: bool,
) -> tuple[np.ndarray, float]:
    rates = np.asarray([lambda_home, lambda_away], dtype=float)
    if not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError("Poisson intensities must be finite and nonnegative")
    if not np.isfinite(rho):
        raise ValueError("Dixon-Coles rho must be finite")
    if not 0 < min_retained_mass <= 1:
        raise ValueError("min_retained_mass must lie in (0, 1]")
    if max_support < 1:
        raise ValueError("max_support must be positive")
    if initial_max_score is None:
        peak = float(rates.max())
        support = max(8, int(np.ceil(peak + 6.0 * np.sqrt(max(peak, 1.0)))))
    else:
        support = int(initial_max_score)
    if support < 1:
        raise ValueError("max_score must be positive")
    support = min(support, max_support)
    while True:
        scores = np.arange(support + 1)
        corrected = np.outer(poisson.pmf(scores, lambda_home), poisson.pmf(scores, lambda_away))
        _apply_dixon_coles(corrected, lambda_home, lambda_away, rho)
        if not np.isfinite(corrected).all() or np.any(corrected < 0):
            raise ValueError("Dixon-Coles correction produced invalid probability mass")
        retained_mass = float(corrected.sum())
        if retained_mass <= 0:
            raise ValueError("Truncated score distribution has zero mass")
        if not adaptive or retained_mass >= min_retained_mass:
            return corrected / retained_mass, retained_mass
        if support >= max_support:
            raise RuntimeError(
                f"Score support {max_support} retained only {retained_mass:.8f} mass, "
                f"below required {min_retained_mass:.8f}"
            )
        support = min(max_support, max(support + 1, int(np.ceil(1.25 * support))))


def _apply_dixon_coles(
    matrix: np.ndarray,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> None:
    tau = np.asarray(
        [
            1.0 - lambda_home * lambda_away * rho,
            1.0 + lambda_home * rho,
            1.0 + lambda_away * rho,
            1.0 - rho,
        ],
        dtype=float,
    )
    if np.any(tau < 0):
        raise ValueError("Dixon-Coles rho is outside the nonnegative low-score correction region")
    matrix[0, 0] *= tau[0]
    matrix[0, 1] *= tau[1]
    matrix[1, 0] *= tau[2]
    matrix[1, 1] *= tau[3]


def _estimate_dixon_coles_rho(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
) -> float:
    def objective(candidate: float) -> float:
        factors = np.ones(home_goals.shape[0], dtype=float)
        factors[(home_goals == 0) & (away_goals == 0)] = (
            1.0
            - lambda_home[(home_goals == 0) & (away_goals == 0)]
            * lambda_away[(home_goals == 0) & (away_goals == 0)]
            * candidate
        )
        factors[(home_goals == 0) & (away_goals == 1)] = (
            1.0 + lambda_home[(home_goals == 0) & (away_goals == 1)] * candidate
        )
        factors[(home_goals == 1) & (away_goals == 0)] = (
            1.0 + lambda_away[(home_goals == 1) & (away_goals == 0)] * candidate
        )
        factors[(home_goals == 1) & (away_goals == 1)] = 1.0 - candidate
        if np.any(factors <= 0) or not np.isfinite(factors).all():
            return float("inf")
        return float(-np.log(factors).sum())

    result = minimize_scalar(objective, bounds=(-0.2, 0.2), method="bounded")
    return float(result.x) if result.success and np.isfinite(result.fun) else 0.0


def _validate_dixon_coles(
    rho: float,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
) -> None:
    tau = np.column_stack(
        [
            1.0 - lambda_home * lambda_away * rho,
            1.0 + lambda_home * rho,
            1.0 + lambda_away * rho,
            np.full_like(lambda_home, 1.0 - rho),
        ]
    )
    if np.any(tau < 0) or not np.isfinite(tau).all():
        raise ValueError("Configured/estimated Dixon-Coles rho is invalid for fitted rates")
