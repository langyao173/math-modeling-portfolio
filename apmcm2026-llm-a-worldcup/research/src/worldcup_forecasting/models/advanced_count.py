"""Symmetry-equivariant penalized count model for World Cup scorelines.

The model is deliberately described as a penalized Poisson GLM.  It is not a
latent Bayesian model: sparse attacker and opponent-defender effects are ordinary
L2-regularized one-hot coefficients whose unseen-category value is the population
intercept.  A single team-perspective design is used for both sides of every match,
which makes the fitted intensities equivariant to exchanging the two teams.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ._advanced_count_utils import (
    DEFAULT_OPTIONAL_CONTEXT,
    DEFAULT_OPTIONAL_EVEN,
    DEFAULT_OPTIONAL_ODD,
    FrozenCountSchema,
    build_component_table,
    build_perspective_frame,
    build_team_effect_table,
    freeze_count_schema,
    joint_batch_from_rates_impl,
    unknown_team_rows,
)
from ._selective_poisson import (
    DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
    DiagonalPenaltyPoissonRegressor,
    SelectivePenaltyProfile,
    build_selective_penalty_profile,
)
from ._utils import dataclass_config, resolve_column
from .count import JointGoalBatch
from .registry import register_model


@dataclass(frozen=True)
class EquivariantCountConfig:
    """Configuration for :class:`EquivariantCountModel`.

    Required odd features are interpreted from the recorded home-team
    perspective.  Optional fields are used only when present, permitting the same
    estimator to operate on the official four-feature table and richer research
    feature tables without silently dropping an explicitly required field.
    """

    odd_numeric_features: tuple[str, ...] = ()
    even_numeric_features: tuple[str, ...] = ()
    even_categorical_features: tuple[str, ...] = ()
    optional_odd_numeric_features: tuple[str, ...] = DEFAULT_OPTIONAL_ODD
    optional_even_numeric_features: tuple[str, ...] = DEFAULT_OPTIONAL_EVEN
    optional_even_categorical_features: tuple[str, ...] = DEFAULT_OPTIONAL_CONTEXT
    paired_numeric_features: tuple[tuple[str, str], ...] = ()
    home_team_column: str = "home_team_initials"
    away_team_column: str = "away_team_initials"
    target_home_column: str = "target_home_goals"
    target_away_column: str = "target_away_goals"
    # Descriptive protocol aliases retained as first-class configuration fields.
    home_goals_column: str | None = None
    away_goals_column: str | None = None
    year_column: str = "Year"
    stage_column: str | None = None
    host_side_column: str | None = None
    common_pace_features: tuple[str, ...] = ()
    alpha: float = 1.0
    # Absolute L2 penalty for attacker/opponent-defender one-hot blocks only.
    # ``None`` means the same penalty as every other non-intercept coefficient;
    # zero removes team-block shrinkage under explicit sum-to-zero constraints.
    team_shrinkage: float | None = None
    max_iter: int = 2_000
    tolerance: float = 1e-8
    time_decay_half_life_years: float | None = 16.0
    shared_score: bool = False
    shared_fraction: float | None = None
    max_shared_fraction: float = 0.75
    max_score: int = 12
    min_retained_mass: float = 0.999
    max_support: int = 100


@register_model(
    "s2_equivariant_poisson_glm",
    "goal_distribution",
    (
        "team_exchange_equivariant",
        "even_odd_decomposition",
        "attacker_defender_shrinkage",
        "optional_shared_score",
        "joint_kernel",
    ),
)
class EquivariantCountModel:
    r"""Fit one regularized team-perspective Poisson intensity model.

    For match :math:`m` and perspective :math:`s\in\{-1,+1\}`,

    .. math::
       \log \lambda_{m,s}=\mu+p(z_m)+s\,b^\top x_m
       +a_{\mathrm{attacker}(m,s)}+d_{\mathrm{defender}(m,s)}.

    The context ``z`` is even and the difference vector ``x`` is odd under team
    exchange.  Swapping the teams and negating ``x`` therefore exchanges the two
    rates exactly.  Categorical attacker/defender effects are separate and L2
    regularized by the same GLM objective.
    """

    def __init__(self, config: EquivariantCountConfig | object | None = None) -> None:
        self.config: EquivariantCountConfig = dataclass_config(EquivariantCountConfig, config)
        self.pipeline_: Pipeline | None = None
        self.odd_columns_: tuple[str, ...] = ()
        self.even_columns_: tuple[str, ...] = ()
        self.context_columns_: tuple[str, ...] = ()
        self.paired_columns_: tuple[tuple[str, str], ...] = ()
        self.home_team_column_: str | None = None
        self.away_team_column_: str | None = None
        self.shared_fraction_: float = 0.0
        self.team_categories_: frozenset[str] = frozenset()
        self.team_effects_: pd.DataFrame = pd.DataFrame()
        self.host_side_column_: str | None = None
        self.common_pace_columns_: tuple[str, ...] = ()
        self.stage_column_: str | None = None
        self.year_column_: str | None = None
        self.schema_: FrozenCountSchema | None = None
        self.penalty_profile_: SelectivePenaltyProfile | None = None
        self.penalty_optimizer_: str = ""

    def fit(self, frame: pd.DataFrame) -> EquivariantCountModel:
        """Fit the GLM on mirrored team perspectives with temporal weights."""
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("EquivariantCountModel.fit requires a nonempty DataFrame")
        self._validate_config()
        self._freeze_schema(frame)
        home_goals, away_goals = self._targets(frame)
        perspective = self._perspectives(frame)
        response = np.concatenate([home_goals, away_goals])
        weights = self._time_weights(frame)
        sample_weight = np.concatenate([weights, weights])
        numeric = [
            *[f"even::{name}" for name in self.even_columns_],
            *[f"odd::{name}" for name in self.odd_columns_],
            *[f"pair_even::{index}" for index in range(len(self.paired_columns_))],
            *[f"pair_odd::{index}" for index in range(len(self.paired_columns_))],
            *(["odd::__host_signed__"] if self.host_side_column_ is not None else []),
        ]
        categorical = [
            *[f"context::{name}" for name in self.context_columns_],
            "attacker_team",
            "opponent_defender_team",
        ]
        transformer = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical,
                ),
            ],
            remainder="drop",
        )
        transformer.fit(perspective, response)
        encoded = np.asarray(transformer.transform(perspective), dtype=float)
        self.penalty_profile_ = build_selective_penalty_profile(
            transformer.get_feature_names_out(),
            base_penalty=self._penalty_alpha(),
            team_penalty=self._team_penalty_alpha(),
        )
        penalties = np.asarray(self.penalty_profile_.effective_penalties, dtype=float)
        if np.all(penalties == self._penalty_alpha()):
            poisson = PoissonRegressor(
                alpha=self._penalty_alpha(),
                max_iter=int(self.config.max_iter),
                tol=float(self.config.tolerance),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                poisson.fit(encoded, response, sample_weight=sample_weight)
            if any(issubclass(item.category, ConvergenceWarning) for item in caught):
                raise RuntimeError("Uniform-penalty Poisson optimization did not converge")
            self.penalty_optimizer_ = "sklearn_uniform_poisson_regressor"
        else:
            poisson = DiagonalPenaltyPoissonRegressor(
                self.penalty_profile_.effective_penalties,
                self.penalty_profile_.coefficient_groups,
                max_iter=int(self.config.max_iter),
                tolerance=float(self.config.tolerance),
                wall_clock_limit_seconds=DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
            ).fit(encoded, response, sample_weight=sample_weight)
            self.penalty_optimizer_ = (
                "direct_diagonal_l2_with_full_rank_zero_sum_team_contrasts"
            )
        self.pipeline_ = Pipeline([("design", transformer), ("poisson", poisson)])
        assert self.schema_ is not None
        self.team_effects_ = build_team_effect_table(
            self.pipeline_, frame, self.schema_, self._team_penalty_alpha()
        )
        fitted_rates = self.predict_rates(frame)
        self.shared_fraction_ = self._fit_shared_fraction(
            home_goals,
            away_goals,
            fitted_rates[:, 0],
            fitted_rates[:, 1],
        )
        return self

    def predict_rates(self, frame: pd.DataFrame) -> np.ndarray:
        """Return home/away marginal goal intensities in input-team order."""
        self._require_fitted()
        perspective = self._perspectives(frame)
        assert self.pipeline_ is not None
        predicted = np.asarray(self.pipeline_.predict(perspective), dtype=float)
        count = len(frame)
        rates = np.column_stack([predicted[:count], predicted[count:]])
        if not np.isfinite(rates).all() or np.any(rates <= 0):
            raise RuntimeError("Penalized Poisson GLM produced invalid intensities")
        return rates

    def predict_joint(self, frame: pd.DataFrame) -> JointGoalBatch:
        """Return coherent score matrices and an auditable model description."""
        rates = self.predict_rates(frame)
        batch = joint_batch_from_rates(
            rates,
            shared_fraction=self.shared_fraction_,
            max_score=self.config.max_score,
            min_retained_mass=self.config.min_retained_mass,
            max_support=self.config.max_support,
        )
        unknown = self._unknown_teams(frame)
        return JointGoalBatch(
            probability=batch.probability,
            support_max=batch.support_max,
            retained_mass=batch.retained_mass,
            expected_home_goals=batch.expected_home_goals,
            expected_away_goals=batch.expected_away_goals,
            model_metadata={
                "name": "s2_equivariant_poisson_glm",
                "estimator": "l2_penalized_poisson_glm",
                "is_bayesian": False,
                "symmetry_group": "S2_team_exchange",
                "even_numeric_features": list(self.even_columns_),
                "odd_numeric_features": list(self.odd_columns_),
                "symmetric_context_features": list(self.context_columns_),
                "attacker_effects": True,
                "opponent_defender_effects": True,
                "unknown_team_policy": "zero_deviation_from_population_intercept",
                "unknown_team_rows": int(unknown.sum()),
                "shared_score_enabled": bool(self.config.shared_score),
                "shared_score_fraction": float(self.shared_fraction_),
                "shared_score_rate_mean": batch.model_metadata["shared_score_rate_mean"],
                "penalty_alpha": self._penalty_alpha(),
                "team_shrinkage_parameter": self._team_penalty_alpha(),
                "selective_optimizer_wall_clock_limit_seconds": getattr(
                    self.pipeline_.named_steps["poisson"],
                    "wall_clock_limit_seconds_",
                    None,
                ),
                "selective_optimizer_wall_clock_seconds": getattr(
                    self.pipeline_.named_steps["poisson"],
                    "wall_clock_seconds_",
                    None,
                ),
                "selective_optimizer_hard_wall_clock_enforced": getattr(
                    self.pipeline_.named_steps["poisson"],
                    "hard_wall_clock_enforced_",
                    False,
                ),
                "penalty_parameterization": getattr(
                    self,
                    "penalty_optimizer_",
                    "sklearn_uniform_poisson_regressor",
                ),
                "non_team_context_penalty": self._penalty_alpha(),
                "team_effect_penalty": self._team_penalty_alpha(),
                "team_effect_identifiability": (
                    "sum_to_zero_blocks_plus_data_penalty_nullspace_projection"
                    if self._team_penalty_alpha() == 0.0
                    else "l2_identified_full_one_hot"
                ),
            },
        )

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Derive home-win/draw/away-win probabilities from the joint kernel."""
        return self.predict_joint(frame).outcome_probabilities()

    def predict_components(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return additive log-rate components and cold-start diagnostics.

        Component columns are an accounting decomposition of the penalized GLM,
        not independently identified causal effects.
        """
        self._require_fitted()
        assert self.pipeline_ is not None
        assert self.schema_ is not None
        return build_component_table(
            self.pipeline_,
            self._perspectives(frame),
            frame,
            self.schema_,
            self.predict_rates(frame),
        )

    def team_effect_table(self) -> pd.DataFrame:
        """Return a defensive copy of the fitted identity-effect audit table."""
        self._require_fitted()
        return self.team_effects_.copy()

    def coefficient_penalty_table(self) -> pd.DataFrame:
        """Return the exhaustive encoded-coefficient penalty assignment."""
        self._require_fitted()
        profile = getattr(self, "penalty_profile_", None)
        if profile is None:
            assert self.pipeline_ is not None
            design = self.pipeline_.named_steps["design"]
            profile = build_selective_penalty_profile(
                design.get_feature_names_out(),
                base_penalty=self._penalty_alpha(),
                team_penalty=self._team_penalty_alpha(),
            )
            self.penalty_profile_ = profile
        return profile.to_frame()

    def _validate_config(self) -> None:
        if not np.isfinite(self.config.alpha) or self.config.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if self.config.team_shrinkage is not None and (
            not np.isfinite(self.config.team_shrinkage) or self.config.team_shrinkage < 0
        ):
            raise ValueError("team_shrinkage must be finite and nonnegative")
        if self.config.max_iter <= 0 or self.config.tolerance <= 0:
            raise ValueError("max_iter and tolerance must be positive")
        if not 0 <= self.config.max_shared_fraction < 1:
            raise ValueError("max_shared_fraction must lie in [0, 1)")
        if self.config.shared_fraction is not None and not (
            0 <= self.config.shared_fraction <= self.config.max_shared_fraction
        ):
            raise ValueError("shared_fraction exceeds the permitted nonnegative range")
        if self.config.max_score < 1 or self.config.max_support < self.config.max_score:
            raise ValueError("score support bounds are inconsistent")
        if not 0 < self.config.min_retained_mass <= 1:
            raise ValueError("min_retained_mass must lie in (0, 1]")

    def _freeze_schema(self, frame: pd.DataFrame) -> None:
        schema = freeze_count_schema(frame, self.config)
        self.schema_ = schema
        self.home_team_column_ = schema.home_team
        self.away_team_column_ = schema.away_team
        self.odd_columns_ = schema.odd
        self.even_columns_ = schema.even
        self.context_columns_ = schema.contexts
        self.paired_columns_ = schema.pairs
        self.host_side_column_ = schema.host_side
        self.common_pace_columns_ = schema.common_pace
        self.stage_column_ = schema.stage
        self.year_column_ = schema.year
        self.team_categories_ = schema.team_categories

    def _perspectives(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.schema_ is None:
            raise RuntimeError("Feature schema is not frozen; fit the model first")
        return build_perspective_frame(frame, self.schema_)

    def _targets(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        home_preferred = self.config.home_goals_column or self.config.target_home_column
        away_preferred = self.config.away_goals_column or self.config.target_away_column
        home_name = resolve_column(frame, home_preferred, ("Home Team Goals", "home_goals"))
        away_name = resolve_column(frame, away_preferred, ("Away Team Goals", "away_goals"))
        home = pd.to_numeric(frame[home_name], errors="coerce").to_numpy(float)
        away = pd.to_numeric(frame[away_name], errors="coerce").to_numpy(float)
        for values in (home, away):
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Goal targets must be complete, finite, and nonnegative")
            if not np.allclose(values, np.round(values)):
                raise ValueError("Poisson goal targets must be integer counts")
        return home, away

    def _time_weights(self, frame: pd.DataFrame) -> np.ndarray:
        half_life = self.config.time_decay_half_life_years
        if half_life is None:
            return np.ones(len(frame), dtype=float)
        if not np.isfinite(half_life) or half_life <= 0:
            raise ValueError("time_decay_half_life_years must be positive or None")
        try:
            year_name = resolve_column(frame, self.config.year_column, ("year",))
        except ValueError:
            return np.ones(len(frame), dtype=float)
        year = pd.to_numeric(frame[year_name], errors="coerce").to_numpy(float)
        if not np.isfinite(year).all():
            raise ValueError("Year values must be finite when time decay is enabled")
        return np.power(0.5, (float(year.max()) - year) / float(half_life))

    def _fit_shared_fraction(
        self,
        home_goal: np.ndarray,
        away_goal: np.ndarray,
        home_rate: np.ndarray,
        away_rate: np.ndarray,
    ) -> float:
        if not self.config.shared_score:
            return 0.0
        if self.config.shared_fraction is not None:
            return float(self.config.shared_fraction)
        residual_covariance = float(np.mean((home_goal - home_rate) * (away_goal - away_rate)))
        scale = float(np.mean(np.minimum(home_rate, away_rate)))
        if not np.isfinite(residual_covariance) or scale <= 0:
            return 0.0
        return float(np.clip(residual_covariance / scale, 0.0, self.config.max_shared_fraction))

    def _penalty_alpha(self) -> float:
        return float(self.config.alpha)

    def _team_penalty_alpha(self) -> float:
        if self.config.team_shrinkage is None:
            return float(self.config.alpha)
        return float(self.config.team_shrinkage)

    def _unknown_teams(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.schema_ is not None
        return unknown_team_rows(frame, self.schema_)

    def _require_fitted(self) -> None:
        if self.pipeline_ is None or self.home_team_column_ is None:
            raise RuntimeError("EquivariantCountModel must be fitted before prediction")


def joint_batch_from_rates(
    rates: np.ndarray,
    *,
    shared_fraction: float = 0.0,
    max_score: int = 12,
    min_retained_mass: float = 0.999,
    max_support: int = 100,
) -> JointGoalBatch:
    """Build the coherent joint PMF after an equivariant rate correction."""
    if max_score < 1 or max_support < max_score:
        raise ValueError("score support bounds are inconsistent")
    if not 0 < min_retained_mass <= 1:
        raise ValueError("min_retained_mass must lie in (0, 1]")
    return joint_batch_from_rates_impl(
        rates,
        shared_fraction=float(shared_fraction),
        max_score=int(max_score),
        min_retained_mass=float(min_retained_mass),
        max_support=int(max_support),
    )
HierarchicalPaceConfig = EquivariantCountConfig
HierarchicalPaceScoreConfig = EquivariantCountConfig
HierarchicalPaceScoreModel = EquivariantCountModel

__all__ = [
    "EquivariantCountConfig",
    "EquivariantCountModel",
    "HierarchicalPaceScoreConfig",
    "HierarchicalPaceConfig",
    "HierarchicalPaceScoreModel",
    "joint_batch_from_rates",
]
