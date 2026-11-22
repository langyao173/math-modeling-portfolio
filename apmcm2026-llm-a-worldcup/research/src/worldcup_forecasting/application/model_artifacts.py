"""Portable fitted-state snapshots used by official prediction certificates."""

from __future__ import annotations

import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import statsmodels

from worldcup_forecasting.models import (
    AggregateOLS,
    HalfTimeIncrementModel,
    JointPoissonModel,
    MatchGoalOLS,
    OutcomeLogit,
)
from worldcup_forecasting.reporting.model_state import (
    ModelStateArtifact,
    write_model_state,
)


def software_versions() -> dict[str, str]:
    """Return the model-runtime versions embedded in every certificate."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
    }


def write_aggregate_state(
    model: AggregateOLS,
    path: Path,
    *,
    random_seed: int,
    data_cutoff: str,
) -> ModelStateArtifact:
    """Persist aggregate-goal and attendance-regression fitted parameters."""
    if model.result_ is None or model.report_ is None or model.attendance_result_ is None:
        raise RuntimeError("Aggregate model and attendance submodel must be fitted")
    return write_model_state(
        path,
        model_type="aggregate_ols",
        feature_schema=list(model.config.feature_columns),
        fitted_state={
            "config": asdict(model.config),
            "design_columns": list(model.report_.design_columns),
            "goal_coefficients": model.result_.params,
            "goal_covariance": model.result_.cov_params(),
            "attendance_coefficients": model.attendance_result_.params,
        },
        training_contract={"data_cutoff": data_cutoff, "formula": "official_problem_1"},
        random_seed=random_seed,
    )


def write_match_ols_state(
    model: MatchGoalOLS,
    path: Path,
    *,
    random_seed: int,
    data_cutoff: str,
) -> ModelStateArtifact:
    """Persist the exact official Question 2 OLS design and coefficients."""
    if model.result_ is None or model.attendance_median_ is None:
        raise RuntimeError("MatchGoalOLS must be fitted")
    return write_model_state(
        path,
        model_type="match_goal_ols",
        feature_schema=["canonical_stage", "half_time_total_goals", "attendance"],
        fitted_state={
            "config": asdict(model.config),
            "design_columns": list(model.design_columns_),
            "stage_levels": list(model.stage_levels_),
            "attendance_median": model.attendance_median_,
            "coefficients": model.result_.params,
            "hc3_covariance": model.result_.cov_params(),
        },
        training_contract={"data_cutoff": data_cutoff, "formula": "official_problem_2"},
        random_seed=random_seed,
    )


def write_half_time_state(
    model: HalfTimeIncrementModel,
    path: Path,
    *,
    random_seed: int,
    data_cutoff: str,
) -> ModelStateArtifact:
    """Persist the enhanced second-half increment model and preprocessing state."""
    if model.pipeline_ is None:
        raise RuntimeError("HalfTimeIncrementModel must be fitted")
    preprocessor = model.pipeline_.named_steps["preprocessor"]
    numeric = preprocessor.named_transformers_["numeric"]
    stage = preprocessor.named_transformers_["stage"]
    poisson = model.pipeline_.named_steps["poisson"]
    return write_model_state(
        path,
        model_type="half_time_increment_poisson",
        feature_schema=["canonical_stage", "half_time_total_goals", "attendance"],
        fitted_state={
            "config": asdict(model.config),
            "stage_levels": list(model.stage_levels_),
            "numeric_mean": numeric.mean_,
            "numeric_scale": numeric.scale_,
            "stage_categories": [category.tolist() for category in stage.categories_],
            "poisson_coefficients": poisson.coef_,
            "poisson_intercept": poisson.intercept_,
            "numeric_location": model.numeric_location_,
            "numeric_training_scale": model.numeric_scale_,
            "numeric_min": model.numeric_min_,
            "numeric_max": model.numeric_max_,
        },
        training_contract={"data_cutoff": data_cutoff, "target": "second_half_increment"},
        random_seed=random_seed,
    )


def write_outcome_state(
    model: OutcomeLogit,
    path: Path,
    *,
    random_seed: int,
    data_cutoff: str,
) -> ModelStateArtifact:
    """Persist the standardized multinomial outcome baseline."""
    if model.scaler_ is None or model.classifier_ is None:
        raise RuntimeError("OutcomeLogit must be fitted")
    return write_model_state(
        path,
        model_type="outcome_logit",
        feature_schema=list(model.config.feature_columns),
        fitted_state={
            "config": asdict(model.config),
            "scaler_mean": model.scaler_.mean_,
            "scaler_scale": model.scaler_.scale_,
            "classes": model.classifier_.classes_,
            "coefficients": model.classifier_.coef_,
            "intercepts": model.classifier_.intercept_,
        },
        training_contract={"data_cutoff": data_cutoff, "swap_augmentation": True},
        random_seed=random_seed,
    )


def write_joint_poisson_state(
    model: JointPoissonModel,
    path: Path,
    *,
    random_seed: int,
    data_cutoff: str,
) -> ModelStateArtifact:
    """Persist a deployable static-history joint-goal model."""
    if model.pipeline_ is None:
        raise RuntimeError("JointPoissonModel must be fitted")
    pipeline = model.pipeline_
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    poisson = pipeline.named_steps["poisson"]
    return write_model_state(
        path,
        model_type="joint_poisson_static_history",
        feature_schema=list(model.feature_columns_),
        fitted_state={
            "config": asdict(model.config),
            "imputer_statistics": imputer.statistics_,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "poisson_coefficients": poisson.coef_,
            "poisson_intercept": poisson.intercept_,
            "dixon_coles_rho": model.rho_,
        },
        training_contract={"data_cutoff": data_cutoff, "dynamic_features": False},
        random_seed=random_seed,
    )
