"""Identical-fold evaluation of the three mandatory match baselines."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from worldcup_forecasting.models.count import JointGoalBatch, JointPoissonConfig, JointPoissonModel
from worldcup_forecasting.models.outcome import OutcomeLogit
from worldcup_forecasting.validation.joint_metrics import joint_forecast_metrics
from worldcup_forecasting.validation.metrics import classification_metrics

_MAXIMUM_SELECTION_YEAR = 2010
_DYNAMIC_FEATURES = ("pre_match_elo_diff", "attack_state_diff", "defense_state_diff")


@dataclass(frozen=True)
class BaselineEvaluationConfig:
    """Frozen baseline fold and estimator controls."""

    outer_test_years: tuple[int, ...] = (1986, 1990, 1994, 1998, 2002, 2006, 2010)
    seed: int = 17
    poisson_alpha: float = 0.5
    time_decay_half_life_years: float = 12.0
    max_score: int = 12
    min_retained_mass: float = 0.999
    max_support: int = 100


@dataclass(frozen=True)
class BaselineSuiteResult:
    """Outcome and joint evidence for exact preregistered baselines."""

    outcome_fold_metrics: pd.DataFrame
    joint_fold_metrics: pd.DataFrame
    prediction_ledger: pd.DataFrame
    joint_grid: pd.DataFrame
    metadata: dict[str, Any]


def run_baseline_suite(
    frame: pd.DataFrame,
    *,
    config: BaselineEvaluationConfig | None = None,
    runtime_clock: Callable[[], float] = time.perf_counter,
) -> BaselineSuiteResult:
    """Fit all baselines on the same chronological outer tournament folds."""
    resolved = config or BaselineEvaluationConfig()
    _validate_inputs(frame, resolved, runtime_clock)
    match_id_column = "MatchID" if "MatchID" in frame else "match_id"
    outcome_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    for test_year in sorted(resolved.outer_test_years):
        train = frame.loc[frame["Year"].lt(test_year)].copy()
        test = frame.loc[frame["Year"].eq(test_year)].copy()
        if train.empty or test.empty:
            raise ValueError(f"Invalid baseline expanding-origin fold for {test_year}")
        train_end = int(train["Year"].max())
        fit_index_hash = _fit_hash(train[match_id_column].tolist(), train_end)
        fold_id = f"outer_{test_year}"
        labels = test["target_outcome"].to_numpy()
        observed = test[["target_home_goals", "target_away_goals"]].to_numpy(int)

        start = runtime_clock()
        logit_probability = OutcomeLogit().fit(train).predict_proba(test)
        logit_runtime = max(0.0, float(runtime_clock() - start))
        _append_outcome_evidence(
            outcome_rows,
            prediction_rows,
            model="official_outcome_logit",
            probability=logit_probability,
            labels=labels,
            test=test,
            match_id_column=match_id_column,
            fold_id=fold_id,
            test_year=test_year,
            train_end=train_end,
            fit_index_hash=fit_index_hash,
            runtime_seconds=logit_runtime,
            seed=resolved.seed,
        )

        model_specs = (
            ("independent_mirrored_poisson", _joint_model(resolved, rho=0.0, dynamic=False)),
            ("legacy_mirrored_dixon_coles", _joint_model(resolved, rho=None, dynamic=True)),
        )
        for model_name, model in model_specs:
            start = runtime_clock()
            batch = model.fit(train).predict_joint(test)
            runtime_seconds = max(0.0, float(runtime_clock() - start))
            probability = batch.outcome_probabilities()
            _append_outcome_evidence(
                outcome_rows,
                prediction_rows,
                model=model_name,
                probability=probability,
                labels=labels,
                test=test,
                match_id_column=match_id_column,
                fold_id=fold_id,
                test_year=test_year,
                train_end=train_end,
                fit_index_hash=fit_index_hash,
                runtime_seconds=runtime_seconds,
                seed=resolved.seed,
            )
            joint_metrics = joint_forecast_metrics(
                observed, batch.probability, pit_seed=resolved.seed + test_year
            )
            joint_rows.append(
                {
                    "outer_fold_id": fold_id,
                    "test_year": test_year,
                    "model": model_name,
                    "runtime_seconds": runtime_seconds,
                    **joint_metrics,
                }
            )
            _append_joint_grid(
                grid_rows,
                model=model_name,
                fold_id=fold_id,
                match_ids=test[match_id_column].tolist(),
                batch=batch,
            )
    outcomes = pd.DataFrame(outcome_rows).sort_values(["test_year", "model"]).reset_index(drop=True)
    joints = pd.DataFrame(joint_rows).sort_values(["test_year", "model"]).reset_index(drop=True)
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["test_year", "model", "match_id"]
    ).reset_index(drop=True)
    grids = pd.DataFrame(grid_rows).sort_values(
        ["outer_fold_id", "model", "match_id", "home_goals", "away_goals"]
    ).reset_index(drop=True)
    metadata = {
        "models": [
            "official_outcome_logit",
            "independent_mirrored_poisson",
            "legacy_mirrored_dixon_coles",
        ],
        "outer_fold_count": len(resolved.outer_test_years),
        "minimum_prediction_year": min(resolved.outer_test_years),
        "maximum_prediction_year": max(resolved.outer_test_years),
        "maximum_selection_year": _MAXIMUM_SELECTION_YEAR,
        "identical_folds": True,
        "audit_year_used": False,
    }
    return BaselineSuiteResult(outcomes, joints, predictions, grids, metadata)


def _append_outcome_evidence(
    metric_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    model: str,
    probability: np.ndarray,
    labels: np.ndarray,
    test: pd.DataFrame,
    match_id_column: str,
    fold_id: str,
    test_year: int,
    train_end: int,
    fit_index_hash: str,
    runtime_seconds: float,
    seed: int,
) -> None:
    metrics, _ = classification_metrics(labels, probability)
    metric_rows.append(
        {
            "outer_fold_id": fold_id,
            "test_year": test_year,
            "model": model,
            "runtime_seconds": runtime_seconds,
            **metrics,
        }
    )
    for position, row in enumerate(test.itertuples(index=False)):
        prediction_rows.append(
            {
                "outer_fold_id": fold_id,
                "test_year": test_year,
                "prediction_time": pd.Timestamp(row.Datetime).isoformat(),
                "train_end_year": train_end,
                "model": model,
                "match_id": getattr(row, match_id_column),
                "fit_index_hash": fit_index_hash,
                "seed": seed,
                "home_win": float(probability[position, 0]),
                "draw": float(probability[position, 1]),
                "away_win": float(probability[position, 2]),
                "target_outcome": row.target_outcome,
                "observed_home_goals": int(row.target_home_goals),
                "observed_away_goals": int(row.target_away_goals),
            }
        )


def _append_joint_grid(
    rows: list[dict[str, Any]],
    *,
    model: str,
    fold_id: str,
    match_ids: list[Any],
    batch: JointGoalBatch,
) -> None:
    for row_index, match_id in enumerate(match_ids):
        for home in range(batch.support_max + 1):
            for away in range(batch.support_max + 1):
                rows.append(
                    {
                        "outer_fold_id": fold_id,
                        "model": model,
                        "match_id": match_id,
                        "home_goals": home,
                        "away_goals": away,
                        "probability": float(batch.probability[row_index, home, away]),
                    }
                )


def _joint_model(
    config: BaselineEvaluationConfig,
    *,
    rho: float | None,
    dynamic: bool,
) -> JointPoissonModel:
    return JointPoissonModel(
        JointPoissonConfig(
            optional_diff_features=_DYNAMIC_FEATURES if dynamic else (),
            alpha=float(config.poisson_alpha),
            time_decay_half_life_years=float(config.time_decay_half_life_years),
            rho=rho,
            max_score=int(config.max_score),
            min_retained_mass=float(config.min_retained_mass),
            max_support=int(config.max_support),
        )
    )


def _fit_hash(match_ids: list[Any], train_end: int) -> str:
    payload = json.dumps(
        {"match_ids": [str(value) for value in match_ids], "train_end_year": int(train_end)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_inputs(
    frame: pd.DataFrame,
    config: BaselineEvaluationConfig,
    runtime_clock: Callable[[], float],
) -> None:
    required = {
        "Year",
        "Datetime",
        "target_outcome",
        "target_home_goals",
        "target_away_goals",
        "historical_appearances_diff",
        "historical_best_finish_diff",
        "historical_mean_goals_for_diff",
        "historical_mean_goals_conceded_diff",
    }
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Baseline suite requires a nonempty DataFrame")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Baseline frame is missing columns: {missing}")
    if "MatchID" not in frame and "match_id" not in frame:
        raise ValueError("Baseline frame requires MatchID or match_id")
    if not callable(runtime_clock):
        raise TypeError("runtime_clock must be callable")
    if not config.outer_test_years:
        raise ValueError("At least one outer selection year is required")
    if any(int(year) > _MAXIMUM_SELECTION_YEAR for year in config.outer_test_years):
        raise ValueError("2014/audit data are forbidden during baseline selection")
    if len(set(config.outer_test_years)) != len(config.outer_test_years):
        raise ValueError("Outer selection years must be unique")
    if config.poisson_alpha < 0 or config.time_decay_half_life_years <= 0:
        raise ValueError("Baseline Poisson controls are invalid")


__all__ = ["BaselineEvaluationConfig", "BaselineSuiteResult", "run_baseline_suite"]
