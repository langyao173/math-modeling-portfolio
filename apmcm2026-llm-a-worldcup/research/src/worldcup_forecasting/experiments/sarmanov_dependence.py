"""Nested exploratory Sarmanov dependence benchmark.

The benchmark keeps the two finite-support score marginals fixed and replaces
their dependence by a one-parameter Sarmanov construction.  A standardized
parameter is fitted only to selected inner OOF predictions inside each outer
fold.  This makes the experiment a chronological contradiction test, not a
promotion-eligible extension of the frozen confirmatory model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from worldcup_forecasting.experiments.evidence import reconstruct_joint_batch

_OUTCOME_COLUMNS = ("home_win", "draw", "away_win")
_THETA_LIMIT = 1.0 - 1e-9
_PROBABILITY_FLOOR = 1e-15


@dataclass(frozen=True)
class SarmanovExplorationResult:
    """Outer OOF predictions and inner-only fitted dependence states."""

    oof_ledger: pd.DataFrame
    joint_grid: pd.DataFrame
    selection_table: pd.DataFrame
    states: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


def sarmanov_joint_from_marginals(
    home_marginal: np.ndarray,
    away_marginal: np.ndarray,
    *,
    theta: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Construct a marginal-preserving joint PMF with standardized dependence.

    ``theta`` lies in ``[-1, 1]``.  Its sign controls the sign of dependence;
    its magnitude is scaled to the exact feasible interval for these two
    marginals.  The bounded kernel ``exp(-score)`` follows the Sarmanov form,
    while finite-support centering makes marginal preservation exact for the
    represented predictive distribution.
    """
    home = _validate_marginal(home_marginal, "home_marginal")
    away = _validate_marginal(away_marginal, "away_marginal")
    value = float(theta)
    if not np.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("theta must be finite and lie in [-1, 1]")

    home_score = np.arange(len(home), dtype=float)
    away_score = np.arange(len(away), dtype=float)
    home_kernel = np.exp(-home_score)
    away_kernel = np.exp(-away_score)
    home_kernel -= float(home @ home_kernel)
    away_kernel -= float(away @ away_kernel)
    product = np.outer(home_kernel, away_kernel)
    base = np.outer(home, away)
    lower, upper = _feasible_omega_interval(product, base)
    omega = value * upper if value >= 0.0 else (-value) * lower
    joint = base * (1.0 + omega * product)
    if not np.isfinite(joint).all() or float(joint.min()) < -1e-13:
        raise RuntimeError("Sarmanov construction violated nonnegativity")
    joint = np.maximum(joint, 0.0)
    mass = float(joint.sum())
    if not np.isclose(mass, 1.0, atol=1e-12):
        raise RuntimeError("Centered Sarmanov construction lost normalization")
    joint /= mass
    if not np.allclose(joint.sum(axis=1), home, atol=2e-12):
        raise RuntimeError("Sarmanov construction changed the home marginal")
    if not np.allclose(joint.sum(axis=0), away, atol=2e-12):
        raise RuntimeError("Sarmanov construction changed the away marginal")
    covariance = _score_covariance(joint)
    return joint, {
        "theta": value,
        "omega": float(omega),
        "omega_lower": float(lower),
        "omega_upper": float(upper),
        "score_covariance": covariance,
    }


def transform_joint_batch_sarmanov(
    base_joint: np.ndarray,
    *,
    theta: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Replace dependence for a batch while preserving every row's marginals."""
    probability = np.asarray(base_joint, dtype=float)
    if probability.ndim != 3 or probability.shape[1] != probability.shape[2]:
        raise ValueError("base_joint must have shape (n, support, support)")
    if len(probability) == 0 or not np.isfinite(probability).all():
        raise ValueError("base_joint must be nonempty and finite")
    if np.any(probability < 0) or not np.allclose(
        probability.sum(axis=(1, 2)), 1.0, atol=1e-8
    ):
        raise ValueError("Every base joint matrix must be a normalized PMF")
    matrices: list[np.ndarray] = []
    diagnostics: list[dict[str, float]] = []
    for matrix in probability:
        joint, row = sarmanov_joint_from_marginals(
            matrix.sum(axis=1), matrix.sum(axis=0), theta=theta
        )
        matrices.append(joint)
        diagnostics.append(row)
    return np.stack(matrices), pd.DataFrame(diagnostics)


def fit_sarmanov_theta(
    base_joint: np.ndarray,
    observed_goals: np.ndarray,
) -> dict[str, Any]:
    """Fit standardized dependence by inner-OOF scoreline log likelihood."""
    probability = np.asarray(base_joint, dtype=float)
    observed = np.asarray(observed_goals)
    if probability.ndim != 3 or probability.shape[1] != probability.shape[2]:
        raise ValueError("base_joint must have shape (n, support, support)")
    if observed.shape != (len(probability), 2):
        raise ValueError("observed_goals must have shape (n, 2) aligned to base_joint")
    numeric = observed.astype(float)
    if not np.isfinite(numeric).all() or np.any(numeric < 0) or not np.equal(
        numeric, np.floor(numeric)
    ).all():
        raise ValueError("observed_goals must be finite nonnegative integers")
    observed = numeric.astype(int)

    def objective(theta: float) -> float:
        transformed, _ = transform_joint_batch_sarmanov(probability, theta=float(theta))
        support = transformed.shape[1]
        inside = (observed[:, 0] < support) & (observed[:, 1] < support)
        assigned = np.full(len(observed), _PROBABILITY_FLOOR, dtype=float)
        row = np.flatnonzero(inside)
        assigned[row] = transformed[row, observed[row, 0], observed[row, 1]]
        return float(-np.log(np.maximum(assigned, _PROBABILITY_FLOOR)).sum())

    optimized = minimize_scalar(
        objective,
        method="bounded",
        bounds=(-_THETA_LIMIT, _THETA_LIMIT),
        options={"xatol": 1e-9, "maxiter": 500},
    )
    candidates = [-_THETA_LIMIT, 0.0, _THETA_LIMIT]
    if optimized.success and np.isfinite(optimized.x):
        candidates.append(float(optimized.x))
    scored = [(float(value), objective(float(value))) for value in candidates]
    theta, loss = min(scored, key=lambda item: (item[1], abs(item[0])))
    independence_loss = objective(0.0)
    transformed, diagnostics = transform_joint_batch_sarmanov(probability, theta=theta)
    del transformed
    return {
        "schema_version": 1,
        "theta": theta,
        "inner_scoreline_nll_sum": loss,
        "inner_scoreline_nll_mean": loss / len(observed),
        "independence_nll_sum": independence_loss,
        "independence_nll_mean": independence_loss / len(observed),
        "inner_nll_gain_vs_independence": (independence_loss - loss) / len(observed),
        "negative_dependence_selected": bool(theta < 0.0),
        "optimizer_success": bool(optimized.success),
        "optimizer_message": str(optimized.message),
        "inner_row_count": int(len(observed)),
        "omega_min": float(diagnostics["omega"].min()),
        "omega_max": float(diagnostics["omega"].max()),
        "covariance_min": float(diagnostics["score_covariance"].min()),
        "covariance_max": float(diagnostics["score_covariance"].max()),
    }


def apply_nested_sarmanov_exploration(
    selected_inner_oof_ledger: pd.DataFrame,
    selected_inner_joint_grid: pd.DataFrame,
    outer_oof_ledger: pd.DataFrame,
    outer_joint_grid: pd.DataFrame,
) -> SarmanovExplorationResult:
    """Fit and apply the exploratory dependence family in every outer fold."""
    inner = selected_inner_oof_ledger.copy()
    outer = outer_oof_ledger.copy()
    _validate_ledgers(inner, outer)
    parent_column = (
        "selection_outer_fold_id" if "selection_outer_fold_id" in inner else "outer_fold_id"
    )
    ledger_parts: list[pd.DataFrame] = []
    grid_parts: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    states: dict[str, dict[str, Any]] = {}
    for outer_fold_id, outer_block in outer.groupby("outer_fold_id", sort=True):
        outer_block = outer_block.sort_values("match_id").copy()
        inner_block = inner.loc[inner[parent_column].eq(outer_fold_id)].sort_values(
            ["prediction_year", "match_id"]
        )
        if inner_block.empty:
            raise ValueError(f"No selected inner OOF rows exist for outer fold {outer_fold_id!r}")
        outer_year = _single_year(outer_block)
        inner_max = int(inner_block["prediction_year"].max())
        if inner_max >= outer_year:
            raise ValueError("Sarmanov inner fits must strictly precede their outer fold")
        inner_joint = reconstruct_joint_batch(
            selected_inner_joint_grid,
            inner_block[["outer_fold_id", "match_id"]],
        )
        state = fit_sarmanov_theta(
            inner_joint,
            inner_block[["observed_home_goals", "observed_away_goals"]].to_numpy(int),
        )
        outer_joint = reconstruct_joint_batch(
            outer_joint_grid,
            outer_block[["outer_fold_id", "match_id"]],
        )
        transformed, diagnostics = transform_joint_batch_sarmanov(
            outer_joint, theta=float(state["theta"])
        )
        outer_block.loc[:, list(_OUTCOME_COLUMNS)] = _outcome_margins(transformed)
        if "model_id" in outer_block:
            outer_block["model_id"] = outer_block["model_id"].astype(str).map(
                lambda value: f"{value}__exploratory_sarmanov"
            )
        ledger_parts.append(outer_block)
        grid_parts.append(
            _long_joint_grid(
                outer_block[["outer_fold_id", "match_id"]], transformed
            )
        )
        state = {
            **state,
            "selection_outer_fold_id": str(outer_fold_id),
            "inner_min_prediction_year": int(inner_block["prediction_year"].min()),
            "inner_max_prediction_year": inner_max,
            "outer_prediction_year": outer_year,
            "fit_scope": "selected_inner_oof_only",
            "outer_omega_min": float(diagnostics["omega"].min()),
            "outer_omega_max": float(diagnostics["omega"].max()),
            "outer_covariance_min": float(diagnostics["score_covariance"].min()),
            "outer_covariance_max": float(diagnostics["score_covariance"].max()),
        }
        states[str(outer_fold_id)] = state
        selection_rows.append(dict(state))

    ledger = pd.concat(ledger_parts, ignore_index=True).sort_values(
        ["prediction_year", "match_id"]
    ).reset_index(drop=True)
    grid = pd.concat(grid_parts, ignore_index=True).sort_values(
        ["outer_fold_id", "match_id", "home_goals", "away_goals"]
    ).reset_index(drop=True)
    selection = pd.DataFrame(selection_rows).sort_values("outer_prediction_year").reset_index(
        drop=True
    )
    metadata = {
        "schema_version": 1,
        "model_id": "exploratory_sarmanov_marginal_preserving",
        "role": "exploratory_contradiction_benchmark_not_promotion_eligible",
        "outer_fold_count": int(ledger["outer_fold_id"].nunique()),
        "maximum_outer_prediction_year": int(ledger["prediction_year"].max()),
        "maximum_inner_prediction_year": int(inner["prediction_year"].max()),
        "fit_scope": "selected_inner_oof_only_within_each_outer_training_window",
        "marginal_contract": "exact_finite_support_marginals_from_frozen_outer_predictor",
        "kernel": "centered_exp_negative_score",
        "parameterization": "row_feasible_interval_standardized_theta",
        "supports_negative_dependence": True,
        "confirmatory_status": "exploratory_only_protocol_does_not_allow_promotion",
        "audit_year_used": False,
        "method_reference_doi": "10.1093/jrsssc/qlae050",
    }
    return SarmanovExplorationResult(ledger, grid, selection, states, metadata)


def _validate_marginal(values: np.ndarray, label: str) -> np.ndarray:
    marginal = np.asarray(values, dtype=float)
    if marginal.ndim != 1 or len(marginal) < 2:
        raise ValueError(f"{label} must be a one-dimensional PMF with at least two cells")
    if not np.isfinite(marginal).all() or np.any(marginal < 0):
        raise ValueError(f"{label} must be finite and nonnegative")
    mass = float(marginal.sum())
    if not np.isclose(mass, 1.0, atol=1e-10):
        raise ValueError(f"{label} must sum to one")
    return marginal / mass


def _feasible_omega_interval(
    kernel_product: np.ndarray,
    base_probability: np.ndarray,
) -> tuple[float, float]:
    active = base_probability > 0.0
    values = kernel_product[active]
    positive = values[values > 0.0]
    negative = values[values < 0.0]
    lower = -np.inf if positive.size == 0 else -1.0 / float(positive.max())
    upper = np.inf if negative.size == 0 else 1.0 / float(np.abs(negative).max())
    if not np.isfinite(lower) or not np.isfinite(upper) or not lower < 0.0 < upper:
        raise RuntimeError("Sarmanov kernel has no finite two-sided feasible interval")
    return lower, upper


def _score_covariance(joint: np.ndarray) -> float:
    score = np.arange(joint.shape[0], dtype=float)
    home_mean = float(joint.sum(axis=1) @ score)
    away_mean = float(joint.sum(axis=0) @ score)
    cross = float(np.sum(joint * np.outer(score, score)))
    return cross - home_mean * away_mean


def _outcome_margins(joint: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()]
            for matrix in joint
        ],
        dtype=float,
    )


def _long_joint_grid(order: pd.DataFrame, joint: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for matrix, (outer_fold_id, match_id) in zip(
        joint, order.itertuples(index=False, name=None), strict=True
    ):
        for home in range(matrix.shape[0]):
            for away in range(matrix.shape[1]):
                rows.append(
                    {
                        "outer_fold_id": outer_fold_id,
                        "match_id": match_id,
                        "home_goals": home,
                        "away_goals": away,
                        "probability": float(matrix[home, away]),
                    }
                )
    return pd.DataFrame(rows)


def _validate_ledgers(inner: pd.DataFrame, outer: pd.DataFrame) -> None:
    common = {
        "outer_fold_id",
        "match_id",
        "prediction_year",
        "train_end_year",
        "is_oof",
        "home_win",
        "draw",
        "away_win",
    }
    required_inner = common | {"observed_home_goals", "observed_away_goals"}
    for label, frame, required in (
        ("inner", inner, required_inner),
        ("outer", outer, common),
    ):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{label} Sarmanov ledger must be a nonempty DataFrame")
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} Sarmanov ledger is missing columns: {missing}")
        if not frame["is_oof"].astype(bool).all():
            raise ValueError(f"{label} Sarmanov ledger must contain OOF rows only")
        train = pd.to_numeric(frame["train_end_year"], errors="coerce")
        predict = pd.to_numeric(frame["prediction_year"], errors="coerce")
        if train.isna().any() or predict.isna().any() or not train.lt(predict).all():
            raise ValueError(f"{label} Sarmanov chronology is invalid")
    if int(pd.to_numeric(outer["prediction_year"], errors="raise").max()) > 2010:
        raise ValueError("2014/audit predictions are forbidden during Sarmanov exploration")


def _single_year(frame: pd.DataFrame) -> int:
    years = pd.to_numeric(frame["prediction_year"], errors="raise").unique()
    if len(years) != 1:
        raise ValueError("Each outer Sarmanov fold must contain one prediction year")
    return int(years[0])


__all__ = [
    "SarmanovExplorationResult",
    "apply_nested_sarmanov_exploration",
    "fit_sarmanov_theta",
    "sarmanov_joint_from_marginals",
    "transform_joint_batch_sarmanov",
]
