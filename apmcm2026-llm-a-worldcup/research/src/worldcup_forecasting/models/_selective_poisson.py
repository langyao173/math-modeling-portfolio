"""Exact diagonal L2 penalties for the team-perspective Poisson design.

``PoissonRegressor`` exposes one scalar penalty. Selective team-block penalties
are therefore optimized directly on the original encoded design. When a team
block has exactly zero penalty, its full one-hot coefficients would otherwise be
unidentified with the intercept. The optimizer first imposes one sum-to-zero
contrast per unpenalized team block, then projects out any remaining joint
data-and-penalty null directions (for example, disconnected opponent graphs).
The returned ``coef_`` is the unique minimum-norm representative in that
identified subspace and multiplies the complete original one-hot design. Each
unpenalized team block sums to zero, so an unseen team has zero team deviation.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.connection import Connection
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin

TEAM_EFFECT_GROUPS = ("attacker_team", "opponent_defender_team")
DEFAULT_WALL_CLOCK_LIMIT_SECONDS = 30.0


class OptimizationTimeoutError(RuntimeError):
    """Raised when a selective-penalty fit exceeds its fixed wall-clock budget."""


@dataclass(frozen=True)
class SelectivePenaltyProfile:
    """Frozen mapping from encoded columns to effective L2 penalties."""

    feature_names: tuple[str, ...]
    coefficient_groups: tuple[str, ...]
    effective_penalties: tuple[float, ...]
    base_penalty: float
    team_penalty: float

    def to_frame(self) -> pd.DataFrame:
        """Return the complete auditable coefficient-penalty assignment."""
        return pd.DataFrame(
            {
                "feature_name": self.feature_names,
                "coefficient_group": self.coefficient_groups,
                "effective_l2_penalty": self.effective_penalties,
            }
        )


class DiagonalPenaltyPoissonRegressor(RegressorMixin, BaseEstimator):
    """Minimize a weighted Poisson objective with per-coefficient L2 penalties.

    The intercept is never penalized. Exact-zero attacker and defender penalties
    activate one sum-to-zero constraint per block through a full-rank contrast
    basis. Any residual data-and-penalty nullspace is projected out before
    optimization. ``coef_`` always multiplies the original encoded design.
    """

    def __init__(
        self,
        effective_penalties: tuple[float, ...],
        coefficient_groups: tuple[str, ...],
        *,
        max_iter: int = 2_000,
        tolerance: float = 1e-8,
        wall_clock_limit_seconds: float = DEFAULT_WALL_CLOCK_LIMIT_SECONDS,
        enforce_hard_wall_clock: bool = True,
    ) -> None:
        self.effective_penalties = effective_penalties
        self.coefficient_groups = coefficient_groups
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.wall_clock_limit_seconds = wall_clock_limit_seconds
        self.enforce_hard_wall_clock = enforce_hard_wall_clock

    def fit(
        self,
        values: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> DiagonalPenaltyPoissonRegressor:
        """Fit the strictly audited convex objective on the original design."""
        if not isinstance(self.enforce_hard_wall_clock, (bool, np.bool_)):
            raise ValueError("enforce_hard_wall_clock must be boolean")
        if self.enforce_hard_wall_clock:
            return self._fit_in_subprocess(values, target, sample_weight)
        matrix = _matrix(values)
        response = _response(target, matrix.shape[0])
        weights = _normalized_weights(sample_weight, matrix.shape[0])
        penalties = _penalties(self.effective_penalties, matrix.shape[1])
        groups = _groups(self.coefficient_groups, matrix.shape[1])
        if int(self.max_iter) != self.max_iter or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if not np.isfinite(self.wall_clock_limit_seconds) or self.wall_clock_limit_seconds <= 0:
            raise ValueError("wall_clock_limit_seconds must be finite and positive")
        fit_started = perf_counter()

        def check_deadline() -> None:
            elapsed = perf_counter() - fit_started
            if elapsed > float(self.wall_clock_limit_seconds):
                raise OptimizationTimeoutError(
                    "Diagonal-penalty Poisson optimization exceeded its fixed "
                    f"{float(self.wall_clock_limit_seconds):.3f}s wall-clock budget"
                )

        team_basis, constraint = _contrast_basis(groups, penalties)
        reduced_design = matrix @ team_basis
        constrained_design = np.column_stack(
            [np.ones(matrix.shape[0], dtype=float), reduced_design]
        )
        reduced_penalty = team_basis.T @ (penalties[:, None] * team_basis)
        constrained_penalty = np.zeros(
            (reduced_design.shape[1] + 1, reduced_design.shape[1] + 1), dtype=float
        )
        constrained_penalty[1:, 1:] = reduced_penalty
        identifiable_basis, residual_nullspace = _identifiable_subspace(
            np.sqrt(weights)[:, None] * constrained_design,
            constrained_penalty,
        )
        optimization_design = constrained_design @ identifiable_basis
        optimization_penalty = identifiable_basis.T @ constrained_penalty @ identifiable_basis
        mean_response = float(weights @ response)
        initial_intercept = float(np.log(max(mean_response, np.finfo(float).tiny)))
        constrained_initial = np.zeros(reduced_design.shape[1] + 1, dtype=float)
        constrained_initial[0] = initial_intercept
        initial = identifiable_basis.T @ constrained_initial

        def objective_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            check_deadline()
            constrained_parameters = identifiable_basis @ parameters
            coefficient = team_basis @ constrained_parameters[1:]
            linear_predictor = optimization_design @ parameters
            mean = _poisson_mean(linear_predictor)
            residual = weights * (mean - response)
            objective = float(
                weights @ (mean - response * linear_predictor)
                + 0.5 * np.dot(penalties, coefficient**2)
            )
            coefficient_gradient = matrix.T @ residual + penalties * coefficient
            constrained_gradient = np.concatenate(
                ([float(residual.sum())], team_basis.T @ coefficient_gradient)
            )
            gradient = identifiable_basis.T @ constrained_gradient
            return objective, gradient

        def objective_hessian(parameters: np.ndarray) -> np.ndarray:
            check_deadline()
            linear_predictor = optimization_design @ parameters
            curvature = weights * _poisson_mean(linear_predictor)
            return (
                optimization_design.T @ (curvature[:, None] * optimization_design)
                + optimization_penalty
            )

        intercept_only_objective, _ = objective_gradient(initial)
        result = minimize(
            objective_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": int(self.max_iter),
                "gtol": float(self.tolerance),
                "ftol": np.finfo(float).eps,
                "maxls": 50,
            },
        )
        check_deadline()
        parameters = np.asarray(result.x, dtype=float)
        objective, gradient = objective_gradient(parameters)
        convergence_bound = max(10.0 * float(self.tolerance), 1e-7)
        refinement_iterations = 0
        while (
            float(np.max(np.abs(gradient), initial=0.0)) > convergence_bound
            and refinement_iterations < 25
        ):
            check_deadline()
            hessian = objective_hessian(parameters)
            try:
                direction = np.linalg.solve(hessian, -gradient)
            except np.linalg.LinAlgError:
                break
            directional_derivative = float(gradient @ direction)
            if (
                not np.isfinite(direction).all()
                or not np.isfinite(directional_derivative)
                or directional_derivative >= 0
            ):
                break
            accepted = False
            step = 1.0
            for _ in range(30):
                candidate = parameters + step * direction
                candidate_objective, candidate_gradient = objective_gradient(candidate)
                if np.isfinite(candidate_objective) and candidate_objective <= (
                    objective + 1e-4 * step * directional_derivative
                ):
                    parameters = candidate
                    objective = candidate_objective
                    gradient = candidate_gradient
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
            refinement_iterations += 1
        constrained_parameters = identifiable_basis @ parameters
        coefficient = team_basis @ constrained_parameters[1:]
        kkt_residual = float(np.max(np.abs(gradient), initial=0.0))
        constraint_residual = float(np.max(np.abs(constraint @ coefficient), initial=0.0))
        gauge_residual = float(
            np.max(np.abs(residual_nullspace.T @ constrained_parameters), initial=0.0)
        )
        audited_hessian = objective_hessian(parameters)
        hessian_rank = int(np.linalg.matrix_rank(audited_hessian))
        if (
            not np.isfinite(objective)
            or not np.isfinite(parameters).all()
            or kkt_residual > convergence_bound
            or constraint_residual > 1e-10
            or gauge_residual > 1e-10
            or hessian_rank != parameters.size
        ):
            raise RuntimeError(
                "Diagonal-penalty Poisson optimization failed its identifiability audit: "
                f"status={result.message!s}, kkt={kkt_residual:.3e}, "
                f"constraint={constraint_residual:.3e}, gauge={gauge_residual:.3e}, "
                f"hessian_rank={hessian_rank}/{parameters.size}"
            )

        self.coef_ = coefficient
        self.intercept_ = float(constrained_parameters[0])
        self.n_features_in_ = int(matrix.shape[1])
        self.n_iter_ = int(result.nit) + refinement_iterations
        self.effective_penalties_ = penalties
        self.coefficient_groups_ = groups
        self.team_contrast_basis_ = team_basis
        self.constraint_matrix_ = constraint
        self.constraint_rank_ = int(np.linalg.matrix_rank(constraint))
        full_contrast = np.zeros((matrix.shape[1] + 1, team_basis.shape[1] + 1), dtype=float)
        full_contrast[0, 0] = 1.0
        full_contrast[1:, 1:] = team_basis
        self.identifiable_parameter_basis_ = full_contrast @ identifiable_basis
        self.parameter_nullspace_ = full_contrast @ residual_nullspace
        self.identifiable_rank_ = int(parameters.size)
        self.hessian_rank_ = hessian_rank
        self.objective_ = objective
        self.intercept_only_objective_ = float(intercept_only_objective)
        self.kkt_residual_ = kkt_residual
        self.constraint_residual_ = constraint_residual
        self.gauge_residual_ = gauge_residual
        self.optimization_message_ = (
            f"{result.message!s}; newton_refinement_iterations={refinement_iterations}"
        )
        self.wall_clock_seconds_ = float(perf_counter() - fit_started)
        self.wall_clock_limit_seconds_ = float(self.wall_clock_limit_seconds)
        return self

    def _fit_in_subprocess(
        self,
        values: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> DiagonalPenaltyPoissonRegressor:
        """Run the complete numerical fit in a killable spawned process."""
        limit = float(self.wall_clock_limit_seconds)
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError("wall_clock_limit_seconds must be finite and positive")
        started = perf_counter()
        context = get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process: Any | None = None
        payload: dict[str, Any] | None = None
        started_process = False
        try:
            process = context.Process(
                target=_fit_worker,
                args=(
                    sender,
                    {
                        "effective_penalties": tuple(self.effective_penalties),
                        "coefficient_groups": tuple(self.coefficient_groups),
                        "max_iter": self.max_iter,
                        "tolerance": self.tolerance,
                        "wall_clock_limit_seconds": limit,
                        "enforce_hard_wall_clock": False,
                    },
                    np.asarray(values),
                    np.asarray(target),
                    None if sample_weight is None else np.asarray(sample_weight),
                ),
                daemon=True,
                name="diagonal-penalty-poisson-fit",
            )
            process.start()
            started_process = True
            sender.close()
            remaining = max(0.0, limit - (perf_counter() - started))
            if receiver.poll(remaining):
                try:
                    received = receiver.recv()
                except EOFError as error:
                    raise RuntimeError(
                        "Diagonal-penalty Poisson worker exited without an audit payload"
                    ) from error
                if not isinstance(received, dict):
                    raise RuntimeError(
                        "Diagonal-penalty Poisson worker returned a malformed payload"
                    )
                if perf_counter() - started > limit:
                    _terminate_process(process)
                    raise OptimizationTimeoutError(
                        "Diagonal-penalty Poisson optimization exceeded its fixed "
                        f"{limit:.3f}s wall-clock budget before result receipt"
                    )
                payload = received
            else:
                _terminate_process(process)
                raise OptimizationTimeoutError(
                    "Diagonal-penalty Poisson optimization exceeded its fixed "
                    f"{limit:.3f}s wall-clock budget and its worker was terminated"
                )
        finally:
            sender.close()
            receiver.close()
            if started_process and process is not None:
                if process.is_alive():
                    process.join(timeout=1.0)
                if process.is_alive():
                    _terminate_process(process)
                else:
                    process.join(timeout=0.0)
                process.close()
        if payload is None:
            raise RuntimeError("Diagonal-penalty Poisson worker produced no result")
        if payload.get("status") != "ok":
            _raise_worker_error(payload)
        state = payload.get("state")
        if not isinstance(state, dict):
            raise RuntimeError("Diagonal-penalty Poisson worker omitted fitted state")
        self.__dict__.update(state)
        self.enforce_hard_wall_clock = True
        self.wall_clock_seconds_ = float(perf_counter() - started)
        self.wall_clock_limit_seconds_ = limit
        self.hard_wall_clock_enforced_ = True
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        """Return finite Poisson means for the original encoded design."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("DiagonalPenaltyPoissonRegressor must be fitted before predict")
        matrix = _matrix(values)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("Encoded design width changed after fitting")
        mean = _poisson_mean(self.intercept_ + matrix @ self.coef_)
        if not np.isfinite(mean).all() or np.any(mean <= 0):
            raise RuntimeError("Diagonal-penalty Poisson model produced invalid means")
        return mean


def build_selective_penalty_profile(
    feature_names: tuple[str, ...] | list[str] | np.ndarray,
    *,
    base_penalty: float,
    team_penalty: float,
    team_groups: tuple[str, ...] = TEAM_EFFECT_GROUPS,
) -> SelectivePenaltyProfile:
    """Build an exact, exhaustive penalty profile for the encoded design.

    Only the two scientifically declared identity blocks may receive the team
    penalty.  Callers cannot silently exempt arbitrary context columns.
    """
    names = tuple(str(name) for name in feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Encoded feature names must be nonempty and unique")
    if len(team_groups) != len(set(team_groups)) or set(team_groups) != set(TEAM_EFFECT_GROUPS):
        raise ValueError("Penalty groups must be exactly attacker_team and opponent_defender_team")
    base = _nonnegative_penalty(base_penalty, "base_penalty")
    team = _nonnegative_penalty(team_penalty, "team_penalty")
    if base == 0.0 and team != 0.0:
        raise ValueError(
            "A positive team penalty with zero base penalty cannot be represented "
            "under the frozen selective-penalty contract"
        )

    groups = tuple(_coefficient_group(name, team_groups) for name in names)
    for group in TEAM_EFFECT_GROUPS:
        if group not in groups:
            raise ValueError(f"Encoded design is missing required penalty group {group!r}")
    penalties = tuple(team if group in TEAM_EFFECT_GROUPS else base for group in groups)
    return SelectivePenaltyProfile(
        feature_names=names,
        coefficient_groups=groups,
        effective_penalties=penalties,
        base_penalty=base,
        team_penalty=team,
    )


def original_scale_coefficients(pipeline: object) -> np.ndarray:
    """Recover coefficients multiplying the unscaled encoded design."""
    steps = getattr(pipeline, "named_steps", {})
    poisson = steps["poisson"]
    coefficients = np.asarray(poisson.coef_, dtype=float)
    scaler = steps.get("penalty_scaler")
    if scaler is None:
        return coefficients
    factors = np.asarray(scaler.scale_factors_, dtype=float)
    if coefficients.shape != factors.shape:
        raise RuntimeError("Fitted coefficient and penalty-scale shapes are inconsistent")
    return coefficients * factors


def _coefficient_group(name: str, team_groups: tuple[str, ...]) -> str:
    matches = [group for group in team_groups if name.startswith(f"categorical__{group}_")]
    if len(matches) > 1:
        raise ValueError(f"Encoded feature {name!r} maps to multiple penalty groups")
    return matches[0] if matches else "non_team_context"


def _nonnegative_penalty(value: float, label: str) -> float:
    penalty = float(value)
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return penalty


def _response(values: np.ndarray, expected_rows: int) -> np.ndarray:
    response = np.asarray(values, dtype=float)
    if (
        response.ndim != 1
        or response.size != expected_rows
        or not np.isfinite(response).all()
        or np.any(response < 0)
    ):
        raise ValueError("Poisson target must be a finite nonnegative vector matching rows")
    return response


def _normalized_weights(values: np.ndarray | None, expected_rows: int) -> np.ndarray:
    weights = (
        np.ones(expected_rows, dtype=float) if values is None else np.asarray(values, dtype=float)
    )
    if (
        weights.ndim != 1
        or weights.size != expected_rows
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or float(weights.sum()) <= 0
    ):
        raise ValueError("sample_weight must be finite, nonnegative, and match rows")
    return weights / float(weights.sum())


def _penalties(values: tuple[float, ...], expected_columns: int) -> np.ndarray:
    penalties = np.asarray(values, dtype=float)
    if (
        penalties.ndim != 1
        or penalties.size != expected_columns
        or not np.isfinite(penalties).all()
        or np.any(penalties < 0)
    ):
        raise ValueError("effective_penalties must be finite, nonnegative, and match columns")
    return penalties


def _groups(values: tuple[str, ...], expected_columns: int) -> tuple[str, ...]:
    groups = tuple(str(value) for value in values)
    if len(groups) != expected_columns:
        raise ValueError("coefficient_groups must match encoded design columns")
    return groups


def _contrast_basis(
    groups: tuple[str, ...], penalties: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic full-rank basis for exact zero-sum team blocks."""
    width = len(groups)
    constrained: list[tuple[int, ...]] = []
    for group in TEAM_EFFECT_GROUPS:
        indices = tuple(index for index, value in enumerate(groups) if value == group)
        if indices and np.all(penalties[list(indices)] == 0.0):
            constrained.append(indices)
    constraint = np.zeros((len(constrained), width), dtype=float)
    references: dict[int, int] = {}
    for row, indices in enumerate(constrained):
        constraint[row, list(indices)] = 1.0
        reference = indices[-1]
        for index in indices[:-1]:
            references[index] = reference
        references[reference] = reference

    columns: list[np.ndarray] = []
    for index in range(width):
        reference = references.get(index)
        if reference == index:
            continue
        column = np.zeros(width, dtype=float)
        column[index] = 1.0
        if reference is not None:
            column[reference] = -1.0
        columns.append(column)
    raw_basis = np.column_stack(columns) if columns else np.empty((width, 0), dtype=float)
    basis = np.linalg.qr(raw_basis, mode="reduced")[0]
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1.0
    expected_rank = width - len(constrained)
    if (
        np.linalg.matrix_rank(basis) != expected_rank
        or np.linalg.matrix_rank(constraint) != len(constrained)
        or not np.allclose(constraint @ basis, 0.0, atol=1e-12, rtol=0.0)
        or not np.allclose(basis.T @ basis, np.eye(expected_rank), atol=1e-12)
    ):
        raise RuntimeError("Failed to construct the frozen team-block contrast basis")
    return basis, constraint


def _identifiable_subspace(
    design: np.ndarray, penalty: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split constrained parameters into identified and exact-null directions."""
    eigenvalue, eigenvector = np.linalg.eigh((penalty + penalty.T) / 2.0)
    scale = max(float(np.max(np.abs(eigenvalue), initial=0.0)), 1.0)
    tolerance = np.finfo(float).eps * max(penalty.shape) * scale
    if np.any(eigenvalue < -tolerance):
        raise RuntimeError("Diagonal penalty became non-positive-semidefinite")
    square_root = (eigenvector * np.sqrt(np.clip(eigenvalue, 0.0, None))) @ eigenvector.T
    stacked = np.vstack([design, square_root])
    _, singular_value, right = np.linalg.svd(stacked, full_matrices=True)
    singular_scale = max(float(np.max(singular_value, initial=0.0)), 1.0)
    singular_tolerance = np.finfo(float).eps * max(stacked.shape) * singular_scale
    rank = int(np.sum(singular_value > singular_tolerance))
    identifiable = right[:rank].T
    nullspace = right[rank:].T
    if (
        rank == 0
        or np.linalg.matrix_rank(identifiable) != rank
        or not np.allclose(identifiable.T @ identifiable, np.eye(rank), atol=1e-12)
        or not np.allclose(identifiable.T @ nullspace, 0.0, atol=1e-12)
        or not np.allclose(stacked @ nullspace, 0.0, atol=1e-10)
    ):
        raise RuntimeError("Failed to isolate the identifiable Poisson parameter subspace")
    return identifiable, nullspace


def _poisson_mean(linear_predictor: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        mean = np.exp(linear_predictor)
    if not np.isfinite(mean).all():
        # This value is outside any plausible football optimum and lets the
        # line search reject the trial without silently clipping the objective.
        return np.full_like(linear_predictor, np.finfo(float).max)
    return mean


def _matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Encoded Poisson design must be a finite matrix")
    return matrix


def _fit_worker(
    connection: Connection,
    parameters: dict[str, Any],
    values: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray | None,
) -> None:
    """Child-process entry point for a fully isolated numerical fit."""
    try:
        fitted = DiagonalPenaltyPoissonRegressor(**parameters).fit(
            values,
            target,
            sample_weight=sample_weight,
        )
        connection.send({"status": "ok", "state": fitted.__dict__})
    except BaseException as error:
        connection.send(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        connection.close()


def _terminate_process(process: Any) -> None:
    """Terminate and, if required, kill one exact spawned worker."""
    if not process.is_alive():
        process.join(timeout=0.0)
        return
    process.terminate()
    process.join(timeout=2.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=2.0)


def _raise_worker_error(payload: dict[str, Any]) -> None:
    """Reconstruct stable public exception classes from a worker failure."""
    error_type = str(payload.get("error_type", "RuntimeError"))
    message = str(payload.get("message", "unknown worker failure"))
    if error_type == "OptimizationTimeoutError":
        raise OptimizationTimeoutError(message)
    if error_type == "ValueError":
        raise ValueError(message)
    raise RuntimeError(f"{error_type}: {message}")


__all__ = [
    "DEFAULT_WALL_CLOCK_LIMIT_SECONDS",
    "DiagonalPenaltyPoissonRegressor",
    "OptimizationTimeoutError",
    "SelectivePenaltyProfile",
    "TEAM_EFFECT_GROUPS",
    "build_selective_penalty_profile",
    "original_scale_coefficients",
]
