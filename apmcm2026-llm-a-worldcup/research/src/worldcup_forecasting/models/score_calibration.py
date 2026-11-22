"""OOF-only calibration of coherent joint football score distributions.

On each W/D/L partition the projection minimizes ``KL(projected || source)`` at
the requested mass by one common cell multiplier. Consequently partition ranks,
nonnegativity, score/outcome consistency, and team-swap symmetry are preserved.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .calibration import TemperatureScaler
from .count import JointGoalBatch

OUTCOME_ORDER = ("home_win", "draw", "away_win")


def outcome_margins(joint: np.ndarray) -> np.ndarray:
    """Return ``[home_win, draw, away_win]`` for each joint score matrix."""
    matrices = _validate_joint(joint)
    return np.column_stack(
        [
            np.tril(matrices, k=-1).sum(axis=(1, 2)),
            np.trace(matrices, axis1=1, axis2=2),
            np.triu(matrices, k=1).sum(axis=(1, 2)),
        ]
    )


def minimal_kl_partition_projection(
    joint: np.ndarray,
    target_outcomes: np.ndarray,
    *,
    zero_mass_policy: str = "uniform",
) -> np.ndarray:
    """Project score grids onto exact W/D/L margins with minimum KL change.

    If a requested partition has zero source mass, finite KL is impossible.  The
    default support-completing convention assigns the requested mass uniformly
    over that all-tied partition; ``zero_mass_policy='raise'`` instead fails closed.
    """
    matrices = _validate_joint(joint)
    targets = _validate_outcome_probabilities(target_outcomes, len(matrices))
    if zero_mass_policy not in {"uniform", "raise"}:
        raise ValueError("zero_mass_policy must be 'uniform' or 'raise'")
    width = matrices.shape[1]
    row, column = np.indices((width, width))
    masks = (row > column, row == column, row < column)
    projected = np.zeros_like(matrices)
    for sample, matrix in enumerate(matrices):
        for outcome_index, mask in enumerate(masks):
            target = float(targets[sample, outcome_index])
            if target == 0:
                continue
            source = float(matrix[mask].sum())
            if source > 0:
                projected[sample][mask] = matrix[mask] * (target / source)
            elif zero_mass_policy == "uniform":
                cell_count = int(mask.sum())
                if cell_count == 0:
                    raise ValueError(f"Score support lacks outcome partition {outcome_index}")
                projected[sample][mask] = target / cell_count
            else:
                raise ValueError(
                    f"Source score grid has zero mass in outcome partition {outcome_index}"
                )
    # Independent partition checks expose mask-orientation errors.
    if not np.allclose(projected.sum(axis=(1, 2)), 1.0, atol=1e-12):
        raise RuntimeError("Partition projection failed to preserve unit mass")
    actual = outcome_margins(projected)
    if not np.allclose(actual, targets, atol=1e-12):
        raise RuntimeError("Partition projection failed to attain requested margins")
    return projected


def minimum_kl_outcome_projection(
    joint: np.ndarray,
    target_outcomes: Mapping[str, float] | np.ndarray,
) -> np.ndarray:
    """Project one score matrix onto exact W/D/L margins, failing on lost support.

    This single-match public wrapper uses the mathematically strict convention:
    requesting positive mass in an outcome partition that has no source support is
    infeasible and raises instead of silently inventing scoreline shape.
    """
    matrix = np.asarray(joint, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A joint score matrix must be square and two-dimensional")
    if isinstance(target_outcomes, Mapping):
        missing = [name for name in OUTCOME_ORDER if name not in target_outcomes]
        extra = sorted(set(target_outcomes).difference(OUTCOME_ORDER))
        if missing or extra:
            raise ValueError(
                f"Outcome target must contain exactly {OUTCOME_ORDER}; "
                f"missing={missing}, extra={extra}"
            )
        target = np.asarray([[target_outcomes[name] for name in OUTCOME_ORDER]], dtype=float)
    else:
        values = np.asarray(target_outcomes, dtype=float)
        target = values.reshape(1, 3) if values.shape == (3,) else values
    return minimal_kl_partition_projection(
        matrix[None, :, :],
        target,
        zero_mass_policy="raise",
    )[0]


@dataclass
class OutcomePartitionCalibrator:
    """Temperature-calibrate OOF outcome margins and project them to score cells."""

    min_temperature: float = 0.05
    max_temperature: float = 20.0
    require_oof: bool = True
    zero_mass_policy: str = "uniform"

    def __post_init__(self) -> None:
        self.scaler_: TemperatureScaler | None = None
        self.selection_report_: dict[str, float | bool | str] = {}

    def fit_oof(
        self,
        joint: np.ndarray | JointGoalBatch,
        observed: np.ndarray,
        *,
        provenance: pd.DataFrame,
    ) -> OutcomePartitionCalibrator:
        """Fit only after proving every calibration row is chronologically OOF."""
        matrices = _joint_array(joint)
        _validate_oof_provenance(provenance, len(matrices))
        labels = _outcome_labels(observed, len(matrices))
        base = outcome_margins(matrices)
        scaler = TemperatureScaler(
            min_temperature=self.min_temperature,
            max_temperature=self.max_temperature,
        ).fit(base, labels)
        candidate = scaler.transform(base)
        baseline_loss, baseline_brier = _proper_scores(base, labels)
        candidate_loss, candidate_brier = _proper_scores(candidate, labels)
        selected = candidate_loss <= baseline_loss + 1e-12
        if not selected:
            scaler.temperature_ = 1.0
            candidate_loss, candidate_brier = baseline_loss, baseline_brier
        self.scaler_ = scaler
        self.selection_report_ = {
            "fit_scope": "training_oof_only",
            "selected": selected,
            "baseline_log_loss": baseline_loss,
            "candidate_log_loss": candidate_loss,
            "baseline_brier": baseline_brier,
            "candidate_brier": candidate_brier,
            "temperature": float(scaler.temperature_),
        }
        return self

    def fit(
        self,
        joint: np.ndarray | JointGoalBatch,
        observed: np.ndarray,
        *,
        provenance: pd.DataFrame | None = None,
        is_oof: bool = False,
    ) -> OutcomePartitionCalibrator:
        """Compatibility fit that still fails closed unless OOF status is proven."""
        if provenance is not None:
            return self.fit_oof(joint, observed, provenance=provenance)
        if self.require_oof and not is_oof:
            raise ValueError("Calibration requires out-of-fold (OOF) predictions")
        matrices = _joint_array(joint)
        labels = _outcome_labels(observed, len(matrices))
        synthetic = pd.DataFrame(
            {
                "is_oof": True,
                "prediction_year": np.ones(len(matrices)),
                "train_end_year": np.zeros(len(matrices)),
            }
        )
        return self.fit_oof(matrices, labels, provenance=synthetic)

    def transform(self, joint: np.ndarray | JointGoalBatch) -> np.ndarray:
        """Return calibrated joint matrices with exact calibrated W/D/L margins."""
        if self.scaler_ is None:
            raise RuntimeError("OutcomePartitionCalibrator must be fitted before transform")
        matrices = _joint_array(joint)
        target = self.scaler_.transform(outcome_margins(matrices))
        return minimal_kl_partition_projection(
            matrices,
            target,
            zero_mass_policy=self.zero_mass_policy,
        )

    def transform_batch(self, batch: JointGoalBatch) -> JointGoalBatch:
        """Preserve the batch interface while recalculating all derived views."""
        probability = self.transform(batch)
        support = np.arange(batch.support_max + 1, dtype=float)
        metadata = dict(batch.model_metadata)
        metadata["outcome_calibration"] = dict(self.selection_report_)
        return JointGoalBatch(
            probability=probability,
            support_max=batch.support_max,
            retained_mass=np.asarray(batch.retained_mass, dtype=float),
            expected_home_goals=np.sum(probability * support[None, :, None], axis=(1, 2)),
            expected_away_goals=np.sum(probability * support[None, None, :], axis=(1, 2)),
            model_metadata=metadata,
        )


@dataclass
class DiagonalScoreCalibrator:
    """Select or reject one OOF-fitted diagonal score-mass tilt.

    The tilt changes probability mass only; it never contains a rule that forces a
    draw point decision.  Selection is based on proper scores and the identity tilt
    is retained when the candidate does not improve without regression.
    """

    log_tilt_bounds: tuple[float, float] = (-4.0, 4.0)
    l2_penalty: float = 1e-4
    paired_score_tolerance: float = 0.01
    minimum_improvement: float = 1e-12

    def __post_init__(self) -> None:
        lower, upper = self.log_tilt_bounds
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("log_tilt_bounds must be finite and increasing")
        if self.l2_penalty < 0 or self.paired_score_tolerance < 0:
            raise ValueError("Calibration penalties/tolerances must be nonnegative")
        self.log_tilt_: float = 0.0
        self.is_fitted_: bool = False
        self.selection_report_: dict[str, float | bool | str] = {}

    def fit_oof(
        self,
        joint: np.ndarray | JointGoalBatch,
        observed: np.ndarray,
        *,
        provenance: pd.DataFrame,
    ) -> DiagonalScoreCalibrator:
        """Fit the diagonal tilt from chronologically valid OOF score grids."""
        matrices = _joint_array(joint)
        _validate_oof_provenance(provenance, len(matrices))
        labels = _outcome_labels(observed, len(matrices))
        baseline = outcome_margins(matrices)
        baseline_loss, baseline_brier = _proper_scores(baseline, labels)

        def objective(log_tilt: float) -> float:
            candidate = outcome_margins(_diagonal_tilt(matrices, log_tilt))
            log_loss, _ = _proper_scores(candidate, labels)
            return log_loss + self.l2_penalty * float(log_tilt) ** 2

        result = minimize_scalar(objective, bounds=self.log_tilt_bounds, method="bounded")
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("Diagonal score calibration optimization failed")
        proposed_tilt = float(result.x)
        proposed = outcome_margins(_diagonal_tilt(matrices, proposed_tilt))
        candidate_loss, candidate_brier = _proper_scores(proposed, labels)
        log_gain = baseline_loss - candidate_loss
        brier_gain = baseline_brier - candidate_brier
        selected = (
            log_gain >= self.minimum_improvement
            and candidate_brier <= baseline_brier + self.paired_score_tolerance
        ) or (
            brier_gain >= self.minimum_improvement
            and candidate_loss <= baseline_loss + self.paired_score_tolerance
        )
        self.log_tilt_ = proposed_tilt if selected else 0.0
        if not selected:
            candidate_loss, candidate_brier = baseline_loss, baseline_brier
        self.is_fitted_ = True
        self.selection_report_ = {
            "fit_scope": "training_oof_only",
            "selected": selected,
            "baseline_log_loss": baseline_loss,
            "candidate_log_loss": candidate_loss,
            "baseline_brier": baseline_brier,
            "candidate_brier": candidate_brier,
            "log_diagonal_tilt": float(self.log_tilt_),
            "forces_draw_decision": False,
        }
        return self

    def fit(
        self,
        joint: np.ndarray | JointGoalBatch,
        observed: np.ndarray,
        *,
        provenance: pd.DataFrame | None = None,
        is_oof: bool = False,
    ) -> DiagonalScoreCalibrator:
        """Fit only when OOF provenance is supplied or explicitly asserted."""
        if provenance is not None:
            return self.fit_oof(joint, observed, provenance=provenance)
        if not is_oof:
            raise ValueError("Diagonal calibration requires out-of-fold (OOF) predictions")
        count = len(_joint_array(joint))
        synthetic = pd.DataFrame(
            {"is_oof": True, "prediction_year": np.ones(count), "train_end_year": np.zeros(count)}
        )
        return self.fit_oof(joint, observed, provenance=synthetic)

    def transform(self, joint: np.ndarray | JointGoalBatch) -> np.ndarray:
        """Apply the selected tilt, or identity when the candidate was rejected."""
        if not self.is_fitted_:
            raise RuntimeError("DiagonalScoreCalibrator must be fitted before transform")
        return _diagonal_tilt(_joint_array(joint), self.log_tilt_)


def _diagonal_tilt(joint: np.ndarray, log_tilt: float) -> np.ndarray:
    matrices = _validate_joint(joint).copy()
    diagonal = np.arange(matrices.shape[1])
    matrices[:, diagonal, diagonal] *= float(np.exp(log_tilt))
    return matrices / matrices.sum(axis=(1, 2), keepdims=True)


def _joint_array(value: np.ndarray | JointGoalBatch) -> np.ndarray:
    return _validate_joint(value.probability if isinstance(value, JointGoalBatch) else value)


def _validate_joint(value: np.ndarray) -> np.ndarray:
    matrices = np.asarray(value, dtype=float)
    if matrices.ndim != 3 or matrices.shape[0] == 0 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("Joint score probabilities must have shape (n, support, support)")
    if not np.isfinite(matrices).all() or np.any(matrices < 0):
        raise ValueError("Joint score probabilities must be finite and nonnegative")
    if not np.allclose(matrices.sum(axis=(1, 2)), 1.0, atol=1e-10):
        raise ValueError("Every joint score matrix must sum to one")
    return matrices


def _validate_outcome_probabilities(value: np.ndarray, count: int) -> np.ndarray:
    probability = np.asarray(value, dtype=float)
    if probability.shape != (count, 3):
        raise ValueError("Outcome probabilities must have shape (n, 3)")
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("Outcome probabilities must be finite and nonnegative")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Every outcome probability row must sum to one")
    return probability


def _outcome_labels(observed: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(observed)
    if values.ndim == 2 and values.shape == (count, 2):
        goal = values.astype(float)
        if not np.isfinite(goal).all() or np.any(goal < 0):
            raise ValueError("Observed score pairs must be finite and nonnegative")
        return np.where(goal[:, 0] > goal[:, 1], 0, np.where(goal[:, 0] == goal[:, 1], 1, 2))
    if values.ndim != 1 or len(values) != count:
        raise ValueError("Observed outcomes must contain one label or score pair per row")
    labels: list[int] = []
    aliases = {"home": 0, "home_win": 0, "draw": 1, "away": 2, "away_win": 2}
    for value in values:
        if isinstance(value, (int, np.integer)):
            label = int(value)
        else:
            label = aliases.get(str(value).strip().casefold().replace("-", "_"), -1)
        if label not in (0, 1, 2):
            raise ValueError(f"Unknown outcome label: {value!r}")
        labels.append(label)
    return np.asarray(labels, dtype=int)


def _validate_oof_provenance(provenance: pd.DataFrame, count: int) -> None:
    if not isinstance(provenance, pd.DataFrame) or len(provenance) != count:
        raise ValueError("OOF provenance must contain one row per prediction")
    required = {"is_oof", "prediction_year", "train_end_year"}
    missing = sorted(required.difference(provenance.columns))
    if missing:
        raise ValueError(f"OOF provenance is missing required columns: {missing}")
    oof = provenance["is_oof"].astype("boolean")
    if oof.isna().any() or not bool(oof.all()):
        raise ValueError("Calibration may use only out-of-fold (OOF) predictions")
    prediction_year = pd.to_numeric(provenance["prediction_year"], errors="coerce")
    train_end = pd.to_numeric(provenance["train_end_year"], errors="coerce")
    if prediction_year.isna().any() or train_end.isna().any():
        raise ValueError("prediction_year and train_end_year must be finite")
    if not bool((train_end < prediction_year).all()):
        raise ValueError("train_end_year must be strictly earlier than prediction_year")


def _proper_scores(probability: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    selected = np.clip(probability[np.arange(len(labels)), labels], 1e-15, 1.0)
    observed = np.eye(3, dtype=float)[labels]
    return (
        float(-np.mean(np.log(selected))),
        float(np.mean(np.sum((probability - observed) ** 2, axis=1))),
    )


__all__ = [
    "DiagonalScoreCalibrator",
    "OutcomePartitionCalibrator",
    "minimal_kl_partition_projection",
    "minimum_kl_outcome_projection",
    "outcome_margins",
]
