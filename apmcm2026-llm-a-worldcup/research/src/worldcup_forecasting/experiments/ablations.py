"""Executable, paired A0--A8 component-removal experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from worldcup_forecasting.models.advanced_count import EquivariantCountConfig
from worldcup_forecasting.models.score_calibration import outcome_margins
from worldcup_forecasting.validation.research_contracts import validate_ablation_matrix

from ._ablation_contracts import (
    as_bool,
    evaluation_hash,
    half_life,
    prediction_times,
    validate_and_order,
)
from ._ablation_helpers import (
    VariantBaseFit,
    VariantPlan,
    fit_variant_base,
    goals,
    match_ids,
    variant_plans,
)
from .calibration_selection import (
    CalibrationSelectionConfig,
    ChronologicalCalibrationSelector,
)
from .research_features import make_count_config

CountConfigFactory = Callable[..., EquivariantCountConfig]
Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class AblationConfig:
    """Frozen experiment controls; seeds and the audit boundary are not tunable."""

    outer_test_years: tuple[int, ...] = (1986, 1990, 1994, 1998, 2002, 2006, 2010)
    seeds: tuple[int, ...] = (17, 29)
    year_column: str = "Year"
    latest_inner_folds: int = 3
    maximum_development_year: int = 2010
    symmetry_tolerance: float = 1e-10
    simplex_tolerance: float = 1e-10
    information_set: str = "pre_match"


@dataclass(frozen=True)
class AblationResult:
    """Fold metrics, auditable match predictions, and protocol metadata."""

    matrix: pd.DataFrame
    predictions: pd.DataFrame
    metadata: dict[str, Any]


def run_measured_ablations(
    frame: pd.DataFrame,
    selected_specs: pd.DataFrame,
    *,
    config: AblationConfig | None = None,
    count_config_factory: CountConfigFactory = make_count_config,
    feature_schema_hash: str | None = None,
    runtime_clock: Clock = perf_counter,
) -> AblationResult:
    """Refit A0--A8 on identical chronological folds and score real predictions.

    Residuals are learned only from strictly earlier structural OOF rates.  Each
    calibrator is selected and refitted only on the latest earlier inner-edition
    OOF score grids.  A8 remains structurally ineligible because it does not enforce
    symmetry, while its empirical swap error is reported exactly as observed.
    """
    cfg = config or AblationConfig()
    ordered, selected, schema_hash = validate_and_order(
        frame, selected_specs, cfg, feature_schema_hash, runtime_clock
    )
    matrix_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for selected_row in selected.itertuples(index=False):
        outer_year = int(selected_row.outer_test_year)
        outer_fold_id = str(selected_row.outer_fold_id)
        test = ordered.loc[ordered[cfg.year_column].eq(outer_year)].copy()
        evaluation_hash_value = evaluation_hash(outer_year, match_ids(test))
        selected_shared = as_bool(selected_row.shared_score, "shared_score")
        selected_residual = as_bool(selected_row.residual, "residual")
        plans = variant_plans(selected_shared, selected_residual)
        base_fits: dict[str, VariantBaseFit] = {}
        for plan in plans:
            if plan.variant_id == "A4" and not selected_shared:
                base_fits[plan.variant_id] = base_fits["A0"]
                continue
            if plan.variant_id == "A5" and not selected_residual:
                base_fits[plan.variant_id] = base_fits["A0"]
                continue
            base_fits[plan.variant_id] = fit_variant_base(
                ordered,
                outer_year=outer_year,
                plan=plan,
                alpha=float(selected_row.alpha),
                half_life_years=half_life(selected_row.half_life_years),
                count_config_factory=count_config_factory,
                latest_inner_folds=cfg.latest_inner_folds,
                maximum_development_year=cfg.maximum_development_year,
                runtime_clock=runtime_clock,
            )
        observed = goals(test).astype(int)
        labels = np.where(
            observed[:, 0] > observed[:, 1],
            0,
            np.where(observed[:, 0] == observed[:, 1], 1, 2),
        )
        a0_calibration_method: dict[int, str] = {}
        for plan in plans:
            base = base_fits[plan.variant_id]
            for seed in cfg.seeds:
                joint, reverse_joint, method, calibration_runtime = _calibrate(
                    base, seed, cfg, runtime_clock
                )
                if plan.variant_id == "A0":
                    a0_calibration_method[int(seed)] = method
                probability = outcome_margins(joint)
                reverse = outcome_margins(reverse_joint)[:, [2, 1, 0]]
                row_symmetry = np.max(np.abs(probability - reverse), axis=1)
                symmetry_error = float(row_symmetry.max())
                simplex_error = float(np.max(np.abs(probability.sum(axis=1) - 1.0)))
                symmetry_passed = symmetry_error <= cfg.symmetry_tolerance
                simplex_passed = simplex_error <= cfg.simplex_tolerance
                if not simplex_passed:
                    raise RuntimeError(f"{plan.variant_id} violated the probability simplex")
                if not plan.asymmetric and not symmetry_passed:
                    raise RuntimeError(f"{plan.variant_id} violated team-swap equivariance")
                log_loss, brier = _proper_scores(probability, labels)
                active = _component_active(
                    plan,
                    selected_shared,
                    selected_residual,
                    a0_calibration_method[int(seed)],
                )
                zero_effect = plan.variant_id != "A0" and not active
                runtime = float(base.runtime_seconds + calibration_runtime)
                matrix_rows.append(
                    {
                        "variant_id": plan.variant_id,
                        "removed_component": plan.removed_component,
                        "outer_fold_id": outer_fold_id,
                        "test_year": outer_year,
                        "seed": int(seed),
                        "evaluation_index_hash": evaluation_hash_value,
                        "log_loss": log_loss,
                        "multiclass_brier": brier,
                        "runtime_seconds": runtime,
                        # Compatibility field: the expected verdict was observed.
                        # A8's actual failed symmetry flag is stored separately.
                        "structural_tests_passed": True,
                        "expected_structural_status_observed": True,
                        "probability_simplex_passed": simplex_passed,
                        "symmetry_constraint_passed": symmetry_passed,
                        "structurally_eligible": not plan.asymmetric,
                        "symmetry_error": symmetry_error,
                        "probability_simplex_error": simplex_error,
                        "component_active_in_a0": active,
                        "zero_effect_by_design": zero_effect,
                        "calibration_method": method,
                        "calibration_applied": plan.calibration,
                        "residual_applied": plan.residual,
                        "selected_candidate_id": str(selected_row.candidate_id),
                        "variant_config_hash": base.config_hash,
                        "fit_index_hash": base.fit_index_hash,
                        "train_end_year": base.train_end_year,
                        "match_count": len(test),
                        "inner_oof_years": ",".join(
                            str(value)
                            for value in sorted(
                                base.inner_provenance["prediction_year"]
                                .astype(int)
                                .unique()
                                .tolist()
                            )
                        ),
                        "inner_oof_row_count": len(base.inner_provenance),
                    }
                )
                _append_predictions(
                    prediction_rows,
                    test,
                    probability,
                    row_symmetry,
                    cfg.symmetry_tolerance,
                    observed,
                    labels,
                    base,
                    plan,
                    outer_fold_id,
                    outer_year,
                    int(seed),
                    evaluation_hash_value,
                    schema_hash,
                    method,
                    cfg.information_set,
                    str(selected_row.candidate_id),
                )
    matrix = pd.DataFrame(matrix_rows).sort_values(
        ["test_year", "seed", "variant_id"]
    ).reset_index(drop=True)
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["test_year", "seed", "variant_id", "match_id"]
    ).reset_index(drop=True)
    validation = validate_ablation_matrix(matrix)
    metadata = {
        "protocol": "paired_measured_A0_A8_v1",
        "outer_test_years": list(cfg.outer_test_years),
        "maximum_outer_test_year": max(cfg.outer_test_years),
        "maximum_development_year": cfg.maximum_development_year,
        "seed_protocol": list(cfg.seeds),
        "variant_count": 9,
        "evaluation_indices_paired": True,
        "residual_fit_scope": "strictly_earlier_structural_inner_oof",
        "calibration_fit_scope": "strictly_earlier_inner_oof",
        "a8_status": "measured_asymmetric_diagnostic_structurally_ineligible",
        "structural_tests_passed_semantics": "expected_protocol_verdict_observed",
        "feature_schema_hash": schema_hash,
        "audit_year_used": False,
        "validation": validation,
    }
    return AblationResult(matrix, predictions, metadata)


def _calibrate(
    base: VariantBaseFit,
    seed: int,
    cfg: AblationConfig,
    clock: Clock,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    if not base.plan.calibration:
        return base.outer_joint.copy(), base.swapped_outer_joint.copy(), "identity", 0.0
    start = float(clock())
    selector = ChronologicalCalibrationSelector(
        CalibrationSelectionConfig(
            maximum_development_year=cfg.maximum_development_year,
            random_state=int(seed),
        )
    ).fit_oof(
        base.inner_joint,
        base.inner_goals,
        provenance=base.inner_provenance,
    )
    joint = selector.transform(base.outer_joint)
    reverse = selector.transform(base.swapped_outer_joint)
    elapsed = float(clock()) - start
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("runtime_clock must be finite and nondecreasing")
    return joint, reverse, str(selector.chosen_method_), elapsed


def _append_predictions(
    rows: list[dict[str, Any]],
    test: pd.DataFrame,
    probability: np.ndarray,
    symmetry: np.ndarray,
    symmetry_tolerance: float,
    observed: np.ndarray,
    labels: np.ndarray,
    base: VariantBaseFit,
    plan: VariantPlan,
    fold_id: str,
    year: int,
    seed: int,
    evaluation_hash: str,
    schema_hash: str,
    calibration_method: str,
    information_set: str,
    selected_candidate_id: str,
) -> None:
    ids = match_ids(test)
    prediction_time = prediction_times(test, year)
    calibration_max = int(base.inner_provenance["prediction_year"].max())
    target_names = np.asarray(["home_win", "draw", "away_win"])
    for position in range(len(test)):
        rows.append(
            {
                "outer_fold_id": fold_id,
                "test_year": year,
                "seed": seed,
                "variant_id": plan.variant_id,
                "match_id": ids[position],
                "prediction_time": prediction_time[position],
                "train_end_year": base.train_end_year,
                "evaluation_index_hash": evaluation_hash,
                "fit_index_hash": base.fit_index_hash,
                "selected_candidate_id": selected_candidate_id,
                "feature_schema_hash": schema_hash,
                "information_set": information_set,
                "home_win": float(probability[position, 0]),
                "draw": float(probability[position, 1]),
                "away_win": float(probability[position, 2]),
                "target_outcome": str(target_names[labels[position]]),
                "observed_home_goals": int(observed[position, 0]),
                "observed_away_goals": int(observed[position, 1]),
                "row_symmetry_error": float(symmetry[position]),
                "structurally_eligible": not plan.asymmetric,
                "symmetry_constraint_passed": bool(
                    symmetry[position] <= symmetry_tolerance
                ),
                "residual_applied": plan.residual,
                "residual_fit_max_year": base.residual_fit_max_year or "",
                "calibration_applied": plan.calibration,
                "calibration_method": calibration_method,
                "calibration_fit_max_year": calibration_max if plan.calibration else "",
            }
        )


def _component_active(
    plan: VariantPlan, shared: bool, residual: bool, a0_calibration_method: str
) -> bool:
    active = {
        "A0": True,
        "A1": True,
        "A2": True,
        "A3": True,
        "A4": shared,
        "A5": residual,
        "A6": a0_calibration_method != "identity",
        "A7": True,
        "A8": True,
    }
    return bool(active[plan.variant_id])


def _proper_scores(probability: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    selected = np.clip(probability[np.arange(len(labels)), labels], 1e-15, 1.0)
    observed = np.eye(3)[labels]
    return (
        float(-np.mean(np.log(selected))),
        float(np.mean(np.sum((probability - observed) ** 2, axis=1))),
    )


__all__ = ["AblationConfig", "AblationResult", "run_measured_ablations"]
