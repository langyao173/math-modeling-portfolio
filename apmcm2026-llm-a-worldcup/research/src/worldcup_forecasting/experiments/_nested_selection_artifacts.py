"""Leakage-auditable calibration and outer-prediction artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from worldcup_forecasting.models.advanced_count import (
    EquivariantCountConfig,
    EquivariantCountModel,
    joint_batch_from_rates,
)
from worldcup_forecasting.validation.nested import (
    InnerTournamentFold,
    NestedTournamentFold,
)

from ._nested_selection_residual import fit_residual, residual_design
from ._nested_selection_types import (
    CandidateSpec,
    Clock,
    NestedSelectionConfig,
    StructuralPrediction,
    elapsed_seconds,
    full_config_hash,
    inner_fit_hash,
    match_ids,
    observed_goals,
    prediction_times,
    sha256_payload,
)


def selected_inner_artifacts(
    frame: pd.DataFrame,
    selection_outer_fold_id: str,
    spec: CandidateSpec,
    model_config: EquivariantCountConfig,
    structural: Sequence[StructuralPrediction],
    evaluation_folds: Sequence[InnerTournamentFold],
    schema_hash: str,
    cfg: NestedSelectionConfig,
    clock: Clock,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Emit selected-candidate inner OOF rows and score PMFs for calibration."""
    by_year = {item.validation_year: item for item in structural}
    design, even, odd = residual_design(frame, model_config)
    identities = match_ids(frame)
    years = pd.to_numeric(frame[cfg.year_column]).to_numpy(int)
    ledgers: list[pd.DataFrame] = []
    grids: list[pd.DataFrame] = []
    for fold in evaluation_folds:
        prediction = by_year[fold.validation_year]
        prior = [item for item in structural if item.validation_year < fold.validation_year]
        rates = prediction.rates
        residual_applied = False
        if spec.residual and prior:
            corrector, _ = fit_residual(frame, design, prior, even, odd, cfg, clock)
            rates = corrector.transform_rates(
                design.iloc[list(prediction.validation_index)], prediction.rates
            )
            residual_applied = True
        batch = joint_batch_from_rates(
            rates,
            shared_fraction=prediction.shared_fraction,
            max_score=model_config.max_score,
            min_retained_mass=model_config.min_retained_mass,
            max_support=model_config.max_support,
        )
        outcome = batch.outcome_probabilities()
        index = np.asarray(prediction.validation_index, dtype=int)
        observed = observed_goals(frame.iloc[index]).astype(int)
        target = np.where(
            observed[:, 0] > observed[:, 1],
            "home_win",
            np.where(observed[:, 0] == observed[:, 1], "draw", "away_win"),
        )
        calibration_id = f"{selection_outer_fold_id}__calibration_inner_{fold.validation_year}"
        fit_hash = inner_fit_hash(spec, model_config, cfg, prediction, prior)
        ledgers.append(
            pd.DataFrame(
                {
                    "match_id": identities[index],
                    "prediction_year": years[index],
                    "prediction_time": prediction_times(frame.iloc[index], cfg.year_column),
                    "train_end_year": prediction.train_end_year,
                    "outer_fold_id": calibration_id,
                    "inner_fold_id": fold.fold_id,
                    "candidate_id": spec.candidate_id,
                    "model_id": _model_id(spec),
                    "seed": cfg.seed,
                    "information_set": cfg.information_set,
                    "fit_index_hash": fit_hash,
                    "feature_schema_hash": schema_hash,
                    "home_win": outcome[:, 0],
                    "draw": outcome[:, 1],
                    "away_win": outcome[:, 2],
                    "target_outcome": target,
                    "observed_home_goals": observed[:, 0],
                    "observed_away_goals": observed[:, 1],
                    "is_oof": True,
                    "selection_outer_fold_id": selection_outer_fold_id,
                    "config_hash": full_config_hash(model_config, spec, cfg),
                    "prediction_role": "selected_inner_oof_for_calibration",
                    "residual_applied": residual_applied,
                    "residual_fit_max_year": (
                        max(item.validation_year for item in prior)
                        if residual_applied
                        else ""
                    ),
                }
            )
        )
        grid = long_joint_grid(batch.probability, identities[index], calibration_id)
        grid.insert(0, "selection_outer_fold_id", selection_outer_fold_id)
        grid["inner_fold_id"] = fold.fold_id
        grid["candidate_id"] = spec.candidate_id
        grid["fit_index_hash"] = fit_hash
        grids.append(grid)
    return pd.concat(ledgers, ignore_index=True), pd.concat(grids, ignore_index=True)


def deploy_outer(
    frame: pd.DataFrame,
    outer: NestedTournamentFold,
    spec: CandidateSpec,
    model_config: EquivariantCountConfig,
    structural: Sequence[StructuralPrediction],
    schema_hash: str,
    cfg: NestedSelectionConfig,
    clock: Clock,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Refit the selected model and emit strict outer OOF evidence."""
    train = np.asarray(outer.train_index, dtype=int)
    valid = np.asarray(outer.validation_index, dtype=int)
    start = clock()
    model = EquivariantCountModel(model_config).fit(frame.iloc[train])
    base_rates = model.predict_rates(frame.iloc[valid])
    rates = base_rates.copy()
    residual_ids: list[Any] = []
    if spec.residual and structural:
        design, even, odd = residual_design(frame, model_config)
        corrector, _ = fit_residual(frame, design, structural, even, odd, cfg, clock)
        rates = corrector.transform_rates(design.iloc[valid], base_rates)
        residual_index = np.concatenate(
            [np.asarray(item.validation_index) for item in structural]
        )
        residual_ids = match_ids(frame)[residual_index].tolist()
    runtime = elapsed_seconds(clock, start)
    batch = joint_batch_from_rates(
        rates,
        shared_fraction=float(model.shared_fraction_),
        max_score=model_config.max_score,
        min_retained_mass=model_config.min_retained_mass,
        max_support=model_config.max_support,
    )
    probabilities = batch.outcome_probabilities()
    years = pd.to_numeric(frame[cfg.year_column]).to_numpy(int)
    train_end = int(years[train].max())
    identities = match_ids(frame)
    fit_hash = sha256_payload(
        {
            "candidate_id": spec.candidate_id,
            "config_hash": full_config_hash(model_config, spec, cfg),
            "structural_train_match_ids": identities[train].tolist(),
            "residual_oof_match_ids": residual_ids,
        }
    )
    observed = observed_goals(frame.iloc[valid]).astype(int)
    target = np.where(
        observed[:, 0] > observed[:, 1],
        "home_win",
        np.where(observed[:, 0] == observed[:, 1], "draw", "away_win"),
    )
    ledger = pd.DataFrame(
        {
            "match_id": identities[valid],
            "prediction_year": years[valid],
            "prediction_time": prediction_times(frame.iloc[valid], cfg.year_column),
            "train_end_year": train_end,
            "outer_fold_id": outer.fold_id,
            "inner_fold_id": "deploy",
            "candidate_id": spec.candidate_id,
            "model_id": _model_id(spec),
            "seed": cfg.seed,
            "information_set": cfg.information_set,
            "fit_index_hash": fit_hash,
            "feature_schema_hash": schema_hash,
            "home_win": probabilities[:, 0],
            "draw": probabilities[:, 1],
            "away_win": probabilities[:, 2],
            "target_outcome": target,
            "observed_home_goals": observed[:, 0],
            "observed_away_goals": observed[:, 1],
            "is_oof": True,
            "config_hash": full_config_hash(model_config, spec, cfg),
            "residual_applied": spec.residual,
        }
    )
    rates_frame = ledger[
        ["outer_fold_id", "match_id", "candidate_id", "config_hash", "fit_index_hash"]
    ].copy()
    rates_frame["base_home_rate"], rates_frame["base_away_rate"] = base_rates.T
    rates_frame["expected_home_goals"], rates_frame["expected_away_goals"] = rates.T
    rates_frame["shared_fraction"] = float(model.shared_fraction_)
    rates_frame["residual_applied"] = spec.residual
    rates_frame = rates_frame[
        [
            "outer_fold_id", "match_id", "expected_home_goals", "expected_away_goals",
            "candidate_id", "config_hash", "fit_index_hash", "base_home_rate",
            "base_away_rate", "shared_fraction", "residual_applied",
        ]
    ]
    grid = long_joint_grid(batch.probability, identities[valid], outer.fold_id)
    deployment = {
        "outer_fit_index_hash": fit_hash,
        "outer_fit_runtime_seconds": runtime,
        "outer_train_end_year": train_end,
        "outer_prediction_rows": len(valid),
    }
    return ledger, rates_frame, grid, deployment


def long_joint_grid(
    probability: np.ndarray, match_identity: np.ndarray, fold_id: str
) -> pd.DataFrame:
    """Convert dense score grids to a deterministic long-form artifact."""
    support = probability.shape[1]
    home = np.repeat(np.arange(support), support)
    away = np.tile(np.arange(support), support)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "outer_fold_id": fold_id,
                    "match_id": match_id,
                    "home_goals": home,
                    "away_goals": away,
                    "probability": probability[row].ravel(),
                }
            )
            for row, match_id in enumerate(match_identity)
        ],
        ignore_index=True,
    )


def _model_id(spec: CandidateSpec) -> str:
    return "equivariant_count_residual" if spec.residual else "equivariant_count"


__all__ = ["deploy_outer", "long_joint_grid", "selected_inner_artifacts"]
