"""Exact final-fit bridge from frozen development selection to later audits."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from worldcup_forecasting.models.advanced_count import (
    EquivariantCountConfig,
    EquivariantCountModel,
    joint_batch_from_rates,
)
from worldcup_forecasting.models.count import JointGoalBatch
from worldcup_forecasting.models.residuals import EquivariantResidualCorrector

from ._nested_selection_fit import (
    development_inner_folds,
    ordered_frame,
    structural_predictions,
    validate_development_boundary,
    validate_inputs,
)
from ._nested_selection_residual import fit_residual, residual_design
from ._nested_selection_types import (
    CandidateSpec,
    Clock,
    FinalDevelopmentSelectionResult,
    StructuralPrediction,
    elapsed_seconds,
    frame_content_hash,
    full_config_hash,
    inner_fit_hash,
    match_ids,
    sha256_payload,
)


@dataclass(frozen=True, slots=True)
class FinalFittedCountPredictor:
    """Immutable serving wrapper for one audited development-to-test lineage."""

    selected_spec: CandidateSpec
    model_config: EquivariantCountConfig
    selected_config_hash: str
    final_fit_hash: str
    development_frame_hash: str
    fit_end_year: int
    residual_oof_max_year: int | None
    fit_runtime_seconds: float
    _model: EquivariantCountModel
    _residual: EquivariantResidualCorrector | None

    @property
    def config(self) -> EquivariantCountConfig:
        """Frozen structural configuration expected by scenario/stress consumers."""
        return self.model_config

    @property
    def home_team_column_(self) -> str | None:
        return self._model.home_team_column_

    @property
    def away_team_column_(self) -> str | None:
        return self._model.away_team_column_

    @property
    def year_column_(self) -> str | None:
        return self._model.year_column_

    @property
    def stage_column_(self) -> str | None:
        return self._model.stage_column_

    @property
    def host_side_column_(self) -> str | None:
        return self._model.host_side_column_

    @property
    def paired_columns_(self) -> tuple[tuple[str, str], ...]:
        return self._model.paired_columns_

    @property
    def odd_columns_(self) -> tuple[str, ...]:
        return self._model.odd_columns_

    @property
    def even_columns_(self) -> tuple[str, ...]:
        return self._model.even_columns_

    @property
    def context_columns_(self) -> tuple[str, ...]:
        return self._model.context_columns_

    @property
    def shared_fraction_(self) -> float:
        return float(self._model.shared_fraction_)

    @property
    def residual_applied(self) -> bool:
        """Whether the frozen selected specification includes correction."""
        return self._residual is not None

    @property
    def metadata(self) -> dict[str, object]:
        """Return exact selection, fit, and residual provenance."""
        return {
            "candidate_id": self.selected_spec.candidate_id,
            "selected_config_hash": self.selected_config_hash,
            "final_fit_hash": self.final_fit_hash,
            "development_frame_hash": self.development_frame_hash,
            "fit_end_year": self.fit_end_year,
            "residual_applied": self.residual_applied,
            "residual_oof_max_year": self.residual_oof_max_year,
            "shared_fraction": float(self._model.shared_fraction_),
            "fit_runtime_seconds": self.fit_runtime_seconds,
        }

    def predict_rates(self, frame: pd.DataFrame) -> np.ndarray:
        """Serve positive paired rates for rows strictly after development."""
        self._validate_later_frame(frame)
        base_rates = self._model.predict_rates(frame)
        if self._residual is None:
            return base_rates
        design, _, _ = residual_design(frame, self.model_config)
        return self._residual.transform_rates(design, base_rates)

    def predict_joint(self, frame: pd.DataFrame) -> JointGoalBatch:
        """Serve the coherent joint score PMF from the final corrected rates."""
        batch = joint_batch_from_rates(
            self.predict_rates(frame),
            shared_fraction=float(self._model.shared_fraction_),
            max_score=self.model_config.max_score,
            min_retained_mass=self.model_config.min_retained_mass,
            max_support=self.model_config.max_support,
        )
        return JointGoalBatch(
            probability=batch.probability,
            support_max=batch.support_max,
            retained_mass=batch.retained_mass,
            expected_home_goals=batch.expected_home_goals,
            expected_away_goals=batch.expected_away_goals,
            model_metadata={**batch.model_metadata, **self.metadata},
        )

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Serve [home win, draw, away win] probabilities from the joint PMF."""
        return self.predict_joint(frame).outcome_probabilities()

    def _validate_later_frame(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("Final prediction requires a nonempty frame")
        year_column = self.model_config.year_column
        if year_column not in frame:
            raise ValueError(f"Missing year column: {year_column}")
        years = pd.to_numeric(frame[year_column], errors="raise").to_numpy(float)
        valid = np.isfinite(years).all() and np.allclose(years, np.rint(years))
        if not valid or np.any(years <= self.fit_end_year):
            raise ValueError("Final predictor only serves editions after its fit cutoff")


def fit_final_selected_model(
    development_frame: pd.DataFrame,
    selection: FinalDevelopmentSelectionResult,
    *,
    runtime_clock: Clock = perf_counter,
) -> FinalFittedCountPredictor:
    """Fit the exact selected structure/residual lineage on all development rows."""
    cfg = selection.selection_config
    validate_development_boundary(development_frame, cfg)
    validate_inputs(development_frame, cfg)
    frame = ordered_frame(development_frame, cfg)
    observed_hash = frame_content_hash(frame)
    expected_hash = selection.metadata.get("development_frame_hash")
    if observed_hash != expected_hash:
        raise ValueError("Development frame differs from the frozen selection frame")
    if selection.selected_candidate_hash != selection.selected_spec.candidate_id:
        raise ValueError("Selected candidate identity is internally inconsistent")
    config_digest = full_config_hash(
        selection.selected_model_config, selection.selected_spec, cfg
    )
    if config_digest != selection.selected_config_hash:
        raise ValueError("Selected model configuration hash is inconsistent")

    all_folds = development_inner_folds(frame, cfg)
    folds = all_folds[-cfg.latest_inner_folds :]
    start = runtime_clock()
    structural = structural_predictions(
        frame,
        folds,
        selection.selected_model_config,
        cfg,
        {},
        runtime_clock,
    )
    lineage_hashes = _validate_selection_lineage(frame, selection, structural)
    residual: EquivariantResidualCorrector | None = None
    if selection.selected_spec.residual:
        design, even, odd = residual_design(frame, selection.selected_model_config)
        residual, _ = fit_residual(
            frame, design, structural, even, odd, cfg, runtime_clock
        )
    model = EquivariantCountModel(selection.selected_model_config).fit(frame)
    fit_end_year = int(pd.to_numeric(frame[cfg.year_column]).max())
    final_hash = sha256_payload(
        {
            "candidate_budget_manifest_hash": selection.metadata[
                "candidate_budget_manifest_hash"
            ],
            "candidate_id": selection.selected_candidate_hash,
            "config_hash": config_digest,
            "development_frame_hash": observed_hash,
            "structural_training_match_ids": match_ids(frame).tolist(),
            "selection_inner_fit_hashes": lineage_hashes,
            "residual_oof_fit_hashes": (
                lineage_hashes if selection.selected_spec.residual else []
            ),
        }
    )
    return FinalFittedCountPredictor(
        selected_spec=selection.selected_spec,
        model_config=selection.selected_model_config,
        selected_config_hash=config_digest,
        final_fit_hash=final_hash,
        development_frame_hash=observed_hash,
        fit_end_year=fit_end_year,
        residual_oof_max_year=(
            max(item.validation_year for item in structural)
            if selection.selected_spec.residual
            else None
        ),
        fit_runtime_seconds=elapsed_seconds(runtime_clock, start),
        _model=model,
        _residual=residual,
    )


def _validate_selection_lineage(
    frame: pd.DataFrame,
    selection: FinalDevelopmentSelectionResult,
    structural: tuple[StructuralPrediction, ...],
) -> list[str]:
    cfg = selection.selection_config
    expected: dict[int, str] = {}
    for prediction in structural:
        prior = [item for item in structural if item.validation_year < prediction.validation_year]
        expected[prediction.validation_year] = inner_fit_hash(
            selection.selected_spec,
            selection.selected_model_config,
            cfg,
            prediction,
            prior,
        )
    ledger = selection.selected_inner_oof_ledger
    observed = ledger.groupby("prediction_year")["fit_index_hash"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    valid = set(observed.index) == set(expected) and all(
        observed[year] == (fit_hash,) for year, fit_hash in expected.items()
    )
    if not valid:
        raise ValueError("Recomputed structural OOF lineage differs from selection")
    if frame_content_hash(frame) != selection.metadata["development_frame_hash"]:
        raise ValueError("Development content changed during final-lineage validation")
    return [expected[year] for year in sorted(expected)]


__all__ = ["FinalFittedCountPredictor", "fit_final_selected_model"]
