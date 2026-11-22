"""Frozen nested and final-development selection for the count-model family."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from worldcup_forecasting.models.advanced_count import EquivariantCountConfig
from worldcup_forecasting.validation.nested import (
    InnerTournamentFold,
    nested_expanding_origin_folds,
)
from worldcup_forecasting.validation.research_contracts import validate_oof_frame

from ._nested_selection_artifacts import deploy_outer, selected_inner_artifacts
from ._nested_selection_final import (
    FinalFittedCountPredictor,
    fit_final_selected_model,
)
from ._nested_selection_fit import (
    development_inner_folds,
    evaluate_spec,
    failed_candidate,
    frame_schema_hash,
    frozen_candidate_specs,
    ordered_frame,
    structural_predictions,
    validate_development_boundary,
    validate_inputs,
)
from ._nested_selection_runtime import (
    REFERENCE_RUNTIME_HASH,
    REFERENCE_RUNTIME_PROTOCOL,
)
from ._nested_selection_types import (
    CandidateSpec,
    Clock,
    CountConfigFactory,
    FinalDevelopmentSelectionResult,
    NestedSelectionConfig,
    NestedSelectionResult,
    StructuralPrediction,
    frame_content_hash,
    full_config_hash,
    validate_candidate_config,
)
from .research_features import config_hash as structural_config_hash
from .research_features import make_count_config


@dataclass(frozen=True)
class _CandidateSearch:
    selected_spec: CandidateSpec
    selected_model_config: EquivariantCountConfig
    structural: tuple[StructuralPrediction, ...]
    selected_summary: dict[str, Any]
    candidate_rows: list[dict[str, Any]]
    fold_rows: list[dict[str, Any]]


def run_nested_selection(
    frame: pd.DataFrame,
    *,
    config: NestedSelectionConfig | None = None,
    feature_schema_hash: str | None = None,
    count_config_factory: CountConfigFactory = make_count_config,
    runtime_clock: Clock = perf_counter,
) -> NestedSelectionResult:
    """Select and deploy one specification per pre-2014 outer tournament."""
    cfg = config or NestedSelectionConfig()
    validate_development_boundary(frame, cfg)
    validate_inputs(frame, cfg)
    frame = ordered_frame(frame, cfg)
    schema_hash = _resolve_schema_hash(frame, feature_schema_hash)
    folds = nested_expanding_origin_folds(
        frame,
        cfg.outer_test_years,
        year_column=cfg.year_column,
        minimum_inner_folds=1,
    )
    specs, budget_hash = frozen_candidate_specs()
    cache: dict[tuple[str, int], StructuralPrediction] = {}
    aggregate_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    inner_oof_parts: list[pd.DataFrame] = []
    inner_grid_parts: list[pd.DataFrame] = []
    oof_parts: list[pd.DataFrame] = []
    rate_parts: list[pd.DataFrame] = []
    grid_parts: list[pd.DataFrame] = []

    for outer in folds:
        evaluation_folds = outer.inner_folds[-cfg.latest_inner_folds :]
        search = _search_candidate_family(
            frame,
            outer.fold_id,
            evaluation_folds,
            specs,
            cfg,
            count_config_factory,
            cache,
            runtime_clock,
        )
        aggregate_rows.extend(search.candidate_rows)
        fold_rows.extend(search.fold_rows)
        inner_oof, inner_grid = selected_inner_artifacts(
            frame,
            outer.fold_id,
            search.selected_spec,
            search.selected_model_config,
            search.structural,
            evaluation_folds,
            schema_hash,
            cfg,
            runtime_clock,
        )
        ledger, rates, grid, deployment = deploy_outer(
            frame,
            outer,
            search.selected_spec,
            search.selected_model_config,
            search.structural,
            schema_hash,
            cfg,
            runtime_clock,
        )
        selected_rows.append(
            {**search.selected_summary, **deployment, "selection_rank": 1}
        )
        inner_oof_parts.append(inner_oof)
        inner_grid_parts.append(inner_grid)
        oof_parts.append(ledger)
        rate_parts.append(rates)
        grid_parts.append(grid)

    oof = pd.concat(oof_parts, ignore_index=True)
    inner_oof = pd.concat(inner_oof_parts, ignore_index=True)
    validate_oof_frame(oof)
    validate_oof_frame(inner_oof)
    return NestedSelectionResult(
        selected_specs=_sorted_frame(selected_rows, ["outer_test_year"]),
        candidate_metrics=_sorted_frame(
            aggregate_rows, ["outer_test_year", "candidate_order"]
        ),
        inner_fold_metrics=_sorted_frame(
            fold_rows, ["outer_test_year", "candidate_order", "inner_test_year"]
        ),
        selected_inner_oof_ledger=inner_oof,
        selected_inner_joint_grid=pd.concat(inner_grid_parts, ignore_index=True),
        oof_ledger=oof,
        rate_ledger=pd.concat(rate_parts, ignore_index=True),
        joint_grid=pd.concat(grid_parts, ignore_index=True),
        metadata={
            **_selection_metadata(cfg, budget_hash),
            "outer_test_years": list(sorted(cfg.outer_test_years)),
        },
    )


def select_final_development_spec(
    frame: pd.DataFrame,
    *,
    latest_inner_folds: int = 3,
    year_column: str = "Year",
    feature_schema_hash: str | None = None,
    count_config_factory: CountConfigFactory = make_count_config,
    runtime_clock: Clock = perf_counter,
    seed: int = 31415,
    information_set: str = "pre_match",
) -> FinalDevelopmentSelectionResult:
    """Choose a later-audit specification from the latest editions at/before 2010.

    The boundary check is deliberately the first data-dependent operation.  An
    input containing 2014, 2018, or any post-2010 row is rejected rather than
    silently filtered, so audit-edition outcomes cannot enter hashes or fitting.
    """
    cfg = NestedSelectionConfig(
        outer_test_years=(2010,),
        year_column=year_column,
        latest_inner_folds=latest_inner_folds,
        seed=seed,
        information_set=information_set,
    )
    validate_development_boundary(frame, cfg)
    validate_inputs(frame, cfg)
    frame = ordered_frame(frame, cfg)
    folds = development_inner_folds(frame, cfg)
    evaluation_folds = folds[-cfg.latest_inner_folds :]
    schema_hash = _resolve_schema_hash(frame, feature_schema_hash)
    specs, budget_hash = frozen_candidate_specs()
    selection_year = max(fold.validation_year for fold in folds)
    selection_id = f"final_development_{selection_year}"
    search = _search_candidate_family(
        frame,
        selection_id,
        evaluation_folds,
        specs,
        cfg,
        count_config_factory,
        {},
        runtime_clock,
    )
    inner_oof, inner_grid = selected_inner_artifacts(
        frame,
        selection_id,
        search.selected_spec,
        search.selected_model_config,
        search.structural,
        evaluation_folds,
        schema_hash,
        cfg,
        runtime_clock,
    )
    validate_oof_frame(inner_oof)
    config_digest = full_config_hash(
        search.selected_model_config, search.selected_spec, cfg
    )
    selected_summary = pd.DataFrame(
        [{**search.selected_summary, "selection_rank": 1}]
    )
    return FinalDevelopmentSelectionResult(
        selection_config=cfg,
        selected_spec=search.selected_spec,
        selected_model_config=search.selected_model_config,
        selected_candidate_hash=search.selected_spec.candidate_id,
        selected_config_hash=config_digest,
        selected_summary=selected_summary,
        candidate_metrics=_sorted_frame(search.candidate_rows, ["candidate_order"]),
        inner_fold_metrics=_sorted_frame(
            search.fold_rows, ["candidate_order", "inner_test_year"]
        ),
        selected_inner_oof_ledger=inner_oof,
        selected_inner_joint_grid=inner_grid,
        metadata={
            **_selection_metadata(cfg, budget_hash),
            "development_cutoff_year": selection_year,
            "development_frame_hash": frame_content_hash(frame),
            "feature_schema_hash": schema_hash,
            "selection_id": selection_id,
            "selection_inner_test_years": [
                fold.validation_year for fold in evaluation_folds
            ],
            "audit_rows_loaded": False,
        },
    )


def _search_candidate_family(
    frame: pd.DataFrame,
    selection_id: str,
    evaluation_folds: Sequence[InnerTournamentFold],
    specs: Sequence[CandidateSpec],
    cfg: NestedSelectionConfig,
    count_config_factory: CountConfigFactory,
    cache: dict[tuple[str, int], StructuralPrediction],
    runtime_clock: Clock,
) -> _CandidateSearch:
    summaries: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    resolved_configs: dict[str, EquivariantCountConfig] = {}
    structural_identity_by_key: dict[tuple[float, float | None, bool], str] = {}
    for spec in specs:
        model_config: EquivariantCountConfig | None = None
        try:
            model_config = count_config_factory(
                alpha=spec.alpha,
                half_life_years=spec.half_life_years,
                shared_score=spec.shared_score,
            )
            validate_candidate_config(spec, model_config)
            structural_key = (
                spec.alpha,
                spec.half_life_years,
                spec.shared_score,
            )
            identity = structural_config_hash(model_config)
            previous_identity = structural_identity_by_key.setdefault(
                structural_key, identity
            )
            if identity != previous_identity:
                raise ValueError(
                    "Count-config factory is nondeterministic for one structural spec"
                )
            resolved_configs[spec.candidate_id] = model_config
            structural = structural_predictions(
                frame, evaluation_folds, model_config, cfg, cache, runtime_clock
            )
            rows, summary = evaluate_spec(
                frame,
                selection_id,
                spec,
                model_config,
                structural,
                evaluation_folds,
                cfg,
                runtime_clock,
            )
        except Exception as error:
            rows, summary = failed_candidate(
                selection_id, spec, model_config, evaluation_folds, cfg, error
            )
        fold_rows.extend(rows)
        aggregate_rows.append(summary)
        summaries.append(summary)
    eligible = [
        row
        for row in summaries
        if row["eligible"] and np.isfinite(row["selection_objective"])
    ]
    if not eligible:
        raise RuntimeError(f"Every candidate failed inside {selection_id}")
    selected = min(
        eligible, key=lambda row: (row["selection_objective"], row["candidate_id"])
    )
    selected_spec = next(
        spec for spec in specs if spec.candidate_id == selected["candidate_id"]
    )
    model_config = resolved_configs[selected_spec.candidate_id]
    structural = structural_predictions(
        frame, evaluation_folds, model_config, cfg, cache, runtime_clock
    )
    return _CandidateSearch(
        selected_spec=selected_spec,
        selected_model_config=model_config,
        structural=structural,
        selected_summary=selected,
        candidate_rows=aggregate_rows,
        fold_rows=fold_rows,
    )


def _resolve_schema_hash(frame: pd.DataFrame, supplied: str | None) -> str:
    schema_hash = supplied or frame_schema_hash(frame)
    nonhex = any(
        character not in "0123456789abcdef" for character in schema_hash.lower()
    )
    if len(schema_hash) != 64 or nonhex:
        raise ValueError("feature_schema_hash must be a 64-character SHA-256 digest")
    return schema_hash


def _selection_metadata(
    cfg: NestedSelectionConfig, budget_hash: str
) -> dict[str, Any]:
    return {
        "candidate_count": 36,
        "candidate_budget_manifest_hash": budget_hash,
        "maximum_selection_year": cfg.maximum_selection_year,
        "forbidden_audit_years": list(cfg.forbidden_audit_years),
        "latest_inner_folds": cfg.latest_inner_folds,
        "year_column": cfg.year_column,
        "seed": cfg.seed,
        "information_set": cfg.information_set,
        "selection_objective": "LL+.25*Brier+.05*ECE+.05*SD+.001*log1p(t)+.0005*C",
        "residual_fit_scope": "strictly_earlier_structural_inner_oof",
        "calibration_fit_source": "selected_candidate_latest_inner_oof_only",
        "runtime_reference_hash": REFERENCE_RUNTIME_HASH,
        "runtime_benchmark_protocol": REFERENCE_RUNTIME_PROTOCOL,
        "live_runtime_role": "diagnostic_only_not_used_for_candidate_ranking",
    }


def _sorted_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(columns).reset_index(drop=True)


__all__ = [
    "CandidateSpec",
    "FinalDevelopmentSelectionResult",
    "FinalFittedCountPredictor",
    "NestedSelectionConfig",
    "NestedSelectionResult",
    "frozen_candidate_specs",
    "fit_final_selected_model",
    "run_nested_selection",
    "select_final_development_spec",
]
