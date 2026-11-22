"""Deterministic runtime references for manifest-bound scientific evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import pandas as pd

from worldcup_forecasting.experiments._nested_selection_runtime import (
    reference_runtime_seconds,
)

LIVE_RUNTIME_COLUMNS = ("measured_runtime_seconds", "outer_fit_runtime_seconds")
BASELINE_REFERENCE_RUNTIME_SECONDS = {
    "official_outcome_logit": 0.01,
    "independent_mirrored_poisson": 0.03,
    "legacy_mirrored_dixon_coles": 0.03,
}
ABLATION_REFERENCE_RUNTIME_SECONDS = {
    "A0": 0.7,
    "A1": 0.6,
    "A2": 0.6,
    "A3": 0.6,
    "A4": 0.7,
    "A5": 0.7,
    "A6": 0.6,
    "A7": 14.0,
    "A8": 0.6,
}
RUNTIME_REFERENCE_PROTOCOL = {
    "schema_version": 1,
    "semantics": "frozen_reference_benchmark_not_live_stage_wall_clock",
    "selection_reference": "existing_candidate_runtime_reference_hash",
    "baseline_reference_seconds": BASELINE_REFERENCE_RUNTIME_SECONDS,
    "ablation_reference_seconds": ABLATION_REFERENCE_RUNTIME_SECONDS,
    "quantization_seconds": {"baseline": 0.01, "ablation": 0.1},
    "source": "preformal_development_only_diagnostic_on_declared_environment",
    "operational_wall_clock": "returned_to_caller_only_not_manifest_bound",
    "live_runtime_columns_excluded": list(LIVE_RUNTIME_COLUMNS),
}
RUNTIME_REFERENCE_PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(
        RUNTIME_REFERENCE_PROTOCOL,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def drop_live_runtime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove wall-clock observations from manifest-bound scientific tables."""
    return frame.drop(columns=list(LIVE_RUNTIME_COLUMNS), errors="ignore").copy()


def replace_runtime_with_reference(
    frame: pd.DataFrame,
    *,
    identity_column: str,
    charges: dict[str, float],
) -> pd.DataFrame:
    """Replace volatile timings with a predeclared reference benchmark charge."""
    if identity_column not in frame or "runtime_seconds" not in frame:
        raise ValueError("Runtime reference normalization requires identity and runtime columns")
    identities = frame[identity_column].astype(str)
    missing = sorted(set(identities).difference(charges))
    if missing:
        raise ValueError(f"Runtime reference is missing identities: {missing}")
    result = frame.copy()
    result["runtime_seconds"] = identities.map(charges).astype(float)
    return result


def seal_nested_runtime_evidence(
    nested: Any,
    final_selection: Any,
    fitted: Any,
) -> tuple[Any, Any, Any]:
    """Return science-equivalent nested objects with no live wall-clock state."""
    nested = replace(
        nested,
        selected_specs=drop_live_runtime_columns(nested.selected_specs),
        candidate_metrics=drop_live_runtime_columns(nested.candidate_metrics),
    )
    final_selection = replace(
        final_selection,
        selected_summary=drop_live_runtime_columns(final_selection.selected_summary),
        candidate_metrics=drop_live_runtime_columns(final_selection.candidate_metrics),
    )
    fitted = replace(
        fitted,
        fit_runtime_seconds=reference_runtime_seconds(fitted.selected_spec.candidate_id),
    )
    return nested, final_selection, fitted


def seal_baseline_runtime_evidence(baseline: Any) -> Any:
    """Replace baseline wall-clock observations by frozen reference charges."""
    metadata = {
        **baseline.metadata,
        "runtime_semantics": RUNTIME_REFERENCE_PROTOCOL["semantics"],
        "runtime_reference_protocol_sha256": RUNTIME_REFERENCE_PROTOCOL_SHA256,
    }
    return replace(
        baseline,
        outcome_fold_metrics=replace_runtime_with_reference(
            baseline.outcome_fold_metrics,
            identity_column="model",
            charges=BASELINE_REFERENCE_RUNTIME_SECONDS,
        ),
        joint_fold_metrics=replace_runtime_with_reference(
            baseline.joint_fold_metrics,
            identity_column="model",
            charges=BASELINE_REFERENCE_RUNTIME_SECONDS,
        ),
        metadata=metadata,
    )


def seal_ablation_runtime_evidence(result: Any) -> Any:
    """Replace ablation wall-clock observations by frozen reference charges."""
    metadata = {
        **result.metadata,
        "runtime_semantics": RUNTIME_REFERENCE_PROTOCOL["semantics"],
        "runtime_reference_protocol_sha256": RUNTIME_REFERENCE_PROTOCOL_SHA256,
    }
    return replace(
        result,
        matrix=replace_runtime_with_reference(
            result.matrix,
            identity_column="variant_id",
            charges=ABLATION_REFERENCE_RUNTIME_SECONDS,
        ),
        metadata=metadata,
    )


__all__ = [
    "ABLATION_REFERENCE_RUNTIME_SECONDS",
    "BASELINE_REFERENCE_RUNTIME_SECONDS",
    "LIVE_RUNTIME_COLUMNS",
    "RUNTIME_REFERENCE_PROTOCOL",
    "RUNTIME_REFERENCE_PROTOCOL_SHA256",
    "drop_live_runtime_columns",
    "replace_runtime_with_reference",
    "seal_ablation_runtime_evidence",
    "seal_baseline_runtime_evidence",
    "seal_nested_runtime_evidence",
]
