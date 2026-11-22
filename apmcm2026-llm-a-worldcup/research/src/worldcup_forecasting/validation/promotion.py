"""Frozen, multi-axis promotion gate for the advanced research system."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_METRICS = ("log_loss", "multiclass_brier")
_EXPECTED_YEARS = (1986, 1990, 1994, 1998, 2002, 2006, 2010)
_REQUIRED_STRUCTURAL_CHECKS = frozenset(("simplex", "symmetry", "leakage"))


class PromotionRejectedError(RuntimeError):
    """Raised after a complete gate evaluation rejects audit promotion."""


def evaluate_advanced_promotion(
    fold_metrics: pd.DataFrame,
    decision_metrics: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    model_identity_hash: str,
    artifact_identity_hash: str,
    structural_checks: Mapping[str, bool],
    expected_years: Sequence[int] = _EXPECTED_YEARS,
    non_degradation_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Evaluate prediction, decision, structure, identity, and cutoff together."""
    years = tuple(int(year) for year in expected_years)
    _validate_fold_metrics(fold_metrics, candidate, baseline, years)
    decision_metrics = normalize_decision_metrics(decision_metrics)
    _validate_decision_metrics(decision_metrics, years)
    if not structural_checks:
        raise ValueError("At least one executable structural check is required")
    missing_structural = sorted(_REQUIRED_STRUCTURAL_CHECKS.difference(structural_checks))
    if missing_structural:
        raise ValueError(
            f"Frozen structural checks are missing required axes: {missing_structural}"
        )
    if any(not isinstance(value, (bool, np.bool_)) for value in structural_checks.values()):
        raise TypeError("Structural check outcomes must be Boolean")
    if not isinstance(non_degradation_tolerance, (int, float)) or not np.isfinite(
        non_degradation_tolerance
    ):
        raise ValueError("non_degradation_tolerance must be finite")
    if float(non_degradation_tolerance) < 0:
        raise ValueError("non_degradation_tolerance cannot be negative")

    subset = fold_metrics.loc[fold_metrics["model"].isin([candidate, baseline])]
    evidence: dict[str, dict[str, float | int]] = {}
    for metric in _METRICS:
        pivot = subset.pivot(index="test_year", columns="model", values=metric).sort_index()
        differences = pivot[candidate] - pivot[baseline]
        evidence[metric] = {
            "mean_candidate_minus_baseline": float(differences.mean()),
            "median_candidate_minus_baseline": float(differences.median()),
            "fold_wins": int(differences.lt(0).sum()),
            "fold_ties": int(differences.eq(0).sum()),
            "fold_losses": int(differences.gt(0).sum()),
        }
    empirically_eligible = []
    for primary, companion in (
        ("log_loss", "multiclass_brier"),
        ("multiclass_brier", "log_loss"),
    ):
        primary_result = evidence[primary]
        companion_result = evidence[companion]
        if (
            float(primary_result["mean_candidate_minus_baseline"]) < 0
            and int(primary_result["fold_wins"]) >= 4
            and float(companion_result["mean_candidate_minus_baseline"])
            <= float(non_degradation_tolerance)
        ):
            empirically_eligible.append(primary)

    decision = decision_metrics.sort_values("test_year")
    positive_draw_folds = int(decision["positive_draw_predictions"].gt(0).sum())
    macro_delta = float(pd.to_numeric(decision["macro_f1_delta"]).mean())
    draw_delta = float(pd.to_numeric(decision["draw_recall_delta"]).mean())
    universal_draw_collapse = bool(
        decision["positive_draw_predictions"].sum() == decision["test_match_count"].sum()
    )
    decision_passed = bool(
        positive_draw_folds >= 4
        and (macro_delta > 0 or draw_delta > 0)
        and not universal_draw_collapse
    )
    maximum_year = int(pd.to_numeric(fold_metrics["test_year"]).max())
    exact_years = set(pd.to_numeric(fold_metrics["test_year"]).astype(int)) == set(years)
    checks = {
        "empirical_probability_gate": bool(empirically_eligible),
        "decision_draw_support": positive_draw_folds >= 4,
        "decision_quality_improved": macro_delta > 0 or draw_delta > 0,
        "no_universal_draw_collapse": not universal_draw_collapse,
        "structural_invariants": bool(all(structural_checks.values())),
        "exact_model_identity": bool(
            _is_sha256(model_identity_hash)
            and _is_sha256(artifact_identity_hash)
            and model_identity_hash == artifact_identity_hash
        ),
        "frozen_2014_excluded": bool(maximum_year <= 2010 and exact_years),
    }
    probability_passed = bool(
        checks["empirical_probability_gate"]
        and checks["structural_invariants"]
        and checks["exact_model_identity"]
        and checks["frozen_2014_excluded"]
    )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "expected_outer_years": list(years),
        "non_degradation_tolerance": float(non_degradation_tolerance),
        "eligible_primary_metrics": empirically_eligible,
        "metric_evidence": evidence,
        "decision_evidence": {
            "positive_draw_folds": positive_draw_folds,
            "mean_macro_f1_delta": macro_delta,
            "mean_draw_recall_delta": draw_delta,
            "universal_draw_collapse": universal_draw_collapse,
        },
        "structural_checks": dict(structural_checks),
        "checks": checks,
        "probability_candidate_passed": probability_passed,
        "decision_layer_passed": decision_passed,
        "system_promoted": bool(probability_passed and decision_passed),
        "interpretation": (
            "development_only_promotion_decision_not_a_2014_or_2026_performance_claim"
        ),
    }


def normalize_decision_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Map persisted nested-decision names to the frozen promotion contract.

    The research stage stores gains under explicit baseline/decision names.  The
    legacy promotion API used generic names.  Accepting both schemas prevents a
    silent, post-hoc reinterpretation while making the mapping auditable.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("decision_metrics must be a pandas DataFrame")
    aliases = {
        "decision_draw_predictions": "positive_draw_predictions",
        "macro_f1_gain": "macro_f1_delta",
        "draw_recall_gain": "draw_recall_delta",
    }
    result = frame.copy()
    for source, target in aliases.items():
        if source in result and target in result:
            left = pd.to_numeric(result[source], errors="coerce")
            right = pd.to_numeric(result[target], errors="coerce")
            if not left.equals(right):
                raise ValueError(f"Conflicting decision columns {source!r} and {target!r}")
        elif source in result:
            result[target] = result[source]
    return result


def evaluate_calibration_retention(
    raw_fold_metrics: pd.DataFrame,
    calibrated_fold_metrics: pd.DataFrame,
    *,
    expected_years: Sequence[int] = _EXPECTED_YEARS,
    paired_primary_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Apply frozen H2 without promoting secondary ECE into a post-hoc gate."""
    years = tuple(int(year) for year in expected_years)
    if not np.isfinite(paired_primary_tolerance) or paired_primary_tolerance < 0:
        raise ValueError("paired_primary_tolerance must be finite and nonnegative")
    required = {"test_year", *_METRICS, "calibration_error"}
    for label, frame in (("raw", raw_fold_metrics), ("calibrated", calibrated_fold_metrics)):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{label} calibration metrics must be nonempty")
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} calibration metrics are missing columns: {missing}")
        if frame["test_year"].duplicated().any():
            raise ValueError(f"{label} calibration metrics require one row per edition")
        if set(pd.to_numeric(frame["test_year"], errors="raise").astype(int)) != set(years):
            raise ValueError(f"{label} calibration metrics have the wrong outer editions")
    joined = raw_fold_metrics[list(required)].merge(
        calibrated_fold_metrics[list(required)],
        on="test_year",
        suffixes=("_raw", "_calibrated"),
        validate="one_to_one",
    )
    numeric = joined.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Calibration-retention metrics must be finite")
    test_years = numeric["test_year"].to_numpy(float)
    if not np.equal(test_years, np.floor(test_years)).all():
        raise ValueError("Calibration-retention test_year values must be integer editions")
    mean_deltas = {
        metric: float((numeric[f"{metric}_calibrated"] - numeric[f"{metric}_raw"]).mean())
        for metric in (*_METRICS, "calibration_error")
    }
    eligible_primary: list[str] = []
    for primary, companion in (
        ("log_loss", "multiclass_brier"),
        ("multiclass_brier", "log_loss"),
    ):
        if mean_deltas[primary] < 0 and mean_deltas[companion] <= paired_primary_tolerance:
            eligible_primary.append(primary)
    ece_regressed = mean_deltas["calibration_error"] > 0
    return {
        "retained_under_frozen_h2": bool(eligible_primary),
        "eligible_primary_metrics": eligible_primary,
        "mean_calibrated_minus_raw": mean_deltas,
        "paired_primary_tolerance": float(paired_primary_tolerance),
        "ece_regressed": bool(ece_regressed),
        "ece_is_secondary_nonblocking": True,
        "claim": (
            "mixed_primary_score_calibration_retained_under_frozen_non_degradation_rule"
            if eligible_primary
            else "calibration_rejected_under_frozen_primary_score_rule"
        ),
    }


def evaluate_promotion_artifacts(output_root: str | Path) -> dict[str, Any]:
    """Validate and evaluate every sealed pre-audit stage artifact."""
    from ._promotion_artifacts import evaluate_promotion_artifacts as _evaluate

    return _evaluate(Path(output_root))


def issue_promotion_certificate(
    output_root: str | Path,
    *,
    authentication_key: bytes | None = None,
    key_id: str | None = None,
) -> dict[str, Any]:
    """Issue the hash-bound promotion record and unopened audit sentinel."""
    from ._promotion_artifacts import issue_promotion_certificate as _issue

    return _issue(Path(output_root), authentication_key=authentication_key, key_id=key_id)


def verify_promotion_certificate(
    certificate_path: str | Path,
    *,
    authentication_key: bytes | None = None,
    revalidate_artifacts: bool = True,
) -> dict[str, Any]:
    """Independently verify certificate integrity and all bound inputs."""
    from ._promotion_artifacts import verify_promotion_certificate as _verify

    return _verify(
        Path(certificate_path),
        authentication_key=authentication_key,
        revalidate_artifacts=revalidate_artifacts,
    )


def _validate_fold_metrics(
    frame: pd.DataFrame,
    candidate: str,
    baseline: str,
    expected_years: tuple[int, ...],
) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("fold_metrics must be a nonempty DataFrame")
    required = {"test_year", "model", *_METRICS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"fold_metrics is missing columns: {missing}")
    subset = frame.loc[frame["model"].isin([candidate, baseline]), list(required)]
    if subset.duplicated(["test_year", "model"]).any():
        raise ValueError("Promotion requires one row per model and tournament")
    pivot = subset.pivot(index="test_year", columns="model", values="log_loss").dropna()
    if candidate not in pivot or baseline not in pivot or len(pivot) != len(expected_years):
        raise ValueError("Promotion requires seven paired outer tournament editions")
    numeric = subset.loc[:, ["test_year", *_METRICS]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Promotion fold metrics must be finite")
    if (numeric["log_loss"] < 0).any():
        raise ValueError("Promotion log_loss values must be nonnegative")
    if ((numeric["multiclass_brier"] < 0) | (numeric["multiclass_brier"] > 2)).any():
        raise ValueError("Promotion multiclass_brier values must lie in [0, 2]")
    test_year = numeric["test_year"].to_numpy(float)
    if not np.equal(test_year, np.floor(test_year)).all() or set(test_year.astype(int)) != set(
        expected_years
    ):
        raise ValueError("Promotion test_year values must be the exact integer editions")


def _validate_decision_metrics(frame: pd.DataFrame, expected_years: tuple[int, ...]) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("decision_metrics must be a nonempty DataFrame")
    required = {
        "test_year",
        "positive_draw_predictions",
        "test_match_count",
        "macro_f1_delta",
        "draw_recall_delta",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"decision_metrics is missing columns: {missing}")
    if frame["test_year"].duplicated().any():
        raise ValueError("Decision metrics require one row per tournament")
    numeric = frame.loc[:, list(required)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Decision metrics must be finite")
    test_year = numeric["test_year"].to_numpy(float)
    if not np.equal(test_year, np.floor(test_year)).all():
        raise ValueError("Decision test_year values must be integer editions")
    years = set(test_year.astype(int))
    if years != set(expected_years):
        raise ValueError("Decision metrics must cover the frozen outer tournaments")
    draw_counts = numeric["positive_draw_predictions"].to_numpy(float)
    match_counts = numeric["test_match_count"].to_numpy(float)
    if (
        np.any(draw_counts < 0)
        or np.any(match_counts <= 0)
        or np.any(draw_counts > match_counts)
        or not np.equal(draw_counts, np.floor(draw_counts)).all()
        or not np.equal(match_counts, np.floor(match_counts)).all()
    ):
        raise ValueError(
            "Draw predictions and test-match totals must be valid bounded integer counts"
        )
    deltas = numeric[["macro_f1_delta", "draw_recall_delta"]]
    if ((deltas < -1) | (deltas > 1)).any().any():
        raise ValueError("Decision metric deltas must lie in [-1, 1]")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "PromotionRejectedError",
    "evaluate_advanced_promotion",
    "evaluate_calibration_retention",
    "evaluate_promotion_artifacts",
    "issue_promotion_certificate",
    "normalize_decision_metrics",
    "verify_promotion_certificate",
]
