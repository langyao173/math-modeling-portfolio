"""Seven-fold statistical closure for the frozen development comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_YEARS = (1986, 1990, 1994, 1998, 2002, 2006, 2010)
EXPECTED_CLASSES = (0, 1, 2)
CANDIDATE = "advanced_nested_calibrated"
BASELINE = "legacy_mirrored_dixon_coles"
METRICS = ("log_loss", "multiclass_brier")
MAXIMUM_FOLD_REGRESSION = {"log_loss": 0.10, "multiclass_brier": 0.05}


@dataclass(frozen=True)
class StatisticalClosureResult:
    """Complete fold, rank, classwise, and promotion summaries."""

    fold_comparison: pd.DataFrame
    model_ranking: pd.DataFrame
    classwise_calibration: pd.DataFrame
    summary: dict[str, Any]


def build_statistical_closure(
    fold_metrics: pd.DataFrame,
    paired_intervals: pd.DataFrame,
    selected_specs: pd.DataFrame,
    raw_classwise: pd.DataFrame,
    calibrated_classwise: pd.DataFrame,
) -> StatisticalClosureResult:
    """Close all pre-2014 comparison views without collapsing estimands."""
    paired = _paired_fold_table(fold_metrics)
    validated_intervals = _validate_paired_intervals(paired_intervals)
    comparison = _fold_comparison(paired)
    metric_summary = {
        metric: _metric_summary(paired, validated_intervals, metric) for metric in METRICS
    }
    calibration = _calibration_summary(paired)
    stability = _selection_stability(selected_specs)
    ranking = _model_ranking(fold_metrics)
    classwise = _classwise_comparison(raw_classwise, calibrated_classwise)
    checks = {
        "median_log_loss_improvement_positive": (
            metric_summary["log_loss"]["median_improvement"] > 0
        ),
        "median_brier_improvement_positive": (
            metric_summary["multiclass_brier"]["median_improvement"] > 0
        ),
        "at_least_five_of_seven_log_loss_wins": (metric_summary["log_loss"]["fold_wins"] >= 5),
        "at_least_five_of_seven_brier_wins": (metric_summary["multiclass_brier"]["fold_wins"] >= 5),
        "no_catastrophic_log_loss_fold": (
            metric_summary["log_loss"]["worst_fold_regression"]
            <= MAXIMUM_FOLD_REGRESSION["log_loss"]
        ),
        "no_catastrophic_brier_fold": (
            metric_summary["multiclass_brier"]["worst_fold_regression"]
            <= MAXIMUM_FOLD_REGRESSION["multiclass_brier"]
        ),
        "calibration_degradation_bounded_and_disclosed": calibration[
            "bounded_under_internal_tolerance"
        ],
        "selected_candidate_modal_in_at_least_five_folds": stability["modal_selected_fold_count"]
        >= 5,
    }
    summary = {
        "schema_version": 1,
        "scope": "seven_outer_development_editions_1986_2010_no_2014",
        "candidate": CANDIDATE,
        "baseline": BASELINE,
        "outer_fold_count": 7,
        "expected_years": list(EXPECTED_YEARS),
        "match_count": int(paired[("n", CANDIDATE)].sum()),
        "metric_summary": metric_summary,
        "calibration_change": calibration,
        "selection_stability": stability,
        "internal_promotion_checks": checks,
        "passed": bool(all(checks.values())),
        "claim_boundary": ("development_nested_validation_not_external_audit_or_2026_performance"),
    }
    return StatisticalClosureResult(comparison, ranking, classwise, summary)


def _paired_fold_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"test_year", "model", "n", *METRICS, "calibration_error"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Fold metrics are missing statistical closure fields: {missing}")
    subset = frame.loc[frame["model"].isin((CANDIDATE, BASELINE))].copy()
    subset["test_year"] = _exact_integers(subset["test_year"], "test_year")
    if subset.duplicated(["test_year", "model"]).any():
        raise ValueError("Statistical closure requires unique model-edition rows")
    years = set(subset["test_year"])
    if years != set(EXPECTED_YEARS) or len(subset) != 14:
        raise ValueError("Statistical closure requires the seven exact paired editions")
    subset["n"] = _exact_integers(subset["n"], "n", positive=True)
    for metric in (*METRICS, "calibration_error"):
        subset[metric] = pd.to_numeric(subset[metric], errors="raise")
    metric_values = subset[[*METRICS, "calibration_error"]].to_numpy(float)
    if not np.isfinite(metric_values).all():
        raise ValueError("Statistical closure metrics must be finite")
    if (
        (subset["log_loss"] < 0).any()
        or not subset["multiclass_brier"].between(0, 2).all()
        or not subset["calibration_error"].between(0, 1).all()
    ):
        raise ValueError("Statistical closure metrics are outside their valid ranges")
    pivot = subset.pivot(index="test_year", columns="model")
    for metric in (*METRICS, "calibration_error", "n"):
        values = pivot[metric][[CANDIDATE, BASELINE]].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Statistical closure metric {metric} is nonfinite")
    if not np.array_equal(
        pivot[("n", CANDIDATE)].to_numpy(),
        pivot[("n", BASELINE)].to_numpy(),
    ):
        raise ValueError("Paired fold match counts disagree between models")
    return pivot.sort_index()


def _validate_paired_intervals(intervals: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate", "baseline", "metric", "replicates", "ci_low", "ci_high"}
    missing = sorted(required.difference(intervals.columns))
    if missing:
        raise ValueError(f"Paired intervals are missing required fields: {missing}")
    validated = intervals.copy()
    relevant = validated.loc[
        validated["candidate"].eq(CANDIDATE)
        & validated["baseline"].eq(BASELINE)
        & validated["metric"].isin(METRICS)
    ].copy()
    if len(relevant) != len(METRICS) or relevant["metric"].nunique() != len(METRICS):
        raise ValueError("Paired intervals require one exact certificate per primary metric")
    relevant["replicates"] = _exact_integers(
        relevant["replicates"], "paired interval replicates", positive=True
    )
    relevant[["ci_low", "ci_high"]] = relevant[["ci_low", "ci_high"]].apply(
        pd.to_numeric, errors="raise"
    )
    interval_values = relevant[["ci_low", "ci_high"]].to_numpy(float)
    if not np.isfinite(interval_values).all():
        raise ValueError("Paired interval bounds must be finite")
    if (relevant["ci_low"] > relevant["ci_high"]).any():
        raise ValueError("Paired interval lower bounds cannot exceed upper bounds")
    return relevant


def _fold_comparison(pivot: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "test_year": pivot.index.astype(int),
            "match_count": pivot[("n", CANDIDATE)].to_numpy(int),
        }
    )
    for metric in (*METRICS, "calibration_error"):
        candidate = pivot[(metric, CANDIDATE)].to_numpy(float)
        baseline = pivot[(metric, BASELINE)].to_numpy(float)
        result[f"candidate_{metric}"] = candidate
        result[f"baseline_{metric}"] = baseline
        result[f"candidate_minus_baseline_{metric}"] = candidate - baseline
        result[f"improvement_{metric}"] = baseline - candidate
    return result


def _metric_summary(
    pivot: pd.DataFrame,
    intervals: pd.DataFrame,
    metric: str,
) -> dict[str, Any]:
    candidate = pivot[(metric, CANDIDATE)].to_numpy(float)
    baseline = pivot[(metric, BASELINE)].to_numpy(float)
    weights = pivot[("n", CANDIDATE)].to_numpy(float)
    difference = candidate - baseline
    certificate = intervals.loc[
        intervals["candidate"].eq(CANDIDATE)
        & intervals["baseline"].eq(BASELINE)
        & intervals["metric"].eq(metric)
    ]
    if len(certificate) != 1:
        raise ValueError(f"Missing unique paired interval for {metric}")
    record = certificate.iloc[0]
    return {
        "candidate_unweighted_mean": float(candidate.mean()),
        "baseline_unweighted_mean": float(baseline.mean()),
        "unweighted_mean_improvement": float((baseline - candidate).mean()),
        "candidate_match_weighted_mean": float(np.average(candidate, weights=weights)),
        "baseline_match_weighted_mean": float(np.average(baseline, weights=weights)),
        "match_weighted_mean_improvement": float(np.average(baseline - candidate, weights=weights)),
        "median_improvement": float(np.median(baseline - candidate)),
        "fold_wins": int(np.sum(difference < 0)),
        "fold_losses": int(np.sum(difference > 0)),
        "worst_fold_year": int(pivot.index[int(np.argmax(difference))]),
        "worst_fold_regression": float(max(float(difference.max()), 0.0)),
        "best_fold_improvement": float(max(float((-difference).max()), 0.0)),
        "paired_cluster_bootstrap_replicates": int(record["replicates"]),
        "paired_cluster_bootstrap_ci_candidate_minus_baseline": [
            float(record["ci_low"]),
            float(record["ci_high"]),
        ],
        "bootstrap_interval_excludes_zero_in_favour_of_candidate": bool(
            float(record["ci_high"]) < 0
        ),
    }


def _calibration_summary(pivot: pd.DataFrame) -> dict[str, Any]:
    difference = (
        pivot[("calibration_error", CANDIDATE)] - pivot[("calibration_error", BASELINE)]
    ).to_numpy(float)
    return {
        "mean_candidate_minus_baseline": float(difference.mean()),
        "median_candidate_minus_baseline": float(np.median(difference)),
        "worsened_folds": int(np.sum(difference > 0)),
        "improved_folds": int(np.sum(difference < 0)),
        "worst_fold_regression": float(max(float(difference.max()), 0.0)),
        "mean_tolerance": 0.02,
        "single_fold_tolerance": 0.05,
        "bounded_under_internal_tolerance": bool(
            difference.mean() <= 0.02 and difference.max() <= 0.05
        ),
        "disclosure": (
            "ECE is a mixed secondary result and is not described as uniformly improved."
        ),
    }


def _selection_stability(selected: pd.DataFrame) -> dict[str, Any]:
    required = {"outer_test_year", "candidate_id"}
    if not required.issubset(selected.columns) or len(selected) != 7:
        raise ValueError("Selected-spec stability requires seven outer selections")
    selected = selected.copy()
    selected["outer_test_year"] = _exact_integers(
        selected["outer_test_year"], "outer_test_year"
    )
    if selected["outer_test_year"].duplicated().any():
        raise ValueError("Selected-spec stability requires one selection per edition")
    years = set(selected["outer_test_year"])
    if years != set(EXPECTED_YEARS):
        raise ValueError("Selected-spec stability has the wrong outer editions")
    if selected["candidate_id"].isna().any():
        raise ValueError("Selected candidate identifiers cannot be missing")
    candidate_ids = selected["candidate_id"].astype(str).str.strip()
    if candidate_ids.eq("").any():
        raise ValueError("Selected candidate identifiers cannot be empty")
    counts = candidate_ids.value_counts()
    return {
        "unique_selected_candidate_count": int(len(counts)),
        "modal_candidate_id": str(counts.index[0]),
        "modal_selected_fold_count": int(counts.iloc[0]),
        "selection_frequencies": {str(key): int(value) for key, value in counts.items()},
    }


def _model_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"test_year", "model", *METRICS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Model-ranking evidence is missing fields: {missing}")
    subset = frame[["test_year", "model", *METRICS]].copy()
    subset["test_year"] = _exact_integers(subset["test_year"], "ranking test_year")
    if set(subset["test_year"]) != set(EXPECTED_YEARS):
        raise ValueError("Model-ranking evidence has the wrong outer editions")
    if subset["model"].isna().any() or subset["model"].astype(str).str.strip().eq("").any():
        raise ValueError("Model-ranking identifiers cannot be missing or empty")
    subset["model"] = subset["model"].astype(str).str.strip()
    if subset.duplicated(["test_year", "model"]).any():
        raise ValueError("Model-ranking evidence requires unique model-edition rows")
    edition_counts = subset.groupby("model")["test_year"].nunique()
    if not edition_counts.eq(len(EXPECTED_YEARS)).all():
        raise ValueError("Every ranked model must cover all seven outer editions")
    subset[list(METRICS)] = subset[list(METRICS)].apply(pd.to_numeric, errors="raise")
    metric_values = subset[list(METRICS)].to_numpy(float)
    if not np.isfinite(metric_values).all() or (metric_values < 0).any():
        raise ValueError("Model-ranking metrics must be finite and nonnegative")
    if not subset["multiclass_brier"].between(0, 2).all():
        raise ValueError("Model-ranking Brier scores must lie in [0, 2]")
    for metric in METRICS:
        subset[f"{metric}_rank"] = subset.groupby("test_year")[metric].rank(
            method="average", ascending=True
        )
    return (
        subset.groupby("model", as_index=False)
        .agg(
            editions=("test_year", "nunique"),
            mean_log_loss=("log_loss", "mean"),
            mean_log_loss_rank=("log_loss_rank", "mean"),
            mean_brier=("multiclass_brier", "mean"),
            mean_brier_rank=("multiclass_brier_rank", "mean"),
        )
        .sort_values(["mean_log_loss_rank", "mean_brier_rank", "model"])
        .reset_index(drop=True)
    )


def _classwise_comparison(raw: pd.DataFrame, calibrated: pd.DataFrame) -> pd.DataFrame:
    required = ("class_index", "n", "prevalence", "intercept", "slope")
    if not set(required).issubset(raw.columns) or not set(required).issubset(calibrated.columns):
        raise ValueError("Classwise calibration tables are incomplete")
    raw = raw[list(required)].copy()
    calibrated = calibrated[list(required)].copy()
    raw["class_index"] = _exact_integers(raw["class_index"], "raw class_index")
    calibrated["class_index"] = _exact_integers(
        calibrated["class_index"], "calibrated class_index"
    )
    if set(raw["class_index"]) != set(EXPECTED_CLASSES) or set(
        calibrated["class_index"]
    ) != set(EXPECTED_CLASSES):
        raise ValueError("Classwise calibration requires the exact classes {0, 1, 2}")
    raw["n"] = _exact_integers(raw["n"], "raw classwise n", positive=True)
    calibrated["n"] = _exact_integers(
        calibrated["n"], "calibrated classwise n", positive=True
    )
    merged = raw.merge(
        calibrated,
        on="class_index",
        suffixes=("_raw", "_calibrated"),
        validate="one_to_one",
    )
    numeric = merged.drop(columns="class_index").apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Classwise calibration evidence must be finite")
    if not np.array_equal(merged["n_raw"].to_numpy(), merged["n_calibrated"].to_numpy()):
        raise ValueError("Raw and calibrated classwise sample counts disagree")
    if merged["n_raw"].nunique() != 1:
        raise ValueError("Each classwise row must describe the same evaluation sample")
    for suffix in ("raw", "calibrated"):
        prevalence = merged[f"prevalence_{suffix}"].to_numpy(float)
        if not np.all((0 <= prevalence) & (prevalence <= 1)) or not np.isclose(
            prevalence.sum(), 1.0, atol=1e-10
        ):
            raise ValueError("Class prevalences must be probabilities summing to one")
    if not np.allclose(
        merged["prevalence_raw"], merged["prevalence_calibrated"], atol=1e-12
    ):
        raise ValueError("Raw and calibrated class prevalences disagree")
    merged["raw_distance_from_ideal"] = (
        merged["intercept_raw"].abs() + (merged["slope_raw"] - 1.0).abs()
    )
    merged["calibrated_distance_from_ideal"] = (
        merged["intercept_calibrated"].abs() + (merged["slope_calibrated"] - 1.0).abs()
    )
    merged["distance_improvement"] = (
        merged["raw_distance_from_ideal"] - merged["calibrated_distance_from_ideal"]
    )
    return merged.sort_values("class_index").reset_index(drop=True)


def _exact_integers(
    values: pd.Series,
    field: str,
    *,
    positive: bool = False,
) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{field} must contain finite exact integers")
    integers = numeric.astype(np.int64)
    if positive and np.any(integers <= 0):
        raise ValueError(f"{field} must contain positive integers")
    return integers


__all__ = ["StatisticalClosureResult", "build_statistical_closure"]
