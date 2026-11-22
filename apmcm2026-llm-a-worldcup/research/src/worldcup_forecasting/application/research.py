"""Leakage-safe expanding-origin research evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from worldcup_forecasting.data import CanonicalDataBundle
from worldcup_forecasting.features import (
    build_dynamic_features,
    build_player_history_features,
    build_team_match_features,
)
from worldcup_forecasting.models import (
    OUTCOME_ORDER,
    ConvexOutcomeEnsemble,
    JointPoissonConfig,
    JointPoissonModel,
    OutcomeLogit,
    TemperatureScaler,
)
from worldcup_forecasting.reporting.figures import plot_model_comparison, plot_reliability
from worldcup_forecasting.validation.bootstrap import paired_fold_comparison
from worldcup_forecasting.validation.feature_availability import (
    assert_dynamic_availability,
    assert_player_availability,
    assert_static_history_availability,
    feature_availability_registry,
)
from worldcup_forecasting.validation.leakage import assert_probability_simplex, symmetry_error
from worldcup_forecasting.validation.metrics import classification_metrics
from worldcup_forecasting.validation.rolling import expanding_origin_folds

from .official import OfficialRunResult
from .runtime import write_csv, write_json


@dataclass(slots=True)
class ResearchRunResult:
    """Research evidence used by reports and the model card."""

    cv_metrics: pd.DataFrame
    comparisons: list[dict[str, Any]]
    selected_model: str
    promotion_decision: str
    figure_catalog: list[dict[str, str]]


def run_research_analysis(
    bundle: CanonicalDataBundle,
    official: OfficialRunResult | None,
    output_dir: Path,
    config: Any,
    logger: Any,
) -> ResearchRunResult:
    """Evaluate structural candidates without touching the frozen 2014 selection target."""
    logger.info("Research: constructing dynamic and player histories")
    features = (
        official.team_features.copy()
        if official is not None
        else build_team_match_features(bundle.matches, bundle.tournaments)
    )
    availability_checks: list[dict[str, int | str]] = [
        assert_static_history_availability(features)
    ]
    dynamic_columns: list[str] = []
    if bool(config.research.enable_dynamic_features):
        dynamic = build_dynamic_features(bundle.matches)
        availability_checks.append(assert_dynamic_availability(dynamic))
        dynamic_columns = [
            "pre_match_elo_diff",
            "attack_state_diff",
            "defense_state_diff",
            "home_dynamic_history_missing",
            "away_dynamic_history_missing",
        ]
        features = features.merge(
            dynamic[["MatchID", *dynamic_columns]], on="MatchID", validate="one_to_one"
        )
    player_schema: list[str] = []
    if bool(config.research.enable_player_history):
        player = build_player_history_features(bundle.players, bundle.matches)
        availability_checks.append(assert_player_availability(player))
        player_schema = [column for column in player if column.endswith("_diff")]
        write_csv(
            player[["MatchID", *player_schema]],
            output_dir / "audit" / "player_history_features.csv",
        )
    write_json(
        {
            "information_set": "pre_match_or_kickoff_as_named",
            "historical_feature_columns": [
                column for column in features if column.endswith("_diff")
            ],
            "player_feature_columns": player_schema,
            "data_cutoff_rule": "strictly earlier match or completed earlier tournament",
            "research_toggles": {
                "dynamic_features": bool(config.research.enable_dynamic_features),
                "player_history_audit": bool(config.research.enable_player_history),
                "dixon_coles": bool(config.research.enable_dixon_coles),
            },
        },
        output_dir / "audit" / "research_feature_schema.json",
    )
    write_csv(
        feature_availability_registry(
            include_dynamic=bool(config.research.enable_dynamic_features),
            include_player=bool(config.research.enable_player_history),
        ),
        output_dir / "audit" / "research_feature_availability_registry.csv",
    )
    write_json(
        {"status": "passed", "checks": availability_checks},
        output_dir / "audit" / "research_feature_availability_report.json",
    )

    test_years = [int(year) for year in config.protocol.selection_test_years]
    folds = expanding_origin_folds(features, test_years)
    metric_rows: list[dict[str, Any]] = []
    pooled: dict[str, list[np.ndarray]] = {"outcome_logit": [], "joint_poisson": []}
    pooled_labels: list[np.ndarray] = []
    logger.info("Research: running %d expanding-origin folds", len(folds))
    for fold in folds:
        train = features.iloc[fold.train_index]
        test = features.iloc[fold.test_index]
        labels = test["target_outcome"].to_numpy()
        pooled_labels.append(labels)
        logit_start = time.perf_counter()
        logit_probability = OutcomeLogit().fit(train).predict_proba(test)
        logit_runtime = time.perf_counter() - logit_start
        poisson_start = time.perf_counter()
        poisson_probability = _joint_model(config).fit(train).predict_proba(test)
        poisson_runtime = time.perf_counter() - poisson_start
        candidates = (
            ("outcome_logit", logit_probability, logit_runtime),
            ("joint_poisson", poisson_probability, poisson_runtime),
        )
        for name, probabilities, runtime_seconds in candidates:
            assert_probability_simplex(probabilities)
            metrics, _ = classification_metrics(labels, probabilities)
            metric_rows.append(
                {
                    "test_year": fold.test_year,
                    "model": name,
                    "runtime_seconds": runtime_seconds,
                    **metrics,
                }
            )
            pooled[name].append(probabilities)

    cv_metrics = pd.DataFrame(metric_rows)
    write_csv(cv_metrics, output_dir / "tables" / "research_cv_metrics.csv")
    plot_model_comparison(
        cv_metrics,
        "log_loss",
        output_dir / "figures" / "research_outer_fold_log_loss",
        int(config.runtime.figure_dpi),
    )

    labels = np.concatenate(pooled_labels)
    label_index = np.asarray([OUTCOME_ORDER.index(str(label)) for label in labels], dtype=int)
    logit_probability = np.vstack(pooled["outcome_logit"])
    poisson_probability = np.vstack(pooled["joint_poisson"])
    logit_scaler = TemperatureScaler().fit(logit_probability, label_index)
    poisson_scaler = TemperatureScaler().fit(poisson_probability, label_index)
    calibrated_logit = logit_scaler.transform(logit_probability)
    calibrated_poisson = poisson_scaler.transform(poisson_probability)
    ensemble = ConvexOutcomeEnsemble().fit(
        np.stack([calibrated_logit, calibrated_poisson]), label_index
    )
    ensemble_probability = ensemble.predict_proba(np.stack([calibrated_logit, calibrated_poisson]))
    development_rows = []
    for name, probability in (
        ("outcome_logit_calibrated", calibrated_logit),
        ("joint_poisson_calibrated", calibrated_poisson),
        ("convex_ensemble", ensemble_probability),
    ):
        metrics, _ = classification_metrics(labels, probability)
        development_rows.append(
            {
                "model": name,
                "evaluation_role": "calibration_fit_diagnostic_not_held_out",
                **metrics,
            }
        )
    write_csv(
        pd.DataFrame(development_rows),
        output_dir / "tables" / "research_development_fit_calibration_metrics.csv",
    )
    write_json(
        {
            "fit_scope": "pooled development OOF base predictions from 1986-2010 only",
            "evaluation_role": "calibration_and_ensemble_fit_diagnostic_not_held_out",
            "logit_temperature": logit_scaler.temperature_,
            "joint_poisson_temperature": poisson_scaler.temperature_,
            "ensemble_weights": ensemble.weights_.tolist(),
            "warning": "The calibrators are fit and diagnosed on the same pooled development "
            "predictions. Only their later application to frozen 2014 is held out.",
        },
        output_dir / "tables" / "research_calibration.json",
    )
    plot_reliability(
        label_index,
        ensemble_probability,
        output_dir / "figures" / "research_development_fit_reliability",
        int(config.runtime.figure_dpi),
    )

    comparisons: list[dict[str, Any]] = []
    for metric in ("log_loss", "multiclass_brier", "ranked_probability_score"):
        comparisons.append(
            paired_fold_comparison(
                cv_metrics,
                candidate="joint_poisson",
                baseline="outcome_logit",
                metric=metric,
            )
        )
    write_csv(pd.DataFrame(comparisons), output_dir / "tables" / "research_paired_comparisons.csv")

    logit_mean = float(cv_metrics.loc[cv_metrics["model"].eq("outcome_logit"), "log_loss"].mean())
    poisson_mean = float(cv_metrics.loc[cv_metrics["model"].eq("joint_poisson"), "log_loss"].mean())
    ablation = pd.DataFrame(
        [
            {
                "variant": "official_outcome_logit",
                "removed_component": "joint score distribution and dynamic strength",
                "mean_outer_log_loss": logit_mean,
            },
            {
                "variant": "joint_poisson",
                "removed_component": "none (structural candidate)",
                "mean_outer_log_loss": poisson_mean,
            },
            {
                "variant": "joint_poisson_without_dynamic_features",
                "removed_component": "dynamic Elo/attack/defence",
                "mean_outer_log_loss": _mandatory_only_cv(features, folds, config),
            },
        ]
    )
    write_csv(ablation, output_dir / "tables" / "research_ablation_matrix.csv")

    # Freeze selection before the 2014 audit. These invariants and repeat fits use
    # only the declared development period through 2010.
    development_train = features["Year"].le(int(config.protocol.train_end_year))
    development_sample = features.loc[
        features["Year"].eq(int(config.protocol.train_end_year))
    ].head(16)
    poisson_model = _joint_model(config).fit(features.loc[development_train])
    direct = poisson_model.predict_proba(development_sample)
    swapped = development_sample.copy()
    for column in poisson_model.feature_columns_:
        swapped[column] = -development_sample[column].to_numpy(float)
    reverse = poisson_model.predict_proba(swapped)
    repeated = _joint_model(config).fit(features.loc[development_train]).predict_proba(
        development_sample
    )
    determinism_error = float(np.max(np.abs(direct - repeated)))
    simplex_error = float(np.max(np.abs(direct.sum(axis=1) - 1.0)))
    swap_error = symmetry_error(direct, reverse)
    point_in_time_ok = _development_point_in_time_ok(
        features, int(config.protocol.train_end_year)
    )
    gate = evaluate_promotion_gate(
        cv_metrics,
        config.research.promotion_gate,
        candidate="joint_poisson",
        baseline="outcome_logit",
        simplex_error=simplex_error,
        symmetry_error_value=swap_error,
        determinism_error=determinism_error,
        point_in_time_ok=point_in_time_ok,
        maximum_selection_year=int(config.protocol.train_end_year),
    )
    gate_passed = bool(gate["passed"])
    selected = (
        "joint_poisson_dynamic_research_candidate_only"
        if gate_passed and bool(config.research.enable_dynamic_features)
        else "joint_poisson_static_research_candidate_only"
        if gate_passed
        else "outcome_logit_official_baseline"
    )
    promotion = (
        "research_gate_passed_candidate_not_deployed_by_research_profile"
        if gate_passed
        else "research_gate_failed_official_baseline_retained"
    )
    gate["selected_research_candidate"] = selected
    gate["decision"] = promotion
    write_json(gate, output_dir / "tables" / "research_promotion_gate.json")
    write_csv(
        pd.DataFrame(
            [
                {
                    "check": name,
                    "passed": passed,
                    "observed": gate["check_evidence"].get(name),
                }
                for name, passed in gate["checks"].items()
            ]
        ),
        output_dir / "tables" / "research_promotion_gate.csv",
    )
    stress = {
        "evaluation_scope": "development data through 2010 only",
        "probability_simplex_max_error": simplex_error,
        "swap_symmetry_max_error": swap_error,
        "repeat_fit_max_probability_error": determinism_error,
        "point_in_time_history_ok": point_in_time_ok,
        "half_time_monotonicity_tested_in_official_layer": True,
        "format_shift_2026_48_teams_104_matches": "scenario metadata only; no 2026 outcomes used",
        "promotion_decision": promotion,
    }
    write_json(stress, output_dir / "tables" / "research_stress_tests.json")

    logger.info("Research selection frozen before 2014 audit; decision=%s", promotion)
    train = features["Year"].le(int(config.protocol.train_end_year))
    test = features["Year"].eq(int(config.protocol.final_test_year))
    logger.info("Research: applying the frozen development decision to the 2014 audit")
    logit_2014 = OutcomeLogit().fit(features.loc[train]).predict_proba(features.loc[test])
    audit_poisson_model = _joint_model(config).fit(features.loc[train])
    poisson_2014 = audit_poisson_model.predict_proba(features.loc[test])
    frozen_ensemble = ensemble.predict_proba(
        np.stack(
            [
                logit_scaler.transform(logit_2014),
                poisson_scaler.transform(poisson_2014),
            ]
        )
    )
    frozen_labels = features.loc[test, "target_outcome"].to_numpy()
    frozen_rows = []
    for name, probability in (
        ("official_outcome_logit", logit_2014),
        ("joint_poisson_research_candidate", poisson_2014),
        ("calibrated_convex_ensemble_development_fitted", frozen_ensemble),
    ):
        metrics, _ = classification_metrics(frozen_labels, probability)
        frozen_rows.append(
            {
                "model": name,
                "selection_role": "audit_only_not_used_for_promotion",
                **metrics,
            }
        )
    write_csv(pd.DataFrame(frozen_rows), output_dir / "tables" / "research_frozen_2014.csv")

    catalog = [
        {
            "filename": "research_outer_fold_log_loss.png",
            "purpose": "Compare proper scores by held-out tournament edition.",
            "data_source": "Expanding-origin folds ending in 2010.",
            "observation": (
                f"Mean log loss: logit={logit_mean:.4f}, joint Poisson={poisson_mean:.4f}."
            ),
            "interpretation": (
                "Lower values indicate better probabilistic sharpness/calibration jointly."
            ),
            "caveat": (
                "Only seven development editions are available; fold estimates are dependent "
                "over time."
            ),
        },
        {
            "filename": "research_development_fit_reliability.png",
            "purpose": "Inspect fitted development calibration behavior.",
            "data_source": "Calibrators fit on pooled 1986-2010 development OOF predictions.",
            "observation": "Marker size is the number of predictions per confidence bin.",
            "interpretation": "The diagonal is perfect top-label calibration.",
            "caveat": (
                "Calibration is diagnosed on its fitting predictions; this is not held-out "
                "evidence."
            ),
        },
    ]
    logger.info("Research analysis complete; promotion decision=%s", promotion)
    return ResearchRunResult(cv_metrics, comparisons, selected, promotion, catalog)


def _mandatory_only_cv(features: pd.DataFrame, folds: list[Any], config: Any) -> float:
    """Return the explicit no-dynamic Joint Poisson ablation."""
    losses: list[float] = []
    for fold in folds:
        train = features.iloc[fold.train_index].drop(
            columns=["pre_match_elo_diff", "attack_state_diff", "defense_state_diff"],
            errors="ignore",
        )
        test = features.iloc[fold.test_index].drop(
            columns=["pre_match_elo_diff", "attack_state_diff", "defense_state_diff"],
            errors="ignore",
        )
        probability = _joint_model(config, include_dynamic=False).fit(train).predict_proba(test)
        metrics, _ = classification_metrics(test["target_outcome"].to_numpy(), probability)
        losses.append(float(metrics["log_loss"]))
    return float(np.mean(losses))


def _joint_model(config: Any, *, include_dynamic: bool | None = None) -> JointPoissonModel:
    """Construct a candidate whose public research toggles all change behavior."""
    use_dynamic = (
        bool(config.research.enable_dynamic_features)
        if include_dynamic is None
        else bool(include_dynamic)
    )
    optional = (
        ("pre_match_elo_diff", "attack_state_diff", "defense_state_diff")
        if use_dynamic
        else ()
    )
    rho = None if bool(config.research.enable_dixon_coles) else 0.0
    model_config = JointPoissonConfig(
        optional_diff_features=optional,
        rho=rho,
        alpha=float(config.models.poisson_alpha),
        time_decay_half_life_years=float(config.models.time_decay_half_life_years),
        max_score=int(config.models.max_score),
    )
    return JointPoissonModel(model_config)


def _development_point_in_time_ok(features: pd.DataFrame, maximum_year: int) -> bool:
    """Check the invariant only on rows eligible to influence research selection."""
    development = features.loc[features["Year"].le(int(maximum_year))]
    if development.empty:
        raise ValueError("Point-in-time validation requires development rows")
    return bool((development["history_cutoff_year"] < development["Year"]).all())


def evaluate_promotion_gate(
    table: pd.DataFrame,
    gate_config: Any,
    *,
    candidate: str,
    baseline: str,
    simplex_error: float,
    symmetry_error_value: float,
    determinism_error: float,
    point_in_time_ok: bool,
    maximum_selection_year: int,
) -> dict[str, Any]:
    """Evaluate an explicit development-only, majority-fold promotion contract."""
    primary = str(gate_config.primary_metric)
    subset = table.loc[table["model"].isin([candidate, baseline])]
    pivot = subset.pivot(index="test_year", columns="model", values=primary).dropna()
    if candidate not in pivot or baseline not in pivot or pivot.empty:
        raise ValueError("Promotion gate requires paired candidate and baseline folds")
    difference = pivot[candidate] - pivot[baseline]
    fold_win_fraction = float((difference < 0).mean())
    candidate_rows = subset.loc[subset["model"].eq(candidate)]
    baseline_rows = subset.loc[subset["model"].eq(baseline)]
    primary_mean_difference = float(difference.mean())
    accuracy_difference = float(
        candidate_rows["accuracy"].mean() - baseline_rows["accuracy"].mean()
    )
    calibration_difference = float(
        candidate_rows["calibration_error"].mean()
        - baseline_rows["calibration_error"].mean()
    )
    candidate_max_runtime = float(candidate_rows["runtime_seconds"].max())
    maximum_observed_year = int(subset["test_year"].max())
    observed = {
        "primary_metric_mean_difference": primary_mean_difference,
        "majority_fold_win_fraction": fold_win_fraction,
        "accuracy_mean_difference": accuracy_difference,
        "calibration_error_mean_difference": calibration_difference,
        "runtime_max_seconds_per_fold": candidate_max_runtime,
        "probability_simplex_max_error": float(simplex_error),
        "swap_symmetry_max_error": float(symmetry_error_value),
        "repeat_fit_max_probability_error": float(determinism_error),
        "point_in_time_history": bool(point_in_time_ok),
        "maximum_selection_year": maximum_observed_year,
    }
    checks = {
        "primary_metric_mean_improved": primary_mean_difference < 0,
        "strict_majority_of_folds_improved": (
            fold_win_fraction > float(gate_config.minimum_fold_win_fraction)
        ),
        "accuracy_not_degraded": (
            accuracy_difference
            >= -float(gate_config.accuracy_non_degradation_tolerance)
        ),
        "calibration_not_degraded": (
            calibration_difference
            <= float(gate_config.calibration_non_degradation_tolerance)
        ),
        "runtime_within_budget": (
            candidate_max_runtime <= float(gate_config.max_runtime_seconds_per_fold)
        ),
        "probability_simplex_invariant": (
            simplex_error <= float(gate_config.max_probability_invariant_error)
        ),
        "swap_symmetry_invariant": (
            symmetry_error_value <= float(gate_config.max_probability_invariant_error)
        ),
        "deterministic_repeat_fit": (
            determinism_error <= float(gate_config.max_determinism_error)
        ),
        "point_in_time_features": bool(point_in_time_ok),
        "frozen_2014_excluded_from_selection": maximum_observed_year <= maximum_selection_year,
    }
    check_evidence = {
        "primary_metric_mean_improved": primary_mean_difference,
        "strict_majority_of_folds_improved": fold_win_fraction,
        "accuracy_not_degraded": accuracy_difference,
        "calibration_not_degraded": calibration_difference,
        "runtime_within_budget": candidate_max_runtime,
        "probability_simplex_invariant": float(simplex_error),
        "swap_symmetry_invariant": float(symmetry_error_value),
        "deterministic_repeat_fit": float(determinism_error),
        "point_in_time_features": bool(point_in_time_ok),
        "frozen_2014_excluded_from_selection": maximum_observed_year,
    }
    return {
        "scope": "development folds only; frozen 2014 excluded",
        "candidate": candidate,
        "baseline": baseline,
        "primary_metric": primary,
        "thresholds": {
            key: value
            for key, value in {
                "minimum_fold_win_fraction": gate_config.minimum_fold_win_fraction,
                "accuracy_non_degradation_tolerance": (
                    gate_config.accuracy_non_degradation_tolerance
                ),
                "calibration_non_degradation_tolerance": (
                    gate_config.calibration_non_degradation_tolerance
                ),
                "max_runtime_seconds_per_fold": gate_config.max_runtime_seconds_per_fold,
                "max_probability_invariant_error": (
                    gate_config.max_probability_invariant_error
                ),
                "max_determinism_error": gate_config.max_determinism_error,
            }.items()
        },
        "observed": observed,
        "check_evidence": check_evidence,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
