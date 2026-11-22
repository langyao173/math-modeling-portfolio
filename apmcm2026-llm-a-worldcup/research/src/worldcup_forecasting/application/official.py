"""Official Q1--Q3 analysis with frozen temporal contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from worldcup_forecasting.config import resolved_config_dict
from worldcup_forecasting.data import CanonicalDataBundle
from worldcup_forecasting.features import build_scenario_team_features, build_team_match_features
from worldcup_forecasting.models import (
    MANDATORY_OUTCOME_FEATURES,
    OUTCOME_ORDER,
    AggregateOLS,
    HalfTimeIncrementModel,
    JointPoissonConfig,
    JointPoissonModel,
    MatchGoalOLS,
    OutcomeLogit,
)
from worldcup_forecasting.reporting.figures import (
    plot_confusion,
    plot_correlation_heatmap,
    plot_halftime_fulltime,
    plot_tournament_trend,
)
from worldcup_forecasting.reporting.model_state import stable_hash, write_model_state
from worldcup_forecasting.reporting.narrative import analyze_trend, correlation_strength
from worldcup_forecasting.reporting.tables import (
    descriptive_statistics,
    outcome_feature_correlations,
    stage_statistics,
)
from worldcup_forecasting.uncertainty.certificates import (
    build_prediction_certificate,
    prediction_payload,
    write_certificate,
)
from worldcup_forecasting.uncertainty.intervals import (
    central_discrete_interval,
    empirical_absolute_interval,
)
from worldcup_forecasting.validation.feature_availability import (
    assert_static_history_availability,
    feature_availability_registry,
)
from worldcup_forecasting.validation.leakage import (
    assert_point_in_time,
    assert_probability_simplex,
    symmetry_error,
)
from worldcup_forecasting.validation.metrics import classification_metrics

from .model_artifacts import (
    software_versions,
    write_aggregate_state,
    write_half_time_state,
    write_joint_poisson_state,
    write_match_ols_state,
    write_outcome_state,
)
from .official_protocol import (
    high_correlation_summary,
    match_scenario_ood,
    rolling_half_time_calibration,
)
from .runtime import write_csv, write_json


@dataclass(slots=True)
class OfficialRunResult:
    """Objects and metrics shared with the optional research layer."""

    team_features: pd.DataFrame
    official_metrics: dict[str, dict[str, Any]]
    figure_catalog: list[dict[str, str]]
    outcome_test_probabilities: np.ndarray
    outcome_test_labels: np.ndarray


def run_official_analysis(
    bundle: CanonicalDataBundle,
    output_dir: Path,
    config: Any,
    logger: Any,
) -> OfficialRunResult:
    """Execute the exact statement baselines and auditable scenario interfaces."""
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    reports = output_dir / "reports"
    predictions = output_dir / "predictions"
    certificates = output_dir / "certificates"
    models = output_dir / "models"
    dpi = int(config.runtime.figure_dpi)
    random_seed = int(config.runtime.primary_seed)
    config_hash = stable_hash(resolved_config_dict(config))
    versions = software_versions()

    logger.info("Q1: tournament statistics, trends, OLS, and 2018 audit")
    tournaments = bundle.tournaments.copy()
    descriptive = descriptive_statistics(tournaments)
    write_csv(descriptive, tables / "problem1_descriptive_statistics.csv")
    total_trend = analyze_trend(tournaments, "GoalsScored")
    rate_trend = analyze_trend(tournaments, "goals_per_match")
    plot_tournament_trend(
        tournaments,
        "GoalsScored",
        "Total goals",
        figures / "problem1_total_goals_trend",
        dpi,
    )
    plot_tournament_trend(
        tournaments,
        "goals_per_match",
        "Goals per match",
        figures / "problem1_goals_per_match_trend",
        dpi,
    )
    aggregate = AggregateOLS().fit(tournaments)
    aggregate_statistics = aggregate.statistics()
    attendance_2018 = aggregate.forecast_attendance(target_year=2018, target_matches_played=64)
    predicted_2018 = aggregate.forecast_total_goals(
        year=2018, qualified_teams=32, matches_played=64
    )
    aggregate_artifact = write_aggregate_state(
        aggregate,
        models / "problem1_aggregate_ols.json",
        random_seed=random_seed,
        data_cutoff="2014-07-13",
    )
    actual_2018 = float(config.protocol["2018_actual_total_goals"])
    signed_error = predicted_2018 - actual_2018
    assert aggregate.loocv_predictions_ is not None
    q1_residuals = (
        tournaments["GoalsScored"].to_numpy(float) - aggregate.loocv_predictions_
    )
    q1_interval = empirical_absolute_interval(predicted_2018, q1_residuals, coverage=0.9)
    q1_feature_schema = list(aggregate.config.feature_columns)
    q1_reference = tournaments.loc[:, q1_feature_schema].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    q1_scenario = np.asarray([32.0, 64.0, attendance_2018], dtype=float)
    q1_scale = q1_reference.std(ddof=0).replace(0, 1).to_numpy(float)
    q1_standardized_distance = (
        q1_scenario - q1_reference.mean().to_numpy(float)
    ) / q1_scale
    q1_ood_components = {
        f"{feature}_absolute_standardized_distance": float(abs(value))
        for feature, value in zip(
            q1_feature_schema, q1_standardized_distance, strict=True
        )
    }
    q1_ood = float(np.linalg.norm(q1_standardized_distance))
    q1_calibration_artifact = write_model_state(
        models / "problem1_loocv_interval_calibration.json",
        model_type="leave_one_tournament_out_absolute_residual_interval",
        feature_schema=q1_feature_schema,
        fitted_state={
            "coverage": 0.9,
            "signed_residuals": q1_residuals,
            "absolute_residuals": np.abs(q1_residuals),
            "prediction_interval": q1_interval,
        },
        training_contract={
            "data_cutoff": "2014-07-13",
            "split": "leave_one_tournament_out_within_data_through_2014",
            "target": "GoalsScored",
        },
        random_seed=random_seed,
    )
    q1_prediction_values = {
        "prediction_year": 2018,
        "data_cutoff": 2014,
        "qualified_teams": 32,
        "matches_played": 64,
        "attendance_strategy": "pre_2018_log_linear_forecast",
        "forecast_attendance": float(attendance_2018),
        "predicted_total_goals": float(predicted_2018),
        "actual_total_goals": float(actual_2018),
        "signed_error": float(signed_error),
        "absolute_error": float(abs(signed_error)),
        "empirical_loocv_interval_90_lower": float(q1_interval[0]),
        "empirical_loocv_interval_90_upper": float(q1_interval[1]),
    }
    q1_prediction = pd.DataFrame(
        [
            {
                **q1_prediction_values,
                "model_hash": aggregate_artifact.sha256,
                "model_artifact": "models/problem1_aggregate_ols.json",
                "audit_timestamp_utc": "2026-07-17T16:00:00+00:00",
            }
        ]
    )
    write_csv(q1_prediction, tables / "problem1_2018_prediction.csv")
    q1_certificate = build_prediction_certificate(
        information_set="pre_match",
        data_cutoff="2014-07-13",
        raw_data_hashes=bundle.raw_hashes,
        feature_schema=q1_feature_schema,
        model_hash=aggregate_artifact.sha256,
        model_artifact="models/problem1_aggregate_ols.json",
        config_hash=config_hash,
        random_seed=random_seed,
        software_versions=versions,
        calibration_method="leave_one_tournament_out_absolute_residual_interval",
        calibration_artifact="models/problem1_loocv_interval_calibration.json",
        prediction=q1_prediction_values,
        goal_interval_90=q1_interval,
        out_of_distribution_score=q1_ood,
        ood_components=q1_ood_components,
        invariant_status={
            "point_prediction_nonnegative": predicted_2018 >= 0,
            "interval_contains_point": q1_interval[0] <= predicted_2018 <= q1_interval[1],
            "2018_excluded_from_fit_and_calibration": int(tournaments["Year"].max()) <= 2014,
            "calibration_artifact_hash_verified": (
                q1_calibration_artifact.sha256
                == q1_calibration_artifact.path_sha256()
            ),
        },
        uncertainty_status="empirical_leave_one_tournament_out_no_coverage_guarantee",
        warning=(
            "The observed 2018 total is included only for the frozen external audit; "
            "it was excluded from model fitting and interval calibration."
        ),
        scenario_label="problem1_frozen_2018_external_audit",
    )
    write_json(
        prediction_payload(q1_certificate),
        predictions / "problem1_2018_total_goals.json",
    )
    write_certificate(
        q1_certificate,
        certificates / "problem1_2018_certificate.json",
    )
    write_json(aggregate_statistics, tables / "problem1_ols_diagnostics.json")
    coefficient_rows = []
    for name, estimate in aggregate_statistics["coefficients"].items():
        coefficient_rows.append(
            {
                "term": name,
                "estimate": estimate,
                "hc3_standard_error": aggregate_statistics["standard_errors_hc3"][name],
                "hc3_p_value": aggregate_statistics["p_values_hc3"][name],
            }
        )
    write_csv(pd.DataFrame(coefficient_rows), tables / "problem1_ols_coefficients.csv")
    (output_dir / "logs" / "problem1_regression.txt").write_text(
        str(aggregate.result_.summary()) + "\n",
        encoding="utf-8",
    )
    significant_terms = [
        name
        for name, value in aggregate_statistics["p_values_hc3"].items()
        if name != "const" and float(value) < 0.05
    ]
    finite_vif = [
        float(value)
        for value in aggregate_statistics["vif"].values()
        if np.isfinite(value)
    ]
    maximum_vif = max(finite_vif) if finite_vif else float("inf")
    summary = "\n".join(
        [
            "# Automated Q1 summary",
            "",
            total_trend.conclusion,
            "",
            rate_trend.conclusion,
            "",
            f"The frozen-formula OLS predicts {predicted_2018:.3f} goals for 2018, "
            f"versus the audit value {actual_2018:.0f} (absolute error "
            f"{abs(signed_error):.3f}).",
            f"LOOCV MAE across the 20 supplied editions is "
            f"{float(aggregate_statistics['loocv']['mae']):.3f}; the empirical 90% "
            f"absolute-residual interval is [{q1_interval[0]:.3f}, {q1_interval[1]:.3f}].",
            "HC3-significant non-intercept predictors at alpha=0.05: "
            + (", ".join(significant_terms) if significant_terms else "none"),
            f"The maximum finite VIF is {maximum_vif:.3f}; collinearity diagnostics "
            "are reported rather than hidden.",
            "This single edition is an external descriptive audit, not a model-selection sample.",
        ]
    )
    (reports / "problem1_automated_summary.md").write_text(summary + "\n", encoding="utf-8")

    logger.info("Q2: stage statistics, half/full-time relation, temporal match regression")
    matches = bundle.matches.copy()
    stage_table = stage_statistics(matches)
    write_csv(stage_table, tables / "problem2_stage_statistics.csv")
    write_csv(stage_table, tables / "problem2_stage_goal_statistics.csv")
    half_full_correlation = float(
        matches[["half_time_total_goals", "full_time_total_goals"]].corr().iloc[0, 1]
    )
    write_json(
        {
            "pearson_correlation": half_full_correlation,
            "strength": correlation_strength(half_full_correlation),
            "n": int(len(matches)),
        },
        tables / "problem2_halftime_fulltime_correlation.json",
    )
    plot_halftime_fulltime(matches, figures / "problem2_halftime_fulltime_scatter", dpi)
    match_ols = MatchGoalOLS()
    temporal_q2 = match_ols.fit_temporal(
        matches,
        train_end_year=int(config.protocol.train_end_year),
        test_year=int(config.protocol.final_test_year),
    )
    q2_feature_schema = ["canonical_stage", "half_time_total_goals", "attendance"]
    q2_schema_hash = stable_hash(q2_feature_schema)
    split_rule = (
        f"train Year <= {int(config.protocol.train_end_year)}; "
        f"test Year == {int(config.protocol.final_test_year)}"
    )
    q2_rows = [
        {
            "model": "official_match_goal_ols",
            "split": split,
            "feature_schema_hash": q2_schema_hash,
            "random_seed": random_seed,
            "split_rule": split_rule,
            **metrics,
        }
        for split, metrics in (("train", temporal_q2["train"]), ("test_2014", temporal_q2["test"]))
    ]
    write_csv(pd.DataFrame(q2_rows), tables / "problem2_metrics.csv")
    write_csv(
        pd.DataFrame(
            [{"term": key, "estimate": value} for key, value in match_ols.coefficients().items()]
        ),
        tables / "problem2_ols_coefficients.csv",
    )
    write_json(
        {
            "feature_schema": q2_feature_schema,
            "feature_schema_hash": q2_schema_hash,
            "random_seed": random_seed,
            "split_rule": split_rule,
            "training_rows": temporal_q2["train"]["n"],
            "test_rows": temporal_q2["test"]["n"],
        },
        tables / "problem2_metrics_metadata.json",
    )

    demo = config.scenarios.half_time_demo
    format_2026 = config.scenarios["2026_format"]
    scenario_match_ols = MatchGoalOLS().fit(matches)
    q2_point_prediction = scenario_match_ols.predict_match(
        stage=str(demo.stage),
        half_time_total_goals=float(demo.half_time_home_goals + demo.half_time_away_goals),
        attendance=float(demo.attendance),
    )
    match_ols_artifact = write_match_ols_state(
        scenario_match_ols,
        models / "problem2_match_goal_ols_2014.json",
        random_seed=random_seed,
        data_cutoff="2014-07-13",
    )
    scenario_ood, scenario_ood_components = match_scenario_ood(
        matches,
        half_time_total_goals=float(demo.half_time_home_goals + demo.half_time_away_goals),
        attendance=float(demo.attendance),
        target_teams=int(format_2026.qualified_teams),
        target_matches=int(format_2026.matches_played),
        historical_teams=int(tournaments["QualifiedTeams"].max()),
        historical_matches=int(tournaments["MatchesPlayed"].max()),
    )
    q2_official_prediction = {
        "stage": str(demo.stage),
        "half_time_total_goals": float(
            demo.half_time_home_goals + demo.half_time_away_goals
        ),
        "attendance": float(demo.attendance),
        "predicted_full_time_total_goals": q2_point_prediction,
    }
    q2_official_certificate = build_prediction_certificate(
        information_set="half_time",
        data_cutoff="2014-07-13",
        raw_data_hashes=bundle.raw_hashes,
        feature_schema=q2_feature_schema,
        model_hash=match_ols_artifact.sha256,
        model_artifact="models/problem2_match_goal_ols_2014.json",
        config_hash=config_hash,
        random_seed=random_seed,
        software_versions=versions,
        calibration_method="none_official_ols_point_prediction",
        calibration_artifact=None,
        prediction=q2_official_prediction,
        goal_interval_90=None,
        out_of_distribution_score=scenario_ood,
        ood_components=scenario_ood_components,
        invariant_status={
            "canonical_stage_known": str(demo.stage) in scenario_match_ols.stage_levels_,
            "nonnegative_prediction": q2_point_prediction >= 0,
            "format_shift_disclosed": True,
        },
        uncertainty_status="official_baseline_point_prediction_no_interval",
        warning=(
            "2026 is a format-shift scenario only: 48 teams and 104 matches are metadata, "
            "not observed outcomes or fixtures."
        ),
        scenario_label="problem2_2026_group_stage_reusable_input",
    )
    write_json(
        prediction_payload(q2_official_certificate),
        predictions / "problem2_2026_scenario.json",
    )
    write_certificate(
        q2_official_certificate,
        certificates / "problem2_2026_certificate.json",
    )

    half_time_calibration = rolling_half_time_calibration(
        matches,
        [int(year) for year in config.protocol.selection_test_years],
        coverage=0.9,
    )
    write_csv(
        half_time_calibration.predictions,
        tables / "problem2_half_time_rolling_oof_predictions.csv",
    )
    write_json(
        half_time_calibration.report,
        tables / "problem2_half_time_interval_diagnostics.json",
    )
    complete_matches = matches.dropna(subset=["Attendance"]).copy()
    half_time_model = HalfTimeIncrementModel().fit(complete_matches)
    half_time_prediction = half_time_model.predict_match(
        stage=str(demo.stage),
        half_time_home_goals=float(demo.half_time_home_goals),
        half_time_away_goals=float(demo.half_time_away_goals),
        attendance=float(demo.attendance),
    )
    half_time_artifact = write_half_time_state(
        half_time_model,
        models / "problem2_half_time_increment_2014.json",
        random_seed=random_seed,
        data_cutoff="2014-07-13",
    )
    calibration_artifact = write_model_state(
        models / "problem2_half_time_oof_calibration.json",
        model_type="empirical_absolute_residual_interval",
        feature_schema=["strictly_earlier_oof_residual"],
        fitted_state={
            "absolute_residuals": np.abs(half_time_calibration.residuals),
            "report": half_time_calibration.report,
        },
        training_contract={
            "maximum_selection_year": int(config.protocol.train_end_year),
            "test_years": [int(year) for year in config.protocol.selection_test_years],
            "coverage": 0.9,
        },
        random_seed=random_seed,
    )
    goal_interval = empirical_absolute_interval(
        half_time_prediction["full_time_total_goals"],
        half_time_calibration.residuals,
        coverage=0.9,
        lower_bound=half_time_prediction["half_time_total_goals"],
    )
    half_certificate = build_prediction_certificate(
        information_set="half_time",
        data_cutoff="2014-07-13",
        raw_data_hashes=bundle.raw_hashes,
        feature_schema=q2_feature_schema,
        model_hash=half_time_artifact.sha256,
        model_artifact="models/problem2_half_time_increment_2014.json",
        config_hash=config_hash,
        random_seed=random_seed,
        software_versions=versions,
        calibration_method="strictly_earlier_rolling_absolute_residual_interval",
        calibration_artifact="models/problem2_half_time_oof_calibration.json",
        prediction=half_time_prediction,
        goal_interval_90=goal_interval,
        out_of_distribution_score=scenario_ood,
        ood_components=scenario_ood_components,
        invariant_status={
            "full_time_not_below_half_time": bool(
                half_time_prediction["full_time_total_goals"]
                >= half_time_prediction["half_time_total_goals"]
            ),
            "interval_not_below_half_time": goal_interval[0]
            >= half_time_prediction["half_time_total_goals"],
            "calibration_strictly_earlier": int(
                half_time_calibration.report["selection_year_max"]
            )
            <= int(config.protocol.train_end_year),
            "calibration_artifact_hash_verified": (
                calibration_artifact.sha256 == calibration_artifact.path_sha256()
            ),
        },
        uncertainty_status="empirical_time_ordered_no_exchangeability_guarantee",
        warning=(
            "The interval is an empirical expanding-origin diagnostic, not an IID conformal "
            "coverage guarantee; the 2026 tournament format is outside the observed format."
        ),
        scenario_label="2026_scenario_only_data_cutoff_2014",
    )
    write_json(prediction_payload(half_certificate), predictions / "2026_half_time_demo.json")
    write_certificate(half_certificate, certificates / "2026_half_time_demo.json")

    logger.info("Q3: pre-tournament historical features and frozen 2014 classification")
    features = build_team_match_features(matches, tournaments)
    # Reporting keeps the audited raw-table name while model code uses target_outcome.
    features["scoreline_outcome"] = features["target_outcome"]
    assert_point_in_time(features)
    static_availability = assert_static_history_availability(features)
    write_csv(
        feature_availability_registry(),
        output_dir / "audit" / "feature_availability_registry.csv",
    )
    write_json(
        {"status": "passed", "checks": [static_availability]},
        output_dir / "audit" / "feature_availability_report.json",
    )
    correlations = outcome_feature_correlations(features)
    write_csv(
        correlations.reset_index().rename(columns={"index": "feature"}),
        tables / "problem3_feature_correlations.csv",
    )
    plot_correlation_heatmap(
        correlations,
        figures / "problem3_feature_correlation_heatmap",
        dpi,
    )
    correlation_records, correlation_markdown = high_correlation_summary(correlations)
    write_json(
        {
            "threshold_absolute_pearson": 0.1,
            "assumption": "descriptive_univariate_not_causal",
            "ranked_associations": correlation_records,
        },
        tables / "problem3_high_correlation_summary.json",
    )
    (reports / "problem3_high_correlation_summary.md").write_text(
        correlation_markdown,
        encoding="utf-8",
    )
    train = features["Year"].le(int(config.protocol.train_end_year))
    test = features["Year"].eq(int(config.protocol.final_test_year))
    outcome_model = OutcomeLogit().fit(features.loc[train])
    train_probability = outcome_model.predict_proba(features.loc[train])
    test_probability = outcome_model.predict_proba(features.loc[test])
    assert_probability_simplex(test_probability)
    train_metrics, _ = classification_metrics(
        features.loc[train, "target_outcome"].to_numpy(), train_probability
    )
    test_labels = features.loc[test, "target_outcome"].to_numpy()
    test_metrics, matrix = classification_metrics(test_labels, test_probability)
    static_audit_model = _static_joint_model(config).fit(features.loc[train])
    static_test_probability = static_audit_model.predict_proba(features.loc[test])
    static_test_metrics, _ = classification_metrics(test_labels, static_test_probability)
    write_csv(
        pd.DataFrame(
            [
                {"model": "outcome_logit", "split": "train", **train_metrics},
                {"model": "outcome_logit", "split": "test_2014", **test_metrics},
                {
                    "model": "joint_poisson_static_history",
                    "split": "test_2014_audit_only",
                    **static_test_metrics,
                },
            ]
        ),
        tables / "problem3_metrics.csv",
    )
    write_csv(
        outcome_model.coefficients().reset_index(names="outcome"),
        tables / "problem3_logit_coefficients.csv",
    )
    plot_confusion(matrix, OUTCOME_ORDER, figures / "problem3_confusion_matrix", dpi)

    scenario = config.scenarios.outcome_demo
    scenario_features = build_scenario_team_features(
        matches,
        tournaments,
        str(scenario.home_team_initials),
        str(scenario.away_team_initials),
        cutoff_year=2014,
    )
    scenario_model = OutcomeLogit().fit(features.loc[features["Year"].le(2014)])
    outcome_artifact = write_outcome_state(
        scenario_model,
        models / "problem3_outcome_logit_2014.json",
        random_seed=random_seed,
        data_cutoff="2014-07-13",
    )
    scenario_probability = scenario_model.predict_proba(scenario_features)
    swapped_probability = scenario_model.predict_proba(
        -scenario_features.loc[:, MANDATORY_OUTCOME_FEATURES].to_numpy(float)
    )
    scenario_array = scenario_features.loc[:, MANDATORY_OUTCOME_FEATURES].to_numpy(float)
    reference = features.loc[features["Year"].le(2014), MANDATORY_OUTCOME_FEATURES]
    scale = reference.std(ddof=0).replace(0, 1).to_numpy(float)
    feature_ood = float(
        np.linalg.norm((scenario_array[0] - reference.mean().to_numpy(float)) / scale)
    )
    pre_match_ood_components = {
        "feature_standardized_distance": feature_ood,
        "relative_team_count_expansion": float(
            int(format_2026.qualified_teams) / int(tournaments["QualifiedTeams"].max()) - 1.0
        ),
        "relative_match_count_expansion": float(
            int(format_2026.matches_played) / int(tournaments["MatchesPlayed"].max()) - 1.0
        ),
    }
    pre_match_ood = float(
        np.sqrt(sum(value * value for value in pre_match_ood_components.values()))
    )
    outcome_prediction = {
        label: float(scenario_probability[0, index]) for index, label in enumerate(OUTCOME_ORDER)
    }
    outcome_prediction["most_likely"] = str(OUTCOME_ORDER[int(scenario_probability.argmax())])
    outcome_certificate = build_prediction_certificate(
        information_set="pre_match",
        data_cutoff="2014-07-13",
        raw_data_hashes=bundle.raw_hashes,
        feature_schema=list(MANDATORY_OUTCOME_FEATURES),
        model_hash=outcome_artifact.sha256,
        model_artifact="models/problem3_outcome_logit_2014.json",
        config_hash=config_hash,
        random_seed=random_seed,
        software_versions=versions,
        calibration_method="none_official_baseline",
        calibration_artifact=None,
        prediction=outcome_prediction,
        goal_interval_90=None,
        out_of_distribution_score=pre_match_ood,
        ood_components=pre_match_ood_components,
        invariant_status={
            "probability_simplex": bool(np.isclose(scenario_probability.sum(), 1.0)),
            "swap_symmetry": symmetry_error(scenario_probability, swapped_probability) < 1e-10,
            "data_cutoff_is_2014": True,
            "format_shift_disclosed": True,
        },
        uncertainty_status="probabilities_uncalibrated_official_baseline",
        warning=(
            "Scenario only: teams are illustrative and no 2026 fixture, squad, result, or "
            "post-2014 performance data were used."
        ),
        scenario_label=str(scenario.label),
    )
    write_json(prediction_payload(outcome_certificate), predictions / "2026_outcome_demo.json")
    write_json(
        prediction_payload(outcome_certificate),
        predictions / "problem3_2026_scenario.json",
    )
    write_certificate(outcome_certificate, certificates / "2026_outcome_demo.json")
    write_certificate(outcome_certificate, certificates / "problem3_2026_certificate.json")

    deployable_joint = _static_joint_model(config).fit(features.loc[features["Year"].le(2014)])
    joint_artifact = write_joint_poisson_state(
        deployable_joint,
        models / "problem3_joint_poisson_static_2014.json",
        random_seed=random_seed,
        data_cutoff="2014-07-13",
    )
    joint_batch = deployable_joint.predict_joint(scenario_features)
    joint_probability = joint_batch.outcome_probabilities()
    joint_total_pmf = joint_batch.total_goal_distribution()[0]
    joint_interval = central_discrete_interval(joint_total_pmf, coverage=0.9)
    score_matrix = joint_batch.probability[0]
    score_index = np.unravel_index(int(score_matrix.argmax()), score_matrix.shape)
    swapped_joint_features = scenario_features.copy()
    for column in deployable_joint.feature_columns_:
        swapped_joint_features[column] = -scenario_features[column].to_numpy(float)
    joint_swapped_probability = deployable_joint.predict_proba(swapped_joint_features)
    joint_prediction: dict[str, Any] = {
        label: float(joint_probability[0, index])
        for index, label in enumerate(OUTCOME_ORDER)
    }
    joint_prediction.update(
        {
            "most_likely": str(OUTCOME_ORDER[int(joint_probability.argmax())]),
            "expected_home_goals": float(joint_batch.expected_home_goals[0]),
            "expected_away_goals": float(joint_batch.expected_away_goals[0]),
            "most_likely_score": f"{score_index[0]}-{score_index[1]}",
            "retained_untruncated_mass": float(joint_batch.retained_mass[0]),
        }
    )
    joint_certificate = build_prediction_certificate(
        information_set="pre_match",
        data_cutoff="2014-07-13",
        raw_data_hashes=bundle.raw_hashes,
        feature_schema=list(deployable_joint.feature_columns_),
        model_hash=joint_artifact.sha256,
        model_artifact="models/problem3_joint_poisson_static_2014.json",
        config_hash=config_hash,
        random_seed=random_seed,
        software_versions=versions,
        calibration_method="none_structural_joint_distribution",
        calibration_artifact=None,
        prediction=joint_prediction,
        goal_interval_90=joint_interval,
        out_of_distribution_score=pre_match_ood,
        ood_components=pre_match_ood_components,
        invariant_status={
            "probability_simplex": bool(np.isclose(joint_probability.sum(), 1.0)),
            "swap_symmetry": symmetry_error(joint_probability, joint_swapped_probability)
            < 1e-10,
            "joint_distribution_normalized": bool(
                np.isclose(joint_batch.probability[0].sum(), 1.0)
            ),
            "data_cutoff_is_2014": True,
            "dynamic_features_excluded_for_deployment": True,
        },
        uncertainty_status="conditional_score_distribution_parameter_uncertainty_not_included",
        warning=(
            "The 90% goal interval is conditional on fitted parameters and supplied historical "
            "features; it is not a parameter-uncertainty or conformal guarantee."
        ),
        scenario_label="deployable_static_history_joint_goal_scenario",
    )
    write_json(
        prediction_payload(joint_certificate),
        predictions / "2026_joint_goal_demo.json",
    )
    write_certificate(
        joint_certificate,
        certificates / "2026_joint_goal_demo.json",
    )

    official_metrics: dict[str, dict[str, Any]] = {
        "problem1_2018": q1_prediction.iloc[0].to_dict(),
        "problem2_2014": temporal_q2["test"],
        "problem3_2014": test_metrics,
        "problem3_static_joint_2014_audit": static_test_metrics,
    }
    catalog = _figure_catalog(total_trend.conclusion, rate_trend.conclusion, half_full_correlation)
    logger.info("Official Q1--Q3 analysis complete")
    return OfficialRunResult(features, official_metrics, catalog, test_probability, test_labels)


def _static_joint_model(config: Any) -> JointPoissonModel:
    """Return the frozen deployable joint-goal specification without dynamic features."""
    return JointPoissonModel(
        JointPoissonConfig(
            optional_diff_features=(),
            alpha=float(config.models.poisson_alpha),
            time_decay_half_life_years=float(config.models.time_decay_half_life_years),
            rho=None,
            max_score=int(config.models.max_score),
        )
    )


def _figure_catalog(total_note: str, rate_note: str, correlation: float) -> list[dict[str, str]]:
    common = {
        "data_source": "Supplied, SHA-256-verified World Cup CSV files",
        "caveat": "Descriptive historical evidence through 2014; no post-2014 observations used.",
    }
    return [
        {
            "filename": "problem1_total_goals_trend.png",
            "purpose": "Show tournament total-goal evolution.",
            "observation": total_note,
            "interpretation": "Total goals combine scoring intensity and tournament size.",
            **common,
        },
        {
            "filename": "problem1_goals_per_match_trend.png",
            "purpose": "Separate scoring intensity from match count.",
            "observation": rate_note,
            "interpretation": "The rate is the appropriate complement to the aggregate series.",
            **common,
        },
        {
            "filename": "problem2_halftime_fulltime_scatter.png",
            "purpose": "Assess half-time information for full-time goals.",
            "observation": f"Pearson correlation is {correlation:.3f}.",
            "interpretation": (
                "Half-time score is predictive but does not determine the final total."
            ),
            **common,
        },
    ]
