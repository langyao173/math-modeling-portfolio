"""Semantic checks over verified pre-audit evidence files."""

from __future__ import annotations

import io
import json
from pathlib import Path
from statistics import NormalDist
from typing import Any

import joblib
import numpy as np
import pandas as pd

from worldcup_forecasting.experiments._nested_selection_runtime import (
    REFERENCE_RUNTIME_HASH,
    reference_runtime_seconds,
)
from worldcup_forecasting.experiments.statistical_closure import (
    build_statistical_closure,
)
from worldcup_forecasting.simulation.adaptive import (
    FROZEN_ADAPTIVE_CONFIG,
    FROZEN_ADAPTIVE_SEED,
    FROZEN_PROTOCOL_DEVIATION_SHA256,
    adaptive_contract_payload,
    adaptive_contract_sha256,
)
from worldcup_forecasting.simulation.format_policy import (
    FIFA_2026_ANNEX_C_CSV_SHA256,
    FIFA_2026_REGULATIONS_SHA256,
)
from worldcup_forecasting.uncertainty.decomposition import (
    hierarchical_variance_decomposition,
)

from .artifact_integrity import read_verified_artifact_bytes, verify_stage_manifest
from .promotion import evaluate_advanced_promotion, evaluate_calibration_retention
from .research_contracts import validate_ablation_matrix, validate_stress_matrix
from .runtime_reproducibility import (
    ABLATION_REFERENCE_RUNTIME_SECONDS,
    BASELINE_REFERENCE_RUNTIME_SECONDS,
    LIVE_RUNTIME_COLUMNS,
    RUNTIME_REFERENCE_PROTOCOL,
    RUNTIME_REFERENCE_PROTOCOL_SHA256,
)

EXPECTED_YEARS = (1986, 1990, 1994, 1998, 2002, 2006, 2010)
_OUTCOMES = ("home_win", "draw", "away_win")


def evaluate_verified_evidence(root: Path) -> dict[str, Any]:
    """Evaluate the complete preregistered gate after byte verification."""
    identity = _json(root / "nested" / "frozen_model_identity.json")
    nested_metadata = _json(root / "nested" / "nested_metadata.json")
    combined = _csv(root / "evaluation" / "combined_fold_metrics.csv")
    raw = _csv(root / "evaluation" / "raw_fold_metrics.csv")
    calibrated = _csv(root / "evaluation" / "calibrated_fold_metrics.csv")
    decision = _csv(root / "evaluation" / "decision_fold_metrics.csv")
    ledger = _csv(root / "evaluation" / "calibrated_oof_ledger.csv")
    calibrated_grid = _csv(root / "evaluation" / "calibrated_joint_grid.csv")
    raw_grid = _csv(root / "nested" / "outer_joint_grid.csv")

    fold_checks = _seven_fold_checks(ledger, calibrated_grid, decision)
    projection = _projection_checks(ledger, raw_grid, calibrated_grid)
    stress = validate_stress_matrix(_csv(root / "stress" / "structural_stress_matrix.csv"))
    stress_frame = _csv(root / "stress" / "structural_stress_matrix.csv")
    ablation = validate_ablation_matrix(_csv(root / "ablation" / "ablation_matrix.csv"))
    simulation = _simulation_checks(root)
    uncertainty = _uncertainty_checks(root)
    provenance = _provenance_checks(root, identity, nested_metadata, stress_frame)
    calibration = evaluate_calibration_retention(raw, calibrated)
    statistical_closure = _statistical_closure_checks(root, combined)
    runtime_reproducibility = _runtime_reproducibility_checks(root)

    structural_checks = {
        "simplex": bool(all(item["simplex"] for item in fold_checks)),
        "symmetry": bool(stress["maximum_symmetry_error"] <= 1e-10),
        "leakage": bool(all(item["oof_and_chronological"] for item in fold_checks)),
        "score_mass": bool(projection["all_joint_masses_normalized"]),
        "score_outcome_coherence": bool(projection["outcome_margins_recovered"]),
        "within_partition_projection": bool(projection["within_partition_ratios_constant"]),
        "halftime_monotonicity": bool(stress["maximum_halftime_violation"] <= 1e-12),
        "seven_fold_chronology": bool(
            len(fold_checks) == 7 and all(item["passed"] for item in fold_checks)
        ),
        "ablation_complete_126": bool(
            ablation["row_count"] == 126 and ablation["evaluation_unit_count"] == 14
        ),
        "tournament_cardinality": bool(simulation["passed"]),
        "deterministic_provenance": bool(provenance["passed"]),
        "deterministic_runtime_references": bool(runtime_reproducibility["passed"]),
    }
    gate = evaluate_advanced_promotion(
        combined,
        decision,
        candidate="advanced_nested_calibrated",
        baseline="legacy_mirrored_dixon_coles",
        # The promoted scientific object is the complete nested-selection
        # procedure.  The final-fit hash is also verified below, but would not
        # by itself identify the cross-fitting and selection rules evaluated by
        # the seven outer folds.
        model_identity_hash=str(identity.get("selection_procedure_hash", "")),
        artifact_identity_hash=str(provenance.get("selection_procedure_hash", "")),
        structural_checks=structural_checks,
    )
    gate["calibration_retention"] = calibration
    gate["seven_fold_checks"] = fold_checks
    gate["projection_checks"] = projection
    gate["stress_checks"] = stress
    gate["ablation_checks"] = ablation
    gate["simulation_checks"] = simulation
    gate["uncertainty_checks"] = uncertainty
    gate["provenance_checks"] = provenance
    gate["statistical_closure"] = statistical_closure
    gate["runtime_reproducibility"] = runtime_reproducibility
    gate["checks"]["calibration_retained_under_frozen_h2"] = bool(
        calibration["retained_under_frozen_h2"]
    )
    gate["checks"]["uncertainty_and_decomposition_complete"] = bool(uncertainty["passed"])
    gate["checks"]["statistical_closure_complete"] = statistical_closure["passed"]
    gate["system_promoted"] = bool(
        gate["system_promoted"]
        and gate["checks"]["calibration_retained_under_frozen_h2"]
        and gate["checks"]["uncertainty_and_decomposition_complete"]
        and gate["checks"]["statistical_closure_complete"]
    )
    gate["warnings"] = []
    if calibration["ece_regressed"]:
        gate["warnings"].append(
            "Secondary ECE regressed; calibration is a mixed result and must not be "
            "described as uniformly beneficial."
        )
    return gate


def _runtime_reproducibility_checks(root: Path) -> dict[str, Any]:
    """Verify that live wall-clock noise cannot enter sealed science artifacts."""
    declaration = _json(root / "nested" / "runtime_reproducibility.json")
    nested_metadata = _json(root / "nested" / "nested_metadata.json")
    baseline_metadata = _json(root / "evaluation" / "baseline_metadata.json")
    ablation_metadata = _json(root / "ablation" / "ablation_metadata.json")
    candidate = _csv(root / "nested" / "candidate_metrics.csv")
    selected = _csv(root / "nested" / "selected_specs.csv")
    final_candidate = _csv(root / "nested" / "final_candidate_metrics.csv")
    final_selected = _csv(root / "nested" / "final_selected_summary.csv")
    baseline_outcome = _csv(root / "evaluation" / "baseline_outcome_fold_metrics.csv")
    baseline_joint = _csv(root / "evaluation" / "baseline_joint_fold_metrics.csv")
    ablation = _csv(root / "ablation" / "ablation_matrix.csv")
    nested_binding = verify_stage_manifest(
        root / "nested" / "artifact_manifest.json",
        required_paths=(
            "nested_result.joblib",
            "final_selection.joblib",
            "final_fitted_model.joblib",
        ),
    )
    evaluation_binding = verify_stage_manifest(
        root / "evaluation" / "artifact_manifest.json",
        required_paths=("baseline_result.joblib",),
    )

    def load_bound_joblib(directory: str, relative_path: str, binding: dict[str, Any]) -> Any:
        return joblib.load(
            io.BytesIO(
                read_verified_artifact_bytes(
                    root / directory,
                    relative_path,
                    binding,
                )
            )
        )

    nested_result = load_bound_joblib("nested", "nested_result.joblib", nested_binding)
    final_selection = load_bound_joblib("nested", "final_selection.joblib", nested_binding)
    final_fitted = load_bound_joblib("nested", "final_fitted_model.joblib", nested_binding)
    baseline_result = load_bound_joblib(
        "evaluation",
        "baseline_result.joblib",
        evaluation_binding,
    )

    expected_declaration = {
        **RUNTIME_REFERENCE_PROTOCOL,
        "runtime_reference_protocol_sha256": RUNTIME_REFERENCE_PROTOCOL_SHA256,
        "candidate_runtime_reference_hash": REFERENCE_RUNTIME_HASH,
    }
    declaration_exact = all(
        declaration.get(key) == value for key, value in expected_declaration.items()
    ) and set(declaration) == {*expected_declaration, "final_fit_runtime_reference_seconds"}
    forbidden_absent = all(
        column not in frame.columns
        for frame in (candidate, selected, final_candidate, final_selected)
        for column in LIVE_RUNTIME_COLUMNS
    )
    def candidate_runtime_exact(frame: pd.DataFrame) -> bool:
        if "candidate_id" not in frame or "runtime_seconds" not in frame:
            return False
        observed = pd.to_numeric(frame["runtime_seconds"], errors="coerce").to_numpy(float)
        expected = np.asarray(
            [reference_runtime_seconds(str(value)) for value in frame["candidate_id"]],
            dtype=float,
        )
        return bool(np.isfinite(observed).all() and np.array_equal(observed, expected))

    candidate_reference_exact = bool(
        nested_metadata.get("runtime_reference_hash") == REFERENCE_RUNTIME_HASH
        and all(
            candidate_runtime_exact(frame)
            for frame in (candidate, selected, final_candidate, final_selected)
        )
    )
    final_reference_exact = bool(
        len(final_selected) == 1
        and "candidate_id" in final_selected
        and "runtime_seconds" in final_selected
        and float(declaration.get("final_fit_runtime_reference_seconds", np.nan))
        == reference_runtime_seconds(str(final_selected["candidate_id"].iloc[0]))
        and float(final_selected["runtime_seconds"].iloc[0])
        == float(declaration["final_fit_runtime_reference_seconds"])
    )

    def mapped_runtime_exact(
        frame: pd.DataFrame,
        identity_column: str,
        mapping: dict[str, float],
    ) -> bool:
        if identity_column not in frame or "runtime_seconds" not in frame:
            return False
        identities = frame[identity_column].astype(str)
        if not set(identities).issubset(mapping):
            return False
        observed = pd.to_numeric(frame["runtime_seconds"], errors="coerce").to_numpy(float)
        expected = identities.map(mapping).to_numpy(float)
        return bool(np.isfinite(observed).all() and np.array_equal(observed, expected))

    baseline_reference_exact = mapped_runtime_exact(
        baseline_outcome,
        "model",
        BASELINE_REFERENCE_RUNTIME_SECONDS,
    ) and mapped_runtime_exact(
        baseline_joint,
        "model",
        BASELINE_REFERENCE_RUNTIME_SECONDS,
    )
    ablation_reference_exact = mapped_runtime_exact(
        ablation,
        "variant_id",
        ABLATION_REFERENCE_RUNTIME_SECONDS,
    )

    def frames_equivalent(left: Any, right: pd.DataFrame) -> bool:
        if not isinstance(left, pd.DataFrame):
            return False
        left = left.reset_index(drop=True)
        right = right.reset_index(drop=True)
        if list(left.columns) != list(right.columns) or len(left) != len(right):
            return False
        for column in left.columns:
            left_numeric = pd.to_numeric(left[column], errors="coerce")
            right_numeric = pd.to_numeric(right[column], errors="coerce")
            numeric_mask = left[column].notna() | right[column].notna()
            numeric_column = bool(
                (~numeric_mask).all()
                or (
                    left_numeric[numeric_mask].notna().all()
                    and right_numeric[numeric_mask].notna().all()
                )
            )
            if numeric_column:
                if not np.allclose(
                    left_numeric.to_numpy(float),
                    right_numeric.to_numpy(float),
                    rtol=0,
                    atol=1e-14,
                    equal_nan=True,
                ):
                    return False
            elif not np.array_equal(
                left[column].fillna("").astype(str).to_numpy(),
                right[column].fillna("").astype(str).to_numpy(),
            ):
                return False
        return True

    joblib_frames_exact = bool(
        frames_equivalent(nested_result.selected_specs, selected)
        and frames_equivalent(nested_result.candidate_metrics, candidate)
        and frames_equivalent(final_selection.selected_summary, final_selected)
        and frames_equivalent(final_selection.candidate_metrics, final_candidate)
        and frames_equivalent(baseline_result.outcome_fold_metrics, baseline_outcome)
        and frames_equivalent(baseline_result.joint_fold_metrics, baseline_joint)
    )
    joblib_runtime_state_exact = bool(
        getattr(final_fitted, "fit_runtime_seconds", None)
        == reference_runtime_seconds(str(final_fitted.selected_spec.candidate_id))
        and final_fitted.metadata.get("fit_runtime_seconds")
        == declaration.get("final_fit_runtime_reference_seconds")
        and baseline_result.metadata.get("runtime_reference_protocol_sha256")
        == RUNTIME_REFERENCE_PROTOCOL_SHA256
    )
    metadata_bound = bool(
        baseline_metadata.get("runtime_semantics")
        == RUNTIME_REFERENCE_PROTOCOL["semantics"]
        and baseline_metadata.get("runtime_reference_protocol_sha256")
        == RUNTIME_REFERENCE_PROTOCOL_SHA256
        and ablation_metadata.get("runtime_semantics")
        == RUNTIME_REFERENCE_PROTOCOL["semantics"]
        and ablation_metadata.get("runtime_reference_protocol_sha256")
        == RUNTIME_REFERENCE_PROTOCOL_SHA256
    )
    checks = {
        "runtime_declaration_exact": declaration_exact,
        "live_runtime_columns_absent": forbidden_absent,
        "candidate_reference_exact": candidate_reference_exact,
        "final_fit_reference_exact": final_reference_exact,
        "baseline_reference_exact": baseline_reference_exact,
        "ablation_reference_exact": ablation_reference_exact,
        "joblib_frames_match_sealed_csv": joblib_frames_exact,
        "joblib_runtime_state_exact": joblib_runtime_state_exact,
        "metadata_hashes_bound": metadata_bound,
    }
    return {**checks, "passed": all(checks.values())}


def _statistical_closure_checks(root: Path, combined_fold_metrics: pd.DataFrame) -> dict[str, Any]:
    """Independently rebuild D-007 from verified source tables."""
    recomputed = build_statistical_closure(
        combined_fold_metrics,
        _csv(root / "evaluation" / "paired_tournament_intervals.csv"),
        _csv(root / "nested" / "selected_specs.csv"),
        _csv(root / "evaluation" / "raw_classwise_calibration.csv"),
        _csv(root / "evaluation" / "calibrated_classwise_calibration.csv"),
    )
    declared_summary = _json(root / "evaluation" / "statistical_closure.json")
    declared_tables = (
        _csv(root / "evaluation" / "statistical_fold_comparison.csv"),
        _csv(root / "evaluation" / "statistical_model_ranking.csv"),
        _csv(root / "evaluation" / "statistical_classwise_calibration.csv"),
    )
    recomputed_tables = (
        recomputed.fold_comparison,
        recomputed.model_ranking,
        recomputed.classwise_calibration,
    )
    tables_match = all(
        _frame_matches(declared, expected)
        for declared, expected in zip(declared_tables, recomputed_tables, strict=True)
    )
    summary_matches = _payload_matches(declared_summary, recomputed.summary)
    return {
        "recomputed": recomputed.summary,
        "declared_summary_matches": summary_matches,
        "declared_tables_match": tables_match,
        "passed": bool(recomputed.summary["passed"] and summary_matches and tables_match),
    }


def _frame_matches(declared: pd.DataFrame, expected: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(
            declared,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
    except AssertionError:
        return False
    return True


def _payload_matches(declared: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(declared, dict)
            and set(declared) == set(expected)
            and all(_payload_matches(declared[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(declared, list)
            and len(declared) == len(expected)
            and all(
                _payload_matches(left, right)
                for left, right in zip(declared, expected, strict=True)
            )
        )
    if isinstance(expected, bool):
        return isinstance(declared, bool) and declared is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(declared, (int, float))
            and not isinstance(declared, bool)
            and bool(np.isclose(float(declared), float(expected), rtol=0.0, atol=1e-12))
        )
    return declared == expected


def _seven_fold_checks(
    ledger: pd.DataFrame,
    grid: pd.DataFrame,
    decision: pd.DataFrame,
) -> list[dict[str, Any]]:
    required = {
        "outer_fold_id",
        "match_id",
        "prediction_year",
        "prediction_time",
        "train_end_year",
        "is_oof",
        "home_win",
        "draw",
        "away_win",
        "target_outcome",
        "observed_home_goals",
        "observed_away_goals",
    }
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise ValueError(f"Calibrated OOF ledger is missing columns: {missing}")
    years = pd.to_numeric(ledger["prediction_year"], errors="raise").astype(int)
    if set(years) != set(EXPECTED_YEARS):
        raise ValueError("Calibrated OOF ledger must contain the seven frozen outer editions")
    decision_year = pd.to_numeric(decision["test_year"], errors="raise").astype(int)
    if set(decision_year) != set(EXPECTED_YEARS):
        raise ValueError("Decision evidence must contain the seven frozen outer editions")
    rows: list[dict[str, Any]] = []
    for year in EXPECTED_YEARS:
        block = ledger.loc[years.eq(year)].copy()
        probability = block[list(_OUTCOMES)].apply(pd.to_numeric, errors="raise").to_numpy(float)
        goals = (
            block[["observed_home_goals", "observed_away_goals"]]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(float)
        )
        train_end = pd.to_numeric(block["train_end_year"], errors="raise").to_numpy(float)
        derived = np.where(
            goals[:, 0] > goals[:, 1],
            "home_win",
            np.where(goals[:, 0] == goals[:, 1], "draw", "away_win"),
        )
        timestamps = pd.to_datetime(block["prediction_time"], errors="coerce")
        grid_keys = grid.loc[grid["outer_fold_id"].eq(f"outer_{year}"), "match_id"]
        decision_row = decision.loc[decision_year.eq(year)]
        checks = {
            "unique_match_keys": bool(
                not block["match_id"].isna().any() and not block["match_id"].duplicated().any()
            ),
            "oof_and_chronological": bool(
                _strict_true(block["is_oof"])
                and np.isfinite(train_end).all()
                and np.all(train_end < year)
            ),
            "prediction_time_valid": bool(
                not timestamps.isna().any() and np.all(timestamps.dt.year.to_numpy() == year)
            ),
            "simplex": bool(
                np.isfinite(probability).all()
                and np.all(probability >= 0)
                and np.allclose(probability.sum(axis=1), 1.0, atol=1e-10)
            ),
            "target_score_consistency": bool(
                np.isfinite(goals).all()
                and np.all(goals >= 0)
                and np.equal(goals, np.floor(goals)).all()
                and np.array_equal(derived, block["target_outcome"].astype(str).to_numpy())
            ),
            "joint_key_coverage": bool(set(grid_keys) == set(block["match_id"])),
            "decision_trained_before_fold": bool(
                len(decision_row) == 1
                and int(decision_row["training_max_prediction_year"].iloc[0]) < year
                and not _truthy(decision_row["probability_layer_changed"].iloc[0])
            ),
        }
        rows.append({"test_year": year, **checks, "passed": bool(all(checks.values()))})
    return rows


def _projection_checks(
    ledger: pd.DataFrame,
    raw_grid: pd.DataFrame,
    calibrated_grid: pd.DataFrame,
) -> dict[str, Any]:
    keys = ["outer_fold_id", "match_id", "home_goals", "away_goals"]
    for label, frame in (("raw", raw_grid), ("calibrated", calibrated_grid)):
        missing = sorted(set([*keys, "probability"]).difference(frame.columns))
        if missing:
            raise ValueError(f"{label} joint grid is missing columns: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} joint grid has duplicate score cells")
    merged = raw_grid[keys + ["probability"]].merge(
        calibrated_grid[keys + ["probability"]],
        on=keys,
        suffixes=("_raw", "_calibrated"),
        how="outer",
        validate="one_to_one",
    )
    numeric_columns = [
        "home_goals",
        "away_goals",
        "probability_raw",
        "probability_calibrated",
    ]
    numeric = merged[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Raw/calibrated joint grids must align with finite cells")
    merged = merged.assign(
        home_goals=numeric["home_goals"].astype(int),
        away_goals=numeric["away_goals"].astype(int),
        probability_raw=numeric["probability_raw"],
        probability_calibrated=numeric["probability_calibrated"],
    )
    merged["outcome"] = np.where(
        merged["home_goals"] > merged["away_goals"],
        "home_win",
        np.where(merged["home_goals"] == merged["away_goals"], "draw", "away_win"),
    )
    masses = merged.groupby(["outer_fold_id", "match_id"], sort=False)[
        "probability_calibrated"
    ].sum()
    margins = (
        merged.groupby(["outer_fold_id", "match_id", "outcome"], sort=False)[
            "probability_calibrated"
        ]
        .sum()
        .unstack("outcome", fill_value=0)
        .reset_index()
    )
    compared = ledger[["outer_fold_id", "match_id", *_OUTCOMES]].merge(
        margins,
        on=["outer_fold_id", "match_id"],
        suffixes=("_ledger", "_grid"),
        validate="one_to_one",
    )
    margin_error = max(
        float(np.max(np.abs(compared[f"{name}_ledger"] - compared[f"{name}_grid"])))
        for name in _OUTCOMES
    )
    positive = merged.loc[merged["probability_raw"].gt(1e-15)].copy()
    positive["ratio"] = positive["probability_calibrated"] / positive["probability_raw"]
    relative_spread = positive.groupby(["outer_fold_id", "match_id", "outcome"], sort=False)[
        "ratio"
    ].agg(lambda values: float(np.ptp(values) / max(1.0, abs(values.mean()))))
    return {
        "all_joint_masses_normalized": bool(np.allclose(masses.to_numpy(), 1.0, atol=1e-10)),
        "maximum_joint_mass_error": float(np.max(np.abs(masses.to_numpy() - 1.0))),
        "outcome_margins_recovered": bool(margin_error <= 1e-10),
        "maximum_outcome_margin_error": margin_error,
        "within_partition_ratios_constant": bool(relative_spread.max() <= 1e-10),
        "maximum_within_partition_relative_ratio_spread": float(relative_spread.max()),
    }


def _simulation_checks(root: Path) -> dict[str, Any]:
    metadata = _json(root / "simulation" / "scenario_metadata.json")
    stage = _csv(root / "simulation" / "scenario_stage_summary.csv")
    teams = _csv(root / "simulation" / "scenario_team_summary.csv")
    stopping = _csv(root / "simulation" / "adaptive_stopping_ledger.csv")
    batch_counts = _csv(root / "simulation" / "adaptive_batch_event_counts.csv.gz")
    sufficient = _csv(root / "simulation" / "continuous_sufficient_statistics.csv")
    summary = _json(root / "simulation_stage_summary.json")
    match_log = pd.read_csv(root / "simulation" / "scenario_match_log.csv.gz")
    config = FROZEN_ADAPTIVE_CONFIG
    expected_stage = {
        "group_stage": 72,
        "round_of_32": 16,
        "round_of_16": 8,
        "quarter_final": 4,
        "semi_final": 2,
        "third_place": 1,
        "final": 1,
    }
    if not {"stage", "matches_per_simulation", "simulated_matches"}.issubset(stage.columns):
        raise ValueError("Simulation stage summary schema is incomplete")
    if stage["stage"].duplicated().any() or set(stage["stage"].astype(str)) != set(expected_stage):
        raise ValueError("Simulation stage names must be the seven exact stages")
    stage_matches = pd.to_numeric(stage["simulated_matches"], errors="raise").to_numpy(float)
    if (
        not np.isfinite(stage_matches).all()
        or not np.equal(stage_matches, np.floor(stage_matches)).all()
    ):
        raise ValueError("Simulation stage counts must be finite integers")
    stage_counts = dict(zip(stage["stage"], stage_matches.astype(int), strict=True))
    simulations = summary.get("simulations")
    if isinstance(simulations, bool) or not isinstance(simulations, int):
        raise ValueError("Simulation count must be an exact integer")
    audit_simulations = config.audit_log_simulations
    exact_stage = all(
        stage_counts.get(name, -1) == count * simulations for name, count in expected_stage.items()
    )
    numeric_ids = pd.to_numeric(match_log["simulation_id"], errors="raise").to_numpy(float)
    if not np.isfinite(numeric_ids).all() or not np.equal(numeric_ids, np.floor(numeric_ids)).all():
        raise ValueError("Simulation identifiers must be finite integers")
    per_simulation = pd.Series(numeric_ids.astype(int)).value_counts(sort=False)
    required_team_columns = {
        "team",
        "expected_group_points",
        "expected_group_points_mcse",
        "expected_tournament_goals",
        "expected_tournament_goals_mcse",
    }
    for event in (
        "knockout",
        "round_of_16",
        "quarter_final",
        "semi_final",
        "final",
        "champion",
    ):
        required_team_columns.update(
            {
                f"{event}_count",
                f"{event}_probability",
                f"{event}_mcse",
                f"{event}_ci_lower",
                f"{event}_ci_upper",
            }
        )
    if not required_team_columns.issubset(teams.columns):
        raise ValueError("Simulation team summary schema is incomplete")
    team_names = set(teams["team"].astype(str))
    event_order = (
        "knockout",
        "round_of_16",
        "quarter_final",
        "semi_final",
        "final",
        "champion",
    )
    expected_event_totals = {
        "knockout": 32 * simulations,
        "round_of_16": 16 * simulations,
        "quarter_final": 8 * simulations,
        "semi_final": 4 * simulations,
        "final": 2 * simulations,
        "champion": simulations,
    }
    event_probabilities = (
        teams[[f"{event}_probability" for event in event_order]]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(float)
    )
    monotone = bool(np.all(np.diff(event_probabilities, axis=1) <= 1e-12))
    mcse_errors: list[float] = []
    interval_errors: list[float] = []
    counts_valid = True
    counts_conserved = True
    confidence_level = config.confidence_level
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    for event in event_order:
        count = pd.to_numeric(teams[f"{event}_count"], errors="raise").to_numpy(float)
        probability = pd.to_numeric(teams[f"{event}_probability"], errors="raise").to_numpy(float)
        observed_mcse = pd.to_numeric(teams[f"{event}_mcse"], errors="raise").to_numpy(float)
        if not (
            np.isfinite(count).all()
            and np.isfinite(probability).all()
            and np.isfinite(observed_mcse).all()
            and np.all((0 <= count) & (count <= simulations))
            and np.equal(count, np.floor(count)).all()
            and np.all((0 <= probability) & (probability <= 1))
            and np.all(observed_mcse >= 0)
        ):
            raise ValueError(f"Simulation {event} counts/probabilities are malformed")
        expected_mcse = np.sqrt(probability * (1.0 - probability) / simulations)
        mcse_errors.append(float(np.max(np.abs(observed_mcse - expected_mcse))))
        lower = pd.to_numeric(teams[f"{event}_ci_lower"], errors="raise").to_numpy(float)
        upper = pd.to_numeric(teams[f"{event}_ci_upper"], errors="raise").to_numpy(float)
        denominator = 1.0 + z**2 / simulations
        center = (probability + z**2 / (2.0 * simulations)) / denominator
        radius = (
            z
            * np.sqrt(
                probability * (1.0 - probability) / simulations + z**2 / (4.0 * simulations**2)
            )
            / denominator
        )
        expected_lower = np.clip(center - radius, 0.0, 1.0)
        expected_upper = np.clip(center + radius, 0.0, 1.0)
        interval_errors.extend(
            [
                float(np.max(np.abs(lower - expected_lower))),
                float(np.max(np.abs(upper - expected_upper))),
            ]
        )
        counts_valid = counts_valid and True
        counts_conserved = counts_conserved and bool(
            int(count.sum()) == expected_event_totals[event]
        )
        if not np.allclose(probability, count / simulations, atol=1e-12):
            raise ValueError(f"Simulation {event} probabilities do not match counts")
    expected_audit_stage = {
        name: count * audit_simulations for name, count in expected_stage.items()
    }
    audit_stage = match_log["stage"].value_counts().to_dict()
    maximum_advancement_mcse = float(
        teams[[f"{event}_mcse" for event in event_order[:-1]]]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(float)
        .max()
    )
    maximum_champion_mcse = float(teams["champion_mcse"].max())
    exact_contract = {
        "minimum_simulations": config.minimum_simulations,
        "batch_size": config.batch_size,
        "maximum_simulations": config.maximum_simulations,
        "advancement_mcse_threshold": config.advancement_mcse_threshold,
        "champion_mcse_threshold": config.champion_mcse_threshold,
        "confidence_level": config.confidence_level,
        "audit_log_simulations": config.audit_log_simulations,
        "seed": FROZEN_ADAPTIVE_SEED,
        "adaptive_contract_sha256": adaptive_contract_sha256(),
        "protocol_deviation_sha256": FROZEN_PROTOCOL_DEVIATION_SHA256,
    }
    contract_locked = (
        all(
            summary.get(key) == value and metadata.get(key) == value
            for key, value in exact_contract.items()
        )
        and metadata.get("adaptive_contract") == adaptive_contract_payload()
    )
    thresholds_satisfied = bool(
        maximum_advancement_mcse <= config.advancement_mcse_threshold + 1e-12
        and maximum_champion_mcse <= config.champion_mcse_threshold + 1e-12
    )
    valid_adaptive_terminal = (
        thresholds_satisfied
        and summary.get("mcse_thresholds_satisfied") is True
        and metadata.get("mcse_thresholds_satisfied") is True
        and summary.get("stopping_reason") == "all_mcse_thresholds_satisfied"
        and metadata.get("stopping_reason") == "all_mcse_thresholds_satisfied"
    ) or (
        not thresholds_satisfied
        and simulations == config.maximum_simulations
        and summary.get("mcse_thresholds_satisfied") is False
        and metadata.get("mcse_thresholds_satisfied") is False
        and summary.get("stopping_reason") == "maximum_simulations_reached"
        and metadata.get("stopping_reason") == "maximum_simulations_reached"
    )
    stopping_required = {
        "batch_index",
        "simulations",
        "maximum_advancement_mcse",
        "maximum_champion_mcse",
        "minimum_reached",
        "advancement_threshold_satisfied",
        "champion_threshold_satisfied",
        "thresholds_satisfied",
        "maximum_reached",
        "decision",
    }
    if not stopping_required.issubset(stopping.columns) or stopping.empty:
        raise ValueError("Adaptive stopping ledger schema is incomplete")
    batch_count_required = {"batch_index", "simulations", "team", "event", "count"}
    if not batch_count_required.issubset(batch_counts.columns) or batch_counts.empty:
        raise ValueError("Adaptive batch-count evidence schema is incomplete")
    ledger_simulations = pd.to_numeric(stopping["simulations"], errors="raise").to_numpy(float)
    ledger_advancement = pd.to_numeric(
        stopping["maximum_advancement_mcse"], errors="raise"
    ).to_numpy(float)
    ledger_champion = pd.to_numeric(stopping["maximum_champion_mcse"], errors="raise").to_numpy(
        float
    )
    ledger_batches = pd.to_numeric(stopping["batch_index"], errors="raise").to_numpy(float)
    ledger_minimum = _strict_bool_values(stopping["minimum_reached"])
    ledger_advancement_satisfied = _strict_bool_values(stopping["advancement_threshold_satisfied"])
    ledger_champion_satisfied = _strict_bool_values(stopping["champion_threshold_satisfied"])
    ledger_thresholds = _strict_bool_values(stopping["thresholds_satisfied"])
    ledger_maximum = _strict_bool_values(stopping["maximum_reached"])
    expected_batches = np.arange(
        config.batch_size, simulations + config.batch_size, config.batch_size, dtype=float
    )
    batch_numeric = batch_counts[["batch_index", "simulations", "count"]].apply(
        pd.to_numeric, errors="raise"
    )
    batch_counts = batch_counts.assign(
        batch_index=batch_numeric["batch_index"],
        simulations=batch_numeric["simulations"],
        count=batch_numeric["count"],
    )
    batch_counts["event"] = batch_counts["event"].astype(str)
    batch_counts["team"] = batch_counts["team"].astype(str)
    batch_index_values = batch_counts["batch_index"].to_numpy(float)
    batch_basic_valid = bool(
        len(batch_counts) == len(stopping) * 48 * len(event_order)
        and not batch_counts.duplicated(["batch_index", "team", "event"]).any()
        and set(batch_counts["team"]) == team_names
        and set(batch_counts["event"]) == set(event_order)
        and np.isfinite(batch_numeric.to_numpy(float)).all()
        and np.equal(batch_numeric.to_numpy(float), np.floor(batch_numeric.to_numpy(float))).all()
        and set(batch_index_values.astype(int)) == set(range(len(stopping)))
        and (batch_counts["count"] >= 0).all()
        and (batch_counts["count"] <= batch_counts["simulations"]).all()
        and batch_counts.groupby("batch_index").size().eq(48 * len(event_order)).all()
        and np.array_equal(
            batch_counts.groupby("batch_index", sort=True)["simulations"].first().to_numpy(float),
            expected_batches,
        )
        and batch_counts.groupby("batch_index")["simulations"].nunique().eq(1).all()
    )
    if not batch_basic_valid:
        raise ValueError("Adaptive batch-count evidence is malformed")
    batch_counts["probability"] = batch_counts["count"] / batch_counts["simulations"]
    batch_counts["mcse"] = np.sqrt(
        batch_counts["probability"]
        * (1.0 - batch_counts["probability"])
        / batch_counts["simulations"]
    )
    recomputed_advancement = (
        batch_counts.loc[batch_counts["event"].ne("champion")]
        .groupby("batch_index", sort=True)["mcse"]
        .max()
        .to_numpy(float)
    )
    recomputed_champion = (
        batch_counts.loc[batch_counts["event"].eq("champion")]
        .groupby("batch_index", sort=True)["mcse"]
        .max()
        .to_numpy(float)
    )
    observed_batch_totals = batch_counts.groupby(["batch_index", "event"], sort=True)["count"].sum()
    batch_totals_valid = all(
        int(observed_batch_totals.loc[(batch_index, event)])
        == slots * int(expected_batches[batch_index])
        for batch_index in range(len(stopping))
        for event, slots in {
            "knockout": 32,
            "round_of_16": 16,
            "quarter_final": 8,
            "semi_final": 4,
            "final": 2,
            "champion": 1,
        }.items()
    )
    terminal_counts = (
        batch_counts.loc[batch_counts["batch_index"].eq(len(stopping) - 1)]
        .pivot(index="team", columns="event", values="count")
        .sort_index()
    )
    terminal_team_counts = (
        teams.assign(team=teams["team"].astype(str)).set_index("team").sort_index()
    )
    terminal_matches_summary = all(
        np.array_equal(
            terminal_counts[event].to_numpy(float),
            pd.to_numeric(terminal_team_counts[f"{event}_count"], errors="raise").to_numpy(float),
        )
        for event in event_order
    )
    batch_counts_valid = bool(
        batch_basic_valid
        and batch_counts.groupby("batch_index").size().eq(48 * len(event_order)).all()
        and batch_counts.sort_values(["team", "event", "batch_index"])
        .groupby(["team", "event"])["count"]
        .diff()
        .dropna()
        .ge(0)
        .all()
        and batch_totals_valid
        and terminal_matches_summary
        and np.allclose(ledger_advancement, recomputed_advancement, atol=1e-15)
        and np.allclose(ledger_champion, recomputed_champion, atol=1e-15)
        and int(summary.get("batch_event_count_rows", -1)) == len(batch_counts)
        and int(metadata.get("batch_event_count_rows", -1)) == len(batch_counts)
    )
    ledger_valid = bool(
        np.array_equal(ledger_simulations, expected_batches)
        and np.array_equal(ledger_batches, np.arange(len(stopping), dtype=float))
        and np.isfinite(ledger_advancement).all()
        and np.isfinite(ledger_champion).all()
        and np.all(ledger_advancement >= 0)
        and np.all(ledger_champion >= 0)
        and np.array_equal(ledger_minimum, ledger_simulations >= config.minimum_simulations)
        and np.array_equal(
            ledger_advancement_satisfied,
            ledger_advancement <= config.advancement_mcse_threshold,
        )
        and np.array_equal(
            ledger_champion_satisfied,
            ledger_champion <= config.champion_mcse_threshold,
        )
        and np.array_equal(
            ledger_thresholds,
            (ledger_simulations >= config.minimum_simulations)
            & (ledger_advancement <= config.advancement_mcse_threshold)
            & (ledger_champion <= config.champion_mcse_threshold),
        )
        and np.array_equal(ledger_maximum, ledger_simulations == config.maximum_simulations)
        and not ledger_thresholds[:-1].any()
        and stopping["decision"].astype(str).iloc[:-1].eq("continue").all()
        and stopping["decision"].astype(str).iloc[-1] == summary.get("stopping_reason")
        and bool(ledger_thresholds[-1] or ledger_maximum[-1])
        and np.isclose(ledger_advancement[-1], maximum_advancement_mcse, atol=1e-15)
        and np.isclose(ledger_champion[-1], maximum_champion_mcse, atol=1e-15)
        and batch_counts_valid
        and int(summary.get("stopping_ledger_rows", -1)) == len(stopping)
        and int(metadata.get("stopping_ledger_rows", -1)) == len(stopping)
    )
    continuous = teams[
        [
            "expected_group_points",
            "expected_group_points_mcse",
            "expected_tournament_goals",
            "expected_tournament_goals_mcse",
        ]
    ].apply(pd.to_numeric, errors="raise")
    continuous_values = continuous.to_numpy(float)
    total_goal_mean = summary.get("mean_tournament_goals")
    total_goal_mcse = summary.get("mean_tournament_goals_mcse")
    sufficient_required = {
        "scope",
        "entity",
        "simulations",
        "group_points_sum",
        "group_points_square_sum",
        "goals_sum",
        "goals_square_sum",
    }
    if not sufficient_required.issubset(sufficient.columns):
        raise ValueError("Continuous sufficient-statistic schema is incomplete")
    sufficient_numeric = sufficient[
        [
            "simulations",
            "group_points_sum",
            "group_points_square_sum",
            "goals_sum",
            "goals_square_sum",
        ]
    ].apply(pd.to_numeric, errors="raise")
    sufficient = sufficient.assign(
        **{column: sufficient_numeric[column] for column in sufficient_numeric}
    )
    team_statistics = (
        sufficient.loc[sufficient["scope"].eq("team")]
        .assign(entity=lambda frame: frame["entity"].astype(str))
        .set_index("entity")
        .sort_index()
    )
    tournament_statistics = sufficient.loc[sufficient["scope"].eq("tournament")]
    ordered_team_continuous = (
        teams.assign(team=teams["team"].astype(str)).set_index("team").sort_index()
    )

    def mean_mcse(total: np.ndarray, square_total: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = total / simulations
        variance = np.maximum(
            (square_total - total**2 / simulations) / (simulations - 1),
            0.0,
        )
        return mean, np.sqrt(variance / simulations)

    point_mean, point_mcse = mean_mcse(
        team_statistics["group_points_sum"].to_numpy(float),
        team_statistics["group_points_square_sum"].to_numpy(float),
    )
    goal_mean, goal_mcse = mean_mcse(
        team_statistics["goals_sum"].to_numpy(float),
        team_statistics["goals_square_sum"].to_numpy(float),
    )
    tournament_goal_mean, tournament_goal_mcse = mean_mcse(
        tournament_statistics["goals_sum"].to_numpy(float),
        tournament_statistics["goals_square_sum"].to_numpy(float),
    )
    continuous_valid = bool(
        np.isfinite(continuous_values).all()
        and continuous["expected_group_points"].between(0, 9).all()
        and (continuous["expected_group_points_mcse"] >= 0).all()
        and (continuous["expected_tournament_goals"] >= 0).all()
        and (continuous["expected_tournament_goals_mcse"] >= 0).all()
        and isinstance(total_goal_mean, (int, float))
        and not isinstance(total_goal_mean, bool)
        and isinstance(total_goal_mcse, (int, float))
        and not isinstance(total_goal_mcse, bool)
        and np.isfinite(float(total_goal_mean))
        and np.isfinite(float(total_goal_mcse))
        and float(total_goal_mean) >= 0
        and float(total_goal_mcse) >= 0
        and np.isclose(
            continuous["expected_tournament_goals"].sum(),
            float(total_goal_mean),
            atol=1e-10,
        )
        and metadata.get("mean_tournament_goals") == total_goal_mean
        and metadata.get("mean_tournament_goals_mcse") == total_goal_mcse
        and len(sufficient) == 49
        and len(team_statistics) == 48
        and set(team_statistics.index) == team_names
        and len(tournament_statistics) == 1
        and tournament_statistics["entity"].astype(str).iloc[0] == "__tournament__"
        and np.isfinite(sufficient_numeric.to_numpy(float)).all()
        and np.equal(
            sufficient_numeric.to_numpy(float),
            np.floor(sufficient_numeric.to_numpy(float)),
        ).all()
        and (sufficient_numeric >= 0).all().all()
        and sufficient["simulations"].eq(simulations).all()
        and (
            team_statistics["group_points_square_sum"]
            >= team_statistics["group_points_sum"] ** 2 / simulations - 1e-9
        ).all()
        and (
            team_statistics["goals_square_sum"]
            >= team_statistics["goals_sum"] ** 2 / simulations - 1e-9
        ).all()
        and (
            tournament_statistics["goals_square_sum"]
            >= tournament_statistics["goals_sum"] ** 2 / simulations - 1e-9
        ).all()
        and np.allclose(
            point_mean,
            pd.to_numeric(
                ordered_team_continuous["expected_group_points"], errors="raise"
            ).to_numpy(float),
            atol=1e-12,
        )
        and np.allclose(
            point_mcse,
            pd.to_numeric(
                ordered_team_continuous["expected_group_points_mcse"], errors="raise"
            ).to_numpy(float),
            atol=1e-12,
        )
        and np.allclose(
            goal_mean,
            pd.to_numeric(
                ordered_team_continuous["expected_tournament_goals"], errors="raise"
            ).to_numpy(float),
            atol=1e-12,
        )
        and np.allclose(
            goal_mcse,
            pd.to_numeric(
                ordered_team_continuous["expected_tournament_goals_mcse"], errors="raise"
            ).to_numpy(float),
            atol=1e-12,
        )
        and np.isclose(tournament_goal_mean[0], float(total_goal_mean), atol=1e-12)
        and np.isclose(tournament_goal_mcse[0], float(total_goal_mcse), atol=1e-12)
        and np.isclose(goal_mean.sum(), tournament_goal_mean[0], atol=1e-12)
        and metadata.get("continuous_sufficient_statistic_rows") == 49
        and summary.get("continuous_sufficient_statistic_rows") == 49
    )
    required_log_columns = {
        "simulation_id",
        "match_number",
        "stage",
        "home_team",
        "away_team",
        "winner",
        "home_source_slot",
        "away_source_slot",
    }
    if not required_log_columns.issubset(match_log.columns):
        raise ValueError("Simulation audit log schema is incomplete")
    match_numbers = pd.to_numeric(match_log["match_number"], errors="raise")
    per_simulation_stage = pd.crosstab(
        pd.Series(numeric_ids.astype(int), index=match_log.index),
        match_log["stage"].astype(str),
    ).reindex(columns=list(expected_stage), fill_value=0)
    exact_per_simulation_stage = bool(
        len(per_simulation_stage) == audit_simulations
        and all(
            per_simulation_stage[stage_name].eq(stage_count).all()
            for stage_name, stage_count in expected_stage.items()
        )
    )
    knockout = match_log["stage"].ne("group_stage")
    knockout_winner_valid = bool(
        match_log.loc[knockout, "winner"].notna().all()
        and (
            match_log.loc[knockout, "winner"]
            .astype(str)
            .eq(match_log.loc[knockout, "home_team"].astype(str))
            | match_log.loc[knockout, "winner"]
            .astype(str)
            .eq(match_log.loc[knockout, "away_team"].astype(str))
        ).all()
    )
    audit_match_structure = bool(
        len(per_simulation) == audit_simulations
        and set(per_simulation.index) == set(range(audit_simulations))
        and (per_simulation == 104).all()
        and np.isfinite(match_numbers.to_numpy(float)).all()
        and np.equal(match_numbers.to_numpy(float), np.floor(match_numbers.to_numpy(float))).all()
        and match_log.assign(_match_number=match_numbers)
        .groupby("simulation_id")["_match_number"]
        .apply(lambda values: set(values.astype(int)) == set(range(1, 105)))
        .all()
        and exact_per_simulation_stage
        and knockout_winner_valid
        and match_log.loc[match_log["stage"].eq("final"), "winner"].notna().all()
    )
    scenario_boundary = bool(
        summary.get("scenario_only_not_factual_forecast") is True
        and metadata.get("factual_2026_entrants_used") is False
        and metadata.get("audit_outcomes_used") is False
    )
    champion_probability = pd.to_numeric(teams["champion_probability"], errors="raise").to_numpy(
        float
    )
    champion_count = pd.to_numeric(teams["champion_count"], errors="raise").to_numpy(float)
    champion_sum = float(champion_probability.sum())
    checks = {
        "team_count_48": len(teams) == 48
        and not teams["team"].isna().any()
        and not teams["team"].duplicated().any()
        and int(metadata.get("team_count", -1)) == 48,
        "format_12_groups_of_4": metadata.get("group_count") == 12
        and metadata.get("group_size") == 4
        and metadata.get("automatic_qualifiers_per_group") == 2
        and metadata.get("best_third_qualifiers") == 8,
        "annex_c_495_certified": metadata.get("annex_c_rows") == 495
        and metadata.get("format_certification")
        == "official_regulations_M73_M104_graph_and_complete_annexe_c_mapping"
        and metadata.get("format_source_sha256") == FIFA_2026_REGULATIONS_SHA256
        and metadata.get("annex_c_csv_sha256") == FIFA_2026_ANNEX_C_CSV_SHA256,
        "frozen_adaptive_contract_exact": contract_locked,
        "simulations_within_predeclared_bounds": (
            config.minimum_simulations <= simulations <= config.maximum_simulations
            and simulations % config.batch_size == 0
        ),
        "audit_matches_104_each": audit_match_structure,
        "audit_stage_cardinality": audit_stage == expected_audit_stage,
        "aggregate_stage_cardinality": exact_stage
        and dict(zip(stage["stage"], stage["matches_per_simulation"], strict=True))
        == expected_stage
        and summary.get("simulated_match_rows") == simulations * 104,
        "no_self_matches_and_known_teams": bool(
            not match_log["home_team"].eq(match_log["away_team"]).any()
            and set(match_log["home_team"].astype(str)).issubset(team_names)
            and set(match_log["away_team"].astype(str)).issubset(team_names)
            and match_log["home_source_slot"].notna().all()
            and match_log["away_source_slot"].notna().all()
        ),
        "advancement_counts_integer_and_conserved": counts_valid and counts_conserved,
        "champion_simplex": bool(
            np.allclose(champion_probability, champion_count / simulations, atol=1e-12)
            and np.isclose(champion_sum, 1.0, atol=1e-12)
        ),
        "advancement_probability_monotonicity": monotone,
        "probability_mcse_intervals_and_continuous_valid": max(mcse_errors) <= 1e-12
        and max(interval_errors) <= 1e-12
        and continuous_valid,
        "adaptive_stopping_and_scenario_boundary": valid_adaptive_terminal
        and ledger_valid
        and scenario_boundary,
    }
    return {
        **checks,
        "scenario_not_factual": scenario_boundary,
        "adaptive_stopping_contract_satisfied": valid_adaptive_terminal and ledger_valid,
        "check_count": len(checks),
        "champion_probability_sum": champion_sum,
        "maximum_recomputed_mcse_error": max(mcse_errors),
        "maximum_recomputed_interval_error": max(interval_errors),
        "passed": len(checks) == 14 and all(checks.values()),
    }


def _strict_bool_values(series: pd.Series) -> np.ndarray:
    """Reject truthy strings and integers in CSV boolean evidence."""
    values = series.to_numpy()
    if not all(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError("Adaptive stopping flags must be exact booleans")
    return values.astype(bool)


def _uncertainty_checks(root: Path) -> dict[str, Any]:
    parameter = _json(root / "uncertainty" / "parameter_metadata.json")
    ledger = _csv(root / "uncertainty" / "parameter_draw_ledger.csv")
    draws = _csv(root / "uncertainty" / "parameter_probability_draws.csv")
    uncertainty_dir = root / "uncertainty"
    decomposition_binding = verify_stage_manifest(
        uncertainty_dir / "decomposition_artifact_manifest.json",
        required_paths=(
            "variance_decomposition.json",
            "variance_draw_cube.joblib",
            "variance_cell_summary.csv",
        ),
    )
    decomposition = json.loads(
        read_verified_artifact_bytes(
            uncertainty_dir,
            "variance_decomposition.json",
            decomposition_binding,
        ).decode("utf-8")
    )
    draw_cube = np.asarray(
        joblib.load(
            io.BytesIO(
                read_verified_artifact_bytes(
                    uncertainty_dir,
                    "variance_draw_cube.joblib",
                    decomposition_binding,
                )
            )
        ),
        dtype=float,
    )
    cell_summary = pd.read_csv(
        io.BytesIO(
            read_verified_artifact_bytes(
                uncertainty_dir,
                "variance_cell_summary.csv",
                decomposition_binding,
            )
        )
    )
    if draw_cube.shape != (20, 3, 100) or not np.isfinite(draw_cube).all():
        raise ValueError("Variance draw cube must be finite with shape (20, 3, 100)")
    recomputed_decomposition = hierarchical_variance_decomposition(draw_cube)
    metadata = decomposition.get("metadata", {})
    simulation = _json(root / "simulation" / "scenario_metadata.json")
    simulation_manifest = root / "simulation" / "artifact_manifest.json"
    components = decomposition.get("components", {})
    shares = decomposition.get("shares", {})
    expected_shares = {
        "parameter",
        "regime_given_parameter",
        "tournament_path_and_match",
    }
    expected_components = expected_shares | {"total", "reconstruction_error"}
    component_values = (
        {name: float(value) for name, value in components.items()}
        if isinstance(components, dict)
        else {}
    )
    share_values = (
        {name: float(value) for name, value in shares.items()} if isinstance(shares, dict) else {}
    )
    finite_components = set(component_values) == expected_components and bool(
        np.isfinite(list(component_values.values())).all()
    )
    finite_shares = set(share_values) == expected_shares and bool(
        np.isfinite(list(share_values.values())).all()
    )
    decomposition_matches_cube = bool(
        finite_components
        and finite_shares
        and all(
            np.isclose(
                component_values[name],
                recomputed_decomposition.components[name],
                rtol=0,
                atol=1e-12,
            )
            for name in expected_components
        )
        and all(
            np.isclose(
                share_values[name],
                recomputed_decomposition.shares[name],
                rtol=0,
                atol=1e-12,
            )
            for name in expected_shares
        )
        and np.isclose(
            float(metadata.get("functional_mean", np.nan)),
            recomputed_decomposition.metadata["functional_mean"],
            rtol=0,
            atol=1e-12,
        )
    )
    cell_required = {
        "parameter_draw_index",
        "bootstrap_draw_id",
        "sample_hash",
        "regime_index",
        "log_total_goal_tilt",
        "path_draws",
        "mean_total_goals",
        "sd_total_goals",
    }
    cell_summary_valid = False
    cell_provenance_valid = False
    draw_ids = pd.to_numeric(ledger.get("bootstrap_draw_id"), errors="coerce")
    ledger_has_provenance = "sample_hash" in ledger and len(ledger) == 200
    if cell_required.issubset(cell_summary.columns) and len(cell_summary) == 60:
        cells = cell_summary.copy()
        cell_numeric = cells[
            [
                "parameter_draw_index",
                "bootstrap_draw_id",
                "regime_index",
                "log_total_goal_tilt",
                "path_draws",
                "mean_total_goals",
                "sd_total_goals",
            ]
        ].apply(pd.to_numeric, errors="coerce")
        parameter_index = cell_numeric["parameter_draw_index"].to_numpy(float)
        bootstrap_draw_id = cell_numeric["bootstrap_draw_id"].to_numpy(float)
        regime_index = cell_numeric["regime_index"].to_numpy(float)
        path_draws = cell_numeric["path_draws"].to_numpy(float)
        integer_indices = bool(
            np.isfinite(cell_numeric.to_numpy(float)).all()
            and np.equal(parameter_index, np.floor(parameter_index)).all()
            and np.equal(bootstrap_draw_id, np.floor(bootstrap_draw_id)).all()
            and np.equal(regime_index, np.floor(regime_index)).all()
            and np.equal(path_draws, np.floor(path_draws)).all()
        )
        if integer_indices:
            cells = cells.assign(
                parameter_draw_index=parameter_index.astype(int),
                bootstrap_draw_id=bootstrap_draw_id.astype(int),
                regime_index=regime_index.astype(int),
            ).sort_values(["parameter_draw_index", "regime_index"])
            expected_pairs = pd.MultiIndex.from_product(
                [range(20), range(3)],
                names=["parameter_draw_index", "regime_index"],
            )
            observed_pairs = pd.MultiIndex.from_frame(
                cells[["parameter_draw_index", "regime_index"]]
            )
            expected_mean = draw_cube.mean(axis=2).reshape(-1)
            expected_sd = draw_cube.std(axis=2, ddof=1).reshape(-1)
            sorted_path_draws = pd.to_numeric(cells["path_draws"], errors="raise").to_numpy(
                float
            )
            cell_summary_valid = bool(
                observed_pairs.equals(expected_pairs)
                and np.equal(sorted_path_draws, 100).all()
                and np.allclose(
                    pd.to_numeric(cells["log_total_goal_tilt"], errors="raise"),
                    np.tile(np.asarray((-0.10, 0.0, 0.10)), 20),
                    rtol=0,
                    atol=1e-15,
                )
                and np.allclose(
                    pd.to_numeric(cells["mean_total_goals"], errors="raise"),
                    expected_mean,
                    rtol=0,
                    atol=1e-12,
                )
                and np.allclose(
                    pd.to_numeric(cells["sd_total_goals"], errors="raise"),
                    expected_sd,
                    rtol=0,
                    atol=1e-12,
                )
                and cells["sample_hash"].notna().all()
                and cells["sample_hash"].astype(str).str.strip().ne("").all()
            )
            if ledger_has_provenance and not draw_ids.isna().any():
                ordered_ledger = ledger.assign(
                    _bootstrap_draw_id=draw_ids.astype(int),
                    _sample_hash=ledger["sample_hash"].astype(str),
                ).sort_values("_bootstrap_draw_id")
                first_twenty = ordered_ledger.head(20)
                cell_provenance_valid = bool(
                    np.array_equal(
                        cells["bootstrap_draw_id"].to_numpy(int),
                        np.repeat(first_twenty["_bootstrap_draw_id"].to_numpy(int), 3),
                    )
                    and np.array_equal(
                        cells["sample_hash"].astype(str).to_numpy(),
                        np.repeat(first_twenty["_sample_hash"].to_numpy(), 3),
                    )
                )
    probability = (
        draws[["home_win", "draw", "away_win"]].apply(pd.to_numeric, errors="coerce")
        if {"home_win", "draw", "away_win"}.issubset(draws.columns)
        else pd.DataFrame()
    )
    checks = {
        "parameter_draws_complete": len(ledger) == 200
        and not draw_ids.isna().any()
        and np.equal(draw_ids.to_numpy(float), np.floor(draw_ids.to_numpy(float))).all()
        and set(draw_ids.astype(int)) == set(range(200))
        and not draw_ids.duplicated().any()
        and ledger_has_provenance
        and ledger["sample_hash"].notna().all()
        and ledger["sample_hash"].astype(str).str.strip().ne("").all()
        and int(parameter.get("replicates_succeeded", -1)) == 200
        and int(parameter.get("replicates_failed", -1)) == 0,
        "probability_draws_complete": len(draws) == 1_600
        and not probability.empty
        and bool(np.isfinite(probability.to_numpy(float)).all())
        and bool((probability >= 0).all().all())
        and bool(np.allclose(probability.sum(axis=1), 1.0, atol=1e-10)),
        "preaudit_only": parameter.get("post_2010_rows_used") is False
        and parameter.get("audit_outcomes_used") is False
        and metadata.get("audit_outcomes_used") is False,
        "decomposition_shape": metadata.get("draw_shape") == [20, 3, 100],
        "decomposition_recomputed_from_manifest_bound_cube": decomposition_matches_cube,
        "variance_cell_summary_recomputed_from_cube": cell_summary_valid,
        "variance_cell_provenance_matches_parameter_ledger": cell_provenance_valid,
        "variance_nonnegative": bool(
            finite_components
            and all(component_values[name] >= 0 for name in expected_shares | {"total"})
        ),
        "variance_reconstructs": bool(
            finite_components
            and abs(component_values["reconstruction_error"]) <= 1e-9
            and np.isclose(
                component_values["total"],
                sum(component_values[name] for name in expected_shares),
                atol=1e-9,
            )
        ),
        "shares_sum_to_one": bool(
            finite_shares
            and all(0 <= share_values[name] <= 1 for name in expected_shares)
            and np.isclose(sum(share_values.values()), 1.0, atol=1e-12)
        ),
        "replacement_simulation_lineage": (
            metadata.get("adaptive_contract_sha256") == adaptive_contract_sha256()
            and metadata.get("protocol_deviation_sha256") == FROZEN_PROTOCOL_DEVIATION_SHA256
            and metadata.get("format_source_sha256") == FIFA_2026_REGULATIONS_SHA256
            and metadata.get("annex_c_csv_sha256") == FIFA_2026_ANNEX_C_CSV_SHA256
            and metadata.get("replacement_simulation_manifest_sha256")
            == _file_sha256(simulation_manifest)
            and simulation.get("adaptive_contract_sha256")
            == metadata.get("adaptive_contract_sha256")
        ),
    }
    return {**checks, "passed": all(checks.values())}


def _provenance_checks(
    root: Path,
    identity: dict[str, Any],
    nested: dict[str, Any],
    stress: pd.DataFrame,
) -> dict[str, Any]:
    evaluation = _json(root / "evaluation" / "evaluation_lineage.json")
    stress_lineage = _json(root / "stress" / "stress_lineage.json")
    ablation = _json(root / "ablation" / "ablation_lineage.json")
    parameter = _json(root / "uncertainty" / "parameter_metadata.json")
    simulation = _json(root / "simulation" / "scenario_metadata.json")
    decomposition = _json(root / "uncertainty" / "variance_decomposition.json")["metadata"]
    sarmanov = _json(root / "evaluation" / "sarmanov_metadata.json")
    sarmanov_selection = _csv(root / "evaluation" / "sarmanov_selection.csv")
    selected = _csv(root / "nested" / "selected_specs.csv")
    same_selection = all(
        item.get("selection_procedure_hash") == identity.get("selection_procedure_hash")
        for item in (evaluation, stress_lineage, ablation, parameter, simulation, decomposition)
    )
    final_fit_items = (stress_lineage, parameter, simulation, decomposition)
    same_fit = all(
        item.get("final_fit_hash") == identity.get("final_fit_hash") for item in final_fit_items
    )
    same_protocol = all(
        item.get("protocol_hash") == identity.get("protocol_hash")
        for item in (evaluation, stress_lineage, ablation, parameter, simulation)
    )
    checks = {
        "candidate_budget_36": int(nested.get("candidate_count", -1)) == 36,
        "outer_years_exact": nested.get("outer_test_years") == list(EXPECTED_YEARS),
        "selected_runtime_eligible": len(selected) == 7 and _strict_true(selected["eligible"]),
        "same_selection_procedure": same_selection,
        "same_final_fit": same_fit,
        "same_protocol": same_protocol,
        "development_boundary_2010": int(identity.get("fit_end_year", -1)) == 2010
        and int(nested.get("maximum_selection_year", -1)) == 2010,
        "audit_not_used": identity.get("audit_outcomes_used_for_selection") is False
        and all(
            item.get("audit_outcomes_used", item.get("audit_year_loaded")) is False
            for item in (evaluation, stress_lineage, ablation, parameter, simulation, decomposition)
        ),
        "snapshot_isolated": identity.get("formal_preaudit_reads_combined_raw") is False
        and isinstance(identity.get("development_snapshot_manifest_sha256"), str)
        and len(identity["development_snapshot_manifest_sha256"]) == 64,
        "sarmanov_exploratory_boundary": sarmanov.get("role")
        == "exploratory_contradiction_benchmark_not_promotion_eligible"
        and str(sarmanov.get("confirmatory_status", "")).startswith("exploratory_only")
        and sarmanov.get("audit_year_used") is False
        and int(sarmanov.get("outer_fold_count", -1)) == 7
        and len(sarmanov_selection) == 7
        and bool(
            (
                pd.to_numeric(sarmanov_selection["inner_max_prediction_year"], errors="raise")
                < pd.to_numeric(sarmanov_selection["outer_prediction_year"], errors="raise")
            ).all()
        )
        and int(pd.to_numeric(sarmanov_selection["outer_prediction_year"], errors="raise").max())
        <= 2010,
        "stress_reproducible": float(
            pd.to_numeric(stress["reproducibility_error"], errors="raise").max()
        )
        <= 1e-12,
        "stress_baseline_provenance": float(
            pd.to_numeric(stress["baseline_provenance_error"], errors="raise").max()
        )
        <= 1e-12,
    }
    return {
        **checks,
        "final_fit_hash": identity.get("final_fit_hash"),
        "selection_procedure_hash": identity.get("selection_procedure_hash"),
        "passed": all(checks.values()),
    }


def _strict_true(series: pd.Series) -> bool:
    return bool(not series.empty and all(_truthy(value) for value in series.tolist()))


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.casefold() in {"true", "1"}
    return isinstance(value, (int, np.integer)) and int(value) == 1


def _csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["EXPECTED_YEARS", "evaluate_verified_evidence"]
