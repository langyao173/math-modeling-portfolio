from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

import worldcup_forecasting.experiments.audit_execution as audit_execution_module
import worldcup_forecasting.validation._promotion_artifacts as promotion_artifacts
import worldcup_forecasting.validation._promotion_checks as promotion_checks
from worldcup_forecasting.data import build_canonical_data
from worldcup_forecasting.experiments._nested_selection_runtime import (
    REFERENCE_RUNTIME_HASH,
    reference_runtime_seconds,
)
from worldcup_forecasting.experiments.audit_baselines import (
    fixed_audit_baseline_identities,
    validate_fixed_audit_baseline_models,
)
from worldcup_forecasting.experiments.audit_execution import (
    FrozenAuditModels,
    evaluate_persisted_blind_audit,
    freeze_audit_models,
    load_batch_blind_2014_features,
    load_isolated_2014_truth,
    load_persisted_blind_audit_predictions,
    persist_audit_result,
    persist_blind_audit_predictions,
    prepare_frozen_audit_predictions,
)
from worldcup_forecasting.models.count import JointPoissonConfig, JointPoissonModel
from worldcup_forecasting.models.outcome import OutcomeLogit
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
from worldcup_forecasting.validation._promotion_artifacts import checkpoint_status
from worldcup_forecasting.validation._promotion_checks import (
    _runtime_reproducibility_checks,
    _simulation_checks,
    _uncertainty_checks,
)
from worldcup_forecasting.validation.artifact_integrity import (
    ArtifactIntegrityError,
    build_stage_manifest,
    integrity_envelope,
    read_verified_artifact_bytes,
    verify_integrity_envelope,
    verify_stage_manifest,
    write_json_atomic,
)
from worldcup_forecasting.validation.audit_orchestration import (
    FrozenAuditPreparation,
    PersistedAudit,
    initialize_unopened_sentinel,
    run_one_time_audit,
    verify_cached_audit,
)
from worldcup_forecasting.validation.promotion import (
    PromotionRejectedError,
    evaluate_calibration_retention,
    issue_promotion_certificate,
    normalize_decision_metrics,
)
from worldcup_forecasting.validation.runtime_reproducibility import (
    ABLATION_REFERENCE_RUNTIME_SECONDS,
    BASELINE_REFERENCE_RUNTIME_SECONDS,
    RUNTIME_REFERENCE_PROTOCOL,
    RUNTIME_REFERENCE_PROTOCOL_SHA256,
)
from worldcup_forecasting.validation.stage_orchestration import run_preaudit_pipeline


def test_fixed_audit_baseline_identities_are_json_canonical() -> None:
    """The recomputed identity must equal its exact persisted JSON representation."""
    identities = fixed_audit_baseline_identities(
        development_frame_hash="d" * 64,
        feature_schema_hash="f" * 64,
    )

    assert identities == json.loads(json.dumps(identities, sort_keys=True))


def test_protocol_deviation_hash_is_lf_stable_and_triply_locked(
    project_root: Path,
) -> None:
    protocol = project_root / "plan" / "PROTOCOL_DEVIATIONS.md"
    attributes = (project_root / ".gitattributes").read_text(encoding="utf-8")
    protocol_bytes = protocol.read_bytes()
    lock = json.loads(
        (project_root / "plan" / "preaudit_amendment_lock.json").read_text(
            encoding="utf-8"
        )
    )

    assert "plan/PROTOCOL_DEVIATIONS.md text eol=lf" in attributes.splitlines()
    assert b"\r" not in protocol_bytes
    assert (
        hashlib.sha256(protocol_bytes).hexdigest()
        == lock["deviation_record_sha256"]
        == FROZEN_PROTOCOL_DEVIATION_SHA256
    )


def test_integrity_envelope_is_independently_hash_verifiable_and_optionally_authenticated() -> None:
    payload = {"decision": "promote", "hash": "a" * 64}
    unsigned = integrity_envelope(payload)
    assert verify_integrity_envelope(unsigned) == payload
    with pytest.raises(ArtifactIntegrityError, match="unsigned"):
        verify_integrity_envelope(unsigned, authentication_key=b"external-test-key-material")
    unsigned["payload"]["decision"] = "reject"
    with pytest.raises(ArtifactIntegrityError, match="digest"):
        verify_integrity_envelope(unsigned)

    key = b"external-test-key-material"
    signed = integrity_envelope(payload, authentication_key=key, key_id="test")
    assert verify_integrity_envelope(signed, authentication_key=key) == payload
    with pytest.raises(ArtifactIntegrityError, match="key|required|HMAC"):
        verify_integrity_envelope(signed)
    with pytest.raises(ArtifactIntegrityError, match="HMAC"):
        verify_integrity_envelope(signed, authentication_key=b"wrong-key-material")


def test_stage_manifest_rejects_tampering_and_unsafe_paths(tmp_path: Path) -> None:
    directory = tmp_path / "stage"
    directory.mkdir()
    artifact = directory / "result.txt"
    artifact.write_text("frozen", encoding="utf-8")
    manifest = write_json_atomic(
        build_stage_manifest(directory, ["result.txt"]),
        directory / "artifact_manifest.json",
    )
    assert verify_stage_manifest(manifest, required_paths=["result.txt"])["artifact_count"] == 1
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="mismatch"):
        verify_stage_manifest(manifest)
    with pytest.raises(ArtifactIntegrityError, match="Unsafe"):
        build_stage_manifest(directory, ["../escape.txt"])


def test_verified_artifact_reader_rejects_verify_then_swap(tmp_path: Path) -> None:
    directory = tmp_path / "stage"
    directory.mkdir()
    artifact = directory / "model.joblib"
    artifact.write_bytes(b"verified-model-bytes")
    manifest = write_json_atomic(
        build_stage_manifest(directory, ["model.joblib"]),
        directory / "artifact_manifest.json",
    )
    binding = verify_stage_manifest(manifest, required_paths=["model.joblib"])
    artifact.write_bytes(b"attacker-swapped-bytes")
    with pytest.raises(ArtifactIntegrityError, match="changed between"):
        read_verified_artifact_bytes(directory, "model.joblib", binding)


def test_stage_manifest_rejects_symlinked_manifest(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    sibling = tmp_path / "sibling"
    stage.mkdir()
    sibling.mkdir()
    (sibling / "result.txt").write_text("foreign", encoding="utf-8")
    foreign_manifest = write_json_atomic(
        build_stage_manifest(sibling, ["result.txt"]),
        sibling / "artifact_manifest.json",
    )
    linked_manifest = stage / "artifact_manifest.json"
    try:
        linked_manifest.symlink_to(foreign_manifest)
    except OSError as error:
        pytest.skip(f"Symlink creation is unavailable on this platform: {error}")
    with pytest.raises(ArtifactIntegrityError, match="Reparse-point"):
        verify_stage_manifest(linked_manifest)


def test_checkpoint_rejects_an_unmanifested_stale_stage_artifact(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "nested"
    directory.mkdir()
    required = (
        "nested_metadata.json",
        "runtime_reproducibility.json",
        "frozen_model_identity.json",
        "candidate_metrics.csv",
        "final_candidate_metrics.csv",
        "final_selected_summary.csv",
        "selected_specs.csv",
        "outer_joint_grid.csv",
        "final_inner_oof_ledger.csv",
        "final_inner_joint_grid.csv",
        "final_fitted_model.joblib",
        "nested_result.joblib",
        "final_selection.joblib",
        "development_features.joblib",
    )
    for name in required:
        (directory / name).write_bytes(b"sealed")
    write_json_atomic(
        build_stage_manifest(directory, required),
        directory / "artifact_manifest.json",
    )
    write_json_atomic(
        {
            "status": "nested_stage_complete_and_locked",
            "formal_preaudit_reads_combined_raw": False,
        },
        tmp_path / "nested_stage_summary.json",
    )
    assert checkpoint_status(tmp_path, "nested")["status"] == "complete"
    (directory / "stale_experiment.txt").write_text("undeclared", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="exact inventory|undeclared"):
        checkpoint_status(tmp_path, "nested")


def test_uncertainty_gate_returns_strict_json_native_booleans(tmp_path: Path) -> None:
    directory = tmp_path / "uncertainty"
    directory.mkdir()
    write_json_atomic(
        {
            "replicates_succeeded": 200,
            "replicates_failed": 0,
            "post_2010_rows_used": False,
            "audit_outcomes_used": False,
        },
        directory / "parameter_metadata.json",
    )
    pd.DataFrame(
        {
            "bootstrap_draw_id": range(200),
            "sample_hash": [f"sample-{index}" for index in range(200)],
        }
    ).to_csv(
        directory / "parameter_draw_ledger.csv", index=False
    )
    pd.DataFrame(
        {
            "home_win": np.full(1_600, 0.4),
            "draw": np.full(1_600, 0.3),
            "away_win": np.full(1_600, 0.3),
        }
    ).to_csv(directory / "parameter_probability_draws.csv", index=False)
    simulation_dir = tmp_path / "simulation"
    simulation_dir.mkdir()
    write_json_atomic({}, simulation_dir / "artifact_manifest.json")
    simulation_manifest_sha256 = hashlib.sha256(
        (simulation_dir / "artifact_manifest.json").read_bytes()
    ).hexdigest()
    write_json_atomic(
        {"adaptive_contract_sha256": adaptive_contract_sha256()},
        simulation_dir / "scenario_metadata.json",
    )
    decomposition_lineage = {
        "adaptive_contract_sha256": adaptive_contract_sha256(),
        "protocol_deviation_sha256": FROZEN_PROTOCOL_DEVIATION_SHA256,
        "format_source_sha256": FIFA_2026_REGULATIONS_SHA256,
        "annex_c_csv_sha256": FIFA_2026_ANNEX_C_CSV_SHA256,
        "replacement_simulation_manifest_sha256": simulation_manifest_sha256,
    }
    rng = np.random.default_rng(1729)
    draw_cube = rng.poisson(lam=240.0, size=(20, 3, 100)).astype(float)
    recomputed = hierarchical_variance_decomposition(draw_cube)
    joblib.dump(draw_cube, directory / "variance_draw_cube.joblib", compress=3)
    cell_rows = []
    for parameter_index in range(20):
        for regime_index, tilt in enumerate((-0.10, 0.0, 0.10)):
            values = draw_cube[parameter_index, regime_index]
            cell_rows.append(
                {
                    "parameter_draw_index": parameter_index,
                    "bootstrap_draw_id": parameter_index,
                    "sample_hash": f"sample-{parameter_index}",
                    "regime_index": regime_index,
                    "log_total_goal_tilt": tilt,
                    "path_draws": 100,
                    "mean_total_goals": float(values.mean()),
                    "sd_total_goals": float(values.std(ddof=1)),
                }
            )
    pd.DataFrame(cell_rows).to_csv(directory / "variance_cell_summary.csv", index=False)
    payload = {
        "components": recomputed.components,
        "shares": recomputed.shares,
        "metadata": {
            **recomputed.metadata,
            "audit_outcomes_used": False,
            **decomposition_lineage,
        },
    }
    write_json_atomic(payload, directory / "variance_decomposition.json")
    decomposition_paths = (
        "variance_decomposition.json",
        "variance_draw_cube.joblib",
        "variance_cell_summary.csv",
    )
    write_json_atomic(
        build_stage_manifest(directory, decomposition_paths),
        directory / "decomposition_artifact_manifest.json",
    )
    result = _uncertainty_checks(tmp_path)
    assert result["passed"] is True
    assert type(result["shares_sum_to_one"]) is bool

    forged_cells = pd.DataFrame(cell_rows).assign(
        bootstrap_draw_id=999,
        sample_hash="forged-but-nonempty",
    )
    forged_cells.to_csv(directory / "variance_cell_summary.csv", index=False)
    write_json_atomic(
        build_stage_manifest(directory, decomposition_paths),
        directory / "decomposition_artifact_manifest.json",
    )
    forged = _uncertainty_checks(tmp_path)
    assert forged["variance_cell_provenance_matches_parameter_ledger"] is False
    assert forged["passed"] is False

    pd.DataFrame(cell_rows).to_csv(directory / "variance_cell_summary.csv", index=False)
    write_json_atomic(
        build_stage_manifest(directory, decomposition_paths),
        directory / "decomposition_artifact_manifest.json",
    )

    malformed = json.loads(json.dumps(payload))
    malformed["components"] = {
        "parameter": 1.0,
        "regime_given_parameter": 1.0,
        "tournament_path_and_match": 2.0,
        "total": 4.0,
        "reconstruction_error": 0.0,
    }
    malformed["shares"] = {
        "parameter": 0.25,
        "regime_given_parameter": 0.25,
        "tournament_path_and_match": 0.5,
    }
    write_json_atomic(malformed, directory / "variance_decomposition.json")
    write_json_atomic(
        build_stage_manifest(directory, decomposition_paths),
        directory / "decomposition_artifact_manifest.json",
    )
    assert _uncertainty_checks(tmp_path)["passed"] is False


def test_runtime_reference_gate_rejects_reintroduced_wall_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_id = "f12f0fc20700df70743ef3b4f27089c45f45989a93fa2bd19c1886c98d609a81"
    reference = reference_runtime_seconds(candidate_id)
    declaration = {
        **RUNTIME_REFERENCE_PROTOCOL,
        "runtime_reference_protocol_sha256": RUNTIME_REFERENCE_PROTOCOL_SHA256,
        "candidate_runtime_reference_hash": REFERENCE_RUNTIME_HASH,
        "final_fit_runtime_reference_seconds": reference,
    }
    runtime_metadata = {
        "runtime_semantics": RUNTIME_REFERENCE_PROTOCOL["semantics"],
        "runtime_reference_protocol_sha256": RUNTIME_REFERENCE_PROTOCOL_SHA256,
    }
    candidate = pd.DataFrame(
        {
            "candidate_id": [candidate_id],
            "runtime_seconds": [reference],
            "failure_reason": [""],
        }
    )
    selected = candidate.copy()
    final_candidate = candidate.copy()
    final_selected = candidate.copy()
    candidate_csv = candidate.copy()
    selected_csv = selected.copy()
    final_candidate_csv = final_candidate.copy()
    final_selected_csv = final_selected.copy()
    for frame in (candidate_csv, selected_csv, final_candidate_csv, final_selected_csv):
        frame.loc[:, "failure_reason"] = np.nan
    baseline = pd.DataFrame(
        {
            "model": list(BASELINE_REFERENCE_RUNTIME_SECONDS),
            "runtime_seconds": list(BASELINE_REFERENCE_RUNTIME_SECONDS.values()),
        }
    )
    ablation = pd.DataFrame(
        {
            "variant_id": list(ABLATION_REFERENCE_RUNTIME_SECONDS),
            "runtime_seconds": list(ABLATION_REFERENCE_RUNTIME_SECONDS.values()),
        }
    )
    json_by_name = {
        "runtime_reproducibility.json": declaration,
        "nested_metadata.json": {"runtime_reference_hash": REFERENCE_RUNTIME_HASH},
        "baseline_metadata.json": runtime_metadata,
        "ablation_metadata.json": runtime_metadata,
    }
    csv_by_name = {
        "candidate_metrics.csv": candidate_csv,
        "selected_specs.csv": selected_csv,
        "final_candidate_metrics.csv": final_candidate_csv,
        "final_selected_summary.csv": final_selected_csv,
        "baseline_outcome_fold_metrics.csv": baseline,
        "baseline_joint_fold_metrics.csv": baseline,
        "ablation_matrix.csv": ablation,
    }
    monkeypatch.setattr(promotion_checks, "_json", lambda path: json_by_name[path.name])
    monkeypatch.setattr(promotion_checks, "_csv", lambda path: csv_by_name[path.name])
    nested_dir = tmp_path / "nested"
    evaluation_dir = tmp_path / "evaluation"
    nested_dir.mkdir()
    evaluation_dir.mkdir()
    joblib.dump(
        SimpleNamespace(selected_specs=selected, candidate_metrics=candidate),
        nested_dir / "nested_result.joblib",
        compress=3,
    )
    joblib.dump(
        SimpleNamespace(
            selected_summary=final_selected,
            candidate_metrics=final_candidate,
        ),
        nested_dir / "final_selection.joblib",
        compress=3,
    )
    joblib.dump(
        SimpleNamespace(
            fit_runtime_seconds=reference,
            selected_spec=SimpleNamespace(candidate_id=candidate_id),
            metadata={"fit_runtime_seconds": reference},
        ),
        nested_dir / "final_fitted_model.joblib",
        compress=3,
    )
    joblib.dump(
        SimpleNamespace(
            outcome_fold_metrics=baseline,
            joint_fold_metrics=baseline,
            metadata={
                "runtime_reference_protocol_sha256": RUNTIME_REFERENCE_PROTOCOL_SHA256
            },
        ),
        evaluation_dir / "baseline_result.joblib",
        compress=3,
    )
    write_json_atomic(
        build_stage_manifest(
            nested_dir,
            (
                "nested_result.joblib",
                "final_selection.joblib",
                "final_fitted_model.joblib",
            ),
        ),
        nested_dir / "artifact_manifest.json",
    )
    write_json_atomic(
        build_stage_manifest(evaluation_dir, ("baseline_result.joblib",)),
        evaluation_dir / "artifact_manifest.json",
    )

    assert _runtime_reproducibility_checks(tmp_path)["passed"] is True
    final_selected_csv["measured_runtime_seconds"] = 987.654
    assert _runtime_reproducibility_checks(tmp_path)["passed"] is False
    final_selected_csv.drop(columns="measured_runtime_seconds", inplace=True)
    ablation.loc[ablation["variant_id"].eq("A7"), "runtime_seconds"] += 0.1
    assert _runtime_reproducibility_checks(tmp_path)["passed"] is False
    ablation.loc[ablation["variant_id"].eq("A7"), "runtime_seconds"] -= 0.1

    final_candidate.loc[:, "runtime_seconds"] = 987.654
    joblib.dump(
        SimpleNamespace(
            selected_summary=final_selected,
            candidate_metrics=final_candidate,
        ),
        nested_dir / "final_selection.joblib",
        compress=3,
    )
    write_json_atomic(
        build_stage_manifest(
            nested_dir,
            (
                "nested_result.joblib",
                "final_selection.joblib",
                "final_fitted_model.joblib",
            ),
        ),
        nested_dir / "artifact_manifest.json",
    )
    assert _runtime_reproducibility_checks(tmp_path)["passed"] is False


def test_simulation_gate_rejects_string_booleans_and_invalid_probabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulations = 60_000
    audit_simulations = FROZEN_ADAPTIVE_CONFIG.audit_log_simulations
    expected = {
        "group_stage": 72,
        "round_of_32": 16,
        "round_of_16": 8,
        "quarter_final": 4,
        "semi_final": 2,
        "third_place": 1,
        "final": 1,
    }
    stage = pd.DataFrame(
        {
            "stage": list(expected),
            "matches_per_simulation": list(expected.values()),
            "simulated_matches": [count * simulations for count in expected.values()],
        }
    )

    def balanced_count_for(n_simulations: int, slots: int) -> np.ndarray:
        total = slots * n_simulations
        values = np.full(48, total // 48, dtype=int)
        values[: total - int(values.sum())] += 1
        return values

    def balanced_count(slots: int) -> np.ndarray:
        return balanced_count_for(simulations, slots)

    teams = pd.DataFrame({"team": [f"team_{index}" for index in range(48)]})
    event_counts = {
        "knockout": balanced_count(32),
        "round_of_16": balanced_count(16),
        "quarter_final": balanced_count(8),
        "semi_final": balanced_count(4),
        "final": balanced_count(2),
        "champion": balanced_count(1),
    }
    z = NormalDist().inv_cdf(0.975)
    for event, count in event_counts.items():
        probability = count / simulations
        teams[f"{event}_count"] = count
        teams[f"{event}_probability"] = probability
        teams[f"{event}_mcse"] = np.sqrt(probability * (1.0 - probability) / simulations)
        denominator = 1.0 + z**2 / simulations
        center = (probability + z**2 / (2 * simulations)) / denominator
        radius = (
            z
            * np.sqrt(probability * (1 - probability) / simulations + z**2 / (4 * simulations**2))
            / denominator
        )
        teams[f"{event}_ci_lower"] = np.clip(center - radius, 0, 1)
        teams[f"{event}_ci_upper"] = np.clip(center + radius, 0, 1)
    teams["expected_group_points"] = 4.0
    teams["expected_group_points_mcse"] = 0.01
    teams["expected_tournament_goals"] = 5.0
    teams["expected_tournament_goals_mcse"] = 0.02
    metadata = {
        "team_count": 48,
        "group_count": 12,
        "group_size": 4,
        "automatic_qualifiers_per_group": 2,
        "best_third_qualifiers": 8,
        "annex_c_rows": 495,
        "format_source_sha256": FIFA_2026_REGULATIONS_SHA256,
        "annex_c_csv_sha256": FIFA_2026_ANNEX_C_CSV_SHA256,
        "format_certification": (
            "official_regulations_M73_M104_graph_and_complete_annexe_c_mapping"
        ),
        "factual_2026_entrants_used": False,
        "audit_outcomes_used": False,
        "mean_tournament_goals": 240.0,
        "mean_tournament_goals_mcse": 0.5,
        "mcse_thresholds_satisfied": True,
        "stopping_reason": "all_mcse_thresholds_satisfied",
        "stopping_ledger_rows": 12,
        "batch_event_count_rows": 12 * 48 * 6,
        "continuous_sufficient_statistic_rows": 49,
        "adaptive_contract": adaptive_contract_payload(),
    }
    exact_contract = {
        "minimum_simulations": 10_000,
        "batch_size": 5_000,
        "maximum_simulations": 250_000,
        "advancement_mcse_threshold": 0.002,
        "champion_mcse_threshold": 0.001,
        "confidence_level": 0.95,
        "audit_log_simulations": audit_simulations,
        "seed": FROZEN_ADAPTIVE_SEED,
        "adaptive_contract_sha256": adaptive_contract_sha256(),
        "protocol_deviation_sha256": FROZEN_PROTOCOL_DEVIATION_SHA256,
    }
    metadata.update(exact_contract)
    summary = {
        "simulations": simulations,
        **exact_contract,
        "simulated_match_rows": simulations * 104,
        "match_log_rows": audit_simulations * 104,
        "advancement_mcse_threshold": 0.002,
        "champion_mcse_threshold": 0.001,
        "mcse_thresholds_satisfied": True,
        "stopping_reason": "all_mcse_thresholds_satisfied",
        "stopping_ledger_rows": 12,
        "batch_event_count_rows": 12 * 48 * 6,
        "continuous_sufficient_statistic_rows": 49,
        "mean_tournament_goals": 240.0,
        "mean_tournament_goals_mcse": 0.5,
        "scenario_only_not_factual_forecast": True,
    }
    stopping = pd.DataFrame(
        {
            "batch_index": range(12),
            "simulations": np.arange(5_000, 65_000, 5_000),
            "maximum_advancement_mcse": [0.01] * 11
            + [
                float(
                    teams[
                        [
                            "knockout_mcse",
                            "round_of_16_mcse",
                            "quarter_final_mcse",
                            "semi_final_mcse",
                            "final_mcse",
                        ]
                    ]
                    .to_numpy(float)
                    .max()
                )
            ],
            "maximum_champion_mcse": [0.01] * 11 + [float(teams["champion_mcse"].max())],
            "minimum_reached": [False] + [True] * 11,
            "advancement_threshold_satisfied": [False] * 11 + [True],
            "champion_threshold_satisfied": [False] * 11 + [True],
            "thresholds_satisfied": [False] * 11 + [True],
            "maximum_reached": [False] * 12,
            "decision": ["continue"] * 11 + ["all_mcse_thresholds_satisfied"],
        }
    )
    batch_rows: list[dict[str, object]] = []
    slot_counts = {
        "knockout": 32,
        "round_of_16": 16,
        "quarter_final": 8,
        "semi_final": 4,
        "final": 2,
        "champion": 1,
    }
    for batch_index, batch_simulations in enumerate(range(5_000, 60_001, 5_000)):
        for event, slots in slot_counts.items():
            counts = balanced_count_for(batch_simulations, slots)
            for team_index, count in enumerate(counts):
                batch_rows.append(
                    {
                        "batch_index": batch_index,
                        "simulations": batch_simulations,
                        "team": f"team_{team_index}",
                        "event": event,
                        "count": int(count),
                    }
                )
    batch_counts = pd.DataFrame(batch_rows)
    for batch_index, block in batch_counts.groupby("batch_index", sort=True):
        probability = block["count"].to_numpy(float) / block["simulations"].iloc[0]
        mcse = np.sqrt(probability * (1 - probability) / block["simulations"].iloc[0])
        is_champion = block["event"].eq("champion").to_numpy()
        stopping.loc[batch_index, "maximum_advancement_mcse"] = float(mcse[~is_champion].max())
        stopping.loc[batch_index, "maximum_champion_mcse"] = float(mcse[is_champion].max())
    stopping["advancement_threshold_satisfied"] = stopping["maximum_advancement_mcse"] <= 0.002
    stopping["champion_threshold_satisfied"] = stopping["maximum_champion_mcse"] <= 0.001
    stopping["thresholds_satisfied"] = (
        stopping["minimum_reached"]
        & stopping["advancement_threshold_satisfied"]
        & stopping["champion_threshold_satisfied"]
    )
    point_sum = 4 * simulations
    point_square_sum = 6 * (simulations - 1) + point_sum**2 / simulations
    goal_sum = 5 * simulations
    goal_square_sum = 24 * (simulations - 1) + goal_sum**2 / simulations
    sufficient = pd.DataFrame(
        [
            {
                "scope": "team",
                "entity": f"team_{team_index}",
                "simulations": simulations,
                "group_points_sum": point_sum,
                "group_points_square_sum": point_square_sum,
                "goals_sum": goal_sum,
                "goals_square_sum": goal_square_sum,
            }
            for team_index in range(48)
        ]
        + [
            {
                "scope": "tournament",
                "entity": "__tournament__",
                "simulations": simulations,
                "group_points_sum": 0,
                "group_points_square_sum": 0,
                "goals_sum": 240 * simulations,
                "goals_square_sum": (
                    15_000 * (simulations - 1) + (240 * simulations) ** 2 / simulations
                ),
            }
        ]
    )
    monkeypatch.setattr(
        promotion_checks,
        "_json",
        lambda path: metadata if path.name == "scenario_metadata.json" else summary,
    )
    monkeypatch.setattr(
        promotion_checks,
        "_csv",
        lambda path: (
            stage
            if path.name == "scenario_stage_summary.csv"
            else stopping
            if path.name == "adaptive_stopping_ledger.csv"
            else batch_counts
            if path.name == "adaptive_batch_event_counts.csv.gz"
            else sufficient
            if path.name == "continuous_sufficient_statistics.csv"
            else teams
        ),
    )
    log_rows = []
    for simulation_id in range(audit_simulations):
        for stage_name, count in expected.items():
            for match in range(count):
                log_rows.append(
                    {
                        "simulation_id": simulation_id,
                        "match_number": sum(
                            list(expected.values())[: list(expected).index(stage_name)]
                        )
                        + match
                        + 1,
                        "stage": stage_name,
                        "home_team": f"team_{match % 48}",
                        "away_team": f"team_{(match + 1) % 48}",
                        "winner": f"team_{match % 48}",
                        "home_source_slot": "source_home",
                        "away_source_slot": "source_away",
                    }
                )
    match_log = pd.DataFrame(log_rows)
    monkeypatch.setattr(
        promotion_checks.pd,
        "read_csv",
        lambda *args, **kwargs: match_log,
    )
    simulation_checks = _simulation_checks(tmp_path)
    assert simulation_checks["passed"] is True
    assert type(simulation_checks["no_self_matches_and_known_teams"]) is bool

    first_group = match_log.index[
        match_log["simulation_id"].eq(0) & match_log["stage"].eq("group_stage")
    ][0]
    second_round = match_log.index[
        match_log["simulation_id"].eq(1) & match_log["stage"].eq("round_of_32")
    ][0]
    match_log.loc[first_group, "stage"] = "round_of_32"
    match_log.loc[second_round, "stage"] = "group_stage"
    assert _simulation_checks(tmp_path)["audit_matches_104_each"] is False
    match_log.loc[first_group, "stage"] = "group_stage"
    match_log.loc[second_round, "stage"] = "round_of_32"

    match_log["match_number"] = match_log["match_number"].astype(float)
    match_log.loc[first_group, "match_number"] += 0.5
    assert _simulation_checks(tmp_path)["audit_matches_104_each"] is False
    match_log.loc[first_group, "match_number"] -= 0.5

    original_winner = match_log.loc[second_round, "winner"]
    match_log.loc[second_round, "winner"] = "not_a_participant"
    assert _simulation_checks(tmp_path)["audit_matches_104_each"] is False
    match_log.loc[second_round, "winner"] = original_winner

    summary["maximum_simulations"] = 1
    assert _simulation_checks(tmp_path)["frozen_adaptive_contract_exact"] is False
    summary["maximum_simulations"] = FROZEN_ADAPTIVE_CONFIG.maximum_simulations

    teams.loc[0, "knockout_count"] += 1
    changed_probability = teams.loc[0, "knockout_count"] / simulations
    teams.loc[0, "knockout_probability"] = changed_probability
    teams.loc[0, "knockout_mcse"] = np.sqrt(
        changed_probability * (1 - changed_probability) / simulations
    )
    denominator = 1.0 + z**2 / simulations
    center = (changed_probability + z**2 / (2 * simulations)) / denominator
    radius = (
        z
        * np.sqrt(
            changed_probability * (1 - changed_probability) / simulations
            + z**2 / (4 * simulations**2)
        )
        / denominator
    )
    teams.loc[0, "knockout_ci_lower"] = max(center - radius, 0)
    teams.loc[0, "knockout_ci_upper"] = min(center + radius, 1)
    assert _simulation_checks(tmp_path)["advancement_counts_integer_and_conserved"] is False
    teams.loc[0, "knockout_count"] -= 1
    original_probability = teams.loc[0, "knockout_count"] / simulations
    teams.loc[0, "knockout_probability"] = original_probability
    teams.loc[0, "knockout_mcse"] = np.sqrt(
        original_probability * (1 - original_probability) / simulations
    )
    center = (original_probability + z**2 / (2 * simulations)) / denominator
    radius = (
        z
        * np.sqrt(
            original_probability * (1 - original_probability) / simulations
            + z**2 / (4 * simulations**2)
        )
        / denominator
    )
    teams.loc[0, "knockout_ci_lower"] = max(center - radius, 0)
    teams.loc[0, "knockout_ci_upper"] = min(center + radius, 1)

    summary["scenario_only_not_factual_forecast"] = "true"
    assert _simulation_checks(tmp_path)["scenario_not_factual"] is False
    summary["scenario_only_not_factual_forecast"] = True
    teams.loc[0, "champion_probability"] = -0.1
    with pytest.raises(ValueError, match="malformed"):
        _simulation_checks(tmp_path)

    teams.loc[0, "champion_probability"] = event_counts["champion"][0] / simulations
    teams.loc[:, "champion_ci_lower"] = 0.0
    teams.loc[:, "champion_ci_upper"] = 1.0
    assert _simulation_checks(tmp_path)["probability_mcse_intervals_and_continuous_valid"] is False


def test_decision_mapping_and_frozen_h2_keep_ece_as_a_disclosed_secondary_warning() -> None:
    years = [1986, 1990, 1994, 1998, 2002, 2006, 2010]
    mapped = normalize_decision_metrics(
        pd.DataFrame(
            {
                "test_year": years,
                "decision_draw_predictions": [1] * 7,
                "test_match_count": [52] * 7,
                "macro_f1_gain": [0.1] * 7,
                "draw_recall_gain": [0.2] * 7,
            }
        )
    )
    assert mapped["positive_draw_predictions"].tolist() == [1] * 7
    raw = pd.DataFrame(
        {
            "test_year": years,
            "log_loss": [0.95] * 7,
            "multiclass_brier": [0.565] * 7,
            "calibration_error": [0.09] * 7,
        }
    )
    calibrated = raw.assign(
        log_loss=0.9502,
        multiclass_brier=0.5647,
        calibration_error=0.11,
    )
    result = evaluate_calibration_retention(raw, calibrated)
    assert result["retained_under_frozen_h2"] is True
    assert result["eligible_primary_metrics"] == ["multiclass_brier"]
    assert result["ece_regressed"] is True
    assert result["ece_is_secondary_nonblocking"] is True


def test_audit_freezes_before_open_and_repeat_only_verifies_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evidence"
    certificate_path = root / "promotion" / "promotion_certificate.json"
    certificate = integrity_envelope({"certificate_type": "preaudit_promotion"})
    write_json_atomic(certificate, certificate_path)
    protocol = {"protocol_bundle_sha256": "p" * 64}
    initialize_unopened_sentinel(
        root,
        certificate_path=certificate_path,
        audit_protocol_identity=protocol,
    )
    monkeypatch.setattr(
        "worldcup_forecasting.validation.promotion.verify_promotion_certificate",
        lambda *args, **kwargs: {"audit_protocol_identity": protocol},
    )
    monkeypatch.setattr(
        audit_execution_module,
        "load_persisted_blind_audit_predictions",
        lambda *args, **kwargs: SimpleNamespace(
            metadata={
                "verified": True,
                "predictions_generated_at_utc": "2026-07-18T12:00:00+00:00",
                "formal_source_commit": "c" * 40,
                "pre2014_freeze_manifest_sha256": "a" * 64,
            }
        ),
    )
    events: list[str] = []

    def freeze() -> FrozenAuditPreparation[str]:
        events.append("freeze")
        return FrozenAuditPreparation(
            "frozen",
            "f" * 64,
            {
                "advanced_predictor": True,
                "calibration": True,
                "decision": True,
                "official_outcome_logit": True,
                "independent_mirrored_poisson": True,
                "legacy_mirrored_dixon_coles": True,
            },
            {
                "fit_end_year": 2010,
                "formal_source_commit": "c" * 40,
                "pre2014_freeze_manifest_sha256": "a" * 64,
                "pre2014_freeze_source_commit": "f" * 40,
            },
        )

    def open_features() -> str:
        events.append("open_features")
        return "features"

    def prepare(frozen: str, features: str) -> str:
        events.append("prepare")
        assert (frozen, features) == ("frozen", "features")
        return "blind"

    def persist_blind(_: str, audit_dir: Path) -> PersistedAudit:
        events.append("persist_blind")
        paths = (
            "2014_predictions_blind.parquet",
            "2014_joint_predictions_blind.parquet",
            "2014_predictions_blind_metadata.json",
        )
        for path in paths:
            (audit_dir / path).write_bytes(b"blind")
        return PersistedAudit(
            paths,
            {
                "status": "blind",
                "predictions_generated_at_utc": "2026-07-18T12:00:00+00:00",
            },
        )

    def open_truth() -> str:
        events.append("open_truth")
        return "targets"

    def evaluate(audit_dir: Path, opened: str, manifest_hash: str) -> str:
        events.append("evaluate")
        assert opened == "targets"
        assert len(manifest_hash) == 64
        assert (audit_dir / "blind_prediction_manifest.json").is_file()
        return "result"

    def persist(result: str, audit_dir: Path) -> PersistedAudit:
        events.append("persist")
        assert result == "result"
        (audit_dir / "metrics.json").write_text("{}", encoding="utf-8")
        return PersistedAudit(("metrics.json",), {"status": "complete"})

    first = run_one_time_audit(
        root,
        freeze=freeze,
        open_blind_features=open_features,
        prepare_blind=prepare,
        persist_blind=persist_blind,
        open_audit_truth=open_truth,
        evaluate_persisted_blind=evaluate,
        persist=persist,
    )
    assert first == {"status": "complete"}
    assert events == [
        "freeze",
        "open_features",
        "prepare",
        "persist_blind",
        "open_truth",
        "evaluate",
        "persist",
    ]

    def forbidden() -> Any:
        raise AssertionError("cached verification must not invoke callbacks")

    second = run_one_time_audit(
        root,
        freeze=forbidden,
        open_blind_features=forbidden,
        prepare_blind=lambda *_: forbidden(),
        persist_blind=lambda *_: forbidden(),
        open_audit_truth=forbidden,
        evaluate_persisted_blind=lambda *_: forbidden(),
        persist=lambda *_: forbidden(),
    )
    assert second == first
    (root / "audit" / "2014_joint_predictions_blind.parquet").unlink()
    with pytest.raises(ArtifactIntegrityError, match="missing|omits"):
        verify_cached_audit(root)


def test_two_phase_audit_persists_blind_predictions_before_truth_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "evidence"
    certificate_path = root / "promotion" / "promotion_certificate.json"
    write_json_atomic(
        integrity_envelope({"certificate_type": "preaudit_promotion"}),
        certificate_path,
    )
    protocol = {"protocol_bundle_sha256": "p" * 64}
    initialize_unopened_sentinel(
        root,
        certificate_path=certificate_path,
        audit_protocol_identity=protocol,
    )
    monkeypatch.setattr(
        "worldcup_forecasting.validation.promotion.verify_promotion_certificate",
        lambda *args, **kwargs: {"audit_protocol_identity": protocol},
    )
    monkeypatch.setattr(
        audit_execution_module,
        "load_persisted_blind_audit_predictions",
        lambda *args, **kwargs: SimpleNamespace(
            metadata={
                "verified": True,
                "predictions_generated_at_utc": "2026-07-18T12:00:00+00:00",
                "formal_source_commit": "c" * 40,
                "pre2014_freeze_manifest_sha256": "a" * 64,
            }
        ),
    )
    events: list[str] = []
    prepared = FrozenAuditPreparation(
        "frozen",
        "f" * 64,
        {
            name: True
            for name in (
                "advanced_predictor",
                "calibration",
                "decision",
                "official_outcome_logit",
                "independent_mirrored_poisson",
                "legacy_mirrored_dixon_coles",
            )
        },
        {
            "fit_end_year": 2010,
            "formal_source_commit": "c" * 40,
            "pre2014_freeze_manifest_sha256": "a" * 64,
            "pre2014_freeze_source_commit": "f" * 40,
        },
    )

    def persist_blind(_: str, audit_dir: Path) -> PersistedAudit:
        events.append("persist_blind")
        paths = (
            "2014_predictions_blind.parquet",
            "2014_joint_predictions_blind.parquet",
            "2014_predictions_blind_metadata.json",
        )
        for path in paths:
            (audit_dir / path).write_bytes(b"blind")
        return PersistedAudit(
            paths,
            {
                "truth_columns_accessed": False,
                "predictions_generated_at_utc": "2026-07-18T12:00:00+00:00",
            },
        )

    def open_truth() -> str:
        events.append("release_truth")
        assert (root / "audit" / "2014_predictions_blind.parquet").is_file()
        assert (root / "audit" / "blind_prediction_manifest.json").is_file()
        state = verify_integrity_envelope(
            json.loads((root / "audit" / "audit_state.json").read_text(encoding="utf-8"))
        )
        assert state["state"] == "opening_truth"
        assert state["truth_released"] is False
        return "truth"

    def release(audit_dir: Path, truth: str, manifest_hash: str) -> str:
        events.append("score_persisted_blind")
        assert audit_dir == root / "audit"
        assert truth == "truth"
        assert len(manifest_hash) == 64
        state = verify_integrity_envelope(
            json.loads((root / "audit" / "audit_state.json").read_text(encoding="utf-8"))
        )
        assert state["state"] == "truth_released"
        assert state["truth_released"] is True
        return "scored"

    def persist_scored(result: str, audit_dir: Path) -> PersistedAudit:
        events.append("persist_scored")
        assert result == "scored"
        (audit_dir / "2014_truth_released.parquet").write_bytes(b"truth")
        (audit_dir / "2014_external_audit.json").write_text("{}", encoding="utf-8")
        return PersistedAudit(
            ("2014_truth_released.parquet", "2014_external_audit.json"),
            {"status": "one_time_2014_audit_complete"},
        )

    summary = run_one_time_audit(
        root,
        freeze=lambda: (events.append("freeze") or prepared),
        open_blind_features=lambda: (events.append("open_features") or "features"),
        prepare_blind=lambda frozen, features: (
            events.append("predict_blind") or "blind"
        ),
        persist_blind=persist_blind,
        open_audit_truth=open_truth,
        evaluate_persisted_blind=release,
        persist=persist_scored,
    )

    assert summary["status"] == "one_time_2014_audit_complete"
    assert events == [
        "freeze",
        "open_features",
        "predict_blind",
        "persist_blind",
        "release_truth",
        "score_persisted_blind",
        "persist_scored",
    ]
    cached = verify_cached_audit(root)
    assert cached == summary
    result_certificate_path = root / "audit" / "audit_result_certificate.json"
    forged_result = verify_integrity_envelope(
        json.loads(result_certificate_path.read_text(encoding="utf-8"))
    )
    forged_result["two_phase_blind_audit"] = False
    write_json_atomic(integrity_envelope(forged_result), result_certificate_path)
    state_path = root / "audit" / "audit_state.json"
    forged_state = verify_integrity_envelope(
        json.loads(state_path.read_text(encoding="utf-8"))
    )
    forged_state["audit_result_certificate_sha256"] = hashlib.sha256(
        result_certificate_path.read_bytes()
    ).hexdigest()
    write_json_atomic(integrity_envelope(forged_state), state_path)
    with pytest.raises(ArtifactIntegrityError, match="authorized two-phase"):
        verify_cached_audit(root)


def test_audit_execution_artifacts_keep_blind_predictions_target_free(tmp_path: Path) -> None:
    joint = np.tile(np.asarray([[[0.2, 0.1], [0.2, 0.5]]]), (64, 1, 1))

    class JointBatch:
        probability = joint

        def outcome_probabilities(self) -> np.ndarray:
            return np.tile(np.asarray([[0.2, 0.7, 0.1]]), (64, 1))

    class JointModel:
        def predict_joint(self, _: pd.DataFrame) -> JointBatch:
            return JointBatch()

    class Selector:
        def transform(self, values: np.ndarray) -> np.ndarray:
            return values.copy()

    class Decision:
        def predict(self, values: np.ndarray) -> np.ndarray:
            return np.asarray(["draw"] * len(values))

    class Official:
        def predict_proba(self, _: pd.DataFrame) -> np.ndarray:
            return np.tile(np.asarray([[0.3, 0.4, 0.3]]), (64, 1))

    frozen = FrozenAuditModels(
        advanced=JointModel(),
        calibration=Selector(),
        decision=Decision(),
        official=Official(),
        independent=JointModel(),
        dixon_coles=JointModel(),
        identity={"final_fit_hash": "f" * 64},
        metadata={
            "fit_end_year": 2010,
            "formal_source_commit": "c" * 40,
            "pre2014_freeze_manifest_sha256": "m" * 64,
            "pre2014_freeze_source_commit": "f" * 40,
        },
    )
    audit = pd.DataFrame(
        {
            "Year": [2014] * 64,
            "MatchID": np.arange(64),
            "Datetime": pd.date_range("2014-06-01", periods=64, freq="h"),
            "target_outcome": ["home_win"] * 64,
            "target_home_goals": [1] * 64,
            "target_away_goals": [0] * 64,
        }
    )
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()

    features = audit.drop(
        columns=["target_outcome", "target_home_goals", "target_away_goals"]
    )
    with pytest.raises(ValueError, match="exposes target columns"):
        prepare_frozen_audit_predictions(frozen, audit)
    blind = prepare_frozen_audit_predictions(frozen, features)
    assert "target_outcome" not in blind.predictions
    persisted = persist_blind_audit_predictions(blind, audit_dir)
    manifest_path = write_json_atomic(
        build_stage_manifest(audit_dir, persisted.relative_paths),
        audit_dir / "blind_prediction_manifest.json",
    )
    verified_manifest = verify_stage_manifest(
        manifest_path,
        required_paths=persisted.relative_paths,
    )
    persisted_blind = pd.read_parquet(audit_dir / "2014_predictions_blind.parquet")
    assert "target_outcome" not in persisted_blind
    assert persisted_blind["prediction_time"].notna().all()
    persisted_joint = pd.read_parquet(
        audit_dir / "2014_joint_predictions_blind.parquet"
    )
    assert len(persisted_joint) == len(blind.joint_grid)
    reloaded = load_persisted_blind_audit_predictions(
        audit_dir,
        expected_manifest_sha256=verified_manifest["manifest_sha256"],
    )
    assert np.array_equal(
        reloaded.joint_models["advanced_calibrated"],
        blind.joint_models["advanced_calibrated"],
    )

    result = evaluate_persisted_blind_audit(
        audit_dir,
        audit,
        verified_manifest["manifest_sha256"],
    )
    persist_audit_result(result, audit_dir)
    truth = pd.read_parquet(audit_dir / "2014_truth_released.parquet")
    assert len(truth) == 64
    assert set(truth["target_outcome"]) == {"home_win"}
    external = json.loads((audit_dir / "2014_external_audit.json").read_text(encoding="utf-8"))
    assert external["truth_released_after_blind_persistence"] is True
    assert external["selection_reopened"] is False


def test_batch_blind_features_are_invariant_to_every_2014_score(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_hash = json.loads(
        (project_root / "artifacts" / "freeze" / "FREEZE_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )["feature_schema_hash"]
    baseline = load_batch_blind_2014_features(
        project_root,
        feature_schema_hash=schema_hash,
    )
    canonical = build_canonical_data(project_root / "data" / "raw")
    changed_matches = canonical.matches.copy()
    audit_rows = changed_matches["Year"].eq(2014)
    changed_matches.loc[audit_rows, "Home Team Goals"] += 7
    changed_matches.loc[audit_rows, "Away Team Goals"] += 3
    changed = replace(canonical, matches=changed_matches)
    monkeypatch.setattr(audit_execution_module, "build_canonical_data", lambda _: changed)

    perturbed = load_batch_blind_2014_features(
        project_root,
        feature_schema_hash=schema_hash,
    )
    pd.testing.assert_frame_equal(baseline, perturbed, check_exact=True)
    assert not set(("target_outcome", "target_home_goals", "target_away_goals")) & set(
        perturbed.columns
    )
    released_truth = load_isolated_2014_truth(project_root)
    assert int(released_truth["target_home_goals"].sum()) > 400


def test_audit_model_deserialization_waits_for_both_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified: list[str] = []

    def verify(path: Path, **_: Any) -> dict[str, Any]:
        verified.append(path.name)
        if len(verified) == 2:
            raise ArtifactIntegrityError("baseline manifest rejected")
        return {"status": "verified"}

    monkeypatch.setattr(
        "worldcup_forecasting.experiments.audit_execution.verify_stage_manifest",
        verify,
    )
    monkeypatch.setattr(
        "worldcup_forecasting.experiments.audit_execution.joblib.load",
        lambda _: (_ for _ in ()).throw(
            AssertionError("joblib must not load before both manifests verify")
        ),
    )
    with pytest.raises(ArtifactIntegrityError, match="baseline manifest"):
        freeze_audit_models(
            tmp_path,
            audit_protocol_identity={"frozen_audit_protocol_hash": "a" * 64},
        )
    assert verified == ["artifact_manifest.json", "baseline_freeze_manifest.json"]


def test_deserialized_audit_baselines_must_keep_exact_classes_and_configs() -> None:
    official = OutcomeLogit()
    official.scaler_ = object()  # type: ignore[assignment]
    official.classifier_ = object()  # type: ignore[assignment]
    independent = JointPoissonModel(
        JointPoissonConfig(
            optional_diff_features=(),
            alpha=0.5,
            time_decay_half_life_years=12.0,
            rho=0.0,
            max_score=12,
            min_retained_mass=0.999,
            max_support=100,
        )
    )
    dixon_coles = JointPoissonModel(
        JointPoissonConfig(
            optional_diff_features=(
                "pre_match_elo_diff",
                "attack_state_diff",
                "defense_state_diff",
            ),
            alpha=0.5,
            time_decay_half_life_years=12.0,
            rho=None,
            max_score=12,
            min_retained_mass=0.999,
            max_support=100,
        )
    )
    for model in (independent, dixon_coles):
        model.pipeline_ = object()  # type: ignore[assignment]
        model.feature_columns_ = ("historical_appearances_diff",)
    validate_fixed_audit_baseline_models(official, independent, dixon_coles)

    independent.config = JointPoissonConfig(rho=0.0)
    with pytest.raises(ValueError, match="independent.*config"):
        validate_fixed_audit_baseline_models(official, independent, dixon_coles)


def test_failed_open_attempt_is_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evidence"
    certificate_path = root / "promotion" / "promotion_certificate.json"
    write_json_atomic(
        integrity_envelope({"certificate_type": "preaudit_promotion"}),
        certificate_path,
    )
    protocol = {"protocol_bundle_sha256": "p" * 64}
    initialize_unopened_sentinel(
        root, certificate_path=certificate_path, audit_protocol_identity=protocol
    )
    monkeypatch.setattr(
        "worldcup_forecasting.validation.promotion.verify_promotion_certificate",
        lambda *args, **kwargs: {"audit_protocol_identity": protocol},
    )
    prepared = FrozenAuditPreparation(
        object(),
        "f" * 64,
        {
            name: True
            for name in (
                "advanced_predictor",
                "calibration",
                "decision",
                "official_outcome_logit",
                "independent_mirrored_poisson",
                "legacy_mirrored_dixon_coles",
            )
        },
        {
            "fit_end_year": 2010,
            "formal_source_commit": "c" * 40,
            "pre2014_freeze_manifest_sha256": "a" * 64,
            "pre2014_freeze_source_commit": "f" * 40,
        },
    )
    with pytest.raises(OSError, match="sealed source"):
        run_one_time_audit(
            root,
            freeze=lambda: prepared,
            open_blind_features=lambda: (_ for _ in ()).throw(
                OSError("sealed source failed")
            ),
            prepare_blind=lambda *_: None,
            persist_blind=lambda *_: PersistedAudit(("blind",), {"status": "never"}),
            open_audit_truth=lambda: None,
            evaluate_persisted_blind=lambda *_: None,
            persist=lambda *_: PersistedAudit(("x",), {"status": "never"}),
        )
    with pytest.raises(ArtifactIntegrityError, match="already opened|non-repeatable"):
        run_one_time_audit(
            root,
            freeze=lambda: prepared,
            open_blind_features=lambda: "must-not-open",
            prepare_blind=lambda *_: None,
            persist_blind=lambda *_: PersistedAudit(("blind",), {"status": "never"}),
            open_audit_truth=lambda: None,
            evaluate_persisted_blind=lambda *_: None,
            persist=lambda *_: PersistedAudit(("x",), {"status": "never"}),
        )


def test_preaudit_pipeline_preserves_strict_order_and_reuses_only_verified_checkpoints(
    tmp_path: Path,
) -> None:
    states = {
        name: "missing"
        for name in (
            "nested",
            "evaluate",
            "stress",
            "ablation",
            "parameter",
            "simulate",
            "decompose",
        )
    }
    events: list[str] = []

    def checkpoint(_: Path, stage: str) -> dict[str, str]:
        return {"stage": stage, "status": states[stage]}

    def runner(stage: str):
        def run(_: Path) -> dict[str, str]:
            events.append(stage)
            states[stage] = "complete"
            return {"status": f"{stage}_complete"}

        return run

    runners = {name: runner(name) for name in states}
    result = run_preaudit_pipeline(
        tmp_path,
        stage_runners=runners,
        promote=lambda _: integrity_envelope(
            {"certificate_type": "preaudit_promotion", "decision": "promote"}
        ),
        checkpoint=checkpoint,
    )
    assert events == list(states)
    assert result["status"] == "preaudit_complete_audit_authorized"
    assert result["audit_executed"] is False


def test_direct_promotion_cannot_bypass_a_sealed_rejection(tmp_path: Path) -> None:
    rejection = integrity_envelope(
        {
            "certificate_type": "preaudit_promotion_rejection",
            "decision": "retain_baseline",
        }
    )
    write_json_atomic(
        rejection,
        tmp_path / "promotion" / "promotion_rejection.json",
    )
    with pytest.raises(PromotionRejectedError, match="rejection|forbidden"):
        issue_promotion_certificate(tmp_path)


def test_existing_certificate_recovers_missing_sentinel_without_second_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evidence"
    certificate_path = root / "promotion" / "promotion_certificate.json"
    envelope = integrity_envelope({"certificate_type": "preaudit_promotion"})
    write_json_atomic(envelope, certificate_path)
    protocol = {"protocol_bundle_sha256": "p" * 64}
    reads = 0
    original_json = promotion_artifacts._json

    def counted_json(path: Path) -> dict[str, Any]:
        nonlocal reads
        reads += 1
        return original_json(path)

    monkeypatch.setattr(promotion_artifacts, "_json", counted_json)
    monkeypatch.setattr(
        promotion_artifacts,
        "_verify_promotion_envelope",
        lambda *args, **kwargs: {"audit_protocol_identity": protocol},
    )
    recovered = promotion_artifacts.issue_promotion_certificate(
        root,
        authentication_key=None,
        key_id=None,
    )
    assert recovered == envelope
    assert reads == 1
    sentinel = root / "audit" / "unopened_sentinel.json"
    assert sentinel.is_file()
    assert (
        initialize_unopened_sentinel(
            root,
            certificate_path=certificate_path,
            audit_protocol_identity=protocol,
        )
        == sentinel
    )


def test_unopened_sentinel_rejects_contradictory_control_states(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    certificate_path = root / "promotion" / "promotion_certificate.json"
    write_json_atomic(
        integrity_envelope({"certificate_type": "preaudit_promotion"}),
        certificate_path,
    )
    protocol = {"protocol_bundle_sha256": "p" * 64}
    sentinel = initialize_unopened_sentinel(
        root,
        certificate_path=certificate_path,
        audit_protocol_identity=protocol,
    )
    write_json_atomic(
        integrity_envelope(
            {
                "state": "opening",
                "targets_opened": True,
                "open_attempts": 1,
                "promotion_certificate_sha256": "x" * 64,
                "audit_protocol_identity": protocol,
            }
        ),
        root / "audit" / "audit_state.json",
    )
    with pytest.raises(ArtifactIntegrityError, match="contradictory"):
        initialize_unopened_sentinel(
            root,
            certificate_path=certificate_path,
            audit_protocol_identity=protocol,
        )
    assert sentinel.is_file()


def test_promotion_verification_cannot_disable_artifact_revalidation(tmp_path: Path) -> None:
    with pytest.raises(ArtifactIntegrityError, match="cannot be disabled"):
        promotion_artifacts.verify_promotion_certificate(
            tmp_path / "missing.json",
            authentication_key=None,
            revalidate_artifacts=False,
        )
