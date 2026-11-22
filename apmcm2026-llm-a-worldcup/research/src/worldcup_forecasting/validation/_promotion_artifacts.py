"""Manifest-bound promotion certificates and checkpoint verification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldcup_forecasting.experiments.audit_baselines import (
    fixed_audit_baseline_identities,
)
from worldcup_forecasting.experiments.development_snapshot import (
    load_development_snapshot,
)
from worldcup_forecasting.experiments.final_audit import frozen_audit_protocol_hash
from worldcup_forecasting.simulation.adaptive import (
    FROZEN_ADAPTIVE_CONFIG,
    FROZEN_ADAPTIVE_SEED,
    FROZEN_PROTOCOL_DEVIATION_SHA256,
    adaptive_contract_sha256,
)

from ._promotion_checks import evaluate_verified_evidence
from .artifact_integrity import (
    ArtifactIntegrityError,
    file_sha256,
    integrity_envelope,
    payload_sha256,
    verify_integrity_envelope,
    verify_stage_manifest,
    write_json_atomic,
)
from .runtime_reproducibility import RUNTIME_REFERENCE_PROTOCOL_SHA256


@dataclass(frozen=True)
class _StageContract:
    directory: str
    manifest: str
    summary: str
    required: tuple[str, ...]
    status_prefix: str


_STAGES: dict[str, _StageContract] = {
    "nested": _StageContract(
        "nested",
        "artifact_manifest.json",
        "nested_stage_summary.json",
        (
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
        ),
        "nested_stage_complete",
    ),
    "evaluate": _StageContract(
        "evaluation",
        "artifact_manifest.json",
        "evaluation_stage_summary.json",
        (
            "combined_fold_metrics.csv",
            "baseline_metadata.json",
            "baseline_outcome_fold_metrics.csv",
            "baseline_joint_fold_metrics.csv",
            "raw_fold_metrics.csv",
            "calibrated_fold_metrics.csv",
            "decision_fold_metrics.csv",
            "calibrated_oof_ledger.csv",
            "calibrated_joint_grid.csv",
            "evaluation_lineage.json",
            "baseline_result.joblib",
            "calibration_result.joblib",
            "decision_result.joblib",
            "sarmanov_oof_ledger.csv",
            "sarmanov_joint_grid.csv",
            "sarmanov_selection.csv",
            "sarmanov_fold_metrics.csv",
            "sarmanov_match_losses.csv",
            "sarmanov_classwise_calibration.csv",
            "sarmanov_metadata.json",
            "sarmanov_states.json",
            "sarmanov_evidence_metadata.json",
            "sarmanov_result.joblib",
            "statistical_closure.json",
            "statistical_fold_comparison.csv",
            "statistical_model_ranking.csv",
            "statistical_classwise_calibration.csv",
        ),
        "evaluation_stage_complete",
    ),
    "stress": _StageContract(
        "stress",
        "artifact_manifest.json",
        "stress_stage_summary.json",
        (
            "structural_stress_matrix.csv",
            "stress_lineage.json",
            "extreme_support_boundary.json",
            "format_transport_audits.csv",
        ),
        "stress_stage_complete",
    ),
    "ablation": _StageContract(
        "ablation",
        "artifact_manifest.json",
        "ablation_stage_summary.json",
        (
            "ablation_matrix.csv",
            "ablation_predictions.csv",
            "ablation_lineage.json",
            "ablation_metadata.json",
        ),
        "ablation_stage_complete",
    ),
    "parameter": _StageContract(
        "uncertainty",
        "parameter_artifact_manifest.json",
        "parameter_uncertainty_stage_summary.json",
        (
            "parameter_metadata.json",
            "parameter_draw_ledger.csv",
            "parameter_probability_draws.csv",
            "parameter_probability_summary.csv",
            "parameter_failures.csv",
        ),
        "parameter_uncertainty_stage_complete",
    ),
    "simulate": _StageContract(
        "simulation",
        "artifact_manifest.json",
        "simulation_stage_summary.json",
        (
            "scenario_metadata.json",
            "scenario_match_log.csv.gz",
            "adaptive_stopping_ledger.csv",
            "adaptive_batch_event_counts.csv.gz",
            "continuous_sufficient_statistics.csv",
            "scenario_stage_summary.csv",
            "scenario_team_summary.csv",
            "scenario_team_profiles.csv",
        ),
        "illustrative_2026_format_scenario_complete",
    ),
    "decompose": _StageContract(
        "uncertainty",
        "decomposition_artifact_manifest.json",
        "variance_decomposition_stage_summary.json",
        (
            "variance_decomposition.json",
            "variance_draw_cube.joblib",
            "variance_cell_summary.csv",
        ),
        "variance_decomposition_stage_complete",
    ),
}


def checkpoint_status(output_root: Path, stage: str) -> dict[str, Any]:
    """Return missing/complete, but raise on partial or corrupted checkpoints."""
    root = output_root.resolve()
    contract = _stage(stage)
    manifest = root / contract.directory / contract.manifest
    summary_path = root / contract.summary
    if not manifest.exists() and not summary_path.exists():
        return {"stage": stage, "status": "missing"}
    if not manifest.is_file() or not summary_path.is_file():
        raise ArtifactIntegrityError(
            f"Stage {stage!r} has a partial checkpoint; refusing to overwrite it"
        )
    binding = verify_stage_manifest(manifest, required_paths=contract.required)
    _verify_exact_stage_inventory(root, stage, contract, binding["declared_paths"])
    summary = _json(summary_path)
    status = str(summary.get("status", ""))
    if not status.startswith(contract.status_prefix):
        raise ArtifactIntegrityError(f"Stage {stage!r} summary status is not complete")
    _verify_summary_boundaries(stage, summary)
    return {
        "stage": stage,
        "status": "complete",
        "manifest": binding,
        "summary_sha256": file_sha256(summary_path),
    }


def verify_all_stage_checkpoints(output_root: Path) -> dict[str, Any]:
    """Verify all seven ordered pre-audit checkpoints."""
    bindings: dict[str, Any] = {}
    shared_uncertainty_paths: set[str] = set()
    for stage in _STAGES:
        result = checkpoint_status(output_root, stage)
        if result["status"] != "complete":
            raise ArtifactIntegrityError(f"Required pre-audit checkpoint is missing: {stage}")
        bindings[stage] = {
            "manifest_path": str(
                Path(result["manifest"]["manifest_path"])
                .relative_to(output_root.resolve())
                .as_posix()
            ),
            "manifest_sha256": result["manifest"]["manifest_sha256"],
            "artifact_count": result["manifest"]["artifact_count"],
            "summary_sha256": result["summary_sha256"],
        }
        if stage in {"parameter", "decompose"}:
            shared_uncertainty_paths.update(result["manifest"]["declared_paths"])
    _verify_shared_uncertainty_inventory(output_root.resolve(), shared_uncertainty_paths)
    return bindings


def evaluate_promotion_artifacts(output_root: Path) -> dict[str, Any]:
    """Verify bytes, governance locks, and all semantic promotion gates."""
    root = output_root.resolve()
    manifests = verify_all_stage_checkpoints(root)
    baseline_binding, baseline_identity = _baseline_freeze_binding(root)
    manifests["audit_baselines"] = baseline_binding
    governance = _governance_identity(root)
    identity = _json(root / "nested" / "frozen_model_identity.json")
    if (
        identity.get("development_snapshot_manifest_sha256")
        != governance["development_snapshot_manifest_sha256"]
        or identity.get("development_frame_hash") != governance["development_snapshot_frame_hash"]
        or identity.get("feature_schema_hash")
        != governance["development_snapshot_feature_schema_hash"]
        or identity.get("formal_preaudit_reads_combined_raw") is not False
    ):
        raise ArtifactIntegrityError(
            "Frozen model identity is not bound to the governed development snapshot"
        )
    gate = evaluate_verified_evidence(root)
    audit_protocol = _audit_protocol_identity(governance)
    return {
        "gate": gate,
        "manifest_bindings": manifests,
        "baseline_identities": baseline_identity,
        "governance": governance,
        "audit_protocol_identity": audit_protocol,
    }


def issue_promotion_certificate(
    output_root: Path,
    *,
    authentication_key: bytes | None,
    key_id: str | None,
) -> dict[str, Any]:
    """Persist a deterministic certificate only when every frozen gate passes."""
    from .audit_orchestration import initialize_unopened_sentinel
    from .promotion import PromotionRejectedError

    root = output_root.resolve()
    certificate_path = root / "promotion" / "promotion_certificate.json"
    rejection_path = root / "promotion" / "promotion_rejection.json"
    if rejection_path.is_file():
        rejection = verify_integrity_envelope(
            _json(rejection_path), authentication_key=authentication_key
        )
        if rejection.get("certificate_type") != "preaudit_promotion_rejection":
            raise ArtifactIntegrityError("Existing promotion rejection is malformed")
        raise PromotionRejectedError(
            "A sealed promotion rejection already exists; audit authorization is forbidden"
        )
    if certificate_path.is_file():
        envelope = _json(certificate_path)
        payload = _verify_promotion_envelope(
            certificate_path.absolute(),
            envelope,
            authentication_key=authentication_key,
        )
        initialize_unopened_sentinel(
            root,
            certificate_path=certificate_path,
            audit_protocol_identity=payload["audit_protocol_identity"],
        )
        return envelope
    evidence = evaluate_promotion_artifacts(root)
    gate = evidence["gate"]
    if not gate["system_promoted"]:
        rejection = integrity_envelope(
            {
                "schema_version": 1,
                "certificate_type": "preaudit_promotion_rejection",
                "decision": "retain_baseline",
                **evidence,
            },
            authentication_key=authentication_key,
            key_id=key_id,
        )
        write_json_atomic(rejection, root / "promotion" / "promotion_rejection.json")
        raise PromotionRejectedError(
            "Frozen promotion gates rejected the advanced system; 2014 audit remains sealed"
        )
    identity = _json(root / "nested" / "frozen_model_identity.json")
    payload = {
        "schema_version": 1,
        "certificate_type": "preaudit_promotion",
        "decision": "promote_to_one_time_audit",
        "model_identity": {
            name: identity[name]
            for name in (
                "final_fit_hash",
                "selected_candidate_hash",
                "selected_config_hash",
                "feature_schema_hash",
                "selection_procedure_hash",
                "development_frame_hash",
                "development_snapshot_manifest_sha256",
            )
        },
        **evidence,
        "decision_column_mapping": {
            "decision_draw_predictions": "positive_draw_predictions",
            "macro_f1_gain": "macro_f1_delta",
            "draw_recall_gain": "draw_recall_delta",
        },
        "claim_boundary": ("development_only_promotion_to_a_single_audit_not_external_validation"),
    }
    payload["evidence_bundle_sha256"] = payload_sha256(
        {
            "manifest_bindings": payload["manifest_bindings"],
            "baseline_identities": payload["baseline_identities"],
            "governance": payload["governance"],
            "audit_protocol_identity": payload["audit_protocol_identity"],
            "gate": payload["gate"],
        }
    )
    certificate = integrity_envelope(
        payload,
        authentication_key=authentication_key,
        key_id=key_id,
    )
    write_json_atomic(certificate, certificate_path)
    initialize_unopened_sentinel(
        root,
        certificate_path=certificate_path,
        audit_protocol_identity=payload["audit_protocol_identity"],
    )
    return certificate


def verify_promotion_certificate(
    certificate_path: Path,
    *,
    authentication_key: bytes | None,
    revalidate_artifacts: bool,
) -> dict[str, Any]:
    """Verify certificate integrity and recompute every bound evidence artifact."""
    if not revalidate_artifacts:
        raise ArtifactIntegrityError(
            "Promotion artifact revalidation cannot be disabled for an audit authorization"
        )
    path = certificate_path.expanduser().absolute()
    envelope = _json(path)
    return _verify_promotion_envelope(
        path,
        envelope,
        authentication_key=authentication_key,
    )


def _verify_promotion_envelope(
    path: Path,
    envelope: dict[str, Any],
    *,
    authentication_key: bytes | None,
) -> dict[str, Any]:
    """Verify one already-read certificate envelope without a second file read."""
    payload = verify_integrity_envelope(envelope, authentication_key=authentication_key)
    if (
        payload.get("certificate_type") != "preaudit_promotion"
        or payload.get("decision") != "promote_to_one_time_audit"
    ):
        raise ArtifactIntegrityError("Certificate is not a positive promotion decision")
    root = path.parent.parent
    governance = _governance_identity(root)
    expected_protocol = _audit_protocol_identity(governance)
    if (
        payload.get("governance") != governance
        or payload.get("audit_protocol_identity") != expected_protocol
    ):
        raise ArtifactIntegrityError("Promotion governance or audit-protocol binding changed")
    current = evaluate_promotion_artifacts(root)
    for name in (
        "manifest_bindings",
        "baseline_identities",
        "governance",
        "audit_protocol_identity",
        "gate",
    ):
        if payload.get(name) != current[name]:
            raise ArtifactIntegrityError(f"Promotion evidence changed after issuance: {name}")
    expected_bundle = payload_sha256(
        {
            name: current[name]
            for name in (
                "manifest_bindings",
                "baseline_identities",
                "governance",
                "audit_protocol_identity",
                "gate",
            )
        }
    )
    if payload.get("evidence_bundle_sha256") != expected_bundle:
        raise ArtifactIntegrityError("Promotion evidence-bundle digest mismatch")
    return payload


def _governance_identity(output_root: Path) -> dict[str, Any]:
    project = _project_root(output_root)
    lock_path = project / "plan" / "preaudit_amendment_lock.json"
    lock = _json(lock_path)
    required = {
        "original_protocol_path",
        "original_protocol_sha256",
        "deviation_record_path",
        "deviation_record_sha256",
        "runtime_reference_protocol_sha256",
        "development_snapshot_manifest_path",
        "development_snapshot_manifest_sha256",
        "formal_preaudit_reads_combined_raw",
        "physical_raw_no_read_claim",
        "maximum_development_year",
        "audit_year",
        "audit_artifacts_present_at_freeze",
    }
    if not required.issubset(lock):
        raise ArtifactIntegrityError("Pre-audit amendment lock is incomplete")
    original = _locked_project_file(project, str(lock["original_protocol_path"]))
    deviations = _locked_project_file(project, str(lock["deviation_record_path"]))
    snapshot_manifest = _locked_project_file(
        project, str(lock["development_snapshot_manifest_path"])
    )
    if file_sha256(original) != lock["original_protocol_sha256"]:
        raise ArtifactIntegrityError("Original preregistration hash changed")
    if file_sha256(deviations) != lock["deviation_record_sha256"]:
        raise ArtifactIntegrityError("Protocol-deviation record hash changed")
    if lock["runtime_reference_protocol_sha256"] != RUNTIME_REFERENCE_PROTOCOL_SHA256:
        raise ArtifactIntegrityError("Runtime reference protocol hash changed")
    if file_sha256(snapshot_manifest) != lock["development_snapshot_manifest_sha256"]:
        raise ArtifactIntegrityError("Development snapshot manifest hash changed")
    snapshot = load_development_snapshot(snapshot_manifest.parent)
    protocol_lock = _json(project / "plan" / "protocol_lock.json")
    if protocol_lock.get("protocol_sha256") != lock["original_protocol_sha256"]:
        raise ArtifactIntegrityError("Original and amendment protocol locks disagree")
    if (
        int(lock["maximum_development_year"]) != 2010
        or int(lock["audit_year"]) != 2014
        or lock["audit_artifacts_present_at_freeze"] is not False
        or lock["formal_preaudit_reads_combined_raw"] is not False
        or lock["physical_raw_no_read_claim"] is not False
        or snapshot.metadata["formal_preaudit_reads_combined_raw"] is not False
        or snapshot.metadata["physical_raw_no_read_claim"] is not False
    ):
        raise ArtifactIntegrityError("Pre-audit amendment boundary is invalid")
    return {
        "amendment_lock_sha256": file_sha256(lock_path),
        "original_protocol_sha256": lock["original_protocol_sha256"],
        "deviation_record_sha256": lock["deviation_record_sha256"],
        "runtime_reference_protocol_sha256": lock["runtime_reference_protocol_sha256"],
        "development_snapshot_manifest_sha256": lock["development_snapshot_manifest_sha256"],
        "development_snapshot_feature_schema_hash": snapshot.feature_schema_hash,
        "development_snapshot_frame_hash": snapshot.development_frame_hash,
        "development_snapshot_rows": int(len(snapshot.features)),
        "maximum_development_year": 2010,
        "audit_year": 2014,
        "audit_artifacts_present_at_freeze": False,
        "formal_preaudit_reads_combined_raw": False,
        "physical_raw_no_read_claim": False,
        "software_isolation_claim": (
            "2014 outcomes were excluded from advanced selection, calibration, "
            "ablation, and promotion; this is not a claim that the supplied raw "
            "tables were never parsed or that researchers were historically blind"
        ),
    }


def _baseline_freeze_binding(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = root / "promotion" / "baselines"
    manifest = directory / "baseline_freeze_manifest.json"
    required = (
        "official_outcome_logit.joblib",
        "independent_mirrored_poisson.joblib",
        "legacy_mirrored_dixon_coles.joblib",
        "baseline_freeze_metadata.json",
    )
    verified = verify_stage_manifest(manifest, required_paths=required)
    discovered = {
        path.name for path in directory.iterdir() if path.is_file() and path.name != manifest.name
    }
    if set(verified["declared_paths"]) != discovered:
        raise ArtifactIntegrityError(
            "Frozen audit baseline manifest is not an exact directory inventory"
        )
    metadata = _json(directory / "baseline_freeze_metadata.json")
    identity = _json(root / "nested" / "frozen_model_identity.json")
    if (
        metadata.get("fit_end_year") != 2010
        or metadata.get("training_row_count") != 772
        or metadata.get("development_frame_hash") != identity.get("development_frame_hash")
        or metadata.get("feature_schema_hash") != identity.get("feature_schema_hash")
        or metadata.get("development_snapshot_manifest_sha256")
        != identity.get("development_snapshot_manifest_sha256")
        or metadata.get("formal_preaudit_reads_combined_raw") is not False
        or metadata.get("audit_outcomes_used") is not False
    ):
        raise ArtifactIntegrityError("Frozen audit baseline lineage is invalid")
    models = metadata.get("models")
    expected = {
        "official_outcome_logit",
        "independent_mirrored_poisson",
        "legacy_mirrored_dixon_coles",
    }
    if not isinstance(models, dict) or set(models) != expected:
        raise ArtifactIntegrityError("Frozen audit baseline model set is incomplete")
    recomputed = fixed_audit_baseline_identities(
        development_frame_hash=identity["development_frame_hash"],
        feature_schema_hash=identity["feature_schema_hash"],
    )
    artifact_binding: dict[str, dict[str, str]] = {}
    for name in sorted(expected):
        actual = models[name]
        if not isinstance(actual, dict):
            raise ArtifactIntegrityError(f"Frozen audit baseline metadata is invalid: {name}")
        expected_path = f"promotion/baselines/{name}.joblib"
        artifact_path = directory / f"{name}.joblib"
        actual_core = {
            key: value
            for key, value in actual.items()
            if key not in {"artifact_path", "artifact_sha256"}
        }
        if (
            actual_core != recomputed[name]
            or actual.get("artifact_path") != expected_path
            or actual.get("artifact_sha256") != file_sha256(artifact_path)
        ):
            raise ArtifactIntegrityError(
                f"Frozen audit baseline identity/config/artifact drifted: {name}"
            )
        artifact_binding[name] = {
            "artifact_path": expected_path,
            "artifact_sha256": actual["artifact_sha256"],
        }
    if metadata.get("baseline_bundle_hash") != payload_sha256(recomputed) or metadata.get(
        "artifact_binding_hash"
    ) != payload_sha256(artifact_binding):
        raise ArtifactIntegrityError("Frozen audit baseline bundle binding is invalid")
    hashes = {
        name: value.get("model_identity_hash") if isinstance(value, dict) else None
        for name, value in models.items()
    }
    if any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
        raise ArtifactIntegrityError("Frozen audit baseline identities are malformed")
    binding = {
        "manifest_path": manifest.relative_to(root).as_posix(),
        "manifest_sha256": verified["manifest_sha256"],
        "artifact_count": verified["artifact_count"],
        "summary_sha256": file_sha256(directory / "baseline_freeze_metadata.json"),
    }
    return binding, hashes


def _audit_protocol_identity(governance: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "frozen_audit_protocol_hash": frozen_audit_protocol_hash(),
        "original_protocol_sha256": governance["original_protocol_sha256"],
        "deviation_record_sha256": governance["deviation_record_sha256"],
        "runtime_reference_protocol_sha256": governance[
            "runtime_reference_protocol_sha256"
        ],
        "development_snapshot_manifest_sha256": governance["development_snapshot_manifest_sha256"],
    }
    return {**payload, "protocol_bundle_sha256": payload_sha256(payload)}


def _verify_summary_boundaries(stage: str, summary: dict[str, Any]) -> None:
    if stage == "nested" and summary.get("formal_preaudit_reads_combined_raw") is not False:
        raise ArtifactIntegrityError("Nested checkpoint is not snapshot-isolated")
    if stage in {"evaluate", "ablation", "parameter", "stress"}:
        used = summary.get("audit_year_loaded", summary.get("audit_outcomes_used"))
        if used is not False:
            raise ArtifactIntegrityError(f"Stage {stage!r} is not development-only")
    if stage == "ablation" and summary.get("matrix_rows") != 126:
        raise ArtifactIntegrityError("Ablation checkpoint is not the exact 126-row matrix")
    if stage == "simulate":
        config = FROZEN_ADAPTIVE_CONFIG
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
        if any(summary.get(key) != value for key, value in exact_contract.items()):
            raise ArtifactIntegrityError("Simulation checkpoint changed the frozen contract")
        simulations = summary.get("simulations")
        if isinstance(simulations, bool) or not isinstance(simulations, int):
            raise ArtifactIntegrityError("Simulation count must be an exact integer")
        minimum = config.minimum_simulations
        maximum = config.maximum_simulations
        stopping_reason = summary.get("stopping_reason")
        valid_terminal = (
            summary.get("mcse_thresholds_satisfied") is True
            and stopping_reason == "all_mcse_thresholds_satisfied"
        ) or (
            simulations == maximum
            and summary.get("mcse_thresholds_satisfied") is False
            and stopping_reason == "maximum_simulations_reached"
        )
        if (
            summary.get("matches_per_simulation") != 104
            or not valid_terminal
            or simulations < minimum
            or simulations > maximum
            or summary.get("simulated_match_rows") != simulations * 104
            or summary.get("match_log_rows") != config.audit_log_simulations * 104
            or not isinstance(summary.get("stopping_ledger_rows"), int)
            or summary.get("stopping_ledger_rows", 0) <= 0
            or summary.get("batch_event_count_rows") != summary.get("stopping_ledger_rows") * 48 * 6
            or summary.get("continuous_sufficient_statistic_rows") != 49
        ):
            raise ArtifactIntegrityError("Simulation checkpoint violates adaptive MCSE/cardinality")
    if stage == "decompose" and summary.get("draw_shape") != [20, 3, 100]:
        raise ArtifactIntegrityError("Variance-decomposition checkpoint has the wrong shape")


def _verify_exact_stage_inventory(
    output_root: Path,
    stage: str,
    contract: _StageContract,
    declared_paths: list[str],
) -> None:
    """Reject undeclared stage outputs, including stale experiment artifacts."""
    directory = output_root / contract.directory
    if stage == "parameter":
        discovered = {
            path.name
            for path in directory.iterdir()
            if path.is_file()
            and path.name.startswith("parameter_")
            and not path.name.endswith("artifact_manifest.json")
        }
    elif stage == "decompose":
        discovered = {
            path.name
            for path in directory.iterdir()
            if path.is_file()
            and path.name.startswith("variance_")
            and not path.name.endswith("artifact_manifest.json")
        }
    else:
        discovered = {
            path.name
            for path in directory.iterdir()
            if path.is_file() and path.name != contract.manifest
        }
    if set(declared_paths) != discovered:
        missing = sorted(discovered.difference(declared_paths))
        stale = sorted(set(declared_paths).difference(discovered))
        raise ArtifactIntegrityError(
            f"Stage {stage!r} manifest is not an exact inventory; "
            f"undeclared={missing}, absent={stale}"
        )


def _verify_shared_uncertainty_inventory(
    output_root: Path,
    declared_data_paths: set[str],
) -> None:
    directory = output_root / "uncertainty"
    allowed = declared_data_paths | {
        _STAGES["parameter"].manifest,
        _STAGES["decompose"].manifest,
    }
    discovered = {path.name for path in directory.iterdir() if path.is_file()}
    if discovered != allowed:
        raise ArtifactIntegrityError(
            "Shared uncertainty directory contains unbound or missing artifacts; "
            f"undeclared={sorted(discovered - allowed)}, "
            f"absent={sorted(allowed - discovered)}"
        )


def _stage(name: str) -> _StageContract:
    try:
        return _STAGES[name]
    except KeyError as error:
        raise ValueError(f"Unknown pre-audit stage: {name}") from error


def _locked_project_file(project: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ArtifactIntegrityError(f"Unsafe governance path: {relative!r}")
    resolved = (project / candidate).resolve()
    try:
        resolved.relative_to(project.resolve())
    except ValueError as error:
        raise ArtifactIntegrityError(
            f"Governance path escapes the project root: {relative!r}"
        ) from error
    if not resolved.is_file():
        raise ArtifactIntegrityError(f"Governance artifact is missing: {relative!r}")
    return resolved


def _project_root(output_root: Path) -> Path:
    for candidate in (output_root, *output_root.parents):
        if (candidate / "plan" / "preaudit_amendment_lock.json").is_file():
            return candidate
    raise FileNotFoundError("Cannot locate the project pre-audit amendment lock")


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"Cannot read JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ArtifactIntegrityError(f"JSON artifact must contain an object: {path}")
    return payload


__all__ = [
    "checkpoint_status",
    "evaluate_promotion_artifacts",
    "issue_promotion_certificate",
    "verify_all_stage_checkpoints",
    "verify_promotion_certificate",
]
