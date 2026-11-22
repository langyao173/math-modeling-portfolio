"""One-time audit state machine with target-opening isolation."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

from .artifact_integrity import (
    ArtifactIntegrityError,
    build_stage_manifest,
    file_sha256,
    integrity_envelope,
    payload_sha256,
    verify_integrity_envelope,
    verify_stage_manifest,
    write_json_atomic,
)

FrozenT = TypeVar("FrozenT")
FeatureT = TypeVar("FeatureT")
TruthT = TypeVar("TruthT")
ResultT = TypeVar("ResultT")
BlindT = TypeVar("BlindT")

_REQUIRED_COMPONENTS = frozenset(
    {
        "advanced_predictor",
        "calibration",
        "decision",
        "official_outcome_logit",
        "independent_mirrored_poisson",
        "legacy_mirrored_dixon_coles",
    }
)


@dataclass(frozen=True)
class FrozenAuditPreparation(Generic[FrozenT]):
    """Opaque frozen objects plus their pre-target provenance."""

    value: FrozenT
    state_hash: str
    components: Mapping[str, bool]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PersistedAudit:
    """Paths and bounded summary written by the audit result persister."""

    relative_paths: tuple[str, ...]
    summary: Mapping[str, Any]


def initialize_unopened_sentinel(
    output_root: Path,
    *,
    certificate_path: Path,
    audit_protocol_identity: dict[str, Any],
) -> Path:
    """Create the single unopened sentinel after positive promotion."""
    root = output_root.resolve()
    audit_dir = root / "audit"
    sentinel = audit_dir / "unopened_sentinel.json"
    consumed = audit_dir / "consumed_unopened_sentinel.json"
    state_path = audit_dir / "audit_state.json"
    certificate_hash = file_sha256(certificate_path)
    if sentinel.is_file():
        if consumed.exists() or state_path.exists():
            raise ArtifactIntegrityError(
                "Audit control state is contradictory: unopened and opened records coexist"
            )
        payload = _verified_json_envelope(sentinel)
        if (
            payload.get("state") != "unopened"
            or payload.get("targets_opened") is not False
            or payload.get("open_attempts") != 0
            or payload.get("promotion_certificate_sha256") != certificate_hash
            or payload.get("audit_protocol_identity") != audit_protocol_identity
        ):
            raise ArtifactIntegrityError("Existing audit sentinel is malformed or stale")
        return sentinel
    if state_path.is_file():
        if not consumed.is_file():
            raise ArtifactIntegrityError(
                "Opened audit state is missing the consumed unopened sentinel"
            )
        state = _verified_json_envelope(state_path)
        consumed_payload = _verified_json_envelope(consumed)
        allowed_states = {
            "preparing_blind",
            "blind_predictions_persisted",
            "opening_truth",
            "truth_released",
            "complete",
            "failed_before_truth_open",
            "failed_after_truth_open",
        }
        truth_opened = state.get("targets_opened") is True
        if (
            state.get("state") not in allowed_states
            or state.get("open_attempts") != (1 if truth_opened else 0)
            or state.get("promotion_certificate_sha256") != certificate_hash
            or state.get("audit_protocol_identity") != audit_protocol_identity
            or consumed_payload.get("state") != "unopened"
            or consumed_payload.get("targets_opened") is not False
            or consumed_payload.get("open_attempts") != 0
            or consumed_payload.get("promotion_certificate_sha256") != certificate_hash
            or consumed_payload.get("audit_protocol_identity") != audit_protocol_identity
        ):
            raise ArtifactIntegrityError("Existing opened audit control state is invalid")
        return state_path
    if consumed.exists():
        raise ArtifactIntegrityError(
            "The audit sentinel was consumed without a durable state; target retry is forbidden"
        )
    if audit_dir.exists() and any(audit_dir.iterdir()):
        raise ArtifactIntegrityError(
            "Audit directory was not empty before unopened-sentinel initialization"
        )
    payload = {
        "schema_version": 1,
        "state": "unopened",
        "promotion_certificate": certificate_path.relative_to(root).as_posix(),
        "promotion_certificate_sha256": certificate_hash,
        "audit_protocol_identity": audit_protocol_identity,
        "targets_opened": False,
        "open_attempts": 0,
        "sentinel_scope": "post_certificate_formal_target_audit_open_attempt",
        "formal_preaudit_reads_combined_raw": False,
        "physical_raw_no_read_claim": False,
    }
    write_json_atomic(integrity_envelope(payload), sentinel)
    return sentinel


def run_one_time_audit(
    output_root: Path,
    *,
    freeze: Callable[[], FrozenAuditPreparation[FrozenT]],
    open_blind_features: Callable[[], FeatureT],
    prepare_blind: Callable[[FrozenT, FeatureT], BlindT],
    persist_blind: Callable[[BlindT, Path], PersistedAudit],
    open_audit_truth: Callable[[], TruthT],
    evaluate_persisted_blind: Callable[[Path, TruthT, str], ResultT],
    persist: Callable[[ResultT, Path], PersistedAudit],
    authentication_key: bytes | None = None,
) -> dict[str, Any]:
    """Seal all blind predictions, then open truth once and score sealed bytes."""
    from .promotion import verify_promotion_certificate

    root = output_root.resolve()
    certificate_path = root / "promotion" / "promotion_certificate.json"
    promotion = verify_promotion_certificate(
        certificate_path,
        authentication_key=authentication_key,
        revalidate_artifacts=True,
    )
    audit_dir = root / "audit"
    state_path = audit_dir / "audit_state.json"
    if state_path.is_file():
        state = _verified_json_envelope(state_path)
        if state.get("state") == "complete":
            return verify_cached_audit(root, authentication_key=authentication_key)
        raise ArtifactIntegrityError(
            f"Audit targets were already opened; non-repeatable state={state.get('state')!r}"
        )
    sentinel = audit_dir / "unopened_sentinel.json"
    sentinel_payload = _verified_json_envelope(sentinel)
    expected_certificate_hash = file_sha256(certificate_path)
    if (
        sentinel_payload.get("state") != "unopened"
        or sentinel_payload.get("targets_opened") is not False
        or sentinel_payload.get("open_attempts") != 0
        or sentinel_payload.get("promotion_certificate_sha256") != expected_certificate_hash
        or sentinel_payload.get("audit_protocol_identity")
        != promotion.get("audit_protocol_identity")
    ):
        raise ArtifactIntegrityError("Unopened audit sentinel is invalid")

    frozen = freeze()
    _validate_frozen_preparation(frozen)
    consumed = audit_dir / "consumed_unopened_sentinel.json"
    os.replace(sentinel, consumed)
    opening_payload = {
        **sentinel_payload,
        "state": "preparing_blind",
        "targets_opened": False,
        "open_attempts": 0,
        "blind_preparation_attempts": 1,
        "frozen_state_hash": frozen.state_hash,
        "frozen_components": dict(frozen.components),
        "frozen_metadata_hash": payload_sha256(dict(frozen.metadata)),
        "formal_source_commit": frozen.metadata["formal_source_commit"],
        "pre2014_freeze_manifest_sha256": frozen.metadata[
            "pre2014_freeze_manifest_sha256"
        ],
        "pre2014_freeze_source_commit": frozen.metadata["pre2014_freeze_source_commit"],
        "two_phase_blind_audit": True,
        "blind_predictions_persisted": False,
        "truth_released": False,
    }
    write_json_atomic(integrity_envelope(opening_payload), state_path)

    truth_opened = False
    truth_released = False
    blind_paths: tuple[str, ...] = ()
    blind_manifest_hash: str | None = None
    try:
        features = open_blind_features()
        blind = prepare_blind(frozen.value, features)
        blind_persisted = persist_blind(blind, audit_dir)
        blind_paths = _validate_persisted(blind_persisted)
        blind_generated_at = blind_persisted.summary.get("predictions_generated_at_utc")
        if not isinstance(blind_generated_at, str):
            raise ValueError("Blind prediction persistence requires a generation timestamp")
        parsed_generated_at = datetime.fromisoformat(blind_generated_at)
        if parsed_generated_at.tzinfo is None:
            raise ValueError("Blind prediction generation timestamp must include a timezone")
        blind_manifest_records = build_stage_manifest(audit_dir, blind_paths)
        blind_manifest_path = write_json_atomic(
            blind_manifest_records,
            audit_dir / "blind_prediction_manifest.json",
        )
        blind_manifest = verify_stage_manifest(
            blind_manifest_path,
            required_paths=blind_paths,
        )
        blind_manifest_hash = blind_manifest["manifest_sha256"]
        blind_state = {
            **opening_payload,
            "state": "blind_predictions_persisted",
            "blind_predictions_persisted": True,
            "blind_prediction_manifest_sha256": blind_manifest_hash,
            "blind_predictions_generated_at_utc": blind_generated_at,
        }
        write_json_atomic(integrity_envelope(blind_state), state_path)

        truth_opening_state = {
            **blind_state,
            "state": "opening_truth",
            "targets_opened": True,
            "open_attempts": 1,
        }
        write_json_atomic(integrity_envelope(truth_opening_state), state_path)
        truth_opened = True
        truth = open_audit_truth()
        truth_state = {
            **truth_opening_state,
            "state": "truth_released",
            "truth_released": True,
        }
        write_json_atomic(integrity_envelope(truth_state), state_path)
        truth_released = True
        assert blind_manifest_hash is not None
        result = evaluate_persisted_blind(audit_dir, truth, blind_manifest_hash)
        persisted = persist(result, audit_dir)
        result_paths = _validate_persisted(persisted)
        relative_paths = (
            *blind_paths,
            "blind_prediction_manifest.json",
            *result_paths,
        )
        if len(relative_paths) != len(set(relative_paths)):
            raise ValueError("Blind and scored audit artifacts must have unique paths")
        manifest_records = build_stage_manifest(audit_dir, relative_paths)
        manifest_path = write_json_atomic(
            manifest_records, audit_dir / "audit_artifact_manifest.json"
        )
        verified_manifest = verify_stage_manifest(manifest_path, required_paths=relative_paths)
        result_payload = {
            "schema_version": 1,
            "certificate_type": "one_time_audit_result",
            "promotion_certificate_sha256": expected_certificate_hash,
            "audit_protocol_identity": promotion["audit_protocol_identity"],
            "frozen_state_hash": frozen.state_hash,
            "artifact_manifest_sha256": verified_manifest["manifest_sha256"],
            "artifact_count": verified_manifest["artifact_count"],
            "summary": dict(persisted.summary),
            "targets_opened_once": True,
            "selection_reopened": False,
            "two_phase_blind_audit": True,
            "blind_prediction_manifest_sha256": blind_manifest_hash,
            "truth_released_after_blind_persistence": True,
            "scored_from_manifest_bound_blind_predictions": True,
            "blind_predictions_generated_at_utc": blind_generated_at,
            "formal_source_commit": frozen.metadata["formal_source_commit"],
            "pre2014_freeze_manifest_sha256": frozen.metadata[
                "pre2014_freeze_manifest_sha256"
            ],
        }
        result_certificate = integrity_envelope(
            result_payload, authentication_key=authentication_key
        )
        result_certificate_path = write_json_atomic(
            result_certificate, audit_dir / "audit_result_certificate.json"
        )
        complete = {
            **opening_payload,
            "state": "complete",
            "audit_result_certificate_sha256": file_sha256(result_certificate_path),
            "audit_artifact_manifest_sha256": verified_manifest["manifest_sha256"],
            "targets_opened": True,
            "open_attempts": 1,
            "blind_predictions_persisted": True,
            "blind_prediction_manifest_sha256": blind_manifest_hash,
            "blind_predictions_generated_at_utc": blind_generated_at,
            "truth_released": True,
        }
        write_json_atomic(integrity_envelope(complete), state_path)
        return dict(persisted.summary)
    except Exception as error:
        failed = {
            **opening_payload,
            "state": (
                "failed_after_truth_open" if truth_opened else "failed_before_truth_open"
            ),
            "targets_opened": truth_opened,
            "open_attempts": 1 if truth_opened else 0,
            "failure_type": type(error).__name__,
            "repeat_target_read_forbidden": True,
            "blind_predictions_persisted": bool(blind_paths),
            "blind_prediction_manifest_sha256": blind_manifest_hash,
            "truth_released": truth_released,
        }
        write_json_atomic(integrity_envelope(failed), state_path)
        raise


def verify_cached_audit(
    output_root: Path,
    *,
    authentication_key: bytes | None = None,
) -> dict[str, Any]:
    """Verify cached results without invoking any target loader."""
    from worldcup_forecasting.experiments.audit_execution import (
        load_persisted_blind_audit_predictions,
    )

    from .promotion import verify_promotion_certificate

    root = output_root.resolve()
    audit_dir = root / "audit"
    state = _verified_json_envelope(audit_dir / "audit_state.json")
    if state.get("state") != "complete" or state.get("open_attempts") != 1:
        raise ArtifactIntegrityError("Cached audit state is not complete and one-time")
    certificate_path = audit_dir / "audit_result_certificate.json"
    if file_sha256(certificate_path) != state.get("audit_result_certificate_sha256"):
        raise ArtifactIntegrityError("Cached audit result certificate hash changed")
    result_payload = verify_integrity_envelope(
        _json(certificate_path), authentication_key=authentication_key
    )
    promotion_certificate = root / "promotion" / "promotion_certificate.json"
    promotion_payload = verify_promotion_certificate(
        promotion_certificate,
        authentication_key=authentication_key,
        revalidate_artifacts=True,
    )
    promotion_hash = file_sha256(promotion_certificate)
    manifest_path = audit_dir / "audit_artifact_manifest.json"
    manifest = verify_stage_manifest(manifest_path)
    if result_payload.get("two_phase_blind_audit") is not True:
        raise ArtifactIntegrityError("Cached audit is not an authorized two-phase audit")
    required_blind_paths = (
        "2014_predictions_blind.parquet",
        "2014_joint_predictions_blind.parquet",
        "2014_predictions_blind_metadata.json",
    )
    blind_manifest = verify_stage_manifest(
        audit_dir / "blind_prediction_manifest.json",
        required_paths=required_blind_paths,
    )
    if set(blind_manifest["declared_paths"]) != set(required_blind_paths):
        raise ArtifactIntegrityError("Cached blind manifest has an unauthorized artifact set")
    blind_manifest_hash = blind_manifest["manifest_sha256"]
    try:
        loaded_blind = load_persisted_blind_audit_predictions(
            audit_dir,
            expected_manifest_sha256=blind_manifest_hash,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        raise ArtifactIntegrityError(
            f"Cached blind prediction semantics are invalid: {error}"
        ) from error
    if (
        state.get("promotion_certificate_sha256") != promotion_hash
        or result_payload.get("promotion_certificate_sha256") != promotion_hash
        or result_payload.get("audit_protocol_identity") != state.get("audit_protocol_identity")
        or result_payload.get("audit_protocol_identity")
        != promotion_payload.get("audit_protocol_identity")
        or result_payload.get("certificate_type") != "one_time_audit_result"
        or manifest["manifest_sha256"] != state.get("audit_artifact_manifest_sha256")
        or manifest["manifest_sha256"] != result_payload.get("artifact_manifest_sha256")
        or manifest["artifact_count"] != result_payload.get("artifact_count")
        or result_payload.get("frozen_state_hash") != state.get("frozen_state_hash")
        or result_payload.get("targets_opened_once") is not True
        or result_payload.get("selection_reopened") is not False
        or result_payload.get("scored_from_manifest_bound_blind_predictions") is not True
        or result_payload.get("blind_predictions_generated_at_utc")
        != state.get("blind_predictions_generated_at_utc")
        or loaded_blind.metadata.get("predictions_generated_at_utc")
        != state.get("blind_predictions_generated_at_utc")
        or result_payload.get("formal_source_commit") != state.get("formal_source_commit")
        or loaded_blind.metadata.get("formal_source_commit")
        != state.get("formal_source_commit")
        or result_payload.get("pre2014_freeze_manifest_sha256")
        != state.get("pre2014_freeze_manifest_sha256")
        or loaded_blind.metadata.get("pre2014_freeze_manifest_sha256")
        != state.get("pre2014_freeze_manifest_sha256")
        or state.get("blind_predictions_persisted") is not True
        or state.get("truth_released") is not True
        or result_payload.get("truth_released_after_blind_persistence") is not True
        or state.get("blind_prediction_manifest_sha256") != blind_manifest_hash
        or result_payload.get("blind_prediction_manifest_sha256") != blind_manifest_hash
    ):
        raise ArtifactIntegrityError("Cached audit provenance binding changed")
    summary = result_payload.get("summary")
    if not isinstance(summary, dict):
        raise ArtifactIntegrityError("Cached audit summary is malformed")
    return dict(summary)


def _validate_frozen_preparation(preparation: FrozenAuditPreparation[Any]) -> None:
    if not isinstance(preparation, FrozenAuditPreparation):
        raise TypeError("freeze must return FrozenAuditPreparation")
    if not isinstance(preparation.state_hash, str) or len(preparation.state_hash) != 64:
        raise ValueError("Frozen audit state_hash must be a SHA-256 digest")
    components = dict(preparation.components)
    if set(components) != _REQUIRED_COMPONENTS or not all(
        isinstance(value, bool) and value for value in components.values()
    ):
        raise ValueError("All six predictor/calibration/decision components must be frozen")
    if not isinstance(preparation.metadata, Mapping) or not preparation.metadata:
        raise ValueError("Frozen audit preparation requires provenance metadata")
    required_metadata = {
        "formal_source_commit": 40,
        "pre2014_freeze_manifest_sha256": 64,
        "pre2014_freeze_source_commit": 40,
    }
    for key, length in required_metadata.items():
        value = preparation.metadata.get(key)
        if not isinstance(value, str) or len(value) != length:
            raise ValueError(f"Frozen audit metadata has invalid {key}")


def _validate_persisted(persisted: PersistedAudit) -> tuple[str, ...]:
    if not isinstance(persisted, PersistedAudit):
        raise TypeError("persist must return PersistedAudit")
    paths = tuple(str(value) for value in persisted.relative_paths)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Persisted audit paths must be nonempty and unique")
    if not isinstance(persisted.summary, Mapping) or not persisted.summary:
        raise ValueError("Persisted audit summary must be nonempty")
    return paths


def _verified_json_envelope(path: Path) -> dict[str, Any]:
    return verify_integrity_envelope(_json(path))


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"Cannot read audit control file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ArtifactIntegrityError(f"Audit control file is not a JSON object: {path}")
    return payload


__all__ = [
    "FrozenAuditPreparation",
    "PersistedAudit",
    "initialize_unopened_sentinel",
    "run_one_time_audit",
    "verify_cached_audit",
]
