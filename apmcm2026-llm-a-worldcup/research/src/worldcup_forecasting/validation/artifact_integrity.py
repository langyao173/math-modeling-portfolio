"""Canonical hashing and fail-closed verification for research artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any


class ArtifactIntegrityError(RuntimeError):
    """Raised when an artifact, manifest, or integrity envelope is invalid."""


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize a JSON-compatible value in the one accepted canonical form."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    """Return the SHA-256 digest of canonical JSON."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integrity_envelope(
    payload: dict[str, Any],
    *,
    authentication_key: bytes | None = None,
    key_id: str | None = None,
) -> dict[str, Any]:
    """Bind a payload to a deterministic digest and optional external HMAC.

    SHA-256 provides independently recomputable integrity, not signer authenticity.
    HMAC authentication is added only when the caller supplies an external key.
    """
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Integrity payload must be a nonempty dictionary")
    encoded = canonical_json_bytes(payload)
    envelope: dict[str, Any] = {
        "payload": payload,
        "integrity": {
            "algorithm": "sha256-canonical-json",
            "payload_sha256": hashlib.sha256(encoded).hexdigest(),
            "authenticity_claimed": authentication_key is not None,
        },
    }
    if authentication_key is not None:
        if not isinstance(authentication_key, bytes) or len(authentication_key) < 16:
            raise ValueError("External HMAC key must contain at least 16 bytes")
        resolved_key_id = key_id or hashlib.sha256(authentication_key).hexdigest()[:16]
        envelope["authentication"] = {
            "algorithm": "hmac-sha256",
            "key_id": str(resolved_key_id),
            "signature": hmac.new(authentication_key, encoded, hashlib.sha256).hexdigest(),
        }
    return envelope


def verify_integrity_envelope(
    envelope: dict[str, Any],
    *,
    authentication_key: bytes | None = None,
) -> dict[str, Any]:
    """Verify canonical digest and, when present, the external HMAC."""
    if not isinstance(envelope, dict) or set(envelope).difference(
        {"payload", "integrity", "authentication"}
    ):
        raise ArtifactIntegrityError("Integrity envelope has an unsupported schema")
    payload = envelope.get("payload")
    integrity = envelope.get("integrity")
    if not isinstance(payload, dict) or not isinstance(integrity, dict):
        raise ArtifactIntegrityError("Integrity envelope is missing payload or digest")
    encoded = canonical_json_bytes(payload)
    observed = hashlib.sha256(encoded).hexdigest()
    if integrity.get("algorithm") != "sha256-canonical-json" or not hmac.compare_digest(
        observed, str(integrity.get("payload_sha256", ""))
    ):
        raise ArtifactIntegrityError("Canonical payload digest mismatch")
    authentication = envelope.get("authentication")
    claimed = bool(integrity.get("authenticity_claimed"))
    if authentication is None:
        if claimed:
            raise ArtifactIntegrityError("Envelope claims authentication without a signature")
        if authentication_key is not None:
            raise ArtifactIntegrityError(
                "External HMAC verification key was supplied for an unsigned envelope"
            )
        return dict(payload)
    if not isinstance(authentication, dict) or authentication.get("algorithm") != "hmac-sha256":
        raise ArtifactIntegrityError("Unsupported certificate authentication")
    if authentication_key is None:
        raise ArtifactIntegrityError("External HMAC key is required for this certificate")
    expected = hmac.new(authentication_key, encoded, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, str(authentication.get("signature", ""))):
        raise ArtifactIntegrityError("Certificate HMAC verification failed")
    if not claimed:
        raise ArtifactIntegrityError("Signed envelope does not declare authentication")
    return dict(payload)


def write_json_atomic(payload: Any, destination: str | Path) -> Path:
    """Write JSON atomically and durably within the destination directory."""
    path = Path(destination).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = stream.name
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise
    return path


def build_stage_manifest(
    directory: str | Path,
    relative_paths: Iterable[str],
) -> list[dict[str, Any]]:
    """Build an exact size/hash inventory for explicitly named stage outputs."""
    root = Path(directory).expanduser().absolute()
    _reject_reparse_point(root, "stage directory")
    normalized = sorted(set(str(item) for item in relative_paths))
    if not normalized:
        raise ValueError("A stage manifest must contain at least one artifact")
    records: list[dict[str, Any]] = []
    for relative in normalized:
        artifact = _safe_path(root, relative)
        if not artifact.is_file():
            raise FileNotFoundError(f"Stage artifact is missing: {relative}")
        records.append(
            {
                "path": relative,
                "bytes": int(artifact.stat().st_size),
                "sha256": file_sha256(artifact),
            }
        )
    return records


def verify_stage_manifest(
    manifest_path: str | Path,
    *,
    required_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Verify every declared stage artifact and required-path inclusion."""
    path = Path(manifest_path).expanduser().absolute()
    if not path.is_file():
        raise ArtifactIntegrityError(f"Stage manifest does not exist: {path}")
    _reject_reparse_point(path.parent, "stage directory")
    _reject_reparse_point(path, "stage manifest")
    try:
        manifest_bytes = path.read_bytes()
        records = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"Stage manifest is unreadable: {error}") from error
    if not isinstance(records, list) or not records:
        raise ArtifactIntegrityError("Stage manifest must be a nonempty record list")
    root = path.parent
    declared: set[str] = set()
    artifact_records: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for raw in records:
        if not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256"}:
            raise ArtifactIntegrityError("Stage manifest record schema is invalid")
        relative = str(raw["path"])
        if relative in declared:
            raise ArtifactIntegrityError(f"Duplicate stage manifest path: {relative}")
        declared.add(relative)
        artifact = _safe_path(root, relative)
        expected_bytes = raw["bytes"]
        expected_hash = str(raw["sha256"])
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or len(expected_hash) != 64
        ):
            raise ArtifactIntegrityError(f"Malformed stage manifest record: {relative}")
        if not artifact.is_file():
            raise ArtifactIntegrityError(f"Declared stage artifact is missing: {relative}")
        if artifact.stat().st_size != expected_bytes or file_sha256(artifact) != expected_hash:
            raise ArtifactIntegrityError(f"Stage artifact hash mismatch: {relative}")
        artifact_records[relative] = {
            "bytes": expected_bytes,
            "sha256": expected_hash,
        }
        total_bytes += expected_bytes
    missing = sorted(set(required_paths).difference(declared))
    if missing:
        raise ArtifactIntegrityError(f"Stage manifest omits required artifacts: {missing}")
    return {
        "manifest_path": str(path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "artifact_count": len(records),
        "total_bytes": total_bytes,
        "declared_paths": sorted(declared),
        "artifact_records": artifact_records,
    }


def read_verified_artifact_bytes(
    directory: str | Path,
    relative_path: str,
    verified_manifest: dict[str, Any],
) -> bytes:
    """Read and rehash the exact bytes that a caller will subsequently consume."""
    root = Path(directory).expanduser().absolute()
    artifact = _safe_path(root, relative_path)
    records = verified_manifest.get("artifact_records")
    record = records.get(relative_path) if isinstance(records, dict) else None
    if not isinstance(record, dict) or set(record) != {"bytes", "sha256"}:
        raise ArtifactIntegrityError(
            f"Verified manifest does not bind requested artifact: {relative_path}"
        )
    try:
        with artifact.open("rb") as stream:
            payload = stream.read()
    except OSError as error:
        raise ArtifactIntegrityError(
            f"Cannot read verified artifact {relative_path}: {error}"
        ) from error
    if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ArtifactIntegrityError(
            f"Artifact changed between manifest verification and consumption: {relative_path}"
        )
    return payload


def _safe_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ArtifactIntegrityError(f"Unsafe stage artifact path: {relative!r}")
    root = root.expanduser().absolute()
    _reject_reparse_point(root, "stage directory")
    candidate = root.joinpath(*pure.parts).absolute()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ArtifactIntegrityError(
            f"Stage artifact escapes its directory: {relative!r}"
        ) from error
    current = root
    for part in pure.parts:
        current = current / part
        if current.exists():
            _reject_reparse_point(current, f"stage artifact {relative!r}")
    return candidate


def _reject_reparse_point(path: Path, label: str) -> None:
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError as error:
        raise ArtifactIntegrityError(f"Cannot inspect {label}: {path}") from error
    is_junction = getattr(path, "is_junction", None)
    if (
        path.is_symlink()
        or bool(callable(is_junction) and is_junction())
        or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    ):
        raise ArtifactIntegrityError(f"Reparse-point {label} is forbidden: {path}")


__all__ = [
    "ArtifactIntegrityError",
    "build_stage_manifest",
    "canonical_json_bytes",
    "file_sha256",
    "integrity_envelope",
    "payload_sha256",
    "read_verified_artifact_bytes",
    "verify_integrity_envelope",
    "verify_stage_manifest",
    "write_json_atomic",
]
