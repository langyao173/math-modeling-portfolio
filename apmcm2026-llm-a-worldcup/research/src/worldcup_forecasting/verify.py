"""Independent verification for a completed output manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from worldcup_forecasting.application.runtime import (
    REQUIRED_ARTIFACTS,
    RESEARCH_REQUIRED_ARTIFACTS,
)
from worldcup_forecasting.reporting.manifest import file_sha256
from worldcup_forecasting.uncertainty.certificates import validate_prediction_certificate


class ManifestVerificationError(RuntimeError):
    """Raised when a declared output cannot be reproduced byte-for-byte."""


@dataclass(frozen=True)
class _ArtifactRecord:
    path: str
    bytes: int
    sha256: str


def verify_manifest(
    manifest_path: str | Path,
    required_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Verify inventory, size, and SHA-256 for every declared artifact.

    The verifier intentionally trusts neither the manifest inventory nor filesystem
    discovery. Both sets must agree exactly, and every path must stay beneath the
    manifest directory.
    """
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise ManifestVerificationError(f"Manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestVerificationError(f"Manifest is unreadable: {error}") from error
    if payload.get("status") != "complete":
        raise ManifestVerificationError("Manifest status is not complete")

    expected_required = _expected_required_paths(payload, required_paths)
    manifest_required = payload.get("required_paths")
    declared_paths = payload.get("relative_paths")
    raw_records = payload.get("artifacts")
    if (
        not isinstance(manifest_required, list)
        or any(not isinstance(item, str) for item in manifest_required)
        or not isinstance(declared_paths, list)
        or not isinstance(raw_records, list)
    ):
        raise ManifestVerificationError("Manifest inventory is malformed")
    if len(manifest_required) != len(set(manifest_required)):
        raise ManifestVerificationError("Manifest required_paths contains duplicates")
    if set(manifest_required) != set(expected_required):
        raise ManifestVerificationError("Manifest required-artifact contract mismatch")
    if len(declared_paths) != len(set(declared_paths)):
        raise ManifestVerificationError("Manifest relative_paths contains duplicates")
    missing_required = sorted(set(expected_required).difference(declared_paths))
    if missing_required:
        raise ManifestVerificationError(
            f"Manifest omits required artifacts: {missing_required}"
        )

    records = [_parse_record(record) for record in raw_records]
    record_paths = [record.path for record in records]
    if len(record_paths) != len(set(record_paths)):
        raise ManifestVerificationError("Manifest artifacts contains duplicate paths")
    if sorted(declared_paths) != sorted(record_paths):
        raise ManifestVerificationError("Manifest inventories mismatch")

    output_dir = path.parent
    discovered = sorted(
        item.relative_to(output_dir).as_posix()
        for item in output_dir.rglob("*")
        if item.is_file() and item != path
    )
    if sorted(declared_paths) != discovered:
        raise ManifestVerificationError("Filesystem and manifest inventories mismatch")

    total_bytes = 0
    for record in records:
        artifact = _safe_artifact_path(output_dir, record.path)
        actual_bytes = artifact.stat().st_size
        actual_hash = file_sha256(artifact)
        if actual_bytes != record.bytes or actual_hash != record.sha256:
            raise ManifestVerificationError(f"Artifact mismatch: {record.path}")
        total_bytes += actual_bytes
    verified_certificates = _verify_certificates(
        output_dir,
        declared_paths,
        payload.get("resolved_config_sha256"),
    )
    return {
        "status": "verified",
        "manifest_path": str(path),
        "verified_artifacts": len(records),
        "verified_certificates": verified_certificates,
        "verified_required_artifacts": len(expected_required),
        "total_bytes": total_bytes,
    }


def _verify_certificates(
    output_dir: Path,
    declared_paths: list[str],
    manifest_config_hash: Any,
) -> int:
    certificate_paths = sorted(
        relative
        for relative in declared_paths
        if relative.startswith("certificates/") and relative.endswith(".json")
    )
    for relative in certificate_paths:
        path = _safe_artifact_path(output_dir, relative)
        try:
            certificate = json.loads(path.read_text(encoding="utf-8"))
            validate_prediction_certificate(certificate)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ManifestVerificationError(
                f"Prediction certificate is invalid: {relative}: {error}"
            ) from error
        if (
            not isinstance(manifest_config_hash, str)
            or certificate["config_hash"] != manifest_config_hash
        ):
            raise ManifestVerificationError(
                f"Certificate config hash mismatch: {relative}"
            )
        model_relative = str(certificate["model_artifact"])
        if model_relative not in declared_paths:
            raise ManifestVerificationError(
                f"Certificate model artifact is not in manifest: {model_relative}"
            )
        model_path = _safe_artifact_path(output_dir, model_relative)
        if file_sha256(model_path) != certificate["model_hash"]:
            raise ManifestVerificationError(
                f"Certificate model hash mismatch: {relative}"
            )
        calibration_relative = certificate["calibration_artifact"]
        if calibration_relative is not None:
            if calibration_relative not in declared_paths:
                raise ManifestVerificationError(
                    "Certificate calibration artifact is not in manifest: "
                    f"{calibration_relative}"
                )
            _safe_artifact_path(output_dir, str(calibration_relative))
    return len(certificate_paths)


def _expected_required_paths(
    payload: dict[str, Any],
    override: set[str] | None,
) -> set[str]:
    if override is not None:
        return set(override)
    profile = payload.get("artifact_profile")
    if profile not in {"official", "research"}:
        raise ManifestVerificationError(
            f"Missing or unknown artifact_profile: {profile!r}"
        )
    run = payload.get("run")
    if not isinstance(run, dict):
        raise ManifestVerificationError("Manifest run metadata is missing")
    config_name = run.get("config_name")
    track = run.get("track")
    if config_name not in {"official", "final", "research"} or not isinstance(
        track, str
    ):
        raise ManifestVerificationError("Manifest run profile metadata is malformed")
    expected_profile = "research" if config_name == "research" else "official"
    track_is_research = "research" in track.casefold()
    if profile != expected_profile or track_is_research != (profile == "research"):
        raise ManifestVerificationError(
            "artifact_profile disagrees with run config_name/track"
        )
    expected = set(REQUIRED_ARTIFACTS)
    if profile == "research":
        expected.update(RESEARCH_REQUIRED_ARTIFACTS)
    return expected


def _parse_record(value: Any) -> _ArtifactRecord:
    if not isinstance(value, dict):
        raise ManifestVerificationError("Artifact record is not an object")
    try:
        record = _ArtifactRecord(
            path=str(value["path"]),
            bytes=int(value["bytes"]),
            sha256=str(value["sha256"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestVerificationError(f"Malformed artifact record: {value!r}") from error
    if record.bytes < 0 or len(record.sha256) != 64:
        raise ManifestVerificationError(f"Malformed artifact metadata: {record.path}")
    return record


def _safe_artifact_path(output_dir: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or relative in {"", "manifest.json"}:
        raise ManifestVerificationError(f"Unsafe artifact path: {relative!r}")
    artifact = (output_dir / Path(*posix.parts)).resolve()
    try:
        artifact.relative_to(output_dir)
    except ValueError as error:
        raise ManifestVerificationError(f"Artifact escapes output directory: {relative}") from error
    if not artifact.is_file():
        raise ManifestVerificationError(f"Artifact is missing: {relative}")
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Verify the default or explicitly supplied output manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        default="outputs/manifest.json",
        help="path to manifest.json (default: outputs/manifest.json)",
    )
    arguments = parser.parse_args(argv)
    summary = verify_manifest(arguments.manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
