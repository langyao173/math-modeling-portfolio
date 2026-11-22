"""Strict post-archive verifier for the TAOLIN-DC-SIM submission package.

The verifier treats the ZIP as the release artifact.  It checks the source
directory and manifest, validates the archive byte-for-byte, extracts it with
path traversal defenses, runs the embedded validator in an isolated Python
process, and compares the independently recomputed results with the packaged
metrics and industrial evidence.

Default outputs are written next to the final ZIP:

    deliverables/SUBMISSION_VERIFICATION.json
    deliverables/SUBMISSION_VERIFICATION.md
    deliverables/SHA256SUMS.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_MANDATORY_CHECKS = frozenset(
    {
        "manifest_integrity",
        "source_package_unmanifested_files",
        "zip_safety",
        "zip_payload_consistency",
        "cleanroom_extraction",
        "package_path_cache_scan",
        "package_secret_scan",
        "package_text_scan_coverage",
        "cleanroom_validator_execution",
        "cross_artifact_consistency",
        "qubo_matrix_consistency",
    }
)
STAGING_MANDATORY_CHECKS = frozenset(
    {
        "staging_manifest_integrity",
        "staging_document_media_structure",
        "staging_zip_safety",
        "staging_zip_membership",
        "staging_nested_archive_scan",
        "staging_trusted_content",
        "staging_package_path_cache_scan",
        "staging_package_secret_scan",
        "staging_package_text_scan_coverage",
        "staging_postscan_hashes",
    }
)
PORTAL_MANDATORY_CHECKS = STAGING_MANDATORY_CHECKS | frozenset(
    {"portal_archive_integrity"}
)
PORTAL_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
PORTAL_MAX_BYTES = 50_000_000
PORTAL_FILENAME_RE = re.compile(
    r"能源调度_(YX26\d{5})_([^_]{2,60})_提交材料\.zip\Z"
)
FINAL_SOLUTION_FIELDS = [
    "job_id", "action_id", "site", "start_slot", "end_slot", "mode",
    "gpu_req", "power_mw", "energy_mwh", "action_cost", "is_dummy",
]
FULL_SOLUTION_FIELDS = [
    "selection_source", "action_id", "job_id", "site", "start_slot", "end_slot",
    "mode", "gpu_req", "power_mw", "energy_mwh", "energy_cost_usd",
    "sla_cost_usd", "extra_mode_cost_usd", "action_cost", "is_dummy",
    "gpu_vector", "power_vector_mw", "conflict_keys", "source",
]
ACTION_FIELDS = [
    "action_id", "job_id", "site", "start_slot", "end_slot", "mode", "gpu_req",
    "power_mw", "energy_mwh", "energy_cost_usd", "sla_cost_usd",
    "extra_mode_cost_usd", "action_cost", "is_dummy", "gpu_vector",
    "power_vector_mw", "conflict_keys", "source",
]
SITE_FIELDS = [
    "slot", "lmp_usd_mwh", "carbon_kgco2_mwh", "penalty_short_usd_mwh",
    "flex_call_prob", "renewable_mw",
]
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".html",
    ".ini",
    ".json",
    ".log",
    ".lock",
    ".md",
    ".py",
    ".rst",
    ".tex",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
NESTED_ARCHIVE_SUFFIXES = {
    ".7z",
    ".bz2",
    ".docx",
    ".gz",
    ".jar",
    ".rar",
    ".tar",
    ".tgz",
    ".whl",
    ".xz",
    ".zip",
}
FORBIDDEN_PATH_PARTS = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "site-packages",
    "venv",
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}
PLACEHOLDER_SECRET_MARKERS = {
    "***",
    "<redacted>",
    "changeme",
    "example",
    "placeholder",
    "redacted",
    "your-",
    "your_",
}


@dataclass
class CheckResult:
    check_id: str
    status: str
    summary: str
    evidence: dict[str, Any]


class Recorder:
    """Collect bounded, JSON-serializable verification evidence."""

    def __init__(self) -> None:
        self.checks: list[CheckResult] = []

    def add(
        self,
        check_id: str,
        status: str,
        summary: str,
        **evidence: Any,
    ) -> None:
        if status not in {"pass", "fail", "warn", "skip"}:
            raise ValueError(f"Unsupported check status: {status}")
        self.checks.append(CheckResult(check_id, status, summary, evidence))

    def passed(self, check_id: str, summary: str, **evidence: Any) -> None:
        self.add(check_id, "pass", summary, **evidence)

    def failed(self, check_id: str, summary: str, **evidence: Any) -> None:
        self.add(check_id, "fail", summary, **evidence)

    def warned(self, check_id: str, summary: str, **evidence: Any) -> None:
        self.add(check_id, "warn", summary, **evidence)

    def skipped(self, check_id: str, summary: str, **evidence: Any) -> None:
        self.add(check_id, "skip", summary, **evidence)

    @property
    def ready(self) -> bool:
        return bool(self.checks) and all(check.status == "pass" for check in self.checks)

    def ready_for(self, mandatory_checks: frozenset[str]) -> bool:
        counts = Counter(check.check_id for check in self.checks)
        return self.ready and counts == Counter({check_id: 1 for check_id in mandatory_checks})

    def counts(self) -> dict[str, int]:
        return dict(Counter(check.status for check in self.checks))

    def has(self, check_id: str) -> bool:
        return any(check.check_id == check_id for check in self.checks)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def path_label(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def is_link_like(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", lambda: False)
    if path.is_symlink() or is_junction():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def validate_unlinked_path_components(path: Path, *, label: str) -> None:
    """Inspect the original lexical path before any resolve() erases link provenance."""
    absolute = path.absolute()
    parts = absolute.parts
    if not parts:
        raise ValueError(f"{label} path is empty")
    cursor = Path(parts[0])
    if is_link_like(cursor):
        raise ValueError(f"{label} path traverses a link or reparse point: {cursor}")
    for part in parts[1:]:
        cursor = cursor / part
        if is_link_like(cursor):
            raise ValueError(f"{label} path traverses a link or reparse point: {cursor}")


def bounded(items: Iterable[Any], limit: int = 25) -> list[Any]:
    return list(items)[:limit]


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return payload


def read_csv_rows(
    path: Path,
    expected_fields: list[str] | None = None,
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path.name}")
        if expected_fields is not None and (
            reader.fieldnames != expected_fields
            or len(set(reader.fieldnames)) != len(expected_fields)
        ):
            raise ValueError(f"CSV has an unexpected or duplicate header: {path.name}")
        return list(reader)


def parse_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def csv_action_rows_match(selected: dict[str, str], canonical: dict[str, str]) -> bool:
    for column in ("job_id", "site", "mode", "source"):
        if selected.get(column) != canonical.get(column):
            return False
    for column in ("start_slot", "end_slot", "gpu_req"):
        try:
            if int(selected.get(column, "")) != int(canonical.get(column, "")):
                return False
        except ValueError:
            return False
    try:
        if parse_bool(selected.get("is_dummy")) != parse_bool(canonical.get("is_dummy")):
            return False
    except ValueError:
        return False
    for column in (
        "power_mw",
        "energy_mwh",
        "energy_cost_usd",
        "sla_cost_usd",
        "extra_mode_cost_usd",
        "action_cost",
    ):
        if not close_number(selected.get(column), canonical.get(column), tolerance=1e-12):
            return False
    for column in ("gpu_vector", "power_vector_mw", "conflict_keys"):
        try:
            if json.loads(selected.get(column, "")) != json.loads(canonical.get(column, "")):
                return False
        except (TypeError, json.JSONDecodeError):
            return False
    return True


def csv_compact_rows_match(compact: dict[str, str], full: dict[str, str]) -> bool:
    for column in ("job_id", "action_id", "site", "mode"):
        if compact.get(column) != full.get(column):
            return False
    try:
        for column in ("start_slot", "end_slot", "gpu_req"):
            if int(compact.get(column, "")) != int(full.get(column, "")):
                return False
        if parse_bool(compact.get("is_dummy")) != parse_bool(full.get("is_dummy")):
            return False
    except (TypeError, ValueError):
        return False
    for column in ("power_mw", "energy_mwh", "action_cost"):
        if not close_number(compact.get(column), full.get(column), tolerance=1e-12):
            return False
    return True


def close_number(left: Any, right: Any, tolerance: float = 5e-7) -> bool:
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=1e-10,
            abs_tol=tolerance,
        )
    except (TypeError, ValueError):
        return False


def validate_relative_path(value: str, *, allow_directory: bool = False) -> str | None:
    if not value:
        return "empty path"
    if "\x00" in value:
        return "NUL byte"
    if "\\" in value:
        return "backslash separator"
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        return "absolute or drive-qualified path"
    if any(ord(character) < 32 for character in value):
        return "control character"
    normalized_value = value[:-1] if allow_directory and value.endswith("/") else value
    if not normalized_value:
        return "empty normalized path"
    parts = PurePosixPath(normalized_value).parts
    if any(part in {"", ".", ".."} for part in parts):
        return "empty, dot, or traversal component"
    for part in parts:
        if ":" in part:
            return "colon or NTFS alternate-data-stream component"
        if part.endswith((".", " ")):
            return "Win32-trimmed trailing dot or space"
        if part.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
            return "reserved Windows device-name component"
    if PurePosixPath(normalized_value).as_posix() != normalized_value:
        return "non-canonical path"
    if len(normalized_value) > 240:
        return "path exceeds 240 characters"
    return None


def verify_manifest(package_dir: Path, recorder: Recorder) -> dict[str, Any] | None:
    manifest_path = package_dir / "MANIFEST.json"
    if not package_dir.is_dir() or not manifest_path.is_file():
        recorder.failed(
            "manifest_presence",
            "Package directory or MANIFEST.json is missing.",
            package_directory_exists=package_dir.is_dir(),
            manifest_exists=manifest_path.is_file(),
        )
        return None

    try:
        payload = load_json(manifest_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        recorder.failed(
            "manifest_parse",
            "MANIFEST.json is not a valid JSON object.",
            error_type=type(error).__name__,
        )
        return None

    raw_records = payload.get("files")
    declared_count = payload.get("file_count")
    issues: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    casefold_paths: dict[str, str] = {}
    if not isinstance(raw_records, list):
        issues.append("files is not a list")
        raw_records = []

    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            issues.append(f"record {index} is not an object")
            continue
        relative = raw_record.get("path")
        byte_count = raw_record.get("bytes")
        digest = raw_record.get("sha256")
        if not isinstance(relative, str):
            issues.append(f"record {index} has a non-string path")
            continue
        path_error = validate_relative_path(relative)
        if path_error:
            issues.append(f"{relative}: {path_error}")
        if relative == "MANIFEST.json":
            issues.append("MANIFEST.json must not hash itself")
        if relative in records:
            issues.append(f"duplicate manifest path: {relative}")
        folded = relative.casefold()
        if folded in casefold_paths and casefold_paths[folded] != relative:
            issues.append(f"case-fold collision: {casefold_paths[folded]} / {relative}")
        casefold_paths[folded] = relative
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            issues.append(f"{relative}: invalid byte count")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            issues.append(f"{relative}: invalid SHA-256")
        records[relative] = raw_record

    if declared_count != len(raw_records):
        issues.append(f"file_count={declared_count!r}, records={len(raw_records)}")

    actual = {
        path.relative_to(package_dir).as_posix(): path
        for path in package_dir.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    missing = sorted(set(records) - set(actual))
    extra = sorted(set(actual) - set(records))
    hash_mismatches: list[str] = []
    size_mismatches: list[str] = []
    for relative in sorted(set(records) & set(actual)):
        record = records[relative]
        path = actual[relative]
        if isinstance(record.get("bytes"), int) and path.stat().st_size != record["bytes"]:
            size_mismatches.append(relative)
        if isinstance(record.get("sha256"), str) and sha256_file(path) != record["sha256"]:
            hash_mismatches.append(relative)

    integrity_failed = bool(issues or missing or hash_mismatches or size_mismatches)
    if integrity_failed:
        recorder.failed(
            "manifest_integrity",
            "Manifest structure or source-package hashes are inconsistent.",
            issue_count=len(issues),
            issues=bounded(issues),
            missing_count=len(missing),
            missing=bounded(missing),
            hash_mismatch_count=len(hash_mismatches),
            hash_mismatches=bounded(hash_mismatches),
            size_mismatch_count=len(size_mismatches),
            size_mismatches=bounded(size_mismatches),
        )
    else:
        recorder.passed(
            "manifest_integrity",
            "Manifest structure, membership, sizes, and SHA-256 hashes match the package directory.",
            file_count=len(records),
            payload_bytes=sum(path.stat().st_size for path in actual.values()),
            manifest_sha256=sha256_file(manifest_path),
        )
    if extra:
        recorder.warned(
            "source_package_unmanifested_files",
            "The working package directory contains files that are not part of the manifest or final ZIP.",
            extra_count=len(extra),
            extra=bounded(extra),
        )
    else:
        recorder.passed(
            "source_package_unmanifested_files",
            "The working package directory has no unmanifested files.",
            extra_count=0,
        )

    return {
        "payload": payload,
        "records": records,
        "actual": actual,
        "manifest_path": manifest_path,
        "valid": not integrity_failed,
    }


def verify_zip(
    zip_path: Path,
    package_dir: Path,
    manifest_state: dict[str, Any] | None,
    recorder: Recorder,
    args: argparse.Namespace,
    *,
    check_payload_consistency: bool = True,
) -> dict[str, Any] | None:
    if not zip_path.is_file():
        recorder.failed("zip_presence", "Final ZIP is missing.", zip_exists=False)
        return None

    archive_sha256 = sha256_file(zip_path)
    archive_bytes = zip_path.stat().st_size
    if archive_bytes > args.max_archive_bytes:
        recorder.failed(
            "zip_safety",
            "ZIP exceeds the compressed archive-size limit; decompression was not attempted.",
            archive_bytes=archive_bytes,
            max_archive_bytes=args.max_archive_bytes,
            crc_test_skipped=True,
        )
        return {
            "archive_sha256": archive_sha256,
            "archive_bytes": archive_bytes,
            "member_count": 0,
            "uncompressed_bytes": 0,
            "safe": False,
            "prefix": package_dir.name,
        }
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            infos = archive.infolist()
            file_infos = [info for info in infos if not info.is_dir()]
            safety_issues: list[str] = []
            seen_names: set[str] = set()
            seen_casefold: dict[str, str] = {}
            total_uncompressed = 0
            total_compressed = 0
            max_ratio = 0.0

            if len(file_infos) > args.max_members:
                safety_issues.append("archive member count exceeds limit")

            for info in infos:
                name = info.filename
                path_error = validate_relative_path(name, allow_directory=info.is_dir())
                if path_error:
                    safety_issues.append(f"{name!r}: {path_error}")
                if name in seen_names:
                    safety_issues.append(f"duplicate member: {name}")
                seen_names.add(name)
                folded = name.rstrip("/").casefold()
                if folded in seen_casefold and seen_casefold[folded] != name.rstrip("/"):
                    safety_issues.append(
                        f"case-fold collision: {seen_casefold[folded]} / {name.rstrip('/')}"
                    )
                seen_casefold[folded] = name.rstrip("/")
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    safety_issues.append(f"symbolic link member: {name}")
                if mode and not info.is_dir() and stat.S_IFMT(mode) not in {0, stat.S_IFREG}:
                    safety_issues.append(f"non-regular member: {name}")
                if info.flag_bits & 0x1:
                    safety_issues.append(f"encrypted member: {name}")
                if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                    safety_issues.append(f"unsupported compression method: {name}")
                if info.is_dir():
                    continue
                total_uncompressed += info.file_size
                total_compressed += info.compress_size
                ratio = info.file_size / max(info.compress_size, 1)
                max_ratio = max(max_ratio, ratio)
                if info.file_size > args.max_member_bytes:
                    safety_issues.append(f"member exceeds size limit: {name}")
                if ratio > args.max_member_compression_ratio:
                    safety_issues.append(f"member compression ratio exceeds limit: {name}")

            total_ratio = total_uncompressed / max(total_compressed, 1)
            if total_uncompressed > args.max_uncompressed_bytes:
                safety_issues.append("archive uncompressed size exceeds limit")
            if total_ratio > args.max_total_compression_ratio:
                safety_issues.append("archive aggregate compression ratio exceeds limit")

            bad_member: str | None = None
            crc_error: str | None = None
            crc_skipped = bool(safety_issues)
            if not safety_issues:
                try:
                    bad_member = archive.testzip()
                except (RuntimeError, OSError, zipfile.BadZipFile) as error:
                    crc_error = type(error).__name__
            if bad_member:
                safety_issues.append(f"CRC failure: {bad_member}")
            if crc_error:
                safety_issues.append(f"CRC test error: {crc_error}")

            if safety_issues:
                recorder.failed(
                    "zip_safety",
                    "ZIP safety, portability, CRC, or expansion limits failed.",
                    issue_count=len(safety_issues),
                    issues=bounded(safety_issues),
                    member_count=len(file_infos),
                    uncompressed_bytes=total_uncompressed,
                    compressed_bytes=total_compressed,
                    aggregate_compression_ratio=round(total_ratio, 3),
                    maximum_member_compression_ratio=round(max_ratio, 3),
                    crc_test_skipped=crc_skipped,
                )
            else:
                recorder.passed(
                    "zip_safety",
                    "ZIP members are canonical, portable, unencrypted, CRC-clean, and within expansion limits.",
                    member_count=len(file_infos),
                    uncompressed_bytes=total_uncompressed,
                    compressed_bytes=total_compressed,
                    aggregate_compression_ratio=round(total_ratio, 3),
                    maximum_member_compression_ratio=round(max_ratio, 3),
                    crc_test_skipped=False,
                )

            if safety_issues:
                return {
                    "archive_sha256": archive_sha256,
                    "archive_bytes": archive_bytes,
                    "member_count": len(file_infos),
                    "uncompressed_bytes": total_uncompressed,
                    "safe": False,
                    "prefix": package_dir.name,
                }

            if not check_payload_consistency:
                return {
                    "archive_sha256": archive_sha256,
                    "archive_bytes": archive_bytes,
                    "member_count": len(file_infos),
                    "uncompressed_bytes": total_uncompressed,
                    "safe": not safety_issues,
                    "prefix": package_dir.name,
                }

            consistency_issues: list[str] = []
            if manifest_state is None:
                consistency_issues.append("manifest was unavailable")
            else:
                prefix = package_dir.name
                expected_names = {
                    f"{prefix}/{relative}" for relative in manifest_state["records"]
                }
                expected_names.add(f"{prefix}/MANIFEST.json")
                actual_names = set(info.filename for info in file_infos)
                missing = sorted(expected_names - actual_names)
                extra = sorted(actual_names - expected_names)
                if missing:
                    consistency_issues.append(f"missing archive members: {missing[:10]}")
                if extra:
                    consistency_issues.append(f"extra archive members: {extra[:10]}")

                if len(actual_names) != len(file_infos):
                    consistency_issues.append("archive contains duplicate file names")
                manifest_member = f"{prefix}/MANIFEST.json"
                if manifest_member in actual_names:
                    with archive.open(manifest_member, "r") as handle:
                        archive_manifest_sha = sha256_stream(handle)
                    local_manifest_sha = sha256_file(manifest_state["manifest_path"])
                    if archive_manifest_sha != local_manifest_sha:
                        consistency_issues.append("archived MANIFEST.json differs from source")

                if not missing:
                    for relative, record in manifest_state["records"].items():
                        member_name = f"{prefix}/{relative}"
                        info = archive.getinfo(member_name)
                        if info.file_size != record.get("bytes"):
                            consistency_issues.append(f"archive size mismatch: {relative}")
                            continue
                        with archive.open(info, "r") as handle:
                            digest = sha256_stream(handle)
                        if digest != record.get("sha256"):
                            consistency_issues.append(f"archive SHA-256 mismatch: {relative}")

            if consistency_issues:
                recorder.failed(
                    "zip_payload_consistency",
                    "Archive members do not exactly reproduce the manifest-bound package.",
                    issue_count=len(consistency_issues),
                    issues=bounded(consistency_issues),
                    archive_sha256=archive_sha256,
                )
            else:
                recorder.passed(
                    "zip_payload_consistency",
                    "Archive membership and every payload hash match the manifest-bound package.",
                    archive_sha256=archive_sha256,
                    archive_bytes=archive_bytes,
                    member_count=len(file_infos),
                )

            return {
                "archive_sha256": archive_sha256,
                "archive_bytes": archive_bytes,
                "member_count": len(file_infos),
                "uncompressed_bytes": total_uncompressed,
                "safe": not safety_issues and not consistency_issues,
                "prefix": package_dir.name,
            }
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        recorder.failed(
            "zip_open",
            "Final ZIP could not be opened.",
            error_type=type(error).__name__,
            archive_sha256=archive_sha256,
        )
        return None


def is_placeholder_secret(value: str) -> bool:
    normalized = value.strip().lower()
    exact_placeholders = {
        "api_key",
        "environment variable",
        "from environment",
        "none",
        "not set",
        "null",
        "password",
        "secret",
        "token",
    }
    return bool(
        not normalized
        or normalized in exact_placeholders
        or normalized.startswith(("${", "$env:", "os.environ", "process.env"))
        or any(marker in normalized for marker in PLACEHOLDER_SECRET_MARKERS)
    )


def read_text_candidate(path: Path, max_scan_bytes: int) -> tuple[bool, str | None, bool]:
    """Return (is_text, decoded_text, coverage_skipped) without lossy decoding."""
    known_text = path.suffix.casefold() in TEXT_SUFFIXES
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            sample = handle.read(min(size, 8192))
    except OSError:
        return known_text, None, known_text
    if not sample and size == 0:
        return known_text, "" if known_text else None, False
    encoding = "utf-8-sig"
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"
    elif b"\x00" in sample:
        return known_text, None, known_text
    try:
        decoded_sample = sample.decode(encoding, errors="strict")
    except UnicodeError:
        return known_text, None, known_text
    controls = sum(
        ord(character) < 32 and character not in "\t\r\n\f"
        for character in decoded_sample
    )
    if decoded_sample and controls / len(decoded_sample) > 0.01:
        return known_text, None, known_text
    if size > max_scan_bytes:
        return True, None, True
    try:
        return True, path.read_text(encoding=encoding, errors="strict"), False
    except (OSError, UnicodeError):
        return True, None, True


def binary_secret_hits(path: Path, max_scan_bytes: int) -> list[str]:
    """Search bounded binary bytes for high-confidence credential signatures."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size <= max_scan_bytes:
                payload = handle.read()
            else:
                window = min(max_scan_bytes // 2, 2 * 1024 * 1024)
                prefix = handle.read(window)
                handle.seek(max(0, size - window))
                payload = prefix + handle.read(window)
    except OSError:
        return []
    patterns = {
        "private_key": re.compile(br"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "aws_access_key": re.compile(br"\bAKIA[0-9A-Z]{16}\b"),
        "github_token": re.compile(br"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        "long_sk_token": re.compile(br"\bsk-[A-Za-z0-9_-]{16,}\b"),
        "credential_url": re.compile(br"https?://[^\s/:@]+:[^\s/@]+@", re.IGNORECASE),
    }
    return [category for category, pattern in patterns.items() if pattern.search(payload)]


def scan_package(package_dir: Path, recorder: Recorder, max_scan_bytes: int) -> None:
    path_hits: list[dict[str, Any]] = []
    secret_hits: list[dict[str, Any]] = []
    skipped_text: list[str] = []
    path_patterns = {
        "windows_absolute_path": re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"),
        "unix_home_path": re.compile(r"/(?:Users|home)/[^/\s]+/"),
        "file_uri": re.compile(r"\bfile://", re.IGNORECASE),
        "cache_dependency": re.compile(
            r"(?:^|[\\/])(?:__pycache__|\.cache|\.venv|venv|site-packages)(?:[\\/]|$)",
            re.IGNORECASE,
        ),
    }
    direct_secret_patterns = {
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        "long_sk_token": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
        "credential_url": re.compile(r"https?://[^\s/:@]+:[^\s/@]+@", re.IGNORECASE),
    }
    assignment_pattern = re.compile(
        r"(?<![A-Za-z0-9_])[\"']?"
        r"(?:api[_-]?key|access[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
        r"authorization|credential|password|passwd|session[_-]?token|token|secret)"
        r"[\"']?\s*[:=]\s*(?:[\"'](?P<quoted>[^\"']{8,})[\"']|"
        r"(?P<bare>[A-Za-z0-9][A-Za-z0-9_.-]{15,}))",
        re.IGNORECASE,
    )

    for path in sorted(package_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package_dir).as_posix()
        parts = {part.casefold() for part in PurePosixPath(relative).parts}
        if parts & {part.casefold() for part in FORBIDDEN_PATH_PARTS}:
            path_hits.append({"file": relative, "category": "cache_or_provenance_path"})
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES or path.name in {".DS_Store", "Thumbs.db"}:
            path_hits.append({"file": relative, "category": "generated_or_system_file"})
        is_text, text, coverage_skipped = read_text_candidate(path, max_scan_bytes)
        if not is_text:
            for category in binary_secret_hits(path, max_scan_bytes):
                secret_hits.append({"file": relative, "category": category})
            continue
        if coverage_skipped or text is None:
            for category in binary_secret_hits(path, max_scan_bytes):
                secret_hits.append({"file": relative, "category": category})
            skipped_text.append(relative)
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for category, pattern in path_patterns.items():
                if pattern.search(line):
                    path_hits.append(
                        {"file": relative, "line": line_number, "category": category}
                    )
            for category, pattern in direct_secret_patterns.items():
                if pattern.search(line):
                    secret_hits.append(
                        {"file": relative, "line": line_number, "category": category}
                    )
            for match in assignment_pattern.finditer(line):
                value = match.group("quoted") or match.group("bare") or ""
                if not is_placeholder_secret(value):
                    secret_hits.append(
                        {
                            "file": relative,
                            "line": line_number,
                            "category": "literal_secret_assignment",
                        }
                    )

    if path_hits:
        recorder.failed(
            "package_path_cache_scan",
            "Package contains machine-specific paths, caches, VCS metadata, or generated files.",
            hit_count=len(path_hits),
            hits=bounded(path_hits),
        )
    else:
        recorder.passed(
            "package_path_cache_scan",
            "No absolute home paths, file URIs, caches, VCS metadata, or generated bytecode were detected.",
            hit_count=0,
        )

    if secret_hits:
        recorder.failed(
            "package_secret_scan",
            "Package contains credential-like material; values are intentionally not reported.",
            hit_count=len(secret_hits),
            hits=bounded(secret_hits),
        )
    else:
        recorder.passed(
            "package_secret_scan",
            "No private keys, provider token shapes, credential URLs, or literal secret assignments were detected.",
            hit_count=0,
        )

    if skipped_text:
        recorder.warned(
            "package_text_scan_coverage",
            "Some text-like files exceeded the configured scan limit or could not be read.",
            skipped_count=len(skipped_text),
            skipped=bounded(skipped_text),
            per_file_limit_bytes=max_scan_bytes,
        )
    else:
        recorder.passed(
            "package_text_scan_coverage",
            "Every decodable text package file was covered by the content scan.",
            per_file_limit_bytes=max_scan_bytes,
        )


def safe_extract(zip_path: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    extracted_files = 0
    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            path_error = validate_relative_path(info.filename, allow_directory=info.is_dir())
            if path_error:
                raise ValueError(f"Unsafe member rejected during extraction: {path_error}")
            target = destination.joinpath(*PurePosixPath(info.filename.rstrip("/")).parts)
            if not is_within(target, destination_root):
                raise ValueError("Archive member escaped the extraction root")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted_files += 1
    return extracted_files


def isolated_environment() -> dict[str, str]:
    keep = {
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "WINDIR",
    }
    environment = {key: value for key, value in os.environ.items() if key.upper() in keep}
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    return environment


def run_embedded_validator(
    package_dir: Path,
    recorder: Recorder,
    timeout_seconds: int,
) -> dict[str, Any] | None:
    validator = package_dir / "validate_final_solution.py"
    recomputed_path = package_dir / "VALIDATION_RECOMPUTED.json"
    if not validator.is_file() or not recomputed_path.is_file():
        recorder.failed(
            "cleanroom_validator_presence",
            "The extracted package lacks the embedded validator or packaged recomputation.",
            validator_exists=validator.is_file(),
            recomputed_exists=recomputed_path.is_file(),
        )
        return None

    try:
        packaged_recomputed = load_json(recomputed_path)
        protected_hashes = {
            path.relative_to(package_dir).as_posix(): sha256_file(path)
            for path in package_dir.rglob("*")
            if path.is_file() and path != recomputed_path
        }
        completed = subprocess.run(
            [sys.executable, "-I", "-B", validator.name],
            cwd=package_dir,
            env=isolated_environment(),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        regenerated = load_json(recomputed_path)
        after_hashes = {
            path.relative_to(package_dir).as_posix(): sha256_file(path)
            for path in package_dir.rglob("*")
            if path.is_file() and path != recomputed_path
        }
    except subprocess.TimeoutExpired:
        recorder.failed(
            "cleanroom_validator_execution",
            "Embedded validator timed out in the isolated extraction.",
            timeout_seconds=timeout_seconds,
        )
        return None
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        recorder.failed(
            "cleanroom_validator_execution",
            "Embedded validator could not be executed or parsed.",
            error_type=type(error).__name__,
        )
        return None

    issues: list[str] = []
    if completed.returncode != 0:
        issues.append(f"validator exit code {completed.returncode}")
    if regenerated.get("status") != "pass":
        issues.append("regenerated status is not pass")
    regenerated_checks = regenerated.get("checks")
    if not isinstance(regenerated_checks, dict) or not all(
        value is True for value in regenerated_checks.values()
    ):
        issues.append("one or more regenerated checks are not true")
    if packaged_recomputed != regenerated:
        issues.append("packaged and clean-room recomputations differ")
    if protected_hashes != after_hashes:
        issues.append("validator changed files other than VALIDATION_RECOMPUTED.json")

    execution_evidence = {
        "exit_code": completed.returncode,
        "stdout_bytes": len(completed.stdout.encode("utf-8")),
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_bytes": len(completed.stderr.encode("utf-8")),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
        "regenerated_status": regenerated.get("status"),
        "regenerated_check_count": len(regenerated_checks or {}),
        "timeout_seconds": timeout_seconds,
    }
    if issues:
        recorder.failed(
            "cleanroom_validator_execution",
            "The isolated embedded validation did not exactly reproduce the packaged result.",
            issues=issues,
            **execution_evidence,
        )
    else:
        recorder.passed(
            "cleanroom_validator_execution",
            "The archived validator ran in isolation and exactly reproduced every packaged validation result.",
            **execution_evidence,
        )
    return {
        "packaged": packaged_recomputed,
        "regenerated": regenerated,
        "valid": not issues,
    }


def verify_cross_artifact_consistency(
    package_dir: Path,
    validator_state: dict[str, Any] | None,
    recorder: Recorder,
) -> dict[str, Any]:
    required = {
        "metrics": package_dir / "FINAL_METRICS.json",
        "validation": package_dir / "VALIDATION.json",
        "solution": package_dir / "FINAL_SOLUTION.csv",
        "full_solution": package_dir / "FULL_SELECTED_ACTIONS.csv",
        "actions": package_dir / "validation_data" / "actions_for_validation.csv",
        "site_timeseries": package_dir / "validation_data" / "site_timeseries_for_validation.csv",
        "metrics_table": package_dir / "artifacts" / "tables" / "final_metrics.csv",
        "industrial_summary": package_dir / "artifacts" / "industrial_ising_evidence_summary.json",
        "oracle_runs": package_dir / "artifacts" / "solver_runs" / "oracle_replacement_summary.csv",
        "cloud_runs": package_dir / "artifacts" / "solver_runs" / "cloud_raw_runs.csv",
        "local_runs": package_dir / "artifacts" / "solver_runs" / "local_solver_runs.csv",
        "cloud_availability": package_dir / "artifacts" / "validation" / "cloud_availability.json",
        "legacy_cleanroom": package_dir / "artifacts" / "validation" / "clean_room_validation.json",
    }
    missing = sorted(name for name, path in required.items() if not path.is_file())
    if missing:
        recorder.failed(
            "cross_artifact_consistency",
            "Required final, validation, or evidence artifacts are missing.",
            missing=missing,
        )
        return {}
    if validator_state is None:
        recorder.failed(
            "cross_artifact_consistency",
            "Cross-artifact checks require a successful clean-room recomputation payload.",
            missing_cleanroom_state=True,
        )
        return {}

    issues: list[str] = []
    summary: dict[str, Any] = {}
    try:
        metrics = load_json(required["metrics"])
        validation = load_json(required["validation"])
        recomputed = validator_state["regenerated"]
        final_rows = read_csv_rows(required["solution"], FINAL_SOLUTION_FIELDS)
        full_rows = read_csv_rows(required["full_solution"], FULL_SOLUTION_FIELDS)
        action_rows = read_csv_rows(required["actions"], ACTION_FIELDS)
        site_rows = read_csv_rows(required["site_timeseries"], SITE_FIELDS)
        metric_rows = read_csv_rows(required["metrics_table"])
        industrial = load_json(required["industrial_summary"])
        oracle_rows = read_csv_rows(required["oracle_runs"])
        cloud_rows = read_csv_rows(required["cloud_runs"])
        local_rows = read_csv_rows(required["local_runs"])
        cloud_availability = load_json(required["cloud_availability"])
        legacy_cleanroom = load_json(required["legacy_cleanroom"])

        if validation.get("status") != "pass" or not all(
            value is True for value in (validation.get("checks") or {}).values()
        ):
            issues.append("VALIDATION.json is not an all-pass result")
        if validator_state["packaged"] != recomputed:
            issues.append("packaged and regenerated recomputation payloads differ")

        selected_jobs = int(metrics["selected_jobs"])
        selected_variables = int(metrics["selected_action_variables"])
        if not final_rows or any(list(row) != FINAL_SOLUTION_FIELDS for row in final_rows):
            issues.append("FINAL_SOLUTION schema is not exact")
        if not full_rows or any(list(row) != FULL_SOLUTION_FIELDS for row in full_rows):
            issues.append("FULL_SELECTED_ACTIONS schema is not exact")
        final_pair_list = [(row.get("job_id"), row.get("action_id")) for row in final_rows]
        full_pair_list = [(row.get("job_id"), row.get("action_id")) for row in full_rows]
        final_pairs = set(final_pair_list)
        full_pairs = set(full_pair_list)
        full_by_pair = {pair: row for pair, row in zip(full_pair_list, full_rows)}
        compact_binding_mismatches = sorted(
            str(row.get("action_id"))
            for pair, row in zip(final_pair_list, final_rows)
            if pair not in full_by_pair or not csv_compact_rows_match(row, full_by_pair[pair])
        )
        action_id_list = [row.get("action_id") for row in action_rows]
        action_ids = set(action_id_list)
        if len(action_ids) != len(action_id_list):
            issues.append("validation actions contain duplicate action IDs")
        canonical_by_id = {row.get("action_id"): row for row in action_rows}
        binding_mismatches = sorted(
            str(row.get("action_id"))
            for row in full_rows
            if row.get("action_id") not in canonical_by_id
            or not csv_action_rows_match(row, canonical_by_id[row.get("action_id")])
        )
        if len(final_rows) != selected_jobs or len(full_rows) != selected_jobs:
            issues.append("solution row count differs from FINAL_METRICS selected_jobs")
        if len({row.get("job_id") for row in final_rows}) != selected_jobs:
            issues.append("FINAL_SOLUTION job IDs are not unique and complete")
        if (
            len(final_pair_list) != len(final_pairs)
            or len(full_pair_list) != len(full_pairs)
            or final_pairs != full_pairs
        ):
            issues.append("FINAL_SOLUTION and FULL_SELECTED_ACTIONS pairs differ")
        if compact_binding_mismatches:
            issues.append(
                "FINAL_SOLUTION fields differ from FULL_SELECTED_ACTIONS: "
                + ", ".join(compact_binding_mismatches[:10])
            )
        if any(action_id not in action_ids for _, action_id in final_pairs):
            issues.append("a selected action is absent from validation actions")
        if binding_mismatches:
            issues.append(
                "selected rows differ from canonical action rows: "
                + ", ".join(binding_mismatches[:10])
            )
        if len(action_rows) != selected_variables:
            issues.append("validation action count differs from selected_action_variables")

        dummy_count = 0
        invalid_boolean_rows: list[str] = []
        for label, rows in (("FINAL_SOLUTION", final_rows), ("FULL_SELECTED_ACTIONS", full_rows)):
            for row_number, row in enumerate(rows, start=2):
                try:
                    parsed = parse_bool(row.get("is_dummy"))
                except ValueError:
                    invalid_boolean_rows.append(f"{label}:{row_number}")
                    continue
                if label == "FULL_SELECTED_ACTIONS":
                    dummy_count += int(parsed)
        if invalid_boolean_rows:
            issues.append(
                "solution contains invalid boolean values: "
                + ", ".join(invalid_boolean_rows[:10])
            )
        mode_counts = dict(Counter(str(row.get("mode")) for row in full_rows))
        action_cost = sum(float(row["action_cost"]) for row in full_rows)
        if {row.get("selection_source") for row in full_rows} != {str(metrics["source"])}:
            issues.append("selection_source differs from FINAL_METRICS source")
        if dummy_count != int(metrics["dummy_actions"]):
            issues.append("dummy count differs between selected actions and metrics")
        if mode_counts != {str(key): int(value) for key, value in metrics["mode_counts"].items()}:
            issues.append("mode counts differ between selected actions and metrics")
        if not close_number(action_cost, metrics["true_action_cost_usd"]):
            issues.append("selected action-cost sum differs from metrics")

        horizon = len(site_rows)
        aggregate_load = [0.0] * horizon
        for row in full_rows:
            vector = [float(value) for value in json.loads(row["power_vector_mw"])]
            if len(vector) != horizon:
                issues.append("a selected power vector has the wrong horizon")
                continue
            if not all(math.isfinite(value) for value in vector):
                issues.append("a selected power vector contains a non-finite value")
                continue
            for index, value in enumerate(vector):
                aggregate_load[index] += value
        if aggregate_load:
            if not close_number(max(aggregate_load), metrics["peak_it_mw"]):
                issues.append("recomputed peak load differs from metrics")
            if not close_number(sum(aggregate_load) / horizon, metrics["mean_it_mw"]):
                issues.append("recomputed mean load differs from metrics")
            if not close_number(sum(aggregate_load), metrics["energy_mwh"]):
                issues.append("recomputed one-hour-slot energy differs from metrics")

        scalar_pairs = [
            (validation.get("row_count"), selected_jobs, "validation row count"),
            (validation.get("unique_jobs"), selected_jobs, "validation unique jobs"),
            (validation.get("selected_action_count"), selected_variables, "validation action count"),
            (validation.get("dummy_actions"), metrics.get("dummy_actions"), "validation dummy count"),
            (validation.get("one_hot_violations"), metrics.get("one_hot_violations"), "validation one-hot count"),
            (recomputed.get("row_count"), selected_jobs, "recomputed row count"),
            (recomputed.get("unique_jobs"), selected_jobs, "recomputed unique jobs"),
            (recomputed.get("dummy_actions"), metrics.get("dummy_actions"), "recomputed dummy count"),
        ]
        for left, right, label in scalar_pairs:
            if left != right:
                issues.append(f"{label} mismatch")
        if not close_number(
            recomputed.get("expected_recourse_cost_usd"),
            metrics.get("expected_recourse_cost_usd"),
        ):
            issues.append("expected recourse differs between recomputation and metrics")
        if not close_number(
            recomputed.get("cvar95_recourse_cost_usd"),
            metrics.get("cvar95_recourse_cost_usd"),
        ):
            issues.append("CVaR95 differs between recomputation and metrics")

        if len(metric_rows) != 1:
            issues.append("final_metrics.csv must contain exactly one row")
        else:
            table = metric_rows[0]
            table_pairs = [
                (table.get("jobs"), selected_jobs, "table jobs"),
                (table.get("spins"), selected_variables, "table spins"),
                (table.get("dummy_actions"), metrics.get("dummy_actions"), "table dummy count"),
                (table.get("one_hot_violations"), metrics.get("one_hot_violations"), "table one-hot count"),
            ]
            for left, right, label in table_pairs:
                try:
                    equal = int(float(str(left))) == int(right)
                except (TypeError, ValueError):
                    equal = False
                if not equal:
                    issues.append(f"{label} mismatch")
            if not close_number(table.get("expected_recourse_cost_usd"), metrics.get("expected_recourse_cost_usd")):
                issues.append("metrics table expected recourse mismatch")
            if not close_number(table.get("cvar95_recourse_cost_usd"), metrics.get("cvar95_recourse_cost_usd")):
                issues.append("metrics table CVaR95 mismatch")

        actual_cloud_rows = sum(
            str(row.get("oracle", "")).casefold().startswith("cloud") for row in oracle_rows
        )
        actual_failed_rows = sum(
            str(row.get("status", "")).casefold() != "success" for row in oracle_rows
        )
        industrial_pairs = [
            (industrial.get("oracle_rows"), len(oracle_rows), "industrial oracle rows"),
            (industrial.get("cloud_rows"), actual_cloud_rows, "industrial cloud rows"),
            (industrial.get("failed_rows"), actual_failed_rows, "industrial failed rows"),
            (len(cloud_rows), actual_cloud_rows, "cloud raw-run rows"),
            (len(local_rows), len(oracle_rows) - actual_cloud_rows, "local run rows"),
            (cloud_availability.get("cloud_rows"), actual_cloud_rows, "cloud availability rows"),
        ]
        for left, right, label in industrial_pairs:
            if left != right:
                issues.append(f"{label} mismatch")
        if industrial.get("cloud_environment") != cloud_availability:
            issues.append("industrial cloud environment differs from cloud availability")

        legacy_pairs = [
            (legacy_cleanroom.get("validation_status"), validation.get("status"), "legacy validation status"),
            (legacy_cleanroom.get("recomputed_status"), recomputed.get("status"), "legacy recomputed status"),
            (legacy_cleanroom.get("jobs"), selected_jobs, "legacy job count"),
            (legacy_cleanroom.get("dummy_actions"), metrics.get("dummy_actions"), "legacy dummy count"),
            (legacy_cleanroom.get("one_hot_violations"), metrics.get("one_hot_violations"), "legacy one-hot count"),
        ]
        for left, right, label in legacy_pairs:
            if left != right:
                issues.append(f"{label} mismatch")
        if not close_number(
            legacy_cleanroom.get("expected_recourse_cost_usd"),
            metrics.get("expected_recourse_cost_usd"),
        ):
            issues.append("legacy expected recourse mismatch")
        if not close_number(
            legacy_cleanroom.get("cvar95_recourse_cost_usd"),
            metrics.get("cvar95_recourse_cost_usd"),
        ):
            issues.append("legacy CVaR95 mismatch")

        summary = {
            "selected_jobs": selected_jobs,
            "selected_action_variables": selected_variables,
            "dummy_actions": dummy_count,
            "one_hot_violations": int(metrics["one_hot_violations"]),
            "true_action_cost_usd": float(metrics["true_action_cost_usd"]),
            "expected_recourse_cost_usd": float(metrics["expected_recourse_cost_usd"]),
            "cvar95_recourse_cost_usd": float(metrics["cvar95_recourse_cost_usd"]),
            "peak_it_mw": float(metrics["peak_it_mw"]),
            "mode_counts": mode_counts,
            "oracle_rows": len(oracle_rows),
            "cloud_rows": actual_cloud_rows,
            "failed_oracle_rows": actual_failed_rows,
        }
    except (KeyError, TypeError, ValueError, OSError, UnicodeError, json.JSONDecodeError) as error:
        issues.append(f"cross-artifact parse/evaluation error: {type(error).__name__}")

    if issues:
        recorder.failed(
            "cross_artifact_consistency",
            "Final solution, metrics, validation, and industrial evidence are inconsistent.",
            issue_count=len(issues),
            issues=bounded(issues),
            **summary,
        )
    else:
        recorder.passed(
            "cross_artifact_consistency",
            "Final solution, recomputed metrics, validation tables, and cloud/local evidence agree.",
            **summary,
        )
    return summary


def verify_qubo_matrices(package_dir: Path, recorder: Recorder, skip: bool) -> None:
    if skip:
        recorder.skipped(
            "qubo_matrix_consistency",
            "QUBO matrix checks were explicitly disabled.",
        )
        return
    try:
        import numpy as np
    except ImportError:
        recorder.warned(
            "qubo_matrix_consistency",
            "NumPy is unavailable; optional QUBO matrix checks were not run.",
        )
        return

    matrix_dir = package_dir / "artifacts" / "qubo_matrices"
    matrix_paths = sorted(matrix_dir.glob("Q_*.npy")) if matrix_dir.is_dir() else []
    if not matrix_paths:
        recorder.failed(
            "qubo_matrix_consistency",
            "No packaged QUBO matrices were found.",
        )
        return

    issues: list[str] = []
    families: dict[str, dict[str, Any]] = {}
    try:
        industrial = load_json(package_dir / "artifacts" / "industrial_ising_evidence_summary.json")
        declared_families = set((industrial.get("families") or {}).keys())
        for matrix_path in matrix_paths:
            family = matrix_path.stem.removeprefix("Q_")
            meta_path = matrix_dir / f"meta_{family}.json"
            index_path = matrix_dir / f"index_{family}.csv"
            if not meta_path.is_file() or not index_path.is_file():
                issues.append(f"{family}: metadata or index is missing")
                continue
            matrix = np.load(matrix_path, allow_pickle=False)
            metadata = load_json(meta_path)
            index_rows = read_csv_rows(index_path)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                issues.append(f"{family}: matrix is not square")
                continue
            spins = int(matrix.shape[0])
            if not np.issubdtype(matrix.dtype, np.number):
                issues.append(f"{family}: matrix dtype is not numeric")
            if not bool(np.isfinite(matrix).all()):
                issues.append(f"{family}: matrix contains non-finite values")
            if not bool(np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12)):
                issues.append(f"{family}: matrix is not symmetric")
            if metadata.get("n_variables") != spins or len(index_rows) != spins:
                issues.append(f"{family}: matrix, metadata, and index sizes differ")
            try:
                indices = [int(row["var_idx"]) for row in index_rows]
            except (KeyError, TypeError, ValueError):
                indices = []
            if indices != list(range(spins)):
                issues.append(f"{family}: var_idx is not contiguous from zero")
            if any(row.get("family") != family for row in index_rows):
                issues.append(f"{family}: index contains a different family label")
            actual_stats = {
                "q_min": float(np.min(matrix)),
                "q_max": float(np.max(matrix)),
                "q_abs_max": float(np.max(np.abs(matrix))),
                "q_nonzero": int(np.count_nonzero(matrix)),
            }
            for key in ("q_min", "q_max", "q_abs_max"):
                if not close_number(actual_stats[key], metadata.get(key), tolerance=1e-10):
                    issues.append(f"{family}: {key} differs from metadata")
            if actual_stats["q_nonzero"] != metadata.get("q_nonzero"):
                issues.append(f"{family}: q_nonzero differs from metadata")
            families[family] = {"spins": spins, **actual_stats}
        if set(families) != declared_families:
            issues.append("matrix families differ from industrial summary families")
    except (OSError, UnicodeError, ValueError, KeyError, json.JSONDecodeError) as error:
        issues.append(f"QUBO parse/evaluation error: {type(error).__name__}")

    if issues:
        recorder.failed(
            "qubo_matrix_consistency",
            "Packaged QUBO matrices, indexes, metadata, or family inventory are inconsistent.",
            issue_count=len(issues),
            issues=bounded(issues),
            families=families,
        )
    else:
        recorder.passed(
            "qubo_matrix_consistency",
            "Every QUBO matrix is safe-loadable, finite, symmetric, and consistent with its index and metadata.",
            family_count=len(families),
            families=families,
        )


def run_cleanroom_phase(
    zip_path: Path,
    zip_state: dict[str, Any] | None,
    recorder: Recorder,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if zip_state is None or not zip_state.get("safe"):
        recorder.skipped(
            "cleanroom_extraction",
            "Clean-room extraction was skipped because archive safety or payload checks failed.",
        )
        recorder.skipped(
            "package_path_cache_scan",
            "Path/cache scan was skipped because no trusted archive extraction was available.",
        )
        recorder.skipped(
            "package_secret_scan",
            "Secret scan was skipped because no trusted archive extraction was available.",
        )
        recorder.skipped(
            "package_text_scan_coverage",
            "Text scan coverage is unavailable because no trusted archive extraction was available.",
        )
        recorder.skipped(
            "cleanroom_validator_execution",
            "Embedded validation was skipped because no trusted extraction was available.",
        )
        recorder.skipped(
            "cross_artifact_consistency",
            "Cross-artifact checks were skipped because no trusted extraction was available.",
        )
        recorder.skipped(
            "qubo_matrix_consistency",
            "QUBO matrix checks were skipped because no trusted extraction was available.",
        )
        return {}

    artifact_summary: dict[str, Any] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="taolin_submission_verify_") as temporary:
            extraction_root = Path(temporary)
            extracted_count = safe_extract(zip_path, extraction_root)
            extracted_package = extraction_root / str(zip_state["prefix"])
            if not extracted_package.is_dir():
                raise ValueError("Expected archive root was not extracted")
            recorder.passed(
                "cleanroom_extraction",
                "The final ZIP was safely extracted into a new temporary directory.",
                extracted_file_count=extracted_count,
                expected_file_count=zip_state["member_count"],
            )
            scan_package(extracted_package, recorder, args.max_scan_bytes)
            validator_state = run_embedded_validator(
                extracted_package,
                recorder,
                timeout_seconds=args.validator_timeout_seconds,
            )
            artifact_summary = verify_cross_artifact_consistency(
                extracted_package,
                validator_state,
                recorder,
            )
            verify_qubo_matrices(extracted_package, recorder, args.skip_qubo_checks)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        recorder.failed(
            "cleanroom_extraction",
            "Safe extraction failed.",
            error_type=type(error).__name__,
        )
        for check_id, summary in (
            (
                "package_path_cache_scan",
                "Path/cache scan was skipped after extraction failure.",
            ),
            (
                "package_secret_scan",
                "Secret scan was skipped after extraction failure.",
            ),
            (
                "package_text_scan_coverage",
                "Text scan coverage is unavailable after extraction failure.",
            ),
        ):
            if not recorder.has(check_id):
                recorder.skipped(check_id, summary)
        recorder.skipped(
            "cleanroom_validator_execution",
            "Embedded validation was skipped after extraction failure.",
        )
        recorder.skipped(
            "cross_artifact_consistency",
            "Cross-artifact checks were skipped after extraction failure.",
        )
        recorder.skipped(
            "qubo_matrix_consistency",
            "QUBO matrix checks were skipped after extraction failure.",
        )
    return artifact_summary


def render_markdown(payload: dict[str, Any]) -> str:
    ready = payload["overall_status"] == "ready"
    lines = [
        "# TAOLIN-DC-SIM Submission Verification",
        "",
        f"Overall status: **{'READY' if ready else 'NOT READY'}**",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Package directory: `{payload['inputs']['package_directory']}`",
        f"Final ZIP: `{payload['inputs']['zip_path']}`",
    ]
    archive = payload.get("archive", {})
    if archive:
        lines.extend(
            [
                f"ZIP SHA-256: `{archive.get('archive_sha256', 'unavailable')}`",
                f"ZIP bytes: {archive.get('archive_bytes', 'unavailable')}",
                f"ZIP members: {archive.get('member_count', 'unavailable')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| Status | Check | Result |",
            "| --- | --- | --- |",
        ]
    )
    for check in payload["checks"]:
        lines.append(
            f"| {check['status'].upper()} | `{check['check_id']}` | {check['summary']} |"
        )

    for status, heading in (("fail", "Failures"), ("warn", "Warnings")):
        selected = [check for check in payload["checks"] if check["status"] == status]
        if not selected:
            continue
        lines.extend(["", f"## {heading}", ""])
        for check in selected:
            evidence = json.dumps(check["evidence"], ensure_ascii=False, sort_keys=True)
            if len(evidence) > 1200:
                evidence = evidence[:1197] + "..."
            lines.append(f"- `{check['check_id']}`: {check['summary']}")
            lines.append(f"  - Evidence: `{evidence}`")

    summary = payload.get("artifact_summary") or {}
    if summary:
        lines.extend(["", "## Verified Final Metrics", ""])
        for key in (
            "selected_jobs",
            "selected_action_variables",
            "dummy_actions",
            "one_hot_violations",
            "true_action_cost_usd",
            "expected_recourse_cost_usd",
            "cvar95_recourse_cost_usd",
            "peak_it_mw",
            "oracle_rows",
            "cloud_rows",
            "failed_oracle_rows",
        ):
            if key in summary:
                lines.append(f"- `{key}`: {summary[key]}")
    lines.extend(
        [
            "",
            "## Gate Meaning",
            "",
            "`READY` means the final ZIP itself passed manifest, hash, safety, secret/path, isolated recomputation, cross-artifact, and QUBO consistency checks. It does not perform the external competition-portal upload.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    payload: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
    sha256_path: Path,
    zip_path: Path,
    manifest_path: Path,
    root: Path,
) -> None:
    atomic_write_text(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )
    atomic_write_text(markdown_path, render_markdown(payload))
    checksum_targets = [zip_path, manifest_path, json_path, markdown_path]
    checksum_lines = [
        f"{sha256_file(path)}  {path_label(path, root)}"
        for path in checksum_targets
        if path.is_file()
    ]
    atomic_write_text(sha256_path, "\n".join(checksum_lines) + "\n")


def load_submission_builder(root: Path) -> Any:
    builder_path = root / "src" / "07_solve_isling" / "build_submission_bundle.py"
    spec = importlib.util.spec_from_file_location("taolin_submission_builder", builder_path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the submission builder for staging validation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trusted_staging_members(builder: Any, root: Path) -> dict[str, set[str]]:
    """Derive exact attachment members from repository allowlists and canonical manifests."""
    code_members = set(builder.CODE_REQUIRED_FILES)
    for relative in builder.CODE_OPTIONAL_FILES:
        path = root / PurePosixPath(relative)
        if path.exists():
            builder.validate_repository_source(path, relative=relative, label="optional code")
            code_members.add(relative)

    data_members = set(builder.DATA_REQUIRED_FILES) | {
        "DATA_PACKAGE_README.md",
        "data/download_manifest.json",
    }
    builder.require_files(builder.DATA_REQUIRED_FILES, label="data")

    final_entries = builder.collect_manifest_bound_tree(
        root / "deliverables" / "final_solution_package",
        label="final solution package",
    )
    report_entries = builder.collect_strict_tree(
        root / "deliverables" / "competition_report",
        builder.COMPETITION_REPORT_FILES,
        label="competition report",
    )
    support_entries = builder.require_files(builder.RESULT_SUPPORT_FILES, label="result support")
    for relative in builder.RESULT_OPTIONAL_FILES:
        path = root / PurePosixPath(relative)
        if path.exists():
            builder.validate_repository_source(path, relative=relative, label="optional result support")
            support_entries.append((relative, path))
    result_members = {
        f"final_solution_package/{relative}" for relative, _ in final_entries
    }
    result_members.update(
        f"competition_report/{relative}" for relative, _ in report_entries
    )
    result_members.update(PurePosixPath(relative).name for relative, _ in support_entries)
    return {
        "code": code_members,
        "data": data_members,
        "calculation_results": result_members,
    }


def nested_archive_kind(path: Path) -> str | None:
    suffix = path.suffix.casefold()
    if suffix in NESTED_ARCHIVE_SUFFIXES:
        return f"suffix:{suffix}"
    try:
        if zipfile.is_zipfile(path):
            return "signature:zip"
    except OSError:
        return "unreadable"
    try:
        with path.open("rb") as handle:
            header = handle.read(600)
    except OSError:
        return "unreadable"
    signatures = (
        (b"PK\x03\x04", "zip"),
        (b"PK\x05\x06", "zip-empty"),
        (b"PK\x07\x08", "zip-spanned"),
        (b"7z\xbc\xaf'\x1c", "7z"),
        (b"Rar!\x1a\x07", "rar"),
        (b"\x1f\x8b", "gzip"),
        (b"BZh", "bzip2"),
        (b"\xfd7zXZ\x00", "xz"),
    )
    for signature, label in signatures:
        if header.startswith(signature):
            return f"signature:{label}"
    if len(header) >= 262 and header[257:262] == b"ustar":
        return "signature:tar"
    return None


def inspect_portal_zip(
    portal_zip: Path,
    *,
    require_video: bool,
    max_members: int = 10_000,
    max_member_bytes: int = 128 * 1024 * 1024,
    max_uncompressed_bytes: int = 512 * 1024 * 1024,
    max_member_compression_ratio: float = 500.0,
    max_total_compression_ratio: float = 100.0,
) -> dict[str, Any]:
    """Validate and safely probe the flat outer archive used for portal upload."""
    validate_unlinked_path_components(portal_zip, label="portal ZIP")
    try:
        resolved = portal_zip.resolve(strict=True)
    except OSError as error:
        raise ValueError("Portal ZIP does not exist") from error
    is_junction = getattr(resolved, "is_junction", lambda: False)
    if not resolved.is_file() or resolved.is_symlink() or is_junction():
        raise ValueError("Portal ZIP must be an unlinked regular file")
    sibling_entries = list(resolved.parent.iterdir())
    if (
        len(sibling_entries) != 1
        or sibling_entries[0].name != resolved.name
        or not sibling_entries[0].is_file()
    ):
        raise ValueError("Portal publish directory must contain exactly one upload ZIP")
    match = PORTAL_FILENAME_RE.fullmatch(resolved.name)
    if match is None:
        raise ValueError("Portal ZIP filename does not match the official single-package contract")
    registration_id, work_name = match.groups()
    archive_bytes = resolved.stat().st_size
    if archive_bytes >= PORTAL_MAX_BYTES:
        raise ValueError("Portal ZIP must be strictly smaller than 50,000,000 bytes")

    prefix = f"能源调度_{registration_id}_{work_name}"
    paper_name = f"{prefix}_论文.pdf"
    code_name = f"{prefix}_代码.zip"
    data_name = f"{prefix}_数据集.zip"
    result_name = f"{prefix}_补充材料.zip"
    video_name = f"{prefix}_演示视频.mp4"
    manifest_name = f"{prefix}_附件清单.json"
    required_names = {
        paper_name,
        code_name,
        data_name,
        result_name,
        manifest_name,
    }

    try:
        with zipfile.ZipFile(resolved, "r") as archive:
            infos = archive.infolist()
            if len(infos) > max_members:
                raise ValueError("Portal ZIP exceeds the configured member-count limit")
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("Portal ZIP contains duplicate member names")
            if len(names) != len({name.casefold() for name in names}):
                raise ValueError("Portal ZIP contains case-insensitive member collisions")
            if names != sorted(names):
                raise ValueError("Portal ZIP members are not in deterministic sorted order")
            for info in infos:
                error = validate_relative_path(info.filename, allow_directory=False)
                if error is not None or info.is_dir() or PurePosixPath(info.filename).name != info.filename:
                    raise ValueError(f"Unsafe or non-root portal ZIP member: {info.filename}")
                if info.flag_bits & 0x1:
                    raise ValueError(f"Encrypted portal ZIP member: {info.filename}")
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                if info.create_system != 3 or not stat.S_ISREG(unix_mode):
                    raise ValueError(f"Non-regular portal ZIP member: {info.filename}")
                if info.date_time != PORTAL_FIXED_ZIP_TIME:
                    raise ValueError(f"Non-deterministic portal ZIP timestamp: {info.filename}")
                if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                    raise ValueError(f"Unsupported portal ZIP compression: {info.filename}")
                if info.file_size > max_member_bytes:
                    raise ValueError(f"Portal ZIP member exceeds the extraction limit: {info.filename}")
                ratio = info.file_size / max(info.compress_size, 1)
                if ratio > max_member_compression_ratio:
                    raise ValueError(f"Portal ZIP member exceeds the compression-ratio limit: {info.filename}")
            total_uncompressed = sum(info.file_size for info in infos)
            if total_uncompressed > max_uncompressed_bytes:
                raise ValueError("Portal ZIP exceeds the total extraction limit")
            total_compressed = sum(info.compress_size for info in infos)
            total_ratio = total_uncompressed / max(total_compressed, 1)
            if total_ratio > max_total_compression_ratio:
                raise ValueError("Portal ZIP exceeds the total compression-ratio limit")
            video_present = video_name in names
            expected_names = set(required_names)
            if video_present:
                expected_names.add(video_name)
            if require_video and not video_present:
                raise ValueError("A demonstration video is required but absent")
            if set(names) != expected_names:
                missing = sorted(expected_names - set(names))
                extra = sorted(set(names) - expected_names)
                raise ValueError(
                    f"Portal ZIP member set mismatch: missing={missing}; extra={extra}"
                )
            bad_crc = archive.testzip()
            if bad_crc is not None:
                raise ValueError(f"Portal ZIP CRC failure: {bad_crc}")
    except ValueError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise ValueError(f"Portal ZIP cannot be safely read: {type(error).__name__}") from error

    expected_inner_zips = {code_name, data_name, result_name}
    try:
        with tempfile.TemporaryDirectory(prefix="taolin-portal-inspect-") as temporary:
            extracted = Path(temporary) / "components"
            extracted_count = safe_extract(resolved, extracted)
            if extracted_count != len(expected_names):
                raise ValueError("Portal ZIP extraction count differs from its member set")
            for name in sorted(expected_names):
                path = extracted / name
                if name in expected_inner_zips:
                    if not zipfile.is_zipfile(path):
                        raise ValueError(f"Expected inner attachment is not a ZIP: {name}")
                    continue
                kind = nested_archive_kind(path)
                if kind is not None:
                    raise ValueError(
                        f"Unexpected nested archive outside the three manifest-bound ZIPs: {name} ({kind})"
                    )
    except ValueError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise ValueError(f"Portal ZIP safe extraction failed: {type(error).__name__}") from error

    return {
        "archive_bytes": archive_bytes,
        "archive_sha256": sha256_file(resolved),
        "member_count": len(expected_names),
        "member_names": sorted(expected_names),
        "registration_id": registration_id,
        "video_present": video_present,
        "work_name": work_name,
    }


def verify_portal_zip(
    portal_zip: Path,
    root: Path,
    recorder: Recorder,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Verify the outer ZIP, then reuse all ten component-level staging gates."""
    try:
        state = inspect_portal_zip(
            portal_zip,
            require_video=bool(args.require_video),
            max_members=int(args.max_members),
            max_member_bytes=int(args.max_member_bytes),
            max_uncompressed_bytes=int(args.max_uncompressed_bytes),
            max_member_compression_ratio=float(args.max_member_compression_ratio),
            max_total_compression_ratio=float(args.max_total_compression_ratio),
        )
    except Exception as error:
        recorder.failed(
            "portal_archive_integrity",
            "The single portal ZIP failed its filename, size, membership, safety, CRC, or determinism gate.",
            error_type=type(error).__name__,
            detail=str(error)[:500],
        )
        return None

    recorder.passed(
        "portal_archive_integrity",
        "The single portal ZIP is flat, deterministic, CRC-clean, unencrypted, and below 50,000,000 bytes.",
        archive_bytes=state["archive_bytes"],
        archive_sha256=state["archive_sha256"],
        member_count=state["member_count"],
        video_present=state["video_present"],
    )
    try:
        with tempfile.TemporaryDirectory(prefix="taolin-portal-verify-") as temporary:
            extracted = Path(temporary) / "components"
            safe_extract(portal_zip, extracted)
            component_recorder = Recorder()
            verify_staging_bundle(extracted, root, component_recorder, args)
            for check in component_recorder.checks:
                recorder.add(
                    check.check_id,
                    check.status,
                    check.summary,
                    **check.evidence,
                )
    except Exception as error:
        recorder.failed(
            "portal_component_verification",
            "Safe extraction or component-level verification failed.",
            error_type=type(error).__name__,
        )
    return state


def compare_extracted_staging_content(
    combined: Path,
    builder: Any,
    root: Path,
) -> list[str]:
    """Compare extracted attachment bytes with trusted working-tree sources."""
    issues: list[str] = []

    code_dir = combined / "code"
    code_relatives = set(builder.CODE_REQUIRED_FILES)
    code_relatives.update(
        relative
        for relative in builder.CODE_OPTIONAL_FILES
        if (root / PurePosixPath(relative)).exists()
    )
    for relative in sorted(code_relatives):
        extracted = code_dir / PurePosixPath(relative)
        source = root / PurePosixPath(relative)
        if not extracted.is_file() or sha256_file(extracted) != sha256_file(source):
            issues.append(f"code payload differs from source: {relative}")

    data_dir = combined / "data"
    for relative in sorted(builder.DATA_REQUIRED_FILES):
        extracted = data_dir / PurePosixPath(relative)
        source = root / PurePosixPath(relative)
        if not extracted.is_file() or sha256_file(extracted) != sha256_file(source):
            issues.append(f"data payload differs from source: {relative}")
    readme_path = data_dir / "DATA_PACKAGE_README.md"
    if not readme_path.is_file() or readme_path.read_bytes() != builder.DATA_PACKAGE_README_BYTES:
        issues.append("data package README differs from the trusted builder payload")
    sanitized_path = data_dir / "data" / "download_manifest.json"
    if not sanitized_path.is_file() or sanitized_path.read_bytes() != builder.sanitized_download_manifest():
        issues.append("sanitized download manifest differs from a fresh trusted sanitization")

    result_dir = combined / "calculation_results"
    nested_package = result_dir / "final_solution_package"
    canonical_manifest = root / "deliverables" / "final_solution_package" / "MANIFEST.json"
    extracted_manifest = nested_package / "MANIFEST.json"
    if (
        not extracted_manifest.is_file()
        or sha256_file(extracted_manifest) != sha256_file(canonical_manifest)
    ):
        issues.append("nested final-solution manifest differs from the canonical manifest")
    nested_recorder = Recorder()
    nested_state = verify_manifest(nested_package, nested_recorder)
    if nested_state is None or any(check.status != "pass" for check in nested_recorder.checks):
        issues.append("nested final-solution manifest does not bind its exact extracted payload")

    for relative in sorted(builder.COMPETITION_REPORT_FILES):
        extracted = result_dir / "competition_report" / PurePosixPath(relative)
        source = root / "deliverables" / "competition_report" / PurePosixPath(relative)
        if not extracted.is_file() or sha256_file(extracted) != sha256_file(source):
            issues.append(f"competition-report payload differs from source: {relative}")
    support_relatives = list(builder.RESULT_SUPPORT_FILES)
    support_relatives.extend(
        relative
        for relative in builder.RESULT_OPTIONAL_FILES
        if (root / PurePosixPath(relative)).exists()
    )
    for relative in sorted(set(support_relatives)):
        extracted = result_dir / PurePosixPath(relative).name
        source = root / PurePosixPath(relative)
        if not extracted.is_file() or sha256_file(extracted) != sha256_file(source):
            issues.append(f"result-support payload differs from source: {relative}")
    return issues


def verify_staging_bundle(
    staging_dir: Path,
    root: Path,
    recorder: Recorder,
    args: argparse.Namespace,
) -> None:
    """Verify the complete portal-facing attachment set without writing outputs."""
    if not staging_dir.is_dir():
        recorder.failed("staging_presence", "Submission staging directory is missing.")
        return
    is_junction = getattr(staging_dir, "is_junction", lambda: False)
    if staging_dir.is_symlink() or is_junction():
        recorder.failed("staging_presence", "Submission staging directory must not be linked.")
        return

    entries = list(staging_dir.iterdir())
    non_regular = [entry.name for entry in entries if not entry.is_file() or entry.is_symlink()]
    manifests = [entry for entry in entries if entry.is_file() and entry.name.endswith("_附件清单.json")]
    if non_regular or len(manifests) != 1:
        recorder.failed(
            "staging_manifest_integrity",
            "Staging must contain regular files and exactly one attachment manifest.",
            non_regular=bounded(sorted(non_regular)),
            manifest_count=len(manifests),
        )
        return

    manifest_path = manifests[0]
    try:
        payload = load_json(manifest_path)
        builder = load_submission_builder(root)
    except Exception as error:
        recorder.failed(
            "staging_manifest_integrity",
            "The attachment manifest or validation helpers could not be loaded.",
            error_type=type(error).__name__,
        )
        return

    issues: list[str] = []
    if payload.get("schema_version") != 1:
        issues.append("schema_version must equal 1")
    if payload.get("track") != "能源调度":
        issues.append("track must equal 能源调度")
    if payload.get("filename_pattern") != "选题方向_报名编号_作品名称_文件类型":
        issues.append("filename_pattern does not match the official four-field contract")
    if payload.get("manifest_self_hash_excluded") is not True:
        issues.append("manifest_self_hash_excluded must be true")
    if not isinstance(payload.get("video_required"), bool):
        issues.append("video_required must be Boolean")

    try:
        registration_id = builder.validate_registration_id(str(payload.get("registration_id", "")))
        work_name = builder.validate_work_name(str(payload.get("work_name", "")))
    except Exception as error:
        issues.append(f"invalid registration/work metadata: {type(error).__name__}")
        registration_id, work_name = "", ""

    prefix = f"能源调度_{registration_id}_{work_name}" if registration_id and work_name else ""
    expected_suffix = {
        "paper": "_论文.pdf",
        "code": "_代码.zip",
        "data": "_数据集.zip",
        "calculation_results": "_补充材料.zip",
        "demonstration_video": "_演示视频.mp4",
    }
    attachments = payload.get("attachments")
    records_by_role: dict[str, dict[str, Any]] = {}
    expected_files: set[str] = {manifest_path.name}
    if not isinstance(attachments, list) or len(attachments) not in {4, 5}:
        issues.append("attachments must contain exactly four files, or five with video")
        attachments = []
    for index, record in enumerate(attachments):
        if not isinstance(record, dict):
            issues.append(f"attachment {index} is not an object")
            continue
        role = record.get("role")
        filename = record.get("file")
        if role not in expected_suffix or role in records_by_role:
            issues.append(f"attachment {index} has an unknown or duplicate role")
            continue
        if not isinstance(filename, str) or filename != prefix + expected_suffix[role]:
            issues.append(f"attachment {index} filename does not match its official role")
            continue
        if validate_relative_path(filename) is not None or PurePosixPath(filename).name != filename:
            issues.append(f"attachment {index} has an unsafe filename")
            continue
        declared_bytes = record.get("bytes")
        declared_hash = record.get("sha256")
        if not isinstance(declared_bytes, int) or declared_bytes <= 0:
            issues.append(f"attachment {index} has an invalid byte count")
        if not isinstance(declared_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", declared_hash):
            issues.append(f"attachment {index} has an invalid SHA-256")
        if role in {"code", "data", "calculation_results"}:
            if not isinstance(record.get("zip_entries"), int) or record["zip_entries"] <= 0:
                issues.append(f"attachment {index} has an invalid ZIP entry count")
        elif "zip_entries" in record:
            issues.append(f"attachment {index} unexpectedly declares ZIP entries")
        records_by_role[role] = record
        expected_files.add(filename)

    required_roles = {"paper", "code", "data", "calculation_results"}
    if not required_roles.issubset(records_by_role):
        issues.append("one or more required attachment roles are missing")
    video_present = "demonstration_video" in records_by_role
    if payload.get("video_required") is True and not video_present:
        issues.append("manifest requires a video but no video attachment exists")
    if args.require_video and (not video_present or payload.get("video_required") is not True):
        issues.append("--require-video requires a present video and video_required=true")
    if manifest_path.name != prefix + "_附件清单.json":
        issues.append("attachment manifest filename does not match official metadata")

    actual_files = {entry.name for entry in entries if entry.is_file()}
    if actual_files != expected_files:
        issues.append("top-level staging membership differs from the manifest")
    for role, record in records_by_role.items():
        attachment = staging_dir / str(record["file"])
        if not attachment.is_file():
            issues.append(f"missing attachment for role {role}")
            continue
        if attachment.stat().st_size != record.get("bytes"):
            issues.append(f"byte count mismatch for role {role}")
        if sha256_file(attachment) != record.get("sha256"):
            issues.append(f"SHA-256 mismatch for role {role}")

    if issues:
        recorder.failed(
            "staging_manifest_integrity",
            "Official attachment names, metadata, membership, or hashes failed.",
            issue_count=len(issues),
            issues=bounded(issues),
        )
        return
    recorder.passed(
        "staging_manifest_integrity",
        "Official attachment metadata, names, membership, sizes, and SHA-256 hashes match.",
        attachment_count=len(records_by_role),
        registration_id=registration_id,
        video_present=video_present,
    )

    structural_issues: list[str] = []
    try:
        builder.validate_paper(staging_dir / str(records_by_role["paper"]["file"]))
    except Exception as error:
        structural_issues.append(f"paper structural gate failed: {type(error).__name__}")
    if video_present:
        try:
            builder.validate_video(
                staging_dir / str(records_by_role["demonstration_video"]["file"]),
                required=True,
            )
        except Exception as error:
            structural_issues.append(f"video structural gate failed: {type(error).__name__}")
    if structural_issues:
        recorder.failed(
            "staging_document_media_structure",
            "Paper or video structure failed.",
            issues=structural_issues,
        )
    else:
        recorder.passed(
            "staging_document_media_structure",
            "The paper is a readable, unencrypted PDF and any video is a structured MP4 stream.",
        )

    zip_states: dict[str, dict[str, Any]] = {}
    zip_issues: list[str] = []
    for role in ("code", "data", "calculation_results"):
        record = records_by_role[role]
        archive_path = staging_dir / str(record["file"])
        local_recorder = Recorder()
        state = verify_zip(
            archive_path,
            staging_dir,
            None,
            local_recorder,
            args,
            check_payload_consistency=False,
        )
        if (
            state is None
            or not state.get("safe")
            or not local_recorder.ready
            or state.get("member_count") != record.get("zip_entries")
        ):
            zip_issues.append(f"{role} ZIP failed safety or entry-count validation")
        else:
            zip_states[role] = state
    if zip_issues:
        recorder.failed(
            "staging_zip_safety",
            "One or more attachment archives failed safety, CRC, expansion, or count gates.",
            issues=zip_issues,
        )
        return
    recorder.passed(
        "staging_zip_safety",
        "All attachment ZIPs are traversal-safe, unencrypted, CRC-clean, bounded, and count-matched.",
        archive_count=len(zip_states),
        member_counts={role: state["member_count"] for role, state in zip_states.items()},
    )

    membership_issues: list[str] = []
    try:
        expected_members = trusted_staging_members(builder, root)
        for role in ("code", "data", "calculation_results"):
            archive_path = staging_dir / str(records_by_role[role]["file"])
            with zipfile.ZipFile(archive_path, "r") as archive:
                names = [info.filename for info in archive.infolist() if not info.is_dir()]
            actual = set(names)
            expected = expected_members[role]
            if len(names) != len(actual):
                membership_issues.append(f"{role} ZIP contains duplicate member names")
            if actual != expected:
                membership_issues.append(
                    f"{role} ZIP membership mismatch: missing={sorted(expected - actual)[:10]}, "
                    f"extra={sorted(actual - expected)[:10]}"
                )
    except Exception as error:
        membership_issues.append(f"trusted member derivation failed: {type(error).__name__}")
    if membership_issues:
        recorder.failed(
            "staging_zip_membership",
            "Attachment ZIPs do not match the trusted repository allowlists and canonical manifest.",
            issues=bounded(membership_issues),
        )
        return
    recorder.passed(
        "staging_zip_membership",
        "Every attachment ZIP has the exact trusted member set.",
        member_counts={role: len(members) for role, members in expected_members.items()},
    )

    try:
        with tempfile.TemporaryDirectory(prefix="taolin-staging-verify-") as temporary:
            combined = Path(temporary) / "combined"
            combined.mkdir(parents=True)
            shutil.copy2(manifest_path, combined / manifest_path.name)
            media_dir = combined / "top_level_media"
            media_dir.mkdir()
            shutil.copy2(
                staging_dir / str(records_by_role["paper"]["file"]),
                media_dir / str(records_by_role["paper"]["file"]),
            )
            if video_present:
                shutil.copy2(
                    staging_dir / str(records_by_role["demonstration_video"]["file"]),
                    media_dir / str(records_by_role["demonstration_video"]["file"]),
                )
            for role in ("code", "data", "calculation_results"):
                safe_extract(
                    staging_dir / str(records_by_role[role]["file"]),
                    combined / role,
                )
            nested_archives: list[dict[str, str]] = []
            for path in combined.rglob("*"):
                if not path.is_file():
                    continue
                kind = nested_archive_kind(path)
                if kind is not None:
                    nested_archives.append(
                        {
                            "file": path.relative_to(combined).as_posix(),
                            "kind": kind,
                        }
                    )
            nested_archives.sort(key=lambda item: item["file"])
            if nested_archives:
                recorder.failed(
                    "staging_nested_archive_scan",
                    "Extracted attachments contain nested archives that were not recursively scanned.",
                    archives=bounded(nested_archives),
                )
            else:
                recorder.passed(
                    "staging_nested_archive_scan",
                    "Extracted attachments contain no unscanned nested archives.",
                )
            content_issues = compare_extracted_staging_content(combined, builder, root)
            if content_issues:
                recorder.failed(
                    "staging_trusted_content",
                    "Extracted attachment bytes differ from trusted source and manifest-bound content.",
                    issue_count=len(content_issues),
                    issues=bounded(content_issues),
                )
            else:
                recorder.passed(
                    "staging_trusted_content",
                    "Every extracted code, data, report, support, and final-solution file matches trusted content.",
                )
            scan_recorder = Recorder()
            scan_package(combined, scan_recorder, args.max_scan_bytes)
            for check in scan_recorder.checks:
                status = "fail" if check.status == "warn" else check.status
                recorder.add(
                    f"staging_{check.check_id}",
                    status,
                    check.summary,
                    **check.evidence,
                )
    except Exception as error:
        recorder.failed(
            "staging_extraction_scan",
            "Attachment extraction or full-content scanning failed.",
            error_type=type(error).__name__,
        )

    changed = []
    for role, record in records_by_role.items():
        attachment = staging_dir / str(record["file"])
        if attachment.stat().st_size != record["bytes"] or sha256_file(attachment) != record["sha256"]:
            changed.append(role)
    if changed:
        recorder.failed(
            "staging_postscan_hashes",
            "One or more attachments changed during verification.",
            roles=changed,
        )
    else:
        recorder.passed(
            "staging_postscan_hashes",
            "All top-level attachment hashes remained stable through verification.",
        )


def run_staging_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.staging_dir is None:
        parser.error("--staging-dir is required for staging mode")
    for attribute in (
        "max_scan_bytes",
        "max_archive_bytes",
        "max_members",
        "max_member_bytes",
        "max_uncompressed_bytes",
        "max_member_compression_ratio",
        "max_total_compression_ratio",
    ):
        if getattr(args, attribute) <= 0:
            parser.error(f"{attribute} must be positive")
    try:
        validate_unlinked_path_components(args.root, label="root")
        validate_unlinked_path_components(args.staging_dir, label="staging directory")
    except ValueError as error:
        parser.error(str(error))
    root = args.root.resolve()
    try:
        staging_dir = args.staging_dir.resolve(strict=True)
    except OSError:
        staging_dir = args.staging_dir.resolve()
    recorder = Recorder()
    try:
        verify_staging_bundle(staging_dir, root, recorder, args)
    except Exception as error:
        recorder.failed(
            "staging_verifier_internal_error",
            "The staging verifier encountered an unexpected internal error.",
            error_type=type(error).__name__,
        )
    is_ready = recorder.ready_for(STAGING_MANDATORY_CHECKS)
    status = "READY" if is_ready else "NOT_READY"
    print(f"Staging overall: {status}")
    print(f"Staging checks: {recorder.counts()}")
    for check in recorder.checks:
        print(f"{check.status.upper()}: {check.check_id}: {check.summary}")
    return 0 if is_ready else 1


def run_portal_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.portal_zip is None:
        parser.error("--portal-zip is required for portal mode")
    for attribute in (
        "max_scan_bytes",
        "max_archive_bytes",
        "max_members",
        "max_member_bytes",
        "max_uncompressed_bytes",
        "max_member_compression_ratio",
        "max_total_compression_ratio",
    ):
        if getattr(args, attribute) <= 0:
            parser.error(f"{attribute} must be positive")
    try:
        validate_unlinked_path_components(args.root, label="root")
        validate_unlinked_path_components(args.portal_zip, label="portal ZIP")
    except ValueError as error:
        parser.error(str(error))
    root = args.root.resolve()
    try:
        portal_zip = args.portal_zip.resolve(strict=True)
    except OSError:
        portal_zip = args.portal_zip.resolve()
    recorder = Recorder()
    state = verify_portal_zip(portal_zip, root, recorder, args)
    is_ready = recorder.ready_for(PORTAL_MANDATORY_CHECKS)
    status = "READY" if is_ready else "NOT_READY"
    print(f"Portal ZIP overall: {status}")
    print(f"Portal ZIP checks: {recorder.counts()}")
    if state is not None:
        print(f"Portal ZIP bytes: {state['archive_bytes']}")
        print(f"Portal ZIP SHA-256: {state['archive_sha256']}")
        print(f"Portal ZIP members: {state['member_count']}")
        print(f"Portal ZIP video present: {state['video_present']}")
    for check in recorder.checks:
        print(f"{check.status.upper()}: {check.check_id}: {check.summary}")
    return 0 if is_ready else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--zip-path", type=Path)
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--portal-zip", type=Path)
    parser.add_argument("--require-video", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--output-sha256", type=Path)
    parser.add_argument("--skip-qubo-checks", action="store_true")
    parser.add_argument("--validator-timeout-seconds", type=int, default=120)
    parser.add_argument("--max-scan-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--max-archive-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--max-members", type=int, default=10_000)
    parser.add_argument("--max-member-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-uncompressed-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--max-member-compression-ratio", type=float, default=500.0)
    parser.add_argument("--max-total-compression-ratio", type=float, default=100.0)
    return parser


def resolve_configuration(args: argparse.Namespace) -> dict[str, Path]:
    validate_unlinked_path_components(args.root, label="root")
    root = args.root.resolve()
    deliverables = root / "deliverables"
    raw_package_dir = args.package_dir or deliverables / "final_solution_package"
    raw_zip_path = args.zip_path or deliverables / "TAOLIN_DC_SIM_final_solution_package.zip"
    raw_output_json = args.output_json or deliverables / "SUBMISSION_VERIFICATION.json"
    raw_output_markdown = args.output_markdown or deliverables / "SUBMISSION_VERIFICATION.md"
    raw_output_sha256 = args.output_sha256 or deliverables / "SHA256SUMS.txt"
    for label, path in (
        ("package directory", raw_package_dir),
        ("ZIP", raw_zip_path),
        ("JSON output", raw_output_json),
        ("Markdown output", raw_output_markdown),
        ("checksum output", raw_output_sha256),
    ):
        validate_unlinked_path_components(path, label=label)
    package_dir = raw_package_dir.resolve()
    zip_path = raw_zip_path.resolve()
    output_json = raw_output_json.resolve()
    output_markdown = raw_output_markdown.resolve()
    output_sha256 = raw_output_sha256.resolve()
    outputs = [output_json, output_markdown, output_sha256]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Verification output paths must be distinct")
    if any(is_within(path, package_dir) for path in outputs):
        raise ValueError("Verification outputs must be outside the manifest-bound package")
    if args.validator_timeout_seconds <= 0:
        raise ValueError("Validator timeout must be positive")
    for attribute in (
        "max_scan_bytes",
        "max_archive_bytes",
        "max_members",
        "max_member_bytes",
        "max_uncompressed_bytes",
        "max_member_compression_ratio",
        "max_total_compression_ratio",
    ):
        if getattr(args, attribute) <= 0:
            raise ValueError(f"{attribute} must be positive")
    return {
        "root": root,
        "package_dir": package_dir,
        "zip_path": zip_path,
        "output_json": output_json,
        "output_markdown": output_markdown,
        "output_sha256": output_sha256,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.staging_dir is not None and args.portal_zip is not None:
        parser.error("--staging-dir and --portal-zip are mutually exclusive")
    if args.portal_zip is not None:
        return run_portal_mode(args, parser)
    if args.staging_dir is not None:
        return run_staging_mode(args, parser)
    if args.require_video:
        parser.error("--require-video is valid only with --staging-dir or --portal-zip")
    try:
        paths = resolve_configuration(args)
    except ValueError as error:
        parser.error(str(error))

    recorder = Recorder()
    manifest_state: dict[str, Any] | None = None
    zip_state: dict[str, Any] | None = None
    artifact_summary: dict[str, Any] = {}
    try:
        manifest_state = verify_manifest(paths["package_dir"], recorder)
        zip_state = verify_zip(
            paths["zip_path"],
            paths["package_dir"],
            manifest_state,
            recorder,
            args,
        )
        artifact_summary = run_cleanroom_phase(
            paths["zip_path"],
            zip_state,
            recorder,
            args,
        )
    except Exception as error:  # Final containment: still emit a non-ready report.
        recorder.failed(
            "verifier_internal_error",
            "The verifier encountered an unexpected internal error.",
            error_type=type(error).__name__,
        )

    is_ready = recorder.ready_for(CANONICAL_MANDATORY_CHECKS)
    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "overall_status": "ready" if is_ready else "not_ready",
        "check_counts": recorder.counts(),
        "inputs": {
            "package_directory": path_label(paths["package_dir"], paths["root"]),
            "zip_path": path_label(paths["zip_path"], paths["root"]),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "isolated_validator": True,
            "qubo_checks_skipped_by_request": bool(args.skip_qubo_checks),
        },
        "archive": zip_state or {},
        "artifact_summary": artifact_summary,
        "checks": [asdict(check) for check in recorder.checks],
    }
    manifest_path = paths["package_dir"] / "MANIFEST.json"
    try:
        write_outputs(
            payload,
            paths["output_json"],
            paths["output_markdown"],
            paths["output_sha256"],
            paths["zip_path"],
            manifest_path,
            paths["root"],
        )
    except OSError as error:
        print(f"Failed to write verification outputs: {type(error).__name__}", file=sys.stderr)
        return 2

    print(f"Overall: {payload['overall_status'].upper()}")
    print(f"Checks: {payload['check_counts']}")
    if zip_state:
        print(f"ZIP SHA-256: {zip_state['archive_sha256']}")
    print(f"Report: {path_label(paths['output_json'], paths['root'])}")
    return 0 if is_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
