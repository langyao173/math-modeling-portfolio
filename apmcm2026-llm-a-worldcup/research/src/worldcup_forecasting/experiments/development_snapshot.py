"""Hash-verified development-only feature snapshot.

The supplied source CSVs combine development and audit editions.  This module
does not pretend that historical human or legacy-code access can be undone.
It creates an enforceable software boundary for the formal advanced rerun:
after bootstrap, pre-audit stages load only the <=2010 feature snapshot and do
not deserialize it until all bound file hashes have been checked.
"""

from __future__ import annotations

import hashlib
import io
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from worldcup_forecasting.experiments._nested_selection_types import frame_content_hash

_REQUIRED_ARTIFACTS = (
    "development_features.joblib",
    "tournaments_through_2010.csv",
)


@dataclass(frozen=True)
class DevelopmentSnapshot:
    """Verified model-development inputs and their immutable identity."""

    features: pd.DataFrame
    tournaments: pd.DataFrame
    feature_schema_hash: str
    development_frame_hash: str
    manifest_sha256: str
    metadata: dict[str, Any]


def load_development_snapshot(snapshot_dir: str | Path) -> DevelopmentSnapshot:
    """Verify exact bytes before loading the development-only joblib payload."""
    directory = Path(snapshot_dir).expanduser().absolute()
    _reject_reparse_point(directory, "development snapshot directory")
    manifest_path = directory / "snapshot_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Development snapshot manifest is missing: {manifest_path}")
    _reject_reparse_point(manifest_path, "development snapshot manifest")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    _validate_manifest_shape(manifest)
    records = manifest["artifacts"]
    if tuple(record["path"] for record in records) != _REQUIRED_ARTIFACTS:
        raise ValueError("Development snapshot artifact inventory is not exact")
    verified_payloads: dict[str, bytes] = {}
    for record in records:
        path = _safe_child(directory, str(record["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"Development snapshot artifact is missing: {path}")
        payload = path.read_bytes()
        if len(payload) != int(record["bytes"]):
            raise ValueError(f"Development snapshot size mismatch: {path.name}")
        if hashlib.sha256(payload).hexdigest() != str(record["sha256"]):
            raise ValueError(f"Development snapshot hash mismatch: {path.name}")
        verified_payloads[str(record["path"])] = payload

    # Consumers deserialize the same byte strings that passed the manifest checks.
    features = joblib.load(io.BytesIO(verified_payloads["development_features.joblib"]))
    tournaments = pd.read_csv(
        io.BytesIO(verified_payloads["tournaments_through_2010.csv"])
    )
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise TypeError("Development feature snapshot must be a nonempty DataFrame")
    if int(manifest["development_rows"]) != len(features) or len(features) != 772:
        raise ValueError("Development feature snapshot must contain the exact 772 manifested rows")
    years = pd.to_numeric(features["Year"], errors="raise")
    if int(years.min()) != 1930 or int(years.max()) != 2010 or bool(years.gt(2010).any()):
        raise ValueError("Development feature snapshot must contain exactly 1930-2010 editions")
    if features["MatchID"].isna().any() or features["MatchID"].duplicated().any():
        raise ValueError("Development feature snapshot requires unique nonmissing MatchIDs")
    observed_hash = frame_content_hash(features)
    if observed_hash != manifest["development_frame_hash"]:
        raise ValueError("Development feature content no longer matches the manifest")
    tournament_years = pd.to_numeric(tournaments["Year"], errors="raise")
    if int(tournament_years.min()) != 1930 or int(tournament_years.max()) != 2010:
        raise ValueError("Development tournament snapshot must contain 1930-2010 only")
    if len(tournaments) != 19 or tournaments["Year"].duplicated().any():
        raise ValueError("Development tournament snapshot must contain 19 unique editions")
    if bool(manifest["audit_outcomes_used"]):
        raise ValueError("Development snapshot cannot authorize audit-outcome use")
    return DevelopmentSnapshot(
        features=features,
        tournaments=tournaments,
        feature_schema_hash=str(manifest["feature_schema_hash"]),
        development_frame_hash=observed_hash,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        metadata=manifest,
    )


def build_snapshot_manifest(
    directory: Path,
    *,
    feature_schema_hash: str,
    development_frame_hash: str,
    source_raw_hashes: dict[str, str],
    development_rows: int,
) -> dict[str, Any]:
    """Build the exact bootstrap manifest after snapshot artifacts are written."""
    records = []
    for relative in _REQUIRED_ARTIFACTS:
        path = _safe_child(directory.resolve(), relative)
        if not path.is_file():
            raise FileNotFoundError(f"Cannot manifest missing snapshot artifact: {path}")
        records.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "snapshot_role": "formal_advanced_development_inputs_through_2010",
        "cutoff_year": 2010,
        "audit_year": 2014,
        "development_rows": int(development_rows),
        "feature_schema_hash": str(feature_schema_hash),
        "development_frame_hash": str(development_frame_hash),
        "source_raw_hashes": dict(sorted(source_raw_hashes.items())),
        "artifacts": records,
        "audit_outcomes_used": False,
        "physical_raw_no_read_claim": False,
        "formal_preaudit_reads_combined_raw": False,
        "origin_disclosure": (
            "Bootstrap was derived once from the combined supplied tables after prior human and "
            "legacy-code access; it creates a prospective software boundary but cannot restore "
            "researcher blinding or prove historical byte-level non-access."
        ),
    }


def _validate_manifest_shape(manifest: Any) -> None:
    expected = {
        "schema_version",
        "snapshot_role",
        "cutoff_year",
        "audit_year",
        "development_rows",
        "feature_schema_hash",
        "development_frame_hash",
        "source_raw_hashes",
        "artifacts",
        "audit_outcomes_used",
        "physical_raw_no_read_claim",
        "formal_preaudit_reads_combined_raw",
        "origin_disclosure",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise ValueError("Development snapshot manifest keys are not exact")
    if manifest["schema_version"] != 1 or manifest["cutoff_year"] != 2010:
        raise ValueError("Unsupported development snapshot version/cutoff")
    if (
        isinstance(manifest["development_rows"], bool)
        or not isinstance(manifest["development_rows"], int)
        or manifest["development_rows"] != 772
    ):
        raise ValueError("Development snapshot manifest must declare exactly 772 rows")
    if manifest["audit_year"] != 2014 or manifest["physical_raw_no_read_claim"] is not False:
        raise ValueError("Development snapshot boundary disclosure is invalid")
    if manifest["formal_preaudit_reads_combined_raw"] is not False:
        raise ValueError("Formal pre-audit must not read the combined raw tables")
    for label in ("feature_schema_hash", "development_frame_hash"):
        value = str(manifest[label])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"Invalid SHA-256 in development manifest: {label}")
    if not isinstance(manifest["artifacts"], list) or len(manifest["artifacts"]) != 2:
        raise ValueError("Development snapshot must bind exactly two artifacts")
    for record in manifest["artifacts"]:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise ValueError("Development snapshot artifact record is malformed")
        digest = str(record["sha256"])
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Development snapshot artifact hash is malformed")


def _safe_child(directory: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative:
        raise ValueError(f"Unsafe development snapshot path: {relative!r}")
    lexical = (directory / candidate).absolute()
    resolved = lexical.resolve()
    if resolved.parent != directory.resolve():
        raise ValueError(f"Development snapshot path escapes its directory: {relative!r}")
    _reject_reparse_point(lexical, f"development snapshot artifact {relative!r}")
    return lexical


def _reject_reparse_point(path: Path, label: str) -> None:
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError as error:
        raise ValueError(f"Cannot inspect {label}: {path}") from error
    is_junction = getattr(path, "is_junction", None)
    if (
        path.is_symlink()
        or bool(callable(is_junction) and is_junction())
        or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    ):
        raise ValueError(f"Reparse-point {label} is forbidden: {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["DevelopmentSnapshot", "build_snapshot_manifest", "load_development_snapshot"]
