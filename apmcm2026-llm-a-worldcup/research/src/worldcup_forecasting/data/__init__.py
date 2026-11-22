"""Data loading, canonicalization, and auditing."""

from .audit import build_canonical_data, write_audit_artifacts
from .canonicalize import canonical_stage, clean_team_name, parse_attendance
from .schemas import CanonicalDataBundle, RawDataIntegrityError

__all__ = [
    "CanonicalDataBundle",
    "RawDataIntegrityError",
    "build_canonical_data",
    "canonical_stage",
    "clean_team_name",
    "parse_attendance",
    "write_audit_artifacts",
]
