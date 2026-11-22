"""Strict loaders for the immutable, closed-track CSV inputs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pandas as pd

from .schemas import (
    EXPECTED_RAW_ROWS,
    EXPECTED_RAW_SHA256,
    RAW_FILE_NAMES,
    REQUIRED_COLUMNS,
    RawDataIntegrityError,
)


@dataclass(frozen=True, slots=True)
class LoadedRawData:
    """Raw frames after byte- and schema-level verification."""

    tournaments: pd.DataFrame
    matches: pd.DataFrame
    players: pd.DataFrame
    hashes: dict[str, str]


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return a streaming SHA-256 digest for *path*."""

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_one(raw_dir: Path, table: str) -> tuple[pd.DataFrame, str]:
    path = raw_dir / RAW_FILE_NAMES[table]
    if not path.is_file():
        raise FileNotFoundError(f"Required raw file is missing: {path}")

    actual_hash = file_sha256(path)
    expected_hash = EXPECTED_RAW_SHA256[table]
    if actual_hash != expected_hash:
        raise RawDataIntegrityError(
            f"SHA-256 mismatch for {path.name}: expected {expected_hash}, got {actual_hash}"
        )

    frame = pd.read_csv(path, low_memory=False)
    missing_columns = sorted(set(REQUIRED_COLUMNS[table]).difference(frame.columns))
    if missing_columns:
        raise RawDataIntegrityError(
            f"{path.name} is missing required columns: {', '.join(missing_columns)}"
        )
    if len(frame) != EXPECTED_RAW_ROWS[table]:
        raise RawDataIntegrityError(
            f"Unexpected row count for {path.name}: "
            f"expected {EXPECTED_RAW_ROWS[table]}, got {len(frame)}"
        )
    return frame, actual_hash


def load_raw_data(raw_dir: str | Path) -> LoadedRawData:
    """Load all supplied CSVs after enforcing their frozen hashes and schemas."""

    directory = Path(raw_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Raw data directory does not exist: {directory}")

    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for table in ("tournaments", "matches", "players"):
        frames[table], hashes[table] = _load_one(directory, table)
    return LoadedRawData(
        tournaments=frames["tournaments"],
        matches=frames["matches"],
        players=frames["players"],
        hashes=hashes,
    )
