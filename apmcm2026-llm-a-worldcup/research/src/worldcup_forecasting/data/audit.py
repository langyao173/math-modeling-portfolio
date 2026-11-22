"""Canonical data assembly, reconciliation, lineage, and audit artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from .canonicalize import (
    canonicalize_matches,
    canonicalize_players,
    canonicalize_tournaments,
    clean_team_name,
    trim_string_columns,
)
from .loaders import LoadedRawData, load_raw_data
from .schemas import (
    EXPECTED_CANONICAL_ROWS,
    EXPECTED_RAW_ROWS,
    EXPECTED_RAW_SHA256,
    RAW_FILE_NAMES,
    CanonicalDataBundle,
    RawDataIntegrityError,
)


def _frame_sha256(frame: pd.DataFrame) -> str:
    content = frame.to_csv(index=False, lineterminator="\n", na_rep="<NA>")
    return sha256(content.encode("utf-8")).hexdigest()


def _count_trimmed_cells(frame: pd.DataFrame) -> int:
    trimmed = trim_string_columns(frame)
    return int(
        sum(
            frame[column].fillna("").ne(trimmed[column].fillna("")).sum()
            for column in frame.select_dtypes(include=["object", "string"]).columns
        )
    )


def _build_reconciliation(tournaments: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    aggregates = (
        matches.groupby("Year", sort=True)
        .agg(
            canonical_matches=("MatchID", "size"),
            canonical_goals=("full_time_total_goals", "sum"),
            canonical_attendance=("Attendance", lambda values: values.sum(min_count=1)),
        )
        .reset_index()
    )
    expected = tournaments[["Year", "MatchesPlayed", "GoalsScored", "Attendance"]].rename(
        columns={
            "MatchesPlayed": "tournament_matches",
            "GoalsScored": "tournament_goals",
            "Attendance": "tournament_attendance",
        }
    )
    result = expected.merge(aggregates, on="Year", how="outer", validate="one_to_one")
    result["match_count_delta"] = result["canonical_matches"] - result["tournament_matches"]
    result["goal_delta"] = result["canonical_goals"] - result["tournament_goals"]
    result["attendance_delta"] = result["canonical_attendance"] - result["tournament_attendance"]
    result["matches_match"] = result["match_count_delta"].eq(0)
    result["goals_match"] = result["goal_delta"].eq(0)
    result["attendance_match"] = result["attendance_delta"].eq(0)
    result["all_match"] = result[["matches_match", "goals_match", "attendance_match"]].all(axis=1)
    return result.sort_values("Year").reset_index(drop=True)


def _build_team_registry(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[
        ["Year", "MatchID", "Home Team Name", "home_team_canonical", "Home Team Initials"]
    ].rename(
        columns={
            "Home Team Name": "raw_name",
            "home_team_canonical": "canonical_name",
            "Home Team Initials": "team_initials",
        }
    )
    home["side"] = "home"
    away = matches[
        ["Year", "MatchID", "Away Team Name", "away_team_canonical", "Away Team Initials"]
    ].rename(
        columns={
            "Away Team Name": "raw_name",
            "away_team_canonical": "canonical_name",
            "Away Team Initials": "team_initials",
        }
    )
    away["side"] = "away"
    rows = pd.concat([home, away], ignore_index=True)
    registry = (
        rows.groupby(["raw_name", "canonical_name", "team_initials"], sort=True)
        .agg(
            first_year=("Year", "min"),
            last_year=("Year", "max"),
            match_rows=("MatchID", "size"),
            home_rows=("side", lambda values: int((values == "home").sum())),
            away_rows=("side", lambda values: int((values == "away").sum())),
        )
        .reset_index()
    )
    if not registry["canonical_name"].map(clean_team_name).eq(registry["canonical_name"]).all():
        raise RawDataIntegrityError("Team-name registry is not idempotently canonical")
    return registry


def _build_stage_registry(matches: pd.DataFrame) -> pd.DataFrame:
    return (
        matches.groupby(["Stage", "canonical_stage"], sort=True)
        .agg(
            first_year=("Year", "min"),
            last_year=("Year", "max"),
            match_rows=("MatchID", "size"),
        )
        .reset_index()
        .rename(columns={"Stage": "raw_stage"})
    )


def _table_schema(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing": {column: int(count) for column, count in frame.isna().sum().items()},
        "content_sha256": _frame_sha256(frame),
    }


def _build_schema_report(
    raw: LoadedRawData,
    tournaments: pd.DataFrame,
    matches: pd.DataFrame,
    players: pd.DataFrame,
    reconciliation: pd.DataFrame,
) -> dict[str, Any]:
    raw_frames = {
        "tournaments": raw.tournaments,
        "matches": raw.matches,
        "players": raw.players,
    }
    canonical = {
        "tournaments": tournaments,
        "matches": matches,
        "players": players,
        "reconciliation": reconciliation,
    }
    return {
        "schema_version": 1,
        "raw_tables": {name: _table_schema(frame) for name, frame in raw_frames.items()},
        "canonical_tables": {name: _table_schema(frame) for name, frame in canonical.items()},
        "validations": {
            "raw_hashes_match": raw.hashes == EXPECTED_RAW_SHA256,
            "canonical_counts_match": all(
                len(canonical[name]) == expected
                for name, expected in EXPECTED_CANONICAL_ROWS.items()
            ),
            "match_key_unique": not matches.duplicated(["RoundID", "MatchID"]).any(),
            "player_rows_unique": not players.duplicated().any(),
            "all_years_reconciled": bool(reconciliation["all_match"].all()),
            "unmapped_stage_rows": int((matches["canonical_stage"] == "unknown").sum()),
            "negative_goal_rows": int(
                (
                    matches[
                        [
                            "Home Team Goals",
                            "Away Team Goals",
                            "Half-time Home Goals",
                            "Half-time Away Goals",
                        ]
                    ]
                    < 0
                )
                .any(axis=1)
                .sum()
            ),
            "half_time_exceeds_full_time_rows": int(
                (matches["half_time_total_goals"] > matches["full_time_total_goals"]).sum()
            ),
        },
    }


def _build_duplicate_report(raw: LoadedRawData, bundle_rows: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "table": "tournaments",
                "deduplication_key": "none (Year must already be unique)",
                "raw_rows": len(raw.tournaments),
                "duplicates_removed": 0,
                "canonical_rows": bundle_rows["tournaments"],
                "conflicting_groups": 0,
            },
            {
                "table": "matches",
                "deduplication_key": "RoundID,MatchID",
                "raw_rows": len(raw.matches),
                "duplicates_removed": len(raw.matches) - bundle_rows["matches"],
                "canonical_rows": bundle_rows["matches"],
                "conflicting_groups": 0,
            },
            {
                "table": "players",
                "deduplication_key": "exact full raw row",
                "raw_rows": len(raw.players),
                "duplicates_removed": len(raw.players) - bundle_rows["players"],
                "canonical_rows": bundle_rows["players"],
                "conflicting_groups": 0,
            },
        ]
    )


def build_canonical_data(raw_dir: str | Path) -> CanonicalDataBundle:
    """Build and fully reconcile the canonical closed-track data bundle."""

    raw = load_raw_data(raw_dir)
    tournaments = canonicalize_tournaments(raw.tournaments)
    matches = canonicalize_matches(raw.matches)
    players = canonicalize_players(raw.players, matches)
    frames = {"tournaments": tournaments, "matches": matches, "players": players}
    for name, expected in EXPECTED_CANONICAL_ROWS.items():
        if len(frames[name]) != expected:
            raise RawDataIntegrityError(
                f"Canonical {name} row count mismatch: expected {expected}, got {len(frames[name])}"
            )

    reconciliation = _build_reconciliation(tournaments, matches)
    if len(reconciliation) != 20 or not reconciliation["all_match"].all():
        failed = reconciliation.loc[~reconciliation["all_match"]].to_dict("records")
        raise RawDataIntegrityError(f"Tournament/match reconciliation failed: {failed}")

    duplicate_report = _build_duplicate_report(
        raw, {name: len(frame) for name, frame in frames.items()}
    )
    team_registry = _build_team_registry(matches)
    stage_registry = _build_stage_registry(matches)
    schema_report = _build_schema_report(raw, tournaments, matches, players, reconciliation)
    lineage = {
        "lineage_version": 1,
        "inputs": {
            name: {
                "file": RAW_FILE_NAMES[name],
                "sha256": raw.hashes[name],
                "expected_sha256": EXPECTED_RAW_SHA256[name],
                "rows": EXPECTED_RAW_ROWS[name],
            }
            for name in ("tournaments", "matches", "players")
        },
        "transformations": [
            "verify frozen SHA-256 and required columns",
            "trim leading and trailing whitespace in string cells",
            "parse tournament attendance as grouped integer",
            "deduplicate matches by (RoundID, MatchID), rejecting conflicts",
            "deduplicate player events by exact full raw row",
            "derive canonical stages, names, goal totals, and scoreline outcomes",
            "join tournament Year to player events by (RoundID, MatchID)",
            "reconcile match count, total goals, and attendance for all 20 editions",
        ],
        "trimmed_cells": {
            "tournaments": _count_trimmed_cells(raw.tournaments),
            "matches": _count_trimmed_cells(raw.matches),
            "players": _count_trimmed_cells(raw.players),
        },
        "outputs": {
            name: {
                "rows": int(len(frame)),
                "columns": int(frame.shape[1]),
                "content_sha256": _frame_sha256(frame),
            }
            for name, frame in {
                **frames,
                "reconciliation": reconciliation,
            }.items()
        },
    }
    return CanonicalDataBundle(
        tournaments=tournaments,
        matches=matches,
        players=players,
        reconciliation=reconciliation,
        raw_hashes=dict(raw.hashes),
        schema_report=schema_report,
        duplicate_report=duplicate_report,
        team_name_registry=team_registry,
        stage_registry=stage_registry,
        data_lineage=lineage,
    )


def _missingness_rows(
    tables: Iterable[tuple[str, str, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, table, frame in tables:
        for column in frame.columns:
            missing = int(frame[column].isna().sum())
            rows.append(
                {
                    "scope": scope,
                    "table": table,
                    "column": column,
                    "rows": int(len(frame)),
                    "missing_count": missing,
                    "missing_fraction": missing / len(frame) if len(frame) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    _write_text(path, frame.to_csv(index=False, lineterminator="\n", na_rep=""))


def write_audit_artifacts(
    bundle: CanonicalDataBundle,
    raw_dir: str | Path,
    audit_dir: str | Path,
    registry_dir: str | Path,
) -> dict[str, Path]:
    """Write deterministic audit and registry artifacts after rechecking inputs."""

    raw = load_raw_data(raw_dir)
    if raw.hashes != bundle.raw_hashes:
        raise RawDataIntegrityError("Raw inputs changed after the canonical bundle was built")
    audit = Path(audit_dir).expanduser().resolve()
    registry = Path(registry_dir).expanduser().resolve()
    audit.mkdir(parents=True, exist_ok=True)
    registry.mkdir(parents=True, exist_ok=True)

    missingness = _missingness_rows(
        [
            ("raw", "tournaments", raw.tournaments),
            ("raw", "matches", raw.matches),
            ("raw", "players", raw.players),
            ("canonical", "tournaments", bundle.tournaments),
            ("canonical", "matches", bundle.matches),
            ("canonical", "players", bundle.players),
        ]
    )
    impossible = pd.DataFrame(
        [
            {"check": key, "value": value, "passed": value == 0}
            for key, value in bundle.schema_report["validations"].items()
            if key.endswith("_rows") and isinstance(value, int)
        ]
    )
    paths = {
        "schema_report": audit / "schema_report.json",
        "data_lineage": audit / "data_lineage.json",
        "raw_hashes": audit / "raw_hashes.json",
        "missingness": audit / "missingness.csv",
        "duplicate_report": audit / "duplicate_report.csv",
        "reconciliation": audit / "reconciliation.csv",
        "impossible_values": audit / "impossible_values.csv",
        "audit_team_name_registry": audit / "team_name_registry.csv",
        "team_name_registry": registry / "team_name_registry.csv",
        "stage_registry": registry / "stage_registry.csv",
        "registry_manifest": registry / "registry_manifest.json",
    }
    _write_json(paths["schema_report"], bundle.schema_report)
    _write_json(paths["data_lineage"], bundle.data_lineage)
    _write_json(paths["raw_hashes"], bundle.raw_hashes)
    _write_csv(paths["missingness"], missingness)
    _write_csv(paths["duplicate_report"], bundle.duplicate_report)
    _write_csv(paths["reconciliation"], bundle.reconciliation)
    _write_csv(paths["impossible_values"], impossible)
    _write_csv(paths["audit_team_name_registry"], bundle.team_name_registry)
    _write_csv(paths["team_name_registry"], bundle.team_name_registry)
    _write_csv(paths["stage_registry"], bundle.stage_registry)
    _write_json(
        paths["registry_manifest"],
        {
            "schema_version": 1,
            "raw_data_hashes": bundle.raw_hashes,
            "registries": {
                "team_name_registry.csv": _frame_sha256(bundle.team_name_registry),
                "stage_registry.csv": _frame_sha256(bundle.stage_registry),
            },
        },
    )
    return paths
