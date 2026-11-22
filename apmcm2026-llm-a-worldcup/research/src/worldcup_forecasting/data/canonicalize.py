"""Deterministic canonicalization rules for the supplied World Cup tables."""

from __future__ import annotations

import html
import re
import unicodedata
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from .schemas import RawDataIntegrityError

_TEAM_NAME_FIXES = {
    "C�te d'Ivoire": "Côte d'Ivoire",
}


def parse_attendance(value: Any) -> int | None:
    """Parse integer attendance, treating dots/commas as thousands separators.

    The source uses values such as ``590.549`` and ``1.045.246``.  A separator
    is accepted only when it forms valid groups of three, so an ambiguous
    decimal value cannot be silently rounded into a crowd count.
    """

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Boolean attendance is invalid")
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"Attendance must be a finite integer: {value!r}")
        result = int(numeric)
    else:
        text = str(value).strip()
        if not text:
            return None
        text = text.replace("\u00a0", " ")
        if re.fullmatch(r"[+-]?\d+", text):
            result = int(text)
        else:
            grouped = re.fullmatch(r"([+-]?\d{1,3})(?:[.,' ](\d{3}))+", text)
            if grouped is None:
                raise ValueError(f"Unrecognized attendance value: {value!r}")
            result = int(re.sub(r"[.,' ]", "", text))
    if result < 0:
        raise ValueError(f"Attendance cannot be negative: {value!r}")
    return result


def clean_team_name(value: Any) -> str:
    """Remove source markup/mojibake and normalize a team name without aliasing history."""

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    name = html.unescape(str(value)).strip()
    name = re.sub(r'^rn["\'>]+', "", name, flags=re.IGNORECASE)
    name = re.sub(r"<[^>]*>", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = _TEAM_NAME_FIXES.get(name, name)
    return unicodedata.normalize("NFC", name)


def canonical_stage(stage: Any, year: int) -> str:
    """Map era-specific stage labels to the six shared competition stages."""

    del year  # The explicit year is retained in the public contract for era extensions.
    if stage is None or (not isinstance(stage, str) and pd.isna(stage)):
        return "unknown"
    normalized = re.sub(r"[_\-]+", " ", str(stage).strip().casefold())
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized.startswith("group ") or normalized in {"group", "group stage"}:
        return "group_stage"
    if normalized in {"preliminary round", "first round", "round of 16", "last 16"}:
        return "round_of_16"
    if normalized in {"quarter final", "quarter finals", "quarterfinals"}:
        return "quarter_final"
    if normalized in {"semi final", "semi finals", "semifinals"}:
        return "semi_final"
    if normalized in {
        "match for third place",
        "play off for third place",
        "third place",
        "third place play off",
    }:
        return "third_place"
    if normalized == "final":
        return "final"
    return "unknown"


def trim_string_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with leading/trailing whitespace removed from string cells."""

    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return result


def _require_nonnegative(frame: pd.DataFrame, columns: tuple[str, ...], table: str) -> None:
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="raise")
        invalid = numeric.notna() & ((numeric < 0) | (numeric % 1 != 0))
        if invalid.any():
            raise RawDataIntegrityError(
                f"{table}.{column} contains negative or non-integral values"
            )


def canonicalize_tournaments(raw: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize one-row-per-edition tournament data."""

    tournaments = trim_string_columns(raw)
    tournaments["Attendance_raw"] = tournaments["Attendance"]
    parsed = tournaments["Attendance_raw"].map(parse_attendance)
    if parsed.isna().any():
        raise RawDataIntegrityError("Tournament attendance contains missing values")
    tournaments["Attendance"] = parsed.astype("int64")
    _require_nonnegative(
        tournaments,
        ("GoalsScored", "QualifiedTeams", "MatchesPlayed", "Attendance"),
        "tournaments",
    )
    if tournaments["Year"].duplicated().any():
        raise RawDataIntegrityError("Tournament years must be unique")
    if (tournaments["MatchesPlayed"] <= 0).any():
        raise RawDataIntegrityError("MatchesPlayed must be positive")
    tournaments["goals_per_match"] = tournaments["GoalsScored"] / tournaments["MatchesPlayed"]
    return tournaments.sort_values("Year", kind="stable").reset_index(drop=True)


def canonicalize_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """Trim, verify, de-duplicate, and enrich match rows."""

    matches = trim_string_columns(raw)
    keys = ["RoundID", "MatchID"]
    duplicate_rows = matches.loc[matches.duplicated(keys, keep=False)]
    if not duplicate_rows.empty:
        conflicts = [
            key
            for key, group in duplicate_rows.groupby(keys, sort=False, dropna=False)
            if len(group.drop_duplicates()) != 1
        ]
        if conflicts:
            raise RawDataIntegrityError(
                f"Conflicting duplicate match keys cannot be resolved safely: {conflicts[:5]}"
            )
    matches = matches.drop_duplicates(keys, keep="first").copy()
    _require_nonnegative(
        matches,
        (
            "Home Team Goals",
            "Away Team Goals",
            "Half-time Home Goals",
            "Half-time Away Goals",
            "Attendance",
        ),
        "matches",
    )
    attendance = pd.to_numeric(matches["Attendance"], errors="raise")
    matches["Attendance"] = attendance.round().astype("Int64")
    matches["match_datetime"] = pd.to_datetime(matches["Datetime"], errors="raise")
    matches["canonical_stage"] = [
        canonical_stage(stage, int(year))
        for stage, year in matches[["Stage", "Year"]].itertuples(index=False)
    ]
    if (matches["canonical_stage"] == "unknown").any():
        labels = sorted(matches.loc[matches["canonical_stage"] == "unknown", "Stage"].unique())
        raise RawDataIntegrityError(f"Unmapped stage labels: {labels}")
    matches["home_team_canonical"] = matches["Home Team Name"].map(clean_team_name)
    matches["away_team_canonical"] = matches["Away Team Name"].map(clean_team_name)
    matches["full_time_total_goals"] = (
        matches["Home Team Goals"] + matches["Away Team Goals"]
    ).astype("int64")
    matches["half_time_total_goals"] = (
        matches["Half-time Home Goals"] + matches["Half-time Away Goals"]
    ).astype("int64")
    if (matches["half_time_total_goals"] > matches["full_time_total_goals"]).any():
        raise RawDataIntegrityError("Half-time goals cannot exceed full-time goals")
    outcome = np.select(
        [
            matches["Home Team Goals"] > matches["Away Team Goals"],
            matches["Home Team Goals"] < matches["Away Team Goals"],
        ],
        ["home_win", "away_win"],
        default="draw",
    )
    matches["scoreline_outcome"] = outcome
    matches["scoreline_result"] = outcome
    return matches.sort_values(["Year", "match_datetime", "MatchID"], kind="stable").reset_index(
        drop=True
    )


def canonicalize_players(raw: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Remove exact player-event duplicates and attach the tournament year."""

    players = trim_string_columns(raw)
    source_columns = list(players.columns)
    players = players.drop_duplicates(subset=source_columns, keep="first").copy()
    match_year = matches[["RoundID", "MatchID", "Year"]]
    players = players.merge(
        match_year,
        on=["RoundID", "MatchID"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if players["Year"].isna().any():
        missing = players.loc[players["Year"].isna(), ["RoundID", "MatchID"]]
        missing_keys = missing.drop_duplicates().head().to_dict("records")
        raise RawDataIntegrityError(f"Player rows refer to unknown matches: {missing_keys}")
    players["Year"] = players["Year"].astype("int64")
    if players.duplicated().any():
        raise RawDataIntegrityError("Canonical player rows must be exact-row unique")
    return players.reset_index(drop=True)
