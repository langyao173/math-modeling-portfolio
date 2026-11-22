"""Small schema adapters shared by point-in-time feature builders.

The audited data layer deliberately retains the supplied competition column names.
Tests and research notebooks, however, often use snake_case fixtures.  Feature code
accepts both forms but always keys teams by their historical three-letter initials.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

_INTEGER_ID = re.compile(r"^[+-]?\d+(?:\.0+)?$")

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "year": ("Year", "year", "tournament_year"),
    "match_id": ("MatchID", "match_id", "matchId"),
    "round_id": ("RoundID", "round_id"),
    "datetime": (
        "match_datetime",
        "Datetime",
        "datetime",
        "timestamp",
        "prediction_time",
        "date",
    ),
    "home_team": (
        "Home Team Initials",
        "home_team_initials",
        "home_initials",
        "home_team",
    ),
    "away_team": (
        "Away Team Initials",
        "away_team_initials",
        "away_initials",
        "away_team",
    ),
    "home_name": ("Home Team Name", "home_team_name", "home_name"),
    "away_name": ("Away Team Name", "away_team_name", "away_name"),
    "home_goals": (
        "Home Team Goals",
        "home_team_goals",
        "home_goals",
        "full_time_home_goals",
    ),
    "away_goals": (
        "Away Team Goals",
        "away_team_goals",
        "away_goals",
        "full_time_away_goals",
    ),
    "stage": ("canonical_stage", "stage", "Stage"),
    "win_conditions": ("Win conditions", "win_conditions"),
    "team": ("Team Initials", "team_initials", "team"),
    "player": ("Player Name", "player_name", "player"),
    "lineup": ("Line-up", "line_up", "lineup"),
    "event": ("Event", "event"),
    "coach": ("Coach Name", "coach_name", "coach"),
}


def find_column(
    frame: pd.DataFrame,
    logical_name: str,
    *,
    required: bool = True,
    extra_aliases: Sequence[str] = (),
) -> str | None:
    """Return the first available alias for a logical column."""

    choices = (*COLUMN_ALIASES.get(logical_name, (logical_name,)), *extra_aliases)
    for column in choices:
        if column in frame.columns:
            return column
    if required:
        raise ValueError(
            f"Missing required {logical_name!r} column; expected one of {list(choices)!r}."
        )
    return None


def numeric_series(frame: pd.DataFrame, logical_name: str) -> pd.Series:
    """Read a required numeric column and reject incomplete match outcomes."""

    column = find_column(frame, logical_name)
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        bad = frame.index[values.isna()].tolist()[:5]
        raise ValueError(f"{logical_name} contains missing/non-numeric values at rows {bad}.")
    return values.astype(float)


def team_series(frame: pd.DataFrame, logical_name: str) -> pd.Series:
    """Read and normalize a team-initial column without merging identities."""

    column = find_column(frame, logical_name)
    values = frame[column].astype("string").str.strip().str.upper()
    invalid = values.isna() | values.eq("")
    if invalid.any():
        bad = frame.index[invalid].tolist()[:5]
        raise ValueError(f"{logical_name} contains missing team initials at rows {bad}.")
    return values.astype(str)


def match_times(matches: pd.DataFrame) -> pd.Series:
    """Parse match timestamps, failing closed when a time is unavailable."""

    column = find_column(matches, "datetime")
    parsed = pd.to_datetime(matches[column], errors="coerce")
    if parsed.isna().any():
        bad = matches.index[parsed.isna()].tolist()[:5]
        raise ValueError(f"Match datetime is missing or invalid at rows {bad}.")
    return parsed


def base_match_frame(matches: pd.DataFrame, *, include_targets: bool = True) -> pd.DataFrame:
    """Create a stable, input-order-aligned match frame used by all builders."""

    if not isinstance(matches, pd.DataFrame) or matches.empty:
        raise ValueError("matches must be a non-empty pandas DataFrame.")

    years = pd.to_numeric(matches[find_column(matches, "year")], errors="coerce")
    if years.isna().any():
        raise ValueError("Match year contains missing or non-numeric values.")
    match_id_column = find_column(matches, "match_id")
    match_ids = matches[match_id_column]
    normalized_match_ids = match_ids.map(canonical_match_id)
    if normalized_match_ids.eq("").any() or normalized_match_ids.duplicated().any():
        raise ValueError("Canonical matches require one unique, non-missing MatchID per row.")

    home = team_series(matches, "home_team")
    away = team_series(matches, "away_team")
    if home.eq(away).any():
        raise ValueError("A canonical match cannot have identical home and away initials.")

    time_column = find_column(matches, "datetime", required=False)
    result = pd.DataFrame(
        {
            "MatchID": match_ids.to_numpy(copy=True),
            "Year": years.astype(int).to_numpy(),
            "home_team_initials": home.to_numpy(),
            "away_team_initials": away.to_numpy(),
            # Raw-name aliases make the output easy to join to the audited table.
            "Home Team Initials": home.to_numpy(),
            "Away Team Initials": away.to_numpy(),
        },
        index=matches.index.copy(),
    )
    if time_column is not None:
        result["Datetime"] = pd.to_datetime(matches[time_column], errors="coerce").to_numpy()

    if include_targets:
        result = add_targets(result, matches)
    return result


def add_targets(base: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Attach the shared goal and outcome targets to an aligned feature frame."""

    home_goals = numeric_series(matches, "home_goals")
    away_goals = numeric_series(matches, "away_goals")
    outcome = np.select(
        [home_goals > away_goals, home_goals < away_goals],
        ["home_win", "away_win"],
        default="draw",
    )
    outcome_code = np.sign(home_goals - away_goals).astype(int)
    result = base.copy()
    result["target_home_goals"] = home_goals.to_numpy()
    result["target_away_goals"] = away_goals.to_numpy()
    result["target_total_goals"] = (home_goals + away_goals).to_numpy()
    result["target_goal_difference"] = (home_goals - away_goals).to_numpy()
    result["target_outcome"] = outcome
    result["target_outcome_code"] = outcome_code.to_numpy()
    return result


def normalize_label(value: object) -> str:
    """Normalize a display name only for within-edition podium lookup."""

    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().casefold().split())


def canonical_match_id(value: object) -> str:
    """Normalize numeric-equivalent IDs before uniqueness checks and joins."""

    if pd.isna(value):
        return ""
    text = str(value).strip()
    if _INTEGER_ID.fullmatch(text):
        integer_text = text.split(".", maxsplit=1)[0]
        return str(int(integer_text))
    return text


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise one concise error for missing concrete columns."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing!r}.")
