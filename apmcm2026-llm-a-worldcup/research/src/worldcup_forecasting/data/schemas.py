"""Typed contracts and frozen schemas for the supplied World Cup data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

RAW_FILE_NAMES: dict[str, str] = {
    "tournaments": "WorldCups_preview.csv",
    "matches": "WorldCupMatches.csv",
    "players": "WorldCupPlayers.csv",
}

EXPECTED_RAW_SHA256: dict[str, str] = {
    "tournaments": "a66240afe2ce2d7df5b701eebf012052193b90f33e003c9874cd3df85941003e",
    "matches": "cb4e68bb7e080a84a82175151a2c1ce5ef4ef43fcb5111a3cf6c53f69bc92363",
    "players": "753a68af0cd4fa38b67c7883c138700dfb37544f1ed898b1d176a03f92748d67",
}

EXPECTED_RAW_ROWS: dict[str, int] = {
    "tournaments": 20,
    "matches": 852,
    "players": 37_784,
}

EXPECTED_CANONICAL_ROWS: dict[str, int] = {
    "tournaments": 20,
    "matches": 836,
    "players": 37_048,
}

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "tournaments": (
        "Year",
        "Country",
        "Winner",
        "Runners-Up",
        "Third",
        "Fourth",
        "GoalsScored",
        "QualifiedTeams",
        "MatchesPlayed",
        "Attendance",
    ),
    "matches": (
        "Year",
        "Datetime",
        "Stage",
        "Stadium",
        "City",
        "Home Team Name",
        "Home Team Goals",
        "Away Team Goals",
        "Away Team Name",
        "Win conditions",
        "Attendance",
        "Half-time Home Goals",
        "Half-time Away Goals",
        "Referee",
        "Assistant 1",
        "Assistant 2",
        "RoundID",
        "MatchID",
        "Home Team Initials",
        "Away Team Initials",
    ),
    "players": (
        "RoundID",
        "MatchID",
        "Team Initials",
        "Coach Name",
        "Line-up",
        "Shirt Number",
        "Player Name",
        "Position",
        "Event",
    ),
}


class RawDataIntegrityError(ValueError):
    """Raised when supplied raw data differs from the frozen closed-track data."""


@dataclass(slots=True)
class CanonicalDataBundle:
    """All canonical tables and the evidence needed to audit their construction."""

    tournaments: pd.DataFrame
    matches: pd.DataFrame
    players: pd.DataFrame
    reconciliation: pd.DataFrame
    raw_hashes: dict[str, str] = field(default_factory=dict)
    schema_report: dict[str, Any] = field(default_factory=dict)
    duplicate_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_name_registry: pd.DataFrame = field(default_factory=pd.DataFrame)
    stage_registry: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_lineage: dict[str, Any] = field(default_factory=dict)
