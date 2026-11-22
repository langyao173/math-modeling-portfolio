"""Point-in-time tournament context and exchange-consistent host/rest features."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

_HOST_ALIASES = {
    "germany": {"germany", "germany fr", "frg"},
    "usa": {"usa", "united states"},
    "south africa": {"south africa"},
    "korea": {"korea republic", "south korea", "korea"},
    "japan": {"japan"},
}


def _normal(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())


def _era(year: int) -> str:
    if year <= 1938:
        return "pre_war"
    if year <= 1958:
        return "post_war_1950s"
    if year <= 1974:
        return "sixteen_team_transition"
    if year <= 1994:
        return "twenty_four_team"
    return "thirty_two_team"


def _format_regime(teams: int) -> str:
    if teams <= 16:
        return "up_to_16"
    if teams <= 24:
        return "17_to_24"
    if teams <= 32:
        return "25_to_32"
    return "above_32"


def _host_names(country: object) -> set[str]:
    normalized = _normal(country)
    pieces = {_normal(part) for part in re.split(r"[/&]", str(country))}
    aliases: set[str] = set()
    for piece in pieces | {normalized}:
        aliases.add(piece)
        aliases.update(_HOST_ALIASES.get(piece, set()))
    return {name for name in aliases if name}


def _is_host(team_name: object, team_initials: object, names: set[str]) -> bool:
    candidates = {_normal(team_name), _normal(team_initials)}
    return bool(candidates & names)


def build_match_context_features(
    matches: pd.DataFrame,
    tournaments: pd.DataFrame,
) -> pd.DataFrame:
    """Return one context row per match without observing the current result.

    Known edition format and fixture order are allowed. Stage scoring baselines use completed
    earlier editions only. Rest snapshots are evaluated before state updates at each timestamp.
    """
    required_matches = {
        "MatchID",
        "Year",
        "Home Team Name",
        "Away Team Name",
        "Home Team Initials",
        "Away Team Initials",
        "canonical_stage",
    }
    required_tournaments = {"Year", "Country", "QualifiedTeams", "MatchesPlayed"}
    missing_matches = sorted(required_matches.difference(matches.columns))
    missing_tournaments = sorted(required_tournaments.difference(tournaments.columns))
    if missing_matches or missing_tournaments:
        raise ValueError(
            f"Context feature inputs are incomplete; matches={missing_matches}, "
            f"tournaments={missing_tournaments}"
        )
    if matches["MatchID"].duplicated().any() or tournaments["Year"].duplicated().any():
        raise ValueError("Match IDs and tournament years must be unique")

    format_table = tournaments[
        ["Year", "Country", "QualifiedTeams", "MatchesPlayed"]
    ].copy()
    result = matches[
        [
            "MatchID",
            "Year",
            "canonical_stage",
            "Home Team Initials",
            "Away Team Initials",
        ]
    ].merge(format_table, on="Year", validate="many_to_one")
    result = result.rename(
        columns={
            "QualifiedTeams": "tournament_teams",
            "MatchesPlayed": "tournament_matches",
        }
    )
    result["matches_per_team"] = (
        2.0 * result["tournament_matches"] / result["tournament_teams"]
    )
    result["competition_era"] = result["Year"].map(lambda value: _era(int(value)))
    result["format_regime"] = result["tournament_teams"].map(
        lambda value: _format_regime(int(value))
    )
    result["is_knockout"] = (~result["canonical_stage"].eq("group_stage")).astype(float)

    times = pd.to_datetime(
        matches["match_datetime"] if "match_datetime" in matches else matches["Datetime"],
        errors="raise",
    )
    fixture = pd.DataFrame(
        {"MatchID": matches["MatchID"], "_time": times, "Year": matches["Year"]}
    ).sort_values(["Year", "_time", "MatchID"], kind="mergesort")
    fixture["tournament_match_ordinal"] = fixture.groupby("Year").cumcount() + 1
    result = result.merge(
        fixture[["MatchID", "_time", "tournament_match_ordinal"]],
        on="MatchID",
        validate="one_to_one",
    )
    result["tournament_progress"] = (
        result["tournament_match_ordinal"] - 1
    ) / result["tournament_matches"].clip(lower=1)

    match_lookup = matches.set_index("MatchID", drop=False)
    host_side: list[str] = []
    for row in result.itertuples(index=False):
        source = match_lookup.loc[row.MatchID]
        names = _host_names(row.Country)
        home_host = _is_host(source["Home Team Name"], source["Home Team Initials"], names)
        away_host = _is_host(source["Away Team Name"], source["Away Team Initials"], names)
        if home_host and away_host:
            host_side.append("neutral")
        elif home_host:
            host_side.append("home")
        elif away_host:
            host_side.append("away")
        else:
            host_side.append("neutral")
    result["host_side"] = host_side
    result["host_mapping_available"] = 1.0

    home_by_id = matches.set_index("MatchID")["Home Team Initials"].astype(str)
    away_by_id = matches.set_index("MatchID")["Away Team Initials"].astype(str)
    last_played: dict[str, pd.Timestamp] = {}
    rest: dict[tuple[object, str], float] = {}
    for timestamp, batch in fixture.groupby("_time", sort=True):
        now = pd.Timestamp(timestamp)
        updates: set[str] = set()
        for identifier in batch["MatchID"]:
            home = home_by_id.loc[identifier]
            away = away_by_id.loc[identifier]
            for side, team in (("home", home), ("away", away)):
                previous = last_played.get(team)
                rest[(identifier, side)] = (
                    float((now - previous).total_seconds() / 86_400.0)
                    if previous is not None
                    else np.nan
                )
                updates.add(team)
        for team in updates:
            last_played[team] = now
    result["home_rest_days"] = [rest[(identifier, "home")] for identifier in result["MatchID"]]
    result["away_rest_days"] = [rest[(identifier, "away")] for identifier in result["MatchID"]]
    result["rest_days_sum"] = result[["home_rest_days", "away_rest_days"]].sum(
        axis=1, min_count=2
    )
    result["rest_days_diff"] = result["home_rest_days"] - result["away_rest_days"]
    result["rest_history_missing"] = result[
        ["home_rest_days", "away_rest_days"]
    ].isna().any(axis=1).astype(float)

    prior_stage: defaultdict[str, list[float]] = defaultdict(list)
    prior_all: list[float] = []
    stage_rate: dict[object, float] = {}
    overall_rate: dict[object, float] = {}
    total_goals = (
        pd.to_numeric(matches["Home Team Goals"], errors="raise")
        + pd.to_numeric(matches["Away Team Goals"], errors="raise")
    )
    target_lookup = pd.DataFrame(
        {
            "MatchID": matches["MatchID"],
            "Year": matches["Year"],
            "stage": matches["canonical_stage"],
            "goals": total_goals,
        }
    )
    for year in sorted(target_lookup["Year"].unique()):
        edition = target_lookup.loc[target_lookup["Year"].eq(year)]
        for row in edition.itertuples(index=False):
            stage_values = prior_stage[str(row.stage)]
            stage_rate[row.MatchID] = float(np.mean(stage_values)) if stage_values else np.nan
            overall_rate[row.MatchID] = float(np.mean(prior_all)) if prior_all else np.nan
        for row in edition.itertuples(index=False):
            prior_stage[str(row.stage)].append(float(row.goals))
            prior_all.append(float(row.goals))
    result["prior_stage_goal_rate"] = result["MatchID"].map(stage_rate)
    result["prior_overall_goal_rate"] = result["MatchID"].map(overall_rate)
    return result.drop(columns=["Country", "_time"]).sort_values("MatchID").reset_index(drop=True)


__all__ = ["build_match_context_features"]
