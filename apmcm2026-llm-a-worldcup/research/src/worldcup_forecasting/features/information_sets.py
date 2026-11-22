"""Leakage-safe player and coach features for nested prediction information sets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InformationSetBundle:
    """Feature frames and machine-readable availability lineage."""

    pre_match: pd.DataFrame
    kickoff: pd.DataFrame
    lineage: pd.DataFrame

    def __getitem__(self, name: str) -> pd.DataFrame:
        return getattr(self, name)


def _column(frame: pd.DataFrame, *candidates: str) -> str:
    for candidate in candidates:
        if candidate in frame:
            return candidate
    raise ValueError(f"Missing required column; expected one of {candidates}")


def _clean_identity(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _event_counts(value: Any) -> tuple[int, int]:
    """Count historical goal and card tokens without using them before their match."""
    text = "" if pd.isna(value) else str(value).upper()
    tokens = text.replace("'", "").split()
    goals = sum(token.startswith(("G", "P", "W")) for token in tokens)
    cards = sum(token.startswith(("Y", "R")) for token in tokens)
    return goals, cards


def _match_clock(matches: pd.DataFrame, match_id_column: str) -> pd.Series:
    if "Datetime" in matches:
        parsed = pd.to_datetime(matches["Datetime"], errors="coerce")
        if parsed.notna().all():
            return parsed
    if "Date" in matches:
        parsed = pd.to_datetime(matches["Date"], errors="coerce")
        if parsed.notna().all():
            return parsed
    if "Year" in matches:
        year = pd.to_numeric(matches["Year"], errors="raise").astype(int)
        order = pd.to_numeric(matches[match_id_column], errors="raise").rank(method="dense")
        return pd.to_datetime(year.astype(str) + "-01-01") + pd.to_timedelta(order, unit="s")
    order = pd.to_numeric(matches[match_id_column], errors="raise").rank(method="dense")
    return pd.Timestamp("1900-01-01") + pd.to_timedelta(order, unit="s")


def _prepare_coaches(
    players: pd.DataFrame,
    coaches: pd.DataFrame | None,
    match_id_column: str,
    player_team_column: str,
) -> dict[tuple[Any, str], str]:
    source = coaches
    if source is None and "Coach Name" in players:
        source = players[[match_id_column, player_team_column, "Coach Name"]].copy()
    if source is None or source.empty:
        return {}
    coach_match = _column(source, "MatchID", "match_id")
    coach_team = _column(source, "Team Initials", "team", "team_initials")
    coach_name = _column(source, "Coach Name", "coach", "coach_name")
    mapping: dict[tuple[Any, str], str] = {}
    for (match_id, team), group in source.groupby([coach_match, coach_team], dropna=False):
        names = {_clean_identity(value) for value in group[coach_name] if _clean_identity(value)}
        if len(names) > 1:
            raise ValueError(f"Conflicting coaches for match/team key {(match_id, team)}: {names}")
        mapping[(match_id, _clean_identity(team))] = next(iter(names), "")
    return mapping


def build_player_coach_information_sets(
    *,
    players: pd.DataFrame,
    matches: pd.DataFrame,
    coaches: pd.DataFrame | None = None,
    state_update_before_year: int | None = None,
) -> InformationSetBundle:
    """Build strictly prior team-history and kickoff-current-personnel features.

    Current-match event tokens are applied only after every feature snapshot at the same
    timestamp. Consequently, changing goals/cards in a match cannot change that match's inputs.
    Current lineup membership is deliberately isolated to the kickoff frame.
    """
    if players.empty or matches.empty:
        raise ValueError("Player and match tables must both be nonempty")
    match_id = _column(matches, "MatchID", "match_id")
    home_team = _column(matches, "Home Team Initials", "home_team", "home_team_initials")
    away_team = _column(matches, "Away Team Initials", "away_team", "away_team_initials")
    player_match = _column(players, "MatchID", "match_id")
    player_team = _column(players, "Team Initials", "team", "team_initials")
    player_name = _column(players, "Player Name", "player", "player_name")
    lineup_column = _column(players, "Line-up", "lineup", "line_up")

    ordered = matches.copy()
    if ordered[match_id].duplicated().any():
        raise ValueError("Match IDs must be unique when constructing information sets")
    ordered["_prediction_time"] = _match_clock(ordered, match_id)
    ordered = ordered.sort_values(["_prediction_time", match_id], kind="mergesort")

    roster_by_key: dict[tuple[Any, str], pd.DataFrame] = {}
    for (identifier, team), group in players.groupby([player_match, player_team], dropna=False):
        roster_by_key[(identifier, _clean_identity(team))] = group.copy()
    coach_by_key = _prepare_coaches(players, coaches, player_match, player_team)

    team_matches: defaultdict[str, int] = defaultdict(int)
    team_player_appearances: defaultdict[str, int] = defaultdict(int)
    team_player_goals: defaultdict[str, int] = defaultdict(int)
    team_player_cards: defaultdict[str, int] = defaultdict(int)
    player_appearances: defaultdict[tuple[str, str], int] = defaultdict(int)
    player_goals: defaultdict[tuple[str, str], int] = defaultdict(int)
    coach_matches: defaultdict[str, int] = defaultdict(int)
    team_last_coach: dict[str, str] = {}

    pre_rows: list[dict[str, Any]] = []
    kickoff_rows: list[dict[str, Any]] = []

    for _, timestamp_group in ordered.groupby("_prediction_time", sort=True):
        pending_updates: list[tuple[Any, str, pd.DataFrame, str, int | None]] = []
        for _, match_values in timestamp_group.iterrows():
            identifier = match_values[match_id]
            teams = (
                ("home", _clean_identity(match_values[home_team])),
                ("away", _clean_identity(match_values[away_team])),
            )
            pre: dict[str, Any] = {match_id: identifier}
            kickoff: dict[str, Any] = {match_id: identifier}
            if "Year" in ordered:
                pre["Year"] = int(match_values["Year"])
                kickoff["Year"] = int(match_values["Year"])
            for side, team in teams:
                roster = roster_by_key.get((identifier, team), players.iloc[0:0])
                current_names = [_clean_identity(value) for value in roster[player_name]]
                unique_names = tuple(sorted({name for name in current_names if name}))
                starter_names = tuple(
                    sorted(
                        {
                            _clean_identity(row[player_name])
                            for _, row in roster.iterrows()
                            if _clean_identity(row[player_name])
                            and str(row[lineup_column]).strip().upper() == "S"
                        }
                    )
                )
                current_coach = coach_by_key.get((identifier, team), "")

                pre[f"{side}_player_prior_team_matches"] = team_matches[team]
                pre[f"{side}_player_prior_appearances"] = team_player_appearances[team]
                pre[f"{side}_player_prior_goals"] = team_player_goals[team]
                pre[f"{side}_player_prior_cards"] = team_player_cards[team]
                pre[f"{side}_coach_prior_team_matches"] = team_matches[team]

                kickoff[f"{side}_current_roster_count"] = len(unique_names)
                kickoff[f"{side}_current_lineup_starters"] = len(starter_names)
                kickoff[f"{side}_current_lineup_prior_appearances"] = sum(
                    player_appearances[(team, name)] for name in starter_names
                )
                kickoff[f"{side}_lineup_historical_scoring_count"] = sum(
                    player_goals[(team, name)] for name in starter_names
                )
                kickoff[f"{side}_current_lineup_continuity"] = (
                    float(
                        np.mean(
                            [player_appearances[(team, name)] > 0 for name in starter_names]
                        )
                    )
                    if starter_names
                    else 0.0
                )
                kickoff[f"{side}_current_coach_experience_matches"] = coach_matches[
                    current_coach
                ]
                kickoff[f"{side}_current_coach_continuity"] = float(
                    bool(current_coach) and team_last_coach.get(team) == current_coach
                )
                kickoff[f"{side}_current_personnel_missing"] = float(
                    not unique_names or not current_coach
                )
                match_year = int(match_values["Year"]) if "Year" in ordered else None
                pending_updates.append(
                    (identifier, team, roster, current_coach, match_year)
                )

            kickoff.update(pre)
            pre_rows.append(pre)
            kickoff_rows.append(kickoff)

        # Predict all equal-time matches before any state update.
        for _, team, roster, current_coach, match_year in pending_updates:
            if (
                state_update_before_year is not None
                and match_year is not None
                and match_year >= int(state_update_before_year)
            ):
                continue
            team_matches[team] += 1
            seen_in_match: set[str] = set()
            for _, row in roster.iterrows():
                name = _clean_identity(row[player_name])
                if not name or name in seen_in_match:
                    continue
                seen_in_match.add(name)
                goals, cards = _event_counts(row.get("Event", ""))
                player_appearances[(team, name)] += 1
                player_goals[(team, name)] += goals
                team_player_appearances[team] += 1
                team_player_goals[team] += goals
                team_player_cards[team] += cards
            if current_coach:
                coach_matches[current_coach] += 1
                team_last_coach[team] = current_coach

    pre_frame = pd.DataFrame(pre_rows)
    kickoff_frame = pd.DataFrame(kickoff_rows)
    lineage_rows: list[dict[str, str]] = []
    for column_name in pre_frame.columns:
        if column_name in {match_id, "Year"}:
            continue
        lineage_rows.append(
            {
                "feature_name": column_name,
                "prediction_information_set": "pre_match",
                "available_time": "strictly_before_prediction",
                "source_table": "players_or_coaches_history",
                "aggregation_cutoff": "prediction_time_exclusive",
            }
        )
    for column_name in kickoff_frame.columns:
        if column_name in set(pre_frame.columns):
            continue
        lineage_rows.append(
            {
                "feature_name": column_name,
                "prediction_information_set": "kickoff",
                "available_time": "declared_at_kickoff",
                "source_table": "current_lineup_or_coach_identity",
                "aggregation_cutoff": "current_identity_prior_events_only",
            }
        )
    return InformationSetBundle(
        pre_match=pre_frame,
        kickoff=kickoff_frame,
        lineage=pd.DataFrame(lineage_rows),
    )
