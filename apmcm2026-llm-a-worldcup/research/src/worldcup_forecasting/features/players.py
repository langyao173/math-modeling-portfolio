"""Point-in-time roster and player World Cup history features."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._schema import (
    base_match_frame,
    canonical_match_id,
    find_column,
    match_times,
    team_series,
)

_GOAL_PATTERN = re.compile(r"(?:^|\s)[GP]\d", flags=re.IGNORECASE)
_CARD_PATTERN = re.compile(r"(?:^|\s)[YR]\d", flags=re.IGNORECASE)
_PLAYER_FEATURES = (
    "player_roster_size",
    "player_known_history_count",
    "player_history_missing_count",
    "player_history_missing_rate",
    "player_prior_appearances_mean",
    "player_prior_starts_mean",
    "player_prior_goals_mean",
    "player_prior_cards_mean",
    "player_lineup_continuity_rate",
)


@dataclass
class _PlayerState:
    appearances: int = 0
    starts: int = 0
    goals: int = 0
    cards: int = 0
    last_time: pd.Timestamp | None = None


@dataclass(frozen=True)
class _RosterEntry:
    player_key: str
    appeared: int
    start: int
    goals: int
    cards: int


def build_player_history_features(
    players: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate current rosters' histories using only strictly earlier matches.

    Player identity is scoped by historical team initials.  Current-match roster
    membership and starting status are treated as information available at kickoff;
    current-match goals/cards are targets for the state update and never enter that
    match's feature row.  Missing roster data and unseen-player history are exposed
    explicitly rather than silently imputed.
    """

    if not isinstance(players, pd.DataFrame):
        raise TypeError("players must be a pandas DataFrame.")
    base = base_match_frame(matches, include_targets=True)
    times = match_times(matches)
    home = team_series(matches, "home_team").to_numpy()
    away = team_series(matches, "away_team").to_numpy()
    rosters = _build_rosters(players)

    n_rows = len(matches)
    values = {
        f"{side}_{name}": np.zeros(n_rows, dtype=float)
        for side in ("home", "away")
        for name in _PLAYER_FEATURES
    }
    missing_values = {
        f"{side}_{name}": np.zeros(n_rows, dtype=np.int8)
        for side in ("home", "away")
        for name in ("player_roster_missing", "player_prior_history_missing")
    }
    cutoff_values: dict[str, list[pd.Timestamp | pd.NaT]] = {
        "home_player_history_cutoff_time": [pd.NaT] * n_rows,
        "away_player_history_cutoff_time": [pd.NaT] * n_rows,
    }

    player_states: dict[tuple[str, str], _PlayerState] = {}
    previous_rosters: dict[str, set[str]] = {}
    team_cutoffs: dict[str, pd.Timestamp] = {}
    match_keys = [canonical_match_id(value) for value in base["MatchID"]]
    order = pd.DataFrame(
        {
            "position": np.arange(n_rows),
            "time": times.to_numpy(),
            "match_id": np.asarray(match_keys),
        }
    ).sort_values(["time", "match_id", "position"], kind="mergesort")

    for current_time, batch in order.groupby("time", sort=True):
        timestamp = pd.Timestamp(current_time)
        positions = batch["position"].to_numpy(dtype=int)

        for position in positions:
            match_key = match_keys[position]
            for side, team_value in (("home", home[position]), ("away", away[position])):
                team = str(team_value)
                current_roster = rosters.get((match_key, team), ())
                _write_snapshot(
                    position,
                    side,
                    team,
                    current_roster,
                    player_states,
                    previous_rosters,
                    team_cutoffs,
                    values,
                    missing_values,
                    cutoff_values,
                )

        # Incorporate all event rows only after every match at this timestamp has
        # received its features.  This also makes tied synthetic fixtures invariant
        # to their input ordering.
        pending_rosters: dict[str, set[str]] = {}
        for position in positions:
            match_key = match_keys[position]
            for team_value in (home[position], away[position]):
                team = str(team_value)
                current_roster = rosters.get((match_key, team), ())
                if not current_roster:
                    continue
                participants = [entry for entry in current_roster if entry.appeared]
                if not participants:
                    continue
                pending_rosters.setdefault(team, set()).update(
                    entry.player_key for entry in participants
                )
                for entry in participants:
                    key = (team, entry.player_key)
                    state = player_states.setdefault(key, _PlayerState())
                    state.appearances += 1
                    state.starts += entry.start
                    state.goals += entry.goals
                    state.cards += entry.cards
                    state.last_time = timestamp
        for team, roster in pending_rosters.items():
            previous_rosters[team] = roster
            team_cutoffs[team] = timestamp

    result = base.copy()
    for column, column_values in values.items():
        result[column] = column_values
    for column, column_values in missing_values.items():
        result[column] = column_values
    for column, column_values in cutoff_values.items():
        result[column] = pd.to_datetime(column_values)
    for name in _PLAYER_FEATURES:
        result[f"{name}_diff"] = result[f"home_{name}"] - result[f"away_{name}"]
    return result.reset_index(drop=True)


def _write_snapshot(
    position: int,
    side: str,
    team: str,
    roster: tuple[_RosterEntry, ...],
    player_states: dict[tuple[str, str], _PlayerState],
    previous_rosters: dict[str, set[str]],
    team_cutoffs: dict[str, pd.Timestamp],
    values: dict[str, np.ndarray],
    missing_values: dict[str, np.ndarray],
    cutoff_values: dict[str, list[pd.Timestamp | pd.NaT]],
) -> None:
    roster_size = len(roster)
    states = [player_states.get((team, entry.player_key)) for entry in roster]
    known = [state for state in states if state is not None and state.appearances > 0]
    known_count = len(known)
    missing_count = roster_size - known_count

    values[f"{side}_player_roster_size"][position] = roster_size
    values[f"{side}_player_known_history_count"][position] = known_count
    values[f"{side}_player_history_missing_count"][position] = missing_count
    values[f"{side}_player_history_missing_rate"][position] = (
        missing_count / roster_size if roster_size else 1.0
    )
    # Means use the full current roster as denominator; unseen players contribute
    # zero prior events, while the explicit missing rate preserves that distinction.
    denominator = roster_size if roster_size else 1
    values[f"{side}_player_prior_appearances_mean"][position] = (
        sum(state.appearances for state in known) / denominator
    )
    values[f"{side}_player_prior_starts_mean"][position] = (
        sum(state.starts for state in known) / denominator
    )
    values[f"{side}_player_prior_goals_mean"][position] = (
        sum(state.goals for state in known) / denominator
    )
    values[f"{side}_player_prior_cards_mean"][position] = (
        sum(state.cards for state in known) / denominator
    )
    prior_roster = previous_rosters.get(team, set())
    current_keys = {entry.player_key for entry in roster}
    values[f"{side}_player_lineup_continuity_rate"][position] = (
        len(current_keys & prior_roster) / roster_size if roster_size and prior_roster else 0.0
    )
    missing_values[f"{side}_player_roster_missing"][position] = int(roster_size == 0)
    missing_values[f"{side}_player_prior_history_missing"][position] = int(known_count == 0)
    cutoff_values[f"{side}_player_history_cutoff_time"][position] = team_cutoffs.get(team, pd.NaT)


def _build_rosters(players: pd.DataFrame) -> dict[tuple[str, str], tuple[_RosterEntry, ...]]:
    if players.empty:
        return {}
    match_id_column = find_column(players, "match_id")
    team_column = find_column(players, "team")
    player_column = find_column(players, "player")
    lineup_column = find_column(players, "lineup", required=False)
    event_column = find_column(players, "event", required=False)
    shirt_column = next(
        (column for column in ("Shirt Number", "shirt_number") if column in players.columns),
        None,
    )

    roster_rows: dict[tuple[str, str], dict[str, _RosterEntry]] = {}
    for row in players.itertuples(index=False, name=None):
        values = dict(zip(players.columns, row, strict=True))
        if pd.isna(values[player_column]) or pd.isna(values[team_column]):
            continue
        shirt_number = None if shirt_column is None else values[shirt_column]
        player_key = _player_key(values[player_column], shirt_number)
        team = str(values[team_column]).strip().upper()
        if not player_key or not team:
            continue
        lineup = "" if lineup_column is None else str(values[lineup_column]).strip().upper()
        event = (
            ""
            if event_column is None or pd.isna(values[event_column])
            else str(values[event_column])
        )
        entry = _RosterEntry(
            player_key=player_key,
            appeared=int(lineup == "S" or bool(event.strip())),
            start=int(lineup == "S"),
            goals=len(_GOAL_PATTERN.findall(event)),
            cards=len(_CARD_PATTERN.findall(event)),
        )
        key = (canonical_match_id(values[match_id_column]), team)
        existing = roster_rows.setdefault(key, {}).get(player_key)
        if existing is not None:
            raise ValueError(
                "Canonical player rows must be unique by normalized "
                "(MatchID, team initials, player name, shirt number); duplicate "
                f"found for {key + (player_key,)!r}."
            )
        roster_rows[key][player_key] = entry
    return {
        key: tuple(sorted(entries.values(), key=lambda entry: entry.player_key))
        for key, entries in roster_rows.items()
    }


def _player_key(value: object, shirt_number: object | None) -> str:
    name = " ".join(str(value).strip().casefold().split())
    if shirt_number is None or pd.isna(shirt_number):
        return name
    return f"{name}|shirt={canonical_match_id(shirt_number)}"
