"""Leakage-safe, completed-tournament team history features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._schema import (
    base_match_frame,
    find_column,
    normalize_label,
    numeric_series,
    team_series,
)

_FEATURES = (
    "historical_appearances",
    "historical_best_finish",
    "historical_mean_goals_for",
    "historical_mean_goals_conceded",
)


@dataclass
class _HistoryState:
    appearances: int = 0
    matches: int = 0
    goals_for: float = 0.0
    goals_conceded: float = 0.0
    best_finish: int = 0

    def values(self) -> tuple[float, float, float, float]:
        if self.matches == 0:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(self.appearances),
            float(self.best_finish),
            self.goals_for / self.matches,
            self.goals_conceded / self.matches,
        )


def build_team_match_features(
    matches: pd.DataFrame,
    tournaments: pd.DataFrame,
) -> pd.DataFrame:
    """Return one feature row per match using completed earlier editions only.

    A World Cup is treated as a single information block: every match in an
    edition receives the same historical team state, and that edition is added
    only after all its matches have been featurized.  Team initials are never
    remapped across eras (for example, ``FRG`` remains distinct from ``GER``).
    """

    base = base_match_frame(matches, include_targets=True)
    tournament_years = _validated_tournament_years(matches, tournaments)
    edition_stats = _edition_team_statistics(matches, tournaments)

    states: dict[str, _HistoryState] = {}
    feature_values: dict[str, np.ndarray] = {
        f"{side}_{name}": np.zeros(len(matches), dtype=float)
        for side in ("home", "away")
        for name in _FEATURES
    }
    cutoffs = np.empty(len(matches), dtype=int)
    positions_by_year: dict[int, list[int]] = {}
    for position, year in enumerate(base["Year"].to_numpy(dtype=int)):
        positions_by_year.setdefault(int(year), []).append(position)

    for year in sorted(positions_by_year):
        cutoff_candidates = tournament_years[tournament_years < year]
        cutoff = int(cutoff_candidates[-1]) if len(cutoff_candidates) else int(year - 1)
        positions = positions_by_year[year]
        cutoffs[positions] = cutoff

        for position in positions:
            for side in ("home", "away"):
                team = str(base.iloc[position][f"{side}_team_initials"])
                values = states.get(team, _HistoryState()).values()
                for name, value in zip(_FEATURES, values, strict=True):
                    feature_values[f"{side}_{name}"][position] = value

        # Update only after the complete edition has been snapshotted.
        year_stats = edition_stats.loc[edition_stats["Year"].eq(year)]
        for row in year_stats.itertuples(index=False):
            state = states.setdefault(str(row.team_initials), _HistoryState())
            state.appearances += 1
            state.matches += int(row.matches)
            state.goals_for += float(row.goals_for)
            state.goals_conceded += float(row.goals_conceded)
            finish = int(row.finish)
            if state.best_finish == 0 or finish < state.best_finish:
                state.best_finish = finish

    result = base.copy()
    result["history_cutoff_year"] = cutoffs
    for column, values in feature_values.items():
        result[column] = values
    for name in _FEATURES:
        result[f"{name}_diff"] = result[f"home_{name}"] - result[f"away_{name}"]
    result["home_history_missing"] = result["home_historical_appearances"].eq(0).astype("int8")
    result["away_history_missing"] = result["away_historical_appearances"].eq(0).astype("int8")
    return result.reset_index(drop=True)


def build_scenario_team_features(
    matches: pd.DataFrame,
    tournaments: pd.DataFrame,
    home_team_initials: str,
    away_team_initials: str,
    cutoff_year: int = 2014,
) -> pd.DataFrame:
    """Build the four frozen team-history feature groups for a 2026 scenario."""

    home = str(home_team_initials).strip().upper()
    away = str(away_team_initials).strip().upper()
    if not home or not away or home == away:
        raise ValueError("Scenario teams must be two distinct, non-empty initials.")

    tournament_years = _validated_tournament_years(matches, tournaments)
    cutoff_candidates = tournament_years[tournament_years <= int(cutoff_year)]
    if len(cutoff_candidates) == 0:
        raise ValueError("cutoff_year precedes the first available completed tournament.")
    effective_cutoff = int(cutoff_candidates[-1])
    edition_stats = _edition_team_statistics(matches, tournaments)
    history = edition_stats.loc[edition_stats["Year"].le(effective_cutoff)]
    states: dict[str, _HistoryState] = {}
    for row in history.sort_values(["Year", "team_initials"]).itertuples(index=False):
        state = states.setdefault(str(row.team_initials), _HistoryState())
        state.appearances += 1
        state.matches += int(row.matches)
        state.goals_for += float(row.goals_for)
        state.goals_conceded += float(row.goals_conceded)
        finish = int(row.finish)
        if state.best_finish == 0 or finish < state.best_finish:
            state.best_finish = finish

    row: dict[str, object] = {
        "Year": 2026,
        "history_cutoff_year": effective_cutoff,
        "home_team_initials": home,
        "away_team_initials": away,
        "Home Team Initials": home,
        "Away Team Initials": away,
    }
    for side, team in (("home", home), ("away", away)):
        values = states.get(team, _HistoryState()).values()
        for name, value in zip(_FEATURES, values, strict=True):
            row[f"{side}_{name}"] = value
        row[f"{side}_history_missing"] = int(states.get(team, _HistoryState()).appearances == 0)
    for name in _FEATURES:
        row[f"{name}_diff"] = float(row[f"home_{name}"]) - float(row[f"away_{name}"])
    return pd.DataFrame([row])


def _tournament_years(tournaments: pd.DataFrame) -> np.ndarray:
    if not isinstance(tournaments, pd.DataFrame) or tournaments.empty:
        raise ValueError("tournaments must be a non-empty pandas DataFrame.")
    year_column = find_column(tournaments, "year")
    years = pd.to_numeric(tournaments[year_column], errors="coerce")
    if years.isna().any() or years.duplicated().any():
        raise ValueError("Tournament years must be unique, non-missing integers.")
    return np.sort(years.astype(int).to_numpy())


def _validated_tournament_years(
    matches: pd.DataFrame,
    tournaments: pd.DataFrame,
) -> np.ndarray:
    tournament_years = _tournament_years(tournaments)
    match_years = pd.to_numeric(matches[find_column(matches, "year")], errors="coerce")
    if match_years.isna().any():
        raise ValueError("Match years must be non-missing integers.")
    observed = set(match_years.astype(int).unique())
    declared = set(tournament_years.tolist())
    undeclared = sorted(observed - declared)
    if undeclared:
        raise ValueError(f"Match years absent from tournaments: {undeclared!r}.")
    required = {year for year in declared if year <= max(observed)}
    missing = sorted(required - observed)
    if missing:
        raise ValueError(
            "Completed tournament history is incomplete; match rows are missing "
            f"for declared editions {missing!r}."
        )
    return tournament_years


def _edition_team_statistics(
    matches: pd.DataFrame,
    tournaments: pd.DataFrame,
) -> pd.DataFrame:
    years = pd.to_numeric(matches[find_column(matches, "year")], errors="raise").astype(int)
    home = team_series(matches, "home_team")
    away = team_series(matches, "away_team")
    home_goals = numeric_series(matches, "home_goals")
    away_goals = numeric_series(matches, "away_goals")
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "Year": years,
                    "team_initials": home,
                    "goals_for": home_goals,
                    "goals_conceded": away_goals,
                }
            ),
            pd.DataFrame(
                {
                    "Year": years,
                    "team_initials": away,
                    "goals_for": away_goals,
                    "goals_conceded": home_goals,
                }
            ),
        ],
        ignore_index=True,
    )
    grouped = long.groupby(["Year", "team_initials"], as_index=False, sort=True).agg(
        matches=("goals_for", "size"),
        goals_for=("goals_for", "sum"),
        goals_conceded=("goals_conceded", "sum"),
    )
    grouped["finish"] = _fallback_finishes(matches, grouped)
    podium = _podium_finishes(matches, tournaments)
    if podium:
        keys = list(zip(grouped["Year"], grouped["team_initials"], strict=True))
        grouped["finish"] = [
            podium.get((int(year), str(team)), finish)
            for (year, team), finish in zip(keys, grouped["finish"], strict=True)
        ]
    return grouped


def _fallback_finishes(matches: pd.DataFrame, grouped: pd.DataFrame) -> np.ndarray:
    stage_column = find_column(matches, "stage", required=False)
    if stage_column is None:
        return np.full(len(grouped), 5, dtype=int)
    stage_rank = {
        "final": 2,
        "third_place": 4,
        "semi_final": 4,
        "quarter_final": 8,
        "round_of_16": 16,
        "group_stage": 32,
    }
    years = pd.to_numeric(matches[find_column(matches, "year")], errors="raise").astype(int)
    home = team_series(matches, "home_team")
    away = team_series(matches, "away_team")
    stages = matches[stage_column].astype(str).map(_normalize_stage)
    observed: dict[tuple[int, str], int] = {}
    for year, home_team, away_team, stage in zip(years, home, away, stages, strict=True):
        rank = stage_rank.get(stage, 32)
        for team in (home_team, away_team):
            key = (int(year), str(team))
            observed[key] = min(observed.get(key, 32), rank)
    return np.asarray(
        [
            observed.get((int(year), str(team)), 32)
            for year, team in grouped[["Year", "team_initials"]].itertuples(index=False, name=None)
        ],
        dtype=int,
    )


def _podium_finishes(
    matches: pd.DataFrame,
    tournaments: pd.DataFrame,
) -> dict[tuple[int, str], int]:
    home_name_column = find_column(matches, "home_name", required=False)
    away_name_column = find_column(matches, "away_name", required=False)
    if home_name_column is None or away_name_column is None:
        return _podium_initial_columns(tournaments)

    years = pd.to_numeric(matches[find_column(matches, "year")], errors="raise").astype(int)
    home = team_series(matches, "home_team")
    away = team_series(matches, "away_team")
    name_lookup: dict[tuple[int, str], str] = {}
    for year, initial, name in zip(years, home, matches[home_name_column], strict=True):
        name_lookup[(int(year), normalize_label(name))] = str(initial)
    for year, initial, name in zip(years, away, matches[away_name_column], strict=True):
        name_lookup[(int(year), normalize_label(name))] = str(initial)

    result = _podium_initial_columns(tournaments)
    year_column = find_column(tournaments, "year")
    fields = (
        (("Winner", "winner"), 1),
        (("Runners-Up", "runners_up", "runners-up"), 2),
        (("Third", "third"), 3),
        (("Fourth", "fourth"), 4),
    )
    for tournament in tournaments.itertuples(index=False, name=None):
        values = dict(zip(tournaments.columns, tournament, strict=True))
        year = int(values[year_column])
        for aliases, rank in fields:
            field = next((alias for alias in aliases if alias in values), None)
            if field is not None:
                initial = name_lookup.get((year, normalize_label(values[field])))
                if initial is not None:
                    result[(year, initial)] = rank
    return result


def _normalize_stage(value: object) -> str:
    label = normalize_label(value).replace("-", "_").replace(" ", "_")
    if "third" in label or "3rd" in label:
        return "third_place"
    if "semi" in label:
        return "semi_final"
    if "quarter" in label:
        return "quarter_final"
    if "round_of_16" in label or "last_16" in label:
        return "round_of_16"
    if label == "final":
        return "final"
    if "group" in label or "first_stage" in label:
        return "group_stage"
    return label


def _podium_initial_columns(tournaments: pd.DataFrame) -> dict[tuple[int, str], int]:
    result: dict[tuple[int, str], int] = {}
    year_column = find_column(tournaments, "year")
    aliases = (
        (("winner_initials", "Winner Initials"), 1),
        (("runners_up_initials", "Runners-Up Initials"), 2),
        (("third_initials", "Third Initials"), 3),
        (("fourth_initials", "Fourth Initials"), 4),
    )
    for choices, rank in aliases:
        column = next((choice for choice in choices if choice in tournaments.columns), None)
        if column is None:
            continue
        for year, initial in tournaments[[year_column, column]].itertuples(index=False, name=None):
            if pd.notna(initial):
                result[(int(year), str(initial).strip().upper())] = rank
    return result
