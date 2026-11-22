"""Leakage-safe prematch team-state features.

The engine supports two explicit protocols:

``transform_history``
    Sequential research features. Each match is read before its result updates
    state.

``transform_history_frozen``
    Deployment-consistent tournament batches. Every match in a World Cup is
    represented from state available before that tournament, and the whole
    batch is incorporated only after all query rows have been constructed.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .team_entities import clean_team_name


CLASS_TO_INT = {"Home Win": 0, "Draw": 1, "Away Win": 2}
INT_TO_CLASS = {value: key for key, value in CLASS_TO_INT.items()}
CLASS_CHINESE = {"Home Win": "主队胜", "Draw": "平局", "Away Win": "客队胜"}
CLASS_CHINESE_BY_INT = {0: "主队胜", 1: "平局", 2: "客队胜"}

STAGE_ORDER = {
    "Group Stage": 1,
    "Round of 16": 2,
    "Quarter-finals": 3,
    "Semi-finals": 4,
    "Final / Third Place": 5,
}
STAGE_PROGRESS = {
    "Group Stage": 0.0,
    "Round of 16": 0.25,
    "Quarter-finals": 0.50,
    "Semi-finals": 0.75,
    "Final / Third Place": 1.0,
}
STAGE_CHINESE = {
    "Group Stage": "小组赛",
    "Round of 16": "十六强赛",
    "Quarter-finals": "四分之一决赛",
    "Semi-finals": "半决赛",
    "Final / Third Place": "决赛/三四名赛",
}

HOSTS_BY_YEAR = {
    1930: {"Uruguay"},
    1934: {"Italy"},
    1938: {"France"},
    1950: {"Brazil"},
    1954: {"Switzerland"},
    1958: {"Sweden"},
    1962: {"Chile"},
    1966: {"England"},
    1970: {"Mexico"},
    1974: {"Germany"},
    1978: {"Argentina"},
    1982: {"Spain"},
    1986: {"Mexico"},
    1990: {"Italy"},
    1994: {"USA"},
    1998: {"France"},
    2002: {"Japan", "South Korea"},
    2006: {"Germany"},
    2010: {"South Africa"},
    2014: {"Brazil"},
    2018: {"Russia"},
    2022: {"Qatar"},
    2026: {"Canada", "Mexico", "USA"},
}


def normalize_stage(stage) -> str:
    text = "" if pd.isna(stage) else str(stage).strip().lower()
    if any(token in text for token in ("三四名", "冠亚军")) or text == "决赛":
        return "Final / Third Place"
    if "半决赛" in text:
        return "Semi-finals"
    if any(token in text for token in ("四分之一", "八强")):
        return "Quarter-finals"
    if any(token in text for token in ("十六强", "1/8")):
        return "Round of 16"
    if "小组" in text:
        return "Group Stage"
    if "final" in text and not any(token in text for token in ("quarter", "semi", "round")):
        return "Final / Third Place"
    if any(token in text for token in ("third", "play off", "play-off")):
        return "Final / Third Place"
    if "semi" in text:
        return "Semi-finals"
    if "quarter" in text:
        return "Quarter-finals"
    if any(token in text for token in ("round of 16", "last 16", "preliminary")):
        return "Round of 16"
    return "Group Stage"


def score_outcome(home_goals, away_goals) -> str:
    home, away = float(home_goals), float(away_goals)
    return "Home Win" if home > away else "Away Win" if home < away else "Draw"


@dataclass
class _TeamState:
    games: int = 0
    wins: int = 0
    draws: int = 0
    goals_for: float = 0.0
    goals_against: float = 0.0
    tournaments: set[int] = field(default_factory=set)
    best_stage: float = 0.0
    elo: float = 1500.0
    recent: deque = field(default_factory=lambda: deque(maxlen=8))


class TeamStateEngine:
    """Maintain Elo, history, form, era context and head-to-head state."""

    def __init__(
        self,
        k_factor: float = 24.0,
        form_decay: float = 0.8,
        entity_policy: str = "conservative",
        era_tournaments: int = 5,
    ):
        if not 0 < form_decay <= 1:
            raise ValueError("近期状态衰减系数必须位于 (0, 1]。")
        if era_tournaments < 1:
            raise ValueError("时代窗口至少包含一届世界杯。")
        self.k_factor = float(k_factor)
        self.form_decay = float(form_decay)
        self.entity_policy = entity_policy
        self.era_tournaments = int(era_tournaments)
        self.states: defaultdict[str, _TeamState] = defaultdict(_TeamState)
        self.h2h: defaultdict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: {
                "games": 0.0,
                "first_points": 0.0,
                "second_points": 0.0,
                "first_gd": 0.0,
            }
        )
        self.era_observations: list[tuple[int, float, float]] = []

    def clean(self, value) -> str:
        return clean_team_name(value, policy=self.entity_policy)

    @staticmethod
    def _ordered(matches: pd.DataFrame) -> pd.DataFrame:
        keys = [name for name in ("Year", "Datetime", "MatchID") if name in matches.columns]
        return (
            matches.sort_values(keys, na_position="last", kind="mergesort")
            if keys
            else matches.copy()
        )

    def _form_values(self, state: _TeamState) -> tuple[float, float, float, float]:
        if not state.recent:
            return 1.0, 0.0, 1.2, 1.2
        values = list(state.recent)
        weights = np.asarray(
            [self.form_decay ** (len(values) - 1 - i) for i in range(len(values))],
            dtype=float,
        )
        points = np.asarray([item[0] for item in values], dtype=float)
        goal_diff = np.asarray([item[1] for item in values], dtype=float)
        goals_for = np.asarray([item[2] for item in values], dtype=float)
        goals_against = np.asarray([item[3] for item in values], dtype=float)
        return (
            float(np.average(points, weights=weights)),
            float(np.average(goal_diff, weights=weights)),
            float(np.average(goals_for, weights=weights)),
            float(np.average(goals_against, weights=weights)),
        )

    def _values(self, state: _TeamState) -> dict[str, float]:
        games = max(state.games, 1)
        form_points, form_gd, recent_gf, recent_ga = self._form_values(state)
        return {
            "Experience": float(state.games),
            "TournamentExperience": float(len(state.tournaments)),
            "BestStage": float(state.best_stage),
            "HistWinRate": float(state.wins / games),
            "HistDrawRate": float(state.draws / games),
            "HistAvgGF": float(state.goals_for / games),
            "HistAvgGA": float(state.goals_against / games),
            "HistGoalDiff": float((state.goals_for - state.goals_against) / games),
            "Elo": float(state.elo),
            "FormPoints": form_points,
            "FormGoalDiff": form_gd,
            "RecentGF": recent_gf,
            "RecentGA": recent_ga,
        }

    def _era_stats(self) -> tuple[float, float, float, float]:
        if not self.era_observations:
            return 1.2, 1.0, 1.2, 1.0
        years = sorted({year for year, _, _ in self.era_observations})
        selected = set(years[-self.era_tournaments :])
        observations = [(gf, ga) for year, gf, ga in self.era_observations if year in selected]
        gf = np.asarray([item[0] for item in observations], dtype=float)
        ga = np.asarray([item[1] for item in observations], dtype=float)
        return (
            float(gf.mean()),
            float(max(gf.std(), 0.25)),
            float(ga.mean()),
            float(max(ga.std(), 0.25)),
        )

    def feature_row(self, home, away, stage="Group Stage", year=2026) -> dict[str, float]:
        home_name, away_name = self.clean(home), self.clean(away)
        hv, av = self._values(self.states[home_name]), self._values(self.states[away_name])
        row: dict[str, float] = {}
        base_keys = (
            "Experience",
            "TournamentExperience",
            "BestStage",
            "HistWinRate",
            "HistDrawRate",
            "HistAvgGF",
            "HistAvgGA",
            "HistGoalDiff",
            "Elo",
            "FormPoints",
            "FormGoalDiff",
            "RecentGF",
            "RecentGA",
        )
        for key in base_keys:
            row[f"{key}Diff"] = hv[key] - av[key]
        for prefix, values in (("Home", hv), ("Away", av)):
            for key in base_keys:
                row[f"{prefix}{key}"] = values[key]

        row["AbsEloDiff"] = abs(row["EloDiff"])
        row["AbsHistWinRateDiff"] = abs(row["HistWinRateDiff"])
        row["AbsHistGoalDiffDiff"] = abs(row["HistGoalDiffDiff"])
        row["AbsFormPointsDiff"] = abs(row["FormPointsDiff"])
        row["DrawRateSum"] = hv["HistDrawRate"] + av["HistDrawRate"]
        row["AttackStrengthSum"] = hv["RecentGF"] + av["RecentGF"]
        row["DefenseStrengthSum"] = hv["RecentGA"] + av["RecentGA"]
        row["ExpectedGoalsProxy"] = 0.5 * (
            row["AttackStrengthSum"] + row["DefenseStrengthSum"]
        )
        row["ExperienceBalance"] = 1.0 / (1.0 + abs(row["ExperienceDiff"]))

        normalized_stage = normalize_stage(stage)
        row["StageCode"] = float(STAGE_ORDER[normalized_stage])
        row["StageProgress"] = float(STAGE_PROGRESS[normalized_stage])
        try:
            query_year = int(float(year))
        except (TypeError, ValueError):
            query_year = 2026
        hosts = HOSTS_BY_YEAR.get(query_year, set())
        row["HostIndicatorDiff"] = float(home_name in hosts) - float(away_name in hosts)

        gf_mean, gf_std, ga_mean, ga_std = self._era_stats()
        home_attack_z = (hv["RecentGF"] - gf_mean) / gf_std
        away_attack_z = (av["RecentGF"] - gf_mean) / gf_std
        home_defense_z = (hv["RecentGA"] - ga_mean) / ga_std
        away_defense_z = (av["RecentGA"] - ga_mean) / ga_std
        row["EraAttackZDiff"] = home_attack_z - away_attack_z
        row["EraDefenseZDiff"] = home_defense_z - away_defense_z
        # Compatibility names now point to genuine rolling-era features rather
        # than a rescaled cumulative-goal difference.
        row["EraAdjustedGFDiff"] = row["EraAttackZDiff"]
        row["EraAdjustedGADiff"] = row["EraDefenseZDiff"]

        first, second = sorted((home_name, away_name))
        record = self.h2h[(first, second)]
        if record["games"]:
            if home_name == first:
                row["H2HPointsDiff"] = (
                    record["first_points"] - record["second_points"]
                ) / record["games"]
                row["H2HGoalDiff"] = record["first_gd"] / record["games"]
            else:
                row["H2HPointsDiff"] = (
                    record["second_points"] - record["first_points"]
                ) / record["games"]
                row["H2HGoalDiff"] = -record["first_gd"] / record["games"]
        else:
            row["H2HPointsDiff"] = 0.0
            row["H2HGoalDiff"] = 0.0
        row["H2HGames"] = record["games"]
        return row

    def update(self, match: pd.Series | dict) -> None:
        home = self.clean(match["Home Team Name"])
        away = self.clean(match["Away Team Name"])
        home_goals = float(match["Home Team Goals"])
        away_goals = float(match["Away Team Goals"])
        outcome = match.get("ScoreOutcome") or score_outcome(home_goals, away_goals)
        if outcome not in CLASS_TO_INT:
            raise ValueError(f"比赛结果类别无法识别：{outcome}")
        year = int(float(match.get("Year", 0)))
        stage = normalize_stage(match.get("StageNormalized", match.get("Stage", "Group Stage")))
        home_state, away_state = self.states[home], self.states[away]

        home_score = 1.0 if outcome == "Home Win" else 0.0 if outcome == "Away Win" else 0.5
        expected = 1.0 / (1.0 + 10 ** ((away_state.elo - home_state.elo) / 400.0))
        change = self.k_factor * (home_score - expected)
        home_state.elo += change
        away_state.elo -= change

        for state, goals_for, goals_against, won, drew in (
            (home_state, home_goals, away_goals, outcome == "Home Win", outcome == "Draw"),
            (away_state, away_goals, home_goals, outcome == "Away Win", outcome == "Draw"),
        ):
            state.games += 1
            state.wins += int(won)
            state.draws += int(drew)
            state.goals_for += goals_for
            state.goals_against += goals_against
            state.tournaments.add(year)
            state.best_stage = max(state.best_stage, float(STAGE_ORDER[stage]))
            state.recent.append(
                (
                    3.0 if won else 1.0 if drew else 0.0,
                    goals_for - goals_against,
                    goals_for,
                    goals_against,
                )
            )
            self.era_observations.append((year, goals_for, goals_against))

        first, second = sorted((home, away))
        record = self.h2h[(first, second)]
        record["games"] += 1.0
        if outcome == "Home Win":
            home_points, away_points = 3.0, 0.0
        elif outcome == "Away Win":
            home_points, away_points = 0.0, 3.0
        else:
            home_points, away_points = 1.0, 1.0
        if home == first:
            record["first_points"] += home_points
            record["second_points"] += away_points
            record["first_gd"] += home_goals - away_goals
        else:
            record["first_points"] += away_points
            record["second_points"] += home_points
            record["first_gd"] += away_goals - home_goals

    def fit(self, matches: pd.DataFrame) -> "TeamStateEngine":
        for _, match in self._ordered(matches).iterrows():
            self.update(match)
        return self

    def _match_feature_record(self, match: pd.Series | dict) -> dict:
        year = int(float(match["Year"]))
        home = self.clean(match["Home Team Name"])
        away = self.clean(match["Away Team Name"])
        row = {
            "Year": year,
            "HomeTeam": home,
            "AwayTeam": away,
            "Outcome": match.get("ScoreOutcome")
            or score_outcome(match["Home Team Goals"], match["Away Team Goals"]),
            "HomeGoals": float(match["Home Team Goals"]),
            "AwayGoals": float(match["Away Team Goals"]),
        }
        row.update(
            self.feature_row(
                home,
                away,
                match.get("StageNormalized", match.get("Stage", "Group Stage")),
                year,
            )
        )
        return row

    def transform_history(self, matches: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        for _, match in self._ordered(matches).iterrows():
            rows.append(self._match_feature_record(match))
            self.update(match)
        return pd.DataFrame(rows)

    def transform_history_frozen(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Build deployment-consistent features with no within-query updates."""
        rows: list[dict] = []
        ordered = self._ordered(matches)
        for year in sorted(pd.to_numeric(ordered["Year"]).dropna().astype(int).unique()):
            batch = self._ordered(ordered[pd.to_numeric(ordered["Year"], errors="coerce") == year])
            rows.extend(self._match_feature_record(match) for _, match in batch.iterrows())
            for _, match in batch.iterrows():
                self.update(match)
        return pd.DataFrame(rows)

    def transform_queries(
        self,
        frame: pd.DataFrame,
        home_col: str,
        away_col: str,
        stage_col: str,
        year_col: str,
    ) -> pd.DataFrame:
        rows = [
            self.feature_row(row[home_col], row[away_col], row[stage_col], row[year_col])
            for _, row in frame.iterrows()
        ]
        return pd.DataFrame(rows, index=frame.index)
