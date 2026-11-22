"""Internal deterministic group and source-graph knockout simulation engine."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .format_policy import first_knockout_source_pairs, later_knockout_source_pairs

if TYPE_CHECKING:
    from .tournament import TournamentFormat

ScoreProvider = Callable[[str, str, str], np.ndarray]
GROUP_FIXTURE_POSITIONS = ((0, 1), (2, 3), (0, 2), (3, 1), (3, 0), (1, 2))


@dataclass(slots=True)
class _Standing:
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass(frozen=True, slots=True)
class _GroupResult:
    home: str
    away: str
    home_goals: int
    away_goals: int


def simulate_once(
    teams: list[str],
    tournament_format: TournamentFormat,
    score_provider: ScoreProvider,
    rng: np.random.Generator,
    *,
    simulation_id: int,
    team_order: dict[str, int],
) -> tuple[str, list[dict[str, Any]]]:
    """Run one group-plus-knockout tournament under a pinned source graph."""
    rows: list[dict[str, Any]] = []
    match_number = 0
    third_place_candidates: list[tuple[str, str, _Standing]] = []
    qualifier_slots: dict[str, str] = {}

    for group_index in range(tournament_format.group_count):
        group_label = chr(ord("A") + group_index)
        start = group_index * tournament_format.group_size
        group = teams[start : start + tournament_format.group_size]
        standings = {team: _Standing() for team in group}
        results: list[_GroupResult] = []
        for home_position, away_position in GROUP_FIXTURE_POSITIONS:
            home = group[home_position]
            away = group[away_position]
            home_goals, away_goals = _sample_score(
                score_provider(home, away, "group_stage"), rng
            )
            result = _GroupResult(home, away, home_goals, away_goals)
            results.append(result)
            _update_standings(standings, result)
            winner = home if home_goals > away_goals else away if away_goals > home_goals else None
            match_number += 1
            rows.append(
                _match_row(
                    simulation_id,
                    match_number,
                    "group_stage",
                    home,
                    away,
                    home_goals,
                    away_goals,
                    winner=winner,
                    home_source_slot=f"{group_label}_draw{home_position + 1}",
                    away_source_slot=f"{group_label}_draw{away_position + 1}",
                    group_label=group_label,
                )
            )
        ranked = _rank_group(group, standings, results, team_order)
        for rank in range(tournament_format.automatic_qualifiers_per_group):
            qualifier_slots[f"{rank + 1}{group_label}"] = ranked[rank]
        if tournament_format.best_third_qualifiers:
            third = ranked[tournament_format.automatic_qualifiers_per_group]
            third_place_candidates.append((third, group_label, standings[third]))

    selected_thirds = sorted(
        third_place_candidates,
        key=lambda item: _standing_key(item[2], team_order[item[0]]),
    )[: tournament_format.best_third_qualifiers]
    for team, group_label, _ in selected_thirds:
        qualifier_slots[f"3{group_label}"] = team
    if len(qualifier_slots) != tournament_format.knockout_team_count:
        raise RuntimeError("Group qualification did not produce the declared knockout field")
    if len(set(qualifier_slots.values())) != len(qualifier_slots):
        raise RuntimeError("A team qualified more than once")

    first_sources = first_knockout_source_pairs(
        tournament_format.bracket_policy,
        third_place_groups=tuple(group for _, group, _ in selected_thirds),
    )
    source_teams = dict(qualifier_slots)
    stages = _main_knockout_stages(tournament_format.first_knockout_stage)
    for stage_index, stage in enumerate(stages):
        source_pairs = (
            first_sources
            if stage_index == 0
            else later_knockout_source_pairs(tournament_format.bracket_policy, stage)
        )
        fixtures = _resolve_fixtures(source_pairs, source_teams, stage)
        winners, losers, stage_rows, match_number = _play_knockout_fixtures(
            fixtures,
            stage,
            score_provider,
            rng,
            simulation_id=simulation_id,
            first_match_number=match_number,
        )
        rows.extend(stage_rows)
        first_stage_match = match_number - len(winners) + 1
        for offset, (winner, loser) in enumerate(zip(winners, losers, strict=True)):
            source_teams[f"W{first_stage_match + offset}"] = winner
            source_teams[f"L{first_stage_match + offset}"] = loser

    third_sources = later_knockout_source_pairs(
        tournament_format.bracket_policy, "third_place"
    )
    third_fixtures = _resolve_fixtures(third_sources, source_teams, "third_place")
    _, _, third_rows, match_number = _play_knockout_fixtures(
        third_fixtures,
        "third_place",
        score_provider,
        rng,
        simulation_id=simulation_id,
        first_match_number=match_number,
    )
    rows.extend(third_rows)

    final_sources = later_knockout_source_pairs(tournament_format.bracket_policy, "final")
    final_fixtures = _resolve_fixtures(final_sources, source_teams, "final")
    champions, _, final_rows, match_number = _play_knockout_fixtures(
        final_fixtures,
        "final",
        score_provider,
        rng,
        simulation_id=simulation_id,
        first_match_number=match_number,
    )
    rows.extend(final_rows)
    if match_number != tournament_format.matches_per_simulation or len(champions) != 1:
        raise RuntimeError("Tournament completion does not match the declared format")
    return champions[0], rows


def _rank_group(
    teams: Sequence[str],
    standings: dict[str, _Standing],
    results: Sequence[_GroupResult],
    team_order: dict[str, int],
) -> list[str]:
    points_groups: dict[int, list[str]] = defaultdict(list)
    for team in teams:
        points_groups[standings[team].points].append(team)
    ranked: list[str] = []
    for points in sorted(points_groups, reverse=True):
        ranked.extend(
            _rank_equal_points(points_groups[points], standings, results, team_order)
        )
    return ranked


def _rank_equal_points(
    tied: Sequence[str],
    standings: dict[str, _Standing],
    results: Sequence[_GroupResult],
    team_order: dict[str, int],
) -> list[str]:
    if len(tied) < 2:
        return list(tied)
    mini = {team: _Standing() for team in tied}
    tied_set = set(tied)
    for result in results:
        if result.home in tied_set and result.away in tied_set:
            _update_standings(mini, result)
    head_to_head_groups: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    for team in tied:
        state = mini[team]
        key = (state.points, state.goal_difference, state.goals_for)
        head_to_head_groups[key].append(team)
    if len(head_to_head_groups) > 1:
        ranked: list[str] = []
        for key in sorted(head_to_head_groups, reverse=True):
            subgroup = head_to_head_groups[key]
            ranked.extend(_rank_equal_points(subgroup, standings, results, team_order))
        return ranked
    # Conduct score is not generated.  It is therefore treated as tied, and
    # ordered scenario input acts as the declared published-ranking proxy.
    return sorted(tied, key=lambda team: _standing_key(standings[team], team_order[team]))


def _update_standings(standings: dict[str, _Standing], result: _GroupResult) -> None:
    home_state = standings[result.home]
    away_state = standings[result.away]
    home_state.goals_for += result.home_goals
    home_state.goals_against += result.away_goals
    away_state.goals_for += result.away_goals
    away_state.goals_against += result.home_goals
    if result.home_goals > result.away_goals:
        home_state.points += 3
    elif result.away_goals > result.home_goals:
        away_state.points += 3
    else:
        home_state.points += 1
        away_state.points += 1


def _standing_key(standing: _Standing, input_position: int) -> tuple[int, int, int, int]:
    return (
        -standing.points,
        -standing.goal_difference,
        -standing.goals_for,
        input_position,
    )


def _main_knockout_stages(first_stage: str) -> tuple[str, ...]:
    if first_stage == "round_of_32":
        return ("round_of_32", "round_of_16", "quarter_final", "semi_final")
    if first_stage == "round_of_16":
        return ("round_of_16", "quarter_final", "semi_final")
    raise RuntimeError(f"Unsupported first knockout stage: {first_stage}")


def _resolve_fixtures(
    source_pairs: Sequence[tuple[str, str]],
    source_teams: dict[str, str],
    stage: str,
) -> list[tuple[str, str, str, str]]:
    fixtures: list[tuple[str, str, str, str]] = []
    for home_source, away_source in source_pairs:
        try:
            home = source_teams[home_source]
            away = source_teams[away_source]
        except KeyError as exc:
            raise RuntimeError(f"Unresolved {stage} bracket source: {exc.args[0]}") from exc
        if home == away:
            raise RuntimeError(f"Bracket source graph produced a self-match in {stage}")
        fixtures.append((home, away, home_source, away_source))
    return fixtures


def _play_knockout_fixtures(
    fixtures: Sequence[tuple[str, str, str, str]],
    stage: str,
    score_provider: ScoreProvider,
    rng: np.random.Generator,
    *,
    simulation_id: int,
    first_match_number: int,
) -> tuple[list[str], list[str], list[dict[str, Any]], int]:
    winners: list[str] = []
    losers: list[str] = []
    rows: list[dict[str, Any]] = []
    match_number = first_match_number
    for home, away, home_source, away_source in fixtures:
        matrix = np.asarray(score_provider(home, away, stage), dtype=float)
        home_goals, away_goals = _sample_score(matrix, rng)
        if home_goals > away_goals:
            winner, loser = home, away
        elif away_goals > home_goals:
            winner, loser = away, home
        elif _sample_knockout_home_win(matrix, rng):
            winner, loser = home, away
        else:
            winner, loser = away, home
        winners.append(winner)
        losers.append(loser)
        match_number += 1
        rows.append(
            _match_row(
                simulation_id,
                match_number,
                stage,
                home,
                away,
                home_goals,
                away_goals,
                winner=winner,
                home_source_slot=home_source,
                away_source_slot=away_source,
                group_label=None,
            )
        )
    return winners, losers, rows, match_number


def _validated_probability(matrix: np.ndarray) -> np.ndarray:
    probability = np.asarray(matrix, dtype=float)
    if probability.ndim != 2 or probability.shape[0] != probability.shape[1]:
        raise ValueError("score_provider must return a square two-dimensional grid")
    if probability.size == 0 or not np.isfinite(probability).all():
        raise ValueError("score_provider returned an empty or nonfinite grid")
    if np.any(probability < 0):
        raise ValueError("score_provider returned negative probability mass")
    mass = float(probability.sum())
    if mass <= 0:
        raise ValueError("score_provider returned zero total probability mass")
    return probability / mass


def _sample_score(matrix: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    probability = _validated_probability(matrix)
    draw = int(rng.choice(probability.size, p=probability.ravel()))
    home_goals, away_goals = np.unravel_index(draw, probability.shape)
    return int(home_goals), int(away_goals)


def _sample_knockout_home_win(matrix: np.ndarray, rng: np.random.Generator) -> bool:
    probability = _validated_probability(matrix)
    home_win_mass = float(np.tril(probability, k=-1).sum())
    away_win_mass = float(np.triu(probability, k=1).sum())
    non_draw_mass = home_win_mass + away_win_mass
    if non_draw_mass <= np.finfo(float).eps:
        return int(rng.integers(0, 2)) == 0
    return bool(rng.random() < home_win_mass / non_draw_mass)


def _match_row(
    simulation_id: int,
    match_number: int,
    stage: str,
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
    *,
    winner: str | None,
    home_source_slot: str,
    away_source_slot: str,
    group_label: str | None,
) -> dict[str, Any]:
    return {
        "simulation_id": simulation_id,
        "match_number": match_number,
        "stage": stage,
        "home_team": home,
        "away_team": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "winner": winner if winner is not None else np.nan,
        "home_source_slot": home_source_slot,
        "away_source_slot": away_source_slot,
        "group_label": group_label if group_label is not None else np.nan,
    }


__all__ = ["GROUP_FIXTURE_POSITIONS", "simulate_once"]
