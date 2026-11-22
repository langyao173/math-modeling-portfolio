"""Auditable tournament simulation for historical and 2026 format scenarios.

The simulator is deliberately format driven.  It does not contain or infer real
2026 entrants, fixtures, rankings, or future information.  Callers provide an
ordered list of scenario teams and a coherent joint-score distribution provider.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._engine import simulate_once as _simulate_once
from .format_policy import (
    FIFA_2026_ANNEX_C_CSV_SHA256,
    FIFA_2026_ANNEX_C_POLICY,
    FIFA_2026_REGULATIONS_SHA256,
    FIFA_2026_REGULATIONS_URL,
    SUPPORTED_BRACKET_POLICIES,
    WORLD_CUP_2022_CROSS_GROUP_POLICY,
)

ScoreProvider = Callable[[str, str, str], np.ndarray]


@dataclass(frozen=True, slots=True)
class TournamentFormat:
    """Declarative group-plus-knockout tournament structure."""

    name: str
    team_count: int
    group_count: int
    group_size: int
    automatic_qualifiers_per_group: int
    best_third_qualifiers: int
    first_knockout_stage: str
    bracket_policy: str
    include_third_place: bool = True

    def __post_init__(self) -> None:
        integer_fields = (
            self.team_count,
            self.group_count,
            self.group_size,
            self.automatic_qualifiers_per_group,
            self.best_third_qualifiers,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
            raise TypeError("Tournament format counts must be integers")
        if (
            not self.name.strip()
            or not self.first_knockout_stage.strip()
            or not self.bracket_policy.strip()
        ):
            raise ValueError("Tournament format and first-stage names must be nonempty")
        if self.bracket_policy not in SUPPORTED_BRACKET_POLICIES:
            raise ValueError(f"Unsupported bracket policy: {self.bracket_policy}")
        if not isinstance(self.include_third_place, bool):
            raise TypeError("include_third_place must be boolean")
        if not self.include_third_place:
            raise ValueError("The supported official bracket policies include a third-place match")
        if self.team_count <= 0 or self.group_count <= 0 or self.group_size < 2:
            raise ValueError("Tournament team/group counts must be positive")
        if self.team_count != self.group_count * self.group_size:
            raise ValueError("team_count must equal group_count * group_size")
        if not 0 < self.automatic_qualifiers_per_group < self.group_size:
            raise ValueError("Automatic qualifiers must be between zero and group size")
        if not 0 <= self.best_third_qualifiers <= self.group_count:
            raise ValueError("best_third_qualifiers is outside the group range")
        knockout = self.knockout_team_count
        if knockout < 2 or knockout & (knockout - 1):
            raise ValueError("The knockout field must contain a power-of-two number of teams")
        expected_stage = {16: "round_of_16", 32: "round_of_32"}.get(knockout)
        if expected_stage is None or self.first_knockout_stage != expected_stage:
            raise ValueError("first_knockout_stage does not match the knockout field size")
        expected_policy = {
            (8, 0): WORLD_CUP_2022_CROSS_GROUP_POLICY,
            (12, 8): FIFA_2026_ANNEX_C_POLICY,
        }.get((self.group_count, self.best_third_qualifiers))
        if self.bracket_policy != expected_policy:
            raise ValueError("bracket_policy does not match the declared group/qualifier format")

    @property
    def knockout_team_count(self) -> int:
        """Number of teams entering the knockout bracket."""
        return (
            self.group_count * self.automatic_qualifiers_per_group
            + self.best_third_qualifiers
        )

    @property
    def group_match_count(self) -> int:
        """Round-robin match count across all groups."""
        return self.group_count * self.group_size * (self.group_size - 1) // 2

    @property
    def matches_per_simulation(self) -> int:
        """Exact number of matches implied by the format."""
        main_knockout = self.knockout_team_count - 1
        third_place = int(self.include_third_place)
        return self.group_match_count + main_knockout + third_place

    @classmethod
    def world_cup_2022(cls) -> TournamentFormat:
        """Return the 32-team, 64-match World Cup format."""
        return cls(
            name="world_cup_2022",
            team_count=32,
            group_count=8,
            group_size=4,
            automatic_qualifiers_per_group=2,
            best_third_qualifiers=0,
            first_knockout_stage="round_of_16",
            bracket_policy=WORLD_CUP_2022_CROSS_GROUP_POLICY,
        )

    @classmethod
    def world_cup_2026(cls) -> TournamentFormat:
        """Return the published 48-team, 104-match World Cup format."""
        return cls(
            name="world_cup_2026",
            team_count=48,
            group_count=12,
            group_size=4,
            automatic_qualifiers_per_group=2,
            best_third_qualifiers=8,
            first_knockout_stage="round_of_32",
            bracket_policy=FIFA_2026_ANNEX_C_POLICY,
        )


@dataclass(frozen=True, slots=True)
class TournamentSimulationResult:
    """Match-level simulation evidence and aggregate championship probabilities."""

    match_log: pd.DataFrame
    team_summary: pd.DataFrame
    metadata: dict[str, Any]


def simulate_tournament(
    *,
    teams: Sequence[str],
    tournament_format: TournamentFormat,
    score_provider: ScoreProvider,
    n_simulations: int,
    seed: int,
) -> TournamentSimulationResult:
    """Simulate a supplied scenario under a declared tournament format.

    The provider returns a square joint score grid with rows representing home
    goals and columns representing away goals.  Every match is sampled from that
    grid.  A tied regulation-time knockout score is resolved from the normalized
    non-draw mass of that same joint score kernel.  This is an explicit
    strength-preserving extra-time/penalty proxy, not a fitted extra-time model.
    """
    scenario_teams = _validate_inputs(
        teams,
        tournament_format=tournament_format,
        score_provider=score_provider,
        n_simulations=n_simulations,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    logs: list[dict[str, Any]] = []
    champion_counts = {team: 0 for team in scenario_teams}
    for simulation_id in range(n_simulations):
        ranking_proxy = dict(
            zip(
                scenario_teams,
                rng.permutation(len(scenario_teams)).tolist(),
                strict=True,
            )
        )
        champion, simulation_rows = _simulate_once(
            scenario_teams,
            tournament_format,
            score_provider,
            rng,
            simulation_id=simulation_id,
            team_order=ranking_proxy,
        )
        logs.extend(simulation_rows)
        champion_counts[champion] += 1

    match_log = pd.DataFrame(
        logs,
        columns=(
            "simulation_id",
            "match_number",
            "stage",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "winner",
            "home_source_slot",
            "away_source_slot",
            "group_label",
        ),
    )
    expected_rows = n_simulations * tournament_format.matches_per_simulation
    if len(match_log) != expected_rows:
        raise RuntimeError(
            f"Format generated {len(match_log)} matches, expected {expected_rows}"
        )
    team_summary = pd.DataFrame(
        {
            "team": scenario_teams,
            "champion_count": [champion_counts[team] for team in scenario_teams],
            "champion_probability": [
                champion_counts[team] / n_simulations for team in scenario_teams
            ],
        }
    )
    metadata: dict[str, Any] = {
        "format_name": tournament_format.name,
        "scenario_status": "illustrative_user_supplied_teams_not_real_2026_entrants",
        "team_count": tournament_format.team_count,
        "group_count": tournament_format.group_count,
        "group_size": tournament_format.group_size,
        "automatic_qualifiers_per_group": (
            tournament_format.automatic_qualifiers_per_group
        ),
        "best_third_qualifiers": tournament_format.best_third_qualifiers,
        "matches_per_simulation": tournament_format.matches_per_simulation,
        "n_simulations": n_simulations,
        "seed": seed,
        "score_sampling": "normalized_joint_score_grid",
        "bracket_policy": tournament_format.bracket_policy,
        "group_fixture_order": "official_draw_positions_1v2_3v4_1v3_4v2_4v1_2v3",
        "knockout_tie_resolution": (
            "normalized_non_draw_mass_from_same_joint_score_kernel_as_"
            "strength_proxy_with_seeded_coin_fallback_only_if_non_draw_mass_is_zero"
        ),
        "knockout_proxy_scope": (
            "not_a_fitted_extra_time_or_penalty_model_and_not_a_claim_about_shootout_mechanics"
        ),
        "group_tie_break": (
            "points_then_recursive_head_to_head_points_goal_difference_goals_for_then_"
            "overall_goal_difference_goals_for_then_conduct_and_published_ranking_"
            "represented_by_a_seeded_random_symmetry_preserving_proxy"
        ),
        "unmodelled_group_tie_break_fields": ["team_conduct_score", "published_fifa_ranking"],
        "published_ranking_proxy": "seeded_random_per_tournament_only_after_modelled_ties",
    }
    if tournament_format.bracket_policy == FIFA_2026_ANNEX_C_POLICY:
        metadata.update(
            {
                "format_source_url": FIFA_2026_REGULATIONS_URL,
                "format_source_sha256": FIFA_2026_REGULATIONS_SHA256,
                "annex_c_csv_sha256": FIFA_2026_ANNEX_C_CSV_SHA256,
                "annex_c_rows": 495,
                "format_certification": (
                    "official_regulations_M73_M104_graph_and_complete_annexe_c_mapping"
                ),
            }
        )
    return TournamentSimulationResult(match_log, team_summary, metadata)


def _validate_inputs(
    teams: Sequence[str],
    *,
    tournament_format: TournamentFormat,
    score_provider: ScoreProvider,
    n_simulations: int,
    seed: int,
) -> list[str]:
    if not isinstance(tournament_format, TournamentFormat):
        raise TypeError("tournament_format must be a TournamentFormat")
    if isinstance(teams, (str, bytes)) or not isinstance(teams, Sequence):
        raise TypeError("teams must be an ordered sequence of names")
    normalized = [str(team).strip() for team in teams]
    if len(normalized) != tournament_format.team_count:
        raise ValueError(
            f"Expected {tournament_format.team_count} teams, received {len(normalized)}"
        )
    if any(not team for team in normalized):
        raise ValueError("Team names must be nonempty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Team names must be unique")
    if not callable(score_provider):
        raise TypeError("score_provider must be callable")
    if isinstance(n_simulations, bool) or not isinstance(n_simulations, int):
        raise TypeError("n_simulations must be an integer")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    return normalized


__all__ = ["TournamentFormat", "TournamentSimulationResult", "simulate_tournament"]
