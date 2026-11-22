"""Source-graph and Annexe C certification tests for tournament simulation."""

from __future__ import annotations

import itertools

import numpy as np

from worldcup_forecasting.simulation._engine import (
    GROUP_FIXTURE_POSITIONS,
    _GroupResult,
    _rank_group,
    _sample_knockout_home_win,
    _Standing,
)
from worldcup_forecasting.simulation.format_policy import (
    FIFA_2026_ANNEX_C_CSV_SHA256,
    FIFA_2026_ANNEX_C_POLICY,
    FIFA_2026_REGULATIONS_SHA256,
    WORLD_CUP_2022_CROSS_GROUP_POLICY,
    first_knockout_source_pairs,
    later_knockout_source_pairs,
    load_fifa_2026_annex_c,
)


def _slot_group(slot: str) -> str:
    assert len(slot) == 2 and slot[0] in "123"
    return slot[1]


def test_annex_c_is_complete_hash_bound_and_covers_all_combinations() -> None:
    table = load_fifa_2026_annex_c()

    assert table.csv_sha256 == FIFA_2026_ANNEX_C_CSV_SHA256
    assert len(FIFA_2026_REGULATIONS_SHA256) == 64
    assert len(table.rows) == 495
    assert [row.option for row in table.rows] == list(range(1, 496))
    assert set(table.by_qualifying_groups) == {
        "".join(groups) for groups in itertools.combinations("ABCDEFGHIJKL", 8)
    }


def test_every_annex_c_option_produces_32_unique_sources_and_no_group_rematch() -> None:
    table = load_fifa_2026_annex_c()
    fixed_sources = {
        0: ("2A", "2B"),
        2: ("1F", "2C"),
        3: ("1C", "2F"),
        5: ("2E", "2I"),
        10: ("2K", "2L"),
        11: ("1H", "2J"),
        13: ("1J", "2H"),
        15: ("2D", "2G"),
    }
    for row in table.rows:
        fixtures = first_knockout_source_pairs(
            FIFA_2026_ANNEX_C_POLICY,
            third_place_groups=tuple(row.qualifying_groups),
        )
        assert len(fixtures) == 16
        sources = [source for fixture in fixtures for source in fixture]
        assert len(sources) == len(set(sources)) == 32
        assert all(_slot_group(home) != _slot_group(away) for home, away in fixtures)
        assert {source for source in sources if source.startswith("3")} == {
            f"3{group}" for group in row.qualifying_groups
        }
        for match_offset, expected in fixed_sources.items():
            assert fixtures[match_offset] == expected


def test_m73_to_m104_source_graph_is_exactly_the_regulations_graph() -> None:
    assert later_knockout_source_pairs(FIFA_2026_ANNEX_C_POLICY, "round_of_16") == (
        ("W74", "W77"),
        ("W73", "W75"),
        ("W76", "W78"),
        ("W79", "W80"),
        ("W83", "W84"),
        ("W81", "W82"),
        ("W86", "W88"),
        ("W85", "W87"),
    )
    assert later_knockout_source_pairs(FIFA_2026_ANNEX_C_POLICY, "quarter_final") == (
        ("W89", "W90"),
        ("W93", "W94"),
        ("W91", "W92"),
        ("W95", "W96"),
    )
    assert later_knockout_source_pairs(FIFA_2026_ANNEX_C_POLICY, "semi_final") == (
        ("W97", "W98"),
        ("W99", "W100"),
    )
    assert later_knockout_source_pairs(FIFA_2026_ANNEX_C_POLICY, "third_place") == (
        ("L101", "L102"),
    )
    assert later_knockout_source_pairs(FIFA_2026_ANNEX_C_POLICY, "final") == (
        ("W101", "W102"),
    )


def test_2022_declarative_policy_also_prevents_same_group_first_round_matches() -> None:
    fixtures = first_knockout_source_pairs(WORLD_CUP_2022_CROSS_GROUP_POLICY)
    assert len(fixtures) == 8
    assert all(_slot_group(home) != _slot_group(away) for home, away in fixtures)
    assert len({source for fixture in fixtures for source in fixture}) == 16


def test_group_fixture_orientation_matches_official_four_team_draw_positions() -> None:
    assert GROUP_FIXTURE_POSITIONS == ((0, 1), (2, 3), (0, 2), (3, 1), (3, 0), (1, 2))


def test_head_to_head_precedes_overall_goal_difference_for_equal_points() -> None:
    teams = ["A", "B", "C", "D"]
    standings = {
        "A": _Standing(points=6, goals_for=20, goals_against=0),
        "B": _Standing(points=6, goals_for=2, goals_against=2),
        "C": _Standing(points=3, goals_for=1, goals_against=4),
        "D": _Standing(points=0, goals_for=0, goals_against=17),
    }
    results = [_GroupResult("A", "B", 0, 1)]

    ranked = _rank_group(teams, standings, results, dict(zip(teams, range(4), strict=True)))

    assert ranked[:2] == ["B", "A"]


def test_knockout_tie_proxy_uses_non_draw_strength_mass_and_is_swap_equivariant() -> None:
    home_strong = np.zeros((3, 3), dtype=float)
    home_strong[0, 0] = 0.9
    home_strong[1, 0] = 0.1
    rng = np.random.default_rng(2026)

    assert _sample_knockout_home_win(home_strong, rng) is True
    assert _sample_knockout_home_win(home_strong.T, rng) is False
