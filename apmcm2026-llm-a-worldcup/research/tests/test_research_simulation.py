"""RED behaviour tests for format-aware World Cup tournament simulation."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
import pytest

from worldcup_forecasting.models.count import score_matrix


def _required_symbol(symbol_name: str) -> Any:
    module_name = "worldcup_forecasting.simulation.tournament"
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(
            f"RED format-aware simulation contract is not implemented: import "
            f"{module_name!r} failed ({exc}).",
            pytrace=False,
        )
    if not hasattr(module, symbol_name):
        pytest.fail(
            f"RED format-aware simulation contract is not implemented: "
            f"{module_name}.{symbol_name} is missing.",
            pytrace=False,
        )
    return getattr(module, symbol_name)


def _symmetric_score_provider(
    home_team: str,
    away_team: str,
    stage: str,
    *_: object,
    **__: object,
) -> np.ndarray:
    del home_team, away_team, stage
    return score_matrix(1.25, 1.25, rho=-0.03, max_score=8, adaptive=False)


def _result_frame(result: Any, name: str) -> pd.DataFrame:
    value = result[name] if isinstance(result, dict) else getattr(result, name)
    assert isinstance(value, pd.DataFrame)
    return value


def _metadata(result: Any) -> dict[str, Any]:
    value = result["metadata"] if isinstance(result, dict) else result.metadata
    assert isinstance(value, dict)
    return value


def test_world_cup_2026_simulation_has_exact_format_and_advancement_invariants() -> None:
    format_type = _required_symbol("TournamentFormat")
    simulate = _required_symbol("simulate_tournament")
    tournament_format = format_type.world_cup_2026()
    teams = [f"T{index:02d}" for index in range(48)]
    result = simulate(
        teams=teams,
        tournament_format=tournament_format,
        score_provider=_symmetric_score_provider,
        n_simulations=3,
        seed=2026,
    )
    log = _result_frame(result, "match_log")
    summary = _result_frame(result, "team_summary")
    metadata = _metadata(result)

    assert metadata["team_count"] == 48
    assert metadata["group_count"] == 12
    assert metadata["best_third_qualifiers"] == 8
    assert metadata["matches_per_simulation"] == 104
    assert metadata["format_name"] == "world_cup_2026"
    assert metadata["bracket_policy"] == "fifa_2026_annex_c"
    assert metadata["annex_c_rows"] == 495
    assert len(metadata["format_source_sha256"]) == 64
    assert "strength_proxy" in metadata["knockout_tie_resolution"]
    assert set(log["simulation_id"]) == {0, 1, 2}
    expected_stage_counts = {
        "group_stage": 72,
        "round_of_32": 16,
        "round_of_16": 8,
        "quarter_final": 4,
        "semi_final": 2,
        "third_place": 1,
        "final": 1,
    }
    for simulation_id, simulation in log.groupby("simulation_id"):
        assert len(simulation) == 104, simulation_id
        assert simulation["stage"].value_counts().to_dict() == expected_stage_counts
        assert not simulation["home_team"].eq(simulation["away_team"]).any()
        assert set(simulation["home_team"]) | set(simulation["away_team"]) <= set(teams)
        group_fixture_sources = simulation.loc[
            simulation["group_label"].eq("A"),
            ["home_source_slot", "away_source_slot"],
        ]
        assert list(group_fixture_sources.itertuples(index=False, name=None)) == [
            ("A_draw1", "A_draw2"),
            ("A_draw3", "A_draw4"),
            ("A_draw1", "A_draw3"),
            ("A_draw4", "A_draw2"),
            ("A_draw4", "A_draw1"),
            ("A_draw2", "A_draw3"),
        ]
        round_of_32 = simulation.loc[simulation["stage"].eq("round_of_32")]
        assert len(round_of_32) == 16
        for match in round_of_32.itertuples(index=False):
            assert match.home_source_slot[1] != match.away_source_slot[1]
        exact_later_sources = {
            89: ("W74", "W77"),
            90: ("W73", "W75"),
            91: ("W76", "W78"),
            92: ("W79", "W80"),
            93: ("W83", "W84"),
            94: ("W81", "W82"),
            95: ("W86", "W88"),
            96: ("W85", "W87"),
            97: ("W89", "W90"),
            98: ("W93", "W94"),
            99: ("W91", "W92"),
            100: ("W95", "W96"),
            101: ("W97", "W98"),
            102: ("W99", "W100"),
            103: ("L101", "L102"),
            104: ("W101", "W102"),
        }
        for match_number, sources in exact_later_sources.items():
            match = simulation.loc[simulation["match_number"].eq(match_number)].iloc[0]
            assert (match["home_source_slot"], match["away_source_slot"]) == sources
        final = simulation.loc[simulation["stage"].eq("final")]
        assert len(final) == 1
        assert final.iloc[0]["winner"] in {
            final.iloc[0]["home_team"],
            final.iloc[0]["away_team"],
        }
        # A knockout loser can appear only in the explicitly separate third-place match.
        knockout_order = {
            "round_of_32": 0,
            "round_of_16": 1,
            "quarter_final": 2,
            "semi_final": 3,
            "final": 4,
        }
        main_knockout = simulation.loc[simulation["stage"].isin(knockout_order)].copy()
        main_knockout["order"] = main_knockout["stage"].map(knockout_order)
        for match in main_knockout.itertuples(index=False):
            loser = match.away_team if match.winner == match.home_team else match.home_team
            later = main_knockout.loc[main_knockout["order"].gt(match.order)]
            assert not (later["home_team"].eq(loser) | later["away_team"].eq(loser)).any()

    assert set(summary["team"]) == set(teams)
    assert (summary["champion_probability"].between(0, 1)).all()
    assert summary["champion_probability"].sum() == pytest.approx(1.0, abs=1e-12)
    assert summary["champion_count"].sum() == 3


def test_simulator_is_format_driven_for_32_and_48_team_world_cups() -> None:
    format_type = _required_symbol("TournamentFormat")
    simulate = _required_symbol("simulate_tournament")
    cases = (
        (format_type.world_cup_2022(), [f"L{index:02d}" for index in range(32)], 64),
        (format_type.world_cup_2026(), [f"N{index:02d}" for index in range(48)], 104),
    )
    for tournament_format, teams, expected_matches in cases:
        result = simulate(
            teams=teams,
            tournament_format=tournament_format,
            score_provider=_symmetric_score_provider,
            n_simulations=1,
            seed=11,
        )
        assert len(_result_frame(result, "match_log")) == expected_matches
        assert _metadata(result)["matches_per_simulation"] == expected_matches


def test_tournament_simulation_is_seed_reproducible_and_label_equivariant() -> None:
    format_type = _required_symbol("TournamentFormat")
    simulate = _required_symbol("simulate_tournament")
    tournament_format = format_type.world_cup_2026()
    teams = [f"T{index:02d}" for index in range(48)]
    renamed = [f"RENAMED_{index:02d}" for index in range(48)]

    first = simulate(
        teams=teams,
        tournament_format=tournament_format,
        score_provider=_symmetric_score_provider,
        n_simulations=2,
        seed=947,
    )
    repeated = simulate(
        teams=teams,
        tournament_format=tournament_format,
        score_provider=_symmetric_score_provider,
        n_simulations=2,
        seed=947,
    )
    changed_seed = simulate(
        teams=teams,
        tournament_format=tournament_format,
        score_provider=_symmetric_score_provider,
        n_simulations=2,
        seed=948,
    )
    renamed_result = simulate(
        teams=renamed,
        tournament_format=tournament_format,
        score_provider=_symmetric_score_provider,
        n_simulations=2,
        seed=947,
    )

    first_log = _result_frame(first, "match_log").reset_index(drop=True)
    repeated_log = _result_frame(repeated, "match_log").reset_index(drop=True)
    changed_log = _result_frame(changed_seed, "match_log").reset_index(drop=True)
    pd.testing.assert_frame_equal(first_log, repeated_log)
    assert not first_log[["home_goals", "away_goals", "winner"]].equals(
        changed_log[["home_goals", "away_goals", "winner"]]
    )

    rename_map = dict(zip(renamed, teams, strict=True))
    normalized = _result_frame(renamed_result, "match_log").copy()
    for column in ("home_team", "away_team", "winner"):
        normalized[column] = normalized[column].map(rename_map)
    pd.testing.assert_frame_equal(first_log, normalized.reset_index(drop=True))

    first_summary = _result_frame(first, "team_summary").sort_values("team").reset_index(drop=True)
    renamed_summary = _result_frame(renamed_result, "team_summary").copy()
    renamed_summary["team"] = renamed_summary["team"].map(rename_map)
    renamed_summary = renamed_summary.sort_values("team").reset_index(drop=True)
    pd.testing.assert_frame_equal(first_summary, renamed_summary)
