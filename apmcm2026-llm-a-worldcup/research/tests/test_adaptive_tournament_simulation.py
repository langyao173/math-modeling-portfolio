"""Adaptive Monte Carlo contracts for the official 2026 scenario engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from worldcup_forecasting.models.count import score_matrix
from worldcup_forecasting.simulation.adaptive import (
    AdaptiveMonteCarloConfig,
    simulate_tournament_adaptive,
)
from worldcup_forecasting.simulation.tournament import TournamentFormat


def _provider(home: str, away: str, stage: str) -> np.ndarray:
    del home, away, stage
    return score_matrix(1.25, 1.25, rho=-0.03, max_score=6, adaptive=False)


def _config(simulations: int, *, audit: int = 2) -> AdaptiveMonteCarloConfig:
    return AdaptiveMonteCarloConfig(
        minimum_simulations=simulations,
        batch_size=simulations,
        maximum_simulations=simulations,
        advancement_mcse_threshold=1.0,
        champion_mcse_threshold=1.0,
        audit_log_simulations=audit,
    )


def _teams() -> list[str]:
    return [f"T{index:02d}" for index in range(48)]


def test_adaptive_summary_contains_full_probability_and_mcse_contract() -> None:
    result = simulate_tournament_adaptive(
        teams=_teams(),
        tournament_format=TournamentFormat.world_cup_2026(),
        score_provider=_provider,
        seed=20260718,
        config=_config(20),
    )

    assert result.metadata["n_simulations"] == 20
    assert result.metadata["confidence_level"] == 0.95
    assert result.metadata["stopping_reason"] == "all_mcse_thresholds_satisfied"
    assert len(result.match_log) == 2 * 104
    assert result.stopping_ledger["decision"].tolist() == ["all_mcse_thresholds_satisfied"]
    assert len(result.batch_event_counts) == 48 * 6
    assert len(result.continuous_sufficient_statistics) == 49
    terminal_counts = result.batch_event_counts.pivot(
        index="team", columns="event", values="count"
    ).sort_index()
    assert np.array_equal(
        terminal_counts["champion"].to_numpy(),
        result.team_summary.sort_values("team")["champion_count"].to_numpy(),
    )
    assert result.stage_summary.set_index("stage")["matches_per_simulation"].to_dict() == {
        "group_stage": 72,
        "round_of_32": 16,
        "round_of_16": 8,
        "quarter_final": 4,
        "semi_final": 2,
        "third_place": 1,
        "final": 1,
    }
    probability_columns = [
        "champion_probability",
        "final_probability",
        "semi_final_probability",
        "quarter_final_probability",
        "round_of_16_probability",
        "knockout_probability",
    ]
    values = result.team_summary[probability_columns].to_numpy(float)
    assert np.all(np.diff(values, axis=1) >= -1e-12)
    assert np.isclose(result.team_summary["champion_probability"].sum(), 1.0)
    assert {
        "expected_group_points",
        "expected_group_points_mcse",
        "expected_tournament_goals",
        "expected_tournament_goals_mcse",
    }.issubset(result.team_summary.columns)


def test_adaptive_run_is_seed_exact_and_different_seed_is_within_mc_error() -> None:
    arguments = {
        "teams": _teams(),
        "tournament_format": TournamentFormat.world_cup_2026(),
        "score_provider": _provider,
        "config": _config(500, audit=1),
    }
    first = simulate_tournament_adaptive(seed=947, **arguments)
    repeated = simulate_tournament_adaptive(seed=947, **arguments)
    changed = simulate_tournament_adaptive(seed=948, **arguments)

    pd.testing.assert_frame_equal(first.team_summary, repeated.team_summary)
    pd.testing.assert_frame_equal(first.match_log, repeated.match_log)
    assert not first.team_summary["champion_count"].equals(changed.team_summary["champion_count"])
    difference = np.abs(
        first.team_summary["champion_probability"].to_numpy()
        - changed.team_summary["champion_probability"].to_numpy()
    )
    combined = np.sqrt(
        first.team_summary["champion_mcse"].to_numpy() ** 2
        + changed.team_summary["champion_mcse"].to_numpy() ** 2
    )
    assert np.all(difference <= 5.0 * combined + 1.0 / 500)


def test_adaptive_run_stops_at_predeclared_maximum_when_precision_is_unmet() -> None:
    config = AdaptiveMonteCarloConfig(
        minimum_simulations=10,
        batch_size=5,
        maximum_simulations=10,
        advancement_mcse_threshold=1e-12,
        champion_mcse_threshold=1e-12,
        audit_log_simulations=0,
    )
    result = simulate_tournament_adaptive(
        teams=_teams(),
        tournament_format=TournamentFormat.world_cup_2026(),
        score_provider=_provider,
        seed=31,
        config=config,
    )

    assert result.metadata["n_simulations"] == 10
    assert result.metadata["stopping_reason"] == "maximum_simulations_reached"
    assert result.metadata["mcse_thresholds_satisfied"] is False
    assert result.match_log.empty
    assert result.stopping_ledger.iloc[-1]["decision"] == "maximum_simulations_reached"


def test_audit_retention_cannot_exceed_minimum_run_length() -> None:
    with np.testing.assert_raises(ValueError):
        AdaptiveMonteCarloConfig(
            minimum_simulations=10,
            batch_size=5,
            maximum_simulations=20,
            advancement_mcse_threshold=0.1,
            champion_mcse_threshold=0.1,
            audit_log_simulations=11,
        )


def test_symmetric_teams_have_approximately_symmetric_champion_probabilities() -> None:
    result = simulate_tournament_adaptive(
        teams=_teams(),
        tournament_format=TournamentFormat.world_cup_2026(),
        score_provider=_provider,
        seed=2718,
        config=_config(2_000, audit=0),
    )
    probabilities = result.team_summary["champion_probability"].to_numpy(float)

    assert float(np.max(np.abs(probabilities - 1.0 / 48.0))) < 0.02
