"""Adaptive Monte Carlo execution for the official 2026 format scenario."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from ._engine import simulate_once
from .format_policy import (
    FIFA_2026_ANNEX_C_CSV_SHA256,
    FIFA_2026_ANNEX_C_POLICY,
    FIFA_2026_REGULATIONS_SHA256,
    FIFA_2026_REGULATIONS_URL,
)
from .tournament import ScoreProvider, TournamentFormat, _validate_inputs

ADVANCEMENT_EVENTS = (
    "knockout",
    "round_of_16",
    "quarter_final",
    "semi_final",
    "final",
)
PROBABILITY_EVENTS = (*ADVANCEMENT_EVENTS, "champion")
STAGES = (
    "group_stage",
    "round_of_32",
    "round_of_16",
    "quarter_final",
    "semi_final",
    "third_place",
    "final",
)
FROZEN_ADAPTIVE_SEED = 20_260_717
FROZEN_PROTOCOL_DEVIATION_SHA256 = (
    "6f487cf1331d45e2591bcc905444bb2f1214481d2fab7a5c3c3c47f40e75cf0c"
)


@dataclass(frozen=True, slots=True)
class AdaptiveMonteCarloConfig:
    """Predeclared stopping and evidence-retention contract."""

    minimum_simulations: int = 10_000
    batch_size: int = 5_000
    maximum_simulations: int = 250_000
    advancement_mcse_threshold: float = 0.002
    champion_mcse_threshold: float = 0.001
    confidence_level: float = 0.95
    audit_log_simulations: int = 25

    def __post_init__(self) -> None:
        integers = (
            self.minimum_simulations,
            self.batch_size,
            self.maximum_simulations,
            self.audit_log_simulations,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise TypeError("Adaptive Monte Carlo counts must be integers")
        if (
            self.minimum_simulations <= 0
            or self.batch_size <= 0
            or self.maximum_simulations < self.minimum_simulations
            or self.audit_log_simulations < 0
            or self.audit_log_simulations > self.minimum_simulations
        ):
            raise ValueError("Adaptive Monte Carlo counts are inconsistent")
        thresholds = (
            self.advancement_mcse_threshold,
            self.champion_mcse_threshold,
        )
        if any(not np.isfinite(value) or value <= 0 for value in thresholds):
            raise ValueError("Monte Carlo standard-error thresholds must be positive")
        if not np.isfinite(self.confidence_level) or not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between zero and one")


@dataclass(frozen=True, slots=True)
class AdaptiveTournamentResult:
    """Bounded logs, full probability summaries, and stopping metadata."""

    match_log: pd.DataFrame
    team_summary: pd.DataFrame
    stage_summary: pd.DataFrame
    stopping_ledger: pd.DataFrame
    batch_event_counts: pd.DataFrame
    continuous_sufficient_statistics: pd.DataFrame
    metadata: dict[str, Any]


FROZEN_ADAPTIVE_CONFIG = AdaptiveMonteCarloConfig(
    minimum_simulations=10_000,
    batch_size=5_000,
    maximum_simulations=250_000,
    advancement_mcse_threshold=0.002,
    champion_mcse_threshold=0.001,
    confidence_level=0.95,
    audit_log_simulations=25,
)


def adaptive_execution_contract_payload(
    config: AdaptiveMonteCarloConfig, *, seed: int
) -> dict[str, Any]:
    """Return a canonical contract for one adaptive execution."""
    return {
        "schema_version": 1,
        "seed": seed,
        "minimum_simulations": config.minimum_simulations,
        "batch_size": config.batch_size,
        "maximum_simulations": config.maximum_simulations,
        "advancement_mcse_threshold": config.advancement_mcse_threshold,
        "champion_mcse_threshold": config.champion_mcse_threshold,
        "confidence_level": config.confidence_level,
        "audit_log_simulations": config.audit_log_simulations,
        "format_source_sha256": FIFA_2026_REGULATIONS_SHA256,
        "annex_c_csv_sha256": FIFA_2026_ANNEX_C_CSV_SHA256,
        "protocol_deviation_sha256": FROZEN_PROTOCOL_DEVIATION_SHA256,
        "probability_interval": "terminal_n_nominal_wilson",
        "group_tie_proxy": "seeded_random_symmetry_preserving",
    }


def adaptive_contract_payload() -> dict[str, Any]:
    """Return the sole predeclared production simulation contract."""
    return adaptive_execution_contract_payload(FROZEN_ADAPTIVE_CONFIG, seed=FROZEN_ADAPTIVE_SEED)


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def adaptive_contract_sha256() -> str:
    """Hash the canonical production simulation contract."""
    return _payload_sha256(adaptive_contract_payload())


def simulate_tournament_adaptive(
    *,
    teams: list[str] | tuple[str, ...],
    tournament_format: TournamentFormat,
    score_provider: ScoreProvider,
    seed: int,
    config: AdaptiveMonteCarloConfig | None = None,
) -> AdaptiveTournamentResult:
    """Simulate until every predeclared probability MCSE gate passes or max is hit."""
    cfg = config or AdaptiveMonteCarloConfig()
    scenario_teams = _validate_inputs(
        teams,
        tournament_format=tournament_format,
        score_provider=score_provider,
        n_simulations=cfg.maximum_simulations,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    team_index = {team: index for index, team in enumerate(scenario_teams)}
    event_counts = np.zeros((len(scenario_teams), len(PROBABILITY_EVENTS)), dtype=np.int64)
    group_point_sum = np.zeros(len(scenario_teams), dtype=float)
    group_point_square_sum = np.zeros(len(scenario_teams), dtype=float)
    goal_sum = np.zeros(len(scenario_teams), dtype=float)
    goal_square_sum = np.zeros(len(scenario_teams), dtype=float)
    tournament_goal_sum = 0.0
    tournament_goal_square_sum = 0.0
    stage_counts = {stage: 0 for stage in STAGES}
    audit_rows: list[dict[str, Any]] = []
    stopping_rows: list[dict[str, Any]] = []
    batch_count_rows: list[dict[str, Any]] = []
    simulations = 0
    stopping_reason = "maximum_simulations_reached"

    while simulations < cfg.maximum_simulations:
        batch_end = min(simulations + cfg.batch_size, cfg.maximum_simulations)
        for simulation_id in range(simulations, batch_end):
            ranking_proxy = dict(
                zip(
                    scenario_teams,
                    rng.permutation(len(scenario_teams)).tolist(),
                    strict=True,
                )
            )
            champion, rows = simulate_once(
                scenario_teams,
                tournament_format,
                score_provider,
                rng,
                simulation_id=simulation_id,
                team_order=ranking_proxy,
            )
            group_points = np.zeros(len(scenario_teams), dtype=float)
            goals = np.zeros(len(scenario_teams), dtype=float)
            participants: dict[str, set[str]] = {event: set() for event in ADVANCEMENT_EVENTS}
            total_goals = 0.0
            for row in rows:
                stage = str(row["stage"])
                stage_counts[stage] += 1
                home = str(row["home_team"])
                away = str(row["away_team"])
                home_index = team_index[home]
                away_index = team_index[away]
                home_goals = int(row["home_goals"])
                away_goals = int(row["away_goals"])
                goals[home_index] += home_goals
                goals[away_index] += away_goals
                total_goals += home_goals + away_goals
                event = "knockout" if stage == "round_of_32" else stage
                if event in participants:
                    participants[event].update((home, away))
                if stage == "group_stage":
                    if home_goals > away_goals:
                        group_points[home_index] += 3
                    elif away_goals > home_goals:
                        group_points[away_index] += 3
                    else:
                        group_points[home_index] += 1
                        group_points[away_index] += 1
            for column, event in enumerate(ADVANCEMENT_EVENTS):
                for team in participants[event]:
                    event_counts[team_index[team], column] += 1
            event_counts[team_index[champion], -1] += 1
            group_point_sum += group_points
            group_point_square_sum += group_points**2
            goal_sum += goals
            goal_square_sum += goals**2
            tournament_goal_sum += total_goals
            tournament_goal_square_sum += total_goals**2
            if simulation_id < cfg.audit_log_simulations:
                audit_rows.extend(rows)
        simulations = batch_end
        advancement_mcse, champion_mcse = _maximum_probability_mcse(event_counts, simulations)
        thresholds_satisfied = (
            simulations >= cfg.minimum_simulations
            and advancement_mcse <= cfg.advancement_mcse_threshold
            and champion_mcse <= cfg.champion_mcse_threshold
        )
        maximum_reached = simulations == cfg.maximum_simulations
        if thresholds_satisfied:
            stopping_reason = "all_mcse_thresholds_satisfied"
        elif maximum_reached:
            stopping_reason = "maximum_simulations_reached"
        else:
            stopping_reason = "continue"
        stopping_rows.append(
            {
                "batch_index": len(stopping_rows),
                "simulations": simulations,
                "maximum_advancement_mcse": advancement_mcse,
                "maximum_champion_mcse": champion_mcse,
                "minimum_reached": simulations >= cfg.minimum_simulations,
                "advancement_threshold_satisfied": (
                    advancement_mcse <= cfg.advancement_mcse_threshold
                ),
                "champion_threshold_satisfied": (champion_mcse <= cfg.champion_mcse_threshold),
                "thresholds_satisfied": thresholds_satisfied,
                "maximum_reached": maximum_reached,
                "decision": stopping_reason,
            }
        )
        batch_index = len(stopping_rows) - 1
        for team_index_value, team in enumerate(scenario_teams):
            for event_index, event in enumerate(PROBABILITY_EVENTS):
                batch_count_rows.append(
                    {
                        "batch_index": batch_index,
                        "simulations": simulations,
                        "team": team,
                        "event": event,
                        "count": int(event_counts[team_index_value, event_index]),
                    }
                )
        if thresholds_satisfied or maximum_reached:
            break

    team_summary = _team_summary(
        scenario_teams,
        event_counts,
        group_point_sum,
        group_point_square_sum,
        goal_sum,
        goal_square_sum,
        simulations,
        cfg.confidence_level,
    )
    advancement_mcse, champion_mcse = _maximum_probability_mcse(event_counts, simulations)
    total_goal_mean, total_goal_mcse = _continuous_summary(
        tournament_goal_sum,
        tournament_goal_square_sum,
        simulations,
    )
    sufficient_statistics = _continuous_sufficient_statistics(
        scenario_teams,
        simulations=simulations,
        group_point_sum=group_point_sum,
        group_point_square_sum=group_point_square_sum,
        goal_sum=goal_sum,
        goal_square_sum=goal_square_sum,
        tournament_goal_sum=tournament_goal_sum,
        tournament_goal_square_sum=tournament_goal_square_sum,
    )
    match_log = pd.DataFrame(audit_rows)
    stopping_ledger = pd.DataFrame(stopping_rows)
    batch_event_counts = pd.DataFrame(batch_count_rows)
    stage_summary = pd.DataFrame(
        {
            "stage": STAGES,
            "matches_per_simulation": [stage_counts[stage] // simulations for stage in STAGES],
            "simulated_matches": [stage_counts[stage] for stage in STAGES],
        }
    )
    metadata = {
        "format_name": tournament_format.name,
        "team_count": tournament_format.team_count,
        "group_count": tournament_format.group_count,
        "group_size": tournament_format.group_size,
        "automatic_qualifiers_per_group": (tournament_format.automatic_qualifiers_per_group),
        "best_third_qualifiers": tournament_format.best_third_qualifiers,
        "matches_per_simulation": tournament_format.matches_per_simulation,
        "bracket_policy": tournament_format.bracket_policy,
        "seed": int(seed),
        "n_simulations": simulations,
        "minimum_simulations": cfg.minimum_simulations,
        "batch_size": cfg.batch_size,
        "maximum_simulations": cfg.maximum_simulations,
        "advancement_mcse_threshold": cfg.advancement_mcse_threshold,
        "champion_mcse_threshold": cfg.champion_mcse_threshold,
        "confidence_level": cfg.confidence_level,
        "maximum_advancement_mcse": advancement_mcse,
        "maximum_champion_mcse": champion_mcse,
        "stopping_reason": stopping_reason,
        "mcse_thresholds_satisfied": (
            advancement_mcse <= cfg.advancement_mcse_threshold
            and champion_mcse <= cfg.champion_mcse_threshold
        ),
        "audit_log_simulations": cfg.audit_log_simulations,
        "audit_log_rows": len(match_log),
        "mean_tournament_goals": total_goal_mean,
        "mean_tournament_goals_mcse": total_goal_mcse,
        "probability_interval": (f"terminal_n_nominal_wilson_{cfg.confidence_level:.6f}"),
        "group_tie_proxy": "seeded_random_symmetry_preserving",
        "adaptive_contract": adaptive_execution_contract_payload(cfg, seed=seed),
        "adaptive_contract_sha256": _payload_sha256(
            adaptive_execution_contract_payload(cfg, seed=seed)
        ),
        "protocol_deviation_sha256": FROZEN_PROTOCOL_DEVIATION_SHA256,
        "stopping_ledger_rows": len(stopping_ledger),
        "batch_event_count_rows": len(batch_event_counts),
        "continuous_sufficient_statistic_rows": len(sufficient_statistics),
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
    return AdaptiveTournamentResult(
        match_log,
        team_summary,
        stage_summary,
        stopping_ledger,
        batch_event_counts,
        sufficient_statistics,
        metadata,
    )


def _maximum_probability_mcse(counts: np.ndarray, simulations: int) -> tuple[float, float]:
    probability = counts / simulations
    mcse = np.sqrt(probability * (1.0 - probability) / simulations)
    return float(np.max(mcse[:, :-1])), float(np.max(mcse[:, -1]))


def _team_summary(
    teams: list[str],
    counts: np.ndarray,
    point_sum: np.ndarray,
    point_square_sum: np.ndarray,
    goal_sum: np.ndarray,
    goal_square_sum: np.ndarray,
    simulations: int,
    confidence_level: float,
) -> pd.DataFrame:
    payload: dict[str, Any] = {"team": teams}
    for column, event in enumerate(PROBABILITY_EVENTS):
        count = counts[:, column]
        probability = count / simulations
        mcse = np.sqrt(probability * (1.0 - probability) / simulations)
        lower, upper = _wilson_interval(count, simulations, confidence_level)
        payload[f"{event}_count"] = count
        payload[f"{event}_probability"] = probability
        payload[f"{event}_mcse"] = mcse
        payload[f"{event}_ci_lower"] = lower
        payload[f"{event}_ci_upper"] = upper
    point_mean, point_mcse = _continuous_vector_summary(point_sum, point_square_sum, simulations)
    goal_mean, goal_mcse = _continuous_vector_summary(goal_sum, goal_square_sum, simulations)
    payload["expected_group_points"] = point_mean
    payload["expected_group_points_mcse"] = point_mcse
    payload["expected_tournament_goals"] = goal_mean
    payload["expected_tournament_goals_mcse"] = goal_mcse
    return pd.DataFrame(payload)


def _wilson_interval(
    count: np.ndarray,
    simulations: int,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    probability = count / simulations
    denominator = 1.0 + z**2 / simulations
    center = (probability + z**2 / (2.0 * simulations)) / denominator
    radius = (
        z
        * np.sqrt(probability * (1.0 - probability) / simulations + z**2 / (4.0 * simulations**2))
        / denominator
    )
    return np.clip(center - radius, 0.0, 1.0), np.clip(center + radius, 0.0, 1.0)


def _continuous_vector_summary(
    total: np.ndarray, square_total: np.ndarray, simulations: int
) -> tuple[np.ndarray, np.ndarray]:
    mean = total / simulations
    if simulations == 1:
        return mean, np.full_like(mean, np.nan)
    variance = np.maximum(
        (square_total - total**2 / simulations) / (simulations - 1),
        0.0,
    )
    return mean, np.sqrt(variance / simulations)


def _continuous_summary(total: float, square_total: float, simulations: int) -> tuple[float, float]:
    mean = total / simulations
    if simulations == 1:
        return float(mean), float("nan")
    variance = max((square_total - total**2 / simulations) / (simulations - 1), 0.0)
    return float(mean), float(sqrt(variance / simulations))


def _continuous_sufficient_statistics(
    teams: list[str],
    *,
    simulations: int,
    group_point_sum: np.ndarray,
    group_point_square_sum: np.ndarray,
    goal_sum: np.ndarray,
    goal_square_sum: np.ndarray,
    tournament_goal_sum: float,
    tournament_goal_square_sum: float,
) -> pd.DataFrame:
    rows = [
        {
            "scope": "team",
            "entity": team,
            "simulations": simulations,
            "group_points_sum": float(group_point_sum[index]),
            "group_points_square_sum": float(group_point_square_sum[index]),
            "goals_sum": float(goal_sum[index]),
            "goals_square_sum": float(goal_square_sum[index]),
        }
        for index, team in enumerate(teams)
    ]
    rows.append(
        {
            "scope": "tournament",
            "entity": "__tournament__",
            "simulations": simulations,
            "group_points_sum": 0.0,
            "group_points_square_sum": 0.0,
            "goals_sum": float(tournament_goal_sum),
            "goals_square_sum": float(tournament_goal_square_sum),
        }
    )
    return pd.DataFrame(rows)


__all__ = [
    "ADVANCEMENT_EVENTS",
    "AdaptiveMonteCarloConfig",
    "AdaptiveTournamentResult",
    "FROZEN_ADAPTIVE_CONFIG",
    "FROZEN_ADAPTIVE_SEED",
    "FROZEN_PROTOCOL_DEVIATION_SHA256",
    "PROBABILITY_EVENTS",
    "adaptive_execution_contract_payload",
    "adaptive_contract_payload",
    "adaptive_contract_sha256",
    "simulate_tournament_adaptive",
]
