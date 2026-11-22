"""Format-aware, scenario-only World Cup tournament simulation."""

from .adaptive import (
    FROZEN_ADAPTIVE_CONFIG,
    FROZEN_ADAPTIVE_SEED,
    AdaptiveMonteCarloConfig,
    AdaptiveTournamentResult,
    adaptive_contract_payload,
    adaptive_contract_sha256,
    adaptive_execution_contract_payload,
    simulate_tournament_adaptive,
)
from .scenario import FormatScenarioProvider, build_format_scenario_provider
from .tournament import TournamentFormat, TournamentSimulationResult, simulate_tournament

__all__ = [
    "FormatScenarioProvider",
    "AdaptiveMonteCarloConfig",
    "AdaptiveTournamentResult",
    "FROZEN_ADAPTIVE_CONFIG",
    "FROZEN_ADAPTIVE_SEED",
    "TournamentFormat",
    "TournamentSimulationResult",
    "build_format_scenario_provider",
    "adaptive_contract_payload",
    "adaptive_contract_sha256",
    "adaptive_execution_contract_payload",
    "simulate_tournament",
    "simulate_tournament_adaptive",
]
