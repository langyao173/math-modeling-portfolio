"""Convenience imports for the three mandatory official baselines."""

from .aggregate import AggregateOLS, AggregateOLSConfig, AggregateOLSReport
from .match import MatchGoalOLS, MatchGoalOLSConfig
from .outcome import MANDATORY_OUTCOME_FEATURES, OUTCOME_ORDER, OutcomeLogit, OutcomeLogitConfig

__all__ = [
    "AggregateOLS",
    "AggregateOLSConfig",
    "AggregateOLSReport",
    "MANDATORY_OUTCOME_FEATURES",
    "MatchGoalOLS",
    "MatchGoalOLSConfig",
    "OUTCOME_ORDER",
    "OutcomeLogit",
    "OutcomeLogitConfig",
]
