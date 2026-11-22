"""Mandatory baselines and research forecasting models.

Built-ins are imported explicitly so registration is deterministic and auditable.
"""

from .advanced_count import (
    EquivariantCountConfig,
    EquivariantCountModel,
    HierarchicalPaceConfig,
    HierarchicalPaceScoreModel,
    joint_batch_from_rates,
)
from .aggregate import AggregateOLS, AggregateOLSConfig, AggregateOLSReport
from .calibration import TemperatureScaler
from .count import (
    JointGoalBatch,
    JointPoissonConfig,
    JointPoissonModel,
    outcome_probabilities,
    score_matrix,
)
from .decision import OOFCostSensitiveDecisionRule
from .ensemble import ConvexOutcomeEnsemble
from .half_time import (
    HalfTimeIncrementConfig,
    HalfTimeIncrementModel,
    enforce_half_time_monotonicity,
)
from .match import MatchGoalOLS, MatchGoalOLSConfig
from .outcome import (
    MANDATORY_OUTCOME_FEATURES,
    OUTCOME_ORDER,
    OutcomeLogit,
    OutcomeLogitConfig,
)
from .registry import (
    MODEL_REGISTRY,
    ModelSpec,
    TaskKind,
    create_model,
    get_model_spec,
    list_models,
    register_model,
)
from .residuals import EquivariantResidualConfig, EquivariantResidualCorrector
from .score_calibration import (
    DiagonalScoreCalibrator,
    OutcomePartitionCalibrator,
    minimal_kl_partition_projection,
    minimum_kl_outcome_projection,
)

__all__ = [
    "AggregateOLS",
    "AggregateOLSConfig",
    "AggregateOLSReport",
    "ConvexOutcomeEnsemble",
    "DiagonalScoreCalibrator",
    "EquivariantCountConfig",
    "EquivariantCountModel",
    "EquivariantResidualConfig",
    "EquivariantResidualCorrector",
    "HalfTimeIncrementConfig",
    "HalfTimeIncrementModel",
    "JointGoalBatch",
    "JointPoissonConfig",
    "JointPoissonModel",
    "HierarchicalPaceConfig",
    "HierarchicalPaceScoreModel",
    "MANDATORY_OUTCOME_FEATURES",
    "MODEL_REGISTRY",
    "MatchGoalOLS",
    "MatchGoalOLSConfig",
    "ModelSpec",
    "OUTCOME_ORDER",
    "OutcomeLogit",
    "OutcomeLogitConfig",
    "OutcomePartitionCalibrator",
    "OOFCostSensitiveDecisionRule",
    "TaskKind",
    "TemperatureScaler",
    "create_model",
    "enforce_half_time_monotonicity",
    "get_model_spec",
    "joint_batch_from_rates",
    "list_models",
    "minimum_kl_outcome_projection",
    "minimal_kl_partition_projection",
    "outcome_probabilities",
    "register_model",
    "score_matrix",
]
