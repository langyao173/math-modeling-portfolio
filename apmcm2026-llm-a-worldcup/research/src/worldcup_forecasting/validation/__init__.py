"""Temporal validation and statistical comparison utilities."""

from .joint_metrics import (
    classwise_calibration_diagnostics,
    joint_forecast_metrics,
    randomized_discrete_pit,
    scoreline_negative_log_likelihood,
    total_goal_crps,
)
from .nested import CandidateBudget, nested_expanding_origin_folds
from .promotion import (
    PromotionRejectedError,
    evaluate_advanced_promotion,
    evaluate_calibration_retention,
    evaluate_promotion_artifacts,
    issue_promotion_certificate,
    verify_promotion_certificate,
)
from .research_contracts import (
    validate_ablation_matrix,
    validate_oof_frame,
    validate_stress_matrix,
    write_oof_predictions,
)
from .transport import kl_edition_robust_risk, paired_tournament_cluster_interval

__all__ = [
    "CandidateBudget",
    "PromotionRejectedError",
    "classwise_calibration_diagnostics",
    "evaluate_advanced_promotion",
    "evaluate_calibration_retention",
    "evaluate_promotion_artifacts",
    "issue_promotion_certificate",
    "joint_forecast_metrics",
    "kl_edition_robust_risk",
    "nested_expanding_origin_folds",
    "paired_tournament_cluster_interval",
    "randomized_discrete_pit",
    "scoreline_negative_log_likelihood",
    "total_goal_crps",
    "validate_ablation_matrix",
    "validate_oof_frame",
    "validate_stress_matrix",
    "verify_promotion_certificate",
    "write_oof_predictions",
]
