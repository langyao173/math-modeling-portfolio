"""Optional cross-language experiments that never enter the production chain."""

from .audit_baselines import FrozenAuditBaselines, fit_fixed_audit_baselines
from .baseline_suite import (
    BaselineEvaluationConfig,
    BaselineSuiteResult,
    run_baseline_suite,
)
from .evidence import ResearchEvidence, reconstruct_joint_batch, summarize_oof_evidence
from .format_transport import FormatTransportResult, run_format_transport_audits
from .matlab_bridge import verify_advanced_matlab_artifacts, verify_matlab_parity
from .nested_calibration import NestedCalibrationResult, apply_nested_calibration
from .nested_decision import NestedDecisionResult, apply_nested_decision
from .nested_selection import (
    CandidateSpec,
    FinalDevelopmentSelectionResult,
    FinalFittedCountPredictor,
    NestedSelectionConfig,
    NestedSelectionResult,
    fit_final_selected_model,
    frozen_candidate_specs,
    run_nested_selection,
    select_final_development_spec,
)
from .research_features import (
    ResearchFeatureBundle,
    build_research_feature_bundle,
    config_hash,
    make_count_config,
)
from .sarmanov_dependence import (
    SarmanovExplorationResult,
    apply_nested_sarmanov_exploration,
    fit_sarmanov_theta,
    sarmanov_joint_from_marginals,
    transform_joint_batch_sarmanov,
)

__all__ = [
    "BaselineEvaluationConfig",
    "BaselineSuiteResult",
    "CandidateSpec",
    "FinalDevelopmentSelectionResult",
    "FinalFittedCountPredictor",
    "FrozenAuditBaselines",
    "FormatTransportResult",
    "NestedCalibrationResult",
    "NestedDecisionResult",
    "NestedSelectionConfig",
    "NestedSelectionResult",
    "ResearchFeatureBundle",
    "ResearchEvidence",
    "SarmanovExplorationResult",
    "build_research_feature_bundle",
    "apply_nested_calibration",
    "apply_nested_decision",
    "apply_nested_sarmanov_exploration",
    "config_hash",
    "fit_final_selected_model",
    "fit_sarmanov_theta",
    "fit_fixed_audit_baselines",
    "frozen_candidate_specs",
    "make_count_config",
    "reconstruct_joint_batch",
    "run_baseline_suite",
    "run_format_transport_audits",
    "run_nested_selection",
    "sarmanov_joint_from_marginals",
    "select_final_development_spec",
    "summarize_oof_evidence",
    "transform_joint_batch_sarmanov",
    "verify_advanced_matlab_artifacts",
    "verify_matlab_parity",
]
