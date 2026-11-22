"""Freeze the three fixed official/independent/DC audit comparators."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from worldcup_forecasting.models.count import JointPoissonConfig, JointPoissonModel
from worldcup_forecasting.models.outcome import OutcomeLogit, OutcomeLogitConfig
from worldcup_forecasting.validation.artifact_integrity import payload_sha256

_DYNAMIC_FEATURES = ("pre_match_elo_diff", "attack_state_diff", "defense_state_diff")


@dataclass(frozen=True)
class FrozenAuditBaselines:
    """Three fitted comparators and deterministic pre-audit identities."""

    official_outcome_logit: OutcomeLogit
    independent_mirrored_poisson: JointPoissonModel
    legacy_mirrored_dixon_coles: JointPoissonModel
    metadata: dict[str, Any]


def fit_fixed_audit_baselines(
    development: pd.DataFrame,
    *,
    development_frame_hash: str,
    feature_schema_hash: str,
) -> FrozenAuditBaselines:
    """Fit all mandatory audit baselines on the exact through-2010 frame."""
    _validate_development(development, development_frame_hash, feature_schema_hash)
    official_config = OutcomeLogitConfig()
    independent_config = _count_config(rho=0.0, dynamic=False)
    dc_config = _count_config(rho=None, dynamic=True)
    official = OutcomeLogit(official_config).fit(development)
    independent = JointPoissonModel(independent_config).fit(development)
    dc = JointPoissonModel(dc_config).fit(development)
    identities = fixed_audit_baseline_identities(
        development_frame_hash=development_frame_hash,
        feature_schema_hash=feature_schema_hash,
    )
    metadata = {
        "schema_version": 1,
        "fit_end_year": 2010,
        "training_row_count": int(len(development)),
        "development_frame_hash": development_frame_hash,
        "feature_schema_hash": feature_schema_hash,
        "audit_outcomes_used": False,
        "models": identities,
        "baseline_bundle_hash": payload_sha256(identities),
    }
    return FrozenAuditBaselines(official, independent, dc, metadata)


def fixed_audit_baseline_identities(
    *,
    development_frame_hash: str,
    feature_schema_hash: str,
) -> dict[str, dict[str, Any]]:
    """Recompute the three exact comparator identities without trusting artifacts."""
    identities = {
        "official_outcome_logit": _identity(
            "official_outcome_logit",
            OutcomeLogitConfig(),
            development_frame_hash,
            feature_schema_hash,
        ),
        "independent_mirrored_poisson": _identity(
            "independent_mirrored_poisson",
            _count_config(rho=0.0, dynamic=False),
            development_frame_hash,
            feature_schema_hash,
        ),
        "legacy_mirrored_dixon_coles": _identity(
            "legacy_mirrored_dixon_coles",
            _count_config(rho=None, dynamic=True),
            development_frame_hash,
            feature_schema_hash,
        ),
    }
    return identities


def validate_fixed_audit_baseline_models(
    official: object,
    independent: object,
    dixon_coles: object,
) -> None:
    """Reject deserialized comparators whose class, config, or fit state drifted."""
    if type(official) is not OutcomeLogit or asdict(official.config) != asdict(
        OutcomeLogitConfig()
    ):
        raise ValueError("Official audit baseline class/config changed")
    if official.scaler_ is None or official.classifier_ is None:
        raise ValueError("Official audit baseline is not fitted")
    expected = (
        (independent, _count_config(rho=0.0, dynamic=False), "independent"),
        (dixon_coles, _count_config(rho=None, dynamic=True), "Dixon-Coles"),
    )
    for model, config, label in expected:
        if type(model) is not JointPoissonModel or asdict(model.config) != asdict(config):
            raise ValueError(f"{label} audit baseline class/config changed")
        if model.pipeline_ is None or not model.feature_columns_:
            raise ValueError(f"{label} audit baseline is not fitted")


def _count_config(*, rho: float | None, dynamic: bool) -> JointPoissonConfig:
    return JointPoissonConfig(
        optional_diff_features=_DYNAMIC_FEATURES if dynamic else (),
        alpha=0.5,
        time_decay_half_life_years=12.0,
        rho=rho,
        max_score=12,
        min_retained_mass=0.999,
        max_support=100,
    )


def _identity(
    model_id: str,
    config: object,
    development_frame_hash: str,
    feature_schema_hash: str,
) -> dict[str, Any]:
    config_payload = json.loads(
        json.dumps(asdict(config), sort_keys=True)  # type: ignore[arg-type]
    )
    payload = {
        "model_id": model_id,
        "model_class": type(config).__name__.removesuffix("Config"),
        "canonical_config": config_payload,
        "config_hash": payload_sha256(config_payload),
        "fit_end_year": 2010,
        "development_frame_hash": development_frame_hash,
        "feature_schema_hash": feature_schema_hash,
        "audit_outcomes_used": False,
    }
    return {**payload, "model_identity_hash": payload_sha256(payload)}


def _validate_development(
    frame: pd.DataFrame,
    development_frame_hash: str,
    feature_schema_hash: str,
) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Baseline freeze requires a nonempty development DataFrame")
    years = pd.to_numeric(frame.get("Year"), errors="coerce")
    if years.isna().any() or int(years.max()) != 2010 or bool(years.gt(2010).any()):
        raise ValueError("Audit baselines must freeze on development data through 2010 only")
    for label, value in (
        ("development_frame_hash", development_frame_hash),
        ("feature_schema_hash", feature_schema_hash),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{label} must be a SHA-256 digest")


__all__ = [
    "FrozenAuditBaselines",
    "fit_fixed_audit_baselines",
    "fixed_audit_baseline_identities",
    "validate_fixed_audit_baseline_models",
]
