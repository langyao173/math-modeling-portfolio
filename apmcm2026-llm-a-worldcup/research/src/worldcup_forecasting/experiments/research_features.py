"""Research-v2 feature assembly and frozen candidate configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import pandas as pd

from worldcup_forecasting.data import CanonicalDataBundle
from worldcup_forecasting.features.context import build_match_context_features
from worldcup_forecasting.features.dynamic import build_dynamic_features
from worldcup_forecasting.features.information_sets import (
    InformationSetBundle,
    build_player_coach_information_sets,
)
from worldcup_forecasting.features.team_history import build_team_match_features
from worldcup_forecasting.models.advanced_count import EquivariantCountConfig


@dataclass(frozen=True)
class ResearchFeatureBundle:
    """Merged pre-match table plus separate kickoff features and lineage."""

    pre_match: pd.DataFrame
    kickoff: pd.DataFrame
    lineage: pd.DataFrame
    schema_hash: str


_DYNAMIC_ODD = ("pre_match_elo_diff", "attack_state_diff", "defense_state_diff")
_CONTEXT_ODD = ("rest_days_diff",)
_COMMON_PACE = (
    "prior_stage_goal_rate",
    "prior_overall_goal_rate",
    "tournament_progress",
)
_CONTEXT_EVEN = (
    "Year",
    "tournament_teams",
    "tournament_matches",
    "matches_per_team",
    "is_knockout",
    "rest_history_missing",
)
_CONTEXT_CATEGORICAL = ("canonical_stage", "competition_era", "format_regime")
_TEAM_PAIRS = (
    ("home_historical_appearances", "away_historical_appearances"),
    ("home_historical_best_finish", "away_historical_best_finish"),
    ("home_historical_mean_goals_for", "away_historical_mean_goals_for"),
    ("home_historical_mean_goals_conceded", "away_historical_mean_goals_conceded"),
)
_DYNAMIC_PAIRS = (
    ("home_pre_match_elo", "away_pre_match_elo"),
    ("home_attack_state", "away_attack_state"),
    ("home_defense_state", "away_defense_state"),
)
_PERSONNEL_PAIRS = (
    ("home_player_prior_appearances", "away_player_prior_appearances"),
    ("home_player_prior_goals", "away_player_prior_goals"),
    ("home_player_prior_cards", "away_player_prior_cards"),
    ("home_coach_prior_team_matches", "away_coach_prior_team_matches"),
)
_REST_PAIRS = (("home_rest_days", "away_rest_days"),)


def _merge_new(base: pd.DataFrame, addition: pd.DataFrame) -> pd.DataFrame:
    columns = ["MatchID", *[name for name in addition if name != "MatchID" and name not in base]]
    return base.merge(addition[columns], on="MatchID", validate="one_to_one")


def build_research_feature_bundle(
    bundle: CanonicalDataBundle,
    *,
    state_update_before_year: int | None = None,
) -> ResearchFeatureBundle:
    """Assemble every feature using the prediction-before-update discipline."""
    base = build_team_match_features(bundle.matches, bundle.tournaments)
    dynamic = build_dynamic_features(
        bundle.matches,
        state_update_before_year=state_update_before_year,
    )
    context = build_match_context_features(bundle.matches, bundle.tournaments)
    personnel: InformationSetBundle = build_player_coach_information_sets(
        players=bundle.players,
        matches=bundle.matches,
        state_update_before_year=state_update_before_year,
    )
    pre_match = base
    for addition in (dynamic, context, personnel.pre_match):
        pre_match = _merge_new(pre_match, addition)
    kickoff = _merge_new(pre_match, personnel.kickoff)
    if not bool((pre_match["history_cutoff_year"] < pre_match["Year"]).all()):
        raise RuntimeError("Static team history is not strictly point-in-time")
    for side in ("home", "away"):
        cutoff = pd.to_datetime(pre_match[f"{side}_dynamic_cutoff_time"], errors="coerce")
        prediction = pd.to_datetime(pre_match["Datetime"], errors="raise")
        valid = cutoff.isna() | cutoff.lt(prediction)
        if not bool(valid.all()):
            raise RuntimeError(f"{side} dynamic state includes current/future matches")
    schema = {
        "columns": sorted(pre_match.columns),
        "dtypes": {name: str(pre_match[name].dtype) for name in sorted(pre_match.columns)},
        "information_sets": ["pre_match", "kickoff"],
        "cutoff": "strictly_before_prediction_except_declared_current_lineup_at_kickoff",
    }
    schema_hash = hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ResearchFeatureBundle(
        pre_match=pre_match.sort_values(["Year", "Datetime", "MatchID"]).reset_index(drop=True),
        kickoff=kickoff.sort_values(["Year", "Datetime", "MatchID"]).reset_index(drop=True),
        lineage=personnel.lineage.copy(),
        schema_hash=schema_hash,
    )


def make_count_config(
    *,
    alpha: float,
    half_life_years: float | None,
    shared_score: bool,
    include_dynamic: bool = True,
    include_personnel: bool = True,
    include_era_format: bool = True,
    enforce_shrinkage: bool = True,
) -> EquivariantCountConfig:
    """Create one member of the frozen structural candidate/ablation family."""
    odd = [*_CONTEXT_ODD]
    pairs = [*_TEAM_PAIRS, *_REST_PAIRS]
    if include_dynamic:
        odd.extend(_DYNAMIC_ODD)
        pairs.extend(_DYNAMIC_PAIRS)
    if include_personnel:
        pairs.extend(_PERSONNEL_PAIRS)
    even = ["is_knockout", "rest_history_missing"]
    categories = ["canonical_stage"]
    if include_era_format:
        even.extend(name for name in _CONTEXT_EVEN if name not in even)
        categories.extend(name for name in _CONTEXT_CATEGORICAL if name not in categories)
    config = EquivariantCountConfig(
        home_team_column="home_team_initials",
        away_team_column="away_team_initials",
        home_goals_column="target_home_goals",
        away_goals_column="target_away_goals",
        year_column="Year",
        stage_column="canonical_stage",
        host_side_column="host_side",
        odd_numeric_features=tuple(odd),
        common_pace_features=_COMMON_PACE,
        even_numeric_features=tuple(even),
        even_categorical_features=tuple(categories),
        optional_odd_numeric_features=(),
        optional_even_numeric_features=(),
        optional_even_categorical_features=(),
        paired_numeric_features=tuple(pairs),
        alpha=float(alpha),
        team_shrinkage=float(alpha if enforce_shrinkage else 0.0),
        time_decay_half_life_years=half_life_years,
        shared_score=bool(shared_score),
        max_iter=2_000,
        tolerance=1e-8,
        max_score=12,
        min_retained_mass=0.999,
        max_support=60,
    )
    return config


def config_hash(config: EquivariantCountConfig) -> str:
    """Return a stable identity for model/config provenance."""
    return hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ResearchFeatureBundle",
    "build_research_feature_bundle",
    "config_hash",
    "make_count_config",
]
