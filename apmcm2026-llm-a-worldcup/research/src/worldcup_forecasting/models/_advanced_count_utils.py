"""Private design, audit, and probability-kernel helpers for advanced counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ._selective_poisson import original_scale_coefficients
from ._utils import resolve_column
from .count import JointGoalBatch

DEFAULT_OPTIONAL_ODD = (
    "pre_match_elo_diff",
    "attack_state_diff",
    "defense_state_diff",
    "host_indicator_diff",
    "rest_days_diff",
    "player_prior_appearances_mean_diff",
    "player_prior_starts_mean_diff",
    "player_prior_goals_mean_diff",
    "player_lineup_continuity_rate_diff",
)
DEFAULT_OPTIONAL_EVEN = (
    "Year",
    "tournament_teams",
    "tournament_matches",
    "matches_per_team",
    "attendance_log",
    "rest_days_sum",
)
DEFAULT_OPTIONAL_CONTEXT = ("canonical_stage", "competition_era", "format_regime")


@dataclass(frozen=True)
class FrozenCountSchema:
    """Resolved feature names and population team support."""

    home_team: str
    away_team: str
    odd: tuple[str, ...]
    even: tuple[str, ...]
    contexts: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    common_pace: tuple[str, ...]
    stage: str | None
    year: str | None
    host_side: str | None
    team_categories: frozenset[str]


def freeze_count_schema(frame: pd.DataFrame, config: Any) -> FrozenCountSchema:
    """Resolve required/optional columns once and validate team identities."""
    home_team = resolve_column(
        frame,
        config.home_team_column,
        ("Home Team Initials", "home_team", "home_team_initials"),
    )
    away_team = resolve_column(
        frame,
        config.away_team_column,
        ("Away Team Initials", "away_team", "away_team_initials"),
    )
    odd = tuple(resolve_column(frame, name) for name in config.odd_numeric_features)
    common_pace = tuple(resolve_column(frame, name) for name in config.common_pace_features)
    even = unique(
        common_pace
        + tuple(resolve_column(frame, name) for name in config.even_numeric_features)
        + present_columns(frame, config.optional_even_numeric_features)
    )
    odd = unique(odd + present_columns(frame, config.optional_odd_numeric_features))
    contexts = unique(
        tuple(resolve_column(frame, name) for name in config.even_categorical_features)
        + present_columns(frame, config.optional_even_categorical_features)
    )
    stage = resolve_column(frame, config.stage_column) if config.stage_column is not None else None
    if stage is not None and stage not in contexts:
        contexts += (stage,)
    host = (
        resolve_column(frame, config.host_side_column)
        if config.host_side_column is not None
        else None
    )
    try:
        year = resolve_column(frame, config.year_column, ("year",))
    except ValueError:
        year = None
    pairs = tuple(
        (resolve_column(frame, home_name), resolve_column(frame, away_name))
        for home_name, away_name in config.paired_numeric_features
    )
    if not odd and not even and not pairs and host is None:
        raise ValueError("At least one numeric pace or signed-difference feature is required")
    teams = pd.concat([frame[home_team], frame[away_team]], ignore_index=True).map(team_key)
    if teams.eq("").any():
        raise ValueError("Training team identities must be nonempty")
    return FrozenCountSchema(
        home_team=home_team,
        away_team=away_team,
        odd=odd,
        even=even,
        contexts=contexts,
        pairs=pairs,
        common_pace=common_pace,
        stage=stage,
        year=year,
        host_side=host,
        team_categories=frozenset(teams),
    )


def build_perspective_frame(frame: pd.DataFrame, schema: FrozenCountSchema) -> pd.DataFrame:
    """Build home and away views with exact even/odd transformation rules."""
    home_team = frame[resolve_column(frame, schema.home_team)].map(team_key)
    away_team = frame[resolve_column(frame, schema.away_team)].map(team_key)
    if home_team.eq("").any() or away_team.eq("").any():
        raise ValueError("Prediction team identities must be nonempty")
    home_view = pd.DataFrame(index=frame.index)
    away_view = pd.DataFrame(index=frame.index)
    for name in schema.even:
        values = pd.to_numeric(frame[resolve_column(frame, name)], errors="coerce")
        home_view[f"even::{name}"] = values
        away_view[f"even::{name}"] = values
    for name in schema.odd:
        values = pd.to_numeric(frame[resolve_column(frame, name)], errors="coerce")
        home_view[f"odd::{name}"] = values
        away_view[f"odd::{name}"] = -values
    if schema.host_side is not None:
        host_side = frame[resolve_column(frame, schema.host_side)].map(host_sign)
        home_view["odd::__host_signed__"] = host_side
        away_view["odd::__host_signed__"] = -host_side
    for index, (home_name, away_name) in enumerate(schema.pairs):
        home_value = pd.to_numeric(frame[resolve_column(frame, home_name)], errors="coerce")
        away_value = pd.to_numeric(frame[resolve_column(frame, away_name)], errors="coerce")
        even = home_value + away_value
        odd = home_value - away_value
        home_view[f"pair_even::{index}"] = even
        away_view[f"pair_even::{index}"] = even
        home_view[f"pair_odd::{index}"] = odd
        away_view[f"pair_odd::{index}"] = -odd
    for name in schema.contexts:
        values = frame[resolve_column(frame, name)].astype("string").fillna("__MISSING__")
        home_view[f"context::{name}"] = values
        away_view[f"context::{name}"] = values
    home_view["attacker_team"] = home_team
    home_view["opponent_defender_team"] = away_team
    away_view["attacker_team"] = away_team
    away_view["opponent_defender_team"] = home_team
    return pd.concat([home_view, away_view], ignore_index=True)


def build_component_table(
    pipeline: Any,
    perspective: pd.DataFrame,
    frame: pd.DataFrame,
    schema: FrozenCountSchema,
    rates: np.ndarray,
) -> pd.DataFrame:
    """Account for fitted log-rate contributions by scientific component."""
    design = pipeline.named_steps["design"]
    poisson_model = pipeline.named_steps["poisson"]
    transformed = np.asarray(design.transform(perspective), dtype=float)
    names = np.asarray(design.get_feature_names_out(), dtype=str)
    contribution = transformed * original_scale_coefficients(pipeline)[None, :]
    count = len(frame)

    def total(row: np.ndarray, fragments: tuple[str, ...]) -> np.ndarray:
        mask = np.asarray([any(fragment in name for fragment in fragments) for name in names])
        return row[:, mask].sum(axis=1) if mask.any() else np.zeros(row.shape[0])

    home = contribution[:count]
    away = contribution[count:]
    pace = tuple(f"even::{name}" for name in schema.common_pace)
    year = (f"even::{schema.year}",) if schema.year is not None else ()
    stage = (f"context::{schema.stage}",) if schema.stage is not None else ()
    home_team = frame[resolve_column(frame, schema.home_team)].map(team_key)
    away_team = frame[resolve_column(frame, schema.away_team)].map(team_key)
    return pd.DataFrame(
        {
            "common_pace": total(home, pace),
            "era_effect": total(home, year),
            "stage_effect": total(home, stage),
            "host_effect": total(home, ("odd::__host_signed__",)),
            "home_attack_effect": total(home, ("attacker_team_",)),
            "away_defence_effect": total(home, ("opponent_defender_team_",)),
            "away_attack_effect": total(away, ("attacker_team_",)),
            "home_defence_effect": total(away, ("opponent_defender_team_",)),
            "home_unknown_team": ~home_team.isin(schema.team_categories),
            "away_unknown_team": ~away_team.isin(schema.team_categories),
            "home_rate": rates[:, 0],
            "away_rate": rates[:, 1],
            "intercept": float(poisson_model.intercept_),
        },
        index=frame.index,
    )


def build_team_effect_table(
    pipeline: Any,
    frame: pd.DataFrame,
    schema: FrozenCountSchema,
    shrinkage: float,
) -> pd.DataFrame:
    """Extract identity coefficients and empirical shrinkage diagnostics."""
    design = pipeline.named_steps["design"]
    names = np.asarray(design.get_feature_names_out(), dtype=str)
    coefficient = original_scale_coefficients(pipeline)
    attack: dict[str, float] = {}
    defence: dict[str, float] = {}
    for name, value in zip(names, coefficient, strict=True):
        if "attacker_team_" in name:
            attack[name.split("attacker_team_", maxsplit=1)[1]] = float(value)
        elif "opponent_defender_team_" in name:
            defence[name.split("opponent_defender_team_", maxsplit=1)[1]] = float(value)
    home = frame[schema.home_team].map(team_key)
    away = frame[schema.away_team].map(team_key)
    sizes = pd.concat([home, away], ignore_index=True).value_counts()
    records = [
        {
            "team": team,
            "attack": attack.get(team, 0.0),
            "defence": defence.get(team, 0.0),
            "sample_size": int(sizes.get(team, 0)),
            "shrinkage_weight": float(sizes.get(team, 0) / (sizes.get(team, 0) + shrinkage)),
        }
        for team in sorted(schema.team_categories)
    ]
    return pd.DataFrame.from_records(records)


def unknown_team_rows(frame: pd.DataFrame, schema: FrozenCountSchema) -> np.ndarray:
    """Mark rows containing identities outside fitted categorical support."""
    home = frame[resolve_column(frame, schema.home_team)].map(team_key)
    away = frame[resolve_column(frame, schema.away_team)].map(team_key)
    return (~home.isin(schema.team_categories) | ~away.isin(schema.team_categories)).to_numpy()


def adaptive_bivariate_poisson(
    *,
    marginal_home: float,
    marginal_away: float,
    shared_rate: float,
    initial_support: int,
    minimum_mass: float,
    maximum_support: int,
) -> tuple[np.ndarray, float]:
    """Return a normalized bivariate-Poisson grid with adaptive truncation."""
    rates = np.asarray([marginal_home, marginal_away, shared_rate], dtype=float)
    if not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError("Bivariate Poisson rates must be finite and nonnegative")
    if shared_rate > min(marginal_home, marginal_away):
        raise ValueError("Shared-score rate cannot exceed either marginal rate")
    independent_home = marginal_home - shared_rate
    independent_away = marginal_away - shared_rate
    support = initial_support
    while True:
        score = np.arange(support + 1)
        home_pmf = poisson.pmf(score, independent_home)
        away_pmf = poisson.pmf(score, independent_away)
        shared_pmf = poisson.pmf(score, shared_rate)
        matrix = np.zeros((support + 1, support + 1), dtype=float)
        for shared_goals, shared_probability in enumerate(shared_pmf):
            width = support - shared_goals + 1
            matrix[shared_goals:, shared_goals:] += shared_probability * np.outer(
                home_pmf[:width], away_pmf[:width]
            )
        retained = float(matrix.sum())
        if retained <= 0 or not np.isfinite(retained):
            raise RuntimeError("Bivariate Poisson score grid retained no finite mass")
        if retained >= minimum_mass:
            return matrix / retained, retained
        if support >= maximum_support:
            raise RuntimeError(
                f"Score support {maximum_support} retained only {retained:.8f} mass, "
                f"below required {minimum_mass:.8f}"
            )
        support = min(maximum_support, max(support + 1, int(np.ceil(1.25 * support))))


def joint_batch_from_rates_impl(
    rates: np.ndarray,
    *,
    shared_fraction: float,
    max_score: int,
    min_retained_mass: float,
    max_support: int,
) -> JointGoalBatch:
    """Convert paired marginal rates into a coherent joint score batch."""
    values = np.asarray(rates, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 2:
        raise ValueError("rates must have shape (n_matches, 2)")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("rates must be finite and strictly positive")
    if not np.isfinite(shared_fraction) or not 0 <= shared_fraction < 1:
        raise ValueError("shared_fraction must lie in [0, 1)")
    matrices: list[np.ndarray] = []
    retained: list[float] = []
    shared_rates: list[float] = []
    for home_rate, away_rate in values:
        shared = shared_fraction * min(float(home_rate), float(away_rate))
        matrix, mass = adaptive_bivariate_poisson(
            marginal_home=float(home_rate),
            marginal_away=float(away_rate),
            shared_rate=shared,
            initial_support=max_score,
            minimum_mass=min_retained_mass,
            maximum_support=max_support,
        )
        matrices.append(matrix)
        retained.append(mass)
        shared_rates.append(shared)
    support_max = max(matrix.shape[0] for matrix in matrices) - 1
    probability = np.zeros((len(matrices), support_max + 1, support_max + 1), dtype=float)
    for row, matrix in enumerate(matrices):
        width = matrix.shape[0]
        probability[row, :width, :width] = matrix
    support = np.arange(support_max + 1, dtype=float)
    return JointGoalBatch(
        probability=probability,
        support_max=support_max,
        retained_mass=np.asarray(retained, dtype=float),
        expected_home_goals=np.sum(probability * support[None, :, None], axis=(1, 2)),
        expected_away_goals=np.sum(probability * support[None, None, :], axis=(1, 2)),
        model_metadata={
            "name": "joint_score_kernel_from_rates",
            "shared_score_fraction": float(shared_fraction),
            "shared_score_rate_mean": float(np.mean(shared_rates)),
        },
    )


def present_columns(frame: pd.DataFrame, candidates: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve optional unambiguous candidates in declared order."""
    present: list[str] = []
    for name in candidates:
        try:
            resolved = resolve_column(frame, name)
        except ValueError:
            continue
        if resolved not in present:
            present.append(resolved)
    return tuple(present)


def unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def team_key(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip().upper()


def host_sign(value: object) -> float:
    text = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    mapping = {
        "home": 1.0,
        "home_team": 1.0,
        "away": -1.0,
        "away_team": -1.0,
        "neutral": 0.0,
        "none": 0.0,
        "0": 0.0,
        "1": 1.0,
        "_1": -1.0,
    }
    if text not in mapping:
        raise ValueError(f"Unknown host-side value: {value!r}")
    return mapping[text]
