"""Construct a bounded 48-team format scenario from pre-audit team profiles."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .format_policy import (
    FIFA_2026_ANNEX_C_CSV_SHA256,
    FIFA_2026_ANNEX_C_POLICY,
    FIFA_2026_REGULATIONS_SHA256,
    FIFA_2026_REGULATIONS_URL,
)

ScoreProvider = Callable[[str, str, str], np.ndarray]


@dataclass(frozen=True)
class FormatScenarioProvider:
    """Illustrative team list, score callback, profiles, and provenance."""

    teams: tuple[str, ...]
    score_provider: ScoreProvider
    profile_table: pd.DataFrame
    metadata: dict[str, Any]


def build_format_scenario_provider(
    fitted_model: Any,
    historical_frame: pd.DataFrame,
    *,
    teams: Sequence[str] | None = None,
    profile_cutoff_year: int = 2010,
    precompute_stages: Sequence[str] | None = None,
) -> FormatScenarioProvider:
    """Build a model-backed 2026-format scenario without using post-cutoff rows.

    Team names are historical stand-ins, not asserted 2026 entrants. The scenario
    changes only known format/stage fields and reconstructs paired team features
    from each team's last lawful pre-cutoff record.
    """
    frame, home_team, away_team, year_column = _validate_inputs(
        fitted_model, historical_frame, profile_cutoff_year
    )
    selected = _select_teams(frame, home_team, away_team, teams)
    pairs = tuple(getattr(fitted_model, "paired_columns_", ()))
    profiles, profile_table, fallback_count = _team_profiles(
        frame,
        selected,
        pairs,
        home_team=home_team,
        away_team=away_team,
        year_column=year_column,
    )
    template = frame.sort_values([year_column, *_optional_sort_columns(frame)]).iloc[-1].copy()
    odd_pair_index = _odd_pair_mapping(tuple(getattr(fitted_model, "odd_columns_", ())), pairs)
    score_cache: dict[tuple[str, str, str], np.ndarray] = {}

    def scenario_row(home: str, away: str, stage: str) -> pd.Series:
        row = template.copy()
        row[home_team] = home
        row[away_team] = away
        row[year_column] = 2026
        _apply_format_context(row, fitted_model, stage)
        for position, (home_column, away_column) in enumerate(pairs):
            row[home_column] = profiles[home][position]
            row[away_column] = profiles[away][position]
        for odd_column in getattr(fitted_model, "odd_columns_", ()):
            pair_position = odd_pair_index.get(str(odd_column))
            row[odd_column] = (
                profiles[home][pair_position] - profiles[away][pair_position]
                if pair_position is not None
                else 0.0
            )
        return row

    def validated_grid(probability: np.ndarray) -> np.ndarray:
        probability = np.asarray(probability, dtype=float)
        if (
            probability.ndim != 2
            or probability.shape[0] != probability.shape[1]
            or not np.isfinite(probability).all()
            or np.any(probability < 0)
            or not np.isclose(probability.sum(), 1.0, atol=1e-10)
        ):
            raise RuntimeError("Scenario model returned an invalid joint score distribution")
        probability.setflags(write=False)
        return probability

    def score_provider(home: str, away: str, stage: str) -> np.ndarray:
        if home not in profiles or away not in profiles or home == away:
            raise ValueError("Scenario score request must use two distinct declared teams")
        cache_key = (home, away, str(stage))
        cached = score_cache.get(cache_key)
        if cached is not None:
            return cached
        batch = fitted_model.predict_joint(pd.DataFrame([scenario_row(*cache_key)]))
        probability = validated_grid(batch.probability[0])
        score_cache[cache_key] = probability
        return probability

    stages = _validated_precompute_stages(precompute_stages)
    precomputed_requests = 0
    if stages:
        keys: list[tuple[str, str, str]] = []
        rows: list[pd.Series] = []
        for stage in stages:
            for home_position, home in enumerate(selected):
                for away in selected[home_position + 1 :]:
                    key = (home, away, stage)
                    keys.append(key)
                    rows.append(scenario_row(*key))
        batch = fitted_model.predict_joint(pd.DataFrame(rows).reset_index(drop=True))
        if len(batch.probability) != len(keys):
            raise RuntimeError("Scenario batch precomputation returned the wrong row count")
        for key, matrix in zip(keys, batch.probability, strict=True):
            probability = validated_grid(matrix)
            score_cache[key] = probability
            score_cache[(key[1], key[0], key[2])] = validated_grid(probability.T)
        precomputed_requests = len(score_cache)

    metadata = {
        "scenario_status": "illustrative_historical_team_stand_ins_not_real_2026_entrants",
        "profile_cutoff_year": int(profile_cutoff_year),
        "source_rows_through_cutoff": int(len(frame)),
        "team_count": len(selected),
        "tournament_teams": 48,
        "tournament_matches": 104,
        "format_regime": "expanded_48_team",
        "bracket_policy": FIFA_2026_ANNEX_C_POLICY,
        "format_source_url": FIFA_2026_REGULATIONS_URL,
        "format_source_sha256": FIFA_2026_REGULATIONS_SHA256,
        "annex_c_csv_sha256": FIFA_2026_ANNEX_C_CSV_SHA256,
        "profile_feature_count": len(pairs),
        "population_fallback_value_count": int(fallback_count),
        "unknown_odd_feature_policy": "symmetry_neutral_zero",
        "score_grid_cache": "deterministic_home_away_stage_memoization",
        "precomputed_stages": list(stages),
        "precomputed_ordered_score_grids": precomputed_requests,
        "post_cutoff_rows_used": False,
        "claim_scope": "format_and_transport_scenario_only_not_factual_forecast",
    }
    return FormatScenarioProvider(tuple(selected), score_provider, profile_table, metadata)


def _validate_inputs(
    model: Any,
    historical_frame: pd.DataFrame,
    profile_cutoff_year: int,
) -> tuple[pd.DataFrame, str, str, str]:
    if not isinstance(historical_frame, pd.DataFrame) or historical_frame.empty:
        raise ValueError("Historical scenario features must be a nonempty DataFrame")
    if not callable(getattr(model, "predict_joint", None)):
        raise TypeError("fitted_model must expose predict_joint")
    home = getattr(model, "home_team_column_", None)
    away = getattr(model, "away_team_column_", None)
    year = getattr(model, "year_column_", None) or getattr(model.config, "year_column", None)
    if not home or not away or not year:
        raise ValueError("Fitted model must expose frozen home/away/year columns")
    missing = [column for column in (home, away, year) if column not in historical_frame]
    if missing:
        raise ValueError(f"Historical scenario frame is missing columns: {missing}")
    if isinstance(profile_cutoff_year, bool) or int(profile_cutoff_year) > 2010:
        raise ValueError("Scenario profile cutoff cannot exceed the frozen 2010 boundary")
    years = pd.to_numeric(historical_frame[year], errors="coerce")
    if years.isna().any():
        raise ValueError("Historical scenario years must be finite")
    filtered = historical_frame.loc[years.le(int(profile_cutoff_year))].copy()
    if filtered.empty:
        raise ValueError("No lawful historical profile rows exist through the cutoff")
    return filtered, str(home), str(away), str(year)


def _select_teams(
    frame: pd.DataFrame,
    home: str,
    away: str,
    teams: Sequence[str] | None,
) -> list[str]:
    appearances = pd.concat([frame[home], frame[away]], ignore_index=True).astype(str)
    available = set(appearances)
    if teams is None:
        counts = appearances.value_counts().rename_axis("team").reset_index(name="appearances")
        counts = counts.sort_values(["appearances", "team"], ascending=[False, True])
        selected = counts["team"].head(48).tolist()
    else:
        if isinstance(teams, (str, bytes)):
            raise TypeError("teams must be an ordered sequence")
        selected = [str(team).strip() for team in teams]
    if len(selected) != 48 or len(set(selected)) != 48 or any(not team for team in selected):
        raise ValueError("The illustrative 2026 format scenario requires 48 unique teams")
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"Scenario team profiles are unavailable through 2010: {unknown}")
    return selected


def _team_profiles(
    frame: pd.DataFrame,
    teams: list[str],
    pairs: tuple[tuple[str, str], ...],
    *,
    home_team: str,
    away_team: str,
    year_column: str,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, int]:
    missing = sorted({column for pair in pairs for column in pair if column not in frame})
    if missing:
        raise ValueError(f"Historical scenario frame lacks paired model features: {missing}")
    fallback = []
    for home_column, away_column in pairs:
        values = pd.concat(
            [
                pd.to_numeric(frame[home_column], errors="coerce"),
                pd.to_numeric(frame[away_column], errors="coerce"),
            ],
            ignore_index=True,
        ).dropna()
        fallback.append(float(values.median()) if not values.empty else 0.0)
    profiles: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    fallback_count = 0
    sort_columns = [year_column, *_optional_sort_columns(frame)]
    for team in teams:
        involved = frame[home_team].astype(str).eq(team) | frame[away_team].astype(str).eq(
            team
        )
        appearances = frame.loc[involved]
        appearances = appearances.sort_values(sort_columns)
        values: list[float] = []
        for position, (home_column, away_column) in enumerate(pairs):
            candidates: list[tuple[pd.Timestamp, float]] = []
            for row in appearances.itertuples(index=False):
                row_dict = row._asdict()
                is_home = str(row_dict[home_team]) == team
                column = home_column if is_home else away_column
                value = pd.to_numeric(pd.Series([row_dict[column]]), errors="coerce").iloc[0]
                if pd.notna(value):
                    timestamp = _row_timestamp(row_dict, year_column)
                    candidates.append((timestamp, float(value)))
            if candidates:
                values.append(max(candidates, key=lambda item: item[0])[1])
            else:
                values.append(fallback[position])
                fallback_count += 1
        profiles[team] = np.asarray(values, dtype=float)
        record: dict[str, Any] = {
            "team": team,
            "profile_max_source_year": int(appearances[year_column].max()),
            "historical_match_rows": int(len(appearances)),
        }
        for position, value in enumerate(values):
            record[f"paired_profile_{position}"] = value
        records.append(record)
    return profiles, pd.DataFrame(records), fallback_count


def _odd_pair_mapping(
    odd_columns: tuple[str, ...],
    pairs: tuple[tuple[str, str], ...],
) -> dict[str, int | None]:
    mapping: dict[str, int | None] = {}
    for odd in odd_columns:
        stem = odd[: -len("_diff")] if odd.endswith("_diff") else odd
        expected = (f"home_{stem}", f"away_{stem}")
        mapping[odd] = pairs.index(expected) if expected in pairs else None
    return mapping


def _apply_format_context(row: pd.Series, model: Any, stage: str) -> None:
    replacements: dict[str, Any] = {
        "Year": 2026,
        "year": 2026,
        "tournament_teams": 48.0,
        "tournament_matches": 104.0,
        "matches_per_team": 5.0,
        "format_regime": "expanded_48_team",
        "competition_era": "expanded_48_team",
        "canonical_stage": str(stage),
        "is_knockout": float(stage != "group_stage"),
        "tournament_progress": 0.0 if stage == "group_stage" else 0.75,
        "host_side": "neutral",
    }
    year_column = getattr(model, "year_column_", None)
    stage_column = getattr(model, "stage_column_", None)
    host_column = getattr(model, "host_side_column_", None)
    if year_column:
        replacements[str(year_column)] = 2026
    if stage_column:
        replacements[str(stage_column)] = str(stage)
    if host_column:
        replacements[str(host_column)] = "neutral"
    for column, value in replacements.items():
        if column in row.index:
            row[column] = value


def _optional_sort_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ("Datetime", "MatchID", "match_id") if column in frame]


def _validated_precompute_stages(stages: Sequence[str] | None) -> tuple[str, ...]:
    if stages is None:
        return ()
    if isinstance(stages, (str, bytes)):
        raise TypeError("precompute_stages must be a sequence of stage labels")
    normalized = tuple(str(stage).strip() for stage in stages)
    if any(not stage for stage in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError("precompute_stages must contain unique nonempty labels")
    return normalized


def _row_timestamp(row: dict[str, Any], year_column: str) -> pd.Timestamp:
    if "Datetime" in row and pd.notna(row["Datetime"]):
        return pd.Timestamp(row["Datetime"])
    return pd.Timestamp(year=int(row[year_column]), month=1, day=1)


__all__ = ["FormatScenarioProvider", "build_format_scenario_provider"]
