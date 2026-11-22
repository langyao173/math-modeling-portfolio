"""Exact tables required by the official problem statement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def descriptive_statistics(tournaments: pd.DataFrame) -> pd.DataFrame:
    """Return the five requested sample statistics for four indicators."""
    indicators = {
        "total_goals": "GoalsScored",
        "goals_per_match": "goals_per_match",
        "qualified_teams": "QualifiedTeams",
        "matches_played": "MatchesPlayed",
    }
    rows: list[dict[str, float | str]] = []
    for indicator, column in indicators.items():
        values = tournaments[column].astype(float)
        rows.append(
            {
                "indicator": indicator,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "sample_std": float(values.std(ddof=1)),
                "n": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def stage_statistics(matches: pd.DataFrame) -> pd.DataFrame:
    """Return canonical-stage counts, total goals, and means."""
    order = [
        "group_stage",
        "round_of_16",
        "quarter_final",
        "semi_final",
        "final",
        "third_place",
    ]
    grouped = (
        matches.groupby("canonical_stage", observed=True)["full_time_total_goals"]
        .agg(matches="size", goal_sum="sum", mean_goals="mean")
        .reindex(order)
        .reset_index()
    )
    if grouped["matches"].sum() != matches.shape[0]:
        raise AssertionError("Canonical stage table does not cover every match")
    return grouped


def outcome_feature_correlations(features: pd.DataFrame) -> pd.DataFrame:
    """Correlate mandatory pairwise features with one-hot outcomes."""
    diff_columns = [
        "historical_appearances_diff",
        "historical_best_finish_diff",
        "historical_mean_goals_for_diff",
        "historical_mean_goals_conceded_diff",
    ]
    missing = [column for column in diff_columns if column not in features]
    if missing:
        raise ValueError(f"Missing mandatory historical features: {missing}")
    outcome = pd.get_dummies(features["scoreline_outcome"], prefix="outcome", dtype=float)
    combined = pd.concat([features[diff_columns].astype(float), outcome], axis=1)
    outcome_columns = [column for column in combined if column.startswith("outcome_")]
    return combined.corr().loc[diff_columns, outcome_columns]


def write_json(payload: dict[str, Any], path: Path) -> Path:
    """Write stable, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    return path


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")
