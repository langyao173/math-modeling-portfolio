"""Input and identity contracts for the measured ablation engine."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import pandas as pd


class _Config(Protocol):
    seeds: tuple[int, ...]
    outer_test_years: tuple[int, ...]
    maximum_development_year: int
    latest_inner_folds: int
    year_column: str


def validate_and_order(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    cfg: _Config,
    schema_hash: str | None,
    clock: Callable[[], float],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Fail closed on selection leakage, duplicates, or unstable identities."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Ablation evaluation requires a nonempty feature frame")
    if not isinstance(selected, pd.DataFrame) or selected.empty:
        raise ValueError("Ablation evaluation requires frozen selected specifications")
    if not callable(clock):
        raise TypeError("runtime_clock must be callable")
    if cfg.seeds != (17, 29):
        raise ValueError("The frozen ablation seeds must be exactly (17, 29)")
    invalid_development_year = (
        isinstance(cfg.maximum_development_year, (bool, np.bool_))
        or not isinstance(cfg.maximum_development_year, (int, np.integer))
        or cfg.maximum_development_year <= 0
        or cfg.maximum_development_year > 2010
    )
    if invalid_development_year:
        raise ValueError("The frozen ablation development boundary cannot exceed 2010")
    if not cfg.outer_test_years or len(set(cfg.outer_test_years)) != len(cfg.outer_test_years):
        raise ValueError("outer_test_years must be nonempty and unique")
    if max(cfg.outer_test_years) > cfg.maximum_development_year:
        raise ValueError("2014/audit years are forbidden in the A0-A8 development experiment")
    if cfg.latest_inner_folds < 1:
        raise ValueError("latest_inner_folds must be positive")
    required = {
        "outer_fold_id",
        "outer_test_year",
        "candidate_id",
        "alpha",
        "half_life_years",
        "shared_score",
        "residual",
    }
    missing = sorted(required.difference(selected.columns))
    if missing:
        raise ValueError(f"selected specifications are missing columns: {missing}")
    selected = selected.loc[selected["outer_test_year"].isin(cfg.outer_test_years)].copy()
    counts = selected.groupby("outer_test_year").size()
    if set(counts.index.astype(int)) != set(cfg.outer_test_years) or not counts.eq(1).all():
        raise ValueError("selected specifications must contain exactly one row per outer year")
    if selected["candidate_id"].astype(str).str.strip().eq("").any():
        raise ValueError("selected candidate_id must be nonempty")
    if cfg.year_column not in frame:
        raise ValueError(f"Ablation frame is missing {cfg.year_column!r}")
    numeric_years = pd.to_numeric(frame[cfg.year_column], errors="raise").to_numpy(float)
    valid_years = np.isfinite(numeric_years).all() and np.allclose(
        numeric_years, np.rint(numeric_years)
    )
    if not valid_years:
        raise ValueError("Ablation frame years must be finite integers")
    years = pd.Series(numeric_years.astype(int), index=frame.index)
    if int(years.max()) > cfg.maximum_development_year:
        raise ValueError(
            "Ablation frame cannot load audit or post-development-year rows"
        )
    for year in cfg.outer_test_years:
        if not years.eq(year).any() or not years.lt(year).any():
            raise ValueError(f"Outer fold {year} lacks training or evaluation rows")
    id_column = "MatchID" if "MatchID" in frame else "match_id"
    if id_column not in frame:
        raise ValueError("Ablation frame requires MatchID or match_id")
    identifiers = frame[id_column]
    if identifiers.isna().any() or identifiers.duplicated().any():
        raise ValueError("Ablation match identifiers must be complete and unique")
    sort_columns = [cfg.year_column]
    if "Datetime" in frame:
        sort_columns.append("Datetime")
    sort_columns.append(id_column)
    ordered = frame.assign(**{cfg.year_column: years}).sort_values(
        sort_columns, kind="stable"
    ).reset_index(drop=True)
    resolved_hash = schema_hash or schema_identity(ordered)
    invalid = len(resolved_hash) != 64 or any(
        char not in "0123456789abcdef" for char in resolved_hash
    )
    if invalid:
        raise ValueError("feature_schema_hash must be a lowercase SHA-256 digest")
    return ordered, selected.sort_values("outer_test_year").reset_index(drop=True), resolved_hash


def evaluation_hash(year: int, ids: np.ndarray) -> str:
    return _hash({"test_year": year, "ordered_match_ids": [str(value) for value in ids]})


def schema_identity(frame: pd.DataFrame) -> str:
    return _hash({"columns": list(frame.columns), "dtypes": frame.dtypes.astype(str).to_dict()})


def prediction_times(frame: pd.DataFrame, year: int) -> list[str]:
    if "Datetime" in frame:
        parsed = pd.to_datetime(frame["Datetime"], errors="raise")
    else:
        parsed = pd.Series(pd.Timestamp(year=year, month=1, day=1), index=frame.index)
    return [value.isoformat() for value in parsed]


def half_life(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def as_bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"selected {label} must be boolean")


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "as_bool",
    "evaluation_hash",
    "half_life",
    "prediction_times",
    "validate_and_order",
]
