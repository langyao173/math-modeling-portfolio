"""Leakage-safe nested validation and preregistered candidate accounting.

The objects in this module deliberately store indices as tuples rather than
NumPy arrays.  A frozen dataclass containing an array is only superficially
immutable because callers can still mutate the array in place.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class InnerTournamentFold:
    """One immutable expanding-origin split inside an outer training window."""

    fold_id: str
    validation_year: int
    train_index: tuple[int, ...]
    validation_index: tuple[int, ...]

    @property
    def test_year(self) -> int:
        """Compatibility alias for validation terminology."""
        return self.validation_year

    @property
    def test_index(self) -> tuple[int, ...]:
        """Compatibility alias for validation terminology."""
        return self.validation_index


@dataclass(frozen=True, slots=True)
class NestedTournamentFold:
    """One immutable outer split and all eligible earlier inner splits."""

    fold_id: str
    validation_year: int
    train_index: tuple[int, ...]
    validation_index: tuple[int, ...]
    inner_folds: tuple[InnerTournamentFold, ...]

    @property
    def test_year(self) -> int:
        """Compatibility alias for validation terminology."""
        return self.validation_year

    @property
    def test_index(self) -> tuple[int, ...]:
        """Compatibility alias for validation terminology."""
        return self.validation_index


def _integer_years(frame: pd.DataFrame, year_column: str) -> np.ndarray:
    if year_column not in frame.columns:
        raise ValueError(f"Missing year column: {year_column}")
    if frame.empty:
        raise ValueError("Nested validation requires a nonempty frame")
    try:
        numeric = pd.to_numeric(frame[year_column], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Year column {year_column!r} must be numeric") from exc
    if not np.isfinite(numeric).all():
        raise ValueError(f"Year column {year_column!r} must contain finite values")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"Year column {year_column!r} must contain integer years")
    return numeric.astype(np.int64)


def _one_year(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must contain integer years, not booleans")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain integer years") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} must contain finite integer years")
    return int(numeric)


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(numeric)


def nested_expanding_origin_folds(
    frame: pd.DataFrame,
    outer_test_years: Iterable[int],
    *,
    year_column: str = "Year",
    minimum_inner_folds: int = 2,
) -> tuple[NestedTournamentFold, ...]:
    """Build nested expanding-origin folds without future-edition leakage.

    Every edition before an outer validation edition is included in its outer
    training set.  Every eligible edition inside that training set (apart from
    the first, which has no earlier observations) becomes an inner validation
    fold.  This maximizes scarce-edition evidence while preserving chronology.
    """
    minimum_inner_folds = _positive_integer(minimum_inner_folds, label="minimum_inner_folds")
    years = _integer_years(frame, year_column)
    requested = tuple(_one_year(year, label="outer_test_years") for year in outer_test_years)
    if not requested:
        raise ValueError("At least one outer validation year is required")
    if len(set(requested)) != len(requested):
        raise ValueError("Outer validation years must be unique")

    folds: list[NestedTournamentFold] = []
    for outer_year in sorted(requested):
        outer_train = np.flatnonzero(years < outer_year)
        outer_validation = np.flatnonzero(years == outer_year)
        if outer_train.size == 0 or outer_validation.size == 0:
            raise ValueError(f"Invalid outer expanding-origin fold for {outer_year}")
        if int(years[outer_train].max()) >= int(years[outer_validation].min()):
            raise AssertionError("Outer training data are not strictly earlier than validation")

        prior_editions = tuple(sorted(int(year) for year in np.unique(years[outer_train])))
        inner_years = prior_editions[1:]
        if len(inner_years) < minimum_inner_folds:
            raise ValueError(
                f"Outer fold {outer_year} provides {len(inner_years)} inner folds; "
                f"at least {minimum_inner_folds} are required"
            )

        inner_folds: list[InnerTournamentFold] = []
        for inner_year in inner_years:
            inner_train = np.flatnonzero(years < inner_year)
            inner_validation = np.flatnonzero(years == inner_year)
            if inner_train.size == 0 or inner_validation.size == 0:
                raise AssertionError(f"Invalid inner expanding-origin fold for {inner_year}")
            if int(years[inner_train].max()) >= int(years[inner_validation].min()):
                raise AssertionError("Inner training data are not strictly earlier than validation")
            if int(years[inner_validation].max()) >= outer_year:
                raise AssertionError("Inner validation data must predate outer validation")
            inner_folds.append(
                InnerTournamentFold(
                    fold_id=f"outer_{outer_year}__inner_{inner_year}",
                    validation_year=inner_year,
                    train_index=tuple(int(index) for index in inner_train),
                    validation_index=tuple(int(index) for index in inner_validation),
                )
            )

        folds.append(
            NestedTournamentFold(
                fold_id=f"outer_{outer_year}",
                validation_year=outer_year,
                train_index=tuple(int(index) for index in outer_train),
                validation_index=tuple(int(index) for index in outer_validation),
                inner_folds=tuple(inner_folds),
            )
        )
    return tuple(folds)


def _canonical_value(value: Any) -> Any:
    """Convert supported scientific configuration values to JSON primitives."""
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return _canonical_value(value.tolist())
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Candidate configuration keys must be strings")
            normalized[key] = _canonical_value(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        return sorted(items, key=lambda item: _canonical_json(item))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Candidate configurations cannot contain NaN or infinity")
        return 0.0 if value == 0.0 else value
    raise TypeError(f"Unsupported candidate configuration value: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


_TEMPORAL_KEY_PARTS = (
    "year",
    "years",
    "edition",
    "season",
    "cutoff",
    "train_end",
    "selection_end",
    "audit",
)


def _temporal_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
        date_match = re.match(r"^([12]\d{3})(?:\D|$)", stripped)
        if date_match is not None:
            return int(date_match.group(1))
    return None


class CandidateBudget:
    """Deduplicate, audit, cap, and freeze a preregistered candidate family."""

    def __init__(
        self,
        *,
        max_configurations: int,
        maximum_selection_year: int,
        forbidden_audit_years: Iterable[int] = (),
    ) -> None:
        self.max_configurations = _positive_integer(max_configurations, label="max_configurations")
        self.maximum_selection_year = _one_year(
            maximum_selection_year, label="maximum_selection_year"
        )
        self.forbidden_audit_years = tuple(
            sorted(
                {_one_year(year, label="forbidden_audit_years") for year in forbidden_audit_years}
            )
        )
        self._canonical_by_id: dict[str, str] = {}
        self._frozen = False
        self._manifest_hash: str | None = None

    @property
    def count(self) -> int:
        """Number of unique registered configurations."""
        return len(self._canonical_by_id)

    @property
    def frozen(self) -> bool:
        """Whether registration has been irreversibly closed."""
        return self._frozen

    @property
    def configuration_ids(self) -> tuple[str, ...]:
        """Candidate identifiers in their deterministic registration order."""
        return tuple(self._canonical_by_id)

    @property
    def manifest_hash(self) -> str | None:
        """Frozen budget-manifest hash, or ``None`` before freezing."""
        return self._manifest_hash

    def _assert_selection_safe(self, value: Any, *, temporal_context: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                folded_key = key.casefold()
                key_temporal = (
                    temporal_context
                    or any(part in folded_key for part in _TEMPORAL_KEY_PARTS)
                    or folded_key.endswith(("_date", "_datetime", "_timestamp", "_as_of"))
                )
                self._assert_selection_safe(item, temporal_context=key_temporal)
            return
        if isinstance(value, list):
            for item in value:
                self._assert_selection_safe(item, temporal_context=temporal_context)
            return
        if not temporal_context:
            return
        year = _temporal_year(value)
        if year is None:
            return
        if year in self.forbidden_audit_years:
            raise ValueError(f"Audit year {year} is forbidden during candidate selection")
        if year > self.maximum_selection_year:
            raise ValueError(
                f"Selection data year {year} exceeds maximum selection year "
                f"{self.maximum_selection_year}"
            )

    def register(self, configuration: Mapping[str, Any]) -> str:
        """Register one safe configuration and return its SHA-256 identity."""
        if self._frozen:
            raise RuntimeError("Candidate budget is frozen")
        if not isinstance(configuration, Mapping):
            raise TypeError("Candidate configuration must be a mapping")
        normalized = _canonical_value(configuration)
        self._assert_selection_safe(normalized)
        canonical = _canonical_json(normalized)
        candidate_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        existing = self._canonical_by_id.get(candidate_id)
        if existing is not None:
            if existing != canonical:
                raise RuntimeError("SHA-256 collision in candidate registry")
            return candidate_id
        if self.count >= self.max_configurations:
            raise ValueError(
                f"Candidate budget of {self.max_configurations} configurations is exhausted"
            )
        self._canonical_by_id[candidate_id] = canonical
        return candidate_id

    def freeze(self) -> str:
        """Close registration and return a deterministic manifest hash."""
        if self._manifest_hash is None:
            manifest = {
                "candidate_ids": sorted(self._canonical_by_id),
                "forbidden_audit_years": list(self.forbidden_audit_years),
                "max_configurations": self.max_configurations,
                "maximum_selection_year": self.maximum_selection_year,
            }
            encoded = _canonical_json(manifest).encode("utf-8")
            self._manifest_hash = hashlib.sha256(encoded).hexdigest()
        self._frozen = True
        return self._manifest_hash


__all__ = [
    "CandidateBudget",
    "InnerTournamentFold",
    "NestedTournamentFold",
    "nested_expanding_origin_folds",
]
