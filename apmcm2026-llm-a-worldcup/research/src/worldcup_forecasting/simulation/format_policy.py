"""Declarative, source-bound World Cup knockout bracket policies."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from io import StringIO
from types import MappingProxyType

FIFA_2026_REGULATIONS_URL = (
    "https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/"
    "FWC2026_regulations_EN.pdf"
)
FIFA_2026_REGULATIONS_SHA256 = (
    "8e4e9f868cb49f2dd8852622c290450da8adaf8cf262a81f08d16c7e5a598f96"
)
FIFA_2026_ANNEX_C_CSV_SHA256 = (
    "883fee7afe8d6e1a00f96370d2172b68ccf9b34fbac851284432466837130d3b"
)
FIFA_2026_ANNEX_C_POLICY = "fifa_2026_annex_c"
WORLD_CUP_2022_CROSS_GROUP_POLICY = "world_cup_2022_cross_group"
SUPPORTED_BRACKET_POLICIES = frozenset(
    {FIFA_2026_ANNEX_C_POLICY, WORLD_CUP_2022_CROSS_GROUP_POLICY}
)

GROUPS_2026 = tuple("ABCDEFGHIJKL")
ANNEX_C_WINNER_COLUMNS = ("1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L")
ANNEX_C_ALLOWED_THIRD_GROUPS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "1A": frozenset("CEFHI"),
        "1B": frozenset("EFGIJ"),
        "1D": frozenset("BEFIJ"),
        "1E": frozenset("ABCDF"),
        "1G": frozenset("AEHIJ"),
        "1I": frozenset("CDFGH"),
        "1K": frozenset("DEIJL"),
        "1L": frozenset("EHIJK"),
    }
)


@dataclass(frozen=True, slots=True)
class AnnexCRow:
    """One FIFA Annexe C assignment for a set of eight third-placed groups."""

    option: int
    qualifying_groups: str
    assignments: tuple[tuple[str, str], ...]

    def assignment_for(self, group_winner_slot: str) -> str:
        for winner, third in self.assignments:
            if winner == group_winner_slot:
                return third
        raise KeyError(group_winner_slot)


@dataclass(frozen=True, slots=True)
class AnnexCTable:
    """Fully validated 495-row FIFA Annexe C table."""

    rows: tuple[AnnexCRow, ...]
    by_qualifying_groups: Mapping[str, AnnexCRow]
    csv_sha256: str

    def row_for(self, qualifying_groups: Sequence[str]) -> AnnexCRow:
        key = _canonical_group_set(qualifying_groups)
        try:
            return self.by_qualifying_groups[key]
        except KeyError as exc:
            raise ValueError(
                "FIFA 2026 Annexe C requires exactly eight distinct groups A-L"
            ) from exc


def _canonical_group_set(groups: Sequence[str]) -> str:
    normalized = tuple(str(group).strip().upper() for group in groups)
    if len(normalized) != 8 or len(set(normalized)) != 8:
        raise ValueError("Exactly eight distinct third-place groups are required")
    if not set(normalized).issubset(GROUPS_2026):
        raise ValueError("Third-place group labels must be drawn from A-L")
    return "".join(sorted(normalized))


def _parse_annex_c(csv_text: str) -> tuple[AnnexCRow, ...]:
    reader = csv.DictReader(StringIO(csv_text))
    expected_fields = ["option", "qualifying_groups", *ANNEX_C_WINNER_COLUMNS]
    if reader.fieldnames != expected_fields:
        raise RuntimeError("FIFA 2026 Annexe C CSV has an invalid schema")
    rows: list[AnnexCRow] = []
    for raw in reader:
        try:
            option = int(raw["option"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Annexe C option is not an integer") from exc
        assignments = tuple(
            (winner, str(raw[winner]).strip().upper())
            for winner in ANNEX_C_WINNER_COLUMNS
        )
        rows.append(
            AnnexCRow(
                option=option,
                qualifying_groups=str(raw["qualifying_groups"]).strip().upper(),
                assignments=assignments,
            )
        )
    return tuple(rows)


def validate_annex_c_rows(rows: Sequence[AnnexCRow]) -> None:
    """Fail closed unless rows are the complete valid choose(12, 8) mapping."""
    if len(rows) != 495:
        raise RuntimeError("FIFA 2026 Annexe C must contain exactly 495 rows")
    expected_combinations = {
        "".join(groups) for groups in itertools.combinations(GROUPS_2026, 8)
    }
    observed: set[str] = set()
    for expected_option, row in enumerate(rows, start=1):
        if row.option != expected_option:
            raise RuntimeError("Annexe C option numbers must be exactly 1 through 495")
        if len(row.assignments) != 8:
            raise RuntimeError(f"Annexe C option {row.option} lacks eight assignments")
        winners = tuple(winner for winner, _ in row.assignments)
        thirds = tuple(third for _, third in row.assignments)
        if winners != ANNEX_C_WINNER_COLUMNS:
            raise RuntimeError(f"Annexe C option {row.option} has reordered winner columns")
        if any(
            len(third) != 2 or third[0] != "3" or third[1] not in GROUPS_2026
            for third in thirds
        ):
            raise RuntimeError(f"Annexe C option {row.option} has an invalid third-place slot")
        third_groups = tuple(third[1] for third in thirds)
        if len(set(third_groups)) != 8:
            raise RuntimeError(f"Annexe C option {row.option} repeats a third-place group")
        if row.qualifying_groups != "".join(sorted(third_groups)):
            raise RuntimeError(f"Annexe C option {row.option} has inconsistent group membership")
        if any(winner[1] == third[1] for winner, third in row.assignments):
            raise RuntimeError(f"Annexe C option {row.option} creates a same-group rematch")
        if any(
            third[1] not in ANNEX_C_ALLOWED_THIRD_GROUPS[winner]
            for winner, third in row.assignments
        ):
            raise RuntimeError(
                f"Annexe C option {row.option} violates an official variable fixture slot"
            )
        observed.add(row.qualifying_groups)
    if observed != expected_combinations:
        raise RuntimeError("Annexe C does not cover each choose(12, 8) combination once")


@lru_cache(maxsize=1)
def load_fifa_2026_annex_c() -> AnnexCTable:
    """Load and cryptographically verify the generated package data."""
    package = resources.files("worldcup_forecasting.simulation.data")
    csv_bytes = package.joinpath("fifa_2026_annex_c.csv").read_bytes()
    metadata = json.loads(
        package.joinpath("fifa_2026_annex_c.metadata.json").read_text(encoding="utf-8")
    )
    csv_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    if csv_sha256 != FIFA_2026_ANNEX_C_CSV_SHA256:
        raise RuntimeError("Packaged FIFA 2026 Annexe C CSV failed its pinned SHA-256")
    required_metadata = {
        "schema_version": 1,
        "source_url": FIFA_2026_REGULATIONS_URL,
        "source_sha256": FIFA_2026_REGULATIONS_SHA256,
        "row_count": 495,
        "combination_count": 495,
        "csv_sha256": FIFA_2026_ANNEX_C_CSV_SHA256,
        "winner_columns": list(ANNEX_C_WINNER_COLUMNS),
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"Packaged FIFA 2026 Annexe C metadata mismatch: {key}")
    rows = _parse_annex_c(csv_bytes.decode("utf-8"))
    validate_annex_c_rows(rows)
    index = MappingProxyType({row.qualifying_groups: row for row in rows})
    return AnnexCTable(rows=rows, by_qualifying_groups=index, csv_sha256=csv_sha256)


def first_knockout_source_pairs(
    bracket_policy: str,
    *,
    third_place_groups: Sequence[str] = (),
) -> tuple[tuple[str, str], ...]:
    """Return the official first-knockout fixture sources in match-number order."""
    if bracket_policy == WORLD_CUP_2022_CROSS_GROUP_POLICY:
        if third_place_groups:
            raise ValueError("The 2022 policy has no best-third qualifiers")
        return (
            ("1A", "2B"),
            ("1C", "2D"),
            ("1B", "2A"),
            ("1D", "2C"),
            ("1E", "2F"),
            ("1G", "2H"),
            ("1F", "2E"),
            ("1H", "2G"),
        )
    if bracket_policy != FIFA_2026_ANNEX_C_POLICY:
        raise ValueError(f"Unsupported bracket policy: {bracket_policy}")
    row = load_fifa_2026_annex_c().row_for(third_place_groups)
    assignment = dict(row.assignments)
    return (
        ("2A", "2B"),
        ("1E", assignment["1E"]),
        ("1F", "2C"),
        ("1C", "2F"),
        ("1I", assignment["1I"]),
        ("2E", "2I"),
        ("1A", assignment["1A"]),
        ("1L", assignment["1L"]),
        ("1D", assignment["1D"]),
        ("1G", assignment["1G"]),
        ("2K", "2L"),
        ("1H", "2J"),
        ("1B", assignment["1B"]),
        ("1J", "2H"),
        ("1K", assignment["1K"]),
        ("2D", "2G"),
    )


_FIFA_2026_LATER_SOURCES: Mapping[str, tuple[tuple[str, str], ...]] = MappingProxyType(
    {
        "round_of_16": (
            ("W74", "W77"),
            ("W73", "W75"),
            ("W76", "W78"),
            ("W79", "W80"),
            ("W83", "W84"),
            ("W81", "W82"),
            ("W86", "W88"),
            ("W85", "W87"),
        ),
        "quarter_final": (
            ("W89", "W90"),
            ("W93", "W94"),
            ("W91", "W92"),
            ("W95", "W96"),
        ),
        "semi_final": (("W97", "W98"), ("W99", "W100")),
        "third_place": (("L101", "L102"),),
        "final": (("W101", "W102"),),
    }
)

_WORLD_CUP_2022_LATER_SOURCES: Mapping[str, tuple[tuple[str, str], ...]] = (
    MappingProxyType(
        {
            "quarter_final": (
                ("W49", "W50"),
                ("W53", "W54"),
                ("W51", "W52"),
                ("W55", "W56"),
            ),
            "semi_final": (("W57", "W58"), ("W59", "W60")),
            "third_place": (("L61", "L62"),),
            "final": (("W61", "W62"),),
        }
    )
)


def later_knockout_source_pairs(
    bracket_policy: str, stage: str
) -> tuple[tuple[str, str], ...]:
    """Return official winner/loser source pairs for a later knockout stage."""
    policies = {
        FIFA_2026_ANNEX_C_POLICY: _FIFA_2026_LATER_SOURCES,
        WORLD_CUP_2022_CROSS_GROUP_POLICY: _WORLD_CUP_2022_LATER_SOURCES,
    }
    try:
        return policies[bracket_policy][stage]
    except KeyError as exc:
        raise ValueError(
            f"No declared source graph for policy={bracket_policy!r}, stage={stage!r}"
        ) from exc


__all__ = [
    "ANNEX_C_ALLOWED_THIRD_GROUPS",
    "ANNEX_C_WINNER_COLUMNS",
    "FIFA_2026_ANNEX_C_CSV_SHA256",
    "FIFA_2026_ANNEX_C_POLICY",
    "FIFA_2026_REGULATIONS_SHA256",
    "FIFA_2026_REGULATIONS_URL",
    "SUPPORTED_BRACKET_POLICIES",
    "WORLD_CUP_2022_CROSS_GROUP_POLICY",
    "AnnexCRow",
    "AnnexCTable",
    "first_knockout_source_pairs",
    "later_knockout_source_pairs",
    "load_fifa_2026_annex_c",
    "validate_annex_c_rows",
]
