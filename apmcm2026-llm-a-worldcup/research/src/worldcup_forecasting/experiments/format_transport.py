"""Historical pseudo-transport audits for tournament format expansion."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning


@dataclass(frozen=True)
class FormatTransportResult:
    """Expansion-edition forecasts and an honestly bounded summary."""

    audits: pd.DataFrame
    summary: dict[str, Any]


def run_format_transport_audits(
    tournaments: pd.DataFrame,
    *,
    minimum_training_editions: int = 4,
) -> FormatTransportResult:
    """Compare a direct total trend with a Poisson match-volume offset model."""
    frame = _validated_frame(tournaments, minimum_training_editions)
    expansion = (
        frame["QualifiedTeams"].gt(frame["QualifiedTeams"].shift(1))
        | frame["MatchesPlayed"].gt(frame["MatchesPlayed"].shift(1))
    )
    rows: list[dict[str, Any]] = []
    for position in np.flatnonzero(expansion.to_numpy()):
        if position < minimum_training_editions:
            continue
        train = frame.iloc[:position]
        test = frame.iloc[position]
        previous = frame.iloc[position - 1]
        origin = float(train["Year"].mean())
        x_train = _design(train["Year"].to_numpy(float), origin)
        x_test = _design(np.asarray([float(test["Year"])]), origin)
        total_model = sm.OLS(train["GoalsScored"].to_numpy(float), x_train).fit()
        direct = float(total_model.predict(x_test)[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PerfectSeparationWarning)
            offset_model = sm.GLM(
                train["GoalsScored"].to_numpy(float),
                x_train,
                family=sm.families.Poisson(),
                offset=np.log(train["MatchesPlayed"].to_numpy(float)),
            ).fit()
        offset = float(
            offset_model.predict(
                x_test,
                offset=np.log(np.asarray([float(test["MatchesPlayed"])])),
            )[0]
        )
        observed = float(test["GoalsScored"])
        rows.append(
            {
                "test_year": int(test["Year"]),
                "train_end_year": int(train["Year"].max()),
                "training_editions": int(len(train)),
                "qualified_teams": int(test["QualifiedTeams"]),
                "matches_played": int(test["MatchesPlayed"]),
                "is_format_expansion": True,
                "is_team_field_expansion": bool(
                    test["QualifiedTeams"] > previous["QualifiedTeams"]
                ),
                "is_match_volume_change": bool(
                    test["MatchesPlayed"] > previous["MatchesPlayed"]
                ),
                "observed_total_goals": observed,
                "direct_total_prediction": direct,
                "offset_prediction": offset,
                "direct_total_absolute_error": abs(direct - observed),
                "offset_absolute_error": abs(offset - observed),
                "offset_improvement": abs(direct - observed) - abs(offset - observed),
                "evaluation_role": "historical_pseudo_transport_audit",
            }
        )
    if not rows:
        raise ValueError("No eligible historical format-expansion audit has enough training data")
    audits = pd.DataFrame(rows).sort_values("test_year").reset_index(drop=True)
    team_expansions = audits.loc[audits["is_team_field_expansion"]]
    summary = {
        "audit_count": int(len(audits)),
        "expansion_years": audits["test_year"].astype(int).tolist(),
        "direct_total_mae": float(audits["direct_total_absolute_error"].mean()),
        "offset_mae": float(audits["offset_absolute_error"].mean()),
        "offset_fold_wins": int(
            (audits["offset_absolute_error"] < audits["direct_total_absolute_error"]).sum()
        ),
        "team_field_expansion_audit_count": int(len(team_expansions)),
        "team_field_expansion_direct_mae": float(
            team_expansions["direct_total_absolute_error"].mean()
        ),
        "team_field_expansion_offset_mae": float(
            team_expansions["offset_absolute_error"].mean()
        ),
        "team_field_expansion_offset_fold_wins": int(
            (
                team_expansions["offset_absolute_error"]
                < team_expansions["direct_total_absolute_error"]
            ).sum()
        ),
        "match_volume_separated": True,
        "claim_scope": "historical_pseudo_transport_not_2026_guarantee",
    }
    return FormatTransportResult(audits, summary)


def _validated_frame(frame: pd.DataFrame, minimum_training_editions: int) -> pd.DataFrame:
    required = {"Year", "QualifiedTeams", "MatchesPlayed", "GoalsScored"}
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Format transport requires a nonempty tournament DataFrame")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Format transport frame is missing columns: {missing}")
    if isinstance(minimum_training_editions, bool) or minimum_training_editions < 3:
        raise ValueError("minimum_training_editions must be an integer of at least three")
    values = frame.loc[:, sorted(required)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Format transport fields must be complete and finite")
    if values["Year"].duplicated().any():
        raise ValueError("Tournament years must be unique for chronological transport audits")
    if (
        (values[["QualifiedTeams", "MatchesPlayed", "GoalsScored"]] <= 0).any().any()
        or not np.equal(values["Year"], np.floor(values["Year"])).all()
    ):
        raise ValueError("Tournament counts must be positive and years must be integers")
    return values.sort_values("Year").reset_index(drop=True)


def _design(year: np.ndarray, origin: float) -> np.ndarray:
    centered = (np.asarray(year, dtype=float) - origin) / 4.0
    return np.column_stack([np.ones(len(centered)), centered])


__all__ = ["FormatTransportResult", "run_format_transport_audits"]
