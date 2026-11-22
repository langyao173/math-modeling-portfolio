"""Tests for separating match volume from per-match scoring intensity."""

from __future__ import annotations

import pandas as pd
import pytest

from worldcup_forecasting.experiments.format_transport import run_format_transport_audits


def _tournaments() -> pd.DataFrame:
    years = [1966, 1970, 1974, 1978, 1982, 1986, 1990, 1994, 1998, 2002]
    matches = [32, 32, 38, 38, 52, 52, 52, 52, 64, 64]
    teams = [16, 16, 16, 16, 24, 24, 24, 24, 32, 32]
    return pd.DataFrame(
        {
            "Year": years,
            "QualifiedTeams": teams,
            "MatchesPlayed": matches,
            "GoalsScored": [round(2.5 * value) for value in matches],
        }
    )


def test_offset_transport_detects_expansions_and_beats_direct_total_trend() -> None:
    result = run_format_transport_audits(_tournaments(), minimum_training_editions=4)

    assert result.audits["test_year"].tolist() == [1982, 1998]
    assert result.audits["is_format_expansion"].all()
    assert result.audits["offset_absolute_error"].max() < 1.0
    assert (
        result.audits["offset_absolute_error"]
        < result.audits["direct_total_absolute_error"]
    ).all()
    assert result.summary["offset_fold_wins"] == 2
    assert result.summary["claim_scope"] == "historical_pseudo_transport_not_2026_guarantee"


def test_format_transport_rejects_nonchronological_or_insufficient_data() -> None:
    duplicated = pd.concat([_tournaments(), _tournaments().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique|year"):
        run_format_transport_audits(duplicated)

    with pytest.raises(ValueError, match="training|audit|expansion"):
        run_format_transport_audits(_tournaments().iloc[:4], minimum_training_editions=4)

