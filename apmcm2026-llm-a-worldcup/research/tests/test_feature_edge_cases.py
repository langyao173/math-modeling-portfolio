"""Focused regression tests for point-in-time feature contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from worldcup_forecasting.features import (
    build_dynamic_features,
    build_player_history_features,
    build_team_match_features,
)


def _matches(
    years: list[int],
    match_ids: list[object],
    times: list[str],
    home: list[str],
    away: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": years,
            "MatchID": match_ids,
            "match_datetime": pd.to_datetime(times),
            "home_team_initials": home,
            "away_team_initials": away,
            "home_goals": [1] * len(years),
            "away_goals": [0] * len(years),
        }
    )


def test_unused_roster_member_does_not_gain_player_appearance() -> None:
    matches = _matches(
        [2000, 2000, 2000],
        [1, 2, 3],
        ["2000-01-01", "2000-01-02", "2000-01-03"],
        ["AAA", "AAA", "AAA"],
        ["BBB", "CCC", "DDD"],
    )
    players = pd.DataFrame(
        {
            "MatchID": [1, 2, 3],
            "Team Initials": ["AAA", "AAA", "AAA"],
            "Player Name": ["Player One"] * 3,
            "Shirt Number": [9, 9, 9],
            "Line-up": ["N", "S", "N"],
            "Event": [None, None, None],
        }
    )

    features = build_player_history_features(players, matches)

    assert features.loc[1, "home_player_prior_appearances_mean"] == 0
    assert features.loc[1, "home_player_prior_history_missing"] == 1
    assert features.loc[2, "home_player_prior_appearances_mean"] == 1


def test_tournament_history_fails_closed_on_missing_or_undeclared_editions() -> None:
    missing = _matches(
        [2000, 2008],
        [1, 2],
        ["2000-01-01", "2008-01-01"],
        ["AAA", "AAA"],
        ["BBB", "BBB"],
    )
    with pytest.raises(ValueError, match="missing.*2004"):
        build_team_match_features(missing, pd.DataFrame({"Year": [2000, 2004, 2008]}))

    undeclared = missing.assign(Year=[2000, 2006])
    with pytest.raises(ValueError, match="absent from tournaments.*2006"):
        build_team_match_features(undeclared, pd.DataFrame({"Year": [2000, 2004]}))


def test_numeric_equivalent_match_ids_are_rejected_before_joining() -> None:
    matches = _matches(
        [2000, 2000],
        [1, "1"],
        ["2000-01-01", "2000-01-02"],
        ["AAA", "CCC"],
        ["BBB", "DDD"],
    )
    with pytest.raises(ValueError, match="unique.*MatchID"):
        build_dynamic_features(matches)


def test_duplicate_player_fragments_are_rejected_explicitly() -> None:
    matches = _matches([2000], [1], ["2000-01-01"], ["AAA"], ["BBB"])
    players = pd.DataFrame(
        {
            "MatchID": [1, 1],
            "Team Initials": ["AAA", "AAA"],
            "Player Name": ["Player One", "Player One"],
            "Shirt Number": [9, 9],
            "Line-up": ["S", "S"],
            "Event": ["G10'", "G20'"],
        }
    )
    with pytest.raises(ValueError, match="player rows must be unique"):
        build_player_history_features(players, matches)


def test_dynamic_state_rejects_same_team_at_identical_timestamp() -> None:
    matches = _matches(
        [2000, 2000],
        [1, 2],
        ["2000-01-01", "2000-01-01"],
        ["AAA", "AAA"],
        ["BBB", "CCC"],
    )
    with pytest.raises(ValueError, match="multiple matches at the same timestamp"):
        build_dynamic_features(matches)
