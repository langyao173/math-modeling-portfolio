"""Per-feature information-set registry and executable availability checks."""

from __future__ import annotations

import pandas as pd

from worldcup_forecasting.models import MANDATORY_OUTCOME_FEATURES

_PLAYER_DIFF_FEATURES = (
    "player_roster_size_diff",
    "player_known_history_count_diff",
    "player_history_missing_count_diff",
    "player_history_missing_rate_diff",
    "player_prior_appearances_mean_diff",
    "player_prior_starts_mean_diff",
    "player_prior_goals_mean_diff",
    "player_prior_cards_mean_diff",
    "player_lineup_continuity_rate_diff",
)


def feature_availability_registry(
    *,
    include_dynamic: bool = False,
    include_player: bool = False,
) -> pd.DataFrame:
    """Return one explicit availability record for every modeled feature family."""
    rows = [
        _record(
            "QualifiedTeams",
            "pre_tournament_known_format",
            "tournaments",
            "declared target-edition format",
            "pre_match",
        ),
        _record(
            "MatchesPlayed",
            "pre_tournament_known_format",
            "tournaments",
            "declared target-edition format",
            "pre_match",
        ),
        _record(
            "Attendance",
            "pre_tournament_forecast",
            "tournaments",
            "forecast from strictly earlier editions for Q1",
            "pre_match",
        ),
        _record(
            "attendance",
            "kickoff_or_earlier_supplied_match_value",
            "matches",
            "supplied match observation available no later than the Q2 prediction cutoff",
            "half_time",
        ),
        _record(
            "canonical_stage",
            "pre_match",
            "matches",
            "known competition stage",
            "half_time",
        ),
        _record(
            "half_time_total_goals",
            "half_time",
            "matches",
            "score observed at half-time only",
            "half_time",
        ),
    ]
    rows.extend(
        _record(
            feature,
            "history_cutoff_year",
            "matches_and_tournaments",
            "completed editions with Year < prediction Year",
            "pre_match",
        )
        for feature in MANDATORY_OUTCOME_FEATURES
    )
    if include_dynamic:
        rows.extend(
            _record(
                feature,
                "home/away_dynamic_cutoff_time",
                "matches",
                "strict predict-then-update; cutoff time < match Datetime",
                "pre_match",
            )
            for feature in ("pre_match_elo_diff", "attack_state_diff", "defense_state_diff")
        )
    if include_player:
        rows.extend(
            _record(
                feature,
                "home/away_player_history_cutoff_time",
                "players_joined_to_matches",
                "prior player events only; cutoff time < match Datetime",
                "kickoff",
            )
            for feature in _PLAYER_DIFF_FEATURES
        )
    return pd.DataFrame(rows)


def assert_static_history_availability(frame: pd.DataFrame) -> dict[str, int | str]:
    """Fail if any tournament-history feature includes the prediction edition."""
    required = {"Year", "history_cutoff_year", *MANDATORY_OUTCOME_FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Static feature availability check missing columns: {missing}")
    invalid = frame["history_cutoff_year"].ge(frame["Year"])
    if invalid.any():
        raise AssertionError("Static history cutoff must be earlier than prediction edition")
    return {
        "check": "static_history_cutoff_before_prediction_year",
        "rows_checked": int(len(frame)),
        "features_checked": len(MANDATORY_OUTCOME_FEATURES),
        "violations": 0,
    }


def assert_dynamic_availability(frame: pd.DataFrame) -> dict[str, int | str]:
    """Fail if a nonempty dynamic state is not strictly earlier than the match."""
    _assert_timestamp_family(
        frame,
        cutoff_suffix="dynamic_cutoff_time",
        history_suffix="dynamic_history_matches",
    )
    return {
        "check": "dynamic_cutoff_before_match_time",
        "rows_checked": int(len(frame)),
        "features_checked": 3,
        "violations": 0,
    }


def assert_player_availability(frame: pd.DataFrame) -> dict[str, int | str]:
    """Fail if known player history is not strictly earlier than the match."""
    _assert_timestamp_family(
        frame,
        cutoff_suffix="player_history_cutoff_time",
        history_suffix="player_known_history_count",
    )
    return {
        "check": "player_history_cutoff_before_match_time",
        "rows_checked": int(len(frame)),
        "features_checked": len(_PLAYER_DIFF_FEATURES),
        "violations": 0,
    }


def _assert_timestamp_family(
    frame: pd.DataFrame,
    *,
    cutoff_suffix: str,
    history_suffix: str,
) -> None:
    if "Datetime" not in frame:
        raise ValueError("Timestamp availability checks require Datetime")
    prediction = pd.to_datetime(frame["Datetime"], errors="coerce")
    if prediction.isna().any():
        raise ValueError("Prediction Datetime must be complete")
    for side in ("home", "away"):
        cutoff_column = f"{side}_{cutoff_suffix}"
        history_column = f"{side}_{history_suffix}"
        if cutoff_column not in frame or history_column not in frame:
            raise ValueError(f"Timestamp availability check missing {cutoff_column}")
        cutoff = pd.to_datetime(frame[cutoff_column], errors="coerce")
        history = pd.to_numeric(frame[history_column], errors="coerce")
        if history.isna().any():
            raise ValueError(f"History count must be numeric: {history_column}")
        invalid_time = cutoff.notna() & cutoff.ge(prediction)
        missing_cutoff = cutoff.isna() & history.gt(0)
        if invalid_time.any() or missing_cutoff.any():
            raise AssertionError(
                f"Feature availability violation for {side}: "
                f"late={int(invalid_time.sum())}, missing={int(missing_cutoff.sum())}"
            )


def _record(
    feature_name: str,
    available_time: str,
    source_table: str,
    aggregation_cutoff: str,
    prediction_information_set: str,
) -> dict[str, str]:
    return {
        "feature_name": feature_name,
        "available_time": available_time,
        "source_table": source_table,
        "aggregation_cutoff": aggregation_cutoff,
        "prediction_information_set": prediction_information_set,
    }
