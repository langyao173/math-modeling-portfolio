import pandas as pd
import pytest

from worldcup_forecasting.models import MANDATORY_OUTCOME_FEATURES
from worldcup_forecasting.validation.feature_availability import (
    assert_dynamic_availability,
    assert_static_history_availability,
    feature_availability_registry,
)


def test_registry_has_required_metadata_for_every_static_outcome_feature() -> None:
    registry = feature_availability_registry(include_dynamic=True, include_player=True)

    assert set(registry.columns) == {
        "feature_name",
        "available_time",
        "source_table",
        "aggregation_cutoff",
        "prediction_information_set",
    }
    assert set(MANDATORY_OUTCOME_FEATURES).issubset(set(registry["feature_name"]))
    assert not registry.isna().any().any()
    q1_attendance = registry.loc[registry["feature_name"].eq("Attendance")].iloc[0]
    q2_attendance = registry.loc[registry["feature_name"].eq("attendance")].iloc[0]
    assert q1_attendance["prediction_information_set"] == "pre_match"
    assert q1_attendance["source_table"] == "tournaments"
    assert q2_attendance["prediction_information_set"] == "half_time"
    assert q2_attendance["source_table"] == "matches"


def test_static_history_availability_fails_on_current_edition() -> None:
    frame = pd.DataFrame(
        {
            "Year": [2010],
            "history_cutoff_year": [2010],
            **{feature: [0.0] for feature in MANDATORY_OUTCOME_FEATURES},
        }
    )

    with pytest.raises(AssertionError, match="earlier"):
        assert_static_history_availability(frame)


def test_dynamic_availability_fails_on_same_timestamp_update() -> None:
    frame = pd.DataFrame(
        {
            "Datetime": ["2010-06-10 18:00:00"],
            "home_dynamic_cutoff_time": ["2010-06-10 18:00:00"],
            "away_dynamic_cutoff_time": [None],
            "home_dynamic_history_matches": [3],
            "away_dynamic_history_matches": [0],
        }
    )

    with pytest.raises(AssertionError, match="availability violation"):
        assert_dynamic_availability(frame)
