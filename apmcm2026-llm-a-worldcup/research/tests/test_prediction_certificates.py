import json
from pathlib import Path

import pytest

from worldcup_forecasting.reporting.model_state import write_model_state
from worldcup_forecasting.uncertainty.certificates import (
    build_prediction_certificate,
    prediction_payload,
)


def _certificate(model_hash: str) -> dict[str, object]:
    return build_prediction_certificate(
        information_set="pre_match",
        data_cutoff="2014-07-13",
        raw_data_hashes={"matches": "a" * 64},
        feature_schema=["feature_a"],
        model_hash=model_hash,
        model_artifact="models/example.json",
        config_hash="b" * 64,
        random_seed=1729,
        software_versions={"python": "3.12.11"},
        calibration_method="none",
        calibration_artifact=None,
        prediction={"home_win": 0.5, "draw": 0.25, "away_win": 0.25},
        goal_interval_90=None,
        out_of_distribution_score=0.2,
        invariant_status={"probability_simplex": True},
        scenario_label="example",
        generated_at_utc="2026-07-18T00:00:00+00:00",
    )


def test_model_state_hashes_the_written_fitted_state(tmp_path: Path) -> None:
    first = write_model_state(
        tmp_path / "model.json",
        model_type="example",
        feature_schema=["x"],
        fitted_state={"coefficient": [1.25], "intercept": 0.5},
        training_contract={"data_cutoff": "2014-07-13"},
        random_seed=1729,
    )
    second = write_model_state(
        tmp_path / "model-copy.json",
        model_type="example",
        feature_schema=["x"],
        fitted_state={"coefficient": [1.25], "intercept": 0.5},
        training_contract={"data_cutoff": "2014-07-13"},
        random_seed=1729,
    )

    assert first.sha256 == second.sha256
    assert first.sha256 == first.path_sha256()
    assert json.loads(first.path.read_text(encoding="utf-8"))["fitted_state"][
        "coefficient"
    ] == [1.25]


def test_prediction_payload_excludes_volatile_generation_time(tmp_path: Path) -> None:
    artifact = write_model_state(
        tmp_path / "model.json",
        model_type="example",
        feature_schema=["feature_a"],
        fitted_state={"coefficient": [1.0]},
        training_contract={"data_cutoff": "2014-07-13"},
        random_seed=1729,
    )
    certificate = _certificate(artifact.sha256)
    payload = prediction_payload(certificate)

    assert "generated_at_utc" not in payload
    assert payload["model_hash"] == artifact.sha256
    assert payload["random_seed"] == 1729


def test_certificate_rejects_untraceable_hash_or_bad_simplex() -> None:
    with pytest.raises(ValueError, match="model_hash"):
        _certificate("metadata-only")

    valid = _certificate("c" * 64)
    valid["prediction"] = {"home_win": 0.8, "draw": 0.3, "away_win": 0.2}
    with pytest.raises(ValueError, match="sum to one"):
        prediction_payload(valid)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generated_at_utc", "not-a-timestamp", "ISO-8601 timestamp"),
        ("generated_at_utc", "2026-07-18T00:00:00", "UTC offset"),
        ("data_cutoff", "2014/07/13", "calendar date"),
        ("calibration_method", "made_up", "Unsupported calibration_method"),
    ],
)
def test_certificate_rejects_invalid_schema_values(
    field: str,
    value: str,
    message: str,
) -> None:
    certificate = _certificate("c" * 64)
    certificate[field] = value

    with pytest.raises(ValueError, match=message):
        prediction_payload(certificate)


def test_certificate_rejects_failed_invariant() -> None:
    certificate = _certificate("c" * 64)
    certificate["invariant_status"] = {"probability_simplex": False}

    with pytest.raises(ValueError, match="failed invariants"):
        prediction_payload(certificate)


def test_certificate_rejects_impossible_half_time_prediction_and_interval() -> None:
    certificate = _certificate("c" * 64)
    certificate["information_set"] = "half_time"
    certificate["prediction"] = {
        "half_time_total_goals": 3,
        "full_time_total_goals": 2,
    }
    certificate["goal_interval_90"] = [0, 1]

    with pytest.raises(ValueError, match="full-time goals"):
        prediction_payload(certificate)


def test_interval_calibration_requires_artifact() -> None:
    certificate = _certificate("c" * 64)
    certificate[
        "calibration_method"
    ] = "strictly_earlier_rolling_absolute_residual_interval"

    with pytest.raises(ValueError, match="requires.*calibration_artifact"):
        prediction_payload(certificate)
