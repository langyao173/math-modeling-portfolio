"""Integrity tests for the prospective formal-development snapshot boundary."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from worldcup_forecasting.experiments._nested_selection_types import frame_content_hash
from worldcup_forecasting.experiments.development_snapshot import (
    build_snapshot_manifest,
    load_development_snapshot,
)


def _write_snapshot(directory: Path) -> Path:
    features = pd.DataFrame(
        {
            "Year": [1930, *([1950] * 770), 2010],
            "MatchID": range(1, 773),
            "value": [position / 772 for position in range(772)],
        }
    )
    years = [
        1930,
        1934,
        1938,
        1950,
        1954,
        1958,
        1962,
        1966,
        1970,
        1974,
        1978,
        1982,
        1986,
        1990,
        1994,
        1998,
        2002,
        2006,
        2010,
    ]
    tournaments = pd.DataFrame({"Year": years, "MatchesPlayed": [1] * len(years)})
    directory.mkdir()
    joblib.dump(features, directory / "development_features.joblib")
    tournaments.to_csv(directory / "tournaments_through_2010.csv", index=False)
    manifest = build_snapshot_manifest(
        directory,
        feature_schema_hash="a" * 64,
        development_frame_hash=frame_content_hash(features),
        source_raw_hashes={"matches": "b" * 64},
        development_rows=772,
    )
    (directory / "snapshot_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def test_development_snapshot_verifies_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _write_snapshot(tmp_path / "snapshot")
    events: list[str] = []
    original = joblib.load

    def tracked(path: Path):
        events.append(str(path))
        return original(path)

    monkeypatch.setattr(
        "worldcup_forecasting.experiments.development_snapshot.joblib.load", tracked
    )
    loaded = load_development_snapshot(directory)
    assert len(loaded.features) == 772
    assert loaded.metadata["physical_raw_no_read_claim"] is False
    assert loaded.metadata["formal_preaudit_reads_combined_raw"] is False
    assert len(events) == 1


def test_development_snapshot_rejects_tampering_before_joblib_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _write_snapshot(tmp_path / "snapshot")
    (directory / "tournaments_through_2010.csv").write_text("Year\n2014\n", encoding="utf-8")
    monkeypatch.setattr(
        "worldcup_forecasting.experiments.development_snapshot.joblib.load",
        lambda _: (_ for _ in ()).throw(AssertionError("must not deserialize")),
    )
    with pytest.raises(ValueError, match="size mismatch|hash mismatch"):
        load_development_snapshot(directory)


def test_development_snapshot_rejects_hash_consistent_truncation(tmp_path: Path) -> None:
    directory = _write_snapshot(tmp_path / "snapshot")
    features = joblib.load(directory / "development_features.joblib").drop(index=1)
    joblib.dump(features, directory / "development_features.joblib")
    manifest = build_snapshot_manifest(
        directory,
        feature_schema_hash="a" * 64,
        development_frame_hash=frame_content_hash(features),
        source_raw_hashes={"matches": "b" * 64},
        development_rows=len(features),
    )
    (directory / "snapshot_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="772 rows"):
        load_development_snapshot(directory)
