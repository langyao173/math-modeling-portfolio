from pathlib import Path

import pandas as pd
import pytest

from worldcup_forecasting.data.audit import build_canonical_data
from worldcup_forecasting.data.canonicalize import canonical_stage, parse_attendance


def test_dot_grouped_attendance_is_not_decimal() -> None:
    assert parse_attendance("590.549") == 590_549
    assert parse_attendance("1.045.246") == 1_045_246
    assert parse_attendance("3.386.810") == 3_386_810


@pytest.mark.integration
def test_canonical_data_reconciles_supplied_files(raw_dir: Path) -> None:
    bundle = build_canonical_data(raw_dir)

    assert bundle.tournaments.shape[0] == 20
    assert bundle.matches.shape[0] == 836
    assert bundle.players.shape[0] == 37_048
    assert not bundle.matches.duplicated(["RoundID", "MatchID"]).any()
    assert not bundle.players.duplicated().any()

    test_2014 = bundle.matches.loc[bundle.matches["Year"] == 2014]
    assert test_2014.shape[0] == 64
    assert int(test_2014["full_time_total_goals"].sum()) == 171
    assert int(test_2014["Attendance"].sum(skipna=True)) == 3_386_810

    reconciliation = bundle.reconciliation.set_index("Year")
    assert reconciliation["matches_match"].all()
    assert reconciliation["goals_match"].all()
    assert reconciliation["attendance_match"].all()


@pytest.mark.integration
def test_stage_mapping_covers_all_observed_labels(raw_dir: Path) -> None:
    raw = pd.read_csv(raw_dir / "WorldCupMatches.csv")
    mapped = {
        canonical_stage(str(stage), int(year))
        for stage, year in raw[["Stage", "Year"]].itertuples(index=False)
    }
    assert "unknown" not in mapped
    assert mapped == {
        "group_stage",
        "round_of_16",
        "quarter_final",
        "semi_final",
        "third_place",
        "final",
    }
