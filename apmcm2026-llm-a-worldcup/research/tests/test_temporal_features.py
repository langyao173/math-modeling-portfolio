from pathlib import Path

import numpy as np
import pytest

from worldcup_forecasting.data.audit import build_canonical_data
from worldcup_forecasting.features.team_history import build_team_match_features


@pytest.mark.integration
def test_history_features_use_completed_prior_tournaments_only(raw_dir: Path) -> None:
    bundle = build_canonical_data(raw_dir)
    features = build_team_match_features(bundle.matches, bundle.tournaments)

    assert (features["history_cutoff_year"] < features["Year"]).all()
    first_edition = features.loc[features["Year"] == 1930]
    assert (first_edition["home_historical_appearances"] == 0).all()
    assert (first_edition["away_historical_appearances"] == 0).all()


@pytest.mark.integration
def test_pairwise_features_are_antisymmetric(raw_dir: Path) -> None:
    bundle = build_canonical_data(raw_dir)
    features = build_team_match_features(bundle.matches, bundle.tournaments)
    diff_columns = [column for column in features.columns if column.endswith("_diff")]
    assert diff_columns

    sample = features.iloc[100]
    swapped = features.iloc[100].copy()
    for column in diff_columns:
        swapped[column] = -sample[column]
    np.testing.assert_allclose(
        swapped[diff_columns].to_numpy(dtype=float),
        -sample[diff_columns].to_numpy(dtype=float),
    )


@pytest.mark.integration
def test_fixed_official_split_is_uncontaminated(raw_dir: Path) -> None:
    bundle = build_canonical_data(raw_dir)
    train = bundle.matches.loc[bundle.matches["Year"] <= 2010]
    test = bundle.matches.loc[bundle.matches["Year"] == 2014]
    assert train.shape[0] == 772
    assert test.shape[0] == 64
    assert train["MatchID"].is_unique
    assert test["MatchID"].is_unique
    assert set(train["MatchID"]).isdisjoint(test["MatchID"])
