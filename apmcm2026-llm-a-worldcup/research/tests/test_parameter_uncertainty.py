import numpy as np
import pandas as pd
import pytest

from worldcup_forecasting.experiments.parameter_uncertainty import (
    TournamentBootstrapConfig,
    run_tournament_cluster_refits,
)


class _MeanModel:
    def __init__(self, mean: float) -> None:
        self.mean = mean

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        home = np.full(len(frame), 1.0 / (1.0 + np.exp(-self.mean)))
        draw = np.full(len(frame), 0.2)
        away = 1.0 - home
        raw = np.column_stack((home, draw, away))
        return raw / raw.sum(axis=1, keepdims=True)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    training = pd.DataFrame(
        {
            "Year": [2002, 2002, 2006, 2006, 2010, 2010],
            "MatchID": [1, 2, 3, 4, 5, 6],
            "x": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        }
    )
    probe = pd.DataFrame({"MatchID": [101, 102], "x": [0.0, 0.0]})
    return training, probe


def test_tournament_cluster_refits_are_deterministic_and_auditable() -> None:
    training, probe = _frames()
    sampled: list[tuple[int, ...]] = []

    def refit(frame: pd.DataFrame, seed: int) -> _MeanModel:
        del seed
        sampled.append(tuple(frame["Year"].astype(int)))
        return _MeanModel(float(frame["x"].mean()))

    config = TournamentBootstrapConfig(replicates=12, seed=17)
    first = run_tournament_cluster_refits(training, probe, refit, config=config)
    sampled.clear()
    second = run_tournament_cluster_refits(training, probe, refit, config=config)

    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    pd.testing.assert_frame_equal(first.draw_ledger, second.draw_ledger)
    assert first.failures.empty
    assert len(first.predictions) == 24
    assert first.predictions.groupby("bootstrap_draw_id").size().eq(2).all()
    assert set(first.predictions["match_id"]) == {101, 102}
    assert np.allclose(
        first.predictions[["home_win", "draw", "away_win"]].sum(axis=1), 1.0
    )
    assert first.draw_ledger["sampled_editions"].str.count(r"\|").eq(2).all()
    assert first.metadata["sampling_unit"] == "tournament_edition"
    assert first.metadata["post_2010_rows_used"] is False


def test_tournament_cluster_refits_reject_post_cutoff_training() -> None:
    training, probe = _frames()
    training.loc[len(training)] = [2014, 7, 4.0]

    with pytest.raises(ValueError, match="2010"):
        run_tournament_cluster_refits(
            training,
            probe,
            lambda frame, seed: _MeanModel(float(frame["x"].mean())),
            config=TournamentBootstrapConfig(replicates=4),
        )

    with pytest.raises(ValueError, match="2010|frozen|training"):
        run_tournament_cluster_refits(
            training,
            probe,
            lambda frame, seed: _MeanModel(float(frame["x"].mean())),
            config=TournamentBootstrapConfig(
                replicates=4, maximum_training_year=2014
            ),
        )


def test_tournament_cluster_refits_require_unambiguous_identifiers() -> None:
    training, probe = _frames()
    training.loc[training.index[-1], "MatchID"] = training.loc[
        training.index[0], "MatchID"
    ]
    with pytest.raises(ValueError, match="training match identifiers.*unique|duplicate"):
        run_tournament_cluster_refits(
            training,
            probe,
            lambda frame, seed: _MeanModel(float(frame["x"].mean())),
            config=TournamentBootstrapConfig(replicates=4),
        )

    training, probe = _frames()
    probe.loc[probe.index[0], "MatchID"] = np.nan
    with pytest.raises(ValueError, match="probe match identifiers.*complete"):
        run_tournament_cluster_refits(
            training,
            probe,
            lambda frame, seed: _MeanModel(float(frame["x"].mean())),
            config=TournamentBootstrapConfig(replicates=4),
        )


def test_tournament_cluster_refits_fail_closed_on_excess_refit_failures() -> None:
    training, probe = _frames()

    def broken(frame: pd.DataFrame, seed: int) -> _MeanModel:
        del frame
        if seed % 2:
            raise RuntimeError("synthetic fit failure")
        return _MeanModel(0.0)

    with pytest.raises(RuntimeError, match="success fraction"):
        run_tournament_cluster_refits(
            training,
            probe,
            broken,
            config=TournamentBootstrapConfig(
                replicates=10,
                seed=5,
                minimum_success_fraction=0.9,
            ),
        )
