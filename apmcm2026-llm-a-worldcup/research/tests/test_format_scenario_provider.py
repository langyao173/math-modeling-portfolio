"""Tests for a lawful, explicitly illustrative 48-team score provider."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from worldcup_forecasting.models.count import JointGoalBatch, score_matrix
from worldcup_forecasting.simulation.scenario import build_format_scenario_provider


class _DummyModel:
    home_team_column_ = "home_team"
    away_team_column_ = "away_team"
    paired_columns_ = (("home_strength", "away_strength"),)
    odd_columns_ = ("strength_diff",)
    even_columns_ = ("tournament_teams", "tournament_matches", "is_knockout")
    context_columns_ = ("format_regime", "canonical_stage")
    host_side_column_ = "host_side"
    stage_column_ = "canonical_stage"
    year_column_ = "Year"
    config = SimpleNamespace(max_score=7)

    def predict_joint(self, frame: pd.DataFrame) -> JointGoalBatch:
        matrices = []
        for row in frame.itertuples(index=False):
            difference = float(row.strength_diff)
            matrices.append(
                score_matrix(
                    np.exp(0.2 + 0.1 * difference),
                    np.exp(0.2 - 0.1 * difference),
                    max_score=7,
                    adaptive=False,
                )
            )
        probability = np.asarray(matrices)
        score = np.arange(probability.shape[1], dtype=float)
        return JointGoalBatch(
            probability=probability,
            support_max=probability.shape[1] - 1,
            retained_mass=np.ones(len(frame)),
            expected_home_goals=np.sum(probability * score[None, :, None], axis=(1, 2)),
            expected_away_goals=np.sum(probability * score[None, None, :], axis=(1, 2)),
            model_metadata={"name": "dummy"},
        )


def _frame() -> pd.DataFrame:
    rows = []
    teams = [f"T{index:02d}" for index in range(52)]
    for index, team in enumerate(teams):
        opponent = teams[(index + 1) % len(teams)]
        rows.append(
            {
                "MatchID": index + 1,
                "Datetime": pd.Timestamp(2010, 6, (index % 20) + 1),
                "Year": 2010,
                "home_team": team,
                "away_team": opponent,
                "home_strength": float(index),
                "away_strength": float(index + 1),
                "strength_diff": -1.0,
                "tournament_teams": 32.0,
                "tournament_matches": 64.0,
                "is_knockout": 0.0,
                "format_regime": "legacy_32",
                "canonical_stage": "group_stage",
                "host_side": "neutral",
            }
        )
    return pd.DataFrame(rows)


def test_format_provider_uses_pre_2014_profiles_and_exact_2026_context() -> None:
    frame = _frame()
    contaminated = pd.concat(
        [
            frame,
            frame.iloc[[0]].assign(
                MatchID=99_999,
                Year=2014,
                Datetime=pd.Timestamp(2014, 7, 1),
                home_strength=99_999.0,
            ),
        ],
        ignore_index=True,
    )

    bundle = build_format_scenario_provider(_DummyModel(), contaminated)

    assert len(bundle.teams) == 48
    assert bundle.metadata["profile_cutoff_year"] == 2010
    assert bundle.metadata["scenario_status"].startswith("illustrative")
    assert bundle.metadata["tournament_teams"] == 48
    assert bundle.metadata["tournament_matches"] == 104
    assert bundle.metadata["bracket_policy"] == "fifa_2026_annex_c"
    assert len(bundle.metadata["format_source_sha256"]) == 64
    assert len(bundle.metadata["annex_c_csv_sha256"]) == 64
    assert bundle.profile_table["profile_max_source_year"].max() == 2010
    direct = bundle.score_provider(bundle.teams[0], bundle.teams[1], "group_stage")
    reverse = bundle.score_provider(bundle.teams[1], bundle.teams[0], "group_stage")
    np.testing.assert_allclose(direct, reverse.T, atol=1e-12)
    assert np.isclose(direct.sum(), 1.0, atol=1e-12)


def test_format_provider_is_deterministic_and_records_fallbacks() -> None:
    first = build_format_scenario_provider(_DummyModel(), _frame())
    second = build_format_scenario_provider(_DummyModel(), _frame())

    assert first.teams == second.teams
    pd.testing.assert_frame_equal(first.profile_table, second.profile_table)
    np.testing.assert_allclose(
        first.score_provider(first.teams[3], first.teams[7], "quarter_final"),
        second.score_provider(second.teams[3], second.teams[7], "quarter_final"),
        atol=0.0,
        rtol=0.0,
    )


def test_format_provider_can_batch_precompute_exact_symmetric_grids() -> None:
    bundle = build_format_scenario_provider(
        _DummyModel(),
        _frame(),
        precompute_stages=("group_stage",),
    )

    assert bundle.metadata["precomputed_stages"] == ["group_stage"]
    assert bundle.metadata["precomputed_ordered_score_grids"] == 48 * 47
    direct = bundle.score_provider(bundle.teams[2], bundle.teams[17], "group_stage")
    reverse = bundle.score_provider(bundle.teams[17], bundle.teams[2], "group_stage")
    np.testing.assert_allclose(direct, reverse.T, atol=0.0, rtol=0.0)
    assert direct.flags.writeable is False
