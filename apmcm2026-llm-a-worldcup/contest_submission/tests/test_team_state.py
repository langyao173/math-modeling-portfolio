from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_io import load_competition_data
from features import (
    CLASS_TO_INT,
    TeamStateEngine,
    clean_team_name,
    normalize_stage,
    score_outcome,
)


class TeamStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matches = load_competition_data(
            Path(__file__).resolve().parents[1] / "data"
        )[1]

    def test_frozen_view_has_one_row_per_match(self):
        sample = self.matches.head(8).copy()
        frozen = TeamStateEngine().transform_history_frozen(sample)
        self.assertEqual(len(frozen), len(sample))

    def test_domain_and_entity_rules(self):
        self.assertEqual(clean_team_name("Germany FR"), "Germany")
        self.assertEqual(clean_team_name("德国"), "Germany")
        self.assertEqual(clean_team_name("Soviet Union"), "Soviet Union")
        self.assertEqual(
            clean_team_name("Soviet Union", policy="legacy_merge"),
            "Russia",
        )
        self.assertEqual(normalize_stage("四分之一决赛"), "Quarter-finals")
        self.assertEqual(score_outcome(1, 1), "Draw")
        self.assertEqual(CLASS_TO_INT, {"Home Win": 0, "Draw": 1, "Away Win": 2})

    def test_first_match_uses_prior_only_and_elo_conserves(self):
        first = self.matches.iloc[[0]]
        engine = TeamStateEngine()
        features = engine.transform_history(first)
        self.assertEqual(float(features.iloc[0]["ExperienceDiff"]), 0.0)
        teams = [
            first.iloc[0]["Home Team Name"],
            first.iloc[0]["Away Team Name"],
        ]
        self.assertAlmostEqual(
            sum(engine.states[name].elo for name in teams),
            3000.0,
            places=7,
        )

    def test_unknown_team_is_finite(self):
        values = TeamStateEngine().feature_row("未知球队甲", "未知球队乙", None, 2026)
        self.assertTrue(np.isfinite(list(values.values())).all())

    def test_frozen_tournament_does_not_update_within_batch(self):
        sample = pd.DataFrame(
            [
                {
                    "Year": 2026,
                    "Stage": "Group Stage",
                    "Home Team Name": "A",
                    "Away Team Name": "B",
                    "Home Team Goals": 1,
                    "Away Team Goals": 0,
                    "ScoreOutcome": "Home Win",
                },
                {
                    "Year": 2026,
                    "Stage": "Group Stage",
                    "Home Team Name": "A",
                    "Away Team Name": "C",
                    "Home Team Goals": 1,
                    "Away Team Goals": 1,
                    "ScoreOutcome": "Draw",
                },
            ]
        )
        features = TeamStateEngine().transform_history_frozen(sample)
        self.assertEqual(features.loc[0, "HomeExperience"], 0.0)
        self.assertEqual(features.loc[1, "HomeExperience"], 0.0)

    def test_future_match_does_not_change_past_features(self):
        base = self.matches.iloc[:80].copy()
        future = self.matches.iloc[[100]].copy()
        future["Year"] = 2030
        before = TeamStateEngine().transform_history_frozen(base)
        after = TeamStateEngine().transform_history_frozen(
            pd.concat([base, future], ignore_index=True)
        ).iloc[: len(base)]
        numeric = [
            name
            for name in before.columns
            if name not in {"HomeTeam", "AwayTeam", "Outcome"}
        ]
        self.assertTrue(
            np.allclose(
                before[numeric].to_numpy(float),
                after[numeric].to_numpy(float),
            )
        )


if __name__ == "__main__":
    unittest.main()
