from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from contracts.input_schema import normalize_query_frame
from contracts.output_schema import build_minimal_submission, write_atomic_submission
from data_io import derive_match_fields
from features import score_outcome
from model_selection import (
    candidate_configs,
    expected_calibration_error,
    multiclass_brier_score,
)


class ProtocolEdgeCaseTests(unittest.TestCase):
    def test_missing_home_or_away_column_fails(self):
        with self.assertRaisesRegex(ValueError, "主队列和客队列"):
            normalize_query_frame(pd.DataFrame({"HomeTeam": ["A"]}), "query.csv")

    def test_empty_team_name_fails(self):
        with self.assertRaisesRegex(ValueError, "空值"):
            normalize_query_frame(
                pd.DataFrame({"HomeTeam": [""], "AwayTeam": ["B"]}),
                "query.csv",
            )

    def test_invalid_year_type_fails(self):
        with self.assertRaisesRegex(ValueError, "无法解析"):
            normalize_query_frame(
                pd.DataFrame(
                    {"HomeTeam": ["A"], "AwayTeam": ["B"], "Year": ["二零二六"]}
                ),
                "query.csv",
            )

    def test_invalid_optional_target_type_fails(self):
        with self.assertRaisesRegex(ValueError, "非数值"):
            normalize_query_frame(
                pd.DataFrame(
                    {
                        "HomeTeam": ["A"],
                        "AwayTeam": ["B"],
                        "Home Team Goals": ["bad"],
                    }
                ),
                "query.csv",
            )

    def test_all_present_ids_keep_input_order(self):
        query = normalize_query_frame(
            pd.DataFrame(
                {
                    "id": ["z", "a", "m"],
                    "HomeTeam": ["A", "B", "C"],
                    "AwayTeam": ["D", "E", "F"],
                }
            ),
            "query.csv",
        )
        rich = pd.DataFrame({"prediction": [2, 0, 1]})
        submission = build_minimal_submission(query, rich)
        self.assertEqual(submission["id"].tolist(), ["z", "a", "m"])

    def test_output_row_count_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "行数"):
            build_minimal_submission(
                pd.DataFrame({"id": [1, 2]}),
                pd.DataFrame({"prediction": [0]}),
            )

    def test_missing_prediction_fails(self):
        with self.assertRaises((KeyError, ValueError)):
            build_minimal_submission(
                pd.DataFrame({"id": [1]}),
                pd.DataFrame({"prediction": [np.nan]}),
            )

    def test_prediction_outside_domain_fails(self):
        with self.assertRaisesRegex(ValueError, "0、1、2"):
            build_minimal_submission(
                pd.DataFrame({"id": [1]}),
                pd.DataFrame({"prediction": [3]}),
            )

    def test_probability_error_above_one_e_minus_ten_fails(self):
        rich = pd.DataFrame(
            {
                "prediction": [0],
                "主队胜概率": [0.6],
                "平局概率": [0.2],
                "客队胜概率": [0.200000001],
            }
        )
        with self.assertRaisesRegex(ValueError, "概率"):
            build_minimal_submission(pd.DataFrame({"id": [1]}), rich)

    def test_penalty_metadata_does_not_rewrite_full_time_draw(self):
        raw = pd.DataFrame(
            {
                "Year": [2014],
                "Home Team Name": ["A"],
                "Away Team Name": ["B"],
                "Home Team Goals": [1],
                "Away Team Goals": [1],
                "Win conditions": ["A win on penalties"],
            }
        )
        derived = derive_match_fields(raw)
        self.assertEqual(derived.loc[0, "ScoreOutcome"], "Draw")

    def test_full_time_label_definitions(self):
        self.assertEqual(score_outcome(2, 1), "Home Win")
        self.assertEqual(score_outcome(1, 1), "Draw")
        self.assertEqual(score_outcome(0, 1), "Away Win")

    def test_atomic_failure_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            blocker = Path(directory) / "blocker"
            blocker.write_text("not a directory", encoding="utf-8")
            with self.assertRaises(OSError):
                write_atomic_submission(
                    pd.DataFrame({"prediction": [1]}),
                    blocker / "cla_pre.csv",
                )
            self.assertFalse((Path(directory) / "cla_pre.tmp.csv").exists())

    def test_candidate_grid_never_mentions_2014(self):
        for config in candidate_configs():
            self.assertNotIn("2014", repr(config))

    def test_probability_quality_metrics_have_expected_bounds(self):
        truth = np.array([0, 1, 2])
        probabilities = np.eye(3) * 0.8 + (1 - np.eye(3)) * 0.1
        self.assertGreaterEqual(multiclass_brier_score(truth, probabilities), 0)
        self.assertLessEqual(expected_calibration_error(truth, probabilities), 1)


if __name__ == "__main__":
    unittest.main()
