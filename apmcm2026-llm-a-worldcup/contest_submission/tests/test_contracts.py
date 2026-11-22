from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from contracts.input_schema import discover_query_frame, normalize_query_frame
from contracts.output_schema import build_minimal_submission, write_atomic_submission
from data_io import read_csv_flexible


class InputOutputContractTests(unittest.TestCase):
    def test_chinese_columns_names_and_missing_stage(self):
        query = normalize_query_frame(
            pd.DataFrame(
                {
                    "编号": [2, 1],
                    "主队": ["德国", "Brazil"],
                    "客队": ["France", "Argentina"],
                }
            ),
            "query.csv",
        )
        self.assertEqual(query["__HomeTeam__"].tolist(), ["Germany", "Brazil"])
        self.assertEqual(
            query["__Stage__"].tolist(),
            ["Group Stage", "Group Stage"],
        )

    def test_duplicate_id_fails(self):
        with self.assertRaisesRegex(ValueError, "重复"):
            normalize_query_frame(
                pd.DataFrame(
                    {
                        "id": [1, 1],
                        "HomeTeam": ["A", "B"],
                        "AwayTeam": ["C", "D"],
                    }
                ),
                "query.csv",
            )

    def test_all_empty_id_is_treated_as_absent(self):
        query = normalize_query_frame(
            pd.DataFrame(
                {
                    "MatchID": [np.nan, np.nan],
                    "HomeTeam": ["A", "B"],
                    "AwayTeam": ["C", "D"],
                }
            ),
            "WorldCupMatches.csv 中的无标签行",
        )
        self.assertNotIn("MatchID", query.columns)
        rich = pd.DataFrame({"prediction": [0, 2]})
        self.assertEqual(
            build_minimal_submission(query, rich).columns.tolist(),
            ["prediction"],
        )

    def test_partially_empty_id_fails(self):
        with self.assertRaisesRegex(ValueError, "部分为空"):
            normalize_query_frame(
                pd.DataFrame(
                    {
                        "MatchID": [101, np.nan],
                        "HomeTeam": ["A", "B"],
                        "AwayTeam": ["C", "D"],
                    }
                ),
                "query.csv",
            )

    def test_empty_query_fails(self):
        with self.assertRaisesRegex(ValueError, "为空"):
            normalize_query_frame(
                pd.DataFrame(columns=["HomeTeam", "AwayTeam"]),
                "query.csv",
            )

    def test_absent_and_conflicting_query_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "未发现待预测数据"):
                discover_query_frame(root, pd.DataFrame(), read_csv_flexible)
            pd.DataFrame(
                {"HomeTeam": ["A"], "AwayTeam": ["B"]}
            ).to_csv(root / "query.csv", index=False)
            with self.assertRaisesRegex(ValueError, "同时发现"):
                discover_query_frame(
                    root,
                    pd.DataFrame({"HomeTeam": ["C"], "AwayTeam": ["D"]}),
                    read_csv_flexible,
                )

    def test_minimal_protocol_and_atomic_write(self):
        query = pd.DataFrame({"id": [7, 5]})
        rich = pd.DataFrame(
            {
                "prediction": [0, 2],
                "主队胜概率": [0.7, 0.1],
                "平局概率": [0.2, 0.2],
                "客队胜概率": [0.1, 0.7],
            }
        )
        minimal = build_minimal_submission(query, rich)
        self.assertEqual(
            minimal.to_dict("list"),
            {"id": [7, 5], "prediction": [0, 2]},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_atomic_submission(
                minimal,
                Path(directory) / "cla_pre.csv",
            )
            self.assertTrue(path.is_file())

    def test_probability_or_argmax_mismatch_fails(self):
        query = pd.DataFrame({"id": [1]})
        bad_sum = pd.DataFrame(
            {
                "prediction": [0],
                "主队胜概率": [0.9],
                "平局概率": [0.9],
                "客队胜概率": [0.1],
            }
        )
        with self.assertRaisesRegex(ValueError, "概率"):
            build_minimal_submission(query, bad_sum)
        bad_argmax = pd.DataFrame(
            {
                "prediction": [2],
                "主队胜概率": [0.6],
                "平局概率": [0.2],
                "客队胜概率": [0.2],
            }
        )
        with self.assertRaisesRegex(ValueError, "argmax"):
            build_minimal_submission(query, bad_argmax)

    def test_unwritable_shape_reports_output_error(self):
        minimal = pd.DataFrame({"prediction": [1]})
        with tempfile.TemporaryDirectory() as directory:
            blocker = Path(directory) / "不是目录"
            blocker.write_text("占位", encoding="utf-8")
            with self.assertRaisesRegex(OSError, "无法写入输出文件"):
                write_atomic_submission(minimal, blocker / "cla_pre.csv")


if __name__ == "__main__":
    unittest.main()
