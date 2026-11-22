from __future__ import annotations

import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_io import load_competition_data
from models import MulticlassLogisticModel, P3_FEATURE_NAMES, P3_FINAL_CONFIG
from prediction_api import MatchGoalPredictor, MatchResultPredictor


PROBABILITY_COLUMNS = ["主队胜概率", "平局概率", "客队胜概率"]


class PredictionInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matches = load_competition_data(
            Path(__file__).resolve().parents[1] / "data"
        )[1]
        cls.predictor = MatchResultPredictor(
            cls.matches,
            2026,
            P3_FINAL_CONFIG,
        )
        cls.query = pd.DataFrame(
            {
                "id": [20, 10],
                "Year": [2026, 2026],
                "Stage": ["Group Stage", "Round of 16"],
                "Home Team Name": ["Brazil", "Germany"],
                "Away Team Name": ["Argentina", "France"],
            }
        )

    def test_repeated_query_is_identical(self):
        first = self.predictor.predict_dataframe(self.query)
        second = self.predictor.predict_dataframe(self.query)
        self.assertTrue(
            np.allclose(
                first[PROBABILITY_COLUMNS],
                second[PROBABILITY_COLUMNS],
            )
        )

    def test_reordering_query_reorders_output(self):
        original = self.predictor.predict_dataframe(self.query)
        reversed_result = self.predictor.predict_dataframe(
            self.query.iloc[::-1].reset_index(drop=True)
        )
        self.assertEqual(
            original["预测结果"].tolist()[::-1],
            reversed_result["预测结果"].tolist(),
        )

    def test_output_row_count_equals_query_count(self):
        result = self.predictor.predict_dataframe(self.query)
        self.assertEqual(len(result), len(self.query))

    def test_duplicate_matchups_do_not_update_state_inside_query(self):
        repeated = pd.concat(
            [self.query.iloc[[0]], self.query.iloc[[0]]],
            ignore_index=True,
        ).drop(columns=["id"])
        result = self.predictor.predict_dataframe(repeated)
        self.assertTrue(
            np.allclose(
                result.loc[0, PROBABILITY_COLUMNS].to_numpy(float),
                result.loc[1, PROBABILITY_COLUMNS].to_numpy(float),
            )
        )

    def test_swapping_teams_swaps_home_away_probabilities(self):
        one = self.query.iloc[[0]].copy()
        swapped = one.copy()
        swapped[["Home Team Name", "Away Team Name"]] = one[
            ["Away Team Name", "Home Team Name"]
        ].to_numpy()
        first = self.predictor.predict_dataframe(one).iloc[0]
        second = self.predictor.predict_dataframe(swapped).iloc[0]
        self.assertAlmostEqual(
            first["主队胜概率"],
            second["客队胜概率"],
            places=7,
        )
        self.assertAlmostEqual(
            first["平局概率"],
            second["平局概率"],
            places=7,
        )

    def test_probability_and_hard_class_are_one_model(self):
        result = self.predictor.predict_dataframe(self.query)
        probabilities = result[PROBABILITY_COLUMNS].to_numpy(float)
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
        self.assertTrue(
            np.array_equal(
                probabilities.argmax(axis=1),
                result["prediction"].to_numpy(int),
            )
        )
        self.assertTrue(result["硬分类与概率一致"].all())
        self.assertTrue((result["平局概率"] > 0).all())

    def test_chinese_team_alias_and_unknown_warning(self):
        known = pd.DataFrame({"主队": ["德国"], "客队": ["法国"]})
        known_result = self.predictor.predict_dataframe(known)
        self.assertEqual(known_result.loc[0, "主队"], "Germany")
        self.assertEqual(known_result.loc[0, "球队实体警告"], "")

        unknown = pd.DataFrame({"主队": ["未知球队甲"], "客队": ["France"]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            unknown_result = self.predictor.predict_dataframe(unknown)
        self.assertTrue(caught)
        self.assertIn("未知球队", unknown_result.loc[0, "球队实体警告"])

    def test_training_max_year_is_actual_history_max(self):
        result = self.predictor.predict_dataframe(self.query)
        self.assertTrue((result["训练数据实际最大年份"] == 2014).all())

    def test_preprocessing_statistics_use_training_rows_only(self):
        names = self.predictor.config["feature_names"]
        matrix = self.predictor.training_features[names].to_numpy(float)
        expected = np.vstack([matrix, -matrix]).mean(axis=0)
        self.assertTrue(np.allclose(self.predictor.model.scaler.mean_, expected))

    def test_problem3_uses_only_official_logistic_model(self):
        self.assertIsInstance(self.predictor.model, MulticlassLogisticModel)
        self.assertEqual(self.predictor.config["feature_names"], P3_FEATURE_NAMES)
        self.assertEqual(len(P3_FEATURE_NAMES), 4)
        self.assertIn("逻辑回归", self.predictor.config["model_name"])

    def test_2014_uses_1930_to_2010_training_only(self):
        predictor = MatchResultPredictor(self.matches, 2014, P3_FINAL_CONFIG)
        test = self.matches[self.matches["Year"] == 2014].copy()
        query = test[["Year", "Stage", "Home Team Name", "Away Team Name"]]
        result = predictor.predict_dataframe(query)
        self.assertEqual(len(result), 64)
        self.assertTrue(result["硬分类与概率一致"].all())
        self.assertEqual(predictor.training_max_year, 2010)
        self.assertTrue((predictor.history["Year"] < 2014).all())

    def test_goal_interval_uses_multiple_oof_years(self):
        predictor = MatchGoalPredictor(self.matches, 2026)
        self.assertGreater(
            predictor.oof_residuals["ValidationYear"].nunique(),
            1,
        )
        self.assertEqual(predictor.training_max_year, 2014)
        self.assertGreater(predictor.empirical_interval_radius, 0)


if __name__ == "__main__":
    unittest.main()
