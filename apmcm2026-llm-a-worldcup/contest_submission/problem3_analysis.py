"""Problem 3: four historical features and official logistic regression."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

try:
    from features import CLASS_CHINESE_BY_INT, CLASS_TO_INT, TeamStateEngine
    from model_selection import (
        evaluate_logistic_fold,
        rolling_validation,
        select_config_on_years,
    )
    from models import P3_FEATURE_NAMES, P3_FINAL_CONFIG
    from visuals import (
        plot_problem3_confusion,
        plot_problem3_feature_correlation,
    )
except ImportError:
    from .features import CLASS_CHINESE_BY_INT, CLASS_TO_INT, TeamStateEngine
    from .model_selection import (
        evaluate_logistic_fold,
        rolling_validation,
        select_config_on_years,
    )
    from .models import P3_FEATURE_NAMES, P3_FINAL_CONFIG
    from .visuals import (
        plot_problem3_confusion,
        plot_problem3_feature_correlation,
    )

OUTCOME_INDICATORS = ("Home Win", "Draw", "Away Win")


DEVELOPMENT_YEARS = [1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010]


def build_training_features(
    matches: pd.DataFrame,
    entity_policy: str = "conservative",
) -> pd.DataFrame:
    """Build one prematch row per match with each Cup frozen as a batch."""
    features = TeamStateEngine(
        form_decay=float(P3_FINAL_CONFIG["form_decay"]),
        entity_policy=entity_policy,
        era_tournaments=5,
    ).transform_history_frozen(matches)
    required = [*P3_FEATURE_NAMES, "Outcome", "Year"]
    missing = [name for name in required if name not in features]
    if missing:
        raise ValueError(f"问题三特征缺失：{', '.join(missing)}")
    if not np.isfinite(features[P3_FEATURE_NAMES].to_numpy(float)).all():
        raise ValueError("问题三四项历史特征中存在非有限值。")
    return features


def _vif_table(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_names:
        others = [name for name in feature_names if name != feature]
        model = LinearRegression().fit(frame[others], frame[feature])
        r2 = float(model.score(frame[others], frame[feature]))
        rows.append(
            {
                "Feature": feature,
                "R2RegressedOnOtherFeatures": r2,
                "VIF": float(1.0 / max(1.0 - r2, 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def _spearman_matrices(frame: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = frame[feature_names].to_numpy(float)
    rho, p_values = stats.spearmanr(matrix, axis=0)
    rho = np.asarray(rho, dtype=float)
    p_values = np.asarray(p_values, dtype=float)
    if rho.ndim == 0:
        rho = np.array([[1.0]], dtype=float)
        p_values = np.array([[0.0]], dtype=float)
    return (
        pd.DataFrame(rho, index=feature_names, columns=feature_names),
        pd.DataFrame(p_values, index=feature_names, columns=feature_names),
    )


def _outcome_indicator_correlation(
    frame: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    indicators = pd.DataFrame(
        {
            name: (frame["Outcome"] == name).astype(float)
            for name in OUTCOME_INDICATORS
        }
    )
    rows = {}
    for feature in feature_names:
        rows[feature] = {
            name: float(frame[feature].corr(indicators[name]))
            for name in OUTCOME_INDICATORS
        }
    return pd.DataFrame.from_dict(rows, orient="index")[list(OUTCOME_INDICATORS)]


def run_problem3(
    matches: pd.DataFrame,
    output_dirs: dict[str, Path],
) -> dict:
    tables_dir = output_dirs["tables"]
    figures_dir = output_dirs["figures"]
    logs_dir = output_dirs["logs"]

    features = build_training_features(matches)
    compact = features[
        ["Year", "HomeTeam", "AwayTeam", *P3_FEATURE_NAMES, "Outcome"]
    ].copy()
    compact["OutcomeCode"] = compact["Outcome"].map(CLASS_TO_INT)
    compact.to_csv(
        tables_dir / "3_1_球队历史四特征.csv",
        index=False,
        encoding="utf-8-sig",
    )

    feature_frame = compact[list(P3_FEATURE_NAMES)].copy()
    spearman_rho, spearman_p = _spearman_matrices(feature_frame, list(P3_FEATURE_NAMES))
    spearman_rho.to_csv(
        tables_dir / "3_1_四特征Spearman相关.csv",
        encoding="utf-8-sig",
    )
    spearman_p.to_csv(
        tables_dir / "3_1_四特征Spearman显著性.csv",
        encoding="utf-8-sig",
    )
    vif = _vif_table(feature_frame, list(P3_FEATURE_NAMES))
    vif.to_csv(
        tables_dir / "3_1_四特征VIF.csv",
        index=False,
        encoding="utf-8-sig",
    )
    outcome_corr = _outcome_indicator_correlation(compact, list(P3_FEATURE_NAMES))
    outcome_corr.to_csv(
        tables_dir / "3_1_四特征与胜负相关性.csv",
        encoding="utf-8-sig",
    )
    plot_problem3_feature_correlation(
        feature_frame,
        spearman_rho,
        spearman_p,
        vif,
        outcome_corr,
        figures_dir / "图5_问题三四特征相关热力图.png",
    )

    selected, selection_table = select_config_on_years(
        features[features["Year"] <= 2010],
        DEVELOPMENT_YEARS,
    )
    selection_table.to_csv(
        tables_dir / "3_2_训练期正则化选择.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if (
        float(selected["C"]) != float(P3_FINAL_CONFIG["C"])
        or selected.get("class_weight") != P3_FINAL_CONFIG["class_weight"]
    ):
        raise RuntimeError(
            "训练期选择结果与冻结配置不一致；请先更新配置再发布。"
        )

    development = rolling_validation(
        features[features["Year"] <= 2010],
        selected,
        DEVELOPMENT_YEARS,
    )
    development.to_csv(
        tables_dir / "3_2_训练期滚动验证.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_row, probabilities = evaluate_logistic_fold(features, 2014, selected)
    test_metrics = pd.DataFrame([test_row])
    test_metrics.insert(0, "TrainingRange", "1930-2010")
    test_metrics.insert(1, "TestRange", "2014")
    test_metrics.to_csv(
        tables_dir / "3_2_2014逻辑回归测试指标.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test = features[features["Year"] == 2014].reset_index(drop=True)
    truth = test["Outcome"].map(CLASS_TO_INT).to_numpy(int)
    prediction = probabilities.argmax(axis=1)
    details = pd.DataFrame(
        {
            "Year": 2014,
            "HomeTeam": test["HomeTeam"],
            "AwayTeam": test["AwayTeam"],
            "ActualCode": truth,
            "Actual": [CLASS_CHINESE_BY_INT[int(value)] for value in truth],
            "HomeWinProbability": probabilities[:, 0],
            "DrawProbability": probabilities[:, 1],
            "AwayWinProbability": probabilities[:, 2],
            "prediction": prediction,
            "Predicted": [
                CLASS_CHINESE_BY_INT[int(value)] for value in prediction
            ],
            "ProbabilityArgmaxMatchesPrediction": (
                probabilities.argmax(axis=1) == prediction
            ),
        }
    )
    details.to_csv(
        tables_dir / "3_2_2014逻辑回归预测明细.csv",
        index=False,
        encoding="utf-8-sig",
    )

    matrix = confusion_matrix(truth, prediction, labels=[0, 1, 2])
    pd.DataFrame(
        matrix,
        index=["真实主胜", "真实平局", "真实客胜"],
        columns=["预测主胜", "预测平局", "预测客胜"],
    ).to_csv(
        tables_dir / "3_2_2014混淆矩阵.csv",
        encoding="utf-8-sig",
    )
    plot_problem3_confusion(
        matrix,
        figures_dir / "图6_2014逻辑回归混淆矩阵.png",
    )

    checks = {
        "official_model_is_logistic": selected["model_name"]
        == "题面四特征多分类逻辑回归",
        "training_ends_at_2010": int(features[features["Year"] < 2014]["Year"].max())
        == 2010,
        "test_is_2014": int(test["Year"].unique()[0]) == 2014,
        "four_required_features": list(selected["feature_names"])
        == list(P3_FEATURE_NAMES),
        "probability_argmax_matches": bool(
            details["ProbabilityArgmaxMatchesPrediction"].all()
        ),
    }
    status = "满足" if all(checks.values()) else "未满足"
    conclusion = (
        f"题面四特征多分类逻辑回归在 2014 测试集正确 "
        f"{test_row['CorrectMatches']}/{test_row['MatchCount']} 场，"
        f"准确率 {test_row['Accuracy']:.3f}，Macro-F1 "
        f"{test_row['MacroF1']:.3f}。训练/测试口径核对结果：{status}。"
    )
    (logs_dir / "3_问题三结论.txt").write_text(conclusion, encoding="utf-8")

    result_row = {
        "Model": selected["model_name"],
        "FeatureCount": len(P3_FEATURE_NAMES),
        "C": selected["C"],
        "ClassWeight": str(selected["class_weight"]),
        "TrainingRange": "1930-2010",
        "TestYear": 2014,
        "Accuracy2014": test_row["Accuracy"],
        "CorrectMatches2014": test_row["CorrectMatches"],
        "MatchCount2014": test_row["MatchCount"],
        "MacroF12014": test_row["MacroF1"],
        "DrawRecall2014": test_row["DrawRecall"],
        "DevelopmentMeanAccuracy": development["Accuracy"].mean(),
        "DevelopmentWorstAccuracy": development["Accuracy"].min(),
        "AllProtocolChecksMet": all(checks.values()),
        "Conclusion": conclusion,
    }
    pd.DataFrame([result_row]).to_csv(
        output_dirs["base"] / "problem3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "features": features,
        "config": dict(selected),
        "selection": selection_table,
        "development": development,
        "test_2014": test_metrics,
        "details_2014": details,
        "confusion_matrix": matrix,
        "correlation": outcome_corr,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "vif": vif,
        "checks": checks,
        "accuracy": float(test_row["Accuracy"]),
        "correct_matches": int(test_row["CorrectMatches"]),
        "match_count": int(test_row["MatchCount"]),
        "macro_f1": float(test_row["MacroF1"]),
        "draw_recall": float(test_row["DrawRecall"]),
        "conclusion": conclusion,
    }
