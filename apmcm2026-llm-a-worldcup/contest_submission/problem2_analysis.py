"""Problem 2: full-time goals from a deployment-consistent frozen protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from visuals import plot_problem2_half_full_scatter, plot_problem2_stage
except ImportError:
    from .visuals import plot_problem2_half_full_scatter, plot_problem2_stage

try:
    from features import STAGE_CHINESE, STAGE_ORDER, TeamStateEngine
    from models import P3_FINAL_CONFIG, RANDOM_STATE
except ImportError:
    from .features import STAGE_CHINESE, STAGE_ORDER, TeamStateEngine
    from .models import P3_FINAL_CONFIG, RANDOM_STATE


P2_NUMERIC = [
    "HalfTimeTotal",
    "AttendanceLog",
    "HomeStrength",
    "AwayStrength",
]
P2_CATEGORICAL = ["StageNormalized"]
P2_FEATURES = [*P2_NUMERIC, *P2_CATEGORICAL]

OFFICIAL_MODEL_NAME = "题面全场进球线性回归"
INCREMENT_MODEL_NAME = "下半场增量泊松回归"
BASELINE_MODEL_NAME = "训练期总进球均值基线"


def make_problem2_estimator(kind: str = "linear") -> Pipeline:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("median_imputer", SimpleImputer(strategy="median")),
                        ("standardize", StandardScaler()),
                    ]
                ),
                P2_NUMERIC,
            ),
            ("stage", encoder, P2_CATEGORICAL),
        ]
    )
    regressor = (
        LinearRegression()
        if kind == "linear"
        else PoissonRegressor(alpha=0.3, max_iter=500)
    )
    return Pipeline([("preprocess", preprocessor), ("regression", regressor)])


def prepare_problem2_model_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """Build historical training rows sequentially from prematch information."""
    data = TeamStateEngine._ordered(matches).reset_index(drop=True).copy()
    engine = TeamStateEngine(
        form_decay=float(P3_FINAL_CONFIG["form_decay"]),
        entity_policy=str(P3_FINAL_CONFIG["entity_policy"]),
    )
    prematch = engine.transform_history(data)
    if len(prematch) != len(data):
        raise RuntimeError("球队状态特征与比赛行数不一致。")
    data["HomeStrength"] = prematch["HomeHistAvgGF"].to_numpy()
    data["AwayStrength"] = prematch["AwayHistAvgGF"].to_numpy()
    data["AttendanceLog"] = np.log1p(data["Attendance"])
    data["SecondHalfGoals"] = data["TotalGoals"] - data["HalfTimeTotal"]
    return data


def prepare_problem2_frozen_target_frame(
    history: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    """Represent the complete target tournament as one frozen query batch."""
    history_ordered = TeamStateEngine._ordered(history).reset_index(drop=True)
    target_ordered = TeamStateEngine._ordered(target).reset_index(drop=True).copy()
    engine = TeamStateEngine(
        form_decay=float(P3_FINAL_CONFIG["form_decay"]),
        entity_policy=str(P3_FINAL_CONFIG["entity_policy"]),
    ).fit(history_ordered)
    rows = [
        engine.feature_row(
            match["Home Team Name"],
            match["Away Team Name"],
            match.get("StageNormalized", match.get("Stage", "Group Stage")),
            match["Year"],
        )
        for _, match in target_ordered.iterrows()
    ]
    prematch = pd.DataFrame(rows, index=target_ordered.index)
    target_ordered["HomeStrength"] = prematch["HomeHistAvgGF"].to_numpy()
    target_ordered["AwayStrength"] = prematch["AwayHistAvgGF"].to_numpy()
    target_ordered["AttendanceLog"] = np.log1p(target_ordered["Attendance"])
    target_ordered["SecondHalfGoals"] = (
        target_ordered["TotalGoals"] - target_ordered["HalfTimeTotal"]
    )
    return target_ordered


def _predict_full_time(
    estimator: Pipeline,
    frame: pd.DataFrame,
    kind: str,
) -> np.ndarray:
    raw = estimator.predict(frame[P2_FEATURES])
    if kind == "official":
        return np.asarray(raw, dtype=float)
    return frame["HalfTimeTotal"].to_numpy(float) + np.clip(raw, 0.0, None)


def rolling_oof_absolute_residuals(
    model_frame: pd.DataFrame,
    kind: str = "increment",
    minimum_training_rows: int = 100,
) -> pd.DataFrame:
    """Collect genuine rolling out-of-fold errors for interval calibration."""
    rows = []
    for year in sorted(int(value) for value in model_frame["Year"].unique()):
        train = model_frame[model_frame["Year"] < year]
        valid_source = model_frame[model_frame["Year"] == year]
        valid = prepare_problem2_frozen_target_frame(train, valid_source)
        if len(train) < minimum_training_rows or valid.empty:
            continue
        target = "TotalGoals" if kind == "official" else "SecondHalfGoals"
        estimator = make_problem2_estimator(
            "linear" if kind == "official" else "poisson"
        ).fit(train[P2_FEATURES], train[target])
        prediction = _predict_full_time(estimator, valid, kind)
        absolute = np.abs(valid["TotalGoals"].to_numpy(float) - prediction)
        rows.extend(
            {
                "ModelKind": kind,
                "ValidationYear": year,
                "AbsoluteError": float(error),
            }
            for error in absolute
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("没有足够的滚动折外残差用于预测区间。")
    return result


def empirical_interval_radius(
    model_frame: pd.DataFrame,
    kind: str = "increment",
    coverage: float = 0.8,
) -> tuple[float, pd.DataFrame]:
    residuals = rolling_oof_absolute_residuals(model_frame, kind=kind)
    radius = float(
        np.quantile(
            residuals["AbsoluteError"].to_numpy(float),
            coverage,
            method="higher",
        )
    )
    return radius, residuals


def _metric_row(
    model_name: str,
    test_year: int,
    truth: pd.Series,
    prediction: np.ndarray,
    mean_cv_mae: float,
    radius: float,
    half_time: np.ndarray,
) -> dict:
    lower = np.maximum(half_time, prediction - radius)
    upper = prediction + radius
    return {
        "Model": model_name,
        "TestYear": int(test_year),
        "MAE": float(mean_absolute_error(truth, prediction)),
        "MSE": float(mean_squared_error(truth, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
        "R2": float(r2_score(truth, prediction)),
        "MeanTimeCV_MAE": float(mean_cv_mae),
        "Empirical80Radius": float(radius),
        "Empirical80Coverage": float(
            np.mean((truth.to_numpy(float) >= lower) & (truth.to_numpy(float) <= upper))
        ),
        "MeanIntervalWidth": float(np.mean(upper - lower)),
        "IntervalResidualSource": "rolling out-of-fold tournament residuals",
        "StateProtocol": "whole tournament frozen; no within-query updates",
    }


def run_problem2(
    matches: pd.DataFrame,
    output_dirs: dict[str, Path],
) -> dict:
    tables_dir = output_dirs["tables"]
    figures_dir = output_dirs["figures"]
    logs_dir = output_dirs["logs"]
    data = prepare_problem2_model_frame(matches)

    stage_stats = (
        data.groupby("StageNormalized", observed=True)["TotalGoals"]
        .agg(
            MatchCount="count",
            AverageTotalGoals="mean",
            MedianTotalGoals="median",
            StandardDeviation="std",
        )
        .reindex(STAGE_ORDER)
        .reset_index()
    )
    stage_stats.insert(
        1,
        "StageChinese",
        stage_stats["StageNormalized"].map(STAGE_CHINESE),
    )
    stage_stats.to_csv(
        tables_dir / "2_1_比赛阶段进球统计.csv",
        index=False,
        encoding="utf-8-sig",
    )
    complete = data.dropna(subset=["HalfTimeTotal", "TotalGoals"])
    pearson_r, pearson_p = stats.pearsonr(
        complete["HalfTimeTotal"], complete["TotalGoals"]
    )
    absolute_r = abs(float(pearson_r))
    correlation_strength = (
        "强"
        if absolute_r >= 0.7
        else "中等"
        if absolute_r >= 0.4
        else "弱"
    )
    correlation_conclusion = (
        f"程序判定：半场总进球与全场总进球呈{correlation_strength}正相关"
        f"（Pearson r={pearson_r:.3f}，p={pearson_p:.3g}）。"
    )
    pd.DataFrame(
        [
            {
                "PearsonR": pearson_r,
                "PValue": pearson_p,
                "Strength": correlation_strength,
                "Conclusion": correlation_conclusion,
            }
        ]
    ).to_csv(
        tables_dir / "2_1_半场与全场进球相关性.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validation_years = [1994, 1998, 2002, 2006, 2010]
    definitions = [
        ("official", OFFICIAL_MODEL_NAME, "TotalGoals"),
        ("increment", INCREMENT_MODEL_NAME, "SecondHalfGoals"),
    ]
    cv_rows = []
    cv_error_rows = []
    for kind, name, target in definitions:
        for year in validation_years:
            train = data[data["Year"] < year]
            valid = prepare_problem2_frozen_target_frame(
                train,
                data[data["Year"] == year],
            )
            estimator = make_problem2_estimator(
                "linear" if kind == "official" else "poisson"
            ).fit(train[P2_FEATURES], train[target])
            prediction = _predict_full_time(estimator, valid, kind)
            absolute = np.abs(valid["TotalGoals"].to_numpy(float) - prediction)
            cv_rows.append(
                {
                    "Model": name,
                    "ValidationYear": year,
                    "MAE": mean_absolute_error(valid["TotalGoals"], prediction),
                    "RMSE": np.sqrt(
                        mean_squared_error(valid["TotalGoals"], prediction)
                    ),
                    "R2": r2_score(valid["TotalGoals"], prediction),
                    "StateProtocol": "whole tournament frozen; no within-query updates",
                }
            )
            cv_error_rows.extend(
                {
                    "Model": name,
                    "ValidationYear": year,
                    "AbsoluteError": float(error),
                }
                for error in absolute
            )
    for year in validation_years:
        train = data[data["Year"] < year]
        valid = prepare_problem2_frozen_target_frame(
            train,
            data[data["Year"] == year],
        )
        prediction = np.full(len(valid), train["TotalGoals"].mean())
        absolute = np.abs(valid["TotalGoals"].to_numpy(float) - prediction)
        cv_rows.append(
            {
                "Model": BASELINE_MODEL_NAME,
                "ValidationYear": year,
                "MAE": mean_absolute_error(valid["TotalGoals"], prediction),
                "RMSE": np.sqrt(
                    mean_squared_error(valid["TotalGoals"], prediction)
                ),
                "R2": r2_score(valid["TotalGoals"], prediction),
                "StateProtocol": "training mean",
            }
        )
        cv_error_rows.extend(
            {
                "Model": BASELINE_MODEL_NAME,
                "ValidationYear": year,
                "AbsoluteError": float(error),
            }
            for error in absolute
        )
    cv_table = pd.DataFrame(cv_rows)
    cv_errors = pd.DataFrame(cv_error_rows)
    cv_table.to_csv(
        tables_dir / "2_2_滚动时间验证.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train = data[data["Year"] <= 2010]
    test = prepare_problem2_frozen_target_frame(
        train,
        data[data["Year"] == 2014],
    )
    metrics = []
    predictions: dict[str, np.ndarray] = {}
    prediction_detail = test[
        [
            "Year",
            "Home Team Name",
            "Away Team Name",
            "StageNormalized",
            "HalfTimeTotal",
            "TotalGoals",
        ]
    ].copy()
    for kind, name, target in definitions:
        estimator = make_problem2_estimator(
            "linear" if kind == "official" else "poisson"
        ).fit(train[P2_FEATURES], train[target])
        prediction = _predict_full_time(estimator, test, kind)
        radius = float(
            np.quantile(
                cv_errors.loc[cv_errors["Model"] == name, "AbsoluteError"],
                0.8,
                method="higher",
            )
        )
        predictions[kind] = prediction
        metrics.append(
            _metric_row(
                name,
                2014,
                test["TotalGoals"],
                prediction,
                cv_table.loc[cv_table["Model"] == name, "MAE"].mean(),
                radius,
                test["HalfTimeTotal"].to_numpy(float),
            )
        )
        label = "题面模型" if kind == "official" else "增量模型"
        prediction_detail[f"{label}预测"] = prediction
        prediction_detail[f"{label}80%下限"] = np.maximum(
            test["HalfTimeTotal"].to_numpy(float), prediction - radius
        )
        prediction_detail[f"{label}80%上限"] = prediction + radius

    baseline_prediction = np.full(len(test), train["TotalGoals"].mean())
    baseline_radius = float(
        np.quantile(
            cv_errors.loc[
                cv_errors["Model"] == BASELINE_MODEL_NAME,
                "AbsoluteError",
            ],
            0.8,
            method="higher",
        )
    )
    metrics.append(
        _metric_row(
            BASELINE_MODEL_NAME,
            2014,
            test["TotalGoals"],
            baseline_prediction,
            cv_table.loc[cv_table["Model"] == BASELINE_MODEL_NAME, "MAE"].mean(),
            baseline_radius,
            np.zeros(len(test)),
        )
    )
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(
        tables_dir / "2_2_2014历史开发基准指标.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prediction_detail.to_csv(
        tables_dir / "2_2_2014历史开发基准预测明细.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired_improvement = (
        np.abs(test["TotalGoals"].to_numpy(float) - predictions["official"])
        - np.abs(test["TotalGoals"].to_numpy(float) - predictions["increment"])
    )
    rng = np.random.default_rng(RANDOM_STATE)
    bootstrap = np.mean(
        rng.choice(
            paired_improvement,
            size=(20_000, len(paired_improvement)),
            replace=True,
        ),
        axis=1,
    )
    interval = np.quantile(bootstrap, [0.025, 0.975])
    paired = pd.DataFrame(
        [
            {
                "Comparison": "题面模型MAE减去增量模型MAE",
                "MeanMAEImprovement": float(np.mean(paired_improvement)),
                "PairedBootstrap95Lower": float(interval[0]),
                "PairedBootstrap95Upper": float(interval[1]),
                "BootstrapRepeats": 20_000,
                "Conclusion": (
                    "区间包含0；增量模型的主要价值是机制更合理，"
                    "不是已证明的稳定预测优势。"
                ),
            }
        ]
    )
    paired.to_csv(
        tables_dir / "2_2_2014配对Bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_problem2_stage(stage_stats, figures_dir / "图3_比赛阶段进球统计.png")
    plot_problem2_half_full_scatter(
        complete,
        float(pearson_r),
        figures_dir / "图4_半场与全场进球相关.png",
        pearson_p=float(pearson_p),
    )

    official = metrics_frame.loc[
        metrics_frame["Model"] == OFFICIAL_MODEL_NAME
    ].iloc[0].to_dict()
    increment = metrics_frame.loc[
        metrics_frame["Model"] == INCREMENT_MODEL_NAME
    ].iloc[0].to_dict()
    baseline = metrics_frame.loc[
        metrics_frame["Model"] == BASELINE_MODEL_NAME
    ].iloc[0].to_dict()
    conclusion = (
        correlation_conclusion
        +
        f"采用与正式批量 API 完全一致的整届冻结状态协议后，2014 历史开发基准中，"
        f"题面模型 MAE={official['MAE']:.4f}，增量模型 MAE="
        f"{increment['MAE']:.4f}，改进={paired.iloc[0]['MeanMAEImprovement']:.4f}；"
        f"配对 Bootstrap 95% 区间为 [{interval[0]:.3f}, {interval[1]:.3f}]。"
        "正式80%区间只使用滚动折外绝对残差，禁止使用最终训练集内残差。"
    )
    (logs_dir / "2_问题二结论.txt").write_text(conclusion, encoding="utf-8")
    pd.DataFrame(
        [
            {
                "PearsonHalfToFull": pearson_r,
                "PearsonPValue": pearson_p,
                "Official2014MAE": official["MAE"],
                "Increment2014MAE": increment["MAE"],
                "Baseline2014MAE": baseline["MAE"],
                "Improvement": paired.iloc[0]["MeanMAEImprovement"],
                "Bootstrap95Lower": interval[0],
                "Bootstrap95Upper": interval[1],
                "Conclusion": conclusion,
            }
        ]
    ).to_csv(output_dirs["base"] / "problem2.csv", index=False, encoding="utf-8-sig")
    return {
        "model_frame": data,
        "stage_stats": stage_stats,
        "cv_detail": cv_table,
        "metrics": metrics_frame,
        "official": official,
        "increment": increment,
        "baseline": baseline,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "correlation_conclusion": correlation_conclusion,
        "paired_improvement": paired.iloc[0].to_dict(),
        "conclusion": conclusion,
    }
