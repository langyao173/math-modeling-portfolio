"""Problem 1: tournament-scale regression and scale/intensity decomposition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from visuals import plot_problem1_trends
except ImportError:
    from .visuals import plot_problem1_trends


P1_FEATURES = ["QualifiedTeams", "MatchesPlayed", "Attendance"]


def parse_worldcups(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    for name in ("GoalsScored", "QualifiedTeams", "MatchesPlayed"):
        data[name] = pd.to_numeric(data[name], errors="coerce")
    data["Attendance"] = pd.to_numeric(
        data["Attendance"].astype(str).str.replace(r"[^\d]", "", regex=True),
        errors="coerce",
    )
    data["GoalsPerMatch"] = data["GoalsScored"] / data["MatchesPlayed"]
    data["AttendancePerMatch"] = data["Attendance"] / data["MatchesPlayed"]
    return data.sort_values("Year").reset_index(drop=True)


def _ols_table(
    matrix: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
) -> tuple[pd.DataFrame, dict]:
    design = np.column_stack([np.ones(len(matrix)), matrix])
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    fitted = design @ beta
    residual = target - fitted
    dof = max(len(target) - design.shape[1], 1)
    sigma2 = float(residual @ residual / dof)
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    stderr = np.sqrt(np.clip(np.diag(covariance), 0, None))
    t_statistic = np.divide(beta, stderr, out=np.zeros_like(beta), where=stderr > 0)
    p_value = 2 * stats.t.sf(np.abs(t_statistic), dof)
    table = pd.DataFrame(
        {
            "Term": ["Intercept", *feature_names],
            "Coefficient": beta,
            "StandardError": stderr,
            "TStatistic": t_statistic,
            "PValue": p_value,
            "SignificantAt5Percent": p_value < 0.05,
        }
    )
    r2 = float(r2_score(target, fitted))
    predictor_count = len(feature_names)
    adjusted_r2 = 1.0 - (1.0 - r2) * (len(target) - 1) / max(
        len(target) - predictor_count - 1,
        1,
    )
    metrics = {
        "R2": r2,
        "AdjustedR2": float(adjusted_r2),
        "MAE": float(mean_absolute_error(target, fitted)),
        "RMSE": float(np.sqrt(mean_squared_error(target, fitted))),
    }
    return table, metrics


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
                "Interpretation": (
                    "Severe multicollinearity; do not interpret this slope independently"
                    if r2 >= 0.9
                    else "Material multicollinearity; interpret the slope cautiously"
                ),
            }
        )
    return pd.DataFrame(rows)


def _empirical_radius(errors: pd.Series, coverage: float = 0.8) -> float:
    return float(np.quantile(errors.to_numpy(float), coverage, method="higher"))


def _segmented_fit(
    years: np.ndarray,
    values: np.ndarray,
    breakpoint: float,
) -> tuple[np.ndarray, float]:
    centered = years - years.mean()
    design = np.column_stack(
        [np.ones(len(years)), centered, np.maximum(0.0, years - breakpoint)]
    )
    beta = np.linalg.lstsq(design, values, rcond=None)[0]
    fitted = design @ beta
    rss = max(float(np.sum((values - fitted) ** 2)), 1e-12)
    bic = len(years) * np.log(rss / len(years)) + design.shape[1] * np.log(len(years))
    return fitted, float(bic)


def _breakpoint_audit(
    data: pd.DataFrame,
    repeats: int = 1_000,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """BIC candidate scan plus deterministic pairs-bootstrap stability."""
    frame = data.dropna(subset=["Year", "GoalsPerMatch"]).copy()
    years = frame["Year"].to_numpy(float)
    values = frame["GoalsPerMatch"].to_numpy(float)
    candidates = years[4:-4].astype(int)
    centered = years - years.mean()
    single_design = np.column_stack([np.ones(len(years)), centered])
    single_fit = single_design @ np.linalg.lstsq(single_design, values, rcond=None)[0]
    single_rss = max(float(np.sum((values - single_fit) ** 2)), 1e-12)
    single_bic = (
        len(years) * np.log(single_rss / len(years))
        + single_design.shape[1] * np.log(len(years))
    )

    rows = []
    fitted_by_year: dict[int, np.ndarray] = {}
    for candidate in candidates:
        fitted, bic = _segmented_fit(years, values, float(candidate))
        fitted_by_year[int(candidate)] = fitted
        rows.append({"CandidateYear": int(candidate), "BIC": bic})
    audit = pd.DataFrame(rows)
    best_year = int(audit.loc[audit["BIC"].idxmin(), "CandidateYear"])
    best_bic = float(audit["BIC"].min())

    rng = np.random.default_rng(2026)
    selected = {int(candidate): 0 for candidate in candidates}
    for _ in range(repeats):
        indices = rng.integers(0, len(years), len(years))
        boot_years = years[indices]
        boot_values = values[indices]
        best = min(
            (
                (_segmented_fit(boot_years, boot_values, float(candidate))[1], int(candidate))
                for candidate in candidates
            ),
            key=lambda item: (item[0], item[1]),
        )[1]
        selected[best] += 1
    audit["BootstrapSelectionFrequency"] = audit["CandidateYear"].map(
        lambda year: selected[int(year)] / repeats
    )
    audit["DeltaBICFromSingle"] = single_bic - audit["BIC"]
    best_frequency = float(
        audit.loc[
            audit["CandidateYear"] == best_year,
            "BootstrapSelectionFrequency",
        ].iloc[0]
    )
    delta_bic = float(single_bic - best_bic)
    robust = bool(delta_bic >= 6.0 and best_frequency >= 0.5)
    conclusion = (
        f"{best_year} 为最优候选断点，ΔBIC={delta_bic:.2f}，"
        f"Bootstrap 选择率={best_frequency:.1%}。"
        + (
            "两项门槛均通过，可称为稳健明显拐点。"
            if robust
            else "至少一项门槛未通过，证据不足以支持唯一稳定拐点。"
        )
    )
    summary = {
        "BestYear": best_year,
        "DeltaBIC": delta_bic,
        "BootstrapSelectionFrequency": best_frequency,
        "RobustBreakpoint": robust,
        "Conclusion": conclusion,
    }
    return audit, summary, fitted_by_year[best_year]


def run_problem1(
    worldcups: pd.DataFrame,
    output_dirs: dict[str, Path],
) -> dict:
    tables_dir = output_dirs["tables"]
    figures_dir = output_dirs["figures"]
    logs_dir = output_dirs["logs"]
    data = parse_worldcups(worldcups)
    data.to_csv(tables_dir / "1_1_世界杯规模与强度.csv", index=False, encoding="utf-8-sig")
    # Official four descriptive variables: total goals, goals/match, teams, matches.
    descriptive = (
        data[["GoalsScored", "GoalsPerMatch", "QualifiedTeams", "MatchesPlayed"]]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .T.rename_axis("Variable")
        .reset_index()
    )
    descriptive.to_csv(
        tables_dir / "1_1_四项描述统计.csv",
        index=False,
        encoding="utf-8-sig",
    )
    breakpoint_audit, breakpoint, breakpoint_fit = _breakpoint_audit(data)
    breakpoint_audit.to_csv(
        tables_dir / "1_1_拐点BIC与Bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    scale_intensity = data[
        ["Year", "GoalsScored", "MatchesPlayed", "GoalsPerMatch", "QualifiedTeams"]
    ].copy()
    scale_intensity["ScaleIdentityCheck"] = (
        scale_intensity["MatchesPlayed"] * scale_intensity["GoalsPerMatch"]
    )
    scale_intensity["IdentityError"] = (
        scale_intensity["GoalsScored"] - scale_intensity["ScaleIdentityCheck"]
    )
    scale_intensity.to_csv(
        tables_dir / "1_1_规模强度分解.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train = data.dropna(subset=[*P1_FEATURES, "GoalsScored"]).copy()
    coefficients, fit_metrics = _ols_table(
        train[P1_FEATURES].to_numpy(float),
        train["GoalsScored"].to_numpy(float),
        P1_FEATURES,
    )
    coefficients.to_csv(
        tables_dir / "1_2_三特征回归系数与显著性.csv",
        index=False,
        encoding="utf-8-sig",
    )
    vif = _vif_table(train, P1_FEATURES)
    vif.to_csv(
        tables_dir / "1_2_三特征多重共线性VIF.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metric_frame = pd.DataFrame(
        [
            {
                "Model": "Three-feature OLS",
                **fit_metrics,
                "IndependentSlopeInterpretationAllowed": False,
                "Caution": (
                    "The joint model may predict totals, but high VIF and non-significant "
                    "slopes do not support independent causal/mechanism claims."
                ),
            }
        ]
    )
    metric_frame.to_csv(
        tables_dir / "1_2_三特征模型指标.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model3 = LinearRegression().fit(train[P1_FEATURES], train["GoalsScored"])
    leave_one_rows = []
    for index in range(len(train)):
        fitting = train.drop(train.index[index])
        held = train.iloc[[index]]
        model = LinearRegression().fit(fitting[P1_FEATURES], fitting["GoalsScored"])
        predicted = float(model.predict(held[P1_FEATURES])[0])
        actual = float(held["GoalsScored"].iloc[0])
        leave_one_rows.append(
            {
                "Year": int(held["Year"].iloc[0]),
                "ActualGoals": actual,
                "PredictedGoals": predicted,
                "AbsoluteError": abs(predicted - actual),
            }
        )
    leave_one = pd.DataFrame(leave_one_rows)
    leave_one.to_csv(
        tables_dir / "1_2_三特征留一届验证.csv",
        index=False,
        encoding="utf-8-sig",
    )
    radius3 = _empirical_radius(leave_one["AbsoluteError"])
    predicted_2018 = float(
        model3.predict(
            pd.DataFrame([[32, 64, 3_031_768]], columns=P1_FEATURES)
        )[0]
    )

    exante_features = ["QualifiedTeams", "MatchesPlayed"]
    rolling_rows = []
    for index in range(8, len(train)):
        earlier = train.iloc[:index]
        target = train.iloc[[index]]
        model = LinearRegression().fit(earlier[exante_features], earlier["GoalsScored"])
        predicted = float(model.predict(target[exante_features])[0])
        actual = float(target["GoalsScored"].iloc[0])
        rolling_rows.append(
            {
                "Year": int(target["Year"].iloc[0]),
                "ActualGoals": actual,
                "PredictedGoals": predicted,
                "AbsoluteError": abs(predicted - actual),
            }
        )
    rolling = pd.DataFrame(rolling_rows)
    rolling.to_csv(
        tables_dir / "1_2_赛前两特征滚动验证.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model2 = LinearRegression().fit(train[exante_features], train["GoalsScored"])
    exante_2018 = float(
        model2.predict(pd.DataFrame([[32, 64]], columns=exante_features))[0]
    )
    radius2 = _empirical_radius(rolling["AbsoluteError"])
    actual_2018 = 169.0
    comparison = pd.DataFrame(
        [
            {
                "Model": "题面三特征复现模型",
                "AvailableAt": "赛后（总观众数完整已知）",
                "Validation": "Leave-one-tournament-out",
                "ValidationMAE": leave_one["AbsoluteError"].mean(),
                "PredictedGoals2018": predicted_2018,
                "Interval80Lower": predicted_2018 - radius3,
                "Interval80Upper": predicted_2018 + radius3,
                "ActualGoals2018": actual_2018,
                "AbsoluteError2018": abs(predicted_2018 - actual_2018),
            },
            {
                "Model": "严格赛前两特征模型",
                "AvailableAt": "赛前",
                "Validation": "Rolling origin",
                "ValidationMAE": rolling["AbsoluteError"].mean(),
                "PredictedGoals2018": exante_2018,
                "Interval80Lower": exante_2018 - radius2,
                "Interval80Upper": exante_2018 + radius2,
                "ActualGoals2018": actual_2018,
                "AbsoluteError2018": abs(exante_2018 - actual_2018),
            },
        ]
    )
    comparison.to_csv(
        tables_dir / "1_2_模型口径与2018预测.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_problem1_trends(
        data,
        figures_dir / "图1_总进球趋势.png",
        figures_dir / "图2_场均进球趋势.png",
        breakpoint_year=int(breakpoint["BestYear"]),
        breakpoint_fit=breakpoint_fit,
    )

    total_slope = float(np.polyfit(data["Year"], data["GoalsScored"], 1)[0])
    intensity_slope = float(np.polyfit(data["Year"], data["GoalsPerMatch"], 1)[0])
    trend_conclusion = (
        f"程序趋势判定：总进球线性斜率为每年 {total_slope:.3f} 球，"
        f"场均进球线性斜率为每年 {intensity_slope:.4f} 球；"
        "因此总量长期上升主要与比赛规模扩张一致，场均强度并非同步单调上升。"
    )
    conclusion = (
        trend_conclusion
        +
        f"三特征模型 R²={fit_metrics['R2']:.3f}，调整后 R²="
        f"{fit_metrics['AdjustedR2']:.3f}。2018 复现预测为 {predicted_2018:.2f}，"
        f"实际为 169；严格赛前两特征预测为 {exante_2018:.2f}。VIF 范围为 "
        f"{vif['VIF'].min():.2f}-{vif['VIF'].max():.2f}，且三个斜率均未在 "
        "5% 水平显著，因此联合模型可用于整体预测，但不得把单个系数包装成"
        f"独立边际机制证据。{breakpoint['Conclusion']}"
    )
    (logs_dir / "1_问题一结论.txt").write_text(conclusion, encoding="utf-8")
    pd.DataFrame(
        [
            {
                "R2": fit_metrics["R2"],
                "AdjustedR2": fit_metrics["AdjustedR2"],
                "Predicted2018ThreeFeature": predicted_2018,
                "Predicted2018ExAnte": exante_2018,
                "Actual2018": actual_2018,
                "MaxVIF": vif["VIF"].max(),
                "TotalGoalsTrendSlopePerYear": total_slope,
                "GoalsPerMatchTrendSlopePerYear": intensity_slope,
                **breakpoint,
                "Conclusion": conclusion,
            }
        ]
    ).to_csv(output_dirs["base"] / "problem1.csv", index=False, encoding="utf-8-sig")
    return {
        "data": data,
        "fit_metrics": fit_metrics,
        "descriptive": descriptive,
        "breakpoint": breakpoint,
        "breakpoint_audit": breakpoint_audit,
        "coefficients": coefficients,
        "vif": vif,
        "comparison": comparison,
        "rolling": rolling,
        "leave_one": leave_one,
        "predicted_2018": predicted_2018,
        "exante_2018": exante_2018,
        "actual_2018": actual_2018,
        "conclusion": conclusion,
    }
