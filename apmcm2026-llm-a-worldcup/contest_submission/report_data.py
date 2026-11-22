"""Single generated fact source for the report and delivery note."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from visuals import font_status
except ImportError:
    from .visuals import font_status


def _number(value) -> float:
    return float(value)


def _required_output_status(output_dirs: dict[str, Path]) -> dict[str, bool]:
    tables = output_dirs["tables"]
    figures = output_dirs["figures"]
    logs = output_dirs["logs"]
    return {
        "q1_descriptive_statistics": (
            tables / "1_1_四项描述统计.csv"
        ).is_file(),
        "q1_two_trend_figures": all(
            (figures / name).is_file()
            for name in ("图1_总进球趋势.png", "图2_场均进球趋势.png")
        ),
        "q1_generated_conclusion": (logs / "1_问题一结论.txt").is_file(),
        "q1_2018_regression": (
            tables / "1_2_模型口径与2018预测.csv"
        ).is_file(),
        "q2_stage_statistics": (
            tables / "2_1_比赛阶段进球统计.csv"
        ).is_file(),
        "q2_half_full_correlation": (
            tables / "2_1_半场与全场进球相关性.csv"
        ).is_file(),
        "q2_single_match_model": (
            tables / "2_2_2014历史开发基准指标.csv"
        ).is_file(),
        "q3_four_features_and_heatmap": all(
            path.is_file()
            for path in (
                tables / "3_1_球队历史四特征.csv",
                tables / "3_1_四特征与胜负相关性.csv",
                tables / "3_1_四特征Spearman相关.csv",
                tables / "3_1_四特征VIF.csv",
                figures / "图5_问题三四特征相关热力图.png",
            )
        ),
        "q3_2014_accuracy_and_confusion": all(
            path.is_file()
            for path in (
                tables / "3_2_2014逻辑回归测试指标.csv",
                figures / "图6_2014逻辑回归混淆矩阵.png",
            )
        ),
    }


def write_report_data(
    analysis: dict,
    output_dirs: dict[str, Path],
) -> dict:
    p1 = analysis["problem1"]
    p2 = analysis["problem2"]
    p3 = analysis["problem3"]
    output_status = _required_output_status(output_dirs)
    plot_font = font_status()
    metrics = {
        "data_min_year": int(analysis["matches"]["Year"].min()),
        "data_max_year": int(analysis["matches"]["Year"].max()),
        "tournament_count": int(analysis["matches"]["Year"].nunique()),
        "match_count": int(len(analysis["matches"])),
        "p1_r2": _number(p1["fit_metrics"]["R2"]),
        "p1_adjusted_r2": _number(p1["fit_metrics"]["AdjustedR2"]),
        "p1_prediction_2018": _number(p1["predicted_2018"]),
        "p1_actual_2018": _number(p1["actual_2018"]),
        "p1_max_vif": _number(p1["vif"]["VIF"].max()),
        "p1_breakpoint_year": int(p1["breakpoint"]["BestYear"]),
        "p2_pearson_half_full": _number(p2["pearson_r"]),
        "p2_pearson_p": _number(p2["pearson_p"]),
        "p2_official_2014_mae": _number(p2["official"]["MAE"]),
        "p2_official_2014_mse": _number(p2["official"]["MSE"]),
        "p2_official_2014_r2": _number(p2["official"]["R2"]),
        "p3_model_name": p3["config"]["model_name"],
        "p3_feature_count": len(p3["config"]["feature_names"]),
        "p3_c": _number(p3["config"]["C"]),
        "p3_class_weight": str(p3["config"]["class_weight"]),
        "p3_development_mean_accuracy": _number(
            p3["development"]["Accuracy"].mean()
        ),
        "p3_development_worst_accuracy": _number(
            p3["development"]["Accuracy"].min()
        ),
        "p3_2014_accuracy": _number(p3["accuracy"]),
        "p3_2014_correct_matches": int(p3["correct_matches"]),
        "p3_2014_match_count": int(p3["match_count"]),
        "p3_2014_macro_f1": _number(p3["macro_f1"]),
        "p3_2014_draw_recall": _number(p3["draw_recall"]),
        "p3_protocol_checks_met": bool(all(p3["checks"].values())),
        "six_question_outputs_complete": bool(all(output_status.values())),
        "chinese_font_available": bool(
            plot_font["chinese_font_available"]
        ),
        "selected_plot_font": str(plot_font["selected_font"]),
        "plot_language": str(plot_font["plot_language"]),
        "training_max_year": int(analysis["matches"]["Year"].max()),
        **output_status,
    }
    base = output_dirs["base"]
    pd.DataFrame([metrics]).to_csv(
        base / "report_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (base / "report_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def typst_literal(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return repr(value)


def csv_to_typst_module(csv_path: Path, destination: Path) -> Path:
    """Generate a temporary Typst dictionary from report_metrics.csv."""
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    if len(frame) != 1:
        raise ValueError("report_metrics.csv 必须且只能包含一行。")
    pairs = []
    for key, value in frame.iloc[0].items():
        if pd.isna(value):
            literal = "none"
        elif isinstance(value, str):
            lowered = value.lower()
            literal = (
                lowered
                if lowered in {"true", "false"}
                else typst_literal(value)
            )
        else:
            literal = typst_literal(value.item() if hasattr(value, "item") else value)
        pairs.append(f'  "{key}": {literal},')
    destination.write_text(
        "#let metrics = (\n" + "\n".join(pairs) + "\n)\n",
        encoding="utf-8",
    )
    return destination


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(text)
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def csv_to_latex_macros(csv_path: Path, destination: Path) -> Path:
    """Generate temporary XeLaTeX macros from report_metrics.csv."""
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    if len(frame) != 1:
        raise ValueError("report_metrics.csv 必须且只能包含一行。")
    row = frame.iloc[0]
    mapping = {
        "MetricDataMinYear": ("data_min_year", "int"),
        "MetricDataMaxYear": ("data_max_year", "int"),
        "MetricTournamentCount": ("tournament_count", "int"),
        "MetricMatchCount": ("match_count", "int"),
        "MetricPOneRTwo": ("p1_r2", "f3"),
        "MetricPOneAdjRTwo": ("p1_adjusted_r2", "f3"),
        "MetricPOnePredTwentyEighteen": ("p1_prediction_2018", "f2"),
        "MetricPOneActualTwentyEighteen": ("p1_actual_2018", "f0"),
        "MetricPOneMaxVif": ("p1_max_vif", "f2"),
        "MetricPTwoPearson": ("p2_pearson_half_full", "f3"),
        "MetricPTwoMae": ("p2_official_2014_mae", "f3"),
        "MetricPTwoMse": ("p2_official_2014_mse", "f3"),
        "MetricPTwoRTwo": ("p2_official_2014_r2", "f3"),
        "MetricPThreeModel": ("p3_model_name", "str"),
        "MetricPThreeC": ("p3_c", "raw"),
        "MetricPThreeAcc": ("p3_2014_accuracy", "f6"),
        "MetricPThreeAccShort": ("p3_2014_accuracy", "f3"),
        "MetricPThreeCorrect": ("p3_2014_correct_matches", "int"),
        "MetricPThreeN": ("p3_2014_match_count", "int"),
        "MetricPThreeMacroFOne": ("p3_2014_macro_f1", "f3"),
        "MetricPThreeDrawRecall": ("p3_2014_draw_recall", "f3"),
    }
    lines = [
        "% Auto-generated from outputs/report_metrics.csv — do not edit.",
        r"\makeatletter",
    ]
    for macro, (key, kind) in mapping.items():
        value = row[key]
        if kind == "int":
            text = str(int(float(value)))
        elif kind == "f0":
            text = f"{float(value):.0f}"
        elif kind == "f2":
            text = f"{float(value):.2f}"
        elif kind == "f3":
            text = f"{float(value):.3f}"
        elif kind == "f6":
            text = f"{float(value):.6f}"
        elif kind == "raw":
            text = _latex_escape(str(value))
        else:
            text = _latex_escape(str(value))
        lines.append(rf"\newcommand{{\{macro}}}{{{text}}}")
    lines.append(r"\makeatother")
    lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def write_delivery_note(
    csv_path: Path,
    destination: Path,
    version: str = "8.0",
) -> Path:
    """Generate the human delivery note from the same metrics CSV."""
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    if len(frame) != 1:
        raise ValueError("report_metrics.csv 必须且只能包含一行。")
    m = frame.iloc[0]

    def yes_no(value) -> str:
        return "通过" if str(value).lower() == "true" else "未通过"

    text = f"""# A题 {version} 交付说明

本说明由 `outputs/report_metrics.csv` 自动生成，不手工抄写指标。

## 正式模型与结果

- 问题三唯一正式模型：{m['p3_model_name']}
- 特征数：{int(float(m['p3_feature_count']))}
- 划分：1930–2010 训练，2014 测试
- 2014：正确 {int(float(m['p3_2014_correct_matches']))}/{int(float(m['p3_2014_match_count']))} 场，准确率 {float(m['p3_2014_accuracy']):.6f}
- 训练期滚动均值/最差折：{float(m['p3_development_mean_accuracy']):.6f} / {float(m['p3_development_worst_accuracy']):.6f}

## 自动输出核对

- 六问规定输出：{yes_no(m['six_question_outputs_complete'])}
- 问题三训练/测试/特征/概率协议：{yes_no(m['p3_protocol_checks_met'])}
- 图表字体：{m['selected_plot_font']}；语言回退：{m['plot_language']}

## 运行命令

```text
python -m pip install -r requirements.txt
python analyze.py
python main.py --demo --details
python release_check.py
cd doc
python build_report.py
```

生成文件包括 `outputs/` 下的 CSV、6 张规定图、文字结论与
`report_metrics.csv`，以及根目录 `cla_pre.csv`。正式预测使用
`python main.py --testdata <目录>`。
"""
    destination.write_text(text, encoding="utf-8")
    return destination


__all__ = [
    "csv_to_typst_module",
    "write_delivery_note",
    "write_report_data",
]
