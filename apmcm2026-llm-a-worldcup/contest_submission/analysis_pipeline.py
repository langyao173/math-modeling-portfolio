"""Thin orchestration layer for the three competition questions."""

from __future__ import annotations

from pathlib import Path

try:
    from config import ensure_output_dirs, resolve_output_root, resolve_training_data_dir
    from data_io import load_competition_data
    from problem1_analysis import run_problem1
    from problem2_analysis import (
        P2_CATEGORICAL,
        P2_NUMERIC,
        make_problem2_estimator,
        prepare_problem2_model_frame,
        run_problem2,
    )
    from problem3_analysis import run_problem3
    from report_data import write_report_data
except ImportError:
    from .config import ensure_output_dirs, resolve_output_root, resolve_training_data_dir
    from .data_io import load_competition_data
    from .problem1_analysis import run_problem1
    from .problem2_analysis import (
        P2_CATEGORICAL,
        P2_NUMERIC,
        make_problem2_estimator,
        prepare_problem2_model_frame,
        run_problem2,
    )
    from .problem3_analysis import run_problem3
    from .report_data import write_report_data


_RUNTIME: dict | None = None


def configure_runtime(testdata_dir=None) -> dict:
    global _RUNTIME
    output_root = resolve_output_root()
    training_dir, discovered_testdata = resolve_training_data_dir(testdata_dir)
    output_dirs = ensure_output_dirs(output_root)
    _RUNTIME = {
        "output_root": output_root,
        "output_dirs": output_dirs,
        "training_data_dir": training_dir,
        "testdata_dir": discovered_testdata,
    }
    return _RUNTIME


def _runtime() -> dict:
    return _RUNTIME if _RUNTIME is not None else configure_runtime()


def run_all_analysis() -> dict:
    runtime = _runtime()
    cups, matches, players, embedded_query = load_competition_data(
        runtime["training_data_dir"],
        entity_policy="conservative",
    )
    problem1 = run_problem1(cups, runtime["output_dirs"])
    problem2 = run_problem2(matches, runtime["output_dirs"])
    problem3 = run_problem3(matches, runtime["output_dirs"])
    analysis = {
        "worldcups": cups,
        "matches": matches,
        "players": players,
        "embedded_query": embedded_query,
        "problem1": problem1,
        "problem2": problem2,
        "problem3": problem3,
    }
    analysis["report_metrics"] = write_report_data(analysis, runtime["output_dirs"])
    return analysis


def write_six_question_summary(
    analysis: dict,
    classification_rows: int,
    cla_path: Path,
) -> str:
    p1 = analysis["problem1"]
    p2 = analysis["problem2"]
    p3 = analysis["problem3"]
    p3_status = "满足" if all(p3["checks"].values()) else "未满足"
    lines = [
        "A题分析与预测摘要",
        "=" * 68,
        (
            "第1问：四项描述统计、总进球趋势图、场均进球趋势图和程序文字结论已生成。"
        ),
        (
            f"第2问：2018 三特征线性回归预测={p1['predicted_2018']:.2f}，"
            f"真实值=169；R²={p1['fit_metrics']['R2']:.3f}，调整后 R²="
            f"{p1['fit_metrics']['AdjustedR2']:.3f}，最大 VIF="
            f"{p1['vif']['VIF'].max():.2f}。"
        ),
        (
            f"第3问：阶段进球统计和半场—全场相关图已生成；Pearson r="
            f"{p2['pearson_r']:.3f}。"
        ),
        (
            f"第4问：单场线性回归工具已封装；2014 题面模型 MAE="
            f"{p2['official']['MAE']:.4f}、MSE={p2['official']['MSE']:.4f}、"
            f"R²={p2['official']['R2']:.3f}。"
        ),
        (
            "第5问：四特征 Spearman corrgram、VIF 与三类结局指示变量"
            "相关热力图已生成（不再使用 OutcomeCode 序数相关）。"
        ),
        (
            f"第6问：1930-2010 训练的多分类逻辑回归在 2014 测试集"
            f"正确 {p3['correct_matches']}/{p3['match_count']} 场，"
            f"准确率={p3['accuracy']:.3f}；混淆矩阵已保存，"
            f"协议核对={p3_status}。"
        ),
        (
            f"2026接口：输出 {classification_rows} 行至 {cla_path}；"
            f"历史训练实际最大年份={int(analysis['matches']['Year'].max())}。"
        ),
    ]
    path = _runtime()["output_dirs"]["logs"] / "分析与预测摘要.txt"
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


__all__ = [
    "P2_CATEGORICAL",
    "P2_NUMERIC",
    "configure_runtime",
    "make_problem2_estimator",
    "prepare_problem2_model_frame",
    "run_all_analysis",
    "write_six_question_summary",
]
