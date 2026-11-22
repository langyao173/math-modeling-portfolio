#!/usr/bin/env python3
"""Formal prediction entry point for Problem A."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
SRC_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SRC_DIR / ".mplconfig"))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="A题世界杯赛果预测")
    parser.add_argument(
        "--testdata",
        default=None,
        help="官方三表及 query.csv 所在目录；也支持比赛表内嵌无标签查询行。",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="显式使用四场内置演示输入；正式模式绝不自动启用。",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="同时输出三分类概率和全场进球区间明细。",
    )
    parser.add_argument(
        "--prediction-only",
        action="store_true",
        help="仅生成 cla_pre.csv，不重跑六问分析（正式一键请勿加此开关）。",
    )
    return parser.parse_args(argv)


def run_analysis(testdata_dir=None) -> dict:
    """Regenerate six-question tables, figures, logs, and report_metrics."""
    import shutil

    from analysis_pipeline import configure_runtime, run_all_analysis
    from config import resolve_output_root

    output_root = resolve_output_root()
    shutil.rmtree(output_root / "outputs", ignore_errors=True)
    configure_runtime(testdata_dir)
    print("正在生成六问分析、规定图表与单一事实源指标……")
    return run_all_analysis()


def run_prediction(args) -> dict:
    from config import ensure_output_dirs, resolve_output_root, resolve_training_data_dir
    from data_io import load_prediction_data
    from models import P3_FINAL_CONFIG
    from prediction_api import generate_evaluator_predictions

    output_root = resolve_output_root()
    training_dir, testdata_dir = resolve_training_data_dir(args.testdata)
    matches, embedded_query = load_prediction_data(training_dir)
    output_dirs = ensure_output_dirs(output_root) if args.details else None
    return generate_evaluator_predictions(
        matches=matches,
        embedded_query=embedded_query,
        testdata_dir=testdata_dir,
        output_root=output_root,
        output_dirs=output_dirs,
        model_config=P3_FINAL_CONFIG,
        demo=args.demo,
        write_details=args.details,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    try:
        analysis = None
        if not args.prediction_only:
            analysis = run_analysis(args.testdata)
        result = run_prediction(args)
        if analysis is not None:
            from analysis_pipeline import write_six_question_summary

            write_six_question_summary(
                analysis,
                len(result["minimal"]),
                Path(result["cla_path"]),
            )
        print(f"预测完成：读取“{result['source']}”，输出 {len(result['minimal'])} 行。")
        print(f"训练数据实际最大年份：{result['training_max_year']}")
        print(f"结果文件：{result['cla_path']}")
        if not args.prediction_only:
            print("六问分析、图表与日志已同步生成。")
        print(f"总运行用时：{time.perf_counter() - started:.2f} 秒。")
        return 0
    except ModuleNotFoundError as exc:
        print(
            f"运行失败：缺少 Python 依赖“{exc.name}”。"
            "请执行 python -m pip install -r requirements.txt。",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
