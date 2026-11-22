#!/usr/bin/env python3
"""Generate all six-question analysis artifacts without running predictions."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
SRC_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SRC_DIR / ".mplconfig"))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="A题六问完整分析")
    parser.add_argument("--testdata", default=None, help="可选官方三表目录。")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    try:
        from analysis_pipeline import configure_runtime, run_all_analysis
        from config import resolve_output_root

        output_root = resolve_output_root()
        shutil.rmtree(output_root / "outputs", ignore_errors=True)
        paths = configure_runtime(args.testdata)
        print("正在生成六问分析、规定图表与单一事实源指标……")
        run_all_analysis()
        print(f"分析完成：{paths['output_dirs']['base']}")
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
