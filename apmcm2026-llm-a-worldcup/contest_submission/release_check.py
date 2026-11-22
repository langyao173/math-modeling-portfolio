#!/usr/bin/env python3
"""Fast release check for tests, six outputs, and prediction contracts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import pandas as pd


SRC_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(SRC_DIR / ".mplconfig"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _run(
    copied_src: Path,
    foreign_cwd: Path,
    environment: dict,
    script: str,
    *arguments: str,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(copied_src / script), *arguments],
        cwd=foreign_cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )


def main() -> int:
    started = time.perf_counter()
    checks: dict[str, bool] = {}

    suite = unittest.defaultTestLoader.discover(
        str(SRC_DIR / "tests"),
        pattern="test_*.py",
    )
    test_result = unittest.TextTestRunner(verbosity=1).run(suite)
    checks["unit_tests"] = bool(
        test_result.wasSuccessful() and test_result.testsRun >= 35
    )

    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in SRC_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".py", ".md", ".typ", ".bat"}
        and "outputs" not in path.parts
        and path.name != "release_check.py"
    )
    checks["single_official_model"] = all(
        token not in source_text
        for token in (
            "TwoStageOutcomeModel",
            "lockbox_evaluation",
            "Fjelstul_WorldCup_matches",
        )
    )
    checks["no_hardcoded_all_passed"] = "全部通过" not in source_text
    checks["no_pregenerated_typst_metrics"] = not (
        SRC_DIR / "doc" / "report_metrics.typ"
    ).exists()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        copied_src = root / "received_submission"
        shutil.copytree(
            SRC_DIR,
            copied_src,
            ignore=shutil.ignore_patterns(
                "outputs",
                "__pycache__",
                "*.pyc",
                ".pytest_cache",
                ".mplconfig",
                "cla_pre.csv",
                "说明文档.pdf",
            ),
        )
        foreign_cwd = root / "examiner_cwd"
        foreign_cwd.mkdir()
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONUTF8": "1",
                "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(root / "mplconfig"),
                "LOKY_MAX_CPU_COUNT": "2",
            }
        )
        fallback_environment = environment.copy()
        fallback_environment["APMCM_PLOT_LANGUAGE"] = "en"
        fallback = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from visuals import font_status;"
                    "s=font_status();"
                    "raise SystemExit(0 if s['plot_language']=='en' "
                    "and s['selected_font']=='DejaVu Sans' else 1)"
                ),
            ],
            cwd=copied_src,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=fallback_environment,
        )
        checks["english_font_fallback"] = fallback.returncode == 0

        analysis = _run(
            copied_src,
            foreign_cwd,
            environment,
            "analyze.py",
        )
        checks["analysis_exit_zero"] = analysis.returncode == 0
        if analysis.returncode:
            print(analysis.stdout + analysis.stderr, file=sys.stderr)

        required = [
            copied_src / "outputs" / "tables" / "1_1_四项描述统计.csv",
            copied_src / "outputs" / "figures" / "图1_总进球趋势.png",
            copied_src / "outputs" / "figures" / "图2_场均进球趋势.png",
            copied_src / "outputs" / "tables" / "1_2_模型口径与2018预测.csv",
            copied_src / "outputs" / "tables" / "2_1_比赛阶段进球统计.csv",
            copied_src / "outputs" / "tables" / "2_1_半场与全场进球相关性.csv",
            copied_src / "outputs" / "figures" / "图4_半场与全场进球相关.png",
            copied_src / "outputs" / "tables" / "3_1_球队历史四特征.csv",
            copied_src / "outputs" / "figures" / "图5_问题三四特征相关热力图.png",
            copied_src / "outputs" / "tables" / "3_2_2014逻辑回归测试指标.csv",
            copied_src / "outputs" / "figures" / "图6_2014逻辑回归混淆矩阵.png",
            copied_src / "outputs" / "report_metrics.csv",
        ]
        checks["required_outputs"] = all(path.is_file() for path in required)
        checks["six_figures"] = (
            len(list((copied_src / "outputs" / "figures").glob("图*.png")))
            == 6
        )

        metrics_path = copied_src / "outputs" / "report_metrics.csv"
        if metrics_path.is_file():
            metrics = pd.read_csv(metrics_path, encoding="utf-8-sig").iloc[0]
            checks["metrics_confirm_outputs"] = bool(
                metrics["six_question_outputs_complete"]
            )
            checks["official_split_and_model"] = bool(
                metrics["p3_protocol_checks_met"]
                and int(metrics["p3_feature_count"]) == 4
                and "逻辑回归" in str(metrics["p3_model_name"])
                and float(metrics["p3_2014_accuracy"]) >= 0.55
            )
        else:
            checks["metrics_confirm_outputs"] = False
            checks["official_split_and_model"] = False

        demo = _run(
            copied_src,
            foreign_cwd,
            environment,
            "main.py",
            "--prediction-only",
            "--demo",
            "--details",
        )
        checks["demo_exit_zero"] = demo.returncode == 0
        if demo.returncode:
            print(demo.stdout + demo.stderr, file=sys.stderr)

        submission_path = copied_src / "cla_pre.csv"
        if submission_path.is_file():
            submission = pd.read_csv(submission_path)
            checks["submission_contract"] = bool(
                submission.columns.tolist() == ["id", "prediction"]
                and len(submission) == 4
                and set(submission["prediction"]).issubset({0, 1, 2})
            )
            first_hash = hashlib.sha256(
                submission_path.read_bytes()
            ).hexdigest()
        else:
            checks["submission_contract"] = False
            first_hash = ""

        probabilities_path = (
            copied_src
            / "outputs"
            / "tables"
            / "3_4_待预测比赛统一概率.csv"
        )
        if probabilities_path.is_file():
            details = pd.read_csv(probabilities_path)
            probability_columns = ["主队胜概率", "平局概率", "客队胜概率"]
            probabilities = details[probability_columns].to_numpy(float)
            checks["probability_contract"] = bool(
                (probabilities.argmax(axis=1) == details["prediction"]).all()
                and details["硬分类与概率一致"].all()
            )
        else:
            checks["probability_contract"] = False

        repeated = _run(
            copied_src,
            foreign_cwd,
            environment,
            "main.py",
            "--prediction-only",
            "--demo",
        )
        checks["deterministic_prediction"] = bool(
            repeated.returncode == 0
            and submission_path.is_file()
            and hashlib.sha256(submission_path.read_bytes()).hexdigest()
            == first_hash
        )

        query_dir = root / "formal_query"
        query_dir.mkdir()
        pd.DataFrame(
            {
                "编号": [9, 3],
                "主队": ["德国", "Brazil"],
                "客队": ["France", "Argentina"],
            }
        ).to_csv(
            query_dir / "query.csv",
            index=False,
            encoding="utf-8-sig",
        )
        formal = _run(
            copied_src,
            foreign_cwd,
            environment,
            "main.py",
            "--prediction-only",
            "--testdata",
            str(query_dir),
        )
        formal_output = (
            pd.read_csv(submission_path)
            if submission_path.is_file()
            else pd.DataFrame()
        )
        checks["formal_query_contract"] = bool(
            formal.returncode == 0
            and formal_output.get("id", pd.Series(dtype=int)).tolist()
            == [9, 3]
        )

        no_query = _run(
            copied_src,
            foreign_cwd,
            environment,
            "main.py",
            "--prediction-only",
        )
        checks["missing_query_fails"] = no_query.returncode != 0
        checks["no_foreign_cwd_writes"] = not any(foreign_cwd.iterdir())

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        print(f"发布检查未满足：{', '.join(failed)}", file=sys.stderr)
        return 1
    print(
        f"发布检查完成：{test_result.testsRun} 项单元测试；"
        f"用时 {time.perf_counter() - started:.2f} 秒。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
