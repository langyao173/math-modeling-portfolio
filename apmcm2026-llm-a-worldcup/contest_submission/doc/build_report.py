#!/usr/bin/env python3
"""Rebuild the concise report from generated analysis metrics."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


DOC_DIR = Path(__file__).resolve().parent
SRC_DIR = DOC_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(SRC_DIR / ".mplconfig"))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis_pipeline import configure_runtime, run_all_analysis  # noqa: E402
from report_data import csv_to_typst_module  # noqa: E402


def _typst_font_choice(typst: str) -> tuple[str, str]:
    completed = subprocess.run(
        [typst, "fonts"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    available = set(completed.stdout.splitlines())
    for name in (
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
    ):
        if name in available:
            return "zh", name
    for name in ("DejaVu Sans", "Arial", "Liberation Sans"):
        if name in available:
            return "en", name
    return "en", "sans-serif"


def main() -> int:
    typst = os.environ.get("TYPST_BIN") or shutil.which("typst")
    if not typst:
        raise RuntimeError(
            "未找到 Typst。请安装 Typst 0.14+，或通过 TYPST_BIN 指定。"
        )

    configure_runtime()
    run_all_analysis()
    metrics_csv = SRC_DIR / "outputs" / "report_metrics.csv"
    generated_module = DOC_DIR / "report_metrics.typ"
    csv_to_typst_module(metrics_csv, generated_module)
    language, font = _typst_font_choice(typst)
    try:
        command = [
            typst,
            "compile",
            "--root",
            str(SRC_DIR),
            "--input",
            "asset-root=../outputs/figures",
            "--input",
            f"doc-lang={language}",
            "--input",
            f"doc-font={font}",
            str(DOC_DIR / "report.typ"),
            str(DOC_DIR / "说明文档.pdf"),
        ]
        subprocess.run(command, check=True)
    finally:
        generated_module.unlink(missing_ok=True)
    print(f"已生成：{DOC_DIR / '说明文档.pdf'}（{language}, {font}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
