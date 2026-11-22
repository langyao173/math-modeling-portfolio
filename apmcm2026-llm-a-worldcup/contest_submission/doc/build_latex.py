#!/usr/bin/env python3
"""Rebuild the XeLaTeX delivery brief from report_metrics.csv."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


DOC_DIR = Path(__file__).resolve().parent
SRC_DIR = DOC_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(SRC_DIR / ".mplconfig"))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis_pipeline import configure_runtime, run_all_analysis  # noqa: E402
from report_data import csv_to_latex_macros  # noqa: E402


def _find_xelatex() -> str:
    env = os.environ.get("XELATEX_BIN")
    if env:
        return env
    found = shutil.which("xelatex")
    if found:
        return found
    raise RuntimeError(
        "未找到 xelatex。请安装 TeX Live / MiKTeX，或设置 XELATEX_BIN。"
    )


def _compile(xelatex: str, tex_name: str = "report.tex") -> None:
    command = [
        xelatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_name,
    ]
    # Two passes for stable references/TOC-less layout.
    for _ in range(2):
        completed = subprocess.run(
            command,
            cwd=str(DOC_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            sys.stderr.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            raise RuntimeError("xelatex 编译失败。")


def main() -> int:
    xelatex = _find_xelatex()
    configure_runtime()
    run_all_analysis()
    metrics_csv = SRC_DIR / "outputs" / "report_metrics.csv"
    generated = DOC_DIR / "report_metrics.tex"
    csv_to_latex_macros(metrics_csv, generated)

    pdf_built = DOC_DIR / "report.pdf"
    pdf_final = DOC_DIR / "说明文档.pdf"
    try:
        _compile(xelatex)
        if not pdf_built.is_file():
            raise RuntimeError("未生成 report.pdf。")
        shutil.copy2(pdf_built, pdf_final)
    finally:
        # Keep report_metrics.tex only if compile fails for debugging? Remove for release.
        generated.unlink(missing_ok=True)
        for suffix in (".aux", ".log", ".out", ".toc"):
            (DOC_DIR / f"report{suffix}").unlink(missing_ok=True)

    print(f"已生成：{pdf_final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
