from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from common import ROOT


CODE_DIRS = ("src", "scripts")
CODE_FILES = ("compiler.py", "pyproject.toml", "README.md", "AGENTS.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a competition submission package skeleton.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "submission_package")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = args.out / "1_Report"
    code = args.out / "2_Code"
    results = args.out / "3_Evaluation_Results"
    for directory in (report, code, results):
        directory.mkdir(parents=True, exist_ok=True)

    for name in CODE_FILES:
        source = args.root / name
        if source.exists():
            shutil.copy2(source, code / source.name)
    for name in CODE_DIRS:
        source = args.root / name
        target = code / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.egg-info"))

    report_source = args.root / "report"
    if report_source.exists():
        for source in report_source.iterdir():
            if source.is_file():
                shutil.copy2(source, report / source.name)

    experiments = args.root / "experiments"
    if experiments.exists():
        target = results / "experiments"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(experiments, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
