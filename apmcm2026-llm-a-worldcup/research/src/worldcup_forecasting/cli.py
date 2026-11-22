"""Command-line interface for reproducible one-click execution."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from worldcup_forecasting.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the World Cup forecasting pipeline")
    parser.add_argument("--config", choices=("official", "research", "final"), default="final")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "override",
        nargs="*",
        help="Optional Hydra overrides such as runtime.figure_dpi=150",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_pipeline(
        project_root=args.project_root,
        config_name=args.config,
        output_override=args.output,
        overrides=args.override,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
