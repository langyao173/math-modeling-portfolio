from __future__ import annotations

import argparse
import subprocess
import sys

from common import ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standard fast/full evaluation sweep.")
    parser.add_argument("--skip-full", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = [[sys.executable, str(ROOT / "scripts" / "fast_eval.py")]]
    if not args.skip_full:
        commands.append([sys.executable, str(ROOT / "scripts" / "full_eval.py")])
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
