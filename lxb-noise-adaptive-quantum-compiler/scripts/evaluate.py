from __future__ import annotations

import argparse
import sys
from pathlib import Path

from noise_adaptive_hackathon.evaluator.submission import evaluate_submission_command

from common import ROOT, append_manifest, summary_payload, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a black-box compiler and log a manifest entry.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--compiler", type=Path, default=ROOT / "compiler.py")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "public_summary.json")
    parser.add_argument("--shots-a", type=int, default=256)
    parser.add_argument("--shots-b", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--decision", default="candidate")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate_submission_command(
        root=args.root,
        command=[sys.executable, str(args.compiler)],
        shots_a=args.shots_a,
        shots_b=args.shots_b,
        seed=args.seed,
        timeout_sec=args.timeout_sec,
    )
    payload = summary_payload(summary)
    write_json(args.out, payload)
    append_manifest(
        summary=payload,
        variant=args.variant,
        seed=args.seed,
        config={
            "shots_a": args.shots_a,
            "shots_b": args.shots_b,
            "timeout_sec": args.timeout_sec,
            "compiler": str(args.compiler),
        },
        decision=args.decision,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
