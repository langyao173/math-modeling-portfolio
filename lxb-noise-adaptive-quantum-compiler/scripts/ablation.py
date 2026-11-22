from __future__ import annotations

import argparse
import sys
from pathlib import Path

from noise_adaptive_hackathon.evaluator.submission import evaluate_submission_command

from common import ROOT, append_manifest, summary_payload, write_json
from noise_adaptive_hackathon.submission.variants import ABLATION_VARIANTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed-variant ablation evaluations.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "ablation_summary.json")
    parser.add_argument("--shots-a", type=int, default=256)
    parser.add_argument("--shots-b", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--variants", nargs="*", default=list(ABLATION_VARIANTS), choices=ABLATION_VARIANTS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    for variant in args.variants:
        summary = evaluate_submission_command(
            root=args.root,
            command=[sys.executable, str(args.root / "scripts" / "variant_compiler.py"), "--variant", variant],
            shots_a=args.shots_a,
            shots_b=args.shots_b,
            seed=args.seed,
            timeout_sec=args.timeout_sec,
        )
        payload = summary_payload(summary)
        summaries[variant] = payload
        samples = payload["sample_results"]
        invalid_count = sum(1 for sample in samples if not sample["valid"])
        row = {
            "variant": variant,
            "track_a_score": payload["track_a_score"],
            "track_b_score": payload["track_b_score"],
            "final_score": payload["final_score"],
            "invalid_count": invalid_count,
            "mean_depth": _mean(sample["depth"] for sample in samples),
            "mean_two_qubit_gate_count": _mean(sample["two_qubit_gate_count"] for sample in samples),
            "mean_estimated_log_reliability": _mean(sample["estimated_log_reliability"] for sample in samples),
            "mean_compile_time_sec": _mean(sample["compile_time_sec"] for sample in samples),
        }
        rows.append(row)
        append_manifest(
            summary=payload,
            variant=f"ablation_{variant}",
            seed=args.seed,
            config={"shots_a": args.shots_a, "shots_b": args.shots_b, "timeout_sec": args.timeout_sec},
            decision="ablation",
        )

    write_json(
        args.out,
        {
            "shots_a": args.shots_a,
            "shots_b": args.shots_b,
            "seed": args.seed,
            "rows": rows,
            "summaries": summaries,
        },
    )
    return 0


def _mean(values) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
