from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

from common import ROOT, write_json

from noise_adaptive_hackathon.evaluator.score_submission import score_submission
from noise_adaptive_hackathon.evaluator.submission import _evaluate_single_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a compiler on generated synthetic holdout benchmarks.")
    parser.add_argument("--root", type=Path, default=ROOT / "experiments" / "synthetic_holdout_v11")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "synthetic_holdout_summary.json")
    parser.add_argument("--compiler", type=Path, default=ROOT / "compiler.py")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--shots-a", type=int, default=128)
    parser.add_argument("--shots-b", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 17])
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--label", default="synthetic_holdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = (
        [sys.executable, str((ROOT / "scripts" / "variant_compiler.py").resolve()), "--variant", args.variant]
        if args.variant
        else [sys.executable, str(args.compiler.resolve())]
    )
    seed_summaries = [
        _evaluate_seed(
            root=args.root,
            command=command,
            shots_a=args.shots_a,
            shots_b=args.shots_b,
            seed=seed,
            timeout_sec=args.timeout_sec,
        )
        for seed in args.seeds
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out,
        {
            "label": args.label,
            "root": str(args.root),
            "compiler": str(args.compiler),
            "variant": args.variant,
            "shots_a": args.shots_a,
            "shots_b": args.shots_b,
            "timeout_sec": args.timeout_sec,
            "seeds": args.seeds,
            "seed_summaries": seed_summaries,
            "mean_final_score": _mean(summary["final_score"] for summary in seed_summaries),
            "mean_track_a_score": _mean(summary["track_a_score"] for summary in seed_summaries),
            "mean_track_b_score": _mean(summary["track_b_score"] for summary in seed_summaries),
            "max_invalid_count": max((summary["invalid_count"] for summary in seed_summaries), default=0),
            "p95_compile_time_sec": _p95(
                sample["compile_time_sec"]
                for summary in seed_summaries
                for sample in summary["sample_results"]
            ),
        },
    )
    return 0


def _evaluate_seed(
    *,
    root: Path,
    command: list[str],
    shots_a: int,
    shots_b: int,
    seed: int,
    timeout_sec: int,
) -> dict[str, object]:
    track_a_dirs = _benchmark_dirs(root, "track_a")
    track_b_dirs = _benchmark_dirs(root, "track_b")
    track_a_results = [
        _evaluate_single_benchmark(
            root=root,
            benchmark_dir=benchmark_dir,
            command=command,
            shots=shots_a,
            seed=seed + index,
            timeout_sec=timeout_sec,
        )
        for index, benchmark_dir in enumerate(track_a_dirs)
    ]
    track_b_results = [
        _evaluate_single_benchmark(
            root=root,
            benchmark_dir=benchmark_dir,
            command=command,
            shots=shots_b,
            seed=seed + 10_000 + index,
            timeout_sec=timeout_sec,
        )
        for index, benchmark_dir in enumerate(track_b_dirs)
    ]
    track_a_score = _mean(result.success_rate for result in track_a_results)
    track_b_score = _mean(result.success_rate for result in track_b_results)
    total = score_submission(track_a_score=track_a_score, track_b_score=track_b_score)
    sample_results = [asdict(result) for result in track_a_results + track_b_results]
    return {
        "seed": seed,
        "track_a_score": track_a_score,
        "track_b_score": track_b_score,
        "final_score": total.final_score,
        "invalid_count": sum(1 for result in track_a_results + track_b_results if not result.valid),
        "sample_results": sample_results,
    }


def _benchmark_dirs(root: Path, track: str) -> list[Path]:
    directory = root / "benchmarks" / track / "holdout"
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_dir())


def _mean(values) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _p95(values) -> float:
    items = sorted(float(value) for value in values)
    if not items:
        return 0.0
    index = min(len(items) - 1, int(0.95 * (len(items) - 1)))
    return items[index]


if __name__ == "__main__":
    raise SystemExit(main())
