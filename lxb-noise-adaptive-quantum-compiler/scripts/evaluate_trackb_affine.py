from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from common import ROOT, write_json

from noise_adaptive_hackathon.evaluator.submission import _evaluate_single_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Track B affine synthetic benchmarks.")
    parser.add_argument("--root", type=Path, default=ROOT / "experiments" / "synthetic_trackb_affine_v13_1")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "synthetic_trackb_affine_v13_1.json")
    parser.add_argument("--compiler", type=Path, default=ROOT / "compiler.py")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 17])
    parser.add_argument("--timeout-sec", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = (
        [sys.executable, str((ROOT / "scripts" / "variant_compiler.py").resolve()), "--variant", args.variant]
        if args.variant
        else [sys.executable, str(args.compiler.resolve())]
    )
    manifest = _manifest(args.root)
    negative_names = {record["name"] for record in manifest if record.get("negative_case")}
    seed_summaries = [
        _evaluate_seed(
            root=args.root,
            command=command,
            shots=args.shots,
            seed=seed,
            timeout_sec=args.timeout_sec,
            negative_names=negative_names,
        )
        for seed in args.seeds
    ]
    write_json(
        args.out,
        {
            "root": str(args.root),
            "variant": args.variant,
            "compiler": str(args.compiler),
            "shots": args.shots,
            "seeds": args.seeds,
            "seed_summaries": seed_summaries,
            "mean_track_b_success": _mean(summary["mean_success"] for summary in seed_summaries),
            "max_invalid_count": max((summary["invalid_count"] for summary in seed_summaries), default=0),
            "max_negative_false_accept_count": max((summary["negative_false_accept_count"] for summary in seed_summaries), default=0),
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
    shots: int,
    seed: int,
    timeout_sec: int,
    negative_names: set[str],
) -> dict[str, object]:
    results = [
        _evaluate_single_benchmark(
            root=root,
            benchmark_dir=benchmark_dir,
            command=command,
            shots=shots,
            seed=seed + 10_000 + index,
            timeout_sec=timeout_sec,
        )
        for index, benchmark_dir in enumerate(_benchmark_dirs(root))
    ]
    sample_results = [asdict(result) for result in results]
    return {
        "seed": seed,
        "mean_success": _mean(result.success_rate for result in results),
        "invalid_count": sum(1 for result in results if not result.valid),
        "negative_false_accept_count": sum(
            1
            for result in results
            if result.benchmark_name in negative_names and result.two_qubit_gate_count < _source_twoq(root, result.benchmark_name)
        ),
        "sample_results": sample_results,
    }


def _benchmark_dirs(root: Path) -> list[Path]:
    directory = root / "benchmarks" / "track_b" / "holdout"
    return sorted(path for path in directory.iterdir() if path.is_dir())


def _manifest(root: Path) -> list[dict[str, object]]:
    path = root / "holdout_manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _source_twoq(root: Path, name: str) -> int:
    for record in _manifest(root):
        if record.get("name") == name:
            return int(record.get("two_qubit_gates", 0))
    return 0


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
