from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.measurement_channel_general import (
    affine_channel_diagnostics,
    measured_affine_channels_equivalent,
)
from noise_adaptive_hackathon.submission.selection import candidate_metrics, select_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump V13.1 measurement-channel candidate diagnostics.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "candidate_dump_v13_1.jsonl")
    parser.add_argument("--include-synthetic", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    benchmark_dirs = _public_dirs(args.root)
    if args.include_synthetic is not None:
        benchmark_dirs.extend(_synthetic_dirs(args.include_synthetic))
    with args.out.open("w", encoding="utf-8") as handle:
        for benchmark_dir in benchmark_dirs:
            benchmark = load_benchmark(benchmark_dir)
            backend = _load_backend(args.root, args.include_synthetic, benchmark)
            compiler = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_measurement_channel_general=True,
                short_circuit_exact_ghz=False,
                short_circuit_legacy_parity=False,
            )
            candidates = compiler.candidate_compilations(benchmark.circuit, backend)
            selected = select_candidate(candidates, backend, benchmark.circuit) if candidates else None
            diagnostics = affine_channel_diagnostics(benchmark.circuit)
            for candidate in candidates:
                metrics = candidate_metrics(candidate, backend, benchmark.circuit)
                record = {
                    "sample": benchmark.name,
                    "track": benchmark.track,
                    "family": benchmark.family,
                    "candidate": candidate.name,
                    "chosen": candidate is selected,
                    "two_qubit_gates": metrics.two_qubit_count,
                    "depth": metrics.depth,
                    "estimated_log_reliability": metrics.log_reliability,
                    "readout_cost": metrics.readout_cost,
                    "compile_time_sec": metrics.compile_time_sec,
                    "affine_supported": diagnostics.supported,
                    "affine_class": diagnostics.channel_class,
                    "affine_random_rank": diagnostics.random_rank,
                    "measured_channel_equivalent": (
                        measured_affine_channels_equivalent(benchmark.circuit, candidate.circuit, candidate.layout)
                        if benchmark.track == "track_b"
                        else False
                    ),
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    return 0


def _public_dirs(root: Path) -> list[Path]:
    return sorted((root / "benchmarks" / "track_a" / "public").iterdir()) + sorted(
        (root / "benchmarks" / "track_b" / "public").iterdir()
    )


def _synthetic_dirs(root: Path) -> list[Path]:
    directory = root / "benchmarks" / "track_b" / "holdout"
    return sorted(path for path in directory.iterdir() if path.is_dir()) if directory.exists() else []


def _load_backend(root: Path, synthetic_root: Path | None, benchmark) -> CompetitionBackend:
    base = synthetic_root if synthetic_root is not None and str(benchmark.name).startswith("affine_b_") else root
    backend_path = (
        base
        / "backends"
        / benchmark.metadata["backend_size_class"]
        / f"{benchmark.metadata['backend_snapshot_id']}.json"
    )
    if not backend_path.exists():
        backend_path = (
            root
            / "backends"
            / benchmark.metadata["backend_size_class"]
            / f"{benchmark.metadata['backend_snapshot_id']}.json"
        )
    return CompetitionBackend.parse_raw(backend_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
