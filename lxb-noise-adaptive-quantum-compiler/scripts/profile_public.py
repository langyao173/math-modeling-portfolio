from __future__ import annotations

import argparse
from pathlib import Path

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.submission.backend_model import BackendNoiseModel
from noise_adaptive_hackathon.submission.circuit_features import extract_circuit_features

from common import ROOT, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile public benchmark interaction and backend pressure.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "public_profile.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = []
    for track in ("track_a", "track_b"):
        for benchmark_dir in sorted((args.root / "benchmarks" / track / "public").iterdir()):
            if not benchmark_dir.is_dir():
                continue
            benchmark = load_benchmark(benchmark_dir)
            backend_path = (
                args.root
                / "backends"
                / benchmark.metadata["backend_size_class"]
                / f"{benchmark.metadata['backend_snapshot_id']}.json"
            )
            backend = CompetitionBackend.parse_raw(backend_path.read_text(encoding="utf-8"))
            model = BackendNoiseModel.from_backend(backend)
            features = extract_circuit_features(benchmark.circuit)
            rows.append(
                {
                    "track": track,
                    "benchmark_name": benchmark.name,
                    "family": benchmark.family,
                    "logical_qubits": len(features.logical_qubits),
                    "two_qubit_interaction_edges": len(features.interactions),
                    "weighted_interaction": sum(item.weight for item in features.interactions),
                    "backend": backend.snapshot_id,
                    "min_edge_error": min(backend.edge_error(*edge) for edge in model.coupling_edges),
                    "max_edge_error": max(backend.edge_error(*edge) for edge in model.coupling_edges),
                    "min_readout_error": min(backend.readout_error.values()),
                    "max_readout_error": max(backend.readout_error.values()),
                }
            )
    write_json(args.out, {"rows": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
