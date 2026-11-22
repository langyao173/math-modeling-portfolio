from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.evaluator.equivalence import LayoutSidecar, validate_compiler_equivalence
from noise_adaptive_hackathon.evaluator.submission import _passes_noiseless_semantics
from noise_adaptive_hackathon.evaluator.validate_circuit import validate_compiled_circuit
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.selection import candidate_metrics, select_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump Track A V14 candidate diagnostics.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "tracka_resynthesis_candidates_v14.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    compiler = NoiseAdaptiveCompiler(enable_parity_resynthesis=True, enable_track_a_resynthesis=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for benchmark_dir in sorted((args.root / "benchmarks" / "track_a" / "public").iterdir()):
            benchmark = load_benchmark(benchmark_dir)
            backend = _load_backend(args.root, benchmark)
            candidates = compiler.candidate_compilations(benchmark.circuit, backend)
            selected = select_candidate(candidates, backend, benchmark.circuit)
            for candidate in candidates:
                metrics = candidate_metrics(candidate, backend, benchmark.circuit)
                validation = validate_compiled_circuit(candidate.circuit, backend, benchmark.track)
                equivalent = False
                noiseless = False
                if validation.valid:
                    layout = LayoutSidecar({logical.x: physical for logical, physical in candidate.layout.items()})
                    equivalent = validate_compiler_equivalence(benchmark, candidate.circuit, layout, seed=7).valid
                    noiseless = equivalent and _passes_noiseless_semantics(benchmark, candidate.circuit, seed=7)
                handle.write(
                    json.dumps(
                        {
                            "sample": benchmark.name,
                            "candidate": candidate.name,
                            "chosen": candidate is selected,
                            "valid": validation.valid,
                            "equivalent": equivalent,
                            "noiseless": noiseless,
                            "two_qubit_gates": metrics.two_qubit_count,
                            "depth": metrics.depth,
                            "estimated_log_reliability": metrics.log_reliability,
                            "readout_cost": metrics.readout_cost,
                            "compile_time_sec": metrics.compile_time_sec,
                            "validation_errors": validation.errors,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    return 0


def _load_backend(root: Path, benchmark) -> CompetitionBackend:
    backend_path = (
        root
        / "backends"
        / benchmark.metadata["backend_size_class"]
        / f"{benchmark.metadata['backend_snapshot_id']}.json"
    )
    return CompetitionBackend.parse_raw(backend_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
