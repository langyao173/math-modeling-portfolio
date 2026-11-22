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
from noise_adaptive_hackathon.submission.track_a_resynthesis.semantic_resynth import semantic_resynthesis_diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump Track A V15 semantic candidate diagnostics.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "tracka_semantic_candidates_v15_1.jsonl")
    parser.add_argument("--comparison", type=Path, default=ROOT / "experiments" / "a007_a008_semantic_comparison_v15_1.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    compiler = NoiseAdaptiveCompiler(
        enable_parity_resynthesis=True,
        enable_exact_small_router=True,
        enable_track_a_resynthesis=True,
        enable_track_a_semantic_resynthesis=True,
        enable_track_b_relabel=True,
    )
    with args.out.open("w", encoding="utf-8") as handle:
        for benchmark_dir in sorted((args.root / "benchmarks" / "track_a" / "public").iterdir()):
            benchmark = load_benchmark(benchmark_dir)
            backend = _load_backend(args.root, benchmark)
            candidates = compiler.candidate_compilations(benchmark.circuit, backend)
            selected = select_candidate(candidates, backend, benchmark.circuit)
            diagnostics = semantic_resynthesis_diagnostics(benchmark.circuit, backend, candidates)
            for candidate in candidates:
                metrics = candidate_metrics(candidate, backend, benchmark.circuit)
                validation = validate_compiled_circuit(candidate.circuit, backend, benchmark.track)
                equivalent = False
                noiseless = False
                if validation.valid:
                    layout = LayoutSidecar({logical.x: physical for logical, physical in candidate.layout.items()})
                    equivalent = validate_compiler_equivalence(benchmark, candidate.circuit, layout, seed=7).valid
                    noiseless = equivalent and _passes_noiseless_semantics(benchmark, candidate.circuit, seed=7)
                row = {
                    "sample": benchmark.name,
                    "candidate": candidate.name,
                    "semantic_candidate": candidate.name.startswith("v15_"),
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
                }
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            for diagnostic in diagnostics:
                row = {
                    "sample": benchmark.name,
                    "candidate": diagnostic["base_candidate"],
                    "semantic_candidate": True,
                    "diagnostic_only": True,
                    **diagnostic,
                }
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    args.comparison.write_text(_comparison(rows), encoding="utf-8")
    return 0


def _comparison(rows: list[dict[str, object]]) -> str:
    lines = ["# V15 A007/A008 Semantic Comparison", ""]
    for sample in ("track_a_007", "track_a_008"):
        sample_rows = [row for row in rows if row["sample"] == sample]
        selected = next((row for row in sample_rows if row.get("chosen")), None)
        semantic = [row for row in sample_rows if row.get("semantic_candidate") and not row.get("diagnostic_only")]
        diagnostics = [row for row in sample_rows if row.get("diagnostic_only")]
        lines.append(f"## {sample}")
        if selected:
            lines.append(
                f"- Selected: {selected['candidate']} | CX={selected['two_qubit_gates']} | "
                f"depth={selected['depth']} | ELR={float(selected['estimated_log_reliability']):.6f}"
            )
        lines.append(f"- Semantic candidates emitted: {len(semantic)}")
        for row in semantic[:10]:
            lines.append(
                f"- {row['candidate']} | chosen={row['chosen']} | valid={row['valid']} | "
                f"equiv={row['equivalent']} | noiseless={row['noiseless']} | "
                f"CX={row['two_qubit_gates']} | depth={row['depth']} | "
                f"ELR={float(row['estimated_log_reliability']):.6f}"
            )
        lines.append(f"- Semantic diagnostic rows: {len(diagnostics)}")
        for row in diagnostics[:10]:
            lines.append(
                f"- {row['detector']} on {row['base_candidate']} | reason={row['rejection_reason']} | "
                f"oldCX={row['old_cx']} | newCX={row.get('new_cx')} | "
                f"oldELR={float(row['old_elr']):.6f}"
            )
        lines.append("")
    return "\n".join(lines)


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
