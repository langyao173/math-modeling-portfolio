from __future__ import annotations

import argparse
import csv
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
from noise_adaptive_hackathon.submission.track_a_resynthesis.bounded_window_synth import (
    bounded_window_synthesis_candidates,
    bounded_window_synthesis_diagnostics,
    rejection_reason_counts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump bounded exact window synthesis attempts for Track A pressure samples.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "window_synth_candidates.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "experiments" / "a007_a008_window_synth_report.md")
    parser.add_argument("--rejected", type=Path, default=ROOT / "experiments" / "rejected_reason_breakdown.csv")
    parser.add_argument("--sample", action="append", default=["track_a_007", "track_a_008"])
    parser.add_argument("--max-seconds", type=float, default=12.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    with args.out.open("w", encoding="utf-8") as handle:
        for sample in args.sample:
            benchmark_dir = args.root / "benchmarks" / "track_a" / "public" / sample
            if not benchmark_dir.exists():
                continue
            benchmark = load_benchmark(benchmark_dir)
            backend = _load_backend(args.root, benchmark)
            compiler = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_track_a_resynthesis=True,
                enable_track_a_semantic_resynthesis=False,
                enable_track_b_relabel=True,
            )
            base_candidates = compiler.candidate_compilations(benchmark.circuit, backend)
            selected = select_candidate(base_candidates, backend, benchmark.circuit)
            diagnostics.extend(
                {
                    "sample": sample,
                    **row,
                }
                for row in bounded_window_synthesis_diagnostics(
                    benchmark.circuit,
                    backend,
                    base_candidates,
                    max_seconds=args.max_seconds,
                )
            )
            additions = bounded_window_synthesis_candidates(
                benchmark.circuit,
                backend,
                base_candidates,
                max_seconds=args.max_seconds,
            )
            for candidate in additions:
                row = _candidate_row(sample, benchmark, backend, selected, candidate)
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if not additions:
                row = {
                    "sample": sample,
                    "accepted": False,
                    "selected_base": selected.name,
                    "reason": "no_accepted_window_candidate",
                }
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    args.report.write_text(_report(rows, diagnostics), encoding="utf-8")
    _write_rejections(args.rejected, diagnostics)
    return 0


def _candidate_row(sample: str, benchmark, backend: CompetitionBackend, selected, candidate) -> dict[str, object]:
    metrics = candidate_metrics(candidate, backend, benchmark.circuit)
    selected_metrics = candidate_metrics(selected, backend, benchmark.circuit)
    validation = validate_compiled_circuit(candidate.circuit, backend, benchmark.track)
    equivalent = False
    noiseless = False
    errors = list(validation.errors)
    if validation.valid:
        layout = LayoutSidecar({logical.x: physical for logical, physical in candidate.layout.items()})
        equivalence = validate_compiler_equivalence(benchmark, candidate.circuit, layout, seed=7)
        equivalent = equivalence.valid
        errors.extend(equivalence.errors)
        noiseless = equivalent and _passes_noiseless_semantics(benchmark, candidate.circuit, seed=7)
    return {
        "sample": sample,
        "accepted": validation.valid and equivalent and noiseless,
        "candidate": candidate.name,
        "selected_base": selected.name,
        "old_cx": selected_metrics.two_qubit_count,
        "new_cx": metrics.two_qubit_count,
        "old_depth": selected_metrics.depth,
        "new_depth": metrics.depth,
        "old_elr": selected_metrics.log_reliability,
        "new_elr": metrics.log_reliability,
        "valid": validation.valid,
        "equivalent": equivalent,
        "noiseless": noiseless,
        "errors": errors,
    }


def _report(rows: list[dict[str, object]], diagnostics: list[dict[str, object]]) -> str:
    lines = ["# A007/A008 Bounded Window Synthesis Report", ""]
    for sample in ("track_a_007", "track_a_008"):
        sample_rows = [row for row in rows if row.get("sample") == sample]
        sample_diag = [row for row in diagnostics if row.get("sample") == sample]
        accepted = [row for row in sample_rows if row.get("accepted") is True]
        lines.append(f"## {sample}")
        lines.append(f"- Accepted candidates: {len(accepted)}")
        for row in accepted[:10]:
            lines.append(
                f"- {row['candidate']} | CX {row['old_cx']} -> {row['new_cx']} | "
                f"depth {row['old_depth']} -> {row['new_depth']} | "
                f"ELR {float(row['old_elr']):.6f} -> {float(row['new_elr']):.6f}"
            )
        counts = rejection_reason_counts([row for row in sample_diag if row.get("accepted") is not True])
        if counts:
            lines.append("- Rejection breakdown: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
        else:
            lines.append("- Rejection breakdown: none")
        lines.append("")
    return "\n".join(lines)


def _write_rejections(path: Path, diagnostics: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample", "rejection_reason", "count"]
    counts: dict[tuple[str, str], int] = {}
    for row in diagnostics:
        if row.get("accepted") is True:
            continue
        key = (str(row.get("sample")), str(row.get("rejection_reason", "unknown")))
        counts[key] = counts.get(key, 0) + 1
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (sample, reason), count in sorted(counts.items()):
            writer.writerow({"sample": sample, "rejection_reason": reason, "count": count})


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
