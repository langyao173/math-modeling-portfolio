from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.selection import rank_candidates_by_proxy
from noise_adaptive_hackathon.submission.track_a_resynthesis.local_lower_bound_analyzer import (
    analyze_local_lower_bounds,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze local CX lower bounds on Track A pressure samples.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--sample", action="append", default=["track_a_007", "track_a_008"])
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "local_lower_bound_a007_a008.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "experiments" / "local_lower_bound_report.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with args.out.open("w", encoding="utf-8") as handle:
        for sample in args.sample:
            benchmark_dir = args.root / "benchmarks" / "track_a" / "public" / sample
            if not benchmark_dir.exists():
                continue
            benchmark = load_benchmark(benchmark_dir)
            backend = _load_backend(args.root, benchmark)
            circuits = [("source", benchmark.circuit)]
            compiler = NoiseAdaptiveCompiler(enable_parity_resynthesis=True, enable_exact_small_router=True)
            candidates = compiler.candidate_compilations(benchmark.circuit, backend)
            for candidate in rank_candidates_by_proxy(candidates, backend, benchmark.circuit)[:4]:
                circuits.append((candidate.name, candidate.circuit))
            for circuit_label, circuit in circuits:
                for record in analyze_local_lower_bounds(circuit, backend, limit=args.limit):
                    row = {"sample": sample, "circuit_label": circuit_label, **record.to_json()}
                    rows.append(row)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
    args.report.write_text(_render_report(rows), encoding="utf-8")
    return 0


def _render_report(rows: list[dict[str, object]]) -> str:
    lines = ["# Local Lower-Bound Analyzer Report", ""]
    for sample in ("track_a_007", "track_a_008"):
        sample_rows = [row for row in rows if row.get("sample") == sample]
        lines.append(f"## {sample}")
        if not sample_rows:
            lines.append("- No high-cost windows found.")
            lines.append("")
            continue
        no_improvement = [
            row
            for row in sample_rows
            if str(row.get("certificate", "")).endswith("no_cx_reduction")
        ]
        open_rows = [row for row in sample_rows if row.get("possible_cx_reduction")]
        lines.append(f"- Windows analyzed: {len(sample_rows)}")
        lines.append(f"- No-improvement certificates: {len(no_improvement)}")
        lines.append(f"- Windows where CX reduction is not ruled out: {len(open_rows)}")
        for row in sample_rows[:12]:
            lines.append(
                "- {circuit_label} window {start}:{stop} q={qubits} old_cx={old_cx} "
                "gf2={gf2_rank_lower_bound} cut={interaction_cut_lower_bound} "
                "template={template_lower_bound} exact={exact_minimal_cx} "
                "possible={possible_cx_reduction} cert={certificate}".format(**row)
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
