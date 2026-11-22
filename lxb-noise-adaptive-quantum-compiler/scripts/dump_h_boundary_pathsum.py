from __future__ import annotations

from collections import Counter
import argparse
import csv
import json
from pathlib import Path

from common import ROOT

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.h_boundary_pathsum_resynth import (
    find_h_boundary_cones,
    h_boundary_pathsum_candidates,
    h_boundary_pathsum_diagnostics,
)
from noise_adaptive_hackathon.submission.selection import candidate_metrics, select_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump H-boundary path-sum diagnostics for Track A.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--sample", default="track_a_007")
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=ROOT / "experiments" / "h_boundary_pathsum_candidates.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "experiments" / "a007_h_boundary_pathsum_report.md",
    )
    parser.add_argument(
        "--rejections",
        type=Path,
        default=ROOT / "experiments" / "rejected_reason_breakdown_h_boundary_pathsum.csv",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark_dir = args.root / "benchmarks" / "track_a" / "public" / args.sample
    benchmark = load_benchmark(benchmark_dir)
    backend = _load_backend(args.root, benchmark)
    compiler = NoiseAdaptiveCompiler(
        enable_parity_resynthesis=True,
        enable_exact_small_router=True,
        enable_route_scar_normalizer=True,
    )
    base_candidates = compiler.candidate_compilations(benchmark.circuit, backend)
    selected = select_candidate(base_candidates, backend, benchmark.circuit)
    selected_metrics = candidate_metrics(selected, backend, benchmark.circuit)
    additions = h_boundary_pathsum_candidates(
        benchmark.circuit,
        backend,
        base_candidates,
        max_seconds=args.max_seconds,
    )
    diagnostics = h_boundary_pathsum_diagnostics(
        benchmark.circuit,
        backend,
        base_candidates,
        max_seconds=args.max_seconds,
    )
    cones = find_h_boundary_cones(list(benchmark.circuit.all_operations()), limit=4)

    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl.open("w", encoding="utf-8") as handle:
        for row in diagnostics:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write_breakdown(args.rejections, diagnostics)
    args.report.write_text(
        _render_report(args.sample, selected.name, selected_metrics, cones, additions, diagnostics),
        encoding="utf-8",
    )
    return 0


def _render_report(
    sample: str,
    selected_name: str,
    selected_metrics,
    cones,
    additions,
    diagnostics: list[dict[str, object]],
) -> str:
    accepted = [row for row in diagnostics if row.get("accepted")]
    best = _best_row(diagnostics)
    lines = [
        "# A007 H-Boundary Path-Sum Report",
        "",
        f"- Sample: {sample}",
        f"- Incumbent: {selected_name}",
        f"- Incumbent metrics: CX={selected_metrics.two_qubit_count}, depth={selected_metrics.depth}, ELR={selected_metrics.log_reliability:.12f}",
        f"- H-boundary cones detected: {len(cones)}",
        f"- Candidate additions accepted vs local base: {len(additions)}",
        f"- Diagnostic rows beating incumbent: {len(accepted)}",
        "",
        "## Path-Sum Cone",
        "",
    ]
    if cones:
        cone = cones[0]
        sig = cone.signature
        lines.extend(
            [
                f"- Cone: {cone.start}:{cone.stop} q={[qubit.x for qubit in cone.qubits]}",
                f"- Cone CX: {sum(1 for op in cone.operations if len(op.qubits) == 2)}",
                f"- H count / path variables: {sig.h_count} / {sig.num_paths}",
                f"- Output input rows: {[format(row, f'0{sig.num_inputs}b') for row in sig.output_input_rows]}",
                f"- Output path rows: {[format(row, f'0{sig.num_paths}b') if sig.num_paths else '' for row in sig.output_path_rows]}",
                f"- Output const: {list(sig.output_const)}",
                f"- Phase/path terms: {len(sig.phase_terms)}",
                f"- H-free phase-affine cores inside cone: {len(cone.phase_affine_intervals)}",
            ]
        )
    if best:
        lines.extend(
            [
                "",
                "## Best Routed Attempt",
                "",
                f"- Method: {best.get('method')}",
                f"- Interval: {best.get('interval_start')}:{best.get('interval_stop')}",
                f"- New routed CX: {best.get('new_cx')}",
                f"- New routed depth: {best.get('new_depth')}",
                f"- New routed ELR: {best.get('new_elr')}",
                f"- Accepted: {best.get('accepted')}",
                f"- Reason: {best.get('rejection_reason')}",
            ]
        )
    lines.extend(["", "## Decision", "", _decision(additions, accepted, best), ""])
    return "\n".join(lines)


def _decision(additions, accepted: list[dict[str, object]], best: dict[str, object] | None) -> str:
    if accepted:
        return "Keep candidate for emulator evaluation; at least one H-boundary path-sum attempt beat the incumbent."
    if additions:
        return (
            "Diagnostic only: H-boundary variants improved weaker local bases, but none beat the incumbent A007 "
            "CX/ELR guard."
        )
    if best:
        return "No-go for promotion: path-sum extraction succeeded, but best exact routed attempt failed the metric guard."
    return "No-go for promotion: no exact H-boundary replacement was generated."


def _best_row(rows: list[dict[str, object]]) -> dict[str, object] | None:
    scored = [row for row in rows if row.get("new_elr") is not None and row.get("new_cx") is not None]
    if not scored:
        return None
    return max(scored, key=lambda row: (float(row["new_elr"]), -int(row["new_cx"]), -int(row.get("new_depth") or 999)))


def _write_breakdown(path: Path, rows: list[dict[str, object]]) -> None:
    counts = Counter(str(row.get("rejection_reason", "unknown")) for row in rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reason", "count"])
        writer.writeheader()
        for reason, count in sorted(counts.items()):
            writer.writerow({"reason": reason, "count": count})


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
