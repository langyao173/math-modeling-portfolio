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
from noise_adaptive_hackathon.submission.phase_affine_cone_resynth import (
    find_phase_affine_intervals,
    phase_affine_cone_candidates,
    phase_affine_cone_diagnostics,
)
from noise_adaptive_hackathon.submission.selection import candidate_metrics, select_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump A007 phase-affine closed-cone diagnostics.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--sample", default="track_a_007")
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=ROOT / "experiments" / "phase_affine_candidates.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "experiments" / "a007_phase_affine_cone_report.md",
    )
    parser.add_argument(
        "--before-after",
        type=Path,
        default=ROOT / "experiments" / "a007_before_after_cone.md",
    )
    parser.add_argument(
        "--rejections",
        type=Path,
        default=ROOT / "experiments" / "rejected_reason_breakdown_phase_affine.csv",
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
    phase_candidates = phase_affine_cone_candidates(
        benchmark.circuit,
        backend,
        base_candidates,
        max_seconds=args.max_seconds,
    )
    diagnostics = phase_affine_cone_diagnostics(
        benchmark.circuit,
        backend,
        base_candidates,
        max_seconds=args.max_seconds,
    )
    intervals = find_phase_affine_intervals(list(benchmark.circuit.all_operations()), max_width=6, limit=12)

    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl.open("w", encoding="utf-8") as handle:
        for row in diagnostics:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    _write_rejection_breakdown(args.rejections, diagnostics)
    args.report.write_text(
        _render_report(
            args.sample,
            selected.name,
            selected_metrics,
            intervals,
            phase_candidates,
            diagnostics,
        ),
        encoding="utf-8",
    )
    args.before_after.write_text(
        _render_before_after(selected.name, selected_metrics, intervals, diagnostics),
        encoding="utf-8",
    )
    return 0


def _write_rejection_breakdown(path: Path, rows: list[dict[str, object]]) -> None:
    counts = Counter(str(row.get("rejection_reason", "unknown")) for row in rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reason", "count"])
        writer.writeheader()
        for reason, count in sorted(counts.items()):
            writer.writerow({"reason": reason, "count": count})


def _render_report(
    sample: str,
    selected_name: str,
    selected_metrics,
    intervals,
    phase_candidates,
    diagnostics: list[dict[str, object]],
) -> str:
    accepted_rows = [row for row in diagnostics if row.get("accepted")]
    best = _best_diagnostic(diagnostics)
    lines = [
        "# A007 Phase-Affine Cone Report",
        "",
        f"- Sample: {sample}",
        f"- Incumbent: {selected_name}",
        f"- Incumbent metrics: CX={selected_metrics.two_qubit_count}, depth={selected_metrics.depth}, ELR={selected_metrics.log_reliability:.12f}",
        f"- Phase-affine intervals detected: {len(intervals)}",
        f"- Candidate additions accepted vs local base: {len(phase_candidates)}",
        f"- Diagnostic rows beating incumbent: {len(accepted_rows)}",
        "",
        "## Cone Boundary Check",
        "",
    ]
    if intervals:
        first = intervals[0]
        lines.extend(
            [
                f"- Top interval: {first.start}:{first.stop} q={[qubit.x for qubit in first.qubits]}",
                f"- Top interval source CX: {sum(1 for op in first.operations if len(op.qubits) == 2)}",
                f"- Corrected closure: {first.closure_start}:{first.closure_stop} q={[qubit.x for qubit in first.closure_qubits]}",
                f"- Closure is phase-affine: {first.closure_phase_affine}",
                f"- Affine rows: {[format(row, f'0{len(first.qubits)}b') for row in first.signature.rows]}",
                f"- Affine const: {list(first.signature.const)}",
                f"- Phase terms: {[(format(mask, f'0{len(first.qubits)}b'), coeff) for mask, coeff in first.signature.phase_terms]}",
            ]
        )
    if best:
        lines.extend(
            [
                "",
                "## Best Routed Attempt",
                "",
                f"- Method: {best.get('method')}",
                f"- Source interval: {best.get('start')}:{best.get('stop')}",
                f"- New routed CX: {best.get('new_cx')}",
                f"- New routed depth: {best.get('new_depth')}",
                f"- New routed ELR: {best.get('new_elr')}",
                f"- Accepted: {best.get('accepted')}",
                f"- Reason: {best.get('rejection_reason')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            _decision_text(phase_candidates, accepted_rows, best),
            "",
        ]
    )
    return "\n".join(lines)


def _render_before_after(
    selected_name: str,
    selected_metrics,
    intervals,
    diagnostics: list[dict[str, object]],
) -> str:
    best = _best_diagnostic(diagnostics)
    lines = [
        "# A007 Before/After Cone",
        "",
        "## Before",
        "",
        f"- Candidate: {selected_name}",
        f"- CX: {selected_metrics.two_qubit_count}",
        f"- Depth: {selected_metrics.depth}",
        f"- ELR: {selected_metrics.log_reliability:.12f}",
        "",
        "## Extracted Cone",
        "",
    ]
    if intervals:
        first = intervals[0]
        lines.extend(
            [
                f"- Interval: {first.start}:{first.stop}",
                f"- Qubits: {[qubit.x for qubit in first.qubits]}",
                f"- Corrected closure: {first.closure_start}:{first.closure_stop}",
                f"- Closure qubits: {[qubit.x for qubit in first.closure_qubits]}",
                f"- Closure phase-affine: {first.closure_phase_affine}",
            ]
        )
    lines.extend(["", "## After", ""])
    if best:
        lines.extend(
            [
                f"- Method: {best.get('method')}",
                f"- CX: {best.get('new_cx')}",
                f"- Depth: {best.get('new_depth')}",
                f"- ELR: {best.get('new_elr')}",
                f"- Accepted: {best.get('accepted')}",
                f"- Rejection reason: {best.get('rejection_reason')}",
            ]
        )
    else:
        lines.append("- No routed phase-affine attempt was produced.")
    return "\n".join(lines) + "\n"


def _decision_text(phase_candidates, accepted_rows, best: dict[str, object] | None) -> str:
    if accepted_rows:
        return "Keep candidate for emulator evaluation; at least one phase-affine routed attempt beat the incumbent metric gate."
    if phase_candidates:
        return (
            "Diagnostic only: some phase-affine variants improved weaker local bases, but none beat the incumbent "
            "CX/ELR guard in the A007 diagnostic."
        )
    if best:
        return (
            "No-go for root promotion: phase-affine extraction succeeded, but the best exact routed attempt did not "
            "beat the incumbent CX/ELR guard."
        )
    return "No-go for root promotion: no exact phase-affine replacement was generated."


def _best_diagnostic(rows: list[dict[str, object]]) -> dict[str, object] | None:
    scored = [row for row in rows if row.get("new_elr") is not None and row.get("new_cx") is not None]
    if not scored:
        return None
    return max(scored, key=lambda row: (float(row["new_elr"]), -int(row["new_cx"]), -int(row.get("new_depth") or 999)))


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
