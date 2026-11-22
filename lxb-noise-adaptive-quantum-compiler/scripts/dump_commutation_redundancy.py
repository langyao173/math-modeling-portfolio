from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.evaluator.equivalence import LayoutSidecar, validate_compiler_equivalence
from noise_adaptive_hackathon.evaluator.submission import _passes_noiseless_semantics
from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation, validate_compiled_circuit
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.selection import candidate_metrics
from noise_adaptive_hackathon.submission.track_a_resynthesis.bounded_window_synth import (
    _window_replacements,
    high_cost_windows,
)
from noise_adaptive_hackathon.submission.track_a_resynthesis.commutation_aware_redundancy import (
    classify_rewrite,
    commutation_aware_redundancy_candidates,
    commutation_aware_redundancy_diagnostics,
    local_unitary_hash,
)
from noise_adaptive_hackathon.submission.track_a_resynthesis.exact_equivalence import unitary_equivalent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump commutation-aware redundancy rewrite diagnostics.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--sample", action="append", default=["track_a_007", "track_a_008"])
    parser.add_argument("--max-seconds", type=float, default=2.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "commutation_redundancy_candidates.jsonl",
    )
    parser.add_argument(
        "--dump",
        type=Path,
        default=ROOT / "experiments" / "a008_window_synth_13_rewrite_dump.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "experiments" / "commutation_redundancy_report.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    first_a008_dump: dict[str, object] | None = None
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
            sample_diagnostics = commutation_aware_redundancy_diagnostics(
                benchmark.circuit,
                backend,
                base_candidates,
                max_seconds=args.max_seconds,
            )
            diagnostics.extend({"sample": sample, **row} for row in sample_diagnostics)
            additions = commutation_aware_redundancy_candidates(
                benchmark.circuit,
                backend,
                base_candidates,
                max_seconds=args.max_seconds,
            )
            for candidate in additions:
                row = _candidate_row(sample, benchmark, backend, candidate)
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if not additions:
                row = {"sample": sample, "accepted": False, "reason": "no_accepted_commutation_redundancy"}
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if sample == "track_a_008" and first_a008_dump is None:
                accepted_diag = next((row for row in sample_diagnostics if row.get("accepted") is True), None)
                if accepted_diag is not None:
                    first_a008_dump = _bounded_window_reference_dump(base_candidates, backend)
                    if first_a008_dump is not None:
                        first_a008_dump["distilled_candidate"] = accepted_diag
    if first_a008_dump is not None:
        args.dump.write_text(json.dumps(first_a008_dump, indent=2, sort_keys=True), encoding="utf-8")
    args.report.write_text(_report(rows, diagnostics, first_a008_dump), encoding="utf-8")
    return 0


def _candidate_row(sample: str, benchmark, backend: CompetitionBackend, candidate) -> dict[str, object]:
    metrics = candidate_metrics(candidate, backend, benchmark.circuit)
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
        "cx": metrics.two_qubit_count,
        "depth": metrics.depth,
        "elr": metrics.log_reliability,
        "valid": validation.valid,
        "equivalent": equivalent,
        "noiseless": noiseless,
        "errors": errors,
    }


def _bounded_window_reference_dump(base_candidates, backend: CompetitionBackend) -> dict[str, object] | None:
    base = next((candidate for candidate in base_candidates if candidate.name == "exact_small_router_r0_l0"), None)
    if base is None:
        return None
    for window in high_cost_windows(base.circuit, backend):
        if window.start != 13:
            continue
        replacement = next(
            (item for item in _window_replacements(window, backend, max_seconds=5.0) if item.reason == "delete_redundant_ops"),
            None,
        )
        if replacement is None:
            continue
        deleted_indices = _deleted_indices(tuple(window.operations), tuple(replacement.operations))
        edges = [
            [op.qubits[0].x, op.qubits[1].x]
            for op in window.operations
            if gate_name_for_operation(op) == "cx"
        ]
        return {
            "sample": "track_a_008",
            "rewrite": "bounded_window_reference",
            "legacy_window_label": "window_synth_13",
            "base_candidate": base.name,
            "start": window.start,
            "stop": window.stop,
            "physical_qubits": [qubit.x for qubit in window.qubits],
            "edges": edges,
            "before_operations": [_operation_record(op) for op in window.operations],
            "after_operations": [_operation_record(op) for op in replacement.operations],
            "deleted_indices": list(deleted_indices),
            "deleted_operations": [_operation_record(window.operations[index]) for index in deleted_indices],
            "classification": classify_rewrite(tuple(window.operations), deleted_indices),
            "local_unitary_hash_before": local_unitary_hash(tuple(window.operations), window.qubits),
            "local_unitary_hash_after": local_unitary_hash(tuple(replacement.operations), window.qubits),
            "exact_equivalent": unitary_equivalent(list(window.operations), list(replacement.operations), window.qubits),
        }
    return None


def _deleted_indices(
    before: tuple[object, ...],
    after: tuple[object, ...],
) -> tuple[int, ...]:
    deleted: list[int] = []
    after_index = 0
    for before_index, op in enumerate(before):
        if after_index < len(after) and str(op) == str(after[after_index]):
            after_index += 1
        else:
            deleted.append(before_index)
    return tuple(deleted)


def _operation_record(op) -> dict[str, object]:
    return {
        "text": str(op),
        "gate": gate_name_for_operation(op),
        "qubits": [qubit.x for qubit in op.qubits],
    }


def _report(
    rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    dump: dict[str, object] | None,
) -> str:
    lines = ["# Commutation-Aware Redundancy Report", ""]
    for sample in ("track_a_007", "track_a_008"):
        sample_rows = [row for row in rows if row.get("sample") == sample]
        accepted = [row for row in sample_rows if row.get("accepted") is True]
        sample_diagnostics = [row for row in diagnostics if row.get("sample") == sample]
        lines.append(f"## {sample}")
        lines.append(f"- Accepted candidates: {len(accepted)}")
        for row in accepted:
            lines.append(
                f"- {row['candidate']} | CX {row['cx']} | depth {row['depth']} | ELR {float(row['elr']):.6f}"
            )
        reasons: dict[str, int] = {}
        for row in sample_diagnostics:
            if row.get("accepted") is True:
                continue
            reason = str(row.get("rejection_reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
        if reasons:
            lines.append("- Rejections: " + ", ".join(f"{key}={value}" for key, value in sorted(reasons.items())))
        else:
            lines.append("- Rejections: none")
        lines.append("")
    if dump is not None:
        lines.append("## A008 Rewrite Classification")
        lines.append(f"- Classification: {dump['classification']}")
        lines.append(f"- Deleted operations: {dump['deleted_operations']}")
        lines.append(f"- Local unitary hash before: `{dump['local_unitary_hash_before']}`")
        lines.append(f"- Local unitary hash after: `{dump['local_unitary_hash_after']}`")
        lines.append(f"- Exact equivalence proof result: {dump['exact_equivalent']}")
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
