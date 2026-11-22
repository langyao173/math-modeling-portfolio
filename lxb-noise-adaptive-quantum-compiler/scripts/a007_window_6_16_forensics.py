from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cirq
import numpy as np

from common import ROOT

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.selection import candidate_metrics, select_candidate
from noise_adaptive_hackathon.submission.track_a_resynthesis.exact_gf2_equivalence import cnot_matrix_bits, matrix_rank
from noise_adaptive_hackathon.submission.track_a_resynthesis.phase_polynomial import phase_polynomial_replacements
from noise_adaptive_hackathon.submission.track_a_resynthesis.reversible_templates import reversible_template_replacements
from noise_adaptive_hackathon.submission.track_a_resynthesis.exact_equivalence import unitary_equivalent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forensic analysis for the A007 logical source window 6:16.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--start", type=int, default=6)
    parser.add_argument("--stop", type=int, default=16)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "a007_window_6_16_forensics.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "experiments" / "a007_window_6_16_forensics.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark_dir = args.root / "benchmarks" / "track_a" / "public" / "track_a_007"
    benchmark = load_benchmark(benchmark_dir)
    backend = _load_backend(args.root, benchmark)
    operations = list(benchmark.circuit.all_operations())
    window = tuple(operations[args.start : args.stop])
    cone_start, cone_stop, cone_qubits = _dependency_cone(operations, args.start, args.stop)
    cone = tuple(operations[cone_start:cone_stop])
    window_qubits = tuple(sorted({qubit for op in window for qubit in op.qubits}, key=lambda qubit: qubit.x))
    compiler = NoiseAdaptiveCompiler(enable_parity_resynthesis=True)
    candidates = compiler.candidate_compilations(benchmark.circuit, backend)
    selected = select_candidate(candidates, backend, benchmark.circuit)
    selected_metrics = candidate_metrics(selected, backend, benchmark.circuit)
    payload = {
        "sample": "track_a_007",
        "window": {
            "start": args.start,
            "stop": args.stop,
            "qubits": [qubit.x for qubit in window_qubits],
            "operations": [_op_record(op) for op in window],
            "dag_closed": cone_start == args.start and cone_stop == args.stop and set(cone_qubits) == set(window_qubits),
            "dependency_cone": {
                "start": cone_start,
                "stop": cone_stop,
                "qubits": [qubit.x for qubit in cone_qubits],
                "operations": [_op_record(op) for op in cone],
            },
            **_classify_window(window, window_qubits),
            **_unitary_payload(window, window_qubits),
            **_synthesis_attempts(window, window_qubits),
        },
        "current_routed_candidate": {
            "name": selected.name,
            "layout": {str(logical.x): physical for logical, physical in selected.layout.items()},
            "metrics": {
                "cx": selected_metrics.two_qubit_count,
                "depth": selected_metrics.depth,
                "elr": selected_metrics.log_reliability,
                "route_overhead": selected_metrics.route_overhead,
            },
            "window_physical_placement": {
                str(qubit.x): selected.layout.get(qubit)
                for qubit in window_qubits
            },
            "used_physical_edges": _used_edges(selected.circuit),
        },
    }
    payload["interpretation"] = _interpret(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.report.write_text(_render_report(payload), encoding="utf-8")
    return 0


def _dependency_cone(
    operations: list[cirq.Operation],
    start: int,
    stop: int,
) -> tuple[int, int, tuple[cirq.Qid, ...]]:
    cone_start, cone_stop = start, stop
    qubits: set[cirq.Qid] = set()
    changed = True
    while changed:
        previous = (cone_start, cone_stop, tuple(sorted(qubits, key=lambda qubit: qubit.x)))
        qubits = {qubit for op in operations[cone_start:cone_stop] for qubit in op.qubits}
        for index, op in enumerate(operations):
            if isinstance(op.gate, cirq.MeasurementGate):
                continue
            op_qubits = set(op.qubits)
            if op_qubits & qubits and (index < cone_start or index >= cone_stop):
                cone_start = min(cone_start, index)
                cone_stop = max(cone_stop, index + 1)
        changed = previous != (cone_start, cone_stop, tuple(sorted(qubits, key=lambda qubit: qubit.x)))
    qubits = {qubit for op in operations[cone_start:cone_stop] for qubit in op.qubits}
    return cone_start, cone_stop, tuple(sorted(qubits, key=lambda qubit: qubit.x))


def _classify_window(
    operations: tuple[cirq.Operation, ...],
    qubits: tuple[cirq.Qid, ...],
) -> dict[str, object]:
    names = [gate_name_for_operation(op) for op in operations]
    allowed_h_free = {"cx", "x", "z", "s", "sdg", "t", "tdg", "rz"}
    clifford = {"cx", "x", "y", "z", "h", "s", "sdg"}
    cnot_ops = [op for op in operations if gate_name_for_operation(op) == "cx"]
    gf2_matrix = None
    gf2_delta_rank = None
    if cnot_ops and all(name in allowed_h_free for name in names):
        matrix = cnot_matrix_bits(cnot_ops, qubits)
        identity = tuple(1 << index for index in range(len(qubits)))
        gf2_matrix = [format(row, f"0{len(qubits)}b") for row in matrix]
        gf2_delta_rank = matrix_rank(tuple(row ^ identity[index] for index, row in enumerate(matrix)))
    return {
        "gate_names": names,
        "old_cx": len(cnot_ops),
        "h_free": "h" not in names,
        "cnot_only": all(name == "cx" for name in names),
        "clifford_only": all(name in clifford for name in names),
        "clifford_t": all(name in clifford | {"t", "tdg", "rz"} for name in names),
        "basis_preserving_h_free": all(name in allowed_h_free for name in names),
        "gf2_matrix_rows": gf2_matrix,
        "gf2_delta_rank": gf2_delta_rank,
    }


def _unitary_payload(
    operations: tuple[cirq.Operation, ...],
    qubits: tuple[cirq.Qid, ...],
) -> dict[str, object]:
    unitary = cirq.Circuit(operations).unitary(qubit_order=qubits)
    flat = unitary.reshape(-1)
    pivot = next((value for value in flat if abs(value) > 1e-9), None)
    normalized = unitary if pivot is None else unitary / (pivot / abs(pivot))
    rounded = np.round(normalized.real, 10) + 1j * np.round(normalized.imag, 10)
    monomial = _monomial_profile(unitary)
    return {
        "unitary_hash": hashlib.sha256(rounded.tobytes()).hexdigest(),
        "unitary_matrix_rounded": [
            [[float(round(value.real, 6)), float(round(value.imag, 6))] for value in row]
            for row in unitary
        ],
        "permutation_diagonal_phase": monomial,
    }


def _monomial_profile(unitary: np.ndarray) -> dict[str, object]:
    permutation: list[int] = []
    phases: list[list[float]] = []
    truth_table: list[dict[str, object]] = []
    width = int(round(np.log2(unitary.shape[0])))
    for column in range(unitary.shape[1]):
        nonzero = [row for row in range(unitary.shape[0]) if abs(unitary[row, column]) > 1e-8]
        if len(nonzero) != 1:
            return {"is_monomial": False, "permutation": [], "phases": [], "truth_table": []}
        row = nonzero[0]
        permutation.append(row)
        value = unitary[row, column]
        phase = [float(round(value.real, 6)), float(round(value.imag, 6))]
        phases.append(phase)
        truth_table.append(
            {
                "input_int": column,
                "input_bits": format(column, f"0{width}b"),
                "output_int": row,
                "output_bits": format(row, f"0{width}b"),
                "phase": phase,
            }
        )
    return {
        "is_monomial": True,
        "permutation": permutation,
        "phases": phases,
        "truth_table": truth_table,
    }


def _synthesis_attempts(
    operations: tuple[cirq.Operation, ...],
    qubits: tuple[cirq.Qid, ...],
) -> dict[str, object]:
    attempts: list[dict[str, object]] = []
    for name, replacements in (
        ("reversible_template", reversible_template_replacements(operations, qubits)),
        ("phase_polynomial", phase_polynomial_replacements(operations, qubits)),
    ):
        for replacement in replacements[:6]:
            attempts.append(
                {
                    "method": name,
                    "cx": sum(1 for op in replacement if gate_name_for_operation(op) == "cx"),
                    "ops": [_op_record(op) for op in replacement],
                    "equivalent": unitary_equivalent(list(operations), replacement, qubits),
                }
            )
    return {
        "candidate_replacements": attempts,
        "best_candidate_cx": min((attempt["cx"] for attempt in attempts if attempt["equivalent"]), default=None),
    }


def _interpret(payload: dict[str, object]) -> str:
    window = payload["window"]
    if not window["dag_closed"]:
        return "The 6:16 interval is not dependency-cone closed; lower bounds on the fixed slice can be optimistic."
    best = window.get("best_candidate_cx")
    if best is not None and int(best) < int(window["old_cx"]):
        return "An exact local replacement exists and should be routed as a guarded candidate."
    if window["permutation_diagonal_phase"]["is_monomial"]:
        return "The window is monomial/permutation-times-phase, but current exact libraries did not find a CX-reducing replacement."
    return "No guarded local replacement was found; continue with logical reversible/carry resynthesis rather than deletion search."


def _render_report(payload: dict[str, object]) -> str:
    window = payload["window"]
    routed = payload["current_routed_candidate"]
    lines = [
        "# A007 Window 6:16 Forensics",
        "",
        f"- Qubits: {window['qubits']}",
        f"- Gates: {window['gate_names']}",
        f"- Old CX: {window['old_cx']}",
        f"- DAG closed: {window['dag_closed']}",
        f"- Dependency cone: {window['dependency_cone']['start']}:{window['dependency_cone']['stop']} q={window['dependency_cone']['qubits']}",
        f"- H-free: {window['h_free']}",
        f"- CNOT-only: {window['cnot_only']}",
        f"- Clifford-only: {window['clifford_only']}",
        f"- Clifford+T: {window['clifford_t']}",
        f"- Permutation x diagonal phase: {window['permutation_diagonal_phase']['is_monomial']}",
        f"- GF(2) matrix rows: {window['gf2_matrix_rows']}",
        f"- GF(2) delta rank: {window['gf2_delta_rank']}",
        f"- Unitary hash: `{window['unitary_hash']}`",
        f"- Current routed candidate: {routed['name']} metrics={routed['metrics']}",
        f"- Window physical placement: {routed['window_physical_placement']}",
        f"- Candidate replacements: {len(window['candidate_replacements'])}",
        f"- Best candidate CX: {window['best_candidate_cx']}",
        "",
        f"Interpretation: {payload['interpretation']}",
        "",
        "## Gate List",
        "",
    ]
    for index, op in enumerate(window["operations"]):
        lines.append(f"- {index}: {op}")
    truth_table = window["permutation_diagonal_phase"].get("truth_table", [])
    if truth_table:
        lines.extend(["", "## Truth Table / Phase", ""])
        for row in truth_table:
            lines.append(
                f"- {row['input_bits']} -> {row['output_bits']} phase={row['phase']}"
            )
    lines.append("")
    return "\n".join(lines)


def _op_record(op: cirq.Operation) -> dict[str, object]:
    return {
        "text": str(op),
        "gate": gate_name_for_operation(op),
        "qubits": [qubit.x for qubit in op.qubits],
    }


def _used_edges(circuit: cirq.Circuit) -> list[list[int]]:
    edges: list[list[int]] = []
    for op in circuit.all_operations():
        if gate_name_for_operation(op) == "cx":
            edges.append([op.qubits[0].x, op.qubits[1].x])
    return edges


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
