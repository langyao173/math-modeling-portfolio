from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .exact_gf2_equivalence import cnot_matrix_bits, identity_matrix_bits, matrix_rank
from .reversible_templates import reversible_template_replacements


@dataclass(frozen=True)
class LocalLowerBoundRecord:
    start: int
    stop: int
    width: int
    qubits: tuple[int, ...]
    family: str
    old_cx: int
    gf2_rank_lower_bound: int | None
    interaction_cut_lower_bound: int | None
    template_lower_bound: int | None
    exact_minimal_cx: int | None
    possible_cx_reduction: bool
    certificate: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def analyze_local_lower_bounds(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    *,
    limit: int = 24,
) -> list[LocalLowerBoundRecord]:
    records: list[LocalLowerBoundRecord] = []
    del backend
    for window in _logical_high_cost_windows(circuit, limit=limit, max_width=4):
        operations = window["operations"]
        qubits = window["qubits"]
        old_cx = int(window["two_qubit_count"])
        gf2_bound = _gf2_rank_lower_bound(operations, qubits)
        cut_bound = _interaction_cut_lower_bound(operations, qubits)
        template_bound = _template_lower_bound(operations, qubits)
        exact_min = _exact_minimal_cnot_count(operations, qubits, old_cx)
        bounds = [
            value
            for value in (gf2_bound, cut_bound, template_bound, exact_min)
            if value is not None
        ]
        possible = any(value < old_cx for value in bounds)
        certificate = _certificate(old_cx, bounds, exact_min, operations)
        records.append(
            LocalLowerBoundRecord(
                start=int(window["start"]),
                stop=int(window["stop"]),
                width=len(qubits),
                qubits=tuple(qubit.x for qubit in qubits),
                family=str(window["family"]),
                old_cx=old_cx,
                gf2_rank_lower_bound=gf2_bound,
                interaction_cut_lower_bound=cut_bound,
                template_lower_bound=template_bound,
                exact_minimal_cx=exact_min,
                possible_cx_reduction=possible,
                certificate=certificate,
            )
        )
    return records


def _logical_high_cost_windows(
    circuit: cirq.Circuit,
    *,
    limit: int,
    max_width: int,
) -> list[dict[str, object]]:
    operations = list(circuit.all_operations())
    windows: list[dict[str, object]] = []
    for start in range(len(operations)):
        if isinstance(operations[start].gate, cirq.MeasurementGate):
            break
        for stop in range(start + 4, min(len(operations), start + 22) + 1):
            block = tuple(operations[start:stop])
            if any(isinstance(op.gate, cirq.MeasurementGate) for op in block):
                break
            cx_count = _cx_count(block)
            if cx_count < 2:
                continue
            qubits = tuple(sorted({qubit for op in block for qubit in op.qubits}, key=lambda qubit: qubit.x))
            if not 3 <= len(qubits) <= max_width:
                continue
            family = _window_family(block)
            score = cx_count + (0.25 if family in {"h_free_phase", "clifford_t_scaffold"} else 0.0)
            windows.append(
                {
                    "start": start,
                    "stop": stop,
                    "operations": block,
                    "qubits": qubits,
                    "two_qubit_count": cx_count,
                    "family": family,
                    "score": score,
                }
            )
    deduped: list[dict[str, object]] = []
    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    for window in sorted(windows, key=lambda row: (-float(row["score"]), int(row["start"]), int(row["stop"]))):
        signature = (
            int(window["start"]),
            int(window["stop"]),
            tuple(qubit.x for qubit in window["qubits"]),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(window)
        if len(deduped) >= limit:
            break
    return deduped


def _window_family(operations: tuple[cirq.Operation, ...]) -> str:
    names = [gate_name_for_operation(op) for op in operations]
    if "h" not in names and all(name in {"cx", "x", "z", "s", "sdg", "t", "tdg", "rz"} for name in names):
        return "h_free_phase" if any(name in {"t", "tdg", "rz"} for name in names) else "h_free_linear"
    if any(name in {"t", "tdg", "rz"} for name in names):
        return "clifford_t_scaffold"
    return "general"


def _gf2_rank_lower_bound(
    operations: tuple[cirq.Operation, ...],
    qubits: tuple[cirq.Qid, ...],
) -> int | None:
    if not _h_free_basis_preserving(operations):
        return None
    cnot_ops = [op for op in operations if gate_name_for_operation(op) == "cx"]
    if not cnot_ops:
        return 0
    matrix = cnot_matrix_bits(cnot_ops, qubits)
    identity = identity_matrix_bits(len(qubits))
    delta = tuple(row ^ identity[index] for index, row in enumerate(matrix))
    return matrix_rank(delta)


def _interaction_cut_lower_bound(
    operations: tuple[cirq.Operation, ...],
    qubits: tuple[cirq.Qid, ...],
) -> int | None:
    if not _h_free_basis_preserving(operations):
        return None
    cnot_ops = [op for op in operations if gate_name_for_operation(op) == "cx"]
    if not cnot_ops:
        return 0
    matrix = cnot_matrix_bits(cnot_ops, qubits)
    width = len(qubits)
    best = 0
    for mask in range(1, (1 << width) - 1):
        if mask > ((1 << width) - 1) ^ mask:
            continue
        crossing_rows: list[int] = []
        for row_index, row in enumerate(matrix):
            row_side = (mask >> row_index) & 1
            outside = row & (~mask if row_side else mask)
            crossing_rows.append(outside & ((1 << width) - 1))
        best = max(best, matrix_rank(tuple(crossing_rows)))
    return best


def _template_lower_bound(
    operations: tuple[cirq.Operation, ...],
    qubits: tuple[cirq.Qid, ...],
) -> int | None:
    replacements = reversible_template_replacements(operations, qubits)
    if not replacements:
        return None
    return min(_cx_count(tuple(replacement)) for replacement in replacements)


def _exact_minimal_cnot_count(
    operations: tuple[cirq.Operation, ...],
    qubits: tuple[cirq.Qid, ...],
    old_cx: int,
) -> int | None:
    if old_cx <= 0 or old_cx > 9 or not all(gate_name_for_operation(op) == "cx" for op in operations):
        return None
    target = cnot_matrix_bits(list(operations), qubits)
    identity = identity_matrix_bits(len(qubits))
    if target == identity:
        return 0
    moves = tuple((control, target_) for control in range(len(qubits)) for target_ in range(len(qubits)) if control != target_)
    queue: deque[tuple[tuple[int, ...], int]] = deque([(identity, 0)])
    seen = {identity}
    while queue:
        matrix, depth = queue.popleft()
        if depth >= old_cx - 1:
            continue
        for control, target_ in moves:
            updated = _apply_cnot(matrix, control, target_)
            if updated in seen:
                continue
            if updated == target:
                return depth + 1
            seen.add(updated)
            queue.append((updated, depth + 1))
    return None


def _apply_cnot(matrix: tuple[int, ...], control: int, target: int) -> tuple[int, ...]:
    rows = list(matrix)
    rows[target] ^= rows[control]
    return tuple(rows)


def _h_free_basis_preserving(operations: tuple[cirq.Operation, ...]) -> bool:
    allowed = {"cx", "x", "z", "s", "sdg", "t", "tdg", "rz"}
    return all(gate_name_for_operation(op) in allowed for op in operations)


def _certificate(
    old_cx: int,
    bounds: list[int],
    exact_min: int | None,
    operations: tuple[cirq.Operation, ...],
) -> str:
    if exact_min is not None and exact_min >= old_cx:
        return "exact_minimal_no_cx_reduction"
    if bounds and max(bounds) >= old_cx and _h_free_basis_preserving(operations):
        return "lower_bound_tight_no_cx_reduction"
    if any(value < old_cx for value in bounds):
        return "cx_reduction_not_ruled_out"
    return "insufficient_certificate"


def _cx_count(operations: tuple[cirq.Operation, ...]) -> int:
    return sum(1 for op in operations if gate_name_for_operation(op) == "cx")


__all__ = ["LocalLowerBoundRecord", "analyze_local_lower_bounds"]
