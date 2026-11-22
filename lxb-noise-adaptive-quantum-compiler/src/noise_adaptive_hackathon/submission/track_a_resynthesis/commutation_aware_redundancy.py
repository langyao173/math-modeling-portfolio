from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import time

import cirq
import numpy as np

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from ..selection import CompilationCandidate, candidate_metrics, rank_candidates_by_proxy
from ..track_b import is_clifford_only
from ..validation import validate_or_raise
from .exact_equivalence import unitary_equivalent


@dataclass(frozen=True)
class RedundancyRewrite:
    start: int
    stop: int
    qubits: tuple[cirq.Qid, ...]
    operations: tuple[cirq.Operation, ...]
    replacement: tuple[cirq.Operation, ...]
    deleted_indices: tuple[int, ...]
    classification: str
    exact_equivalent: bool

    @property
    def deleted_operations(self) -> tuple[cirq.Operation, ...]:
        return tuple(self.operations[index] for index in self.deleted_indices)


def commutation_aware_redundancy_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    *,
    limit: int = 1,
    max_seconds: float = 2.0,
) -> list[CompilationCandidate]:
    if is_clifford_only(source) or not candidates:
        return []
    started = time.perf_counter()
    additions: list[CompilationCandidate] = []
    seen: set[tuple[str, ...]] = set()
    ranked = rank_candidates_by_proxy(candidates, backend, source)[:4]
    for base in ranked:
        if time.perf_counter() - started >= max_seconds:
            break
        for rewrite in find_redundancy_rewrites(
            base.circuit,
            backend,
            max_seconds=max(0.05, max_seconds - (time.perf_counter() - started)),
            limit=1,
        ):
            if time.perf_counter() - started >= max_seconds:
                break
            rewritten = _replace_window(base.circuit, rewrite)
            signature = tuple(str(op) for op in rewritten.all_operations())
            if signature in seen:
                continue
            seen.add(signature)
            try:
                validate_or_raise(source, rewritten, backend)
            except Exception:
                continue
            trial = CompilationCandidate(
                f"{base.name}_commute_redundancy_{rewrite.start}_{rewrite.classification.replace(' ', '_')}",
                rewritten,
                dict(base.layout),
                base.compile_time_sec + time.perf_counter() - started,
            )
            if _accepted_metric_delta(trial, base, backend, source):
                additions.append(trial)
                if len(additions) >= limit:
                    return additions
    return additions


def commutation_aware_redundancy_diagnostics(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    *,
    max_seconds: float = 2.0,
) -> list[dict[str, object]]:
    if is_clifford_only(source) or not candidates:
        return []
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for base in rank_candidates_by_proxy(candidates, backend, source)[:4]:
        if time.perf_counter() - started >= max_seconds:
            break
        base_metrics = candidate_metrics(base, backend, source)
        rewrites = find_redundancy_rewrites(
            base.circuit,
            backend,
            max_seconds=max(0.05, max_seconds - (time.perf_counter() - started)),
            limit=8,
        )
        if not rewrites:
            rows.append(
                {
                    "detector": "commutation_aware_redundancy",
                    "base_candidate": base.name,
                    "accepted": False,
                    "rejection_reason": "no_redundant_pair",
                    "old_cx": base_metrics.two_qubit_count,
                    "old_depth": base_metrics.depth,
                    "old_elr": base_metrics.log_reliability,
                }
            )
            continue
        for rewrite in rewrites:
            rewritten = _replace_window(base.circuit, rewrite)
            trial = CompilationCandidate(
                "diagnostic",
                rewritten,
                dict(base.layout),
                base.compile_time_sec,
            )
            trial_metrics = candidate_metrics(trial, backend, source)
            valid = False
            metric_guard = False
            reason = "accepted"
            try:
                validate_or_raise(source, rewritten, backend)
                valid = True
                metric_guard = _accepted_metric_delta(trial, base, backend, source)
                if not metric_guard:
                    reason = "metric_guard_failed"
            except Exception as error:
                reason = type(error).__name__
            rows.append(
                {
                    "detector": "commutation_aware_redundancy",
                    "base_candidate": base.name,
                    "accepted": valid and metric_guard,
                    "valid": valid,
                    "metric_guard": metric_guard,
                    "rejection_reason": reason,
                    "start": rewrite.start,
                    "stop": rewrite.stop,
                    "width": len(rewrite.qubits),
                    "physical_qubits": [qubit.x for qubit in rewrite.qubits],
                    "edges": _edge_list(rewrite.operations),
                    "deleted_indices": list(rewrite.deleted_indices),
                    "deleted_operations": [_operation_record(op) for op in rewrite.deleted_operations],
                    "before_operations": [_operation_record(op) for op in rewrite.operations],
                    "after_operations": [_operation_record(op) for op in rewrite.replacement],
                    "classification": rewrite.classification,
                    "exact_equivalent": rewrite.exact_equivalent,
                    "local_unitary_hash_before": local_unitary_hash(rewrite.operations, rewrite.qubits),
                    "local_unitary_hash_after": local_unitary_hash(rewrite.replacement, rewrite.qubits),
                    "old_cx": base_metrics.two_qubit_count,
                    "old_depth": base_metrics.depth,
                    "old_elr": base_metrics.log_reliability,
                    "new_cx": trial_metrics.two_qubit_count,
                    "new_depth": trial_metrics.depth,
                    "new_elr": trial_metrics.log_reliability,
                }
            )
    return rows


def find_redundancy_rewrites(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    *,
    limit: int = 4,
    max_seconds: float = 1.0,
) -> list[RedundancyRewrite]:
    started = time.perf_counter()
    rewrites: list[RedundancyRewrite] = []
    seen: set[tuple[int, int, tuple[int, ...], tuple[str, ...]]] = set()
    for start, stop, qubits, operations, score in _candidate_windows(circuit, backend):
        del score
        if time.perf_counter() - started >= max_seconds:
            break
        for deleted_indices in _priority_deletion_pairs(operations)[:64]:
            if time.perf_counter() - started >= max_seconds:
                break
            replacement = tuple(
                op for index, op in enumerate(operations) if index not in set(deleted_indices)
            )
            if not _operations_use_legal_edges(replacement, backend):
                continue
            signature = (start, stop, deleted_indices, tuple(str(op) for op in replacement))
            if signature in seen:
                continue
            seen.add(signature)
            equivalent = unitary_equivalent(list(operations), list(replacement), qubits)
            if not equivalent:
                continue
            rewrites.append(
                RedundancyRewrite(
                    start=start,
                    stop=stop,
                    qubits=qubits,
                    operations=operations,
                    replacement=replacement,
                    deleted_indices=deleted_indices,
                    classification=classify_rewrite(operations, deleted_indices),
                    exact_equivalent=True,
                )
            )
            if len(rewrites) >= limit:
                return rewrites
    return rewrites


def classify_rewrite(
    operations: tuple[cirq.Operation, ...],
    deleted_indices: tuple[int, ...],
) -> str:
    deleted = tuple(operations[index] for index in deleted_indices)
    if len(deleted) == 2 and all(gate_name_for_operation(op) == "cx" for op in deleted):
        if str(deleted[0]) == str(deleted[1]):
            return "inverse cancellation" if deleted_indices[1] == deleted_indices[0] + 1 else "commuting CX cancellation"
        return "compute-uncompute"
    if len(deleted) == 2 and _phase_inverse_pair(deleted[0], deleted[1]):
        return "inverse cancellation"
    names = {gate_name_for_operation(op) for op in deleted}
    if names <= {"t", "tdg", "s", "sdg", "rz"}:
        return "phase-gadget"
    if any(gate_name_for_operation(op) == "cx" for op in deleted):
        return "route artifact"
    return "other"


def local_unitary_hash(
    operations: tuple[cirq.Operation, ...],
    qubits: tuple[cirq.Qid, ...],
) -> str:
    unitary = cirq.Circuit(operations).unitary(qubit_order=qubits)
    flat = unitary.reshape(-1)
    pivot = next((value for value in flat if abs(value) > 1e-9), None)
    if pivot is not None:
        unitary = unitary / (pivot / abs(pivot))
    rounded = np.round(unitary.real, 10) + 1j * np.round(unitary.imag, 10)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def _candidate_windows(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    *,
    limit: int = 28,
    max_width: int = 4,
    max_len: int = 22,
) -> list[tuple[int, int, tuple[cirq.Qid, ...], tuple[cirq.Operation, ...], float]]:
    operations = list(circuit.all_operations())
    windows: list[tuple[int, int, tuple[cirq.Qid, ...], tuple[cirq.Operation, ...], float]] = []
    for start in range(len(operations)):
        if isinstance(operations[start].gate, cirq.MeasurementGate):
            break
        for stop in range(start + 3, min(len(operations), start + max_len) + 1):
            block = tuple(operations[start:stop])
            if any(isinstance(op.gate, cirq.MeasurementGate) for op in block):
                break
            pairs = _priority_deletion_pairs(block)
            if not pairs:
                continue
            qubits = tuple(sorted({qubit for op in block for qubit in op.qubits}, key=lambda qubit: qubit.x))
            if not 2 <= len(qubits) <= max_width:
                continue
            cx_count = sum(1 for op in block if gate_name_for_operation(op) == "cx")
            if cx_count < 2:
                continue
            windows.append((start, stop, qubits, block, _window_score(block, backend, pairs)))
    deduped: list[tuple[int, int, tuple[cirq.Qid, ...], tuple[cirq.Operation, ...], float]] = []
    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    for window in sorted(windows, key=lambda item: (-item[4], item[0], item[1])):
        signature = (window[0], window[1], tuple(qubit.x for qubit in window[2]))
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(window)
        if len(deduped) >= limit:
            break
    return deduped


def _priority_deletion_pairs(operations: tuple[cirq.Operation, ...]) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int, int, int]] = []
    removable = [
        index
        for index, op in enumerate(operations)
        if not isinstance(op.gate, cirq.MeasurementGate)
    ]
    for left, right in itertools.combinations(removable, 2):
        left_op = operations[left]
        right_op = operations[right]
        left_name = gate_name_for_operation(left_op)
        right_name = gate_name_for_operation(right_op)
        same_self_inverse = str(left_op) == str(right_op) and left_name in {"cx", "h", "x", "y", "z"}
        inverse_phase = _phase_inverse_pair(left_op, right_op)
        if not same_self_inverse and not inverse_phase:
            continue
        twoq_priority = 0 if left_name == "cx" and right_name == "cx" else 1
        phase_priority = 1 if inverse_phase else 0
        candidates.append((twoq_priority, phase_priority, right - left, left * 128 + right))
    ordered: list[tuple[int, int]] = []
    for _, _, _, encoded in sorted(candidates):
        ordered.append((encoded // 128, encoded % 128))
    return ordered


def _phase_inverse_pair(left: cirq.Operation, right: cirq.Operation) -> bool:
    if len(left.qubits) != 1 or left.qubits != right.qubits:
        return False
    names = {gate_name_for_operation(left), gate_name_for_operation(right)}
    return names in ({"t", "tdg"}, {"s", "sdg"})


def _window_score(
    operations: tuple[cirq.Operation, ...],
    backend: CompetitionBackend,
    pairs: list[tuple[int, int]],
) -> float:
    score = 0.25 * len(pairs)
    for op in operations:
        if gate_name_for_operation(op) != "cx":
            continue
        left, right = op.qubits
        score += 1.0 + _safe_edge_error(backend, left.x, right.x)
    return score


def _safe_edge_error(backend: CompetitionBackend, left: int, right: int) -> float:
    try:
        return backend.edge_error(left, right)
    except KeyError:
        return 1.0


def _operations_use_legal_edges(
    operations: tuple[cirq.Operation, ...],
    backend: CompetitionBackend,
) -> bool:
    legal_edges = {tuple(sorted(edge)) for edge in backend.coupling_map}
    for op in operations:
        if gate_name_for_operation(op) != "cx":
            continue
        if tuple(sorted((op.qubits[0].x, op.qubits[1].x))) not in legal_edges:
            return False
    return True


def _replace_window(circuit: cirq.Circuit, rewrite: RedundancyRewrite) -> cirq.Circuit:
    operations = list(circuit.all_operations())
    return cirq.Circuit([*operations[: rewrite.start], *rewrite.replacement, *operations[rewrite.stop :]])


def _accepted_metric_delta(
    trial: CompilationCandidate,
    base: CompilationCandidate,
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> bool:
    trial_metrics = candidate_metrics(trial, backend, source)
    base_metrics = candidate_metrics(base, backend, source)
    cx_improves = trial_metrics.two_qubit_count <= base_metrics.two_qubit_count - 2
    reliability_improves = (
        trial_metrics.log_reliability >= base_metrics.log_reliability + 0.02
        and trial_metrics.two_qubit_count <= base_metrics.two_qubit_count + 1
    )
    return (
        (cx_improves or reliability_improves)
        and trial_metrics.depth <= base_metrics.depth + 1
        and trial_metrics.readout_cost <= base_metrics.readout_cost + 1e-12
        and trial_metrics.compile_time_sec <= max(12.0, base_metrics.compile_time_sec + 2.0)
    )


def _operation_record(op: cirq.Operation) -> dict[str, object]:
    return {
        "text": str(op),
        "gate": gate_name_for_operation(op),
        "qubits": [qubit.x for qubit in op.qubits],
    }


def _edge_list(operations: tuple[cirq.Operation, ...]) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for op in operations:
        if gate_name_for_operation(op) == "cx":
            edges.append((op.qubits[0].x, op.qubits[1].x))
    return edges


__all__ = [
    "RedundancyRewrite",
    "classify_rewrite",
    "commutation_aware_redundancy_candidates",
    "commutation_aware_redundancy_diagnostics",
    "find_redundancy_rewrites",
    "local_unitary_hash",
]
