from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from ..backend_model import BackendNoiseModel
from ..clifford_resynthesis import _remote_cnot_operations, synthesize_cnot_matrix
from .exact_gf2_equivalence import (
    MatrixBits,
    apply_cnot_to_matrix,
    cnot_matrix_bits,
    cnot_networks_equivalent,
    identity_matrix_bits,
)


@dataclass(frozen=True)
class WeightedCnotSynthesisResult:
    operations: list[cirq.Operation]
    cost: float
    method: str


@dataclass(frozen=True)
class WeightedCnotReplacement:
    start: int
    stop: int
    qubits: tuple[cirq.Qid, ...]
    operations: list[cirq.Operation]
    method: str


def weighted_cnot_replacements(
    operations: list[cirq.Operation],
    backend: CompetitionBackend,
    *,
    max_width: int = 7,
    limit: int = 8,
    max_seconds: float = 3.0,
) -> list[WeightedCnotReplacement]:
    started = time.perf_counter()
    replacements: list[WeightedCnotReplacement] = []
    index = 0
    while index < len(operations):
        if time.perf_counter() - started >= max_seconds:
            break
        if gate_name_for_operation(operations[index]) != "cx":
            index += 1
            continue
        start = index
        while index < len(operations) and gate_name_for_operation(operations[index]) == "cx":
            index += 1
        block = operations[start:index]
        if len(block) < 3:
            continue
        qubits = tuple(sorted({qubit for op in block for qubit in op.qubits}, key=lambda qubit: qubit.x))
        if not 2 <= len(qubits) <= max_width:
            continue
        result = synthesize_weighted_cnot_block(
            block,
            backend,
            beam_width=512,
            max_seconds=max(0.1, max_seconds - (time.perf_counter() - started)),
        )
        if result is None or len(result.operations) >= len(block):
            continue
        check_qubits = tuple(sorted({qubit for op in [*block, *result.operations] for qubit in op.qubits}, key=lambda qubit: qubit.x))
        if not cnot_networks_equivalent(block, result.operations, check_qubits):
            continue
        replacements.append(
            WeightedCnotReplacement(
                start=start,
                stop=index,
                qubits=check_qubits,
                operations=result.operations,
                method=result.method,
            )
        )
        if len(replacements) >= limit:
            break
    return replacements


def synthesize_weighted_cnot_block(
    block: list[cirq.Operation],
    backend: CompetitionBackend,
    *,
    beam_width: int = 2048,
    max_steps: int | None = None,
    max_seconds: float = 2.0,
) -> WeightedCnotSynthesisResult | None:
    if any(gate_name_for_operation(op) != "cx" for op in block):
        return None
    qubits = tuple(sorted({qubit for op in block for qubit in op.qubits}, key=lambda qubit: qubit.x))
    if len(qubits) < 2:
        return None
    target = cnot_matrix_bits(block, qubits)
    if target == identity_matrix_bits(len(qubits)):
        return WeightedCnotSynthesisResult([], 0.0, "identity")

    deadline = time.perf_counter() + max_seconds
    if len(qubits) <= 7:
        beam = _beam_synthesize(target, qubits, backend, beam_width=beam_width, max_steps=max_steps, deadline=deadline)
        if beam is not None:
            return beam
    return _remote_gauss_synthesize(target, qubits, backend)


def _beam_synthesize(
    target: MatrixBits,
    qubits: tuple[cirq.Qid, ...],
    backend: CompetitionBackend,
    *,
    beam_width: int,
    max_steps: int | None,
    deadline: float,
) -> WeightedCnotSynthesisResult | None:
    width = len(qubits)
    qubit_ids = [qubit.x for qubit in qubits]
    index_by_physical = {physical: index for index, physical in enumerate(qubit_ids)}
    actions: list[tuple[float, int, int, cirq.Operation]] = []
    for left, right in backend.coupling_map:
        if left not in index_by_physical or right not in index_by_physical:
            continue
        for control_physical, target_physical in ((left, right), (right, left)):
            control = index_by_physical[control_physical]
            target_index = index_by_physical[target_physical]
            edge_cost = -math.log(max(1.0 - backend.edge_error(control_physical, target_physical), 1e-12))
            actions.append(
                (
                    edge_cost + 0.0002,
                    control,
                    target_index,
                    cirq.CNOT(cirq.LineQubit(control_physical), cirq.LineQubit(target_physical)),
                )
            )
    if not actions:
        return None
    if max_steps is None:
        max_steps = max(8, min(36, 2 * width * width + 4))

    start = identity_matrix_bits(width)
    beams: list[tuple[float, MatrixBits, tuple[cirq.Operation, ...]]] = [(0.0, start, ())]
    best_seen: dict[MatrixBits, float] = {start: 0.0}
    min_edge = min(action[0] for action in actions)
    for _ in range(max_steps):
        if time.perf_counter() >= deadline:
            return None
        next_items: list[tuple[float, float, MatrixBits, tuple[cirq.Operation, ...]]] = []
        for current_cost, matrix, ops in beams:
            for action_cost, control, target_index, op in actions:
                updated = apply_cnot_to_matrix(matrix, control, target_index)
                cost = current_cost + action_cost
                if cost >= best_seen.get(updated, float("inf")) - 1e-12:
                    continue
                best_seen[updated] = cost
                updated_ops = (*ops, op)
                if updated == target:
                    return WeightedCnotSynthesisResult(list(updated_ops), cost, "beam_dijkstra")
                priority = cost + min_edge * _matrix_hamming_distance(updated, target)
                next_items.append((priority, cost, updated, updated_ops))
        if not next_items:
            return None
        beams = [
            (cost, matrix, ops)
            for _, cost, matrix, ops in heapq.nsmallest(
                beam_width,
                next_items,
                key=lambda item: (item[0], len(item[3]), tuple(str(op) for op in item[3])),
            )
        ]
    return None


def _remote_gauss_synthesize(
    target: MatrixBits,
    qubits: tuple[cirq.Qid, ...],
    backend: CompetitionBackend,
) -> WeightedCnotSynthesisResult | None:
    matrix = [[(row >> column) & 1 for column in range(len(qubits))] for row in target]
    try:
        logical_cnots = synthesize_cnot_matrix(matrix)
    except ValueError:
        return None
    model = BackendNoiseModel.from_backend(backend)
    physical_qubits = tuple(cirq.LineQubit.range(backend.num_qubits))
    operations: list[cirq.Operation] = []
    cost = 0.0
    for control_index, target_index in logical_cnots:
        control = qubits[control_index].x
        target_physical = qubits[target_index].x
        routed = _remote_cnot_operations(control, target_physical, model, physical_qubits)
        operations.extend(routed)
        cost += _operations_cost(routed, backend)
    return WeightedCnotSynthesisResult(operations, cost, "weighted_remote_gauss")


def _matrix_hamming_distance(left: MatrixBits, right: MatrixBits) -> int:
    return sum((left_row ^ right_row).bit_count() for left_row, right_row in zip(left, right))


def _operations_cost(operations: list[cirq.Operation], backend: CompetitionBackend) -> float:
    total = 0.0
    for op in operations:
        if gate_name_for_operation(op) == "cx":
            total += -math.log(max(1.0 - backend.edge_error(op.qubits[0].x, op.qubits[1].x), 1e-12))
    return total
