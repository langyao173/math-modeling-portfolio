from __future__ import annotations

import cirq

from noise_adaptive_hackathon.baseline.common import _decomposed_swap
from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .backend_model import BackendNoiseModel
from .router import _append_bridge


Matrix = list[list[int]]
Cnot = tuple[int, int]


def resynthesize_cnot_blocks(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
) -> tuple[cirq.Circuit, bool]:
    model = BackendNoiseModel.from_backend(backend)
    physical_qubits = tuple(cirq.LineQubit.range(backend.num_qubits))
    operations = list(circuit.all_operations())
    rewritten: list[cirq.Operation] = []
    changed = False
    index = 0
    while index < len(operations):
        if gate_name_for_operation(operations[index]) != "cx":
            rewritten.append(operations[index])
            index += 1
            continue

        start = index
        while index < len(operations) and gate_name_for_operation(operations[index]) == "cx":
            index += 1
        block = operations[start:index]
        replacement = _resynthesize_block(block, model, physical_qubits)
        if replacement is not None and len(replacement) < len(block):
            rewritten.extend(replacement)
            changed = True
        else:
            rewritten.extend(block)

    return cirq.Circuit(rewritten), changed


def cnot_block_matrix(block: list[cirq.Operation], qubits: list[cirq.LineQubit]) -> Matrix:
    index = {qubit: offset for offset, qubit in enumerate(qubits)}
    matrix = _identity(len(qubits))
    for op in block:
        if gate_name_for_operation(op) != "cx":
            raise ValueError("CNOT block matrix can only be built from cx operations")
        control = index[op.qubits[0]]
        target = index[op.qubits[1]]
        _row_add(matrix, control, target)
    return matrix


def synthesize_cnot_matrix(matrix: Matrix) -> list[Cnot]:
    work = [row[:] for row in matrix]
    size = len(work)
    reductions: list[Cnot] = []
    for column in range(size):
        pivot = _find_pivot(work, column)
        if pivot is None:
            raise ValueError("CNOT matrix is singular")
        if pivot != column:
            for control, target in ((pivot, column), (column, pivot), (pivot, column)):
                _row_add(work, control, target)
                reductions.append((control, target))
        for row in range(size):
            if row != column and work[row][column]:
                _row_add(work, column, row)
                reductions.append((column, row))
    if work != _identity(size):
        raise ValueError("CNOT synthesis failed to reduce matrix")
    return list(reversed(reductions))


def _resynthesize_block(
    block: list[cirq.Operation],
    model: BackendNoiseModel,
    physical_qubits: tuple[cirq.LineQubit, ...],
) -> list[cirq.Operation] | None:
    qubits = sorted({qubit for op in block for qubit in op.qubits}, key=lambda qubit: qubit.x)
    if len(qubits) < 2:
        return None
    matrix = cnot_block_matrix(block, qubits)
    try:
        all_to_all = synthesize_cnot_matrix(matrix)
    except ValueError:
        return None

    replacement: list[cirq.Operation] = []
    for control_index, target_index in all_to_all:
        control = qubits[control_index].x
        target = qubits[target_index].x
        replacement.extend(_remote_cnot_operations(control, target, model, physical_qubits))
        if len(replacement) >= len(block):
            return None
    return replacement


def _remote_cnot_operations(
    control: int,
    target: int,
    model: BackendNoiseModel,
    physical_qubits: tuple[cirq.LineQubit, ...],
) -> list[cirq.Operation]:
    if control == target:
        return []
    path = model.noise_path(control, target)
    if len(path) == 2:
        return [cirq.CNOT(physical_qubits[control], physical_qubits[target])]
    if len(path) == 3:
        circuit = cirq.Circuit()
        _append_bridge(circuit, physical_qubits, path[0], path[1], path[2])
        return list(circuit.all_operations())

    operations: list[cirq.Operation] = []
    forward_edges = list(zip(path[:-2], path[1:-1]))
    for left, right in forward_edges:
        operations.extend(_decomposed_swap(physical_qubits[left], physical_qubits[right]))
    operations.append(cirq.CNOT(physical_qubits[path[-2]], physical_qubits[path[-1]]))
    for left, right in reversed(forward_edges):
        operations.extend(_decomposed_swap(physical_qubits[left], physical_qubits[right]))
    return operations


def _identity(size: int) -> Matrix:
    return [[1 if row == column else 0 for column in range(size)] for row in range(size)]


def _row_add(matrix: Matrix, control: int, target: int) -> None:
    matrix[target] = [left ^ right for left, right in zip(matrix[target], matrix[control])]


def _find_pivot(matrix: Matrix, column: int) -> int | None:
    size = len(matrix)
    for row in range(column, size):
        if matrix[row][column]:
            return row
    for row in range(column):
        if matrix[row][column]:
            return row
    return None
