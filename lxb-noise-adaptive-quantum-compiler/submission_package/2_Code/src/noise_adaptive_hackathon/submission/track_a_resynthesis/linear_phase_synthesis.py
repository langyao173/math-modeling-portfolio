from __future__ import annotations

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation


PHASE_GATES = {"z", "s", "sdg", "t", "tdg", "rz"}


def is_cnot_phase_circuit(operations: tuple[cirq.Operation, ...]) -> bool:
    return all(gate_name_for_operation(op) in PHASE_GATES | {"cx"} for op in operations)


def cnot_linear_matrix(
    operations: tuple[cirq.Operation, ...],
    qubits: tuple[cirq.Qid, ...],
) -> tuple[tuple[int, ...], ...] | None:
    qubit_index = {qubit: index for index, qubit in enumerate(qubits)}
    matrix = [[1 if row == column else 0 for column in range(len(qubits))] for row in range(len(qubits))]
    for op in operations:
        name = gate_name_for_operation(op)
        if name in PHASE_GATES:
            continue
        if name != "cx" or len(op.qubits) != 2:
            return None
        control = qubit_index[op.qubits[0]]
        target = qubit_index[op.qubits[1]]
        matrix[target] = [(left ^ right) for left, right in zip(matrix[target], matrix[control])]
    return tuple(tuple(row) for row in matrix)


def parity_phase_gadget_candidates(
    operations: tuple[cirq.Operation, ...],
) -> list[list[cirq.Operation]]:
    if not is_cnot_phase_circuit(operations):
        return []
    ops = list(operations)
    candidates: list[list[cirq.Operation]] = []
    for phase_index, op in enumerate(ops):
        if gate_name_for_operation(op) not in PHASE_GATES or len(op.qubits) != 1:
            continue
        prefix = ops[:phase_index]
        suffix = ops[phase_index + 1 :]
        if len(prefix) != len(suffix) or not prefix:
            continue
        if not all(gate_name_for_operation(item) == "cx" for item in prefix + suffix):
            continue
        if _inverse_cnot_ladder(prefix, suffix):
            reversed_prefix = list(reversed(prefix))
            reversed_suffix = list(reversed(suffix))
            candidates.append([*reversed_prefix, op, *reversed_suffix])
    return candidates


def _inverse_cnot_ladder(prefix: list[cirq.Operation], suffix: list[cirq.Operation]) -> bool:
    if len(prefix) != len(suffix):
        return False
    for left, right in zip(prefix, reversed(suffix)):
        if tuple(left.qubits) != tuple(right.qubits):
            return False
    return True
