from __future__ import annotations

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation


def conservative_peephole(circuit: cirq.Circuit, *, clifford_only: bool) -> cirq.Circuit:
    del clifford_only
    simplified: list[cirq.Operation] = []
    for op in circuit.all_operations():
        if simplified and _cancels(simplified[-1], op):
            simplified.pop()
            continue
        simplified.append(op)
    return cirq.Circuit(_cancel_commuting_identical_cx(simplified))


def _cancels(left: cirq.Operation, right: cirq.Operation) -> bool:
    if left.qubits != right.qubits:
        return False
    left_name = gate_name_for_operation(left)
    right_name = gate_name_for_operation(right)
    if left_name is None or right_name is None:
        return False
    if (left_name, right_name) in {("h", "h"), ("x", "x"), ("s", "sdg"), ("sdg", "s")}:
        return True
    return left_name == right_name == "cx"


def _cancel_commuting_identical_cx(operations: list[cirq.Operation]) -> list[cirq.Operation]:
    ops = list(operations)
    changed = True
    while changed:
        changed = False
        rewritten: list[cirq.Operation] = []
        index = 0
        while index < len(ops):
            op = ops[index]
            if gate_name_for_operation(op) != "cx":
                rewritten.append(op)
                index += 1
                continue

            match = _next_touching_operation(ops, index)
            if match is not None and _same_cx(op, ops[match]):
                rewritten.extend(ops[index + 1 : match])
                index = match + 1
                changed = True
                continue

            rewritten.append(op)
            index += 1
        ops = rewritten
    return ops


def _next_touching_operation(operations: list[cirq.Operation], index: int) -> int | None:
    touched = set(operations[index].qubits)
    for candidate in range(index + 1, len(operations)):
        if touched & set(operations[candidate].qubits):
            return candidate
    return None


def _same_cx(left: cirq.Operation, right: cirq.Operation) -> bool:
    return gate_name_for_operation(right) == "cx" and left.qubits == right.qubits
