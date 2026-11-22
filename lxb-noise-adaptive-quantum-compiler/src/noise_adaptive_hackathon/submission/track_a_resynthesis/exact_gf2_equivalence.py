from __future__ import annotations

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation


MatrixBits = tuple[int, ...]


def cnot_matrix_bits(
    operations: list[cirq.Operation],
    qubits: tuple[cirq.Qid, ...],
) -> MatrixBits:
    index = {qubit: offset for offset, qubit in enumerate(qubits)}
    rows = list(identity_matrix_bits(len(qubits)))
    for op in operations:
        if gate_name_for_operation(op) != "cx":
            raise ValueError("GF(2) CNOT equivalence only supports cx operations")
        control = index[op.qubits[0]]
        target = index[op.qubits[1]]
        rows[target] ^= rows[control]
    return tuple(rows)


def cnot_networks_equivalent(
    left: list[cirq.Operation],
    right: list[cirq.Operation],
    qubits: tuple[cirq.Qid, ...],
) -> bool:
    try:
        return cnot_matrix_bits(left, qubits) == cnot_matrix_bits(right, qubits)
    except Exception:
        return False


def identity_matrix_bits(width: int) -> MatrixBits:
    return tuple(1 << index for index in range(width))


def apply_cnot_to_matrix(matrix: MatrixBits, control: int, target: int) -> MatrixBits:
    rows = list(matrix)
    rows[target] ^= rows[control]
    return tuple(rows)


def matrix_rank(matrix: MatrixBits) -> int:
    basis: dict[int, int] = {}
    rank = 0
    for row in matrix:
        current = row
        while current:
            pivot = current.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = current
                rank += 1
                break
            current ^= basis[pivot]
    return rank


def is_invertible_matrix(matrix: MatrixBits) -> bool:
    return matrix_rank(matrix) == len(matrix)
