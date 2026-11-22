from __future__ import annotations

from dataclasses import dataclass

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation

from ..clifford_resynthesis import cnot_block_matrix, synthesize_cnot_matrix
from .exact_equivalence import unitary_equivalent


@dataclass(frozen=True)
class CnotBlockReplacement:
    start: int
    stop: int
    qubits: tuple[cirq.Qid, ...]
    operations: list[cirq.Operation]


def cnot_block_gf2_replacements(
    operations: list[cirq.Operation],
    *,
    max_width: int = 5,
    limit: int = 8,
) -> list[CnotBlockReplacement]:
    replacements: list[CnotBlockReplacement] = []
    index = 0
    while index < len(operations):
        if gate_name_for_operation(operations[index]) != "cx":
            index += 1
            continue
        start = index
        while index < len(operations) and gate_name_for_operation(operations[index]) == "cx":
            index += 1
        block = operations[start:index]
        qubits = tuple(sorted({qubit for op in block for qubit in op.qubits}, key=lambda qubit: qubit.x))
        if 2 <= len(qubits) <= max_width and len(block) >= 3:
            replacement = _synthesize_logical_cnot_block(block, qubits)
            if replacement is not None:
                replacements.append(CnotBlockReplacement(start, index, qubits, replacement))
                if len(replacements) >= limit:
                    return replacements
    return replacements


def _synthesize_logical_cnot_block(
    block: list[cirq.Operation],
    qubits: tuple[cirq.Qid, ...],
) -> list[cirq.Operation] | None:
    matrix = cnot_block_matrix(block, list(qubits))
    try:
        synthesized = synthesize_cnot_matrix(matrix)
    except ValueError:
        return None
    replacement = [cirq.CNOT(qubits[control], qubits[target]) for control, target in synthesized]
    if len(replacement) >= len(block):
        return None
    if not unitary_equivalent(block, replacement, qubits):
        return None
    return replacement
