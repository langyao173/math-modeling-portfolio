from __future__ import annotations

import cirq

from .exact_equivalence import unitary_equivalent
from .linear_phase_synthesis import cnot_linear_matrix, is_cnot_phase_circuit, parity_phase_gadget_candidates


def phase_polynomial_replacements(
    operations: tuple[cirq.Operation, ...],
    qubits: tuple[cirq.Qid, ...],
) -> list[list[cirq.Operation]]:
    if not 3 <= len(qubits) <= 5 or not is_cnot_phase_circuit(operations):
        return []
    if cnot_linear_matrix(operations, qubits) is None:
        return []
    candidates: list[list[cirq.Operation]] = []
    candidates.extend(parity_phase_gadget_candidates(operations))
    return _dedupe_exact(list(operations), candidates, qubits)


def _dedupe_exact(
    source: list[cirq.Operation],
    candidates: list[list[cirq.Operation]],
    qubits: tuple[cirq.Qid, ...],
) -> list[list[cirq.Operation]]:
    accepted: list[list[cirq.Operation]] = []
    seen: set[tuple[str, ...]] = {tuple(str(op) for op in source)}
    for candidate in candidates:
        signature = tuple(str(op) for op in candidate)
        if signature in seen:
            continue
        seen.add(signature)
        if unitary_equivalent(source, candidate, qubits):
            accepted.append(candidate)
    return accepted
