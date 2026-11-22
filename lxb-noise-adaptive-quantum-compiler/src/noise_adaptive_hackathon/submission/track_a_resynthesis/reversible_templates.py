from __future__ import annotations

import cirq

from .exact_equivalence import unitary_equivalent


def toffoli_6cx(control_a: cirq.Qid, control_b: cirq.Qid, target: cirq.Qid) -> list[cirq.Operation]:
    return [
        cirq.H(target),
        cirq.CNOT(control_b, target),
        cirq.T(target) ** -1,
        cirq.CNOT(control_a, target),
        cirq.T(target),
        cirq.CNOT(control_b, target),
        cirq.T(target) ** -1,
        cirq.T(control_b),
        cirq.CNOT(control_a, target),
        cirq.T(target),
        cirq.CNOT(control_a, control_b),
        cirq.H(target),
        cirq.T(control_a),
        cirq.T(control_b) ** -1,
        cirq.CNOT(control_a, control_b),
    ]


def fredkin_from_toffoli(control: cirq.Qid, left: cirq.Qid, right: cirq.Qid) -> list[cirq.Operation]:
    return [
        cirq.CNOT(right, left),
        *toffoli_6cx(control, left, right),
        cirq.CNOT(right, left),
    ]


def reversible_template_replacements(
    operations: tuple[cirq.Operation, ...],
    qubits: tuple[cirq.Qid, ...],
) -> list[list[cirq.Operation]]:
    if not 3 <= len(qubits) <= 5:
        return []
    source = list(operations)
    candidates: list[list[cirq.Operation]] = []

    if len(qubits) == 3:
        candidates.extend(_toffoli_candidates(source, qubits))
        candidates.extend(_fredkin_candidates(source, qubits))
        candidates.extend(_peres_candidates(source, qubits))

    return _dedupe_exact(source, candidates, qubits)


def _toffoli_candidates(
    operations: list[cirq.Operation],
    qubits: tuple[cirq.Qid, ...],
) -> list[list[cirq.Operation]]:
    candidates: list[list[cirq.Operation]] = []
    for target in qubits:
        controls = tuple(qubit for qubit in qubits if qubit != target)
        if unitary_equivalent(operations, [cirq.CCX(controls[0], controls[1], target)], qubits):
            candidates.append(toffoli_6cx(controls[0], controls[1], target))
            candidates.append(toffoli_6cx(controls[1], controls[0], target))
    return candidates


def _fredkin_candidates(
    operations: list[cirq.Operation],
    qubits: tuple[cirq.Qid, ...],
) -> list[list[cirq.Operation]]:
    candidates: list[list[cirq.Operation]] = []
    for control in qubits:
        left, right = tuple(qubit for qubit in qubits if qubit != control)
        if unitary_equivalent(operations, [cirq.CSWAP(control, left, right)], qubits):
            candidates.append(fredkin_from_toffoli(control, left, right))
            candidates.append(fredkin_from_toffoli(control, right, left))
    return candidates


def _peres_candidates(
    operations: list[cirq.Operation],
    qubits: tuple[cirq.Qid, ...],
) -> list[list[cirq.Operation]]:
    candidates: list[list[cirq.Operation]] = []
    for target in qubits:
        controls = tuple(qubit for qubit in qubits if qubit != target)
        peres = [cirq.CCX(controls[0], controls[1], target), cirq.CNOT(controls[0], controls[1])]
        inverse_peres = [cirq.CNOT(controls[0], controls[1]), cirq.CCX(controls[0], controls[1], target)]
        if unitary_equivalent(operations, peres, qubits):
            candidates.append([*toffoli_6cx(controls[0], controls[1], target), cirq.CNOT(controls[0], controls[1])])
            candidates.append([*toffoli_6cx(controls[1], controls[0], target), cirq.CNOT(controls[0], controls[1])])
        if unitary_equivalent(operations, inverse_peres, qubits):
            candidates.append([cirq.CNOT(controls[0], controls[1]), *toffoli_6cx(controls[0], controls[1], target)])
            candidates.append([cirq.CNOT(controls[0], controls[1]), *toffoli_6cx(controls[1], controls[0], target)])
    return candidates


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
