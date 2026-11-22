from __future__ import annotations

from dataclasses import dataclass

import cirq

from .exact_equivalence import unitary_equivalent
from .parity_gadget_synthesis import ParityGadget, find_parity_gadgets, gadget_operations


@dataclass(frozen=True)
class PhasePolynomialReplacement:
    start: int
    stop: int
    qubits: tuple[cirq.Qid, ...]
    operations: list[cirq.Operation]


def phase_polynomial_rewrite_candidates(
    operations: list[cirq.Operation],
    *,
    max_width: int = 5,
    limit: int = 8,
) -> list[PhasePolynomialReplacement]:
    gadgets = find_parity_gadgets(operations)
    if len(gadgets) < 2:
        return []
    replacements: list[PhasePolynomialReplacement] = []
    for left, right in zip(gadgets, gadgets[1:]):
        candidate = _combine_adjacent_equal_gadgets(operations, left, right, max_width=max_width)
        if candidate is not None:
            replacements.append(candidate)
            if len(replacements) >= limit:
                return replacements
    return replacements


def _combine_adjacent_equal_gadgets(
    operations: list[cirq.Operation],
    left: ParityGadget,
    right: ParityGadget,
    *,
    max_width: int,
) -> PhasePolynomialReplacement | None:
    if left.stop != right.start:
        return None
    if left.ladder != right.ladder or left.phase_qubit != right.phase_qubit:
        return None
    source = operations[left.start : right.stop]
    qubits = tuple(sorted({qubit for op in source for qubit in op.qubits}, key=lambda qubit: qubit.x))
    if not 2 <= len(qubits) <= max_width:
        return None
    combined = gadget_operations(left, left.exponent + right.exponent)
    if len(combined) >= len(source):
        return None
    if not unitary_equivalent(source, combined, qubits):
        return None
    return PhasePolynomialReplacement(left.start, right.stop, qubits, combined)
