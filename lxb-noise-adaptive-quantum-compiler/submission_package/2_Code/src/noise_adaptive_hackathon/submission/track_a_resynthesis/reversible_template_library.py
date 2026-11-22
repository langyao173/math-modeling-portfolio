from __future__ import annotations

import cirq

from .reversible_templates import (
    fredkin_from_toffoli,
    reversible_template_replacements,
    toffoli_6cx,
)


def reversible_library_replacements(
    operations: tuple[cirq.Operation, ...],
    qubits: tuple[cirq.Qid, ...],
) -> list[list[cirq.Operation]]:
    return reversible_template_replacements(operations, qubits)


__all__ = [
    "fredkin_from_toffoli",
    "reversible_library_replacements",
    "toffoli_6cx",
]
