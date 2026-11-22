from __future__ import annotations

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend


def basis_and_coupling_legal(operations: list[cirq.Operation], backend: CompetitionBackend) -> bool:
    allowed_edges = {tuple(sorted(edge)) for edge in backend.coupling_map}
    for op in operations:
        name = gate_name_for_operation(op)
        if name is None or name not in backend.basis_gates:
            return False
        if len(op.qubits) == 2 and tuple(sorted(qubit.x for qubit in op.qubits)) not in allowed_edges:
            return False
    return True
