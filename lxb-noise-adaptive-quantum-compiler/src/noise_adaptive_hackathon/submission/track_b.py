from __future__ import annotations

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import TRACK_B_CLIFFORD_BASIS


def is_clifford_only(circuit: cirq.Circuit) -> bool:
    for op in circuit.all_operations():
        if isinstance(op.gate, cirq.IdentityGate):
            continue
        gate_name = gate_name_for_operation(op)
        if gate_name not in TRACK_B_CLIFFORD_BASIS:
            return False
    return True
