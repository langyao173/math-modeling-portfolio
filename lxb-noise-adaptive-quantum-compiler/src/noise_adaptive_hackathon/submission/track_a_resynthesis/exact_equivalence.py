from __future__ import annotations

import cirq


def unitary_equivalent(
    left: list[cirq.Operation],
    right: list[cirq.Operation],
    qubits: tuple[cirq.Qid, ...],
    *,
    atol: float = 1e-8,
) -> bool:
    try:
        left_unitary = cirq.Circuit(left).unitary(qubit_order=list(qubits))
        right_unitary = cirq.Circuit(right).unitary(qubit_order=list(qubits))
        cirq.testing.assert_allclose_up_to_global_phase(left_unitary, right_unitary, atol=atol)
    except Exception:
        return False
    return True


def circuit_without_measurements(circuit: cirq.Circuit) -> cirq.Circuit:
    return cirq.Circuit(op for op in circuit.all_operations() if not isinstance(op.gate, cirq.MeasurementGate))
