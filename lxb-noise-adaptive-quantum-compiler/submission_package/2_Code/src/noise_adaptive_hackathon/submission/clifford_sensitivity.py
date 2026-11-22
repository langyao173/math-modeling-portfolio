from __future__ import annotations

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend


def clifford_sensitivity_cost(circuit: cirq.Circuit, backend: CompetitionBackend) -> float:
    operations = [
        op for op in circuit.all_operations()
        if not isinstance(op.gate, cirq.IdentityGate)
    ]
    measured = _measured_physical_qubits(operations)
    if not measured:
        return 0.0

    cost = sum(backend.readout(qubit) for qubit in measured)
    for index, op in enumerate(operations):
        if gate_name_for_operation(op) != "cx" or len(op.qubits) != 2:
            continue
        control, target = op.qubits[0].x, op.qubits[1].x
        influence = 0.25 * sum(
            _propagated_x_support(
                x_bits={x_qubit} if x_qubit is not None else set(),
                z_bits={z_qubit} if z_qubit is not None else set(),
                suffix=operations[index + 1 :],
                measured=measured,
            )
            for x_qubit, z_qubit in (
                (control, None),
                (None, control),
                (target, None),
                (None, target),
            )
        )
        cost += backend.edge_error(control, target) * influence
    return cost


def _measured_physical_qubits(operations: list[cirq.Operation]) -> tuple[int, ...]:
    measurements = [op for op in operations if isinstance(op.gate, cirq.MeasurementGate)]
    if not measurements:
        return ()
    return tuple(qubit.x for qubit in measurements[-1].qubits)


def _propagated_x_support(
    x_bits: set[int],
    z_bits: set[int],
    suffix: list[cirq.Operation],
    measured: tuple[int, ...],
) -> int:
    x = set(x_bits)
    z = set(z_bits)
    for op in suffix:
        gate_name = gate_name_for_operation(op)
        if gate_name == "measure":
            break
        if len(op.qubits) == 1:
            qubit = op.qubits[0].x
            if gate_name == "h":
                x_has = qubit in x
                z_has = qubit in z
                if x_has:
                    x.remove(qubit)
                    z.add(qubit)
                if z_has:
                    z.remove(qubit)
                    x.add(qubit)
            elif gate_name in {"s", "sdg"} and qubit in x:
                if qubit in z:
                    z.remove(qubit)
                else:
                    z.add(qubit)
            continue
        if gate_name != "cx" or len(op.qubits) != 2:
            continue
        control, target = op.qubits[0].x, op.qubits[1].x
        if control in x:
            if target in x:
                x.remove(target)
            else:
                x.add(target)
        if target in z:
            if control in z:
                z.remove(control)
            else:
                z.add(control)
    return sum(1 for qubit in measured if qubit in x)
