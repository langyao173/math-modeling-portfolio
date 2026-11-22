from __future__ import annotations

from dataclasses import dataclass

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation


TRACK_A_TAIL_DIAGONAL = {"z", "s", "sdg", "t", "tdg", "rz"}
TRACK_B_TAIL_DIAGONAL = {"z", "s", "sdg"}


@dataclass(frozen=True)
class MeasurementTailPruneResult:
    circuit: cirq.Circuit
    removed_operations: tuple[cirq.Operation, ...]

    @property
    def changed(self) -> bool:
        return bool(self.removed_operations)


def prune_measurement_tail(
    circuit: cirq.Circuit,
    *,
    clifford_only: bool,
) -> MeasurementTailPruneResult:
    """Remove final Z-basis diagonal gates that cannot affect the final z measurement.

    The rule is deliberately narrow: it only walks backward from the one final
    measurement and deletes single-qubit diagonal Z-family gates on measured
    qubits until a non-diagonal, two-qubit, or measurement boundary is reached.
    """

    operations = list(circuit.all_operations())
    measurement_indices = [
        index for index, op in enumerate(operations) if isinstance(op.gate, cirq.MeasurementGate)
    ]
    if len(measurement_indices) != 1 or measurement_indices[0] != len(operations) - 1:
        return MeasurementTailPruneResult(circuit, ())

    measurement = operations[measurement_indices[0]]
    if cirq.measurement_key_name(measurement) != "z":
        return MeasurementTailPruneResult(circuit, ())

    allowed = TRACK_B_TAIL_DIAGONAL if clifford_only else TRACK_A_TAIL_DIAGONAL
    open_qubits = set(measurement.qubits)
    remove_indices: set[int] = set()
    removed: list[cirq.Operation] = []

    for index in range(measurement_indices[0] - 1, -1, -1):
        if not open_qubits:
            break
        op = operations[index]
        op_qubits = set(op.qubits)
        touched = op_qubits & open_qubits
        if not touched:
            continue
        if isinstance(op.gate, cirq.MeasurementGate) or len(op.qubits) != 1:
            open_qubits -= touched
            continue
        name = gate_name_for_operation(op)
        if name in allowed and op.qubits[0] in open_qubits:
            remove_indices.add(index)
            removed.append(op)
            continue
        open_qubits.remove(op.qubits[0])

    if not remove_indices:
        return MeasurementTailPruneResult(circuit, ())
    pruned = cirq.Circuit(op for index, op in enumerate(operations) if index not in remove_indices)
    return MeasurementTailPruneResult(pruned, tuple(reversed(removed)))


__all__ = ["MeasurementTailPruneResult", "prune_measurement_tail"]
