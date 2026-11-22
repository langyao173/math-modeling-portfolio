from __future__ import annotations

from dataclasses import dataclass

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation


PHASE_EXPONENTS = {
    "z": 1.0,
    "s": 0.5,
    "sdg": -0.5,
    "t": 0.25,
    "tdg": -0.25,
}


@dataclass(frozen=True)
class ParityGadget:
    start: int
    stop: int
    ladder: tuple[tuple[cirq.Qid, cirq.Qid], ...]
    phase_qubit: cirq.Qid
    exponent: float


def find_parity_gadgets(operations: list[cirq.Operation]) -> list[ParityGadget]:
    gadgets: list[ParityGadget] = []
    index = 0
    while index < len(operations):
        found = _match_gadget(operations, index)
        if found is None:
            index += 1
            continue
        gadgets.append(found)
        index = found.stop
    return gadgets


def gadget_operations(gadget: ParityGadget, exponent: float | None = None) -> list[cirq.Operation]:
    phase_exponent = gadget.exponent if exponent is None else exponent
    operations: list[cirq.Operation] = [cirq.CNOT(control, target) for control, target in gadget.ladder]
    phase = _phase_operation(gadget.phase_qubit, phase_exponent)
    if phase is not None:
        operations.append(phase)
    operations.extend(cirq.CNOT(control, target) for control, target in reversed(gadget.ladder))
    return operations


def _match_gadget(operations: list[cirq.Operation], start: int) -> ParityGadget | None:
    ladder: list[tuple[cirq.Qid, cirq.Qid]] = []
    index = start
    while index < len(operations) and gate_name_for_operation(operations[index]) == "cx":
        control, target = operations[index].qubits
        ladder.append((control, target))
        index += 1
    if not ladder or index >= len(operations):
        return None
    phase_name = gate_name_for_operation(operations[index])
    if phase_name == "rz":
        try:
            exponent = float(operations[index].gate.exponent)
        except Exception:
            return None
    else:
        exponent = PHASE_EXPONENTS.get(phase_name)
    if exponent is None or len(operations[index].qubits) != 1:
        return None
    phase_qubit = operations[index].qubits[0]
    index += 1
    for control, target in reversed(ladder):
        if index >= len(operations) or gate_name_for_operation(operations[index]) != "cx":
            return None
        if tuple(operations[index].qubits) != (control, target):
            return None
        index += 1
    return ParityGadget(start, index, tuple(ladder), phase_qubit, exponent)


def _phase_operation(qubit: cirq.Qid, exponent: float) -> cirq.Operation | None:
    normalized = exponent % 2.0
    if abs(normalized) < 1e-9:
        return None
    if abs(normalized - 1.0) < 1e-9:
        return cirq.Z(qubit)
    if abs(normalized - 0.5) < 1e-9:
        return cirq.S(qubit)
    if abs(normalized - 1.5) < 1e-9:
        return cirq.S(qubit) ** -1
    if abs(normalized - 0.25) < 1e-9:
        return cirq.T(qubit)
    if abs(normalized - 1.75) < 1e-9:
        return cirq.T(qubit) ** -1
    return cirq.ZPowGate(exponent=normalized).on(qubit)
