from __future__ import annotations

from dataclasses import dataclass

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation


@dataclass(frozen=True)
class OperationWindow:
    start: int
    stop: int
    qubits: tuple[cirq.Qid, ...]
    operations: tuple[cirq.Operation, ...]


def pair_windows(circuit: cirq.Circuit) -> list[OperationWindow]:
    operations = list(circuit.all_operations())
    windows: list[OperationWindow] = []
    index = 0
    while index < len(operations):
        op = operations[index]
        if gate_name_for_operation(op) == "measure" or len(op.qubits) != 2:
            index += 1
            continue
        pair = tuple(op.qubits)
        pair_set = set(pair)
        start = index
        block: list[cirq.Operation] = []
        while index < len(operations):
            current = operations[index]
            if gate_name_for_operation(current) == "measure" or not set(current.qubits) <= pair_set:
                break
            block.append(current)
            index += 1
        if len(block) >= 2:
            windows.append(OperationWindow(start, index, pair, tuple(block)))
    return windows


def replace_window(
    circuit: cirq.Circuit,
    window: OperationWindow,
    replacement: list[cirq.Operation],
) -> cirq.Circuit:
    operations = list(circuit.all_operations())
    return cirq.Circuit([*operations[: window.start], *replacement, *operations[window.stop :]])
