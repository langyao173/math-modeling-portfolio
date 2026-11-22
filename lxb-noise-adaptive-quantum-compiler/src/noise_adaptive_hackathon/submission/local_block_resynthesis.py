from __future__ import annotations

import cirq
from cirq.transformers.analytical_decompositions import two_qubit_matrix_to_cz_operations

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend


def resynthesize_local_two_qubit_blocks(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
) -> tuple[cirq.Circuit, bool]:
    operations = list(circuit.all_operations())
    rewritten: list[cirq.Operation] = []
    changed = False
    index = 0
    while index < len(operations):
        op = operations[index]
        if gate_name_for_operation(op) == "measure" or len(op.qubits) != 2:
            rewritten.append(op)
            index += 1
            continue

        pair = tuple(op.qubits)
        block, stop = _collect_pair_block(operations, index, pair)
        replacement = _synthesize_pair_block(block, pair, backend)
        if replacement is not None and _cx_count(replacement) < _cx_count(block):
            rewritten.extend(replacement)
            changed = True
        else:
            rewritten.extend(block)
        index = stop

    return cirq.Circuit(rewritten), changed


def _collect_pair_block(
    operations: list[cirq.Operation],
    start: int,
    pair: tuple[cirq.Qid, cirq.Qid],
) -> tuple[list[cirq.Operation], int]:
    pair_set = set(pair)
    block: list[cirq.Operation] = []
    index = start
    while index < len(operations):
        op = operations[index]
        if gate_name_for_operation(op) == "measure":
            break
        if not set(op.qubits) <= pair_set:
            break
        block.append(op)
        index += 1
    return block, index


def _synthesize_pair_block(
    block: list[cirq.Operation],
    pair: tuple[cirq.Qid, cirq.Qid],
    backend: CompetitionBackend,
) -> list[cirq.Operation] | None:
    if _cx_count(block) < 2:
        return None
    try:
        unitary = cirq.Circuit(block).unitary(qubit_order=list(pair))
        raw_ops = two_qubit_matrix_to_cz_operations(pair[0], pair[1], unitary, allow_partial_czs=False)
    except Exception:
        return None

    replacement: list[cirq.Operation] = []
    for op in raw_ops:
        replacement.extend(_to_supported_basis(op))
    if not replacement:
        return []
    if any(gate_name_for_operation(op) not in backend.basis_gates for op in replacement):
        return None
    if any(len(op.qubits) == 2 and not _is_allowed_edge(op, backend) for op in replacement):
        return None
    return replacement


def _to_supported_basis(op: cirq.Operation) -> list[cirq.Operation]:
    if isinstance(op.gate, cirq.CZPowGate) and abs((float(op.gate.exponent) % 2) - 1.0) < 1e-9:
        control, target = op.qubits
        return [cirq.H(target), cirq.CNOT(control, target), cirq.H(target)]
    gate_name = gate_name_for_operation(op)
    if gate_name is not None:
        return [op]

    converted: list[cirq.Operation] = []
    for part in cirq.decompose(op):
        if isinstance(part.gate, cirq.CZPowGate) and abs((float(part.gate.exponent) % 2) - 1.0) < 1e-9:
            control, target = part.qubits
            converted.extend([cirq.H(target), cirq.CNOT(control, target), cirq.H(target)])
        elif gate_name_for_operation(part) is not None:
            converted.append(part)
        else:
            return []
    return converted


def _cx_count(operations: list[cirq.Operation]) -> int:
    return sum(1 for op in operations if gate_name_for_operation(op) == "cx")


def _is_allowed_edge(op: cirq.Operation, backend: CompetitionBackend) -> bool:
    edge = tuple(sorted(qubit.x for qubit in op.qubits))
    return edge in {tuple(sorted(item)) for item in backend.coupling_map}
