from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cirq
import numpy as np

from noise_adaptive_hackathon.evaluator.validate_circuit import count_two_qubit_gates, gate_name_for_operation

from .track_a_resynthesis.reversible_templates import toffoli_6cx


@dataclass(frozen=True)
class RelativePhaseToffoliRewrite:
    name: str
    circuit: cirq.Circuit
    replacements: int
    old_cx: int
    new_cx: int


@dataclass(frozen=True)
class _Match:
    start: int
    stop: int
    matched_indices: tuple[int, ...]
    replacement: tuple[cirq.Operation, ...]
    label: str


def relative_phase_toffoli_rewrites(
    circuit: cirq.Circuit,
    *,
    max_replacements: int = 2,
) -> list[RelativePhaseToffoliRewrite]:
    operations = tuple(circuit.all_operations())
    if not _has_single_final_measurement(operations):
        return []
    active_qubits = tuple(sorted(circuit.all_qubits(), key=lambda qubit: qubit.x))
    if not (3 <= len(active_qubits) <= 7):
        return []

    matches = _find_matches(operations)
    if not matches:
        return []

    rewrites: list[RelativePhaseToffoliRewrite] = []
    seen: set[tuple[str, ...]] = set()
    for count in range(1, min(max_replacements, len(matches)) + 1):
        for selected in combinations(matches, count):
            if _overlaps(selected):
                continue
            rewritten = cirq.Circuit(_apply_matches(operations, selected))
            signature = tuple(str(op) for op in rewritten.all_operations())
            if signature in seen:
                continue
            seen.add(signature)
            old_cx = count_two_qubit_gates(circuit)
            new_cx = count_two_qubit_gates(rewritten)
            if new_cx >= old_cx:
                continue
            if not final_z_measurement_equivalent(circuit, rewritten):
                continue
            labels = "_".join(match.label for match in selected)
            rewrites.append(
                RelativePhaseToffoliRewrite(
                    name=f"relative_phase_toffoli_{labels}",
                    circuit=rewritten,
                    replacements=len(selected),
                    old_cx=old_cx,
                    new_cx=new_cx,
                )
            )
    return sorted(rewrites, key=lambda item: (item.new_cx, -item.replacements, item.name))


def final_z_measurement_equivalent(
    left: cirq.Circuit,
    right: cirq.Circuit,
    *,
    atol: float = 1e-8,
) -> bool:
    left_measurement = _single_final_measurement(left)
    right_measurement = _single_final_measurement(right)
    if left_measurement is None or right_measurement is None:
        return False
    if tuple(left_measurement.qubits) != tuple(right_measurement.qubits):
        return False
    qubits = tuple(left_measurement.qubits)
    if set(left.all_qubits()) != set(qubits) or set(right.all_qubits()) != set(qubits):
        return False
    if len(qubits) > 7:
        return False
    try:
        left_unitary = _unitary_without_measurement(left, qubits)
        right_unitary = _unitary_without_measurement(right, qubits)
    except Exception:
        return False
    if left_unitary.shape != right_unitary.shape:
        return False
    for row_index in range(left_unitary.shape[0]):
        left_row = left_unitary[row_index, :]
        right_row = right_unitary[row_index, :]
        pivot_index = int(np.argmax(np.abs(left_row)))
        if abs(left_row[pivot_index]) < atol:
            if np.max(np.abs(right_row)) >= atol:
                return False
            continue
        phase = right_row[pivot_index] / left_row[pivot_index]
        if abs(phase) < atol:
            return False
        phase /= abs(phase)
        if np.max(np.abs(right_row - phase * left_row)) > atol:
            return False
    return True


def _find_matches(operations: tuple[cirq.Operation, ...]) -> list[_Match]:
    non_measure_qubits = sorted(
        {qubit for op in operations if gate_name_for_operation(op) != "measure" for qubit in op.qubits},
        key=lambda qubit: qubit.x,
    )
    matches: list[_Match] = []
    for target in non_measure_qubits:
        controls = [qubit for qubit in non_measure_qubits if qubit != target]
        for control_a in controls:
            for control_b in controls:
                if control_a == control_b:
                    continue
                template = tuple(toffoli_6cx(control_a, control_b, target))
                replacement = tuple(_relative_phase_toffoli(control_a, control_b, target))
                for start in range(len(operations)):
                    match = _match_template(operations, start, template, replacement)
                    if match is not None:
                        matches.append(
                            _Match(
                                match.start,
                                match.stop,
                                match.matched_indices,
                                match.replacement,
                                f"{control_a.x}_{control_b.x}_{target.x}",
                            )
                        )
    return _dedupe_matches(matches)


def _match_template(
    operations: tuple[cirq.Operation, ...],
    start: int,
    template: tuple[cirq.Operation, ...],
    replacement: tuple[cirq.Operation, ...],
) -> _Match | None:
    template_qubits = {qubit for op in template for qubit in op.qubits}
    matched: list[int] = []
    template_index = 0
    index = start
    while index < len(operations) and template_index < len(template):
        op = operations[index]
        gate_name = gate_name_for_operation(op)
        if gate_name == "measure":
            return None
        if _same_operation(op, template[template_index]):
            matched.append(index)
            template_index += 1
            index += 1
            continue
        if set(op.qubits).isdisjoint(template_qubits):
            index += 1
            continue
        return None
    if template_index != len(template):
        return None
    return _Match(start=matched[0], stop=index, matched_indices=tuple(matched), replacement=replacement, label="")


def _apply_matches(
    operations: tuple[cirq.Operation, ...],
    selected: tuple[_Match, ...],
) -> list[cirq.Operation]:
    by_start = {match.matched_indices[0]: match for match in selected}
    skipped = {index for match in selected for index in match.matched_indices}
    rewritten: list[cirq.Operation] = []
    for index, op in enumerate(operations):
        match = by_start.get(index)
        if match is not None:
            rewritten.extend(match.replacement)
            continue
        if index in skipped:
            continue
        rewritten.append(op)
    return rewritten


def _relative_phase_toffoli(control_a: cirq.Qid, control_b: cirq.Qid, target: cirq.Qid) -> list[cirq.Operation]:
    return [
        cirq.H(target),
        cirq.T(target),
        cirq.CNOT(control_b, target),
        cirq.T(target) ** -1,
        cirq.CNOT(control_a, target),
        cirq.T(target),
        cirq.CNOT(control_b, target),
        cirq.T(target) ** -1,
        cirq.H(target),
    ]


def _single_final_measurement(circuit: cirq.Circuit) -> cirq.Operation | None:
    operations = list(circuit.all_operations())
    measurements = [op for op in operations if gate_name_for_operation(op) == "measure"]
    if len(measurements) != 1:
        return None
    measurement = measurements[0]
    if operations[-1] is not measurement:
        return None
    if cirq.measurement_key_name(measurement) != "z":
        return None
    return measurement


def _has_single_final_measurement(operations: tuple[cirq.Operation, ...]) -> bool:
    measurements = [op for op in operations if gate_name_for_operation(op) == "measure"]
    return len(measurements) == 1 and operations and operations[-1] is measurements[0]


def _unitary_without_measurement(circuit: cirq.Circuit, qubits: tuple[cirq.Qid, ...]) -> np.ndarray:
    return cirq.Circuit(
        op for op in circuit.all_operations() if gate_name_for_operation(op) != "measure"
    ).unitary(qubit_order=list(qubits))


def _same_operation(left: cirq.Operation, right: cirq.Operation) -> bool:
    if gate_name_for_operation(left) != gate_name_for_operation(right):
        return False
    return tuple(left.qubits) == tuple(right.qubits)


def _dedupe_matches(matches: list[_Match]) -> list[_Match]:
    seen: set[tuple[int, tuple[int, ...], tuple[str, ...]]] = set()
    deduped: list[_Match] = []
    for match in sorted(matches, key=lambda item: (item.start, item.stop, item.label)):
        key = (match.start, match.matched_indices, tuple(str(op) for op in match.replacement))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(match)
    return deduped


def _overlaps(matches: tuple[_Match, ...]) -> bool:
    used: set[int] = set()
    for match in matches:
        current = set(match.matched_indices)
        if used & current:
            return True
        used |= current
    return False
