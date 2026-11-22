from __future__ import annotations

from dataclasses import dataclass

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation

from .cnot_block_gf2 import cnot_block_gf2_replacements
from .exact_equivalence import circuit_without_measurements, unitary_equivalent
from .phase_polynomial_real import phase_polynomial_rewrite_candidates
from .reversible_template_library import reversible_library_replacements


@dataclass(frozen=True)
class SemanticSourceVariant:
    name: str
    circuit: cirq.Circuit
    start: int
    stop: int
    qubits: tuple[cirq.Qid, ...]
    reason: str


def semantic_source_variants(
    source: cirq.Circuit,
    *,
    limit: int = 10,
) -> list[SemanticSourceVariant]:
    operations = list(source.all_operations())
    if any(isinstance(op.gate, cirq.MeasurementGate) for op in operations[:-1]):
        return []
    logical_qubits = tuple(sorted(source.all_qubits(), key=lambda qubit: qubit.x))
    source_ops = list(circuit_without_measurements(source).all_operations())
    variants: list[SemanticSourceVariant] = []
    seen: set[tuple[str, ...]] = {tuple(str(op) for op in operations)}

    for replacement in cnot_block_gf2_replacements(operations, limit=limit):
        _append_variant(
            variants,
            seen,
            operations,
            source_ops,
            logical_qubits,
            replacement.start,
            replacement.stop,
            replacement.qubits,
            replacement.operations,
            reason="cnot_block_gf2",
            limit=limit,
        )
        if len(variants) >= limit:
            return variants

    for replacement in phase_polynomial_rewrite_candidates(operations, limit=limit):
        _append_variant(
            variants,
            seen,
            operations,
            source_ops,
            logical_qubits,
            replacement.start,
            replacement.stop,
            replacement.qubits,
            replacement.operations,
            reason="phase_polynomial",
            limit=limit,
        )
        if len(variants) >= limit:
            return variants

    for start in range(len(operations)):
        if isinstance(operations[start].gate, cirq.MeasurementGate):
            break
        for stop in range(start + 4, min(len(operations), start + 20) + 1):
            block = tuple(operations[start:stop])
            if any(isinstance(op.gate, cirq.MeasurementGate) for op in block):
                break
            if sum(1 for op in block if gate_name_for_operation(op) == "cx") < 4:
                continue
            qubits = tuple(sorted({qubit for op in block for qubit in op.qubits}, key=lambda qubit: qubit.x))
            if not 3 <= len(qubits) <= 5:
                continue
            for index, replacement in enumerate(reversible_library_replacements(block, qubits)):
                _append_variant(
                    variants,
                    seen,
                    operations,
                    source_ops,
                    logical_qubits,
                    start,
                    stop,
                    qubits,
                    replacement,
                    reason=f"reversible_template_{index}",
                    limit=limit,
                )
                if len(variants) >= limit:
                    return variants
    return variants


def _append_variant(
    variants: list[SemanticSourceVariant],
    seen: set[tuple[str, ...]],
    operations: list[cirq.Operation],
    source_ops: list[cirq.Operation],
    logical_qubits: tuple[cirq.Qid, ...],
    start: int,
    stop: int,
    qubits: tuple[cirq.Qid, ...],
    replacement: list[cirq.Operation],
    *,
    reason: str,
    limit: int,
) -> None:
    if len(variants) >= limit:
        return
    block = operations[start:stop]
    if not unitary_equivalent(block, replacement, qubits):
        return
    candidate_ops = [*operations[:start], *replacement, *operations[stop:]]
    signature = tuple(str(op) for op in candidate_ops)
    if signature in seen:
        return
    seen.add(signature)
    candidate_without_measurement = [
        op for op in candidate_ops if not isinstance(op.gate, cirq.MeasurementGate)
    ]
    if not unitary_equivalent(source_ops, candidate_without_measurement, logical_qubits):
        return
    variants.append(
        SemanticSourceVariant(
            name=f"semantic_{reason}_s{start}_e{stop}",
            circuit=cirq.Circuit(candidate_ops),
            start=start,
            stop=stop,
            qubits=qubits,
            reason=reason,
        )
    )
