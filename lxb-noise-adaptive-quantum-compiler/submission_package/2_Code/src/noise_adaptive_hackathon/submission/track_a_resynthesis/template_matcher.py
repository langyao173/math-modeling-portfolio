from __future__ import annotations

from dataclasses import dataclass

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation

from .exact_equivalence import circuit_without_measurements, unitary_equivalent
from .phase_polynomial import phase_polynomial_replacements
from .reversible_templates import reversible_template_replacements


@dataclass(frozen=True)
class SourceTemplateVariant:
    name: str
    circuit: cirq.Circuit
    start: int
    stop: int
    qubits: tuple[cirq.Qid, ...]


def source_template_variants(
    source: cirq.Circuit,
    *,
    limit: int = 6,
) -> list[SourceTemplateVariant]:
    operations = list(source.all_operations())
    if any(isinstance(op.gate, cirq.MeasurementGate) for op in operations[:-1]):
        return []
    logical_qubits = tuple(sorted(source.all_qubits(), key=lambda qubit: qubit.x))
    source_unitary_ops = list(circuit_without_measurements(source).all_operations())
    variants: list[SourceTemplateVariant] = []
    seen: set[tuple[str, ...]] = {tuple(str(op) for op in operations)}

    for start in range(len(operations)):
        if isinstance(operations[start].gate, cirq.MeasurementGate):
            break
        for stop in range(start + 2, min(len(operations), start + 20) + 1):
            block = tuple(operations[start:stop])
            if any(isinstance(op.gate, cirq.MeasurementGate) for op in block):
                break
            if not _interesting_block(block):
                continue
            qubits = tuple(sorted({qubit for op in block for qubit in op.qubits}, key=lambda qubit: qubit.x))
            if not 3 <= len(qubits) <= 5:
                continue
            replacements = [
                *reversible_template_replacements(block, qubits),
                *phase_polynomial_replacements(block, qubits),
            ]
            for replacement_index, replacement in enumerate(replacements):
                candidate_ops = [*operations[:start], *replacement, *operations[stop:]]
                signature = tuple(str(op) for op in candidate_ops)
                if signature in seen:
                    continue
                seen.add(signature)
                if not unitary_equivalent(list(block), replacement, qubits):
                    continue
                candidate_without_measurement = [
                    op for op in candidate_ops if not isinstance(op.gate, cirq.MeasurementGate)
                ]
                if not unitary_equivalent(source_unitary_ops, candidate_without_measurement, logical_qubits):
                    continue
                variants.append(
                    SourceTemplateVariant(
                        name=f"template_s{start}_e{stop}_r{replacement_index}",
                        circuit=cirq.Circuit(candidate_ops),
                        start=start,
                        stop=stop,
                        qubits=qubits,
                    )
                )
                if len(variants) >= limit:
                    return variants
    return variants


def _interesting_block(operations: tuple[cirq.Operation, ...]) -> bool:
    names = [gate_name_for_operation(op) for op in operations]
    if any(name is None for name in names):
        return False
    cx_count = sum(1 for name in names if name == "cx")
    if cx_count < 2:
        return False
    has_phase = any(name in {"t", "tdg", "s", "sdg", "z", "rz"} for name in names)
    return cx_count >= 4 or has_phase
