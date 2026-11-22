from __future__ import annotations

import random
import time

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import count_two_qubit_gates, estimate_log_reliability, gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .circuit_features import extract_circuit_features
from .clifford_resynthesis import cnot_block_matrix, synthesize_cnot_matrix
from .clifford_sensitivity import clifford_sensitivity_cost
from .legacy_parity import _compile_with_cirq_cqc, _is_large_hub_clifford, _legacy_hub_aware_layout
from .peephole import conservative_peephole
from .selection import CompilationCandidate, candidate_metrics
from .simulation_selector import SimulationSelectorConfig, _ideal_output_support, _noisy_support_success
from .validation import validate_or_raise


def gf2_parity_resynthesis_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    legacy: CompilationCandidate | None,
    *,
    seed: int = 2026,
    max_seconds: float = 4.0,
    max_orders: int = 48,
) -> list[CompilationCandidate]:
    if legacy is None or not _is_large_hub_clifford(source, backend):
        return []
    deadline = time.perf_counter() + max_seconds
    support_config = SimulationSelectorConfig(track_b_lowshot=512, track_b_timeout_sec=1.0)
    support = _ideal_output_support(source, track_b=True, seed=7, config=support_config)
    if not support:
        return []

    legacy_metrics = candidate_metrics(legacy, backend, source)
    legacy_support = _noisy_support_success(
        legacy.circuit,
        backend,
        support,
        track_b=True,
        shots=support_config.track_b_lowshot,
        seed=7,
        config=support_config,
    )
    if legacy_support is None:
        return []

    candidates: list[CompilationCandidate] = []
    seen_circuits: set[tuple[str, ...]] = set()
    for index, order in enumerate(_cnot_block_order_portfolio(source, seed=seed, max_orders=max_orders)):
        if time.perf_counter() >= deadline:
            break
        rewritten, changed = compress_logical_cnot_blocks(source, first_block_order=order)
        if not changed:
            continue
        try:
            layout = _legacy_hub_aware_layout(rewritten, backend)
            if layout is None:
                continue
            started = time.perf_counter()
            compiled, routed_layout = _compile_with_cirq_cqc(rewritten, backend, 12, layout)
            compiled = conservative_peephole(compiled, clifford_only=True)
            validate_or_raise(source, compiled, backend)
            signature = tuple(str(op) for op in compiled.all_operations())
            if signature in seen_circuits:
                continue
            seen_circuits.add(signature)
        except Exception:
            continue

        if not _accept_gf2_candidate(compiled, backend, source, legacy_metrics, legacy_support, support, support_config):
            continue
        candidates.append(
            CompilationCandidate(
                f"v13a_gf2_parity_order{index}",
                compiled,
                routed_layout,
                time.perf_counter() - started,
            )
        )
    return candidates


def compress_logical_cnot_blocks(
    circuit: cirq.Circuit,
    *,
    first_block_order: tuple[cirq.LineQubit, ...] | None = None,
) -> tuple[cirq.Circuit, bool]:
    operations = list(circuit.all_operations())
    rewritten: list[cirq.Operation] = []
    changed = False
    block_index = 0
    index = 0
    while index < len(operations):
        if gate_name_for_operation(operations[index]) != "cx":
            rewritten.append(operations[index])
            index += 1
            continue

        start = index
        while index < len(operations) and gate_name_for_operation(operations[index]) == "cx":
            index += 1
        block = operations[start:index]
        replacement = _compress_cnot_block(block, first_block_order if block_index == 0 else None)
        block_index += 1
        if replacement is not None and len(replacement) < len(block):
            rewritten.extend(replacement)
            changed = True
        else:
            rewritten.extend(block)
    return cirq.Circuit(rewritten), changed


def _compress_cnot_block(
    block: list[cirq.Operation],
    order: tuple[cirq.LineQubit, ...] | None,
) -> list[cirq.Operation] | None:
    if len(block) < 3:
        return None
    qubits = sorted({qubit for op in block for qubit in op.qubits}, key=lambda qubit: qubit.x)
    if order is not None:
        order_set = set(order)
        if order_set != set(qubits):
            return None
        qubits = list(order)
    try:
        matrix = cnot_block_matrix(block, qubits)
        synthesized = synthesize_cnot_matrix(matrix)
    except Exception:
        return None
    if len(synthesized) >= len(block):
        return None
    return [cirq.CNOT(qubits[control], qubits[target]) for control, target in synthesized]


def _cnot_block_order_portfolio(
    source: cirq.Circuit,
    *,
    seed: int,
    max_orders: int,
) -> list[tuple[cirq.LineQubit, ...]]:
    first_block = _first_large_cnot_block(source)
    if not first_block:
        return []
    qubits = tuple(sorted({qubit for op in first_block for qubit in op.qubits}, key=lambda qubit: qubit.x))
    if len(qubits) < 3:
        return []

    features = extract_circuit_features(source)
    weighted_degree = {qubit: features.weighted_degrees.get(qubit.x, 0.0) for qubit in qubits}
    orders: list[tuple[cirq.LineQubit, ...]] = [
        qubits,
        tuple(sorted(qubits, key=lambda qubit: (weighted_degree[qubit], -qubit.x), reverse=True)),
        tuple(sorted(qubits, key=lambda qubit: (weighted_degree[qubit], qubit.x))),
    ]
    rng = random.Random(seed)
    while len(orders) < max_orders:
        shuffled = list(qubits)
        rng.shuffle(shuffled)
        candidate = tuple(shuffled)
        if candidate not in orders:
            orders.append(candidate)
    return orders


def _first_large_cnot_block(source: cirq.Circuit) -> list[cirq.Operation]:
    operations = list(source.all_operations())
    index = 0
    while index < len(operations):
        if gate_name_for_operation(operations[index]) != "cx":
            index += 1
            continue
        start = index
        while index < len(operations) and gate_name_for_operation(operations[index]) == "cx":
            index += 1
        block = operations[start:index]
        if len(block) >= 3:
            return block
    return []


def _accept_gf2_candidate(
    compiled: cirq.Circuit,
    backend: CompetitionBackend,
    source: cirq.Circuit,
    legacy_metrics,
    legacy_support: float,
    support: set[str],
    support_config: SimulationSelectorConfig,
) -> bool:
    cx = count_two_qubit_gates(compiled)
    depth = len(compiled)
    elr = estimate_log_reliability(compiled, backend)
    sensitivity = clifford_sensitivity_cost(compiled, backend)
    support_success = _noisy_support_success(
        compiled,
        backend,
        support,
        track_b=True,
        shots=support_config.track_b_lowshot,
        seed=7,
        config=support_config,
    )
    if support_success is None:
        return False
    if support_success >= legacy_support + 0.03:
        return True
    cx_win = cx <= legacy_metrics.two_qubit_count - 8
    sensitivity_win = sensitivity <= legacy_metrics.clifford_sensitivity_cost - 0.35
    proxy_not_worse = (
        depth <= legacy_metrics.depth + 8
        and elr >= legacy_metrics.log_reliability - 0.08
        and support_success >= legacy_support - 0.005
    )
    return (cx_win or sensitivity_win) and proxy_not_worse
