from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import itertools
import math
import time

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import (
    count_two_qubit_gates,
    estimate_log_reliability,
    gate_name_for_operation,
)
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .clifford_sensitivity import clifford_sensitivity_cost
from .legacy_parity import _is_large_hub_clifford
from .selection import CompilationCandidate, candidate_metrics
from .track_b import is_clifford_only
from .validation import validate_or_raise


@dataclass(frozen=True)
class AffineParityStar:
    logical_qubits: tuple[cirq.LineQubit, ...]
    hubs: tuple[int, ...]
    constants: tuple[int, ...]
    leaf_assignments: tuple[tuple[int, int, int], ...]


def measurement_channel_parity_candidate(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    legacy: CompilationCandidate | None,
    *,
    max_seconds: float = 2.0,
) -> CompilationCandidate | None:
    if legacy is None or not _is_large_hub_clifford(source, backend):
        return None
    started = time.perf_counter()
    star = detect_affine_parity_star(source)
    if star is None:
        return None
    layout = _place_physical_stars(star, backend, max_seconds=max_seconds)
    if layout is None:
        return None
    compiled = _build_physical_star_circuit(star, layout)
    try:
        validate_or_raise(source, compiled, backend)
    except Exception:
        return None
    candidate = CompilationCandidate(
        "v13c_measurement_parity_star",
        compiled,
        dict(layout),
        time.perf_counter() - started,
    )
    if not _accept_measurement_channel_candidate(candidate, legacy, backend, source):
        return None
    return candidate


def detect_affine_parity_star(circuit: cirq.Circuit) -> AffineParityStar | None:
    if not is_clifford_only(circuit):
        return None
    operations = list(circuit.all_operations())
    measurements = [op for op in operations if isinstance(op.gate, cirq.MeasurementGate)]
    if len(measurements) != 1 or operations[-1] is not measurements[0]:
        return None
    if cirq.measurement_key_name(measurements[0]) != "z":
        return None

    logical_qubits = tuple(sorted(circuit.all_qubits(), key=lambda qubit: qubit.x))
    if len(logical_qubits) < 16 or tuple(measurements[0].qubits) != logical_qubits:
        return None
    qubit_index = {qubit: index for index, qubit in enumerate(logical_qubits)}
    width = len(logical_qubits)
    rows = [1 << index for index in range(width)]
    constants = [0 for _ in range(width)]
    hubs: list[int] = []
    seen_body = False

    for op in operations[:-1]:
        name = gate_name_for_operation(op)
        if name == "h":
            if seen_body or len(op.qubits) != 1:
                return None
            hub = qubit_index[op.qubits[0]]
            if hub in hubs:
                return None
            rows[hub] = 1 << hub
            constants[hub] = 0
            hubs.append(hub)
        elif name == "x":
            seen_body = True
            constants[qubit_index[op.qubits[0]]] ^= 1
        elif name == "cx":
            seen_body = True
            control = qubit_index[op.qubits[0]]
            target = qubit_index[op.qubits[1]]
            rows[target] ^= rows[control]
            constants[target] ^= constants[control]
        else:
            return None

    if len(hubs) < 3 or len(hubs) > width // 2:
        return None
    hub_mask = sum(1 << hub for hub in hubs)
    non_hub_mask = ((1 << width) - 1) ^ hub_mask
    if any(rows[hub] & non_hub_mask for hub in hubs):
        return None
    if _gf2_rank([rows[hub] & hub_mask for hub in hubs]) != len(hubs):
        return None

    assignments: list[tuple[int, int, int]] = []
    leaf_counts = {hub: 0 for hub in hubs}
    for leaf in range(width):
        if leaf in leaf_counts:
            continue
        matches = [
            hub
            for hub in hubs
            if rows[leaf] ^ rows[hub] == (1 << leaf)
        ]
        if len(matches) != 1:
            return None
        hub = matches[0]
        leaf_counts[hub] += 1
        assignments.append((hub, leaf, constants[leaf] ^ constants[hub]))

    if any(count < 2 for count in leaf_counts.values()):
        return None
    return AffineParityStar(
        logical_qubits=logical_qubits,
        hubs=tuple(hubs),
        constants=tuple(constants),
        leaf_assignments=tuple(assignments),
    )


def _gf2_rank(values: list[int]) -> int:
    basis: dict[int, int] = {}
    rank = 0
    for value in values:
        current = value
        while current:
            pivot = current.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = current
                rank += 1
                break
            current ^= basis[pivot]
    return rank


def _place_physical_stars(
    star: AffineParityStar,
    backend: CompetitionBackend,
    *,
    max_seconds: float,
    option_limit: int = 96,
    beam_width: int = 512,
) -> dict[cirq.LineQubit, int] | None:
    deadline = time.perf_counter() + max_seconds
    leaves_by_hub: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for hub, leaf, flip in star.leaf_assignments:
        leaves_by_hub[hub].append((leaf, flip))

    neighbors: dict[int, list[int]] = defaultdict(list)
    edge_cost: dict[tuple[int, int], float] = {}
    for left, right in backend.coupling_map:
        neighbors[left].append(right)
        neighbors[right].append(left)
        edge_cost[tuple(sorted((left, right)))] = -math.log(max(1.0 - backend.edge_error(left, right), 1e-12))

    option_groups: list[tuple[int, list[tuple[float, int, tuple[int, ...]]]]] = []
    for hub in sorted(star.hubs, key=lambda item: (-len(leaves_by_hub[item]), item)):
        leaf_count = len(leaves_by_hub[hub])
        options: list[tuple[float, int, tuple[int, ...]]] = []
        for center in range(backend.num_qubits):
            if len(neighbors[center]) < leaf_count:
                continue
            nearby = sorted(
                neighbors[center],
                key=lambda node: (
                    edge_cost[tuple(sorted((center, node)))],
                    backend.readout(node),
                    -len(neighbors[node]),
                    node,
                ),
            )
            for leaves in itertools.combinations(nearby, leaf_count):
                cost = _physical_star_cost(center, leaves, backend, edge_cost)
                options.append((cost, center, tuple(leaves)))
        if not options:
            return None
        option_groups.append((hub, sorted(options)[:option_limit]))

    beams: list[tuple[float, frozenset[int], dict[cirq.LineQubit, int]]] = [(0.0, frozenset(), {})]
    for hub, options in option_groups:
        if time.perf_counter() >= deadline:
            return None
        next_beams: list[tuple[float, frozenset[int], dict[cirq.LineQubit, int]]] = []
        logical_hub = star.logical_qubits[hub]
        logical_leaves = [star.logical_qubits[leaf] for leaf, _ in leaves_by_hub[hub]]
        for current_cost, used, current_layout in beams:
            for option_cost, center, leaves in options:
                nodes = frozenset((center, *leaves))
                if used & nodes:
                    continue
                layout = dict(current_layout)
                layout[logical_hub] = center
                for logical_leaf, physical_leaf in zip(logical_leaves, leaves):
                    layout[logical_leaf] = physical_leaf
                next_beams.append((current_cost + option_cost, used | nodes, layout))
        if not next_beams:
            return None
        beams = sorted(next_beams, key=lambda item: (item[0], tuple(sorted(item[1]))))[:beam_width]

    if not beams:
        return None
    return dict(min(beams, key=lambda item: item[0])[2])


def _physical_star_cost(
    center: int,
    leaves: tuple[int, ...],
    backend: CompetitionBackend,
    edge_cost: dict[tuple[int, int], float],
) -> float:
    readout = 0.18 * _readout_cost(backend, center)
    readout += 0.06 * sum(_readout_cost(backend, leaf) for leaf in leaves)
    two_qubit = sum(edge_cost[tuple(sorted((center, leaf)))] for leaf in leaves)
    degree_bonus = -0.001 * len(leaves)
    return two_qubit + readout + degree_bonus


def _readout_cost(backend: CompetitionBackend, qubit: int) -> float:
    return -math.log(max(1.0 - backend.readout(qubit), 1e-12))


def _build_physical_star_circuit(
    star: AffineParityStar,
    layout: dict[cirq.LineQubit, int],
) -> cirq.Circuit:
    physical = {
        logical: cirq.LineQubit(physical_index)
        for logical, physical_index in layout.items()
    }
    circuit = cirq.Circuit()
    for hub in star.hubs:
        logical_hub = star.logical_qubits[hub]
        circuit.append(cirq.H(physical[logical_hub]))
        if star.constants[hub]:
            circuit.append(cirq.X(physical[logical_hub]))

    leaves_by_hub: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for hub, leaf, flip in star.leaf_assignments:
        leaves_by_hub[hub].append((leaf, flip))
    max_fanout = max((len(leaves) for leaves in leaves_by_hub.values()), default=0)
    for offset in range(max_fanout):
        for hub in star.hubs:
            leaves = leaves_by_hub[hub]
            if offset >= len(leaves):
                continue
            leaf, flip = leaves[offset]
            logical_hub = star.logical_qubits[hub]
            logical_leaf = star.logical_qubits[leaf]
            circuit.append(cirq.CNOT(physical[logical_hub], physical[logical_leaf]))
            if flip:
                circuit.append(cirq.X(physical[logical_leaf]))

    circuit.append(cirq.measure(*(physical[logical] for logical in star.logical_qubits), key="z"))
    return circuit


def _accept_measurement_channel_candidate(
    candidate: CompilationCandidate,
    legacy: CompilationCandidate,
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> bool:
    candidate_value = candidate_metrics(candidate, backend, source)
    legacy_value = candidate_metrics(legacy, backend, source)
    return (
        candidate_value.two_qubit_count <= legacy_value.two_qubit_count - 8
        and candidate_value.depth <= legacy_value.depth
        and candidate_value.log_reliability >= legacy_value.log_reliability + 0.05
        and candidate_value.clifford_sensitivity_cost <= legacy_value.clifford_sensitivity_cost
        and count_two_qubit_gates(candidate.circuit) <= count_two_qubit_gates(source)
        and estimate_log_reliability(candidate.circuit, backend) > estimate_log_reliability(legacy.circuit, backend)
        and clifford_sensitivity_cost(candidate.circuit, backend) <= clifford_sensitivity_cost(legacy.circuit, backend)
    )
