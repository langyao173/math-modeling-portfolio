from __future__ import annotations

from collections import Counter
from itertools import combinations
import math
import random
import time

import cirq
import networkx as nx

from noise_adaptive_hackathon.baseline.common import decompose_swaps
from noise_adaptive_hackathon.evaluator.validate_circuit import count_two_qubit_gates, estimate_log_reliability
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .circuit_features import extract_circuit_features
from .ghz_tree import detect_exact_ghz_star
from .track_b import is_clifford_only


def compile_legacy_hub_parity(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    *,
    max_seconds: float = 8.0,
) -> tuple[cirq.Circuit, dict[cirq.LineQubit, int], float] | None:
    if not _is_large_hub_clifford(circuit, backend):
        return None
    deadline = time.perf_counter() + max_seconds
    started = time.perf_counter()
    initial_layout = _legacy_hub_aware_layout(circuit, backend)
    if initial_layout is None:
        return None

    try:
        best_circuit, _ = _compile_with_cirq_cqc(circuit, backend, 12, initial_layout)
    except Exception:
        return None
    hub_compiled = best_circuit
    best_layout = dict(initial_layout)
    best_cx = count_two_qubit_gates(best_circuit)
    best_elr = estimate_log_reliability(best_circuit, backend)

    logical_qubits = sorted(circuit.all_qubits(), key=lambda qubit: qubit.x)

    rng = random.Random(100)
    current_layout = dict(best_layout)
    current_circuit = best_circuit
    current_cx = best_cx
    current_elr = best_elr

    for _ in range(100):
        if time.perf_counter() >= deadline:
            return best_circuit, best_layout, time.perf_counter() - started
        proposal = _try_uniform_layout_swap(circuit, backend, best_layout, rng, logical_qubits)
        if proposal is None:
            continue
        layout, compiled, cx, elr = proposal
        if cx < best_cx or (cx == best_cx and elr > best_elr):
            best_layout, best_circuit, best_cx, best_elr = layout, compiled, cx, elr
            current_layout, current_circuit, current_cx, current_elr = layout, compiled, cx, elr

    rng = random.Random(7)
    current_layout = dict(best_layout)
    current_circuit = best_circuit
    current_cx = best_cx
    current_elr = best_elr
    for temp in (2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2):
        for _ in range(50):
            if time.perf_counter() >= deadline:
                return best_circuit, best_layout, time.perf_counter() - started
            proposal = _try_uniform_layout_swap(circuit, backend, current_layout, rng, logical_qubits)
            if proposal is None:
                continue
            layout, compiled, cx, elr = proposal
            delta = (cx - current_cx) * 10.0 + (current_elr - elr)
            if delta < 0.0 or rng.random() < math.exp(-delta / temp):
                current_layout, current_circuit, current_cx, current_elr = layout, compiled, cx, elr
                if cx < best_cx or (cx == best_cx and elr > best_elr):
                    best_layout, best_circuit, best_cx, best_elr = layout, compiled, cx, elr

    remaining = deadline - time.perf_counter()
    if remaining > 0.0:
        pruned_circuit, pruned_layout = _pruned_weighted_local_search(
            circuit=circuit,
            backend=backend,
            initial_layout=initial_layout,
            initial_circuit=hub_compiled,
            max_steps=80,
            ls_seed=100,
            sa_seed=7,
            sa_temps=(2.0, 1.5, 1.0),
            sa_steps_per_temp=60,
            max_seconds=min(remaining, 8.0),
        )
        if _accept_pruned_large_clifford(best_circuit, pruned_circuit, backend):
            best_circuit, best_layout = pruned_circuit, pruned_layout

    return best_circuit, best_layout, time.perf_counter() - started


def _is_large_hub_clifford(circuit: cirq.Circuit, backend: CompetitionBackend) -> bool:
    del backend
    if not is_clifford_only(circuit) or detect_exact_ghz_star(circuit) is not None:
        return False
    features = extract_circuit_features(circuit)
    if len(features.logical_qubits) < 16:
        return False
    graph = nx.Graph()
    for interaction in features.interactions:
        graph.add_edge(*interaction.pair)
    if graph.number_of_nodes() == 0:
        return False
    max_degree = max((graph.degree(node) for node in graph.nodes), default=0)
    hub_threshold = max(4, int(max_degree * 0.6))
    hubs = [node for node in graph.nodes if graph.degree(node) >= hub_threshold]
    hub_ratio = len(hubs) / graph.number_of_nodes()
    leaves = [node for node in graph.nodes if graph.degree(node) <= 2 and node not in hubs]
    leaf_fraction = len(leaves) / graph.number_of_nodes()
    return len(hubs) >= 3 and 0.15 <= hub_ratio <= 0.60 and leaf_fraction >= 0.45


def _legacy_hub_aware_layout(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
) -> dict[cirq.LineQubit, int] | None:
    graph = nx.Graph()
    for op in circuit.all_operations():
        if not isinstance(op.gate, cirq.CXPowGate):
            continue
        graph.add_edge(op.qubits[0].x, op.qubits[1].x)
    if graph.number_of_nodes() == 0:
        return None

    max_degree = max(graph.degree(node) for node in graph.nodes)
    hub_threshold = max(4, int(max_degree * 0.6))
    hubs = [node for node in graph.nodes if graph.degree(node) >= hub_threshold]
    node_count = graph.number_of_nodes()
    if len(hubs) < 3 or len(hubs) > node_count * 0.6 or len(hubs) < node_count * 0.15:
        return None

    leaves = [node for node in graph.nodes if node not in hubs]
    hub_leaves = {hub: [node for node in graph.neighbors(hub) if node in leaves] for hub in hubs}
    hub_edges = [(left, right) for left, right in combinations(hubs, 2) if graph.has_edge(left, right)]

    hardware = nx.Graph()
    hardware.add_edges_from((int(left), int(right)) for left, right in backend.coupling_map)
    shortest = dict(nx.all_pairs_shortest_path_length(hardware))

    def evaluate(hub_map: dict[int, int]) -> float:
        score = 0.0
        for left, right in hub_edges:
            score += shortest[hub_map[left]][hub_map[right]]
        for hub in hubs:
            nearest = min(shortest[hub_map[hub]][node] for node in hardware.nodes if node not in hub_map.values())
            score += len(hub_leaves[hub]) * nearest
        return score

    sorted_hubs = sorted(hubs, key=lambda hub: -graph.degree(hub))
    best_initial: dict[int, int] | None = None
    best_initial_score = float("inf")
    for center in list(hardware.nodes())[:10]:
        nodes = [center]
        seen = {center}
        for neighbor in hardware.neighbors(center):
            if neighbor not in seen:
                nodes.append(neighbor)
                seen.add(neighbor)
            for two_hop in hardware.neighbors(neighbor):
                if two_hop not in seen:
                    nodes.append(two_hop)
                    seen.add(two_hop)
        if len(nodes) < len(hubs):
            continue
        physical_subset = sorted(nodes[:10], key=lambda node: -hardware.degree(node))[: len(hubs)]
        mapping = {sorted_hubs[index]: physical_subset[index] for index in range(len(hubs))}
        score = evaluate(mapping)
        if score < best_initial_score:
            best_initial = mapping
            best_initial_score = score
    if best_initial is None:
        return None

    rng = random.Random(42)
    current = dict(best_initial)
    current_score = best_initial_score
    best = dict(current)
    best_score = current_score
    for temp in (5.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.3, 0.1):
        for _ in range(300):
            left, right = rng.sample(hubs, 2)
            trial = dict(current)
            trial[left], trial[right] = trial[right], trial[left]
            score = evaluate(trial)
            delta = score - current_score
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                current = trial
                current_score = score
                if score < best_score:
                    best = dict(trial)
                    best_score = score

    used = set(best.values())
    layout: dict[int, int] = dict(best)
    for hub in hubs:
        hub_physical = best[hub]
        available = [(shortest[hub_physical][node], node) for node in hardware.nodes if node not in used]
        available.sort()
        for leaf in hub_leaves[hub]:
            if not available:
                break
            _, physical = available.pop(0)
            layout[leaf] = physical
            used.add(physical)
    for node in graph.nodes:
        if node in layout:
            continue
        available = [(shortest[best[hubs[0]]][physical], physical) for physical in hardware.nodes if physical not in used]
        available.sort()
        if not available:
            break
        _, physical = available.pop(0)
        layout[node] = physical
        used.add(physical)
    return {cirq.LineQubit(logical): physical for logical, physical in layout.items()}


def _try_uniform_layout_swap(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    base_layout: dict[cirq.LineQubit, int],
    rng: random.Random,
    logical_qubits: list[cirq.LineQubit],
) -> tuple[dict[cirq.LineQubit, int], cirq.Circuit, int, float] | None:
    left, right = rng.sample(logical_qubits, 2)
    layout = dict(base_layout)
    layout[left], layout[right] = layout[right], layout[left]
    return _compile_trial_layout(circuit, backend, layout)


def _try_weighted_layout_swap(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    base_layout: dict[cirq.LineQubit, int],
    rng: random.Random,
    proposal_pairs: list[tuple[cirq.LineQubit, cirq.LineQubit]],
    proposal_weights: list[float],
    uniform_pairs: list[tuple[cirq.LineQubit, cirq.LineQubit]],
) -> tuple[dict[cirq.LineQubit, int], cirq.Circuit, int, float] | None:
    left, right = _sample_weighted_pair(rng, proposal_pairs, proposal_weights, uniform_pairs)
    layout = dict(base_layout)
    layout[left], layout[right] = layout[right], layout[left]
    return _compile_trial_layout(circuit, backend, layout)


def _compile_trial_layout(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    layout: dict[cirq.LineQubit, int],
) -> tuple[dict[cirq.LineQubit, int], cirq.Circuit, int, float] | None:
    try:
        compiled, _ = _compile_with_cirq_cqc(circuit, backend, 12, layout)
    except Exception:
        return None
    return dict(layout), compiled, count_two_qubit_gates(compiled), estimate_log_reliability(compiled, backend)


def _pruned_weighted_local_search(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    initial_layout: dict[cirq.LineQubit, int],
    initial_circuit: cirq.Circuit,
    max_steps: int,
    ls_seed: int,
    sa_seed: int,
    sa_temps: tuple[float, ...],
    sa_steps_per_temp: int,
    max_seconds: float,
) -> tuple[cirq.Circuit, dict[cirq.LineQubit, int]]:
    logical_qubits = sorted(circuit.all_qubits(), key=lambda qubit: qubit.x)
    proposal_pairs, proposal_weights, uniform_pairs = _build_weighted_proposal_pairs(circuit, logical_qubits)

    best_layout = dict(initial_layout)
    best_circuit = initial_circuit
    best_cx = count_two_qubit_gates(best_circuit)
    best_elr = estimate_log_reliability(best_circuit, backend)
    deadline = time.perf_counter() + max_seconds

    rng = random.Random(ls_seed)
    for _ in range(max_steps):
        if time.perf_counter() >= deadline:
            return best_circuit, best_layout
        proposal = _try_weighted_layout_swap(circuit, backend, best_layout, rng, proposal_pairs, proposal_weights, uniform_pairs)
        if proposal is None:
            continue
        layout, compiled, cx, elr = proposal
        if cx < best_cx or (cx == best_cx and elr > best_elr):
            best_layout, best_circuit, best_cx, best_elr = layout, compiled, cx, elr

    current_layout = dict(best_layout)
    current_cx = best_cx
    current_elr = best_elr
    rng = random.Random(sa_seed)
    for temp in sa_temps:
        for _ in range(sa_steps_per_temp):
            if time.perf_counter() >= deadline:
                return best_circuit, best_layout
            proposal = _try_weighted_layout_swap(circuit, backend, current_layout, rng, proposal_pairs, proposal_weights, uniform_pairs)
            if proposal is None:
                continue
            layout, compiled, cx, elr = proposal
            delta = (cx - current_cx) * 10.0 + (current_elr - elr)
            if delta < 0.0 or rng.random() < math.exp(-delta / temp):
                current_layout = layout
                current_cx = cx
                current_elr = elr
                if cx < best_cx or (cx == best_cx and elr > best_elr):
                    best_layout, best_circuit, best_cx, best_elr = layout, compiled, cx, elr
    return best_circuit, best_layout


def _accept_pruned_large_clifford(
    current_circuit: cirq.Circuit,
    pruned_circuit: cirq.Circuit,
    backend: CompetitionBackend,
) -> bool:
    current_cx = count_two_qubit_gates(current_circuit)
    pruned_cx = count_two_qubit_gates(pruned_circuit)
    if pruned_cx > current_cx - 8:
        return False
    if len(pruned_circuit) > len(current_circuit):
        return False
    current_elr = estimate_log_reliability(current_circuit, backend)
    pruned_elr = estimate_log_reliability(pruned_circuit, backend)
    return pruned_elr > current_elr + 0.03


def _build_weighted_proposal_pairs(
    circuit: cirq.Circuit,
    logical_qubits: list[cirq.LineQubit],
) -> tuple[list[tuple[cirq.LineQubit, cirq.LineQubit]], list[float], list[tuple[cirq.LineQubit, cirq.LineQubit]]]:
    weighted_degree: Counter[cirq.LineQubit] = Counter()
    graph_degree: Counter[cirq.LineQubit] = Counter()
    edge_weight: Counter[tuple[cirq.LineQubit, cirq.LineQubit]] = Counter()
    for op in circuit.all_operations():
        if len(op.qubits) != 2:
            continue
        left, right = sorted(op.qubits, key=lambda qubit: qubit.x)
        edge_weight[(left, right)] += 1
    for (left, right), weight in edge_weight.items():
        weighted_degree[left] += weight
        weighted_degree[right] += weight
        graph_degree[left] += 1
        graph_degree[right] += 1

    hot_count = min(len(logical_qubits), max(6, len(logical_qubits) // 3))
    hot = set(
        sorted(
            logical_qubits,
            key=lambda qubit: (weighted_degree[qubit], graph_degree[qubit], -qubit.x),
            reverse=True,
        )[:hot_count]
    )

    uniform_pairs: list[tuple[cirq.LineQubit, cirq.LineQubit]] = []
    pairs: list[tuple[cirq.LineQubit, cirq.LineQubit]] = []
    weights: list[float] = []
    for index, left in enumerate(logical_qubits):
        for right in logical_qubits[index + 1 :]:
            uniform_pairs.append((left, right))
            edge = tuple(sorted((left, right), key=lambda qubit: qubit.x))
            direct = edge_weight[edge] > 0
            hot_endpoint = left in hot or right in hot
            if not direct and not hot_endpoint:
                continue
            pair_weight = 1.0 + 2.0 * edge_weight[edge] + 0.25 * (weighted_degree[left] + weighted_degree[right])
            if hot_endpoint:
                pair_weight += 4.0
            if left in hot and right in hot:
                pair_weight += 3.0
            pairs.append((left, right))
            weights.append(pair_weight)
    if not pairs:
        return uniform_pairs, [1.0] * len(uniform_pairs), uniform_pairs
    return pairs, weights, uniform_pairs


def _sample_weighted_pair(
    rng: random.Random,
    pairs: list[tuple[cirq.LineQubit, cirq.LineQubit]],
    weights: list[float],
    uniform_pairs: list[tuple[cirq.LineQubit, cirq.LineQubit]],
) -> tuple[cirq.LineQubit, cirq.LineQubit]:
    if uniform_pairs and rng.random() < 0.25:
        return rng.choice(uniform_pairs)
    return rng.choices(pairs, weights=weights, k=1)[0]


def _compile_with_cirq_cqc(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    lookahead_radius: int,
    initial_layout: dict[cirq.LineQubit, int],
) -> tuple[cirq.Circuit, dict[cirq.LineQubit, int]]:
    physical_qubits = tuple(cirq.LineQubit.range(backend.num_qubits))
    device_graph = nx.Graph()
    device_graph.add_nodes_from(physical_qubits)
    device_graph.add_edges_from(
        (physical_qubits[left], physical_qubits[right])
        for left, right in backend.coupling_map
    )
    mapping = {
        logical: physical_qubits[physical]
        for logical, physical in initial_layout.items()
    }
    routed, initial_map, _ = cirq.RouteCQC(device_graph).route_circuit(
        circuit,
        lookahead_radius=lookahead_radius,
        initial_mapper=cirq.HardCodedInitialMapper(mapping),
    )
    layout = {logical: physical.x for logical, physical in initial_map.items()}
    return decompose_swaps(cirq.Circuit(routed)), layout
