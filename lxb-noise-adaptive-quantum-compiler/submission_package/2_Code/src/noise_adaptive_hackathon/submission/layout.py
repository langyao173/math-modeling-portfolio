from __future__ import annotations

from collections import Counter
from itertools import permutations
import math
import random

import cirq
import networkx as nx

from noise_adaptive_hackathon.baseline.common import movement_aware_layout, reliability_layout, trivial_layout
from noise_adaptive_hackathon.baseline.paper_greedy import paper_greedy_edge_layout, paper_greedy_vertex_layout
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .backend_model import BackendNoiseModel
from .circuit_features import CircuitFeatures


Layout = dict[cirq.LineQubit, int]


def choose_initial_layout(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    model: BackendNoiseModel,
    features: CircuitFeatures,
) -> Layout:
    candidates = candidate_initial_layouts(circuit, backend, model, features)
    return min(
        candidates,
        key=lambda layout: (
            score_layout(layout, model, features),
            tuple(layout[qubit] for qubit in features.logical_qubits),
        ),
    )


def candidate_initial_layouts(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    model: BackendNoiseModel,
    features: CircuitFeatures,
    *,
    limit: int | None = None,
) -> list[Layout]:
    candidates = [
        trivial_layout(circuit, backend),
        movement_aware_layout(circuit, backend),
        reliability_layout(circuit, backend),
        paper_greedy_vertex_layout(circuit, backend),
        paper_greedy_edge_layout(circuit, backend),
        greedy_noise_layout(model, features),
        low_readout_layout(model, features),
        low_noise_subgraph_layout(model, features),
        annealed_layout(model, features),
    ]
    hub_layout = hub_aware_layout(circuit, backend, model)
    if hub_layout is not None:
        candidates.append(hub_layout)
    if len(features.logical_qubits) <= 7:
        candidates.extend(small_simulation_layouts(circuit, backend, model, features, limit=10))
    unique: list[Layout] = []
    seen: set[tuple[int, ...]] = set()
    for layout in candidates:
        if not _is_valid_layout(layout, backend, features):
            continue
        signature = tuple(layout[qubit] for qubit in features.logical_qubits)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(layout)
    ranked = sorted(
        unique,
        key=lambda layout: (
            score_layout(layout, model, features),
            tuple(layout[qubit] for qubit in features.logical_qubits),
        ),
    )
    return ranked[:limit] if limit is not None else ranked


def greedy_noise_layout(model: BackendNoiseModel, features: CircuitFeatures) -> Layout:
    available = set(range(model.backend.num_qubits))
    unplaced = set(features.logical_qubits)
    layout: Layout = {}

    if features.interactions:
        seed = max(
            features.interactions,
            key=lambda item: (
                item.weight,
                features.weighted_degrees[item.pair[0]] + features.weighted_degrees[item.pair[1]],
                -item.pair[0],
                -item.pair[1],
            ),
        )
        physical_edge = min(
            model.coupling_edges,
            key=lambda edge: (
                model.edge_cost(*edge),
                model.readout_cost(edge[0]) + model.readout_cost(edge[1]),
                -len(model.neighbors[edge[0]]) - len(model.neighbors[edge[1]]),
                edge,
            ),
        )
        logical_pair = sorted(
            (cirq.LineQubit(seed.pair[0]), cirq.LineQubit(seed.pair[1])),
            key=lambda qubit: (features.weighted_degrees[qubit.x], -qubit.x),
            reverse=True,
        )
        physical_pair = sorted(
            physical_edge,
            key=lambda node: (model.readout_cost(node), -len(model.neighbors[node]), node),
        )
        for logical, physical in zip(logical_pair, physical_pair):
            _place(layout, unplaced, available, logical, physical)

    while unplaced:
        logical = max(
            unplaced,
            key=lambda qubit: (
                _placed_neighbor_weight(qubit.x, layout, features),
                features.weighted_degrees[qubit.x],
                -qubit.x,
            ),
        )
        physical = min(
            available,
            key=lambda node: (
                _incremental_layout_cost(logical.x, node, layout, model, features),
                model.readout_cost(node),
                -len(model.neighbors[node]),
                node,
            ),
        )
        _place(layout, unplaced, available, logical, physical)

    return layout


def low_readout_layout(model: BackendNoiseModel, features: CircuitFeatures) -> Layout:
    logical_order = sorted(
        features.logical_qubits,
        key=lambda qubit: (features.weighted_degrees[qubit.x], -qubit.x),
        reverse=True,
    )
    physical_order = sorted(
        range(model.backend.num_qubits),
        key=lambda node: (model.readout_cost(node), -len(model.neighbors[node]), node),
    )
    return {
        logical: physical
        for logical, physical in zip(logical_order, physical_order)
    }


def low_noise_subgraph_layout(model: BackendNoiseModel, features: CircuitFeatures) -> Layout:
    width = len(features.logical_qubits)
    if width == 0:
        return {}
    center = min(
        range(model.backend.num_qubits),
        key=lambda node: (
            _local_subgraph_cost(node, width, model),
            model.readout_cost(node),
            node,
        ),
    )
    physical_order = sorted(
        range(model.backend.num_qubits),
        key=lambda node: (
            model.noise_distance(center, node) + 0.2 * model.hop_distance(center, node),
            model.readout_cost(node),
            node,
        ),
    )[:width]
    logical_order = sorted(
        features.logical_qubits,
        key=lambda qubit: (features.weighted_degrees[qubit.x], -qubit.x),
        reverse=True,
    )
    return {
        logical: physical
        for logical, physical in zip(logical_order, physical_order)
    }


def annealed_layout(model: BackendNoiseModel, features: CircuitFeatures) -> Layout:
    current = greedy_noise_layout(model, features)
    best = dict(current)
    best_score = score_layout(best, model, features)
    logicals = list(features.logical_qubits)
    if len(logicals) < 2:
        return best

    for step in range(96):
        left = logicals[(17 * step + 3) % len(logicals)]
        right = logicals[(31 * step + 7) % len(logicals)]
        if left == right:
            continue
        trial = dict(current)
        trial[left], trial[right] = trial[right], trial[left]
        trial_score = score_layout(trial, model, features)
        temperature = max(0.0005, 0.02 * (1.0 - step / 96.0))
        deterministic_jitter = ((step * 1103515245 + 12345) % 997) / 997.0
        accept_uphill = trial_score < best_score + temperature * deterministic_jitter
        if trial_score <= score_layout(current, model, features) or accept_uphill:
            current = trial
        if trial_score < best_score:
            best = trial
            best_score = trial_score
    return best


def hub_aware_layout(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    model: BackendNoiseModel,
) -> Layout | None:
    graph = nx.Graph()
    edge_weights: Counter[tuple[int, int]] = Counter()
    for op in circuit.all_operations():
        if len(op.qubits) != 2:
            continue
        left, right = sorted((op.qubits[0].x, op.qubits[1].x))
        graph.add_edge(left, right)
        edge_weights[(left, right)] += 1
    if graph.number_of_nodes() == 0:
        return None

    max_degree = max((graph.degree(node) for node in graph.nodes), default=0)
    hub_threshold = max(4, int(max_degree * 0.6))
    hubs = sorted(node for node in graph.nodes if graph.degree(node) >= hub_threshold)
    node_count = graph.number_of_nodes()
    if len(hubs) < 3 or len(hubs) > node_count * 0.6 or len(hubs) < node_count * 0.15:
        return None

    leaves = [node for node in sorted(graph.nodes) if node not in hubs]
    hub_leaves = {hub: [node for node in graph.neighbors(hub) if node in leaves] for hub in hubs}
    hub_edges = [(left, right) for left, right in edge_weights if left in hubs and right in hubs]
    if not hub_edges and not any(hub_leaves.values()):
        return None

    initial = _initial_hub_mapping(hubs, hub_edges, hub_leaves, model)
    if initial is None:
        return None
    best = _refine_hub_mapping(initial, hubs, hub_edges, hub_leaves, model)
    return _assign_hub_leaves(best, graph, hubs, hub_leaves, model)


def small_simulation_layouts(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    model: BackendNoiseModel,
    features: CircuitFeatures,
    *,
    limit: int = 10,
) -> list[Layout]:
    logical_qubits = list(features.logical_qubits)
    width = len(logical_qubits)
    if width == 0 or width > 7:
        return []

    candidates: list[Layout] = []
    seen: set[tuple[int, ...]] = set()

    def add(layout: Layout) -> None:
        if not _is_valid_layout(layout, backend, features):
            return
        signature = tuple(layout[qubit] for qubit in logical_qubits)
        if signature in seen:
            return
        seen.add(signature)
        candidates.append(layout)

    for layout in [
        trivial_layout(circuit, backend),
        movement_aware_layout(circuit, backend),
        reliability_layout(circuit, backend),
        paper_greedy_vertex_layout(circuit, backend),
        paper_greedy_edge_layout(circuit, backend),
        greedy_noise_layout(model, features),
        low_noise_subgraph_layout(model, features),
        low_readout_layout(model, features),
    ]:
        add(layout)

    nodes = list(range(backend.num_qubits))
    rng = random.Random(42)
    max_random = 300 if backend.num_qubits <= 16 else 120
    for _ in range(max_random):
        shuffled = nodes[:]
        rng.shuffle(shuffled)
        add({logical: physical for logical, physical in zip(logical_qubits, shuffled[:width])})

    if backend.num_qubits <= 10:
        top_nodes = sorted(
            nodes,
            key=lambda node: (
                -len(model.neighbors[node]),
                model.readout_cost(node),
                node,
            ),
        )[: min(len(nodes), width + 2)]
        for index, perm in enumerate(permutations(top_nodes, width)):
            if index >= 500:
                break
            add({logical: physical for logical, physical in zip(logical_qubits, perm)})

    return sorted(
        candidates,
        key=lambda layout: (
            score_layout(layout, model, features),
            tuple(layout[qubit] for qubit in logical_qubits),
        ),
    )[:limit]


def _initial_hub_mapping(
    hubs: list[int],
    hub_edges: list[tuple[int, int]],
    hub_leaves: dict[int, list[int]],
    model: BackendNoiseModel,
) -> dict[int, int] | None:
    sorted_hubs = sorted(hubs, key=lambda hub: (len(hub_leaves[hub]), hub), reverse=True)
    best_mapping: dict[int, int] | None = None
    best_score = float("inf")
    centers = sorted(
        range(model.backend.num_qubits),
        key=lambda node: (-len(model.neighbors[node]), model.readout_cost(node), node),
    )[:12]
    for center in centers:
        cluster = sorted(
            range(model.backend.num_qubits),
            key=lambda node: (
                model.hop_distance(center, node),
                model.noise_distance(center, node),
                -len(model.neighbors[node]),
                model.readout_cost(node),
                node,
            ),
        )
        if len(cluster) < len(hubs):
            continue
        mapping = {hub: cluster[index] for index, hub in enumerate(sorted_hubs)}
        score = _hub_mapping_score(mapping, hub_edges, hub_leaves, model)
        if score < best_score:
            best_mapping = mapping
            best_score = score
    return best_mapping


def _refine_hub_mapping(
    initial: dict[int, int],
    hubs: list[int],
    hub_edges: list[tuple[int, int]],
    hub_leaves: dict[int, list[int]],
    model: BackendNoiseModel,
) -> dict[int, int]:
    current = dict(initial)
    best = dict(initial)
    current_score = _hub_mapping_score(current, hub_edges, hub_leaves, model)
    best_score = current_score
    rng = random.Random(42)
    temperatures = (2.0, 1.0, 0.5, 0.25)
    for temp in temperatures:
        for _ in range(35):
            left, right = rng.sample(hubs, 2)
            trial = dict(current)
            trial[left], trial[right] = trial[right], trial[left]
            score = _hub_mapping_score(trial, hub_edges, hub_leaves, model)
            delta = score - current_score
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                current = trial
                current_score = score
                if score < best_score:
                    best = trial
                    best_score = score
    return best


def _hub_mapping_score(
    mapping: dict[int, int],
    hub_edges: list[tuple[int, int]],
    hub_leaves: dict[int, list[int]],
    model: BackendNoiseModel,
) -> float:
    score = 0.0
    for left, right in hub_edges:
        score += model.noise_distance(mapping[left], mapping[right]) + 0.35 * model.hop_distance(mapping[left], mapping[right])
    used = set(mapping.values())
    available = [node for node in range(model.backend.num_qubits) if node not in used]
    for hub, leaves in hub_leaves.items():
        if not available:
            score += 100.0 * len(leaves)
            continue
        nearest = min(model.noise_distance(mapping[hub], node) + 0.25 * model.hop_distance(mapping[hub], node) for node in available)
        score += len(leaves) * nearest
    score += 0.05 * sum(model.readout_cost(physical) for physical in mapping.values())
    return score


def _assign_hub_leaves(
    hub_mapping: dict[int, int],
    graph: nx.Graph,
    hubs: list[int],
    hub_leaves: dict[int, list[int]],
    model: BackendNoiseModel,
) -> Layout:
    used = set(hub_mapping.values())
    assigned: dict[int, int] = dict(hub_mapping)
    for hub in sorted(hubs, key=lambda item: len(hub_leaves[item]), reverse=True):
        for leaf in sorted(hub_leaves[hub]):
            if leaf in assigned:
                continue
            available = [node for node in range(model.backend.num_qubits) if node not in used]
            if not available:
                break
            physical = min(
                available,
                key=lambda node: (
                    model.noise_distance(hub_mapping[hub], node),
                    model.hop_distance(hub_mapping[hub], node),
                    model.readout_cost(node),
                    -len(model.neighbors[node]),
                    node,
                ),
            )
            assigned[leaf] = physical
            used.add(physical)

    weighted_degrees = {
        node: sum(1 for _ in graph.neighbors(node))
        for node in graph.nodes
    }
    while len(assigned) < graph.number_of_nodes():
        unassigned = [node for node in sorted(graph.nodes) if node not in assigned]
        logical = max(
            unassigned,
            key=lambda node: (
                sum(1 for neighbor in graph.neighbors(node) if neighbor in assigned),
                weighted_degrees[node],
                -node,
            ),
        )
        available = [node for node in range(model.backend.num_qubits) if node not in used]
        physical = min(
            available,
            key=lambda node: (
                sum(
                    model.noise_distance(node, assigned[neighbor])
                    for neighbor in graph.neighbors(logical)
                    if neighbor in assigned
                ),
                model.readout_cost(node),
                node,
            ),
        )
        assigned[logical] = physical
        used.add(physical)
    return {cirq.LineQubit(logical): physical for logical, physical in assigned.items()}


def score_layout(layout: Layout, model: BackendNoiseModel, features: CircuitFeatures) -> float:
    interaction_cost = 0.0
    congestion: Counter[tuple[int, int]] = Counter()
    by_id = {logical.x: physical for logical, physical in layout.items()}
    for interaction in features.interactions:
        left, right = interaction.pair
        physical_left = by_id[left]
        physical_right = by_id[right]
        hop = model.hop_distance(physical_left, physical_right)
        noise = model.noise_distance(physical_left, physical_right)
        interaction_cost += interaction.weight * (noise + 0.35 * hop)
        path = model.noise_path(physical_left, physical_right)
        for a, b in zip(path, path[1:]):
            congestion[tuple(sorted((a, b)))] += 1

    readout_cost = sum(model.readout_cost(physical) for physical in by_id.values())
    congestion_cost = 0.02 * sum(count * count for count in congestion.values())
    future_cost = 0.0
    for interaction in features.interactions:
        left, right = interaction.pair
        hop = model.hop_distance(by_id[left], by_id[right])
        future_cost += interaction.weight * max(0, hop - 1) * 0.08

    return interaction_cost + 0.2 * readout_cost + congestion_cost + future_cost


def _local_subgraph_cost(center: int, width: int, model: BackendNoiseModel) -> float:
    closest = sorted(
        range(model.backend.num_qubits),
        key=lambda node: (
            model.noise_distance(center, node) + 0.35 * model.hop_distance(center, node),
            model.readout_cost(node),
            node,
        ),
    )[:width]
    edge_cost = 0.0
    for index, left in enumerate(closest):
        for right in closest[index + 1 :]:
            edge_cost += model.noise_distance(left, right)
    readout = sum(model.readout_cost(node) for node in closest)
    return edge_cost + 0.2 * readout


def _incremental_layout_cost(
    logical: int,
    physical: int,
    layout: Layout,
    model: BackendNoiseModel,
    features: CircuitFeatures,
) -> float:
    total = 0.15 * model.readout_cost(physical)
    by_id = {qubit.x: mapped for qubit, mapped in layout.items()}
    for interaction in features.interactions:
        left, right = interaction.pair
        if logical not in interaction.pair:
            continue
        other = right if left == logical else left
        if other not in by_id:
            continue
        total += interaction.weight * (
            model.noise_distance(physical, by_id[other])
            + 0.35 * model.hop_distance(physical, by_id[other])
        )
    if math.isclose(total, 0.0):
        total = model.readout_cost(physical) - 0.02 * len(model.neighbors[physical])
    return total


def _placed_neighbor_weight(logical: int, layout: Layout, features: CircuitFeatures) -> float:
    placed_ids = {qubit.x for qubit in layout}
    total = 0.0
    for interaction in features.interactions:
        if logical in interaction.pair:
            other = interaction.pair[1] if interaction.pair[0] == logical else interaction.pair[0]
            if other in placed_ids:
                total += interaction.weight
    return total


def _place(
    layout: Layout,
    unplaced: set[cirq.LineQubit],
    available: set[int],
    logical: cirq.LineQubit,
    physical: int,
) -> None:
    layout[logical] = physical
    unplaced.remove(logical)
    available.remove(physical)


def _is_valid_layout(layout: Layout, backend: CompetitionBackend, features: CircuitFeatures) -> bool:
    if set(layout) != set(features.logical_qubits):
        return False
    physical = list(layout.values())
    return len(set(physical)) == len(physical) and all(0 <= item < backend.num_qubits for item in physical)
