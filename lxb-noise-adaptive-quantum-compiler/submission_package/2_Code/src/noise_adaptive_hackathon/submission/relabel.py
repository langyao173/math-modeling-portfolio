from __future__ import annotations

from collections import Counter
from collections import deque
import time

import cirq
import networkx as nx
from networkx.algorithms import isomorphism

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .backend_model import BackendNoiseModel


def optimize_physical_relabeling(
    circuit: cirq.Circuit,
    layout: dict[cirq.LineQubit, int],
    backend: CompetitionBackend,
    *,
    max_mappings: int = 20_000,
    max_seconds: float = 1.5,
    readout_weight: float = 0.35,
) -> tuple[cirq.Circuit, dict[cirq.LineQubit, int]]:
    model = BackendNoiseModel.from_backend(backend)
    non_identity_ops = [op for op in circuit.all_operations() if not isinstance(op.gate, cirq.IdentityGate)]
    edge_counts, measured = _usage(non_identity_ops)
    used_nodes = set(measured)
    for left, right in edge_counts:
        used_nodes.add(left)
        used_nodes.add(right)
    for op in non_identity_ops:
        for qubit in op.qubits:
            if isinstance(qubit, cirq.LineQubit):
                used_nodes.add(qubit.x)

    if len(used_nodes) <= 1 or len(used_nodes) > backend.num_qubits or not edge_counts:
        return cirq.Circuit(non_identity_ops), dict(layout)

    pattern = nx.Graph()
    pattern.add_nodes_from(used_nodes)
    pattern.add_edges_from(edge_counts)

    current = {node: node for node in used_nodes}
    best_mapping = current
    best_score = _mapping_score(current, edge_counts, measured, model, readout_weight=readout_weight)
    tree_mapping = _optimize_tree_mapping(pattern, edge_counts, measured, model, readout_weight=readout_weight)
    if tree_mapping is not None:
        tree_score = _mapping_score(tree_mapping, edge_counts, measured, model, readout_weight=readout_weight)
        if tree_score < best_score:
            return _transform_circuit(non_identity_ops, tree_mapping), {
                logical: tree_mapping.get(physical, physical)
                for logical, physical in layout.items()
            }
        return cirq.Circuit(non_identity_ops), dict(layout)

    graph_matcher = isomorphism.GraphMatcher(model.graph, pattern)
    started = time.perf_counter()
    seen = 0
    for host_to_pattern in graph_matcher.subgraph_monomorphisms_iter():
        seen += 1
        pattern_to_host = {pattern_node: host_node for host_node, pattern_node in host_to_pattern.items()}
        score = _mapping_score(pattern_to_host, edge_counts, measured, model, readout_weight=readout_weight)
        if score < best_score:
            best_score = score
            best_mapping = pattern_to_host
        if seen >= max_mappings or time.perf_counter() - started >= max_seconds:
            break

    if best_mapping == current:
        return cirq.Circuit(non_identity_ops), dict(layout)

    return _transform_circuit(non_identity_ops, best_mapping), {
        logical: best_mapping.get(physical, physical)
        for logical, physical in layout.items()
    }


def _usage(operations: list[cirq.Operation]) -> tuple[Counter[tuple[int, int]], set[int]]:
    edge_counts: Counter[tuple[int, int]] = Counter()
    measured: set[int] = set()
    for op in operations:
        gate_name = gate_name_for_operation(op)
        if gate_name == "measure":
            measured.update(qubit.x for qubit in op.qubits if isinstance(qubit, cirq.LineQubit))
        elif gate_name == "cx" and len(op.qubits) == 2:
            edge_counts[tuple(sorted(qubit.x for qubit in op.qubits))] += 1
    return edge_counts, measured


def _mapping_score(
    mapping: dict[int, int],
    edge_counts: Counter[tuple[int, int]],
    measured: set[int],
    model: BackendNoiseModel,
    *,
    readout_weight: float,
) -> float:
    edge_cost = sum(
        count * model.edge_cost(mapping[left], mapping[right])
        for (left, right), count in edge_counts.items()
    )
    readout_cost = sum(model.readout_cost(mapping[qubit]) for qubit in measured)
    return edge_cost + readout_weight * readout_cost


def _optimize_tree_mapping(
    pattern: nx.Graph,
    edge_counts: Counter[tuple[int, int]],
    measured: set[int],
    model: BackendNoiseModel,
    *,
    readout_weight: float,
    max_states: int = 2_000,
    max_root_hosts: int = 64,
) -> dict[int, int] | None:
    if not nx.is_tree(pattern):
        return None
    if not pattern.nodes:
        return None

    root = max(pattern.nodes, key=lambda node: (pattern.degree(node), node in measured, -node))
    order, parent = _tree_order(pattern, root, measured)

    states: list[tuple[float, dict[int, int], frozenset[int]]] = []
    root_hosts = sorted(
        model.graph.nodes,
        key=lambda host: (
            model.graph.degree(host) < pattern.degree(root),
            -model.graph.degree(host),
            model.readout_cost(host) if root in measured else 0.0,
            host,
        ),
    )[:max_root_hosts]
    for host in root_hosts:
        if model.graph.degree(host) < pattern.degree(root):
            continue
        score = readout_weight * model.readout_cost(host) if root in measured else 0.0
        states.append((score, {root: host}, frozenset((host,))))
    if not states:
        return None

    for pattern_node in order[1:]:
        pattern_parent = parent[pattern_node]
        assert pattern_parent is not None
        next_states: list[tuple[float, dict[int, int], frozenset[int]]] = []
        for score, mapping, used_hosts in states:
            host_parent = mapping[pattern_parent]
            for host in model.neighbors[host_parent]:
                if host in used_hosts:
                    continue
                if model.graph.degree(host) < pattern.degree(pattern_node):
                    continue
                added = edge_counts[tuple(sorted((pattern_node, pattern_parent)))] * model.edge_cost(host, host_parent)
                if pattern_node in measured:
                    added += readout_weight * model.readout_cost(host)
                next_mapping = dict(mapping)
                next_mapping[pattern_node] = host
                next_states.append((score + added, next_mapping, used_hosts | frozenset((host,))))
        if not next_states:
            return None
        states = sorted(next_states, key=lambda item: item[0])[:max_states]

    return min(states, key=lambda item: item[0])[1]


def _tree_order(pattern: nx.Graph, root: int, measured: set[int]) -> tuple[list[int], dict[int, int | None]]:
    parent: dict[int, int | None] = {root: None}
    order = [root]
    queue: deque[int] = deque([root])
    while queue:
        node = queue.popleft()
        children = sorted(
            (neighbor for neighbor in pattern.neighbors(node) if neighbor not in parent),
            key=lambda neighbor: (-pattern.degree(neighbor), neighbor not in measured, neighbor),
        )
        for child in children:
            parent[child] = node
            order.append(child)
            queue.append(child)
    return order, parent


def _transform_circuit(operations: list[cirq.Operation], mapping: dict[int, int]) -> cirq.Circuit:
    def transform(qubit: cirq.Qid) -> cirq.Qid:
        if isinstance(qubit, cirq.LineQubit):
            return cirq.LineQubit(mapping.get(qubit.x, qubit.x))
        return qubit

    return cirq.Circuit(op.transform_qubits(transform) for op in operations)
