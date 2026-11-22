from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

import cirq
import networkx as nx

from noise_adaptive_hackathon.evaluator.validate_circuit import (
    count_two_qubit_gates,
    estimate_log_reliability,
    gate_name_for_operation,
)
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .selection import terminal_readout_cost
from .track_b import is_clifford_only


@dataclass(frozen=True)
class GHZStar:
    hub: cirq.LineQubit
    leaves: tuple[cirq.LineQubit, ...]
    logical_qubits: tuple[cirq.LineQubit, ...]


@dataclass(frozen=True)
class RootedTree:
    root: int
    parent: dict[int, int]
    children: dict[int, tuple[int, ...]]
    depth: dict[int, int]
    terminals: tuple[int, ...]


def compile_exact_ghz_tree_fanout(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
) -> tuple[cirq.Circuit, dict[cirq.LineQubit, int]] | None:
    ghz = detect_exact_ghz_star(circuit)
    if ghz is None:
        return None

    candidates: list[tuple[cirq.Circuit, dict[cirq.LineQubit, int]]] = []
    for tree in _candidate_ghz_trees(backend, ghz):
        compiled, layout = _compile_tree(circuit, backend, ghz, tree)
        if compiled is not None and _cnot_linear_equivalent(circuit, compiled, layout, ghz):
            candidates.append((compiled, layout))
    if not candidates:
        return None
    return max(candidates, key=lambda item: _candidate_key(item[0], backend))


def detect_exact_ghz_star(circuit: cirq.Circuit) -> GHZStar | None:
    if not is_clifford_only(circuit):
        return None
    logical_qubits = tuple(sorted(circuit.all_qubits(), key=lambda qubit: qubit.x))
    if len(logical_qubits) < 3:
        return None

    measurements = [op for op in circuit.all_operations() if isinstance(op.gate, cirq.MeasurementGate)]
    if len(measurements) != 1 or cirq.measurement_key_name(measurements[0]) != "z":
        return None
    if tuple(measurements[0].qubits) != logical_qubits:
        return None
    if list(circuit.all_operations())[-1] != measurements[0]:
        return None

    non_measure_ops = [
        op for op in circuit.all_operations() if not isinstance(op.gate, cirq.MeasurementGate)
    ]
    if len(non_measure_ops) != len(logical_qubits):
        return None
    h_ops = [op for op in non_measure_ops if gate_name_for_operation(op) == "h"]
    cx_ops = [op for op in non_measure_ops if gate_name_for_operation(op) == "cx"]
    if len(h_ops) != 1 or len(cx_ops) != len(logical_qubits) - 1:
        return None
    if non_measure_ops[0] != h_ops[0]:
        return None

    hub = h_ops[0].qubits[0]
    if not isinstance(hub, cirq.LineQubit):
        return None
    leaves: list[cirq.LineQubit] = []
    seen: set[cirq.LineQubit] = set()
    for op in cx_ops:
        if op.qubits[0] != hub or not isinstance(op.qubits[1], cirq.LineQubit):
            return None
        target = op.qubits[1]
        if target == hub or target in seen:
            return None
        seen.add(target)
        leaves.append(target)
    if set(leaves) != set(logical_qubits) - {hub}:
        return None
    return GHZStar(hub=hub, leaves=tuple(leaves), logical_qubits=logical_qubits)


def _candidate_ghz_trees(
    backend: CompetitionBackend,
    ghz: GHZStar,
) -> list[RootedTree]:
    width = len(ghz.logical_qubits)
    hardware = backend.graph()
    root_candidates = sorted(
        hardware.nodes,
        key=lambda node: (
            backend.readout(node),
            -_incident_reliability(hardware, node),
            -hardware.degree(node),
            node,
        ),
    )[:12]

    trees: list[RootedTree] = []
    seen: set[tuple[int, tuple[tuple[int, int], ...], tuple[int, ...]]] = set()
    for root in root_candidates:
        for mode in ("balanced", "reliability", "readout", "breadth"):
            tree = _grow_tree(hardware, root, width, mode)
            if tree is None:
                continue
            key = (tree.root, tuple(sorted(tree.parent.items())), tree.terminals)
            if key in seen:
                continue
            seen.add(key)
            trees.append(tree)
    return sorted(trees, key=lambda tree: _static_tree_key(tree, hardware, backend), reverse=True)[:12]


def _grow_tree(hardware: nx.Graph, root: int, width: int, mode: str) -> RootedTree | None:
    selected = {root}
    parent: dict[int, int] = {}
    depth = {root: 0}

    while len(selected) < width:
        frontier: list[tuple[tuple[float, ...], int, int]] = []
        for source in selected:
            for target in hardware.neighbors(source):
                if target in selected:
                    continue
                frontier.append((_growth_priority(hardware, source, target, depth[source], mode), source, target))
        if not frontier:
            return None
        _, source, target = min(frontier, key=lambda item: item[0])
        selected.add(target)
        parent[target] = source
        depth[target] = depth[source] + 1

    children_map: dict[int, list[int]] = defaultdict(list)
    for child, source in parent.items():
        children_map[source].append(child)
    children = {
        node: tuple(sorted(values, key=lambda child: _edge_priority(hardware, node, child)))
        for node, values in children_map.items()
    }
    terminals = tuple(
        sorted(
            selected,
            key=lambda node: (-hardware.nodes[node]["readout_reliability"], node),
        )
    )
    return RootedTree(root=root, parent=parent, children=children, depth=depth, terminals=terminals)


def _growth_priority(
    hardware: nx.Graph,
    source: int,
    target: int,
    source_depth: int,
    mode: str,
) -> tuple[float, ...]:
    edge_cost = hardware.edges[source, target]["cost"]
    readout_cost = -hardware.nodes[target]["readout_reliability"]
    degree_cost = -hardware.degree(target)
    if mode == "reliability":
        return (edge_cost, source_depth, readout_cost, degree_cost, target)
    if mode == "readout":
        return (readout_cost, edge_cost, source_depth, degree_cost, target)
    if mode == "breadth":
        return (source_depth, edge_cost, readout_cost, degree_cost, target)
    return (edge_cost + 0.15 * source_depth, readout_cost, degree_cost, target)


def _compile_tree(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    ghz: GHZStar,
    tree: RootedTree,
) -> tuple[cirq.Circuit | None, dict[cirq.LineQubit, int]]:
    del circuit
    layout: dict[cirq.LineQubit, int] = {ghz.hub: tree.root}
    leaf_nodes = [node for node in tree.terminals if node != tree.root]
    if len(leaf_nodes) < len(ghz.leaves):
        return None, {}
    for logical, physical in zip(sorted(ghz.leaves, key=lambda qubit: qubit.x), leaf_nodes):
        layout[logical] = physical

    physical = cirq.LineQubit.range(backend.num_qubits)
    compiled = cirq.Circuit()
    compiled.append(cirq.H(physical[tree.root]))

    by_parent_depth: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for child, parent in tree.parent.items():
        if parent == tree.root:
            continue
        by_parent_depth[tree.depth[parent]].append((parent, child))

    for depth in sorted(by_parent_depth, reverse=True):
        compiled.append(
            [cirq.CNOT(physical[parent], physical[child]) for parent, child in _order_edges(by_parent_depth[depth], backend)]
        )

    root_edges = [(tree.root, child) for child in tree.children.get(tree.root, ())]
    for parent, child in _order_edges(root_edges, backend):
        compiled.append(cirq.CNOT(physical[parent], physical[child]))

    for depth in sorted(by_parent_depth):
        compiled.append(
            [cirq.CNOT(physical[parent], physical[child]) for parent, child in _order_edges(by_parent_depth[depth], backend)]
        )

    compiled.append(cirq.measure(*(physical[layout[logical]] for logical in ghz.logical_qubits), key="z"))
    return compiled, layout


def _cnot_linear_equivalent(
    source: cirq.Circuit,
    compiled: cirq.Circuit,
    layout: dict[cirq.LineQubit, int],
    ghz: GHZStar,
) -> bool:
    physical_to_logical = {physical: index for index, logical in enumerate(ghz.logical_qubits) for physical in [layout[logical]]}
    source_rows = _cnot_rows(source, {logical.x: index for index, logical in enumerate(ghz.logical_qubits)})
    compiled_rows = _physical_cnot_rows(compiled, physical_to_logical, len(ghz.logical_qubits))
    for index, logical in enumerate(ghz.logical_qubits):
        physical = layout[logical]
        if compiled_rows.get(physical) != source_rows[index]:
            return False
    return True


def _cnot_rows(circuit: cirq.Circuit, logical_positions: dict[int, int]) -> dict[int, tuple[int, ...]]:
    size = len(logical_positions)
    rows = {
        position: tuple(1 if index == position else 0 for index in range(size))
        for position in logical_positions.values()
    }
    for op in circuit.all_operations():
        if gate_name_for_operation(op) != "cx":
            continue
        control = logical_positions[op.qubits[0].x]
        target = logical_positions[op.qubits[1].x]
        rows[target] = _xor_rows(rows[target], rows[control])
    return rows


def _physical_cnot_rows(
    circuit: cirq.Circuit,
    physical_to_logical: dict[int, int],
    size: int,
) -> dict[int, tuple[int, ...]]:
    rows = {
        physical: tuple(1 if index == logical else 0 for index in range(size))
        for physical, logical in physical_to_logical.items()
    }
    for op in circuit.all_operations():
        if gate_name_for_operation(op) != "cx":
            continue
        control = op.qubits[0].x
        target = op.qubits[1].x
        if control not in rows or target not in rows:
            return {}
        rows[target] = _xor_rows(rows[target], rows[control])
    return rows


def _xor_rows(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a ^ b for a, b in zip(left, right))


def _order_edges(edges: list[tuple[int, int]], backend: CompetitionBackend) -> list[tuple[int, int]]:
    return sorted(edges, key=lambda edge: (backend.edge_error(edge[0], edge[1]), edge[0], edge[1]))


def _static_tree_key(tree: RootedTree, hardware: nx.Graph, backend: CompetitionBackend) -> tuple[float, int, float, float]:
    log_reliability = 0.0
    for child, parent in tree.parent.items():
        repeats = 1 if parent == tree.root else 2
        log_reliability += repeats * math.log(max(1.0 - backend.edge_error(parent, child), 1e-12))
    for node in tree.terminals:
        log_reliability += math.log(max(1.0 - backend.readout(node), 1e-12))
    root_children = len(tree.children.get(tree.root, ()))
    cnot_count = root_children + 2 * (len(tree.parent) - root_children)
    return (log_reliability, -cnot_count, _incident_reliability(hardware, tree.root), -max(tree.depth.values()))


def _edge_priority(hardware: nx.Graph, source: int, target: int) -> tuple[float, int]:
    return (hardware.edges[source, target]["cost"], target)


def _incident_reliability(hardware: nx.Graph, node: int) -> float:
    return sum(hardware.edges[node, neighbor]["reliability"] for neighbor in hardware.neighbors(node))


def _candidate_key(circuit: cirq.Circuit, backend: CompetitionBackend) -> tuple[float, int, int, float]:
    return (
        estimate_log_reliability(circuit, backend),
        -count_two_qubit_gates(circuit),
        -len(circuit),
        -terminal_readout_cost(circuit, backend),
    )
