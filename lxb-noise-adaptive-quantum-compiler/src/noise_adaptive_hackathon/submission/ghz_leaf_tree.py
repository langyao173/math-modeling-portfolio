from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import (
    count_two_qubit_gates,
    estimate_log_reliability,
    gate_name_for_operation,
)
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .ghz_tree import GHZStar, detect_exact_ghz_star
from .selection import CompilationCandidate, candidate_metrics
from .validation import validate_or_raise


@dataclass(frozen=True)
class _LeafTree:
    root: int
    parent: dict[int, int]
    children: dict[int, tuple[int, ...]]
    leaves: tuple[int, ...]


def ghz_leaf_tree_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    *,
    limit: int = 4,
) -> list[CompilationCandidate]:
    """Use unmeasured Steiner leaves for final-Z GHZ measurement channels.

    Logical leaf qubits are placed only on physical leaves. Internal physical
    qubits are clean ancillas that carry the fanout but are not measured. This
    preserves the source CNOT linear measurement channel while often reducing
    the number of CX gates versus compute/uncompute tree constructions.
    """
    ghz = detect_exact_ghz_star(source)
    if ghz is None:
        return []
    accepted: list[CompilationCandidate] = []
    seen: set[tuple[str, ...]] = set()
    for tree in _candidate_leaf_trees(backend, len(ghz.logical_qubits)):
        compiled, layout = _compile_leaf_tree(backend, ghz, tree)
        signature = tuple(str(op) for op in compiled.all_operations())
        if signature in seen:
            continue
        seen.add(signature)
        if not _cnot_linear_equivalent(source, compiled, layout, ghz):
            continue
        try:
            validate_or_raise(source, compiled, backend)
        except Exception:
            continue
        candidate = CompilationCandidate("ghz_leaf_steiner_tree", compiled, layout, 0.0)
        if _leaf_tree_guard(candidate, source, backend):
            accepted.append(candidate)
            if len(accepted) >= limit:
                break
    return accepted


def _candidate_leaf_trees(backend: CompetitionBackend, width: int) -> list[_LeafTree]:
    trees: list[_LeafTree] = []
    seen: set[tuple[int, tuple[tuple[int, int], ...], tuple[int, ...]]] = set()
    roots = sorted(
        range(backend.num_qubits),
        key=lambda node: (
            backend.readout(node),
            -backend.graph().degree(node),
            node,
        ),
    )
    for root in roots:
        for mode in ("degree", "mixed", "edge", "readout", "depth"):
            tree = _grow_leaf_tree(root, backend, width, mode)
            if tree is None:
                continue
            key = (tree.root, tuple(sorted(tree.parent.items())), tree.leaves)
            if key in seen:
                continue
            seen.add(key)
            trees.append(tree)
    return sorted(trees, key=lambda tree: _tree_key(tree, backend), reverse=True)[:16]


def _grow_leaf_tree(
    root: int,
    backend: CompetitionBackend,
    width: int,
    mode: str,
    *,
    max_nodes: int = 32,
) -> _LeafTree | None:
    graph = backend.graph()
    selected = {root}
    parent: dict[int, int] = {}
    children: dict[int, list[int]] = defaultdict(list)
    leaves = {root}
    depth = {root: 0}

    while len(leaves - {root}) < width - 1 and len(selected) < max_nodes:
        frontier: list[tuple[tuple[float, ...], int, int]] = []
        for source in selected:
            for target in graph.neighbors(source):
                if target in selected:
                    continue
                frontier.append((_growth_key(source, target, leaves, depth, backend, mode), source, target))
        if not frontier:
            return None
        _, source, target = min(frontier, key=lambda item: item[0])
        selected.add(target)
        parent[target] = source
        children[source].append(target)
        depth[target] = depth[source] + 1
        if source in leaves:
            leaves.remove(source)
        leaves.add(target)

    logical_leaves = tuple(sorted(leaves - {root}))
    if len(logical_leaves) < width - 1:
        return None
    return _LeafTree(
        root=root,
        parent=parent,
        children={node: tuple(sorted(values)) for node, values in children.items()},
        leaves=logical_leaves,
    )


def _growth_key(
    source: int,
    target: int,
    leaves: set[int],
    depth: dict[int, int],
    backend: CompetitionBackend,
    mode: str,
) -> tuple[float, ...]:
    graph = backend.graph()
    leaf_gain = 1 if source not in leaves else 0
    edge = -math.log(max(1.0 - backend.edge_error(source, target), 1e-12))
    readout = -math.log(max(1.0 - backend.readout(target), 1e-12))
    degree = graph.degree(target)
    source_depth = depth[source]
    if mode == "degree":
        return (-leaf_gain, -degree, edge, readout, source_depth, source, target)
    if mode == "edge":
        return (-leaf_gain, edge, readout, source_depth, -degree, source, target)
    if mode == "readout":
        return (-leaf_gain, readout, edge, source_depth, -degree, source, target)
    if mode == "depth":
        return (-leaf_gain, source_depth, edge, readout, -degree, source, target)
    return (-leaf_gain, edge + 0.35 * readout + 0.01 * source_depth - 0.003 * degree, source, target)


def _compile_leaf_tree(
    backend: CompetitionBackend,
    ghz: GHZStar,
    tree: _LeafTree,
) -> tuple[cirq.Circuit, dict[cirq.LineQubit, int]]:
    physical = cirq.LineQubit.range(backend.num_qubits)
    logical_qubits = tuple(sorted(ghz.logical_qubits, key=lambda qubit: qubit.x))
    layout: dict[cirq.LineQubit, int] = {ghz.hub: tree.root}
    leaves = sorted(tree.leaves, key=lambda node: (-backend.graph().degree(node), backend.readout(node), node))
    for logical, physical_leaf in zip(sorted(ghz.leaves, key=lambda qubit: qubit.x), leaves):
        layout[logical] = physical_leaf

    depth = _depths(tree)
    circuit = cirq.Circuit(cirq.H(physical[tree.root]))
    for level in sorted(set(depth.values())):
        operations = [
            cirq.CNOT(physical[parent], physical[child])
            for child, parent in sorted(tree.parent.items())
            if depth[parent] == level
        ]
        if operations:
            circuit.append(operations)
    circuit.append(cirq.measure(*(physical[layout[logical]] for logical in logical_qubits), key="z"))
    return circuit, layout


def _depths(tree: _LeafTree) -> dict[int, int]:
    depth = {tree.root: 0}
    queue: deque[int] = deque([tree.root])
    while queue:
        node = queue.popleft()
        for child in tree.children.get(node, ()):
            depth[child] = depth[node] + 1
            queue.append(child)
    return depth


def _cnot_linear_equivalent(
    source: cirq.Circuit,
    compiled: cirq.Circuit,
    layout: dict[cirq.LineQubit, int],
    ghz: GHZStar,
) -> bool:
    logical_qubits = tuple(sorted(ghz.logical_qubits, key=lambda qubit: qubit.x))
    logical_position = {logical.x: index for index, logical in enumerate(logical_qubits)}
    source_rows = _logical_rows(source, logical_position)
    physical_to_logical = {physical: logical_position[logical.x] for logical, physical in layout.items()}
    compiled_rows = _physical_rows(compiled, physical_to_logical, len(logical_qubits))
    for logical_index, logical in enumerate(logical_qubits):
        physical = layout[logical]
        if compiled_rows.get(physical) != source_rows[logical_index]:
            return False
    return True


def _logical_rows(circuit: cirq.Circuit, logical_position: dict[int, int]) -> dict[int, tuple[int, ...]]:
    width = len(logical_position)
    rows = {
        position: tuple(1 if index == position else 0 for index in range(width))
        for position in logical_position.values()
    }
    for op in circuit.all_operations():
        if gate_name_for_operation(op) != "cx":
            continue
        control = logical_position[op.qubits[0].x]
        target = logical_position[op.qubits[1].x]
        rows[target] = tuple(left ^ right for left, right in zip(rows[target], rows[control]))
    return rows


def _physical_rows(
    circuit: cirq.Circuit,
    physical_to_logical: dict[int, int],
    width: int,
) -> dict[int, tuple[int, ...]]:
    rows: dict[int, tuple[int, ...]] = {}
    for qubit in circuit.all_qubits():
        if not isinstance(qubit, cirq.LineQubit):
            return {}
        logical = physical_to_logical.get(qubit.x)
        if logical is None:
            rows[qubit.x] = tuple(0 for _ in range(width))
        else:
            rows[qubit.x] = tuple(1 if index == logical else 0 for index in range(width))
    for op in circuit.all_operations():
        if gate_name_for_operation(op) != "cx":
            continue
        control, target = op.qubits[0].x, op.qubits[1].x
        rows[target] = tuple(left ^ right for left, right in zip(rows[target], rows[control]))
    return rows


def _leaf_tree_guard(candidate: CompilationCandidate, source: cirq.Circuit, backend: CompetitionBackend) -> bool:
    metrics = candidate_metrics(candidate, backend, source)
    source_cx = count_two_qubit_gates(source)
    return (
        metrics.two_qubit_count <= max(source_cx + 6, 24)
        and metrics.depth <= 16
        and metrics.log_reliability >= -0.58
        and metrics.readout_cost <= 0.25
        and estimate_log_reliability(candidate.circuit, backend) >= -0.58
    )


def _tree_key(tree: _LeafTree, backend: CompetitionBackend) -> tuple[float, int, int, float]:
    circuit, layout = _compile_leaf_tree(
        backend,
        GHZStar(
            hub=cirq.LineQubit(0),
            leaves=tuple(cirq.LineQubit.range(1, len(tree.leaves) + 1)),
            logical_qubits=tuple(cirq.LineQubit.range(len(tree.leaves) + 1)),
        ),
        tree,
    )
    del layout
    return (
        estimate_log_reliability(circuit, backend),
        -count_two_qubit_gates(circuit),
        -len(circuit),
        -sum(backend.readout(node) for node in tree.leaves),
    )
