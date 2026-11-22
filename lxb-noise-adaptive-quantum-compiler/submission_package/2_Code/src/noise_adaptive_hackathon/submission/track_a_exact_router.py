from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq
import itertools
import math
import random
import time

import cirq
import networkx as nx

from noise_adaptive_hackathon.baseline.common import _decomposed_swap
from noise_adaptive_hackathon.evaluator.validate_circuit import count_two_qubit_gates, estimate_log_reliability, gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .backend_model import BackendNoiseModel
from .circuit_features import CircuitFeatures, extract_circuit_features, future_two_qubit_pairs
from .layout import candidate_initial_layouts
from .peephole import conservative_peephole
from .selection import CompilationCandidate, candidate_metrics
from .track_b import is_clifford_only
from .validation import validate_or_raise


@dataclass(frozen=True)
class _SearchNode:
    index: int
    mapping: tuple[int, ...]


@dataclass(frozen=True)
class _Step:
    previous: _SearchNode | None
    operations: tuple[cirq.Operation, ...]


def exact_small_router_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    incumbent: CompilationCandidate | None,
    *,
    seed: int = 7,
    max_candidates: int = 8,
    max_seconds: float = 8.0,
    incumbent_only: bool = False,
    min_two_qubit_count: int = 12,
) -> list[CompilationCandidate]:
    logical_qubits = tuple(sorted(source.all_qubits(), key=lambda qubit: qubit.x))
    source_two_qubit_count = count_two_qubit_gates(source)
    if (
        is_clifford_only(source)
        or not (6 <= len(logical_qubits) <= 7)
        or source_two_qubit_count < min_two_qubit_count
    ):
        return []
    if incumbent is not None and candidate_metrics(incumbent, backend, source).route_overhead <= 4:
        return []
    started = time.perf_counter()
    deadline = started + max_seconds
    model = BackendNoiseModel.from_backend(backend)
    features = extract_circuit_features(source)
    candidates: list[CompilationCandidate] = []
    seen: set[tuple[str, ...]] = set()

    jobs = _search_jobs(source, backend, model, features, incumbent, seed=seed, incumbent_only=incumbent_only)
    for region_index, (region, layouts) in enumerate(jobs):
        if time.perf_counter() >= deadline:
            break
        for layout_index, layout in enumerate(layouts):
            if time.perf_counter() >= deadline or len(candidates) >= max_candidates:
                break
            remaining = max(0.2, deadline - time.perf_counter())
            result = _beam_route(
                source,
                backend,
                model,
                tuple(sorted(region)),
                layout,
                max_seconds=min(2.0 if region_index < 2 else 1.0, remaining),
            )
            if result is None:
                continue
            compiled, routed_layout = result
            compiled = conservative_peephole(compiled, clifford_only=False)
            try:
                validate_or_raise(source, compiled, backend)
            except Exception:
                continue
            signature = tuple(str(op) for op in compiled.all_operations())
            if signature in seen:
                continue
            seen.add(signature)
            candidate = CompilationCandidate(
                f"exact_small_router_r{region_index}_l{layout_index}",
                compiled,
                routed_layout,
                time.perf_counter() - started,
            )
            if incumbent is not None and not _promising(candidate, incumbent, backend, source):
                continue
            candidates.append(candidate)
    return candidates


def _search_jobs(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    model: BackendNoiseModel,
    features: CircuitFeatures,
    incumbent: CompilationCandidate | None,
    *,
    seed: int,
    incumbent_only: bool = False,
) -> list[tuple[tuple[int, ...], list[dict[cirq.LineQubit, int]]]]:
    jobs: list[tuple[tuple[int, ...], list[dict[cirq.LineQubit, int]]]] = []
    seen: set[tuple[int, ...]] = set()

    def add(region: tuple[int, ...], layouts: list[dict[cirq.LineQubit, int]]) -> None:
        signature = tuple(sorted(region))
        if signature in seen or not layouts:
            return
        seen.add(signature)
        jobs.append((signature, layouts))

    if incumbent is not None and incumbent.layout:
        used = set(incumbent.layout.values())
        neighborhood = set(used)
        for physical in tuple(used):
            neighborhood.update(model.graph.neighbors(physical))
        add(tuple(sorted(neighborhood)), [dict(incumbent.layout)])
        add(tuple(range(backend.num_qubits)), [dict(incumbent.layout)])
        if incumbent_only:
            return jobs

    width = len(features.logical_qubits)
    for region_index, region in enumerate(_ranked_regions(backend, model, width, limit=18)):
        layouts = _region_layouts(source, backend, model, features, region, seed=seed + region_index, limit=4)
        add(region, layouts)
    return jobs


def _ranked_regions(
    backend: CompetitionBackend,
    model: BackendNoiseModel,
    width: int,
    *,
    limit: int,
) -> list[tuple[int, ...]]:
    graph = backend.graph()
    sizes = tuple(sorted({min(backend.num_qubits, width + 1), min(backend.num_qubits, width + 2), min(backend.num_qubits, width + 3)}))
    regions: list[tuple[float, tuple[int, ...]]] = []
    for size in sizes:
        for nodes in itertools.combinations(range(backend.num_qubits), size):
            subgraph = graph.subgraph(nodes)
            if not nx.is_connected(subgraph):
                continue
            edge_costs = [model.edge_cost(left, right) for left, right in subgraph.edges]
            if not edge_costs:
                continue
            diameter = nx.diameter(subgraph)
            readout = sum(model.readout_cost(node) for node in nodes) / len(nodes)
            mean_edge = sum(edge_costs) / len(edge_costs)
            connectivity = nx.edge_connectivity(subgraph) if subgraph.number_of_nodes() > 1 else 0
            score = mean_edge + 0.015 * diameter + 0.15 * readout - 0.003 * connectivity
            regions.append((score, tuple(nodes)))
    ranked: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for _, region in sorted(regions, key=lambda item: (item[0], item[1])):
        if region in seen:
            continue
        seen.add(region)
        ranked.append(region)
        if len(ranked) >= limit:
            break
    return ranked


def _region_layouts(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    model: BackendNoiseModel,
    features: CircuitFeatures,
    region: tuple[int, ...],
    *,
    seed: int,
    limit: int,
) -> list[dict[cirq.LineQubit, int]]:
    region_set = set(region)
    logical_qubits = tuple(features.logical_qubits)
    layouts: list[dict[cirq.LineQubit, int]] = []
    seen: set[tuple[int, ...]] = set()

    def add(layout: dict[cirq.LineQubit, int]) -> None:
        if set(layout) != set(logical_qubits) or any(physical not in region_set for physical in layout.values()):
            return
        if len(set(layout.values())) != len(layout):
            return
        signature = tuple(layout[logical] for logical in logical_qubits)
        if signature in seen:
            return
        seen.add(signature)
        layouts.append(dict(layout))

    for layout in candidate_initial_layouts(source, backend, model, features, limit=16):
        add(layout)

    logical_order = sorted(logical_qubits, key=lambda qubit: (features.weighted_degrees[qubit.x], -qubit.x), reverse=True)
    subgraph = backend.graph().subgraph(region)
    physical_order = sorted(
        region,
        key=lambda node: (
            -subgraph.degree(node),
            model.readout_cost(node),
            sum(model.edge_cost(node, neighbor) for neighbor in subgraph.neighbors(node)),
            node,
        ),
    )
    add({logical: physical for logical, physical in zip(logical_order, physical_order)})
    add({logical: physical for logical, physical in zip(logical_order, reversed(physical_order))})

    rng = random.Random(seed)
    for _ in range(12):
        shuffled = list(region)
        rng.shuffle(shuffled)
        add({logical: physical for logical, physical in zip(logical_order, shuffled)})
        if len(layouts) >= limit:
            break
    return layouts[:limit]


def _beam_route(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    model: BackendNoiseModel,
    region: tuple[int, ...],
    layout: dict[cirq.LineQubit, int],
    *,
    max_seconds: float,
    max_expansions: int = 9000,
    beam_per_index: int = 220,
) -> tuple[cirq.Circuit, dict[cirq.LineQubit, int]] | None:
    del backend
    deadline = time.perf_counter() + max_seconds
    logical_qubits = tuple(sorted(source.all_qubits(), key=lambda qubit: qubit.x))
    logical_index = {qubit.x: index for index, qubit in enumerate(logical_qubits)}
    operations = tuple(source.all_operations())
    physical_qubits = tuple(cirq.LineQubit.range(model.backend.num_qubits))
    initial_mapping = tuple(layout[logical] for logical in logical_qubits)
    start = _SearchNode(0, initial_mapping)
    costs: dict[_SearchNode, float] = {start: 0.0}
    counts: dict[_SearchNode, int] = {start: 0}
    steps: dict[_SearchNode, _Step] = {start: _Step(None, ())}
    by_index: dict[int, list[float]] = defaultdict(list)
    queue: list[tuple[float, float, int, _SearchNode]] = []
    sequence = 0
    heapq.heappush(queue, (_heuristic(start, operations, model, logical_index), 0.0, sequence, start))
    expansions = 0
    region_edges = tuple(sorted(edge for edge in model.coupling_edges if edge[0] in region and edge[1] in region))
    subgraph = model.graph.subgraph(region).copy()
    min_edge_cost = min((model.edge_cost(*edge) for edge in region_edges), default=0.0)

    while queue and expansions < max_expansions and time.perf_counter() < deadline:
        _, cost, _, node = heapq.heappop(queue)
        if cost > costs.get(node, float("inf")) + 1e-12:
            continue
        if node.index >= len(operations):
            return _reconstruct(node, steps, physical_qubits, logical_qubits, source)
        expansions += 1
        for next_node, action_ops, action_cost, action_count in _transitions(
            node,
            operations,
            model,
            subgraph,
            region_edges,
            physical_qubits,
            logical_index,
            min_edge_cost,
        ):
            next_cost = cost + action_cost
            previous = costs.get(next_node)
            if previous is not None and next_cost >= previous - 1e-12:
                continue
            bucket = by_index[next_node.index]
            if len(bucket) >= beam_per_index and next_cost > max(bucket):
                continue
            costs[next_node] = next_cost
            counts[next_node] = counts[node] + action_count
            steps[next_node] = _Step(node, action_ops)
            bucket.append(next_cost)
            if len(bucket) > beam_per_index:
                bucket.remove(max(bucket))
            sequence += 1
            priority = next_cost + _heuristic(next_node, operations, model, logical_index)
            heapq.heappush(queue, (priority, next_cost, sequence, next_node))
    return None


def _transitions(
    node: _SearchNode,
    operations: tuple[cirq.Operation, ...],
    model: BackendNoiseModel,
    subgraph: nx.Graph,
    region_edges: tuple[tuple[int, int], ...],
    physical_qubits: tuple[cirq.LineQubit, ...],
    logical_index: dict[int, int],
    min_edge_cost: float,
) -> list[tuple[_SearchNode, tuple[cirq.Operation, ...], float, int]]:
    op = operations[node.index]
    if isinstance(op.gate, cirq.MeasurementGate):
        return [(_SearchNode(node.index + 1, node.mapping), (), 0.0, 0)]
    if len(op.qubits) == 1:
        physical = node.mapping[logical_index[op.qubits[0].x]]
        return [
            (
                _SearchNode(node.index + 1, node.mapping),
                (op.transform_qubits(lambda _: physical_qubits[physical]),),
                0.00005,
                0,
            )
        ]
    if len(op.qubits) != 2:
        return []

    left_logical = logical_index[op.qubits[0].x]
    right_logical = logical_index[op.qubits[1].x]
    left_physical = node.mapping[left_logical]
    right_physical = node.mapping[right_logical]
    transitions: list[tuple[_SearchNode, tuple[cirq.Operation, ...], float, int]] = []

    if model.is_coupled(left_physical, right_physical):
        transformed = op.transform_qubits(lambda qubit: physical_qubits[node.mapping[logical_index[qubit.x]]])
        transitions.append(
            (
                _SearchNode(node.index + 1, node.mapping),
                (transformed,),
                model.edge_cost(left_physical, right_physical) + 0.0008,
                1,
            )
        )
        if gate_name_for_operation(op) == "cx":
            swapped = _swap_mapping(node.mapping, left_physical, right_physical)
            transitions.append(
                (
                    _SearchNode(node.index + 1, swapped),
                    (
                        cirq.CNOT(physical_qubits[right_physical], physical_qubits[left_physical]),
                        cirq.CNOT(physical_qubits[left_physical], physical_qubits[right_physical]),
                    ),
                    2.0 * model.edge_cost(left_physical, right_physical) + 0.0018,
                    2,
                )
            )
        return transitions

    if gate_name_for_operation(op) == "cx" and _has_path(subgraph, left_physical, right_physical):
        path = nx.shortest_path(subgraph, left_physical, right_physical, weight="cost")
        if len(path) == 3:
            middle = path[1]
            bridge_ops = (
                cirq.CNOT(physical_qubits[left_physical], physical_qubits[middle]),
                cirq.CNOT(physical_qubits[middle], physical_qubits[right_physical]),
                cirq.CNOT(physical_qubits[left_physical], physical_qubits[middle]),
                cirq.CNOT(physical_qubits[middle], physical_qubits[right_physical]),
            )
            bridge_cost = 2.0 * model.edge_cost(left_physical, middle) + 2.0 * model.edge_cost(middle, right_physical) + 0.0035
            transitions.append((_SearchNode(node.index + 1, node.mapping), bridge_ops, bridge_cost, 4))

    for edge in _candidate_swap_edges(node, operations, model, subgraph, region_edges, logical_index):
        swapped = _swap_mapping(node.mapping, edge[0], edge[1])
        transitions.append(
            (
                _SearchNode(node.index, swapped),
                tuple(_decomposed_swap(physical_qubits[edge[0]], physical_qubits[edge[1]])),
                3.0 * model.edge_cost(*edge) + 0.0025 + 0.0002 * min_edge_cost,
                3,
            )
        )
    return transitions


def _candidate_swap_edges(
    node: _SearchNode,
    operations: tuple[cirq.Operation, ...],
    model: BackendNoiseModel,
    subgraph: nx.Graph,
    region_edges: tuple[tuple[int, int], ...],
    logical_index: dict[int, int],
) -> tuple[tuple[int, int], ...]:
    op = operations[node.index]
    active = {node.mapping[logical_index[qubit.x]] for qubit in op.qubits}
    candidates: set[tuple[int, int]] = set()
    for physical in active:
        for neighbor in subgraph.neighbors(physical):
            candidates.add(tuple(sorted((physical, neighbor))))
    future = future_two_qubit_pairs(operations, node.index, limit=6)
    for left, right in future[:4]:
        if left not in logical_index or right not in logical_index:
            continue
        source = node.mapping[logical_index[left]]
        target = node.mapping[logical_index[right]]
        if not _has_path(subgraph, source, target):
            continue
        for path in (
            nx.shortest_path(subgraph, source, target),
            nx.shortest_path(subgraph, source, target, weight="cost"),
        ):
            for edge in zip(path, path[1:]):
                candidates.add(tuple(sorted(edge)))
    if not candidates:
        candidates.update(region_edges)
    return tuple(sorted(candidates))


def _swap_mapping(mapping: tuple[int, ...], left: int, right: int) -> tuple[int, ...]:
    updated = list(mapping)
    for index, physical in enumerate(updated):
        if physical == left:
            updated[index] = right
        elif physical == right:
            updated[index] = left
    return tuple(updated)


def _heuristic(
    node: _SearchNode,
    operations: tuple[cirq.Operation, ...],
    model: BackendNoiseModel,
    logical_index: dict[int, int],
) -> float:
    del node, operations, model, logical_index
    return 0.0


def _reconstruct(
    node: _SearchNode,
    steps: dict[_SearchNode, _Step],
    physical_qubits: tuple[cirq.LineQubit, ...],
    logical_qubits: tuple[cirq.LineQubit, ...],
    source: cirq.Circuit,
) -> tuple[cirq.Circuit, dict[cirq.LineQubit, int]]:
    chunks: list[tuple[cirq.Operation, ...]] = []
    cursor = node
    while True:
        step = steps[cursor]
        chunks.append(step.operations)
        if step.previous is None:
            break
        cursor = step.previous
    operations: list[cirq.Operation] = []
    for chunk in reversed(chunks):
        operations.extend(chunk)
    final_mapping = node.mapping
    measurements = [
        qubit
        for op in source.all_operations()
        if isinstance(op.gate, cirq.MeasurementGate)
        for qubit in op.qubits
    ]
    measured = measurements or list(logical_qubits)
    logical_index = {logical.x: index for index, logical in enumerate(logical_qubits)}
    operations.append(cirq.measure(*(physical_qubits[final_mapping[logical_index[qubit.x]]] for qubit in measured), key="z"))
    initial = cursor.mapping
    layout = {logical: initial[index] for index, logical in enumerate(logical_qubits)}
    return cirq.Circuit(operations), layout


def _has_path(graph: nx.Graph, left: int, right: int) -> bool:
    return left in graph and right in graph and nx.has_path(graph, left, right)


def _promising(
    candidate: CompilationCandidate,
    incumbent: CompilationCandidate,
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> bool:
    candidate_value = candidate_metrics(candidate, backend, source)
    incumbent_value = candidate_metrics(incumbent, backend, source)
    return (
        candidate_value.two_qubit_count <= incumbent_value.two_qubit_count - 1
        or (
            candidate_value.log_reliability >= incumbent_value.log_reliability + 0.025
            and candidate_value.two_qubit_count <= incumbent_value.two_qubit_count + 1
        )
        or (
            candidate_value.two_qubit_count <= incumbent_value.two_qubit_count
            and candidate_value.depth <= incumbent_value.depth
            and estimate_log_reliability(candidate.circuit, backend) >= estimate_log_reliability(incumbent.circuit, backend) - 0.005
            and count_two_qubit_gates(candidate.circuit) <= count_two_qubit_gates(incumbent.circuit)
        )
    )
