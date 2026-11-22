from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cirq

from noise_adaptive_hackathon.baseline.common import _decomposed_swap
from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .backend_model import BackendNoiseModel
from .circuit_features import CircuitFeatures, LogicalPair, future_two_qubit_pairs


@dataclass(frozen=True)
class RoutingResult:
    circuit: cirq.Circuit
    final_mapping: dict[int, int]


def route_circuit(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    model: BackendNoiseModel,
    features: CircuitFeatures,
    initial_layout: dict[cirq.LineQubit, int],
    *,
    enable_execute_and_swap: bool = True,
    enable_bridge: bool = True,
    terminal_remap: bool = True,
    routing_objective: str = "balanced",
) -> RoutingResult:
    physical_qubits = tuple(cirq.LineQubit.range(backend.num_qubits))
    operations = list(circuit.all_operations())
    mapping = {logical.x: physical for logical, physical in initial_layout.items()}
    reverse: dict[int, int | None] = {physical: None for physical in range(backend.num_qubits)}
    for logical, physical in mapping.items():
        reverse[physical] = logical

    routed = cirq.Circuit()
    measurements: list[cirq.Qid] = []
    initial_measurements = [
        qubit
        for op in operations
        if isinstance(op.gate, cirq.MeasurementGate)
        for qubit in op.qubits
    ]
    measurement_qubits = _measurement_qubits(circuit, initial_measurements)
    measured_logicals = tuple(qubit.x for qubit in measurement_qubits)
    edge_usage: Counter[tuple[int, int]] = Counter()

    for index, op in enumerate(operations):
        if isinstance(op.gate, cirq.MeasurementGate):
            measurements.extend(op.qubits)
            continue
        if len(op.qubits) == 1:
            logical = op.qubits[0].x
            routed.append(op.transform_qubits(lambda _: physical_qubits[mapping[logical]]))
            continue
        if len(op.qubits) != 2:
            raise ValueError("only one- and two-qubit operations are supported")

        left, right = op.qubits[0].x, op.qubits[1].x
        future_pairs = future_two_qubit_pairs(operations, index, limit=16)
        if _is_cx(op):
            control_physical = mapping[left]
            target_physical = mapping[right]
            if model.is_coupled(control_physical, target_physical):
                if enable_execute_and_swap and _try_execute_and_swap(
                    routed,
                    physical_qubits,
                    (left, right),
                    future_pairs,
                    mapping,
                    reverse,
                    model,
                    edge_usage,
                    measured_logicals,
                    routing_objective,
                ):
                    continue
            elif enable_bridge and model.hop_distance(control_physical, target_physical) == 2 and _should_bridge(
                (left, right),
                future_pairs,
                mapping,
                reverse,
                model,
                measured_logicals,
                routing_objective,
            ):
                middle = model.hop_path(control_physical, target_physical)[1]
                _append_bridge(routed, physical_qubits, control_physical, middle, target_physical)
                edge_usage[tuple(sorted((control_physical, middle)))] += 2
                edge_usage[tuple(sorted((middle, target_physical)))] += 2
                continue

        _route_pair(
            routed=routed,
            physical_qubits=physical_qubits,
            mapping=mapping,
            reverse=reverse,
            logical_pair=(left, right),
            future_pairs=future_pairs,
            model=model,
            edge_usage=edge_usage,
            measured_logicals=measured_logicals,
            routing_objective=routing_objective,
        )
        if _is_cx(op) and enable_execute_and_swap and _try_execute_and_swap(
            routed,
            physical_qubits,
            (left, right),
            future_pairs,
            mapping,
            reverse,
            model,
            edge_usage,
            measured_logicals,
            routing_objective,
        ):
            continue
        routed.append(op.transform_qubits(lambda qubit: physical_qubits[mapping[qubit.x]]))

    if terminal_remap:
        _apply_terminal_remap(routed, physical_qubits, mapping, reverse, measured_logicals, model)
    routed.append(cirq.measure(*[physical_qubits[mapping[qubit.x]] for qubit in measurement_qubits], key="z"))
    return RoutingResult(circuit=routed, final_mapping=dict(mapping))


def _route_pair(
    routed: cirq.Circuit,
    physical_qubits: tuple[cirq.LineQubit, ...],
    mapping: dict[int, int],
    reverse: dict[int, int | None],
    logical_pair: LogicalPair,
    future_pairs: list[LogicalPair],
    model: BackendNoiseModel,
    edge_usage: Counter[tuple[int, int]],
    measured_logicals: tuple[int, ...],
    routing_objective: str,
) -> None:
    max_steps = max(4, model.backend.num_qubits * 4)
    steps = 0
    while not model.is_coupled(mapping[logical_pair[0]], mapping[logical_pair[1]]):
        if steps >= max_steps:
            raise RuntimeError(f"routing failed to connect logical pair {logical_pair}")
        swap_edge = _choose_swap(
            logical_pair,
            future_pairs,
            mapping,
            reverse,
            model,
            edge_usage,
            measured_logicals,
            routing_objective,
        )
        routed.append(_decomposed_swap(physical_qubits[swap_edge[0]], physical_qubits[swap_edge[1]]))
        _apply_swap(swap_edge, mapping, reverse)
        edge_usage[tuple(sorted(swap_edge))] += 1
        steps += 1


def _choose_swap(
    logical_pair: LogicalPair,
    future_pairs: list[LogicalPair],
    mapping: dict[int, int],
    reverse: dict[int, int | None],
    model: BackendNoiseModel,
    edge_usage: Counter[tuple[int, int]],
    measured_logicals: tuple[int, ...],
    routing_objective: str,
) -> tuple[int, int]:
    candidates = _candidate_swaps(logical_pair, future_pairs, mapping, model, routing_objective)
    return min(
        candidates,
        key=lambda edge: (
            _swap_score(
                edge,
                logical_pair,
                future_pairs,
                mapping,
                reverse,
                model,
                edge_usage,
                measured_logicals,
                routing_objective,
            ),
            edge_usage[tuple(sorted(edge))],
            tuple(sorted(edge)),
        ),
    )


def _candidate_swaps(
    logical_pair: LogicalPair,
    future_pairs: list[LogicalPair],
    mapping: dict[int, int],
    model: BackendNoiseModel,
    routing_objective: str,
) -> set[tuple[int, int]]:
    candidates: set[tuple[int, int]] = set()
    active_physical = [mapping[logical_pair[0]], mapping[logical_pair[1]]]
    for physical in active_physical:
        for neighbor in model.neighbors[physical]:
            candidates.add(tuple(sorted((physical, neighbor))))

    for path in (
        _primary_path(mapping[logical_pair[0]], mapping[logical_pair[1]], model, routing_objective),
        _secondary_path(mapping[logical_pair[0]], mapping[logical_pair[1]], model, routing_objective),
    ):
        for left, right in zip(path, path[1:]):
            candidates.add(tuple(sorted((left, right))))

    for left, right in future_pairs[:6]:
        if left not in mapping or right not in mapping:
            continue
        path = _primary_path(mapping[left], mapping[right], model, routing_objective)
        for a, b in zip(path, path[1:]):
            candidates.add(tuple(sorted((a, b))))

    return {edge for edge in candidates if model.is_coupled(*edge)}


def _swap_score(
    edge: tuple[int, int],
    logical_pair: LogicalPair,
    future_pairs: list[LogicalPair],
    mapping: dict[int, int],
    reverse: dict[int, int | None],
    model: BackendNoiseModel,
    edge_usage: Counter[tuple[int, int]],
    measured_logicals: tuple[int, ...],
    routing_objective: str,
) -> float:
    trial = dict(mapping)
    trial_reverse = dict(reverse)
    _apply_swap(edge, trial, trial_reverse)
    active_left, active_right = trial[logical_pair[0]], trial[logical_pair[1]]
    active_cost = _distance_cost(active_left, active_right, model, routing_objective)
    future_cost = _future_layout_cost(trial, future_pairs, model, routing_objective)
    readout_cost = _terminal_readout_cost(trial, measured_logicals, model)
    reuse_penalty = 0.03 * edge_usage[tuple(sorted(edge))]
    depth_penalty = 0.01
    return model.swap_cost(*edge) + active_cost + 0.25 * future_cost + 0.08 * readout_cost + reuse_penalty + depth_penalty


def _should_execute_and_swap(
    logical_pair: LogicalPair,
    future_pairs: list[LogicalPair],
    mapping: dict[int, int],
    reverse: dict[int, int | None],
    model: BackendNoiseModel,
    measured_logicals: tuple[int, ...],
    routing_objective: str,
) -> bool:
    if not future_pairs:
        return False
    control, target = logical_pair
    control_physical = mapping[control]
    target_physical = mapping[target]
    if not model.is_coupled(control_physical, target_physical):
        return False

    current_score = (
        model.edge_cost(control_physical, target_physical)
        + 0.35 * _future_layout_cost(mapping, future_pairs, model, routing_objective)
        + 0.08 * _terminal_readout_cost(mapping, measured_logicals, model)
    )
    trial = dict(mapping)
    trial_reverse = dict(reverse)
    _apply_swap((control_physical, target_physical), trial, trial_reverse)
    swapped_score = (
        2.0 * model.edge_cost(control_physical, target_physical)
        + 0.35 * _future_layout_cost(trial, future_pairs, model, routing_objective)
        + 0.08 * _terminal_readout_cost(trial, measured_logicals, model)
        + 0.005
    )
    return swapped_score < current_score


def _try_execute_and_swap(
    routed: cirq.Circuit,
    physical_qubits: tuple[cirq.LineQubit, ...],
    logical_pair: LogicalPair,
    future_pairs: list[LogicalPair],
    mapping: dict[int, int],
    reverse: dict[int, int | None],
    model: BackendNoiseModel,
    edge_usage: Counter[tuple[int, int]],
    measured_logicals: tuple[int, ...],
    routing_objective: str,
) -> bool:
    control_physical = mapping[logical_pair[0]]
    target_physical = mapping[logical_pair[1]]
    if not _should_execute_and_swap(logical_pair, future_pairs, mapping, reverse, model, measured_logicals, routing_objective):
        return False
    _append_execute_and_swap(
        routed,
        physical_qubits,
        control_physical,
        target_physical,
    )
    _apply_swap((control_physical, target_physical), mapping, reverse)
    edge_usage[tuple(sorted((control_physical, target_physical)))] += 2
    return True


def _should_bridge(
    logical_pair: LogicalPair,
    future_pairs: list[LogicalPair],
    mapping: dict[int, int],
    reverse: dict[int, int | None],
    model: BackendNoiseModel,
    measured_logicals: tuple[int, ...],
    routing_objective: str,
) -> bool:
    control, target = logical_pair
    control_physical = mapping[control]
    target_physical = mapping[target]
    if model.hop_distance(control_physical, target_physical) != 2:
        return False

    path = model.hop_path(control_physical, target_physical)
    middle = path[1]
    bridge_score = (
        2.0 * model.edge_cost(control_physical, middle)
        + 2.0 * model.edge_cost(middle, target_physical)
        + 0.35 * _future_layout_cost(mapping, future_pairs, model, routing_objective)
        + 0.08 * _terminal_readout_cost(mapping, measured_logicals, model)
        + _bridge_middle_penalty(middle, logical_pair, future_pairs, reverse)
        + 0.02
    )
    swap_scores = []
    for edge in ((control_physical, middle), (middle, target_physical)):
        trial = dict(mapping)
        trial_reverse = dict(reverse)
        _apply_swap(edge, trial, trial_reverse)
        if not model.is_coupled(trial[control], trial[target]):
            continue
        swap_scores.append(
            model.swap_cost(*edge)
            + model.edge_cost(trial[control], trial[target])
            + 0.35 * _future_layout_cost(trial, future_pairs, model, routing_objective)
            + 0.08 * _terminal_readout_cost(trial, measured_logicals, model)
            + 0.01
        )
    if not swap_scores:
        return True
    return bridge_score + 0.005 < min(swap_scores)


def _future_layout_cost(
    mapping: dict[int, int],
    future_pairs: list[LogicalPair],
    model: BackendNoiseModel,
    routing_objective: str = "balanced",
) -> float:
    total = 0.0
    for offset, (left, right) in enumerate(future_pairs):
        if left not in mapping or right not in mapping:
            continue
        decay = 1.0 / (offset + 1)
        total += decay * _distance_cost(mapping[left], mapping[right], model, routing_objective)
    return total


def _distance_cost(left: int, right: int, model: BackendNoiseModel, routing_objective: str) -> float:
    noise = model.noise_distance(left, right)
    hop = model.hop_distance(left, right)
    if routing_objective == "noise":
        return noise + 0.1 * hop
    if routing_objective == "hop":
        return 0.2 * noise + 0.55 * hop
    return noise + 0.25 * hop


def _primary_path(left: int, right: int, model: BackendNoiseModel, routing_objective: str) -> list[int]:
    if routing_objective == "hop":
        return model.hop_path(left, right)
    return model.noise_path(left, right)


def _secondary_path(left: int, right: int, model: BackendNoiseModel, routing_objective: str) -> list[int]:
    if routing_objective == "hop":
        return model.noise_path(left, right)
    return model.hop_path(left, right)


def _terminal_readout_cost(
    mapping: dict[int, int],
    measured_logicals: tuple[int, ...],
    model: BackendNoiseModel,
) -> float:
    return sum(
        model.readout_cost(mapping[logical])
        for logical in measured_logicals
        if logical in mapping
    )


def _apply_terminal_remap(
    routed: cirq.Circuit,
    physical_qubits: tuple[cirq.LineQubit, ...],
    mapping: dict[int, int],
    reverse: dict[int, int | None],
    measured_logicals: tuple[int, ...],
    model: BackendNoiseModel,
    *,
    max_passes: int = 2,
    max_swaps_per_pass: int = 3,
) -> int:
    applied = 0
    for _ in range(max_passes):
        plan = _best_terminal_remap_plan(mapping, reverse, measured_logicals, model, max_swaps_per_pass)
        if plan is None:
            break
        for edge in plan:
            routed.append(_decomposed_swap(physical_qubits[edge[0]], physical_qubits[edge[1]]))
            _apply_swap(edge, mapping, reverse)
            applied += 1
    return applied


def _best_terminal_remap_plan(
    mapping: dict[int, int],
    reverse: dict[int, int | None],
    measured_logicals: tuple[int, ...],
    model: BackendNoiseModel,
    max_swaps: int,
) -> tuple[tuple[int, int], ...] | None:
    current = _terminal_readout_cost(mapping, measured_logicals, model)
    unused = sorted(physical for physical, logical in reverse.items() if logical is None)
    if not unused:
        return None

    best_plan: tuple[tuple[int, int], ...] | None = None
    best_margin = 0.0
    measured_physical = sorted({mapping[logical] for logical in measured_logicals if logical in mapping})
    for start in measured_physical:
        for target in unused:
            for path in (model.noise_path(start, target), model.hop_path(start, target)):
                if len(path) < 2 or len(path) - 1 > max_swaps:
                    continue
                plan = tuple(tuple(sorted(edge)) for edge in zip(path, path[1:]))
                trial = dict(mapping)
                trial_reverse = dict(reverse)
                added_cost = 0.0
                for edge in plan:
                    added_cost += model.swap_cost(*edge)
                    _apply_swap(edge, trial, trial_reverse)
                gain = current - _terminal_readout_cost(trial, measured_logicals, model)
                depth_penalty = 0.004 * len(plan)
                margin = gain - added_cost - depth_penalty
                if margin > best_margin + 1e-12:
                    best_margin = margin
                    best_plan = plan
    return best_plan


def _bridge_middle_penalty(
    middle: int,
    logical_pair: LogicalPair,
    future_pairs: list[LogicalPair],
    reverse: dict[int, int | None],
) -> float:
    middle_logical = reverse.get(middle)
    if middle_logical is None or middle_logical in logical_pair:
        return 0.0
    return 0.02 * sum(
        1.0 / (offset + 1)
        for offset, pair in enumerate(future_pairs)
        if middle_logical in pair
    )


def _is_cx(op: cirq.Operation) -> bool:
    return gate_name_for_operation(op) == "cx"


def _append_execute_and_swap(
    routed: cirq.Circuit,
    physical_qubits: tuple[cirq.LineQubit, ...],
    control: int,
    target: int,
) -> None:
    routed.append(cirq.CNOT(physical_qubits[target], physical_qubits[control]))
    routed.append(cirq.CNOT(physical_qubits[control], physical_qubits[target]))


def _append_bridge(
    routed: cirq.Circuit,
    physical_qubits: tuple[cirq.LineQubit, ...],
    control: int,
    middle: int,
    target: int,
) -> None:
    routed.append(cirq.CNOT(physical_qubits[control], physical_qubits[middle]))
    routed.append(cirq.CNOT(physical_qubits[middle], physical_qubits[target]))
    routed.append(cirq.CNOT(physical_qubits[control], physical_qubits[middle]))
    routed.append(cirq.CNOT(physical_qubits[middle], physical_qubits[target]))


def _apply_swap(
    edge: tuple[int, int],
    mapping: dict[int, int],
    reverse: dict[int, int | None],
) -> None:
    left, right = edge
    logical_left = reverse[left]
    logical_right = reverse[right]
    reverse[left], reverse[right] = logical_right, logical_left
    if logical_left is not None:
        mapping[logical_left] = right
    if logical_right is not None:
        mapping[logical_right] = left


def _measurement_qubits(circuit: cirq.Circuit, measurements: list[cirq.Qid]) -> tuple[cirq.LineQubit, ...]:
    if measurements:
        unique: list[cirq.LineQubit] = []
        seen: set[cirq.LineQubit] = set()
        for qubit in measurements:
            if not isinstance(qubit, cirq.LineQubit):
                raise ValueError("competition circuits must use LineQubit inputs")
            if qubit not in seen:
                unique.append(qubit)
                seen.add(qubit)
        return tuple(unique)
    return tuple(sorted(circuit.all_qubits(), key=lambda qubit: qubit.x))
