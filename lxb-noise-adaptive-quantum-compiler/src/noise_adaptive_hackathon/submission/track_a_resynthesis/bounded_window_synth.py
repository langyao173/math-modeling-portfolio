from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
import itertools
import time

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from ..peephole import conservative_peephole
from ..selection import CompilationCandidate, candidate_metrics, rank_candidates_by_proxy
from ..track_b import is_clifford_only
from ..validation import validate_or_raise
from .cnot_block_gf2 import cnot_block_gf2_replacements
from .exact_equivalence import unitary_equivalent
from .phase_polynomial_real import phase_polynomial_rewrite_candidates
from .reversible_template_library import reversible_library_replacements
from .weighted_steiner_cnot import weighted_cnot_replacements


@dataclass(frozen=True)
class WindowSpec:
    start: int
    stop: int
    qubits: tuple[cirq.Qid, ...]
    operations: tuple[cirq.Operation, ...]
    two_qubit_count: int
    score: float
    family: str


@dataclass(frozen=True)
class WindowReplacement:
    operations: tuple[cirq.Operation, ...]
    reason: str


def bounded_window_synthesis_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    *,
    limit: int = 6,
    max_seconds: float = 10.0,
) -> list[CompilationCandidate]:
    if is_clifford_only(source) or not candidates:
        return []
    started = time.perf_counter()
    additions: list[CompilationCandidate] = []
    seen: set[tuple[str, ...]] = set()
    ranked = rank_candidates_by_proxy(candidates, backend, source)[:6]
    for base in ranked:
        if time.perf_counter() - started >= max_seconds:
            break
        base_metrics = candidate_metrics(base, backend, source)
        for attempt in _attempt_candidate_windows(base, backend, max_seconds=max_seconds - (time.perf_counter() - started)):
            if time.perf_counter() - started >= max_seconds:
                break
            if attempt["accepted"] is not True:
                continue
            circuit = attempt["circuit"]
            assert isinstance(circuit, cirq.Circuit)
            try:
                validate_or_raise(source, circuit, backend)
            except Exception:
                continue
            signature = tuple(str(op) for op in circuit.all_operations())
            if signature in seen:
                continue
            seen.add(signature)
            trial = CompilationCandidate(
                f"{base.name}_window_synth_{attempt['start']}_{attempt['reason']}",
                circuit,
                dict(base.layout),
                base.compile_time_sec + time.perf_counter() - started,
            )
            if _candidate_improves(trial, base, backend, source):
                additions.append(trial)
                if len(additions) >= limit:
                    return additions
    return additions


def bounded_window_synthesis_diagnostics(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    *,
    max_seconds: float = 10.0,
) -> list[dict[str, object]]:
    if is_clifford_only(source) or not candidates:
        return []
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for base in rank_candidates_by_proxy(candidates, backend, source)[:6]:
        if time.perf_counter() - started >= max_seconds:
            break
        base_metrics = candidate_metrics(base, backend, source)
        attempts = _attempt_candidate_windows(
            base,
            backend,
            max_seconds=max(0.1, max_seconds - (time.perf_counter() - started)),
        )
        if not attempts:
            rows.append(
                _diagnostic_row(
                    base,
                    base_metrics,
                    None,
                    None,
                    "bounded_window",
                    "no_candidate_window",
                )
            )
            continue
        for attempt in attempts:
            trial_metrics = None
            if isinstance(attempt.get("circuit"), cirq.Circuit):
                trial = CompilationCandidate(
                    "diagnostic",
                    attempt["circuit"],
                    dict(base.layout),
                    base.compile_time_sec,
                )
                trial_metrics = candidate_metrics(trial, backend, source)
            rows.append(
                _diagnostic_row(
                    base,
                    base_metrics,
                    trial_metrics,
                    attempt,
                    str(attempt.get("detector", "bounded_window")),
                    str(attempt.get("rejection_reason", "unknown")),
                )
            )
    return rows


def high_cost_windows(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    *,
    limit: int = 24,
    max_width: int = 4,
) -> list[WindowSpec]:
    operations = list(circuit.all_operations())
    windows: list[WindowSpec] = []
    for start in range(len(operations)):
        if isinstance(operations[start].gate, cirq.MeasurementGate):
            break
        for stop in range(start + 4, min(len(operations), start + 22) + 1):
            block = tuple(operations[start:stop])
            if any(isinstance(op.gate, cirq.MeasurementGate) for op in block):
                break
            cx_count = sum(1 for op in block if gate_name_for_operation(op) == "cx")
            if cx_count < 2:
                continue
            qubits = tuple(sorted({qubit for op in block for qubit in op.qubits}, key=lambda qubit: qubit.x))
            if not 3 <= len(qubits) <= max_width:
                continue
            family = _window_family(block)
            score = _window_score(block, backend, family)
            windows.append(WindowSpec(start, stop, qubits, block, cx_count, score, family))
    deduped: list[WindowSpec] = []
    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    for window in sorted(windows, key=lambda item: (-item.score, item.start, item.stop)):
        signature = (window.start, window.stop, tuple(qubit.x for qubit in window.qubits))
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(window)
        if len(deduped) >= limit:
            break
    return deduped


def _attempt_candidate_windows(
    base: CompilationCandidate,
    backend: CompetitionBackend,
    *,
    max_seconds: float,
) -> list[dict[str, object]]:
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    operations = list(base.circuit.all_operations())
    for window in high_cost_windows(base.circuit, backend):
        if time.perf_counter() - started >= max_seconds:
            break
        replacements = _window_replacements(window, backend, max_seconds=max_seconds - (time.perf_counter() - started))
        if not replacements:
            rows.append(_attempt_row(base, window, None, "no_replacement"))
            continue
        for replacement in replacements:
            if time.perf_counter() - started >= max_seconds:
                break
            circuit = cirq.Circuit([*operations[: window.start], *replacement.operations, *operations[window.stop :]])
            if not unitary_equivalent(list(window.operations), list(replacement.operations), window.qubits):
                rows.append(_attempt_row(base, window, replacement, "local_unitary_failed"))
                continue
            if not _circuit_uses_legal_edges(circuit, backend):
                rows.append(_attempt_row(base, window, replacement, "illegal_edge"))
                continue
            rows.append(_attempt_row(base, window, replacement, "accepted", circuit=circuit))
            return rows
    return rows


def _window_replacements(
    window: WindowSpec,
    backend: CompetitionBackend,
    *,
    max_seconds: float,
) -> list[WindowReplacement]:
    started = time.perf_counter()
    replacements: list[WindowReplacement] = []
    seen: set[tuple[str, ...]] = {tuple(str(op) for op in window.operations)}

    def add(ops: list[cirq.Operation], reason: str) -> None:
        cleaned = list(conservative_peephole(cirq.Circuit(ops), clifford_only=False).all_operations())
        signature = tuple(str(op) for op in cleaned)
        if signature in seen:
            return
        seen.add(signature)
        if not unitary_equivalent(list(window.operations), cleaned, window.qubits):
            return
        replacements.append(WindowReplacement(tuple(cleaned), reason))

    for replacement in _deletion_replacements(window):
        add(replacement, "delete_redundant_ops")

    for replacement in cnot_block_gf2_replacements(list(window.operations), max_width=len(window.qubits), limit=4):
        candidate = [
            *window.operations[: replacement.start],
            *replacement.operations,
            *window.operations[replacement.stop :],
        ]
        add(candidate, "gf2_cnot_window")

    for replacement in phase_polynomial_rewrite_candidates(list(window.operations), max_width=len(window.qubits), limit=4):
        candidate = [
            *window.operations[: replacement.start],
            *replacement.operations,
            *window.operations[replacement.stop :],
        ]
        add(candidate, "phase_polynomial_window")

    for replacement in reversible_library_replacements(tuple(window.operations), window.qubits):
        add(replacement, "reversible_template_direct")
        if time.perf_counter() - started < max_seconds:
            routed = _route_local_template(
                replacement,
                window.qubits,
                backend,
                max_seconds=max(0.1, max_seconds - (time.perf_counter() - started)),
            )
            if routed is not None:
                add(routed, "reversible_template_local_route")

    if time.perf_counter() - started < max_seconds:
        for replacement in weighted_cnot_replacements(
            list(window.operations),
            backend,
            max_width=len(window.qubits),
            limit=4,
            max_seconds=max(0.1, max_seconds - (time.perf_counter() - started)),
        ):
            candidate = [
                *window.operations[: replacement.start],
                *replacement.operations,
                *window.operations[replacement.stop :],
            ]
            add(candidate, f"weighted_cnot_{replacement.method}")

    return sorted(
        replacements,
        key=lambda item: (
            _two_qubit_count(item.operations),
            len(item.operations),
            item.reason,
        ),
    )


def _deletion_replacements(window: WindowSpec) -> list[list[cirq.Operation]]:
    operations = list(window.operations)
    replacements: list[list[cirq.Operation]] = []
    removable = [index for index, op in enumerate(operations) if not isinstance(op.gate, cirq.MeasurementGate)]
    priority_single = sorted(removable, key=lambda index: (gate_name_for_operation(operations[index]) != "cx", index))
    for index in priority_single[:8]:
        candidate = [op for offset, op in enumerate(operations) if offset != index]
        if _two_qubit_count(candidate) <= window.two_qubit_count and unitary_equivalent(operations, candidate, window.qubits):
            replacements.append(candidate)

    pair_options = _priority_deletion_pairs(operations, removable)
    for left, right in pair_options[:32]:
        candidate = [op for offset, op in enumerate(operations) if offset not in {left, right}]
        if _two_qubit_count(candidate) <= window.two_qubit_count and unitary_equivalent(operations, candidate, window.qubits):
            replacements.append(candidate)
            if len(replacements) >= 8:
                break
    return replacements


def _priority_deletion_pairs(
    operations: list[cirq.Operation],
    removable: list[int],
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int, int, int]] = []
    for left, right in itertools.combinations(removable, 2):
        left_name = gate_name_for_operation(operations[left])
        right_name = gate_name_for_operation(operations[right])
        same_operation = str(operations[left]) == str(operations[right])
        inverse_phase = _phase_inverse_pair(operations[left], operations[right])
        if not same_operation and not inverse_phase:
            continue
        twoq_bonus = 0 if left_name == "cx" and right_name == "cx" else 1
        pairs.append((twoq_bonus, right - left, left, right))
    return [(left, right) for _, _, left, right in sorted(pairs)]


def _phase_inverse_pair(left: cirq.Operation, right: cirq.Operation) -> bool:
    if len(left.qubits) != 1 or left.qubits != right.qubits:
        return False
    names = {gate_name_for_operation(left), gate_name_for_operation(right)}
    return names in ({"t", "tdg"}, {"s", "sdg"})


def _route_local_template(
    logical_operations: list[cirq.Operation],
    qubits: tuple[cirq.Qid, ...],
    backend: CompetitionBackend,
    *,
    max_seconds: float,
    max_expansions: int = 40_000,
) -> list[cirq.Operation] | None:
    physical_ids = tuple(qubit.x for qubit in qubits)
    region_edges = tuple(
        tuple(edge)
        for edge in backend.coupling_map
        if edge[0] in physical_ids and edge[1] in physical_ids
    )
    if not region_edges:
        return None
    edge_set = {tuple(sorted(edge)) for edge in region_edges}
    qubits_by_id = {physical: cirq.LineQubit(physical) for physical in physical_ids}
    logical_index = {qubit: index for index, qubit in enumerate(qubits)}
    identity_mapping = physical_ids
    start = (0, identity_mapping)
    queue: list[tuple[float, int, tuple[int, tuple[int, ...]]]] = [(0.0, 0, start)]
    distances = {start: 0.0}
    previous: dict[tuple[int, tuple[int, ...]], tuple[tuple[int, tuple[int, ...]] | None, tuple[cirq.Operation, ...]]] = {
        start: (None, ())
    }
    sequence = 0
    expansions = 0
    deadline = time.perf_counter() + max_seconds
    while queue and expansions < max_expansions and time.perf_counter() < deadline:
        cost, _, state = heapq.heappop(queue)
        if cost > distances.get(state, float("inf")) + 1e-12:
            continue
        index, mapping = state
        if index == len(logical_operations) and mapping == identity_mapping:
            return _reconstruct_local_route(state, previous)
        expansions += 1
        if index < len(logical_operations):
            for next_state, action_ops, action_cost in _local_execute_transitions(
                state,
                logical_operations[index],
                logical_index,
                edge_set,
                qubits_by_id,
            ):
                next_cost = cost + action_cost
                if next_cost < distances.get(next_state, float("inf")) - 1e-12:
                    distances[next_state] = next_cost
                    previous[next_state] = (state, action_ops)
                    sequence += 1
                    heapq.heappush(queue, (next_cost + 0.001 * (len(logical_operations) - next_state[0]), sequence, next_state))
        for next_state, swap_ops, swap_cost in _local_swap_transitions(state, region_edges, qubits_by_id):
            next_cost = cost + swap_cost
            if next_cost < distances.get(next_state, float("inf")) - 1e-12:
                distances[next_state] = next_cost
                previous[next_state] = (state, swap_ops)
                sequence += 1
                heapq.heappush(queue, (next_cost + 0.001 * (len(logical_operations) - next_state[0]), sequence, next_state))
    return None


def _local_execute_transitions(
    state: tuple[int, tuple[int, ...]],
    op: cirq.Operation,
    logical_index: dict[cirq.Qid, int],
    edge_set: set[tuple[int, int]],
    qubits_by_id: dict[int, cirq.LineQubit],
) -> list[tuple[tuple[int, tuple[int, ...]], tuple[cirq.Operation, ...], float]]:
    index, mapping = state
    if len(op.qubits) == 1:
        physical = mapping[logical_index[op.qubits[0]]]
        transformed = op.transform_qubits(lambda _: qubits_by_id[physical])
        return [((index + 1, mapping), (transformed,), 0.0001)]
    if gate_name_for_operation(op) != "cx":
        return []
    control_index = logical_index[op.qubits[0]]
    target_index = logical_index[op.qubits[1]]
    control = mapping[control_index]
    target = mapping[target_index]
    if tuple(sorted((control, target))) not in edge_set:
        return []
    direct = (
        (index + 1, mapping),
        (cirq.CNOT(qubits_by_id[control], qubits_by_id[target]),),
        1.0,
    )
    swapped = list(mapping)
    swapped[control_index], swapped[target_index] = swapped[target_index], swapped[control_index]
    execute_swap = (
        (index + 1, tuple(swapped)),
        (
            cirq.CNOT(qubits_by_id[target], qubits_by_id[control]),
            cirq.CNOT(qubits_by_id[control], qubits_by_id[target]),
        ),
        2.02,
    )
    return [direct, execute_swap]


def _local_swap_transitions(
    state: tuple[int, tuple[int, ...]],
    edges: tuple[tuple[int, int], ...],
    qubits_by_id: dict[int, cirq.LineQubit],
) -> list[tuple[tuple[int, tuple[int, ...]], tuple[cirq.Operation, ...], float]]:
    index, mapping = state
    transitions: list[tuple[tuple[int, tuple[int, ...]], tuple[cirq.Operation, ...], float]] = []
    for left, right in edges:
        updated = list(mapping)
        left_index = updated.index(left)
        right_index = updated.index(right)
        updated[left_index], updated[right_index] = updated[right_index], updated[left_index]
        transitions.append(
            (
                (index, tuple(updated)),
                _swap_ops(qubits_by_id[left], qubits_by_id[right]),
                3.05,
            )
        )
    return transitions


def _swap_ops(left: cirq.Qid, right: cirq.Qid) -> tuple[cirq.Operation, ...]:
    return (
        cirq.CNOT(left, right),
        cirq.CNOT(right, left),
        cirq.CNOT(left, right),
    )


def _reconstruct_local_route(
    state: tuple[int, tuple[int, ...]],
    previous: dict[tuple[int, tuple[int, ...]], tuple[tuple[int, tuple[int, ...]] | None, tuple[cirq.Operation, ...]]],
) -> list[cirq.Operation]:
    operations: list[cirq.Operation] = []
    current = state
    while previous[current][0] is not None:
        parent, action_ops = previous[current]
        operations.extend(reversed(action_ops))
        assert parent is not None
        current = parent
    return list(reversed(operations))


def _candidate_improves(
    trial: CompilationCandidate,
    base: CompilationCandidate,
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> bool:
    trial_metrics = candidate_metrics(trial, backend, source)
    base_metrics = candidate_metrics(base, backend, source)
    return (
        (
            trial_metrics.two_qubit_count <= base_metrics.two_qubit_count - 1
            or trial_metrics.log_reliability >= base_metrics.log_reliability + 0.03
        )
        and trial_metrics.depth <= base_metrics.depth + 2
        and trial_metrics.readout_cost <= base_metrics.readout_cost + 1e-12
        and trial_metrics.compile_time_sec <= 12.0
    )


def _attempt_row(
    base: CompilationCandidate,
    window: WindowSpec,
    replacement: WindowReplacement | None,
    reason: str,
    *,
    circuit: cirq.Circuit | None = None,
) -> dict[str, object]:
    return {
        "base_candidate": base.name,
        "detector": "bounded_window",
        "start": window.start,
        "stop": window.stop,
        "width": len(window.qubits),
        "family": window.family,
        "old_window_cx": window.two_qubit_count,
        "new_window_cx": _two_qubit_count(replacement.operations) if replacement is not None else None,
        "reason": replacement.reason if replacement is not None else None,
        "accepted": reason == "accepted",
        "rejection_reason": reason,
        "circuit": circuit,
    }


def _diagnostic_row(
    base: CompilationCandidate,
    base_metrics,
    trial_metrics,
    attempt: dict[str, object] | None,
    detector: str,
    reason: str,
) -> dict[str, object]:
    return {
        "detector": detector,
        "base_candidate": base.name,
        "start": None if attempt is None else attempt.get("start"),
        "stop": None if attempt is None else attempt.get("stop"),
        "width": None if attempt is None else attempt.get("width"),
        "family": None if attempt is None else attempt.get("family"),
        "old_window_cx": None if attempt is None else attempt.get("old_window_cx"),
        "new_window_cx": None if attempt is None else attempt.get("new_window_cx"),
        "old_cx": base_metrics.two_qubit_count,
        "old_elr": base_metrics.log_reliability,
        "old_depth": base_metrics.depth,
        "new_cx": getattr(trial_metrics, "two_qubit_count", None),
        "new_elr": getattr(trial_metrics, "log_reliability", None),
        "new_depth": getattr(trial_metrics, "depth", None),
        "validity": attempt is not None and attempt.get("accepted") is True,
        "accepted": attempt is not None and attempt.get("accepted") is True,
        "rejection_reason": reason,
    }


def _window_family(operations: tuple[cirq.Operation, ...]) -> str:
    names = [gate_name_for_operation(op) for op in operations]
    if "h" not in names and all(name in {"cx", "x", "z", "s", "sdg", "t", "tdg", "rz"} for name in names):
        if any(name in {"t", "tdg", "rz"} for name in names):
            return "h_free_phase"
        return "h_free_linear"
    if any(name in {"t", "tdg", "rz"} for name in names):
        return "clifford_t_scaffold"
    return "general"


def _window_score(
    operations: tuple[cirq.Operation, ...],
    backend: CompetitionBackend,
    family: str,
) -> float:
    score = 0.0
    for op in operations:
        if gate_name_for_operation(op) == "cx":
            score += 1.0 + backend.edge_error(op.qubits[0].x, op.qubits[1].x)
        elif gate_name_for_operation(op) in {"t", "tdg", "rz"}:
            score += 0.10
    if family in {"h_free_phase", "clifford_t_scaffold"}:
        score += 0.25
    return score


def _circuit_uses_legal_edges(circuit: cirq.Circuit, backend: CompetitionBackend) -> bool:
    edges = {tuple(sorted(edge)) for edge in backend.coupling_map}
    for op in circuit.all_operations():
        if gate_name_for_operation(op) == "cx" and tuple(sorted((op.qubits[0].x, op.qubits[1].x))) not in edges:
            return False
    return True


def _two_qubit_count(operations: tuple[cirq.Operation, ...] | list[cirq.Operation]) -> int:
    return sum(1 for op in operations if len(op.qubits) == 2)


def rejection_reason_counts(rows: list[dict[str, object]]) -> Counter[str]:
    return Counter(str(row.get("rejection_reason", "unknown")) for row in rows)
