from __future__ import annotations

from dataclasses import dataclass
import math

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import count_two_qubit_gates, estimate_log_reliability
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .clifford_sensitivity import clifford_sensitivity_cost
from .track_b import is_clifford_only


@dataclass(frozen=True)
class CompilationCandidate:
    name: str
    circuit: cirq.Circuit
    layout: dict[cirq.LineQubit, int]
    compile_time_sec: float = 0.0


@dataclass(frozen=True)
class CandidateMetrics:
    log_reliability: float
    two_qubit_count: int
    depth: int
    readout_cost: float
    route_overhead: int
    clifford_sensitivity_cost: float
    compile_time_sec: float


def candidate_metrics(
    candidate: CompilationCandidate,
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> CandidateMetrics:
    two_qubit_count = count_two_qubit_gates(candidate.circuit)
    return CandidateMetrics(
        log_reliability=estimate_log_reliability(candidate.circuit, backend),
        two_qubit_count=two_qubit_count,
        depth=len(candidate.circuit),
        readout_cost=terminal_readout_cost(candidate.circuit, backend),
        route_overhead=two_qubit_count - count_two_qubit_gates(source),
        clifford_sensitivity_cost=clifford_sensitivity_cost(candidate.circuit, backend) if is_clifford_only(source) else 0.0,
        compile_time_sec=candidate.compile_time_sec,
    )


def terminal_readout_cost(circuit: cirq.Circuit, backend: CompetitionBackend) -> float:
    measurements = [op for op in circuit.all_operations() if isinstance(op.gate, cirq.MeasurementGate)]
    if not measurements:
        return 0.0
    final_measurement = measurements[-1]
    return sum(-math.log(max(1.0 - backend.readout(qubit.x), 1e-12)) for qubit in final_measurement.qubits)


def pareto_frontier(
    candidates: list[CompilationCandidate],
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> list[CompilationCandidate]:
    metrics = {id(candidate): candidate_metrics(candidate, backend, source) for candidate in candidates}
    frontier: list[CompilationCandidate] = []
    for candidate in candidates:
        candidate_metrics_value = metrics[id(candidate)]
        dominated = any(
            other is not candidate and _dominates(metrics[id(other)], candidate_metrics_value)
            for other in candidates
        )
        if not dominated:
            frontier.append(candidate)
    return frontier


def select_candidate(
    candidates: list[CompilationCandidate],
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> CompilationCandidate:
    ordered = rank_candidates_by_proxy(candidates, backend, source)
    if not ordered:
        raise ValueError("no candidate survived Pareto selection")
    metrics = {id(candidate): candidate_metrics(candidate, backend, source) for candidate in ordered}
    selected = ordered[0]
    for candidate in ordered[1:]:
        if _guarded_better(metrics[id(candidate)], metrics[id(selected)]):
            selected = candidate
    return selected


def rank_candidates_by_proxy(
    candidates: list[CompilationCandidate],
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> list[CompilationCandidate]:
    frontier = pareto_frontier(candidates, backend, source)
    metrics = {
        id(candidate): candidate_metrics(candidate, backend, source)
        for candidate in frontier
    }
    return sorted(
        frontier,
        key=lambda candidate: (
            _selector_score(metrics[id(candidate)]),
            -metrics[id(candidate)].two_qubit_count,
            -metrics[id(candidate)].depth,
            -metrics[id(candidate)].readout_cost,
            candidate.name,
        ),
        reverse=True,
    )


def _selector_score(metrics: CandidateMetrics) -> float:
    return (
        metrics.log_reliability
        - 0.0015 * metrics.two_qubit_count
        - 0.0004 * metrics.depth
        - 0.05 * metrics.readout_cost
        - 0.0005 * max(metrics.route_overhead, 0)
        - 0.006 * metrics.clifford_sensitivity_cost
        - 0.0001 * metrics.compile_time_sec
    )


def _guarded_better(left: CandidateMetrics, right: CandidateMetrics) -> bool:
    if (
        left.two_qubit_count <= right.two_qubit_count - 2
        and left.log_reliability >= right.log_reliability - 0.03
        and left.clifford_sensitivity_cost <= right.clifford_sensitivity_cost + 1.5
    ):
        return True
    if (
        left.log_reliability >= right.log_reliability + 0.05
        and left.two_qubit_count <= right.two_qubit_count + 2
        and left.clifford_sensitivity_cost <= right.clifford_sensitivity_cost + 1.0
    ):
        return True
    if abs(left.log_reliability - right.log_reliability) <= 0.02 and abs(left.two_qubit_count - right.two_qubit_count) <= 1:
        return (
            left.clifford_sensitivity_cost,
            left.depth,
            left.readout_cost,
            left.compile_time_sec,
        ) < (
            right.clifford_sensitivity_cost,
            right.depth,
            right.readout_cost,
            right.compile_time_sec,
        )
    return False


def _dominates(left: CandidateMetrics, right: CandidateMetrics) -> bool:
    comparisons = (
        left.two_qubit_count <= right.two_qubit_count,
        left.log_reliability >= right.log_reliability - 1e-12,
        left.depth <= right.depth,
        left.readout_cost <= right.readout_cost + 1e-12,
        left.clifford_sensitivity_cost <= right.clifford_sensitivity_cost + 1e-12,
        left.compile_time_sec <= right.compile_time_sec + 0.05,
    )
    if not all(comparisons):
        return False
    return (
        left.two_qubit_count < right.two_qubit_count
        or left.log_reliability > right.log_reliability + 1e-12
        or left.depth < right.depth
        or left.readout_cost < right.readout_cost - 1e-12
        or left.clifford_sensitivity_cost < right.clifford_sensitivity_cost - 1e-12
        or left.compile_time_sec < right.compile_time_sec - 0.05
    )
