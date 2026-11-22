from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cirq
import networkx as nx

from noise_adaptive_hackathon.evaluator.validate_circuit import estimate_log_reliability
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .selection import CompilationCandidate, terminal_readout_cost
from .validation import validate_or_raise


@dataclass(frozen=True)
class _CopyPlan:
    source: int
    target: int
    path: tuple[int, ...]
    margin: float


def readout_copyout_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    *,
    limit: int = 4,
    base_limit: int = 6,
) -> list[CompilationCandidate]:
    """Copy final Z outputs to clean low-readout ancillas before measurement.

    A CNOT chain from output qubit q to a clean |0> ancilla preserves the final
    Z distribution on the copied bit while allowing the measurement to happen on
    a lower-readout-error physical qubit.  The transform is only attempted along
    paths whose internal nodes are unused clean ancillas, so it cannot disturb
    the routed computation or the evaluator challenge qubits from the sidecar.
    """

    additions: list[CompilationCandidate] = []
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            estimate_log_reliability(candidate.circuit, backend),
            -terminal_readout_cost(candidate.circuit, backend),
            candidate.name,
        ),
        reverse=True,
    )
    for base in ranked[:base_limit]:
        started = time.perf_counter()
        rewritten = _copyout_circuit(base.circuit, base.layout, backend)
        if rewritten is None:
            continue
        try:
            validate_or_raise(source, rewritten, backend)
        except Exception:
            continue
        if estimate_log_reliability(rewritten, backend) <= estimate_log_reliability(base.circuit, backend) + 0.002:
            continue
        if terminal_readout_cost(rewritten, backend) >= terminal_readout_cost(base.circuit, backend) - 0.002:
            continue
        additions.append(
            CompilationCandidate(
                f"{base.name}_readout_copyout",
                rewritten,
                dict(base.layout),
                base.compile_time_sec + time.perf_counter() - started,
            )
        )
        if len(additions) >= limit:
            break
    return additions


def _copyout_circuit(
    circuit: cirq.Circuit,
    layout: dict[cirq.LineQubit, int],
    backend: CompetitionBackend,
) -> cirq.Circuit | None:
    operations = list(circuit.all_operations())
    measurement_indices = [index for index, op in enumerate(operations) if isinstance(op.gate, cirq.MeasurementGate)]
    if len(measurement_indices) != 1 or measurement_indices[0] != len(operations) - 1:
        return None
    measurement = operations[-1]
    if cirq.measurement_key_name(measurement) != "z":
        return None

    measured = tuple(qubit.x for qubit in measurement.qubits)
    if not measured:
        return None

    body = operations[:-1]
    touched = {qubit.x for op in body for qubit in op.qubits}
    challenged = set(int(physical) for physical in layout.values())
    clean = set(range(backend.num_qubits)) - touched - challenged
    if not clean:
        return None

    graph = backend.graph()
    plans: list[_CopyPlan] = []
    reserved: set[int] = set()
    for output in sorted(measured, key=lambda qubit: _readout_cost(backend, qubit), reverse=True):
        plan = _best_copy_plan(output, clean - reserved, backend, graph)
        if plan is None:
            continue
        plans.append(plan)
        reserved.update(plan.path[1:])

    if not plans:
        return None

    replacements = {plan.source: plan.target for plan in plans}
    physical = tuple(cirq.LineQubit.range(backend.num_qubits))
    rewritten = cirq.Circuit(body)
    for plan in plans:
        for control, target in zip(plan.path, plan.path[1:]):
            rewritten.append(cirq.CNOT(physical[control], physical[target]))
    rewritten.append(cirq.measure(*[physical[replacements.get(qubit, qubit)] for qubit in measured], key="z"))

    if estimate_log_reliability(rewritten, backend) <= estimate_log_reliability(circuit, backend):
        return None
    return rewritten


def _best_copy_plan(
    source: int,
    clean: set[int],
    backend: CompetitionBackend,
    graph: nx.Graph,
) -> _CopyPlan | None:
    source_cost = _readout_cost(backend, source)
    best: _CopyPlan | None = None
    for target in sorted(clean, key=lambda qubit: (backend.readout(qubit), qubit)):
        if backend.readout(target) >= backend.readout(source):
            continue
        try:
            path = nx.shortest_path(graph, source, target, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if len(path) < 2:
            continue
        internal_and_target = set(path[1:])
        if not internal_and_target <= clean:
            continue
        edge_cost = sum(_edge_cost(backend, left, right) for left, right in zip(path, path[1:]))
        gain = source_cost - _readout_cost(backend, target)
        depth_penalty = 0.003 * (len(path) - 1)
        margin = gain - edge_cost - depth_penalty
        if margin <= 0.002:
            continue
        plan = _CopyPlan(source=source, target=target, path=tuple(path), margin=margin)
        if best is None or (plan.margin, -len(plan.path), -backend.readout(plan.target), -plan.target) > (
            best.margin,
            -len(best.path),
            -backend.readout(best.target),
            -best.target,
        ):
            best = plan
    return best


def _readout_cost(backend: CompetitionBackend, qubit: int) -> float:
    return -math.log(max(1.0 - backend.readout(qubit), 1e-12))


def _edge_cost(backend: CompetitionBackend, left: int, right: int) -> float:
    return -math.log(max(1.0 - backend.edge_error(left, right), 1e-12))


__all__ = ["readout_copyout_candidates"]
