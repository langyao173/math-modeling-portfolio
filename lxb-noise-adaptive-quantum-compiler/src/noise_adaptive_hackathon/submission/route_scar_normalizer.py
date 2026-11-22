from __future__ import annotations

from dataclasses import dataclass, field
import time

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .selection import CompilationCandidate, candidate_metrics, rank_candidates_by_proxy
from .track_a_resynthesis.exact_equivalence import unitary_equivalent
from .track_b import is_clifford_only
from .validation import validate_or_raise


@dataclass(frozen=True)
class RouteScarRewrite:
    kind: str
    deleted_indices: tuple[int, ...]
    deleted_operations: tuple[cirq.Operation, ...]
    qubits: tuple[cirq.Qid, ...]
    exact_validated: bool


@dataclass(frozen=True)
class RouteScarResult:
    circuit: cirq.Circuit
    rewrites: tuple[RouteScarRewrite, ...]
    profile: dict[str, object] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.rewrites)


def normalize_route_scars(
    circuit: cirq.Circuit,
    *,
    max_passes: int = 4,
    max_ladder_width: int = 4,
    max_ladder_span: int = 10,
    enable_ladder_validation: bool = True,
) -> RouteScarResult:
    started = time.perf_counter()
    profile: dict[str, object] = {
        "num_ops": len(list(circuit.all_operations())),
        "num_cx": sum(1 for op in circuit.all_operations() if gate_name_for_operation(op) == "cx"),
        "num_pairs_scanned": 0,
        "num_commutation_checks": 0,
        "num_exact_validations": 0,
        "exact_validation_time_ms": 0.0,
        "num_rewrites": 0,
        "reason_no_rewrite": "none",
        "time_total_ms": 0.0,
    }
    operations = list(circuit.all_operations())
    rewrites: list[RouteScarRewrite] = []
    for _ in range(max_passes):
        rewrite = _find_commuting_same_cx_rewrite(operations, profile)
        if rewrite is None and enable_ladder_validation:
            rewrite = _find_ladder_scar_rewrite(
                operations,
                max_width=max_ladder_width,
                max_span=max_ladder_span,
                profile=profile,
            )
        if rewrite is None:
            break
        remove = set(rewrite.deleted_indices)
        operations = [op for index, op in enumerate(operations) if index not in remove]
        rewrites.append(rewrite)
    profile["num_rewrites"] = len(rewrites)
    if not rewrites and profile.get("reason_no_rewrite") in {None, "none"}:
        profile["reason_no_rewrite"] = "no_pair"
    profile["time_total_ms"] = 1000.0 * (time.perf_counter() - started)
    return RouteScarResult(cirq.Circuit(operations), tuple(rewrites), profile)


def route_scar_normalizer_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    *,
    limit: int = 8,
    max_seconds: float = 0.75,
) -> list[CompilationCandidate]:
    if is_clifford_only(source) or not candidates:
        return []
    started = time.perf_counter()
    additions: list[CompilationCandidate] = []
    seen: set[tuple[str, ...]] = {
        tuple(str(op) for op in candidate.circuit.all_operations())
        for candidate in candidates
    }
    for base in rank_candidates_by_proxy(candidates, backend, source)[:6]:
        if time.perf_counter() - started >= max_seconds:
            break
        base_metrics = candidate_metrics(base, backend, source)
        if _skip_low_pressure_candidate(base_metrics):
            continue
        result = normalize_route_scars(base.circuit)
        if not result.changed:
            continue
        signature = tuple(str(op) for op in result.circuit.all_operations())
        if signature in seen:
            continue
        seen.add(signature)
        try:
            validate_or_raise(source, result.circuit, backend)
        except Exception:
            continue
        trial = CompilationCandidate(
            f"{base.name}_route_scar_normalized",
            result.circuit,
            dict(base.layout),
            base.compile_time_sec + time.perf_counter() - started,
        )
        if _normalizer_improves(trial, base, backend, source):
            additions.append(trial)
            if len(additions) >= limit:
                break
    return additions


def route_scar_normalizer_diagnostics(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    *,
    max_seconds: float = 1.0,
) -> list[dict[str, object]]:
    if is_clifford_only(source) or not candidates:
        return []
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for base in candidates:
        if time.perf_counter() - started >= max_seconds:
            break
        base_metrics = candidate_metrics(base, backend, source)
        if _skip_low_pressure_candidate(base_metrics):
            rows.append(
                {
                    "base_candidate": base.name,
                    "accepted": False,
                    "reason": "low_route_pressure",
                    "profile": {
                        "num_ops": len(list(base.circuit.all_operations())),
                        "num_cx": base_metrics.two_qubit_count,
                        "num_pairs_scanned": 0,
                        "num_commutation_checks": 0,
                        "num_exact_validations": 0,
                        "exact_validation_time_ms": 0.0,
                        "num_rewrites": 0,
                        "reason_no_rewrite": "low_route_pressure",
                        "time_total_ms": 0.0,
                    },
                    "old_cx": base_metrics.two_qubit_count,
                    "old_depth": base_metrics.depth,
                    "old_elr": base_metrics.log_reliability,
                }
            )
            continue
        result = normalize_route_scars(base.circuit)
        if not result.changed:
            rows.append(
                {
                    "base_candidate": base.name,
                    "accepted": False,
                    "reason": result.profile.get("reason_no_rewrite", "no_route_scar"),
                    "profile": result.profile,
                    "old_cx": base_metrics.two_qubit_count,
                    "old_depth": base_metrics.depth,
                    "old_elr": base_metrics.log_reliability,
                }
            )
            continue
        trial = CompilationCandidate("diagnostic", result.circuit, dict(base.layout), base.compile_time_sec)
        trial_metrics = candidate_metrics(trial, backend, source)
        valid = False
        accepted = False
        reason = "accepted"
        try:
            validate_or_raise(source, result.circuit, backend)
            valid = True
            accepted = _normalizer_improves(trial, base, backend, source)
            if not accepted:
                reason = "metric_guard_failed"
        except Exception as error:
            reason = type(error).__name__
        rows.append(
            {
                "base_candidate": base.name,
                "accepted": accepted,
                "valid": valid,
                "reason": reason,
                "profile": result.profile,
                "rewrites": [_rewrite_record(rewrite) for rewrite in result.rewrites],
                "old_cx": base_metrics.two_qubit_count,
                "old_depth": base_metrics.depth,
                "old_elr": base_metrics.log_reliability,
                "new_cx": trial_metrics.two_qubit_count,
                "new_depth": trial_metrics.depth,
                "new_elr": trial_metrics.log_reliability,
            }
        )
    return rows


def cx_commutes_with_operation(control: cirq.Qid, target: cirq.Qid, op: cirq.Operation) -> bool:
    name = gate_name_for_operation(op)
    if name == "measure":
        return False
    if not set(op.qubits) & {control, target}:
        return True
    if name == "cx" and len(op.qubits) == 2:
        other_control, other_target = op.qubits
        return target != other_control and other_target != control
    if len(op.qubits) != 1:
        return False
    qubit = op.qubits[0]
    if qubit == control:
        return name in {"z", "s", "sdg", "t", "tdg", "rz"}
    if qubit == target:
        return name == "x"
    return True


def _find_commuting_same_cx_rewrite(
    operations: list[cirq.Operation],
    profile: dict[str, object],
) -> RouteScarRewrite | None:
    pending: dict[tuple[cirq.Qid, cirq.Qid], int] = {}
    saw_same_edge = False
    for index, op in enumerate(operations):
        if isinstance(op.gate, cirq.MeasurementGate):
            pending.clear()
            continue
        for edge, left_index in list(pending.items()):
            profile["num_commutation_checks"] = int(profile["num_commutation_checks"]) + 1
            if not cx_commutes_with_operation(edge[0], edge[1], op):
                del pending[edge]
                profile["reason_no_rewrite"] = "barrier"
        if gate_name_for_operation(op) != "cx":
            continue
        edge = (op.qubits[0], op.qubits[1])
        if edge in pending:
            left = pending[edge]
            saw_same_edge = True
            profile["num_pairs_scanned"] = int(profile["num_pairs_scanned"]) + 1
            qubits = _operation_qubits(operations[left : index + 1])
            return RouteScarRewrite(
                kind="commuting_same_cx",
                deleted_indices=(left, index),
                deleted_operations=(operations[left], op),
                qubits=qubits,
                exact_validated=False,
            )
        pending[edge] = index
    if not saw_same_edge and profile.get("reason_no_rewrite") == "none":
        profile["reason_no_rewrite"] = "no_pair"
    return None


def _find_ladder_scar_rewrite(
    operations: list[cirq.Operation],
    *,
    max_width: int,
    max_span: int,
    profile: dict[str, object],
) -> RouteScarRewrite | None:
    for left, op in enumerate(operations):
        if gate_name_for_operation(op) != "cx":
            continue
        for right in range(left + 2, min(len(operations), left + max_span + 1)):
            other = operations[right]
            if isinstance(other.gate, cirq.MeasurementGate):
                break
            if gate_name_for_operation(other) != "cx" or other.qubits != op.qubits:
                continue
            profile["num_pairs_scanned"] = int(profile["num_pairs_scanned"]) + 1
            span = tuple(operations[left : right + 1])
            qubits = _operation_qubits(span)
            if len(qubits) > max_width or not _looks_like_ladder_compute_uncompute(span):
                continue
            replacement = list(span[1:-1])
            if _memoized_unitary_equivalent(span, tuple(replacement), qubits, profile):
                return RouteScarRewrite(
                    kind="ladder_compute_uncompute",
                    deleted_indices=(left, right),
                    deleted_operations=(op, other),
                    qubits=qubits,
                    exact_validated=True,
                )
    return None


_EXACT_LADDER_CACHE: dict[tuple[tuple[str, ...], tuple[int, ...]], bool] = {}


def _memoized_unitary_equivalent(
    left: tuple[cirq.Operation, ...],
    right: tuple[cirq.Operation, ...],
    qubits: tuple[cirq.Qid, ...],
    profile: dict[str, object],
) -> bool:
    signature = (
        tuple(str(op) for op in left) + ("=>",) + tuple(str(op) for op in right),
        tuple(qubit.x for qubit in qubits),
    )
    cached = _EXACT_LADDER_CACHE.get(signature)
    if cached is not None:
        return cached
    profile["num_exact_validations"] = int(profile["num_exact_validations"]) + 1
    started = time.perf_counter()
    equivalent = unitary_equivalent(list(left), list(right), qubits)
    profile["exact_validation_time_ms"] = float(profile["exact_validation_time_ms"]) + 1000.0 * (
        time.perf_counter() - started
    )
    _EXACT_LADDER_CACHE[signature] = equivalent
    return equivalent


def _looks_like_ladder_compute_uncompute(operations: tuple[cirq.Operation, ...]) -> bool:
    cxs = [op for op in operations[1:-1] if gate_name_for_operation(op) == "cx"]
    if len(cxs) < 2:
        return False
    counts: dict[tuple[cirq.Qid, cirq.Qid], int] = {}
    for op in cxs:
        counts[(op.qubits[0], op.qubits[1])] = counts.get((op.qubits[0], op.qubits[1]), 0) + 1
    return any(count >= 2 for count in counts.values())


def _normalizer_improves(
    trial: CompilationCandidate,
    base: CompilationCandidate,
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> bool:
    trial_metrics = candidate_metrics(trial, backend, source)
    base_metrics = candidate_metrics(base, backend, source)
    return (
        trial_metrics.two_qubit_count <= base_metrics.two_qubit_count - 2
        and trial_metrics.depth <= base_metrics.depth + 1
        and trial_metrics.log_reliability >= base_metrics.log_reliability + 0.005
        and trial_metrics.readout_cost <= base_metrics.readout_cost + 1e-12
    )


def _skip_low_pressure_candidate(metrics) -> bool:
    return metrics.two_qubit_count < 12 or metrics.route_overhead < 5


def _operation_qubits(operations: tuple[cirq.Operation, ...] | list[cirq.Operation]) -> tuple[cirq.Qid, ...]:
    return tuple(sorted({qubit for op in operations for qubit in op.qubits}, key=lambda qubit: qubit.x))


def _rewrite_record(rewrite: RouteScarRewrite) -> dict[str, object]:
    return {
        "kind": rewrite.kind,
        "deleted_indices": list(rewrite.deleted_indices),
        "deleted_operations": [
            {
                "gate": gate_name_for_operation(op),
                "qubits": [qubit.x for qubit in op.qubits],
                "text": str(op),
            }
            for op in rewrite.deleted_operations
        ],
        "physical_qubits": [qubit.x for qubit in rewrite.qubits],
        "exact_validated": rewrite.exact_validated,
    }


__all__ = [
    "RouteScarResult",
    "RouteScarRewrite",
    "cx_commutes_with_operation",
    "normalize_route_scars",
    "route_scar_normalizer_candidates",
    "route_scar_normalizer_diagnostics",
]
