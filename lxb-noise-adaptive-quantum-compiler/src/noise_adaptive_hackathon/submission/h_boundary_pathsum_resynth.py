from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import time

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .backend_model import BackendNoiseModel
from .circuit_features import extract_circuit_features
from .layout import candidate_initial_layouts
from .peephole import conservative_peephole
from .phase_affine_cone_resynth import (
    PHASE_AFFINE_GATES,
    PhaseAffineInterval,
    _replacement_stream,
    closed_dependency_cone,
    find_phase_affine_intervals,
)
from .router import route_circuit
from .selection import CompilationCandidate, candidate_metrics, rank_candidates_by_proxy
from .track_b import is_clifford_only
from .validation import validate_or_raise


H_BOUNDARY_GATES = PHASE_AFFINE_GATES | {"h"}


@dataclass(frozen=True)
class LinearExpression:
    mask: int
    const: int = 0

    def xor(self, other: "LinearExpression") -> "LinearExpression":
        return LinearExpression(self.mask ^ other.mask, self.const ^ other.const)


@dataclass(frozen=True)
class PathSumTerm:
    kind: str
    coeff_mod8: int
    mask: int
    const: int
    path_variable: int | None = None


@dataclass(frozen=True)
class PathSumSignature:
    qubits: tuple[cirq.Qid, ...]
    num_inputs: int
    num_paths: int
    output_input_rows: tuple[int, ...]
    output_path_rows: tuple[int, ...]
    output_const: tuple[int, ...]
    phase_terms: tuple[PathSumTerm, ...]
    h_count: int


@dataclass(frozen=True)
class HBoundaryCone:
    start: int
    stop: int
    qubits: tuple[cirq.Qid, ...]
    operations: tuple[cirq.Operation, ...]
    signature: PathSumSignature
    phase_affine_intervals: tuple[PhaseAffineInterval, ...]


def h_boundary_pathsum_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    *,
    limit: int = 6,
    max_seconds: float = 8.0,
) -> list[CompilationCandidate]:
    if is_clifford_only(source) or not candidates or len(source.all_qubits()) > 7:
        return []
    started = time.perf_counter()
    ranked = rank_candidates_by_proxy(candidates, backend, source)[:3]
    cones = find_h_boundary_cones(list(source.all_operations()), limit=4)
    if not cones or not ranked:
        return []
    model = BackendNoiseModel.from_backend(backend)
    additions: list[CompilationCandidate] = []
    seen: set[tuple[str, ...]] = set()
    for cone in cones:
        for interval in cone.phase_affine_intervals:
            if time.perf_counter() - started >= max_seconds or len(additions) >= limit:
                return additions
            replacements = _replacement_stream(
                [interval],
                max_seconds=max(0.1, max_seconds - (time.perf_counter() - started)),
                limit=12,
            )
            for replacement in replacements:
                if time.perf_counter() - started >= max_seconds or len(additions) >= limit:
                    return additions
                if not replacement.exact_equivalent:
                    continue
                source_variant = _replace_operations(
                    source,
                    replacement.interval.start,
                    replacement.interval.stop,
                    list(replacement.operations),
                )
                features = extract_circuit_features(source_variant)
                layouts = _candidate_layouts(source_variant, backend, model, features, ranked)
                for layout_index, layout in enumerate(layouts[:6]):
                    if time.perf_counter() - started >= max_seconds or len(additions) >= limit:
                        return additions
                    try:
                        routed = route_circuit(source_variant, backend, model, features, layout).circuit
                        routed = conservative_peephole(routed, clifford_only=False)
                        validate_or_raise(source, routed, backend)
                    except Exception:
                        continue
                    signature = tuple(str(op) for op in routed.all_operations())
                    if signature in seen:
                        continue
                    seen.add(signature)
                    trial = CompilationCandidate(
                        f"h_pathsum_shell_{interval.start}_{interval.stop}_l{layout_index}",
                        routed,
                        dict(layout),
                        time.perf_counter() - started,
                    )
                    if any(_strictly_improves(trial, base, backend, source) for base in ranked):
                        additions.append(trial)
    return additions


def h_boundary_pathsum_diagnostics(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    *,
    max_seconds: float = 8.0,
) -> list[dict[str, object]]:
    started = time.perf_counter()
    ranked = rank_candidates_by_proxy(candidates, backend, source)[:3] if candidates else []
    cones = find_h_boundary_cones(list(source.all_operations()), limit=4)
    if not cones:
        return [{"detector": "h_boundary_pathsum", "accepted": False, "rejection_reason": "no_h_boundary_cone"}]
    rows: list[dict[str, object]] = []
    for cone in cones:
        cone_row = _cone_row(cone)
        if not cone.phase_affine_intervals:
            rows.append({**cone_row, "accepted": False, "rejection_reason": "no_h_free_core"})
            continue
        for interval in cone.phase_affine_intervals:
            if time.perf_counter() - started >= max_seconds:
                break
            replacements = _replacement_stream(
                [interval],
                max_seconds=max(0.1, max_seconds - (time.perf_counter() - started)),
                limit=12,
            )
            if not replacements:
                rows.append(
                    {
                        **cone_row,
                        **_interval_row(interval),
                        "accepted": False,
                        "rejection_reason": "no_exact_core_replacement",
                    }
                )
                continue
            for replacement in replacements:
                row = {
                    **cone_row,
                    **_interval_row(interval),
                    "method": f"h_shell_{replacement.method}",
                    "new_source_cx": replacement.new_cx,
                    "exact_equivalent": replacement.exact_equivalent,
                }
                if not replacement.exact_equivalent:
                    rows.append({**row, "accepted": False, "rejection_reason": "local_unitary_mismatch"})
                    continue
                routed = _route_replacement_diagnostic(
                    source,
                    backend,
                    ranked,
                    replacement.interval.start,
                    replacement.interval.stop,
                    list(replacement.operations),
                    max_seconds=max(0.1, max_seconds - (time.perf_counter() - started)),
                )
                rows.append({**row, **routed})
    return rows


def find_h_boundary_cones(
    operations: list[cirq.Operation],
    *,
    max_h_count: int = 6,
    max_width: int = 7,
    min_cx: int = 5,
    limit: int = 8,
) -> list[HBoundaryCone]:
    non_measure_ops = [op for op in operations if not isinstance(op.gate, cirq.MeasurementGate)]
    cones: list[HBoundaryCone] = []
    seen: set[tuple[int, int]] = set()
    for start in range(len(non_measure_ops)):
        for stop in range(start + 1, len(non_measure_ops) + 1):
            block = tuple(non_measure_ops[start:stop])
            if _cx_count(block) < min_cx or _h_count(block) == 0:
                continue
            if any(gate_name_for_operation(op) not in H_BOUNDARY_GATES for op in block):
                continue
            cone_start, cone_stop, cone_qubits = closed_dependency_cone(non_measure_ops, start, stop)
            if (cone_start, cone_stop) in seen:
                continue
            cone_ops = tuple(non_measure_ops[cone_start:cone_stop])
            if any(gate_name_for_operation(op) not in H_BOUNDARY_GATES for op in cone_ops):
                continue
            if not (1 <= _h_count(cone_ops) <= max_h_count):
                continue
            if _cx_count(cone_ops) < min_cx or len(cone_qubits) > max_width:
                continue
            signature = extract_pathsum_signature(cone_ops, cone_qubits)
            if signature is None:
                continue
            phase_intervals = tuple(
                interval
                for interval in find_phase_affine_intervals(non_measure_ops, max_width=max_width, limit=24)
                if cone_start <= interval.start and interval.stop <= cone_stop
            )
            seen.add((cone_start, cone_stop))
            cones.append(
                HBoundaryCone(
                    start=cone_start,
                    stop=cone_stop,
                    qubits=cone_qubits,
                    operations=cone_ops,
                    signature=signature,
                    phase_affine_intervals=phase_intervals,
                )
            )
    cones.sort(
        key=lambda cone: (
            _cx_count(cone.operations),
            -cone.signature.h_count,
            -(cone.stop - cone.start),
            -len(cone.qubits),
        ),
        reverse=True,
    )
    return cones[:limit]


def extract_pathsum_signature(
    operations: tuple[cirq.Operation, ...],
    qubits: tuple[cirq.Qid, ...],
) -> PathSumSignature | None:
    qubit_index = {qubit: index for index, qubit in enumerate(qubits)}
    expressions = [LinearExpression(1 << index, 0) for index in range(len(qubits))]
    phase_terms: list[PathSumTerm] = []
    next_variable = len(qubits)
    h_count = 0
    for op in operations:
        name = gate_name_for_operation(op)
        if name == "cx":
            control = qubit_index[op.qubits[0]]
            target = qubit_index[op.qubits[1]]
            expressions[target] = expressions[target].xor(expressions[control])
        elif name == "x":
            wire = qubit_index[op.qubits[0]]
            expressions[wire] = LinearExpression(expressions[wire].mask, expressions[wire].const ^ 1)
        elif name in {"t", "tdg", "s", "sdg", "z"}:
            coeff = {"t": 1, "tdg": 7, "s": 2, "sdg": 6, "z": 4}[name]
            expression = expressions[qubit_index[op.qubits[0]]]
            phase_terms.append(PathSumTerm("linear_phase", coeff, expression.mask, expression.const))
        elif name == "h":
            wire = qubit_index[op.qubits[0]]
            old = expressions[wire]
            path_variable = next_variable
            next_variable += 1
            h_count += 1
            phase_terms.append(
                PathSumTerm(
                    "hadamard_kernel",
                    4,
                    old.mask,
                    old.const,
                    path_variable=path_variable,
                )
            )
            expressions[wire] = LinearExpression(1 << path_variable, 0)
        else:
            return None
    input_mask = (1 << len(qubits)) - 1
    output_input_rows = tuple(expression.mask & input_mask for expression in expressions)
    output_path_rows = tuple(expression.mask >> len(qubits) for expression in expressions)
    return PathSumSignature(
        qubits=qubits,
        num_inputs=len(qubits),
        num_paths=next_variable - len(qubits),
        output_input_rows=output_input_rows,
        output_path_rows=output_path_rows,
        output_const=tuple(expression.const for expression in expressions),
        phase_terms=tuple(phase_terms),
        h_count=h_count,
    )


def _route_replacement_diagnostic(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    ranked: list[CompilationCandidate],
    start: int,
    stop: int,
    replacement: list[cirq.Operation],
    *,
    max_seconds: float,
) -> dict[str, object]:
    if not ranked:
        return {"accepted": False, "rejection_reason": "no_base_candidate"}
    started = time.perf_counter()
    source_variant = _replace_operations(source, start, stop, replacement)
    model = BackendNoiseModel.from_backend(backend)
    features = extract_circuit_features(source_variant)
    layouts = _candidate_layouts(source_variant, backend, model, features, ranked)
    best: dict[str, object] = {"accepted": False, "rejection_reason": "no_valid_route"}
    base_metrics = candidate_metrics(ranked[0], backend, source)
    for layout_index, layout in enumerate(layouts[:6]):
        if time.perf_counter() - started >= max_seconds:
            break
        try:
            routed = route_circuit(source_variant, backend, model, features, layout).circuit
            routed = conservative_peephole(routed, clifford_only=False)
            validate_or_raise(source, routed, backend)
        except Exception as error:
            best = {"accepted": False, "rejection_reason": type(error).__name__}
            continue
        trial = CompilationCandidate("h_pathsum_diag", routed, dict(layout), time.perf_counter() - started)
        trial_metrics = candidate_metrics(trial, backend, source)
        improves = _strictly_improves(trial, ranked[0], backend, source)
        row = {
            "layout_index": layout_index,
            "base_cx": base_metrics.two_qubit_count,
            "base_depth": base_metrics.depth,
            "base_elr": base_metrics.log_reliability,
            "new_cx": trial_metrics.two_qubit_count,
            "new_depth": trial_metrics.depth,
            "new_elr": trial_metrics.log_reliability,
            "accepted": improves,
            "rejection_reason": "accepted" if improves else "metric_guard_failed",
        }
        if improves:
            return row
        if (
            "new_elr" not in best
            or (trial_metrics.log_reliability, -trial_metrics.two_qubit_count, -trial_metrics.depth)
            > (float(best["new_elr"]), -int(best["new_cx"]), -int(best["new_depth"]))
        ):
            best = row
    return best


def _candidate_layouts(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    model: BackendNoiseModel,
    features,
    ranked: list[CompilationCandidate],
) -> list[dict[cirq.LineQubit, int]]:
    layouts: list[dict[cirq.LineQubit, int]] = []
    seen: set[tuple[int, ...]] = set()
    logical_qubits = tuple(sorted(source.all_qubits(), key=lambda qubit: qubit.x))

    def add(layout: dict[cirq.LineQubit, int]) -> None:
        signature = tuple(layout.get(logical, -1) for logical in logical_qubits)
        if -1 in signature or signature in seen:
            return
        seen.add(signature)
        layouts.append(dict(layout))

    for base in ranked:
        add(base.layout)
    for layout in candidate_initial_layouts(source, backend, model, features, limit=8):
        add(layout)
    return layouts


def _replace_operations(
    circuit: cirq.Circuit,
    start: int,
    stop: int,
    replacement: list[cirq.Operation],
) -> cirq.Circuit:
    operations = list(circuit.all_operations())
    return cirq.Circuit([*operations[:start], *replacement, *operations[stop:]])


def _cone_row(cone: HBoundaryCone) -> dict[str, object]:
    return {
        "detector": "h_boundary_pathsum",
        "cone_start": cone.start,
        "cone_stop": cone.stop,
        "cone_qubits": [qubit.x for qubit in cone.qubits],
        "cone_cx": _cx_count(cone.operations),
        "cone_h": cone.signature.h_count,
        "num_path_variables": cone.signature.num_paths,
        "output_input_rows": [
            format(row, f"0{cone.signature.num_inputs}b") for row in cone.signature.output_input_rows
        ],
        "output_path_rows": [
            format(row, f"0{cone.signature.num_paths}b") if cone.signature.num_paths else ""
            for row in cone.signature.output_path_rows
        ],
        "output_const": list(cone.signature.output_const),
        "pathsum_terms": [
            {
                "kind": term.kind,
                "coeff_mod8": term.coeff_mod8,
                "mask": format(term.mask, f"0{cone.signature.num_inputs + cone.signature.num_paths}b"),
                "const": term.const,
                "path_variable": term.path_variable,
            }
            for term in cone.signature.phase_terms
        ],
    }


def _interval_row(interval: PhaseAffineInterval) -> dict[str, object]:
    return {
        "interval_start": interval.start,
        "interval_stop": interval.stop,
        "interval_qubits": [qubit.x for qubit in interval.qubits],
        "old_source_cx": _cx_count(interval.operations),
    }


def _strictly_improves(
    trial: CompilationCandidate,
    base: CompilationCandidate,
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> bool:
    trial_metrics = candidate_metrics(trial, backend, source)
    base_metrics = candidate_metrics(base, backend, source)
    cx_improves = trial_metrics.two_qubit_count <= base_metrics.two_qubit_count - 1
    reliability_improves = (
        trial_metrics.log_reliability >= base_metrics.log_reliability + 0.015
        and trial_metrics.two_qubit_count <= base_metrics.two_qubit_count + 1
    )
    return (
        (cx_improves or reliability_improves)
        and trial_metrics.depth <= base_metrics.depth + 2
        and trial_metrics.compile_time_sec <= max(12.0, base_metrics.compile_time_sec + 8.0)
    )


def _cx_count(operations) -> int:
    return sum(1 for op in operations if gate_name_for_operation(op) == "cx")


def _h_count(operations) -> int:
    return sum(1 for op in operations if gate_name_for_operation(op) == "h")


__all__ = [
    "HBoundaryCone",
    "LinearExpression",
    "PathSumSignature",
    "PathSumTerm",
    "extract_pathsum_signature",
    "find_h_boundary_cones",
    "h_boundary_pathsum_candidates",
    "h_boundary_pathsum_diagnostics",
]
