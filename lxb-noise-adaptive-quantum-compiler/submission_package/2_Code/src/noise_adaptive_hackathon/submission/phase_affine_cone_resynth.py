from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
import time

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .backend_model import BackendNoiseModel
from .circuit_features import extract_circuit_features
from .clifford_resynthesis import synthesize_cnot_matrix
from .layout import candidate_initial_layouts
from .peephole import conservative_peephole
from .router import route_circuit
from .selection import CompilationCandidate, candidate_metrics, rank_candidates_by_proxy
from .track_a_resynthesis.exact_equivalence import unitary_equivalent
from .track_b import is_clifford_only
from .validation import validate_or_raise


PHASE_AFFINE_GATES = {"cx", "x", "z", "s", "sdg", "t", "tdg"}
PHASE_COEFFICIENTS = {"t": 1, "tdg": -1, "s": 2, "sdg": -2, "z": 4}


@dataclass(frozen=True)
class PhaseAffineSignature:
    qubits: tuple[cirq.Qid, ...]
    rows: tuple[int, ...]
    const: tuple[int, ...]
    phase_terms: tuple[tuple[int, int], ...]
    global_phase: int


@dataclass(frozen=True)
class PhaseAffineInterval:
    start: int
    stop: int
    qubits: tuple[cirq.Qid, ...]
    operations: tuple[cirq.Operation, ...]
    signature: PhaseAffineSignature
    closure_start: int
    closure_stop: int
    closure_qubits: tuple[cirq.Qid, ...]
    closure_phase_affine: bool


@dataclass(frozen=True)
class PhaseAffineReplacement:
    interval: PhaseAffineInterval
    operations: tuple[cirq.Operation, ...]
    method: str
    exact_equivalent: bool

    @property
    def old_cx(self) -> int:
        return _cx_count(self.interval.operations)

    @property
    def new_cx(self) -> int:
        return _cx_count(self.operations)


def phase_affine_cone_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    *,
    limit: int = 6,
    max_seconds: float = 8.0,
) -> list[CompilationCandidate]:
    if is_clifford_only(source) or not candidates:
        return []
    if len(source.all_qubits()) > 7:
        return []
    started = time.perf_counter()
    ranked = rank_candidates_by_proxy(candidates, backend, source)[:3]
    if not ranked:
        return []
    intervals = find_phase_affine_intervals(list(source.all_operations()), max_width=6, limit=12)
    if not intervals:
        return []

    model = BackendNoiseModel.from_backend(backend)
    additions: list[CompilationCandidate] = []
    seen: set[tuple[str, ...]] = set()
    for replacement in _replacement_stream(intervals, max_seconds=max(0.1, max_seconds - (time.perf_counter() - started))):
        if time.perf_counter() - started >= max_seconds or len(additions) >= limit:
            break
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
                break
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
                (
                    f"phase_affine_{replacement.method}_"
                    f"{replacement.interval.start}_{replacement.interval.stop}_l{layout_index}"
                ),
                routed,
                dict(layout),
                time.perf_counter() - started,
            )
            if any(_strictly_improves(trial, base, backend, source) for base in ranked):
                additions.append(trial)
    return additions


def phase_affine_cone_diagnostics(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    *,
    max_seconds: float = 8.0,
) -> list[dict[str, object]]:
    started = time.perf_counter()
    operations = list(source.all_operations())
    intervals = find_phase_affine_intervals(operations, max_width=6, limit=24)
    ranked = rank_candidates_by_proxy(candidates, backend, source)[:3] if candidates else []
    rows: list[dict[str, object]] = []
    if not intervals:
        rows.append({"detector": "phase_affine", "accepted": False, "rejection_reason": "no_interval"})
        return rows

    for interval in intervals:
        if time.perf_counter() - started >= max_seconds:
            break
        base_row = _interval_row(interval)
        replacements = list(
            _replacement_stream(
                [interval],
                max_seconds=max(0.1, max_seconds - (time.perf_counter() - started)),
                limit=24,
            )
        )
        if not replacements:
            rows.append({**base_row, "accepted": False, "rejection_reason": "no_replacement"})
            continue
        for replacement in replacements:
            row = {
                **base_row,
                "method": replacement.method,
                "new_source_cx": replacement.new_cx,
                "exact_equivalent": replacement.exact_equivalent,
                "accepted": False,
                "rejection_reason": "not_routed",
            }
            if not replacement.exact_equivalent:
                row["rejection_reason"] = "local_unitary_mismatch"
                rows.append(row)
                continue
            routed_row = _route_diagnostic(
                source,
                backend,
                ranked,
                replacement,
                max_seconds=max(0.1, max_seconds - (time.perf_counter() - started)),
            )
            rows.append({**row, **routed_row})
    return rows


def find_phase_affine_intervals(
    operations: list[cirq.Operation],
    *,
    max_width: int = 6,
    min_cx: int = 5,
    limit: int = 16,
) -> list[PhaseAffineInterval]:
    non_measure_ops = [op for op in operations if not isinstance(op.gate, cirq.MeasurementGate)]
    intervals: list[PhaseAffineInterval] = []
    seen: set[tuple[int, int]] = set()
    for start in range(len(non_measure_ops)):
        if not _is_phase_affine_operation(non_measure_ops[start]):
            continue
        stop = start
        while stop < len(non_measure_ops) and _is_phase_affine_operation(non_measure_ops[stop]):
            stop += 1
        for sub_start in range(start, stop):
            for sub_stop in range(sub_start + 2, stop + 1):
                if (sub_start, sub_stop) in seen:
                    continue
                block = tuple(non_measure_ops[sub_start:sub_stop])
                if _cx_count(block) < min_cx:
                    continue
                qubits = tuple(sorted({qubit for op in block for qubit in op.qubits}, key=lambda qubit: qubit.x))
                if not 2 <= len(qubits) <= max_width:
                    continue
                signature = extract_phase_affine_signature(block, qubits)
                if signature is None:
                    continue
                closure_start, closure_stop, closure_qubits = closed_dependency_cone(
                    non_measure_ops,
                    sub_start,
                    sub_stop,
                )
                closure_ops = tuple(non_measure_ops[closure_start:closure_stop])
                closure_phase_affine = all(_is_phase_affine_operation(op) for op in closure_ops)
                seen.add((sub_start, sub_stop))
                intervals.append(
                    PhaseAffineInterval(
                        start=sub_start,
                        stop=sub_stop,
                        qubits=qubits,
                        operations=block,
                        signature=signature,
                        closure_start=closure_start,
                        closure_stop=closure_stop,
                        closure_qubits=closure_qubits,
                        closure_phase_affine=closure_phase_affine,
                    )
                )
    intervals.sort(
        key=lambda interval: (
            interval.closure_phase_affine,
            _cx_count(interval.operations),
            interval.stop - interval.start,
            -len(interval.qubits),
        ),
        reverse=True,
    )
    return intervals[:limit]


def closed_dependency_cone(
    operations: list[cirq.Operation],
    start: int,
    stop: int,
) -> tuple[int, int, tuple[cirq.Qid, ...]]:
    cone_start, cone_stop = start, stop
    qubits: set[cirq.Qid] = set()
    changed = True
    while changed:
        previous = (cone_start, cone_stop, tuple(sorted(qubits, key=lambda qubit: qubit.x)))
        qubits = {qubit for op in operations[cone_start:cone_stop] for qubit in op.qubits}
        for index, op in enumerate(operations):
            if isinstance(op.gate, cirq.MeasurementGate):
                continue
            if set(op.qubits) & qubits and (index < cone_start or index >= cone_stop):
                cone_start = min(cone_start, index)
                cone_stop = max(cone_stop, index + 1)
        changed = previous != (cone_start, cone_stop, tuple(sorted(qubits, key=lambda qubit: qubit.x)))
    qubits = {qubit for op in operations[cone_start:cone_stop] for qubit in op.qubits}
    return cone_start, cone_stop, tuple(sorted(qubits, key=lambda qubit: qubit.x))


def extract_phase_affine_signature(
    operations: tuple[cirq.Operation, ...],
    qubits: tuple[cirq.Qid, ...],
) -> PhaseAffineSignature | None:
    index = {qubit: offset for offset, qubit in enumerate(qubits)}
    rows = [1 << offset for offset in range(len(qubits))]
    const = [0 for _ in qubits]
    phase_terms: Counter[int] = Counter()
    global_phase = 0
    for op in operations:
        name = gate_name_for_operation(op)
        if name == "cx":
            control = index[op.qubits[0]]
            target = index[op.qubits[1]]
            rows[target] ^= rows[control]
            const[target] ^= const[control]
        elif name == "x":
            const[index[op.qubits[0]]] ^= 1
        elif name in PHASE_COEFFICIENTS:
            wire = index[op.qubits[0]]
            coeff = PHASE_COEFFICIENTS[name] % 8
            mask = rows[wire]
            bit = const[wire]
            if mask:
                phase_terms[mask] = (phase_terms[mask] + (coeff if bit == 0 else -coeff)) % 8
            if bit:
                global_phase = (global_phase + coeff) % 8
        else:
            return None
    terms = tuple(sorted((mask, coeff % 8) for mask, coeff in phase_terms.items() if coeff % 8))
    return PhaseAffineSignature(
        qubits=qubits,
        rows=tuple(rows),
        const=tuple(const),
        phase_terms=terms,
        global_phase=global_phase % 8,
    )


def _replacement_stream(
    intervals: list[PhaseAffineInterval],
    *,
    max_seconds: float,
    limit: int = 48,
) -> list[PhaseAffineReplacement]:
    started = time.perf_counter()
    replacements: list[PhaseAffineReplacement] = []
    seen: set[tuple[str, ...]] = set()
    for interval in intervals:
        if time.perf_counter() - started >= max_seconds or len(replacements) >= limit:
            break
        for method, ops in _synthesize_interval_replacements(
            interval,
            max_seconds=max(0.1, max_seconds - (time.perf_counter() - started)),
        ):
            signature = tuple(str(op) for op in ops)
            if signature in seen:
                continue
            seen.add(signature)
            exact = unitary_equivalent(list(interval.operations), list(ops), interval.qubits)
            replacements.append(PhaseAffineReplacement(interval, tuple(ops), method, exact))
            if len(replacements) >= limit:
                break
    return replacements


def _synthesize_interval_replacements(
    interval: PhaseAffineInterval,
    *,
    max_seconds: float,
) -> list[tuple[str, list[cirq.Operation]]]:
    started = time.perf_counter()
    outputs: list[tuple[str, list[cirq.Operation]]] = []
    direct = _direct_phase_gadget_synthesis(interval.signature)
    if direct is not None:
        outputs.append(("phase_parity_gadget", direct))
    if time.perf_counter() - started < max_seconds:
        outputs.extend(
            _hybrid_parity_reuse_synthesis(
                interval,
                max_seconds=max(0.1, max_seconds - (time.perf_counter() - started)),
                limit=24,
            )
        )
    return outputs


def _direct_phase_gadget_synthesis(signature: PhaseAffineSignature) -> list[cirq.Operation] | None:
    ops: list[cirq.Operation] = []
    for mask, coeff in sorted(signature.phase_terms, key=lambda item: (item[0].bit_count(), item[0], item[1])):
        bits = [index for index in range(len(signature.qubits)) if (mask >> index) & 1]
        if not bits:
            continue
        target = bits[-1]
        for bit in bits[:-1]:
            ops.append(cirq.CNOT(signature.qubits[bit], signature.qubits[target]))
        ops.extend(_phase_ops(signature.qubits[target], coeff))
        for bit in reversed(bits[:-1]):
            ops.append(cirq.CNOT(signature.qubits[bit], signature.qubits[target]))
    linear = _linear_affine_ops(signature)
    if linear is None:
        return None
    ops.extend(linear)
    return ops


def _hybrid_parity_reuse_synthesis(
    interval: PhaseAffineInterval,
    *,
    max_seconds: float,
    limit: int,
) -> list[tuple[str, list[cirq.Operation]]]:
    signature = interval.signature
    width = len(signature.qubits)
    old_cx = _cx_count(interval.operations)
    if width > 3 or old_cx > 6:
        return []
    if any(signature.const):
        return []
    started = time.perf_counter()
    actions = [(control, target) for control in range(width) for target in range(width) if control != target]
    outputs: list[tuple[str, list[cirq.Operation]]] = []
    seen: set[tuple[str, ...]] = set()
    for length in range(max(0, old_cx - 1), old_cx + 1):
        if time.perf_counter() - started >= max_seconds or len(outputs) >= limit:
            break
        for skeleton in product(actions, repeat=length):
            if time.perf_counter() - started >= max_seconds or len(outputs) >= limit:
                break
            if not _skeleton_can_host_signature(skeleton, signature):
                continue
            ops = _build_hybrid_from_skeleton(skeleton, signature)
            if ops is None:
                continue
            sig = tuple(str(op) for op in ops)
            if sig in seen:
                continue
            seen.add(sig)
            outputs.append((f"hybrid_parity_reuse_cx{length}", ops))
    return outputs


def _skeleton_can_host_signature(
    skeleton: tuple[tuple[int, int], ...],
    signature: PhaseAffineSignature,
) -> bool:
    rows = [1 << index for index in range(len(signature.qubits))]
    seen = set(rows)
    for control, target in skeleton:
        rows[target] ^= rows[control]
        seen.update(rows)
    required = {mask for mask, _ in signature.phase_terms}
    return tuple(rows) == signature.rows and required <= seen


def _build_hybrid_from_skeleton(
    skeleton: tuple[tuple[int, int], ...],
    signature: PhaseAffineSignature,
) -> list[cirq.Operation] | None:
    rows = [1 << index for index in range(len(signature.qubits))]
    pending = dict(signature.phase_terms)
    ops: list[cirq.Operation] = []

    def emit_available() -> None:
        changed = True
        while changed:
            changed = False
            for mask, coeff in list(pending.items()):
                for wire, row in enumerate(rows):
                    if row == mask:
                        ops.extend(_phase_ops(signature.qubits[wire], coeff))
                        del pending[mask]
                        changed = True
                        break
                if changed:
                    break

    emit_available()
    for control, target in skeleton:
        ops.append(cirq.CNOT(signature.qubits[control], signature.qubits[target]))
        rows[target] ^= rows[control]
        emit_available()
    if pending:
        return None
    return ops


def _linear_affine_ops(signature: PhaseAffineSignature) -> list[cirq.Operation] | None:
    matrix = [
        [(signature.rows[row] >> column) & 1 for column in range(len(signature.qubits))]
        for row in range(len(signature.qubits))
    ]
    try:
        cnots = synthesize_cnot_matrix(matrix)
    except ValueError:
        return None
    ops = [cirq.CNOT(signature.qubits[control], signature.qubits[target]) for control, target in cnots]
    for index, bit in enumerate(signature.const):
        if bit:
            ops.append(cirq.X(signature.qubits[index]))
    return ops


def _phase_ops(qubit: cirq.Qid, coeff: int) -> list[cirq.Operation]:
    coeff %= 8
    if coeff == 0:
        return []
    if coeff == 1:
        return [cirq.T(qubit)]
    if coeff == 2:
        return [cirq.S(qubit)]
    if coeff == 3:
        return [cirq.T(qubit), cirq.S(qubit)]
    if coeff == 4:
        return [cirq.Z(qubit)]
    if coeff == 5:
        return [cirq.T(qubit), cirq.Z(qubit)]
    if coeff == 6:
        return [cirq.S(qubit) ** -1]
    return [cirq.T(qubit) ** -1]


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
        if base.layout:
            add(base.layout)
    for layout in candidate_initial_layouts(source, backend, model, features, limit=8):
        add(layout)
    return layouts


def _route_diagnostic(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    ranked: list[CompilationCandidate],
    replacement: PhaseAffineReplacement,
    *,
    max_seconds: float,
) -> dict[str, object]:
    if not ranked:
        return {"rejection_reason": "no_base_candidate"}
    started = time.perf_counter()
    source_variant = _replace_operations(
        source,
        replacement.interval.start,
        replacement.interval.stop,
        list(replacement.operations),
    )
    model = BackendNoiseModel.from_backend(backend)
    features = extract_circuit_features(source_variant)
    layouts = _candidate_layouts(source_variant, backend, model, features, ranked)
    best_row: dict[str, object] = {"rejection_reason": "no_valid_route"}
    for layout_index, layout in enumerate(layouts[:6]):
        if time.perf_counter() - started >= max_seconds:
            break
        try:
            routed = route_circuit(source_variant, backend, model, features, layout).circuit
            routed = conservative_peephole(routed, clifford_only=False)
            validate_or_raise(source, routed, backend)
        except Exception as error:
            best_row = {"rejection_reason": type(error).__name__}
            continue
        trial = CompilationCandidate(
            f"phase_affine_diag_l{layout_index}",
            routed,
            dict(layout),
            time.perf_counter() - started,
        )
        trial_metrics = candidate_metrics(trial, backend, source)
        base_metrics = candidate_metrics(ranked[0], backend, source)
        improves = _strictly_improves(trial, ranked[0], backend, source)
        row = {
            "layout_index": layout_index,
            "new_cx": trial_metrics.two_qubit_count,
            "new_depth": trial_metrics.depth,
            "new_elr": trial_metrics.log_reliability,
            "base_cx": base_metrics.two_qubit_count,
            "base_depth": base_metrics.depth,
            "base_elr": base_metrics.log_reliability,
            "accepted": improves,
            "rejection_reason": "accepted" if improves else "metric_guard_failed",
        }
        if improves:
            return row
        if (
            "new_elr" not in best_row
            or (trial_metrics.log_reliability, -trial_metrics.two_qubit_count)
            > (float(best_row["new_elr"]), -int(best_row["new_cx"]))
        ):
            best_row = row
    return best_row


def _replace_operations(
    circuit: cirq.Circuit,
    start: int,
    stop: int,
    replacement: list[cirq.Operation],
) -> cirq.Circuit:
    operations = list(circuit.all_operations())
    return cirq.Circuit([*operations[:start], *replacement, *operations[stop:]])


def _interval_row(interval: PhaseAffineInterval) -> dict[str, object]:
    return {
        "detector": "phase_affine",
        "start": interval.start,
        "stop": interval.stop,
        "qubits": [qubit.x for qubit in interval.qubits],
        "old_source_cx": _cx_count(interval.operations),
        "gate_names": [gate_name_for_operation(op) for op in interval.operations],
        "closure_start": interval.closure_start,
        "closure_stop": interval.closure_stop,
        "closure_qubits": [qubit.x for qubit in interval.closure_qubits],
        "closure_phase_affine": interval.closure_phase_affine,
        "affine_rows": [format(row, f"0{len(interval.qubits)}b") for row in interval.signature.rows],
        "affine_const": list(interval.signature.const),
        "phase_terms": [
            {"mask": format(mask, f"0{len(interval.qubits)}b"), "coeff_mod8": coeff}
            for mask, coeff in interval.signature.phase_terms
        ],
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


def _is_phase_affine_operation(op: cirq.Operation) -> bool:
    return gate_name_for_operation(op) in PHASE_AFFINE_GATES


def _cx_count(operations) -> int:
    return sum(1 for op in operations if gate_name_for_operation(op) == "cx")


__all__ = [
    "PhaseAffineInterval",
    "PhaseAffineReplacement",
    "PhaseAffineSignature",
    "closed_dependency_cone",
    "extract_phase_affine_signature",
    "find_phase_affine_intervals",
    "phase_affine_cone_candidates",
    "phase_affine_cone_diagnostics",
]
