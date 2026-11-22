from __future__ import annotations

import time

import cirq

from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from ..backend_model import BackendNoiseModel
from ..circuit_features import extract_circuit_features
from ..layout import candidate_initial_layouts
from ..peephole import conservative_peephole
from ..router import route_circuit
from ..selection import CompilationCandidate, candidate_metrics, rank_candidates_by_proxy
from ..track_b import is_clifford_only
from ..validation import validate_or_raise
from .bounded_window_synth import bounded_window_synthesis_candidates, bounded_window_synthesis_diagnostics
from .exact_window_search import semantic_source_variants
from .weighted_steiner_cnot import weighted_cnot_replacements


def track_a_semantic_resynthesis_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    *,
    limit: int = 8,
    max_seconds: float = 10.0,
) -> list[CompilationCandidate]:
    if is_clifford_only(source) or not candidates:
        return []
    started = time.perf_counter()
    ranked = rank_candidates_by_proxy(candidates, backend, source)[:4]
    if not _semantic_budget_worth_spending(ranked, backend, source):
        return []
    additions: list[CompilationCandidate] = []
    for base in ranked:
        if time.perf_counter() - started >= max_seconds:
            break
        _extend_candidates(
            additions,
            bounded_window_synthesis_candidates(
                source,
                backend,
                [base],
                limit=max(1, limit - len(additions)),
                max_seconds=min(3.0, max(0.1, max_seconds - (time.perf_counter() - started))),
            ),
            limit,
        )
        if len(additions) >= limit:
            return additions
    model = BackendNoiseModel.from_backend(backend)
    variants = semantic_source_variants(source, limit=8)
    seen: set[tuple[str, ...]] = set()
    for variant in variants:
        if time.perf_counter() - started >= max_seconds:
            break
        features = extract_circuit_features(variant.circuit)
        layouts = list(candidate_initial_layouts(variant.circuit, backend, model, features, limit=5))
        for base in ranked:
            if base.layout:
                layouts.insert(0, dict(base.layout))
        layout_seen: set[tuple[int, ...]] = set()
        logical_qubits = tuple(sorted(variant.circuit.all_qubits(), key=lambda qubit: qubit.x))
        for layout_index, layout in enumerate(layouts[:8]):
            if time.perf_counter() - started >= max_seconds:
                break
            layout_signature = tuple(layout.get(qubit, -1) for qubit in logical_qubits)
            if layout_signature in layout_seen:
                continue
            layout_seen.add(layout_signature)
            try:
                routed = route_circuit(variant.circuit, backend, model, features, layout).circuit
                routed = conservative_peephole(routed, clifford_only=False)
                validate_or_raise(source, routed, backend)
            except Exception:
                continue
            signature = tuple(str(op) for op in routed.all_operations())
            if signature in seen:
                continue
            seen.add(signature)
            trial = CompilationCandidate(
                f"v15_{variant.name}_l{layout_index}",
                routed,
                dict(layout),
                time.perf_counter() - started,
            )
            if any(_strictly_improves(trial, base, backend, source) for base in ranked):
                additions.append(trial)
                if len(additions) >= limit:
                    return additions
    for base in ranked:
        if time.perf_counter() - started >= max_seconds:
            break
        _extend_weighted_cnot_candidates(
            additions,
            source,
            backend,
            base,
            started=started,
            limit=limit,
            max_seconds=max_seconds,
        )
        if len(additions) >= limit:
            return additions
    return additions


def _semantic_budget_worth_spending(
    ranked: list[CompilationCandidate],
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> bool:
    if not ranked:
        return False
    best = candidate_metrics(ranked[0], backend, source)
    return best.route_overhead >= 5 or best.two_qubit_count >= 22


def semantic_resynthesis_diagnostics(
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
    ranked = rank_candidates_by_proxy(candidates, backend, source)[:4]
    for base in ranked:
        if time.perf_counter() - started >= max_seconds:
            break
        base_metrics = candidate_metrics(base, backend, source)
        try:
            replacements = weighted_cnot_replacements(
                list(base.circuit.all_operations()),
                backend,
                max_seconds=max(0.1, max_seconds - (time.perf_counter() - started)),
            )
        except Exception as error:
            rows.append(_diagnostic_row("weighted_cnot", base.name, base_metrics, None, False, False, type(error).__name__))
            continue
        if not replacements:
            rows.append(_diagnostic_row("weighted_cnot", base.name, base_metrics, None, False, False, "no_replacement"))
            continue
        for replacement in replacements:
            rewritten = _replace_operations(base.circuit, replacement.start, replacement.stop, replacement.operations)
            valid = False
            improves = False
            reason = "accepted"
            try:
                validate_or_raise(source, rewritten, backend)
                valid = True
                trial = CompilationCandidate(
                    f"{base.name}_v15_1_weighted_cnot_{replacement.start}_{replacement.method}",
                    rewritten,
                    dict(base.layout),
                    base.compile_time_sec + time.perf_counter() - started,
                )
                improves = _strictly_improves(trial, base, backend, source)
                if not improves:
                    reason = "metric_guard_failed"
                rows.append(_diagnostic_row("weighted_cnot", base.name, base_metrics, candidate_metrics(trial, backend, source), valid, improves, reason))
            except Exception as error:
                rows.append(_diagnostic_row("weighted_cnot", base.name, base_metrics, None, valid, improves, type(error).__name__))
    rows.extend(
        bounded_window_synthesis_diagnostics(
            source,
            backend,
            ranked,
            max_seconds=max(0.1, max_seconds - (time.perf_counter() - started)),
        )
    )
    return rows


def _extend_candidates(
    target: list[CompilationCandidate],
    additions: list[CompilationCandidate],
    limit: int,
) -> None:
    seen = {tuple(str(op) for op in candidate.circuit.all_operations()) for candidate in target}
    for candidate in additions:
        signature = tuple(str(op) for op in candidate.circuit.all_operations())
        if signature in seen:
            continue
        seen.add(signature)
        target.append(candidate)
        if len(target) >= limit:
            return


def _extend_weighted_cnot_candidates(
    additions: list[CompilationCandidate],
    source: cirq.Circuit,
    backend: CompetitionBackend,
    base: CompilationCandidate,
    *,
    started: float,
    limit: int,
    max_seconds: float,
) -> None:
    try:
        replacements = weighted_cnot_replacements(
            list(base.circuit.all_operations()),
            backend,
            max_seconds=max(0.1, max_seconds - (time.perf_counter() - started)),
        )
    except Exception:
        return
    for replacement in replacements:
        if time.perf_counter() - started >= max_seconds:
            return
        try:
            rewritten = _replace_operations(base.circuit, replacement.start, replacement.stop, replacement.operations)
            validate_or_raise(source, rewritten, backend)
        except Exception:
            continue
        trial = CompilationCandidate(
            f"{base.name}_v15_1_weighted_cnot_{replacement.start}_{replacement.method}",
            rewritten,
            dict(base.layout),
            base.compile_time_sec + time.perf_counter() - started,
        )
        if _strictly_improves(trial, base, backend, source):
            additions.append(trial)
            if len(additions) >= limit:
                return


def _replace_operations(
    circuit: cirq.Circuit,
    start: int,
    stop: int,
    replacement: list[cirq.Operation],
) -> cirq.Circuit:
    operations = list(circuit.all_operations())
    return cirq.Circuit([*operations[:start], *replacement, *operations[stop:]])


def _diagnostic_row(
    detector: str,
    base_name: str,
    base_metrics,
    trial_metrics,
    valid: bool,
    accepted: bool,
    reason: str,
) -> dict[str, object]:
    return {
        "detector": detector,
        "base_candidate": base_name,
        "old_cx": base_metrics.two_qubit_count,
        "old_elr": base_metrics.log_reliability,
        "old_depth": base_metrics.depth,
        "new_cx": getattr(trial_metrics, "two_qubit_count", None),
        "new_elr": getattr(trial_metrics, "log_reliability", None),
        "new_depth": getattr(trial_metrics, "depth", None),
        "validity": valid,
        "accepted": accepted,
        "rejection_reason": reason,
    }


def _strictly_improves(
    trial: CompilationCandidate,
    base: CompilationCandidate,
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> bool:
    trial_metrics = candidate_metrics(trial, backend, source)
    base_metrics = candidate_metrics(base, backend, source)
    cx_improves = trial_metrics.two_qubit_count <= base_metrics.two_qubit_count - 2
    reliability_improves = (
        trial_metrics.log_reliability >= base_metrics.log_reliability + 0.02
        and trial_metrics.two_qubit_count <= base_metrics.two_qubit_count + 1
    )
    return (
        (cx_improves or reliability_improves)
        and trial_metrics.depth <= base_metrics.depth + 2
        and trial_metrics.readout_cost <= base_metrics.readout_cost + 1e-12
        and trial_metrics.compile_time_sec <= max(12.0, base_metrics.compile_time_sec + 10.0)
    )
