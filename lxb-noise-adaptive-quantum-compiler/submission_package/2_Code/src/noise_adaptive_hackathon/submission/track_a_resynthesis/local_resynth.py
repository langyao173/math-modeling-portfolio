from __future__ import annotations

import time

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import count_two_qubit_gates, estimate_log_reliability
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from ..backend_model import BackendNoiseModel
from ..circuit_features import extract_circuit_features
from ..layout import candidate_initial_layouts
from ..local_block_resynthesis import resynthesize_local_two_qubit_blocks
from ..peephole import conservative_peephole
from ..router import route_circuit
from ..selection import CompilationCandidate, candidate_metrics, rank_candidates_by_proxy
from ..track_b import is_clifford_only
from ..validation import validate_or_raise
from .template_matcher import source_template_variants


def track_a_resynthesis_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    *,
    limit: int = 6,
) -> list[CompilationCandidate]:
    if is_clifford_only(source) or not candidates:
        return []
    additions: list[CompilationCandidate] = []
    ranked = rank_candidates_by_proxy(candidates, backend, source)[:limit]
    if _template_budget_available(ranked):
        additions.extend(_source_template_candidates(source, backend, ranked[:3], limit=limit))
    for candidate in ranked:
        try:
            started = time.perf_counter()
            rewritten, changed = resynthesize_local_two_qubit_blocks(candidate.circuit, backend)
            if not changed:
                continue
            validate_or_raise(source, rewritten, backend)
        except Exception:
            continue
        trial = CompilationCandidate(
            f"{candidate.name}_v14_pair_kak",
            rewritten,
            dict(candidate.layout),
            candidate.compile_time_sec + time.perf_counter() - started,
        )
        if _strictly_improves(trial, candidate, backend, source):
            additions.append(trial)
    return additions


def _template_budget_available(ranked: list[CompilationCandidate]) -> bool:
    if not ranked:
        return False
    best = ranked[0]
    if best.name.startswith("exact_small_router"):
        return False
    return best.compile_time_sec <= 4.0


def _source_template_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    ranked: list[CompilationCandidate],
    *,
    limit: int,
) -> list[CompilationCandidate]:
    if not ranked:
        return []
    started = time.perf_counter()
    additions: list[CompilationCandidate] = []
    model = BackendNoiseModel.from_backend(backend)
    variants = source_template_variants(source, limit=4)
    seen: set[tuple[str, ...]] = set()
    for variant_index, variant in enumerate(variants):
        if time.perf_counter() - started > 8.0:
            break
        features = extract_circuit_features(variant.circuit)
        layouts = list(candidate_initial_layouts(variant.circuit, backend, model, features, limit=5))
        for base in ranked:
            if base.layout:
                layouts.insert(0, dict(base.layout))
        layout_seen: set[tuple[int, ...]] = set()
        logical_qubits = tuple(sorted(variant.circuit.all_qubits(), key=lambda qubit: qubit.x))
        for layout_index, layout in enumerate(layouts[:8]):
            signature_layout = tuple(layout.get(qubit, -1) for qubit in logical_qubits)
            if signature_layout in layout_seen:
                continue
            layout_seen.add(signature_layout)
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
                f"v14_{variant.name}_l{layout_index}",
                routed,
                dict(layout),
                time.perf_counter() - started,
            )
            if any(_strictly_improves(trial, base, backend, source) for base in ranked):
                additions.append(trial)
                if len(additions) >= limit:
                    return additions
    return additions


def _strictly_improves(
    trial: CompilationCandidate,
    base: CompilationCandidate,
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> bool:
    trial_metrics = candidate_metrics(trial, backend, source)
    base_metrics = candidate_metrics(base, backend, source)
    return (
        trial_metrics.two_qubit_count < base_metrics.two_qubit_count
        and trial_metrics.depth <= base_metrics.depth + 1
        and trial_metrics.readout_cost <= base_metrics.readout_cost + 1e-12
        and trial_metrics.log_reliability >= base_metrics.log_reliability - 0.01
        and trial_metrics.compile_time_sec <= 12.0
        and count_two_qubit_gates(trial.circuit) < count_two_qubit_gates(base.circuit)
        and estimate_log_reliability(trial.circuit, backend) >= estimate_log_reliability(base.circuit, backend) - 0.01
    )
