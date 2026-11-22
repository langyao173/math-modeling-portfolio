from __future__ import annotations

import time

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import count_two_qubit_gates, estimate_log_reliability
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .clifford_sensitivity import clifford_sensitivity_cost
from .measurement_channel_general import measured_affine_channels_equivalent
from .relabel import optimize_physical_relabeling
from .selection import CompilationCandidate, candidate_metrics
from .track_b import is_clifford_only
from .validation import validate_or_raise


def measurement_channel_placement_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    *,
    limit: int = 4,
    max_seconds: float = 2.0,
) -> list[CompilationCandidate]:
    """Relabel already-valid affine measurement-channel circuits without changing topology.

    This is deliberately narrower than the generic relabel pass. It only considers
    candidates whose measured affine channel is already equivalent to the source,
    then accepts a physical relabeling only when CX/depth stay fixed and reliability
    and sensitivity do not regress.
    """
    if not is_clifford_only(source) or not candidates:
        return []
    started = time.perf_counter()
    accepted: list[CompilationCandidate] = []
    for base in _rank_affine_candidates(source, backend, candidates):
        if time.perf_counter() - started >= max_seconds:
            break
        try:
            relabeled, layout = optimize_physical_relabeling(
                base.circuit,
                base.layout,
                backend,
                max_mappings=30_000,
                max_seconds=max(0.2, max_seconds - (time.perf_counter() - started)),
            )
            validate_or_raise(source, relabeled, backend)
        except Exception:
            continue
        trial = CompilationCandidate(
            f"{base.name}_mc_placement",
            relabeled,
            layout,
            base.compile_time_sec + time.perf_counter() - started,
        )
        if _placement_improves(trial, base, backend, source):
            accepted.append(trial)
            if len(accepted) >= limit:
                break
    return accepted


def _rank_affine_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
) -> list[CompilationCandidate]:
    affine: list[CompilationCandidate] = []
    for candidate in candidates:
        if not _looks_like_measurement_channel(candidate.name):
            continue
        if not measured_affine_channels_equivalent(source, candidate.circuit, candidate.layout):
            continue
        affine.append(candidate)
    return sorted(
        affine,
        key=lambda candidate: (
            candidate_metrics(candidate, backend, source).two_qubit_count,
            -candidate_metrics(candidate, backend, source).log_reliability,
            candidate.name,
        ),
    )


def _looks_like_measurement_channel(name: str) -> bool:
    return (
        name.startswith("v13c_measurement")
        or name.startswith("v13_1")
        or "measurement_parity" in name
        or "affine" in name
    )


def _placement_improves(
    trial: CompilationCandidate,
    base: CompilationCandidate,
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> bool:
    trial_metrics = candidate_metrics(trial, backend, source)
    base_metrics = candidate_metrics(base, backend, source)
    return (
        trial_metrics.two_qubit_count <= base_metrics.two_qubit_count
        and trial_metrics.depth <= base_metrics.depth
        and trial_metrics.log_reliability >= base_metrics.log_reliability + 0.001
        and trial_metrics.readout_cost <= base_metrics.readout_cost + 0.002
        and trial_metrics.clifford_sensitivity_cost <= base_metrics.clifford_sensitivity_cost + 1e-12
        and count_two_qubit_gates(trial.circuit) == count_two_qubit_gates(base.circuit)
        and len(trial.circuit) == len(base.circuit)
        and estimate_log_reliability(trial.circuit, backend) > estimate_log_reliability(base.circuit, backend)
        and clifford_sensitivity_cost(trial.circuit, backend) <= clifford_sensitivity_cost(base.circuit, backend)
        and measured_affine_channels_equivalent(source, trial.circuit, trial.layout)
    )
