from __future__ import annotations

from dataclasses import dataclass
import time

import cirq

from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.simulator.cirq_runner import _inject_cirq_noise
from noise_adaptive_hackathon.simulator.stim_runner import _cirq_to_stim

from .selection import CompilationCandidate, rank_candidates_by_proxy, select_candidate
from .track_b import is_clifford_only


SELECTOR_PROXY_ONLY = "proxy_only"
SELECTOR_TRACK_A_RANKED_PROXY = "track_a_ranked_proxy_only"
SELECTOR_LOWSHOT = "proxy_plus_lowshot_noise"
SELECTOR_COMMON_RANDOM = "proxy_plus_common_random_noise"
SELECTOR_TRACK_A_COMMON_RANDOM = "track_a_proxy_plus_common_random_noise"
SELECTOR_TRACK_B_COMMON_RANDOM = "track_b_proxy_plus_common_random_noise"
SELECTOR_MODES = (
    SELECTOR_PROXY_ONLY,
    SELECTOR_TRACK_A_RANKED_PROXY,
    SELECTOR_LOWSHOT,
    SELECTOR_COMMON_RANDOM,
    SELECTOR_TRACK_A_COMMON_RANDOM,
    SELECTOR_TRACK_B_COMMON_RANDOM,
)


@dataclass(frozen=True)
class SimulationSelectorConfig:
    mode: str = SELECTOR_COMMON_RANDOM
    track_a_top_k: int = 4
    track_b_top_k: int = 6
    track_a_lowshot: int = 128
    track_b_lowshot: int = 512
    track_a_timeout_sec: float = 4.0
    track_b_timeout_sec: float = 2.0
    track_a_margin: float = 0.03
    track_b_margin: float = 0.015
    max_track_a_active_qubits: int = 11
    ideal_sample_shots: int = 4096
    max_sampled_support: int = 2048


def select_candidate_with_simulation(
    candidates: list[CompilationCandidate],
    backend: CompetitionBackend,
    source: cirq.Circuit,
    *,
    seed: int,
    config: SimulationSelectorConfig | None = None,
) -> CompilationCandidate:
    config = config or SimulationSelectorConfig()
    if config.mode == SELECTOR_PROXY_ONLY:
        return select_candidate(candidates, backend, source)
    if config.mode not in SELECTOR_MODES:
        raise ValueError(f"unknown selector mode: {config.mode}")

    proxy_selected = select_candidate(candidates, backend, source)
    ranked = rank_candidates_by_proxy(candidates, backend, source)
    track_b = is_clifford_only(source)
    if config.mode == SELECTOR_TRACK_A_RANKED_PROXY:
        if track_b:
            return proxy_selected
        return ranked[0] if ranked else proxy_selected
    if config.mode == SELECTOR_TRACK_A_COMMON_RANDOM and track_b:
        return proxy_selected
    if config.mode == SELECTOR_TRACK_B_COMMON_RANDOM and not track_b:
        return proxy_selected
    if len(ranked) <= 1:
        return proxy_selected

    top_k = config.track_b_top_k if track_b else config.track_a_top_k
    shots = config.track_b_lowshot if track_b else config.track_a_lowshot
    timeout_sec = config.track_b_timeout_sec if track_b else config.track_a_timeout_sec
    deadline = time.perf_counter() + timeout_sec

    scored_candidates = _dedupe_preserving_order([proxy_selected, *ranked[:top_k]])
    support = _ideal_output_support(source, track_b=track_b, seed=seed, config=config)
    if not support:
        return proxy_selected

    scores: dict[int, float] = {}
    for index, candidate in enumerate(scored_candidates):
        if time.perf_counter() >= deadline:
            break
        candidate_seed = (
            seed
            if config.mode in {SELECTOR_COMMON_RANDOM, SELECTOR_TRACK_A_COMMON_RANDOM, SELECTOR_TRACK_B_COMMON_RANDOM}
            else seed + 997 * (index + 1)
        )
        score = _noisy_support_success(
            candidate.circuit,
            backend,
            support,
            track_b=track_b,
            shots=shots,
            seed=candidate_seed,
            config=config,
        )
        if score is not None:
            scores[id(candidate)] = score

    if id(proxy_selected) not in scores or not scores:
        return proxy_selected
    best = max(
        (candidate for candidate in scored_candidates if id(candidate) in scores),
        key=lambda candidate: (
            scores[id(candidate)],
            -ranked.index(candidate) if candidate in ranked else 0,
        ),
    )
    if best is proxy_selected:
        return proxy_selected

    margin = config.track_b_margin if track_b else config.track_a_margin
    if scores[id(best)] >= scores[id(proxy_selected)] + margin:
        return best
    return proxy_selected


def _dedupe_preserving_order(candidates: list[CompilationCandidate]) -> list[CompilationCandidate]:
    seen: set[int] = set()
    unique: list[CompilationCandidate] = []
    for candidate in candidates:
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        unique.append(candidate)
    return unique


def _ideal_output_support(
    source: cirq.Circuit,
    *,
    track_b: bool,
    seed: int,
    config: SimulationSelectorConfig,
) -> set[str] | None:
    if track_b:
        return _sampled_ideal_support(source, seed=seed, config=config)
    exact = _exact_ideal_support(source)
    if exact is not None:
        return exact
    return _sampled_ideal_support(source, seed=seed, config=config)


def _exact_ideal_support(source: cirq.Circuit) -> set[str] | None:
    measurements = [op for op in source.all_operations() if isinstance(op.gate, cirq.MeasurementGate)]
    if len(measurements) != 1 or cirq.measurement_key_name(measurements[0]) != "z":
        return None
    unitary_part = cirq.Circuit(op for op in source.all_operations() if not isinstance(op.gate, cirq.MeasurementGate))
    qubits = sorted(unitary_part.all_qubits() | set(measurements[0].qubits))
    if len(qubits) > 12:
        return None
    positions = {qubit: index for index, qubit in enumerate(qubits)}
    measured_positions = [positions[qubit] for qubit in measurements[0].qubits]
    try:
        result = cirq.Simulator(seed=1234).simulate(unitary_part, qubit_order=qubits)
    except Exception:
        return None
    support: set[str] = set()
    for basis_index, amplitude in enumerate(result.final_state_vector):
        probability = abs(amplitude) ** 2
        if probability < 1e-8:
            continue
        bits = cirq.big_endian_int_to_bits(basis_index, bit_count=len(qubits))
        support.add("".join(str(bits[position]) for position in measured_positions))
    return support or None


def _sampled_ideal_support(
    source: cirq.Circuit,
    *,
    seed: int,
    config: SimulationSelectorConfig,
) -> set[str] | None:
    try:
        simulator = cirq.CliffordSimulator(seed=seed) if is_clifford_only(source) else cirq.Simulator(seed=seed)
        result = simulator.run(source, repetitions=config.ideal_sample_shots)
    except Exception:
        return None
    support = {"".join(str(bit) for bit in row) for row in result.measurements["z"]}
    if not support or len(support) > config.max_sampled_support:
        return None
    return support


def _noisy_support_success(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    support: set[str],
    *,
    track_b: bool,
    shots: int,
    seed: int,
    config: SimulationSelectorConfig,
) -> float | None:
    try:
        if track_b:
            stim_circuit = _cirq_to_stim(circuit, backend)
            samples = stim_circuit.compile_sampler(seed=seed).sample(shots)
            correct = sum(1 for row in samples if "".join(str(int(bit)) for bit in row) in support)
        else:
            if len(circuit.all_qubits()) > config.max_track_a_active_qubits:
                return None
            noisy = _inject_cirq_noise(circuit, backend)
            result = cirq.DensityMatrixSimulator(seed=seed).run(noisy, repetitions=shots)
            correct = sum(1 for row in result.measurements["z"] if "".join(str(bit) for bit in row) in support)
    except Exception:
        return None
    return correct / shots
