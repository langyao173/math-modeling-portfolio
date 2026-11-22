from __future__ import annotations

import itertools
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cirq

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.benchmarks.model import CompetitionBenchmark, ExpectedOutputRule
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.selection import CompilationCandidate
from noise_adaptive_hackathon.submission.simulation_selector import (
    SELECTOR_TRACK_A_COMMON_RANDOM,
    SimulationSelectorConfig,
)


PUBLIC_TARGETS = (
    "track_a_001",
    "track_a_002",
    "track_a_003",
    "track_a_004",
    "track_a_005",
    "track_a_006",
    "track_a_007",
    "track_a_008",
    "track_b_001",
    "track_b_002",
)


@dataclass(frozen=True)
class LoadedSample:
    name: str
    benchmark: CompetitionBenchmark
    backend: CompetitionBackend
    benchmark_dir: Path


def public_sample_dirs(root: Path = ROOT) -> list[Path]:
    out: list[Path] = []
    for track in ("track_a", "track_b"):
        out.extend(sorted(path for path in (root / "benchmarks" / track / "public").iterdir() if path.is_dir()))
    return out


def load_public_samples(root: Path = ROOT, targets: Iterable[str] | None = None) -> list[LoadedSample]:
    target_set = set(targets) if targets is not None else None
    samples: list[LoadedSample] = []
    for directory in public_sample_dirs(root):
        benchmark = load_benchmark(directory)
        if target_set is not None and benchmark.name not in target_set:
            continue
        backend = load_backend_for_benchmark(root, benchmark)
        samples.append(LoadedSample(benchmark.name, benchmark, backend, directory))
    return samples


def load_backend_for_benchmark(root: Path, benchmark: CompetitionBenchmark) -> CompetitionBackend:
    path = (
        root
        / "backends"
        / benchmark.metadata["backend_size_class"]
        / f"{benchmark.metadata['backend_snapshot_id']}.json"
    )
    return CompetitionBackend.parse_raw(path.read_text(encoding="utf-8"))


def current_root_compiler() -> NoiseAdaptiveCompiler:
    return NoiseAdaptiveCompiler(
        enable_parity_resynthesis=True,
        enable_exact_small_router=True,
        enable_route_scar_normalizer=True,
        enable_measurement_tail_pruning=True,
        enable_relative_phase_toffoli_rewrite=True,
        enable_measurement_channel_physical_placement_v2=True,
        measurement_channel_physical_placement_v2_limit=2,
        enable_ghz_leaf_tree=True,
        enable_readout_copyout=True,
        enable_ghz_relabel=True,
        selector_mode=SELECTOR_TRACK_A_COMMON_RANDOM,
        selector_config=SimulationSelectorConfig(
            mode=SELECTOR_TRACK_A_COMMON_RANDOM,
            track_a_top_k=8,
            track_a_lowshot=512,
            track_a_timeout_sec=2.5,
            track_a_margin=0.006,
        ),
    )


def compile_current_root(source: cirq.Circuit, backend: CompetitionBackend, *, seed: int) -> CompilationCandidate:
    compiler = current_root_compiler()
    started = time.perf_counter()
    circuit = compiler.compile(source, backend, seed=seed)
    return CompilationCandidate("current_root", circuit, dict(compiler.last_initial_layout), time.perf_counter() - started)


def zero_readout_backend(backend: CompetitionBackend) -> CompetitionBackend:
    return backend.copy(update={"readout_error": {str(index): 0.0 for index in range(backend.num_qubits)}})


def measurement_qubits(circuit: cirq.Circuit) -> tuple[int, ...]:
    measurements = [op for op in circuit.all_operations() if isinstance(op.gate, cirq.MeasurementGate)]
    if not measurements:
        return ()
    return tuple(qubit.x for qubit in measurements[-1].qubits)


def sorted_readout_errors(backend: CompetitionBackend) -> list[tuple[float, int]]:
    return sorted((backend.readout(index), index) for index in range(backend.num_qubits))


def best_readout_qubits(backend: CompetitionBackend, width: int) -> tuple[int, ...]:
    return tuple(index for _, index in sorted_readout_errors(backend)[:width])


def cap_for_rule_on_physical(rule: ExpectedOutputRule, physical_qubits: Iterable[int], backend: CompetitionBackend) -> float:
    return cap_for_rule_with_errors(rule, [backend.readout(index) for index in physical_qubits])


def cap_for_rule_with_errors(rule: ExpectedOutputRule, errors: list[float]) -> float:
    if rule.rule_type == "bitstring":
        return math.prod(1.0 - error for error in errors)
    if rule.rule_type == "allowed_bitstrings":
        allowed = tuple(rule.allowed_bitstrings)
        if not allowed:
            return 0.0
        best = 0.0
        for ideal in allowed:
            success = 0.0
            for observed in allowed:
                probability = 1.0
                for ideal_bit, observed_bit, error in zip(ideal, observed, errors):
                    probability *= (1.0 - error) if ideal_bit == observed_bit else error
                success += probability
            best = max(best, success)
        return best
    return parity_success_probability(rule.parity_checks, errors)


def parity_success_probability(checks: tuple[tuple[tuple[int, ...], int], ...], errors: list[float]) -> float:
    if not checks:
        return 1.0
    syndrome_columns: list[int] = []
    for bit in range(len(errors)):
        mask = 0
        for check_index, (indices, _) in enumerate(checks):
            if bit in indices:
                mask ^= 1 << check_index
        syndrome_columns.append(mask)
    probs = {0: 1.0}
    for error, column in zip(errors, syndrome_columns):
        next_probs: dict[int, float] = {}
        for syndrome, probability in probs.items():
            next_probs[syndrome] = next_probs.get(syndrome, 0.0) + probability * (1.0 - error)
            flipped = syndrome ^ column
            next_probs[flipped] = next_probs.get(flipped, 0.0) + probability * error
        probs = next_probs
    return probs.get(0, 0.0)


def best_same_width_readout_cap(rule: ExpectedOutputRule, backend: CompetitionBackend, width: int) -> tuple[float, tuple[int, ...], str]:
    if rule.rule_type == "parity_checks" and pairwise_parity_components(rule.parity_checks, width):
        components = pairwise_parity_components(rule.parity_checks, width)
        qubits = best_readout_qubits(backend, width)
        ordered = sorted((backend.readout(index), index) for index in qubits)
        high_first = list(reversed(ordered))
        # Equal-size public parity components are optimized by concentrating the
        # larger selected readout errors in the same component; the dominant
        # no-flip term is invariant to grouping and this maximizes all-flip
        # correction terms.
        assigned: list[int] = []
        grouped_errors: list[list[float]] = []
        offset = 0
        for size in sorted((len(component) for component in components), reverse=True):
            chunk = high_first[offset : offset + size]
            assigned.extend(index for _, index in chunk)
            grouped_errors.append([error for error, _ in chunk])
            offset += size
        success = math.prod(math.prod(1.0 - e for e in group) + math.prod(e for e in group) for group in grouped_errors)
        return success, tuple(assigned), "best_low_readout_grouped_pairwise_parity"
    qubits = best_readout_qubits(backend, width)
    return cap_for_rule_on_physical(rule, qubits, backend), qubits, "best_low_readout_same_width"


def pairwise_parity_components(
    checks: tuple[tuple[tuple[int, ...], int], ...],
    width: int,
) -> list[tuple[int, ...]]:
    if not checks:
        return []
    graph: dict[int, set[int]] = {index: set() for index in range(width)}
    for indices, _ in checks:
        if len(indices) != 2:
            return []
        left, right = indices
        graph[left].add(right)
        graph[right].add(left)
    components: list[tuple[int, ...]] = []
    remaining = set(range(width))
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = {start}
        while stack:
            node = stack.pop()
            for neighbor in graph[node]:
                if neighbor in component:
                    continue
                component.add(neighbor)
                remaining.discard(neighbor)
                stack.append(neighbor)
        if len(component) > 1:
            components.append(tuple(sorted(component)))
    if not components:
        return []
    return components


def legal_semantic_compressed_cap(rule: ExpectedOutputRule, backend: CompetitionBackend, width: int) -> tuple[float, str]:
    cap, _, _ = best_same_width_readout_cap(rule, backend, width)
    return cap, "same_as_same_width_cap_final_measurement_width_is_mandatory"


def illegal_anchor_oracle_cap(rule: ExpectedOutputRule, backend: CompetitionBackend, width: int) -> tuple[float, int, str]:
    if rule.rule_type == "allowed_bitstrings" and len(rule.allowed_bitstrings) == 2:
        left, right = rule.allowed_bitstrings
        if left == "0" * width and right == "1" * width:
            return 1.0, 1, "illegal_classical_anchor_copy_for_ghz"
    if rule.rule_type == "parity_checks":
        components = pairwise_parity_components(rule.parity_checks, width)
        if components:
            return 1.0, len(components), "illegal_classical_anchor_copy_for_pairwise_parity_components"
    return best_same_width_readout_cap(rule, backend, width)[0], width, "no_compression_detected"


def anf_terms_for_outputs(outputs: tuple[int, ...], width: int) -> list[list[tuple[int, ...]]]:
    per_bit: list[list[tuple[int, ...]]] = []
    for bit in range(width):
        values = [((output >> (width - 1 - bit)) & 1) for output in outputs]
        coeffs = values[:]
        for axis in range(width):
            step = 1 << axis
            for mask in range(1 << width):
                if mask & step:
                    coeffs[mask] ^= coeffs[mask ^ step]
        terms: list[tuple[int, ...]] = []
        for mask, coeff in enumerate(coeffs):
            if not coeff:
                continue
            terms.append(tuple(index for index in range(width) if mask & (1 << (width - 1 - index))))
        per_bit.append(terms)
    return per_bit


def learn_basis_truth_table(circuit: cirq.Circuit) -> tuple[tuple[int, ...], bool]:
    qubits = tuple(sorted(circuit.all_qubits(), key=lambda qubit: qubit.x))
    width = len(qubits)
    outputs: list[int] = []
    deterministic = True
    for value in range(1 << width):
        histogram = exact_output_histogram(prepend_basis_state(circuit, qubits, value), qubits)
        if len(histogram) != 1:
            deterministic = False
            outputs.append(max(histogram, key=histogram.get))
        else:
            outputs.append(next(iter(histogram)))
    return tuple(outputs), deterministic


def exact_output_histogram(circuit: cirq.Circuit, measured_qubits: tuple[cirq.LineQubit, ...]) -> dict[int, float]:
    unitary = cirq.Circuit(op for op in circuit.all_operations() if not isinstance(op.gate, cirq.MeasurementGate))
    qubits = sorted(unitary.all_qubits() | set(measured_qubits), key=lambda qubit: qubit.x)
    positions = {qubit: index for index, qubit in enumerate(qubits)}
    measured_positions = [positions[qubit] for qubit in measured_qubits]
    result = cirq.Simulator().simulate(unitary, qubit_order=qubits)
    histogram: dict[int, float] = {}
    width = len(measured_positions)
    for basis_index, amplitude in enumerate(result.final_state_vector):
        probability = abs(amplitude) ** 2
        if probability < 1e-10:
            continue
        bits = cirq.big_endian_int_to_bits(basis_index, bit_count=len(qubits))
        out = 0
        for index, position in enumerate(measured_positions):
            out |= bits[position] << (width - 1 - index)
        histogram[out] = histogram.get(out, 0.0) + float(probability)
    return histogram


def prepend_basis_state(circuit: cirq.Circuit, qubits: tuple[cirq.LineQubit, ...], value: int) -> cirq.Circuit:
    width = len(qubits)
    challenged = cirq.Circuit()
    for index, qubit in enumerate(qubits):
        if (value >> (width - 1 - index)) & 1:
            challenged.append(cirq.X(qubit))
    challenged += circuit
    return challenged


def affine_model(outputs: tuple[int, ...], width: int) -> tuple[bool, int | None, tuple[int, ...]]:
    constant = outputs[0]
    columns = tuple(outputs[1 << (width - 1 - index)] ^ constant for index in range(width))
    for value, output in enumerate(outputs):
        predicted = constant
        for index, column in enumerate(columns):
            if (value >> (width - 1 - index)) & 1:
                predicted ^= column
        if predicted != output:
            return False, None, ()
    return True, constant, columns


def anf_degree(terms: list[tuple[int, ...]]) -> int:
    return max((len(term) for term in terms), default=0)


def format_anf_terms(terms: list[tuple[int, ...]]) -> str:
    if not terms:
        return "0"
    parts: list[str] = []
    for term in terms:
        if not term:
            parts.append("1")
        else:
            parts.append("*".join(f"x{index}" for index in term))
    return " xor ".join(parts)


def invert_gf2_matrix(rows: tuple[int, ...], width: int) -> tuple[int, ...] | None:
    left = list(rows)
    right = [1 << (width - 1 - index) for index in range(width)]
    for col in range(width):
        bit = 1 << (width - 1 - col)
        pivot = next((row for row in range(col, width) if left[row] & bit), None)
        if pivot is None:
            return None
        if pivot != col:
            left[col], left[pivot] = left[pivot], left[col]
            right[col], right[pivot] = right[pivot], right[col]
        for row in range(width):
            if row != col and (left[row] & bit):
                left[row] ^= left[col]
                right[row] ^= right[col]
    return tuple(right)


def multiply_gf2_matrices(left: tuple[int, ...], right: tuple[int, ...], width: int) -> tuple[int, ...]:
    out: list[int] = []
    for row in left:
        value = 0
        for col in range(width):
            if row & (1 << (width - 1 - col)):
                value ^= right[col]
        out.append(value)
    return tuple(out)


def apply_cnot_to_matrix(rows: tuple[int, ...], control: int, target: int) -> tuple[int, ...]:
    updated = list(rows)
    updated[target] ^= updated[control]
    return tuple(updated)


def cnot_gate_set(width: int) -> list[tuple[int, int]]:
    return [(control, target) for control in range(width) for target in range(width) if control != target]


def identity_matrix(width: int) -> tuple[int, ...]:
    return tuple(1 << (width - 1 - index) for index in range(width))


def enumerate_cnot_matrices(width: int, max_depth: int) -> dict[tuple[int, ...], tuple[tuple[int, int], ...]]:
    gates = cnot_gate_set(width)
    identity = identity_matrix(width)
    seen: dict[tuple[int, ...], tuple[tuple[int, int], ...]] = {identity: ()}
    frontier = {identity: ()}
    for _ in range(max_depth):
        next_frontier: dict[tuple[int, ...], tuple[tuple[int, int], ...]] = {}
        for matrix, sequence in frontier.items():
            for gate in gates:
                new_matrix = apply_cnot_to_matrix(matrix, gate[0], gate[1])
                if new_matrix in seen:
                    continue
                new_sequence = sequence + (gate,)
                seen[new_matrix] = new_sequence
                next_frontier[new_matrix] = new_sequence
        frontier = next_frontier
    return seen


def exact_cnot_synthesis_mitm(
    target_rows: tuple[int, ...],
    width: int,
    max_gates: int,
) -> tuple[tuple[int, int], ...] | None:
    left_depth = max_gates // 2
    right_depth = max_gates - left_depth
    left_seen = enumerate_cnot_matrices(width, left_depth)
    right_seen = enumerate_cnot_matrices(width, right_depth)
    for prefix_matrix, prefix_sequence in sorted(left_seen.items(), key=lambda item: (len(item[1]), item[1])):
        inverse = invert_gf2_matrix(prefix_matrix, width)
        if inverse is None:
            continue
        needed_suffix = multiply_gf2_matrices(target_rows, inverse, width)
        suffix_sequence = right_seen.get(needed_suffix)
        if suffix_sequence is None:
            continue
        if len(prefix_sequence) + len(suffix_sequence) <= max_gates:
            return prefix_sequence + suffix_sequence
    return None


def affine_rows_from_columns(columns: tuple[int, ...], width: int) -> tuple[int, ...]:
    rows: list[int] = []
    for output in range(width):
        row = 0
        for input_index, column in enumerate(columns):
            if column & (1 << (width - 1 - output)):
                row |= 1 << (width - 1 - input_index)
        rows.append(row)
    return tuple(rows)


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

