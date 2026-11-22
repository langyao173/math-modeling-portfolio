from __future__ import annotations

from dataclasses import dataclass
import time

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import count_two_qubit_gates, estimate_log_reliability
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .circuit_features import extract_circuit_features
from .clifford_sensitivity import clifford_sensitivity_cost
from .layout import hub_aware_layout
from .legacy_parity import _compile_with_cirq_cqc
from .measurement_channel import measurement_channel_parity_candidate
from .peephole import conservative_peephole
from .selection import CompilationCandidate, candidate_metrics
from .track_b import is_clifford_only
from .validation import validate_or_raise


@dataclass(frozen=True)
class AffineMeasurementChannel:
    width: int
    random_columns: tuple[int, ...]
    input_columns: tuple[int, ...]
    constant: int


@dataclass(frozen=True)
class AffineChannelDiagnostics:
    supported: bool
    width: int
    random_rank: int
    input_rank: int
    channel_class: str
    reason: str = ""


def measurement_channel_general_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    legacy: CompilationCandidate | None,
    *,
    max_seconds: float = 3.0,
) -> list[CompilationCandidate]:
    started = time.perf_counter()
    if legacy is None or not is_clifford_only(source) or len(source.all_qubits()) < 8:
        return []
    source_channel = extract_affine_measurement_channel(source)
    if source_channel is None:
        return []

    candidates: list[CompilationCandidate] = []
    fallback = measurement_channel_parity_candidate(
        source,
        backend,
        legacy,
        max_seconds=max(0.2, min(1.5, max_seconds * 0.5)),
    )
    if fallback is not None and measured_affine_channels_equivalent(source, fallback.circuit, fallback.layout):
        candidates.append(
            CompilationCandidate(
                "v13_1_disjoint_star_fallback",
                fallback.circuit,
                fallback.layout,
                time.perf_counter() - started,
            )
        )

    remaining = max_seconds - (time.perf_counter() - started)
    if remaining <= 0.2:
        return candidates

    synthesized = _synthesize_simple_affine_logical(source_channel)
    if synthesized is None:
        return candidates
    if not channels_equivalent(source_channel, extract_affine_measurement_channel(synthesized)):
        return candidates

    layout = hub_aware_layout(synthesized, backend, BackendNoiseModelShim(backend))
    if layout is None:
        layout = _low_readout_identity_layout(synthesized, backend)
    if layout is None:
        return candidates

    try:
        compiled, routed_layout = _compile_with_cirq_cqc(synthesized, backend, 12, layout)
        compiled = conservative_peephole(compiled, clifford_only=True)
        validate_or_raise(source, compiled, backend)
    except Exception:
        return candidates

    candidate = CompilationCandidate(
        "v13_1_low_noise_affine_route",
        compiled,
        routed_layout,
        time.perf_counter() - started,
    )
    if (
        measured_affine_channels_equivalent(source, candidate.circuit, candidate.layout)
        and _accept_general_candidate(candidate, legacy, backend, source)
    ):
        candidates.append(candidate)
    return candidates


def extract_affine_measurement_channel(
    circuit: cirq.Circuit,
    *,
    input_qubit_map: dict[int, int] | None = None,
) -> AffineMeasurementChannel | None:
    if not is_clifford_only(circuit):
        return None
    operations = list(circuit.all_operations())
    measurements = [op for op in operations if isinstance(op.gate, cirq.MeasurementGate)]
    if len(measurements) != 1 or operations[-1] is not measurements[0]:
        return None
    measurement = measurements[0]
    if cirq.measurement_key_name(measurement) != "z":
        return None
    unitary_ops = operations[:-1]
    width = len(measurement.qubits)
    if width == 0:
        return None
    if input_qubit_map is None:
        logical_ids = sorted(qubit.x for qubit in circuit.all_qubits())
        input_qubit_map = {logical_id: logical_id for logical_id in logical_ids}
    logical_width = max(input_qubit_map.values(), default=-1) + 1
    if logical_width != width:
        return None

    random_columns = [0 for _ in range(width)]
    input_columns = [0 for _ in range(width)]
    constant = 0
    for row, measured_qubit in enumerate(measurement.qubits):
        try:
            pauli = cirq.PauliString({measured_qubit: cirq.Z}).before(unitary_ops)
        except Exception:
            return None
        coefficient = complex(pauli.coefficient)
        if abs(coefficient.imag) > 1e-8 or abs(abs(coefficient.real) - 1.0) > 1e-8:
            return None
        if coefficient.real < 0:
            constant ^= 1 << row
        for qubit, term in pauli.items():
            logical = input_qubit_map.get(qubit.x)
            if logical is None or logical < 0 or logical >= width:
                return None
            if term == cirq.X:
                random_columns[logical] ^= 1 << row
            elif term == cirq.Z:
                input_columns[logical] ^= 1 << row
            else:
                return None
    return AffineMeasurementChannel(
        width=width,
        random_columns=tuple(random_columns),
        input_columns=tuple(input_columns),
        constant=constant,
    )


def measured_affine_channels_equivalent(
    source: cirq.Circuit,
    compiled: cirq.Circuit,
    layout: dict[cirq.LineQubit, int],
) -> bool:
    source_channel = extract_affine_measurement_channel(source)
    inverse_layout = {physical: logical.x for logical, physical in layout.items()}
    compiled_channel = extract_affine_measurement_channel(compiled, input_qubit_map=inverse_layout)
    return channels_equivalent(source_channel, compiled_channel)


def channels_equivalent(
    left: AffineMeasurementChannel | None,
    right: AffineMeasurementChannel | None,
) -> bool:
    if left is None or right is None or left.width != right.width:
        return False
    left_random = _BitSpace(left.random_columns)
    right_random = _BitSpace(right.random_columns)
    if not left_random.same_space(right.random_columns):
        return False
    if not right_random.same_space(left.random_columns):
        return False
    if not left_random.contains(left.constant ^ right.constant):
        return False
    for left_column, right_column in zip(left.input_columns, right.input_columns):
        if not left_random.contains(left_column ^ right_column):
            return False
    return True


def affine_channel_diagnostics(circuit: cirq.Circuit) -> AffineChannelDiagnostics:
    channel = extract_affine_measurement_channel(circuit)
    if channel is None:
        return AffineChannelDiagnostics(False, len(circuit.all_qubits()), 0, 0, "unsupported", "not an affine X/Z measurement channel")
    random_rank = _BitSpace(channel.random_columns).rank
    input_rank = _BitSpace(channel.input_columns).rank
    return AffineChannelDiagnostics(
        supported=True,
        width=channel.width,
        random_rank=random_rank,
        input_rank=input_rank,
        channel_class=_classify_channel(channel),
    )


def _synthesize_simple_affine_logical(channel: AffineMeasurementChannel) -> cirq.Circuit | None:
    random_basis = _independent_columns(channel.random_columns)
    latent_sources = _choose_latent_sources(random_basis, channel)
    if latent_sources is None:
        return None
    random_space = _BitSpace(random_basis)
    for logical, input_column in enumerate(channel.input_columns):
        if logical in latent_sources:
            if not random_space.contains(input_column):
                return None
        else:
            if not random_space.contains(input_column ^ (1 << logical)):
                return None

    qubits = cirq.LineQubit.range(channel.width)
    operations: list[cirq.Operation] = []
    for basis_vector, source_row in zip(random_basis, latent_sources):
        operations.append(cirq.H(qubits[source_row]))
        for target in range(channel.width):
            if target != source_row and (basis_vector >> target) & 1:
                operations.append(cirq.CNOT(qubits[source_row], qubits[target]))
    for target in range(channel.width):
        if (channel.constant >> target) & 1:
            operations.append(cirq.X(qubits[target]))
    operations.append(cirq.measure(*qubits, key="z"))
    return cirq.Circuit(operations)


def _independent_columns(columns: tuple[int, ...]) -> tuple[int, ...]:
    basis: dict[int, int] = {}
    ordered: list[int] = []
    for column in sorted((item for item in columns if item), key=lambda item: (item.bit_count(), item)):
        reduced = column
        while reduced:
            pivot = reduced.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = reduced
                ordered.append(column)
                break
            reduced ^= basis[pivot]
    return tuple(ordered)


def _choose_latent_sources(
    basis: tuple[int, ...],
    channel: AffineMeasurementChannel,
) -> tuple[int, ...] | None:
    used: set[int] = set()
    sources: list[int] = []
    random_space = _BitSpace(basis)
    for vector in basis:
        options = [
            row
            for row in range(channel.width)
            if ((vector >> row) & 1) and row not in used and random_space.contains(channel.input_columns[row])
        ]
        if not options:
            return None
        source = min(options, key=lambda row: (channel.input_columns[row] != 0, row))
        used.add(source)
        sources.append(source)
    return tuple(sources)


class _BitSpace:
    def __init__(self, columns: tuple[int, ...] | list[int]) -> None:
        self.basis: dict[int, int] = {}
        self.rank = 0
        for column in columns:
            self._add(column)

    def contains(self, value: int) -> bool:
        current = value
        while current:
            pivot = current.bit_length() - 1
            if pivot not in self.basis:
                return False
            current ^= self.basis[pivot]
        return True

    def same_space(self, columns: tuple[int, ...] | list[int]) -> bool:
        other = _BitSpace(columns)
        if other.rank != self.rank:
            return False
        return all(self.contains(column) for column in columns)

    def _add(self, value: int) -> None:
        current = value
        while current:
            pivot = current.bit_length() - 1
            if pivot not in self.basis:
                self.basis[pivot] = current
                self.rank += 1
                return
            current ^= self.basis[pivot]


def _classify_channel(channel: AffineMeasurementChannel) -> str:
    random_rank = _BitSpace(channel.random_columns).rank
    if random_rank == 0:
        return "deterministic_affine"
    row_random_weight = [
        sum(1 for column in channel.random_columns if (column >> row) & 1)
        for row in range(channel.width)
    ]
    max_weight = max(row_random_weight, default=0)
    if max_weight <= 1:
        groups = sum(1 for column in channel.random_columns if column.bit_count() >= 3)
        if groups >= 2:
            return "parity_star_or_forest"
        return "single_parity_tree"
    if random_rank < channel.width // 2:
        return "shared_subexpression_parity"
    return "multi_row_affine_parity"


def _accept_general_candidate(
    candidate: CompilationCandidate,
    legacy: CompilationCandidate,
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> bool:
    candidate_value = candidate_metrics(candidate, backend, source)
    legacy_value = candidate_metrics(legacy, backend, source)
    if candidate_value.compile_time_sec > 12.0:
        return False
    return (
        candidate_value.two_qubit_count <= legacy_value.two_qubit_count - 8
        and candidate_value.depth <= legacy_value.depth
        and candidate_value.log_reliability >= legacy_value.log_reliability + 0.03
        and candidate_value.clifford_sensitivity_cost <= legacy_value.clifford_sensitivity_cost
        and count_two_qubit_gates(candidate.circuit) <= count_two_qubit_gates(source)
        and estimate_log_reliability(candidate.circuit, backend) > estimate_log_reliability(legacy.circuit, backend)
        and clifford_sensitivity_cost(candidate.circuit, backend) <= clifford_sensitivity_cost(legacy.circuit, backend)
    )


class BackendNoiseModelShim:
    def __init__(self, backend: CompetitionBackend) -> None:
        from .backend_model import BackendNoiseModel

        self._model = BackendNoiseModel.from_backend(backend)

    def __getattr__(self, name: str):
        return getattr(self._model, name)


def _low_readout_identity_layout(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
) -> dict[cirq.LineQubit, int] | None:
    qubits = tuple(sorted(circuit.all_qubits(), key=lambda qubit: qubit.x))
    if len(qubits) > backend.num_qubits:
        return None
    physical = sorted(range(backend.num_qubits), key=lambda qubit: (backend.readout(qubit), qubit))[: len(qubits)]
    return {logical: physical[index] for index, logical in enumerate(qubits)}
