from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import cirq

from noise_adaptive_hackathon.evaluator.validate_circuit import count_two_qubit_gates, validate_compiled_circuit
from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend, TRACK_A_BASIS
from noise_adaptive_hackathon.submission.backend_model import BackendNoiseModel
from noise_adaptive_hackathon.submission.clifford_resynthesis import (
    _remote_cnot_operations,
    cnot_block_matrix,
    resynthesize_cnot_blocks,
    synthesize_cnot_matrix,
)
from noise_adaptive_hackathon.submission.circuit_features import extract_circuit_features
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.compiler import _protect_legacy_parity
from noise_adaptive_hackathon.submission.ghz_tree import compile_exact_ghz_tree_fanout, detect_exact_ghz_star
from noise_adaptive_hackathon.submission.h_boundary_pathsum_resynth import (
    extract_pathsum_signature,
    find_h_boundary_cones,
    h_boundary_pathsum_diagnostics,
)
from noise_adaptive_hackathon.submission.layout import choose_initial_layout
from noise_adaptive_hackathon.submission.local_block_resynthesis import resynthesize_local_two_qubit_blocks
from noise_adaptive_hackathon.submission.measurement_channel import detect_affine_parity_star
from noise_adaptive_hackathon.submission.measurement_channel_placement import measurement_channel_placement_candidates
from noise_adaptive_hackathon.submission.measurement_channel_general import measured_affine_channels_equivalent
from noise_adaptive_hackathon.submission.measurement_tail_pruner import prune_measurement_tail
from noise_adaptive_hackathon.submission.parity_resynthesis import compress_logical_cnot_blocks
from noise_adaptive_hackathon.submission.peephole import conservative_peephole
from noise_adaptive_hackathon.submission.phase_affine_cone_resynth import (
    closed_dependency_cone,
    extract_phase_affine_signature,
    find_phase_affine_intervals,
    phase_affine_cone_diagnostics,
)
from noise_adaptive_hackathon.submission.relative_phase_toffoli import (
    final_z_measurement_equivalent,
    relative_phase_toffoli_rewrites,
)
from noise_adaptive_hackathon.submission.readout_copyout import readout_copyout_candidates
from noise_adaptive_hackathon.submission.relabel import optimize_physical_relabeling
from noise_adaptive_hackathon.submission.router import _append_bridge, _append_execute_and_swap, route_circuit
from noise_adaptive_hackathon.submission.route_scar_normalizer import (
    cx_commutes_with_operation,
    normalize_route_scars,
    route_scar_normalizer_candidates,
)
from noise_adaptive_hackathon.submission.selection import (
    CompilationCandidate,
    candidate_metrics,
    pareto_frontier,
    rank_candidates_by_proxy,
    select_candidate,
    terminal_readout_cost,
)
from noise_adaptive_hackathon.submission.simulation_selector import (
    SELECTOR_LOWSHOT,
    SELECTOR_PROXY_ONLY,
    SELECTOR_TRACK_A_RANKED_PROXY,
    SimulationSelectorConfig,
    select_candidate_with_simulation,
)
from noise_adaptive_hackathon.submission.track_a_exact_router import exact_small_router_candidates
from noise_adaptive_hackathon.submission.track_a_resynthesis.exact_equivalence import unitary_equivalent
from noise_adaptive_hackathon.submission.track_a_resynthesis.local_lower_bound_analyzer import analyze_local_lower_bounds
from noise_adaptive_hackathon.submission.track_a_resynthesis.bounded_window_synth import (
    bounded_window_synthesis_candidates,
    high_cost_windows,
)
from noise_adaptive_hackathon.submission.track_a_resynthesis.commutation_aware_redundancy import (
    commutation_aware_redundancy_candidates,
    find_redundancy_rewrites,
)
from noise_adaptive_hackathon.submission.track_a_resynthesis.cnot_block_gf2 import cnot_block_gf2_replacements
from noise_adaptive_hackathon.submission.track_a_resynthesis.exact_gf2_equivalence import cnot_networks_equivalent
from noise_adaptive_hackathon.submission.track_a_resynthesis.phase_polynomial_real import phase_polynomial_rewrite_candidates
from noise_adaptive_hackathon.submission.track_a_resynthesis.phase_polynomial import phase_polynomial_replacements
from noise_adaptive_hackathon.submission.track_a_resynthesis.reversible_templates import (
    reversible_template_replacements,
    toffoli_6cx,
)
from noise_adaptive_hackathon.submission.track_a_resynthesis.template_matcher import source_template_variants
from noise_adaptive_hackathon.submission.track_a_resynthesis.weighted_steiner_cnot import synthesize_weighted_cnot_block
from noise_adaptive_hackathon.submission.track_b import is_clifford_only


def line_backend(num_qubits: int = 5) -> CompetitionBackend:
    coupling_map = [(index, index + 1) for index in range(num_qubits - 1)]
    return CompetitionBackend.parse_obj(
        {
            "schema_version": "2.0",
            "snapshot_id": "line_test",
            "size_class": "small",
            "num_qubits": num_qubits,
            "basis_gates": list(TRACK_A_BASIS),
            "coupling_map": coupling_map,
            "two_qubit_error": {
                f"{left}-{right}": 0.006 + 0.01 * index
                for index, (left, right) in enumerate(coupling_map)
            },
            "readout_error": {str(index): 0.005 + 0.002 * index for index in range(num_qubits)},
            "single_qubit_error": 0.001,
        }
    )


def readout_skew_backend(two_qubit_error: float = 0.001) -> CompetitionBackend:
    return CompetitionBackend.parse_obj(
        {
            "schema_version": "2.0",
            "snapshot_id": "readout_skew",
            "size_class": "small",
            "num_qubits": 2,
            "basis_gates": list(TRACK_A_BASIS),
            "coupling_map": [(0, 1)],
            "two_qubit_error": {"0-1": two_qubit_error},
            "readout_error": {"0": 0.08, "1": 0.001},
            "single_qubit_error": 0.001,
        }
    )


def star_backend(width: int = 5) -> CompetitionBackend:
    center = 0
    coupling_map = [(center, node) for node in range(1, width)]
    return CompetitionBackend.parse_obj(
        {
            "schema_version": "2.0",
            "snapshot_id": "star_test",
            "size_class": "small",
            "num_qubits": width,
            "basis_gates": list(TRACK_A_BASIS),
            "coupling_map": coupling_map,
            "two_qubit_error": {f"{left}-{right}": 0.004 + 0.001 * right for left, right in coupling_map},
            "readout_error": {str(index): 0.004 + 0.001 * index for index in range(width)},
            "single_qubit_error": 0.001,
        }
    )


def test_backend_noise_model_precomputes_costs() -> None:
    backend = line_backend()
    model = BackendNoiseModel.from_backend(backend)

    assert model.is_coupled(0, 1)
    assert not model.is_coupled(0, 2)
    assert model.hop_distance(0, 4) == 4
    assert model.edge_cost(0, 1) < model.edge_cost(3, 4)
    assert model.noise_path(0, 4) == [0, 1, 2, 3, 4]


def test_exact_ghz_detector_requires_structural_fanout() -> None:
    q = cirq.LineQubit.range(5)
    ghz = cirq.Circuit(cirq.H(q[0]), [cirq.CNOT(q[0], qubit) for qubit in q[1:]], cirq.measure(*q, key="z"))
    almost_ghz = cirq.Circuit(cirq.H(q[0]), cirq.X(q[1]), [cirq.CNOT(q[0], qubit) for qubit in q[1:]], cirq.measure(*q, key="z"))

    detected = detect_exact_ghz_star(ghz)

    assert detected is not None
    assert detected.hub == q[0]
    assert detect_exact_ghz_star(almost_ghz) is None


def test_exact_ghz_tree_candidate_is_valid_and_clifford_safe() -> None:
    backend = star_backend(width=5)
    q = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(cirq.H(q[0]), [cirq.CNOT(q[0], qubit) for qubit in q[1:]], cirq.measure(*q, key="z"))

    result = compile_exact_ghz_tree_fanout(circuit, backend)

    assert result is not None
    compiled, layout = result
    assert set(layout) == set(q)
    assert is_clifford_only(compiled)
    assert count_two_qubit_gates(compiled) == 4
    assert validate_compiled_circuit(compiled, backend, "track_b").valid


def test_noise_aware_layout_is_unique_and_in_range() -> None:
    backend = line_backend()
    model = BackendNoiseModel.from_backend(backend)
    q = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.CNOT(q[0], q[3]), cirq.CNOT(q[0], q[1]), cirq.measure(*q, key="z"))
    features = extract_circuit_features(circuit)

    layout = choose_initial_layout(circuit, backend, model, features)

    assert set(layout) == set(q)
    assert len(set(layout.values())) == len(q)
    assert all(0 <= physical < backend.num_qubits for physical in layout.values())


def test_router_decomposes_movement_into_legal_cx() -> None:
    backend = line_backend()
    model = BackendNoiseModel.from_backend(backend)
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(q0, q1), cirq.measure(q0, q1, key="z"))
    features = extract_circuit_features(circuit)

    routed = route_circuit(circuit, backend, model, features, {q0: 0, q1: 4}).circuit

    assert all(not isinstance(op.gate, cirq.SwapPowGate) for op in routed.all_operations())
    assert count_two_qubit_gates(routed) > 1
    assert validate_compiled_circuit(routed, backend, "track_b").valid


def test_router_uses_execute_and_swap_when_lookahead_improves_layout() -> None:
    backend = line_backend(num_qubits=3)
    model = BackendNoiseModel.from_backend(backend)
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.CNOT(q[0], q[1]), cirq.CNOT(q[0], q[2]), cirq.measure(*q, key="z"))
    features = extract_circuit_features(circuit)

    routed = route_circuit(circuit, backend, model, features, {q[0]: 0, q[1]: 1, q[2]: 2})
    operations = list(routed.circuit.all_operations())

    assert operations[:3] == [
        cirq.CNOT(cirq.LineQubit(1), cirq.LineQubit(0)),
        cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)),
        cirq.CNOT(cirq.LineQubit(1), cirq.LineQubit(2)),
    ]
    assert routed.final_mapping == {0: 1, 1: 0, 2: 2}
    assert count_two_qubit_gates(routed.circuit) == 3
    assert validate_compiled_circuit(routed.circuit, backend, "track_b").valid


def test_terminal_remap_moves_measurement_when_readout_gain_pays_for_swap() -> None:
    backend = readout_skew_backend(two_qubit_error=0.001)
    model = BackendNoiseModel.from_backend(backend)
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key="z"))
    features = extract_circuit_features(circuit)

    routed = route_circuit(circuit, backend, model, features, {q0: 0})
    operations = list(routed.circuit.all_operations())

    assert sum(1 for op in operations if gate_name_for_operation(op) == "cx") == 3
    assert operations[-1] == cirq.measure(cirq.LineQubit(1), key="z")
    assert routed.final_mapping == {0: 1}
    assert validate_compiled_circuit(routed.circuit, backend, "track_b").valid


def test_terminal_remap_skips_when_added_cx_noise_exceeds_readout_gain() -> None:
    backend = readout_skew_backend(two_qubit_error=0.05)
    model = BackendNoiseModel.from_backend(backend)
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key="z"))
    features = extract_circuit_features(circuit)

    routed = route_circuit(circuit, backend, model, features, {q0: 0})

    assert sum(1 for op in routed.circuit.all_operations() if gate_name_for_operation(op) == "cx") == 0
    assert list(routed.circuit.all_operations())[-1] == cirq.measure(cirq.LineQubit(0), key="z")
    assert routed.final_mapping == {0: 0}


def test_readout_copyout_uses_clean_ancilla_path_when_reliability_improves() -> None:
    backend = CompetitionBackend.parse_obj(
        {
            "schema_version": "2.0",
            "snapshot_id": "copyout_line",
            "size_class": "small",
            "num_qubits": 3,
            "basis_gates": list(TRACK_A_BASIS),
            "coupling_map": [(0, 1), (1, 2)],
            "two_qubit_error": {"0-1": 0.001, "1-2": 0.001},
            "readout_error": {"0": 0.10, "1": 0.08, "2": 0.001},
            "single_qubit_error": 0.001,
        }
    )
    q0 = cirq.LineQubit(0)
    source = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key="z"))
    base = CompilationCandidate("base", source, {q0: 0}, 0.1)

    additions = readout_copyout_candidates(source, backend, [base])

    assert additions
    copied = additions[0].circuit
    assert count_two_qubit_gates(copied) == 2
    assert list(copied.all_operations())[-1] == cirq.measure(cirq.LineQubit(2), key="z")
    assert terminal_readout_cost(copied, backend) < terminal_readout_cost(source, backend)
    assert validate_compiled_circuit(copied, backend, "track_b").valid


def test_readout_copyout_refuses_dirty_internal_path() -> None:
    backend = CompetitionBackend.parse_obj(
        {
            "schema_version": "2.0",
            "snapshot_id": "copyout_dirty_path",
            "size_class": "small",
            "num_qubits": 3,
            "basis_gates": list(TRACK_A_BASIS),
            "coupling_map": [(0, 1), (1, 2)],
            "two_qubit_error": {"0-1": 0.001, "1-2": 0.001},
            "readout_error": {"0": 0.10, "1": 0.08, "2": 0.001},
            "single_qubit_error": 0.001,
        }
    )
    q0, q1 = cirq.LineQubit.range(2)
    source = cirq.Circuit(cirq.X(q1), cirq.X(q0), cirq.measure(q0, key="z"))
    base = CompilationCandidate("base", source, {q0: 0}, 0.1)

    assert readout_copyout_candidates(source, backend, [base]) == []


def test_tree_relabel_moves_tree_pattern_to_lower_noise_subgraph() -> None:
    backend = CompetitionBackend.parse_obj(
        {
            "schema_version": "2.0",
            "snapshot_id": "tree_relabel",
            "size_class": "small",
            "num_qubits": 4,
            "basis_gates": list(TRACK_A_BASIS),
            "coupling_map": [(0, 1), (1, 2), (2, 3)],
            "two_qubit_error": {"0-1": 0.050, "1-2": 0.002, "2-3": 0.002},
            "readout_error": {"0": 0.100, "1": 0.050, "2": 0.002, "3": 0.001},
            "single_qubit_error": 0.001,
        }
    )
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.CNOT(q0, q1), cirq.CNOT(q1, q2), cirq.measure(q0, q2, key="z"))

    optimized, layout = optimize_physical_relabeling(circuit, {q0: 0, q1: 1, q2: 2}, backend)

    assert validate_compiled_circuit(optimized, backend, "track_b").valid
    assert count_two_qubit_gates(optimized) == count_two_qubit_gates(circuit)
    assert candidate_metrics(CompilationCandidate("optimized", optimized, layout), backend, circuit).log_reliability > (
        candidate_metrics(CompilationCandidate("base", circuit, {q0: 0, q1: 1, q2: 2}), backend, circuit).log_reliability
    )
    assert 0 not in {qubit.x for op in optimized.all_operations() for qubit in op.qubits}


def test_execute_and_swap_matches_cx_then_logical_swap_on_basis_inputs() -> None:
    q = cirq.LineQubit.range(2)
    physical = tuple(cirq.LineQubit.range(2))
    fused_ops = cirq.Circuit()
    _append_execute_and_swap(fused_ops, physical, control=0, target=1)

    for bits in ((0, 0), (0, 1), (1, 0), (1, 1)):
        logical = cirq.Circuit(cirq.CNOT(q[0], q[1]), cirq.measure(q[0], q[1], key="z"))
        fused = cirq.Circuit(fused_ops, cirq.measure(physical[1], physical[0], key="z"))
        assert _run_basis(logical, bits) == _run_basis(fused, bits)


def test_bridge_cx_matches_remote_cx_and_preserves_middle_on_basis_inputs() -> None:
    q = cirq.LineQubit.range(3)
    physical = tuple(cirq.LineQubit.range(3))
    bridge_ops = cirq.Circuit()
    _append_bridge(bridge_ops, physical, control=0, middle=1, target=2)

    for value in range(8):
        bits = tuple((value >> shift) & 1 for shift in (2, 1, 0))
        logical = cirq.Circuit(cirq.CNOT(q[0], q[2]), cirq.measure(q[0], q[1], q[2], key="z"))
        bridged = cirq.Circuit(bridge_ops, cirq.measure(*physical, key="z"))
        assert _run_basis(logical, bits) == _run_basis(bridged, bits)


def test_remote_cnot_decomposition_preserves_mapping_on_longer_path() -> None:
    backend = line_backend(num_qubits=4)
    model = BackendNoiseModel.from_backend(backend)
    q = cirq.LineQubit.range(4)
    remote = cirq.Circuit(_remote_cnot_operations(0, 3, model, tuple(q)), cirq.measure(*q, key="z"))
    logical = cirq.Circuit(cirq.CNOT(q[0], q[3]), cirq.measure(*q, key="z"))

    for value in range(16):
        bits = tuple((value >> shift) & 1 for shift in (3, 2, 1, 0))
        assert _run_basis(remote, bits) == _run_basis(logical, bits)


def test_cnot_matrix_synthesis_reconstructs_linear_transform() -> None:
    q = cirq.LineQubit.range(4)
    block = [
        cirq.CNOT(q[0], q[2]),
        cirq.CNOT(q[2], q[3]),
        cirq.CNOT(q[1], q[3]),
        cirq.CNOT(q[0], q[1]),
    ]
    matrix = cnot_block_matrix(block, list(q))
    synthesized = synthesize_cnot_matrix(matrix)
    synthesized_block = [cirq.CNOT(q[control], q[target]) for control, target in synthesized]

    assert cnot_block_matrix(synthesized_block, list(q)) == matrix


def test_cnot_resynthesis_replaces_strictly_smaller_blocks_only() -> None:
    backend = line_backend(num_qubits=3)
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[0], q[1]),
        cirq.measure(*q, key="z"),
    )

    rewritten, changed = resynthesize_cnot_blocks(circuit, backend)

    assert changed
    assert sum(1 for op in rewritten.all_operations() if gate_name_for_operation(op) == "cx") == 0
    assert validate_compiled_circuit(rewritten, backend, "track_b").valid


def test_v13a_logical_gf2_compresses_redundant_cnot_block() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.CNOT(q0, q1),
        cirq.CNOT(q0, q1),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key="z"),
    )

    rewritten, changed = compress_logical_cnot_blocks(circuit)

    assert changed
    assert sum(1 for op in rewritten.all_operations() if gate_name_for_operation(op) == "cx") == 1


def test_measurement_channel_detects_affine_parity_star() -> None:
    q = cirq.LineQubit.range(16)
    hubs = (0, 4, 8, 12)
    circuit = cirq.Circuit(cirq.H(q[hub]) for hub in hubs)
    for hub in hubs:
        for leaf in range(hub + 1, hub + 4):
            circuit.append(cirq.CNOT(q[hub], q[leaf]))
    circuit.append(cirq.X(q[3]))
    circuit.append(cirq.X(q[14]))
    circuit.append(cirq.measure(*q, key="z"))

    star = detect_affine_parity_star(circuit)

    assert star is not None
    assert star.hubs == hubs
    assert len(star.leaf_assignments) == 12
    assert (0, 3, 1) in star.leaf_assignments
    assert (12, 14, 1) in star.leaf_assignments


def test_measurement_channel_equivalence_accepts_remapped_star() -> None:
    q = cirq.LineQubit.range(4)
    p = cirq.LineQubit.range(10, 14)
    source = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.CNOT(q[0], q[2]), cirq.X(q[3]), cirq.measure(*q, key="z"))
    compiled = cirq.Circuit(cirq.H(p[0]), cirq.CNOT(p[0], p[1]), cirq.CNOT(p[0], p[2]), cirq.X(p[3]), cirq.measure(*p, key="z"))
    layout = {q[index]: p[index].x for index in range(4)}

    assert measured_affine_channels_equivalent(source, compiled, layout)


def test_measurement_channel_equivalence_rejects_wrong_order_and_sign() -> None:
    q = cirq.LineQubit.range(4)
    source = cirq.Circuit(cirq.X(q[0]), cirq.measure(*q, key="z"))
    wrong_order = cirq.Circuit(cirq.X(q[0]), cirq.measure(q[1], q[0], q[2], q[3], key="z"))
    wrong_sign = cirq.Circuit(cirq.measure(*q, key="z"))
    layout = {qubit: qubit.x for qubit in q}

    assert not measured_affine_channels_equivalent(source, wrong_order, layout)
    assert not measured_affine_channels_equivalent(source, wrong_sign, layout)


def test_measurement_channel_equivalence_rejects_non_layout_dependence() -> None:
    q = cirq.LineQubit.range(2)
    p0, p1, ancilla = cirq.LineQubit.range(3)
    source = cirq.Circuit(cirq.measure(*q, key="z"))
    compiled = cirq.Circuit(cirq.CNOT(ancilla, p0), cirq.measure(p0, p1, key="z"))
    layout = {q[0]: p0.x, q[1]: p1.x}

    assert not measured_affine_channels_equivalent(source, compiled, layout)


def test_measurement_channel_placement_preserves_cx_depth_and_improves_metrics() -> None:
    backend = CompetitionBackend.parse_obj(
        {
            "schema_version": "2.0",
            "snapshot_id": "placement_star",
            "size_class": "medium",
            "num_qubits": 6,
            "basis_gates": list(TRACK_A_BASIS),
            "coupling_map": [(0, 1), (0, 2), (3, 4), (3, 5), (1, 2), (4, 5)],
            "two_qubit_error": {
                "0-1": 0.050,
                "0-2": 0.050,
                "3-4": 0.002,
                "3-5": 0.002,
                "1-2": 0.040,
                "4-5": 0.003,
            },
            "readout_error": {str(index): (0.050 if index < 3 else 0.002) for index in range(6)},
            "single_qubit_error": 0.001,
        }
    )
    q = cirq.LineQubit.range(3)
    source = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.CNOT(q[0], q[2]), cirq.measure(*q, key="z"))
    p = cirq.LineQubit.range(3)
    base = CompilationCandidate(
        "v13c_measurement_parity_star",
        cirq.Circuit(cirq.H(p[0]), cirq.CNOT(p[0], p[1]), cirq.CNOT(p[0], p[2]), cirq.measure(*p, key="z")),
        {q[0]: 0, q[1]: 1, q[2]: 2},
        0.1,
    )

    additions = measurement_channel_placement_candidates(source, backend, [base])

    assert additions
    placed = additions[0]
    assert count_two_qubit_gates(placed.circuit) == count_two_qubit_gates(base.circuit)
    assert len(placed.circuit) == len(base.circuit)
    assert candidate_metrics(placed, backend, source).log_reliability > candidate_metrics(base, backend, source).log_reliability
    assert measured_affine_channels_equivalent(source, placed.circuit, placed.layout)


def test_legacy_protection_allows_strict_measurement_channel_placement_gain() -> None:
    backend = CompetitionBackend.parse_obj(
        {
            "schema_version": "2.0",
            "snapshot_id": "placement_protect",
            "size_class": "medium",
            "num_qubits": 6,
            "basis_gates": list(TRACK_A_BASIS),
            "coupling_map": [(0, 1), (0, 2), (3, 4), (3, 5), (1, 2), (4, 5)],
            "two_qubit_error": {
                "0-1": 0.050,
                "0-2": 0.050,
                "3-4": 0.002,
                "3-5": 0.002,
                "1-2": 0.040,
                "4-5": 0.003,
            },
            "readout_error": {str(index): (0.050 if index < 3 else 0.002) for index in range(6)},
            "single_qubit_error": 0.001,
        }
    )
    q = cirq.LineQubit.range(3)
    source = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.CNOT(q[0], q[2]), cirq.measure(*q, key="z"))
    legacy = CompilationCandidate(
        "legacy_hub_parity_seed100",
        cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.CNOT(q[0], q[2]), cirq.measure(*q, key="z")),
        {q[0]: 0, q[1]: 1, q[2]: 2},
        0.1,
    )
    placed_qubits = [cirq.LineQubit(index) for index in (3, 4, 5)]
    placed = CompilationCandidate(
        "v13c_measurement_parity_star_mc_placement",
        cirq.Circuit(
            cirq.H(placed_qubits[0]),
            cirq.CNOT(placed_qubits[0], placed_qubits[1]),
            cirq.CNOT(placed_qubits[0], placed_qubits[2]),
            cirq.measure(*placed_qubits, key="z"),
        ),
        {q[0]: 3, q[1]: 4, q[2]: 5},
        0.2,
    )

    selected = _protect_legacy_parity(placed, [legacy, placed], backend, source)

    assert selected is placed


def test_local_two_qubit_resynthesis_reduces_exact_track_a_block() -> None:
    backend = line_backend(num_qubits=2)
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.CNOT(q[0], q[1]),
        cirq.Z(q[0]),
        cirq.CNOT(q[0], q[1]),
        cirq.measure(*q, key="z"),
    )

    rewritten, changed = resynthesize_local_two_qubit_blocks(circuit, backend)
    source_unitary = cirq.Circuit(op for op in circuit.all_operations() if gate_name_for_operation(op) != "measure").unitary(qubit_order=q)
    rewritten_unitary = cirq.Circuit(op for op in rewritten.all_operations() if gate_name_for_operation(op) != "measure").unitary(qubit_order=q)

    assert changed
    assert sum(1 for op in rewritten.all_operations() if gate_name_for_operation(op) == "cx") == 0
    cirq.testing.assert_allclose_up_to_global_phase(source_unitary, rewritten_unitary, atol=1e-8)
    assert validate_compiled_circuit(rewritten, backend, "track_a").valid


def test_exact_small_router_skips_low_pressure_track_a_circuits() -> None:
    backend = line_backend(num_qubits=8)
    q = cirq.LineQubit.range(6)
    circuit = cirq.Circuit(cirq.T(q[0]), cirq.CNOT(q[0], q[1]), cirq.CNOT(q[2], q[3]), cirq.measure(*q, key="z"))

    assert exact_small_router_candidates(circuit, backend, None) == []


def test_reversible_template_generates_equivalent_toffoli_variants() -> None:
    q = cirq.LineQubit.range(3)
    source = tuple(toffoli_6cx(q[0], q[1], q[2]))

    replacements = reversible_template_replacements(source, tuple(q))

    assert replacements
    assert all(unitary_equivalent(list(source), replacement, tuple(q)) for replacement in replacements)


def test_phase_polynomial_generates_equivalent_parity_gadget_variant() -> None:
    q = cirq.LineQubit.range(3)
    source = (
        cirq.CNOT(q[0], q[2]),
        cirq.CNOT(q[1], q[2]),
        cirq.T(q[2]),
        cirq.CNOT(q[1], q[2]),
        cirq.CNOT(q[0], q[2]),
    )

    replacements = phase_polynomial_replacements(source, tuple(q))

    assert replacements
    assert all(unitary_equivalent(list(source), replacement, tuple(q)) for replacement in replacements)


def test_source_template_matcher_finds_toffoli_scaffold_variant() -> None:
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(*toffoli_6cx(q[0], q[1], q[2]), cirq.measure(*q, key="z"))

    variants = source_template_variants(circuit, limit=4)

    assert variants
    assert all(
        unitary_equivalent(
            list(cirq.Circuit(op for op in circuit.all_operations() if not isinstance(op.gate, cirq.MeasurementGate)).all_operations()),
            list(cirq.Circuit(op for op in variant.circuit.all_operations() if not isinstance(op.gate, cirq.MeasurementGate)).all_operations()),
            tuple(q),
        )
        for variant in variants
    )


def test_v15_cnot_block_gf2_compresses_redundant_linear_block() -> None:
    q = cirq.LineQubit.range(3)
    operations = [
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[0], q[1]),
        cirq.measure(*q, key="z"),
    ]

    replacements = cnot_block_gf2_replacements(operations)

    assert replacements
    replacement = replacements[0]
    assert len(replacement.operations) == 1
    assert unitary_equivalent(operations[:3], replacement.operations, tuple(q[:2]))


def test_exact_gf2_equivalence_rejects_altered_cnot_network() -> None:
    q = cirq.LineQubit.range(3)
    left = [cirq.CNOT(q[0], q[1]), cirq.CNOT(q[1], q[2])]
    same = [cirq.CNOT(q[0], q[1]), cirq.CNOT(q[1], q[2])]
    altered = [cirq.CNOT(q[1], q[2]), cirq.CNOT(q[0], q[1])]

    assert cnot_networks_equivalent(left, same, tuple(q))
    assert not cnot_networks_equivalent(left, altered, tuple(q))


def test_weighted_cnot_synthesis_emits_legal_equivalent_network() -> None:
    backend = line_backend(num_qubits=3)
    q = cirq.LineQubit.range(3)
    block = [
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[0], q[1]),
    ]

    result = synthesize_weighted_cnot_block(block, backend, max_seconds=1.0)

    assert result is not None
    assert len(result.operations) == 1
    assert all(validate_compiled_circuit(cirq.Circuit(*result.operations, cirq.measure(*q, key="z")), backend, "track_a").valid for _ in [0])
    assert cnot_networks_equivalent(block, result.operations, tuple(q[:2]))


def test_v15_phase_polynomial_combines_equal_adjacent_gadgets() -> None:
    q = cirq.LineQubit.range(3)
    gadget = [
        cirq.CNOT(q[0], q[2]),
        cirq.CNOT(q[1], q[2]),
        cirq.T(q[2]),
        cirq.CNOT(q[1], q[2]),
        cirq.CNOT(q[0], q[2]),
    ]
    operations = [*gadget, *gadget, cirq.measure(*q, key="z")]

    replacements = phase_polynomial_rewrite_candidates(operations)

    assert replacements
    replacement = replacements[0]
    assert len(replacement.operations) < 2 * len(gadget)
    assert unitary_equivalent(operations[:10], replacement.operations, tuple(q))


def test_bounded_window_synth_accepts_exact_redundant_window_candidate() -> None:
    backend = line_backend(num_qubits=3)
    q = cirq.LineQubit.range(3)
    source = cirq.Circuit(
        cirq.H(q[0]),
        cirq.CNOT(q[0], q[1]),
        cirq.X(q[2]),
        cirq.CNOT(q[0], q[1]),
        cirq.T(q[2]),
        cirq.measure(*q, key="z"),
    )
    base = CompilationCandidate("base", source, {qubit: qubit.x for qubit in q}, 0.1)

    windows = high_cost_windows(source, backend)
    additions = bounded_window_synthesis_candidates(source, backend, [base], max_seconds=2.0)

    assert windows
    assert additions
    assert count_two_qubit_gates(additions[0].circuit) < count_two_qubit_gates(source)
    assert validate_compiled_circuit(additions[0].circuit, backend, "track_a").valid


def test_commutation_redundancy_removes_non_adjacent_redundant_cx_pair() -> None:
    backend = line_backend(num_qubits=3)
    q = cirq.LineQubit.range(3)
    source = cirq.Circuit(
        cirq.CNOT(q[0], q[1]),
        cirq.T(q[2]),
        cirq.CNOT(q[0], q[1]),
        cirq.measure(*q, key="z"),
    )
    base = CompilationCandidate("base", source, {qubit: qubit.x for qubit in q}, 0.1)

    additions = commutation_aware_redundancy_candidates(source, backend, [base], max_seconds=1.0)

    assert additions
    assert count_two_qubit_gates(additions[0].circuit) == 0
    assert validate_compiled_circuit(additions[0].circuit, backend, "track_a").valid


def test_commutation_redundancy_keeps_noncommuting_cx_pair() -> None:
    backend = line_backend(num_qubits=3)
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.CNOT(q[0], q[1]),
        cirq.H(q[0]),
        cirq.CNOT(q[0], q[1]),
        cirq.T(q[2]),
        cirq.measure(*q, key="z"),
    )

    rewrites = find_redundancy_rewrites(circuit, backend, max_seconds=1.0)

    assert rewrites == []


def test_commutation_redundancy_filters_illegal_replacement_edges_before_candidate() -> None:
    backend = line_backend(num_qubits=3)
    q = cirq.LineQubit.range(3)
    source = cirq.Circuit(
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[0], q[2]),
        cirq.T(q[2]),
        cirq.CNOT(q[0], q[1]),
        cirq.measure(*q, key="z"),
    )

    rewrites = find_redundancy_rewrites(source, backend, max_seconds=1.0)

    assert rewrites == []


def test_route_scar_normalizer_cancels_only_provably_commuting_same_cx() -> None:
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.CNOT(q[0], q[1]),
        cirq.T(q[0]),
        cirq.X(q[2]),
        cirq.CNOT(q[0], q[1]),
        cirq.measure(*q, key="z"),
    )

    result = normalize_route_scars(circuit, enable_ladder_validation=False)

    assert result.changed
    names = [gate_name_for_operation(op) for op in result.circuit.all_operations()]
    assert names.count("cx") == 0
    assert sorted(names) == ["measure", "t", "x"]


def test_route_scar_normalizer_refuses_uncertain_single_qubit_crossing() -> None:
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.CNOT(q[0], q[1]),
        cirq.H(q[1]),
        cirq.CNOT(q[0], q[1]),
        cirq.measure(*q, key="z"),
    )

    result = normalize_route_scars(circuit, enable_ladder_validation=False)

    assert not result.changed
    assert sum(1 for op in result.circuit.all_operations() if gate_name_for_operation(op) == "cx") == 2


def test_route_scar_cnot_commutation_condition_matches_dependency_rule() -> None:
    q = cirq.LineQubit.range(4)

    assert cx_commutes_with_operation(q[0], q[1], cirq.CNOT(q[2], q[3]))
    assert cx_commutes_with_operation(q[0], q[1], cirq.CNOT(q[0], q[2]))
    assert not cx_commutes_with_operation(q[0], q[1], cirq.CNOT(q[1], q[2]))
    assert not cx_commutes_with_operation(q[0], q[1], cirq.CNOT(q[2], q[0]))


def test_route_scar_ladder_guard_accepts_exact_validated_compute_uncompute() -> None:
    q = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        cirq.CNOT(q[1], q[2]),
        cirq.CNOT(q[0], q[1]),
        cirq.H(q[2]),
        cirq.T(q[0]),
        cirq.T(q[1]) ** -1,
        cirq.CNOT(q[2], q[3]),
        cirq.CNOT(q[0], q[1]),
        cirq.H(q[2]),
        cirq.CNOT(q[1], q[2]),
        cirq.measure(*q, key="z"),
    )

    result = normalize_route_scars(circuit)

    assert result.changed
    assert result.rewrites[0].kind == "ladder_compute_uncompute"
    assert result.rewrites[0].exact_validated
    assert sum(1 for op in result.circuit.all_operations() if gate_name_for_operation(op) == "cx") == 3


def test_route_scar_candidate_reproduces_redundant_ladder_metrics() -> None:
    backend = line_backend(num_qubits=4)
    q = cirq.LineQubit.range(4)
    source = cirq.Circuit(
        cirq.H(q[2]),
        cirq.T(q[0]),
        cirq.T(q[1]) ** -1,
        cirq.CNOT(q[2], q[3]),
        cirq.CNOT(q[0], q[1]),
        cirq.H(q[2]),
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[1], q[2]),
        cirq.CNOT(q[1], q[2]),
        cirq.measure(*q, key="z"),
    )
    routed = cirq.Circuit(
        cirq.CNOT(q[1], q[2]),
        cirq.CNOT(q[0], q[1]),
        cirq.H(q[2]),
        cirq.T(q[0]),
        cirq.T(q[1]) ** -1,
        cirq.CNOT(q[2], q[3]),
        cirq.CNOT(q[0], q[1]),
        cirq.H(q[2]),
        cirq.CNOT(q[1], q[2]),
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[1], q[2]),
        cirq.CNOT(q[1], q[2]),
        cirq.CNOT(q[2], q[3]),
        cirq.CNOT(q[2], q[3]),
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[0], q[1]),
        cirq.measure(*q, key="z"),
    )
    base = CompilationCandidate("base", routed, {qubit: qubit.x for qubit in q}, 0.1)

    additions = route_scar_normalizer_candidates(source, backend, [base])

    assert additions
    assert count_two_qubit_gates(additions[0].circuit) <= count_two_qubit_gates(routed) - 2
    assert validate_compiled_circuit(additions[0].circuit, backend, "track_a").valid


def test_local_lower_bound_analyzer_certifies_redundant_cnot_identity() -> None:
    backend = line_backend(num_qubits=3)
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[1], q[2]),
        cirq.CNOT(q[1], q[2]),
        cirq.measure(*q, key="z"),
    )

    rows = analyze_local_lower_bounds(circuit, backend, limit=4)

    assert rows
    assert any(row.exact_minimal_cx == 0 for row in rows)
    assert any(row.possible_cx_reduction for row in rows)


def test_closed_dependency_cone_recomputes_qubits_after_interval_expands() -> None:
    q = cirq.LineQubit.range(4)
    operations = [
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[1], q[2]),
        cirq.CNOT(q[0], q[3]),
    ]

    start, stop, qubits = closed_dependency_cone(operations, 2, 3)

    assert (start, stop) == (0, 3)
    assert [qubit.x for qubit in qubits] == [0, 1, 2, 3]


def test_phase_affine_signature_tracks_affine_rows_and_phase_terms() -> None:
    q = cirq.LineQubit.range(2)
    operations = (
        cirq.T(q[0]),
        cirq.CNOT(q[0], q[1]),
        cirq.T(q[1]) ** -1,
        cirq.CNOT(q[0], q[1]),
    )

    signature = extract_phase_affine_signature(operations, tuple(q))

    assert signature is not None
    assert signature.rows == (0b01, 0b10)
    assert signature.const == (0, 0)
    assert dict(signature.phase_terms) == {0b01: 1, 0b11: 7}


def test_phase_affine_diagnostics_finds_exact_hybrid_interval() -> None:
    backend = line_backend(num_qubits=4)
    q = cirq.LineQubit.range(3)
    source = cirq.Circuit(
        cirq.T(q[1]),
        cirq.T(q[2]),
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[0], q[2]),
        cirq.T(q[2]) ** -1,
        cirq.CNOT(q[1], q[2]),
        cirq.T(q[2]) ** -1,
        cirq.CNOT(q[0], q[2]),
        cirq.T(q[2]),
        cirq.CNOT(q[1], q[2]),
        cirq.measure(*q, key="z"),
    )
    compiler = NoiseAdaptiveCompiler(enable_parity_resynthesis=True)
    base_candidates = compiler.candidate_compilations(source, backend)

    rows = phase_affine_cone_diagnostics(source, backend, base_candidates, max_seconds=2.0)

    assert any(row.get("exact_equivalent") for row in rows)
    assert any(str(row.get("method", "")).startswith("hybrid_parity_reuse") for row in rows)


def test_pathsum_signature_introduces_path_variable_for_h() -> None:
    q = cirq.LineQubit.range(2)
    operations = (
        cirq.H(q[0]),
        cirq.CNOT(q[0], q[1]),
        cirq.T(q[1]),
    )

    signature = extract_pathsum_signature(operations, tuple(q))

    assert signature is not None
    assert signature.num_inputs == 2
    assert signature.num_paths == 1
    assert signature.h_count == 1
    assert signature.output_input_rows == (0b00, 0b10)
    assert signature.output_path_rows == (0b1, 0b1)
    assert any(term.kind == "hadamard_kernel" for term in signature.phase_terms)


def test_h_boundary_cone_detector_finds_h_shell_core() -> None:
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H(q[2]),
        cirq.CNOT(q[0], q[2]),
        cirq.T(q[2]),
        cirq.CNOT(q[1], q[2]),
        cirq.T(q[2]) ** -1,
        cirq.CNOT(q[0], q[2]),
        cirq.CNOT(q[1], q[2]),
        cirq.CNOT(q[0], q[1]),
        cirq.H(q[2]),
        cirq.measure(*q, key="z"),
    )

    cones = find_h_boundary_cones(list(circuit.all_operations()), min_cx=4)

    assert cones
    assert cones[0].signature.num_paths == 2
    assert cones[0].phase_affine_intervals


def test_h_boundary_pathsum_diagnostics_reports_exact_replacements() -> None:
    backend = line_backend(num_qubits=4)
    q = cirq.LineQubit.range(3)
    source = cirq.Circuit(
        cirq.H(q[2]),
        cirq.T(q[1]),
        cirq.T(q[2]),
        cirq.CNOT(q[0], q[1]),
        cirq.CNOT(q[0], q[2]),
        cirq.T(q[2]) ** -1,
        cirq.CNOT(q[1], q[2]),
        cirq.T(q[2]) ** -1,
        cirq.CNOT(q[0], q[2]),
        cirq.T(q[2]),
        cirq.CNOT(q[1], q[2]),
        cirq.H(q[2]),
        cirq.measure(*q, key="z"),
    )
    compiler = NoiseAdaptiveCompiler(enable_parity_resynthesis=True)
    base_candidates = compiler.candidate_compilations(source, backend)

    rows = h_boundary_pathsum_diagnostics(source, backend, base_candidates, max_seconds=2.0)

    assert any(row.get("exact_equivalent") for row in rows)
    assert any(row.get("num_path_variables") == 2 for row in rows)


def test_measurement_tail_pruner_removes_track_a_diagonal_tail() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.T(q0),
        cirq.S(q1) ** -1,
        cirq.measure(q0, q1, key="z"),
    )

    result = prune_measurement_tail(circuit, clifford_only=False)

    assert result.changed
    assert sorted(gate_name_for_operation(op) for op in result.removed_operations) == ["sdg", "t"]
    assert list(result.circuit.all_operations()) == [cirq.H(q0), cirq.measure(q0, q1, key="z")]


def test_measurement_tail_pruner_track_b_keeps_non_clifford_tail() -> None:
    (q0,) = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.T(q0), cirq.S(q0), cirq.measure(q0, key="z"))

    result = prune_measurement_tail(circuit, clifford_only=True)

    assert [gate_name_for_operation(op) for op in result.removed_operations] == ["s"]
    assert [gate_name_for_operation(op) for op in result.circuit.all_operations()] == ["t", "measure"]


def test_measurement_tail_pruner_stops_at_non_commuting_boundary() -> None:
    (q0,) = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.Z(q0), cirq.X(q0), cirq.S(q0), cirq.measure(q0, key="z"))

    result = prune_measurement_tail(circuit, clifford_only=False)

    assert [gate_name_for_operation(op) for op in result.removed_operations] == ["s"]
    assert [gate_name_for_operation(op) for op in result.circuit.all_operations()] == ["z", "x", "measure"]


def test_relative_phase_toffoli_rewrite_preserves_final_z_probabilities() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit([*toffoli_6cx(q0, q1, q2), cirq.measure(q0, q1, q2, key="z")])

    rewrites = relative_phase_toffoli_rewrites(circuit)

    assert rewrites
    best = rewrites[0]
    assert best.old_cx == 6
    assert best.new_cx == 3
    assert final_z_measurement_equivalent(circuit, best.circuit)


def test_relative_phase_toffoli_rewrite_rejects_phase_exposing_h() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit([*toffoli_6cx(q0, q1, q2), cirq.H(q2), cirq.measure(q0, q1, q2, key="z")])

    assert relative_phase_toffoli_rewrites(circuit) == []


def test_terminal_readout_cost_uses_final_measurement_qubits() -> None:
    backend = readout_skew_backend()
    q0, q1 = cirq.LineQubit.range(2)
    high_readout = cirq.Circuit(cirq.measure(q0, key="z"))
    low_readout = cirq.Circuit(cirq.measure(q1, key="z"))

    assert terminal_readout_cost(low_readout, backend) < terminal_readout_cost(high_readout, backend)


def test_pareto_frontier_drops_dominated_candidate() -> None:
    backend = readout_skew_backend()
    q0, q1 = cirq.LineQubit.range(2)
    source = cirq.Circuit(cirq.measure(q0, key="z"))
    dominated = CompilationCandidate(
        "dominated",
        cirq.Circuit(cirq.CNOT(q0, q1), cirq.measure(q0, key="z")),
        {q0: 0},
        0.2,
    )
    better = CompilationCandidate("better", cirq.Circuit(cirq.measure(q1, key="z")), {q0: 1}, 0.1)

    frontier = pareto_frontier([dominated, better], backend, source)

    assert frontier == [better]


def test_selector_prefers_lower_readout_cost_when_other_metrics_match() -> None:
    backend = readout_skew_backend()
    q0, q1 = cirq.LineQubit.range(2)
    source = cirq.Circuit(cirq.measure(q0, key="z"))
    high_readout = CompilationCandidate("high", cirq.Circuit(cirq.measure(q0, key="z")), {q0: 0}, 0.1)
    low_readout = CompilationCandidate("low", cirq.Circuit(cirq.measure(q1, key="z")), {q0: 1}, 0.1)

    selected = select_candidate([high_readout, low_readout], backend, source)

    assert selected is low_readout


def test_simulation_selector_proxy_only_matches_proxy_selector() -> None:
    backend = readout_skew_backend()
    q0, q1 = cirq.LineQubit.range(2)
    source = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key="z"))
    high_readout = CompilationCandidate("high", cirq.Circuit(cirq.X(q0), cirq.measure(q0, key="z")), {q0: 0}, 0.1)
    low_readout = CompilationCandidate("low", cirq.Circuit(cirq.X(q1), cirq.measure(q1, key="z")), {q0: 1}, 0.1)

    selected = select_candidate_with_simulation(
        [high_readout, low_readout],
        backend,
        source,
        seed=7,
        config=SimulationSelectorConfig(mode=SELECTOR_PROXY_ONLY),
    )

    assert selected is select_candidate([high_readout, low_readout], backend, source)


def test_simulation_selector_track_a_ranked_proxy_returns_top_proxy_candidate() -> None:
    backend = readout_skew_backend()
    q0, q1 = cirq.LineQubit.range(2)
    source = cirq.Circuit(cirq.T(q0), cirq.measure(q0, key="z"))
    high_readout = CompilationCandidate("high", cirq.Circuit(cirq.T(q0), cirq.measure(q0, key="z")), {q0: 0}, 0.1)
    low_readout = CompilationCandidate("low", cirq.Circuit(cirq.T(q1), cirq.measure(q1, key="z")), {q0: 1}, 0.1)
    candidates = [high_readout, low_readout]

    selected = select_candidate_with_simulation(
        candidates,
        backend,
        source,
        seed=7,
        config=SimulationSelectorConfig(mode=SELECTOR_TRACK_A_RANKED_PROXY),
    )

    assert selected is rank_candidates_by_proxy(candidates, backend, source)[0]


def test_simulation_selector_uses_noisy_support_without_expected_rule() -> None:
    backend = readout_skew_backend()
    q0, q1 = cirq.LineQubit.range(2)
    source = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key="z"))
    high_readout = CompilationCandidate("high", cirq.Circuit(cirq.X(q0), cirq.measure(q0, key="z")), {q0: 0}, 0.1)
    low_readout = CompilationCandidate("low", cirq.Circuit(cirq.X(q1), cirq.measure(q1, key="z")), {q0: 1}, 0.1)

    selected = select_candidate_with_simulation(
        [high_readout, low_readout],
        backend,
        source,
        seed=7,
        config=SimulationSelectorConfig(mode=SELECTOR_LOWSHOT, track_a_lowshot=128, track_a_timeout_sec=1.0),
    )

    assert selected is low_readout


def test_peephole_only_removes_safe_adjacent_pairs() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.H(q0),
        cirq.S(q0),
        cirq.S(q0) ** -1,
        cirq.CNOT(q0, q1),
        cirq.CNOT(q0, q1),
        cirq.X(q1),
        cirq.measure(q0, q1, key="z"),
    )

    simplified = conservative_peephole(circuit, clifford_only=True)

    assert [op for op in simplified.all_operations() if not isinstance(op.gate, cirq.MeasurementGate)] == [cirq.X(q1)]


def test_peephole_cancels_identical_cx_across_disjoint_operations() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.CNOT(q0, q1),
        cirq.X(q2),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, q2, key="z"),
    )

    simplified = conservative_peephole(circuit, clifford_only=True)

    assert list(simplified.all_operations()) == [cirq.X(q2), cirq.measure(q0, q1, q2, key="z")]


def test_peephole_keeps_cx_when_intervening_operation_touches_pair() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.CNOT(q0, q1),
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key="z"),
    )

    simplified = conservative_peephole(circuit, clifford_only=True)

    assert sum(1 for op in simplified.all_operations() if gate_name_for_operation(op) == "cx") == 2


def test_submission_compiler_keeps_clifford_outputs_valid() -> None:
    backend = line_backend()
    q = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[3]), cirq.measure(*q, key="z"))

    compiler = NoiseAdaptiveCompiler()
    compiled = compiler.compile(circuit, backend)

    assert is_clifford_only(compiled)
    assert validate_compiled_circuit(compiled, backend, "track_b").valid


def test_root_cli_writes_cirq_json_and_layout_sidecar(tmp_path: Path) -> None:
    backend = line_backend()
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.X(q[0]), cirq.CNOT(q[0], q[2]), cirq.measure(*q, key="z"))
    circuit_path = tmp_path / "circuit.json"
    backend_path = tmp_path / "backend.json"
    out_path = tmp_path / "compiled.json"
    cirq.to_json(circuit, circuit_path)
    backend_path.write_text(backend.json(), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(Path.cwd() / "compiler.py"),
            "--circuit",
            str(circuit_path),
            "--backend",
            str(backend_path),
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=tmp_path,
    )

    compiled = cirq.read_json(out_path)
    sidecar = json.loads(out_path.with_name("compiled.layout.json").read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == "1.0"
    assert set(sidecar["initial_layout"]) == {"0", "1", "2"}
    assert validate_compiled_circuit(compiled, backend, "track_b").valid


def _run_basis(circuit: cirq.Circuit, bits: tuple[int, ...]) -> tuple[int, ...]:
    qubits = sorted(circuit.all_qubits(), key=lambda qubit: qubit.x)
    prepared = cirq.Circuit(cirq.X(qubit) for qubit, bit in zip(qubits, bits) if bit)
    prepared += circuit
    result = cirq.Simulator(seed=123).run(prepared, repetitions=1)
    return tuple(int(bit) for bit in result.measurements["z"][0])
