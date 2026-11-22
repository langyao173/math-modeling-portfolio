from __future__ import annotations

import cirq
import networkx as nx
import time

from noise_adaptive_hackathon.baseline.cirq_cqc import CirqCQCCompiler
from noise_adaptive_hackathon.baseline.common import decompose_swaps
from noise_adaptive_hackathon.baseline.movement_aware import MovementAwareCompiler
from noise_adaptive_hackathon.baseline.paper_greedy import PaperGreedyEdgeCompiler, PaperGreedyVertexCompiler
from noise_adaptive_hackathon.baseline.reliability_greedy import ReliabilityGreedyCompiler
from noise_adaptive_hackathon.evaluator.validate_circuit import count_two_qubit_gates, estimate_log_reliability, gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .backend_model import BackendNoiseModel
from .clifford_resynthesis import resynthesize_cnot_blocks
from .circuit_features import extract_circuit_features
from .ghz_tree import compile_exact_ghz_tree_fanout
from .ghz_leaf_tree import ghz_leaf_tree_candidates
from .h_boundary_pathsum_resynth import h_boundary_pathsum_candidates
from .legacy_parity import compile_legacy_hub_parity
from .layout import candidate_initial_layouts, choose_initial_layout, hub_aware_layout
from .local_block_resynthesis import resynthesize_local_two_qubit_blocks
from .measurement_channel import measurement_channel_parity_candidate
from .measurement_channel_general import measurement_channel_general_candidates
from .measurement_channel_placement import measurement_channel_placement_candidates
from .measurement_channel_physical_placement_v2 import measurement_channel_physical_placement_v2_candidates
from .measurement_tail_pruner import prune_measurement_tail
from .parity_resynthesis import gf2_parity_resynthesis_candidates
from .peephole import conservative_peephole
from .phase_affine_cone_resynth import phase_affine_cone_candidates
from .relabel import optimize_physical_relabeling
from .relative_phase_toffoli import relative_phase_toffoli_rewrites
from .readout_copyout import readout_copyout_candidates
from .router import route_circuit
from .route_scar_normalizer import route_scar_normalizer_candidates
from .selection import CompilationCandidate, candidate_metrics, select_candidate
from .simulation_selector import SELECTOR_PROXY_ONLY, SimulationSelectorConfig, select_candidate_with_simulation
from .track_a_affine_collapse import track_a_affine_collapse_candidates
from .track_a_exact_router import exact_small_router_candidates
from .track_a_resynthesis import track_a_resynthesis_candidates, track_a_semantic_resynthesis_candidates
from .track_b import is_clifford_only
from .validation import validate_or_raise


MAX_CANDIDATES = 32
LEGACY_PARITY_BUDGET_SEC = 18.0


class NoiseAdaptiveCompiler:
    name = "noise_adaptive"

    def __init__(
        self,
        *,
        include_legacy_parity: bool = True,
        short_circuit_exact_ghz: bool = True,
        short_circuit_legacy_parity: bool = True,
        enable_ghz_leaf_tree: bool = False,
        enable_parity_resynthesis: bool = False,
        enable_measurement_channel_general: bool = False,
        enable_gf2_parity_resynthesis: bool = False,
        enable_exact_small_router: bool = False,
        enable_broad_exact_small_router: bool = False,
        enable_track_a_resynthesis: bool = False,
        enable_track_a_redundancy_rewrite: bool = False,
        enable_route_scar_normalizer: bool = False,
        enable_phase_affine_cone_resynthesis: bool = False,
        enable_h_boundary_pathsum_resynthesis: bool = False,
        enable_measurement_tail_pruning: bool = False,
        enable_relative_phase_toffoli_rewrite: bool = False,
        enable_relative_phase_exact_router: bool = False,
        enable_track_a_affine_collapse: bool = False,
        enable_track_a_semantic_resynthesis: bool = False,
        enable_track_b_relabel: bool = False,
        enable_ghz_relabel: bool = False,
        enable_readout_heavy_relabel: bool = False,
        enable_readout_copyout: bool = False,
        enable_measurement_channel_placement: bool = False,
        enable_measurement_channel_physical_placement_v2: bool = False,
        measurement_channel_physical_placement_v2_limit: int = 1,
        measurement_channel_physical_placement_v2_max_seconds: float = 3.0,
        selector_mode: str = SELECTOR_PROXY_ONLY,
        selector_config: SimulationSelectorConfig | None = None,
    ) -> None:
        self.last_initial_layout: dict[cirq.LineQubit, int] = {}
        self.include_legacy_parity = include_legacy_parity
        self.short_circuit_exact_ghz = short_circuit_exact_ghz
        self.short_circuit_legacy_parity = short_circuit_legacy_parity
        self.enable_ghz_leaf_tree = enable_ghz_leaf_tree
        self.enable_parity_resynthesis = enable_parity_resynthesis
        self.enable_measurement_channel_general = enable_measurement_channel_general
        self.enable_gf2_parity_resynthesis = enable_gf2_parity_resynthesis
        self.enable_exact_small_router = enable_exact_small_router
        self.enable_broad_exact_small_router = enable_broad_exact_small_router
        self.enable_track_a_resynthesis = enable_track_a_resynthesis
        self.enable_route_scar_normalizer = enable_route_scar_normalizer or enable_track_a_redundancy_rewrite
        self.enable_phase_affine_cone_resynthesis = enable_phase_affine_cone_resynthesis
        self.enable_h_boundary_pathsum_resynthesis = enable_h_boundary_pathsum_resynthesis
        self.enable_measurement_tail_pruning = enable_measurement_tail_pruning
        self.enable_relative_phase_toffoli_rewrite = enable_relative_phase_toffoli_rewrite
        self.enable_relative_phase_exact_router = enable_relative_phase_exact_router
        self.enable_track_a_affine_collapse = enable_track_a_affine_collapse
        self.enable_track_a_semantic_resynthesis = enable_track_a_semantic_resynthesis
        self.enable_track_b_relabel = enable_track_b_relabel
        self.enable_ghz_relabel = enable_ghz_relabel
        self.enable_readout_heavy_relabel = enable_readout_heavy_relabel
        self.enable_readout_copyout = enable_readout_copyout
        self.enable_measurement_channel_placement = enable_measurement_channel_placement
        self.enable_measurement_channel_physical_placement_v2 = enable_measurement_channel_physical_placement_v2
        self.measurement_channel_physical_placement_v2_limit = measurement_channel_physical_placement_v2_limit
        self.measurement_channel_physical_placement_v2_max_seconds = measurement_channel_physical_placement_v2_max_seconds
        self.selector_config = selector_config or SimulationSelectorConfig(mode=selector_mode)

    def compile(self, circuit: cirq.Circuit, backend: CompetitionBackend, seed: int = 7) -> cirq.Circuit:
        logical_qubits = tuple(sorted(circuit.all_qubits(), key=lambda qubit: qubit.x))
        if len(logical_qubits) > backend.num_qubits:
            raise ValueError("program circuit does not fit on backend")

        candidates = self._candidate_compilations(circuit, backend, seed)
        if not candidates:
            raise ValueError("no valid compilation candidate was produced")
        if self.enable_measurement_tail_pruning:
            _extend_candidates(candidates, _measurement_tail_pruned_candidates(circuit, backend, candidates))
        if self.enable_readout_copyout:
            _extend_candidates(candidates, readout_copyout_candidates(circuit, backend, candidates))
        selected = _protect_legacy_parity(
            select_candidate_with_simulation(
                candidates,
                backend,
                circuit,
                seed=seed,
                config=self.selector_config,
            ),
            candidates,
            backend,
            circuit,
        )
        self.last_initial_layout = selected.layout
        return selected.circuit

    def candidate_compilations(
        self,
        circuit: cirq.Circuit,
        backend: CompetitionBackend,
        seed: int = 7,
    ) -> list[CompilationCandidate]:
        return self._candidate_compilations(circuit, backend, seed)

    def _candidate_compilations(
        self,
        circuit: cirq.Circuit,
        backend: CompetitionBackend,
        seed: int,
    ) -> list[CompilationCandidate]:
        candidates: list[CompilationCandidate] = []
        ghz_candidate = self._exact_ghz_candidate(circuit, backend)
        if ghz_candidate is not None:
            _append_candidate(candidates, ghz_candidate)
            if self.enable_ghz_leaf_tree:
                _extend_candidates(candidates, ghz_leaf_tree_candidates(circuit, backend))
            if self.enable_track_b_relabel or self.enable_ghz_relabel:
                if self.enable_ghz_relabel and not self.enable_track_b_relabel:
                    _extend_candidates(
                        candidates,
                        _guarded_relabel_candidates(
                            circuit,
                            backend,
                            candidates,
                            max_mappings=20_000,
                            max_seconds=1.5,
                        ),
                    )
                else:
                    _extend_candidates(candidates, _guarded_relabel_candidates(circuit, backend, candidates))
            if self.short_circuit_exact_ghz:
                return candidates
        legacy_candidate = self._legacy_hub_parity_candidate(circuit, backend)
        if legacy_candidate is not None:
            _append_candidate(candidates, legacy_candidate)
            if self.enable_parity_resynthesis:
                if self.enable_measurement_channel_general:
                    _extend_candidates(
                        candidates,
                        measurement_channel_general_candidates(circuit, backend, legacy_candidate),
                    )
                if not any(candidate.name.startswith("v13_1") for candidate in candidates):
                    measurement_candidate = measurement_channel_parity_candidate(circuit, backend, legacy_candidate)
                    if measurement_candidate is not None:
                        _append_candidate(candidates, measurement_candidate)
                if self.enable_gf2_parity_resynthesis and not any(candidate.name.startswith("v13") for candidate in candidates):
                    _extend_candidates(
                        candidates,
                        gf2_parity_resynthesis_candidates(circuit, backend, legacy_candidate, seed=seed),
                    )
            if self.enable_measurement_channel_placement:
                _extend_candidates(
                    candidates,
                    measurement_channel_placement_candidates(circuit, backend, candidates),
                )
            if self.enable_measurement_channel_physical_placement_v2:
                _extend_candidates(
                    candidates,
                    measurement_channel_physical_placement_v2_candidates(
                        circuit,
                        backend,
                        candidates,
                        limit=self.measurement_channel_physical_placement_v2_limit,
                        max_seconds=self.measurement_channel_physical_placement_v2_max_seconds,
                    ),
                )
            if self.enable_track_b_relabel:
                _extend_candidates(candidates, _guarded_relabel_candidates(circuit, backend, candidates))
            if self.short_circuit_legacy_parity and _legacy_parity_candidate_clears_guard(
                legacy_candidate,
                backend,
                circuit,
            ):
                return candidates
        if _is_large_general_clifford(circuit):
            _extend_candidates(candidates, self._cirq_cqc_layout_ensemble(circuit, backend))
            _extend_candidates(
                candidates,
                self._noise_adaptive_ensemble(
                    circuit,
                    backend,
                    layout_limit=3,
                    max_candidates=6,
                    routing_variants=(
                        ("balanced", {"routing_objective": "balanced", "enable_execute_and_swap": True, "enable_bridge": True}),
                        ("noise", {"routing_objective": "noise", "enable_execute_and_swap": True, "enable_bridge": True}),
                    ),
                ),
            )
            if candidates:
                return candidates[:MAX_CANDIDATES]
        _extend_candidates(candidates, self._noise_adaptive_ensemble(circuit, backend))
        _extend_candidates(candidates, self._cirq_cqc_layout_ensemble(circuit, backend))
        _extend_candidates(candidates, self._hub_local_search_ensemble(circuit, backend))
        if self.enable_track_a_affine_collapse:
            _extend_candidates(candidates, track_a_affine_collapse_candidates(circuit, backend))
        if self.enable_relative_phase_toffoli_rewrite and not is_clifford_only(circuit):
            _extend_candidates(candidates, self._relative_phase_toffoli_ensemble(circuit, backend, seed=seed))
        exact_router_added = False
        track_a_semantic_added = False
        if self.enable_exact_small_router and not is_clifford_only(circuit) and candidates:
            try:
                incumbent = select_candidate(candidates, backend, circuit)
            except Exception:
                incumbent = None
            if (
                self.enable_route_scar_normalizer
                and not self.enable_broad_exact_small_router
                and not _route_scar_exact_router_worth_spending(circuit, backend, incumbent)
            ):
                exact_router_added = True
            else:
                exact_router_kwargs = self._exact_router_kwargs()
                _extend_candidates(
                    candidates,
                    exact_small_router_candidates(circuit, backend, incumbent, seed=seed, **exact_router_kwargs),
                )
                exact_router_added = True
        if self.enable_track_a_semantic_resynthesis and not is_clifford_only(circuit) and candidates:
            before_semantic = len(candidates)
            _extend_candidates(candidates, track_a_semantic_resynthesis_candidates(circuit, backend, candidates))
            track_a_semantic_added = True
            if len(candidates) > before_semantic:
                return candidates[:MAX_CANDIDATES]
        if self.enable_route_scar_normalizer and not is_clifford_only(circuit) and candidates:
            before_redundancy = len(candidates)
            _extend_candidates(candidates, route_scar_normalizer_candidates(circuit, backend, candidates))
            if len(candidates) > before_redundancy:
                return candidates[:MAX_CANDIDATES]
        if self.enable_phase_affine_cone_resynthesis and not is_clifford_only(circuit) and candidates:
            before_phase_affine = len(candidates)
            _extend_candidates(candidates, phase_affine_cone_candidates(circuit, backend, candidates))
            if len(candidates) > before_phase_affine:
                return candidates[:MAX_CANDIDATES]
        if self.enable_h_boundary_pathsum_resynthesis and not is_clifford_only(circuit) and candidates:
            before_h_pathsum = len(candidates)
            _extend_candidates(candidates, h_boundary_pathsum_candidates(circuit, backend, candidates))
            if len(candidates) > before_h_pathsum:
                return candidates[:MAX_CANDIDATES]
        for name, compile_fn in [
            ("reliability_greedy", lambda source, target: _compile_baseline(ReliabilityGreedyCompiler(), source, target, seed)),
            ("movement_aware", lambda source, target: _compile_baseline(MovementAwareCompiler(), source, target, seed)),
            ("paper_greedy_vertex", lambda source, target: _compile_baseline(PaperGreedyVertexCompiler(), source, target, seed)),
            ("paper_greedy_edge", lambda source, target: _compile_baseline(PaperGreedyEdgeCompiler(), source, target, seed)),
            ("cirq_cqc", lambda source, target: _compile_baseline(CirqCQCCompiler(), source, target, seed)),
        ]:
            try:
                started = time.perf_counter()
                compiled, layout = compile_fn(circuit, backend)
                elapsed = time.perf_counter() - started
                compiled = conservative_peephole(compiled, clifford_only=is_clifford_only(circuit))
                validate_or_raise(circuit, compiled, backend)
            except Exception:
                continue
            _append_candidate(candidates, CompilationCandidate(name, compiled, layout, elapsed))
            if is_clifford_only(circuit):
                try:
                    started = time.perf_counter()
                    resynthesized, changed = resynthesize_cnot_blocks(compiled, backend)
                    if changed:
                        validate_or_raise(circuit, resynthesized, backend)
                        _append_candidate(
                            candidates,
                            CompilationCandidate(f"{name}_cnot_resynth", resynthesized, layout, elapsed + time.perf_counter() - started),
                        )
                except Exception:
                    pass
            else:
                try:
                    started = time.perf_counter()
                    resynthesized, changed = resynthesize_local_two_qubit_blocks(compiled, backend)
                    if changed:
                        validate_or_raise(circuit, resynthesized, backend)
                        _append_candidate(
                            candidates,
                            CompilationCandidate(f"{name}_local_2q_resynth", resynthesized, layout, elapsed + time.perf_counter() - started),
                        )
                except Exception:
                    pass
        if is_clifford_only(circuit):
            _extend_candidates(candidates, _relabel_best_candidates(circuit, backend, candidates))
            if self.enable_readout_heavy_relabel:
                _extend_candidates(candidates, _readout_heavy_relabel_candidates(circuit, backend, candidates))
        else:
            if self.enable_readout_heavy_relabel:
                _extend_candidates(candidates, _readout_heavy_relabel_candidates(circuit, backend, candidates))
            if self.enable_exact_small_router and not exact_router_added and candidates:
                try:
                    incumbent = select_candidate(candidates, backend, circuit)
                except Exception:
                    incumbent = None
                if (
                    self.enable_broad_exact_small_router
                    or not self.enable_route_scar_normalizer
                    or _route_scar_exact_router_worth_spending(circuit, backend, incumbent)
                ):
                    exact_router_kwargs = self._exact_router_kwargs()
                    _extend_candidates(
                        candidates,
                        exact_small_router_candidates(circuit, backend, incumbent, seed=seed, **exact_router_kwargs),
                    )
            if self.enable_track_a_resynthesis:
                _extend_candidates(candidates, track_a_resynthesis_candidates(circuit, backend, candidates))
            if self.enable_route_scar_normalizer:
                _extend_candidates(candidates, route_scar_normalizer_candidates(circuit, backend, candidates))
            if self.enable_phase_affine_cone_resynthesis:
                _extend_candidates(candidates, phase_affine_cone_candidates(circuit, backend, candidates))
            if self.enable_h_boundary_pathsum_resynthesis:
                _extend_candidates(candidates, h_boundary_pathsum_candidates(circuit, backend, candidates))
            if self.enable_track_a_semantic_resynthesis:
                if not track_a_semantic_added:
                    _extend_candidates(candidates, track_a_semantic_resynthesis_candidates(circuit, backend, candidates))
        return candidates[:MAX_CANDIDATES]

    def _exact_router_kwargs(self) -> dict[str, object]:
        if self.enable_broad_exact_small_router:
            return {"max_candidates": 8, "max_seconds": 8.0}
        if self.enable_route_scar_normalizer:
            return {"max_candidates": 1, "max_seconds": 8.0, "incumbent_only": True}
        return {}

    def _exact_ghz_candidate(
        self,
        circuit: cirq.Circuit,
        backend: CompetitionBackend,
    ) -> CompilationCandidate | None:
        try:
            started = time.perf_counter()
            result = compile_exact_ghz_tree_fanout(circuit, backend)
            if result is None:
                return None
            compiled, layout = result
            validate_or_raise(circuit, compiled, backend)
            return CompilationCandidate("exact_ghz_tree", compiled, layout, time.perf_counter() - started)
        except Exception:
            return None

    def _legacy_hub_parity_candidate(
        self,
        circuit: cirq.Circuit,
        backend: CompetitionBackend,
    ) -> CompilationCandidate | None:
        if not self.include_legacy_parity:
            return None
        try:
            result = compile_legacy_hub_parity(circuit, backend, max_seconds=LEGACY_PARITY_BUDGET_SEC)
            if result is None:
                return None
            compiled, layout, elapsed = result
            compiled = conservative_peephole(compiled, clifford_only=True)
            validate_or_raise(circuit, compiled, backend)
            return CompilationCandidate("legacy_hub_parity_seed100", compiled, layout, elapsed)
        except Exception:
            return None

    def _noise_adaptive_ensemble(
        self,
        circuit: cirq.Circuit,
        backend: CompetitionBackend,
        *,
        layout_limit: int = 8,
        max_candidates: int = 20,
        routing_variants: tuple[tuple[str, dict[str, object]], ...] | None = None,
    ) -> list[CompilationCandidate]:
        candidates: list[CompilationCandidate] = []
        model = BackendNoiseModel.from_backend(backend)
        features = extract_circuit_features(circuit)
        layouts = candidate_initial_layouts(circuit, backend, model, features, limit=layout_limit)
        routing_variants = routing_variants or (
            ("balanced", {"routing_objective": "balanced", "enable_execute_and_swap": True, "enable_bridge": True}),
            ("noise", {"routing_objective": "noise", "enable_execute_and_swap": True, "enable_bridge": True}),
            ("hop", {"routing_objective": "hop", "enable_execute_and_swap": True, "enable_bridge": True}),
            ("conservative", {"routing_objective": "balanced", "enable_execute_and_swap": False, "enable_bridge": False}),
        )
        for layout_index, layout in enumerate(layouts):
            for route_name, route_kwargs in routing_variants:
                if len(candidates) >= max_candidates:
                    return candidates
                name = f"noise_adaptive_l{layout_index}_{route_name}"
                try:
                    started = time.perf_counter()
                    routed = route_circuit(circuit, backend, model, features, layout, **route_kwargs).circuit
                    routed = conservative_peephole(routed, clifford_only=is_clifford_only(circuit))
                    validate_or_raise(circuit, routed, backend)
                    elapsed = time.perf_counter() - started
                except Exception:
                    continue
                _append_candidate(candidates, CompilationCandidate(name, routed, dict(layout), elapsed))
        return candidates

    def _relative_phase_toffoli_ensemble(
        self,
        circuit: cirq.Circuit,
        backend: CompetitionBackend,
        *,
        seed: int,
    ) -> list[CompilationCandidate]:
        candidates: list[CompilationCandidate] = []
        for rewrite in relative_phase_toffoli_rewrites(circuit, max_replacements=2)[:3]:
            routed_candidates = self._noise_adaptive_ensemble(
                rewrite.circuit,
                backend,
                layout_limit=6,
                max_candidates=8,
                routing_variants=(
                    ("balanced", {"routing_objective": "balanced", "enable_execute_and_swap": True, "enable_bridge": True}),
                    ("noise", {"routing_objective": "noise", "enable_execute_and_swap": True, "enable_bridge": True}),
                ),
            )
            for candidate in routed_candidates:
                try:
                    validate_or_raise(circuit, candidate.circuit, backend)
                except Exception:
                    continue
                _append_candidate(
                    candidates,
                    CompilationCandidate(
                        f"{rewrite.name}_{candidate.name}",
                        candidate.circuit,
                        dict(candidate.layout),
                        candidate.compile_time_sec,
                    ),
                )
            if self.enable_relative_phase_exact_router and routed_candidates:
                try:
                    incumbent = select_candidate(routed_candidates, backend, rewrite.circuit)
                except Exception:
                    incumbent = None
                for candidate in exact_small_router_candidates(
                    rewrite.circuit,
                    backend,
                    incumbent,
                    seed=seed,
                    max_candidates=3,
                    max_seconds=6.0,
                    incumbent_only=True,
                    min_two_qubit_count=8,
                ):
                    try:
                        validate_or_raise(circuit, candidate.circuit, backend)
                    except Exception:
                        continue
                    if incumbent is not None and not _relative_phase_exact_router_guard(
                        candidate,
                        incumbent,
                        backend,
                        rewrite.circuit,
                    ):
                        continue
                    _append_candidate(
                        candidates,
                        CompilationCandidate(
                            f"{rewrite.name}_{candidate.name}",
                            candidate.circuit,
                            dict(candidate.layout),
                            candidate.compile_time_sec,
                        ),
                    )
        return candidates

    def _cirq_cqc_layout_ensemble(
        self,
        circuit: cirq.Circuit,
        backend: CompetitionBackend,
    ) -> list[CompilationCandidate]:
        candidates: list[CompilationCandidate] = []
        model = BackendNoiseModel.from_backend(backend)
        features = extract_circuit_features(circuit)
        layouts = candidate_initial_layouts(circuit, backend, model, features, limit=4)
        lookaheads = (8, 12, 16) if is_clifford_only(circuit) else (12,)
        if len(features.logical_qubits) > 16:
            lookaheads = (12,)
            layouts = layouts[:2]
        for layout_index, layout in enumerate(layouts):
            for lookahead in lookaheads:
                if len(candidates) >= 8:
                    return candidates
                try:
                    started = time.perf_counter()
                    compiled, routed_layout = _compile_with_cirq_cqc_layout(circuit, backend, layout, lookahead)
                    compiled = conservative_peephole(compiled, clifford_only=is_clifford_only(circuit))
                    validate_or_raise(circuit, compiled, backend)
                    elapsed = time.perf_counter() - started
                except Exception:
                    continue
                _append_candidate(
                    candidates,
                    CompilationCandidate(f"cirq_cqc_l{layout_index}_r{lookahead}", compiled, routed_layout, elapsed),
                )
        return candidates

    def _hub_local_search_ensemble(
        self,
        circuit: cirq.Circuit,
        backend: CompetitionBackend,
    ) -> list[CompilationCandidate]:
        if not is_clifford_only(circuit):
            return []
        model = BackendNoiseModel.from_backend(backend)
        features = extract_circuit_features(circuit)
        initial = hub_aware_layout(circuit, backend, model)
        if initial is None:
            return []

        candidates: list[CompilationCandidate] = []
        try:
            started = time.perf_counter()
            compiled = route_circuit(circuit, backend, model, features, initial, routing_objective="balanced").circuit
            compiled = conservative_peephole(compiled, clifford_only=True)
            validate_or_raise(circuit, compiled, backend)
            elapsed = time.perf_counter() - started
            base_candidate = CompilationCandidate("hub_aware_layout", compiled, dict(initial), elapsed)
            _append_candidate(candidates, base_candidate)
        except Exception:
            return candidates

        best_layout = dict(initial)
        best_circuit = candidates[0].circuit
        best_key = _candidate_score(best_circuit, backend)
        hot = sorted(
            features.logical_qubits,
            key=lambda qubit: (features.weighted_degrees[qubit.x], -qubit.x),
            reverse=True,
        )[: max(4, min(10, len(features.logical_qubits)))]
        if len(hot) < 2:
            return candidates
        steps = 28 if len(features.logical_qubits) > 16 else 16
        for step in range(steps):
            left = hot[(17 * step + 1) % len(hot)]
            right = hot[(29 * step + 3) % len(hot)]
            if left == right:
                continue
            trial = dict(best_layout)
            trial[left], trial[right] = trial[right], trial[left]
            try:
                started = time.perf_counter()
                compiled = route_circuit(circuit, backend, model, features, trial, routing_objective="balanced").circuit
                compiled = conservative_peephole(compiled, clifford_only=True)
                validate_or_raise(circuit, compiled, backend)
                elapsed = time.perf_counter() - started
            except Exception:
                continue
            key = _candidate_score(compiled, backend)
            if key > best_key + 1e-12:
                best_key = key
                best_layout = trial
                best_circuit = compiled
                _append_candidate(candidates, CompilationCandidate(f"hub_aware_ls_{step}", compiled, dict(trial), elapsed))
            if len(candidates) >= 6:
                break
        return candidates

    def _compile_noise_adaptive(
        self,
        circuit: cirq.Circuit,
        backend: CompetitionBackend,
    ) -> tuple[cirq.Circuit, dict[cirq.LineQubit, int]]:
        model = BackendNoiseModel.from_backend(backend)
        features = extract_circuit_features(circuit)
        initial_layout = choose_initial_layout(circuit, backend, model, features)
        routed = route_circuit(circuit, backend, model, features, initial_layout).circuit
        return routed, dict(initial_layout)


def _compile_baseline(compiler, circuit: cirq.Circuit, backend: CompetitionBackend, seed: int):
    compiled = compiler.compile(circuit, backend, seed=seed)
    layout = getattr(compiler, "last_initial_layout", None)
    if layout is None and hasattr(compiler, "initial_layout"):
        layout = compiler.initial_layout(circuit, backend)
    if layout is None:
        logical_qubits = sorted(circuit.all_qubits(), key=lambda qubit: qubit.x)
        layout = {qubit: qubit.x for qubit in logical_qubits}
    return compiled, dict(layout)


def _compile_with_cirq_cqc_layout(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    layout: dict[cirq.LineQubit, int],
    lookahead_radius: int,
) -> tuple[cirq.Circuit, dict[cirq.LineQubit, int]]:
    physical_qubits = tuple(cirq.LineQubit.range(backend.num_qubits))
    device_graph = nx.Graph()
    device_graph.add_nodes_from(physical_qubits)
    device_graph.add_edges_from(
        (physical_qubits[left], physical_qubits[right])
        for left, right in backend.coupling_map
    )
    initial_mapping = {
        logical: physical_qubits[physical]
        for logical, physical in layout.items()
    }
    routed, initial_map, _ = cirq.RouteCQC(device_graph).route_circuit(
        circuit,
        lookahead_radius=lookahead_radius,
        initial_mapper=cirq.HardCodedInitialMapper(initial_mapping),
    )
    initial_layout = {
        logical: physical.x
        for logical, physical in initial_map.items()
    }
    return decompose_swaps(cirq.Circuit(routed)), initial_layout


def _candidate_score(circuit: cirq.Circuit, backend: CompetitionBackend) -> float:
    return (
        estimate_log_reliability(circuit, backend)
        - 0.002 * count_two_qubit_gates(circuit)
        - 0.0005 * len(circuit)
    )


def _relative_phase_exact_router_guard(
    candidate: CompilationCandidate,
    incumbent: CompilationCandidate,
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> bool:
    candidate_value = candidate_metrics(candidate, backend, source)
    incumbent_value = candidate_metrics(incumbent, backend, source)
    if (
        candidate_value.two_qubit_count <= incumbent_value.two_qubit_count - 2
        and candidate_value.depth <= incumbent_value.depth
        and candidate_value.log_reliability >= incumbent_value.log_reliability - 0.001
    ):
        return True
    if (
        candidate_value.log_reliability >= incumbent_value.log_reliability + 0.01
        and candidate_value.two_qubit_count <= incumbent_value.two_qubit_count
    ):
        return True
    return False


def _relabel_best_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    limit: int = 3,
    max_mappings: int = 20_000,
    max_seconds: float = 1.5,
) -> list[CompilationCandidate]:
    relabeled: list[CompilationCandidate] = []
    ranked = sorted(candidates, key=lambda item: _candidate_score(item.circuit, backend), reverse=True)
    for candidate in ranked[:limit]:
        try:
            started = time.perf_counter()
            optimized, optimized_layout = optimize_physical_relabeling(
                candidate.circuit,
                candidate.layout,
                backend,
                max_mappings=max_mappings,
                max_seconds=max_seconds,
            )
            validate_or_raise(source, optimized, backend)
        except Exception:
            continue
        relabeled.append(
            CompilationCandidate(
                f"{candidate.name}_relabel",
                optimized,
                optimized_layout,
                candidate.compile_time_sec + time.perf_counter() - started,
            )
        )
    return relabeled


def _guarded_relabel_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    limit: int = 4,
    max_mappings: int = 20_000,
    max_seconds: float = 1.5,
) -> list[CompilationCandidate]:
    by_name = {candidate.name: candidate for candidate in candidates}
    accepted: list[CompilationCandidate] = []
    for candidate in _relabel_best_candidates(
        source,
        backend,
        candidates,
        limit=limit,
        max_mappings=max_mappings,
        max_seconds=max_seconds,
    ):
        if not candidate.name.endswith("_relabel"):
            continue
        base = by_name.get(candidate.name[: -len("_relabel")])
        if base is None:
            continue
        if _relabel_improves(candidate, base, backend, source):
            accepted.append(candidate)
    return accepted


def _readout_heavy_relabel_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
    limit: int = 4,
) -> list[CompilationCandidate]:
    accepted: list[CompilationCandidate] = []
    ranked = sorted(candidates, key=lambda item: _candidate_score(item.circuit, backend), reverse=True)
    for base in ranked[:limit]:
        try:
            started = time.perf_counter()
            optimized, optimized_layout = optimize_physical_relabeling(
                base.circuit,
                base.layout,
                backend,
                max_mappings=50_000,
                max_seconds=2.5,
                readout_weight=1.0,
            )
            validate_or_raise(source, optimized, backend)
        except Exception:
            continue
        candidate = CompilationCandidate(
            f"{base.name}_readout_relabel",
            optimized,
            optimized_layout,
            base.compile_time_sec + time.perf_counter() - started,
        )
        if _readout_relabel_improves(candidate, base, backend, source):
            accepted.append(candidate)
            if len(accepted) >= limit:
                break
    return accepted


def _readout_relabel_improves(
    candidate: CompilationCandidate,
    base: CompilationCandidate,
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> bool:
    candidate_value = candidate_metrics(candidate, backend, source)
    base_value = candidate_metrics(base, backend, source)
    return (
        candidate_value.two_qubit_count == base_value.two_qubit_count
        and candidate_value.depth == base_value.depth
        and candidate_value.log_reliability >= base_value.log_reliability + 0.002
        and candidate_value.readout_cost <= base_value.readout_cost + 0.002
        and candidate_value.clifford_sensitivity_cost <= base_value.clifford_sensitivity_cost + 0.05
        and candidate_value.compile_time_sec <= 12.0
    )


def _relabel_improves(
    candidate: CompilationCandidate,
    base: CompilationCandidate,
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> bool:
    candidate_value = candidate_metrics(candidate, backend, source)
    base_value = candidate_metrics(base, backend, source)
    return (
        candidate_value.two_qubit_count <= base_value.two_qubit_count
        and candidate_value.depth <= base_value.depth
        and candidate_value.log_reliability >= base_value.log_reliability + 0.005
        and candidate_value.readout_cost <= base_value.readout_cost + 1e-12
        and candidate_value.clifford_sensitivity_cost <= base_value.clifford_sensitivity_cost + 0.05
    )


def _append_candidate(candidates: list[CompilationCandidate], candidate: CompilationCandidate) -> None:
    if len(candidates) >= MAX_CANDIDATES:
        return
    signature = (
        tuple(str(op) for op in candidate.circuit.all_operations()),
        tuple(sorted((logical.x, physical) for logical, physical in candidate.layout.items())),
    )
    for existing in candidates:
        existing_signature = (
            tuple(str(op) for op in existing.circuit.all_operations()),
            tuple(sorted((logical.x, physical) for logical, physical in existing.layout.items())),
        )
        if signature == existing_signature:
            return
    candidates.append(candidate)


def _extend_candidates(candidates: list[CompilationCandidate], additions: list[CompilationCandidate]) -> None:
    for candidate in additions:
        _append_candidate(candidates, candidate)


def _measurement_tail_pruned_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    candidates: list[CompilationCandidate],
) -> list[CompilationCandidate]:
    additions: list[CompilationCandidate] = []
    seen = {tuple(str(op) for op in candidate.circuit.all_operations()) for candidate in candidates}
    for candidate in candidates:
        result = prune_measurement_tail(candidate.circuit, clifford_only=is_clifford_only(source))
        if not result.changed:
            continue
        signature = tuple(str(op) for op in result.circuit.all_operations())
        if signature in seen:
            continue
        try:
            validate_or_raise(source, result.circuit, backend)
        except Exception:
            continue
        seen.add(signature)
        additions.append(
            CompilationCandidate(
                f"{candidate.name}_measurement_tail_pruned",
                result.circuit,
                dict(candidate.layout),
                candidate.compile_time_sec,
            )
        )
    return additions


def _protect_legacy_parity(
    selected: CompilationCandidate,
    candidates: list[CompilationCandidate],
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> CompilationCandidate:
    legacy = next((candidate for candidate in candidates if candidate.name == "legacy_hub_parity_seed100"), None)
    if legacy is None or selected is legacy:
        return selected
    selected_metrics = candidate_metrics(selected, backend, source)
    legacy_metrics = candidate_metrics(legacy, backend, source)
    clearly_better = (
        selected_metrics.two_qubit_count <= legacy_metrics.two_qubit_count - 8
        and selected_metrics.depth <= legacy_metrics.depth
        and selected_metrics.log_reliability >= legacy_metrics.log_reliability + 0.05
        and selected_metrics.clifford_sensitivity_cost <= legacy_metrics.clifford_sensitivity_cost
    )
    placement_better = (
        selected.name.endswith("_mc_placement")
        and selected_metrics.two_qubit_count <= legacy_metrics.two_qubit_count
        and selected_metrics.depth <= legacy_metrics.depth
        and selected_metrics.log_reliability >= legacy_metrics.log_reliability + 0.001
        and selected_metrics.readout_cost <= legacy_metrics.readout_cost + 0.002
        and selected_metrics.clifford_sensitivity_cost <= legacy_metrics.clifford_sensitivity_cost + 1e-12
    )
    return selected if clearly_better or placement_better else legacy


def _legacy_parity_candidate_clears_guard(
    candidate: CompilationCandidate,
    backend: CompetitionBackend,
    source: cirq.Circuit,
) -> bool:
    logical_width = len(source.all_qubits())
    if logical_width <= 16 or not is_clifford_only(source):
        return False
    metrics = candidate_metrics(candidate, backend, source)
    return (
        metrics.two_qubit_count <= 115
        and metrics.depth <= 80
        and metrics.log_reliability >= -2.10
        and metrics.compile_time_sec <= LEGACY_PARITY_BUDGET_SEC + 1.0
    )


def _route_scar_exact_router_worth_spending(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    incumbent: CompilationCandidate | None,
) -> bool:
    logical_width = len(circuit.all_qubits())
    source_cx = count_two_qubit_gates(circuit)
    if logical_width != 7 or source_cx < 12 or source_cx > 20:
        return False
    names = [gate_name_for_operation(op) for op in circuit.all_operations()]
    phase_count = sum(1 for name in names if name in {"t", "tdg", "s", "sdg", "rz"})
    if names.count("h") < 2 or phase_count < 8:
        return False
    if incumbent is None:
        return True
    incumbent_metrics = candidate_metrics(incumbent, backend, circuit)
    return incumbent_metrics.route_overhead >= 5


def _is_large_general_clifford(circuit: cirq.Circuit) -> bool:
    return is_clifford_only(circuit) and len(circuit.all_qubits()) >= 20
