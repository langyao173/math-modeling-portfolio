from __future__ import annotations

import cirq

from noise_adaptive_hackathon.baseline.cirq_cqc import CirqCQCCompiler
from noise_adaptive_hackathon.baseline.common import BaseCompetitionCompiler
from noise_adaptive_hackathon.evaluator.validate_circuit import count_two_qubit_gates, estimate_log_reliability
from noise_adaptive_hackathon.formats.backend import CompetitionBackend

from .backend_model import BackendNoiseModel
from .circuit_features import extract_circuit_features
from .compiler import NoiseAdaptiveCompiler
from .layout import choose_initial_layout
from .peephole import conservative_peephole
from .router import route_circuit
from .track_b import is_clifford_only
from .simulation_selector import SELECTOR_COMMON_RANDOM, SELECTOR_PROXY_ONLY
from .simulation_selector import SELECTOR_TRACK_A_COMMON_RANDOM, SELECTOR_TRACK_A_RANKED_PROXY
from .simulation_selector import SELECTOR_TRACK_B_COMMON_RANDOM, SimulationSelectorConfig
from .validation import validate_or_raise


ABLATION_VARIANTS = (
    "baseline",
    "layout_only",
    "layout_routing",
    "layout_routing_peephole",
    "track_b_branch",
    "track_b_branch_proxy_only",
    "track_b_branch_sim_selector",
    "track_b_branch_v13",
    "track_b_branch_v13_1",
    "track_b_branch_v13a",
    "track_b_branch_v13_relabel",
    "track_a_exact_router",
    "track_a_v14",
    "v13_root",
    "v13_plus_b_relabel_only",
    "v13_plus_tracka_exact_router_only",
    "v13_plus_tracka_templates_only",
    "v14_combined",
    "track_a_v15",
    "v15_tracka_semantic",
    "v15_2_redundancy",
    "v16_phase_affine",
    "v17_h_boundary_pathsum",
    "v18_measurement_tail",
    "v19_relative_phase_toffoli",
    "v15_1_b002_placement",
    "v20_bplacement_v2",
    "v20_aggressive",
    "v20_aggressive_fast",
    "v21_affine_collapse",
    "v22_ghz_leaf_tree",
    "v23_readout_copyout",
    "v24_trackb_lowshot_selector",
    "v25_b_relabel_copyout",
    "v26_ghz_relabel_copyout",
    "v27_b002_frontier_copyout",
    "v28_b002_wide_frontier_copyout",
    "v29_tracka_lowshot_selector",
    "v30_tracka_ranked_proxy_selector",
    "v31_ghz_relabel_tracka_ranked",
    "v32_relative_phase_exact_router",
    "final_c0_conservative",
    "final_c1_b_relabel",
    "final_c2_tracka_exact",
    "final_c3_b_relabel_tracka_exact",
    "final_c4_all_safe",
)


class NoiseAwareLayoutShortestPathCompiler(BaseCompetitionCompiler):
    name = "layout_only"

    def initial_layout(self, circuit: cirq.Circuit, backend: CompetitionBackend) -> dict[cirq.LineQubit, int]:
        model = BackendNoiseModel.from_backend(backend)
        features = extract_circuit_features(circuit)
        return choose_initial_layout(circuit, backend, model, features)

    def route_path(self, source: int, target: int, backend: CompetitionBackend) -> list[int]:
        return backend.shortest_path(source, target)


class VariantCompiler:
    def __init__(self, variant: str) -> None:
        if variant not in ABLATION_VARIANTS:
            raise ValueError(f"unknown ablation variant: {variant}")
        self.variant = variant
        self.last_initial_layout: dict[cirq.LineQubit, int] = {}

    def compile(self, circuit: cirq.Circuit, backend: CompetitionBackend, seed: int = 7) -> cirq.Circuit:
        if self.variant == "baseline":
            compiled, layout = _compile_with_layout(CirqCQCCompiler(), circuit, backend, seed)
        elif self.variant == "layout_only":
            compiled, layout = _compile_with_layout(NoiseAwareLayoutShortestPathCompiler(), circuit, backend, seed)
        elif self.variant == "layout_routing":
            compiled, layout = _compile_pass_pipeline(circuit, backend, peephole=False)
        elif self.variant == "layout_routing_peephole":
            compiled, layout = _compile_pass_pipeline(circuit, backend, peephole=True)
        elif self.variant == "track_b_branch":
            final = NoiseAdaptiveCompiler()
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "track_b_branch_proxy_only":
            final = NoiseAdaptiveCompiler(selector_mode=SELECTOR_PROXY_ONLY)
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant in {"track_b_branch_v13", "track_b_branch_v13a", "v13_root"}:
            final = NoiseAdaptiveCompiler(enable_parity_resynthesis=True)
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant in {"track_b_branch_v13_relabel", "v13_plus_b_relabel_only"}:
            final = NoiseAdaptiveCompiler(enable_parity_resynthesis=True, enable_track_b_relabel=True)
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "track_b_branch_v13_1":
            final = NoiseAdaptiveCompiler(enable_parity_resynthesis=True, enable_measurement_channel_general=True)
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v15_1_b002_placement":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_measurement_channel_placement=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v20_bplacement_v2":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_measurement_channel_physical_placement_v2=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant in {"v20_aggressive", "v20_aggressive_fast"}:
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_measurement_tail_pruning=True,
                enable_relative_phase_toffoli_rewrite=True,
                enable_measurement_channel_physical_placement_v2=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v21_affine_collapse":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_measurement_tail_pruning=True,
                enable_relative_phase_toffoli_rewrite=True,
                enable_measurement_channel_physical_placement_v2=True,
                enable_track_a_affine_collapse=True,
                enable_readout_heavy_relabel=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v22_ghz_leaf_tree":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_measurement_tail_pruning=True,
                enable_relative_phase_toffoli_rewrite=True,
                enable_measurement_channel_physical_placement_v2=True,
                enable_ghz_leaf_tree=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v23_readout_copyout":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_measurement_tail_pruning=True,
                enable_relative_phase_toffoli_rewrite=True,
                enable_measurement_channel_physical_placement_v2=True,
                enable_ghz_leaf_tree=True,
                enable_readout_copyout=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v24_trackb_lowshot_selector":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_measurement_tail_pruning=True,
                enable_relative_phase_toffoli_rewrite=True,
                enable_measurement_channel_physical_placement_v2=True,
                enable_ghz_leaf_tree=True,
                enable_readout_copyout=True,
                selector_mode=SELECTOR_TRACK_B_COMMON_RANDOM,
                selector_config=SimulationSelectorConfig(
                    mode=SELECTOR_TRACK_B_COMMON_RANDOM,
                    track_b_top_k=8,
                    track_b_lowshot=2048,
                    track_b_timeout_sec=3.0,
                    track_b_margin=0.004,
                ),
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v25_b_relabel_copyout":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_measurement_tail_pruning=True,
                enable_relative_phase_toffoli_rewrite=True,
                enable_measurement_channel_physical_placement_v2=True,
                enable_ghz_leaf_tree=True,
                enable_readout_copyout=True,
                enable_track_b_relabel=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v26_ghz_relabel_copyout":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_measurement_tail_pruning=True,
                enable_relative_phase_toffoli_rewrite=True,
                enable_measurement_channel_physical_placement_v2=True,
                enable_ghz_leaf_tree=True,
                enable_readout_copyout=True,
                enable_ghz_relabel=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v27_b002_frontier_copyout":
            final = NoiseAdaptiveCompiler(
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
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v28_b002_wide_frontier_copyout":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_measurement_tail_pruning=True,
                enable_relative_phase_toffoli_rewrite=True,
                enable_measurement_channel_physical_placement_v2=True,
                measurement_channel_physical_placement_v2_limit=12,
                measurement_channel_physical_placement_v2_max_seconds=8.0,
                enable_ghz_leaf_tree=True,
                enable_readout_copyout=True,
                enable_ghz_relabel=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v29_tracka_lowshot_selector":
            final = NoiseAdaptiveCompiler(
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
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v30_tracka_ranked_proxy_selector":
            final = NoiseAdaptiveCompiler(
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
                selector_mode=SELECTOR_TRACK_A_RANKED_PROXY,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v31_ghz_relabel_tracka_ranked":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_measurement_tail_pruning=True,
                enable_relative_phase_toffoli_rewrite=True,
                enable_measurement_channel_physical_placement_v2=True,
                enable_ghz_leaf_tree=True,
                enable_readout_copyout=True,
                enable_ghz_relabel=True,
                selector_mode=SELECTOR_TRACK_A_RANKED_PROXY,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v32_relative_phase_exact_router":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_measurement_tail_pruning=True,
                enable_relative_phase_toffoli_rewrite=True,
                enable_relative_phase_exact_router=True,
                enable_measurement_channel_physical_placement_v2=True,
                measurement_channel_physical_placement_v2_limit=2,
                enable_ghz_leaf_tree=True,
                enable_readout_copyout=True,
                enable_ghz_relabel=True,
                selector_mode=SELECTOR_TRACK_A_RANKED_PROXY,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "final_c0_conservative":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_measurement_tail_pruning=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "final_c1_b_relabel":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_measurement_tail_pruning=True,
                enable_track_b_relabel=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "final_c2_tracka_exact":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_broad_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_measurement_tail_pruning=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "final_c3_b_relabel_tracka_exact":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_broad_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_measurement_tail_pruning=True,
                enable_track_b_relabel=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "final_c4_all_safe":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_broad_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_measurement_tail_pruning=True,
                enable_track_b_relabel=True,
                enable_measurement_channel_placement=True,
                enable_track_a_resynthesis=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant in {"track_a_v14", "v14_combined"}:
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_track_a_resynthesis=True,
                enable_track_b_relabel=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant in {"track_a_exact_router", "v13_plus_tracka_exact_router_only"}:
            final = NoiseAdaptiveCompiler(enable_parity_resynthesis=True, enable_exact_small_router=True)
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v13_plus_tracka_templates_only":
            final = NoiseAdaptiveCompiler(enable_parity_resynthesis=True, enable_track_a_resynthesis=True)
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant in {"track_a_v15", "v15_tracka_semantic"}:
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_track_a_resynthesis=True,
                enable_track_a_semantic_resynthesis=True,
                enable_track_b_relabel=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v15_2_redundancy":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_route_scar_normalizer=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v16_phase_affine":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_phase_affine_cone_resynthesis=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v17_h_boundary_pathsum":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_h_boundary_pathsum_resynthesis=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v18_measurement_tail":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_measurement_tail_pruning=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        elif self.variant == "v19_relative_phase_toffoli":
            final = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
                enable_route_scar_normalizer=True,
                enable_measurement_tail_pruning=True,
                enable_relative_phase_toffoli_rewrite=True,
            )
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout
        else:
            final = NoiseAdaptiveCompiler(selector_mode=SELECTOR_COMMON_RANDOM)
            compiled = final.compile(circuit, backend, seed=seed)
            layout = final.last_initial_layout

        validate_or_raise(circuit, compiled, backend)
        self.last_initial_layout = dict(layout)
        return compiled


def candidate_objective(circuit: cirq.Circuit, backend: CompetitionBackend) -> float:
    return (
        estimate_log_reliability(circuit, backend)
        - 0.002 * count_two_qubit_gates(circuit)
        - 0.0005 * len(circuit)
    )


def _compile_pass_pipeline(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    *,
    peephole: bool,
) -> tuple[cirq.Circuit, dict[cirq.LineQubit, int]]:
    model = BackendNoiseModel.from_backend(backend)
    features = extract_circuit_features(circuit)
    layout = choose_initial_layout(circuit, backend, model, features)
    compiled = route_circuit(circuit, backend, model, features, layout).circuit
    if peephole:
        compiled = conservative_peephole(compiled, clifford_only=is_clifford_only(circuit))
    return compiled, dict(layout)


def _compile_with_layout(
    compiler,
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    seed: int,
) -> tuple[cirq.Circuit, dict[cirq.LineQubit, int]]:
    compiled = compiler.compile(circuit, backend, seed=seed)
    if self_layout := getattr(compiler, "last_initial_layout", None):
        return compiled, dict(self_layout)
    if hasattr(compiler, "initial_layout"):
        return compiled, dict(compiler.initial_layout(circuit, backend))
    logical_qubits = sorted(circuit.all_qubits(), key=lambda qubit: qubit.x)
    return compiled, {qubit: qubit.x for qubit in logical_qubits}
