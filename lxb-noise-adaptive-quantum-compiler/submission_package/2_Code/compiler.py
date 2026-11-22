from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cirq

from noise_adaptive_hackathon.evaluator.equivalence import write_layout_sidecar
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.simulation_selector import (
    SELECTOR_TRACK_A_COMMON_RANDOM,
    SimulationSelectorConfig,
)


FIXED_SEED = 7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Noise-adaptive quantum compiler submission.")
    parser.add_argument("--circuit", required=True, type=Path)
    parser.add_argument("--backend", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    circuit = cirq.read_json(args.circuit)
    backend = CompetitionBackend.parse_raw(args.backend.read_text(encoding="utf-8"))

    compiler = NoiseAdaptiveCompiler(
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
    compiled = compiler.compile(circuit, backend, seed=FIXED_SEED)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cirq.to_json(compiled, args.out)
    write_layout_sidecar(args.out, compiler.last_initial_layout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
