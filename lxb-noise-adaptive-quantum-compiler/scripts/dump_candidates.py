from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.evaluator.equivalence import LayoutSidecar, validate_compiler_equivalence
from noise_adaptive_hackathon.evaluator.submission import _passes_noiseless_semantics
from noise_adaptive_hackathon.evaluator.validate_circuit import validate_compiled_circuit
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.simulator.cirq_runner import run_track_a
from noise_adaptive_hackathon.simulator.stim_runner import run_track_b
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler, _protect_legacy_parity
from noise_adaptive_hackathon.submission.selection import candidate_metrics
from noise_adaptive_hackathon.submission.simulation_selector import (
    SELECTOR_COMMON_RANDOM,
    SELECTOR_MODES,
    SimulationSelectorConfig,
    select_candidate_with_simulation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump all validated compiler candidates for public samples.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "candidate_dump_v11.jsonl")
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--shots-a", type=int, default=256)
    parser.add_argument("--shots-b", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--selector-mode", choices=SELECTOR_MODES, default=SELECTOR_COMMON_RANDOM)
    parser.add_argument("--debug-success", choices=("selected", "all", "none"), default="selected")
    parser.add_argument("--deep-check", choices=("selected", "all", "none"), default="selected")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    samples = set(args.sample)
    with args.out.open("w", encoding="utf-8") as handle:
        for benchmark_dir in _benchmark_dirs(args.root):
            benchmark = load_benchmark(benchmark_dir)
            if samples and benchmark.name not in samples:
                continue
            backend = _load_backend(args.root, benchmark)
            compiler = NoiseAdaptiveCompiler(
                short_circuit_exact_ghz=False,
                short_circuit_legacy_parity=False,
                selector_mode=args.selector_mode,
            )
            candidates = compiler.candidate_compilations(benchmark.circuit, backend, seed=args.seed)
            selected = _protect_legacy_parity(
                select_candidate_with_simulation(
                    candidates,
                    backend,
                    benchmark.circuit,
                    seed=args.seed,
                    config=SimulationSelectorConfig(mode=args.selector_mode),
                ),
                candidates,
                backend,
                benchmark.circuit,
            )
            for index, candidate in enumerate(candidates):
                record = _candidate_record(
                    benchmark=benchmark,
                    backend=backend,
                    candidate=candidate,
                    candidate_id=f"{index:02d}_{candidate.name}",
                    chosen=candidate is selected,
                    shots=args.shots_a if benchmark.track == "track_a" else args.shots_b,
                    seed=args.seed,
                    debug_success=args.debug_success,
                    deep_check=args.deep_check,
                )
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    return 0


def _benchmark_dirs(root: Path) -> list[Path]:
    return sorted((root / "benchmarks" / "track_a" / "public").iterdir()) + sorted(
        (root / "benchmarks" / "track_b" / "public").iterdir()
    )


def _load_backend(root: Path, benchmark) -> CompetitionBackend:
    backend_path = (
        root
        / "backends"
        / benchmark.metadata["backend_size_class"]
        / f"{benchmark.metadata['backend_snapshot_id']}.json"
    )
    return CompetitionBackend.parse_raw(backend_path.read_text(encoding="utf-8"))


def _candidate_record(
    benchmark,
    backend: CompetitionBackend,
    candidate,
    candidate_id: str,
    chosen: bool,
    shots: int,
    seed: int,
    debug_success: str,
    deep_check: str,
) -> dict[str, object]:
    metrics = candidate_metrics(candidate, backend, benchmark.circuit)
    validation = validate_compiled_circuit(candidate.circuit, backend, benchmark.track)
    valid = validation.valid
    errors = list(validation.errors)
    should_deep_check = deep_check == "all" or (
        deep_check == "selected"
        and (chosen or candidate.name == "legacy_hub_parity_seed100" or candidate.name.startswith("exact_ghz"))
    )
    if valid and should_deep_check:
        layout = LayoutSidecar(initial_layout={logical.x: physical for logical, physical in candidate.layout.items()})
        equivalence = validate_compiler_equivalence(benchmark, candidate.circuit, layout, seed)
        valid = equivalence.valid
        errors.extend(equivalence.errors)
    if valid and should_deep_check and not _passes_noiseless_semantics(benchmark, candidate.circuit, seed):
        valid = False
        errors.append("noiseless semantic check failed")

    success = 0.0
    should_score = debug_success == "all" or (
        debug_success == "selected"
        and (chosen or candidate.name == "legacy_hub_parity_seed100" or candidate.name.startswith("exact_ghz"))
    )
    if valid and should_score:
        run = (
            run_track_a(benchmark, candidate.circuit, backend, shots=shots, seed=seed)
            if benchmark.track == "track_a"
            else run_track_b(benchmark, candidate.circuit, backend, shots=shots, seed=seed)
        )
        success = run.success_rate

    return {
        "sample": benchmark.name,
        "track": benchmark.track,
        "candidate_id": candidate_id,
        "strategy": _strategy(candidate.name),
        "valid": valid,
        "chosen": chosen,
        "depth": metrics.depth,
        "two_qubit_gates": metrics.two_qubit_count,
        "estimated_log_reliability": metrics.log_reliability,
        "terminal_readout_cost": metrics.readout_cost,
        "route_overhead": metrics.route_overhead,
        "clifford_sensitivity_cost": metrics.clifford_sensitivity_cost,
        "compile_time_sec": candidate.compile_time_sec,
        "public_success_debug_only": success,
        "validation_errors": errors,
    }


def _strategy(name: str) -> str:
    if name.startswith("legacy_hub"):
        return "hub_aware_ls_sa"
    if name.startswith("exact_ghz"):
        return "exact_ghz_tree_fanout"
    if name.startswith("noise_adaptive"):
        return "v10_noise_adaptive_ensemble"
    if name.startswith("cirq_cqc"):
        return "fixed_layout_route_cqc"
    if name.startswith("hub_aware"):
        return "hub_aware_layout"
    if "resynth" in name:
        return "guarded_resynthesis"
    if "relabel" in name:
        return "physical_relabeling"
    return "baseline_or_generic"


if __name__ == "__main__":
    raise SystemExit(main())
