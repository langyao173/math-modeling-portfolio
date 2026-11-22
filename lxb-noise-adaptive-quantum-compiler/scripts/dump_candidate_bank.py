from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cirq

from common import ROOT

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.evaluator.equivalence import LayoutSidecar, validate_compiler_equivalence
from noise_adaptive_hackathon.evaluator.submission import _passes_noiseless_semantics
from noise_adaptive_hackathon.evaluator.validate_circuit import (
    count_two_qubit_gates,
    estimate_log_reliability,
    validate_compiled_circuit,
)
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.simulator.cirq_runner import run_track_a
from noise_adaptive_hackathon.simulator.stim_runner import run_track_b
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler, _protect_legacy_parity
from noise_adaptive_hackathon.submission.selection import candidate_metrics, pareto_frontier, select_candidate
from noise_adaptive_hackathon.submission.track_b import is_clifford_only
from noise_adaptive_hackathon.submission.variants import ABLATION_VARIANTS, VariantCompiler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump dev-time candidate-bank features and labels.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "candidate_bank_v15_1.jsonl")
    parser.add_argument("--variant", default="v13_root", choices=("root", *ABLATION_VARIANTS))
    parser.add_argument("--split", default="public")
    parser.add_argument("--track", action="append", default=[])
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shots-a", type=int, default=256)
    parser.add_argument("--shots-b", type=int, default=1024)
    parser.add_argument("--success-labels", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    samples = set(args.sample)
    tracks = tuple(args.track or ("track_a", "track_b"))
    with args.out.open("w", encoding="utf-8") as handle:
        for benchmark_dir in _benchmark_dirs(args.root, tracks, args.split):
            benchmark = load_benchmark(benchmark_dir)
            if samples and benchmark.name not in samples:
                continue
            backend = _load_backend(args.root, benchmark)
            candidates = _candidates_for_variant(args.variant, benchmark.circuit, backend, args.seed)
            if not candidates:
                continue
            selected = select_candidate(candidates, backend, benchmark.circuit)
            selected = _protect_legacy_parity(selected, candidates, backend, benchmark.circuit)
            frontier_ids = {id(candidate) for candidate in pareto_frontier(candidates, backend, benchmark.circuit)}
            for index, candidate in enumerate(candidates):
                handle.write(
                    json.dumps(
                        _record(
                            benchmark=benchmark,
                            backend=backend,
                            candidate=candidate,
                            candidate_index=index,
                            selected=candidate is selected,
                            pareto=id(candidate) in frontier_ids,
                            seed=args.seed,
                            shots=args.shots_a if benchmark.track == "track_a" else args.shots_b,
                            success_labels=args.success_labels,
                        ),
                        sort_keys=True,
                    )
                    + "\n"
                )
    return 0


def _benchmark_dirs(root: Path, tracks: tuple[str, ...], split: str) -> list[Path]:
    dirs: list[Path] = []
    for track in tracks:
        track_dir = root / "benchmarks" / track / split
        if track_dir.exists():
            dirs.extend(sorted(path for path in track_dir.iterdir() if path.is_dir()))
    return dirs


def _load_backend(root: Path, benchmark) -> CompetitionBackend:
    backend_path = (
        root
        / "backends"
        / benchmark.metadata["backend_size_class"]
        / f"{benchmark.metadata['backend_snapshot_id']}.json"
    )
    return CompetitionBackend.parse_raw(backend_path.read_text(encoding="utf-8"))


def _candidates_for_variant(
    variant: str,
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    seed: int,
):
    if variant == "root":
        compiler = NoiseAdaptiveCompiler()
    elif variant == "v13_root":
        compiler = NoiseAdaptiveCompiler(enable_parity_resynthesis=True)
    elif variant == "v13_plus_b_relabel_only":
        compiler = NoiseAdaptiveCompiler(enable_parity_resynthesis=True, enable_track_b_relabel=True)
    elif variant == "v13_plus_tracka_exact_router_only":
        compiler = NoiseAdaptiveCompiler(enable_parity_resynthesis=True, enable_exact_small_router=True)
    elif variant == "v15_1_b002_placement":
        compiler = NoiseAdaptiveCompiler(enable_parity_resynthesis=True, enable_measurement_channel_placement=True)
    elif variant == "v15_tracka_semantic":
        compiler = NoiseAdaptiveCompiler(
            enable_parity_resynthesis=True,
            enable_exact_small_router=True,
            enable_track_a_resynthesis=True,
            enable_track_a_semantic_resynthesis=True,
            enable_track_b_relabel=True,
        )
    else:
        compiler = VariantCompiler(variant)
        compiled = compiler.compile(circuit, backend, seed=seed)
        return [type("Candidate", (), {
            "name": variant,
            "circuit": compiled,
            "layout": dict(compiler.last_initial_layout),
            "compile_time_sec": 0.0,
        })()]
    return compiler.candidate_compilations(circuit, backend, seed=seed)


def _record(
    *,
    benchmark,
    backend: CompetitionBackend,
    candidate,
    candidate_index: int,
    selected: bool,
    pareto: bool,
    seed: int,
    shots: int,
    success_labels: bool,
) -> dict[str, object]:
    metrics = candidate_metrics(candidate, backend, benchmark.circuit)
    validation = validate_compiled_circuit(candidate.circuit, backend, benchmark.track)
    equivalent = False
    noiseless = False
    errors = list(validation.errors)
    if validation.valid:
        layout = LayoutSidecar({logical.x: physical for logical, physical in candidate.layout.items()})
        equivalence = validate_compiler_equivalence(benchmark, candidate.circuit, layout, seed)
        equivalent = equivalence.valid
        errors.extend(equivalence.errors)
        noiseless = equivalent and _passes_noiseless_semantics(benchmark, candidate.circuit, seed)
    success = None
    if success_labels and validation.valid and equivalent and noiseless:
        run = (
            run_track_a(benchmark, candidate.circuit, backend, shots=shots, seed=seed)
            if benchmark.track == "track_a"
            else run_track_b(benchmark, candidate.circuit, backend, shots=shots, seed=seed)
        )
        success = run.success_rate
    return {
        "schema_version": "candidate_bank.v1",
        "benchmark_name": benchmark.name,
        "track": benchmark.track,
        "family_dev_only": benchmark.family,
        "structural_features": _structural_features(benchmark.circuit, backend),
        "candidate_index": candidate_index,
        "candidate_name": candidate.name,
        "candidate_family": _candidate_family(candidate.name),
        "candidate_hash": _circuit_hash(candidate.circuit),
        "layout": {str(logical.x): physical for logical, physical in candidate.layout.items()},
        "selected_by_runtime_proxy": selected,
        "pareto_frontier": pareto,
        "metrics": {
            "two_qubit_count": metrics.two_qubit_count,
            "depth": metrics.depth,
            "estimated_log_reliability": metrics.log_reliability,
            "terminal_readout_cost": metrics.readout_cost,
            "route_overhead": metrics.route_overhead,
            "clifford_sensitivity_cost": metrics.clifford_sensitivity_cost,
            "compile_time_sec": metrics.compile_time_sec,
        },
        "validation": {
            "basic_valid": validation.valid,
            "equivalence_valid": equivalent,
            "noiseless_valid": noiseless,
            "errors": errors,
        },
        "dev_success_label": success,
    }


def _structural_features(circuit: cirq.Circuit, backend: CompetitionBackend) -> dict[str, object]:
    ops = list(circuit.all_operations())
    twoq_errors = [backend.edge_error(left, right) for left, right in backend.coupling_map]
    readouts = [backend.readout(index) for index in range(backend.num_qubits)]
    return {
        "logical_qubits": len(circuit.all_qubits()),
        "operation_count": len(ops),
        "source_two_qubit_count": count_two_qubit_gates(circuit),
        "source_depth": len(circuit),
        "clifford_only": is_clifford_only(circuit),
        "source_estimated_log_reliability_identity": _safe_source_reliability(circuit, backend),
        "backend_num_qubits": backend.num_qubits,
        "backend_coupling_edges": len(backend.coupling_map),
        "mean_two_qubit_error": sum(twoq_errors) / len(twoq_errors) if twoq_errors else 0.0,
        "mean_readout_error": sum(readouts) / len(readouts) if readouts else 0.0,
    }


def _candidate_family(name: str) -> str:
    if name.startswith("exact_ghz"):
        return "exact_ghz"
    if name.startswith("legacy_hub"):
        return "legacy_parity"
    if "measurement" in name or name.startswith("v13"):
        return "measurement_channel"
    if "relabel" in name or "placement" in name:
        return "physical_placement"
    if "exact_small_router" in name:
        return "exact_small_router"
    if name.startswith("v14") or name.startswith("v15"):
        return "track_a_semantic"
    return "general"


def _safe_source_reliability(circuit: cirq.Circuit, backend: CompetitionBackend) -> float | None:
    try:
        return estimate_log_reliability(circuit, backend)
    except Exception:
        return None


def _circuit_hash(circuit: cirq.Circuit) -> str:
    payload = "\n".join(str(op) for op in circuit.all_operations()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
