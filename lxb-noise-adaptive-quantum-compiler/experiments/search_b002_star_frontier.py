from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cirq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.evaluator.validate_circuit import (
    count_two_qubit_gates,
    estimate_log_reliability,
)
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.simulator.stim_runner import run_track_b
from noise_adaptive_hackathon.submission.measurement_channel import (
    _build_physical_star_circuit,
    detect_affine_parity_star,
)
from noise_adaptive_hackathon.submission.measurement_channel_general import (
    measured_affine_channels_equivalent,
)
from noise_adaptive_hackathon.submission.measurement_channel_physical_placement_v2 import (
    _enumerate_star_layouts,
)
from noise_adaptive_hackathon.submission.readout_copyout import readout_copyout_candidates
from noise_adaptive_hackathon.submission.selection import CompilationCandidate, candidate_metrics
from noise_adaptive_hackathon.submission.validation import validate_or_raise
from noise_adaptive_hackathon.submission.variants import VariantCompiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline B002 affine-star placement frontier search."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmarks/track_b/public/track_b_002"),
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/b002_star_frontier_search.json"))
    parser.add_argument("--layout-limit", type=int, default=120)
    parser.add_argument("--simulate-top", type=int, default=60)
    parser.add_argument("--layout-seconds", type=float, default=18.0)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed-bases", default="7,11,17,23,31")
    parser.add_argument("--per-hub-option-limit", type=int, default=128)
    parser.add_argument("--beam-width", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    benchmark_dir = (root / args.benchmark).resolve()
    benchmark = load_benchmark(benchmark_dir)
    backend_path = (
        root
        / "backends"
        / benchmark.metadata["backend_size_class"]
        / f"{benchmark.metadata['backend_snapshot_id']}.json"
    )
    backend = CompetitionBackend.parse_raw(backend_path.read_text(encoding="utf-8"))
    star = detect_affine_parity_star(benchmark.circuit)
    if star is None:
        raise SystemExit("benchmark is not an affine parity star")

    started = time.perf_counter()
    layouts = _enumerate_star_layouts(
        star,
        backend,
        max_seconds=args.layout_seconds,
        limit=args.layout_limit,
        per_hub_option_limit=args.per_hub_option_limit,
        beam_width=args.beam_width,
    )
    candidates: list[CompilationCandidate] = []
    rejected: dict[str, int] = {}
    seen: set[tuple[str, ...]] = set()

    def add_candidate(candidate: CompilationCandidate, *, require_channel: bool) -> None:
        signature = tuple(str(op) for op in candidate.circuit.all_operations())
        if signature in seen:
            rejected["duplicate"] = rejected.get("duplicate", 0) + 1
            return
        seen.add(signature)
        try:
            validate_or_raise(benchmark.circuit, candidate.circuit, backend)
        except Exception:
            rejected["validator"] = rejected.get("validator", 0) + 1
            return
        if require_channel:
            if not measured_affine_channels_equivalent(
                benchmark.circuit,
                candidate.circuit,
                candidate.layout,
            ):
                rejected["channel"] = rejected.get("channel", 0) + 1
                return
        candidates.append(candidate)

    try:
        v27 = VariantCompiler("v27_b002_frontier_copyout")
        current = v27.compile(benchmark.circuit, backend, seed=7)
        add_candidate(
            CompilationCandidate(
                "current_v27_b002_frontier_copyout",
                current,
                dict(v27.last_initial_layout),
                time.perf_counter() - started,
            ),
            require_channel=False,
        )
    except Exception as error:
        rejected[f"current_v27_{type(error).__name__}"] = 1

    for index, layout in enumerate(layouts):
        circuit = _build_physical_star_circuit(star, layout)
        add_candidate(
            CompilationCandidate(
                f"offline_star_{index:03d}",
                circuit,
                dict(layout),
                time.perf_counter() - started,
            ),
            require_channel=True,
        )

    for base in list(candidates):
        for copyout in readout_copyout_candidates(
            benchmark.circuit,
            backend,
            [base],
            limit=1,
            base_limit=1,
        ):
            add_candidate(copyout, require_channel=False)

    ranked = sorted(
        candidates,
        key=lambda candidate: _ranking_key(candidate, backend, benchmark.circuit),
        reverse=True,
    )
    simulated = ranked[: args.simulate_top]
    seed_bases = [int(item) for item in args.seed_bases.split(",") if item.strip()]
    results = []
    for candidate in simulated:
        metrics = candidate_metrics(candidate, backend, benchmark.circuit)
        successes = []
        for seed_base in seed_bases:
            # Public evaluator offsets Track B by 10_000 and B002 is the second
            # sorted public Track B sample, so use +10_001 for comparable noise.
            seed = seed_base + 10_001
            run = run_track_b(
                benchmark,
                candidate.circuit,
                backend,
                shots=args.shots,
                seed=seed,
            )
            successes.append(run.success_rate)
        results.append(
            {
                "name": candidate.name,
                "cx": metrics.two_qubit_count,
                "depth": metrics.depth,
                "elr": metrics.log_reliability,
                "readout_cost": metrics.readout_cost,
                "sensitivity": metrics.clifford_sensitivity_cost,
                "mean_success": statistics.fmean(successes),
                "min_success": min(successes),
                "max_success": max(successes),
                "successes": dict(zip([str(seed) for seed in seed_bases], successes)),
                "measurement_qubits": _measurement_qubits(candidate.circuit),
                "layout": {str(logical.x): physical for logical, physical in sorted(candidate.layout.items())},
            }
        )

    payload = {
        "benchmark": benchmark.name,
        "shots": args.shots,
        "seed_bases": seed_bases,
        "layout_count": len(layouts),
        "candidate_count": len(candidates),
        "simulated_count": len(results),
        "rejected": rejected,
        "elapsed_sec": time.perf_counter() - started,
        "top_by_success": sorted(
            results,
            key=lambda item: (
                item["mean_success"],
                item["min_success"],
                item["elr"],
                -item["cx"],
                item["name"],
            ),
            reverse=True,
        )[:20],
        "top_by_proxy": results[:20],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["top_by_success"][:5], indent=2))
    print(f"wrote {args.out}")
    return 0


def _ranking_key(candidate: CompilationCandidate, backend: CompetitionBackend, source: cirq.Circuit) -> tuple:
    metrics = candidate_metrics(candidate, backend, source)
    return (
        metrics.log_reliability - 0.04 * metrics.readout_cost - 0.004 * metrics.clifford_sensitivity_cost,
        -metrics.two_qubit_count,
        -metrics.depth,
        -metrics.readout_cost,
        candidate.name,
    )


def _measurement_qubits(circuit: cirq.Circuit) -> list[int]:
    measurements = [op for op in circuit.all_operations() if isinstance(op.gate, cirq.MeasurementGate)]
    if not measurements:
        return []
    return [qubit.x for qubit in measurements[-1].qubits]


if __name__ == "__main__":
    raise SystemExit(main())
