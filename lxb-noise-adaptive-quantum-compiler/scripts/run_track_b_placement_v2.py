from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cirq

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.evaluator.validate_circuit import (
    count_two_qubit_gates,
    estimate_log_reliability,
)
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.simulator.stim_runner import run_track_b
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.ghz_leaf_tree import (
    _candidate_leaf_trees,
    _cnot_linear_equivalent,
    _compile_leaf_tree,
)
from noise_adaptive_hackathon.submission.ghz_tree import detect_exact_ghz_star
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
from noise_adaptive_hackathon.submission.selection import (
    CompilationCandidate,
    candidate_metrics,
    terminal_readout_cost,
)
from noise_adaptive_hackathon.submission.simulation_selector import (
    SELECTOR_TRACK_A_COMMON_RANDOM,
    SimulationSelectorConfig,
)
from noise_adaptive_hackathon.submission.validation import validate_or_raise


OUT_JSON = ROOT / "experiments" / "track_b_placement_v2_candidates.json"
OUT_REPORT = ROOT / "experiments" / "track_b_placement_v2_report.md"
OUT_COMPARE = ROOT / "experiments" / "root_vs_bplacement_v2_seed7.md"


BAD_B001_EDGES = {tuple(sorted(edge)) for edge in ((5, 14), (11, 12), (14, 15), (4, 12))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dev-time Track B placement/readout v2 search.")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--b001-limit", type=int, default=80)
    parser.add_argument("--b002-limit", type=int, default=160)
    parser.add_argument("--simulate-top", type=int, default=24)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-report", type=Path, default=OUT_REPORT)
    parser.add_argument("--out-compare", type=Path, default=OUT_COMPARE)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    payload = {
        "variant": "track_b_placement_readout_v2_dev_only",
        "seed": args.seed,
        "shots": args.shots,
        "samples": {},
    }
    for sample in ("track_b_001", "track_b_002"):
        benchmark = load_benchmark(ROOT / "benchmarks" / "track_b" / "public" / sample)
        backend = _load_backend(benchmark)
        root_candidate = _compile_root_candidate(benchmark.circuit, backend, seed=args.seed)
        if sample == "track_b_001":
            candidates = _b001_candidates(benchmark.circuit, backend, root_candidate, limit=args.b001_limit)
        else:
            candidates = _b002_candidates(benchmark.circuit, backend, root_candidate, limit=args.b002_limit)
        candidates = _dedupe_candidates([root_candidate, *candidates])
        ranked = sorted(
            candidates,
            key=lambda candidate: _placement_rank(candidate, backend, benchmark.circuit, sample),
            reverse=True,
        )
        simulated = []
        simulate_set = [root_candidate]
        for candidate in ranked:
            if candidate.name == "root_current":
                continue
            simulate_set.append(candidate)
            if len(simulate_set) >= args.simulate_top:
                break
        for candidate in simulate_set:
            simulated.append(_simulate_candidate(benchmark, backend, candidate, seed=args.seed, shots=args.shots))
        payload["samples"][sample] = {
            "family": benchmark.family,
            "candidate_count": len(candidates),
            "simulated_count": len(simulated),
            "root": next(item for item in simulated if item["name"] == "root_current"),
            "best_by_success": sorted(
                simulated,
                key=lambda item: (item["success_rate"], item["estimated_log_reliability"], -item["cx"], item["name"]),
                reverse=True,
            )[0],
            "top_candidates": sorted(
                simulated,
                key=lambda item: (item["success_rate"], item["estimated_log_reliability"], -item["cx"], item["name"]),
                reverse=True,
            ),
        }

    payload["elapsed_sec"] = time.perf_counter() - started
    payload["seed7_gate"] = _seed7_gate(payload)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_report.write_text(_render_report(payload), encoding="utf-8")
    args.out_compare.write_text(_render_compare(payload), encoding="utf-8")
    print(json.dumps(payload["seed7_gate"], indent=2, sort_keys=True))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_report}")
    print(f"wrote {args.out_compare}")
    return 0


def _current_root_compiler() -> NoiseAdaptiveCompiler:
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


def _compile_root_candidate(source: cirq.Circuit, backend: CompetitionBackend, *, seed: int) -> CompilationCandidate:
    compiler = _current_root_compiler()
    started = time.perf_counter()
    compiled = compiler.compile(source, backend, seed=seed)
    return CompilationCandidate("root_current", compiled, dict(compiler.last_initial_layout), time.perf_counter() - started)


def _b001_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    root: CompilationCandidate,
    *,
    limit: int,
) -> list[CompilationCandidate]:
    ghz = detect_exact_ghz_star(source)
    if ghz is None:
        return []
    out: list[CompilationCandidate] = []
    started = time.perf_counter()
    trees = _candidate_leaf_trees(backend, len(ghz.logical_qubits))
    extra = _candidate_leaf_trees(backend, len(ghz.logical_qubits))
    for index, tree in enumerate([*trees, *extra]):
        compiled, layout = _compile_leaf_tree(backend, ghz, tree)
        if not _cnot_linear_equivalent(source, compiled, layout, ghz):
            continue
        try:
            validate_or_raise(source, compiled, backend)
        except Exception:
            continue
        candidate = CompilationCandidate(
            f"b001_leaf_tree_{index:03d}",
            compiled,
            layout,
            time.perf_counter() - started,
        )
        if _bad_edge_count(candidate.circuit) > 1:
            continue
        out.append(candidate)
        for copyout in readout_copyout_candidates(source, backend, [candidate], limit=1, base_limit=1):
            out.append(
                CompilationCandidate(
                    f"{candidate.name}_copyout",
                    copyout.circuit,
                    copyout.layout,
                    copyout.compile_time_sec,
                )
            )
        if len(out) >= limit:
            break
    # Keep the existing root in the comparison set, but do not emit clones that are
    # clearly worse on both ELR and readout.
    root_metrics = candidate_metrics(root, backend, source)
    return [
        candidate
        for candidate in out
        if (
            candidate_metrics(candidate, backend, source).log_reliability >= root_metrics.log_reliability - 0.06
            or candidate_metrics(candidate, backend, source).readout_cost <= root_metrics.readout_cost - 0.015
        )
    ][:limit]


def _b002_candidates(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    root: CompilationCandidate,
    *,
    limit: int,
) -> list[CompilationCandidate]:
    star = detect_affine_parity_star(source)
    if star is None:
        return []
    started = time.perf_counter()
    layouts = _enumerate_star_layouts(
        star,
        backend,
        max_seconds=20.0,
        limit=limit,
        per_hub_option_limit=192,
        beam_width=2048,
    )
    out: list[CompilationCandidate] = []
    root_metrics = candidate_metrics(root, backend, source)
    for index, layout in enumerate(layouts):
        compiled = _build_physical_star_circuit(star, layout)
        try:
            validate_or_raise(source, compiled, backend)
        except Exception:
            continue
        if not measured_affine_channels_equivalent(source, compiled, layout):
            continue
        candidate = CompilationCandidate(
            f"b002_affine_star_{index:03d}",
            compiled,
            dict(layout),
            time.perf_counter() - started,
        )
        metrics = candidate_metrics(candidate, backend, source)
        if metrics.two_qubit_count > root_metrics.two_qubit_count:
            continue
        if metrics.depth > root_metrics.depth + 1:
            continue
        if metrics.log_reliability < root_metrics.log_reliability - 0.08:
            continue
        out.append(candidate)
        for copyout in readout_copyout_candidates(source, backend, [candidate], limit=1, base_limit=1):
            try:
                validate_or_raise(source, copyout.circuit, backend)
            except Exception:
                continue
            out.append(
                CompilationCandidate(
                    f"{candidate.name}_copyout",
                    copyout.circuit,
                    dict(copyout.layout),
                    copyout.compile_time_sec,
                )
            )
        if len(out) >= limit:
            break
    return out[:limit]


def _simulate_candidate(
    benchmark,
    backend: CompetitionBackend,
    candidate: CompilationCandidate,
    *,
    seed: int,
    shots: int,
) -> dict[str, Any]:
    run = run_track_b(benchmark, candidate.circuit, backend, shots=shots, seed=seed + 10_000 + _track_b_index(benchmark.name))
    metrics = candidate_metrics(candidate, backend, benchmark.circuit)
    return {
        "name": candidate.name,
        "success_rate": run.success_rate,
        "correct_shots": run.correct_shots,
        "shots": shots,
        "cx": metrics.two_qubit_count,
        "depth": metrics.depth,
        "estimated_log_reliability": metrics.log_reliability,
        "terminal_readout_cost": metrics.readout_cost,
        "sensitivity": metrics.clifford_sensitivity_cost,
        "compile_time_sec": candidate.compile_time_sec,
        "bad_b001_edge_count": _bad_edge_count(candidate.circuit),
        "measurement_qubits": _measurement_qubits(candidate.circuit),
        "readout_sum": _readout_sum(candidate.circuit, backend),
        "edge_summary": _edge_summary(candidate.circuit, backend),
        "layout": {str(logical.x): physical for logical, physical in sorted(candidate.layout.items(), key=lambda item: item[0].x)},
    }


def _seed7_gate(payload: dict[str, Any]) -> dict[str, Any]:
    b001 = payload["samples"]["track_b_001"]
    b002 = payload["samples"]["track_b_002"]
    root_b001 = b001["root"]["success_rate"]
    root_b002 = b002["root"]["success_rate"]
    best_b001 = b001["best_by_success"]["success_rate"]
    best_b002 = b002["best_by_success"]["success_rate"]
    track_b_root = (root_b001 + root_b002) / 2.0
    track_b_best = (best_b001 + best_b002) / 2.0
    track_b_delta = track_b_best - track_b_root
    final_delta = 0.25 * track_b_delta
    return {
        "b001_delta": best_b001 - root_b001,
        "b002_delta": best_b002 - root_b002,
        "track_b_delta": track_b_delta,
        "final_delta": final_delta,
        "raw_success_plus_0_03": (best_b001 - root_b001 >= 0.03) or (best_b002 - root_b002 >= 0.03),
        "track_b_plus_0_015": track_b_delta >= 0.015,
        "final_plus_0_0035": final_delta >= 0.0035,
        "keep": (
            ((best_b001 - root_b001 >= 0.03) or (best_b002 - root_b002 >= 0.03))
            and track_b_delta >= 0.015
            and final_delta >= 0.0035
        ),
    }


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Track B Placement/Readout V2 Report",
        "",
        "This is a development-only search. It does not modify root `compiler.py`.",
        "",
        f"- Seed: `{payload['seed']}`",
        f"- Shots per candidate: `{payload['shots']}`",
        f"- Keep decision: `{payload['seed7_gate']['keep']}`",
        f"- Track B delta from best per-sample candidates: `{payload['seed7_gate']['track_b_delta']:.6f}`",
        f"- Final delta estimate: `{payload['seed7_gate']['final_delta']:.6f}`",
        "",
    ]
    for sample, data in payload["samples"].items():
        root = data["root"]
        best = data["best_by_success"]
        lines.extend(
            [
                f"## {sample}",
                "",
                f"- Candidates/simulated: `{data['candidate_count']}` / `{data['simulated_count']}`",
                f"- Root: success `{root['success_rate']:.6f}`, CX `{root['cx']}`, depth `{root['depth']}`, ELR `{root['estimated_log_reliability']:.6f}`, readout `{root['terminal_readout_cost']:.6f}`.",
                f"- Best: `{best['name']}`, success `{best['success_rate']:.6f}`, CX `{best['cx']}`, depth `{best['depth']}`, ELR `{best['estimated_log_reliability']:.6f}`, readout `{best['terminal_readout_cost']:.6f}`.",
                "",
                "| Candidate | Success | CX | Depth | ELR | Readout | Bad B001 Edges |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in data["top_candidates"][:10]:
            lines.append(
                f"| `{row['name']}` | {row['success_rate']:.6f} | {row['cx']} | {row['depth']} | {row['estimated_log_reliability']:.6f} | {row['terminal_readout_cost']:.6f} | {row['bad_b001_edge_count']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            "Do not merge unless all keep-gate thresholds pass and compile tail is acceptable.",
        ]
    )
    return "\n".join(lines)


def _render_compare(payload: dict[str, Any]) -> str:
    gate = payload["seed7_gate"]
    lines = [
        "# Root vs Track B Placement V2 Seed7",
        "",
        "| Metric | Value | Pass |",
        "|---|---:|---:|",
        f"| B001 best raw delta | {gate['b001_delta']:.6f} | {gate['b001_delta'] >= 0.03} |",
        f"| B002 best raw delta | {gate['b002_delta']:.6f} | {gate['b002_delta'] >= 0.03} |",
        f"| Track B delta | {gate['track_b_delta']:.6f} | {gate['track_b_plus_0_015']} |",
        f"| Final delta estimate | {gate['final_delta']:.6f} | {gate['final_plus_0_0035']} |",
        f"| Keep | {1 if gate['keep'] else 0} | {gate['keep']} |",
        "",
    ]
    for sample, data in payload["samples"].items():
        root = data["root"]
        best = data["best_by_success"]
        lines.extend(
            [
                f"## {sample}",
                "",
                f"- Root candidate: `{root['name']}` success `{root['success_rate']:.6f}`.",
                f"- Best candidate: `{best['name']}` success `{best['success_rate']:.6f}`.",
                f"- Delta: `{best['success_rate'] - root['success_rate']:.6f}`.",
                "",
            ]
        )
    return "\n".join(lines)


def _placement_rank(candidate: CompilationCandidate, backend: CompetitionBackend, source: cirq.Circuit, sample: str) -> tuple:
    metrics = candidate_metrics(candidate, backend, source)
    bad_penalty = _bad_edge_count(candidate.circuit) if sample == "track_b_001" else 0
    return (
        metrics.log_reliability
        - 0.04 * metrics.readout_cost
        - 0.006 * metrics.clifford_sensitivity_cost
        - 0.02 * bad_penalty,
        -metrics.two_qubit_count,
        -metrics.depth,
        -metrics.compile_time_sec,
        candidate.name,
    )


def _load_backend(benchmark) -> CompetitionBackend:
    return CompetitionBackend.parse_raw(
        (
            ROOT
            / "backends"
            / benchmark.metadata["backend_size_class"]
            / f"{benchmark.metadata['backend_snapshot_id']}.json"
        ).read_text(encoding="utf-8")
    )


def _dedupe_candidates(candidates: list[CompilationCandidate]) -> list[CompilationCandidate]:
    out: list[CompilationCandidate] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        signature = tuple(str(op) for op in candidate.circuit.all_operations())
        if signature in seen:
            continue
        seen.add(signature)
        out.append(candidate)
    return out


def _measurement_qubits(circuit: cirq.Circuit) -> list[int]:
    measurements = [op for op in circuit.all_operations() if isinstance(op.gate, cirq.MeasurementGate)]
    if not measurements:
        return []
    return [qubit.x for qubit in measurements[-1].qubits]


def _readout_sum(circuit: cirq.Circuit, backend: CompetitionBackend) -> float:
    return sum(backend.readout(qubit) for qubit in _measurement_qubits(circuit))


def _edge_summary(circuit: cirq.Circuit, backend: CompetitionBackend) -> list[dict[str, Any]]:
    counts: dict[tuple[int, int], int] = {}
    for op in circuit.all_operations():
        if len(op.qubits) != 2:
            continue
        edge = tuple(sorted((op.qubits[0].x, op.qubits[1].x)))
        counts[edge] = counts.get(edge, 0) + 1
    return [
        {"edge": list(edge), "count": count, "error": backend.edge_error(*edge)}
        for edge, count in sorted(counts.items(), key=lambda item: (-item[1], -backend.edge_error(*item[0]), item[0]))
    ]


def _bad_edge_count(circuit: cirq.Circuit) -> int:
    total = 0
    for op in circuit.all_operations():
        if len(op.qubits) != 2:
            continue
        if tuple(sorted((op.qubits[0].x, op.qubits[1].x))) in BAD_B001_EDGES:
            total += 1
    return total


def _track_b_index(name: str) -> int:
    return 0 if name.endswith("_001") else 1


if __name__ == "__main__":
    raise SystemExit(main())
