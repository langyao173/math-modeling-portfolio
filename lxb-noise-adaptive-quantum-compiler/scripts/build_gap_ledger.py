from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
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
    gate_name_for_operation,
)
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.selection import candidate_metrics, terminal_readout_cost
from noise_adaptive_hackathon.submission.simulation_selector import (
    SELECTOR_TRACK_A_COMMON_RANDOM,
    SimulationSelectorConfig,
)


DEFAULT_SUMMARY = ROOT / "experiments" / "public_summary_promoted_root_seed7.json"
DEFAULT_HEADROOM = ROOT / "experiments" / "readout_ceiling_promoted_root_seed7_headroom.json"
DEFAULT_JSON = ROOT / "experiments" / "gap_ledger_current_root.json"
DEFAULT_MD = ROOT / "experiments" / "gap_ledger_current_root.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build per-sample score-loss gap ledger for the current root compiler.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--headroom", type=Path, default=DEFAULT_HEADROOM)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    summary = _read_json(args.summary)
    headroom = _read_json(args.headroom)
    ledger = build_gap_ledger(summary=summary, headroom=headroom, seed=args.seed)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render_markdown(ledger), encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


def build_gap_ledger(*, summary: dict[str, Any], headroom: dict[str, Any], seed: int) -> dict[str, Any]:
    summary_by_name = {item["benchmark_name"]: item for item in summary["sample_results"]}
    cap_by_name = {item["benchmark"]: item for item in headroom["sample_headroom"]}
    row_phase_usage = _load_row_phase_usage()
    route_scar_usage = _load_route_scar_usage()

    samples: list[dict[str, Any]] = []
    for benchmark_dir in _public_benchmark_dirs():
        benchmark = load_benchmark(benchmark_dir)
        backend = _load_backend_for_benchmark(benchmark)
        compiler = _current_root_compiler()

        started = time.perf_counter()
        compiled = compiler.compile(benchmark.circuit, backend, seed=seed)
        elapsed = time.perf_counter() - started

        candidate_names: list[str] = []
        selected_candidate = "not_matched_after_tail_or_copyout"
        try:
            probe = _current_root_compiler()
            candidates = probe.candidate_compilations(benchmark.circuit, backend, seed=seed)
            candidate_names = [candidate.name for candidate in candidates]
            final_signature = _circuit_signature(compiled)
            for candidate in candidates:
                if _circuit_signature(candidate.circuit) == final_signature:
                    selected_candidate = candidate.name
                    break
        except Exception as exc:
            candidate_names = [f"candidate_probe_failed:{type(exc).__name__}"]

        summary_item = summary_by_name[benchmark.name]
        cap_item = cap_by_name[benchmark.name]
        edge_stats = _edge_stats(compiled, backend)
        readout_stats = _readout_stats(compiled, backend)
        op_counts = _op_counts(compiled)
        source_op_counts = _op_counts(benchmark.circuit)
        log_reliability = estimate_log_reliability(compiled, backend)
        readout_cost = terminal_readout_cost(compiled, backend)
        single_qubit_cost = _single_qubit_cost(compiled, backend)
        two_qubit_cost = edge_stats["twoq_log_cost"]
        total_positive_log_cost = -log_reliability
        sample_weight = 0.75 / 8.0 if benchmark.track == "track_a" else 0.25 / 2.0
        cap = float(cap_item["cap"])
        current = float(summary_item["success_rate"])
        sample = {
            "benchmark": benchmark.name,
            "track": benchmark.track,
            "family": benchmark.family,
            "width": len(benchmark.logical_qubits),
            "readout_only_upper_bound": cap,
            "current_success": current,
            "raw_headroom_to_readout_cap": cap - current,
            "weighted_final_headroom": (cap - current) * sample_weight,
            "sample_weight": sample_weight,
            "valid": bool(summary_item["valid"]),
            "shots": int(summary_item["shots"]),
            "correct_shots": int(summary_item["correct_shots"]),
            "source_cx_count": int(source_op_counts.get("cx", 0)),
            "current_cx_count": count_two_qubit_gates(compiled),
            "current_depth": len(compiled),
            "summary_depth": int(summary_item["depth"]),
            "estimated_log_reliability": log_reliability,
            "summary_estimated_log_reliability": float(summary_item["estimated_log_reliability"]),
            "positive_log_cost_total": total_positive_log_cost,
            "twoq_log_cost": two_qubit_cost,
            "singleq_log_cost": single_qubit_cost,
            "terminal_readout_cost": readout_cost,
            "log_cost_share": _cost_share(total_positive_log_cost, two_qubit_cost, single_qubit_cost, readout_cost),
            "used_edges": edge_stats,
            "measurement_readout": readout_stats,
            "op_counts": dict(op_counts),
            "source_op_counts": dict(source_op_counts),
            "compile_time_sec": float(summary_item["compile_time_sec"]),
            "fresh_compile_time_sec": elapsed,
            "selected_candidate": selected_candidate,
            "candidate_count": len(candidate_names),
            "candidate_families": _candidate_families(candidate_names),
            "semantic_rewrite_triggered": _semantic_markers(
                benchmark_name=benchmark.name,
                track=benchmark.track,
                family=benchmark.family,
                candidate_names=candidate_names,
                row_phase_usage=row_phase_usage,
                route_scar_usage=route_scar_usage,
            ),
            "recommended_next_action": _recommended_next_action(
                track=benchmark.track,
                benchmark_name=benchmark.name,
                headroom=cap - current,
                cx_count=count_two_qubit_gates(compiled),
                readout_cost=readout_cost,
                twoq_cost=two_qubit_cost,
            ),
        }
        samples.append(sample)

    samples.sort(key=lambda item: item["weighted_final_headroom"], reverse=True)
    track_a_samples = [item for item in samples if item["track"] == "track_a"]
    track_b_samples = [item for item in samples if item["track"] == "track_b"]
    return {
        "variant": "current_promoted_root",
        "seed": seed,
        "summary_scores": {
            "track_a": summary["track_a_score"],
            "track_b": summary["track_b_score"],
            "final": summary["final_score"],
            "invalid_count": sum(1 for item in summary["sample_results"] if not item["valid"]),
        },
        "readout_only_caps": {
            "track_a": headroom["track_a_cap"],
            "track_b": headroom["track_b_cap"],
            "final": headroom["final_cap"],
        },
        "aggregate_headroom": {
            "track_a": sum(item["weighted_final_headroom"] for item in track_a_samples),
            "track_b": sum(item["weighted_final_headroom"] for item in track_b_samples),
            "final": sum(item["weighted_final_headroom"] for item in samples),
        },
        "largest_residuals": [
            {
                "benchmark": item["benchmark"],
                "track": item["track"],
                "weighted_final_headroom": item["weighted_final_headroom"],
                "raw_headroom_to_readout_cap": item["raw_headroom_to_readout_cap"],
                "current_success": item["current_success"],
                "readout_only_upper_bound": item["readout_only_upper_bound"],
                "current_cx_count": item["current_cx_count"],
                "current_depth": item["current_depth"],
                "twoq_log_cost": item["twoq_log_cost"],
                "terminal_readout_cost": item["terminal_readout_cost"],
                "recommended_next_action": item["recommended_next_action"],
            }
            for item in samples[:6]
        ],
        "samples": samples,
    }


def render_markdown(ledger: dict[str, Any]) -> str:
    lines = [
        "# Current Root Gap Ledger",
        "",
        f"- Variant: `{ledger['variant']}`",
        f"- Seed: `{ledger['seed']}`",
        f"- Current Final: `{ledger['summary_scores']['final']:.12f}`",
        f"- Current Track A: `{ledger['summary_scores']['track_a']:.12f}`",
        f"- Current Track B: `{ledger['summary_scores']['track_b']:.12f}`",
        f"- Invalid count: `{ledger['summary_scores']['invalid_count']}`",
        f"- Readout-only Final cap: `{ledger['readout_only_caps']['final']:.12f}`",
        f"- Remaining weighted Final headroom to cap: `{ledger['aggregate_headroom']['final']:.12f}`",
        "",
        "## Largest Residuals",
        "",
        "| Rank | Sample | Track | Cap | Current | Raw Gap | Weighted Final Gap | CX | Depth | 2Q Cost | Readout Cost | Next action |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, item in enumerate(ledger["samples"], 1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    item["benchmark"],
                    item["track"],
                    f"{item['readout_only_upper_bound']:.6f}",
                    f"{item['current_success']:.6f}",
                    f"{item['raw_headroom_to_readout_cap']:.6f}",
                    f"{item['weighted_final_headroom']:.6f}",
                    str(item["current_cx_count"]),
                    str(item["current_depth"]),
                    f"{item['twoq_log_cost']:.6f}",
                    f"{item['terminal_readout_cost']:.6f}",
                    item["recommended_next_action"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-Sample Physical Ledger",
            "",
        ]
    )
    for item in ledger["samples"]:
        edge = item["used_edges"]
        meas = item["measurement_readout"]
        markers = ", ".join(item["semantic_rewrite_triggered"]) or "none"
        families = ", ".join(item["candidate_families"]) or "none"
        lines.extend(
            [
                f"### {item['benchmark']}",
                "",
                f"- Family: `{item['family']}`, width `{item['width']}`.",
                f"- Success/cap/gap: `{item['current_success']:.6f}` / `{item['readout_only_upper_bound']:.6f}` / `{item['raw_headroom_to_readout_cap']:.6f}`.",
                f"- CX/depth/ELR: `{item['current_cx_count']}` / `{item['current_depth']}` / `{item['estimated_log_reliability']:.6f}`.",
                f"- Log-cost split: 2Q `{item['twoq_log_cost']:.6f}`, 1Q `{item['singleq_log_cost']:.6f}`, readout `{item['terminal_readout_cost']:.6f}`.",
                f"- Used edges: `{edge['unique_edge_count']}` unique, worst `{edge['worst_edge']}`, hottest `{edge['hottest_edges'][:5]}`.",
                f"- Measurement qubits: `{meas['qubits']}`, readout sum `{meas['readout_error_sum']:.6f}`, max readout `{meas['max_readout_error']:.6f}`.",
                f"- Selected candidate: `{item['selected_candidate']}`; candidate families: `{families}`.",
                f"- Semantic markers: `{markers}`.",
                f"- Recommendation: {item['recommended_next_action']}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "- The ledger ranks residuals by final-score weight, not by raw success alone.",
            "- Track B residuals are now mainly physical/readout/edge placement residuals because the final-Z measurement-channel semantics are already compressed.",
            "- Track A residuals that still have high CX and high headroom should be handled by semantic row-phase/Peres/carry-style rewrites, not generic routing passes.",
            "- The readout-only cap is a ceiling for a zero-gate legal implementation, not a reachable promise for circuits that still require noisy CX gates.",
            "",
        ]
    )
    return "\n".join(lines)


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


def _public_benchmark_dirs() -> list[Path]:
    dirs: list[Path] = []
    for track in ("track_a", "track_b"):
        root = ROOT / "benchmarks" / track / "public"
        dirs.extend(sorted(path for path in root.iterdir() if path.is_dir()))
    return dirs


def _load_backend_for_benchmark(benchmark) -> CompetitionBackend:
    size_class = benchmark.metadata.get("backend_size_class", "medium")
    snapshot_id = benchmark.metadata.get("backend_snapshot_id", "medium_snapshot_a")
    path = ROOT / "backends" / size_class / f"{snapshot_id}.json"
    return CompetitionBackend.parse_raw(path.read_text(encoding="utf-8"))


def _edge_stats(circuit: cirq.Circuit, backend: CompetitionBackend) -> dict[str, Any]:
    counts: Counter[tuple[int, int]] = Counter()
    directed_counts: Counter[tuple[int, int]] = Counter()
    total_log_cost = 0.0
    total_error = 0.0
    for op in circuit.all_operations():
        if len(op.qubits) != 2:
            continue
        left, right = op.qubits[0].x, op.qubits[1].x
        directed_counts[(left, right)] += 1
        edge = tuple(sorted((left, right)))
        counts[edge] += 1
        error = backend.edge_error(left, right)
        total_error += error
        total_log_cost += -math.log(max(1.0 - error, 1e-12))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -backend.edge_error(*item[0]), item[0]))
    worst_edge = None
    if counts:
        worst = max(counts, key=lambda edge: backend.edge_error(*edge))
        worst_edge = {
            "edge": list(worst),
            "count": counts[worst],
            "error": backend.edge_error(*worst),
        }
    return {
        "twoq_log_cost": total_log_cost,
        "twoq_error_sum": total_error,
        "mean_twoq_error_per_gate": total_error / sum(counts.values()) if counts else 0.0,
        "unique_edge_count": len(counts),
        "edge_counts": [
            {"edge": list(edge), "count": count, "error": backend.edge_error(*edge)}
            for edge, count in ranked
        ],
        "directed_edge_counts": [
            {"edge": list(edge), "count": count, "error": backend.edge_error(*edge)}
            for edge, count in sorted(directed_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "hottest_edges": [
            {"edge": list(edge), "count": count, "error": backend.edge_error(*edge)}
            for edge, count in ranked[:8]
        ],
        "worst_edge": worst_edge,
    }


def _readout_stats(circuit: cirq.Circuit, backend: CompetitionBackend) -> dict[str, Any]:
    measurements = [op for op in circuit.all_operations() if isinstance(op.gate, cirq.MeasurementGate)]
    if not measurements:
        return {
            "qubits": [],
            "readout_errors": [],
            "readout_error_sum": 0.0,
            "max_readout_error": 0.0,
            "readout_log_cost": 0.0,
        }
    final = measurements[-1]
    qubits = [qubit.x for qubit in final.qubits]
    errors = [backend.readout(qubit) for qubit in qubits]
    return {
        "qubits": qubits,
        "readout_errors": errors,
        "readout_error_sum": sum(errors),
        "mean_readout_error": sum(errors) / len(errors) if errors else 0.0,
        "max_readout_error": max(errors) if errors else 0.0,
        "readout_log_cost": sum(-math.log(max(1.0 - error, 1e-12)) for error in errors),
    }


def _single_qubit_cost(circuit: cirq.Circuit, backend: CompetitionBackend) -> float:
    total = 0.0
    for op in circuit.all_operations():
        if len(op.qubits) == 1 and gate_name_for_operation(op) != "measure":
            total += -math.log(max(1.0 - backend.single_qubit_error, 1e-12))
    return total


def _op_counts(circuit: cirq.Circuit) -> Counter[str]:
    counts: Counter[str] = Counter()
    for op in circuit.all_operations():
        name = gate_name_for_operation(op) or type(op.gate).__name__
        counts[name] += 1
    return counts


def _cost_share(total: float, twoq: float, singleq: float, readout: float) -> dict[str, float]:
    if total <= 0:
        return {"twoq": 0.0, "singleq": 0.0, "readout": 0.0}
    return {
        "twoq": twoq / total,
        "singleq": singleq / total,
        "readout": readout / total,
    }


def _candidate_families(names: list[str]) -> list[str]:
    families: set[str] = set()
    for name in names:
        if name.startswith("relative_phase_toffoli"):
            families.add("row_phase_relative_toffoli")
        elif "route_scar" in name or "scar" in name:
            families.add("route_scar")
        elif name.startswith("exact_small_router"):
            families.add("exact_small_router")
        elif name.startswith("v13") or "measurement" in name or "mc_" in name:
            families.add("measurement_channel")
        elif "ghz" in name:
            families.add("ghz_tree")
        elif "relabel" in name:
            families.add("physical_relabel")
        elif name.startswith("noise_adaptive"):
            families.add("noise_adaptive_router")
        elif name.startswith("cirq_cqc"):
            families.add("cirq_cqc")
        elif name.startswith("legacy"):
            families.add("legacy_track_b")
    return sorted(families)


def _semantic_markers(
    *,
    benchmark_name: str,
    track: str,
    family: str,
    candidate_names: list[str],
    row_phase_usage: set[str],
    route_scar_usage: set[str],
) -> list[str]:
    markers: list[str] = []
    if track == "track_b" and family == "ghz":
        markers.append("final-Z GHZ tree/leaf-tree measurement-channel")
    if track == "track_b" and family == "parity":
        markers.append("affine parity measurement-channel physical placement/copyout")
    if benchmark_name in row_phase_usage:
        markers.append("final-Z row-phase relative-Toffoli rewrite")
    if benchmark_name in route_scar_usage:
        markers.append("route-scar compute/uncompute cancellation")
    if any(name.startswith("exact_small_router") for name in candidate_names):
        markers.append("exact small-router candidate bank")
    if any(name.startswith("relative_phase_toffoli") for name in candidate_names) and benchmark_name not in row_phase_usage:
        markers.append("relative-phase candidates generated")
    if any("measurement_tail" in name for name in candidate_names):
        markers.append("measurement-tail pruning candidate")
    return markers


def _recommended_next_action(
    *,
    track: str,
    benchmark_name: str,
    headroom: float,
    cx_count: int,
    readout_cost: float,
    twoq_cost: float,
) -> str:
    if track == "track_b":
        if headroom >= 0.12:
            return "Track B: only low-risk physical placement/readout assignment; do not rewrite semantics"
        return "Track B: hold; residual is below main priority"
    if benchmark_name in {"track_a_007", "track_a_008"} and cx_count >= 10:
        return "Track A: semantic row-phase/Peres/carry/H-boundary mining, not generic router"
    if twoq_cost > readout_cost * 0.75 and cx_count >= 6:
        return "Track A: inspect semantic rewrite or lower-noise exact route"
    if readout_cost >= twoq_cost:
        return "Readout placement/tail cleanup only"
    return "Low priority; keep current root"


def _load_row_phase_usage() -> set[str]:
    path = ROOT / "experiments" / "row_phase_rewrite_usage_report.md"
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"\|\s*(track_[ab]_\d{3})\s*\|", text))


def _load_route_scar_usage() -> set[str]:
    path = ROOT / "experiments" / "normalized_candidate_delta.jsonl"
    if not path.exists():
        return set()
    accepted: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("accepted"):
            accepted.add(str(payload.get("sample")))
    return accepted


def _circuit_signature(circuit: cirq.Circuit) -> tuple[str, ...]:
    return tuple(str(op) for op in circuit.all_operations())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
