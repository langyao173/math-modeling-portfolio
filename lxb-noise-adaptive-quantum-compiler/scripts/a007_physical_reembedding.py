from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cirq
import networkx as nx

from common import ROOT

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.submission.backend_model import BackendNoiseModel
from noise_adaptive_hackathon.submission.circuit_features import extract_circuit_features
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.peephole import conservative_peephole
from noise_adaptive_hackathon.submission.router import route_circuit
from noise_adaptive_hackathon.submission.selection import CompilationCandidate, candidate_metrics, select_candidate
from noise_adaptive_hackathon.submission.validation import validate_or_raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A007-like fixed-logic physical reembedding diagnostic.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--sample", default="track_a_007")
    parser.add_argument("--subgraph-limit", type=int, default=80)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--jsonl", type=Path, default=ROOT / "experiments" / "a007_physical_reembedding_candidates.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "experiments" / "a007_physical_reembedding_report.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark = load_benchmark(args.root / "benchmarks" / "track_a" / "public" / args.sample)
    backend = _load_backend(args.root, benchmark)
    compiler = NoiseAdaptiveCompiler(
        enable_parity_resynthesis=True,
        enable_exact_small_router=True,
        enable_route_scar_normalizer=True,
        enable_measurement_tail_pruning=True,
    )
    base_candidates = compiler.candidate_compilations(benchmark.circuit, backend)
    selected = select_candidate(base_candidates, backend, benchmark.circuit)
    base_metrics = candidate_metrics(selected, backend, benchmark.circuit)
    rows = _reembed(
        benchmark.circuit,
        backend,
        base_metrics=base_metrics,
        subgraph_limit=args.subgraph_limit,
        max_seconds=args.max_seconds,
    )
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    args.report.write_text(_render_report(args.sample, selected.name, base_metrics, rows), encoding="utf-8")
    return 0


def _reembed(
    circuit: cirq.Circuit,
    backend: CompetitionBackend,
    *,
    base_metrics,
    subgraph_limit: int,
    max_seconds: float,
) -> list[dict[str, object]]:
    started = time.perf_counter()
    model = BackendNoiseModel.from_backend(backend)
    features = extract_circuit_features(circuit)
    width = len(circuit.all_qubits())
    logical_qubits = tuple(sorted(circuit.all_qubits(), key=lambda qubit: qubit.x))
    rows: list[dict[str, object]] = []
    for region_index, region in enumerate(_rank_regions(backend, model, width, limit=subgraph_limit)):
        if time.perf_counter() - started >= max_seconds:
            break
        for layout_index, layout in enumerate(_layouts_for_region(logical_qubits, features, model, region)[:4]):
            if time.perf_counter() - started >= max_seconds:
                break
            try:
                routed = route_circuit(circuit, backend, model, features, layout).circuit
                routed = conservative_peephole(routed, clifford_only=False)
                validate_or_raise(circuit, routed, backend)
            except Exception as error:
                rows.append(
                    {
                        "region_index": region_index,
                        "layout_index": layout_index,
                        "region": list(region),
                        "accepted": False,
                        "rejection_reason": type(error).__name__,
                    }
                )
                continue
            trial = CompilationCandidate("physical_reembedding", routed, dict(layout), time.perf_counter() - started)
            metrics = candidate_metrics(trial, backend, circuit)
            accepted = (
                metrics.two_qubit_count <= base_metrics.two_qubit_count
                and metrics.depth <= base_metrics.depth + 1
                and metrics.log_reliability >= base_metrics.log_reliability + 0.015
            )
            rows.append(
                {
                    "region_index": region_index,
                    "layout_index": layout_index,
                    "region": list(region),
                    "layout": {str(logical.x): physical for logical, physical in layout.items()},
                    "cx": metrics.two_qubit_count,
                    "depth": metrics.depth,
                    "elr": metrics.log_reliability,
                    "readout_cost": metrics.readout_cost,
                    "accepted": accepted,
                    "rejection_reason": "accepted" if accepted else "metric_guard_failed",
                }
            )
    return rows


def _rank_regions(
    backend: CompetitionBackend,
    model: BackendNoiseModel,
    width: int,
    *,
    limit: int,
) -> list[tuple[int, ...]]:
    graph = backend.graph()
    scored: list[tuple[float, tuple[int, ...]]] = []
    for nodes in __import__("itertools").combinations(range(backend.num_qubits), width):
        subgraph = graph.subgraph(nodes)
        if not nx.is_connected(subgraph):
            continue
        edge_costs = [model.edge_cost(left, right) for left, right in subgraph.edges]
        if not edge_costs:
            continue
        readout = sum(model.readout_cost(node) for node in nodes) / width
        diameter = nx.diameter(subgraph)
        score = sum(edge_costs) / len(edge_costs) + 0.02 * diameter + 0.12 * readout
        scored.append((score, tuple(nodes)))
    return [region for _, region in sorted(scored, key=lambda item: (item[0], item[1]))[:limit]]


def _layouts_for_region(
    logical_qubits: tuple[cirq.LineQubit, ...],
    features,
    model: BackendNoiseModel,
    region: tuple[int, ...],
) -> list[dict[cirq.LineQubit, int]]:
    subgraph = model.graph.subgraph(region)
    logical_orders = [
        sorted(logical_qubits, key=lambda qubit: (features.weighted_degrees[qubit.x], -qubit.x), reverse=True),
        list(logical_qubits),
    ]
    physical_orders = [
        sorted(region, key=lambda node: _physical_score(node, subgraph, model)),
        sorted(region, key=lambda node: (-subgraph.degree(node), _physical_score(node, subgraph, model))),
    ]
    layouts: list[dict[cirq.LineQubit, int]] = []
    seen: set[tuple[int, ...]] = set()
    for logical_order in logical_orders:
        for physical_order in physical_orders:
            layout = {logical: physical for logical, physical in zip(logical_order, physical_order)}
            signature = tuple(layout[logical] for logical in logical_qubits)
            if signature in seen:
                continue
            seen.add(signature)
            layouts.append(layout)
    return layouts


def _physical_score(node: int, subgraph, model: BackendNoiseModel) -> float:
    incident = [model.edge_cost(node, neighbor) for neighbor in subgraph.neighbors(node)]
    edge = sum(incident) / max(len(incident), 1)
    return edge + 0.25 * model.readout_cost(node) - 0.01 * math.log1p(subgraph.degree(node))


def _render_report(sample: str, selected_name: str, base_metrics, rows: list[dict[str, object]]) -> str:
    valid = [row for row in rows if row.get("elr") is not None]
    accepted = [row for row in valid if row.get("accepted")]
    best = max(valid, key=lambda row: (float(row["elr"]), -int(row["cx"]), -int(row["depth"])), default=None)
    lines = [
        "# A007 Physical Reembedding Report",
        "",
        f"- Sample: {sample}",
        f"- Incumbent: {selected_name}",
        f"- Incumbent metrics: CX={base_metrics.two_qubit_count}, depth={base_metrics.depth}, ELR={base_metrics.log_reliability:.12f}",
        f"- Routed candidates checked: {len(valid)}",
        f"- Accepted candidates: {len(accepted)}",
        "",
    ]
    if best:
        lines.extend(
            [
                "## Best Candidate",
                "",
                f"- Region: {best['region']}",
                f"- CX: {best['cx']}",
                f"- Depth: {best['depth']}",
                f"- ELR: {best['elr']}",
                f"- Accepted: {best['accepted']}",
                f"- Reason: {best['rejection_reason']}",
                "",
            ]
        )
    lines.extend(["## Decision", ""])
    if accepted:
        lines.append("Keep for emulator evaluation; at least one fixed-logic physical reembedding candidate improved ELR under CX/depth constraints.")
    else:
        lines.append("Diagnostic no-go: no fixed-logic reembedding candidate reached ELR improvement under CX<=incumbent and depth<=incumbent+1.")
    return "\n".join(lines) + "\n"


def _load_backend(root: Path, benchmark) -> CompetitionBackend:
    backend_path = (
        root
        / "backends"
        / benchmark.metadata["backend_size_class"]
        / f"{benchmark.metadata['backend_snapshot_id']}.json"
    )
    return CompetitionBackend.parse_raw(backend_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
