from __future__ import annotations

import argparse
from pathlib import Path
import json
import math

import cirq

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.submission.backend_model import BackendNoiseModel
from noise_adaptive_hackathon.submission.circuit_features import extract_circuit_features
from noise_adaptive_hackathon.submission.layout import choose_initial_layout, score_layout

from common import ROOT, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attribute remaining public-score loss by routing, noise, and depth.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summary", type=Path, default=ROOT / "experiments" / "full_public_summary_v3.json")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "loss_attribution_v3.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    sample_results = {sample["benchmark_name"]: sample for sample in summary["sample_results"]}
    rows = []
    for track in ("track_a", "track_b"):
        for benchmark_dir in sorted((args.root / "benchmarks" / track / "public").iterdir()):
            if not benchmark_dir.is_dir():
                continue
            benchmark = load_benchmark(benchmark_dir)
            sample = sample_results[benchmark.name]
            backend = _load_backend(args.root, benchmark)
            logical_stats = _logical_stats(benchmark.circuit)
            best_log_reliability = _best_possible_subgraph_log_reliability(benchmark.circuit, backend)
            rows.append(
                {
                    "sample": benchmark.name,
                    "track": track,
                    "success": sample["success_rate"],
                    "contribution_gap": _contribution_gap(track, sample["success_rate"]),
                    "logical_2q_gates": logical_stats["logical_2q_gates"],
                    "compiled_2q_gates": sample["two_qubit_gate_count"],
                    "route_overhead": sample["two_qubit_gate_count"] - logical_stats["logical_2q_gates"],
                    "best_possible_subgraph_log_reliability": best_log_reliability,
                    "compiled_log_reliability": sample["estimated_log_reliability"],
                    "noise_excess": sample["estimated_log_reliability"] - best_log_reliability,
                    "logical_depth_lower_bound": logical_stats["logical_depth_lower_bound"],
                    "compiled_depth": sample["depth"],
                    "depth_excess": sample["depth"] - logical_stats["logical_depth_lower_bound"],
                }
            )
    rows.sort(key=lambda row: (row["contribution_gap"], row["route_overhead"], -row["noise_excess"]), reverse=True)
    write_json(args.out, {"summary": str(args.summary), "rows": rows})
    print_table(rows)
    return 0


def print_table(rows: list[dict[str, object]]) -> None:
    print("sample track success contribution_gap route_overhead noise_excess depth_excess")
    for row in rows:
        print(
            "{sample} {track} {success:.6f} {contribution_gap:.6f} {route_overhead} {noise_excess:.6f} {depth_excess}".format(
                **row
            )
        )


def _load_backend(root: Path, benchmark) -> CompetitionBackend:
    return CompetitionBackend.parse_raw(
        (
            root
            / "backends"
            / benchmark.metadata["backend_size_class"]
            / f"{benchmark.metadata['backend_snapshot_id']}.json"
        ).read_text(encoding="utf-8")
    )


def _contribution_gap(track: str, success: float) -> float:
    if track == "track_a":
        return (1.0 - success) * 0.75 / 8.0
    return (1.0 - success) * 0.25 / 2.0


def _logical_stats(circuit: cirq.Circuit) -> dict[str, int]:
    operations = [op for op in circuit.all_operations() if not isinstance(op.gate, cirq.MeasurementGate)]
    logical_2q = sum(1 for op in operations if len(op.qubits) == 2)
    per_qubit_depth: dict[cirq.Qid, int] = {qubit: 0 for qubit in circuit.all_qubits()}
    for op in operations:
        next_layer = max((per_qubit_depth[qubit] for qubit in op.qubits), default=0) + 1
        for qubit in op.qubits:
            per_qubit_depth[qubit] = next_layer
    return {
        "logical_2q_gates": logical_2q,
        "logical_depth_lower_bound": max(per_qubit_depth.values(), default=0),
    }


def _best_possible_subgraph_log_reliability(circuit: cirq.Circuit, backend: CompetitionBackend) -> float:
    model = BackendNoiseModel.from_backend(backend)
    features = extract_circuit_features(circuit)
    layout = choose_initial_layout(circuit, backend, model, features)
    layout_score = score_layout(layout, model, features)
    optimistic_2q = _optimistic_two_qubit_cost(circuit, model)
    layout_2q = _layout_two_qubit_cost(circuit, model, {logical.x: physical for logical, physical in layout.items()})
    readout_cost = _best_readout_cost(circuit, model)
    single_cost = _single_qubit_cost(circuit, backend)
    # Use the better of an optimistic independent-edge bound and the selected subgraph layout.
    two_qubit_cost = min(optimistic_2q, layout_2q, layout_score)
    return -(single_cost + two_qubit_cost + readout_cost)


def _single_qubit_cost(circuit: cirq.Circuit, backend: CompetitionBackend) -> float:
    total = 0.0
    for op in circuit.all_operations():
        if isinstance(op.gate, (cirq.MeasurementGate, cirq.IdentityGate)):
            continue
        if len(op.qubits) == 1:
            total += -math.log(max(1.0 - backend.single_qubit_error, 1e-12))
    return total


def _layout_two_qubit_cost(circuit: cirq.Circuit, model: BackendNoiseModel, layout: dict[int, int]) -> float:
    total = 0.0
    for op in circuit.all_operations():
        if len(op.qubits) != 2:
            continue
        total += model.noise_distance(layout[op.qubits[0].x], layout[op.qubits[1].x])
    return total


def _optimistic_two_qubit_cost(circuit: cirq.Circuit, model: BackendNoiseModel) -> float:
    edge_costs = sorted(model.edge_cost(*edge) for edge in model.coupling_edges)
    best_edge = edge_costs[0] if edge_costs else 0.0
    return sum(best_edge for op in circuit.all_operations() if len(op.qubits) == 2)


def _best_readout_cost(circuit: cirq.Circuit, model: BackendNoiseModel) -> float:
    measured_width = 0
    for op in circuit.all_operations():
        if gate_name_for_operation(op) == "measure":
            measured_width = max(measured_width, len(op.qubits))
    readout_costs = sorted(model.readout_cost(qubit) for qubit in range(model.backend.num_qubits))
    return sum(readout_costs[:measured_width])


if __name__ == "__main__":
    raise SystemExit(main())
