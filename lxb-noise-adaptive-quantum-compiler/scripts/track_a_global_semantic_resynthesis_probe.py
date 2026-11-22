from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cirq
import numpy as np

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.evaluator.equivalence import (
    LayoutSidecar,
    validate_compiler_equivalence,
)
from noise_adaptive_hackathon.evaluator.validate_circuit import count_two_qubit_gates, estimate_log_reliability
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.simulator.cirq_runner import run_track_a
from noise_adaptive_hackathon.submission.backend_model import BackendNoiseModel
from noise_adaptive_hackathon.submission.circuit_features import extract_circuit_features
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.layout import candidate_initial_layouts
from noise_adaptive_hackathon.submission.peephole import conservative_peephole
from noise_adaptive_hackathon.submission.relative_phase_toffoli import final_z_measurement_equivalent
from noise_adaptive_hackathon.submission.router import route_circuit
from noise_adaptive_hackathon.submission.selection import CompilationCandidate, candidate_metrics
from noise_adaptive_hackathon.submission.simulation_selector import (
    SELECTOR_TRACK_A_COMMON_RANDOM,
    SimulationSelectorConfig,
)
from noise_adaptive_hackathon.submission.track_a_affine_collapse import _synthesize_affine_logical
from noise_adaptive_hackathon.submission.track_a_resynthesis.reversible_templates import toffoli_6cx
from noise_adaptive_hackathon.submission.validation import validate_or_raise


TARGETS = ("track_a_002", "track_a_007", "track_a_008")
OUT_JSON = ROOT / "experiments" / "track_a_global_semantic_resynthesis_probe.json"
OUT_REPORT = ROOT / "experiments" / "track_a_global_semantic_resynthesis_probe.md"


@dataclass(frozen=True)
class TruthModel:
    width: int
    outputs: tuple[int, ...]
    deterministic: bool
    is_permutation: bool
    is_affine: bool
    affine_constant: int | None
    affine_columns: tuple[int, ...]
    anf_degrees: tuple[int, ...]
    max_degree: int
    likely_class: str


@dataclass(frozen=True)
class SourceCandidate:
    name: str
    circuit: cirq.Circuit
    method: str
    source_cx: int
    proof: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Global Track A semantic resynthesis probe.")
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--public-shots", type=int, default=256)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-report", type=Path, default=OUT_REPORT)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    payload: dict[str, Any] = {
        "variant": "track_a_global_semantic_resynthesis_probe_dev_only",
        "seed": args.seed,
        "shots": args.shots,
        "public_shots": args.public_shots,
        "targets": {},
    }
    for sample in [item.strip() for item in args.targets.split(",") if item.strip()]:
        benchmark = load_benchmark(ROOT / "benchmarks" / "track_a" / "public" / sample)
        backend = _load_backend(benchmark)
        root = _compile_root(benchmark.circuit, backend, seed=args.seed)
        payload["targets"][sample] = probe_sample(
            benchmark=benchmark,
            backend=backend,
            root=root,
            seed=args.seed,
            shots=args.shots,
            public_shots=args.public_shots,
            max_candidates=args.max_candidates,
        )
    payload["elapsed_sec"] = time.perf_counter() - started
    payload["seed7_gate"] = _seed7_gate(payload)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_report.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload["seed7_gate"], indent=2, sort_keys=True))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_report}")
    return 0


def probe_sample(
    *,
    benchmark,
    backend: CompetitionBackend,
    root: CompilationCandidate,
    seed: int,
    shots: int,
    public_shots: int,
    max_candidates: int,
) -> dict[str, Any]:
    model = learn_truth_model(benchmark.circuit)
    root_row = _score_candidate(benchmark, backend, root, seed=seed, shots=shots, public_shots=public_shots)
    source_candidates = generate_source_candidates(benchmark.circuit, model, max_candidates=max_candidates)
    rows: list[dict[str, Any]] = []
    for candidate in source_candidates:
        rows.append(
            route_and_score_source_candidate(
                benchmark=benchmark,
                backend=backend,
                root=root,
                source_candidate=candidate,
                seed=seed,
                shots=shots,
                public_shots=public_shots,
            )
        )
    valid_rows = [row for row in rows if row["valid"]]
    best = max(
        [root_row, *valid_rows],
        key=lambda row: (
            row.get("diagnostic_success", -1.0),
            row.get("elr", -999.0),
            -row.get("cx", 999),
            row.get("name", ""),
        ),
    )
    return {
        "family": benchmark.family,
        "truth_model": truth_model_payload(model),
        "root": root_row,
        "source_candidate_count": len(source_candidates),
        "routed_candidate_count": len(rows),
        "valid_count": len(valid_rows),
        "accepted_metric_guard_count": sum(1 for row in rows if row.get("accepted_metric_guard")),
        "best": best,
        "top_candidates": sorted(
            rows,
            key=lambda row: (
                row.get("valid", False),
                row.get("diagnostic_success", -1.0),
                row.get("elr", -999.0),
                -row.get("cx", 999),
                row.get("name", ""),
            ),
            reverse=True,
        )[:30],
        "rejection_counts": _rejection_counts(rows),
    }


def learn_truth_model(circuit: cirq.Circuit) -> TruthModel:
    qubits = tuple(sorted(circuit.all_qubits(), key=lambda qubit: qubit.x))
    width = len(qubits)
    outputs: list[int] = []
    deterministic = True
    for value in range(1 << width):
        histogram = _exact_histogram(_prepend_basis_state(circuit, qubits, value), qubits)
        if len(histogram) != 1:
            deterministic = False
            outputs.append(max(histogram, key=histogram.get))
            continue
        outputs.append(next(iter(histogram)))

    is_perm = len(set(outputs)) == len(outputs)
    is_affine, constant, columns = _affine_model(outputs, width)
    degrees = tuple(_anf_degree(outputs, width, bit) for bit in range(width))
    likely_class = classify_truth_model(width, outputs, is_perm, is_affine, degrees)
    return TruthModel(
        width=width,
        outputs=tuple(outputs),
        deterministic=deterministic,
        is_permutation=is_perm,
        is_affine=is_affine,
        affine_constant=constant,
        affine_columns=columns,
        anf_degrees=degrees,
        max_degree=max(degrees) if degrees else 0,
        likely_class=likely_class,
    )


def generate_source_candidates(source: cirq.Circuit, model: TruthModel, *, max_candidates: int) -> list[SourceCandidate]:
    qubits = tuple(sorted(source.all_qubits(), key=lambda qubit: qubit.x))
    candidates: list[SourceCandidate] = []
    seen: set[tuple[str, ...]] = set()

    def add(name: str, circuit: cirq.Circuit, method: str, proof: str) -> None:
        if len(candidates) >= max_candidates:
            return
        signature = tuple(str(op) for op in circuit.all_operations())
        if signature in seen:
            return
        seen.add(signature)
        if not final_z_measurement_equivalent(source, circuit):
            return
        candidates.append(
            SourceCandidate(
                name=name,
                circuit=circuit,
                method=method,
                source_cx=count_two_qubit_gates(circuit),
                proof=proof,
            )
        )

    if model.is_affine and model.affine_constant is not None:
        affine_candidate = _affine_source_candidate(source, model)
        if affine_candidate is not None:
            add("global_affine_synthesis", affine_candidate, "affine_truth_table", "exact_final_z_truth_table")

    for name, candidate in _functional_candidates(source, model):
        add(name, candidate, "recognized_reversible_truth_table", "exact_final_z_truth_table")

    # Retain existing row-phase source candidates as baselines for this global
    # probe; they are not window-bound and are verified by the full final-Z miter.
    from noise_adaptive_hackathon.submission.relative_phase_toffoli import relative_phase_toffoli_rewrites

    for rewrite in relative_phase_toffoli_rewrites(source, max_replacements=3):
        add(rewrite.name, rewrite.circuit, "relative_phase_toffoli_global", "full_final_z_row_phase_miter")
    return candidates[:max_candidates]


def route_and_score_source_candidate(
    *,
    benchmark,
    backend: CompetitionBackend,
    root: CompilationCandidate,
    source_candidate: SourceCandidate,
    seed: int,
    shots: int,
    public_shots: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    root_metrics = candidate_metrics(root, backend, benchmark.circuit)
    try:
        routed_candidates = route_source_candidate(source_candidate.circuit, backend, root.layout, seed=seed)
    except Exception as exc:
        return {
            "name": source_candidate.name,
            "method": source_candidate.method,
            "valid": False,
            "accepted_metric_guard": False,
            "rejection_reason": f"route_error:{type(exc).__name__}",
            "source_cx": source_candidate.source_cx,
            "proof": source_candidate.proof,
        }
    rows: list[dict[str, Any]] = []
    for index, routed in enumerate(routed_candidates):
        try:
            validate_or_raise(benchmark.circuit, routed.circuit, backend)
        except Exception as exc:
            rows.append(
                {
                    "name": f"{source_candidate.name}_r{index}",
                    "method": source_candidate.method,
                    "valid": False,
                    "accepted_metric_guard": False,
                    "rejection_reason": f"validator:{type(exc).__name__}",
                    "source_cx": source_candidate.source_cx,
                    "proof": source_candidate.proof,
                }
            )
            continue
        if not final_z_measurement_equivalent(_physicalize_source_for_miter(benchmark.circuit, routed.layout), routed.circuit):
            rows.append(
                {
                    "name": f"{source_candidate.name}_r{index}",
                    "method": source_candidate.method,
                    "valid": False,
                    "accepted_metric_guard": False,
                    "rejection_reason": "physical_final_z_miter_failed",
                    "source_cx": source_candidate.source_cx,
                    "proof": source_candidate.proof,
                }
            )
            continue
        equivalence = validate_compiler_equivalence(
            benchmark,
            routed.circuit,
            LayoutSidecar(initial_layout={logical.x: physical for logical, physical in routed.layout.items()}),
            seed=seed,
        )
        if not equivalence.valid:
            rows.append(
                {
                    "name": f"{source_candidate.name}_r{index}",
                    "method": source_candidate.method,
                    "valid": False,
                    "accepted_metric_guard": False,
                    "rejection_reason": "official_equivalence_failed",
                    "source_cx": source_candidate.source_cx,
                    "proof": source_candidate.proof,
                    "errors": equivalence.errors,
                }
            )
            continue
        if not _passes_noiseless_semantics(benchmark, routed.circuit, seed=seed):
            rows.append(
                {
                    "name": f"{source_candidate.name}_r{index}",
                    "method": source_candidate.method,
                    "valid": False,
                    "accepted_metric_guard": False,
                    "rejection_reason": "noiseless_semantic_failed",
                    "source_cx": source_candidate.source_cx,
                    "proof": source_candidate.proof,
                }
            )
            continue
        metrics = candidate_metrics(routed, backend, benchmark.circuit)
        accepted = _metric_guard(metrics, root_metrics)
        diagnostic = run_track_a(benchmark, routed.circuit, backend, shots=shots, seed=seed + _track_a_index(benchmark.name))
        public = run_track_a(benchmark, routed.circuit, backend, shots=public_shots, seed=seed + _track_a_index(benchmark.name))
        rows.append(
            {
                "name": f"{source_candidate.name}_r{index}",
                "method": source_candidate.method,
                "valid": True,
                "accepted_metric_guard": accepted,
                "rejection_reason": "accepted" if accepted else "metric_guard_failed",
                "source_cx": source_candidate.source_cx,
                "proof": source_candidate.proof,
                "cx": metrics.two_qubit_count,
                "depth": metrics.depth,
                "elr": metrics.log_reliability,
                "readout_cost": metrics.readout_cost,
                "compile_time_sec": routed.compile_time_sec + time.perf_counter() - started,
                "diagnostic_success": diagnostic.success_rate,
                "diagnostic_correct": diagnostic.correct_shots,
                "diagnostic_shots": shots,
                "public_success": public.success_rate,
                "public_correct": public.correct_shots,
                "public_shots": public_shots,
                "cx_delta": metrics.two_qubit_count - root_metrics.two_qubit_count,
                "elr_delta": metrics.log_reliability - root_metrics.log_reliability,
                "layout": {str(logical.x): physical for logical, physical in sorted(routed.layout.items(), key=lambda item: item[0].x)},
            }
        )
    if not rows:
        return {
            "name": source_candidate.name,
            "method": source_candidate.method,
            "valid": False,
            "accepted_metric_guard": False,
            "rejection_reason": "no_route_candidates",
            "source_cx": source_candidate.source_cx,
            "proof": source_candidate.proof,
        }
    return max(
        rows,
        key=lambda row: (
            row.get("valid", False),
            row.get("diagnostic_success", -1.0),
            row.get("elr", -999.0),
            -row.get("cx", 999),
            row.get("name", ""),
        ),
    )


def route_source_candidate(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    root_layout: dict[cirq.LineQubit, int],
    *,
    seed: int,
) -> list[CompilationCandidate]:
    del seed
    noise_model = BackendNoiseModel.from_backend(backend)
    features = extract_circuit_features(source)
    layouts = _candidate_layouts(source, backend, noise_model, features, root_layout)
    out: list[CompilationCandidate] = []
    seen: set[tuple[str, ...]] = set()
    for index, layout in enumerate(layouts[:10]):
        started = time.perf_counter()
        try:
            routed = route_circuit(
                source,
                backend,
                noise_model,
                features,
                layout,
                routing_objective="balanced",
                enable_execute_and_swap=True,
                enable_bridge=True,
            ).circuit
            routed = conservative_peephole(routed, clifford_only=False)
        except Exception:
            continue
        signature = tuple(str(op) for op in routed.all_operations())
        if signature in seen:
            continue
        seen.add(signature)
        out.append(CompilationCandidate(f"global_semantic_l{index}", routed, dict(layout), time.perf_counter() - started))
    return out


def render_report(payload: dict[str, Any]) -> str:
    gate = payload["seed7_gate"]
    lines = [
        "# Track A Global Semantic Resynthesis Probe",
        "",
        "This is a development-only global semantic probe. It does not modify root `compiler.py`.",
        "",
        f"- Seed: `{payload['seed']}`",
        f"- Diagnostic shots: `{payload['shots']}`",
        f"- Public-style shots: `{payload['public_shots']}`",
        f"- Keep decision: `{gate['keep']}`",
        f"- Final seed7 delta estimate: `{gate['final_seed7_delta_estimate']:.6f}`",
        "",
        "## Keep Gate",
        "",
        "| Metric | Value | Pass |",
        "|---|---:|---:|",
    ]
    for sample, delta in gate["raw_deltas"].items():
        threshold = 0.03 if sample == "track_a_002" else 0.05
        lines.append(f"| {sample} raw diagnostic delta | {delta:.6f} | {delta >= threshold} |")
    lines.append(f"| Final seed7 delta estimate | {gate['final_seed7_delta_estimate']:.6f} | {gate['final_plus_0_004']} |")
    lines.append(f"| Keep | {1 if gate['keep'] else 0} | {gate['keep']} |")
    lines.append("")

    for sample, data in payload["targets"].items():
        root = data["root"]
        best = data["best"]
        truth = data["truth_model"]
        lines.extend(
            [
                f"## {sample}",
                "",
                f"- Family: `{data['family']}`",
                f"- Truth classification: `{truth['likely_class']}`, deterministic `{truth['deterministic']}`, permutation `{truth['is_permutation']}`, affine `{truth['is_affine']}`, ANF degrees `{truth['anf_degrees']}`.",
                f"- Source candidates/routed/valid/accepted: `{data['source_candidate_count']}` / `{data['routed_candidate_count']}` / `{data['valid_count']}` / `{data['accepted_metric_guard_count']}`.",
                f"- Root: diagnostic `{root['diagnostic_success']:.6f}`, public-style `{root['public_success']:.6f}`, CX `{root['cx']}`, depth `{root['depth']}`, ELR `{root['elr']:.6f}`.",
                f"- Best: `{best['name']}`, diagnostic `{best.get('diagnostic_success', 0):.6f}`, public-style `{best.get('public_success', 0):.6f}`, CX `{best.get('cx')}`, depth `{best.get('depth')}`, ELR `{best.get('elr', 0):.6f}`.",
                f"- Rejections: `{data['rejection_counts']}`",
                "",
                "| Candidate | Method | Valid | Guard | Source CX | Routed CX | Depth | ELR | Diag Success | Public Success | Reason |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in data["top_candidates"][:16]:
            lines.append(
                f"| `{row['name']}` | `{row.get('method')}` | {row.get('valid')} | {row.get('accepted_metric_guard')} | {row.get('source_cx')} | {row.get('cx')} | {row.get('depth')} | {row.get('elr', 0):.6f} | {row.get('diagnostic_success', 0):.6f} | {row.get('public_success', 0):.6f} | `{row.get('rejection_reason')}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            "If this probe does not clear the keep gate, stop algorithmic sprinting and keep the current root as the release candidate.",
        ]
    )
    return "\n".join(lines)


def _seed7_gate(payload: dict[str, Any]) -> dict[str, Any]:
    raw_deltas: dict[str, float] = {}
    public_deltas: dict[str, float] = {}
    best_cx: dict[str, int | None] = {}
    for sample, data in payload["targets"].items():
        root = data["root"]
        best = data["best"]
        raw_deltas[sample] = float(best.get("diagnostic_success", 0.0)) - float(root["diagnostic_success"])
        public_deltas[sample] = float(best.get("public_success", 0.0)) - float(root["public_success"])
        best_cx[sample] = best.get("cx")
    final_delta = sum(public_deltas.values()) * (0.75 / 8.0)
    a002_gate = (best_cx.get("track_a_002") or 999) <= 6 or raw_deltas.get("track_a_002", 0.0) >= 0.03
    a007_gate = (best_cx.get("track_a_007") or 999) <= 11 or raw_deltas.get("track_a_007", 0.0) >= 0.05
    a008_gate = (best_cx.get("track_a_008") or 999) <= 11 or raw_deltas.get("track_a_008", 0.0) >= 0.05
    target_gate = a002_gate or a007_gate or a008_gate
    return {
        "raw_deltas": raw_deltas,
        "public_deltas": public_deltas,
        "best_cx": best_cx,
        "target_gate": target_gate,
        "final_seed7_delta_estimate": final_delta,
        "final_plus_0_004": final_delta >= 0.004,
        "keep": target_gate and final_delta >= 0.004,
    }


def _affine_source_candidate(source: cirq.Circuit, model: TruthModel) -> cirq.Circuit | None:
    from noise_adaptive_hackathon.submission.track_a_affine_collapse import _AffineModel

    qubits = tuple(sorted(source.all_qubits(), key=lambda qubit: qubit.x))
    if model.affine_constant is None:
        return None
    affine = _AffineModel(
        qubits=qubits,
        constant=model.affine_constant,
        columns=model.affine_columns,
    )
    return _synthesize_affine_logical(affine)


def _functional_candidates(source: cirq.Circuit, model: TruthModel) -> list[tuple[str, cirq.Circuit]]:
    qubits = tuple(sorted(source.all_qubits(), key=lambda qubit: qubit.x))
    outputs = model.outputs
    out: list[tuple[str, cirq.Circuit]] = []
    if not model.deterministic or not model.is_permutation:
        return out
    for controls, target in _quadratic_toggle_patterns(model):
        circuit = _candidate_from_quadratic_toggles(qubits, controls, target, model)
        if circuit is not None:
            out.append((f"quadratic_toggle_{controls[0]}_{controls[1]}_{target}", circuit))
    return out[:8]


def _quadratic_toggle_patterns(model: TruthModel) -> list[tuple[tuple[int, int], int]]:
    patterns: list[tuple[tuple[int, int], int]] = []
    width = model.width
    base_affine, constant, columns = _affine_model([value ^ model.outputs[0] for value in model.outputs], width)
    del base_affine, constant, columns
    # Try every CCX-like toggle against identity/affine-looking outputs. This is
    # conservative and later exact final-Z miter decides.
    for target in range(width):
        for controls in itertools.combinations([index for index in range(width) if index != target], 2):
            patterns.append((controls, target))
    return patterns


def _candidate_from_quadratic_toggles(
    qubits: tuple[cirq.LineQubit, ...],
    controls: tuple[int, int],
    target: int,
    model: TruthModel,
) -> cirq.Circuit | None:
    width = model.width
    transformed = []
    for value in range(1 << width):
        bits = [(value >> (width - 1 - index)) & 1 for index in range(width)]
        if bits[controls[0]] & bits[controls[1]]:
            bits[target] ^= 1
        transformed.append(sum(bit << (width - 1 - index) for index, bit in enumerate(bits)))
    residual = [model.outputs[value] ^ transformed[value] ^ value for value in range(1 << width)]
    is_affine, constant, columns = _affine_model(residual, width)
    if not is_affine or constant is None:
        return None
    ops: list[cirq.Operation] = []
    ops.extend(_affine_delta_ops(qubits, constant, columns))
    ops.extend(toffoli_6cx(qubits[controls[0]], qubits[controls[1]], qubits[target]))
    ops.append(cirq.measure(*qubits, key="z"))
    return cirq.Circuit(ops)


def _affine_delta_ops(qubits: tuple[cirq.LineQubit, ...], constant: int, columns: tuple[int, ...]) -> list[cirq.Operation]:
    width = len(qubits)
    ops: list[cirq.Operation] = []
    # residual output = input xor affine_delta(input), so implement row XORs in a
    # simple conservative order. Exact final-Z miter validates correctness.
    for input_index, column in enumerate(columns):
        for output_index in range(width):
            if input_index == output_index:
                continue
            if (column >> (width - 1 - output_index)) & 1:
                ops.append(cirq.CNOT(qubits[input_index], qubits[output_index]))
    for output_index in range(width):
        if (constant >> (width - 1 - output_index)) & 1:
            ops.append(cirq.X(qubits[output_index]))
    return ops


def _exact_histogram(circuit: cirq.Circuit, measured_qubits: tuple[cirq.LineQubit, ...]) -> dict[int, float]:
    unitary = cirq.Circuit(op for op in circuit.all_operations() if not isinstance(op.gate, cirq.MeasurementGate))
    qubits = sorted(unitary.all_qubits() | set(measured_qubits), key=lambda qubit: qubit.x)
    positions = {qubit: index for index, qubit in enumerate(qubits)}
    measured_positions = [positions[qubit] for qubit in measured_qubits]
    result = cirq.Simulator().simulate(unitary, qubit_order=qubits)
    histogram: dict[int, float] = {}
    width = len(measured_positions)
    for basis_index, amplitude in enumerate(result.final_state_vector):
        probability = abs(amplitude) ** 2
        if probability < 1e-10:
            continue
        bits = cirq.big_endian_int_to_bits(basis_index, bit_count=len(qubits))
        out = 0
        for index, position in enumerate(measured_positions):
            out |= bits[position] << (width - 1 - index)
        histogram[out] = histogram.get(out, 0.0) + float(probability)
    return histogram


def _prepend_basis_state(circuit: cirq.Circuit, qubits: tuple[cirq.LineQubit, ...], value: int) -> cirq.Circuit:
    width = len(qubits)
    challenged = cirq.Circuit()
    for index, qubit in enumerate(qubits):
        if (value >> (width - 1 - index)) & 1:
            challenged.append(cirq.X(qubit))
    challenged += circuit
    return challenged


def _affine_model(outputs: list[int] | tuple[int, ...], width: int) -> tuple[bool, int | None, tuple[int, ...]]:
    constant = outputs[0]
    columns = tuple(outputs[1 << (width - 1 - index)] ^ constant for index in range(width))
    for value, output in enumerate(outputs):
        predicted = constant
        for index, column in enumerate(columns):
            if (value >> (width - 1 - index)) & 1:
                predicted ^= column
        if predicted != output:
            return False, None, ()
    return True, constant, columns


def _anf_degree(outputs: tuple[int, ...], width: int, bit: int) -> int:
    values = [((output >> (width - 1 - bit)) & 1) for output in outputs]
    coeffs = values[:]
    for axis in range(width):
        step = 1 << axis
        for mask in range(1 << width):
            if mask & step:
                coeffs[mask] ^= coeffs[mask ^ step]
    degree = 0
    for mask, coeff in enumerate(coeffs):
        if coeff:
            degree = max(degree, mask.bit_count())
    return degree


def classify_truth_model(
    width: int,
    outputs: tuple[int, ...],
    is_perm: bool,
    is_affine: bool,
    degrees: tuple[int, ...],
) -> str:
    if is_affine:
        return "affine_linear_reversible"
    if is_perm and max(degrees, default=0) <= 2:
        if _looks_like_two_carry(width, outputs):
            return "quadratic_reversible_carry_like"
        return "quadratic_reversible"
    if is_perm:
        return "nonlinear_reversible"
    return "non_permutation_distributional"


def _looks_like_two_carry(width: int, outputs: tuple[int, ...]) -> bool:
    changed_degrees = []
    for bit in range(width):
        values = [((outputs[x] ^ x) >> (width - 1 - bit)) & 1 for x in range(1 << width)]
        if any(values):
            coeffs = values[:]
            for axis in range(width):
                step = 1 << axis
                for mask in range(1 << width):
                    if mask & step:
                        coeffs[mask] ^= coeffs[mask ^ step]
            changed_degrees.append(max((mask.bit_count() for mask, coeff in enumerate(coeffs) if coeff), default=0))
    return changed_degrees.count(2) >= 1 and len(changed_degrees) <= 3


def _candidate_layouts(
    source: cirq.Circuit,
    backend: CompetitionBackend,
    model: BackendNoiseModel,
    features,
    root_layout: dict[cirq.LineQubit, int],
) -> list[dict[cirq.LineQubit, int]]:
    layouts: list[dict[cirq.LineQubit, int]] = []
    seen: set[tuple[int, ...]] = set()
    logical = tuple(sorted(source.all_qubits(), key=lambda qubit: qubit.x))

    def add(layout: dict[cirq.LineQubit, int]) -> None:
        signature = tuple(layout.get(qubit, -1) for qubit in logical)
        if -1 in signature or signature in seen:
            return
        seen.add(signature)
        layouts.append(dict(layout))

    add(root_layout)
    for layout in candidate_initial_layouts(source, backend, model, features, limit=10):
        add(layout)
    return layouts


def _physicalize_source_for_miter(source: cirq.Circuit, layout: dict[cirq.LineQubit, int]) -> cirq.Circuit:
    mapping = {logical: cirq.LineQubit(physical) for logical, physical in layout.items()}
    return source.transform_qubits(lambda qubit: mapping.get(qubit, qubit))


def _metric_guard(metrics, root_metrics) -> bool:
    if metrics.two_qubit_count <= root_metrics.two_qubit_count - 2 and metrics.log_reliability >= root_metrics.log_reliability - 0.035:
        return True
    if metrics.log_reliability >= root_metrics.log_reliability + 0.02 and metrics.two_qubit_count <= root_metrics.two_qubit_count + 1:
        return True
    return False


def _passes_noiseless_semantics(benchmark, circuit: cirq.Circuit, *, seed: int) -> bool:
    try:
        result = cirq.Simulator(seed=seed).run(circuit, repetitions=128)
        samples = result.measurements["z"]
    except Exception:
        return False
    return all(benchmark.expected_output_rule.is_correct("".join(str(bit) for bit in row)) for row in samples)


def _compile_root(source: cirq.Circuit, backend: CompetitionBackend, *, seed: int) -> CompilationCandidate:
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
    started = time.perf_counter()
    compiled = compiler.compile(source, backend, seed=seed)
    return CompilationCandidate("root_current", compiled, dict(compiler.last_initial_layout), time.perf_counter() - started)


def _score_candidate(
    benchmark,
    backend: CompetitionBackend,
    candidate: CompilationCandidate,
    *,
    seed: int,
    shots: int,
    public_shots: int,
) -> dict[str, Any]:
    metrics = candidate_metrics(candidate, backend, benchmark.circuit)
    diagnostic = run_track_a(benchmark, candidate.circuit, backend, shots=shots, seed=seed + _track_a_index(benchmark.name))
    public = run_track_a(benchmark, candidate.circuit, backend, shots=public_shots, seed=seed + _track_a_index(benchmark.name))
    return {
        "name": candidate.name,
        "valid": True,
        "cx": metrics.two_qubit_count,
        "depth": metrics.depth,
        "elr": metrics.log_reliability,
        "readout_cost": metrics.readout_cost,
        "compile_time_sec": candidate.compile_time_sec,
        "diagnostic_success": diagnostic.success_rate,
        "diagnostic_correct": diagnostic.correct_shots,
        "diagnostic_shots": shots,
        "public_success": public.success_rate,
        "public_correct": public.correct_shots,
        "public_shots": public_shots,
    }


def truth_model_payload(model: TruthModel) -> dict[str, Any]:
    return {
        "width": model.width,
        "deterministic": model.deterministic,
        "is_permutation": model.is_permutation,
        "is_affine": model.is_affine,
        "affine_constant": model.affine_constant,
        "affine_columns": list(model.affine_columns),
        "anf_degrees": list(model.anf_degrees),
        "max_degree": model.max_degree,
        "likely_class": model.likely_class,
    }


def _rejection_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("rejection_reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _load_backend(benchmark) -> CompetitionBackend:
    return CompetitionBackend.parse_raw(
        (
            ROOT
            / "backends"
            / benchmark.metadata["backend_size_class"]
            / f"{benchmark.metadata['backend_snapshot_id']}.json"
        ).read_text(encoding="utf-8")
    )


def _track_a_index(name: str) -> int:
    return int(name.rsplit("_", 1)[1]) - 1


if __name__ == "__main__":
    raise SystemExit(main())
