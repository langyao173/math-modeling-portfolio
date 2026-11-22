from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cirq

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.evaluator.equivalence import (
    LayoutSidecar,
    validate_compiler_equivalence,
)
from noise_adaptive_hackathon.evaluator.validate_circuit import (
    count_two_qubit_gates,
    estimate_log_reliability,
    gate_name_for_operation,
    validate_compiled_circuit,
)
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.simulator.cirq_runner import run_track_a
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.relative_phase_toffoli import final_z_measurement_equivalent
from noise_adaptive_hackathon.submission.selection import CompilationCandidate, candidate_metrics
from noise_adaptive_hackathon.submission.simulation_selector import (
    SELECTOR_TRACK_A_COMMON_RANDOM,
    SimulationSelectorConfig,
)


TARGETS = ("track_a_002", "track_a_007", "track_a_008")
RULESET = ROOT / "experiments" / "verified_ruleset_row_phase.json"
OUT_JSON = ROOT / "experiments" / "contextual_window_candidates.json"
OUT_REPORT = ROOT / "experiments" / "contextual_window_report.md"
OUT_REPORT_ALIAS = ROOT / "experiments" / "contextual_row_phase_window_miner_report.md"


@dataclass(frozen=True)
class Window:
    start: int
    stop: int
    qubits: tuple[cirq.LineQubit, ...]
    score: float
    cx_count: int
    h_count: int
    phase_count: int


@dataclass(frozen=True)
class WindowCandidate:
    name: str
    kind: str
    start: int
    stop: int
    replacement_ops: tuple[cirq.Operation, ...]
    deleted_indices: tuple[int, ...]
    metadata: dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Contextual final-Z row-phase window miner for Track A residuals.")
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--public-shots", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=90)
    parser.add_argument("--max-miters-per-sample", type=int, default=900)
    parser.add_argument("--max-candidates-per-window", type=int, default=20)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-report", type=Path, default=OUT_REPORT)
    parser.add_argument("--out-report-alias", type=Path, default=OUT_REPORT_ALIAS)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    rules = _load_rules()
    payload: dict[str, Any] = {
        "variant": "contextual_row_phase_window_miner_dev_only",
        "seed": args.seed,
        "shots": args.shots,
        "public_shots": args.public_shots,
        "ruleset": str(RULESET),
        "rule_count": len(rules),
        "targets": {},
    }
    targets = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    for sample in targets:
        benchmark = load_benchmark(ROOT / "benchmarks" / "track_a" / "public" / sample)
        backend = _load_backend(benchmark)
        root = _compile_root(benchmark.circuit, backend, seed=args.seed)
        sample_payload = mine_sample(
            benchmark=benchmark,
            backend=backend,
            root=root,
            rules=rules,
            seed=args.seed,
            shots=args.shots,
            public_shots=args.public_shots,
            max_windows=args.max_windows,
            max_miters=args.max_miters_per_sample,
            max_candidates_per_window=args.max_candidates_per_window,
        )
        payload["targets"][sample] = sample_payload
    payload["elapsed_sec"] = time.perf_counter() - started
    payload["seed7_gate"] = _seed7_gate(payload)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report = render_report(payload)
    args.out_report.write_text(report, encoding="utf-8")
    args.out_report_alias.write_text(report, encoding="utf-8")
    print(json.dumps(payload["seed7_gate"], indent=2, sort_keys=True))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_report}")
    print(f"wrote {args.out_report_alias}")
    return 0


def mine_sample(
    *,
    benchmark,
    backend: CompetitionBackend,
    root: CompilationCandidate,
    rules: list[dict[str, Any]],
    seed: int,
    shots: int,
    public_shots: int,
    max_windows: int,
    max_miters: int,
    max_candidates_per_window: int,
) -> dict[str, Any]:
    root_metrics = candidate_metrics(root, backend, benchmark.circuit)
    root_diag = _score_compiled(benchmark, backend, root, seed=seed, shots=shots, public_shots=public_shots)
    operations = list(root.circuit.all_operations())
    windows = candidate_windows(root.circuit, backend, max_windows=max_windows)
    rows: list[dict[str, Any]] = []
    seen_circuits: set[tuple[str, ...]] = {_signature(root.circuit)}
    miter_count = 0
    rejection_counts: Counter[str] = Counter()

    for candidate in generate_global_candidates(operations, backend, rules, max_candidates=min(420, max_miters)):
        if miter_count >= max_miters:
            break
        rewritten = apply_window_candidate(operations, candidate)
        signature = _signature(rewritten)
        if signature in seen_circuits:
            rejection_counts["duplicate"] += 1
            continue
        seen_circuits.add(signature)
        if count_two_qubit_gates(rewritten) > root_metrics.two_qubit_count:
            rejection_counts["cx_increase"] += 1
            continue
        miter_count += 1
        row = evaluate_contextual_candidate(
            benchmark=benchmark,
            backend=backend,
            root=root,
            root_metrics=root_metrics,
            rewritten=rewritten,
            candidate=candidate,
            seed=seed,
            shots=shots,
            public_shots=public_shots,
        )
        rows.append(row)
        rejection_counts[row["rejection_reason"]] += 1

    for window in windows:
        if miter_count >= max_miters:
            break
        candidates = generate_window_candidates(
            operations,
            window,
            backend,
            rules,
            max_candidates=max_candidates_per_window,
        )
        for candidate in candidates:
            if miter_count >= max_miters:
                break
            rewritten = apply_window_candidate(operations, candidate)
            signature = _signature(rewritten)
            if signature in seen_circuits:
                rejection_counts["duplicate"] += 1
                continue
            seen_circuits.add(signature)
            if count_two_qubit_gates(rewritten) > root_metrics.two_qubit_count:
                rejection_counts["cx_increase"] += 1
                continue
            miter_count += 1
            row = evaluate_contextual_candidate(
                benchmark=benchmark,
                backend=backend,
                root=root,
                root_metrics=root_metrics,
                rewritten=rewritten,
                candidate=candidate,
                seed=seed,
                shots=shots,
                public_shots=public_shots,
            )
            rows.append(row)
            rejection_counts[row["rejection_reason"]] += 1

    valid_rows = [row for row in rows if row["valid"]]
    best = max(
        [root_diag, *valid_rows],
        key=lambda row: (
            row.get("diagnostic_success", -1.0),
            row.get("elr", -999.0),
            -row.get("cx", 999),
            row.get("name", ""),
        ),
    )
    accepted = [row for row in rows if row.get("accepted_metric_guard")]
    return {
        "family": benchmark.family,
        "root": root_diag,
        "best": best,
        "window_count": len(windows),
        "windows": [
            {
                "start": window.start,
                "stop": window.stop,
                "qubits": [qubit.x for qubit in window.qubits],
                "score": window.score,
                "cx_count": window.cx_count,
                "h_count": window.h_count,
                "phase_count": window.phase_count,
            }
            for window in windows[:25]
        ],
        "tested_contextual_candidates": len(rows),
        "miter_count": miter_count,
        "valid_count": len(valid_rows),
        "accepted_metric_guard_count": len(accepted),
        "accepted": sorted(
            accepted,
            key=lambda row: (
                row.get("diagnostic_success", -1.0),
                row.get("elr", -999.0),
                -row.get("cx", 999),
                row.get("name", ""),
            ),
            reverse=True,
        )[:20],
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
        )[:40],
        "rejection_counts": dict(rejection_counts),
    }


def candidate_windows(circuit: cirq.Circuit, backend: CompetitionBackend, *, max_windows: int) -> list[Window]:
    ops = [op for op in circuit.all_operations() if not isinstance(op.gate, cirq.MeasurementGate)]
    windows: list[Window] = []
    for start in range(len(ops)):
        qubits: set[cirq.LineQubit] = set()
        cx_count = 0
        h_count = 0
        phase_count = 0
        twoq_cost = 0.0
        repeated_edges: Counter[tuple[int, int]] = Counter()
        for stop in range(start + 1, min(len(ops), start + 22) + 1):
            op = ops[stop - 1]
            qubits.update(op.qubits)
            if len(qubits) > 5:
                break
            name = gate_name_for_operation(op)
            if name == "cx":
                cx_count += 1
                edge = tuple(sorted((op.qubits[0].x, op.qubits[1].x)))
                repeated_edges[edge] += 1
                twoq_cost += -math.log(max(1.0 - backend.edge_error(*edge), 1e-12))
            elif name == "h":
                h_count += 1
            elif name in {"s", "sdg", "t", "tdg", "rz", "z"}:
                phase_count += 1
            width = len(qubits)
            if not 3 <= width <= 5:
                continue
            if cx_count < 2:
                continue
            score = (
                6.0 * cx_count
                + 2.5 * h_count
                + 1.4 * phase_count
                + 20.0 * twoq_cost
                + 2.0 * sum(max(0, count - 1) for count in repeated_edges.values())
                - 0.05 * (stop - start)
            )
            windows.append(
                Window(
                    start=start,
                    stop=stop,
                    qubits=tuple(sorted(qubits, key=lambda qubit: qubit.x)),
                    score=score,
                    cx_count=cx_count,
                    h_count=h_count,
                    phase_count=phase_count,
                )
            )
    deduped: list[Window] = []
    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    for window in sorted(windows, key=lambda item: (item.score, item.cx_count, -item.stop + item.start), reverse=True):
        key = (window.start, window.stop, tuple(qubit.x for qubit in window.qubits))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(window)
        if len(deduped) >= max_windows:
            break
    return deduped


def generate_window_candidates(
    operations: list[cirq.Operation],
    window: Window,
    backend: CompetitionBackend,
    rules: list[dict[str, Any]],
    *,
    max_candidates: int,
) -> list[WindowCandidate]:
    body = operations[window.start : window.stop]
    out: list[WindowCandidate] = []
    seen: set[tuple[str, ...]] = set()

    def add(candidate: WindowCandidate) -> None:
        if len(out) >= max_candidates:
            return
        replacement = list(candidate.replacement_ops)
        if _replacement_illegal(replacement, backend):
            return
        key = (candidate.kind, tuple(str(op) for op in replacement), candidate.deleted_indices)
        if key in seen:
            return
        seen.add(key)
        old_cx = sum(1 for op in body if gate_name_for_operation(op) == "cx")
        new_cx = sum(1 for op in replacement if gate_name_for_operation(op) == "cx")
        if new_cx > old_cx:
            return
        if new_cx == old_cx and len(replacement) >= len(body):
            return
        out.append(candidate)

    for candidate in _deletion_candidates(body, window):
        add(candidate)
    for candidate in _local_cleanup_candidates(body, window):
        add(candidate)
    for candidate in _verified_rule_candidates(body, window, rules):
        add(candidate)
    return out


def generate_global_candidates(
    operations: list[cirq.Operation],
    backend: CompetitionBackend,
    rules: list[dict[str, Any]],
    *,
    max_candidates: int,
) -> list[WindowCandidate]:
    body_len = len(operations) - 1 if operations and isinstance(operations[-1].gate, cirq.MeasurementGate) else len(operations)
    body = operations[:body_len]
    out: list[WindowCandidate] = []
    seen: set[tuple[str, ...]] = set()

    def add(candidate: WindowCandidate) -> None:
        if len(out) >= max_candidates:
            return
        replacement = list(candidate.replacement_ops)
        if _replacement_illegal(replacement, backend):
            return
        signature = tuple(str(op) for op in replacement)
        if signature in seen:
            return
        seen.add(signature)
        out.append(candidate)

    cx_indices = [index for index, op in enumerate(body) if gate_name_for_operation(op) == "cx"]
    for left, right in _ranked_cx_pairs(body, cx_indices):
        add(_global_delete_candidate(body, body_len, (left, right), "global_delete_cx_pair"))
    # Triples are expensive but can expose compute/uncompute scars where no pair
    # alone is context-equivalent. Keep a deterministic bounded prefix.
    triple_count = 0
    for i, left in enumerate(cx_indices):
        for j, middle in enumerate(cx_indices[i + 1 :], i + 1):
            for right in cx_indices[j + 1 :]:
                add(_global_delete_candidate(body, body_len, (left, middle, right), "global_delete_cx_triple"))
                triple_count += 1
                if triple_count >= 180 or len(out) >= max_candidates:
                    break
            if triple_count >= 180 or len(out) >= max_candidates:
                break
        if triple_count >= 180 or len(out) >= max_candidates:
            break

    for candidate in _global_verified_rule_candidates(body, body_len, rules, max_candidates=max(0, max_candidates - len(out))):
        add(candidate)
    return out[:max_candidates]


def _global_delete_candidate(
    body: list[cirq.Operation],
    stop: int,
    indices: tuple[int, ...],
    kind: str,
) -> WindowCandidate:
    deleted = set(indices)
    replacement = tuple(op for index, op in enumerate(body) if index not in deleted)
    return WindowCandidate(
        name=f"context_{kind}_{'_'.join(str(index) for index in indices)}",
        kind=kind,
        start=0,
        stop=stop,
        replacement_ops=replacement,
        deleted_indices=indices,
        metadata={"global_deleted_indices": list(indices)},
    )


def _global_verified_rule_candidates(
    body: list[cirq.Operation],
    stop: int,
    rules: list[dict[str, Any]],
    *,
    max_candidates: int,
) -> list[WindowCandidate]:
    if max_candidates <= 0:
        return []
    out: list[WindowCandidate] = []
    qubits = tuple(sorted({qubit for op in body for qubit in op.qubits}, key=lambda qubit: qubit.x))
    for rule_index, rule in enumerate(_ranked_rules(rules)):
        before = rule.get("before", [])
        after = rule.get("after", [])
        if not before:
            continue
        rule_qubits = sorted({qubit for item in [*before, *after] for qubit in item["qubits"]})
        if len(rule_qubits) > len(qubits):
            continue
        for mapped in _limited_permutations(qubits, len(rule_qubits), limit=240):
            mapping = dict(zip(rule_qubits, mapped))
            match = _match_rule_subsequence(body, before, mapping)
            if match is None:
                continue
            replacement = _apply_subsequence_replacement(body, match, after, mapping)
            out.append(
                WindowCandidate(
                    name=f"context_global_rule_{rule.get('family')}_{rule_index}_{'_'.join(str(index) for index in match)}",
                    kind=f"global_verified_rule:{rule.get('family')}",
                    start=0,
                    stop=stop,
                    replacement_ops=tuple(replacement),
                    deleted_indices=match,
                    metadata={
                        "rule_index": rule_index,
                        "rule_family": rule.get("family"),
                        "mapped_qubits": {str(key): value.x for key, value in mapping.items()},
                    },
                )
            )
            if len(out) >= max_candidates:
                return out
    return out


def _deletion_candidates(body: list[cirq.Operation], window: Window) -> list[WindowCandidate]:
    cx_indices = [index for index, op in enumerate(body) if gate_name_for_operation(op) == "cx"]
    phase_indices = [index for index, op in enumerate(body) if gate_name_for_operation(op) in {"s", "sdg", "t", "tdg", "rz", "z"}]
    out: list[WindowCandidate] = []

    # Single and paired CX deletion are not standalone-safe, but contextual miter
    # can prove whether suffix/prefix makes them final-Z redundant.
    for local_index in cx_indices[:10]:
        out.append(_delete_indices_candidate(body, window, (local_index,), "delete_single_cx"))
    for left_index, right_index in _ranked_cx_pairs(body, cx_indices)[:18]:
        out.append(_delete_indices_candidate(body, window, (left_index, right_index), "delete_cx_pair"))
    for local_index in phase_indices[:8]:
        out.append(_delete_indices_candidate(body, window, (local_index,), "delete_phase"))
    return out


def _local_cleanup_candidates(body: list[cirq.Operation], window: Window) -> list[WindowCandidate]:
    out: list[WindowCandidate] = []
    for index in range(len(body) - 1):
        left, right = body[index], body[index + 1]
        left_name = gate_name_for_operation(left)
        right_name = gate_name_for_operation(right)
        if tuple(left.qubits) != tuple(right.qubits):
            continue
        if (left_name, right_name) in {
            ("h", "h"),
            ("x", "x"),
            ("z", "z"),
            ("s", "sdg"),
            ("sdg", "s"),
            ("t", "tdg"),
            ("tdg", "t"),
        }:
            out.append(_delete_indices_candidate(body, window, (index, index + 1), f"cleanup_{left_name}_{right_name}"))
    for index in range(len(body) - 4):
        candidate = _h_cx_h_boundary_candidate(body, window, index)
        if candidate is not None:
            out.append(candidate)
    return out


def _verified_rule_candidates(
    body: list[cirq.Operation],
    window: Window,
    rules: list[dict[str, Any]],
) -> list[WindowCandidate]:
    out: list[WindowCandidate] = []
    qubits = window.qubits
    for rule_index, rule in enumerate(_ranked_rules(rules)):
        before = rule.get("before", [])
        after = rule.get("after", [])
        if not before:
            continue
        rule_qubits = sorted({qubit for item in [*before, *after] for qubit in item["qubits"]})
        if len(rule_qubits) > len(qubits):
            continue
        for mapped in _limited_permutations(qubits, len(rule_qubits), limit=120):
            mapping = dict(zip(rule_qubits, mapped))
            match = _match_rule_subsequence(body, before, mapping)
            if match is None:
                continue
            replacement = _apply_subsequence_replacement(body, match, after, mapping)
            out.append(
                WindowCandidate(
                    name=f"context_rule_{rule.get('family')}_{rule_index}_s{window.start}",
                    kind=f"verified_rule:{rule.get('family')}",
                    start=window.start,
                    stop=window.stop,
                    replacement_ops=tuple(replacement),
                    deleted_indices=tuple(window.start + index for index in match),
                    metadata={
                        "rule_index": rule_index,
                        "rule_family": rule.get("family"),
                        "rule_before_cx": rule.get("before_cx"),
                        "rule_after_cx": rule.get("after_cx"),
                        "mapped_qubits": {str(key): value.x for key, value in mapping.items()},
                    },
                )
            )
            if len(out) >= 10:
                return out
    return out


def evaluate_contextual_candidate(
    *,
    benchmark,
    backend: CompetitionBackend,
    root: CompilationCandidate,
    root_metrics,
    rewritten: cirq.Circuit,
    candidate: WindowCandidate,
    seed: int,
    shots: int,
    public_shots: int,
) -> dict[str, Any]:
    base = {
        "name": candidate.name,
        "kind": candidate.kind,
        "window": {
            "start": candidate.start,
            "stop": candidate.stop,
            "deleted_indices": list(candidate.deleted_indices),
        },
        "metadata": candidate.metadata,
    }
    if not final_z_measurement_equivalent(root.circuit, rewritten):
        return {**base, "valid": False, "accepted_metric_guard": False, "rejection_reason": "contextual_final_z_miter_failed"}
    validation = validate_compiled_circuit(rewritten, backend, benchmark.track)
    if not validation.valid:
        return {
            **base,
            "valid": False,
            "accepted_metric_guard": False,
            "rejection_reason": "validator_failed",
            "validation_errors": validation.errors,
        }
    equivalence = validate_compiler_equivalence(
        benchmark,
        rewritten,
        LayoutSidecar(initial_layout={logical.x: physical for logical, physical in root.layout.items()}),
        seed=seed,
    )
    if not equivalence.valid:
        return {
            **base,
            "valid": False,
            "accepted_metric_guard": False,
            "rejection_reason": "official_equivalence_failed",
            "validation_errors": equivalence.errors,
        }
    if not _passes_noiseless_semantics(benchmark, rewritten, seed=seed):
        return {**base, "valid": False, "accepted_metric_guard": False, "rejection_reason": "noiseless_semantic_failed"}

    trial = CompilationCandidate(candidate.name, rewritten, dict(root.layout), root.compile_time_sec)
    metrics = candidate_metrics(trial, backend, benchmark.circuit)
    accepted = _metric_guard(metrics, root_metrics)
    diagnostic = run_track_a(benchmark, rewritten, backend, shots=shots, seed=seed + _track_a_index(benchmark.name))
    public = run_track_a(benchmark, rewritten, backend, shots=public_shots, seed=seed + _track_a_index(benchmark.name))
    return {
        **base,
        "valid": True,
        "accepted_metric_guard": accepted,
        "rejection_reason": "accepted" if accepted else "metric_guard_failed",
        "cx": metrics.two_qubit_count,
        "depth": metrics.depth,
        "elr": metrics.log_reliability,
        "readout_cost": metrics.readout_cost,
        "route_overhead": metrics.route_overhead,
        "diagnostic_success": diagnostic.success_rate,
        "diagnostic_correct": diagnostic.correct_shots,
        "diagnostic_shots": shots,
        "public_success": public.success_rate,
        "public_correct": public.correct_shots,
        "public_shots": public_shots,
        "cx_delta": metrics.two_qubit_count - root_metrics.two_qubit_count,
        "depth_delta": metrics.depth - root_metrics.depth,
        "elr_delta": metrics.log_reliability - root_metrics.log_reliability,
    }


def apply_window_candidate(operations: list[cirq.Operation], candidate: WindowCandidate) -> cirq.Circuit:
    rewritten_ops = [
        *operations[: candidate.start],
        *candidate.replacement_ops,
        *operations[candidate.stop :],
    ]
    return cirq.Circuit(rewritten_ops)


def render_report(payload: dict[str, Any]) -> str:
    gate = payload["seed7_gate"]
    lines = [
        "# Contextual Row-Phase Window Miner Report",
        "",
        "This is a development-only contextual miner. It does not modify root `compiler.py`.",
        "",
        f"- Seed: `{payload['seed']}`",
        f"- Diagnostic shots: `{payload['shots']}`",
        f"- Public-style shots: `{payload['public_shots']}`",
        f"- Rules loaded: `{payload['rule_count']}`",
        f"- Keep decision: `{gate['keep']}`",
        f"- Final seed7 delta estimate: `{gate['final_seed7_delta_estimate']:.6f}`",
        "",
        "## Keep Gate",
        "",
        "| Metric | Value | Pass |",
        "|---|---:|---:|",
    ]
    for sample, delta in gate["raw_deltas"].items():
        threshold = 0.03 if sample == "track_a_002" else 0.04
        lines.append(f"| {sample} raw diagnostic delta | {delta:.6f} | {delta >= threshold} |")
    lines.append(f"| Final seed7 delta estimate | {gate['final_seed7_delta_estimate']:.6f} | {gate['final_plus_0_003']} |")
    lines.append(f"| Keep | {1 if gate['keep'] else 0} | {gate['keep']} |")
    lines.append("")

    for sample, data in payload["targets"].items():
        root = data["root"]
        best = data["best"]
        lines.extend(
            [
                f"## {sample}",
                "",
                f"- Family: `{data['family']}`",
                f"- Windows: `{data['window_count']}`, tested contextual candidates: `{data['tested_contextual_candidates']}`, miter count: `{data['miter_count']}`.",
                f"- Valid candidates: `{data['valid_count']}`, metric-guard accepted: `{data['accepted_metric_guard_count']}`.",
                f"- Root: diagnostic `{root['diagnostic_success']:.6f}`, public-style `{root['public_success']:.6f}`, CX `{root['cx']}`, depth `{root['depth']}`, ELR `{root['elr']:.6f}`.",
                f"- Best: `{best['name']}`, diagnostic `{best.get('diagnostic_success', 0):.6f}`, public-style `{best.get('public_success', 0):.6f}`, CX `{best.get('cx')}`, depth `{best.get('depth')}`, ELR `{best.get('elr', 0):.6f}`.",
                f"- Rejections: `{data['rejection_counts']}`",
                "",
                "| Candidate | Kind | Valid | Guard | Diag Success | Public Success | CX | Depth | ELR | Reason |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in data["top_candidates"][:14]:
            lines.append(
                f"| `{row['name']}` | `{row.get('kind', '')}` | {row.get('valid')} | {row.get('accepted_metric_guard')} | {row.get('diagnostic_success', 0):.6f} | {row.get('public_success', 0):.6f} | {row.get('cx')} | {row.get('depth')} | {row.get('elr', 0):.6f} | `{row.get('rejection_reason')}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            "Do not merge unless A007/A008 reach `CX <= 12` with raw success `+0.04`, or A002 raw success improves by `+0.03`, and the public-style final delta is at least `+0.003`.",
        ]
    )
    return "\n".join(lines)


def _seed7_gate(payload: dict[str, Any]) -> dict[str, Any]:
    raw_deltas: dict[str, float] = {}
    public_deltas: dict[str, float] = {}
    cx_values: dict[str, int | None] = {}
    for sample, data in payload["targets"].items():
        root = data["root"]
        best = data["best"]
        raw_deltas[sample] = float(best.get("diagnostic_success", 0.0)) - float(root["diagnostic_success"])
        public_deltas[sample] = float(best.get("public_success", 0.0)) - float(root["public_success"])
        cx_values[sample] = best.get("cx")
    final_delta = sum(public_deltas.values()) * (0.75 / 8.0)
    a007_gate = raw_deltas.get("track_a_007", 0.0) >= 0.04 and (cx_values.get("track_a_007") or 999) <= 12
    a008_gate = raw_deltas.get("track_a_008", 0.0) >= 0.04 and (cx_values.get("track_a_008") or 999) <= 12
    a002_gate = raw_deltas.get("track_a_002", 0.0) >= 0.03
    return {
        "raw_deltas": raw_deltas,
        "public_deltas": public_deltas,
        "best_cx": cx_values,
        "target_gate": a007_gate or a008_gate or a002_gate,
        "final_seed7_delta_estimate": final_delta,
        "final_plus_0_003": final_delta >= 0.003,
        "keep": (a007_gate or a008_gate or a002_gate) and final_delta >= 0.003,
    }


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


def _score_compiled(
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


def _ranked_cx_pairs(body: list[cirq.Operation], cx_indices: list[int]) -> list[tuple[int, int]]:
    pairs: list[tuple[tuple[int, int, int], tuple[int, int]]] = []
    for i, left in enumerate(cx_indices):
        for right in cx_indices[i + 1 :]:
            left_op, right_op = body[left], body[right]
            same_edge = tuple(left_op.qubits) == tuple(right_op.qubits)
            same_undirected = set(left_op.qubits) == set(right_op.qubits)
            distance = right - left
            priority = (0 if same_edge else 1 if same_undirected else 2, distance, left)
            pairs.append((priority, (left, right)))
    return [pair for _, pair in sorted(pairs, key=lambda item: item[0])]


def _delete_indices_candidate(
    body: list[cirq.Operation],
    window: Window,
    indices: tuple[int, ...],
    kind: str,
) -> WindowCandidate:
    deleted = set(indices)
    replacement = tuple(op for index, op in enumerate(body) if index not in deleted)
    return WindowCandidate(
        name=f"context_{kind}_s{window.start}_e{window.stop}_{'_'.join(str(window.start + i) for i in indices)}",
        kind=kind,
        start=window.start,
        stop=window.stop,
        replacement_ops=replacement,
        deleted_indices=tuple(window.start + index for index in indices),
        metadata={"local_deleted_indices": list(indices), "window_qubits": [qubit.x for qubit in window.qubits]},
    )


def _h_cx_h_boundary_candidate(body: list[cirq.Operation], window: Window, index: int) -> WindowCandidate | None:
    chunk = body[index : index + 5]
    names = [gate_name_for_operation(op) for op in chunk]
    if names != ["h", "h", "cx", "h", "h"]:
        return None
    first_h, second_h, cx, third_h, fourth_h = chunk
    if first_h.qubits != third_h.qubits or second_h.qubits != fourth_h.qubits:
        return None
    if set(first_h.qubits + second_h.qubits) != set(cx.qubits):
        return None
    control, target = cx.qubits
    replacement = [
        *body[:index],
        cirq.CNOT(target, control),
        *body[index + 5 :],
    ]
    return WindowCandidate(
        name=f"context_h_cx_h_reversal_s{window.start}_i{index}",
        kind="h_cx_h_reversal",
        start=window.start,
        stop=window.stop,
        replacement_ops=tuple(replacement),
        deleted_indices=tuple(window.start + index + offset for offset in range(5)),
        metadata={"window_qubits": [qubit.x for qubit in window.qubits]},
    )


def _ranked_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    interesting = [
        rule
        for rule in rules
        if rule.get("family") in {"relative_phase_toffoli", "mined_row_phase", "h_cx_h_reversal"}
        or int(rule.get("before_cx", 0)) > int(rule.get("after_cx", 0))
        or int(rule.get("before_ops", 0)) > int(rule.get("after_ops", 0))
    ]
    return sorted(
        interesting,
        key=lambda rule: (
            int(rule.get("after_cx", 0)) - int(rule.get("before_cx", 0)),
            int(rule.get("after_ops", 0)) - int(rule.get("before_ops", 0)),
            str(rule.get("family", "")),
        ),
    )


def _match_rule_subsequence(
    body: list[cirq.Operation],
    pattern: list[dict[str, Any]],
    mapping: dict[int, cirq.LineQubit],
) -> tuple[int, ...] | None:
    matched: list[int] = []
    pattern_index = 0
    touched = {mapping[index] for item in pattern for index in item["qubits"]}
    for index, op in enumerate(body):
        if pattern_index >= len(pattern):
            break
        item = pattern[pattern_index]
        if _same_rule_op(op, item, mapping):
            matched.append(index)
            pattern_index += 1
            continue
        if not set(op.qubits).isdisjoint(touched):
            if matched:
                return None
    if pattern_index != len(pattern):
        return None
    return tuple(matched)


def _apply_subsequence_replacement(
    body: list[cirq.Operation],
    matched: tuple[int, ...],
    after: list[dict[str, Any]],
    mapping: dict[int, cirq.LineQubit],
) -> list[cirq.Operation]:
    matched_set = set(matched)
    first = matched[0]
    out: list[cirq.Operation] = []
    for index, op in enumerate(body):
        if index == first:
            out.extend(_operation_from_rule_op(item, mapping) for item in after)
        if index in matched_set:
            continue
        out.append(op)
    return out


def _same_rule_op(op: cirq.Operation, item: dict[str, Any], mapping: dict[int, cirq.LineQubit]) -> bool:
    if gate_name_for_operation(op) != item["gate"]:
        return False
    return tuple(op.qubits) == tuple(mapping[index] for index in item["qubits"])


def _operation_from_rule_op(item: dict[str, Any], mapping: dict[int, cirq.LineQubit]) -> cirq.Operation:
    qubits = [mapping[index] for index in item["qubits"]]
    gate = item["gate"]
    if gate == "cx":
        return cirq.CNOT(qubits[0], qubits[1])
    if gate == "h":
        return cirq.H(qubits[0])
    if gate == "x":
        return cirq.X(qubits[0])
    if gate == "y":
        return cirq.Y(qubits[0])
    if gate == "z":
        return cirq.Z(qubits[0])
    if gate == "s":
        return cirq.S(qubits[0])
    if gate == "sdg":
        return cirq.S(qubits[0]) ** -1
    if gate == "t":
        return cirq.T(qubits[0])
    if gate == "tdg":
        return cirq.T(qubits[0]) ** -1
    raise ValueError(f"unsupported rule gate: {gate}")


def _replacement_illegal(ops: list[cirq.Operation], backend: CompetitionBackend) -> bool:
    edges = {tuple(sorted(edge)) for edge in backend.coupling_map}
    for op in ops:
        if len(op.qubits) == 2 and tuple(sorted((op.qubits[0].x, op.qubits[1].x))) not in edges:
            return True
    return False


def _limited_permutations(items: tuple[cirq.LineQubit, ...], size: int, *, limit: int):
    count = 0
    for item in __import__("itertools").permutations(items, size):
        yield item
        count += 1
        if count >= limit:
            return


def _signature(circuit: cirq.Circuit) -> tuple[str, ...]:
    return tuple(str(op) for op in circuit.all_operations())


def _load_rules() -> list[dict[str, Any]]:
    payload = json.loads(RULESET.read_text(encoding="utf-8"))
    return [rule for rule in payload.get("rules", []) if rule.get("verified")]


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
