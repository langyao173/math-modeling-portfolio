from __future__ import annotations

import argparse
import itertools
import json
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

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.evaluator.validate_circuit import (
    count_two_qubit_gates,
    estimate_log_reliability,
    gate_name_for_operation,
)
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.simulator.cirq_runner import run_track_a
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.relative_phase_toffoli import (
    final_z_measurement_equivalent,
    relative_phase_toffoli_rewrites,
)
from noise_adaptive_hackathon.submission.selection import CompilationCandidate, candidate_metrics
from noise_adaptive_hackathon.submission.simulation_selector import (
    SELECTOR_TRACK_A_COMMON_RANDOM,
    SimulationSelectorConfig,
)
from noise_adaptive_hackathon.submission.validation import validate_or_raise


RULESET = ROOT / "experiments" / "verified_ruleset_row_phase.json"
OUT_JSON = ROOT / "experiments" / "tracka_semantic_mining_candidates.json"
OUT_REPORT = ROOT / "experiments" / "tracka_semantic_mining_report.md"
OUT_COMPARE = ROOT / "experiments" / "root_vs_tracka_semantic_seed7.md"
TARGETS = ("track_a_002", "track_a_007", "track_a_008")


@dataclass(frozen=True)
class SourceVariant:
    name: str
    circuit: cirq.Circuit
    kind: str
    old_source_cx: int
    new_source_cx: int
    proof: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dev-time targeted Track A row-phase semantic mining.")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--public-shots", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-source-variants", type=int, default=32)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-report", type=Path, default=OUT_REPORT)
    parser.add_argument("--out-compare", type=Path, default=OUT_COMPARE)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    rules = _load_rules()
    payload = {
        "variant": "tracka_semantic_mining_dev_only",
        "seed": args.seed,
        "diagnostic_shots": args.shots,
        "public_shots": args.public_shots,
        "ruleset": str(RULESET),
        "rule_count": len(rules),
        "targets": {},
    }
    for sample in TARGETS:
        benchmark = load_benchmark(ROOT / "benchmarks" / "track_a" / "public" / sample)
        backend = _load_backend(benchmark)
        root_candidate = _compile_root_candidate(benchmark.circuit, backend, seed=args.seed)
        root_diag = _simulate_candidate(
            benchmark,
            backend,
            root_candidate,
            seed=args.seed,
            shots=args.shots,
            public_shots=args.public_shots,
        )
        variants = _source_variants(
            benchmark.circuit,
            rules,
            limit=args.max_source_variants,
        )
        routed_rows: list[dict[str, Any]] = []
        for variant in variants:
            row = _route_and_score_variant(
                benchmark,
                backend,
                root_candidate,
                variant,
                seed=args.seed,
                shots=args.shots,
                public_shots=args.public_shots,
            )
            routed_rows.append(row)
        routed_rows.sort(
            key=lambda item: (
                item.get("accepted_metric_guard", False),
                item.get("diagnostic_success", -1.0),
                item.get("elr", -999.0),
                -item.get("cx", 999),
                item["name"],
            ),
            reverse=True,
        )
        accepted = [row for row in routed_rows if row.get("accepted_metric_guard")]
        best = max(
            [root_diag, *[row for row in routed_rows if row.get("valid")]],
            key=lambda item: (
                item.get("diagnostic_success", item.get("success_rate", -1.0)),
                item.get("elr", item.get("estimated_log_reliability", -999.0)),
                -item.get("cx", 999),
                item["name"],
            ),
        )
        payload["targets"][sample] = {
            "family": benchmark.family,
            "source_variant_count": len(variants),
            "routed_candidate_count": len(routed_rows),
            "accepted_metric_guard_count": len(accepted),
            "root": root_diag,
            "best": best,
            "accepted": accepted[:20],
            "top_candidates": routed_rows[:30],
            "rejection_counts": _rejection_counts(routed_rows),
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


def _semantic_route_compiler() -> NoiseAdaptiveCompiler:
    return NoiseAdaptiveCompiler(
        enable_parity_resynthesis=True,
        enable_exact_small_router=True,
        enable_route_scar_normalizer=True,
        enable_measurement_tail_pruning=True,
        enable_relative_phase_toffoli_rewrite=False,
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


def _source_variants(source: cirq.Circuit, rules: list[dict[str, Any]], *, limit: int) -> list[SourceVariant]:
    variants: list[SourceVariant] = []
    seen: set[tuple[str, ...]] = {tuple(str(op) for op in source.all_operations())}

    for rewrite in relative_phase_toffoli_rewrites(source, max_replacements=3):
        signature = tuple(str(op) for op in rewrite.circuit.all_operations())
        if signature in seen:
            continue
        seen.add(signature)
        variants.append(
            SourceVariant(
                name=rewrite.name,
                circuit=rewrite.circuit,
                kind="relative_phase_toffoli",
                old_source_cx=rewrite.old_cx,
                new_source_cx=rewrite.new_cx,
                proof="final_z_row_phase_miter",
            )
        )
        if len(variants) >= limit:
            return variants

    for variant in _verified_rule_variants(source, rules, limit=limit - len(variants)):
        signature = tuple(str(op) for op in variant.circuit.all_operations())
        if signature in seen:
            continue
        seen.add(signature)
        variants.append(variant)
        if len(variants) >= limit:
            break
    return variants


def _verified_rule_variants(
    source: cirq.Circuit,
    rules: list[dict[str, Any]],
    *,
    limit: int,
) -> list[SourceVariant]:
    if limit <= 0:
        return []
    ops = list(source.all_operations())
    if not ops or not isinstance(ops[-1].gate, cirq.MeasurementGate):
        return []
    body = ops[:-1]
    logical_qubits = tuple(sorted(source.all_qubits(), key=lambda qubit: qubit.x))
    out: list[SourceVariant] = []
    for rule_index, rule in enumerate(_interesting_rules(rules)):
        before = rule["before"]
        after = rule["after"]
        if not before:
            continue
        before_len = len(before)
        rule_qubits = sorted({qubit for item in [*before, *after] for qubit in item["qubits"]})
        if not rule_qubits:
            continue
        for start in range(0, len(body) - before_len + 1):
            window = body[start : start + before_len]
            for mapped in itertools.permutations(logical_qubits, len(rule_qubits)):
                mapping = dict(zip(rule_qubits, mapped))
                if not _window_matches(window, before, mapping):
                    continue
                replacement = [_operation_from_rule_op(item, mapping) for item in after]
                candidate = cirq.Circuit([*body[:start], *replacement, *body[start + before_len :], ops[-1]])
                if count_two_qubit_gates(candidate) > count_two_qubit_gates(source):
                    continue
                if not final_z_measurement_equivalent(source, candidate):
                    continue
                out.append(
                    SourceVariant(
                        name=f"verified_row_phase_{rule['family']}_{rule_index}_s{start}",
                        circuit=candidate,
                        kind=f"verified_rule:{rule['family']}",
                        old_source_cx=count_two_qubit_gates(source),
                        new_source_cx=count_two_qubit_gates(candidate),
                        proof="verified_ruleset_plus_full_final_z_miter",
                    )
                )
                if len(out) >= limit:
                    return out
    return out


def _interesting_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preferred = []
    rest = []
    for rule in rules:
        family = str(rule.get("family", ""))
        if family in {"relative_phase_toffoli", "mined_row_phase", "h_cx_h_reversal"}:
            preferred.append(rule)
        elif int(rule.get("before_ops", 0)) > int(rule.get("after_ops", 0)):
            rest.append(rule)
    return sorted(
        [*preferred, *rest],
        key=lambda rule: (
            int(rule.get("after_cx", 0)) - int(rule.get("before_cx", 0)),
            int(rule.get("after_ops", 0)) - int(rule.get("before_ops", 0)),
            str(rule.get("family", "")),
        ),
    )


def _route_and_score_variant(
    benchmark,
    backend: CompetitionBackend,
    root: CompilationCandidate,
    variant: SourceVariant,
    *,
    seed: int,
    shots: int,
    public_shots: int,
) -> dict[str, Any]:
    root_metrics = candidate_metrics(root, backend, benchmark.circuit)
    started = time.perf_counter()
    try:
        compiler = _semantic_route_compiler()
        compiled = compiler.compile(variant.circuit, backend, seed=seed)
        validate_or_raise(benchmark.circuit, compiled, backend)
    except Exception as exc:
        return {
            "name": variant.name,
            "kind": variant.kind,
            "valid": False,
            "accepted_metric_guard": False,
            "rejection_reason": f"{type(exc).__name__}: {exc}",
            "old_source_cx": variant.old_source_cx,
            "new_source_cx": variant.new_source_cx,
            "proof": variant.proof,
        }
    candidate = CompilationCandidate(
        variant.name,
        compiled,
        dict(compiler.last_initial_layout),
        time.perf_counter() - started,
    )
    metrics = candidate_metrics(candidate, backend, benchmark.circuit)
    accepted_metric = _metric_guard(metrics, root_metrics)
    diagnostic = run_track_a(benchmark, compiled, backend, shots=shots, seed=seed + _track_a_index(benchmark.name))
    public = run_track_a(benchmark, compiled, backend, shots=public_shots, seed=seed + _track_a_index(benchmark.name))
    return {
        "name": variant.name,
        "kind": variant.kind,
        "valid": True,
        "accepted_metric_guard": accepted_metric,
        "rejection_reason": "accepted" if accepted_metric else "metric_guard_failed",
        "old_source_cx": variant.old_source_cx,
        "new_source_cx": variant.new_source_cx,
        "proof": variant.proof,
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
        "layout": {str(logical.x): physical for logical, physical in sorted(candidate.layout.items(), key=lambda item: item[0].x)},
    }


def _metric_guard(metrics, root_metrics) -> bool:
    if metrics.two_qubit_count <= root_metrics.two_qubit_count - 2 and metrics.log_reliability >= root_metrics.log_reliability - 0.03:
        return True
    if metrics.log_reliability >= root_metrics.log_reliability + 0.02 and metrics.two_qubit_count <= root_metrics.two_qubit_count + 1:
        return True
    return False


def _simulate_candidate(
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
        "layout": {str(logical.x): physical for logical, physical in sorted(candidate.layout.items(), key=lambda item: item[0].x)},
    }


def _seed7_gate(payload: dict[str, Any]) -> dict[str, Any]:
    raw_deltas = {}
    public_deltas = {}
    for sample, data in payload["targets"].items():
        root = data["root"]
        best = data["best"]
        raw_deltas[sample] = best.get("diagnostic_success", 0.0) - root["diagnostic_success"]
        public_deltas[sample] = best.get("public_success", 0.0) - root["public_success"]
    # Track A has 8 equal samples with 0.75 total final weight.
    final_public_delta = sum(public_deltas.values()) * (0.75 / 8.0)
    return {
        "raw_deltas": raw_deltas,
        "public_deltas": public_deltas,
        "any_target_raw_plus_gate": (
            raw_deltas.get("track_a_007", 0.0) >= 0.04
            or raw_deltas.get("track_a_008", 0.0) >= 0.04
            or raw_deltas.get("track_a_002", 0.0) >= 0.03
        ),
        "final_seed7_delta_estimate": final_public_delta,
        "final_plus_0_003": final_public_delta >= 0.003,
        "keep": (
            (
                raw_deltas.get("track_a_007", 0.0) >= 0.04
                or raw_deltas.get("track_a_008", 0.0) >= 0.04
                or raw_deltas.get("track_a_002", 0.0) >= 0.03
            )
            and final_public_delta >= 0.003
        ),
    }


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Track A Semantic Mining Report",
        "",
        "This is a development-only targeted semantic mining run. It does not modify root `compiler.py`.",
        "",
        f"- Targets: `{', '.join(TARGETS)}`",
        f"- Rules loaded: `{payload['rule_count']}`",
        f"- Diagnostic shots: `{payload['diagnostic_shots']}`",
        f"- Public-style shots: `{payload['public_shots']}`",
        f"- Keep decision: `{payload['seed7_gate']['keep']}`",
        "",
    ]
    for sample, data in payload["targets"].items():
        root = data["root"]
        best = data["best"]
        lines.extend(
            [
                f"## {sample}",
                "",
                f"- Family: `{data['family']}`",
                f"- Source variants/routed candidates: `{data['source_variant_count']}` / `{data['routed_candidate_count']}`",
                f"- Metric-guard accepted: `{data['accepted_metric_guard_count']}`",
                f"- Root: diagnostic `{root['diagnostic_success']:.6f}`, public-style `{root['public_success']:.6f}`, CX `{root['cx']}`, depth `{root['depth']}`, ELR `{root['elr']:.6f}`.",
                f"- Best: `{best['name']}`, diagnostic `{best.get('diagnostic_success', 0):.6f}`, public-style `{best.get('public_success', 0):.6f}`, CX `{best.get('cx')}`, depth `{best.get('depth')}`, ELR `{best.get('elr', 0):.6f}`.",
                f"- Rejections: `{data['rejection_counts']}`",
                "",
                "| Candidate | Kind | Valid | Guard | Diag Success | Public Success | CX | Depth | ELR | Reason |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in data["top_candidates"][:12]:
            lines.append(
                f"| `{row['name']}` | `{row.get('kind', '')}` | {row.get('valid')} | {row.get('accepted_metric_guard')} | {row.get('diagnostic_success', 0):.6f} | {row.get('public_success', 0):.6f} | {row.get('cx')} | {row.get('depth')} | {row.get('elr', 0):.6f} | `{row.get('rejection_reason')}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            "Do not merge unless a target clears the raw-success gate and the public-style final delta clears `+0.003` with zero invalid candidates.",
        ]
    )
    return "\n".join(lines)


def _render_compare(payload: dict[str, Any]) -> str:
    gate = payload["seed7_gate"]
    lines = [
        "# Root vs Track A Semantic Mining Seed7",
        "",
        "| Metric | Value | Pass |",
        "|---|---:|---:|",
    ]
    for sample, delta in gate["raw_deltas"].items():
        threshold = 0.03 if sample == "track_a_002" else 0.04
        lines.append(f"| {sample} diagnostic raw delta | {delta:.6f} | {delta >= threshold} |")
    lines.append(f"| Final seed7 delta estimate | {gate['final_seed7_delta_estimate']:.6f} | {gate['final_plus_0_003']} |")
    lines.append(f"| Keep | {1 if gate['keep'] else 0} | {gate['keep']} |")
    lines.append("")
    return "\n".join(lines)


def _window_matches(window: list[cirq.Operation], pattern: list[dict[str, Any]], mapping: dict[int, cirq.Qid]) -> bool:
    if len(window) != len(pattern):
        return False
    for op, item in zip(window, pattern):
        if gate_name_for_operation(op) != item["gate"]:
            return False
        mapped_qubits = tuple(mapping[index] for index in item["qubits"])
        if tuple(op.qubits) != mapped_qubits:
            return False
    return True


def _operation_from_rule_op(item: dict[str, Any], mapping: dict[int, cirq.Qid]) -> cirq.Operation:
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


def _rejection_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("rejection_reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


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
