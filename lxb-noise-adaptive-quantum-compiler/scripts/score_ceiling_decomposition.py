from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from noise_adaptive_hackathon.evaluator.score_submission import score_submission
from noise_adaptive_hackathon.simulator.cirq_runner import run_track_a
from noise_adaptive_hackathon.simulator.stim_runner import run_track_b

from upper_bound_common import (
    ROOT,
    best_same_width_readout_cap,
    cap_for_rule_on_physical,
    compile_current_root,
    illegal_anchor_oracle_cap,
    legal_semantic_compressed_cap,
    load_public_samples,
    measurement_qubits,
    zero_readout_backend,
)


DEFAULT_SUMMARY = ROOT / "experiments" / "final_public_summary_seed7.json"
DEFAULT_JSON = ROOT / "experiments" / "upper_bound_decomposition.json"
DEFAULT_MD = ROOT / "experiments" / "upper_bound_decomposition.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Decompose public score ceilings by readout, output rule, and oracle assumptions.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shots-a", type=int, default=2048)
    parser.add_argument("--shots-b", type=int, default=4096)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    current_summary = json.loads(args.summary.read_text(encoding="utf-8"))
    current_by_name = {item["benchmark_name"]: item for item in current_summary["sample_results"]}
    samples = []
    for loaded in load_public_samples(ROOT):
        benchmark = loaded.benchmark
        backend = loaded.backend
        width = len(benchmark.logical_qubits)
        current = current_by_name[benchmark.name]
        compiled = compile_current_root(benchmark.circuit, backend, seed=args.seed)
        measured = measurement_qubits(compiled.circuit)
        current_measured_cap = cap_for_rule_on_physical(benchmark.expected_output_rule, measured, backend)
        best_cap, best_physical, best_method = best_same_width_readout_cap(benchmark.expected_output_rule, backend, width)
        legal_compressed_cap, legal_compressed_reason = legal_semantic_compressed_cap(
            benchmark.expected_output_rule, backend, width
        )
        illegal_cap, illegal_anchor_count, illegal_reason = illegal_anchor_oracle_cap(
            benchmark.expected_output_rule, backend, width
        )
        gate_only_backend = zero_readout_backend(backend)
        shots = args.shots_a if benchmark.track == "track_a" else args.shots_b
        runner = run_track_a if benchmark.track == "track_a" else run_track_b
        gate_only = runner(benchmark, compiled.circuit, gate_only_backend, shots=shots, seed=args.seed)
        samples.append(
            {
                "benchmark": benchmark.name,
                "track": benchmark.track,
                "family": benchmark.family,
                "rule_type": benchmark.expected_output_rule.rule_type,
                "width": width,
                "current_score": float(current["success_rate"]),
                "current_cx": int(current["two_qubit_gate_count"]),
                "current_depth": int(current["depth"]),
                "current_measured_qubits": list(measured),
                "current_measurement_readout_cap": current_measured_cap,
                "best_physical_readout_remap_ceiling": best_cap,
                "best_physical_qubits": list(best_physical),
                "best_physical_method": best_method,
                "zero_cx_same_output_ceiling": best_cap,
                "semantic_compressed_output_ceiling_legal": legal_compressed_cap,
                "semantic_compressed_output_legal_reason": legal_compressed_reason,
                "semantic_compressed_output_ceiling_illegal_oracle": illegal_cap,
                "semantic_compressed_illegal_anchor_count": illegal_anchor_count,
                "semantic_compressed_illegal_reason": illegal_reason,
                "oracle_readout_current_circuit_success": gate_only.success_rate,
                "oracle_readout_zero_cx_semantic_ceiling": 1.0,
                "gap_to_best_readout_remap": best_cap - float(current["success_rate"]),
                "gap_to_oracle_readout_current_circuit": gate_only.success_rate - float(current["success_rate"]),
            }
        )

    payload = build_payload(current_summary, samples)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["decision_gate"], indent=2, sort_keys=True))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


def build_payload(current_summary: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    same_width_track_a = statistics.fmean(
        row["best_physical_readout_remap_ceiling"] for row in samples if row["track"] == "track_a"
    )
    same_width_track_b = statistics.fmean(
        row["best_physical_readout_remap_ceiling"] for row in samples if row["track"] == "track_b"
    )
    same_width_score = score_submission(track_a_score=same_width_track_a, track_b_score=same_width_track_b)
    illegal_track_a = statistics.fmean(
        row["semantic_compressed_output_ceiling_illegal_oracle"] for row in samples if row["track"] == "track_a"
    )
    illegal_track_b = statistics.fmean(
        row["semantic_compressed_output_ceiling_illegal_oracle"] for row in samples if row["track"] == "track_b"
    )
    illegal_score = score_submission(track_a_score=illegal_track_a, track_b_score=illegal_track_b)
    gate_only_track_a = statistics.fmean(
        row["oracle_readout_current_circuit_success"] for row in samples if row["track"] == "track_a"
    )
    gate_only_track_b = statistics.fmean(
        row["oracle_readout_current_circuit_success"] for row in samples if row["track"] == "track_b"
    )
    gate_only_score = score_submission(track_a_score=gate_only_track_a, track_b_score=gate_only_track_b)
    b_rows = {row["benchmark"]: row for row in samples if row["track"] == "track_b"}
    track_b_illegal_final_lift = 0.25 * (illegal_track_b - same_width_track_b)
    return {
        "current_summary": {
            "track_a": current_summary["track_a_score"],
            "track_b": current_summary["track_b_score"],
            "final": current_summary["final_score"],
            "invalid": sum(1 for item in current_summary["sample_results"] if not item["valid"]),
        },
        "same_width_best_readout": {
            "track_a": same_width_track_a,
            "track_b": same_width_track_b,
            "final": same_width_score.final_score,
        },
        "oracle_readout_current_circuit": {
            "track_a": gate_only_track_a,
            "track_b": gate_only_track_b,
            "final": gate_only_score.final_score,
        },
        "illegal_compressed_oracle": {
            "track_a": illegal_track_a,
            "track_b": illegal_track_b,
            "final": illegal_score.final_score,
            "final_lift_over_legal_same_width": illegal_score.final_score - same_width_score.final_score,
        },
        "samples": samples,
        "answers": {
            "b001_b002_cap_is_fixed_output_width_or_measurement_set": _track_b_cap_answer(b_rows),
            "a007_a008_cap_can_final_measurement_remap_raise": _track_a_remap_answer(samples),
            "legal_fewer_readout_sensitive_bits_possible": "No. validate_compiler_equivalence requires compiled final measurement width to match logical width; expected-output rules index the full measured bitstring. Fewer anchor bits are only an illegal/classical-postprocessing oracle in this evaluator.",
        },
        "decision_gate": {
            "legal_same_width_final_ceiling": same_width_score.final_score,
            "legal_ceiling_below_0_92": same_width_score.final_score < 0.92,
            "track_b_illegal_semantic_readout_final_lift": track_b_illegal_final_lift,
            "track_b_illegal_lift_ge_0_04_final": track_b_illegal_final_lift >= 0.04,
        },
    }


def _track_b_cap_answer(rows: dict[str, dict[str, Any]]) -> str:
    parts = []
    for name in ("track_b_001", "track_b_002"):
        row = rows[name]
        current_cap = row["current_measurement_readout_cap"]
        best_cap = row["best_physical_readout_remap_ceiling"]
        delta = best_cap - current_cap
        parts.append(f"{name}: current measurement cap {current_cap:.6f}, best same-width cap {best_cap:.6f}, lift {delta:.6f}")
    return "; ".join(parts) + ". The remaining gap is dominated by mandatory same-width output readout, not by missing a large readout remap."


def _track_a_remap_answer(samples: list[dict[str, Any]]) -> str:
    parts = []
    for name in ("track_a_007", "track_a_008"):
        row = next(item for item in samples if item["benchmark"] == name)
        delta = row["best_physical_readout_remap_ceiling"] - row["current_measurement_readout_cap"]
        parts.append(f"{name}: current measurement cap {row['current_measurement_readout_cap']:.6f}, best same-width cap {row['best_physical_readout_remap_ceiling']:.6f}, lift {delta:.6f}")
    return "; ".join(parts)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Upper-Bound Score Ceiling Decomposition",
        "",
        "This is a development-only upper-bound analysis. It does not modify root `compiler.py`.",
        "",
        "## Aggregate Ceilings",
        "",
        "| Scenario | Track A | Track B | Final |",
        "|---|---:|---:|---:|",
        f"| Current seed7 | {payload['current_summary']['track_a']:.12f} | {payload['current_summary']['track_b']:.12f} | {payload['current_summary']['final']:.12f} |",
        f"| Best legal same-width readout remap, zero gate noise | {payload['same_width_best_readout']['track_a']:.12f} | {payload['same_width_best_readout']['track_b']:.12f} | {payload['same_width_best_readout']['final']:.12f} |",
        f"| Current circuit with oracle readout | {payload['oracle_readout_current_circuit']['track_a']:.12f} | {payload['oracle_readout_current_circuit']['track_b']:.12f} | {payload['oracle_readout_current_circuit']['final']:.12f} |",
        f"| Illegal compressed-output oracle | {payload['illegal_compressed_oracle']['track_a']:.12f} | {payload['illegal_compressed_oracle']['track_b']:.12f} | {payload['illegal_compressed_oracle']['final']:.12f} |",
        "",
        "## Direct Answers",
        "",
        f"- B001/B002 cap source: {payload['answers']['b001_b002_cap_is_fixed_output_width_or_measurement_set']}",
        f"- A007/A008 remap headroom: {payload['answers']['a007_a008_cap_can_final_measurement_remap_raise']}",
        f"- Fewer readout-sensitive bits: {payload['answers']['legal_fewer_readout_sensitive_bits_possible']}",
        "",
        "## Per-Sample Decomposition",
        "",
        "| Sample | Current | Current measured cap | Best same-width cap | Zero-CX same-output cap | Legal compressed cap | Illegal compressed oracle | Oracle-readout current | CX |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["samples"]:
        lines.append(
            f"| `{row['benchmark']}` | {row['current_score']:.6f} | {row['current_measurement_readout_cap']:.6f} | {row['best_physical_readout_remap_ceiling']:.6f} | {row['zero_cx_same_output_ceiling']:.6f} | {row['semantic_compressed_output_ceiling_legal']:.6f} | {row['semantic_compressed_output_ceiling_illegal_oracle']:.6f} | {row['oracle_readout_current_circuit_success']:.6f} | {row['current_cx']} |"
        )
    lines.extend(
        [
            "",
            "## Decision Gate",
            "",
            f"- Legal same-width final ceiling: `{payload['decision_gate']['legal_same_width_final_ceiling']:.12f}`.",
            f"- Legal ceiling below 0.92: `{payload['decision_gate']['legal_ceiling_below_0_92']}`.",
            f"- Track B illegal semantic-readout final lift: `{payload['decision_gate']['track_b_illegal_semantic_readout_final_lift']:.12f}`.",
            f"- Track B illegal lift >= 0.04 Final: `{payload['decision_gate']['track_b_illegal_lift_ge_0_04_final']}`.",
            "",
            "Legal conclusion: same-width final measurement is a hard evaluator constraint. Compressed anchor-bit output is useful as an oracle diagnostic, but not a valid submission route unless the evaluator contract changes.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

