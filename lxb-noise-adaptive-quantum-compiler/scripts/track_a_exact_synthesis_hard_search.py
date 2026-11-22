from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cirq

from noise_adaptive_hackathon.evaluator.equivalence import LayoutSidecar, validate_compiler_equivalence
from noise_adaptive_hackathon.evaluator.validate_circuit import validate_compiled_circuit
from noise_adaptive_hackathon.simulator.cirq_runner import run_track_a
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.relative_phase_toffoli import (
    final_z_measurement_equivalent,
    relative_phase_toffoli_rewrites,
)
from noise_adaptive_hackathon.submission.selection import CompilationCandidate, candidate_metrics
from noise_adaptive_hackathon.submission.simulation_selector import SELECTOR_PROXY_ONLY, SimulationSelectorConfig
from noise_adaptive_hackathon.submission.validation import validate_or_raise

from upper_bound_common import (
    ROOT as COMMON_ROOT,
    affine_model,
    affine_rows_from_columns,
    compile_current_root,
    exact_cnot_synthesis_mitm,
    learn_basis_truth_table,
    load_public_samples,
)


TARGETS = ("track_a_002", "track_a_007", "track_a_008")
DEFAULT_OUT_JSON = ROOT / "experiments" / "track_a_exact_synthesis_hard_search.json"
DEFAULT_OUT_MD = ROOT / "experiments" / "track_a_exact_synthesis_hard_search.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Development-only hard exact synthesis search for A002/A007/A008.")
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    targets = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    rows: dict[str, Any] = {}
    for loaded in load_public_samples(COMMON_ROOT, targets=targets):
        rows[loaded.name] = probe_sample(loaded.benchmark, loaded.backend, seed=args.seed, shots=args.shots)
    payload = {
        "variant": "track_a_exact_synthesis_hard_search_dev_only",
        "seed": args.seed,
        "shots": args.shots,
        "samples": rows,
        "decision": decision(rows),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


def probe_sample(benchmark, backend, *, seed: int, shots: int) -> dict[str, Any]:
    root = compile_current_root(benchmark.circuit, backend, seed=seed)
    root_row = score_track_a_candidate(benchmark, backend, root, seed=seed, shots=shots)

    truth_outputs, deterministic = learn_basis_truth_table(benchmark.circuit)
    is_affine, constant, columns = affine_model(truth_outputs, len(benchmark.logical_qubits))
    source_candidates = []
    if benchmark.name == "track_a_002" and is_affine and columns:
        rows = affine_rows_from_columns(columns, len(benchmark.logical_qubits))
        for max_cx in (6, 7, 8):
            sequence = exact_cnot_synthesis_mitm(rows, len(benchmark.logical_qubits), max_cx)
            source_candidates.append(
                {
                    "name": f"a002_affine_mitm_le_{max_cx}",
                    "kind": "affine_cnot_mitm",
                    "max_cx": max_cx,
                    "found": sequence is not None,
                    "source_cx": len(sequence) if sequence is not None else None,
                    "sequence": list(sequence) if sequence is not None else [],
                }
            )
    if benchmark.name in {"track_a_007", "track_a_008"}:
        for rewrite in relative_phase_toffoli_rewrites(benchmark.circuit, max_replacements=4)[:16]:
            source_candidates.append(
                {
                    "name": rewrite.name,
                    "kind": "row_phase_relative_toffoli",
                    "found": True,
                    "source_cx": rewrite.new_cx,
                    "old_source_cx": rewrite.old_cx,
                    "replacements": rewrite.replacements,
                }
            )

    routed_rows = []
    for candidate in route_source_candidates(benchmark, backend, seed=seed):
        routed_rows.append(validate_score_routed(benchmark, backend, root, candidate, seed=seed, shots=shots))
    best = max(
        [root_row] + [row for row in routed_rows if row["valid"]],
        key=lambda row: (
            -row["cx"],
            row["success"],
            row["log_reliability"],
            -row["depth"],
            row["name"],
        ),
    )
    return {
        "root": root_row,
        "truth": {
            "deterministic_basis_map": deterministic,
            "is_affine": is_affine,
            "affine_constant": constant,
            "affine_columns": list(columns),
        },
        "source_exact_candidates": source_candidates,
        "routed_candidate_count": len(routed_rows),
        "valid_routed_count": sum(1 for row in routed_rows if row["valid"]),
        "best": best,
        "target_hit": target_hit(benchmark.name, best, root_row),
        "top_routed": sorted(
            routed_rows,
            key=lambda row: (row["valid"], -row.get("cx", 999), row.get("success", -1), row.get("log_reliability", -99), row.get("name", "")),
            reverse=True,
        )[:20],
        "rejection_counts": rejection_counts(routed_rows),
    }


def aggressive_track_a_compiler() -> NoiseAdaptiveCompiler:
    return NoiseAdaptiveCompiler(
        enable_exact_small_router=True,
        enable_broad_exact_small_router=True,
        enable_route_scar_normalizer=True,
        enable_relative_phase_toffoli_rewrite=True,
        enable_relative_phase_exact_router=True,
        enable_track_a_affine_collapse=True,
        enable_track_a_semantic_resynthesis=True,
        enable_phase_affine_cone_resynthesis=True,
        enable_h_boundary_pathsum_resynthesis=True,
        enable_measurement_tail_pruning=True,
        selector_mode=SELECTOR_PROXY_ONLY,
        selector_config=SimulationSelectorConfig(mode=SELECTOR_PROXY_ONLY),
    )


def route_source_candidates(benchmark, backend, *, seed: int) -> list[CompilationCandidate]:
    compiler = aggressive_track_a_compiler()
    candidates: list[CompilationCandidate] = []
    try:
        candidates.extend(compiler.candidate_compilations(benchmark.circuit, backend, seed=seed))
    except Exception:
        pass
    rewrites = relative_phase_toffoli_rewrites(benchmark.circuit, max_replacements=4)[:8]
    for rewrite in rewrites:
        try:
            for candidate in compiler.candidate_compilations(rewrite.circuit, backend, seed=seed):
                candidates.append(CompilationCandidate(f"{rewrite.name}:{candidate.name}", candidate.circuit, candidate.layout, candidate.compile_time_sec))
        except Exception:
            continue
    return dedupe(candidates)


def validate_score_routed(benchmark, backend, root: CompilationCandidate, candidate: CompilationCandidate, *, seed: int, shots: int) -> dict[str, Any]:
    validation = validate_compiled_circuit(candidate.circuit, backend, benchmark.track)
    if not validation.valid:
        return invalid_row(candidate.name, "basic_validator_failed", validation.errors)
    try:
        validate_or_raise(benchmark.circuit, candidate.circuit, backend)
    except Exception as exc:
        return invalid_row(candidate.name, f"validate_or_raise:{type(exc).__name__}")
    physicalized = physicalize_source(benchmark.circuit, candidate.layout)
    if len(benchmark.logical_qubits) <= 7 and not final_z_measurement_equivalent(physicalized, candidate.circuit):
        return invalid_row(candidate.name, "physical_final_z_miter_failed")
    equivalence = validate_compiler_equivalence(
        benchmark,
        candidate.circuit,
        LayoutSidecar({logical.x: physical for logical, physical in candidate.layout.items()}),
        seed=seed,
    )
    if not equivalence.valid:
        return invalid_row(candidate.name, "official_equivalence_failed", equivalence.errors)
    return score_track_a_candidate(benchmark, backend, candidate, seed=seed, shots=shots)


def score_track_a_candidate(benchmark, backend, candidate: CompilationCandidate, *, seed: int, shots: int) -> dict[str, Any]:
    metrics = candidate_metrics(candidate, backend, benchmark.circuit)
    run = run_track_a(benchmark, candidate.circuit, backend, shots=shots, seed=seed)
    return {
        "name": candidate.name,
        "valid": True,
        "success": run.success_rate,
        "correct": run.correct_shots,
        "shots": run.total_shots,
        "cx": metrics.two_qubit_count,
        "depth": metrics.depth,
        "log_reliability": metrics.log_reliability,
        "readout_cost": metrics.readout_cost,
        "compile_time_sec": metrics.compile_time_sec,
    }


def physicalize_source(source: cirq.Circuit, layout: dict[cirq.LineQubit, int]) -> cirq.Circuit:
    mapping = {logical: cirq.LineQubit(physical) for logical, physical in layout.items()}
    return source.transform_qubits(lambda qubit: mapping.get(qubit, qubit))


def target_hit(name: str, row: dict[str, Any], root: dict[str, Any]) -> bool:
    if not row.get("valid", False):
        return False
    if name == "track_a_002":
        return row["cx"] <= 7 and row["cx"] < root["cx"]
    return row["cx"] <= 13 and row["cx"] < root["cx"]


def invalid_row(name: str, reason: str, errors: list[str] | None = None) -> dict[str, Any]:
    return {"name": name, "valid": False, "rejection_reason": reason, "errors": errors or [], "cx": 999, "success": -1.0}


def dedupe(candidates: list[CompilationCandidate]) -> list[CompilationCandidate]:
    out: list[CompilationCandidate] = []
    seen: set[tuple[tuple[str, ...], tuple[tuple[int, int], ...]]] = set()
    for candidate in candidates:
        key = (
            tuple(str(op) for op in candidate.circuit.all_operations()),
            tuple(sorted((logical.x, physical) for logical, physical in candidate.layout.items())),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def rejection_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row["valid"]:
            continue
        reason = row.get("rejection_reason", "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def decision(rows: dict[str, Any]) -> dict[str, Any]:
    hits = {name: row["target_hit"] for name, row in rows.items()}
    return {
        "target_hits": hits,
        "any_target_hit": any(hits.values()),
        "recommendation": "Do not merge; no A002 <=7 or A007/A008 <=13 valid candidate found." if not any(hits.values()) else "Target hit found; run public multi-seed before any root change.",
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Track A Exact Synthesis Hard Search",
        "",
        "Development-only hard search for A002/A007/A008. Root compiler was not modified.",
        "",
        f"- Decision: `{payload['decision']['recommendation']}`",
        "",
        "| Sample | Root CX | Best CX | Root success | Best success | Target hit | Best candidate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in payload["samples"].items():
        root = row["root"]
        best = row["best"]
        lines.append(
            f"| `{name}` | {root['cx']} | {best.get('cx')} | {root['success']:.6f} | {best.get('success', -1):.6f} | {row['target_hit']} | `{best.get('name')}` |"
        )
    for name, row in payload["samples"].items():
        lines.extend(
            [
                "",
                f"## {name}",
                "",
                f"- Truth: affine `{row['truth']['is_affine']}`, deterministic `{row['truth']['deterministic_basis_map']}`.",
                f"- Routed candidates: `{row['routed_candidate_count']}`, valid `{row['valid_routed_count']}`.",
                f"- Rejections: `{row['rejection_counts']}`.",
                "",
                "### Source Exact Candidates",
                "",
                "| Candidate | Kind | Found | Source CX | Max CX |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for candidate in row["source_exact_candidates"]:
            lines.append(
                f"| `{candidate['name']}` | `{candidate['kind']}` | {candidate.get('found')} | {candidate.get('source_cx')} | {candidate.get('max_cx')} |"
            )
        lines.extend(["", "### Top Routed", "", "| Candidate | Valid | CX | Success | Depth | ELR | Reason |", "|---|---:|---:|---:|---:|---:|---|"])
        for item in row["top_routed"][:12]:
            lines.append(
                f"| `{item['name']}` | {item['valid']} | {item.get('cx')} | {item.get('success', -1):.6f} | {item.get('depth')} | {item.get('log_reliability', 0):.6f} | `{item.get('rejection_reason', '')}` |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
