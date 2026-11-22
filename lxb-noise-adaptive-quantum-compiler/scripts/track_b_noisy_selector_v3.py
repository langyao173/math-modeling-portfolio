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
from noise_adaptive_hackathon.simulator.stim_runner import run_track_b
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.readout_copyout import readout_copyout_candidates
from noise_adaptive_hackathon.submission.selection import CompilationCandidate, candidate_metrics
from noise_adaptive_hackathon.submission.simulation_selector import SELECTOR_PROXY_ONLY, SimulationSelectorConfig
from noise_adaptive_hackathon.submission.validation import validate_or_raise

from upper_bound_common import compile_current_root, load_public_samples


DEFAULT_OUT_JSON = ROOT / "experiments" / "track_b_noisy_selector_v3.json"
DEFAULT_OUT_MD = ROOT / "experiments" / "track_b_noisy_selector_v3.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Development-only Track B same-width noisy candidate selector.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lowshot", type=int, default=4096)
    parser.add_argument("--public-shots", type=int, default=1024)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    samples: dict[str, Any] = {}
    public_deltas: dict[str, float] = {}
    for loaded in load_public_samples(ROOT, targets=("track_b_001", "track_b_002")):
        row = probe_track_b_sample(
            loaded.benchmark,
            loaded.backend,
            seed=args.seed,
            lowshot=args.lowshot,
            public_shots=args.public_shots,
        )
        samples[loaded.name] = row
        public_deltas[loaded.name] = row["selected"]["public_success"] - row["root"]["public_success"]
    track_b_delta = sum(public_deltas.values()) / 2.0
    final_delta = 0.25 * track_b_delta
    payload = {
        "variant": "track_b_noisy_selector_v3_dev_only",
        "seed": args.seed,
        "lowshot": args.lowshot,
        "public_shots": args.public_shots,
        "samples": samples,
        "decision": {
            "public_deltas": public_deltas,
            "track_b_seed7_delta": track_b_delta,
            "final_seed7_delta": final_delta,
            "keep": final_delta >= 0.004,
            "recommendation": "Do not merge" if final_delta < 0.004 else "Candidate selector branch clears seed7 gate; run multi-seed before root change.",
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


def probe_track_b_sample(benchmark, backend, *, seed: int, lowshot: int, public_shots: int) -> dict[str, Any]:
    root_candidate = compile_current_root(benchmark.circuit, backend, seed=seed)
    root_row = score_candidate(benchmark, backend, root_candidate, seed=seed, lowshot=lowshot, public_shots=public_shots)
    candidates = candidate_bank(benchmark.circuit, backend, seed=seed)
    candidates.insert(0, root_candidate)
    candidates = dedupe_candidates(candidates)

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = validate_and_score(benchmark, backend, candidate, seed=seed, lowshot=lowshot, public_shots=public_shots)
        rows.append(row)
    valid = [row for row in rows if row["valid"]]
    selected = max(
        valid or [root_row],
        key=lambda row: (
            row["lowshot_success"],
            row["public_success"],
            row["log_reliability"],
            -row["cx"],
            -row["depth"],
            -row["readout_cost"],
            row["name"],
        ),
    )
    if selected["public_success"] < root_row["public_success"] - 1e-12:
        selected = root_row | {"selection_note": "root_fallback_public_guard"}
    return {
        "root": root_row,
        "selected": selected,
        "candidate_count": len(rows),
        "valid_count": len(valid),
        "top_candidates": sorted(
            rows,
            key=lambda row: (
                row["valid"],
                row.get("lowshot_success", -1.0),
                row.get("public_success", -1.0),
                row.get("log_reliability", -99.0),
                -row.get("cx", 999),
                row.get("name", ""),
            ),
            reverse=True,
        )[:20],
        "rejection_counts": rejection_counts(rows),
    }


def candidate_bank(source: cirq.Circuit, backend, *, seed: int) -> list[CompilationCandidate]:
    configs = [
        ("root_bank", dict(
            enable_parity_resynthesis=True,
            enable_ghz_leaf_tree=True,
            enable_measurement_channel_physical_placement_v2=True,
            measurement_channel_physical_placement_v2_limit=2,
            enable_ghz_relabel=True,
            enable_readout_copyout=True,
        )),
        ("wide_placement", dict(
            enable_parity_resynthesis=True,
            enable_measurement_channel_general=True,
            enable_measurement_channel_physical_placement_v2=True,
            measurement_channel_physical_placement_v2_limit=8,
            measurement_channel_physical_placement_v2_max_seconds=8.0,
            enable_ghz_leaf_tree=True,
            enable_ghz_relabel=True,
            enable_track_b_relabel=True,
            enable_readout_heavy_relabel=True,
        )),
        ("readout_proxy", dict(
            enable_parity_resynthesis=True,
            enable_measurement_channel_placement=True,
            enable_measurement_channel_physical_placement_v2=True,
            measurement_channel_physical_placement_v2_limit=4,
            measurement_channel_physical_placement_v2_max_seconds=5.0,
            enable_ghz_leaf_tree=True,
            enable_ghz_relabel=True,
            enable_readout_heavy_relabel=True,
        )),
    ]
    out: list[CompilationCandidate] = []
    for prefix, kwargs in configs:
        compiler = NoiseAdaptiveCompiler(
            **kwargs,
            selector_mode=SELECTOR_PROXY_ONLY,
            selector_config=SimulationSelectorConfig(mode=SELECTOR_PROXY_ONLY),
        )
        try:
            candidates = compiler.candidate_compilations(source, backend, seed=seed)
        except Exception:
            continue
        for candidate in candidates:
            out.append(CompilationCandidate(f"{prefix}:{candidate.name}", candidate.circuit, candidate.layout, candidate.compile_time_sec))
        try:
            additions = readout_copyout_candidates(source, backend, candidates)
            for candidate in additions:
                out.append(CompilationCandidate(f"{prefix}:{candidate.name}", candidate.circuit, candidate.layout, candidate.compile_time_sec))
        except Exception:
            pass
    return out


def validate_and_score(benchmark, backend, candidate: CompilationCandidate, *, seed: int, lowshot: int, public_shots: int) -> dict[str, Any]:
    width = len(benchmark.logical_qubits)
    measurements = [op for op in candidate.circuit.all_operations() if isinstance(op.gate, cirq.MeasurementGate)]
    if not measurements or len(measurements[-1].qubits) != width:
        return invalid_row(candidate.name, "measurement_width_mismatch")
    validation = validate_compiled_circuit(candidate.circuit, backend, benchmark.track)
    if not validation.valid:
        return invalid_row(candidate.name, "basic_validator_failed", validation.errors)
    try:
        validate_or_raise(benchmark.circuit, candidate.circuit, backend)
    except Exception as exc:
        return invalid_row(candidate.name, f"submission_validate_or_raise:{type(exc).__name__}")
    equivalence = validate_compiler_equivalence(
        benchmark,
        candidate.circuit,
        LayoutSidecar({logical.x: physical for logical, physical in candidate.layout.items()}),
        seed=seed,
    )
    if not equivalence.valid:
        return invalid_row(candidate.name, "official_equivalence_failed", equivalence.errors)
    try:
        result = cirq.CliffordSimulator(seed=seed).run(candidate.circuit, repetitions=256)
        if not all(benchmark.expected_output_rule.is_correct("".join(str(int(bit)) for bit in row)) for row in result.measurements["z"]):
            return invalid_row(candidate.name, "noiseless_rule_failed")
    except Exception as exc:
        return invalid_row(candidate.name, f"noiseless_error:{type(exc).__name__}")
    return score_candidate(benchmark, backend, candidate, seed=seed, lowshot=lowshot, public_shots=public_shots)


def score_candidate(benchmark, backend, candidate: CompilationCandidate, *, seed: int, lowshot: int, public_shots: int) -> dict[str, Any]:
    metrics = candidate_metrics(candidate, backend, benchmark.circuit)
    low = run_track_b(benchmark, candidate.circuit, backend, shots=lowshot, seed=seed)
    public = run_track_b(benchmark, candidate.circuit, backend, shots=public_shots, seed=seed)
    return {
        "name": candidate.name,
        "valid": True,
        "lowshot_success": low.success_rate,
        "lowshot_correct": low.correct_shots,
        "lowshot_shots": low.total_shots,
        "public_success": public.success_rate,
        "public_correct": public.correct_shots,
        "public_shots": public.total_shots,
        "cx": metrics.two_qubit_count,
        "depth": metrics.depth,
        "log_reliability": metrics.log_reliability,
        "readout_cost": metrics.readout_cost,
        "sensitivity": metrics.clifford_sensitivity_cost,
        "compile_time_sec": metrics.compile_time_sec,
    }


def invalid_row(name: str, reason: str, errors: list[str] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "valid": False,
        "rejection_reason": reason,
        "errors": errors or [],
        "lowshot_success": -1.0,
        "public_success": -1.0,
    }


def dedupe_candidates(candidates: list[CompilationCandidate]) -> list[CompilationCandidate]:
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


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Track B Noisy Selector V3",
        "",
        "Development-only same-width candidate selector. Root compiler was not modified.",
        "",
        f"- Seed: `{payload['seed']}`",
        f"- Lowshot: `{payload['lowshot']}`",
        f"- Public shots: `{payload['public_shots']}`",
        f"- Final seed7 delta estimate: `{payload['decision']['final_seed7_delta']:.6f}`",
        f"- Keep: `{payload['decision']['keep']}`",
        "",
        "| Sample | Root public | Selected public | Delta | Root CX | Selected CX | Selected |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in payload["samples"].items():
        root = row["root"]
        selected = row["selected"]
        lines.append(
            f"| `{name}` | {root['public_success']:.6f} | {selected['public_success']:.6f} | {selected['public_success'] - root['public_success']:.6f} | {root['cx']} | {selected['cx']} | `{selected['name']}` |"
        )
    lines.extend(["", "## Candidate Details", ""])
    for name, row in payload["samples"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Candidate count: `{row['candidate_count']}`, valid `{row['valid_count']}`.",
                f"- Rejections: `{row['rejection_counts']}`.",
                "",
                "| Candidate | Valid | Lowshot | Public | CX | Depth | ELR | Readout | Sensitivity | Reason |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for item in row["top_candidates"][:12]:
            lines.append(
                f"| `{item['name']}` | {item['valid']} | {item.get('lowshot_success', -1):.6f} | {item.get('public_success', -1):.6f} | {item.get('cx')} | {item.get('depth')} | {item.get('log_reliability', 0):.6f} | {item.get('readout_cost', 0):.6f} | {item.get('sensitivity', 0):.6f} | `{item.get('rejection_reason', '')}` |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

