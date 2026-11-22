from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.evaluator.equivalence import LayoutSidecar, validate_compiler_equivalence
from noise_adaptive_hackathon.evaluator.submission import _passes_noiseless_semantics
from noise_adaptive_hackathon.evaluator.validate_circuit import validate_compiled_circuit
from noise_adaptive_hackathon.formats.backend import CompetitionBackend
from noise_adaptive_hackathon.submission.compiler import NoiseAdaptiveCompiler
from noise_adaptive_hackathon.submission.route_scar_normalizer import route_scar_normalizer_diagnostics
from noise_adaptive_hackathon.submission.selection import candidate_metrics, rank_candidates_by_proxy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump route-scar normalizer diagnostics.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--sample", action="append", default=["track_a_007", "track_a_008"])
    parser.add_argument("--max-seconds", type=float, default=2.0)
    parser.add_argument(
        "--delta",
        type=Path,
        default=ROOT / "experiments" / "normalized_candidate_delta.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "experiments" / "route_scar_normalizer_report.md",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=ROOT / "experiments" / "a007_a008_normalizer_comparison.md",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "experiments" / "route_scar_cost_profile.jsonl",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.delta.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with args.delta.open("w", encoding="utf-8") as handle, args.profile.open("w", encoding="utf-8") as profile_handle:
        for sample in args.sample:
            benchmark_dir = args.root / "benchmarks" / "track_a" / "public" / sample
            if not benchmark_dir.exists():
                continue
            benchmark = load_benchmark(benchmark_dir)
            backend = _load_backend(args.root, benchmark)
            compiler = NoiseAdaptiveCompiler(
                enable_parity_resynthesis=True,
                enable_exact_small_router=True,
            )
            base_candidates = compiler.candidate_compilations(benchmark.circuit, backend)
            base_ranked = rank_candidates_by_proxy(base_candidates, backend, benchmark.circuit)
            diagnostics = route_scar_normalizer_diagnostics(
                benchmark.circuit,
                backend,
                base_ranked,
                max_seconds=args.max_seconds,
            )
            for row in diagnostics:
                payload = {
                    "sample": sample,
                    **row,
                }
                if row.get("accepted") is True:
                    payload.update(_semantic_labels(benchmark, backend, base_candidates, row))
                rows.append(payload)
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
                profile = dict(payload.get("profile", {}))
                profile.update(
                    {
                        "sample": sample,
                        "candidate_id": payload.get("base_candidate"),
                        "accepted": payload.get("accepted", False),
                        "reason_no_rewrite": payload.get("reason", profile.get("reason_no_rewrite", "unknown")),
                    }
                )
                profile_handle.write(json.dumps(profile, sort_keys=True) + "\n")
    args.report.write_text(_render_report(rows), encoding="utf-8")
    args.comparison.write_text(_render_comparison(rows), encoding="utf-8")
    return 0


def _semantic_labels(benchmark, backend: CompetitionBackend, base_candidates, row: dict[str, object]) -> dict[str, object]:
    base = next((candidate for candidate in base_candidates if candidate.name == row.get("base_candidate")), None)
    if base is None:
        return {"valid": False, "equivalent": False, "noiseless": False, "errors": ["missing base candidate"]}
    from noise_adaptive_hackathon.submission.route_scar_normalizer import normalize_route_scars

    result = normalize_route_scars(base.circuit)
    validation = validate_compiled_circuit(result.circuit, backend, benchmark.track)
    equivalent = False
    noiseless = False
    errors = list(validation.errors)
    if validation.valid:
        layout = LayoutSidecar({logical.x: physical for logical, physical in base.layout.items()})
        equivalence = validate_compiler_equivalence(benchmark, result.circuit, layout, seed=7)
        equivalent = equivalence.valid
        errors.extend(equivalence.errors)
        noiseless = equivalent and _passes_noiseless_semantics(benchmark, result.circuit, seed=7)
    return {
        "valid": validation.valid,
        "equivalent": equivalent,
        "noiseless": noiseless,
        "errors": errors,
    }


def _render_report(rows: list[dict[str, object]]) -> str:
    lines = ["# Route-Scar Normalizer Report", ""]
    accepted = [row for row in rows if row.get("accepted") is True]
    lines.append(f"- Accepted normalized candidates: {len(accepted)}")
    lines.append(f"- Invalid accepted candidates: {sum(1 for row in accepted if row.get('valid') is False)}")
    lines.append("")
    for row in accepted:
        lines.append(
            "- {sample} / {base}: CX {old_cx} -> {new_cx}, depth {old_depth} -> {new_depth}, "
            "ELR {old_elr:.6f} -> {new_elr:.6f}, rewrites={rewrites}".format(
                sample=row["sample"],
                base=row["base_candidate"],
                old_cx=int(row["old_cx"]),
                new_cx=int(row["new_cx"]),
                old_depth=int(row["old_depth"]),
                new_depth=int(row["new_depth"]),
                old_elr=float(row["old_elr"]),
                new_elr=float(row["new_elr"]),
                rewrites=row["rewrites"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _render_comparison(rows: list[dict[str, object]]) -> str:
    lines = ["# A007/A008 Route-Scar Normalizer Comparison", ""]
    for sample in ("track_a_007", "track_a_008"):
        sample_rows = [row for row in rows if row.get("sample") == sample]
        accepted = [row for row in sample_rows if row.get("accepted") is True]
        lines.append(f"## {sample}")
        lines.append(f"- Accepted candidates: {len(accepted)}")
        if accepted:
            best = min(accepted, key=lambda row: (int(row["new_cx"]), -float(row["new_elr"])))
            lines.append(
                f"- Best: {best['base_candidate']} | CX {best['old_cx']} -> {best['new_cx']} | "
                f"depth {best['old_depth']} -> {best['new_depth']} | "
                f"ELR {float(best['old_elr']):.6f} -> {float(best['new_elr']):.6f}"
            )
            lines.append(f"- Validator/equivalence/noiseless: {best.get('valid')}/{best.get('equivalent')}/{best.get('noiseless')}")
        reasons: dict[str, int] = {}
        for row in sample_rows:
            if row.get("accepted") is True:
                continue
            reason = str(row.get("reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
        lines.append("- Rejections: " + (", ".join(f"{key}={value}" for key, value in sorted(reasons.items())) or "none"))
        lines.append("")
    return "\n".join(lines)


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
