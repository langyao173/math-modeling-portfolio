from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from common import ROOT, load_json


FINAL_VARIANTS = (
    "final_c0_conservative",
    "v19_relative_phase_toffoli",
    "final_c1_b_relabel",
    "final_c2_tracka_exact",
    "final_c3_b_relabel_tracka_exact",
    "final_c4_all_safe",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final-submission choice scoreboard for C0-C4.")
    parser.add_argument("--assets", type=Path, default=ROOT / "experiments" / "private_emulator_v1")
    parser.add_argument("--baseline", default="final_c0_conservative")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-report", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    variants = tuple(args.variant or FINAL_VARIANTS)
    manifest = load_json(args.assets / "suite_manifest.json")
    matrices = _load_matrices(args.assets, manifest)
    rows = [_build_row(variant, matrices, args.baseline) for variant in variants]
    rows = [row for row in rows if row is not None]
    rows.sort(key=lambda row: float(row["choice_score"]), reverse=True)

    out_csv = args.out_csv or args.assets / "submission_choice_scoreboard.csv"
    out_report = args.out_report or args.assets / "submission_choice_report.md"
    _write_csv(out_csv, rows)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(_render_report(rows, args.baseline), encoding="utf-8")
    return 0


def _load_matrices(assets: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    matrices: dict[str, dict[str, Any]] = {}
    for suite_payload in manifest.get("suites", []):
        path = assets / str(suite_payload["matrix_output"])
        if path.exists():
            matrices[str(suite_payload["name"])] = load_json(path)
    return matrices


def _build_row(
    variant: str,
    matrices: dict[str, dict[str, Any]],
    baseline: str,
) -> dict[str, Any] | None:
    public = _payload(matrices, "public_replay", variant)
    tracka = _payload(matrices, "tracka_structural", variant)
    trackb = _payload(matrices, "trackb_affine", variant)
    jitter = _payload(matrices, "backend_jitter", variant)
    if public is None:
        return None

    baseline_public = _payload(matrices, "public_replay", baseline)
    baseline_tracka = _payload(matrices, "tracka_structural", baseline)
    baseline_trackb = _payload(matrices, "trackb_affine", baseline)
    baseline_jitter = _payload(matrices, "backend_jitter", baseline)
    public_mean = _suite_score(public, "final")
    worst_seed = _worst_seed(public)
    synthetic_hard = _suite_score(tracka, "track_a")
    synthetic_affine = _suite_score(trackb, "track_b")
    backend_jitter = _suite_score(jitter, "final")
    public_invalid_count = _total_invalid(public)
    aggregate_invalid_count = _total_invalid(public, tracka, trackb, jitter)
    p95_compile = max(_float(payload, "p95_compile_time_sec") for payload in (public, tracka, trackb, jitter) if payload)
    max_compile = max(_float(payload, "max_compile_time_sec") for payload in (public, tracka, trackb, jitter) if payload)
    compile_risk = max(0.0, p95_compile - 12.0) / 12.0 + max(0.0, max_compile - 30.0) / 30.0
    invalid_penalty = 1.0 if aggregate_invalid_count else 0.0
    choice_score = (
        0.35 * public_mean
        + 0.20 * worst_seed
        + 0.20 * synthetic_hard
        + 0.10 * synthetic_affine
        - 0.10 * invalid_penalty
        - 0.05 * compile_risk
    )

    public_delta = public_mean - (_suite_score(baseline_public, "final") if baseline_public else 0.0)
    baseline_aggregate_invalid_count = _total_invalid(
        baseline_public,
        baseline_tracka,
        baseline_trackb,
        baseline_jitter,
    )
    negative_public_seed_count = _negative_seed_count(public, baseline_public)
    public_seed_deltas = _seed_deltas(public, baseline_public)
    aggressive_gate = (
        aggregate_invalid_count == 0
        or aggregate_invalid_count <= baseline_aggregate_invalid_count
    ) and (
        public_invalid_count == 0
        and p95_compile < 12.0
        and public_delta >= 0.004
        and negative_public_seed_count == 0
    )
    gate_label = "baseline" if variant == baseline else ("pass" if aggressive_gate else "fail")
    return {
        "variant": variant,
        "choice_score": choice_score,
        "public_5seed_mean": public_mean,
        "public_delta_vs_baseline": public_delta,
        "worst_seed_final": worst_seed,
        "synthetic_hard_mean": synthetic_hard,
        "synthetic_affine_trackb": synthetic_affine,
        "backend_jitter_mean": backend_jitter,
        "public_invalid_count": public_invalid_count,
        "aggregate_invalid_count": aggregate_invalid_count,
        "negative_public_seed_count": negative_public_seed_count,
        "public_seed_deltas": ";".join(f"{delta:.6f}" for delta in public_seed_deltas),
        "p95_compile_time_sec": p95_compile,
        "max_compile_time_sec": max_compile,
        "compile_time_risk": compile_risk,
        "aggressive_gate": gate_label,
    }


def _payload(matrices: dict[str, dict[str, Any]], suite: str, variant: str) -> dict[str, Any] | None:
    matrix = matrices.get(suite)
    if not matrix:
        return None
    payload = matrix.get("variants", {}).get(variant)
    return payload if isinstance(payload, dict) else None


def _suite_score(payload: dict[str, Any] | None, preferred: str) -> float:
    if not payload:
        return 0.0
    key_by_preference = {
        "final": "mean_final_score",
        "track_a": "mean_track_a_score",
        "track_b": "mean_track_b_score",
    }
    key = key_by_preference[preferred]
    value = _float(payload, key)
    if value:
        return value
    return max(
        _float(payload, "mean_final_score"),
        _float(payload, "mean_track_a_score"),
        _float(payload, "mean_track_b_score"),
    )


def _worst_seed(payload: dict[str, Any]) -> float:
    seeds = payload.get("seeds", [])
    scores = [_float(seed, "final_score") for seed in seeds if isinstance(seed, dict)]
    return min(scores) if scores else _float(payload, "mean_final_score")


def _total_invalid(*payloads: dict[str, Any] | None) -> int:
    total = 0
    for payload in payloads:
        if not payload:
            continue
        for seed in payload.get("seeds", []):
            if isinstance(seed, dict):
                total += int(seed.get("invalid_count", 0))
    return total


def _seed_deltas(payload: dict[str, Any], baseline: dict[str, Any] | None) -> list[float]:
    if baseline is None:
        return []
    base_by_seed = {
        int(seed["seed"]): _float(seed, "final_score")
        for seed in baseline.get("seeds", [])
        if isinstance(seed, dict) and "seed" in seed
    }
    deltas: list[float] = []
    for seed in payload.get("seeds", []):
        if not isinstance(seed, dict) or "seed" not in seed:
            continue
        seed_id = int(seed["seed"])
        if seed_id in base_by_seed:
            deltas.append(_float(seed, "final_score") - base_by_seed[seed_id])
    return deltas


def _negative_seed_count(payload: dict[str, Any], baseline: dict[str, Any] | None) -> int:
    return sum(1 for delta in _seed_deltas(payload, baseline) if delta < 0.0)


def _float(payload: dict[str, Any], key: str) -> float:
    try:
        return float(payload.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "choice_score",
        "public_5seed_mean",
        "public_delta_vs_baseline",
        "worst_seed_final",
        "synthetic_hard_mean",
        "synthetic_affine_trackb",
        "backend_jitter_mean",
        "public_invalid_count",
        "aggregate_invalid_count",
        "negative_public_seed_count",
        "public_seed_deltas",
        "p95_compile_time_sec",
        "max_compile_time_sec",
        "compile_time_risk",
        "aggressive_gate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(rows: list[dict[str, Any]], baseline: str) -> str:
    recommended = _recommended_variant(rows, baseline)
    lines = [
        "# Final Submission Choice Report",
        "",
        f"Baseline: `{baseline}`",
        f"Recommended submission: `{recommended}`",
        "",
        "Aggressive gate: public invalids = 0, no additional aggregate holdout invalids versus baseline, p95 compile < 12s, public 5-seed mean delta >= 0.004, and no negative public seed deltas.",
        "",
        "Choice score:",
        "`0.35 * public_5seed_mean + 0.20 * worst_seed_final + 0.20 * synthetic_hard_mean + 0.10 * synthetic_affine_trackb - 0.10 * invalid_penalty - 0.05 * compile_time_risk`",
        "",
        "| Variant | Score | Public mean | Delta | Worst seed | Synthetic A | Synthetic B | Public invalid | Aggregate invalid | p95 | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {choice_score:.6f} | {public_5seed_mean:.6f} | "
            "{public_delta_vs_baseline:.6f} | {worst_seed_final:.6f} | "
            "{synthetic_hard_mean:.6f} | {synthetic_affine_trackb:.6f} | "
            "{public_invalid_count} | {aggregate_invalid_count} | "
            "{p95_compile_time_sec:.3f}s | {aggressive_gate} |".format(**row)
        )
    lines.extend(["", "## Public Seed Deltas", ""])
    for row in rows:
        lines.append(f"- `{row['variant']}`: {row['public_seed_deltas'] or 'n/a'}")
    lines.append("")
    return "\n".join(lines)


def _recommended_variant(rows: list[dict[str, Any]], baseline: str) -> str:
    if not rows:
        return baseline
    for row in rows:
        if row["variant"] != baseline and row["aggressive_gate"] == "pass":
            return str(row["variant"])
    return baseline


if __name__ == "__main__":
    raise SystemExit(main())
