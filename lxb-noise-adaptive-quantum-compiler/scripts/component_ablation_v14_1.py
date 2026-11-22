from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT, write_json


VARIANTS = (
    "v13_root",
    "v13_plus_b_relabel_only",
    "v13_plus_tracka_exact_router_only",
    "v13_plus_tracka_templates_only",
    "v14_combined",
)
SEEDS = (7, 11, 17, 23, 31)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate V14.1 component ablation outputs.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "component_ablation_v14_1.json")
    parser.add_argument("--decision", type=Path, default=ROOT / "experiments" / "promote_decision_v14_1.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    public = _public_results(args.root)
    affine = _holdout_results(args.root, prefix="synthetic_trackb_affine_v14_1")
    small_tracka = _holdout_results(args.root, prefix="synthetic_small_tracka_v14_1")
    gates = _gate_summary(public, affine)
    payload = {
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "public": public,
        "synthetic_affine_trackb": affine,
        "synthetic_small_tracka": small_tracka,
        "gate_summary": gates,
    }
    write_json(args.out, payload)
    args.decision.parent.mkdir(parents=True, exist_ok=True)
    args.decision.write_text(_decision_markdown(gates, public, affine, small_tracka), encoding="utf-8")
    return 0


def _public_results(root: Path) -> dict[str, object]:
    results: dict[str, object] = {}
    for variant in VARIANTS:
        seed_rows = []
        for seed in SEEDS:
            path = root / "experiments" / f"component_v14_1_{variant}_seed{seed}.json"
            if not path.exists():
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))["summaries"][variant]
            samples = summary["sample_results"]
            b002 = next((sample for sample in samples if sample["benchmark_name"] == "track_b_002"), None)
            seed_rows.append(
                {
                    "seed": seed,
                    "track_a_score": summary["track_a_score"],
                    "track_b_score": summary["track_b_score"],
                    "final_score": summary["final_score"],
                    "invalid_count": sum(1 for sample in samples if not sample["valid"]),
                    "b002_success": b002["success_rate"] if b002 else None,
                    "b002_two_qubit_gate_count": b002["two_qubit_gate_count"] if b002 else None,
                    "b002_depth": b002["depth"] if b002 else None,
                    "b002_estimated_log_reliability": b002["estimated_log_reliability"] if b002 else None,
                    "p95_compile_time_sec": _p95(sample["compile_time_sec"] for sample in samples),
                    "max_compile_time_sec": max((sample["compile_time_sec"] for sample in samples), default=0.0),
                }
            )
        results[variant] = {
            "seeds": seed_rows,
            "mean_final_score": _mean(row["final_score"] for row in seed_rows),
            "mean_track_a_score": _mean(row["track_a_score"] for row in seed_rows),
            "mean_track_b_score": _mean(row["track_b_score"] for row in seed_rows),
            "mean_b002_success": _mean(row["b002_success"] for row in seed_rows if row["b002_success"] is not None),
            "max_invalid_count": max((row["invalid_count"] for row in seed_rows), default=0),
            "max_p95_compile_time_sec": max((row["p95_compile_time_sec"] for row in seed_rows), default=0.0),
        }
    return results


def _holdout_results(root: Path, *, prefix: str) -> dict[str, object]:
    results: dict[str, object] = {}
    for variant in VARIANTS:
        path = root / "experiments" / f"{prefix}_{variant}.json"
        if path.exists():
            results[variant] = json.loads(path.read_text(encoding="utf-8"))
    return results


def _gate_summary(public: dict[str, object], affine: dict[str, object]) -> dict[str, object]:
    baseline = public.get("v13_root", {})
    relabel = public.get("v13_plus_b_relabel_only", {})
    baseline_final = float(baseline.get("mean_final_score", 0.0))
    relabel_final = float(relabel.get("mean_final_score", 0.0))
    baseline_b002 = float(baseline.get("mean_b002_success", 0.0))
    relabel_b002 = float(relabel.get("mean_b002_success", 0.0))
    baseline_affine = float(affine.get("v13_root", {}).get("mean_track_b_success", 0.0))
    relabel_affine = float(affine.get("v13_plus_b_relabel_only", {}).get("mean_track_b_success", 0.0))
    relabel_gate = {
        "mean_final_delta": relabel_final - baseline_final,
        "mean_b002_delta": relabel_b002 - baseline_b002,
        "synthetic_affine_delta": relabel_affine - baseline_affine,
        "invalid_ok": int(relabel.get("max_invalid_count", 1)) == 0,
        "p95_compile_ok": float(relabel.get("max_p95_compile_time_sec", 999.0)) < 10.0,
    }
    relabel_gate["promote"] = (
        relabel_gate["mean_final_delta"] >= 0.004
        and relabel_gate["mean_b002_delta"] >= 0.015
        and relabel_gate["synthetic_affine_delta"] >= -1e-12
        and relabel_gate["invalid_ok"]
        and relabel_gate["p95_compile_ok"]
    )
    return {"relabel_only": relabel_gate}


def _decision_markdown(
    gates: dict[str, object],
    public: dict[str, object],
    affine: dict[str, object],
    small_tracka: dict[str, object],
) -> str:
    relabel = gates["relabel_only"]
    lines = [
        "# V14.1 Promote Decision",
        "",
        f"Decision: {'promote v13_plus_b_relabel_only to root' if relabel['promote'] else 'do not promote; keep opt-in'}",
        "",
        "## Relabel-Only Gate",
        "",
        f"- Mean Final delta: {relabel['mean_final_delta']:.6f}",
        f"- Mean B002 success delta: {relabel['mean_b002_delta']:.6f}",
        f"- Synthetic affine TrackB delta: {relabel['synthetic_affine_delta']:.6f}",
        f"- Invalid count OK: {relabel['invalid_ok']}",
        f"- p95 compile OK: {relabel['p95_compile_ok']}",
        "",
        "## Public Means",
        "",
    ]
    for variant in VARIANTS:
        row = public.get(variant, {})
        lines.append(
            f"- {variant}: Final={float(row.get('mean_final_score', 0.0)):.6f}, "
            f"TrackA={float(row.get('mean_track_a_score', 0.0)):.6f}, "
            f"TrackB={float(row.get('mean_track_b_score', 0.0)):.6f}, "
            f"B002={float(row.get('mean_b002_success', 0.0)):.6f}, "
            f"invalid={int(row.get('max_invalid_count', 0))}, "
            f"max_p95_compile={float(row.get('max_p95_compile_time_sec', 0.0)):.3f}s"
        )
    lines.extend(["", "## Holdouts", ""])
    for variant in VARIANTS:
        affine_row = affine.get(variant, {})
        small_row = small_tracka.get(variant, {})
        lines.append(
            f"- {variant}: affine_mean={float(affine_row.get('mean_track_b_success', 0.0)):.6f}, "
            f"small_tracka_mean={float(small_row.get('mean_track_a_score', 0.0)):.6f}"
        )
    lines.append("")
    return "\n".join(lines)


def _mean(values) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _p95(values) -> float:
    items = sorted(float(value) for value in values)
    if not items:
        return 0.0
    index = min(len(items) - 1, int(0.95 * (len(items) - 1)))
    return items[index]


if __name__ == "__main__":
    raise SystemExit(main())
