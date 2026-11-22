from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from common import ROOT, load_json

from noise_adaptive_hackathon.evaluator.paired_stats import paired_delta_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build private-emulator paired reports and promotion scoreboard.")
    parser.add_argument("--assets", type=Path, default=ROOT / "experiments" / "private_emulator_v1")
    parser.add_argument("--baseline", default="v13_root")
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-report", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_json(args.assets / "suite_manifest.json")
    out_csv = args.out_csv or args.assets / "final_choice_scoreboard.csv"
    out_report = args.out_report or args.assets / "paired_delta_report.md"
    summaries = _load_summaries(args.assets, manifest, args.baseline)
    rows = _scoreboard_rows(summaries, manifest)
    _write_csv(out_csv, rows)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(_render_report(rows, summaries, manifest), encoding="utf-8")
    return 0


def _load_summaries(assets: Path, manifest: dict[str, object], baseline: str) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for suite_payload in manifest["suites"]:
        matrix_path = assets / suite_payload["matrix_output"]
        if not matrix_path.exists():
            continue
        matrix = load_json(matrix_path)
        baseline_payload = matrix["variants"].get(baseline)
        if baseline_payload is None:
            continue
        for variant, variant_payload in sorted(matrix["variants"].items()):
            if variant == baseline:
                continue
            summary = paired_delta_summary(
                baseline_payload,
                variant_payload,
                baseline_name=baseline,
                variant_name=variant,
                suite=suite_payload["name"],
            )
            item = asdict(summary)
            item["suite_weight"] = float(suite_payload.get("weight", 1.0))
            item["near_miss_false_accept"] = _near_miss_false_accepts(
                suite_payload=suite_payload,
                variant_payload=variant_payload,
                assets=assets,
            )
            summaries.append(item)
    return summaries


def _scoreboard_rows(
    summaries: list[dict[str, object]],
    manifest: dict[str, object],
) -> list[dict[str, object]]:
    by_variant: dict[str, list[dict[str, object]]] = {}
    for summary in summaries:
        by_variant.setdefault(str(summary["variant"]), []).append(summary)
    rule = manifest.get("promotion_rule", {})
    rows: list[dict[str, object]] = []
    for variant, items in sorted(by_variant.items()):
        weight_total = sum(float(item["suite_weight"]) for item in items) or 1.0
        weighted_delta = sum(float(item["suite_weight"]) * float(item["mean_delta"]) for item in items) / weight_total
        weighted_negative_fraction = (
            sum(float(item["suite_weight"]) * float(item["negative_delta_fraction"]) for item in items) / weight_total
        )
        invalid_count = sum(int(item["invalid_count"]) for item in items)
        near_miss = sum(int(item["near_miss_false_accept"]) for item in items)
        p95_compile = max((float(item["p95_compile_time_sec"]) for item in items), default=0.0)
        max_compile = max((float(item["max_compile_time_sec"]) for item in items), default=0.0)
        worst_seed_delta = min((float(item["worst_seed_delta"]) for item in items), default=0.0)
        promoted = (
            invalid_count <= int(rule.get("invalid_count", 0))
            and p95_compile < float(rule.get("p95_compile_lt", 12.0))
            and weighted_delta >= float(rule.get("paired_mean_delta_gte", 0.003))
            and weighted_negative_fraction <= float(rule.get("negative_delta_fraction_lte", 0.20))
            and worst_seed_delta >= float(rule.get("worst_seed_delta_gte", -0.004))
            and near_miss <= int(rule.get("near_miss_false_accept", 0))
        )
        rows.append(
            {
                "variant": variant,
                "weighted_mean_delta": weighted_delta,
                "weighted_negative_fraction": weighted_negative_fraction,
                "worst_seed_delta": worst_seed_delta,
                "invalid_count": invalid_count,
                "near_miss_false_accept": near_miss,
                "p95_compile_time_sec": p95_compile,
                "max_compile_time_sec": max_compile,
                "promotion_gate": "pass" if promoted else "fail",
            }
        )
    rows.sort(key=lambda row: (row["promotion_gate"] == "pass", row["weighted_mean_delta"]), reverse=True)
    return rows


def _near_miss_false_accepts(
    *,
    suite_payload: dict[str, object],
    variant_payload: dict[str, object],
    assets: Path,
) -> int:
    if suite_payload.get("name") != "trackb_affine":
        return 0
    manifest_path = Path(suite_payload["root"]) / "holdout_manifest.json"
    if not manifest_path.exists():
        manifest_path = assets / "trackb_affine" / "holdout_manifest.json"
    if not manifest_path.exists():
        return 0
    negative = {
        str(item["name"])
        for item in load_json(manifest_path)
        if bool(item.get("negative_case", False))
    }
    count = 0
    for seed in variant_payload.get("seeds", []):
        for sample in seed.get("sample_results", []):
            if sample.get("benchmark_name") in negative and not sample.get("valid", False):
                count += 1
    return count


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "weighted_mean_delta",
        "weighted_negative_fraction",
        "worst_seed_delta",
        "invalid_count",
        "near_miss_false_accept",
        "p95_compile_time_sec",
        "max_compile_time_sec",
        "promotion_gate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(
    rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    manifest: dict[str, object],
) -> str:
    lines = [
        "# Private Emulator Paired Delta Report",
        "",
        "Promotion rule:",
        f"- `{manifest.get('promotion_rule', {})}`",
        "",
        "## Final Choice Scoreboard",
        "",
        "| Variant | Weighted delta | Negative frac | Worst seed | Invalid | Near-miss | p95 compile | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {weighted_mean_delta:.6f} | {weighted_negative_fraction:.3f} | "
            "{worst_seed_delta:.6f} | {invalid_count} | {near_miss_false_accept} | "
            "{p95_compile_time_sec:.3f} | {promotion_gate} |".format(**row)
        )
    lines.extend(["", "## Suite Details", ""])
    for item in sorted(summaries, key=lambda value: (value["variant"], value["suite"])):
        lines.append(
            "- {variant} / {suite}: mean_delta={mean_delta:.6f}, std={std_delta:.6f}, "
            "worst_seed={worst_seed_delta:.6f}, negatives={negative_delta_count}/{paired_count}, "
            "invalid={invalid_count}, p95={p95_compile_time_sec:.3f}s".format(**item)
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
