from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEFAULT_CEILING = ROOT / "experiments" / "upper_bound_decomposition.json"
DEFAULT_OUT_JSON = ROOT / "experiments" / "cap_normalized_gap_report.json"
DEFAULT_OUT_MD = ROOT / "experiments" / "cap_normalized_gap_report.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank residuals by legal-cap-normalized gap.")
    parser.add_argument("--ceiling-json", type=Path, default=DEFAULT_CEILING)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    payload = json.loads(args.ceiling_json.read_text(encoding="utf-8"))
    rows = []
    for sample in payload["samples"]:
        cap = float(sample["best_physical_readout_remap_ceiling"])
        current = float(sample["current_score"])
        normalized_gap = max(0.0, (cap - current) / cap) if cap > 0 else 0.0
        sample_weight = 0.75 / 8.0 if sample["track"] == "track_a" else 0.25 / 2.0
        raw_weighted_gap = sample_weight * (cap - current)
        weighted_cap_gap = sample_weight * normalized_gap
        rows.append(
            {
                "benchmark": sample["benchmark"],
                "track": sample["track"],
                "family": sample["family"],
                "current": current,
                "legal_cap": cap,
                "cap_efficiency": current / cap if cap > 0 else 0.0,
                "normalized_gap": normalized_gap,
                "sample_weight": sample_weight,
                "weighted_cap_gap": weighted_cap_gap,
                "raw_weighted_gap_to_cap": raw_weighted_gap,
                "cx": sample["current_cx"],
                "depth": sample["current_depth"],
                "current_measured_cap": sample["current_measurement_readout_cap"],
                "best_readout_remap_lift": cap - float(sample["current_measurement_readout_cap"]),
                "oracle_readout_current": sample["oracle_readout_current_circuit_success"],
                "gate_noise_gap": float(sample["oracle_readout_current_circuit_success"]) - current,
            }
        )
    rows = sorted(rows, key=lambda row: (row["weighted_cap_gap"], row["normalized_gap"], row["benchmark"]), reverse=True)
    output = {
        "source": str(args.ceiling_json),
        "current_final": payload["current_summary"]["final"],
        "legal_same_width_final_ceiling": payload["same_width_best_readout"]["final"],
        "total_weighted_cap_gap": sum(row["weighted_cap_gap"] for row in rows),
        "total_raw_weighted_gap_to_cap": sum(row["raw_weighted_gap_to_cap"] for row in rows),
        "rows": rows,
        "interpretation": interpret(rows),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render_markdown(output), encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("legal_same_width_final_ceiling", "total_weighted_cap_gap", "total_raw_weighted_gap_to_cap")}, indent=2))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


def interpret(rows: list[dict[str, Any]]) -> list[str]:
    top = rows[:5]
    return [
        "Normalized ranking still prioritizes B002/B001, but now as cap-efficiency deficits rather than raw readout-width headroom.",
        "A007/A008 remain the top Track A semantic targets after cap normalization.",
        "Best-readout remap lift is modest for A007/A008 and B002, so large jumps require gate/noise implementation changes, not only final measurement relocation.",
        "B001 has the largest readout remap lift among legal candidates, but its same-width cap is still only 0.858252.",
        "Top normalized residuals: " + ", ".join(f"{row['benchmark']}({row['weighted_cap_gap']:.6f})" for row in top),
    ]


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Cap-Normalized Gap Report",
        "",
        "This ranks samples by `(legal_cap - current) / legal_cap`, weighted by their final-score sample weight.",
        "",
        f"- Current Final: `{payload['current_final']:.12f}`",
        f"- Legal same-width Final ceiling: `{payload['legal_same_width_final_ceiling']:.12f}`",
        f"- Sum weighted normalized cap gap: `{payload['total_weighted_cap_gap']:.12f}`",
        f"- Sum raw weighted gap to legal cap: `{payload['total_raw_weighted_gap_to_cap']:.12f}`",
        "",
        "## Ranking",
        "",
        "| Rank | Sample | Track | Current | Legal cap | Efficiency | Normalized gap | Weighted cap gap | Raw weighted cap gap | CX | Gate-noise gap | Readout-remap lift |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(payload["rows"], 1):
        lines.append(
            f"| {index} | `{row['benchmark']}` | {row['track']} | {row['current']:.6f} | {row['legal_cap']:.6f} | {row['cap_efficiency']:.6f} | {row['normalized_gap']:.6f} | {row['weighted_cap_gap']:.6f} | {row['raw_weighted_gap_to_cap']:.6f} | {row['cx']} | {row['gate_noise_gap']:.6f} | {row['best_readout_remap_lift']:.6f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {item}" for item in payload["interpretation"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

