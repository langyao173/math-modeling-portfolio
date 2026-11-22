from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import statistics
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "manifest.jsonl"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_payload(summary: Any) -> dict[str, Any]:
    if hasattr(summary, "__dataclass_fields__"):
        return asdict(summary)
    return dict(summary)


def append_manifest(
    *,
    summary: dict[str, Any],
    variant: str,
    seed: int,
    config: dict[str, Any],
    decision: str,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    samples = summary.get("sample_results", [])
    invalid_count = sum(1 for sample in samples if not sample.get("valid", False))
    entry = {
        "timestamp_unix": int(time.time()),
        "variant": variant,
        "seed": seed,
        "config": config,
        "track_a_score": summary.get("track_a_score", 0.0),
        "track_b_score": summary.get("track_b_score", 0.0),
        "final_score": summary.get("final_score", 0.0),
        "invalid_count": invalid_count,
        "mean_depth": _mean(sample.get("depth", 0) for sample in samples),
        "mean_two_qubit_gate_count": _mean(sample.get("two_qubit_gate_count", 0) for sample in samples),
        "mean_estimated_log_reliability": _mean(sample.get("estimated_log_reliability", 0.0) for sample in samples),
        "mean_compile_time_sec": _mean(sample.get("compile_time_sec", 0.0) for sample in samples),
        "decision": decision,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def best_manifest_entry(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries:
        return None
    return max(entries, key=lambda item: (item.get("final_score", 0.0), -item.get("invalid_count", 0)))


def _mean(values) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return float(statistics.fmean(items))
