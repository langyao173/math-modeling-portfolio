from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.benchmarks.model import CompetitionBenchmark
from noise_adaptive_hackathon.evaluator.score_submission import score_submission
from noise_adaptive_hackathon.formats.backend import CompetitionBackend


@dataclass(frozen=True)
class CeilingResult:
    benchmark: str
    track: str
    rule_type: str
    width: int
    backend: str
    cap: float
    method: str
    selected_physical: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute optimistic readout-only public score ceilings."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-json", type=Path, default=Path("experiments/readout_ceiling_exact.json"))
    parser.add_argument("--out-md", type=Path, default=Path("experiments/readout_ceiling_exact.md"))
    parser.add_argument(
        "--current-summary",
        type=Path,
        default=None,
        help="Optional evaluation summary JSON to report current headroom.",
    )
    parser.add_argument(
        "--current-matrix",
        type=Path,
        default=None,
        help="Optional public 5-seed matrix JSON to report current headroom.",
    )
    parser.add_argument(
        "--current-variant",
        default=None,
        help="Variant name inside --current-matrix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    results: list[CeilingResult] = []
    for track in ("track_a", "track_b"):
        for benchmark_dir in sorted((root / "benchmarks" / track / "public").iterdir()):
            if not benchmark_dir.is_dir():
                continue
            benchmark = load_benchmark(benchmark_dir)
            backend = _load_backend(root, benchmark)
            results.append(_ceiling_for_benchmark(benchmark, backend))

    track_a = _mean(result.cap for result in results if result.track == "track_a")
    track_b = _mean(result.cap for result in results if result.track == "track_b")
    score = score_submission(track_a_score=track_a, track_b_score=track_b)
    current = _load_current_summary(args.current_summary)
    if current is None:
        current = _load_current_matrix(args.current_matrix, args.current_variant)
    payload = {
        "track_a_cap": track_a,
        "track_b_cap": track_b,
        "final_cap": score.final_score,
        "current_summary": current,
        "headroom": _headroom(current, track_a, track_b, score.final_score),
        "sample_headroom": _sample_headroom(current, results),
        "samples": [
            {
                "benchmark": result.benchmark,
                "track": result.track,
                "rule_type": result.rule_type,
                "width": result.width,
                "backend": result.backend,
                "cap": result.cap,
                "method": result.method,
                "selected_physical": list(result.selected_physical),
            }
            for result in results
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("track_a_cap", "track_b_cap", "final_cap", "headroom")}, indent=2))
    return 0


def _load_backend(root: Path, benchmark: CompetitionBenchmark) -> CompetitionBackend:
    path = (
        root
        / "backends"
        / benchmark.metadata["backend_size_class"]
        / f"{benchmark.metadata['backend_snapshot_id']}.json"
    )
    return CompetitionBackend.parse_raw(path.read_text(encoding="utf-8"))


def _ceiling_for_benchmark(benchmark: CompetitionBenchmark, backend: CompetitionBackend) -> CeilingResult:
    rule = benchmark.expected_output_rule
    width = len(benchmark.logical_qubits)
    readouts = {index: backend.readout(index) for index in range(backend.num_qubits)}
    if rule.rule_type == "bitstring":
        physical = _best_reliability_qubits(readouts, width)
        cap = math.prod(1.0 - readouts[index] for index in physical)
        method = "deterministic_bitstring_best_readout"
    elif rule.rule_type == "allowed_bitstrings":
        physical = _best_reliability_qubits(readouts, width)
        cap = _allowed_bitstrings_cap(rule.allowed_bitstrings, [readouts[index] for index in physical])
        method = "allowed_bitstrings_best_readout"
    else:
        cap, physical = _parity_checks_cap(rule.parity_checks, readouts, width)
        method = "parity_checks_readout_dominant_grouped_cap" if _is_pairwise_star_parity(rule.parity_checks) else "parity_checks_best_readout_pattern"
    return CeilingResult(
        benchmark=benchmark.name,
        track=benchmark.track,
        rule_type=rule.rule_type,
        width=width,
        backend=backend.snapshot_id,
        cap=cap,
        method=method,
        selected_physical=tuple(physical),
    )


def _best_reliability_qubits(readouts: dict[int, float], width: int) -> tuple[int, ...]:
    return tuple(sorted(readouts, key=lambda index: (readouts[index], index))[:width])


def _allowed_bitstrings_cap(allowed: tuple[str, ...], readouts: list[float]) -> float:
    if not allowed:
        return 0.0
    # Optimistic zero-gate cap: choose the allowed noiseless output that is most
    # likely to still decode into the allowed set after independent readout flips.
    best = 0.0
    allowed_set = set(allowed)
    for ideal in allowed:
        success = 0.0
        for observed in allowed_set:
            probability = 1.0
            for bit, target, error in zip(ideal, observed, readouts):
                probability *= (1.0 - error) if bit == target else error
            success += probability
        best = max(best, success)
    return best


def _parity_checks_cap(
    checks: tuple[tuple[tuple[int, ...], int], ...],
    readouts: dict[int, float],
    width: int,
) -> tuple[float, tuple[int, ...]]:
    if _is_pairwise_star_parity(checks):
        groups = _pairwise_star_groups(checks, width)
        return _pairwise_star_cap(groups, readouts)
    physical = _best_reliability_qubits(readouts, width)
    return _parity_rule_success(checks, [readouts[index] for index in physical]), physical


def _is_pairwise_star_parity(checks: tuple[tuple[tuple[int, ...], int], ...]) -> bool:
    if not checks:
        return False
    graph: dict[int, set[int]] = {}
    for indices, _ in checks:
        if len(indices) != 2:
            return False
        left, right = indices
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    return all(len(component) >= 2 and max(len(graph[node]) for node in component) == len(component) - 1 for component in _components(graph))


def _pairwise_star_groups(
    checks: tuple[tuple[tuple[int, ...], int], ...],
    width: int,
) -> list[tuple[int, ...]]:
    graph: dict[int, set[int]] = {index: set() for index in range(width)}
    for indices, _ in checks:
        left, right = indices
        graph[left].add(right)
        graph[right].add(left)
    return [tuple(sorted(component)) for component in _components(graph) if len(component) > 1]


def _components(graph: dict[int, set[int]]) -> list[set[int]]:
    remaining = set(graph)
    components: list[set[int]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = {start}
        while stack:
            node = stack.pop()
            for neighbor in graph[node]:
                if neighbor in component:
                    continue
                component.add(neighbor)
                remaining.discard(neighbor)
                stack.append(neighbor)
        components.append(component)
    return components


def _pairwise_star_cap(
    groups: list[tuple[int, ...]],
    readouts: dict[int, float],
) -> tuple[float, tuple[int, ...]]:
    # For a connected pairwise parity component, all readout error bits in that
    # component must be equal. Its success probability is therefore:
    # prod(1-p_i) + prod(p_i). For B002 public this gives an exact assignment
    # problem over six disjoint 4-sets.
    group_sizes = tuple(sorted((len(group) for group in groups), reverse=True))
    physical = _best_reliability_qubits(readouts, sum(group_sizes))
    # The no-readout-flip term dominates and is maximized by the lowest readout
    # qubits. The all-flip correction within each parity component is tiny but
    # positive; grouping the larger readout errors together maximizes that
    # correction for the selected physical set.
    ordered_errors = sorted((readouts[index], index) for index in physical)
    high_error_first = list(reversed(ordered_errors))
    grouped_errors: list[list[float]] = []
    selected_order: list[int] = []
    offset = 0
    for size in group_sizes:
        chunk = high_error_first[offset : offset + size]
        grouped_errors.append([error for error, _ in chunk])
        selected_order.extend(index for _, index in chunk)
        offset += size
    success = math.prod(_equal_error_group_success(errors) for errors in grouped_errors)
    return success, tuple(selected_order)


def _grouped_equal_error_success(group_sizes: Iterable[int], readouts: list[float]) -> float:
    offset = 0
    success = 1.0
    for size in group_sizes:
        success *= _equal_error_group_success(readouts[offset : offset + size])
        offset += size
    return success


def _equal_error_group_success(errors: list[float]) -> float:
    return math.prod(1.0 - error for error in errors) + math.prod(error for error in errors)


def _parity_rule_success(checks: tuple[tuple[tuple[int, ...], int], ...], readouts: list[float]) -> float:
    success = 0.0
    for error_bits in itertools.product((0, 1), repeat=len(readouts)):
        if all(sum(error_bits[index] for index in indices) % 2 == 0 for indices, _ in checks):
            probability = 1.0
            for bit, error in zip(error_bits, readouts):
                probability *= error if bit else 1.0 - error
            success += probability
    return success


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _load_current_summary(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample_success = {
        sample["benchmark_name"]: float(sample["success_rate"])
        for sample in payload.get("sample_results", [])
    }
    result = {
        "track_a_score": payload.get("track_a_score"),
        "track_b_score": payload.get("track_b_score"),
        "final_score": payload.get("final_score"),
    }
    if sample_success:
        result["sample_success"] = sample_success
    return result


def _load_current_matrix(path: Path | None, variant: str | None) -> dict | None:
    if path is None or not path.exists() or not variant:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload.get("variants", {}).get(variant)
    if not entry:
        return None
    seeds = entry.get("seeds", [])
    if not seeds:
        return None
    sample_totals: dict[str, list[float]] = {}
    for seed in seeds:
        for sample in seed.get("sample_results", []):
            sample_totals.setdefault(sample["benchmark_name"], []).append(float(sample["success_rate"]))
    return {
        "variant": variant,
        "track_a_score": _mean(float(seed["track_a_score"]) for seed in seeds),
        "track_b_score": _mean(float(seed["track_b_score"]) for seed in seeds),
        "final_score": _mean(float(seed["final_score"]) for seed in seeds),
        "invalid_count": sum(int(seed.get("invalid_count", 0)) for seed in seeds),
        "seed_count": len(seeds),
        "sample_success": {
            name: _mean(values)
            for name, values in sorted(sample_totals.items())
        },
    }


def _headroom(current: dict | None, track_a_cap: float, track_b_cap: float, final_cap: float) -> dict | None:
    if not current:
        return None
    return {
        "track_a": track_a_cap - float(current["track_a_score"]),
        "track_b": track_b_cap - float(current["track_b_score"]),
        "final": final_cap - float(current["final_score"]),
    }


def _sample_headroom(current: dict | None, results: list[CeilingResult]) -> list[dict]:
    if not current or "sample_success" not in current:
        return []
    success = current["sample_success"]
    rows = []
    for result in results:
        current_value = success.get(result.benchmark)
        if current_value is None:
            continue
        rows.append(
            {
                "benchmark": result.benchmark,
                "track": result.track,
                "cap": result.cap,
                "current": current_value,
                "headroom": result.cap - current_value,
            }
        )
    return sorted(rows, key=lambda row: (row["headroom"], row["benchmark"]), reverse=True)


def _markdown(payload: dict) -> str:
    lines = [
        "# Exact Readout Ceiling",
        "",
        "This diagnostic computes the optimistic score if all gates were perfect and only mandatory final readout noise remained.",
        "",
        f"- Track A cap: `{payload['track_a_cap']:.12f}`",
        f"- Track B cap: `{payload['track_b_cap']:.12f}`",
        f"- Final cap: `{payload['final_cap']:.12f}`",
        "",
    ]
    if payload["current_summary"]:
        lines.extend(
            [
                "## Current Headroom",
                "",
                f"- Track A: `{payload['headroom']['track_a']:.12f}`",
                f"- Track B: `{payload['headroom']['track_b']:.12f}`",
                f"- Final: `{payload['headroom']['final']:.12f}`",
                "",
            ]
        )
    if payload["sample_headroom"]:
        lines.extend(
            [
                "## Per-Sample Headroom",
                "",
                "| sample | track | current | cap | headroom |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in payload["sample_headroom"]:
            lines.append(
                f"| `{row['benchmark']}` | {row['track']} | {row['current']:.12f} | "
                f"{row['cap']:.12f} | {row['headroom']:.12f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Samples",
            "",
            "| sample | track | rule | width | cap | method |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for sample in payload["samples"]:
        lines.append(
            f"| `{sample['benchmark']}` | {sample['track']} | {sample['rule_type']} | {sample['width']} | "
            f"{sample['cap']:.12f} | {sample['method']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
