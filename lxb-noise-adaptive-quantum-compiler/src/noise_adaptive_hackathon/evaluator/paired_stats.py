from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class PairedDeltaSummary:
    baseline: str
    variant: str
    suite: str
    paired_count: int
    mean_delta: float
    std_delta: float
    worst_seed_delta: float
    negative_delta_count: int
    negative_delta_fraction: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    invalid_count: int
    p95_compile_time_sec: float
    max_compile_time_sec: float


def paired_delta_summary(
    baseline_payload: dict[str, object],
    variant_payload: dict[str, object],
    *,
    baseline_name: str,
    variant_name: str,
    suite: str,
    bootstrap_seed: int = 7,
) -> PairedDeltaSummary:
    baseline_samples = _sample_index(baseline_payload)
    variant_samples = _sample_index(variant_payload)
    keys = sorted(set(baseline_samples) & set(variant_samples))
    deltas = [variant_samples[key]["success_rate"] - baseline_samples[key]["success_rate"] for key in keys]
    seed_deltas = _seed_deltas(baseline_payload, variant_payload)
    compiles = [
        float(sample["compile_time_sec"])
        for seed in variant_payload.get("seeds", [])
        for sample in seed.get("sample_results", [])
    ]
    invalid_count = sum(
        1
        for seed in variant_payload.get("seeds", [])
        for sample in seed.get("sample_results", [])
        if not sample.get("valid", False)
    )
    ci_low, ci_high = _bootstrap_ci(deltas, seed=bootstrap_seed)
    return PairedDeltaSummary(
        baseline=baseline_name,
        variant=variant_name,
        suite=suite,
        paired_count=len(deltas),
        mean_delta=_mean(deltas),
        std_delta=_std(deltas),
        worst_seed_delta=min(seed_deltas) if seed_deltas else 0.0,
        negative_delta_count=sum(1 for delta in deltas if delta < -1e-12),
        negative_delta_fraction=(sum(1 for delta in deltas if delta < -1e-12) / len(deltas)) if deltas else 0.0,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        invalid_count=invalid_count,
        p95_compile_time_sec=_p95(compiles),
        max_compile_time_sec=max(compiles, default=0.0),
    )


def _sample_index(payload: dict[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    indexed: dict[tuple[str, str], dict[str, object]] = {}
    for seed in payload.get("seeds", []):
        seed_value = str(seed.get("seed"))
        for sample in seed.get("sample_results", []):
            indexed[(seed_value, str(sample.get("benchmark_name")))] = sample
    return indexed


def _seed_deltas(baseline_payload: dict[str, object], variant_payload: dict[str, object]) -> list[float]:
    baseline = {int(seed["seed"]): float(seed.get("final_score", 0.0)) for seed in baseline_payload.get("seeds", [])}
    variant = {int(seed["seed"]): float(seed.get("final_score", 0.0)) for seed in variant_payload.get("seeds", [])}
    return [variant[seed] - baseline[seed] for seed in sorted(set(baseline) & set(variant))]


def _bootstrap_ci(values: list[float], *, seed: int, iterations: int = 1000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(_mean(sample))
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def _mean(values) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _std(values) -> float:
    items = [float(value) for value in values]
    if len(items) < 2:
        return 0.0
    mean = _mean(items)
    return (sum((value - mean) ** 2 for value in items) / (len(items) - 1)) ** 0.5


def _p95(values) -> float:
    items = sorted(float(value) for value in values)
    if not items:
        return 0.0
    return items[min(len(items) - 1, int(0.95 * (len(items) - 1)))]
