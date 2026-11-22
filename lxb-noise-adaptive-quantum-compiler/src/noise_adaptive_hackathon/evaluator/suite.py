from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from noise_adaptive_hackathon.evaluator.score_submission import score_submission
from noise_adaptive_hackathon.evaluator.submission import SubmissionSampleResult, evaluate_single_benchmark


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    root: Path
    split: str = "public"
    tracks: tuple[str, ...] = ("track_a", "track_b")
    weight: float = 1.0


@dataclass(frozen=True)
class SeedResult:
    suite: str
    seed: int
    track_a_score: float
    track_b_score: float
    final_score: float
    invalid_count: int
    sample_results: list[SubmissionSampleResult]


@dataclass(frozen=True)
class SuiteResult:
    suite: str
    variant: str
    seeds: list[SeedResult]


def evaluate_suite(
    spec: SuiteSpec,
    command: list[str],
    *,
    shots_a: int,
    shots_b: int,
    seeds: list[int],
    timeout_sec: int,
) -> SuiteResult:
    seed_results = [
        evaluate_suite_seed(
            spec,
            command,
            shots_a=shots_a,
            shots_b=shots_b,
            seed=seed,
            timeout_sec=timeout_sec,
        )
        for seed in seeds
    ]
    return SuiteResult(suite=spec.name, variant="", seeds=seed_results)


def evaluate_suite_seed(
    spec: SuiteSpec,
    command: list[str],
    *,
    shots_a: int,
    shots_b: int,
    seed: int,
    timeout_sec: int,
) -> SeedResult:
    results: list[SubmissionSampleResult] = []
    for track in spec.tracks:
        track_dir = spec.root / "benchmarks" / track / spec.split
        if not track_dir.exists():
            continue
        shots = shots_a if track == "track_a" else shots_b
        offset = 0 if track == "track_a" else 10_000
        for index, benchmark_dir in enumerate(sorted(path for path in track_dir.iterdir() if path.is_dir())):
            results.append(
                evaluate_single_benchmark(
                    root=spec.root,
                    benchmark_dir=benchmark_dir,
                    command=command,
                    shots=shots,
                    seed=seed + offset + index,
                    timeout_sec=timeout_sec,
                )
            )
    track_a = [result for result in results if result.track == "track_a"]
    track_b = [result for result in results if result.track == "track_b"]
    track_a_score = _mean(result.success_rate for result in track_a)
    track_b_score = _mean(result.success_rate for result in track_b)
    if track_a and track_b:
        final_score = score_submission(track_a_score=track_a_score, track_b_score=track_b_score).final_score
    elif track_a:
        final_score = track_a_score
    elif track_b:
        final_score = track_b_score
    else:
        final_score = 0.0
    return SeedResult(
        suite=spec.name,
        seed=seed,
        track_a_score=track_a_score,
        track_b_score=track_b_score,
        final_score=final_score,
        invalid_count=sum(1 for result in results if not result.valid),
        sample_results=results,
    )


def suite_result_payload(result: SuiteResult, *, variant: str) -> dict[str, object]:
    return {
        "suite": result.suite,
        "variant": variant,
        "seeds": [
            {
                **{key: value for key, value in asdict(seed).items() if key != "sample_results"},
                "sample_results": [asdict(sample) for sample in seed.sample_results],
            }
            for seed in result.seeds
        ],
        "mean_final_score": _mean(seed.final_score for seed in result.seeds),
        "mean_track_a_score": _mean(seed.track_a_score for seed in result.seeds),
        "mean_track_b_score": _mean(seed.track_b_score for seed in result.seeds),
        "max_invalid_count": max((seed.invalid_count for seed in result.seeds), default=0),
        "p95_compile_time_sec": _p95(
            sample.compile_time_sec for seed in result.seeds for sample in seed.sample_results
        ),
        "max_compile_time_sec": max(
            (sample.compile_time_sec for seed in result.seeds for sample in seed.sample_results),
            default=0.0,
        ),
    }


def _mean(values) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _p95(values) -> float:
    items = sorted(float(value) for value in values)
    if not items:
        return 0.0
    return items[min(len(items) - 1, int(0.95 * (len(items) - 1)))]
