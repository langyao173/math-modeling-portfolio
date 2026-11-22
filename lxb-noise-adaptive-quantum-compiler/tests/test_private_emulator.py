from __future__ import annotations

from dataclasses import asdict
import random

from noise_adaptive_hackathon.evaluator.backend_jitter import JITTER_VARIANTS, jitter_backend_payload
from noise_adaptive_hackathon.evaluator.paired_stats import paired_delta_summary
from noise_adaptive_hackathon.evaluator.submission import SubmissionSampleResult
from noise_adaptive_hackathon.evaluator.suite import SeedResult, SuiteResult, suite_result_payload


def test_backend_jitter_variants_are_deterministic_and_valid_shape() -> None:
    payload = {
        "schema_version": "2.0",
        "snapshot_id": "demo",
        "size_class": "medium",
        "num_qubits": 3,
        "basis_gates": ["x", "h", "cx", "measure"],
        "coupling_map": [(0, 1), (1, 2)],
        "two_qubit_error": {"0-1": 0.01, "1-2": 0.02},
        "readout_error": {"0": 0.01, "1": 0.02, "2": 0.03},
        "single_qubit_error": 0.001,
    }

    first = [jitter_backend_payload(payload, variant, rng=random.Random(7)) for variant in JITTER_VARIANTS]
    second = [jitter_backend_payload(payload, variant, rng=random.Random(7)) for variant in JITTER_VARIANTS]

    assert first == second
    assert first[0]["two_qubit_error"] == payload["two_qubit_error"]
    assert first[0]["readout_error"] == payload["readout_error"]
    assert first[-1]["two_qubit_error"]["0-1"] == payload["two_qubit_error"]["1-2"]


def test_suite_payload_reports_invalid_and_compile_quantiles() -> None:
    samples = [
        SubmissionSampleResult("a", "track_a", 0.5, True, 10, 5, 3, 2, -0.1, 0.2, []),
        SubmissionSampleResult("b", "track_b", 0.0, False, 10, 0, 0, 0, 0.0, 1.0, ["bad"]),
    ]
    result = SuiteResult("demo", "variant", [SeedResult("demo", 7, 0.5, 0.0, 0.375, 1, samples)])

    payload = suite_result_payload(result, variant="variant")

    assert payload["max_invalid_count"] == 1
    assert payload["p95_compile_time_sec"] == 0.2
    assert payload["max_compile_time_sec"] == 1.0


def test_paired_delta_summary_pairs_by_seed_and_benchmark() -> None:
    baseline = {
        "seeds": [
            {
                "seed": 7,
                "final_score": 0.5,
                "sample_results": [
                    {"benchmark_name": "a", "success_rate": 0.4, "valid": True, "compile_time_sec": 0.1},
                    {"benchmark_name": "b", "success_rate": 0.6, "valid": True, "compile_time_sec": 0.1},
                ],
            }
        ]
    }
    variant = {
        "seeds": [
            {
                "seed": 7,
                "final_score": 0.55,
                "sample_results": [
                    {"benchmark_name": "a", "success_rate": 0.5, "valid": True, "compile_time_sec": 0.2},
                    {"benchmark_name": "b", "success_rate": 0.5, "valid": True, "compile_time_sec": 0.3},
                ],
            }
        ]
    }

    summary = paired_delta_summary(
        baseline,
        variant,
        baseline_name="base",
        variant_name="variant",
        suite="demo",
    )

    assert summary.paired_count == 2
    assert summary.mean_delta == 0.0
    assert summary.negative_delta_count == 1
    assert summary.worst_seed_delta > 0.0
