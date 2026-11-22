from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import ROOT, load_json, write_json

from noise_adaptive_hackathon.evaluator.suite import SuiteSpec, evaluate_suite, suite_result_payload


DEFAULT_SEEDS = (7, 11, 17, 23, 31)
DEFAULT_VARIANTS = (
    "v13_root",
    "v13_plus_b_relabel_only",
    "v13_plus_tracka_exact_router_only",
    "v15_tracka_semantic",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the private-leaderboard emulator matrix.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--assets", type=Path, default=ROOT / "experiments" / "private_emulator_v1")
    parser.add_argument("--suite", action="append", default=[])
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--baseline", default="v13_root")
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--shots-a", type=int, default=256)
    parser.add_argument("--shots-b", type=int, default=1024)
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--build-scoreboard", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.assets / "suite_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} is missing; run scripts/gen_private_emulator_assets.py first"
        )
    manifest = load_json(manifest_path)
    suite_filter = set(args.suite)
    variants = tuple(args.variant or DEFAULT_VARIANTS)
    if args.baseline not in variants:
        variants = (args.baseline, *variants)
    seeds = tuple(args.seed or DEFAULT_SEEDS)

    for suite_payload in manifest["suites"]:
        if suite_filter and suite_payload["name"] not in suite_filter:
            continue
        spec = SuiteSpec(
            name=suite_payload["name"],
            root=Path(suite_payload["root"]),
            split=suite_payload.get("split", "public"),
            tracks=tuple(suite_payload.get("tracks", ("track_a", "track_b"))),
            weight=float(suite_payload.get("weight", 1.0)),
        )
        matrix_path = args.assets / suite_payload["matrix_output"]
        matrix = _load_resume_matrix(
            matrix_path,
            suite=spec.name,
            baseline=args.baseline,
            seeds=seeds,
            shots_a=args.shots_a,
            shots_b=args.shots_b,
            timeout_sec=args.timeout_sec,
        )
        for variant in variants:
            if variant in matrix["variants"]:
                continue
            result = evaluate_suite(
                spec,
                _variant_command(args.root, variant),
                shots_a=args.shots_a,
                shots_b=args.shots_b,
                seeds=list(seeds),
                timeout_sec=args.timeout_sec,
            )
            matrix["variants"][variant] = suite_result_payload(result, variant=variant)
            write_json(matrix_path, matrix)

    if args.build_scoreboard:
        from build_final_choice_scoreboard import main as build_scoreboard_main

        return build_scoreboard_main(["--assets", str(args.assets), "--baseline", args.baseline])
    return 0


def _variant_command(root: Path, variant: str) -> list[str]:
    if variant == "root":
        return [sys.executable, str(root / "compiler.py")]
    return [sys.executable, str(root / "scripts" / "variant_compiler.py"), "--variant", variant]


def _load_resume_matrix(
    path: Path,
    *,
    suite: str,
    baseline: str,
    seeds: tuple[int, ...],
    shots_a: int,
    shots_b: int,
    timeout_sec: int,
) -> dict[str, object]:
    if path.exists():
        try:
            existing = load_json(path)
        except Exception:
            existing = {}
        if (
            existing.get("schema_version") == "private_emulator.matrix.v1"
            and existing.get("suite") == suite
            and existing.get("baseline") == baseline
            and tuple(existing.get("seeds", [])) == tuple(seeds)
            and int(existing.get("shots_a", -1)) == int(shots_a)
            and int(existing.get("shots_b", -1)) == int(shots_b)
            and int(existing.get("timeout_sec", -1)) == int(timeout_sec)
            and isinstance(existing.get("variants"), dict)
        ):
            return existing
    return {
        "schema_version": "private_emulator.matrix.v1",
        "suite": suite,
        "baseline": baseline,
        "seeds": list(seeds),
        "shots_a": shots_a,
        "shots_b": shots_b,
        "timeout_sec": timeout_sec,
        "variants": {},
    }


if __name__ == "__main__":
    raise SystemExit(main())
