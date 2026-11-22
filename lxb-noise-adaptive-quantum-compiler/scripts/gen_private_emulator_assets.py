from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cirq

from common import ROOT, write_json
from noise_adaptive_hackathon.benchmarks.io import load_benchmark, write_benchmark
from noise_adaptive_hackathon.benchmarks.model import CompetitionBenchmark, ExpectedOutputRule
from noise_adaptive_hackathon.evaluator.backend_jitter import JITTER_VARIANTS, write_jittered_backends
from noise_adaptive_hackathon.formats.backend import CompetitionBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate private-leaderboard emulator assets.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "private_emulator_v1")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    tracka_root = args.out / "tracka_structural"
    trackb_root = args.out / "trackb_affine"
    jitter_root = args.out / "backend_jitter"
    _reset_roots(tracka_root, trackb_root, jitter_root)

    _copy_backend_tree(args.root, tracka_root)
    _copy_backend_tree(args.root, trackb_root)
    jitter_records = write_jittered_backends(args.root, jitter_root, seed=args.seed)

    tracka_manifest = _write_tracka_structural(tracka_root, rng)
    trackb_manifest = _write_trackb_affine(trackb_root, rng)
    jitter_manifest = _write_jitter_public_replay(args.root, jitter_root, jitter_records)

    suite_manifest = {
        "schema_version": "private_emulator.v1",
        "seed": args.seed,
        "suites": [
            {
                "name": "public_replay",
                "root": str(args.root),
                "split": "public",
                "tracks": ["track_a", "track_b"],
                "weight": 0.25,
                "matrix_output": "public_5seed_matrix.json",
            },
            {
                "name": "tracka_structural",
                "root": str(tracka_root),
                "split": "holdout",
                "tracks": ["track_a"],
                "weight": 0.35,
                "matrix_output": "synthetic_tracka_structural_matrix.json",
            },
            {
                "name": "trackb_affine",
                "root": str(trackb_root),
                "split": "holdout",
                "tracks": ["track_b"],
                "weight": 0.25,
                "matrix_output": "synthetic_trackb_affine_matrix.json",
            },
            {
                "name": "backend_jitter",
                "root": str(jitter_root),
                "split": "holdout",
                "tracks": ["track_a", "track_b"],
                "weight": 0.15,
                "matrix_output": "backend_jitter_matrix.json",
            },
        ],
        "promotion_rule": {
            "invalid_count": 0,
            "p95_compile_lt": 12.0,
            "paired_mean_delta_gte": 0.003,
            "negative_delta_fraction_lte": 0.20,
            "worst_seed_delta_gte": -0.004,
            "near_miss_false_accept": 0,
        },
        "manifests": {
            "tracka_structural": tracka_manifest,
            "trackb_affine": trackb_manifest,
            "backend_jitter": jitter_manifest,
        },
    }
    write_json(args.out / "suite_manifest.json", suite_manifest)
    return 0


def _reset_roots(*roots: Path) -> None:
    for root in roots:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)


def _copy_backend_tree(source_root: Path, target_root: Path) -> None:
    for source in (source_root / "backends").glob("*/*.json"):
        target = target_root / "backends" / source.parent.name / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _write_tracka_structural(root: Path, rng: random.Random) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    backend_refs = _backend_refs(root, "medium")
    index = 1
    for family, circuit in _tracka_cases(rng):
        backend_size, snapshot = rng.choice(backend_refs)
        name = f"tracka_struct_{index:03d}"
        benchmark = CompetitionBenchmark(
            track="track_a",
            name=name,
            family=family,
            circuit=circuit,
            logical_qubits=tuple(sorted(circuit.all_qubits(), key=lambda qubit: qubit.x)),
            expected_output_rule=_deterministic_rule(circuit, rng),
            metadata={"backend_size_class": backend_size, "backend_snapshot_id": snapshot, "synthetic_family": family},
        )
        write_benchmark(benchmark, root / "benchmarks" / "track_a" / "holdout" / name)
        records.append(_record(benchmark))
        index += 1
    write_json(root / "holdout_manifest.json", records)
    return records


def _tracka_cases(rng: random.Random):
    for width in (6, 7):
        q = cirq.LineQubit.range(width)
        circuit = cirq.Circuit()
        for target in range(2, width):
            circuit.append(cirq.CNOT(q[0], q[target]))
            circuit.append(cirq.T(q[target]) if target % 2 == 0 else cirq.T(q[target]) ** -1)
            circuit.append(cirq.CNOT(q[1], q[target]))
        circuit.append(cirq.measure(*q, key="z"))
        yield "cnot_ladder_phase", circuit

    for width in (6, 7):
        q = cirq.LineQubit.range(width)
        circuit = cirq.Circuit()
        for target in range(2, min(width, 6)):
            circuit.append(_toffoli_6cx(q[0], q[1], q[target]))
            if target + 1 < width:
                circuit.append(cirq.CNOT(q[target], q[target + 1]))
        circuit.append(cirq.measure(*q, key="z"))
        yield "toffoli_carry_chain", circuit

    for width in (6, 7):
        q = cirq.LineQubit.range(width)
        circuit = cirq.Circuit()
        for round_index in range(4):
            for left in range(width - 1):
                if (left + round_index) % 2 == 0:
                    circuit.append(cirq.CNOT(q[left], q[left + 1]))
            circuit.append(cirq.T(q[(2 * round_index + 1) % width]))
            circuit.append(cirq.CNOT(q[round_index % width], q[-1]))
        circuit.append(cirq.measure(*q, key="z"))
        yield "structured_random_phase", circuit

    for width in (6, 7):
        q = cirq.LineQubit.range(width)
        circuit = cirq.Circuit()
        for _ in range(18 if width == 7 else 14):
            left, right = rng.sample(range(width), 2)
            circuit.append(cirq.CNOT(q[left], q[right]))
            if rng.random() < 0.35:
                phase = cirq.T(q[right]) if rng.random() < 0.5 else cirq.T(q[right]) ** -1
                circuit.append(phase)
        circuit.append(cirq.measure(*q, key="z"))
        yield "routing_pressure_random", circuit


def _write_trackb_affine(root: Path, rng: random.Random) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    backend_refs = _backend_refs(root, "large")
    index = 1
    for circuit, family, negative in _trackb_cases(rng):
        backend_size, snapshot = rng.choice(backend_refs)
        name = f"affine_struct_{index:03d}"
        benchmark = CompetitionBenchmark(
            track="track_b",
            name=name,
            family=family,
            circuit=circuit,
            logical_qubits=tuple(sorted(circuit.all_qubits(), key=lambda qubit: qubit.x)),
            expected_output_rule=_support_rule(circuit, rng),
            metadata={
                "backend_size_class": backend_size,
                "backend_snapshot_id": snapshot,
                "synthetic_family": family,
                "negative_case": negative,
            },
        )
        write_benchmark(benchmark, root / "benchmarks" / "track_b" / "holdout" / name)
        record = _record(benchmark)
        record["negative_case"] = negative
        record["two_qubit_gates"] = sum(1 for op in circuit.all_operations() if len(op.qubits) == 2)
        records.append(record)
        index += 1
    write_json(root / "holdout_manifest.json", records)
    return records


def _trackb_cases(rng: random.Random):
    for hubs, leaves_per_hub in ((3, 4), (4, 4), (6, 3), (8, 2)):
        width = hubs * (leaves_per_hub + 1)
        q = cirq.LineQubit.range(width)
        circuit = cirq.Circuit(cirq.H(q[hub]) for hub in range(hubs))
        for hub in range(hubs):
            leaves = list(range(hubs + hub * leaves_per_hub, hubs + (hub + 1) * leaves_per_hub))
            rng.shuffle(leaves)
            for leaf in leaves:
                circuit.append(cirq.CNOT(q[hub], q[leaf]))
                if rng.random() < 0.30:
                    circuit.append(cirq.X(q[leaf]))
        circuit.append(cirq.measure(*q, key="z"))
        yield circuit, "exact_affine_parity_star", False

    for width, roots in ((16, (0, 8)), (24, (0, 8, 16))):
        q = cirq.LineQubit.range(width)
        circuit = cirq.Circuit(cirq.H(q[root]) for root in roots)
        for root in roots:
            frontier = [root]
            targets = [item for item in range(width) if item not in roots][root % len(roots) :: len(roots)]
            for target in targets:
                parent = rng.choice(frontier)
                circuit.append(cirq.CNOT(q[parent], q[target]))
                frontier.append(target)
        circuit.append(cirq.measure(*q, key="z"))
        yield circuit, "parity_tree", False

    for width, layers in ((12, 18), (18, 28), (24, 40)):
        q = cirq.LineQubit.range(width)
        random_sources = sorted(rng.sample(range(width), max(3, width // 4)))
        circuit = cirq.Circuit(cirq.H(q[source]) for source in random_sources)
        for _ in range(layers):
            control, target = rng.sample(range(width), 2)
            circuit.append(cirq.CNOT(q[control], q[target]))
            if rng.random() < 0.10:
                circuit.append(cirq.X(q[target]))
        circuit.append(cirq.measure(*q, key="z"))
        yield circuit, "multi_row_affine", False

    for width in (12, 18, 24):
        q = cirq.LineQubit.range(width)
        circuit = cirq.Circuit(cirq.H(q[0]), cirq.S(q[0]), cirq.CNOT(q[0], q[1]))
        for _ in range(width + 4):
            control, target = rng.sample(range(width), 2)
            circuit.append(cirq.CNOT(q[control], q[target]))
            if rng.random() < 0.20:
                circuit.append(cirq.S(q[target]))
        target = rng.randrange(width)
        circuit.append(cirq.H(q[target]))
        circuit.append(cirq.measure(*q, key="z"))
        yield circuit, "near_miss_non_affine_clifford", True


def _write_jitter_public_replay(source_root: Path, root: Path, backend_records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_source = {(record["size_class"], record["source_snapshot_id"]): [] for record in backend_records}
    for record in backend_records:
        by_source[(record["size_class"], record["source_snapshot_id"])].append(record)
    records: list[dict[str, object]] = []
    for track in ("track_a", "track_b"):
        for source_dir in sorted((source_root / "benchmarks" / track / "public").iterdir()):
            if not source_dir.is_dir():
                continue
            benchmark = load_benchmark(source_dir)
            key = (benchmark.metadata["backend_size_class"], benchmark.metadata["backend_snapshot_id"])
            for backend_record in by_source.get(key, []):
                name = f"{benchmark.name}_{backend_record['variant']}"
                replay = CompetitionBenchmark(
                    track=benchmark.track,
                    name=name,
                    family=f"{benchmark.family}_jitter",
                    circuit=benchmark.circuit,
                    logical_qubits=benchmark.logical_qubits,
                    expected_output_rule=benchmark.expected_output_rule,
                    metadata={
                        **benchmark.metadata,
                        "backend_snapshot_id": backend_record["snapshot_id"],
                        "jitter_variant": backend_record["variant"],
                    },
                )
                write_benchmark(replay, root / "benchmarks" / track / "holdout" / name)
                records.append(_record(replay))
    write_json(root / "holdout_manifest.json", records)
    return records


def _backend_refs(root: Path, size_class: str) -> list[tuple[str, str]]:
    return [(size_class, path.stem) for path in sorted((root / "backends" / size_class).glob("*.json"))]


def _toffoli_6cx(control_a: cirq.Qid, control_b: cirq.Qid, target: cirq.Qid) -> list[cirq.Operation]:
    return [
        cirq.H(target),
        cirq.CNOT(control_b, target),
        cirq.T(target) ** -1,
        cirq.CNOT(control_a, target),
        cirq.T(target),
        cirq.CNOT(control_b, target),
        cirq.T(target) ** -1,
        cirq.T(control_b),
        cirq.CNOT(control_a, target),
        cirq.T(target),
        cirq.CNOT(control_a, control_b),
        cirq.H(target),
        cirq.T(control_a),
        cirq.T(control_b) ** -1,
        cirq.CNOT(control_a, control_b),
    ]


def _deterministic_rule(circuit: cirq.Circuit, rng: random.Random) -> ExpectedOutputRule:
    result = cirq.Simulator(seed=rng.randrange(1_000_000)).run(circuit, repetitions=1)
    return ExpectedOutputRule(rule_type="bitstring", bitstring="".join(str(bit) for bit in result.measurements["z"][0]))


def _support_rule(circuit: cirq.Circuit, rng: random.Random) -> ExpectedOutputRule:
    simulator = cirq.CliffordSimulator(seed=rng.randrange(1_000_000))
    result = simulator.run(circuit, repetitions=2048)
    support = sorted({"".join(str(bit) for bit in row) for row in result.measurements["z"]})
    return ExpectedOutputRule(rule_type="allowed_bitstrings", allowed_bitstrings=tuple(support[:512]))


def _record(benchmark: CompetitionBenchmark) -> dict[str, object]:
    return {
        "name": benchmark.name,
        "track": benchmark.track,
        "family": benchmark.family,
        "backend_size_class": benchmark.metadata["backend_size_class"],
        "backend_snapshot_id": benchmark.metadata["backend_snapshot_id"],
        "logical_qubits": len(benchmark.logical_qubits),
    }


if __name__ == "__main__":
    raise SystemExit(main())
