from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cirq

from common import ROOT

from noise_adaptive_hackathon.benchmarks.io import write_benchmark
from noise_adaptive_hackathon.benchmarks.model import CompetitionBenchmark, ExpectedOutputRule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic Track B affine measurement-channel benchmarks.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "synthetic_trackb_affine_v13_1")
    parser.add_argument("--seed", type=int, default=131)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    _copy_backend(args.root, args.out)
    records: list[dict[str, object]] = []
    index = 1
    for circuit, family, negative in _cases(rng):
        name = f"affine_b_{index:03d}"
        benchmark = CompetitionBenchmark(
            track="track_b",
            name=name,
            family=family,
            circuit=circuit,
            logical_qubits=tuple(sorted(circuit.all_qubits(), key=lambda qubit: qubit.x)),
            expected_output_rule=_support_rule(circuit, rng),
            metadata={
                "backend_size_class": "large",
                "backend_snapshot_id": "large_snapshot_a",
                "synthetic_family": family,
                "negative_case": negative,
            },
        )
        write_benchmark(benchmark, args.out / "benchmarks" / "track_b" / "holdout" / name)
        records.append(
            {
                "name": name,
                "family": family,
                "negative_case": negative,
                "logical_qubits": len(benchmark.logical_qubits),
                "two_qubit_gates": sum(1 for op in circuit.all_operations() if len(op.qubits) == 2),
            }
        )
        index += 1
    (args.out / "holdout_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return 0


def _cases(rng: random.Random):
    for hubs, leaves in ((3, 4), (4, 4), (6, 3)):
        yield _affine_star(hubs, leaves, rng), "exact_affine_parity_star", False
    for width, roots in ((16, (0, 8)), (24, (0, 8, 16))):
        yield _parity_tree(width, roots, rng), "parity_tree", False
    for width, layers in ((12, 16), (18, 24), (24, 32)):
        yield _multi_row_affine(width, layers, rng), "multi_row_affine_parity", False
    for width in (12, 18):
        yield _near_miss_non_affine(width, rng), "near_miss_non_affine_clifford", True


def _affine_star(hubs: int, leaves_per_hub: int, rng: random.Random) -> cirq.Circuit:
    width = hubs * (leaves_per_hub + 1)
    qubits = cirq.LineQubit.range(width)
    circuit = cirq.Circuit(cirq.H(qubits[hub]) for hub in range(hubs))
    for hub in range(hubs):
        base = hubs + hub * leaves_per_hub
        leaves = list(range(base, base + leaves_per_hub))
        rng.shuffle(leaves)
        for leaf in leaves:
            circuit.append(cirq.CNOT(qubits[hub], qubits[leaf]))
            if rng.random() < 0.35:
                circuit.append(cirq.X(qubits[leaf]))
    circuit.append(cirq.measure(*qubits, key="z"))
    return circuit


def _parity_tree(width: int, roots: tuple[int, ...], rng: random.Random) -> cirq.Circuit:
    qubits = cirq.LineQubit.range(width)
    circuit = cirq.Circuit(cirq.H(qubits[root]) for root in roots)
    remaining = [index for index in range(width) if index not in roots]
    rng.shuffle(remaining)
    chunks = [remaining[index :: len(roots)] for index in range(len(roots))]
    for root, chunk in zip(roots, chunks):
        frontier = [root]
        for target in chunk:
            parent = rng.choice(frontier)
            circuit.append(cirq.CNOT(qubits[parent], qubits[target]))
            frontier.append(target)
            if rng.random() < 0.25:
                circuit.append(cirq.X(qubits[target]))
    circuit.append(cirq.measure(*qubits, key="z"))
    return circuit


def _multi_row_affine(width: int, layers: int, rng: random.Random) -> cirq.Circuit:
    qubits = cirq.LineQubit.range(width)
    random_sources = sorted(rng.sample(range(width), max(3, width // 4)))
    circuit = cirq.Circuit(cirq.H(qubits[source]) for source in random_sources)
    for _ in range(layers):
        control, target = rng.sample(range(width), 2)
        circuit.append(cirq.CNOT(qubits[control], qubits[target]))
        if rng.random() < 0.12:
            circuit.append(cirq.X(qubits[target]))
    circuit.append(cirq.measure(*qubits, key="z"))
    return circuit


def _near_miss_non_affine(width: int, rng: random.Random) -> cirq.Circuit:
    qubits = cirq.LineQubit.range(width)
    circuit = cirq.Circuit(cirq.H(qubits[0]), cirq.S(qubits[0]), cirq.CNOT(qubits[0], qubits[1]))
    for _ in range(max(8, width)):
        control, target = rng.sample(range(width), 2)
        circuit.append(cirq.CNOT(qubits[control], qubits[target]))
        if rng.random() < 0.20:
            circuit.append(cirq.S(qubits[target]))
    target = rng.randrange(width)
    circuit.append(cirq.S(qubits[target]))
    circuit.append(cirq.H(qubits[target]))
    circuit.append(cirq.measure(*qubits, key="z"))
    return circuit


def _support_rule(circuit: cirq.Circuit, rng: random.Random) -> ExpectedOutputRule:
    simulator = cirq.CliffordSimulator(seed=rng.randrange(1_000_000))
    result = simulator.run(circuit, repetitions=4096)
    support = sorted({"".join(str(bit) for bit in row) for row in result.measurements["z"]})
    if len(support) <= 512:
        return ExpectedOutputRule(rule_type="allowed_bitstrings", allowed_bitstrings=tuple(support))
    sample = support[:512]
    return ExpectedOutputRule(rule_type="allowed_bitstrings", allowed_bitstrings=tuple(sample))


def _copy_backend(root: Path, out: Path) -> None:
    source = root / "backends" / "large" / "large_snapshot_a.json"
    target = out / "backends" / "large" / "large_snapshot_a.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


if __name__ == "__main__":
    raise SystemExit(main())
