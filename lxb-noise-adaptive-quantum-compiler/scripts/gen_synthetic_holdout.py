from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cirq

from common import ROOT

from noise_adaptive_hackathon.benchmarks.io import write_benchmark
from noise_adaptive_hackathon.benchmarks.model import CompetitionBenchmark, ExpectedOutputRule
from noise_adaptive_hackathon.formats.backend import CompetitionBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic holdout circuits and perturbed backend snapshots.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "synthetic_holdout")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    backend_refs = _write_backend_variants(args.root, args.out, rng)

    manifest: list[dict[str, object]] = []
    manifest.extend(_write_track_b_holdout(args.out, backend_refs, rng))
    manifest.extend(_write_track_a_holdout(args.out, backend_refs, rng))
    (args.out / "holdout_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


def _write_backend_variants(root: Path, out: Path, rng: random.Random) -> list[tuple[str, str, int]]:
    refs: list[tuple[str, str, int]] = []
    sources = [
        ("medium", root / "backends" / "medium" / "medium_snapshot_a.json"),
        ("large", root / "backends" / "large" / "large_snapshot_a.json"),
    ]
    for size_class, source_path in sources:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        for variant in ("original", "twoq_lognormal", "readout_jitter", "worst_edge_relocated"):
            mutated = _mutate_backend_payload(payload, variant, rng)
            snapshot_id = f"synthetic_{size_class}_{variant}"
            mutated["snapshot_id"] = snapshot_id
            backend = CompetitionBackend.parse_obj(mutated)
            target = out / "backends" / size_class / f"{snapshot_id}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(backend.json(indent=2), encoding="utf-8")
            refs.append((size_class, snapshot_id, backend.num_qubits))
    return refs


def _mutate_backend_payload(payload: dict[str, object], variant: str, rng: random.Random) -> dict[str, object]:
    mutated = json.loads(json.dumps(payload))
    if variant == "original":
        return mutated
    if variant == "twoq_lognormal":
        mutated["two_qubit_error"] = {
            edge: _clamp_error(float(error) * math.exp(rng.gauss(0.0, 0.25)))
            for edge, error in mutated["two_qubit_error"].items()
        }
    elif variant == "readout_jitter":
        mutated["readout_error"] = {
            qubit: _clamp_error(float(error) * math.exp(rng.gauss(0.0, 0.30)))
            for qubit, error in mutated["readout_error"].items()
        }
    elif variant == "worst_edge_relocated":
        errors = dict(mutated["two_qubit_error"])
        worst = max(errors, key=lambda edge: errors[edge])
        best = min(errors, key=lambda edge: errors[edge])
        errors[worst], errors[best] = errors[best], errors[worst]
        mutated["two_qubit_error"] = errors
    else:
        raise ValueError(f"unknown backend variant: {variant}")
    return mutated


def _clamp_error(value: float) -> float:
    return min(max(value, 1e-5), 0.20)


def _write_track_b_holdout(
    out: Path,
    backend_refs: list[tuple[str, str, int]],
    rng: random.Random,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    sample_index = 1

    for width in (8, 16, 24):
        root = rng.randrange(width)
        leaves = [index for index in range(width) if index != root]
        rng.shuffle(leaves)
        qubits = cirq.LineQubit.range(width)
        circuit = cirq.Circuit(cirq.H(qubits[root]), (cirq.CNOT(qubits[root], qubits[leaf]) for leaf in leaves), cirq.measure(*qubits, key="z"))
        sample_index = _write_benchmark(records, out, backend_refs, "track_b", sample_index, "ghz_star", circuit, _ghz_rule(width), rng)

    for width in (8, 16, 24):
        root = rng.randrange(width)
        order = [root] + [index for index in range(width) if index != root]
        rng.shuffle(order[1:])
        qubits = cirq.LineQubit.range(width)
        circuit = cirq.Circuit(cirq.H(qubits[root]))
        for left, right in zip(order, order[1:]):
            circuit.append(cirq.CNOT(qubits[left], qubits[right]))
        circuit.append(cirq.measure(*qubits, key="z"))
        sample_index = _write_benchmark(records, out, backend_refs, "track_b", sample_index, "ghz_chain", circuit, _ghz_rule(width), rng)

    for hubs, leaves_per_hub in ((3, 5), (4, 4), (6, 3), (8, 2)):
        width = hubs * (leaves_per_hub + 1)
        qubits = cirq.LineQubit.range(width)
        circuit = cirq.Circuit()
        for hub_index in range(hubs):
            hub = hub_index
            if rng.random() < 0.5:
                circuit.append(cirq.X(qubits[hub]))
            for leaf_offset in range(leaves_per_hub):
                leaf = hubs + hub_index * leaves_per_hub + leaf_offset
                if rng.random() < 0.25:
                    circuit.append(cirq.X(qubits[leaf]))
                circuit.append(cirq.CNOT(qubits[hub], qubits[leaf]))
        for hub_index in range(hubs - 1):
            circuit.append(cirq.CNOT(qubits[hub_index], qubits[hub_index + 1]))
        circuit.append(cirq.measure(*qubits, key="z"))
        sample_index = _write_benchmark(records, out, backend_refs, "track_b", sample_index, "parity_hub_leaf", circuit, _deterministic_rule(circuit, rng), rng)

    for width, density in ((8, 16), (16, 32), (24, 56)):
        qubits = cirq.LineQubit.range(width)
        circuit = cirq.Circuit()
        for qubit in qubits:
            if rng.random() < 0.25:
                circuit.append(cirq.H(qubit))
                circuit.append(cirq.H(qubit))
            if rng.random() < 0.25:
                circuit.append(cirq.S(qubit))
            if rng.random() < 0.20:
                circuit.append(cirq.X(qubit))
            if rng.random() < 0.20:
                circuit.append(cirq.Z(qubit))
        for _ in range(density):
            left, right = rng.sample(range(width), 2)
            circuit.append(cirq.CNOT(qubits[left], qubits[right]))
        circuit.append(cirq.measure(*qubits, key="z"))
        sample_index = _write_benchmark(records, out, backend_refs, "track_b", sample_index, "random_clifford_cnot", circuit, _deterministic_rule(circuit, rng), rng)

    return records


def _write_track_a_holdout(
    out: Path,
    backend_refs: list[tuple[str, str, int]],
    rng: random.Random,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    sample_index = 1
    for width, t_count in ((6, 3), (7, 6)):
        qubits = cirq.LineQubit.range(width)
        circuit = cirq.Circuit()
        for index in range(t_count):
            qubit = qubits[index % width]
            if index % 2 == 0:
                circuit.append(cirq.X(qubit))
            circuit.append(cirq.T(qubit))
        for left in range(width - 1):
            circuit.append(cirq.CNOT(qubits[left], qubits[-1]))
        circuit.append(cirq.measure(*qubits, key="z"))
        sample_index = _write_benchmark(records, out, backend_refs, "track_a", sample_index, "small_clifford_t", circuit, _deterministic_rule(circuit, rng), rng)

    for width, rounds in ((6, 3), (7, 4)):
        qubits = cirq.LineQubit.range(width)
        circuit = cirq.Circuit()
        for round_index in range(rounds):
            for left in range(width - 1):
                if (left + round_index) % 2 == 0:
                    circuit.append(cirq.CNOT(qubits[left], qubits[left + 1]))
            circuit.append(cirq.T(qubits[round_index % width]))
            circuit.append(cirq.X(qubits[(2 * round_index + 1) % width]))
        circuit.append(cirq.measure(*qubits, key="z"))
        sample_index = _write_benchmark(records, out, backend_refs, "track_a", sample_index, "structured_routing_pressure", circuit, _deterministic_rule(circuit, rng), rng)
    return records


def _write_benchmark(
    records: list[dict[str, object]],
    out: Path,
    backend_refs: list[tuple[str, str, int]],
    track: str,
    sample_index: int,
    family: str,
    circuit: cirq.Circuit,
    rule: ExpectedOutputRule,
    rng: random.Random,
) -> int:
    width = len(circuit.all_qubits())
    compatible = [ref for ref in backend_refs if ref[2] >= width]
    if not compatible:
        raise ValueError(f"no synthetic backend can host {width} logical qubits")
    size_class, snapshot_id, _ = rng.choice(compatible)
    name = f"synthetic_{track[-1]}_{sample_index:03d}"
    benchmark = CompetitionBenchmark(
        track=track,
        name=name,
        family=family,
        circuit=circuit,
        logical_qubits=tuple(sorted(circuit.all_qubits(), key=lambda qubit: qubit.x)),
        expected_output_rule=rule,
        metadata={
            "backend_size_class": size_class,
            "backend_snapshot_id": snapshot_id,
            "synthetic_family": family,
        },
    )
    write_benchmark(benchmark, out / "benchmarks" / track / "holdout" / name)
    records.append(
        {
            "name": name,
            "track": track,
            "family": family,
            "backend_size_class": size_class,
            "backend_snapshot_id": snapshot_id,
            "logical_qubits": len(benchmark.logical_qubits),
        }
    )
    return sample_index + 1


def _ghz_rule(width: int) -> ExpectedOutputRule:
    return ExpectedOutputRule(rule_type="allowed_bitstrings", allowed_bitstrings=("0" * width, "1" * width))


def _deterministic_rule(circuit: cirq.Circuit, rng: random.Random) -> ExpectedOutputRule:
    result = cirq.Simulator(seed=rng.randrange(1_000_000)).run(circuit, repetitions=1)
    bitstring = "".join(str(bit) for bit in result.measurements["z"][0])
    return ExpectedOutputRule(rule_type="bitstring", bitstring=bitstring)


if __name__ == "__main__":
    raise SystemExit(main())
