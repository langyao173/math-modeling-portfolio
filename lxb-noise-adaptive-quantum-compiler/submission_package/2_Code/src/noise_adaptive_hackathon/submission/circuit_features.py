from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import cirq


LogicalPair = tuple[int, int]


@dataclass(frozen=True)
class Interaction:
    pair: LogicalPair
    count: int
    weight: float
    first_index: int
    last_index: int


@dataclass(frozen=True)
class CircuitFeatures:
    logical_qubits: tuple[cirq.LineQubit, ...]
    interactions: tuple[Interaction, ...]
    weighted_degrees: dict[int, float]
    two_qubit_pairs_by_op: dict[int, LogicalPair]

    @property
    def logical_ids(self) -> tuple[int, ...]:
        return tuple(qubit.x for qubit in self.logical_qubits)


def extract_circuit_features(circuit: cirq.Circuit) -> CircuitFeatures:
    logical_qubits = tuple(sorted(circuit.all_qubits(), key=lambda qubit: qubit.x))
    operations = list(circuit.all_operations())
    max_index = max(len(operations) - 1, 1)
    stats: dict[LogicalPair, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "weight": 0.0, "first": float("inf"), "last": -1.0}
    )
    pairs_by_op: dict[int, LogicalPair] = {}

    for index, op in enumerate(operations):
        if len(op.qubits) != 2:
            continue
        left, right = sorted(qubit.x for qubit in op.qubits)
        pair = (left, right)
        recency = 1.0 + 0.35 * (1.0 - index / max_index)
        criticality = 1.0 + 0.15 * (index / max_index)
        stats[pair]["count"] += 1.0
        stats[pair]["weight"] += recency * criticality
        stats[pair]["first"] = min(stats[pair]["first"], float(index))
        stats[pair]["last"] = max(stats[pair]["last"], float(index))
        pairs_by_op[index] = pair

    interactions = tuple(
        Interaction(
            pair=pair,
            count=int(values["count"]),
            weight=float(values["weight"]),
            first_index=int(values["first"]),
            last_index=int(values["last"]),
        )
        for pair, values in sorted(stats.items())
    )
    weighted_degrees = {qubit.x: 0.0 for qubit in logical_qubits}
    for interaction in interactions:
        left, right = interaction.pair
        weighted_degrees[left] += interaction.weight
        weighted_degrees[right] += interaction.weight

    return CircuitFeatures(
        logical_qubits=logical_qubits,
        interactions=interactions,
        weighted_degrees=weighted_degrees,
        two_qubit_pairs_by_op=pairs_by_op,
    )


def future_two_qubit_pairs(
    operations: list[cirq.Operation],
    start_index: int,
    limit: int = 16,
) -> list[LogicalPair]:
    pairs: list[LogicalPair] = []
    for op in operations[start_index + 1 :]:
        if len(op.qubits) != 2:
            continue
        pairs.append(tuple(sorted((op.qubits[0].x, op.qubits[1].x))))
        if len(pairs) >= limit:
            break
    return pairs
