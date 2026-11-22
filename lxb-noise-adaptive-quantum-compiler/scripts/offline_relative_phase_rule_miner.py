from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import cirq
import numpy as np

from common import ROOT

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.submission.track_a_resynthesis.exact_equivalence import unitary_equivalent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline exact rewrite miner for small H-boundary Clifford+T rules.")
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-enumerated", type=int, default=20000)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "verified_ruleset.json")
    parser.add_argument("--report", type=Path, default=ROOT / "experiments" / "relative_phase_rule_miner_report.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    qubits = tuple(cirq.LineQubit.range(args.width))
    seeded = _seed_rules(qubits)
    mined = _mine_exact_rules(
        qubits,
        max_depth=args.max_depth,
        max_enumerated=args.max_enumerated,
    )
    rules = _dedupe_rules([*seeded, *mined], qubits)
    payload = {
        "schema_version": "verified_ruleset.v1",
        "width": args.width,
        "max_depth": args.max_depth,
        "rule_count": len(rules),
        "rules": rules,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.report.write_text(_render_report(payload), encoding="utf-8")
    return 0


def _seed_rules(qubits: tuple[cirq.Qid, ...]) -> list[dict[str, object]]:
    rules: list[dict[str, object]] = []
    for control in qubits:
        for target in qubits:
            if control == target:
                continue
            before = [cirq.H(control), cirq.H(target), cirq.CNOT(control, target), cirq.H(control), cirq.H(target)]
            after = [cirq.CNOT(target, control)]
            rules.append(_rule("h_cx_h_reversal", before, after, qubits))
    for qubit in qubits:
        rules.append(_rule("hh_cancel", [cirq.H(qubit), cirq.H(qubit)], [], qubits))
        rules.append(_rule("t_tdg_cancel", [cirq.T(qubit), cirq.T(qubit) ** -1], [], qubits))
        rules.append(_rule("s_sdg_cancel", [cirq.S(qubit), cirq.S(qubit) ** -1], [], qubits))
    return [rule for rule in rules if rule["verified"]]


def _mine_exact_rules(
    qubits: tuple[cirq.Qid, ...],
    *,
    max_depth: int,
    max_enumerated: int,
) -> list[dict[str, object]]:
    alphabet = _alphabet(qubits)
    buckets: dict[str, list[list[cirq.Operation]]] = {}
    count = 0
    for depth in range(1, max_depth + 1):
        for indices in product(range(len(alphabet)), repeat=depth):
            ops = [alphabet[index] for index in indices]
            if not _interesting(ops):
                continue
            key = _unitary_key(ops, qubits)
            buckets.setdefault(key, []).append(ops)
            count += 1
            if count >= max_enumerated:
                return _representative_rules(buckets, qubits)
    return _representative_rules(buckets, qubits)


def _representative_rules(
    buckets: dict[str, list[list[cirq.Operation]]],
    qubits: tuple[cirq.Qid, ...],
) -> list[dict[str, object]]:
    rules: list[dict[str, object]] = []
    for ops_list in buckets.values():
        if len(ops_list) < 2:
            continue
        representatives = sorted(ops_list, key=lambda ops: (_cx_count(ops), len(ops), tuple(str(op) for op in ops)))
        best = representatives[0]
        for candidate in representatives[1:8]:
            if _cx_count(best) < _cx_count(candidate) or len(best) + 1 < len(candidate):
                rule = _rule("mined_exact_unitary", candidate, best, qubits)
                if rule["verified"]:
                    rules.append(rule)
    return rules


def _rule(
    family: str,
    before: list[cirq.Operation],
    after: list[cirq.Operation],
    qubits: tuple[cirq.Qid, ...],
) -> dict[str, object]:
    verified = unitary_equivalent(before, after, qubits)
    return {
        "family": family,
        "width": len(qubits),
        "before": [_op_record(op) for op in before],
        "after": [_op_record(op) for op in after],
        "before_cx": _cx_count(before),
        "after_cx": _cx_count(after),
        "before_ops": len(before),
        "after_ops": len(after),
        "unitary_hash": _unitary_key(before, qubits),
        "verified": verified,
        "relative_phase": False,
    }


def _dedupe_rules(rules: list[dict[str, object]], qubits: tuple[cirq.Qid, ...]) -> list[dict[str, object]]:
    del qubits
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, object]] = []
    for rule in rules:
        key = (
            json.dumps(rule["before"], sort_keys=True),
            json.dumps(rule["after"], sort_keys=True),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rule)
    return sorted(deduped, key=lambda rule: (str(rule["family"]), int(rule["after_cx"]), int(rule["after_ops"])))


def _alphabet(qubits: tuple[cirq.Qid, ...]) -> list[cirq.Operation]:
    ops: list[cirq.Operation] = []
    for qubit in qubits:
        ops.extend([cirq.H(qubit), cirq.T(qubit), cirq.T(qubit) ** -1, cirq.S(qubit), cirq.S(qubit) ** -1])
    for control in qubits:
        for target in qubits:
            if control != target:
                ops.append(cirq.CNOT(control, target))
    return ops


def _interesting(ops: list[cirq.Operation]) -> bool:
    names = [gate_name_for_operation(op) for op in ops]
    return "h" in names and "cx" in names


def _unitary_key(ops: list[cirq.Operation], qubits: tuple[cirq.Qid, ...]) -> str:
    unitary = cirq.Circuit(ops).unitary(qubit_order=list(qubits))
    flat = unitary.reshape(-1)
    pivot = next((value for value in flat if abs(value) > 1e-9), None)
    normalized = unitary if pivot is None else unitary / (pivot / abs(pivot))
    rounded = np.round(normalized.real, 10) + 1j * np.round(normalized.imag, 10)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def _op_record(op: cirq.Operation) -> dict[str, object]:
    return {
        "gate": gate_name_for_operation(op),
        "qubits": [qubit.x for qubit in op.qubits],
        "text": str(op),
    }


def _cx_count(ops: list[cirq.Operation]) -> int:
    return sum(1 for op in ops if gate_name_for_operation(op) == "cx")


def _render_report(payload: dict[str, object]) -> str:
    rules = payload["rules"]
    improving_cx = [rule for rule in rules if int(rule["after_cx"]) < int(rule["before_cx"])]
    lines = [
        "# Relative-Phase Rule Miner Report",
        "",
        f"- Width: {payload['width']}",
        f"- Max depth: {payload['max_depth']}",
        f"- Verified rules: {payload['rule_count']}",
        f"- CX-reducing rules: {len(improving_cx)}",
        "",
        "## Rule Families",
        "",
    ]
    families: dict[str, int] = {}
    for rule in rules:
        families[str(rule["family"])] = families.get(str(rule["family"]), 0) + 1
    for family, count in sorted(families.items()):
        lines.append(f"- {family}: {count}")
    lines.extend(["", "## Decision", ""])
    lines.append(
        "Rules are exported for offline inspection only. Runtime application still requires pattern matching plus an exact local miter and metric guard."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
