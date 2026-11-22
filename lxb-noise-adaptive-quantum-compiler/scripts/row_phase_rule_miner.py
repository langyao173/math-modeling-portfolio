from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

import cirq
import numpy as np

from common import ROOT

from noise_adaptive_hackathon.evaluator.validate_circuit import gate_name_for_operation
from noise_adaptive_hackathon.submission.relative_phase_toffoli import final_z_measurement_equivalent
from noise_adaptive_hackathon.submission.track_a_resynthesis.reversible_templates import toffoli_6cx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline row-phase rewrite miner for final-Z circuits.")
    parser.add_argument("--width", action="append", type=int, default=[])
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-enumerated", type=int, default=20000)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "verified_ruleset_row_phase.json")
    parser.add_argument("--report", type=Path, default=ROOT / "experiments" / "row_phase_rule_miner_report.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    widths = tuple(args.width or (3, 4))
    all_rules: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for width in widths:
        qubits = tuple(cirq.LineQubit.range(width))
        seeded = _seed_rules(qubits)
        mined = _mine_row_phase_rules(
            qubits,
            max_depth=args.max_depth,
            max_enumerated=args.max_enumerated,
        )
        rules = _dedupe_rules([*seeded, *mined])
        summaries.append(
            {
                "width": width,
                "seeded_rules": len(seeded),
                "mined_rules": len(mined),
                "deduped_rules": len(rules),
                "cx_reducing_rules": sum(1 for rule in rules if int(rule["after_cx"]) < int(rule["before_cx"])),
            }
        )
        all_rules.extend(rules)
    all_rules = _dedupe_rules(all_rules)
    payload = {
        "schema_version": "verified_row_phase_ruleset.v1",
        "widths": list(widths),
        "max_depth": args.max_depth,
        "max_enumerated_per_width": args.max_enumerated,
        "rule_count": len(all_rules),
        "summaries": summaries,
        "rules": all_rules,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.report.write_text(_render_report(payload), encoding="utf-8")
    return 0


def _seed_rules(qubits: tuple[cirq.Qid, ...]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for target in qubits:
        controls = [qubit for qubit in qubits if qubit != target]
        for control_a in controls:
            for control_b in controls:
                if control_a == control_b:
                    continue
                before = toffoli_6cx(control_a, control_b, target)
                after = _relative_phase_toffoli(control_a, control_b, target)
                rules.append(_rule("relative_phase_toffoli", before, after, qubits))
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


def _mine_row_phase_rules(
    qubits: tuple[cirq.Qid, ...],
    *,
    max_depth: int,
    max_enumerated: int,
) -> list[dict[str, Any]]:
    alphabet = _alphabet(qubits)
    buckets: dict[str, list[list[cirq.Operation]]] = {}
    count = 0
    for depth in range(1, max_depth + 1):
        for indices in product(range(len(alphabet)), repeat=depth):
            ops = [alphabet[index] for index in indices]
            if not _interesting(ops):
                continue
            key = _row_phase_key(ops, qubits)
            buckets.setdefault(key, []).append(ops)
            count += 1
            if count >= max_enumerated:
                return _representative_rules(buckets, qubits)
    return _representative_rules(buckets, qubits)


def _representative_rules(
    buckets: dict[str, list[list[cirq.Operation]]],
    qubits: tuple[cirq.Qid, ...],
) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for ops_list in buckets.values():
        if len(ops_list) < 2:
            continue
        representatives = sorted(ops_list, key=lambda ops: (_cx_count(ops), len(ops), tuple(str(op) for op in ops)))
        best = representatives[0]
        for candidate in representatives[1:10]:
            if _cx_count(best) < _cx_count(candidate) or len(best) + 1 < len(candidate):
                rule = _rule("mined_row_phase", candidate, best, qubits)
                if rule["verified"]:
                    rules.append(rule)
    return rules


def _rule(
    family: str,
    before: list[cirq.Operation],
    after: list[cirq.Operation],
    qubits: tuple[cirq.Qid, ...],
) -> dict[str, Any]:
    verified = _row_phase_equivalent_ops(before, after, qubits)
    return {
        "family": family,
        "width": len(qubits),
        "before": [_op_record(op) for op in before],
        "after": [_op_record(op) for op in after],
        "before_cx": _cx_count(before),
        "after_cx": _cx_count(after),
        "before_ops": len(before),
        "after_ops": len(after),
        "row_phase_hash": _row_phase_key(before, qubits),
        "verified": verified,
        "runtime_status": "offline_only_requires_pattern_match_and_exact_miter",
    }


def _row_phase_equivalent_ops(
    before: list[cirq.Operation],
    after: list[cirq.Operation],
    qubits: tuple[cirq.Qid, ...],
) -> bool:
    return final_z_measurement_equivalent(
        cirq.Circuit([*before, cirq.measure(*qubits, key="z")]),
        cirq.Circuit([*after, cirq.measure(*qubits, key="z")]),
    )


def _row_phase_key(ops: list[cirq.Operation], qubits: tuple[cirq.Qid, ...]) -> str:
    unitary = cirq.Circuit(ops).unitary(qubit_order=list(qubits))
    canonical = np.zeros_like(unitary, dtype=np.complex128)
    for row_index in range(unitary.shape[0]):
        row = unitary[row_index, :]
        pivot = next((value for value in row if abs(value) > 1e-9), None)
        if pivot is None:
            canonical[row_index, :] = row
            continue
        canonical[row_index, :] = row / (pivot / abs(pivot))
    rounded = np.round(canonical.real, 10) + 1j * np.round(canonical.imag, 10)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def _alphabet(qubits: tuple[cirq.Qid, ...]) -> list[cirq.Operation]:
    ops: list[cirq.Operation] = []
    for qubit in qubits:
        ops.extend(
            [
                cirq.H(qubit),
                cirq.T(qubit),
                cirq.T(qubit) ** -1,
                cirq.S(qubit),
                cirq.S(qubit) ** -1,
                cirq.X(qubit),
                cirq.Z(qubit),
            ]
        )
    for control in qubits:
        for target in qubits:
            if control != target:
                ops.append(cirq.CNOT(control, target))
    return ops


def _interesting(ops: list[cirq.Operation]) -> bool:
    names = [gate_name_for_operation(op) for op in ops]
    return "h" in names and "cx" in names and any(name in {"t", "tdg", "s", "sdg"} for name in names)


def _relative_phase_toffoli(control_a: cirq.Qid, control_b: cirq.Qid, target: cirq.Qid) -> list[cirq.Operation]:
    return [
        cirq.H(target),
        cirq.T(target),
        cirq.CNOT(control_b, target),
        cirq.T(target) ** -1,
        cirq.CNOT(control_a, target),
        cirq.T(target),
        cirq.CNOT(control_b, target),
        cirq.T(target) ** -1,
        cirq.H(target),
    ]


def _dedupe_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, int]] = set()
    deduped: list[dict[str, Any]] = []
    for rule in rules:
        key = (
            json.dumps(rule["before"], sort_keys=True),
            json.dumps(rule["after"], sort_keys=True),
            int(rule["width"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rule)
    return sorted(
        deduped,
        key=lambda rule: (
            str(rule["family"]),
            int(rule["width"]),
            int(rule["after_cx"]),
            int(rule["after_ops"]),
            json.dumps(rule["before"], sort_keys=True),
        ),
    )


def _op_record(op: cirq.Operation) -> dict[str, Any]:
    return {
        "gate": gate_name_for_operation(op),
        "qubits": [qubit.x for qubit in op.qubits],
        "text": str(op),
    }


def _cx_count(ops: list[cirq.Operation]) -> int:
    return sum(1 for op in ops if gate_name_for_operation(op) == "cx")


def _render_report(payload: dict[str, Any]) -> str:
    rules = payload["rules"]
    improving_cx = [rule for rule in rules if int(rule["after_cx"]) < int(rule["before_cx"])]
    families: dict[str, int] = {}
    for rule in rules:
        families[str(rule["family"])] = families.get(str(rule["family"]), 0) + 1
    lines = [
        "# Row-Phase Rule Miner Report",
        "",
        f"- Widths: {payload['widths']}",
        f"- Max depth: {payload['max_depth']}",
        f"- Max enumerated per width: {payload['max_enumerated_per_width']}",
        f"- Verified row-phase rules: {payload['rule_count']}",
        f"- CX-reducing rules: {len(improving_cx)}",
        "",
        "## Width Summaries",
        "",
        "| Width | Seeded | Mined | Deduped | CX-reducing |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in payload["summaries"]:
        lines.append(
            f"| {item['width']} | {item['seeded_rules']} | {item['mined_rules']} | "
            f"{item['deduped_rules']} | {item['cx_reducing_rules']} |"
        )
    lines.extend(["", "## Rule Families", ""])
    for family, count in sorted(families.items()):
        lines.append(f"- {family}: {count}")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Rules are exported for offline inspection only. Runtime promotion still requires structural pattern matching, exact full-circuit row-phase miter, validator/equivalence/noiseless checks, and a metric guard.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
