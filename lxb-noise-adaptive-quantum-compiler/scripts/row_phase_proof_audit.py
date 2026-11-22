from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from typing import Any

import cirq
import numpy as np

from common import ROOT

from noise_adaptive_hackathon.benchmarks.io import load_benchmark
from noise_adaptive_hackathon.submission.relative_phase_toffoli import (
    final_z_measurement_equivalent,
    relative_phase_toffoli_rewrites,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit final-Z row-phase equivalence for V19 rewrites.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "row_phase_proof_audit_v19.md")
    parser.add_argument(
        "--negative-out",
        type=Path,
        default=ROOT / "experiments" / "row_phase_negative_cases_v19.json",
    )
    parser.add_argument("--random-states", type=int, default=32)
    parser.add_argument("--hidden-preps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2027)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _audit_public_track_a(args.root, random_states=args.random_states, hidden_preps=args.hidden_preps, seed=args.seed)
    negatives = _negative_cases(seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_render_report(rows, negatives), encoding="utf-8")
    args.negative_out.write_text(json.dumps(negatives, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def _audit_public_track_a(
    root: Path,
    *,
    random_states: int,
    hidden_preps: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    benchmark_root = root / "benchmarks" / "track_a" / "public"
    for benchmark_dir in sorted(path for path in benchmark_root.iterdir() if path.is_dir()):
        benchmark = load_benchmark(benchmark_dir)
        rewrites = relative_phase_toffoli_rewrites(benchmark.circuit)
        for rewrite in rewrites:
            rows.append(
                _audit_pair(
                    sample=benchmark.name,
                    rewrite_name=rewrite.name,
                    original=benchmark.circuit,
                    rewritten=rewrite.circuit,
                    random_states=random_states,
                    hidden_preps=hidden_preps,
                    seed=seed,
                    old_cx=rewrite.old_cx,
                    new_cx=rewrite.new_cx,
                    replacements=rewrite.replacements,
                )
            )
    return rows


def _audit_pair(
    *,
    sample: str,
    rewrite_name: str,
    original: cirq.Circuit,
    rewritten: cirq.Circuit,
    random_states: int,
    hidden_preps: int,
    seed: int,
    old_cx: int,
    new_cx: int,
    replacements: int,
) -> dict[str, Any]:
    qubits = tuple(_measurement_qubits(original))
    left = _unitary_without_measurement(original, qubits)
    right = _unitary_without_measurement(rewritten, qubits)
    diagonal = _row_phase_diagonal(left, right)
    rng = np.random.default_rng(seed + hash((sample, rewrite_name)) % 100_000)
    return {
        "sample": sample,
        "rewrite": rewrite_name,
        "replacements": replacements,
        "old_cx": old_cx,
        "new_cx": new_cx,
        "cx_delta": new_cx - old_cx,
        "row_phase_equivalent": diagonal is not None,
        "diagonal_max_residual": _diagonal_residual(left, right, diagonal) if diagonal is not None else None,
        "basis_probability_max_delta": _basis_probability_max_delta(left, right),
        "random_state_probability_max_delta": _random_state_probability_max_delta(left, right, rng, random_states),
        "hidden_prep_probability_max_delta": _hidden_prep_probability_max_delta(left, right, qubits, hidden_preps, seed),
        "api_miter": final_z_measurement_equivalent(original, rewritten),
    }


def _negative_cases(*, seed: int) -> list[dict[str, Any]]:
    q0, q1, q2 = cirq.LineQubit.range(3)
    exact = _exact_toffoli(q0, q1, q2)
    relative = _relative_phase_toffoli(q0, q1, q2)
    cases = [
        {
            "name": "relative_phase_followed_by_h",
            "expect_accept": False,
            "accepted": final_z_measurement_equivalent(
                cirq.Circuit([*exact, cirq.H(q2), cirq.measure(q0, q1, q2, key="z")]),
                cirq.Circuit([*relative, cirq.H(q2), cirq.measure(q0, q1, q2, key="z")]),
            ),
        },
        {
            "name": "non_final_measurement",
            "expect_accept": False,
            "accepted": final_z_measurement_equivalent(
                cirq.Circuit([cirq.measure(q0, key="z"), cirq.H(q0)]),
                cirq.Circuit([cirq.measure(q0, key="z"), cirq.T(q0)]),
            ),
        },
        {
            "name": "non_z_measurement",
            "expect_accept": False,
            "accepted": final_z_measurement_equivalent(
                cirq.Circuit([cirq.H(q0), cirq.measure(q0, key="m")]),
                cirq.Circuit([cirq.T(q0), cirq.measure(q0, key="m")]),
            ),
        },
        {
            "name": "row_phase_false_case",
            "expect_accept": False,
            "accepted": final_z_measurement_equivalent(
                cirq.Circuit([cirq.H(q0), cirq.measure(q0, key="z")]),
                cirq.Circuit([cirq.measure(q0, key="z")]),
            ),
        },
    ]
    # Include a randomized false case so accidental permissiveness is easier to catch.
    rng = Random(seed)
    ops_left = [rng.choice([cirq.H, cirq.X, cirq.S, cirq.T])(q0) for _ in range(3)]
    ops_right = [rng.choice([cirq.H, cirq.X, cirq.S, cirq.T])(q0) for _ in range(3)]
    cases.append(
        {
            "name": "random_single_qubit_false_case",
            "expect_accept": False,
            "accepted": final_z_measurement_equivalent(
                cirq.Circuit([*ops_left, cirq.measure(q0, key="z")]),
                cirq.Circuit([*ops_right, cirq.measure(q0, key="z")]),
            ),
        }
    )
    for case in cases:
        case["passed"] = bool(case["accepted"]) == bool(case["expect_accept"])
    return cases


def _row_phase_diagonal(left: np.ndarray, right: np.ndarray, *, atol: float = 1e-8) -> np.ndarray | None:
    phases: list[complex] = []
    for row_index in range(left.shape[0]):
        left_row = left[row_index, :]
        right_row = right[row_index, :]
        pivot_index = int(np.argmax(np.abs(left_row)))
        if abs(left_row[pivot_index]) < atol:
            if np.max(np.abs(right_row)) >= atol:
                return None
            phases.append(1.0 + 0.0j)
            continue
        phase = right_row[pivot_index] / left_row[pivot_index]
        if abs(phase) < atol:
            return None
        phase /= abs(phase)
        if np.max(np.abs(right_row - phase * left_row)) > atol:
            return None
        phases.append(phase)
    return np.array(phases, dtype=np.complex128)


def _diagonal_residual(left: np.ndarray, right: np.ndarray, diagonal: np.ndarray) -> float:
    reconstructed = diagonal[:, np.newaxis] * left
    return float(np.max(np.abs(right - reconstructed)))


def _basis_probability_max_delta(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.abs(left) ** 2 - np.abs(right) ** 2)))


def _random_state_probability_max_delta(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
    count: int,
) -> float:
    worst = 0.0
    for _ in range(count):
        state = rng.normal(size=left.shape[1]) + 1j * rng.normal(size=left.shape[1])
        state /= np.linalg.norm(state)
        worst = max(worst, _probability_delta(left @ state, right @ state))
    return float(worst)


def _hidden_prep_probability_max_delta(
    left: np.ndarray,
    right: np.ndarray,
    qubits: tuple[cirq.Qid, ...],
    count: int,
    seed: int,
) -> float:
    rng = Random(seed)
    gates = ("i", "x", "h", "xh", "s", "t", "hs", "ht")
    worst = 0.0
    for _ in range(count):
        prep = cirq.Circuit()
        for qubit in qubits:
            choice = rng.choice(gates)
            prep.append(_prep_ops(qubit, choice))
        state = cirq.Simulator(seed=seed).simulate(prep, qubit_order=list(qubits)).final_state_vector
        worst = max(worst, _probability_delta(left @ state, right @ state))
    return float(worst)


def _probability_delta(left_state: np.ndarray, right_state: np.ndarray) -> float:
    return float(np.max(np.abs(np.abs(left_state) ** 2 - np.abs(right_state) ** 2)))


def _measurement_qubits(circuit: cirq.Circuit) -> tuple[cirq.Qid, ...]:
    measurements = [op for op in circuit.all_operations() if isinstance(op.gate, cirq.MeasurementGate)]
    if len(measurements) != 1:
        raise ValueError("expected exactly one measurement")
    return tuple(measurements[0].qubits)


def _unitary_without_measurement(circuit: cirq.Circuit, qubits: tuple[cirq.Qid, ...]) -> np.ndarray:
    return cirq.Circuit(op for op in circuit.all_operations() if not isinstance(op.gate, cirq.MeasurementGate)).unitary(
        qubit_order=list(qubits)
    )


def _exact_toffoli(control_a: cirq.Qid, control_b: cirq.Qid, target: cirq.Qid) -> list[cirq.Operation]:
    from noise_adaptive_hackathon.submission.track_a_resynthesis.reversible_templates import toffoli_6cx

    return toffoli_6cx(control_a, control_b, target)


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


def _prep_ops(qubit: cirq.Qid, choice: str) -> list[cirq.Operation]:
    mapping = {
        "i": [],
        "x": [cirq.X(qubit)],
        "h": [cirq.H(qubit)],
        "xh": [cirq.X(qubit), cirq.H(qubit)],
        "s": [cirq.S(qubit)],
        "t": [cirq.T(qubit)],
        "hs": [cirq.H(qubit), cirq.S(qubit)],
        "ht": [cirq.H(qubit), cirq.T(qubit)],
    }
    return mapping[choice]


def _render_report(rows: list[dict[str, Any]], negatives: list[dict[str, Any]]) -> str:
    accepted = [row for row in rows if row["row_phase_equivalent"] and row["api_miter"]]
    max_basis = max((float(row["basis_probability_max_delta"]) for row in rows), default=0.0)
    max_random = max((float(row["random_state_probability_max_delta"]) for row in rows), default=0.0)
    max_hidden = max((float(row["hidden_prep_probability_max_delta"]) for row in rows), default=0.0)
    negative_failures = [case for case in negatives if not case["passed"]]
    lines = [
        "# Row-Phase Proof Audit V19",
        "",
        f"- Rewrites audited: {len(rows)}",
        f"- Row-phase accepted: {len(accepted)}",
        f"- Negative cases: {len(negatives)}",
        f"- Negative failures: {len(negative_failures)}",
        f"- Max basis-input probability delta: {max_basis:.3e}",
        f"- Max random-state probability delta: {max_random:.3e}",
        f"- Max hidden-prep probability delta: {max_hidden:.3e}",
        "",
        "## Audited Rewrites",
        "",
        "| Sample | Rewrite | Replacements | CX | Row-phase | Residual | Random delta | Hidden delta |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {sample} | {rewrite} | {replacements} | {old_cx}->{new_cx} | {row_phase_equivalent}/{api_miter} | "
            "{diagonal_max_residual:.3e} | {random_state_probability_max_delta:.3e} | "
            "{hidden_prep_probability_max_delta:.3e} |".format(**row)
        )
    lines.extend(["", "## Negative Cases", "", "| Case | Expected | Accepted | Passed |", "|---|---|---|---|"])
    for case in negatives:
        lines.append(f"| {case['name']} | {case['expect_accept']} | {case['accepted']} | {case['passed']} |")
    decision = "PASS" if rows and not negative_failures and len(accepted) == len(rows) else "FAIL"
    lines.extend(["", "## Decision", "", f"`{decision}`: V19 row-phase rewrites preserve final-Z probabilities under the audited conditions."])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
