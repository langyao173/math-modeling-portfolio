from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cirq

from noise_adaptive_hackathon.evaluator.equivalence import LayoutSidecar, validate_compiler_equivalence
from noise_adaptive_hackathon.evaluator.validate_circuit import validate_compiled_circuit
from noise_adaptive_hackathon.simulator.stim_runner import run_track_b

from upper_bound_common import (
    ROOT,
    best_readout_qubits,
    best_same_width_readout_cap,
    cap_for_rule_on_physical,
    compile_current_root,
    illegal_anchor_oracle_cap,
    load_public_samples,
)


DEFAULT_JSON = ROOT / "experiments" / "track_b_output_representation_probe.json"
DEFAULT_MD = ROOT / "experiments" / "track_b_output_representation_report.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe whether Track B output representation/readout compression is legal.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows: dict[str, Any] = {}
    for loaded in load_public_samples(ROOT, targets=("track_b_001", "track_b_002")):
        rows[loaded.name] = probe_sample(loaded.benchmark, loaded.backend, seed=args.seed, shots=args.shots)
    payload = {
        "variant": "track_b_output_representation_probe_dev_only",
        "seed": args.seed,
        "shots": args.shots,
        "samples": rows,
        "decision": decision(rows),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


def probe_sample(benchmark, backend, *, seed: int, shots: int) -> dict[str, Any]:
    width = len(benchmark.logical_qubits)
    root = compile_current_root(benchmark.circuit, backend, seed=seed)
    root_run = run_track_b(benchmark, root.circuit, backend, shots=shots, seed=seed)
    same_width_cap, best_physical, method = best_same_width_readout_cap(benchmark.expected_output_rule, backend, width)
    illegal_cap, anchors, illegal_reason = illegal_anchor_oracle_cap(benchmark.expected_output_rule, backend, width)

    candidates = []
    candidates.append(evaluate_candidate(
        benchmark,
        backend,
        name="zero_cx_best_same_width_valid_string",
        circuit=zero_cx_valid_string_circuit(benchmark, backend),
        layout={logical.x: physical for logical, physical in zip(benchmark.logical_qubits, best_readout_qubits(backend, width))},
        seed=seed,
        shots=shots,
    ))
    candidates.append(evaluate_candidate(
        benchmark,
        backend,
        name="illegal_anchor_measurement_width",
        circuit=illegal_anchor_circuit(benchmark, backend),
        layout={logical.x: physical for logical, physical in zip(benchmark.logical_qubits, best_readout_qubits(backend, width))},
        seed=seed,
        shots=shots,
    ))
    return {
        "family": benchmark.family,
        "width": width,
        "root_success": root_run.success_rate,
        "root_cx": sum(1 for op in root.circuit.all_operations() if len(op.qubits) == 2),
        "same_width_best_readout_cap": same_width_cap,
        "same_width_best_physical": list(best_physical),
        "same_width_method": method,
        "illegal_compressed_oracle_cap": illegal_cap,
        "illegal_anchor_count": anchors,
        "illegal_reason": illegal_reason,
        "root_gap_to_same_width_cap": same_width_cap - root_run.success_rate,
        "same_width_cap_to_illegal_oracle_gap": illegal_cap - same_width_cap,
        "representation_candidates": candidates,
    }


def zero_cx_valid_string_circuit(benchmark, backend) -> cirq.Circuit:
    width = len(benchmark.logical_qubits)
    physical = best_readout_qubits(backend, width)
    bitstring = valid_representative_bitstring(benchmark)
    ops: list[cirq.Operation] = []
    for bit, qubit in zip(bitstring, physical):
        if bit == "1":
            ops.append(cirq.X(cirq.LineQubit(qubit)))
    ops.append(cirq.measure(*(cirq.LineQubit(index) for index in physical), key="z"))
    return cirq.Circuit(ops)


def illegal_anchor_circuit(benchmark, backend) -> cirq.Circuit:
    physical = best_readout_qubits(backend, 1)
    return cirq.Circuit(cirq.measure(cirq.LineQubit(physical[0]), key="z"))


def valid_representative_bitstring(benchmark) -> str:
    rule = benchmark.expected_output_rule
    width = len(benchmark.logical_qubits)
    if rule.rule_type == "allowed_bitstrings":
        return rule.allowed_bitstrings[0]
    bits = [0] * width
    for indices, value in rule.parity_checks:
        if len(indices) == 2 and indices[0] < indices[1]:
            bits[indices[1]] = bits[indices[0]] ^ value
    out = "".join(str(bit) for bit in bits)
    if rule.is_correct(out):
        return out
    for value in range(1 << width):
        candidate = format(value, f"0{width}b")
        if rule.is_correct(candidate):
            return candidate
    raise ValueError("no valid representative bitstring")


def evaluate_candidate(benchmark, backend, *, name: str, circuit: cirq.Circuit, layout: dict[int, int], seed: int, shots: int) -> dict[str, Any]:
    validation = validate_compiled_circuit(circuit, backend, benchmark.track)
    equivalence = validate_compiler_equivalence(benchmark, circuit, LayoutSidecar(layout), seed=seed) if validation.valid else None
    noiseless_ok = False
    noisy_success = None
    if validation.valid:
        try:
            result = cirq.CliffordSimulator(seed=seed).run(circuit, repetitions=128)
            noiseless_ok = all(benchmark.expected_output_rule.is_correct("".join(str(int(bit)) for bit in row)) for row in result.measurements["z"])
        except Exception:
            noiseless_ok = False
        try:
            noisy_success = run_track_b(benchmark, circuit, backend, shots=shots, seed=seed).success_rate
        except Exception:
            noisy_success = None
    return {
        "name": name,
        "measurement_width": len([op for op in circuit.all_operations() if isinstance(op.gate, cirq.MeasurementGate)][-1].qubits) if any(isinstance(op.gate, cirq.MeasurementGate) for op in circuit.all_operations()) else 0,
        "valid_basic": validation.valid,
        "validation_errors": validation.errors,
        "equivalence_valid": equivalence.valid if equivalence is not None else False,
        "equivalence_errors": equivalence.errors if equivalence is not None else ["skipped_basic_invalid"],
        "noiseless_rule_ok": noiseless_ok,
        "noisy_success": noisy_success,
        "cx": sum(1 for op in circuit.all_operations() if len(op.qubits) == 2),
    }


def decision(rows: dict[str, Any]) -> dict[str, Any]:
    illegal_lifts = {
        name: row["same_width_cap_to_illegal_oracle_gap"]
        for name, row in rows.items()
    }
    any_candidate_equivalent = any(
        candidate["equivalence_valid"]
        for row in rows.values()
        for candidate in row["representation_candidates"]
    )
    return {
        "any_compressed_or_zero_cx_candidate_equivalent": any_candidate_equivalent,
        "illegal_oracle_lifts": illegal_lifts,
        "recommendation": "Do not pursue Track B output compression in runtime unless evaluator width/equivalence contract changes." if not any_candidate_equivalent else "Investigate equivalent representation candidate.",
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Track B Output Representation Probe",
        "",
        "This probe tests whether the high Track B readout ceiling can be bypassed by changing output representation instead of placement.",
        "",
        f"- Decision: `{payload['decision']['recommendation']}`",
        "",
        "| Sample | Root success | Legal same-width cap | Illegal oracle cap | Illegal lift | Root CX |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in payload["samples"].items():
        lines.append(
            f"| `{name}` | {row['root_success']:.6f} | {row['same_width_best_readout_cap']:.6f} | {row['illegal_compressed_oracle_cap']:.6f} | {row['same_width_cap_to_illegal_oracle_gap']:.6f} | {row['root_cx']} |"
        )
    lines.extend(["", "## Candidate Legality", "", "| Sample | Candidate | Width | Basic valid | Equivalence | Noiseless rule | Noisy success | Errors |", "|---|---|---:|---:|---:|---:|---:|---|"])
    for name, row in payload["samples"].items():
        for candidate in row["representation_candidates"]:
            errors = "; ".join(candidate["validation_errors"] or candidate["equivalence_errors"])
            lines.append(
                f"| `{name}` | `{candidate['name']}` | {candidate['measurement_width']} | {candidate['valid_basic']} | {candidate['equivalence_valid']} | {candidate['noiseless_rule_ok']} | {candidate['noisy_success']} | `{errors}` |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Width-preserving zero-CX valid-output circuits can satisfy the scoring rule but fail official equivalence because they do not reproduce the source distribution under challenges.",
            "- Anchor-bit compressed circuits fail because the compiled final measurement width must match the program width.",
            "- Therefore the large illegal oracle lift is not a legal submission path under the current evaluator.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

