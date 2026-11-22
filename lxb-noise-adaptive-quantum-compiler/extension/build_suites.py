"""Build two generalisation suites in the challenge's benchmark format.

S1  official generator, unseen sizes: every Track A family at 5, 8 and 11 program
    qubits (public set is 6-7 qubits), two generator seeds, medium backend (16q).
S2  external circuits from QASMBench (pnnl/QASMBench, small/), measurements stripped,
    re-measured on all qubits with key 'z'. Deterministic outputs use the
    'bitstring' rule; small-support outputs use 'allowed_bitstrings'.
"""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path
import numpy as np
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
from noise_adaptive_hackathon.benchmarks.track_a import generate_track_a_benchmark, TRACK_A_FAMILIES
from noise_adaptive_hackathon.benchmarks.model import CompetitionBenchmark, ExpectedOutputRule
from noise_adaptive_hackathon.benchmarks.io import write_benchmark

HERE = Path(__file__).resolve().parent
REPO = Path(sys.argv[1]); QASM = Path(sys.argv[2]); OUT = Path(sys.argv[3])


def make_root(name):
    root = OUT / name
    if root.exists():
        shutil.rmtree(root)
    (root / "benchmarks" / "track_a" / "public").mkdir(parents=True)
    (root / "benchmarks" / "track_b" / "public").mkdir(parents=True)
    shutil.copytree(REPO / "backends", root / "backends")
    return root


def meta(n, family, backend="medium"):
    return {"family": family, "logical_qubits": n, "clifford_only": False, "split": "public",
            "backend_snapshot_id": f"{backend}_snapshot_a", "backend_size_class": backend}


def s1():
    root = make_root("S1_unseen_sizes")
    for fam in TRACK_A_FAMILIES:
        for n in (5, 8, 11):
            for seed in (101, 202):
                try:
                    b = generate_track_a_benchmark(fam, n, seed, sample_id=f"{fam}_{n}q_s{seed}")
                except ValueError:          # family has a minimum width (adder >= 6)
                    continue
                b = CompetitionBenchmark(track=b.track, name=b.name, family=b.family, circuit=b.circuit,
                                         logical_qubits=b.logical_qubits, expected_output_rule=b.expected_output_rule,
                                         metadata=meta(n, fam))
                write_benchmark(b, root / "benchmarks" / "track_a" / "public" / b.name)


KEEP = (cirq.CXPowGate, cirq.HPowGate, cirq.ZPowGate, cirq.XPowGate, cirq.YPowGate)


def to_basis(c: cirq.Circuit) -> cirq.Circuit:
    ops = [op for op in c.all_operations() if not cirq.is_measurement(op)]
    flat = cirq.decompose(ops, keep=lambda op: isinstance(op.gate, KEEP) and
                          (not isinstance(op.gate, (cirq.CXPowGate, cirq.HPowGate)) or op.gate.exponent == 1))
    qmap = {q: cirq.LineQubit(i) for i, q in enumerate(sorted(c.all_qubits()))}
    return cirq.Circuit(op.transform_qubits(qmap) for op in flat)


def s2():
    root = make_root("S2_qasmbench")
    log = []
    for f in sorted(QASM.glob("*.qasm")):
        try:
            c = to_basis(circuit_from_qasm(f.read_text()))
        except Exception as e:  # unsupported QASM construct
            log.append({"circuit": f.stem, "status": f"parse_failed: {type(e).__name__}"}); continue
        qs = sorted(c.all_qubits()); n = len(qs)
        if n < 2 or n > 11:
            log.append({"circuit": f.stem, "status": f"skipped_n={n}"}); continue
        p = np.abs(cirq.final_state_vector(c, qubit_order=qs, dtype=np.complex128)) ** 2
        support = np.flatnonzero(p > 1e-9)
        bits = [format(i, f"0{n}b") for i in support]
        if p.max() > 0.999:
            rule = ExpectedOutputRule(rule_type="bitstring", bitstring=bits[int(np.argmax(p[support]))])
        elif len(support) <= 2 ** (n - 2) and len(support) <= 16:
            rule = ExpectedOutputRule(rule_type="allowed_bitstrings", allowed_bitstrings=tuple(bits))
        else:
            log.append({"circuit": f.stem, "status": f"skipped_support={len(support)}"}); continue
        c.append(cirq.measure(*qs, key="z"))
        b = CompetitionBenchmark(track="track_a", name=f"qasmbench_{f.stem}", family="qasmbench", circuit=c,
                                 logical_qubits=tuple(qs), expected_output_rule=rule, metadata=meta(n, "qasmbench"))
        write_benchmark(b, root / "benchmarks" / "track_a" / "public" / b.name)
        log.append({"circuit": f.stem, "status": "included", "n": n, "rule": rule.rule_type,
                    "support": int(len(support)), "cx": sum(isinstance(o.gate, cirq.CXPowGate) for o in c.all_operations())})
    (root / "suite_log.json").write_text(json.dumps(log, indent=2))
    for r in log: print(r)


if __name__ == "__main__":
    s1(); s2()
