# Final Rule-Assumption Audit

Status: development audit only. Root `compiler.py` was not modified.

## Answers

| Assumption | Confirmed answer | Evidence |
|---|---|---|
| Final measurement width must match source program width | Yes. | `validate_compiler_equivalence` rejects width mismatch with "compiled final measurement width must match program circuit width" at `src/noise_adaptive_hackathon/evaluator/equivalence.py:97-105`. |
| Extra ancilla may be measured but evaluator compares only a classical subset | No. | `_compiled_measurement_qubits` requires exactly one measurement operation with key `z`; equivalence then requires its width to equal logical width. See `equivalence.py:142-148` and `equivalence.py:97-105`. |
| Measurement after classical copy/remap is allowed | No classical post-measurement mechanism exists. | `validate_compiled_circuit` rejects classically controlled operations and any non-measurement operation after measurement. See `validate_circuit.py:42` and `validate_circuit.py:50`. |
| Evaluator compares full bitstring distribution | Yes for equivalence; score checks the full measured `z` row against the expected-output rule. | Equivalence builds histograms from all measured `z` bits at `equivalence.py:117-121` and `equivalence.py:213+`; scoring reads `result.measurements["z"]` and passes the full row string to `is_correct` in `cirq_runner.py:21-22` and `stim_runner.py:22`. |
| `layout.json` can encode output classical-bit permutation | No. | `LayoutSidecar` contains only `initial_layout: dict[int, int]`; writer emits only schema version and `initial_layout`. See `equivalence.py:21-22` and `equivalence.py:35-41`. |
| Final measured physical qubit order may define output bit order | Yes, but only as part of the compiled circuit distribution, not as sidecar metadata. | `_compiled_measurement_qubits` returns the measured qubits in the final `z` measurement order at `equivalence.py:142-148`; there is no sidecar output permutation. |

## Practical Consequence

The evaluator contract permits physical qubit selection and final measurement ordering inside the compiled circuit, but it does not permit measuring fewer bits, comparing a subset, or using layout sidecar metadata to remap classical output bits.

Therefore:

- A legal compiler must emit one final `z` measurement whose width matches the logical program width.
- Any output permutation must be implemented by the quantum circuit and pass equivalence; it cannot be declared in `layout.json`.
- Anchor-bit/classical-copy output compression is not a valid submission path under the current evaluator.

## Direct Track B Check

The output-representation probe verified this empirically:

- B001 zero-CX same-width valid string: basic valid and scoring-rule valid, but official equivalence fails with total variation 1.000.
- B001 1-bit anchor: rejected because final measurement width is 1 instead of 16.
- B002 zero-CX same-width valid string: basic valid and scoring-rule valid, but official equivalence fails with total variation 1.000.
- B002 1-bit anchor: rejected because final measurement width is 1 instead of 24.

See `experiments/track_b_output_representation_report.md`.

## Decision

All requested assumptions are effectively negative for output-compression attacks. The public 0.95+ target should be removed unless the evaluator contract changes or a completely legal same-width circuit family with much lower gate noise is found.

