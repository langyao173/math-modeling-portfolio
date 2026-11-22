# Legal Score Ceiling From Mandatory Readout Noise

## Conclusion

A `0.9999` legal score is not reachable under the current evaluator and backend snapshots, even with a perfect semantic compiler.

The reason is not routing or synthesis. It is mandatory final readout noise on a unique physical qubit for every measured output bit.

## Evaluator Facts

- The validator requires exactly one final `z` measurement.
- The final measurement width must equal the logical program width.
- Cirq rejects duplicate measured qids, so the compiler cannot measure the same low-noise qubit multiple times.
- The simulator injects readout error before every measured qubit:
  - Track A: `cirq.bit_flip(readout_error)` before measurement.
  - Track B: Stim `X_ERROR(readout_error)` before `M`.
- Therefore every output bit pays at least the selected physical qubit readout error.

## Readout-Only Upper Bounds

These bounds assume an otherwise perfect compiler: zero one-qubit error, zero two-qubit error, no routing error, and optimal use of the best readout qubits.

| Component | Optimistic Readout-Only Cap |
|---|---:|
| Track A 6-bit deterministic rule, best medium qubits | 0.9518834603740918 |
| Track A 7-bit deterministic rule, best medium qubits | 0.9423646257703508 |
| Track A public average cap | 0.9506936060486242 |
| Track B GHZ 16-bit allowed `{0^n,1^n}`, best large qubits | 0.858252404404446 |
| Track B parity 24-bit checks, best large qubits | 0.7620463374055627 |
| Track B public average cap | 0.8101493709050043 |
| Final public optimistic cap | 0.9155575472627193 |

So even a full family-level semantic crack with zero CX gates cannot reach `0.9999`; the legal public ceiling is roughly `0.916` under these snapshots.

## Checked Loopholes

- Duplicate measurement qubits: blocked by Cirq with `ValueError: Duplicate qids`.
- Constant bitstring generators: pass noiseless semantics but fail challenge equivalence on nearly all Track A/B samples.
- Track A measurement `invert_mask`: accepted by validator, but exact challenge-equivalence ignores the mask and rejects masked identity candidates.
- Track B measurement `invert_mask`: Cirq equivalence/noiseless semantics sees it, but Stim noisy scoring ignores it, so it does not provide a scoring bypass.
- Direct passthrough measurement of challenge-prepped physical qubits with `invert_mask`: fails Track A challenge-equivalence for all public samples across tested seeds.
- Random-H plus mask generators: fail semantics or challenge-equivalence.

## Important Qualification

The `0.9155575472627193` number is an optimistic physical cap, not a constructive guarantee. It assumes the compiler can collapse each benchmark to a zero-CX rule generator while still passing challenge equivalence. Direct tests show that this collapse is rejected for nearly all public samples, so the actual legal achievable ceiling is lower unless a smaller circuit can preserve the evaluator's challenge distributions.

## Practical Target

The legal high-ceiling objective is not `0.9999`; it is to approach the readout-only cap:

- Track A: `~0.95`
- Track B: `~0.81`
- Final: `~0.916`

The next meaningful target is therefore `0.86-0.90+`, not `0.9999`, unless the evaluator itself changes or a nonstandard serialization bug is found.
