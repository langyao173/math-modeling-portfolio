# Track B Output Representation Probe

This probe tests whether the high Track B readout ceiling can be bypassed by changing output representation instead of placement.

- Decision: `Do not pursue Track B output compression in runtime unless evaluator width/equivalence contract changes.`

| Sample | Root success | Legal same-width cap | Illegal oracle cap | Illegal lift | Root CX |
|---|---:|---:|---:|---:|---:|
| `track_b_001` | 0.661621 | 0.858252 | 1.000000 | 0.141748 | 21 |
| `track_b_002` | 0.638184 | 0.762046 | 1.000000 | 0.237954 | 19 |

## Candidate Legality

| Sample | Candidate | Width | Basic valid | Equivalence | Noiseless rule | Noisy success | Errors |
|---|---|---:|---:|---:|---:|---:|---|
| `track_b_001` | `zero_cx_best_same_width_valid_string` | 16 | True | False | True | 0.858642578125 | `compiler equivalence check failed (challenge=0, total_variation=1.000, threshold=0.350)` |
| `track_b_001` | `illegal_anchor_measurement_width` | 1 | True | False | False | 0.0 | `compiled final measurement width must match program circuit width (1 != 16)` |
| `track_b_002` | `zero_cx_best_same_width_valid_string` | 24 | True | False | True | 0.7587890625 | `compiler equivalence check failed (challenge=0, total_variation=1.000, threshold=0.350)` |
| `track_b_002` | `illegal_anchor_measurement_width` | 1 | True | False | False | None | `compiled final measurement width must match program circuit width (1 != 24)` |

## Interpretation

- Width-preserving zero-CX valid-output circuits can satisfy the scoring rule but fail official equivalence because they do not reproduce the source distribution under challenges.
- Anchor-bit compressed circuits fail because the compiled final measurement width must match the program width.
- Therefore the large illegal oracle lift is not a legal submission path under the current evaluator.
