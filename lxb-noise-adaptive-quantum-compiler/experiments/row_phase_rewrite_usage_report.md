# Row-Phase Rewrite Usage Report V19

This report lists every current V19 relative-phase rewrite audited by the full-circuit final-Z row-phase miter. It is explanatory only; no additional mined rules are promoted.

| Sample | Matched structure | Replacements | CX delta | Row-phase proof | Residual | Random delta | Hidden delta |
|---|---|---:|---:|---|---:|---:|---:|
| track_a_003 | `relative_phase_toffoli_0_1_2` | 1 | 6->3 | True/True | 3.252e-17 | 2.776e-17 | 6.939e-18 |
| track_a_004 | `relative_phase_toffoli_0_2_1` | 1 | 8->5 | True/True | 3.252e-17 | 2.776e-17 | 1.057e-33 |
| track_a_005 | `relative_phase_toffoli_0_1_2` | 1 | 7->4 | True/True | 3.252e-17 | 2.776e-17 | 2.776e-17 |
| track_a_006 | `relative_phase_toffoli_0_1_2` | 1 | 6->3 | True/True | 3.252e-17 | 2.776e-17 | 2.776e-17 |
| track_a_007 | `relative_phase_toffoli_0_1_3_2_3_5` | 2 | 15->9 | True/True | 6.661e-16 | 1.249e-16 | 8.882e-16 |
| track_a_007 | `relative_phase_toffoli_0_1_3` | 1 | 15->12 | True/True | 6.661e-16 | 9.714e-17 | 7.772e-16 |
| track_a_007 | `relative_phase_toffoli_2_3_5` | 1 | 15->12 | True/True | 4.441e-16 | 6.939e-17 | 4.441e-16 |
| track_a_008 | `relative_phase_toffoli_3_4_5_0_4_5` | 2 | 16->10 | True/True | 3.418e-17 | 1.388e-17 | 6.939e-18 |
| track_a_008 | `relative_phase_toffoli_0_4_5` | 1 | 16->13 | True/True | 3.252e-17 | 2.082e-17 | 6.939e-18 |
| track_a_008 | `relative_phase_toffoli_3_4_5` | 1 | 16->13 | True/True | 6.670e-17 | 2.776e-17 | 6.939e-18 |

## Boundary Checks

- Relative phase followed by `H`: rejected.
- Non-final measurement: rejected.
- Non-`z` measurement: rejected.
- Row-phase false case: rejected.
- Random single-qubit false case: rejected.
