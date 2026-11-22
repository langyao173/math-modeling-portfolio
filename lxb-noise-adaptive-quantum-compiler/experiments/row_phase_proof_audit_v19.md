# Row-Phase Proof Audit V19

- Rewrites audited: 10
- Row-phase accepted: 10
- Negative cases: 5
- Negative failures: 0
- Max basis-input probability delta: 1.332e-15
- Max random-state probability delta: 1.249e-16
- Max hidden-prep probability delta: 8.882e-16

## Audited Rewrites

| Sample | Rewrite | Replacements | CX | Row-phase | Residual | Random delta | Hidden delta |
|---|---|---:|---:|---|---:|---:|---:|
| track_a_003 | relative_phase_toffoli_0_1_2 | 1 | 6->3 | True/True | 3.252e-17 | 2.776e-17 | 6.939e-18 |
| track_a_004 | relative_phase_toffoli_0_2_1 | 1 | 8->5 | True/True | 3.252e-17 | 2.776e-17 | 1.057e-33 |
| track_a_005 | relative_phase_toffoli_0_1_2 | 1 | 7->4 | True/True | 3.252e-17 | 2.776e-17 | 2.776e-17 |
| track_a_006 | relative_phase_toffoli_0_1_2 | 1 | 6->3 | True/True | 3.252e-17 | 2.776e-17 | 2.776e-17 |
| track_a_007 | relative_phase_toffoli_0_1_3_2_3_5 | 2 | 15->9 | True/True | 6.661e-16 | 1.249e-16 | 8.882e-16 |
| track_a_007 | relative_phase_toffoli_0_1_3 | 1 | 15->12 | True/True | 6.661e-16 | 9.714e-17 | 7.772e-16 |
| track_a_007 | relative_phase_toffoli_2_3_5 | 1 | 15->12 | True/True | 4.441e-16 | 6.939e-17 | 4.441e-16 |
| track_a_008 | relative_phase_toffoli_3_4_5_0_4_5 | 2 | 16->10 | True/True | 3.418e-17 | 1.388e-17 | 6.939e-18 |
| track_a_008 | relative_phase_toffoli_0_4_5 | 1 | 16->13 | True/True | 3.252e-17 | 2.082e-17 | 6.939e-18 |
| track_a_008 | relative_phase_toffoli_3_4_5 | 1 | 16->13 | True/True | 6.670e-17 | 2.776e-17 | 6.939e-18 |

## Negative Cases

| Case | Expected | Accepted | Passed |
|---|---|---|---|
| relative_phase_followed_by_h | False | False | True |
| non_final_measurement | False | False | True |
| non_z_measurement | False | False | True |
| row_phase_false_case | False | False | True |
| random_single_qubit_false_case | False | False | True |

## Decision

`PASS`: V19 row-phase rewrites preserve final-Z probabilities under the audited conditions.
