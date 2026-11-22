# Track A Exact Synthesis Hard Search

Development-only hard search for A002/A007/A008. Root compiler was not modified.

- Decision: `Do not merge; no A002 <=7 or A007/A008 <=13 valid candidate found.`

| Sample | Root CX | Best CX | Root success | Best success | Target hit | Best candidate |
|---|---:|---:|---:|---:|---:|---|
| `track_a_002` | 8 | 8 | 0.875488 | 0.875488 | False | `noise_adaptive_l0_balanced_relabel` |
| `track_a_007` | 14 | 14 | 0.822754 | 0.822754 | False | `relative_phase_toffoli_0_1_3_2_3_5_noise_adaptive_l0_balanced` |
| `track_a_008` | 14 | 14 | 0.814453 | 0.814453 | False | `current_root` |

## track_a_002

- Truth: affine `True`, deterministic `True`.
- Routed candidates: `19`, valid `4`.
- Rejections: `{'physical_final_z_miter_failed': 15}`.

### Source Exact Candidates

| Candidate | Kind | Found | Source CX | Max CX |
|---|---|---:|---:|---:|
| `a002_affine_mitm_le_6` | `affine_cnot_mitm` | False | None | 6 |
| `a002_affine_mitm_le_7` | `affine_cnot_mitm` | False | None | 7 |
| `a002_affine_mitm_le_8` | `affine_cnot_mitm` | True | 8 | 8 |

### Top Routed

| Candidate | Valid | CX | Success | Depth | ELR | Reason |
|---|---:|---:|---:|---:|---:|---|
| `noise_adaptive_l0_balanced_relabel` | True | 8 | 0.875488 | 8 | -0.150430 | `` |
| `noise_adaptive_l0_balanced` | True | 8 | 0.872559 | 8 | -0.152064 | `` |
| `noise_adaptive_l1_balanced` | True | 8 | 0.803223 | 8 | -0.245098 | `` |
| `cirq_cqc` | True | 8 | 0.750488 | 8 | -0.337676 | `` |
| `reliability_greedy` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
| `paper_greedy_vertex` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
| `paper_greedy_edge` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
| `noise_adaptive_l7_balanced` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
| `noise_adaptive_l6_noise` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
| `noise_adaptive_l6_balanced` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
| `noise_adaptive_l5_noise` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
| `noise_adaptive_l5_balanced` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |

## track_a_007

- Truth: affine `False`, deterministic `True`.
- Routed candidates: `108`, valid `9`.
- Rejections: `{'physical_final_z_miter_failed': 99}`.

### Source Exact Candidates

| Candidate | Kind | Found | Source CX | Max CX |
|---|---|---:|---:|---:|
| `relative_phase_toffoli_0_1_3_2_3_5` | `row_phase_relative_toffoli` | True | 9 | None |
| `relative_phase_toffoli_0_1_3` | `row_phase_relative_toffoli` | True | 12 | None |
| `relative_phase_toffoli_2_3_5` | `row_phase_relative_toffoli` | True | 12 | None |

### Top Routed

| Candidate | Valid | CX | Success | Depth | ELR | Reason |
|---|---:|---:|---:|---:|---:|---|
| `relative_phase_toffoli_0_1_3_2_3_5_noise_adaptive_l0_balanced` | True | 14 | 0.822754 | 22 | -0.223123 | `` |
| `relative_phase_toffoli_0_1_3_2_3_5_noise_adaptive_l1_balanced` | True | 17 | 0.796875 | 22 | -0.254358 | `` |
| `relative_phase_toffoli_0_1_3:phase_affine_hybrid_parity_reuse_cx5_15_25_l0` | True | 18 | 0.804688 | 26 | -0.245700 | `` |
| `relative_phase_toffoli_0_1_3:noise_adaptive_l3_balanced` | True | 19 | 0.788086 | 26 | -0.271010 | `` |
| `relative_phase_toffoli_0_1_3:noise_adaptive_l1_noise` | True | 20 | 0.728516 | 26 | -0.377224 | `` |
| `relative_phase_toffoli_0_1_3:cirq_cqc_l3_r12` | True | 24 | 0.741211 | 28 | -0.345835 | `` |
| `relative_phase_toffoli_0_1_3:cirq_cqc_l1_r12` | True | 24 | 0.702637 | 33 | -0.425351 | `` |
| `relative_phase_toffoli_0_1_3:noise_adaptive_l4_hop` | True | 25 | 0.756836 | 33 | -0.336752 | `` |
| `relative_phase_toffoli_0_1_3:noise_adaptive_l4_noise` | True | 26 | 0.761719 | 31 | -0.326582 | `` |
| `relative_phase_toffoli_2_3_5:noise_adaptive_l6_noise` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
| `relative_phase_toffoli_2_3_5:noise_adaptive_l6_balanced` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
| `relative_phase_toffoli_2_3_5:noise_adaptive_l5_noise` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |

## track_a_008

- Truth: affine `False`, deterministic `True`.
- Routed candidates: `102`, valid `3`.
- Rejections: `{'physical_final_z_miter_failed': 99}`.

### Source Exact Candidates

| Candidate | Kind | Found | Source CX | Max CX |
|---|---|---:|---:|---:|
| `relative_phase_toffoli_3_4_5_0_4_5` | `row_phase_relative_toffoli` | True | 10 | None |
| `relative_phase_toffoli_0_4_5` | `row_phase_relative_toffoli` | True | 13 | None |
| `relative_phase_toffoli_3_4_5` | `row_phase_relative_toffoli` | True | 13 | None |

### Top Routed

| Candidate | Valid | CX | Success | Depth | ELR | Reason |
|---|---:|---:|---:|---:|---:|---|
| `relative_phase_toffoli_0_4_5:noise_adaptive_l0_conservative` | True | 19 | 0.786621 | 28 | -0.280693 | `` |
| `relative_phase_toffoli_0_4_5:cirq_cqc_l0_r12` | True | 19 | 0.786621 | 28 | -0.280693 | `` |
| `relative_phase_toffoli_3_4_5:cirq_cqc_l0_r12` | True | 25 | 0.760254 | 31 | -0.331125 | `` |
| `relative_phase_toffoli_3_4_5_0_4_5_noise_adaptive_l4_balanced` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
| `relative_phase_toffoli_3_4_5_0_4_5_noise_adaptive_l3_noise` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
| `relative_phase_toffoli_3_4_5_0_4_5_noise_adaptive_l3_balanced` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
| `relative_phase_toffoli_3_4_5_0_4_5_noise_adaptive_l2_noise` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
| `relative_phase_toffoli_3_4_5_0_4_5_noise_adaptive_l2_balanced` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
| `relative_phase_toffoli_3_4_5_0_4_5_noise_adaptive_l1_noise` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
| `relative_phase_toffoli_3_4_5_0_4_5_noise_adaptive_l1_balanced` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
| `relative_phase_toffoli_3_4_5_0_4_5_noise_adaptive_l0_balanced` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
| `relative_phase_toffoli_3_4_5_0_4_5:reliability_greedy` | False | 999 | -1.000000 | None | 0.000000 | `physical_final_z_miter_failed` |
