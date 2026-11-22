# Current Root Gap Ledger

- Variant: `current_promoted_root`
- Seed: `7`
- Current Final: `0.827636718750`
- Current Track A: `0.890136718750`
- Current Track B: `0.640136718750`
- Invalid count: `0`
- Readout-only Final cap: `0.915557547263`
- Remaining weighted Final headroom to cap: `0.087920828513`

## Largest Residuals

| Rank | Sample | Track | Cap | Current | Raw Gap | Weighted Final Gap | CX | Depth | 2Q Cost | Readout Cost | Next action |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | track_b_001 | track_b | 0.858252 | 0.666992 | 0.191260 | 0.023908 | 21 | 16 | 0.294296 | 0.211167 | Track B: only low-risk physical placement/readout assignment; do not rewrite semantics |
| 2 | track_b_002 | track_b | 0.762046 | 0.613281 | 0.148765 | 0.018596 | 19 | 6 | 0.246330 | 0.290018 | Track B: only low-risk physical placement/readout assignment; do not rewrite semantics |
| 3 | track_a_007 | track_a | 0.951883 | 0.804688 | 0.147196 | 0.013800 | 14 | 22 | 0.148685 | 0.060430 | Track A: semantic row-phase/Peres/carry/H-boundary mining, not generic router |
| 4 | track_a_008 | track_a | 0.942365 | 0.804688 | 0.137677 | 0.012907 | 14 | 28 | 0.149509 | 0.074529 | Track A: semantic row-phase/Peres/carry/H-boundary mining, not generic router |
| 5 | track_a_002 | track_a | 0.951883 | 0.871094 | 0.080790 | 0.007574 | 8 | 8 | 0.085997 | 0.060430 | Track A: inspect semantic rewrite or lower-noise exact route |
| 6 | track_a_004 | track_a | 0.951883 | 0.914062 | 0.037821 | 0.003546 | 5 | 13 | 0.035497 | 0.050323 | Readout placement/tail cleanup only |
| 7 | track_a_005 | track_a | 0.951883 | 0.925781 | 0.026102 | 0.002447 | 5 | 12 | 0.035496 | 0.054473 | Readout placement/tail cleanup only |
| 8 | track_a_003 | track_a | 0.951883 | 0.929688 | 0.022196 | 0.002081 | 3 | 11 | 0.020862 | 0.050323 | Readout placement/tail cleanup only |
| 9 | track_a_006 | track_a | 0.951883 | 0.929688 | 0.022196 | 0.002081 | 3 | 11 | 0.020862 | 0.050323 | Readout placement/tail cleanup only |
| 10 | track_a_001 | track_a | 0.951883 | 0.941406 | 0.010477 | 0.000982 | 2 | 7 | 0.014090 | 0.050323 | Readout placement/tail cleanup only |

## Per-Sample Physical Ledger

### track_b_001

- Family: `ghz`, width `16`.
- Success/cap/gap: `0.666992` / `0.858252` / `0.191260`.
- CX/depth/ELR: `21` / `16` / `-0.506463`.
- Log-cost split: 2Q `0.294296`, 1Q `0.001001`, readout `0.211167`.
- Used edges: `21` unique, worst `{'edge': [11, 12], 'count': 1, 'error': 0.016999999999999998}`, hottest `[{'edge': [5, 14], 'count': 1, 'error': 0.016999999999999998}, {'edge': [11, 12], 'count': 1, 'error': 0.016999999999999998}, {'edge': [14, 15], 'count': 1, 'error': 0.016999999999999998}, {'edge': [4, 12], 'count': 1, 'error': 0.015799999999999998}, {'edge': [13, 21], 'count': 1, 'error': 0.015799999999999998}]`.
- Measurement qubits: `[10, 13, 9, 23, 1, 30, 18, 2, 5, 4, 20, 29, 15, 3, 27, 6]`, readout sum `0.209700`, max readout `0.020000`.
- Selected candidate: `ghz_leaf_steiner_tree_relabel`; candidate families: `ghz_tree`.
- Semantic markers: `final-Z GHZ tree/leaf-tree measurement-channel`.
- Recommendation: Track B: only low-risk physical placement/readout assignment; do not rewrite semantics.

### track_b_002

- Family: `parity`, width `24`.
- Success/cap/gap: `0.613281` / `0.762046` / `0.148765`.
- CX/depth/ELR: `19` / `6` / `-0.552356`.
- Log-cost split: 2Q `0.246330`, 1Q `0.016008`, readout `0.290018`.
- Used edges: `19` unique, worst `{'edge': [14, 15], 'count': 1, 'error': 0.016999999999999998}`, hottest `[{'edge': [14, 15], 'count': 1, 'error': 0.016999999999999998}, {'edge': [4, 12], 'count': 1, 'error': 0.015799999999999998}, {'edge': [16, 24], 'count': 1, 'error': 0.015799999999999998}, {'edge': [22, 30], 'count': 1, 'error': 0.015799999999999998}, {'edge': [3, 4], 'count': 1, 'error': 0.014599999999999998}]`.
- Measurement qubits: `[1, 10, 0, 2, 4, 3, 12, 5, 14, 6, 15, 13, 28, 20, 27, 29, 17, 9, 24, 25, 30, 21, 22, 31]`, readout sum `0.288100`, max readout `0.020000`.
- Selected candidate: `not_matched_after_tail_or_copyout`; candidate families: `legacy_track_b, measurement_channel`.
- Semantic markers: `affine parity measurement-channel physical placement/copyout`.
- Recommendation: Track B: only low-risk physical placement/readout assignment; do not rewrite semantics.

### track_a_007

- Family: `adder`, width `6`.
- Success/cap/gap: `0.804688` / `0.951883` / `0.147196`.
- CX/depth/ELR: `14` / `22` / `-0.223123`.
- Log-cost split: 2Q `0.148685`, 1Q `0.014007`, readout `0.060430`.
- Used edges: `6` unique, worst `{'edge': [5, 6], 'count': 1, 'error': 0.020999999999999998}`, hottest `[{'edge': [0, 4], 'count': 4, 'error': 0.00675}, {'edge': [5, 9], 'count': 3, 'error': 0.0174}, {'edge': [4, 5], 'count': 3, 'error': 0.00729}, {'edge': [0, 1], 'count': 2, 'error': 0.008369999999999999}, {'edge': [5, 6], 'count': 1, 'error': 0.020999999999999998}]`.
- Measurement qubits: `[9, 4, 5, 0, 6, 1]`, readout sum `0.060100`, max readout `0.014000`.
- Selected candidate: `relative_phase_toffoli_0_1_3_2_3_5_noise_adaptive_l0_balanced`; candidate families: `cirq_cqc, noise_adaptive_router, row_phase_relative_toffoli`.
- Semantic markers: `final-Z row-phase relative-Toffoli rewrite`.
- Recommendation: Track A: semantic row-phase/Peres/carry/H-boundary mining, not generic router.

### track_a_008

- Family: `structured_random`, width `7`.
- Success/cap/gap: `0.804688` / `0.942365` / `0.137677`.
- CX/depth/ELR: `14` / `28` / `-0.243048`.
- Log-cost split: 2Q `0.149509`, 1Q `0.019010`, readout `0.074529`.
- Used edges: `7` unique, worst `{'edge': [5, 6], 'count': 1, 'error': 0.020999999999999998}`, hottest `[{'edge': [4, 5], 'count': 5, 'error': 0.00729}, {'edge': [1, 2], 'count': 2, 'error': 0.0162}, {'edge': [1, 5], 'count': 2, 'error': 0.00891}, {'edge': [0, 1], 'count': 2, 'error': 0.008369999999999999}, {'edge': [5, 6], 'count': 1, 'error': 0.020999999999999998}]`.
- Measurement qubits: `[0, 2, 9, 1, 5, 4, 6]`, readout sum `0.074100`, max readout `0.014000`.
- Selected candidate: `relative_phase_toffoli_3_4_5_0_4_5_noise_adaptive_l1_balanced`; candidate families: `cirq_cqc, noise_adaptive_router, row_phase_relative_toffoli`.
- Semantic markers: `final-Z row-phase relative-Toffoli rewrite, route-scar compute/uncompute cancellation`.
- Recommendation: Track A: semantic row-phase/Peres/carry/H-boundary mining, not generic router.

### track_a_002

- Family: `hs`, width `6`.
- Success/cap/gap: `0.871094` / `0.951883` / `0.080790`.
- CX/depth/ELR: `8` / `8` / `-0.150430`.
- Log-cost split: 2Q `0.085997`, 1Q `0.004002`, readout `0.060430`.
- Used edges: `5` unique, worst `{'edge': [5, 9], 'count': 1, 'error': 0.0174}`, hottest `[{'edge': [9, 13], 'count': 2, 'error': 0.015}, {'edge': [0, 1], 'count': 2, 'error': 0.008369999999999999}, {'edge': [4, 5], 'count': 2, 'error': 0.00729}, {'edge': [5, 9], 'count': 1, 'error': 0.0174}, {'edge': [0, 4], 'count': 1, 'error': 0.00675}]`.
- Measurement qubits: `[1, 0, 4, 5, 9, 13]`, readout sum `0.060100`, max readout `0.014000`.
- Selected candidate: `noise_adaptive_l0_balanced_relabel`; candidate families: `cirq_cqc, noise_adaptive_router, physical_relabel`.
- Semantic markers: `none`.
- Recommendation: Track A: inspect semantic rewrite or lower-noise exact route.

### track_a_004

- Family: `fredkin`, width `6`.
- Success/cap/gap: `0.914062` / `0.951883` / `0.037821`.
- CX/depth/ELR: `5` / `13` / `-0.095825`.
- Log-cost split: 2Q `0.035497`, 1Q `0.010005`, readout `0.050323`.
- Used edges: `2` unique, worst `{'edge': [0, 1], 'count': 1, 'error': 0.008369999999999999}`, hottest `[{'edge': [0, 4], 'count': 4, 'error': 0.00675}, {'edge': [0, 1], 'count': 1, 'error': 0.008369999999999999}]`.
- Measurement qubits: `[1, 0, 4, 7, 14, 5]`, readout sum `0.050100`, max readout `0.011000`.
- Selected candidate: `relative_phase_toffoli_0_2_1_noise_adaptive_l0_balanced`; candidate families: `cirq_cqc, noise_adaptive_router, row_phase_relative_toffoli`.
- Semantic markers: `final-Z row-phase relative-Toffoli rewrite`.
- Recommendation: Readout placement/tail cleanup only.

### track_a_005

- Family: `peres`, width `6`.
- Success/cap/gap: `0.925781` / `0.951883` / `0.026102`.
- CX/depth/ELR: `5` / `12` / `-0.099974`.
- Log-cost split: 2Q `0.035496`, 1Q `0.010005`, readout `0.054473`.
- Used edges: `2` unique, worst `{'edge': [4, 5], 'count': 3, 'error': 0.00729}`, hottest `[{'edge': [4, 5], 'count': 3, 'error': 0.00729}, {'edge': [0, 4], 'count': 2, 'error': 0.00675}]`.
- Measurement qubits: `[0, 5, 4, 2, 7, 14]`, readout sum `0.054200`, max readout `0.014000`.
- Selected candidate: `relative_phase_toffoli_0_1_2_noise_adaptive_l5_balanced`; candidate families: `cirq_cqc, noise_adaptive_router, row_phase_relative_toffoli`.
- Semantic markers: `final-Z row-phase relative-Toffoli rewrite`.
- Recommendation: Readout placement/tail cleanup only.

### track_a_003

- Family: `toffoli`, width `6`.
- Success/cap/gap: `0.929688` / `0.951883` / `0.022196`.
- CX/depth/ELR: `3` / `11` / `-0.083192`.
- Log-cost split: 2Q `0.020862`, 1Q `0.012006`, readout `0.050323`.
- Used edges: `2` unique, worst `{'edge': [4, 5], 'count': 1, 'error': 0.00729}`, hottest `[{'edge': [0, 4], 'count': 2, 'error': 0.00675}, {'edge': [4, 5], 'count': 1, 'error': 0.00729}]`.
- Measurement qubits: `[5, 0, 4, 7, 14, 1]`, readout sum `0.050100`, max readout `0.011000`.
- Selected candidate: `relative_phase_toffoli_0_1_2_noise_adaptive_l0_balanced`; candidate families: `cirq_cqc, noise_adaptive_router, row_phase_relative_toffoli`.
- Semantic markers: `final-Z row-phase relative-Toffoli rewrite`.
- Recommendation: Readout placement/tail cleanup only.

### track_a_006

- Family: `or`, width `6`.
- Success/cap/gap: `0.929688` / `0.951883` / `0.022196`.
- CX/depth/ELR: `3` / `11` / `-0.085193`.
- Log-cost split: 2Q `0.020862`, 1Q `0.014007`, readout `0.050323`.
- Used edges: `2` unique, worst `{'edge': [4, 5], 'count': 1, 'error': 0.00729}`, hottest `[{'edge': [0, 4], 'count': 2, 'error': 0.00675}, {'edge': [4, 5], 'count': 1, 'error': 0.00729}]`.
- Measurement qubits: `[5, 0, 4, 7, 14, 1]`, readout sum `0.050100`, max readout `0.011000`.
- Selected candidate: `relative_phase_toffoli_0_1_2_noise_adaptive_l0_balanced`; candidate families: `cirq_cqc, noise_adaptive_router, row_phase_relative_toffoli`.
- Semantic markers: `final-Z row-phase relative-Toffoli rewrite`.
- Recommendation: Readout placement/tail cleanup only.

### track_a_001

- Family: `bv`, width `6`.
- Success/cap/gap: `0.941406` / `0.951883` / `0.010477`.
- CX/depth/ELR: `2` / `7` / `-0.078420`.
- Log-cost split: 2Q `0.014090`, 1Q `0.014007`, readout `0.050323`.
- Used edges: `2` unique, worst `{'edge': [4, 5], 'count': 1, 'error': 0.00729}`, hottest `[{'edge': [4, 5], 'count': 1, 'error': 0.00729}, {'edge': [0, 4], 'count': 1, 'error': 0.00675}]`.
- Measurement qubits: `[0, 7, 5, 14, 1, 4]`, readout sum `0.050100`, max readout `0.011000`.
- Selected candidate: `noise_adaptive_l0_balanced`; candidate families: `cirq_cqc, noise_adaptive_router`.
- Semantic markers: `none`.
- Recommendation: Readout placement/tail cleanup only.

## Interpretation

- The ledger ranks residuals by final-score weight, not by raw success alone.
- Track B residuals are now mainly physical/readout/edge placement residuals because the final-Z measurement-channel semantics are already compressed.
- Track A residuals that still have high CX and high headroom should be handled by semantic row-phase/Peres/carry-style rewrites, not generic routing passes.
- The readout-only cap is a ceiling for a zero-gate legal implementation, not a reachable promise for circuits that still require noisy CX gates.
