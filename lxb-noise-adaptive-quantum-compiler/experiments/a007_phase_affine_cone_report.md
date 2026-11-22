# A007 Phase-Affine Cone Report

- Sample: track_a_007
- Incumbent: noise_adaptive_l1_balanced
- Incumbent metrics: CX=18, depth=25, ELR=-0.277106863312
- Phase-affine intervals detected: 5
- Candidate additions accepted vs local base: 6
- Diagnostic rows beating incumbent: 0

## Cone Boundary Check

- Top interval: 5:16 q=[0, 1, 2, 3, 4]
- Top interval source CX: 6
- Corrected closure: 0:35 q=[0, 1, 2, 3, 4, 5]
- Closure is phase-affine: False
- Affine rows: ['00001', '00011', '00100', '01000', '10100']
- Affine const: [0, 0, 0, 0, 0]
- Phase terms: [('00010', 1), ('01000', 1), ('01001', 7), ('01010', 7), ('01011', 1)]

## Best Routed Attempt

- Method: hybrid_parity_reuse_cx5
- Source interval: 6:16
- New routed CX: 18
- New routed depth: 25
- New routed ELR: -0.27710686331225304
- Accepted: False
- Reason: metric_guard_failed

## Decision

Diagnostic only: some phase-affine variants improved weaker local bases, but none beat the incumbent CX/ELR guard in the A007 diagnostic.
