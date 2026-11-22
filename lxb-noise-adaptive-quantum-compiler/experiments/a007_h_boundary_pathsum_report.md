# A007 H-Boundary Path-Sum Report

- Sample: track_a_007
- Incumbent: noise_adaptive_l1_balanced
- Incumbent metrics: CX=18, depth=25, ELR=-0.277106863312
- H-boundary cones detected: 1
- Candidate additions accepted vs local base: 6
- Diagnostic rows beating incumbent: 0

## Path-Sum Cone

- Cone: 0:35 q=[0, 1, 2, 3, 4, 5]
- Cone CX: 15
- H count / path variables: 4 / 4
- Output input rows: ['000001', '000010', '000111', '000000', '010111', '000000']
- Output path rows: ['0000', '0000', '0000', '0010', '0000', '1000']
- Output const: [0, 1, 1, 0, 1, 0]
- Phase/path terms: 18
- H-free phase-affine cores inside cone: 5

## Best Routed Attempt

- Method: h_shell_hybrid_parity_reuse_cx5
- Interval: 6:16
- New routed CX: 18
- New routed depth: 25
- New routed ELR: -0.27710686331225304
- Accepted: False
- Reason: metric_guard_failed

## Decision

Diagnostic only: H-boundary variants improved weaker local bases, but none beat the incumbent A007 CX/ELR guard.
