# V21 Affine Collapse Diagnostic

## Summary

V21 tested two ceiling-attack ideas:

1. Exact affine final-Z measurement collapse for small Track A circuits.
2. Readout-heavy physical relabeling for already minimal CNOT networks.

The pass is opt-in only through `scripts/variant_compiler.py --variant v21_affine_collapse`; it is not promoted to the root compiler.

## Findings

- Track A 001 and 002 are affine-measurement under exact challenge-distribution tests.
- Track A 003-008 reject the affine model at weight-2/3 challenge patterns, confirming that the nonlinear carry/Toffoli structure is real.
- A001 is already at the minimal all-to-all affine CNOT count: 2 CX.
- A002 is already at the minimal all-to-all affine CNOT count: 8 CX.
- A readout-heavy relabel found a slightly better A002 physical embedding:
  - ELR: `-0.150430 -> -0.150052`
  - readout: `0.060430 -> 0.056382`
  - CX/depth unchanged: `8 / 8`

## Public Five-Seed Result

| Variant | Mean Final | Track A | Track B | Invalid | p95 Compile | Max Compile |
|---|---:|---:|---:|---:|---:|---:|
| V20 aggressive fast | 0.8236083984375 | 0.88896484375 | 0.6275390625 | 0 | 10.194s | 10.608s |
| V21 affine collapse | 0.8234619140625 | 0.88876953125 | 0.6275390625 | 0 | 10.448s | 11.817s |

The small A002 metric gain did not survive noisy scoring; A002 five-seed mean moved from `0.86015625` to `0.85859375`. V21 is therefore diagnostic only.

## Implication

The affine-collapse route does not unlock a jump toward the readout-only ceiling. A001/A002 are already CNOT-minimal at the affine level; the remaining gap is physical CNOT noise and nonlinear Track A structure, especially A007/A008.
