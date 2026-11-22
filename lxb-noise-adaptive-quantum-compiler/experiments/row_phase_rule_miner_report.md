# Row-Phase Rule Miner Report

- Widths: [3, 4]
- Max depth: 4
- Max enumerated per width: 12000
- Verified row-phase rules: 97
- CX-reducing rules: 58

## Width Summaries

| Width | Seeded | Mined | Deduped | CX-reducing |
|---:|---:|---:|---:|---:|
| 3 | 21 | 28 | 49 | 34 |
| 4 | 48 | 0 | 48 | 24 |

## Rule Families

- h_cx_h_reversal: 18
- hh_cancel: 7
- mined_row_phase: 28
- relative_phase_toffoli: 30
- s_sdg_cancel: 7
- t_tdg_cancel: 7

## Decision

Rules are exported for offline inspection only. Runtime promotion still requires structural pattern matching, exact full-circuit row-phase miter, validator/equivalence/noiseless checks, and a metric guard.
