# Final Submission Choice Report

Baseline: `final_c0_conservative`
Recommended submission: `v19_relative_phase_toffoli`

Aggressive gate: public invalids = 0, no additional aggregate holdout invalids versus baseline, p95 compile < 12s, public 5-seed mean delta >= 0.004, and no negative public seed deltas.

Choice score:
`0.35 * public_5seed_mean + 0.20 * worst_seed_final + 0.20 * synthetic_hard_mean + 0.10 * synthetic_affine_trackb - 0.10 * invalid_penalty - 0.05 * compile_time_risk`

| Variant | Score | Public mean | Delta | Worst seed | Synthetic A | Synthetic B | Public invalid | Aggregate invalid | p95 | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v19_relative_phase_toffoli | 0.551657 | 0.819067 | 0.013257 | 0.808472 | 0.763574 | 0.505745 | 0 | 5 | 8.222s | pass |
| final_c0_conservative | 0.540880 | 0.805811 | 0.000000 | 0.793823 | 0.747656 | 0.505501 | 0 | 5 | 8.694s | baseline |

## Public Seed Deltas

- `v19_relative_phase_toffoli`: 0.012085;0.016479;0.014648;0.013916;0.009155
- `final_c0_conservative`: 0.000000;0.000000;0.000000;0.000000;0.000000
