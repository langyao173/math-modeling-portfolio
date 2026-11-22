# Final Residual And No-Go Summary

This archive records why recent branches were not merged into the release-candidate root.

## Current RC Baseline

Public seed 7:

- Final: 0.827636718750
- Track A: 0.890136718750
- Track B: 0.640136718750
- invalid: 0

Readout-only public ceiling estimate:

- Track A cap: 0.950693606049
- Track B cap: 0.810149370905
- Final cap: 0.915557547263

This frames the remaining residual under the legal same-width/full-distribution evaluator contract. The separate output-compressed oracle is useful as a diagnostic upper bound, but it is not a legal submission path.

## Gap Ledger

The largest residual public gaps are still:

| sample | readout-only cap | current success | raw gap | weighted final gap | diagnosis |
|---|---:|---:|---:|---:|---|
| track_b_001 | 0.858252 | 0.666992 | 0.191260 | 0.023908 | GHZ physical placement/readout residual |
| track_b_002 | 0.762046 | 0.613281 | 0.148765 | 0.018596 | affine parity placement/readout residual |
| track_a_007 | 0.951883 | 0.804688 | 0.147196 | 0.013800 | H-boundary / reversible-carry semantic residual |
| track_a_008 | 0.942365 | 0.804688 | 0.137677 | 0.012907 | semantic residual plus route-scar already normalized |
| track_a_002 | 0.951884 | 0.871094 | 0.080790 | 0.007574 | small affine/reversible residual |

## Rejected Branches

### Track B placement/readout v2

Artifacts:

- `experiments/track_b_placement_v2_candidates.json`
- `experiments/track_b_placement_v2_report.md`
- `experiments/root_vs_bplacement_v2_seed7.md`

Result:

- B001 best diagnostic raw delta: +0.007812
- B002 best diagnostic raw delta: +0.002197
- Track B diagnostic delta: +0.005005
- Final estimate delta: +0.001251
- Keep gate failed: no sample reached +0.03 raw success, Track B did not reach +0.015, Final did not reach +0.0035.

Decision: not merged.

### Track B noisy selector v3

Artifacts:

- `experiments/track_b_noisy_selector_v3.json`
- `experiments/track_b_noisy_selector_v3.md`

Result:

- B001 selected `current_root`.
- B002 selected `current_root`.
- Track B seed7 delta: 0.
- Final seed7 delta: 0.

Decision: not merged.

### A source-level semantic mining

Artifacts:

- `experiments/tracka_semantic_mining_candidates.json`
- `experiments/tracka_semantic_mining_report.md`
- `experiments/root_vs_tracka_semantic_seed7.md`

Result:

- A002 produced no accepted source variants.
- A007 relative-phase variants tied or failed metric guard.
- A008 relative-phase variants tied or failed metric guard.
- Final estimate delta: 0.

Decision: not merged.

### Contextual row-phase window miner

Artifacts:

- `experiments/contextual_window_candidates.json`
- `experiments/contextual_window_report.md`
- `experiments/contextual_row_phase_window_miner_report.md`

Result:

- A002: 92 contextual candidates, 0 valid.
- A007: 281 contextual candidates, 0 valid.
- A008: 269 contextual candidates, 0 valid.
- Rejection mode: full-circuit final-Z contextual miter failed.
- Final estimate delta: 0.

Decision: not merged.

### Global semantic resynthesis probe

Artifacts:

- `experiments/track_a_global_semantic_resynthesis_probe.json`
- `experiments/track_a_global_semantic_resynthesis_probe.md`

Result:

- A002 was classified as affine linear reversible, but from-scratch affine synthesis routed to 15 CX, weaker than current 8 CX.
- A007 was classified as nonlinear reversible; best relative-phase global attempt tied current 14 CX and failed metric gain.
- A008 was classified as quadratic reversible; physical candidates failed final-Z miter.
- Final estimate delta: 0.

Decision: not merged.

### Legal-cap gap closing branch

Artifacts:

- `experiments/cap_normalized_gap_report.md`
- `experiments/track_b_noisy_selector_v3.md`
- `experiments/track_a_exact_synthesis_hard_search.md`

Result:

- Cap-normalized ranking identified B001, B002, A007, A008, and A002 as the top residuals.
- Track B same-width selector did not improve the root.
- Track A hard search found no valid A002 <=7 CX candidate and no valid A007/A008 <=13 CX candidate.

Decision: not merged.

## Final Interpretation

The recent evidence says the obvious low-risk residuals have been tested:

- Track B semantic compression is already strong; remaining placement/readout search was too thin to merge.
- A008 route-scar was already distilled into the root; contextual mining did not find additional safe replacements.
- A007/A008/A002 global semantic probes did not beat the current root under strict final-Z, validator, and metric guards.
- Legal-cap gap closing did not find a branch meeting the required Final improvement gate.

The release-candidate root is therefore frozen. Further work should be reporting, packaging, and reproducibility rather than new runtime passes.
