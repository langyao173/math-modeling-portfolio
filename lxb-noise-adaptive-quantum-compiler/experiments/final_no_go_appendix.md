# Final No-Go Appendix

This appendix records branches that were intentionally not merged into the release-candidate compiler. These are audit evidence, not runtime dependencies.

## B Placement / Readout V2

Artifacts:

- `experiments/track_b_placement_v2_candidates.json`
- `experiments/track_b_placement_v2_report.md`
- `experiments/root_vs_bplacement_v2_seed7.md`

Result:

- B001 best diagnostic raw delta: `+0.007812`.
- B002 best diagnostic raw delta: `+0.002197`.
- Track B diagnostic delta: `+0.005005`.
- Final estimate delta: `+0.001251`.

Decision: not merged. It did not meet the gate of B001/B002 raw success `+0.03`, Track B seed7 `+0.015`, or Final seed7 `+0.0035`.

## B Noisy Selector V3

Artifacts:

- `experiments/track_b_noisy_selector_v3.json`
- `experiments/track_b_noisy_selector_v3.md`

Result:

- B001 selected `current_root`.
- B002 selected `current_root`.
- Track B seed7 delta: `0.000000`.
- Final seed7 delta: `0.000000`.

Decision: not merged. Same-width low-shot/readout-sensitive selection did not improve the frozen root.

## A Source-Level Semantic Mining

Artifacts:

- `experiments/tracka_semantic_mining_candidates.json`
- `experiments/tracka_semantic_mining_report.md`
- `experiments/root_vs_tracka_semantic_seed7.md`

Result:

- A002 produced no accepted source variants.
- A007 relative-phase variants tied or failed metric guard.
- A008 relative-phase variants tied or failed metric guard.
- Final estimate delta: `0`.

Decision: not merged.

## Contextual Row-Phase Window Miner

Artifacts:

- `experiments/contextual_window_candidates.json`
- `experiments/contextual_window_report.md`
- `experiments/contextual_row_phase_window_miner_report.md`

Result:

- A002: 92 contextual candidates, 0 valid.
- A007: 281 contextual candidates, 0 valid.
- A008: 269 contextual candidates, 0 valid.
- Dominant rejection: full-circuit final-Z contextual miter failed.

Decision: not merged.

## Global Semantic Resynthesis Probe

Artifacts:

- `experiments/track_a_global_semantic_resynthesis_probe.json`
- `experiments/track_a_global_semantic_resynthesis_probe.md`

Result:

- A002 was classified as affine linear reversible, but from-scratch affine synthesis routed to 15 CX, weaker than current 8 CX.
- A007 was classified as nonlinear reversible; best relative-phase global attempt tied current 14 CX and failed metric gain.
- A008 was classified as quadratic reversible; physical candidates failed final-Z miter.
- Final estimate delta: `0`.

Decision: not merged.

## Legal-Cap Gap Closing

Artifacts:

- `experiments/cap_normalized_gap_report.json`
- `experiments/cap_normalized_gap_report.md`
- `experiments/track_b_noisy_selector_v3.json`
- `experiments/track_b_noisy_selector_v3.md`
- `experiments/track_a_exact_synthesis_hard_search.json`
- `experiments/track_a_exact_synthesis_hard_search.md`

Result:

- Legal same-width Final ceiling estimate: `0.915557547263`.
- Top normalized residuals: B001, B002, A007, A008, A002.
- Track B same-width noisy selector did not improve root.
- Track A hard search found no valid A002 `<=7` CX candidate and no valid A007/A008 `<=13` CX candidate.
- Tests still passed: `93 passed`.

Decision: not merged. Neither Track B nor Track A branch improved Final by the required `+0.006`.

## Final Interpretation

The rejected branches consistently point to the same residual structure:

- Track B is limited primarily by mandatory full-width readout noise and the remaining physical implementation cost under same-width distribution equivalence.
- Track A residuals are concentrated in low-CX nonlinear reversible/row-phase synthesis, especially A007/A008/A002.
- No branch found a valid, lower-noise replacement that cleared validator, equivalence, noiseless semantics, and metric gates.

The current root remains the release candidate.
