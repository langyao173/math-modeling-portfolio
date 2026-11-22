# Final Report Polish Notes

Use the final report as a system description rather than a version diary.

## Recommended Framing

The final solution should emphasize four contributions:

1. Noise-aware candidate bank and Pareto selection.
2. Track B final-Z measurement-channel resynthesis.
3. Track A final-Z row-phase equivalence for relative-phase reversible rewrites.
4. Residual analysis with legality and no-go audits.

## Ceiling Language

Use:

- The legal same-width readout-only ceiling estimate is approximately `0.9156` on the public backend snapshots.
- An output-compressed oracle can reach approximately `0.9630`, but it is not a legal submission path because the evaluator compares the required full final `z` bitstring distribution.
- The current method is a stable strong solution under legal same-width/full-distribution equivalence.
- The remaining residual is primarily from mandatory output width, readout noise, and low-CX synthesis lower bounds.

Avoid:

- presenting the work as a failure to reach a target score;
- implying that illegal compressed-output oracles are candidate submissions;
- using public sample names as runtime logic;
- describing experimental miners as runtime dependencies.

## Evidence To Cite

Core files:

- `experiments/final_multiseed_reproduction.md`
- `experiments/final_package_audit.md`
- `experiments/final_no_go_appendix.md`
- `experiments/upper_bound_decomposition.md`
- `experiments/lower_bound_no_go_report.md`
- `experiments/cap_normalized_gap_report.md`
- `experiments/row_phase_proof_audit_v19.md`

Key values:

- Seven-seed public mean Final: `0.832746233259`.
- Seven-seed mean Track A: `0.893763950893`.
- Seven-seed mean Track B: `0.649693080357`.
- Public invalid count: `0`.
- Overall p95 compile: `13.982924` seconds.
- Overall max compile: `17.451201` seconds.
- Legal same-width Final ceiling: `0.915557547263`.
- Illegal compressed-output oracle: `0.963020204536`.

## Final Positioning

The correct final claim is not that every public residual is eliminated. The correct claim is that the compiler is a deterministic, validator-first, noise-aware, semantics-preserving system whose major legal structural opportunities were implemented and whose remaining residuals were audited with explicit no-go evidence.
