# Lower-Bound And No-Go Report

Status: development evidence archive. Root `compiler.py` was not modified.

## Executive Conclusion

The current public upper-bound attack did not find a legal path to 0.95+. The decisive split is:

- Legal same-width readout ceiling: `0.915557547263`.
- Illegal compressed-output oracle ceiling: `0.963020204536`.

The gap between those numbers is real, but it comes from an output-compression oracle that the evaluator rejects.

## Evidence Chain

### 1. Legal readout ceiling is about 0.9156

From `experiments/upper_bound_decomposition.md`:

| Scenario | Track A | Track B | Final |
|---|---:|---:|---:|
| Current seed7 | 0.890136718750 | 0.640136718750 | 0.827636718750 |
| Best legal same-width readout remap, zero gate noise | 0.950693606049 | 0.810149370905 | 0.915557547263 |
| Current circuit with oracle readout | 0.945495605469 | 0.832031250000 | 0.917129516602 |
| Illegal compressed-output oracle | 0.950693606049 | 1.000000000000 | 0.963020204536 |

Interpretation:

- The legal ceiling already assumes zero gate noise and the best physical readout qubits, while preserving required output width.
- It still lands below 0.92.
- The only 0.95+ number comes from illegal output compression in Track B.

### 2. Track B anchor/copy compression is rejected

From `experiments/track_b_output_representation_report.md`:

| Sample | Candidate | Width | Basic valid | Equivalence | Noiseless rule | Reason |
|---|---|---:|---:|---:|---:|---|
| B001 | zero-CX same-width valid string | 16 | true | false | true | source distribution not reproduced under equivalence challenges |
| B001 | 1-bit anchor measurement | 1 | true | false | false | final measurement width must match program width, 1 != 16 |
| B002 | zero-CX same-width valid string | 24 | true | false | true | source distribution not reproduced under equivalence challenges |
| B002 | 1-bit anchor measurement | 1 | true | false | false | final measurement width must match program width, 1 != 24 |

Interpretation:

- B001/B002 can be made to satisfy the final scoring rule with fewer conceptual output bits only in an oracle model.
- The actual evaluator checks equivalence before noisy scoring, so these shortcuts score 0 if submitted.

### 3. B001/B002 caps are mostly fixed-width readout limits, not missed placement

From `experiments/upper_bound_decomposition.md`:

- B001 current measured cap: `0.809639`; best same-width cap: `0.858252`; lift: `0.048613`.
- B002 current measured cap: `0.748250`; best same-width cap: `0.762046`; lift: `0.013796`.

Interpretation:

- Some readout remap headroom exists, especially B001.
- But even the best same-width readout ceiling is `Track B = 0.810149`, not close to 1.0.
- The remaining large Track B headroom requires illegal output compression or a fundamentally better legal same-width low-noise circuit.

### 4. A007/A008/A002 exact synthesis probes did not break through

From `experiments/track_a_truth_table_anf_report.md`:

| Sample | Classification | Result |
|---|---|---|
| A002 | affine linear reversible | no `<=6 CX` exact CNOT solution found; `8 CX` diagnostic solution exists |
| A007 | nonlinear reversible, carry/comparator-like, nested dependencies | CNOT-only impossible; no `<=11 CX` candidate found |
| A008 | quadratic reversible / controlled-increment-like boundary | CNOT-only impossible; no `<=11 CX` candidate found |

Interpretation:

- A002 is already at the relevant affine lower region.
- A007 requires nonlinear reversible resources; local deletion/window mining is the wrong abstraction.
- A008 has quadratic reversible structure, but this probe did not find a shorter legal exact implementation.

### 5. Earlier failure archive is consistent with this bound

From `experiments/final_residual_no_go_summary.md` and recent probes:

- B placement/readout v2 failed keep gate.
- A source-level semantic mining failed keep gate.
- Contextual row-phase window miner produced no accepted candidates.
- Global semantic resynthesis probe produced no accepted improvement.
- Output representation attack produced only illegal oracle gains.

## Final No-Go Statement

Under the current evaluator:

1. final measurement width must equal source logical width;
2. equivalence is checked before noisy scoring;
3. the layout sidecar encodes only initial qubit placement, not classical output remapping;
4. post-measurement classical copy/remap is unavailable;
5. fewer-bit output compression is rejected.

Therefore the 0.95+ route through Track B anchor/copy output compression is not legal. The current method family can still improve only by finding a genuinely legal same-width lower-noise circuit, especially for B001/B002 physical implementation or A007/A008 nonlinear reversible synthesis. No such candidate was found in this upper-bound branch.

## Recommendation

Remove 0.95+ from the current actionable target list. Keep the current root frozen and use this report as the formal explanation for why the remaining gap is structural rather than a missing small optimization pass.

