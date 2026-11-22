# Upper-Bound Score Ceiling Decomposition

This is a development-only upper-bound analysis. It does not modify root `compiler.py`.

## Aggregate Ceilings

| Scenario | Track A | Track B | Final |
|---|---:|---:|---:|
| Current seed7 | 0.890136718750 | 0.640136718750 | 0.827636718750 |
| Best legal same-width readout remap, zero gate noise | 0.950693606049 | 0.810149370905 | 0.915557547263 |
| Current circuit with oracle readout | 0.945495605469 | 0.832031250000 | 0.917129516602 |
| Illegal compressed-output oracle | 0.950693606049 | 1.000000000000 | 0.963020204536 |

## Direct Answers

- B001/B002 cap source: track_b_001: current measurement cap 0.809639, best same-width cap 0.858252, lift 0.048613; track_b_002: current measurement cap 0.748250, best same-width cap 0.762046, lift 0.013796. The remaining gap is dominated by mandatory same-width output readout, not by missing a large readout remap.
- A007/A008 remap headroom: track_a_007: current measurement cap 0.941359, best same-width cap 0.951883, lift 0.010524; track_a_008: current measurement cap 0.928180, best same-width cap 0.942365, lift 0.014184
- Fewer readout-sensitive bits: No. validate_compiler_equivalence requires compiled final measurement width to match logical width; expected-output rules index the full measured bitstring. Fewer anchor bits are only an illegal/classical-postprocessing oracle in this evaluator.

## Per-Sample Decomposition

| Sample | Current | Current measured cap | Best same-width cap | Zero-CX same-output cap | Legal compressed cap | Illegal compressed oracle | Oracle-readout current | CX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `track_a_001` | 0.941406 | 0.950922 | 0.951883 | 0.951883 | 0.951883 | 0.951883 | 0.980469 | 2 |
| `track_a_002` | 0.871094 | 0.941359 | 0.951883 | 0.951883 | 0.951883 | 0.951883 | 0.936035 | 8 |
| `track_a_003` | 0.929688 | 0.950922 | 0.951883 | 0.951883 | 0.951883 | 0.951883 | 0.977051 | 3 |
| `track_a_004` | 0.914062 | 0.950922 | 0.951883 | 0.951883 | 0.951883 | 0.951883 | 0.970215 | 5 |
| `track_a_005` | 0.925781 | 0.946984 | 0.951883 | 0.951883 | 0.951883 | 0.951883 | 0.967285 | 5 |
| `track_a_006` | 0.929688 | 0.950922 | 0.951883 | 0.951883 | 0.951883 | 0.951883 | 0.977539 | 3 |
| `track_a_007` | 0.804688 | 0.941359 | 0.951883 | 0.951883 | 0.951883 | 0.951883 | 0.879395 | 14 |
| `track_a_008` | 0.804688 | 0.928180 | 0.942365 | 0.942365 | 0.942365 | 0.942365 | 0.875977 | 14 |
| `track_b_001` | 0.666992 | 0.809639 | 0.858252 | 0.858252 | 0.858252 | 1.000000 | 0.821533 | 21 |
| `track_b_002` | 0.613281 | 0.748250 | 0.762046 | 0.762046 | 0.762046 | 1.000000 | 0.842529 | 19 |

## Decision Gate

- Legal same-width final ceiling: `0.915557547263`.
- Legal ceiling below 0.92: `True`.
- Track B illegal semantic-readout final lift: `0.047462657274`.
- Track B illegal lift >= 0.04 Final: `True`.

Legal conclusion: same-width final measurement is a hard evaluator constraint. Compressed anchor-bit output is useful as an oracle diagnostic, but not a valid submission route unless the evaluator contract changes.
