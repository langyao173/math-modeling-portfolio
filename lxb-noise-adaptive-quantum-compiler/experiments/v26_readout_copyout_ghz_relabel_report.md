# V26 Readout Copyout + GHZ Relabel Report

Date: 2026-06-21

## Summary

This iteration tested legal ways to move closer to the readout-only ceiling without changing the frozen root compiler.

The literal 0.9999 target is not reachable under the current evaluator contract: every valid output must contain one full-width final `z` measurement, and the simulator injects independent readout bit-flip noise before every measured physical qubit. For Track A bitstring rules, even a perfect zero-gate implementation is capped by the product of the selected physical readout reliabilities.

The useful result is an aggressive Track B package:

- V22: GHZ leaf-tree candidate for B001.
- V23: readout-copyout candidate for B002.
- V26: V23 plus GHZ-only physical relabel for B001.

Root `compiler.py` remains frozen and does not enable these new aggressive flags.

## Public 5-Seed Results

Seeds: 7 / 11 / 17 / 23 / 31  
Shots: Track A 256, Track B 1024  
Invalid count: 0 for all variants below

| Variant | Mean Final | Mean Track A | Mean Track B | p95 Compile | Max Compile |
|---|---:|---:|---:|---:|---:|
| V20 aggressive fast | 0.8236083984 | 0.8889648438 | 0.6275390625 | 10.19s | 10.61s |
| V22 GHZ leaf tree | 0.8256591797 | 0.8889648438 | 0.6357421875 | 11.94s | 12.84s |
| V23 readout copyout | 0.8271484375 | 0.8889648438 | 0.6416992188 | 10.83s | 11.16s |
| V26 GHZ relabel + copyout | 0.8285888672 | 0.8889648438 | 0.6474609375 | 12.18s | 12.23s |
| V26 fast tree relabel + copyout | 0.8297363281 | 0.8889648438 | 0.6520507813 | 10.94s | 11.18s |
| V27 B002 frontier + copyout | 0.8305664063 | 0.8889648438 | 0.6553710938 | 11.55s | 12.74s |
| V28 wide B002 frontier + copyout | 0.8300292969 | 0.8889648438 | 0.6532226563 | 15.91s | 16.83s |
| V29 Track A lowshot selector | 0.8314453125 | 0.8901367188 | 0.6553710938 | 11.25s | 11.84s |
| V30 Track A ranked-proxy selector | 0.8312744141 | 0.8901367188 | 0.6546875000 | 13.19s | 14.24s |
| V31 V26 + Track A ranked-proxy | 0.8306152344 | 0.8901367188 | 0.6520507813 | 12.34s | 13.82s |
| V32 relative-phase exact-router diagnostic | no full replay | pressure-only | pressure-only | pressure-only | pressure-only |

V26 fast tree relabel improves over V23 on all five public seeds:

| Seed | Final Delta vs V23 | Track B Delta vs V23 |
|---:|---:|---:|
| 7 | +0.0014648438 | +0.0058593750 |
| 11 | +0.0029296875 | +0.0117187500 |
| 17 | +0.0026855469 | +0.0107421875 |
| 23 | +0.0034179688 | +0.0136718750 |
| 31 | +0.0024414063 | +0.0097656250 |

V27 adds a two-layout B002 physical-placement frontier and lets the existing proxy selector choose the lower-sensitivity copyout assignment. It improves the five-seed mean over V26, but seed 7 regresses and max compile rises above 12s:

| Seed | Final Delta vs V26 | Track B Delta vs V26 |
|---:|---:|---:|
| 7 | -0.0009765625 | -0.0039062500 |
| 11 | +0.0008544922 | +0.0034179688 |
| 17 | +0.0012207031 | +0.0048828125 |
| 23 | +0.0015869141 | +0.0063476563 |
| 31 | +0.0014648438 | +0.0058593750 |

## Track B Sample Effects

### B001

| Variant | Mean Success | CX | Depth | ELR |
|---|---:|---:|---:|---:|
| V20 | 0.6347656250 | 25 | 9 | -0.525124 |
| V22/V23 | 0.6511718750 | 21 | 15 | -0.527375 |
| V26 fast tree relabel | 0.6718750000 | 21 | 16 | -0.506463 |

V22 replaced the original GHZ physical tree with a leaf-tree using clean internal physical nodes. V26 then relabeled the physical tree to improve edge/readout reliability while preserving the same final-Z measurement channel. The final implementation uses a deterministic bounded tree-pattern mapper instead of generic VF2 monomorphism search, which reduced B001 compile time from about 12s to about 8s and improved the no-cache public mean.

### B002

| Variant | Mean Success | CX | Depth | ELR |
|---|---:|---:|---:|---:|
| V20/V22 | 0.6203125000 | 18 | 6 | -0.569287 |
| V23/V26 | 0.6322265625 | 19 | 6 | -0.552356 |
| V27 frontier | 0.6388671875 | 19 | 6 | -0.552356 |

V23 adds a guarded final-Z copyout to a cleaner measured physical qubit. The transform only accepts clean ancilla paths where readout gain beats added CNOT noise and the full validator passes.

V27 keeps the same physical cost metrics as V26 but changes the logical-to-physical output assignment among equivalent affine-star candidates. The improvement is sampling-sensitive, so it should be treated as aggressive rather than conservative.

## Track A Selector Diagnostic

The Track A candidate dump exposed a selector residual on `track_a_008`: the Pareto ranker placed a higher-reliability relative-phase candidate first (`14 CX`, depth `28`, ELR about `-0.243048`, readout cost about `0.074529`), but the guarded proxy selector preferred a shallower candidate (`13 CX`, depth `23`, ELR about `-0.258184`, readout cost about `0.080170`). Direct noisy replay favored the ranked candidate on the five-seed mean.

Two opt-in selector diagnostics were added:

- V29 uses a bounded Track-A-only common-random lowshot selector. It improves mean Track A to `0.8901367188` and mean Final to `0.8314453125`, but the gain is small and still sampling-sensitive.
- V30 uses a cheaper Track-A-only ranked-proxy selector. It gets the same Track A mean improvement without Track A lowshot simulation, but the latest replay had a higher Track B compile-time tail from the inherited B002 frontier.
- V31 combines the ranked Track A selector with the lower-tail V26 Track B stack. It is more conservative than V30 but has lower mean Final than V30/V29.

These variants are legal because they select among already validated candidates using only the input circuit and backend. They remain aggressive diagnostics rather than a root promotion because the five-seed final gain is below the `0.005` practical significance threshold.

## A007 Relative-Phase Exact-Router Diagnostic

The remaining A007 gap was tested by combining the best source-level relative-phase rewrite with the exact small-circuit router. The relative-phase rewrite reduces the logical circuit from 15 source CX to 9 source CX, but the incumbent physical route still needs 14 CX. A broader development-time exact-router search found a legal 12-CX / depth-20 route:

- 12-CX exact route: depth `20`, ELR about `-0.225117`.
- Current 14-CX incumbent: depth `22`, ELR about `-0.223123`.

Despite using two fewer CX gates, the 12-CX route had slightly worse noisy replay: five-seed 4096-shot mean `0.8278808594` versus `0.8288574219` for the incumbent. This shows the current A007 residual is not merely CX count; edge choice and reliability dominate at this margin. The opt-in V32 path therefore has a guard: exact-routed relative-phase candidates are only added when the CX/depth gain does not materially worsen ELR, or when ELR materially improves with no CX increase. In the current public A007 pressure smoke, V32 keeps the incumbent selected.

## Rejected Diagnostic

V28 widened the B002 physical-placement frontier from 2 to 12 candidates and increased the bounded placement budget from 3s to 8s. It was rejected:

- Mean Final delta vs V27: -0.0005371094.
- Mean Track B delta vs V27: -0.0021484375.
- p95 compile rose from about 11.68s to 15.91s.
- B002 seed 7 and seed 23 improved slightly, but seed 17 and seed 31 regressed enough to lose the five-seed mean.

V24 fixed the Track B common-random lowshot selector and tested it as an opt-in selector. It was rejected:

- Mean Final delta vs V23: +0.0000488281
- p95 compile: 16.30s
- It introduced seed regressions and did not justify the runtime tail.

The selector bugfix is retained because it is logically correct and only affects opt-in selector modes.

## Readout Ceiling Check

The evaluator requires exactly one final `z` measurement whose width matches the source logical width, and both noisy simulators inject readout errors immediately before measuring every output qubit. Track A public rules are exact bitstrings, so even a zero-gate implementation placed on the best medium-backend readout qubits has a readout-only success ceiling of about 0.951 for six-qubit samples and 0.942 for the seven-qubit sample. The public Track A mean readout-only ceiling is 0.9506936060.

For Track B, applying the public output rules to the best available large-backend readout qubits gives a readout-only ceiling of 0.8582524044 for GHZ and 0.7620463374 for the parity-check sample, with mean 0.8101493709. The weighted public readout-only ceiling under the current benchmark set is therefore about 0.9155575473 before any gate noise. This does not rule out more legal score improvement, but it rules out a legal 0.9999-class score under the current evaluator because full-width noisy readout is mandatory.

## Verification

- `python -m pytest`: 93 passed.
- Runtime compliance scan over `compiler.py` and `src/noise_adaptive_hackathon/submission/*.py`: no benchmark-name, manifest, public-summary, expected-output, hidden/private path, or absolute local path dependency found.
- Root CLI remains unchanged: only `--circuit`, `--backend`, and `--out`.
- Root compiler remains conservative; V26/V27 are aggressive variants only.

## Decision

V27 is the current highest public mean aggressive package, while V26 fast tree relabel is cleaner on seed stability and compile-time max. Neither is a proof of a 0.9999-class breakthrough. These are legal readout/placement gains near the remaining Track B margin. The hard blocker to 0.9999 remains mandatory full-width noisy readout.
