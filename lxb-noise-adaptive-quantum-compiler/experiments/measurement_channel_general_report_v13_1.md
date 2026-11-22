# V13.1 Measurement-Channel Generalization And V14 Track A Resynthesis Report

## Baseline

Current promoted root compiler remains V13:

| Metric | Value |
|---|---:|
| Seed 7 Final | 0.8060302734375 |
| Seed 7 Track A | 0.8720703125 |
| Seed 7 Track B | 0.60791015625 |
| B001 | 0.650390625, 25 CX |
| B002 | 0.5654296875, 18 CX, depth 6 |
| Three-seed mean Final | 0.798828125 |
| Invalid | 0 |

## V13.1 Implementation

Added `measurement_channel_general.py` with:

- final-Z measured-observable extraction via `cirq.PauliString.before`, equivalent to propagating `U^\dagger Z_i U`;
- affine channel representation `y = A r + B x + b`, where `r` are H-created random variables and `x` are computational-basis challenge bits;
- exact channel equivalence up to the same random affine subspace, including measurement order, signs, and layout remapping;
- strict rejection of unsupported Y observables and dependence on physical qubits outside the layout;
- opt-in `track_b_branch_v13_1` variant;
- conservative candidate generation that currently admits only proved disjoint-star fallback or a simple affine route candidate when exact affine-channel equivalence and metric guards pass.

The existing V13 B002 18-CX candidate remains the fallback and is still selected.

## V13.1 Public Results

| Seed | Track A | Track B | Final | Invalid |
|---:|---:|---:|---:|---:|
| 7 | 0.8720703125 | 0.60791015625 | 0.8060302734375 | 0 |
| 11 | 0.8662109375 | 0.60546875 | 0.801025390625 | 0 |
| 17 | 0.85400390625 | 0.595703125 | 0.7894287109375 | 0 |

Three-seed mean Final: `0.798828125`.

## Synthetic TrackB Affine Results

Generated `experiments/synthetic_trackb_affine_v13_1` with:

- exact affine parity-star variants;
- parity-tree variants;
- multi-row affine parity variants;
- near-miss unsupported Clifford negative cases with Y observables.

| Variant | Mean TrackB Success | Invalid | Negative False Accepts | p95 Compile |
|---|---:|---:|---:|---:|
| V11 proxy | 0.5938802083333333 | 0 | 0 | 9.016273200046271 |
| V12 sim selector | 0.5938802083333333 | 0 | 0 | 17.289938599977177 |
| V13 | 0.6046874999999999 | 0 | 0 | 9.265591100032907 |
| V13.1 | 0.6046874999999999 | 0 | 0 | 9.004237900022417 |

V13.1 is better than V11/V12 on the affine synthetic suite and clears invalid, negative-case, and timing gates. It does not improve beyond V13 because the generalized candidate synthesis is still intentionally conservative; only the already-proven star fallback is accepted.

Artifacts:

- `experiments/synthetic_trackb_affine_v13_1.json`
- `experiments/synthetic_trackb_affine_v11_proxy.json`
- `experiments/synthetic_trackb_affine_v12_sim_selector.json`
- `experiments/synthetic_trackb_affine_v13_baseline.json`
- `experiments/candidate_dump_v13_1.jsonl`

## V13.1 Keep Decision

V13.1 clears:

- public seed 7 not below V13;
- three-seed mean not below V13 - 0.003;
- synthetic affine TrackB better than V11/V12;
- near-miss negative cases zero false accept;
- invalid = 0;
- p95 compile < 12 seconds.

Decision: keep the code and `track_b_branch_v13_1` variant, but do not change the root compiler beyond current V13 because public score is unchanged.

## V14 Track A Implementation

Added `track_a_resynthesis/` with:

- `window_extract.py`: dependency-local pair windows and replacement helper;
- `exact_equivalence.py`: unitary equality up to global phase;
- `phase_polynomial.py`: guarded placeholder returning no candidates;
- `reversible_templates.py`: guarded placeholder returning no candidates;
- `topology_templates.py`: basis/coupling legality helper;
- `local_resynth.py`: opt-in candidate pass applying exact 2Q KAK/CZ local resynthesis to top existing candidates, keeping only strict metric improvements.

Added `track_a_v14` variant and `scripts/dump_tracka_v14_candidates.py`.

## V14 Results

Seed 7 public result:

| Metric | Value |
|---|---:|
| Track A | 0.8720703125 |
| Track B | 0.60791015625 |
| Final | 0.8060302734375 |
| Invalid | 0 |

Pressure samples:

| Sample | Success | 2Q | Depth |
|---|---:|---:|---:|
| A007 | 0.74609375 | 18 | 25 |
| A008 | 0.7578125 | 25 | 32 |
| B001 | 0.650390625 | 25 | 9 |
| B002 | 0.5654296875 | 18 | 6 |

Three-seed mean Final: `0.798828125`.

Synthetic holdout: mean Final `0.7630208333333334`, invalid `0`, p95 compile `7.654853500018362`.

The V14 candidate dump contains zero accepted `v14_pair_kak` candidates. Existing pair KAK resynthesis does not find a strict 2Q/depth/reliability improvement on the selected A007/A008 circuits.

Artifacts:

- `experiments/tracka_resynthesis_candidates_v14.jsonl`
- `experiments/full_public_summary_v14_seed7.json`
- `experiments/seed_robustness_v14_7_11_17.json`
- `experiments/synthetic_small_tracka_v14.json`
- `experiments/loss_attribution_v13_measurement_channel.json`

## V14 Keep Decision

V14 does not clear:

- TrackA seed7 >= 0.895;
- Final seed7 >= 0.823;
- three-seed mean Final >= 0.815.

Decision: keep V14 modules as guarded diagnostic scaffolding only. Root compiler remains V13.

