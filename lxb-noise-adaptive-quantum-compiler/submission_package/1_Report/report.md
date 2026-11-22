# Noise-Adaptive Quantum Compiler Submission Report

## Summary

This submission implements an offline, deterministic, black-box quantum compiler for the dual-track NISQ mapping challenge. The runtime entrypoint is `compiler.py`; it accepts only `--circuit`, `--backend`, and `--out`, writes Cirq JSON plus the required `.layout.json` sidecar, and uses a fixed seed of `7`.

The compiler uses a pass-style pipeline:

- backend noise graph construction with edge cost `-log(1 - two_qubit_error)` and readout cost `-log(1 - readout_error)`;
- circuit interaction extraction with weighted two-qubit interaction counts;
- noise-aware initial layout over current baseline candidates, greedy high-interaction placement, low-readout, low-noise-subgraph, deterministic annealed, hub-aware, and small-circuit random-pruned candidates;
- bounded lookahead routing with legal `cx`-decomposed movement, CX-and-swap fusion, distance-2 bridge candidates, and guarded terminal readout remapping;
- exact GHZ tree fanout candidate generation when the input structurally matches one root `H`, root-to-leaf fanout `CX`, final `z` measurement, and Clifford-only Track B constraints;
- fixed-layout `RouteCQC` candidates with multiple lookahead radii, plus bounded hub-aware local search for hub-leaf Clifford circuits;
- conservative peephole cancellation for adjacent self-inverse pairs and identical CNOTs separated only by disjoint commuting operations;
- guarded GF(2) CNOT-block resynthesis for Clifford-only circuits, accepted only when the synthesized topology-respecting block reduces CX count and passes validation;
- guarded Track A local two-qubit block resynthesis via KAK/CZ decomposition, accepted only when CX count decreases and validation passes;
- measurement-tail pruning for final Z-basis diagonal gates, accepted only after full validation;
- a proof-carrying Track A relative-phase Toffoli rewrite: exact 6-CX Toffoli decompositions are replaced by 3-CX relative-phase Toffoli templates only when the entire final-Z measurement channel is row-phase equivalent, so all final computational-basis measurement probabilities are preserved for arbitrary input states;
- Clifford-only detection and validation for Track B-compatible inputs;
- Clifford error-sensitivity scoring for Clifford candidates, estimating how CX-local Pauli faults propagate into final measured bits;
- a guarded legacy hub-parity candidate that restores the old fixed-budget hub-aware LS/SA search for large non-GHZ Clifford hub-leaf circuits, without using sample names;
- V13 measurement-channel parity resynthesis for large non-GHZ Clifford hub-leaf circuits: it detects an affine H-prefix/CNOT/X/final-Z parity-star measurement channel, directly places disjoint physical stars, and accepts the candidate only when it strictly improves CX count, depth, reliability, and Clifford sensitivity over the legacy fallback;
- Pareto-frontier candidate selection over estimated log reliability, two-qubit count, depth, terminal readout cost, route overhead, Clifford sensitivity cost, and compile-time proxy after local validation;
- an experimental simulation-in-the-loop selector variant that estimates low-shot noisy support success from the input circuit semantics only. This variant is retained for diagnostics but is not promoted as the default because its public score is unchanged and its compile-time overhead is higher.

The implementation does not use network access, API keys, hidden data, benchmark metadata, or sample-name hardcoding at runtime.

## Legality And Residual Framing

The compiler is optimized under the legal same-width final-Z bitstring equivalence contract: the output circuit preserves the evaluator-visible final `z` measurement distribution, writes the required layout sidecar, and keeps the measured classical bitstring width required by the source program. Within that contract, the system combines noise-aware routing, Track B measurement-channel resynthesis, and Track A row-phase equivalence rewrites. The no-go audit excludes compressed-output oracle variants and candidates that fail validator, equivalence, noiseless semantics, or full-circuit final-Z row-phase checks.

The remaining residual is therefore treated as a physical and synthesis residual rather than as an unchecked optimization gap: mandatory output width exposes readout noise, Track B is sensitive to the selected physical readout set and tree/star edges, and Track A residuals concentrate in low-CX reversible/row-phase synthesis lower bounds.

## Reproduction Commands

Environment setup:

```bash
uv venv
uv pip install -e ".[dev]"
```

Run tests:

```bash
.venv\Scripts\python -m pytest
```

Run the final public evaluation for the final root and the C0 baseline:

```bash
.venv\Scripts\python scripts\run_private_leaderboard_emulator.py --suite public_replay --variant final_c0_conservative --variant v19_relative_phase_toffoli --baseline final_c0_conservative --seed 7 --seed 11 --seed 17 --seed 23 --seed 31 --shots-a 256 --shots-b 1024 --timeout-sec 30
```

Previous stable default summary:

```bash
.venv\Scripts\python scripts\evaluate.py --out experiments\full_public_summary_v12_stable_default.json --variant v12_stable_default --decision keep_proxy_default --shots-a 256 --shots-b 1024 --seed 7 --timeout-sec 30
```

Dump all public candidates for selector diagnosis:

```bash
.venv\Scripts\python scripts\dump_candidates.py --out experiments\candidate_dump_v11.jsonl --shots-a 256 --shots-b 1024 --seed 7 --debug-success selected --deep-check selected
```

Generate the synthetic holdout set:

```bash
.venv\Scripts\python scripts\gen_synthetic_holdout.py --out experiments\synthetic_holdout_v11 --seed 7
```

Run synthetic holdout scoring:

```bash
.venv\Scripts\python scripts\evaluate_synthetic_holdout.py --root experiments\synthetic_holdout_v11 --out experiments\synthetic_holdout_summary_v11_proxy_only.json --variant track_b_branch_proxy_only --shots-a 64 --shots-b 256 --seeds 7 11 17 --timeout-sec 30 --label v11_proxy_only
.venv\Scripts\python scripts\evaluate_synthetic_holdout.py --root experiments\synthetic_holdout_v11 --out experiments\synthetic_holdout_summary_v12_sim_selector.json --variant track_b_branch_sim_selector --shots-a 64 --shots-b 256 --seeds 7 11 17 --timeout-sec 30 --label v12_sim_selector
.venv\Scripts\python scripts\evaluate_synthetic_holdout.py --root experiments\synthetic_holdout_v11 --out experiments\synthetic_holdout_summary_v13_measurement_channel_nogf2.json --variant track_b_branch_v13 --shots-a 64 --shots-b 256 --seeds 7 11 17 --timeout-sec 30 --label v13_measurement_channel_nogf2
```

Run the ablation study:

```bash
.venv\Scripts\python scripts\ablation.py --out experiments\ablation_summary_v10_structural_replay.json --shots-a 256 --shots-b 1024 --timeout-sec 30
```

Run remaining-loss attribution for the previous V10 baseline:

```bash
.venv\Scripts\python scripts\attribute_losses.py --summary experiments\full_public_summary_v10_structural_replay.json --out experiments\loss_attribution_v10_structural_replay.json
```

Package the submission:

```bash
.venv\Scripts\python scripts\package.py --out submission_package
```

## Final Public Evaluation Result

Raw final matrix JSON: `experiments/private_emulator_v1/public_5seed_matrix.json`.

| Metric | Value |
|---|---:|
| Seed-7 Track A | 0.89208984375 |
| Seed-7 Track B | 0.60791015625 |
| Seed-7 Final Score | 0.821044921875 |
| Public 5-seed mean Track A | 0.88896484375 |
| Public 5-seed mean Track B | 0.609375 |
| Public 5-seed mean Final | 0.8190673828125 |
| Invalid samples | 0 |
| A007 seed-7 two-qubit gates / depth / log reliability | 14 / 22 / -0.2231225844058102 |
| A008 seed-7 two-qubit gates / depth / log reliability | 13 / 23 / -0.25818449829740375 |

The public baseline reference is Final `0.6626`, Track A `0.7500`, Track B `0.4004`. The final root improves the seed-7 public final score by about `0.1584` and the five-seed mean final score by about `0.1565`, while preserving zero public invalid samples.

The final submission root is `v19_relative_phase_toffoli`: C0 conservative plus the guarded relative-phase Toffoli rewrite. The actual black-box `compiler.py` entrypoint enables the same configuration and matches the V19 five-seed public result. This is the Track A structural breakthrough that was missing from the earlier conservative root.

| Variant | Public 5-seed Final | Delta vs C0 | Track A Mean | Track B Mean | Public Invalid | p95 Compile |
|---|---:|---:|---:|---:|---:|---:|
| C0 conservative | 0.805810546875 | 0.000000000000 | 0.8712890625 | 0.609375 | 0 | 8.240160600049421 |
| V19 relative-phase Toffoli | 0.8190673828125 | 0.013256835938 | 0.88896484375 | 0.609375 | 0 | 8.221962200012058 |

The final aggressive gate was: public five-seed mean delta at least `+0.004`, public invalid count `0`, no added aggregate holdout invalids versus C0, p95 compile below `12s`, and no negative public seed deltas. V19 clears this gate with public seed deltas `+0.012085`, `+0.016479`, `+0.014648`, `+0.013916`, and `+0.009155`. Full decision details are in `experiments/private_emulator_v1/submission_choice_report.md`.

Historical V13 seed-7 score before route-scar and measurement-tail finalization:

| Metric | Value |
|---|---:|
| Track A | 0.8720703125 |
| Track B | 0.60791015625 |
| Final Score | 0.8060302734375 |
| Invalid samples | 0 |
| Mean depth | 14.8 |
| Mean two-qubit gates | 12.9 |
| Mean estimated log reliability | -0.25116064259701004 |

Track B pressure-point comparison against the released public baseline summary:

| Sample | Baseline 2Q Gates | Final 2Q Gates | Baseline Log Reliability | Final Log Reliability |
|---|---:|---:|---:|---:|
| GHZ (`track_b_001`) | 51 | 25 | -0.9715223045306494 | -0.5251238029904381 |
| Parity (`track_b_002`) | 100 | 18 | -2.0154799221000546 | -0.6181661479233818 |

Candidate-level diagnostics are stored in `experiments/candidate_dump_v11.jsonl` for the pre-V13 selector and in `experiments/full_public_summary_v13_measurement_channel_root.json` for the promoted run. For `track_b_002`, V13 detects that the final Z distribution is an affine hub-leaf parity-star measurement channel. It replaces the legacy 97-CX routed circuit with a directly placed 18-CX physical-star circuit, depth 6, estimated log reliability `-0.6181661479233818`, and public seed-7 success `0.5654296875`. Exact GHZ tree fanout keeps `track_b_001` at 25 two-qubit gates and public seed-7 success `0.650390625`.

V12 adds an explicit simulation-in-the-loop selector variant. Public ablation on seed 7 shows the same score as the stable proxy selector:

| Variant | Track A | Track B | Final | Invalid | Mean Compile Time |
|---|---:|---:|---:|---:|---:|
| stable proxy selector | 0.8720703125 | 0.55859375 | 0.793701171875 | 0 | 4.540547999995761 |
| simulation selector | 0.8720703125 | 0.55859375 | 0.793701171875 | 0 | 4.939099700003863 |

Because the simulation selector does not clear the `+0.005` public keep threshold and increases mean compile time, it remains diagnostic-only. V13 keeps the stable proxy selector and changes the candidate pool by adding the guarded measurement-channel parity candidate.

V13 public ablation on seed 7:

| Variant | Track A | Track B | Final | Invalid | Mean Compile Time |
|---|---:|---:|---:|---:|---:|
| V11/V12 stable proxy selector | 0.8720703125 | 0.55859375 | 0.793701171875 | 0 | 3.9927734000084456 |
| V13 measurement-channel parity | 0.8720703125 | 0.60791015625 | 0.8060302734375 | 0 | 4.061597239994444 |

Final root public seed robustness:

| Seed | Track A | Track B | Final | Invalid |
|---:|---:|---:|---:|---:|
| 7 | 0.89208984375 | 0.60791015625 | 0.821044921875 | 0 |
| 11 | 0.8916015625 | 0.60546875 | 0.820068359375 | 0 |
| 17 | 0.87939453125 | 0.595703125 | 0.8084716796875 | 0 |
| 23 | 0.88916015625 | 0.60791015625 | 0.81884765625 | 0 |
| 31 | 0.892578125 | 0.6298828125 | 0.826904296875 | 0 |

The five-seed mean final score is `0.8190673828125`, with zero public invalid samples for all seeds.

## Ablation Study

Raw ablation JSON: `experiments/ablation_summary_v10_structural_replay.json`.

| Variant | Track A | Track B | Final | Invalid | Mean Depth | Mean 2Q Gates | Mean Log Reliability |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.75830078125 | 0.40673828125 | 0.67041015625 | 0 | 24.8 | 25.5 | -0.578991716597065 |
| layout_only | 0.8115234375 | 0.30810546875 | 0.6856689453125 | 0 | 34.7 | 43.2 | -0.7089886519472255 |
| layout_routing | 0.83935546875 | 0.4287109375 | 0.7366943359375 | 0 | 22.7 | 25.8 | -0.45184707818458875 |
| layout_routing_peephole | 0.83935546875 | 0.4287109375 | 0.7366943359375 | 0 | 22.7 | 25.8 | -0.45184707818458875 |
| track_b_branch (V10) | 0.8720703125 | 0.51806640625 | 0.7835693359375 | 0 | 19.2 | 20.7 | -0.38546160249054356 |
| track_b_branch (V11) | 0.8720703125 | 0.55859375 | 0.793701171875 | 0 | 19.4 | 20.8 | -0.36354954052741045 |
| track_b_branch (V13) | 0.8720703125 | 0.60791015625 | 0.8060302734375 | 0 | 14.8 | 12.9 | -0.25116064259701004 |

The ablation indicates that layout alone improves Track A but can over-route Track B. The routing pass recovers reliability and final score. V11 adds candidate-level Clifford sensitivity and a guarded legacy hub-parity candidate, restoring Track B parity without materially changing Track A. V12 adds simulation-selector infrastructure but keeps the stable proxy selector as the default. V13 adds guarded measurement-channel resynthesis and is promoted because it clears the public, robustness, invalid-count, and timing gates.

## Synthetic Holdout

Synthetic holdout generation and scoring are now implemented. The holdout contains GHZ-star, GHZ-chain/tree, parity hub-leaf, random Clifford CNOT, small Clifford+T, and structured routing-pressure circuits with jittered backend snapshots.

| Variant | Mean Track A | Mean Track B | Mean Final | Max Invalid | p95 Compile Time |
|---|---:|---:|---:|---:|---:|
| stable proxy selector | 0.83984375 | 0.5325520833333334 | 0.7630208333333334 | 0 | 8.897578999982215 |
| simulation selector | 0.83984375 | 0.5326522435897436 | 0.7630458733974358 | 0 | 12.253149700001813 |
| V13 measurement-channel parity | 0.83984375 | 0.5325520833333334 | 0.7630208333333334 | 0 | 8.010450700006913 |

The older broad synthetic result is neutral for V13 because that generated holdout does not contain the exact affine parity-star measurement-channel structure that triggers the promoted pass. The newer private-emulator holdouts are more discriminative for the final root: V19 improves synthetic structural Track A from `0.74765625` to `0.76357421875` with zero invalid samples, while synthetic affine Track B remains effectively unchanged versus C0 and does not introduce additional invalid samples. The GF(2) CNOT-block diagnostic path is available behind an explicit constructor flag but is disabled in the runtime candidate ensemble because it did not improve B002 and increased synthetic timing.

## Residual Analysis and No-Go Diagnostics

The final development phase shifted from adding generic passes to residual analysis. The resulting residual map is:

| Residual | Diagnosis | Decision |
|---|---|---|
| Track B GHZ | Structured fanout state | Solved by exact GHZ tree fanout |
| Track B parity | Affine final-Z measurement channel | Solved by V13 measurement-channel parity resynthesis |
| Track A A008 | Routing scar plus exact Toffoli phase structure | Solved by route-scar normalizer and relative-phase Toffoli rewrite |
| Track A A007 | H-boundary Clifford+T / reversible-carry structure | Solved by final-Z row-phase-equivalent relative-phase Toffoli rewrite |

For A008, route-scar normalization first distilled a bounded-window accepted candidate into a cheap runtime rule. The later relative-phase Toffoli rewrite improves the final root further: A008 reaches 13 CX gates, depth 23, and estimated log reliability `-0.25818449829740375` on public seed 7.

For A007, earlier targeted attempts failed to beat the incumbent:

- Route-scar normalization produced zero accepted candidates.
- Phase-affine cone analysis found that the full closed cone is not a pure phase-affine `permutation x diagonal phase` block.
- H-boundary path-sum analysis detected one cone over qubits `[0, 1, 2, 3, 4, 5]`, 4 path variables, and 18 phase/path terms, but the best routed exact attempt only tied the incumbent at 18 CX, depth 25, estimated log reliability `-0.27710686331225304`, and failed the metric guard.
- Physical reembedding checked 280 routed candidates and accepted none; the best candidate had 21 CX, depth 29, and estimated log reliability `-0.299`, worse than the incumbent 18 CX, depth 25, `-0.277`.

The breakthrough is that A007 is not unitary-rewrite limited; it is final-measurement limited. Its two exact 6-CX Toffoli decompositions can be replaced by 3-CX relative-phase Toffoli templates because the whole circuit after replacement is row-phase equivalent at the final Z measurement: every output row differs only by a phase, so all final bitstring probabilities are exactly preserved for arbitrary input states. This is stronger than passing the sampled evaluator challenges and is checked directly before the rewrite is allowed.

The promoted rewrite reduces A007 from 18 routed CX, depth 25, estimated log reliability `-0.27710686331225304` to 14 routed CX, depth 22, estimated log reliability `-0.2231225844058102`, with validator, evaluator equivalence, and noiseless checks passing. It also improves A008 from 22 routed CX after route-scar normalization to 13 routed CX. An offline relative-phase rule miner remains available for future expansion; it enumerated width-3 H/CX/T/TDG/S/SDG circuits up to depth 4, verified 199 exact-unitary rules, and found 48 CX-reducing rules.

## Row-Phase Proof Audit

Raw proof report: `experiments/row_phase_proof_audit_v19.md`.

The V19 rewrite is guarded by a full-circuit final-Z row-phase miter. The audit explicitly constructs the original and rewritten full-circuit unitaries before final measurement and verifies that `U' = D U` for a diagonal phase matrix `D` in the measured computational basis. It then compares final-Z probabilities for every computational-basis input, random complex statevectors, and randomized hidden-prep circuits composed from `H`, `S`, `T`, and `X`.

| Audit item | Value |
|---|---:|
| Rewrites audited | 10 |
| Row-phase accepted | 10 |
| Negative cases | 5 |
| Negative failures | 0 |
| Max basis-input probability delta | 1.332e-15 |
| Max random-state probability delta | 1.665e-16 |
| Max hidden-prep probability delta | 8.882e-16 |

Negative cases include relative phase followed by `H`, non-final measurement, non-`z` measurement, an explicit row-phase false case, and a randomized single-qubit false case. All are rejected. This is the main hidden-safety argument for promoting V19.

## Row-Phase Rule Miner

Raw miner output: `experiments/verified_ruleset_row_phase.json`; report: `experiments/row_phase_rule_miner_report.md`.

The previous exact-unitary miner was extended to mine final-Z row-phase equivalence classes. It row-normalizes each unitary to canonicalize the equivalence relation `U ~ U' iff U' = D U`, where `D` is diagonal in the final measured basis. A bounded offline run over widths 3 and 4, gate set `H/CX/T/TDG/S/SDG/X/Z`, max depth 4, and 12,000 enumerated candidates per width exported 97 verified row-phase rules, including 58 CX-reducing rules.

| Width | Seeded | Mined | Deduped | CX-reducing |
|---:|---:|---:|---:|---:|
| 3 | 21 | 28 | 49 | 34 |
| 4 | 48 | 0 | 48 | 24 |

The mined row-phase rules are kept offline-only. Runtime promotion still requires structural pattern matching, exact full-circuit row-phase miter, validator/equivalence/noiseless checks, and a metric guard. No additional mined rule was promoted into V19 after the release-candidate freeze.

## Private Data Isolation Audit

- `compiler.py` reads only the paths provided by `--circuit`, `--backend`, and `--out`.
- The runtime compiler does not read `benchmark.json`, manifests, public summaries, file names, hidden paths, or expected-output rules.
- Experiment scripts may read public benchmark metadata for local analysis, but those scripts are not needed by the final black-box compiler command.
- No network calls, online services, API keys, environment secrets, or external account credentials are used.
- A static runtime scan of `compiler.py` and `src/noise_adaptive_hackathon/submission/*.py` found no benchmark-name, public-summary, network, subprocess, directory-enumeration, or hidden-data access; the only file read is the user-provided backend path in `compiler.py`.
- The output is validated through the repository evaluator before noisy scoring, including sidecar, basis, coupling, Clifford-only, equivalence, and noiseless semantic checks.

## Limitations

- The router is heuristic and bounded; it is not a global optimum solver.
- Public-score changes below `0.005` to `0.01` should be interpreted cautiously because shot noise can dominate small differences.
- The resynthesis passes are intentionally conservative; V13 handles affine hub-leaf parity measurement channels but does not yet implement full weighted GF(2) Steiner-Gauss synthesis or broad multi-qubit Clifford tableau reduction.
- Weighted GF(2)/Steiner-Gauss synthesis and broader Track A phase-polynomial/local Clifford+T resynthesis remain future work; the promoted Track A relative-phase Toffoli rewrite handles the released A007/A008 carry-like residual without enabling a broad unguarded synthesizer.
- Candidate selection optimizes an estimated reliability proxy, not the exact hidden noisy score.
- The compiler is designed for the released schema and basis gates; future directed-CNOT or pulse-level noise models would require extending the backend model.
