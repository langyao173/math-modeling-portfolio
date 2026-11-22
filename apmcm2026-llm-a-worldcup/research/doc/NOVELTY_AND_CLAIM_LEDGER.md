# Novelty, Falsifier, and Claim-Evidence Ledger

## 1. Claim policy

No component is called original merely because it is absent from the official APMCM baseline.
Novelty is assessed against the nearest verified literature, and every retained claim requires an
executable artifact. Integration and auditability can be contributions even when the statistical
building blocks are established.

## 2. Nearest-work novelty matrix

| Proposed component | Nearest prior work | What is not new | Potentially differentiating element | Falsifier | Required evidence |
|---|---|---|---|---|---|
| Joint score kernel | Maher; Dixon-Coles; Karlis-Ntzoufras; Koopman-Lit | Paired count distribution and derived WDL | One frozen kernel used and numerically certified across match, half-time, and tournament outputs | Any derived output disagrees with score-matrix aggregation | Cross-task invariant suite and simulator ledger |
| Dynamic attack/defence | Rue-Salvesen; Crowder; Baker-McHale | Time-varying team strength | Strict predict-then-update filtration with explicit feature availability certificates | Future or same-match state changes earlier predictions | Mutation/leakage tests and fold states |
| Sparse-team shrinkage | Baio-Blangiardo; Groll et al. | Hierarchical/regularized pooling | Explicit unknown-team stress and calibrated abstention/OOD policy | New-team predictions become unstable or falsely certain | Sparse/new-team outer strata and bootstrap uncertainty |
| Player/coach layer | Whitaker; Kharrat et al.; Arntzen-Hvattum | Player abilities and lineup aggregation | Formal separation of predicted lineup, confirmed lineup, and historical player state | Same-match events leak or no repeated external gain | Kickoff-only ledger, missingness bias audit, A2 |
| Half-time increment | In-play Bayesian literature | Conditioning on current score and nonnegative remaining goals | Same kernel updated across declared filtrations with structural monotonicity | FT samples fall below HT or no conditional calibration | State-stratified coverage and monotonicity tests |
| Calibration | Guo; Kull; Gneiting literature | Temperature/Dirichlet/isotonic methods | Calibration selected only through nested temporal OOF and attached to the exact model hash | Outer test influences fit or classwise reliability regresses | Inner/outer calibration ledger and reliability intervals |
| Nonexchangeable intervals | Xu-Xie; Gibbs-Candès; Barber et al. | Weighted/adaptive conformal ideas | Honest separation of empirical, dependence-qualified, regime, and bracket uncertainty | Unqualified IID 90% claim or poor historical coverage | Coverage by era/stage/OOD plus assumption statement |
| 2026 simulation | Dyte-Clarke; Koning et al.; Scarf-Yusof | Monte Carlo tournament simulation | Transport decomposition and partial-identification envelope under a new format | 2026 total is direct linear extrapolation or bracket totals do not reconcile | Pseudo-expansion audits and simulation invariants |
| Prediction certificate | model cards/datasheets; engineering provenance | Documentation and hashes in general | Machine-verifiable binding of data cutoff, information set, schema, model, calibration, OOD, and invariants per forecast | Any hash cannot resolve or certificate/model identities differ | Independent fail-closed verifier |
| Promotion gate | future-match challenge practice | Temporal evaluation and model comparison | Same deployable identity must win repeated outer folds and pass structural/reproducibility gates | Evaluated dynamic model differs from deployed static model | Exact gate-model-schema-artifact identity |

## 3. Current claim ledger after development, audit, and scenario closure

The selected candidate is bound by selection-procedure hash
`751895a4397c9a7597d40b69144ed699a74ace38d6589ab9990ccca346537a29`, configuration hash
`77ba14cb38a364b145be15ec4474811bc6bd0914452e438d34d415ee604c366b`, and final-fit hash
`10238a4c6d368e50acb9ac28e4523dbd1b4f1b64710667ba0e8eb511b8774f53`. These identities must
be refreshed if the deterministic selection rerun changes any frozen byte.

| Claim | Measured evidence | Final bounded decision |
|---|---|---|
| The advanced model is a dynamic Bayesian hierarchical model | The executable model is an equivariant penalized count model with attack/defence team blocks and point-in-time covariates; the selected specification has no time decay, shared-score term, or ML residual | **Withdraw that label.** Describe the implemented likelihood and selected components exactly. |
| Calibrated advanced WDL probabilities improve over the legacy mirrored Dixon--Coles baseline | Seven-edition OOF mean log loss 0.954824 versus 0.975247 (paired edition-cluster 95% interval for the difference: -0.036524 to -0.003451); in the one-shot 2014 audit, advanced log loss is 1.022728 versus 0.985056 | **Retain only as 1930--2010 temporal-OOF evidence.** The external audit is adverse and falsifies a general-superiority claim. |
| The same advance is established for exact scorelines and total-goal distributions | Scoreline-NLL difference -0.007952 with interval [-0.028227, 0.010524]; total-goal CRPS difference +0.011554 with interval [-0.008383, 0.035073] | **Narrow/negative.** Joint-score improvement is inconclusive and CRPS is slightly worse on average. |
| The three-class decision layer cures draw collapse without trade-offs | The nested cost-sensitive layer produces draw predictions and improves macro-F1 in all seven folds, but often reduces accuracy | **Retain the draw-recall/macro-F1 result and disclose the accuracy trade-off.** It is a decision rule, not a probability-model improvement. |
| Player-history features improve prediction | Removing the player block (A2) changes mean log loss by -0.003106 relative to A0 | **Negative result.** The available player-history representation is not promoted. |
| Era and format covariates are indispensable | Removing them (A3) changes mean log loss by -0.002146 relative to A0 | **Negative result for this encoding.** This does not erase the independently demonstrated 2026 format shift. |
| Positive score dependence and ML residuals add predictive value | A4 and A5 have exactly zero measured effect because nested selection excluded shared score and residual correction | **Negative selection result.** Do not generalize this to all dependence or ML models. |
| Nested calibration strictly improves every proper score | Removing calibration (A6) improves log loss by 0.000262 but worsens Brier by 0.000221 | **Mixed.** Retain calibration only under the frozen multi-metric non-degradation gate; do not claim uniform gain. |
| Team-effect regularization is optional | Exact removal of team-block L2 shrinkage under identified sum-to-zero/nullspace constraints (A7) worsens log loss by 0.148942 and Brier by 0.049105 | **Retain regularization.** The large degradation is a substantive sparse-environment result. |
| Swap equivariance is merely aesthetic | Ordered-side asymmetric Poisson A8 has maximum empirical swap error 0.470830 and worsens log loss by 0.045833 | **Retain the symmetry constraint.** A8 remains structurally ineligible regardless of predictive score. |
| The complete A0--A8 and stress program exists | 126/126 paired ablation cells, 7,416 match predictions, 24 measured stress rows, A0--A7 symmetry error <= 1.81e-15, and an explicit safe refusal beyond score support | **Retain as an executable evaluation contribution.** Artifact paths: `temp/advanced_evidence/ablation/` and `temp/advanced_evidence/stress/`. |
| One kernel supports match and tournament simulation | The first 5,000-run adjacency-bracket artifact is invalid and remains superseded. The replacement engine passed all 495 mapping/source-graph tests and completed 220,000 tournaments with 104 matches each. | **Retain the replacement as an illustrative mechanism-certified scenario only.** Never reuse the old numbers or call the replacement a factual entrant forecast. |
| Uncertainty output has frequentist 90% coverage | The replacement decomposition reconstructs scenario variance to 1.14e-13, with parameter/regime/path shares 54.99%/24.48%/20.54%. | **Retain only as model-conditional scenario decomposition.** No frequentist coverage guarantee is claimed. |
| Negative dependence is the missing score mechanism | A marginal-preserving Sarmanov family is fitted only on selected inner OOF predictions; the development replay chooses positive standardized dependence in all seven outer folds. | **Negative exploratory result.** The broader family is reported, but negative dependence is not promoted as the explanation. |
| The 2026 output is a factual forecast | No post-2014 results or real 2026 lineups/states enter the closed track | **Scenario only.** Never use factual or retrospective 2026 wording. |

| The advanced candidate generalizes better to 2014 | One-shot audit: advanced calibrated log loss/Brier 1.022728/0.608988; legacy Dixon--Coles 0.985056/0.587594. Target reads = 1 and selection reopened = false. | **Negative external result.** Preserve the frozen model and report the reversal; do not retune on 2014. |

The one-time 2014 audit is complete and may only be cache-verified. Promotion eligibility recorded
before the audit is not evidence of external superiority after the adverse result.

## 4. Review-time decision rules

- **Retain:** evidence directly supports the bounded wording and all linked gates pass.
- **Narrow:** the mechanism is valid but empirical superiority is weak, mixed, or non-significant.
- **Negative result:** the component was fairly tested and failed; retain the test and remove it from
  deployment.
- **Withdraw:** likelihood/code contradicts the claim, information leakage exists, or the evidence
  is not independently recomputable.
- **Exploratory hypothesis:** concept is plausible but identification or sample size is insufficient.

The final manuscript must update this ledger with artifact paths and hashes. No abstract or
conclusion statement may be stronger than its ledger decision.
