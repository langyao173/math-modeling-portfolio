# Model Card: World Cup Joint-Goal Forecasting System

## Model identity

- **Deployed enhanced model:** `joint_poisson_static_history`
- **Required baselines retained:** aggregate OLS, match-level OLS, multinomial logistic regression
- **Frozen advanced candidate:** equivariant penalized count model selected only through 2010
- **Advanced candidate ID:** `f12f0fc20700df70743ef3b4f27089c45f45989a93fa2bd19c1886c98d609a81`
- **Data cutoff:** 2014-07-13
- **Production runtime:** deterministic Python
- **Optional workbench:** MATLAB R2026a, never required at inference

## Intended use

The system is intended for reproducible competition analysis of the supplied historical World Cup tables, methodological comparison, and demonstration of audited prediction interfaces. It is not intended for betting, live operations, real 2026 forecasting, roster decisions, or causal claims.

## Inputs and information sets

- `pre_match`: four strictly historical team-difference features for deployable outcome/joint-goal prediction.
- `half_time`: canonical stage, observed half-time total goals, and attendance for match-total updates.
- 2026 format metadata: 48 teams and 104 matches, used only for OOD disclosure.

No post-2014 ranking, fixture, squad, injury, coach, or match result is ingested.

## Frozen performance

| Task/model | Test | Key results |
|---|---|---|
| Q1 aggregate OLS | 2018 audit | prediction 152.553 vs 169; absolute error 16.447; LOOCV MAE 14.811 |
| Q2 official OLS | 2014, 64 matches | MAE 0.974; RMSE 1.169; \(R^2=0.554\) |
| Q3 official Logit | 2014, 64 matches | accuracy 0.531; log loss 0.993; Brier 0.592 |
| Static Joint Poisson | legacy 2014 audit-only | accuracy 0.531; log loss 0.989; Brier 0.589 |
| Frozen advanced calibrated | one-shot 2014 external audit | accuracy 0.531; log loss 1.023; Brier 0.609 |
| Legacy Dixon--Coles comparator | same one-shot 2014 audit | accuracy 0.547; log loss 0.985; Brier 0.588 |

In the separately frozen seven-fold advanced study, the calibrated candidate beat the legacy
Dixon--Coles comparator on WDL log loss and Brier score in 5/7 editions. Equal-edition mean
candidate-minus-baseline differences were -0.020423 and -0.013022. The one-shot 2014 audit reversed
the ranking. This is an adverse external result and no post-audit model reselection is permitted.

## Deployment decision

The zero-argument judge-facing profile deploys the scenario-compatible static-history joint
distribution and never reruns model search. The separately frozen advanced candidate is retained as
an audited research result, not silently substituted into the judge-facing profile after seeing
2014. Its pre-2014 promotion certificate records engineering eligibility; the later adverse audit
limits the performance claim rather than changing the frozen identity.

## Example scenario

BRA–GER is an illustrative stale-cutoff interface test, not an asserted 2026 fixture:

- home win 0.4805;
- draw 0.2635;
- away win 0.2560;
- expected goals 1.4904–1.0144;
- most likely score 1–1;
- conditional total-goal 90% interval [0, 5];
- OOD score 2.1949, including historical-feature and tournament-format shift.

## Invariants and traceability

- probabilities are nonnegative and sum to one;
- swapping teams swaps home/away probabilities;
- one normalized joint score matrix produces outcome and total-goal outputs;
- full-time total cannot be below observed half-time total;
- model and calibration files are listed in the manifest;
- every certificate hashes the actual fitted-state artifact, not a model name;
- deterministic prediction payloads exclude generation timestamps.

## Limitations and failure modes

- Historical data end in 2014 and contain strong era/format shift.
- Only 20 tournaments and 7 development folds are available.
- Draw recall is 0 for the advanced probability argmax and legacy Dixon--Coles classifier in the
  one-shot 2014 audit. The separate cost-sensitive decision rule reaches 0.385 draw recall with an
  accuracy trade-off and is not a probability-score improvement.
- Calibration-fit development diagnostics are not independent held-out evidence.
- Empirical half-time intervals make no exchangeability guarantee.
- Joint-goal intervals are conditional on fitted parameters and omit full parameter uncertainty.
- National-team identity inheritance and geopolitical history remain imperfectly represented.

## Artifacts

- Final manifest: `outputs/manifest.json`
- Deployed state: `outputs/models/problem3_joint_poisson_static_2014.json`
- Prediction: `outputs/predictions/2026_joint_goal_demo.json`
- Certificate: `outputs/certificates/2026_joint_goal_demo.json`
- Research gate: `research_outputs/tables/research_promotion_gate.json`
- Frozen advanced evidence: `temp/advanced_evidence/`
- One-shot audit: `temp/advanced_evidence/audit/2014_external_audit.json`
- Manuscript evidence ledger: `doc/tables/paper_evidence.json`
- Full audit: `doc/AUDIT.md`
