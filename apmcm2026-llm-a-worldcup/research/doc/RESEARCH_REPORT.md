# Research Report: Evidence-Bound World Cup Forecasting

**Research state:** frozen development study, completed one-shot 2014 external audit, and promoted
official-format 2026 scenario simulation

**Evidence date:** 18 July 2026

**Primary manuscript:** `doc/PAPER.tex`

**Machine-readable ledger:** `doc/tables/paper_evidence.json`

**Artifact manifest:** `doc/artifact_manifest.json`

## 1. Executive judgement

This project reports three different estimands and never pools them into one headline number:

1. seven expanding-origin development folds ending in 2010;
2. a one-shot 64-match 2014 external audit; and
3. an illustrative 2026-format Monte Carlo scenario using frozen historical team profiles.

The development evidence supports a bounded improvement in WDL log loss and Brier score over the
legacy mirrored Dixon--Coles comparator. The external audit does not reproduce that advantage:
advanced calibrated log loss is 1.02273 versus 0.98506 for legacy Dixon--Coles, and Brier score is
0.60899 versus 0.58759. This adverse result is retained. The 2014 labels were read exactly once,
selection was not reopened, and all later audit access is cached verification only.

The corrected tournament engine implements all 495 Annex C third-place mappings and the official
M73--M104 successor graph. Its 220,000-tournament output is a mechanism-certified scenario, not a
factual forecast of the 2026 entrants.

## 2. Frozen identities and evidence boundary

| Item | Frozen value |
|---|---|
| Pre-2014 Git tag | `rc0-pre2014` |
| Frozen commit | `fd9186e3277efb5ab8125c6c8348dc1b337f3ef7` |
| Development matches | 772, through 2010 |
| Outer OOF rows | 412 across seven editions, 1986--2010 |
| Candidate rows | 252 from a 36-configuration contract |
| Maximum selection year | 2010 |
| Final candidate ID | `f12f0fc20700df70743ef3b4f27089c45f45989a93fa2bd19c1886c98d609a81` |
| Final-fit hash | `10238a4c6d368e50acb9ac28e4523dbd1b4f1b64710667ba0e8eb511b8774f53` |
| Protocol-deviation ledger hash | `6f487cf1331d45e2591bcc905444bb2f1214481d2fab7a5c3c3c47f40e75cf0c` |

No 2014 outcome entered preprocessing, feature selection, hyperparameter search, calibration,
promotion, or the final blind-prediction file. The audit certificate binds the prediction file,
truth file, model state, configuration, source commit, and target-read count.

## 3. Development validation

Differences below are candidate minus legacy Dixon--Coles; lower losses are better.

| Metric | Equal-edition mean difference | 95% edition-bootstrap interval | Fold wins |
|---|---:|---:|---:|
| WDL log loss | -0.020423 | [-0.036524, -0.003451] | 5/7 |
| Multiclass Brier | -0.013022 | [-0.023749, -0.002668] | 5/7 |
| Scoreline NLL | -0.007952 | [-0.028227, 0.010524] | 5/7 |
| Total-goal CRPS | +0.011554 | [-0.008383, 0.035073] | 4/7 |

The development claim is limited to WDL proper scores. Exact-score evidence is inconclusive,
total-goal CRPS is adverse on average, and calibration changes are mixed. With seven outer
environments, intervals are edition-cluster sensitivity summaries, not large-sample proof.

The A0--A8 program contains 126 paired cells and 7,416 predictions. Removing team-effect shrinkage
(A7) worsens mean log loss by about 0.149, while the asymmetric A8 diagnostic has maximum swap error
0.47083 and is structurally ineligible. A2/A3 are negative results for the available player/era
representations; A4/A5 are zero because the corresponding components were not selected.

## 4. One-shot 2014 external audit

| Model/output | Accuracy | Macro-F1 | Log loss | Brier | Calibration error |
|---|---:|---:|---:|---:|---:|
| Advanced calibrated | 0.53125 | 0.38899 | 1.02273 | 0.60899 | 0.11278 |
| Advanced raw | 0.53125 | 0.38899 | 1.02273 | 0.60899 | 0.11278 |
| Legacy Dixon--Coles | 0.54688 | 0.39909 | 0.98506 | 0.58759 | 0.06627 |
| Independent Poisson | -- | -- | 0.98807 | 0.58848 | -- |
| Official multinomial logit | -- | -- | 0.99264 | 0.59193 | -- |

The advanced model also has scoreline NLL 3.17821 and total-goal CRPS 1.03001, versus 3.02363 and
0.97825 for legacy Dixon--Coles. Both probability argmax rules have zero draw recall. The frozen
cost-sensitive decision rule raises draw recall to 0.38462 and macro-F1 to 0.40378, but lowers
accuracy to 0.42188; it is a utility decision and has no proper-score claim.

This audit falsifies external superiority. It does not authorize a new model search. Any improved
model must receive a new version and a new future holdout; the completed 2014 audit cannot be reused
as if it were blind.

## 5. Corrected 2026-format scenario

The first 5,000-run simulator was invalid because it used an adjacency knockout construction. Its
numbers remain permanently superseded. The replacement engine was separately implemented and had
to pass exact Annex C and M73--M104 graph checks before simulation.

The promoted run completed 220,000 tournaments (22,880,000 simulated matches). Maximum advancement
and champion Monte Carlo standard errors were 0.001066 and 0.000999, respectively; champion mass
sums to one. Mean tournament goals were 181.1653 with Monte Carlo standard error 0.02875. The result
uses historical profile labels as stand-ins, so champion/team probabilities are scenario outputs,
not factual 2026 forecasts.

The uncertainty decomposition reconstructs the declared total to numerical tolerance:

| Component | Variance share |
|---|---:|
| Parameter | 54.99% |
| Regime | 24.48% |
| Path/match | 20.54% |

These are model-conditional scenario shares, not frequentist coverage guarantees.

## 6. Structural and reproducibility gates

- 24 stress rows across 12 scenarios and two seeds passed.
- Maximum swap and simplex errors were 1.67e-16 and 2.22e-16.
- Extreme finite-grid mass loss triggers an explicit refusal.
- Whole-edition parameter bootstrap completed 200/200 refits.
- The simulator uses 48 teams, 12 groups of four, 104 matches, a round of 32, and one champion.
- Same seed reproduces the same payload; different seeds vary only within declared Monte Carlo error.
- The formal audit records one target read, no reselection, and cached-only repeat access.
- The manuscript builder hashes every source and all 33 generated table/figure artifacts.

## 7. Claim boundaries

Supported: a bounded historical-development WDL proper-score gain, a coherent and equivariant joint
probability interface, exact tournament-graph implementation, and reproducible provenance.

Not supported: universal model superiority, 2014 external superiority, factual 2026 entrants or
outcomes, causal claims, distribution-free coverage, or reuse of 2014 for tuning. The adverse audit
and null ablations are part of the result, not defects to hide.

## 8. Reproduction

From the repository root:

```powershell
uv sync --frozen --extra dev
uv run python main.py
uv run python verify.py
uv run python scripts/build_paper_artifacts.py --project-root . --output-root doc
uv run pytest -q
uv run ruff check src tests scripts
Set-Location doc
latexmk -xelatex -interaction=nonstopmode -halt-on-error PAPER.tex
```

The formal advanced evidence under `temp/advanced_evidence` is append-only scientific evidence and
must not be regenerated after the one-shot audit. Verification must use the cached certificates.

## 9. Completion statement

The development, external-audit, and official-format scenario layers are complete under their
declared boundaries. The final scientific conclusion is deliberately asymmetric: the method looked
better in historical nested validation, failed to beat the legacy comparator in the one-shot 2014
audit, and remains useful as a rigorously bounded scenario and reproducibility system.
