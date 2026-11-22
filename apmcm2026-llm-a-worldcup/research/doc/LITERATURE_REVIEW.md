# Literature Review and Blind-Spot Scan

## 1. Scope, search protocol, and evidence boundary

This document records a targeted, source-verified scoping review for the World Cup forecasting
study. It is not presented as a PRISMA systematic review and does not claim exhaustive database
coverage. Searches were conducted on 2026-07-18 through DOI/Crossref resolution, publisher and
institutional repositories, arXiv for explicitly labelled preprints, PMLR, NeurIPS, JMLR, ACM,
and FIFA's official site.

Search families covered:

- football score models: Poisson, Dixon-Coles, bivariate count, copula, Weibull, Skellam;
- dynamic/state-space strength, Elo, hierarchical and regularized team effects;
- player, coach, lineup, plus-minus, and in-play information;
- probabilistic calibration, proper scoring, classwise reliability, and sharpness;
- conformal/predictive intervals under time dependence and distribution shift;
- World Cup tournament simulation, seeding, incentives, and the 2026 format;
- temporal leakage, model cards, dataset documentation, and scientific reproducibility.

Inclusion required an original methodological paper or official source, a verified bibliographic
identity, and direct relevance. Blogs, Wikipedia, Reddit, betting-tip pages, and low-quality
application papers were excluded from the evidence base. Publications after the locked prediction
date can inform methods and the open research discussion, but cannot supply closed-track features,
selection labels, calibration data, or performance evidence.

Evidence grades:

- **A:** strong, directly relevant peer-reviewed primary evidence;
- **A-:** strong but narrower, older, or context-specific primary evidence;
- **B+:** useful peer-reviewed evidence with direct relevance but narrower identification or scope;
- **B:** useful supporting evidence, not sufficient for a central claim;
- **B/P:** useful supporting evidence currently verified through a preprint record;
- **P:** preprint/frontier evidence only;
- **O:** official rule or format source.

## 2. Verified evidence matrix

### 2.1 Joint score-count models and dependence

| Source | Main verified contribution | Grade |
|---|---|---:|
| Maher, 1982, [DOI](https://doi.org/10.1111/j.1467-9574.1982.tb00782.x) | Attack/defence Poisson model and departures from independent scoring. | A |
| Dixon & Coles, 1997, [DOI](https://doi.org/10.1111/1467-9876.00065) | Time weighting and low-score dependence correction. | A |
| Rue & Salvesen, 2000, [DOI](https://doi.org/10.1111/1467-9884.00243) | Bayesian dynamic GLM with time-varying team skill. | A |
| Crowder et al., 2002, [DOI](https://doi.org/10.1111/1467-9884.00308) | Latent stochastic attack/defence evolution and approximate dynamic inference. | A |
| Karlis & Ntzoufras, 2003, [DOI](https://doi.org/10.1111/1467-9884.00366) | Correlated bivariate Poisson scores and diagonal inflation for draws. | A |
| McHale & Scarf, 2007, [DOI](https://doi.org/10.1111/j.1467-9574.2007.00368.x) | Bivariate discrete distributions with a general dependence structure. | A |
| McHale & Scarf, 2011, [DOI](https://doi.org/10.1177/1471082X1001100303) | International-score dependence can vary with competitive imbalance and become negative. | A |
| Baio & Blangiardo, 2010, [DOI](https://doi.org/10.1080/02664760802684177) | Hierarchical Poisson attack/defence model and mixture remedy for overshrinkage. | A- |
| Koopman & Lit, 2015, [DOI](https://doi.org/10.1111/rssa.12042) | Dynamic bivariate Poisson model in state-space form. | A |
| Karlis & Ntzoufras, 2009, [DOI](https://doi.org/10.1093/imaman/dpn026) | Bayesian goal-difference modelling using the Skellam distribution. | A- |
| Boshnakov, Kharrat & McHale, 2017, [DOI](https://doi.org/10.1016/j.ijforecast.2016.11.006) | Copula-linked bivariate Weibull count model with out-of-sample evaluation. | A |
| Groll et al., 2018, [DOI](https://doi.org/10.1515/jqas-2017-0067) | Sparse bivariate Poisson modelling and tournament simulation. | A- |
| Michels, Ötting & Karlis, 2025, [DOI](https://doi.org/10.1093/jrsssc/qlae050) | Shows Dixon--Coles is a Sarmanov-family special case and develops broader marginal/dependence constructions. | A |

### 2.2 Dynamic ratings, international tournaments, and ML hybrids

| Source | Main verified contribution | Grade |
|---|---|---:|
| Hvattum & Arntzen, 2010, [DOI](https://doi.org/10.1016/j.ijforecast.2009.10.002) | Elo plus ordered logit; strong non-market baseline, but weaker than market odds. | A |
| Baker & McHale, 2015, [DOI](https://doi.org/10.1111/rssa.12060) | Longitudinal time-varying football ratings. | A- |
| Dyte & Clarke, 2000, [DOI](https://doi.org/10.1057/palgrave.jors.2600997) | Ratings-based Poisson World Cup simulation. | B+ |
| Groll, Schauberger & Tutz, 2015, [DOI](https://doi.org/10.1515/jqas-2014-0051) | Team-specific regularized Poisson regression for World Cup 2014. | A- |
| Tsokos et al., 2019, [DOI](https://doi.org/10.1007/s10994-018-5741-1) | Bradley-Terry and hierarchical Poisson comparisons under future-oriented validation. | A |
| Berrar, Lopes & Dubitzky, 2019, [DOI](https://doi.org/10.1007/s10994-018-5747-8) | Domain knowledge, recency-aware features, kNN/XGBoost, and a future-event challenge design. | A |
| Schauberger & Groll, 2018, [DOI](https://doi.org/10.1177/1471082X18799934) | Random forests adapted for international score and ordinal outcome prediction. | A- |
| Groll et al., 2019, [DOI](https://doi.org/10.1515/jqas-2018-0060) | Hybrid random forest with Poisson ability parameters and future World Cup validation. | A |
| Berrar, Lopes & Dubitzky, 2024, [DOI](https://doi.org/10.1007/s10994-024-06625-9) | Exact-score and WDL probabilities for 736 genuinely future matches. | A |
| Wheatcroft, 2021, [DOI](https://doi.org/10.3233/JSA-200462) | Predicting intermediate match statistics for probabilistic outcomes. | A- |
| Macrì-Demartino, Egidi & Torelli, 2026, [DOI](https://doi.org/10.1093/jrsssc/qlag032) | Weighted dynamic Bayesian model with separate time-varying attack/defence commensurability. | A- |
| Rezaei & Samadi, 2026, [arXiv](https://arxiv.org/abs/2606.24171) | Direct 2026 World Cup comparator using recent Elo histories, sufficient dimension reduction, Poisson double regression, and 2018/2022 out-of-sample RPS evaluation. | P |
| Nguyen & Tran, 2026, [arXiv](https://arxiv.org/abs/2607.01722) | Adaptive Glicko-2 preprint combining rating uncertainty, structural shocks, ordered-logit draws, and season simulation. | P |

### 2.3 Player, lineup, coach-adjacent, and in-play information

| Source | Main verified contribution | Grade |
|---|---|---:|
| Whitaker et al., 2021, [DOI](https://doi.org/10.1111/rssc.12454) | Variational Bayesian player abilities and lineup-dependent goal modelling. | A |
| Kharrat, McHale & Peña, 2020, [DOI](https://doi.org/10.1016/j.ejor.2019.11.026) | Ridge-regularized plus-minus on goal/xG/expected-points segments. | A |
| Arntzen & Hvattum, 2021, [DOI](https://doi.org/10.1177/1471082X20929881) | Player ratings do not consistently outperform team ratings; combination is not automatically useful. | A- |
| Zou, Song & Shi, 2020, [DOI](https://doi.org/10.3390/app10082904) | Bayesian in-play outcome updating. | B |
| Klemp, Wunderlich & Memmert, 2021, [DOI](https://doi.org/10.1038/s41598-021-03157-3) | Conditional value of rich event and positional information for in-play forecasts. | A- |
| Robberechts, Van Haaren & Davis, 2019, [preprint](https://arxiv.org/abs/1906.05029) | Bayesian in-game win probability. | B/P |
| Clegg, Song & Cartlidge, 2026, [arXiv](https://arxiv.org/abs/2605.16066) | Market-calibrated accelerated-failure-time in-play model; useful evidence that lawful kickoff-market and post-shot information can dominate model-form refinements. | P |

### 2.4 Proper scoring and calibration

| Source | Main verified contribution | Grade |
|---|---|---:|
| Brier, 1950, [DOI](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2) | Canonical proper probability score. | A |
| Murphy, 1973, [DOI](https://doi.org/10.1175/1520-0450(1973)012%3C0595:ANVPOT%3E2.0.CO;2) | Reliability-resolution-uncertainty decomposition. | A |
| Gneiting & Raftery, 2007, [DOI](https://doi.org/10.1198/016214506000001437) | General theory of strictly proper scoring rules. | A |
| Gneiting, Balabdaoui & Raftery, 2007, [DOI](https://doi.org/10.1111/j.1467-9868.2007.00587.x) | Calibration, sharpness, PIT, and diagnostic principles. | A |
| Guo et al., 2017, [PMLR](https://proceedings.mlr.press/v70/guo17a.html) | Temperature scaling and modern-network miscalibration. | A |
| Kull et al., 2019, [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html) | Dirichlet calibration for multiclass probabilities. | A |
| Wheatcroft, 2021, [DOI](https://doi.org/10.1515/jqas-2019-0089) | RPS is not automatically preferable to ignorance/log score in football evaluation. | A- |
| Ekstrøm et al., 2021, [DOI](https://doi.org/10.3233/JSA-200454) | Tournament Rank Probability Score. | B+ |

### 2.5 Predictive intervals under dependence and shift

| Source | Main verified contribution | Grade |
|---|---|---:|
| Xu & Xie, 2021, [PMLR](https://proceedings.mlr.press/v139/xu21h.html) | Sequential conformal intervals with time-series-qualified approximate coverage. | A |
| Gibbs & Candès, 2021, [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html) | Adaptive conformal miscoverage control under change. | A |
| Barber et al., 2023, [DOI](https://doi.org/10.1214/23-AOS2276) | Weighted conformal prediction and degradation bounds beyond exchangeability. | A |
| Gibbs & Candès, 2024, [JMLR](https://jmlr.org/papers/v25/22-1218.html) | Online conformal inference under arbitrary distribution shifts. | A |

### 2.6 Tournament mechanism and the 2026 format

| Source | Main verified contribution | Grade |
|---|---|---:|
| FIFA, [official format](https://www.fifa.com/en/articles/article-fifa-world-cup-2026-mexico-canada-usa-new-format-tournament-football-soccer) | 48 teams, 12 groups of four, 104 matches, and a round of 32. | O |
| FIFA, 2026, [official regulations](https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/FWC2026_regulations_EN.pdf) | Match M73--M104 topology and all 495 Annex C assignments for the eight best third-placed teams. | O |
| Koning et al., 2003, [DOI](https://doi.org/10.1016/S0377-2217(02)00683-5) | Championship simulation from match-scoring intensities. | A- |
| Scarf & Yusof, 2011, [DOI](https://doi.org/10.1111/j.1467-9574.2010.00471.x) | Interactions among strength, seeding, and World Cup tournament structure. | A- |
| Guyon, 2020, [DOI](https://doi.org/10.3233/JSA-200414) | Formal analysis of group-design collusion incentives. | B+ |
| Devriesere, Csató & Goossens, 2025, [DOI](https://doi.org/10.1016/j.ejor.2024.10.044) | Operational-research review separating tournament format, seeding, draw, scheduling, and ranking under efficacy, fairness, and attractiveness criteria. | A- |
| Csató & Gyimesi, 2026, [DOI](https://doi.org/10.1016/j.ejor.2025.11.025) | Monte Carlo study of stakeless matches and an imbalanced-group alternative for the expanded format. | A- |
| Csató, 2026, [DOI](https://doi.org/10.1111/itor.70194) | Parametric optimization of constrained World Cup group draws. | A- |
| Qi, 2026, [arXiv](https://arxiv.org/abs/2606.19554) | Identifies the 495-configuration coupling and proposes an alternative four-section bracket; preprint evidence only. | P |

### 2.7 Leakage, auditability, and reporting

| Source | Main verified contribution | Grade |
|---|---|---:|
| Kaufman et al., 2012, [DOI](https://doi.org/10.1145/2382577.2382579) | Leakage taxonomy and detection. | A |
| Kapoor & Narayanan, 2023, [DOI](https://doi.org/10.1016/j.patter.2023.100804) | Temporal/preprocessing leakage as a reproducibility failure in ML science. | A |
| Mitchell et al., 2019, [DOI](https://doi.org/10.1145/3287560.3287596) | Model cards and structured limitation reporting. | A |
| Gebru et al., 2021, [DOI](https://doi.org/10.1145/3458723) | Dataset provenance and intended-use documentation. | A |

## 3. What is known, what remains open

### 3.1 Known knowns

- One joint score distribution is the coherent source for scorelines, WDL probabilities, expected
  goals, totals, and tournament simulations.
- Independent Poisson, Dixon-Coles, bivariate Poisson, dynamic ratings, hierarchical shrinkage,
  Elo, and direct outcome models are mandatory prior-art baselines.
- Player and lineup features are plausible but do not have a universal empirical advantage.
- Accuracy is insufficient; proper scores, classwise reliability, and score-distribution metrics
  are necessary.
- Random match-level K-fold validation is invalid for a temporally evolving World Cup process.
- Ordinary split-conformal finite-sample coverage does not survive this nonexchangeable setting by
  assertion alone.
- The 2026 48-team/104-match mechanism is a structural distribution shift, not merely a longer
  32-team edition.

### 3.2 Known unknowns

- Which dependence family transfers best to sparse international tournament data.
- Whether attack and defence evolve at distinct rates or exhibit discontinuous regime changes.
- Whether recent rating trajectories contain stable information beyond current rating level once
  candidate selection is nested by tournament edition.
- Stable incremental value of player/coach/lineup information at each lawful information set.
- Calibration stability with only a few independent tournament environments.
- Transportability of per-match scoring intensity to an expanded field and new knockout round.
- Useful interval coverage under temporal and institutional shift.
- Whether ensemble weights are stable when candidate models are strongly correlated.
- Whether apparent tournament gains reflect score-model skill or bracket luck.
- Whether the official 495-case Annex C mapping creates material path inequality relative to
  counterfactual bracket designs; this is a mechanism question, not evidence that an alternative
  rule is normatively superior.

### 3.3 Unknown knowns that must become tests

- Source `home`/`away` often means row order on neutral ground; true host status is separate.
- Confirmed lineups are a `kickoff` input, not a generic pre-match feature.
- Same-match player events are post-treatment leakage for pre-match prediction.
- Missing player information is informative and historically nonuniform.
- Penalty advancement is not the statistical full-time score result.
- Extra time changes exposure and should not be silently treated as 90-minute regulation.
- Group-to-knockout selection changes opponent distribution.
- Tactical incentives can depend on stage, current score, and other group results.
- Team naming, predecessor states, and historical record revisions can alter dynamic ratings.
- Host, travel, altitude, rest, and era are partially confounded in this small dataset.

### 3.4 Candidate unknown unknowns, explicitly hypotheses

- An executable **filtration graph** that converts information availability into a tested model
  invariant.
- Dependence conditioned on competitive imbalance and tournament incentive rather than a single
  global parameter.
- A **format-transport decomposition** separating team-strength, scoring-regime, and tournament
  mechanism uncertainty.
- Bracket-functional calibration for advancement/champion probabilities.
- Selective prediction or abstention for out-of-support team/format states.
- Partial-identification envelopes when 2026 transport assumptions are not identifiable.
- Stress-test optimization against downstream tournament decisions rather than average match loss.

## 4. Contradictions that require experiments

| Tension | Evidence | Required experiment |
|---|---|---|
| Positive versus negative score dependence | Karlis/Ntzoufras versus McHale/Scarf and the broader Sarmanov construction | Compare independent, DC, bivariate-positive, and a marginal-preserving Sarmanov family on identical nested folds. |
| Coherent score versus direct result model | Score coherence versus similar empirical performance in Tsokos et al. | Preserve coherence but do not presume better WDL accuracy. |
| Player ratings should help versus inconsistent gain | Whitaker/Kharrat versus Arntzen/Hvattum | Team-only, player-only, and combined outer-fold comparison with missing-lineup stress. |
| RPS convention versus criticism | Sports convention versus Wheatcroft | Predeclare log/Brier primary; retain RPS as secondary. |
| “Distribution-free” conformal versus temporal shift | Classical slogan versus Xu/Xie, Gibbs/Candès, Barber et al. | Report exact assumptions and empirical/long-run/weighted coverage separately. |
| Flexible ML versus parsimonious structural models | Hybrid RF evidence versus Elo/count simplicity | Bound the ML residual and require repeated outer-fold benefit. |
| Current rating versus rating trajectory | Classical Elo baselines versus Rezaei--Samadi SDR and adaptive Glicko proposals | Compare only inside nested temporal folds; do not import post-2014 match outcomes into the closed track. |
| Historical continuity versus 2026 shift | 32-team eras versus the official 48-team regulations and 495 Annex C assignments | Separate per-match transport from a format-certified tournament simulator and its uncertainty. |

## 5. Central synthesis

The literature rejects a claim that this project invented joint score modelling, dynamic team
strength, hierarchical shrinkage, player ratings, in-play updating, calibration, or tournament
simulation. Those are established building blocks.

The strongest defensible research contribution is their audited integration:

\[
\text{filtration integrity}
+\text{joint probabilistic coherence}
+\text{format-aware transport}
+\text{external temporal validation}
+\text{prediction-level certification}.
\]

With only about twenty pre-2014 tournament environments, the most scientifically valuable result
may be a negative one: a sophisticated component that fails the repeated external gate. The study
must therefore prefer bounded claims and complete failure reporting over architectural spectacle.
