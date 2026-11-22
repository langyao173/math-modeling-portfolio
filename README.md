# Mathematical modelling portfolio

Competition work in operations research, statistics, machine learning and simulation by **Haotong (Welt) Yang**
(University of Southampton / Jiangxi University of Finance and Economics), released with code, data
pointers and — for every project — a post-competition extension that tests the original work against
evidence it had not seen: new data, external benchmarks, exact solvers or stricter validation protocols.

Each folder is a self-contained project with its own README, figures
that summarise the extensions, a licence and reproduction commands. Every project also ships an **extended
edition** of its paper (`*_extended.pdf`, linked at the top of each README): the submitted document, unchanged,
followed by a Part II, in English, that writes up the post-competition extension with its tables, figures and references.

The extensions use four kinds of evidence. **Outcomes that arrived later**: the 2018–2026 World Cups, the
2024–2025 pet-population statistics and China's official 2025 energy statistics. **Exact references**: a
MILP certificate for the scheduling QUBO, exact density-matrix scoring of 1 520 candidate compilations, and exact
optima and LP bounds for every dispatch period. **Independent data**: three public randomised marketing
experiments, 1 088 annual series of the M4 competition, and the published study behind the beer data
together with its consumer ratings. **Stricter validation**: leave-one-level-out and leave-one-environment-out
tests, conformal calibration under shift, customer-level bootstraps and 300 fresh simulation seeds. Each
extension states where the original conclusion holds, where it fails, and the boundary of its own evidence.

| Repository | Competition | Core methods | Post-competition extension (headline) |
|---|---|---|---|
| [`apmcm2026-b-chip-microchannel`](apmcm2026-b-chip-microchannel) | APMCM 2026 (Chinese track), Problem B | grey-box heat-transfer model, RSM-3 + GPR surrogates, NSGA-II, min–max regret, Sobol | leave-one-level-out validation shows $R^{\ast}$ is an exact cubic and GPR intervals collapse on unseen levels; the recommendation holds in 100 % of surrogate-posterior draws; redoing the tolerance analysis on 1 000 posterior surfaces confirms it (CV ≤ 1.06 %, γ dominant in every draw) but shows the paper's GPR understates the $R^{\ast}$ and φ spread by 16–28 % |
| [`qpool-quantum-ridepooling`](qpool-quantum-ridepooling) | 1st Quantum Computing Application Scenario Challenge 2026 | conflict-graph QUBO, exact reduction, int8 Ising compilation (SPQC-550), equal-energy polish | precision audit of 435 dispatch periods: 68 % solved by reduction alone; polish cuts quantisation loss from 1 554 to 15 vehicle-minutes; finer precision is not monotonically better; at 4–12.5× the public trace's demand the reduced core reaches thousands of binaries and the 549-spin device fits only 9 % of the larger fleet's 60-s periods at 8×, while HiGHS proves every one of 1 331 optima within 1.8 s |
| [`lxb-noise-adaptive-quantum-compiler`](lxb-noise-adaptive-quantum-compiler) | Noise-adaptive quantum compilation challenge (量旋杯) 2026 | noise-aware layout and routing, measurement-channel resynthesis, row-phase-equivalent Toffoli rewriting | generalisation to unseen circuit sizes and to QASMBench: +0.108 and +0.065 success over the RouteCQC baseline; deadline guard removes timeout failures; exact density-matrix scoring of 1 520 candidates in 57 circuits shows selection is saturated (best in pool +0.0004 over the submitted choice), so further gains must come from candidate generation |
| [`apmcm2026-llm-a-worldcup`](apmcm2026-llm-a-worldcup) | APMCM LLM Competition 2026, Problem A | regression, multinomial logit, joint-goal Poisson/Dixon–Coles models | prospective test on all 232 matches of the 2018, 2022 and 2026 World Cups: the frozen contest logit beats the base rate (RPS 0.210 vs 0.235) but trails an Elo benchmark (0.191); the gap comes from balanced class weights — refitting the same four features for probabilities gains 0.064 log loss [0.017, 0.112] and ties Elo, and a history + Elo logit is best on every proper score (not significant against Elo) |
| [`fintechathon2026-credit-score-stability`](fintechathon2026-credit-score-stability) | Shenzhen International FinTechathon 2026, Task 1 | reference-scale features, near-stable updates, capacity-protected review lists, review workbench | the guard's separating threshold is a free design choice (exact-L2 rule: 68 scores touched vs 14; "fewest broken promises": 14 → 11 broken ±1 pp promises, identical AUC); replaying the stabiliser variants shows the ±1 pp radius sits on a flat stretch, and clipping every update at ±2 pp instead of gating keeps AUC while cutting worst-cohort tail instability by 39 % (5.29 → 3.25 pp) |
| [`mathorcup2026-bigdata-a-beer`](mathorcup2026-bigdata-a-beer) | MathorCup 2026 Big Data, Problem A | dual-representation TabICL fusion, within-style stability, copula generation, margin-controlled transfer | under the protocol of the published reference study (Nat. Commun. 2024) the fusion design beats a re-run GBR in 9 of 10 splits; with Extra Trees branches and a beer-style prior it reaches macro $R^2$ 0.238 (published GBR: 0.21), and the paper's key compound 4-vinylguaiacol also tracks independent RateBeer consumers' clove descriptors within style (ρ ≈ 0.25) |
| [`simulation-challenge2026-a-energy-transition`](simulation-challenge2026-a-energy-transition) | National Simulation Modeling Application Challenge 2026, Problem A | multi-agent energy-transition model (Simulink / AnyMath), security-constrained investment, hourly coordination | 2025 out-of-sample check: aggregates within 1.4 %, wind and solar build-out under-predicted by 8.5–10 %; the gap identifies the initial pipeline throughput, not the construction lag, and re-anchoring to 2025 lowers 2060 net CO₂ by 7.3 % but raises unserved service 2.7× because the investment budget binds in every year |
| [`ising-challenge2026-datacenter-qubo`](ising-challenge2026-datacenter-qubo) | Ising Intelligent Computing Future Challenge 2026 | risk-aware compute–power co-scheduling as QUBO/Ising, LNS, scenario CVaR recourse | exact MILP certificate for all 588 jobs: the submitted schedule is within 1.6 % of the optimum at the same peak load and 2.8 % of the global optimum (37 % higher peak); annealing diagnostics locate the limits of the Ising route — the base energy anti-correlates with the objective (Spearman −0.89), and annealing even the exact objective with single-job moves reaches 0.39 % at best but 1.8 % in the median, because the path to the optimum climbs a 0.68-USD barrier |
| [`meituan2026-coupon-uplift`](meituan2026-coupon-uplift) | 6th Meituan Business Analytics Elite Contest | IPTW T-learner, isotonic CATE calibration, fairness-constrained budget allocation | on a real randomised e-mail experiment (Hillstrom, 64 000 customers) uplift targeting captures no more incremental visits than response targeting (36.1 % vs 36.1 % of the total at the top 30 %); across three public randomised campaigns (64 000–687 029 customers) and seven targeting rules, uplift wins only where likely buyers are not the persuadable ones (X5: every uplift learner beats response targeting in 10/10 splits), while the proposal's isotonic step gives the best-calibrated effect estimates on all three |
| [`shuwei2026-c-restocking`](shuwei2026-c-restocking) | 12th ShuWei Cup 2026 (autumn), Problem C | Kalman state-space model with HMM phases, split-conformal calibration, rollout scheduling with a hierarchical duration model | 300 fresh seeds per pool: the validation–test reversal of the scheduling policies is an environment effect, traced to one test environment; in a held-out environment the 90 % prediction ball of Problem 2 covers as little as 0.76, and adaptive conformal inference restores ≥ 0.86 everywhere for a 4 % larger mean radius |
| [`apmcm2024-c-pet-industry`](apmcm2024-c-pet-industry) | APMCM 2024, Problem C — **Third Prize** | combined trend forecasts, global demand and coupling models | one-year-ahead errors within ±3 %, but the trend extrapolation over-predicts cats by 7.3 % after two years and gets the sign of the dog trend wrong; on 1 088 real five-point series (M4 Demographic) the paper's linear trend is the weakest trended method and its OLS 95 % intervals cover only 72 % at three years, against 93 % for class-calibrated intervals |

## Conventions

* `paper/` holds the paper or technical write-up of the solution; competition code is kept as submitted
  (folder names may be Chinese), apart from fixes that the README of the project documents.
* Submitted documents are never edited. The extended edition next to each one reproduces it page for page as
  Part I and adds Part II (pages E-1, E-2, …) with the post-competition work; tables, figures and equations in
  Part II are numbered E1, E2, … so that they cannot be confused with those of the original. The LaTeX source of
  each extended edition is in the `extended_edition_src/` folder next to it.
* `extension/`, `results/extension/` and `figures/extension/` hold everything added after the contest;
  every number in an extension section is produced by the scripts there.
* Third-party data are referenced or downloaded by script unless their licence allows redistribution;
  row-level data derived from restricted sources are not included.

Code is released under the MIT licence; papers and figures under CC BY 4.0, unless a folder states
otherwise.
