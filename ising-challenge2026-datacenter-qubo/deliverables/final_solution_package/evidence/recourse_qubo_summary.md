# Recourse-Aware QUBO Summary

> **Historical snapshot (superseded).** This 2026-06-30 note records an earlier
> solver state. Any “best”, “current”, or “final” wording below is historical.
> The authoritative submitted schedule is the later `full_cloudseed_peakguard`
> result in `FINAL_REPORT.md` and `FINAL_METRICS.json`.

Generated: `2026-06-30T06:50:00+00:00`

## Model Change

The solver now supports recourse-aware QUBO coefficients. When
`--recourse-weight` or `--recourse-cvar-weight` is nonzero, the existing
closed-form renewable/grid/flex recourse evaluator is finite-differenced around
the baseline schedule to produce per-slot marginal recourse prices. Those
expected and CVaR marginal prices are normalized at the action level and added
as QUBO linear terms.

This is stronger than the earlier structural slot-risk proxy because the QUBO
coefficients now come from the same scenario dispatch model used for post-solve
expected recourse and CVaR reporting.

## Guardrail Added

The risk-accepted view now requires:

- one-hot feasible candidate,
- dummy actions no greater than baseline,
- true action cost within `--risk-max-action-cost-increase` of baseline,
- then minimum CVaR95.

This prevents a raw solver candidate from being selected as risk-accepted only
because it buys a small CVaR reduction with very large degrade/SLA action cost.

## Candidate-Aware Recourse Polish Added

The solver now also supports a true recourse local search through
`--recourse-polish-passes`. This pass keeps the one-hot encoding fixed and
searches one job's action at a time, but it evaluates candidates with the exact
closed-form expected/CVaR recourse evaluator instead of the structural QUBO
proxy. It uses the same true action-cost and dummy-action guard as the
risk-accepted selector.

The final tested setting is:

```powershell
--recourse-cvar-weight 0.10 --recourse-cut-iterations 3 --recourse-polish-passes 3
```

## Local Cut + True-Polish Ladder

| Jobs | View | Source | QUBO | True action cost | Expected recourse | CVaR95 | Peak MW | Dummy |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 120 | risk_accepted | greedy_polish | -9497.010175 | 14.824472 | 16.211732 | 21.123815 | 0.021465 | 0 |
| 240 | risk_accepted | recourse_true_polish | -18960.938353 | 22.781784 | 25.190561 | 32.570919 | 0.039758 | 0 |
| 360 | risk_accepted | recourse_true_polish | -28483.933935 | 29.539147 | 30.999830 | 41.193138 | 0.032237 | 0 |
| full | risk_accepted | recourse_true_polish | -46669.827287 | 7000038.294745 | 36.496139 | 48.764804 | 0.032237 | 7 |

This fixes the earlier 240/360 weakness: the QUBO-only recourse-linear model
kept choosing either the baseline under the cost guard or a high-cost degrade
schedule outside the guard, while exact recourse polish finds low-cost
delay/normal adjustments that lower CVaR directly.

## Final Full Cloud Run With Cut + True Polish

| Candidate | QUBO | True action cost | Expected recourse | CVaR95 | Peak MW | Dummy | One-hot |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | -46672.449222 | 7000038.159726 | 41.325326 | 56.042082 | 0.036735 | 7 | 0 |
| greedy_polish | -46676.609178 | 7000038.265897 | 38.096278 | 51.611607 | 0.027833 | 7 | 0 |
| recourse_cut_greedy | -46676.610233 | 7000038.267242 | 38.000084 | 51.613422 | 0.027833 | 7 | 0 |
| recourse_true_polish | -46669.827287 | 7000038.294745 | 36.496139 | 48.764804 | 0.032237 | 7 | 0 |
| cloud | -46623.291732 | 7000479.115777 | 38.439552 | 51.549123 | 0.033845 | 7 | 0 |
| cloud_polished | -46676.465425 | 7000038.282893 | 37.581577 | 51.234398 | 0.027115 | 7 | 0 |

Final accepted solution by QUBO energy: `recourse_cut_greedy`.

Final risk-accepted solution by guarded CVaR: `recourse_true_polish`.

## Local Multi-Start Recourse Polish

To preserve platform balance, the next refinement was run locally only with:

```powershell
--recourse-cvar-weight 0.10 --recourse-cut-iterations 3 --recourse-polish-passes 3 --recourse-polish-starts all --recourse-polish-orders all
```

| Jobs | Start mode | Risk source | True action cost | Expected recourse | CVaR95 | Peak MW | Dummy | One-hot | Attempts |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 120 | single-start | greedy_polish | 14.824472 | 16.211732 | 21.123815 | 0.021465 | 0 | 0 | 1 |
| 120 | multi-start | recourse_true_polish | 14.852758 | 15.353054 | 19.740421 | 0.023863 | 0 | 0 | 8 |
| 240 | single-start | recourse_true_polish | 22.781784 | 25.190561 | 32.570919 | 0.039758 | 0 | 0 | 1 |
| 240 | multi-start | recourse_true_polish | 22.761385 | 24.694846 | 32.074622 | 0.033906 | 0 | 0 | 8 |
| 360 | single-start | recourse_true_polish | 29.539147 | 30.999830 | 41.193138 | 0.032237 | 0 | 0 | 1 |
| 360 | multi-start | recourse_true_polish | 29.521798 | 30.464940 | 40.509510 | 0.040521 | 0 | 0 | 8 |
| full | single-start | recourse_true_polish | 7000038.294745 | 36.496139 | 48.764804 | 0.032237 | 7 | 0 | 1 |
| full | multi-start | recourse_true_polish | 7000038.331944 | 35.273095 | 47.109637 | 0.032157 | 7 | 0 | 12 |

The full multi-start result is local-only and does not consume platform
balance. It is the best current risk-facing schedule in this workspace.

## Cloud Ladder: `--recourse-cvar-weight 0.10`

All cloud runs used `--use-credit --use-coupon`, `--dummy-cost 120`, and
`--dummy-only-cost 25`.

| Run | Spins | Cloud raw one-hot | Cloud repairs | Cloud QUBO | Cloud CVaR95 | Cloud-polished QUBO | Cloud-polished CVaR95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 120 | 375 | 0 | 0 | -9496.039234 | 20.914057 | -9497.010175 | 21.123815 |
| 240 | 735 | 0 | 0 | -18964.536152 | 32.263935 | -18966.342054 | 33.705120 |
| 360 | 1122 | 0 | 0 | -28484.974540 | 42.651973 | -28488.745659 | 42.908051 |
| full | 1922 | 0 | 0 | -46627.231559 | 51.506281 | -46677.869737 | 51.304567 |

## Structural Versus Recourse-Embedded Acceptance

| Jobs | Model | View | Source | One-hot | Dummy | QUBO | True action cost | Expected recourse | CVaR95 | Peak MW |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 120 | structural | accepted | greedy_polish | 0 | 0 | -9518.511525 | 14.825329 | 16.212546 | 21.124971 | 0.020773 |
| 120 | rec_cvar0.10 | accepted | greedy_polish | 0 | 0 | -9497.010175 | 14.824472 | 16.211732 | 21.123815 | 0.021465 |
| 120 | rec_cvar0.10 | risk_accepted | cloud | 0 | 0 | -9496.039234 | 14.867231 | 16.164183 | 20.914057 | 0.024316 |
| 240 | structural | accepted | greedy_polish | 0 | 0 | -18999.582202 | 162.399845 | 25.589420 | 33.643970 | 0.025041 |
| 240 | rec_cvar0.10 | accepted | greedy_polish | 0 | 0 | -18966.343372 | 162.399902 | 25.588879 | 33.643446 | 0.025041 |
| 240 | rec_cvar0.10 | risk_accepted | baseline | 0 | 0 | -18963.533763 | 22.701681 | 28.005516 | 36.711471 | 0.035980 |
| 360 | structural | accepted | greedy_polish | 0 | 0 | -28538.019227 | 169.149440 | 31.657711 | 42.913854 | 0.026059 |
| 360 | rec_cvar0.10 | accepted | greedy_polish | 0 | 0 | -28488.750562 | 169.147591 | 31.712134 | 43.020988 | 0.026059 |
| 360 | rec_cvar0.10 | risk_accepted | baseline | 0 | 0 | -28485.645691 | 29.442025 | 34.745894 | 46.707629 | 0.035980 |
| full | structural | accepted | greedy_polish | 0 | 7 | -46743.602159 | 7000038.270618 | 37.963716 | 51.520419 | 0.027583 |
| full | structural | risk_accepted | cloud_polished | 0 | 7 | -46743.381217 | 7000038.291803 | 37.825453 | 51.512721 | 0.026255 |
| full | rec_cvar0.10 | accepted | greedy_polish | 0 | 7 | -46678.018194 | 7000038.265897 | 38.096278 | 51.611607 | 0.027833 |
| full | rec_cvar0.10 | risk_accepted | cloud_polished | 0 | 7 | -46677.869737 | 7000038.281454 | 37.648627 | 51.304567 | 0.027090 |

## Economic Sanity Check

I also tested a strict cost-plus-recourse local objective with
`--risk-weight 0`, `--peak-weight 0`, `--one-hot-penalty 3000`,
`--cost-cap 2000`, and `--recourse-cvar-weight 1.00`.

| Jobs | Source | True action cost | Expected recourse | CVaR95 | Peak MW | Dummy |
|---:|---|---:|---:|---:|---:|---:|
| 120 | baseline | 14.813795 | 17.865190 | 23.380378 | 0.027685 | 0 |
| 240 | baseline | 22.701681 | 28.005516 | 36.711471 | 0.035980 | 0 |
| 360 | baseline | 29.442025 | 34.745894 | 46.707629 | 0.035980 | 0 |

This indicates that the current scenario recourse savings are smaller than the
large real action-cost penalties for aggressive degrade/SLA moves. The
structural risk/peak objective can still prefer those moves, but the
cost-guarded risk view blocks them from becoming risk-accepted unless the user
explicitly raises the action-cost tolerance.

## Interpretation

- The recourse-aware cloud pipeline succeeded on 120, 240, 360, and full 1922
  spins with zero raw one-hot violations.
- On the full instance, the best cloud-integrated risk-accepted result after
  cut + true polish is `recourse_true_polish`: CVaR95 `48.764804`, expected
  recourse `36.496139`, and 7 unavoidable dummy actions.
- The best current local-only risk-accepted result is also
  `recourse_true_polish`: CVaR95 `47.109637`, expected recourse `35.273095`,
  and 7 unavoidable dummy actions.
- The full cut + true-polish risk-accepted CVaR95 improves over the earlier
  structural risk-accepted CVaR95 (`51.512721`) by `2.747917`.
- It also improves over the first recourse-linear cloud-polished CVaR95
  (`51.304567`) by `2.539763`.
- The local multi-start result improves over the earlier structural
  risk-accepted CVaR95 (`51.512721`) by `4.403084`, without using cloud balance.
- The QUBO-accepted recourse-aware solution is not the risk-best solution; the
  report preserves both views.
- The next modeling improvement should add a wider beam or stochastic restarts
  locally before any additional cloud spending.

## Artifacts

- `reports/qubo_solver_report_120_rec_cvar0p10_cloud.md`
- `reports/qubo_solver_results_120_rec_cvar0p10_cloud.json`
- `reports/qubo_solver_report_240_rec_cvar0p10_cloud.md`
- `reports/qubo_solver_results_240_rec_cvar0p10_cloud.json`
- `reports/qubo_solver_report_360_rec_cvar0p10_cloud.md`
- `reports/qubo_solver_results_360_rec_cvar0p10_cloud.json`
- `reports/qubo_solver_report_full_rec_cvar0p10_cloud.md`
- `reports/qubo_solver_results_full_rec_cvar0p10_cloud.json`
- `reports/qubo_accepted_actions_full_rec_cvar0p10_cloud.csv`
- `reports/qubo_risk_accepted_actions_full_rec_cvar0p10_cloud.csv`
- `reports/qubo_solver_report_full_rec_cvar0p10_cut3_truepolish3_cloud.md`
- `reports/qubo_solver_results_full_rec_cvar0p10_cut3_truepolish3_cloud.json`
- `reports/qubo_accepted_actions_full_rec_cvar0p10_cut3_truepolish3_cloud.csv`
- `reports/qubo_risk_accepted_actions_full_rec_cvar0p10_cut3_truepolish3_cloud.csv`
- `reports/local_recourse_multistart_summary.md`
- `reports/qubo_solver_results_full_rec_cvar0p10_cut3_truepolish3_ms_local.json`
- `reports/qubo_accepted_actions_full_rec_cvar0p10_cut3_truepolish3_ms_local.csv`
- `reports/qubo_risk_accepted_actions_full_rec_cvar0p10_cut3_truepolish3_ms_local.csv`
