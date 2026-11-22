# Cloud QUBO Scale Summary

> **Historical snapshot (superseded).** This 2026-06-30 note records predecessor
> scale experiments. Any “current” or “final” wording below applies only to that
> snapshot. The authoritative submitted schedule is the later
> `full_cloudseed_peakguard` result in `FINAL_REPORT.md` and `FINAL_METRICS.json`.

Generated: `2026-06-30T06:05:00+00:00`

## Payment Flag Correction

The earlier cloud failures did not indicate that the issued compute balance had
been consumed. They occurred because the task was submitted with `useCoupon`
enabled but `useCredit` disabled. The platform-provided compute balance is
account credit, so cloud runs must pass both `--use-credit` and `--use-coupon`
when using the current contest allocation.

## Dummy Penalty Correction

The original `dummy_cost=25` made avoidable dummy actions too cheap for the
cloud solver. Raising all dummy actions to `dummy_cost=120` removed avoidable
dummies in smaller neighborhoods, but made dummy-only jobs unattractive because
their one-hot-selected contribution became positive. The final tuned setting
therefore splits dummy costs:

- `dummy_cost=120` for jobs that have at least one real action.
- `dummy_only_cost=25` for jobs whose only feasible action is dummy.

This keeps dummy-only jobs selectable while preventing the solver from using
dummy actions as a cheap load-avoidance shortcut.

## Tuned Cloud Ladder

| Run | Jobs | Spins | Cloud raw one-hot violations | Cloud repairs | Cloud dummy actions | Cloud QUBO | Cloud-polished QUBO | Greedy QUBO | Cloud-polished CVaR95 | Greedy CVaR95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 120 | 120 | 375 | 0 | 0 | 0 | -9517.704745 | -9518.511525 | -9518.511525 | 21.124971 | 21.124971 |
| 240 | 240 | 735 | 0 | 0 | 0 | -18997.656748 | -18999.578925 | -18999.582202 | 33.739750 | 33.643970 |
| 360 | 360 | 1122 | 0 | 0 | 0 | -28534.030007 | -28538.010665 | -28538.019227 | 42.842726 | 42.913854 |
| full | 588 | 1922 | 0 | 0 | 7 | -46694.932249 | -46743.381217 | -46743.602159 | 51.512721 | 51.520419 |

## Before / After On Full 1922-Spin Instance

| Setting | Cloud raw one-hot violations | Cloud repairs | Cloud dummy actions | Cloud QUBO | Cloud CVaR95 |
|---|---:|---:|---:|---:|---:|
| `dummy_cost=25` | 0 | 0 | 40 | -45839.603281 | 48.685109 |
| `dummy_cost=120`, no split | 7 | 7 | 7 | -46031.830235 | 52.167963 |
| `dummy_cost=120`, `dummy_only_cost=25` | 0 | 0 | 7 | -46698.133515 | 51.300144 |

## Acceptance Views

The report now preserves two selected schedules:

- QUBO-accepted: lowest repaired QUBO energy.
- Risk-accepted: lowest CVaR95 among one-hot-feasible candidates whose dummy
  count is no greater than the baseline dummy count.

For the full 1922-spin run:

| View | Source | QUBO | Peak IT MW | Expected recourse | CVaR95 | Dummy actions |
|---|---|---:|---:|---:|---:|---:|
| QUBO-accepted | greedy_polish | -46743.602159 | 0.027583 | 37.963716 | 51.520419 | 7 |
| Risk-accepted | cloud_polished | -46743.381217 | 0.026255 | 37.825453 | 51.512721 | 7 |

## Interpretation

- All credit-enabled cloud runs succeeded, including the full 1922-spin instance.
- Tuned cloud runs eliminated avoidable dummy actions. The remaining 7 full-run
  dummy actions are unavoidable because those jobs have no real feasible action
  in the current action table.
- Cloud alone remains behind the deterministic one-job descent on QUBO energy,
  but cloud followed by the same greedy polish reaches essentially the same
  QUBO local optimum and gives a slightly better risk-facing schedule on the
  full instance.

## Recourse-Aware QUBO Follow-Up

The next modeling step has been implemented and measured in
`reports/recourse_qubo_summary.md`. The solver now supports
`--recourse-weight` and `--recourse-cvar-weight`, which embed finite-difference
expected/CVaR recourse marginal prices into QUBO linear coefficients. It also
supports `--recourse-cut-iterations` for iterative marginalization updates and
`--recourse-polish-passes` for exact candidate-aware recourse/CVaR coordinate
polish under the same action-cost guard.

The final full 1922-spin cloud-integrated run used
`--recourse-cvar-weight 0.10 --recourse-cut-iterations 3
--recourse-polish-passes 3`. Cloud task creation succeeded and the raw cloud
candidate had zero one-hot violations. The best guarded risk-accepted result is
`recourse_true_polish`: expected recourse `36.496139`, CVaR95 `48.764804`, and
7 unavoidable dummy actions. This improves the earlier full structural
risk-accepted CVaR95 `51.512721` by `2.747917`.

The recourse report also records a strict cost-plus-recourse sanity check. Under
that economic setting, the baseline schedule remains selected on 120/240/360
because the available recourse savings are smaller than the high real
degrade/SLA action-cost penalties.

## Balance-Preserving Local Follow-Up

After the cloud-integrated run, the exact recourse polish was extended locally
with `--recourse-polish-starts all --recourse-polish-orders all`. This did not
consume platform balance. The best current local-only full result is
`recourse_true_polish`: expected recourse `35.273095`, CVaR95 `47.109637`, and
7 unavoidable dummy actions. See `reports/local_recourse_multistart_summary.md`.

## Artifacts

- `reports/qubo_solver_report_120_d120_do25_cloudpolish.md`
- `reports/qubo_solver_report_240_d120_do25_cloudpolish.md`
- `reports/qubo_solver_report_360_d120_do25_cloudpolish.md`
- `reports/qubo_solver_report_full_d120_do25_cloudpolish.md`
- `reports/qubo_accepted_actions_full_d120_do25_cloudpolish.csv`
- `reports/qubo_risk_accepted_actions_full_d120_do25_cloudpolish.csv`
- `reports/recourse_qubo_summary.md`
- `reports/qubo_solver_report_full_rec_cvar0p10_cloud.md`
- `reports/qubo_accepted_actions_full_rec_cvar0p10_cloud.csv`
- `reports/qubo_risk_accepted_actions_full_rec_cvar0p10_cloud.csv`
- `reports/qubo_solver_report_full_rec_cvar0p10_cut3_truepolish3_cloud.md`
- `reports/qubo_accepted_actions_full_rec_cvar0p10_cut3_truepolish3_cloud.csv`
- `reports/qubo_risk_accepted_actions_full_rec_cvar0p10_cut3_truepolish3_cloud.csv`
- `reports/local_recourse_multistart_summary.md`
