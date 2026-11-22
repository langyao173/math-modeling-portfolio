# Local Recourse Multi-Start Summary

> **Historical snapshot (superseded).** This 2026-06-30 note documents the local
> predecessor used by the promotion guard. Any “current” or “final” wording
> below does not describe the submitted schedule. Use `FINAL_REPORT.md` and
> `FINAL_METRICS.json` for the authoritative `full_cloudseed_peakguard` result.

Generated: `2026-06-30T06:50:00+00:00`

## Purpose

This run intentionally did not call the cloud platform. It extends the exact
recourse/CVaR polish stage with local multi-start and multiple job-order scans
so platform balance is preserved while improving the risk-facing schedule.

Command pattern:

```powershell
python src/07_solve_isling/solve_qubo_platform.py --solver local --gear balanced --max-jobs 0 `
  --dummy-cost 120 --dummy-only-cost 25 `
  --recourse-cvar-weight 0.10 `
  --recourse-cut-iterations 3 `
  --recourse-polish-passes 3 `
  --recourse-polish-starts all `
  --recourse-polish-orders all `
  --output-json qubo_solver_results_full_rec_cvar0p10_cut3_truepolish3_ms_local.json `
  --output-md qubo_solver_report_full_rec_cvar0p10_cut3_truepolish3_ms_local.md `
  --output-solution-csv qubo_accepted_actions_full_rec_cvar0p10_cut3_truepolish3_ms_local.csv `
  --output-risk-solution-csv qubo_risk_accepted_actions_full_rec_cvar0p10_cut3_truepolish3_ms_local.csv
```

## Local Ladder

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

## Full 1922-Spin Validation

The final local multi-start risk-accepted CSV has:

- rows: `588`
- unique jobs: `588`
- dummy actions: `7`
- one-hot violations: `0`
- risk-accepted source: `recourse_true_polish`
- CVaR95: `47.109637`
- expected recourse: `35.273095`

The 7 dummy actions remain unavoidable under the current action table because
those jobs have no real feasible action.

## Interpretation

- No platform balance was used for this refinement.
- Multi-start exact recourse polish improves the full CVaR95 from the previous
  cloud-integrated guarded result `48.764804` to `47.109637`.
- It improves over the earlier structural risk-accepted CVaR95 `51.512721` by
  `4.403084`.
- The result is local-only. A future cloud run is not necessary unless the
  contest workflow requires every final artifact to include a fresh platform
  task ID.

## Artifacts

- `reports/qubo_solver_results_120_rec_cvar0p10_cut3_truepolish3_ms_local.json`
- `reports/qubo_solver_results_240_rec_cvar0p10_cut3_truepolish3_ms_local.json`
- `reports/qubo_solver_results_360_rec_cvar0p10_cut3_truepolish3_ms_local.json`
- `reports/qubo_solver_results_full_rec_cvar0p10_cut3_truepolish3_ms_local.json`
- `reports/qubo_accepted_actions_full_rec_cvar0p10_cut3_truepolish3_ms_local.csv`
- `reports/qubo_risk_accepted_actions_full_rec_cvar0p10_cut3_truepolish3_ms_local.csv`
