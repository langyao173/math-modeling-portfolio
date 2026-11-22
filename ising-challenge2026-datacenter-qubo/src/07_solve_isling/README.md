# QUBO Platform Solver Stage

This stage builds a task-action QUBO over `data/processed/actions.parquet`.
Each candidate action is one binary variable. The model includes:

- one-hot penalties so each job selects one action,
- time-risk linear terms from price, carbon, flexibility-call, and renewable-shortfall signals,
- a peak-load quadratic term over selected action power vectors,
- optional first-order expected/CVaR recourse marginal-price linear terms,
- optional exact recourse/CVaR coordinate polish under a true action-cost guard.

Continuous grid/renewable/flex-shortfall recourse is evaluated after decoding.
The current recourse evaluator uses fixed stress scenarios, exact per-slot
closed-form dispatch, expected recourse cost, VaR, and CVaR. When
`--recourse-weight` or `--recourse-cvar-weight` is enabled, the same evaluator
is finite-differenced around the baseline schedule and embedded into QUBO
linear coefficients. `--recourse-cut-iterations` can update that
marginalization point with deterministic candidates, and
`--recourse-polish-passes` can do a final candidate-aware local search using the
true recourse/CVaR evaluator directly. Storage, unit commitment, DC power flow,
and integer redispatch remain outside this MVP recourse model until those
subproblems have separate validation.

## Run

From `TAOLIN-DC-SIM`:

```powershell
$wheel = Resolve-Path "..\介绍、模板、平台使用\plectrum_core-0.1.1-py3-none-any.whl"
$env:PYTHONPATH = $wheel.Path
python src/07_solve_isling/solve_qubo_platform.py --solver local --max-jobs 0 `
  --dummy-cost 120 --dummy-only-cost 25 `
  --output-json qubo_solver_results_full_local.json `
  --output-md qubo_solver_report_full_local.md `
  --output-solution-csv qubo_accepted_actions_full_local.csv `
  --output-risk-solution-csv qubo_risk_accepted_actions_full_local.csv
```

For a cloud attempt, set `PLECTRUM_API_KEY` in the current process and use
both `--use-credit` and `--use-coupon`. The platform-provided compute balance
is account credit, so omitting `--use-credit` can make the platform reject task
creation with an insufficient-balance error even when compute has been issued.

```powershell
python src/07_solve_isling/solve_qubo_platform.py --solver both --max-jobs 120 `
  --dummy-cost 120 --dummy-only-cost 25 --use-credit --use-coupon `
  --output-json qubo_solver_results_120.json `
  --output-md qubo_solver_report_120.md `
  --output-solution-csv qubo_accepted_actions_120.csv `
  --output-risk-solution-csv qubo_risk_accepted_actions_120.csv
```

For the predecessor balance-preserving local recourse refinement used by the
promotion guard:

```powershell
python src/07_solve_isling/solve_qubo_platform.py --solver local --gear balanced --max-jobs 0 `
  --dummy-cost 120 --dummy-only-cost 25 `
  --recourse-cvar-weight 0.10 `
  --recourse-cut-iterations 3 --recourse-polish-passes 5 `
  --recourse-polish-starts all --recourse-polish-orders all `
  --output-json qubo_solver_results_full_rec_cvar0p10_cut3_truepolish5_ms_local.json `
  --output-md qubo_solver_report_full_rec_cvar0p10_cut3_truepolish5_ms_local.md `
  --output-solution-csv qubo_accepted_actions_full_rec_cvar0p10_cut3_truepolish5_ms_local.csv `
  --output-risk-solution-csv qubo_risk_accepted_actions_full_rec_cvar0p10_cut3_truepolish5_ms_local.csv
```

For the historical full cloud-integrated recourse-aware QUBO experiment:

```powershell
python src/07_solve_isling/solve_qubo_platform.py --solver cloud --gear balanced --max-jobs 0 `
  --dummy-cost 120 --dummy-only-cost 25 `
  --recourse-cvar-weight 0.10 `
  --recourse-cut-iterations 3 --recourse-polish-passes 3 `
  --use-credit --use-coupon `
  --output-json qubo_solver_results_full_rec_cvar0p10_cut3_truepolish3_cloud.json `
  --output-md qubo_solver_report_full_rec_cvar0p10_cut3_truepolish3_cloud.md `
  --output-solution-csv qubo_accepted_actions_full_rec_cvar0p10_cut3_truepolish3_cloud.csv `
  --output-risk-solution-csv qubo_risk_accepted_actions_full_rec_cvar0p10_cut3_truepolish3_cloud.csv
```

The script records cloud failures in the report instead of failing the whole
run. Reports include both QUBO metrics and recourse/CVaR metrics for baseline,
greedy-polished, local Plectrum, and cloud candidates when available. The
risk-accepted view now requires one-hot feasibility, no more dummy actions than
baseline, and true action cost within `--risk-max-action-cost-increase` of the
baseline cost before minimizing CVaR.

## Controlled Exact MILP Baseline

The 30/60/100-job controlled baseline opens every real action for a nested,
seeded job subset, fixes all other jobs to the submitted schedule, and solves
the same expected-recourse plus discrete-CVaR95 objective with SciPy/HiGHS.
It also solves a comparator constrained not to exceed the submitted peak:

```powershell
python src/07_solve_isling/build_exact_controlled_baseline.py
python -m pytest tests/test_exact_controlled_baseline.py -q
```

The outputs are `deliverables/competition_report/tables/15_exact_controlled_baseline.csv`
and its companion JSON. They are a hash-frozen evidence snapshot containing
measured wall times and a generation timestamp, so repeated runs should be
numerically checked rather than expected to be byte-identical. These models
certify only the controlled neighborhoods; they do not certify global
optimality of the full 588-job problem.

## Final Deliverable

The current final descends from a historical 700-spin
`peak_shave_lns__cloud_balanced` run. The frozen evidence retains its task ID,
returned length, and energy summary, but not the raw spin vector; consequently,
the original cloud assignment cannot be replayed bit by bit from the package.
Promotion independently replays the resulting full 588-job post-polish
schedule and requires strict improvement in expected recourse, CVaR95, and
peak IT load. Re-run promotion and build the package without using cloud
balance:

```powershell
python src/07_solve_isling/promote_cloud_candidate.py
python src/07_solve_isling/build_competition_report.py
python src/07_solve_isling/build_final_deliverable.py
```

The builder validates the final risk-accepted schedule, writes
`deliverables/final_solution_package`, records file hashes in
`MANIFEST.json`, and creates
`deliverables/TAOLIN_DC_SIM_final_solution_package.zip`.
This evidence supports a bounded 700-spin same-QUBO CloudSolver contribution,
not a global cloud solve of the complete 1,929-variable schedule.

## Current Outputs

- `deliverables/final_solution_package/FINAL_SOLUTION.csv`
- `deliverables/final_solution_package/FINAL_REPORT.md`
- `deliverables/final_solution_package/VALIDATION.json`
- `deliverables/final_solution_package/MANIFEST.json`
- `deliverables/TAOLIN_DC_SIM_final_solution_package.zip`
- `reports/cloud_scale_summary.md`
- `reports/recourse_qubo_summary.md`
- `reports/local_recourse_multistart_summary.md`
- `reports/qubo_solver_report_120_d120_do25_cloudpolish.md`
- `reports/qubo_solver_results_120_d120_do25_cloudpolish.json`
- `reports/qubo_accepted_actions_120_d120_do25_cloudpolish.csv`
- `reports/qubo_solver_report_240_d120_do25_cloudpolish.md`
- `reports/qubo_solver_results_240_d120_do25_cloudpolish.json`
- `reports/qubo_accepted_actions_240_d120_do25_cloudpolish.csv`
- `reports/qubo_solver_report_360_d120_do25_cloudpolish.md`
- `reports/qubo_solver_results_360_d120_do25_cloudpolish.json`
- `reports/qubo_accepted_actions_360_d120_do25_cloudpolish.csv`
- `reports/qubo_solver_report_full_d120_do25_cloudpolish.md`
- `reports/qubo_solver_results_full_d120_do25_cloudpolish.json`
- `reports/qubo_accepted_actions_full_d120_do25_cloudpolish.csv`
- `reports/qubo_risk_accepted_actions_full_d120_do25_cloudpolish.csv`
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
- `reports/qubo_solver_report_full_rec_cvar0p10_cut3_truepolish3_ms_local.md`
- `reports/qubo_solver_results_full_rec_cvar0p10_cut3_truepolish3_ms_local.json`
- `reports/qubo_accepted_actions_full_rec_cvar0p10_cut3_truepolish3_ms_local.csv`
- `reports/qubo_risk_accepted_actions_full_rec_cvar0p10_cut3_truepolish3_ms_local.csv`
- `reports/qubo_solver_report_full_rec_cvar0p10_cut3_truepolish5_ms_local.md`
- `reports/qubo_solver_results_full_rec_cvar0p10_cut3_truepolish5_ms_local.json`
- `reports/qubo_accepted_actions_full_rec_cvar0p10_cut3_truepolish5_ms_local.csv`
- `reports/qubo_risk_accepted_actions_full_rec_cvar0p10_cut3_truepolish5_ms_local.csv`
- `reports/qubo_solver_report_full_local.md`
- `reports/qubo_solver_results_full_local.json`
- `reports/qubo_accepted_actions_full_local.csv`
