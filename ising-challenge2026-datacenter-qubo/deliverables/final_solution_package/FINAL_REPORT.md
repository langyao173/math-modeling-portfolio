# TAOLIN-DC-SIM Final Solution Package

Generated: `2026-07-26T13:39:59+00:00`

## Final Solution

- Selected solution: `FINAL_SOLUTION.csv`
- Full selected action table: `FULL_SELECTED_ACTIONS.csv`
- Risk-accepted source: `cloudseed_peak_shave_contextual_recourse_polish`
- The final 588-job schedule descends from the verified 700-spin CloudSolver neighborhood task `260706111252601`; the frozen evidence retains its task ID, returned length, and energy summary, but not the raw spin vector.
- The retained post-polish schedule is independently replayable, but the original cloud assignment cannot be reproduced bit by bit or causally separated from the subsequent contextual polish.
- Promotion was gated on zero one-hot violations, zero dummy actions, the action-cost guard, and strict improvement over the prior final in expected recourse, CVaR95, and peak IT load.

## Metrics

- Jobs selected: 588
- QUBO variables / spins: 1929
- One-hot violations: 0
- Dummy actions: 0
- True action cost: 5414.858289
- QUBO energy (proxy only, not final business objective): -46716.177985
- Expected recourse cost: 35.394524
- CVaR95 recourse cost: 47.718285
- Peak IT MW: 0.031646
- Mode counts: `{'delay': 294, 'horizon_rescue': 7, 'normal': 287}`

## Validation

- Status: `pass`
- Rows: 588
- Unique jobs: 588
- Expected jobs: 588
- Missing action ids: 0
- Duplicate jobs: 0

## Industrial Ising Evidence

- Positioning: Ising-assisted bounded large-neighborhood optimization for binary action selection; exact recourse replay remains the business objective.
- LNS portfolio: base_lns (700 spins), aligned_recourse_lns (700 spins), tail_cvar_lns (700 spins), cloud_sparse_tail_lns (520 spins), dummy_rescue_lns (701 spins), peak_shave_lns (700 spins), renewable_absorb_lns (702 spins).
- Same-QUBO oracle replacement rows: 22 total, 1 CloudSolver rows, 0 failed rows.
- Cloud status: CloudSolver rows are included in the same-QUBO oracle replacement table.
- Required evidence tables are packaged under `artifacts/solver_runs/`, `artifacts/tables/`, and `artifacts/validation/`.
- QUBO energy is retained only as solver/proxy evidence; final performance is recomputed from exact recourse and CVaR replay.

## Reproduce

```powershell
python src/07_solve_isling/build_industrial_ising_evidence.py --target-spins 700 --use-cloud `
  --cloud-families peak_shave_lns --reuse-cloud-task-id 260706111252601
python src/07_solve_isling/promote_cloud_candidate.py
python src/07_solve_isling/build_final_deliverable.py
```

Inside the extracted package, recompute validation and recourse metrics with:

```powershell
python validate_final_solution.py
```

## Caveats

- CAISO price/demand data remains synthetic in the current cleaned dataset.
- Recourse uses fixed stress scenarios and normalized renewable availability.
- Storage, unit commitment, DC power flow, and integer redispatch are not included in this MVP recourse evaluator.
- The former dummy-only horizon-edge jobs are covered by `horizon_rescue` candidate actions; replacing that engineering proxy with real cross-site/migration actions remains future work.
- The Ising solver is claimed only for bounded binary neighborhoods, not for global solution of the full compute-power recourse problem.
