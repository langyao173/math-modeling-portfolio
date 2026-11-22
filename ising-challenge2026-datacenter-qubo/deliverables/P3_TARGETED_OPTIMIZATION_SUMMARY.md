# Historical P3 Targeted Optimization Summary (Superseded)

> **Do not submit this file as a current result.** It records a predecessor
> stage and is excluded from the official attachment builder. The authoritative
> promoted result is in `final_solution_package/FINAL_REPORT.md`.

## Scope

This round did not call CloudSolver or consume platform balance. It only reran the local full-instance recourse polish with the same QUBO/recourse settings and increased `--recourse-polish-passes` from 3 to 5.

## Result

| Variant | Expected recourse | CVaR95 | Dummy actions | Worst tail scenario |
|---|---:|---:|---:|---|
| pass3 local multi-start | 35.273095 | 47.109637 | 7 | green_short_peak_price |
| pass5 local multi-start | 34.953755 | 46.793822 | 7 | green_short_peak_price |

Improvements from pass3 to pass5:

- Expected recourse decreased by 0.319339.
- CVaR95 decreased by 0.315815.
- Dummy actions stayed at 7.

## Dummy Action Audit

All 7 final dummy jobs are dummy-only in the candidate action table:

- `job_00346`
- `job_00570`
- `job_00575`
- `job_00577`
- `job_00578`
- `job_00581`
- `job_00582`

Each of these jobs has 1 candidate action, and that candidate is the dummy action. Therefore the solver cannot reduce the dummy count below 7 without changing the upstream candidate-action generation model or adding new feasible real actions.

## Tail Scenario Audit

The CVaR95 tail is driven by `green_short_peak_price`:

- Total recourse cost: 46.793822
- Grid cost: 43.129882
- Curtailment cost: 0.115980
- Flex shortfall cost: 3.547960
- Grid import: 0.346120 MWh
- Flex shortfall: 0.014192 MWh

The tail is primarily grid-cost dominated under a green-short/high-price stress scenario, with a smaller flex-shortfall penalty. The pass5 local polish reduced both CVaR95 and expected recourse while preserving full job coverage and one-hot feasibility.
