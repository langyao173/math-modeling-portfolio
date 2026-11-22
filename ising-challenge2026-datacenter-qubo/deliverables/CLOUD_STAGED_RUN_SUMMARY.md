# Historical Cloud Staged Run Summary (Superseded)

> **Do not submit this file as a current result.** It records a predecessor
> stage and is excluded from the official attachment builder. The authoritative
> promoted result is in `final_solution_package/FINAL_REPORT.md`.

## Scope

After the local `plectrum-core` SDK was installed from the provided offline wheel, cloud checks were rerun with the platform API key injected only into the current process environment. The key was removed after each command.

No full 1922-spin cloud run was launched in this round. The decision gate was: run 120, then 240/360 only if the previous stage succeeds; attempt full only if staged cloud results show a credible improvement over the current local true-recourse polish route.

## Commands

All staged checks used:

```powershell
python src/07_solve_isling/solve_qubo_platform.py --solver cloud --gear balanced `
  --dummy-cost 120 --dummy-only-cost 25 --use-credit --use-coupon `
  --recourse-cvar-weight 0.10 --recourse-cut-iterations 3 `
  --recourse-polish-passes 5 --recourse-polish-starts all --recourse-polish-orders all
```

with `--max-jobs` set to 120, 240, and 360 respectively.

## Results

| Jobs | Best staged source | Expected recourse | CVaR95 | Cloud raw CVaR95 | One-hot violations |
|---:|---|---:|---:|---:|---:|
| 120 | recourse_true_polish | 15.353054 | 19.740421 | 20.987425 | 0 |
| 240 | recourse_true_polish | 24.493723 | 31.728852 | 31.842015 | 0 |
| 360 | recourse_true_polish | 30.431646 | 40.424192 | 41.184857 | 0 |

## Decision

The staged cloud path is functional after installing the SDK, but cloud raw solutions did not outperform the local exact recourse polish on 120/240/360. Therefore the full 1922-spin cloud run was intentionally skipped to avoid spending platform balance without evidence of likely improvement.

The current final deliverable remains the local full-instance pass5 true-recourse-polished solution:

- Expected recourse: 34.953755
- CVaR95: 46.793822
- One-hot violations: 0
- Dummy actions: 7
