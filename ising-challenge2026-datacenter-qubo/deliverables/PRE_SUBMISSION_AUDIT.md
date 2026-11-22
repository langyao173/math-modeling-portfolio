# Historical Pre-Submission Audit (Superseded)

> **Do not use this audit for submission.** It predates the promoted
> `full_cloudseed_peakguard` package and is excluded from the official
> attachment builder. Use `SUBMISSION_VERIFICATION.md` instead.

## Package

- Package path: `deliverables/TAOLIN_DC_SIM_final_solution_package.zip`
- Top-level folder after extraction: `final_solution_package/`
- Required final CSV: `FINAL_SOLUTION.csv`
- Recompute script inside package: `validate_final_solution.py`
- Embedded validation output: `VALIDATION_RECOMPUTED.json`

## Clean-Room Acceptance

The ZIP was extracted into a fresh temporary directory and validated from inside the extracted package.

| Check | Status |
|---|---|
| ZIP integrity `testzip()` | pass |
| `FINAL_SOLUTION.csv` found after extraction | pass |
| `validate_final_solution.py` runs in extracted package | pass |
| `VALIDATION.json` status | pass |
| `VALIDATION_RECOMPUTED.json` status | pass |
| Jobs covered | 588/588 |
| One-hot violations | 0 |
| Dummy actions | 7 |
| Expected recourse | 34.953755 |
| CVaR95 | 46.793822 |
| Absolute path / account / cache / token scan | 0 hits |

## CSV Schema

`FINAL_SOLUTION.csv` columns:

```text
job_id, action_id, site, start_slot, end_slot, mode, gpu_req, power_mw, energy_mwh, action_cost, is_dummy
```

The CSV has 588 data rows and 588 unique `job_id` values.

## File Hygiene

- No Windows absolute paths are present in the package payload.
- No user account path, AppData path, temporary directory path, or cache directory is present.
- No platform key value or token assignment is present.
- No `__pycache__`, virtual environment, package cache, or local dependency folder is included.
- Package manifest uses `package_root: "."`.

## Report / Material Status

- Competition report draft: `deliverables/competition_report/COMPETITION_REPORT.md`
- Experiment tables: `deliverables/competition_report/tables/`
- P3 optimization note: `deliverables/P3_TARGETED_OPTIMIZATION_SUMMARY.md`

## Remaining External Check

The platform-specific submission filename and CSV schema should be compared against the final competition portal rule page before upload. The local package is internally consistent and reproducible, but platform-side naming requirements can only be finally confirmed against the portal.
