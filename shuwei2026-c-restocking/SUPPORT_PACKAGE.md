# Support material - Problem C, team SWBMCM2639496

State estimation and decision optimisation for restocking operations in unmanned supermarkets
(2026 ShuWei Cup, autumn contest). This package contains every program used in the report, the
frozen models and decision rules, the result files behind every table and figure, and the run and
process logs. Raw data are **not** included.

## Contents

| folder | content |
|---|---|
| `code/` | all Python programs (see the table in Appendix A of the report) |
| `results/` | result tables (`*.csv`, `*.json`), frozen models and rules (`frozen_*.json`), environment files, comparison of the two runs (`repro_report.json`) |
| `logs/` | run logs of every step, process logs of the final policy for all 450 validation and 450 test trials (`p3_process_log_FROZEN_*.jsonl.gz`), run times |
| `figures/` | report figures (PDF and PNG) and the figure manifest `figures.md` |
| `tables/` | LaTeX tables of the report, written by `code/make_tables.py` from `results/` |
| `interim/` | small derived index tables (`records.csv`, `segments.csv`, SHA-1 manifest of the raw table members) |

## Requirements

Python 3.10 or later with NumPy, pandas, SciPy, scikit-learn, statsmodels, Numba, PyArrow and
Matplotlib. Two tested environments are recorded in `results/environment_cloud.json` (Linux; the
reported results) and `results/environment_laptop.json` (Windows 11; the independent rerun).

## How to run

1. Place this folder next to the official data folder, so that the layout is
   `<root>/Restocking_Problem_and_Data_EN/{Raw_Data,Restocking_Tasks_and_Constraints}` and
   `<root>/<this folder>/code`. Other locations can be given through the environment variables
   `RAW_DIR` (folder with the three raw archives), `TASK_DIR` (task folder) and `TABLE_DIR`
   (folder with table-only archives).
2. Optional, to avoid reading the 5.8 GB archives repeatedly: `python code/s01_repack_tables.py`
   copies all non-video members into `interim/<Dataset>_tables.zip` (190 MB, payload unchanged).
3. `python code/run_all.py --procs 8` runs the whole pipeline in order (about 15 minutes on a laptop):
   data harmonisation, Problem 1 (audit, state model, reliability model, factors), Problem 2
   (masking, request policies, natural occlusion), Problem 3 (duration model, index-constant grid,
   validation replay of the candidate policies, selection of the final policy with the pool-level
   rule on 60 emulated pools, validation replay of the final policy with its ablations, test replay,
   analysis, schedules by case and decision trace), the figures and the LaTeX tables.
   `python code/run_all.py --list` shows the steps; `--from <step>` resumes; `--only <steps>` runs a subset.
   All temporary files and caches are written inside this folder (`_tmp/`, `_cache/`).
4. `python code/make_tables.py` alone rewrites the LaTeX tables from the result files (it is also the last step of `run_all.py`).
5. `python code/s99_compare_runs.py <results_A> <results_B>` compares two runs cell by cell.

## Protocol

All models are fitted on the 16 training environments, selected and calibrated on the 4 validation
environments, frozen (`results/frozen_p1_state.json`, `frozen_p1_quality.json`,
`frozen_p2_calibration.json`, `frozen_p3_duration.json`, `frozen_p3_policy.json`) and evaluated on the
4 test environments. Random streams are seeded by constants in the programs; the trial seed of
Problem 3 is never an input of a policy.

Seeds of Problem 3: the validation pool is replayed with the 30 public seeds of `common_conditions.json`
(2026111700-2026111729). The seeds of the organisers' final test are not released, so the test pool is
replayed with 30 further seeds (2026112400-2026112429); the frozen rule can be replayed on any other
seed list with the `custom` command given below.

The decision rule of Problem 3 (look-ahead with the family belief, policy name `FROZEN`) is selected by a
one-standard-error rule whose standard error is the pool-to-pool standard deviation measured on 60
emulated pools of training environments (`code/p3_pooltransfer.py`). The results of all candidate
configurations on both pools are in `results/p3_trials_validation.csv` and `results/p3_trials_test.csv`.

## Material required for Problem 3

* frozen duration model and decision rule: `results/frozen_p3_duration.json`, `results/frozen_p3_policy.json`
  (policy name `FROZEN` in all result files)
* trial-level results for the five mappings (case, split, trial, seed, alpha, beta, all common metrics):
  `results/p3_trials_validation.csv` (21 policies x 450 trials) and `results/p3_trials_test.csv`
  (16 policies x 450 trials; 30 seeds chosen by the team, 2026112400-2026112429)
* schedules of the final policy in the cases S1-S3 (order, reservations, occupancy, waiting, stopping):
  `results/p3_schedule_by_case.csv`, `results/p3_schedule_by_job.csv`, `results/p3_gantt_trials.json`;
  candidate values of every decision of one replayed trial per case: `results/p3_decision_trace.json`
* pool-level selection rule: `results/p3_pooltransfer_pools.csv`,
  `p3_pooltransfer_summary.csv`, `p3_pooltransfer.json`, `p3_duration_by_env.csv`
* process logs (decision time, action, job, visible information, reservation or waiting duration, service start,
  completion, overrun, idle, censoring, zone and cost before and after, source reference after execution):
  `logs/p3_process_log_FROZEN_validation_stage2.jsonl.gz`, `logs/p3_process_log_FROZEN_test.jsonl.gz`
* replay of the rule on other seeds (for instance the organisers' final seeds):
  `python code/p3_run.py custom --split test --seeds <s1,s2,...> --policies FROZEN --tag final --procs 8`
  writes `results/p3_trials_final.csv` and the process log `logs/p3_process_log_FROZEN_final.jsonl.gz`
* environment information: `results/environment_*.json`

## Reproduction check

The reported results come from one complete run on Linux (16.4 minutes on two cores, `logs/run_all_timing.json`).
This folder was then unpacked on a second machine (Windows 11 laptop, other library versions) and executed with the
single command of step 3 (11.3 minutes with eight processes, `logs/run_all_timing_laptop.json`).
`results/repro_report.json` holds the cell-by-cell comparison of the two runs: 76 of 78 result files (964 058 numeric
cells) agree to a relative tolerance of 1e-6 and the largest relative difference in the other two is 3e-5.
