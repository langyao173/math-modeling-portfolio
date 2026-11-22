# TAOLIN-DC-SIM / TAILOR-DC-SAFE

TAOLIN-DC-SIM is the reproducibility repository for a risk-aware compute-power
co-scheduling solution expressed as bounded QUBO/Ising action-selection
problems. The solver energy is treated as a neighborhood-search proxy. Final
business metrics are recomputed by an independent decode, repair, guard, and
scenario-replay path.

This repository supports two different reproducibility contracts:

1. **Frozen-result replay** revalidates the submitted schedule from packaged
   CSV/JSON inputs. It is the acceptance path and does not need the Plectrum
   SDK, a cloud account, raw-data downloads, or third-party Python packages.
2. **Full solver reproduction** rebuilds processed data and reruns the local or
   cloud-assisted optimization pipeline. It needs the scientific Python
   environment and the organizer-provided Plectrum SDK. Remote data and cloud
   services can change, so this path is not guaranteed to reproduce the frozen
   solution bit-for-bit.

## Frozen result snapshot

The canonical package is
[`deliverables/final_solution_package`](deliverables/final_solution_package/).
Its independently replayed result is:

| Metric | Frozen value |
| --- | ---: |
| Jobs | 588 |
| Binary action variables / spins | 1,929 |
| One-hot violations | 0 |
| Dummy actions | 0 |
| Expected recourse cost (USD) | 35.39452387390732 |
| CVaR95 recourse cost (USD) | 47.718284903473034 |
| Peak IT load (MW) | 0.031646304 |

The authoritative files are:

- [final schedule](deliverables/final_solution_package/FINAL_SOLUTION.csv)
- [declared metrics](deliverables/final_solution_package/FINAL_METRICS.json)
- [independent replay result](deliverables/final_solution_package/VALIDATION_RECOMPUTED.json)
- [package manifest](deliverables/final_solution_package/MANIFEST.json)
- [final report](deliverables/final_solution_package/FINAL_REPORT.md)
- [competition report](deliverables/competition_report/COMPETITION_REPORT.md)

## Fast acceptance replay

Run from the repository root with Python 3.10 or newer:

```powershell
python deliverables/final_solution_package/validate_final_solution.py
python -m zipfile -t deliverables/TAOLIN_DC_SIM_final_solution_package.zip
```

The validator overwrites
`deliverables/final_solution_package/VALIDATION_RECOMPUTED.json` with the newly
computed checks and metrics. It uses only the Python standard library and the
validation CSV/JSON files inside the package. A `pass` result means the frozen
schedule, row counts, action mapping, dummy count, expected recourse cost, and
CVaR agree with the packaged declarations within the validator's numerical
tolerances. It does not rerun optimization or prove global optimality.

## Scientific Python environment

The project is a script repository rather than an installable library. Create
the environment with `uv`:

```powershell
uv sync
```

The dependency ranges are declared in [`pyproject.toml`](pyproject.toml). The
environment audited on 2026-07-20 was:

| Component | Audited version |
| --- | ---: |
| Python | 3.12.11 |
| NumPy | 2.5.1 |
| SciPy | 1.18.0 |
| Pandas | 2.3.3 |
| PyArrow | 24.0.0 |
| pypdf | 6.14.2 |
| PyYAML | 6.0.3 |
| Requests | 2.34.2 |

NumPy and Pandas are imported directly by the cleaning, solver, evidence, and
report scripts. SciPy/HiGHS solves the controlled 30/60/100-job exact MILP
certificates. PyArrow is the Parquet engine. PyYAML reads the instance
configuration. pypdf performs the submission-paper structural gate. Requests
is declared for API/client interoperability, although the current raw-data
downloader itself uses Python's `urllib`. The committed
[`uv.lock`](uv.lock) freezes the registry-resolved environment; the
organizer-provided Plectrum wheel remains an explicit offline dependency.

## Plectrum SDK wheel

`plectrum_core` is not declared as a registry dependency because the available
SDK is an organizer-provided wheel and its redistribution terms are not asserted
by this repository. Obtain `plectrum_core-0.1.1-py3-none-any.whl` from the
competition materials. If redistribution is permitted for the intended bundle,
place it at:

```text
vendor/plectrum_core-0.1.1-py3-none-any.whl
```

Then either install it into the `uv` environment:

```powershell
uv pip install vendor/plectrum_core-0.1.1-py3-none-any.whl
```

or expose the wheel only for the current PowerShell process:

```powershell
$env:PYTHONPATH = (Resolve-Path "vendor\plectrum_core-0.1.1-py3-none-any.whl").Path
```

Do not commit credentials. A cloud run reads its API credential from the
current process environment. Cloud availability, account balance, platform
version, and stochastic solver behavior are external to this repository.

## Rebuild processed inputs

The canonical cleaning configuration is
[`instances/config.yaml`](instances/config.yaml). Existing raw data can be
transformed and checked with:

```powershell
uv run python src/02_clean/clean_mvp_data.py
uv run python src/02_clean/validate_cleaned_data.py
```

This produces the 588-job manifest, 24-slot site time series, IEEE 14-bus grid
case, and 1,929 candidate actions used by the current solver. See
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) before interpreting these inputs:
price and demand are synthetic fallbacks, workload timestamps use a synthetic
calendar anchor, and several power/grid mappings are engineering proxies.

Raw-source download and validation instructions are in
[`src/01_download/README.md`](src/01_download/README.md). API or repository
redownloads are not expected to be byte-identical to the frozen local snapshot.

## Rebuild the predecessor baseline and promoted final

After the processed inputs and Plectrum wheel are available, the following
command rebuilds the predecessor local multi-start schedule used by the final
promotion guard. It does not by itself produce the current promoted final:

```powershell
uv run python src/07_solve_isling/solve_qubo_platform.py --solver local --gear balanced --max-jobs 0 `
  --dummy-cost 120 --dummy-only-cost 25 `
  --recourse-cvar-weight 0.10 `
  --recourse-cut-iterations 3 --recourse-polish-passes 5 `
  --recourse-polish-starts all --recourse-polish-orders all `
  --output-json qubo_solver_results_full_rec_cvar0p10_cut3_truepolish5_ms_local.json `
  --output-md qubo_solver_report_full_rec_cvar0p10_cut3_truepolish5_ms_local.md `
  --output-solution-csv qubo_accepted_actions_full_rec_cvar0p10_cut3_truepolish5_ms_local.csv `
  --output-risk-solution-csv qubo_risk_accepted_actions_full_rec_cvar0p10_cut3_truepolish5_ms_local.csv
```

The script writes these named outputs under `reports/`. The default random seed
for this solver entry point is 42. The industrial evidence builder has a
separate default seed of 20260702.

The current final uses the persisted 700-spin `peak_shave_lns` CloudSolver
assignment. The assignment is energy-checked against the reconstructed QUBO,
merged into the 588-job schedule, independently replayed, and accepted only if
expected recourse, CVaR95, and peak IT load all strictly improve. If the
persisted candidate is already present, promotion and packaging are offline:

```powershell
uv run python src/07_solve_isling/promote_cloud_candidate.py
uv run python src/07_solve_isling/build_competition_report.py
uv run python src/07_solve_isling/build_final_deliverable.py
```

To refresh the persisted candidate from the historical cloud task, inject the
cloud credential only into the current process and run:

```powershell
uv run python src/07_solve_isling/build_industrial_ising_evidence.py `
  --target-spins 700 --use-cloud --cloud-families peak_shave_lns `
  --reuse-cloud-task-id 260706111252601
```

These commands rewrite files below `reports/` and `deliverables/`. Cloud access
is not required for replaying the packaged schedule or re-running promotion
from the persisted candidate. Additional solver details are in
[`src/07_solve_isling/README.md`](src/07_solve_isling/README.md).

## Submission gates

Run the preserved security regression suite with:

```powershell
uv run python -I -B -m pytest -q
```

The official attachment builder validates the canonical archive first, then
creates deterministic paper/code/data/supplement/video attachments and performs
an independent staging verification. It enforces trusted member sets and
content hashes, NTFS-safe names, archive-expansion limits, active-content PDF
rejection, sampled MP4 structure, nested-archive rejection, and full text/secret
scanning. See
[`deliverables/SUBMISSION_OPERATOR_CHECKLIST.md`](deliverables/SUBMISSION_OPERATOR_CHECKLIST.md)
for the exact final command and portal procedure.

Automated technical artifacts are ready, registration ID `YX2600013` is
confirmed, and the official-template DOCX/PDF has passed visual, academic, and
active-content checks. The portal accepts one ZIP/RAR smaller than 50 MB; a
cloud demonstration video is an optional attachment and is not part of the
default submission gate. The deterministic five-member outer ZIP passes all
11 portal checks without a video. Portal readiness remains conditional only on
the exact registered work name, the complete real team-member list, and the
human upload receipt. Synthetic QA metadata must never be uploaded.

## Scope and limitations

- The frozen final schedule was seeded by one successful 700-spin CloudSolver
  call in a bounded same-QUBO neighborhood, then independently replayed and
  Pareto-guarded locally. This is not a claim of a cloud or global solve of the
  full 1,929-variable scheduling and recourse problem.
- The complete cloud assignment and merged selected-action vector are persisted
  in the final package with energy, feasibility, metadata-replay, and promotion
  evidence. Only one cloud call is represented, so no repeated-run variance or
  stochastic robustness claim is made.
- Market price and demand are synthetic. The reported USD values support
  structural comparison under the stated stress model, not real-market savings
  claims.
- The five recourse scenarios are fixed stress cases, not a statistically
  estimated market distribution. With the current probabilities, the 95% tail
  is coarser than the worst scenario's probability mass, and the reported
  VaR95/CVaR95 equals the worst scenario cost.
- GPU power is a request-based proxy rather than measured accelerator telemetry.
- Wind generation uses a simple wind-speed proxy; grid and renewable bus
  assignments are deterministic placeholders.
- Storage, unit commitment, AC/DC dispatch optimization, integer redispatch,
  and a real multi-site migration model are outside the current evaluator.
- `horizon_rescue` prevents dummy-only edge cases but is an engineering proxy,
  not a validated migration action.
- The repository contains no global-optimality certificate and no repeated-run
  uncertainty estimate for the cloud solver.

Data lineage and rights boundaries are documented in
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) and [`NOTICE.md`](NOTICE.md).
