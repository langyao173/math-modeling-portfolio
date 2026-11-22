# Data provenance and transformation record

This document describes the data lineage of the frozen TAOLIN-DC-SIM result.
It distinguishes locally present raw material, canonical processed inputs,
synthetic fields, and the reduced validation snapshot shipped with the final
solution.

The recorded source inventory was generated on 2026-06-29 and the canonical
cleaning/validation outputs were regenerated on 2026-07-06. Paths below are
relative to the repository root.

## Canonical lineage

```text
Alibaba GPU workload trace ─┐
NSRDB solar sample ─────────┤
Wind Toolkit sample ────────┼─> cleaning and proxy construction
PGLib IEEE 14-bus case ─────┤        │
synthetic price/demand ─────┘        v
                         jobs_manifest.parquet
                         site_timeseries.parquet
                         grid_case.json
                         actions.parquet
                                  │
                                  v
                       QUBO/Ising action selection
                                  │
                                  v
                    exact scenario replay and validation
                                  │
                                  v
                      frozen final-solution package
```

## Raw-source inventory used by the MVP

| Source | Local role | Recorded version or identifier | Canonical-use status |
| --- | --- | --- | --- |
| Alibaba Cluster Data GPU v2023 | Workload jobs and relative timing | repository commit `0d0f3f1efdbf`; canonical CSV `cluster-trace-gpu-v2023/csv/openb_pod_list_default.csv` | Used |
| NLR/NREL NSRDB GOES aggregated sample | Solar irradiance for PV capacity factor | local manifest identifier `39caa85d22f3`; 2024 SF Bay NP15 sample | Used |
| NLR/NREL Wind Toolkit sample | Wind speed at 100 m | local manifest identifier `080c3ded39ef`; 2014 SF Bay NP15 sample | Used through a proxy curve |
| PGLib-OPF | Grid-case structure | repository commit `dc6be4b2f85c`; `pglib_opf_case14_ieee.m` | Used as a structural case |
| Google Cluster Data | Auxiliary power-trace context | repository commit `3f6a61d380dc` | Present but not used as the canonical GPU workload |
| CAISO OASIS | Intended LMP and demand source | no local files; two recorded download failures | Not used; replaced by synthetic fallback |

Other Alibaba trace generations and the downloaded v2020 archives are present
in the raw tree but are not the canonical workload for this result. The v2026
bulk archives were not downloaded because the upstream compressed payload was
recorded as approximately 356 GB. The canonical experiment must not be
described as covering the v2026 bulk trace.

At the raw-data audit snapshot, `data/raw` contained 381 files totaling
1,927,260,980 bytes. The detailed source/status record is in
[`reports/data_card.md`](reports/data_card.md), and the machine-readable raw
validation evidence is in
[`reports/data_validation_results.json`](reports/data_validation_results.json).

## Transformation policy

The canonical configuration is
[`instances/config.yaml`](instances/config.yaml), implemented by
[`src/02_clean/clean_mvp_data.py`](src/02_clean/clean_mvp_data.py).

### Workload

- The cleaner read 8,152 raw workload rows, retained 6,202 after structural
  rules, and selected 588 jobs intersecting the 24-slot MVP horizon.
- Alibaba timestamps are relative trace seconds. They were aligned to the
  synthetic UTC anchor `2024-01-01T00:00:00Z`; the resulting calendar dates are
  not observed execution dates.
- GPU milli-request is used as a utilization proxy. Power is a calibrated proxy
  based on requested GPU resources, not measured A100 or facility telemetry.
- Thirty-two long jobs are preserved separately in
  `data/interim/raw_long_jobs.parquet`.

### Renewable and site time series

- The first 24 rows of the NSRDB and Wind Toolkit samples are aligned to the
  same synthetic one-hour horizon despite originating from different years.
- PV capacity factor is `clip(GHI / 1000, 0, 1)` with a nominal 120 MW PV
  capacity.
- Wind capacity factor is the cubic proxy
  `clip((speed_100m - 3) / (12 - 3), 0, 1)^3` with a nominal 80 MW wind
  capacity. This is not a documented turbine power curve.
- Demand, LMP, carbon intensity, and flexibility-call probability are generated
  by deterministic formulas from slot position and renewable scarcity.
- Every canonical site-time-series row is flagged with
  `is_synthetic_price = true` and
  `price_source = synthetic_fallback_caiso_missing`.

### Grid structure

- The structural grid input is `pglib_opf_case14_ieee.m`, parsed into 14 buses
  and 20 branches.
- The data-center bus and renewable bus mappings are deterministic placeholders.
- The grid case is an input to an MVP approximation; the recourse evaluator
  does not solve AC OPF, unit commitment, storage dispatch, or integer
  redispatch.

### Candidate actions

- The action generator produced 1,929 candidate rows for 588 jobs.
- Candidate modes before optimization include normal, delay, degrade, dummy,
  and seven `horizon_rescue` actions.
- The final schedule contains 287 normal, 294 delay, and 7 horizon-rescue
  actions, with no selected dummy action.
- `horizon_rescue` is an engineering feasibility proxy for jobs whose initial
  candidate set would otherwise be dummy-only; it is not a real cross-site or
  migration trace.

## Canonical processed outputs

| File | Rows/items | Purpose |
| --- | ---: | --- |
| `data/interim/jobs_manifest.parquet` | 588 rows | Cleaned job-level scheduling inputs |
| `data/interim/site_timeseries.parquet` | 24 rows | One-site, one-hour-slot renewable and synthetic market signals |
| `data/interim/grid_case.json` | 14 buses, 20 branches | Parsed structural PGLib case and placeholder mappings |
| `data/interim/raw_long_jobs.parquet` | 32 rows | Long-job audit subset |
| `data/processed/actions.parquet` | 1,929 rows | Candidate action table and solver variables |

The cleaned-data validator reported the processed data as ready for structural
MVP tests with caveats, not as market-realistic operational data. See
[`reports/cleaned_data_validation_report.md`](reports/cleaned_data_validation_report.md)
and [`reports/cleaning_audit.json`](reports/cleaning_audit.json).

## Scenario and metric provenance

The recourse evaluator replays five fixed stress scenarios with probabilities
0.50, 0.20, 0.15, 0.10, and 0.05. These probabilities are model inputs, not
frequencies estimated from the raw datasets. They are intended to exercise
nominal, green-short/peak-price, high-green/low-price, evening-stress, and
solar-midday-shortfall behavior.

Because the worst scenario has probability 0.20 while the CVaR95 tail mass is
0.05, the discrete implementation's VaR95 and CVaR95 both equal the worst
scenario cost for the frozen result. The CVaR figure therefore provides a
coarse stress boundary, not a high-resolution empirical tail estimate.

QUBO energy is retained only as solver evidence. Expected recourse, VaR, CVaR,
grid import, curtailment, flexibility shortfall, and peak IT load are recomputed
from the selected action table and validation time series.

## Frozen validation subset

The final package contains only the data needed to replay the accepted result:

- `validation_data/actions_for_validation.csv`
- `validation_data/site_timeseries_for_validation.csv`
- `FINAL_SOLUTION.csv`
- `FULL_SELECTED_ACTIONS.csv`
- `FINAL_METRICS.json`
- `validate_final_solution.py`

This subset is intentionally different from an end-to-end raw-data archive. It
allows deterministic acceptance replay without downloading approximately 1.9
GB of raw material or contacting any API. File hashes for the packaged snapshot
are recorded in
[`deliverables/final_solution_package/MANIFEST.json`](deliverables/final_solution_package/MANIFEST.json).

## Known gaps and claim boundaries

- CAISO price and demand data are absent; no real-market savings claim is
  supported.
- Weather sources use different historical years and are synthetically aligned.
- Only a one-site, 24-hour structural MVP is represented.
- Workload power, wind power, grid bus assignments, and horizon rescue contain
  explicit proxy assumptions.
- The five stress scenarios are not statistically calibrated and do not support
  population-level confidence intervals.
- Raw APIs and upstream repositories can change; end-to-end redownload is not a
  byte-for-byte reproduction contract.
- The promoted final is seeded by one persisted 700-spin cloud assignment in a
  bounded `peak_shave_lns` neighborhood. Its full selected-action vector is
  present in the package and independently replayable; one call does not
  support repeated-run variance or global-optimality claims.

Upstream rights and SDK redistribution boundaries are recorded in
[`NOTICE.md`](NOTICE.md). This project does not relicense third-party data.
