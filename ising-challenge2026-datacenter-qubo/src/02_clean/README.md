# MVP Data Cleaning

Run from `TAOLIN-DC-SIM`:

```powershell
python src/02_clean/clean_mvp_data.py
python src/02_clean/validate_cleaned_data.py
```

Outputs:

- `data/interim/jobs_manifest.parquet`
- `data/interim/site_timeseries.parquet`
- `data/interim/grid_case.json`
- `data/interim/raw_long_jobs.parquet`
- `data/processed/actions.parquet`
- `reports/cleaning_audit.json`
- `reports/cleaning_report.md`
- `reports/cleaned_data_validation_report.md`

The first MVP uses `T=24` one-hour slots. Alibaba workload timestamps are
relative, so they are aligned to a synthetic UTC anchor. CAISO market data is
currently missing locally, so `site_timeseries.parquet` contains synthetic
price/demand fields flagged by `is_synthetic_price = true`.
