# Data Download

This folder contains reproducible download helpers for the raw datasets listed
in the project data requirements.

## GitHub Sources

Run from `TAOLIN-DC-SIM`:

```powershell
git clone --depth 1 https://github.com/alibaba/clusterdata.git data/raw/workload/alibaba_gpu
git clone --depth 1 https://github.com/google/cluster-data.git data/raw/workload/google_cluster
git clone --depth 1 https://github.com/power-grid-lib/pglib-opf.git data/raw/grid/pglib
```

Alibaba v2020 archives are hosted outside GitHub under the upstream
`cluster-trace-gpu-v2020/data/download_data.sh` URLs. The v2026 bulk archives
are documented upstream but are not downloaded by default because they require
hundreds of GB.

## API Sources

```powershell
python src/01_download/download_api_samples.py
python src/01_download/build_manifest.py
```

Set `NLR_API_KEY`/`NREL_API_KEY` and `NLR_API_EMAIL`/`NREL_API_EMAIL` for
production NLR/NREL pulls (optionally `NLR_API_NAME` for the requester name).
The API key defaults to `DEMO_KEY` for small samples; the NSRDB and WIND Toolkit
endpoints require your own contact e-mail, so the script stops if none is given.

If CAISO DNS fails locally, rerun only that source after switching DNS/network:

```powershell
python src/01_download/download_api_samples.py --only caiso
python src/01_download/build_manifest.py
```
