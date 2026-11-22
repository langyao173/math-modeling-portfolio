"""Download small API-based raw-data samples for the TAOLIN-DC-SIM MVP.

The GitHub-hosted datasets are cloned separately. This script handles sources
that should be pulled through APIs rather than manual browser downloads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


USER_AGENT = "dc-qubo-research-downloader/1.0"
SITE = {
    "site_id": "sf_bay_np15_sample",
    "wkt": "POINT(-122.4194 37.7749)",
    "caiso_node": "TH_NP15_GEN-APND",
    "lat": 37.7749,
    "lon": -122.4194,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int | None:
    if path.suffix.lower() != ".csv":
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            return sum(1 for _ in csv.reader(handle))
    except OSError:
        return None


def redacted_params(params: dict[str, Any]) -> dict[str, Any]:
    safe = dict(params)
    if "api_key" in safe and safe["api_key"] != "DEMO_KEY":
        safe["api_key"] = "***"
    return safe


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_event(log_path: Path, event: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def build_url(endpoint: str, params: dict[str, Any]) -> str:
    return endpoint + "?" + urlencode(params, safe="(),")


def download_file(
    *,
    source_name: str,
    endpoint: str,
    params: dict[str, Any],
    source_url: str,
    output_path: Path,
    log_path: Path,
    license_text: str,
    time_zone: str,
    unit: str,
    timeout: int,
    force: bool,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = now_utc()
    url = build_url(endpoint, params)

    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        event = {
            "source_name": source_name,
            "source_url": source_url,
            "download_date": started,
            "api_endpoint": endpoint,
            "query_parameters": redacted_params(params),
            "local_path": str(output_path),
            "rows_files_size": {
                "rows": count_csv_rows(output_path),
                "files": 1,
                "bytes": output_path.stat().st_size,
            },
            "sha256": sha256_file(output_path),
            "license": license_text,
            "time_zone": time_zone,
            "unit": unit,
            "status": "already_exists",
        }
        write_event(log_path, event)
        return event

    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
        output_path.write_bytes(payload)
        event = {
            "source_name": source_name,
            "source_url": source_url,
            "download_date": started,
            "api_endpoint": endpoint,
            "query_parameters": redacted_params(params),
            "local_path": str(output_path),
            "rows_files_size": {
                "rows": count_csv_rows(output_path),
                "files": 1,
                "bytes": output_path.stat().st_size,
            },
            "sha256": sha256_file(output_path),
            "license": license_text,
            "time_zone": time_zone,
            "unit": unit,
            "status": "downloaded",
        }
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        event = {
            "source_name": source_name,
            "source_url": source_url,
            "download_date": started,
            "api_endpoint": endpoint,
            "query_parameters": redacted_params(params),
            "local_path": str(output_path),
            "rows_files_size": {"rows": None, "files": 0, "bytes": 0},
            "sha256": None,
            "license": license_text,
            "time_zone": time_zone,
            "unit": unit,
            "status": "failed",
            "error": str(exc),
        }
    write_event(log_path, event)
    return event


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(args.root).resolve()
    log_path = root / "reports" / "download_events.jsonl"
    api_key = args.api_key
    email = args.email
    full_name = args.full_name
    if args.only in ("all", "nsrdb", "wind") and not email:
        raise SystemExit("NSRDB/WIND Toolkit downloads need a contact e-mail: pass --email or set NLR_API_EMAIL / NREL_API_EMAIL.")
    events: list[dict[str, Any]] = []

    if args.only in ("all", "nsrdb"):
        nsrdb_params = {
            "api_key": api_key,
            "wkt": SITE["wkt"],
            "names": args.nsrdb_year,
            "leap_day": "false",
            "interval": "60",
            "utc": "true",
            "full_name": full_name,
            "email": email,
            "affiliation": "research",
            "mailing_list": "false",
            "reason": "research",
            "attributes": "ghi,dhi,dni,air_temperature,wind_speed",
        }
        events.append(
            download_file(
                source_name="nlr_nsrdb_goes_aggregated",
                endpoint="https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv",
                params=nsrdb_params,
                source_url="https://developer.nlr.gov/docs/solar/nsrdb/nsrdb-GOES-aggregated-v4-0-0-download/",
                output_path=root
                / "data"
                / "raw"
                / "energy"
                / "nsrdb"
                / f"nsrdb_goes_aggregated_v4_0_0_{args.nsrdb_year}_{SITE['site_id']}.csv",
                log_path=log_path,
                license_text="NLR/NREL API terms",
                time_zone="UTC",
                unit="GHI/DHI/DNI W/m2, temperature C, wind speed m/s",
                timeout=args.timeout,
                force=args.force,
            )
        )

    if args.only in ("all", "wind"):
        wind_params = {
            "api_key": api_key,
            "wkt": SITE["wkt"],
            "names": args.wind_year,
            "leap_day": "false",
            "interval": "60",
            "utc": "true",
            "full_name": full_name,
            "email": email,
            "affiliation": "research",
            "mailing_list": "false",
            "reason": "research",
            "attributes": "windspeed_100m,winddirection_100m,power",
        }
        events.append(
            download_file(
                source_name="nlr_wind_toolkit",
                endpoint="https://developer.nlr.gov/api/wind-toolkit/v2/wind/wtk-download.csv",
                params=wind_params,
                source_url="https://developer.nlr.gov/docs/wind/wind-toolkit/wtk-download/",
                output_path=root
                / "data"
                / "raw"
                / "energy"
                / "wind_toolkit"
                / f"wind_toolkit_{args.wind_year}_{SITE['site_id']}_100m.csv",
                log_path=log_path,
                license_text="NLR/NREL API terms",
                time_zone="UTC",
                unit="wind speed m/s, wind direction degrees, power field when available",
                timeout=args.timeout,
                force=args.force,
            )
        )

    if args.only in ("all", "caiso"):
        start = args.caiso_start.replace("-", "") + "T00:00-0000"
        end = args.caiso_end.replace("-", "") + "T00:00-0000"
        caiso_endpoint = "https://oasis.caiso.com/oasisapi/SingleZip"
        caiso_jobs = [
            (
                "caiso_oasis_prc_lmp_dam",
                {
                    "queryname": "PRC_LMP",
                    "startdatetime": start,
                    "enddatetime": end,
                    "version": "1",
                    "market_run_id": "DAM",
                    "node": SITE["caiso_node"],
                },
                root
                / "data"
                / "raw"
                / "price"
                / "caiso_oasis"
                / f"prc_lmp_dam_{SITE['caiso_node'].lower()}_{args.caiso_start}.zip",
                "LMP $/MWh",
            ),
            (
                "caiso_oasis_sld_fcst_dam",
                {
                    "queryname": "SLD_FCST",
                    "startdatetime": start,
                    "enddatetime": end,
                    "version": "1",
                    "market_run_id": "DAM",
                    "forecast_type": "SYS",
                },
                root
                / "data"
                / "raw"
                / "price"
                / "caiso_oasis"
                / f"sld_fcst_dam_sys_{args.caiso_start}.zip",
                "demand forecast MW",
            ),
        ]
        for source_name, params, output_path, unit in caiso_jobs:
            events.append(
                download_file(
                    source_name=source_name,
                    endpoint=caiso_endpoint,
                    params=params,
                    source_url="https://oasis.caiso.com/oasisapi/",
                    output_path=output_path,
                    log_path=log_path,
                    license_text="CAISO OASIS terms",
                    time_zone="UTC query timestamps; source-defined market intervals",
                    unit=unit,
                    timeout=args.timeout,
                    force=args.force,
                )
            )

    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[2])
    parser.add_argument("--only", choices=("all", "nsrdb", "wind", "caiso"), default="all")
    parser.add_argument("--api-key", default=os.getenv("NLR_API_KEY") or os.getenv("NREL_API_KEY") or "DEMO_KEY")
    parser.add_argument("--email", default=os.getenv("NLR_API_EMAIL") or os.getenv("NREL_API_EMAIL") or "")
    parser.add_argument("--full-name", default=os.getenv("NLR_API_NAME") or os.getenv("NREL_API_NAME") or "research user")
    parser.add_argument("--nsrdb-year", default="2024")
    parser.add_argument("--wind-year", default="2014")
    parser.add_argument("--caiso-start", default="2024-06-01")
    parser.add_argument("--caiso-end", default="2024-06-02")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    events = run(parse_args())
    print(json.dumps(events, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
