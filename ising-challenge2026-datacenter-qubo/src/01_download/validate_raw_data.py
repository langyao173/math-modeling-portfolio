"""Validate downloaded raw data for the TAOLIN-DC-SIM MVP."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data" / "download_manifest.json"
REPORT_PATH = ROOT / "reports" / "data_validation_report.md"
RESULTS_PATH = ROOT / "reports" / "data_validation_results.json"


@dataclass
class Finding:
    severity: str
    confidence: str
    area: str
    title: str
    evidence: str
    impact: str
    remediation: str


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(path: Path):
    for item in sorted(path.rglob("*")):
        if item.is_file() and ".git" not in item.parts:
            yield item


def count_files_bytes(path: Path) -> tuple[int, int]:
    if path.is_file():
        return 1, path.stat().st_size
    if not path.exists():
        return 0, 0
    count = 0
    total = 0
    for item in iter_files(path):
        count += 1
        total += item.stat().st_size
    return count, total


def git_info(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        return {"is_git_repo": False}
    result: dict[str, Any] = {"is_git_repo": True}
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
        result["head"] = head.stdout.strip()
        result["status_short"] = [line for line in status.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError) as exc:
        result["error"] = str(exc)
    return result


def load_csv(path: Path, skip_rows: int = 0) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for _ in range(skip_rows):
            next(reader)
        header = next(reader)
        rows = [dict(zip(header, row)) for row in reader]
    return header, rows


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text in ("", "N/A", "NA", "nan", "None"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def to_int(value: str | None) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def null_like_count(rows: list[dict[str, str]], columns: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in columns:
        counts[column] = sum(1 for row in rows if row.get(column, "").strip() in ("", "N/A", "NA", "nan", "None"))
    return counts


def validate_manifest(findings: list[Finding]) -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    status_counts = Counter(entry.get("status") for entry in entries)
    reconciled = []

    for entry in entries:
        local_path = Path(entry["local_path"])
        full_path = ROOT / local_path
        files, bytes_total = count_files_bytes(full_path)
        status = entry.get("status")
        expected_files = entry.get("rows_files_size", {}).get("files")
        expected_bytes = entry.get("rows_files_size", {}).get("bytes")
        exists = full_path.exists()
        reconciled.append(
            {
                "source_name": entry.get("source_name"),
                "status": status,
                "local_path": local_path.as_posix(),
                "exists": exists,
                "actual_files": files,
                "actual_bytes": bytes_total,
                "manifest_files": expected_files,
                "manifest_bytes": expected_bytes,
            }
        )
        if status in {"present", "downloaded", "already_exists"} and not exists:
            findings.append(
                Finding(
                    "Critical",
                    "High",
                    "manifest",
                    f"Manifest source is marked {status} but path is missing",
                    f"{entry.get('source_name')} -> {local_path.as_posix()}",
                    "Downstream code may assume data exists when it does not.",
                    "Restore the file/directory or change the manifest status before using the source.",
                )
            )
        if status == "empty" and files == 0 and entry.get("source_name") == "caiso_oasis":
            findings.append(
                Finding(
                    "High",
                    "High",
                    "price_demand",
                    "CAISO OASIS data is absent",
                    "Manifest status is empty and data/raw/price/caiso_oasis contains 0 files.",
                    "LMP, demand forecast, and market-price features cannot be built from local raw data.",
                    "Rerun src/01_download/download_api_samples.py --only caiso on a network/DNS path that can reach oasis.caiso.com, then rebuild the manifest.",
                )
            )
        if status == "not_downloaded_capacity":
            findings.append(
                Finding(
                    "Medium",
                    "High",
                    "workload",
                    "Alibaba v2026 bulk archives were intentionally skipped",
                    "Manifest records required upstream compressed size as 356,275,833,963 bytes.",
                    "Experiments cannot claim v2026 bulk-trace coverage; only repository metadata/scripts are local.",
                    "Keep v2026 out of MVP claims unless a high-capacity storage location is used and the archives are downloaded and validated.",
                )
            )

    return {
        "manifest_generated_at": manifest.get("generated_at"),
        "entry_count": len(entries),
        "status_counts": dict(status_counts),
        "reconciled_entries": reconciled,
    }


def validate_alibaba_v2020(findings: list[Finding]) -> dict[str, Any]:
    base = ROOT / "data/raw/workload/alibaba_gpu/cluster-trace-gpu-v2020/data"
    expected = {
        "pai_group_tag_table.tar.gz": "pai_group_tag_table.csv",
        "pai_instance_table.tar.gz": "pai_instance_table.csv",
        "pai_job_table.tar.gz": "pai_job_table.csv",
        "pai_machine_metric.tar.gz": "pai_machine_metric.csv",
        "pai_machine_spec.tar.gz": "pai_machine_spec.csv",
        "pai_sensor_table.tar.gz": "pai_sensor_table.csv",
        "pai_task_table.tar.gz": "pai_task_table.csv",
    }
    archives = []
    for archive_name, expected_member in expected.items():
        path = base / archive_name
        record: dict[str, Any] = {
            "name": archive_name,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "sha256": sha256_file(path) if path.exists() else None,
            "tar_ok": False,
            "members": [],
        }
        if not path.exists():
            findings.append(
                Finding(
                    "High",
                    "High",
                    "workload",
                    "Expected Alibaba v2020 archive is missing",
                    archive_name,
                    "The v2020 workload trace cannot be fully reconstructed.",
                    "Download the missing archive from the upstream v2020GPUTraces URL.",
                )
            )
        else:
            try:
                with tarfile.open(path, "r:gz") as tar:
                    members = [member.name for member in tar.getmembers()]
                record["tar_ok"] = True
                record["members"] = members[:5]
                if expected_member not in members:
                    findings.append(
                        Finding(
                            "High",
                            "High",
                            "workload",
                            "Alibaba v2020 archive does not contain the expected CSV",
                            f"{archive_name} members start with {members[:5]}",
                            "The archive may be incomplete or not the expected table.",
                            "Redownload the archive and re-run validation.",
                        )
                    )
            except (tarfile.TarError, OSError) as exc:
                findings.append(
                    Finding(
                        "High",
                        "High",
                        "workload",
                        "Alibaba v2020 archive failed tar/gzip validation",
                        f"{archive_name}: {exc}",
                        "The file may be truncated or corrupt.",
                        "Redownload the archive and re-run validation.",
                    )
                )
        archives.append(record)
    return {"archive_count": len(archives), "archives": archives}


def validate_alibaba_v2023(findings: list[Finding]) -> dict[str, Any]:
    base = ROOT / "data/raw/workload/alibaba_gpu/cluster-trace-gpu-v2023/csv"
    csv_files = sorted(base.glob("*.csv"))
    expected_pod_columns = {
        "name",
        "cpu_milli",
        "memory_mib",
        "num_gpu",
        "gpu_milli",
        "gpu_spec",
        "qos",
        "pod_phase",
        "creation_time",
        "deletion_time",
        "scheduled_time",
    }
    expected_node_columns = {"sn", "cpu_milli", "memory_mib", "gpu", "model"}
    file_profiles = []
    total_data_rows = 0
    aggregate_invalid = Counter()
    missing_schema_files: list[dict[str, Any]] = []
    phases = Counter()
    qos_values = Counter()

    for path in csv_files:
        header, rows = load_csv(path)
        total_data_rows += len(rows)
        name = path.name
        profile: dict[str, Any] = {
            "file": name,
            "rows": len(rows),
            "columns": header,
            "duplicate_key_count": None,
            "invalid_counts": {},
        }

        if name.startswith("openb_node_list"):
            missing = sorted(expected_node_columns.difference(header))
            key = "sn"
            keys = [row.get(key, "") for row in rows]
            duplicates = len(keys) - len(set(keys))
            profile["duplicate_key_count"] = duplicates
            invalid_counts = {
                "missing_required_columns": len(missing),
                "duplicate_sn": duplicates,
                "cpu_milli_nonpositive": sum(1 for row in rows if (to_float(row.get("cpu_milli")) or 0) <= 0),
                "memory_mib_nonpositive": sum(1 for row in rows if (to_float(row.get("memory_mib")) or 0) <= 0),
                "gpu_negative": sum(1 for row in rows if (to_float(row.get("gpu")) or 0) < 0),
                "gpu_model_missing_when_gpu_positive": sum(
                    1
                    for row in rows
                    if (to_float(row.get("gpu")) or 0) > 0 and row.get("model", "").strip() == ""
                ),
            }
        else:
            missing = sorted(expected_pod_columns.difference(header))
            key = "name"
            keys = [row.get(key, "") for row in rows]
            duplicates = len(keys) - len(set(keys))
            profile["duplicate_key_count"] = duplicates
            phases.update(row.get("pod_phase", "") for row in rows)
            qos_values.update(row.get("qos", "") for row in rows)
            invalid_counts = {
                "missing_required_columns": len(missing),
                "duplicate_name": duplicates,
                "cpu_milli_negative": sum(1 for row in rows if (to_float(row.get("cpu_milli")) or 0) < 0),
                "memory_mib_nonpositive": sum(1 for row in rows if (to_float(row.get("memory_mib")) or 0) <= 0),
                "num_gpu_negative": sum(1 for row in rows if (to_float(row.get("num_gpu")) or 0) < 0),
                "gpu_milli_negative": sum(1 for row in rows if (to_float(row.get("gpu_milli")) or 0) < 0),
                "deletion_before_creation": sum(
                    1
                    for row in rows
                    if (to_int(row.get("deletion_time")) is not None)
                    and (to_int(row.get("creation_time")) is not None)
                    and to_int(row.get("deletion_time")) < to_int(row.get("creation_time"))
                ),
                "scheduled_before_creation_nonzero": sum(
                    1
                    for row in rows
                    if (to_int(row.get("scheduled_time")) or 0) != 0
                    and (to_int(row.get("creation_time")) is not None)
                    and (to_int(row.get("scheduled_time")) or 0) < (to_int(row.get("creation_time")) or 0)
                ),
            }
        profile["invalid_counts"] = invalid_counts
        if invalid_counts["missing_required_columns"]:
            profile["missing_columns"] = missing
            missing_schema_files.append(
                {
                    "file": name,
                    "rows": len(rows),
                    "missing_columns": missing,
                    "is_multigpu_scenario": name.startswith("openb_pod_list_multigpu"),
                }
            )
        aggregate_invalid.update(invalid_counts)
        file_profiles.append(profile)

    non_multigpu_schema_mismatches = [item for item in missing_schema_files if not item["is_multigpu_scenario"]]
    multigpu_schema_mismatches = [item for item in missing_schema_files if item["is_multigpu_scenario"]]

    if not csv_files:
        findings.append(
            Finding(
                "High",
                "High",
                "workload",
                "Alibaba v2023 CSV directory is empty",
                str(base.relative_to(ROOT)),
                "The preferred MVP workload source is unavailable.",
                "Restore the cloned Alibaba clusterdata repository.",
            )
        )
    if non_multigpu_schema_mismatches:
        findings.append(
            Finding(
                "High",
                "High",
                "workload",
                "Alibaba v2023 CSV schema mismatch",
                json.dumps(non_multigpu_schema_mismatches, ensure_ascii=False),
                "The CSVs cannot be transformed reliably into jobs_manifest/actions without source-specific handling.",
                "Inspect file profiles in data_validation_results.json and update the transform schema.",
            )
        )
    if multigpu_schema_mismatches:
        findings.append(
            Finding(
                "Medium",
                "High",
                "workload",
                "Alibaba v2023 multigpu scenario CSVs omit timing/status columns",
                f"{len(multigpu_schema_mismatches)} files, {sum(item['rows'] for item in multigpu_schema_mismatches)} rows, missing columns include qos/pod_phase/creation/deletion/scheduled timing fields.",
                "Default and most scenario files are schedulable, but multigpu scenario rows cannot directly produce time-windowed actions without joining or imputing timing/status fields.",
                "Use `openb_pod_list_default.csv` for the first MVP workload transform, or derive multigpu timing/status fields from a compatible full pod file before including those scenarios.",
            )
        )
    for check_name, value in aggregate_invalid.items():
        if check_name != "missing_required_columns" and value > 0:
            severity = "Low" if check_name == "memory_mib_nonpositive" else "Medium"
            findings.append(
                Finding(
                    severity,
                    "Medium",
                    "workload",
                    f"Alibaba v2023 has {check_name} anomalies",
                    f"{check_name}: {value}",
                    "May affect workload scheduling grain or action feasibility if not handled in transforms.",
                    "Filter or normalize these records in the ETL, and add an automated check at the generated jobs_manifest grain.",
                )
            )

    return {
        "csv_file_count": len(csv_files),
        "total_data_rows": total_data_rows,
        "aggregate_invalid_counts": dict(aggregate_invalid),
        "missing_schema_files": missing_schema_files,
        "pod_phase_counts": dict(phases),
        "qos_counts": dict(qos_values),
        "file_profiles": file_profiles,
    }


def timestamp_key(row: dict[str, str]) -> tuple[int, int, int, int, int] | None:
    parts = [to_int(row.get(name)) for name in ["Year", "Month", "Day", "Hour", "Minute"]]
    if any(part is None for part in parts):
        return None
    return tuple(part for part in parts if part is not None)  # type: ignore[return-value]


def validate_nsrdb(findings: list[Finding]) -> dict[str, Any]:
    path = ROOT / "data/raw/energy/nsrdb/nsrdb_goes_aggregated_v4_0_0_2024_sf_bay_np15_sample.csv"
    header, rows = load_csv(path, skip_rows=2)
    required = ["Year", "Month", "Day", "Hour", "Minute", "GHI", "DHI", "DNI", "Temperature", "Wind Speed"]
    missing = sorted(set(required).difference(header))
    keys = [timestamp_key(row) for row in rows]
    unique_keys = len(set(keys))
    nulls = null_like_count(rows, required)
    data_rows = len(rows)
    ranges = {
        "GHI_negative": sum(1 for row in rows if (to_float(row.get("GHI")) or 0) < 0),
        "DHI_negative": sum(1 for row in rows if (to_float(row.get("DHI")) or 0) < 0),
        "DNI_negative": sum(1 for row in rows if (to_float(row.get("DNI")) or 0) < 0),
        "irradiance_over_1500": sum(
            1
            for row in rows
            if any((to_float(row.get(column)) or 0) > 1500 for column in ("GHI", "DHI", "DNI"))
        ),
        "temperature_outside_minus80_80": sum(
            1
            for row in rows
            if (to_float(row.get("Temperature")) is not None)
            and not (-80 <= (to_float(row.get("Temperature")) or 0) <= 80)
        ),
        "wind_speed_negative": sum(1 for row in rows if (to_float(row.get("Wind Speed")) or 0) < 0),
    }
    months = Counter(to_int(row.get("Month")) for row in rows)
    feb_29_rows = sum(1 for row in rows if row.get("Month") == "2" and row.get("Day") == "29")
    minutes = Counter(row.get("Minute") for row in rows)

    if data_rows != 8760:
        findings.append(
            Finding(
                "Medium",
                "High",
                "renewable_solar",
                "NSRDB sample does not have 8760 hourly rows",
                f"Rows after metadata/header: {data_rows}",
                "Annual hourly solar features may have missing or duplicated time slots.",
                "Re-download with interval=60 and leap_day=false, or document the expected row count.",
            )
        )
    if missing:
        findings.append(
            Finding(
                "High",
                "High",
                "renewable_solar",
                "NSRDB missing required columns",
                ", ".join(missing),
                "site_timeseries cannot compute solar availability as expected.",
                "Update the API attributes or transform schema.",
            )
        )
    if unique_keys != data_rows:
        findings.append(
            Finding(
                "High",
                "High",
                "renewable_solar",
                "NSRDB duplicate or malformed timestamps",
                f"unique timestamp keys {unique_keys} vs rows {data_rows}",
                "Hourly joins may duplicate or lose records.",
                "Deduplicate or redownload before generating site_timeseries.",
            )
        )
    invalid_total = sum(ranges.values()) + sum(nulls.values())
    if invalid_total > 0:
        findings.append(
            Finding(
                "Medium",
                "Medium",
                "renewable_solar",
                "NSRDB has range/null anomalies",
                f"nulls={nulls}, ranges={ranges}",
                "May distort renewable capacity-factor generation.",
                "Inspect affected rows and apply source-specific cleaning before modeling.",
            )
        )

    return {
        "path": path.relative_to(ROOT).as_posix(),
        "rows": data_rows,
        "columns": header,
        "missing_required_columns": missing,
        "unique_timestamp_keys": unique_keys,
        "null_like_counts": nulls,
        "range_checks": ranges,
        "minute_values": dict(minutes),
        "month_counts": dict(months),
        "feb_29_rows": feb_29_rows,
        "min_timestamp": list(min(key for key in keys if key is not None)),
        "max_timestamp": list(max(key for key in keys if key is not None)),
    }


def validate_wind(findings: list[Finding]) -> dict[str, Any]:
    path = ROOT / "data/raw/energy/wind_toolkit/wind_toolkit_2014_sf_bay_np15_sample_100m.csv"
    header, rows = load_csv(path, skip_rows=1)
    speed_col = "wind speed at 100m (m/s)"
    direction_col = "wind direction at 100m (deg)"
    power_col = "power - DEPRECATED"
    required = ["Year", "Month", "Day", "Hour", "Minute", speed_col, direction_col]
    missing = sorted(set(required).difference(header))
    keys = [timestamp_key(row) for row in rows]
    unique_keys = len(set(keys))
    nulls = null_like_count(rows, required)
    data_rows = len(rows)
    ranges = {
        "wind_speed_negative": sum(1 for row in rows if (to_float(row.get(speed_col)) or 0) < 0),
        "wind_speed_over_75": sum(1 for row in rows if (to_float(row.get(speed_col)) or 0) > 75),
        "wind_direction_outside_0_360": sum(
            1
            for row in rows
            if (to_float(row.get(direction_col)) is not None)
            and not (0 <= (to_float(row.get(direction_col)) or 0) <= 360)
        ),
    }
    power_na_count = sum(1 for row in rows if row.get(power_col, "").strip() in ("", "N/A", "NA"))
    minutes = Counter(row.get("Minute") for row in rows)

    if data_rows != 8760:
        findings.append(
            Finding(
                "Medium",
                "High",
                "renewable_wind",
                "Wind Toolkit sample does not have 8760 hourly rows",
                f"Rows after metadata/header: {data_rows}",
                "Annual hourly wind features may have missing or duplicated time slots.",
                "Re-download with interval=60 and a full-year name, or document the expected row count.",
            )
        )
    if missing:
        findings.append(
            Finding(
                "High",
                "High",
                "renewable_wind",
                "Wind Toolkit missing required columns",
                ", ".join(missing),
                "site_timeseries cannot compute wind availability as expected.",
                "Update the API attributes or transform schema.",
            )
        )
    if unique_keys != data_rows:
        findings.append(
            Finding(
                "High",
                "High",
                "renewable_wind",
                "Wind Toolkit duplicate or malformed timestamps",
                f"unique timestamp keys {unique_keys} vs rows {data_rows}",
                "Hourly joins may duplicate or lose records.",
                "Deduplicate or redownload before generating site_timeseries.",
            )
        )
    if sum(ranges.values()) + sum(nulls.values()) > 0:
        findings.append(
            Finding(
                "Medium",
                "Medium",
                "renewable_wind",
                "Wind Toolkit has range/null anomalies",
                f"nulls={nulls}, ranges={ranges}",
                "May distort renewable capacity-factor generation.",
                "Inspect affected rows and apply source-specific cleaning before modeling.",
            )
        )
    if power_na_count == data_rows:
        findings.append(
            Finding(
                "Low",
                "High",
                "renewable_wind",
                "Wind Toolkit power field is entirely N/A",
                f"{power_na_count}/{data_rows} rows have N/A in `{power_col}`.",
                "The local file is usable for wind-speed features, but not direct wind-power MW without a turbine model.",
                "Compute wind power from speed and a documented turbine curve, or request a non-deprecated power attribute if available.",
            )
        )

    return {
        "path": path.relative_to(ROOT).as_posix(),
        "rows": data_rows,
        "columns": header,
        "missing_required_columns": missing,
        "unique_timestamp_keys": unique_keys,
        "null_like_counts": nulls,
        "range_checks": ranges,
        "minute_values": dict(minutes),
        "power_na_count": power_na_count,
        "min_timestamp": list(min(key for key in keys if key is not None)),
        "max_timestamp": list(max(key for key in keys if key is not None)),
    }


def validate_pglib(findings: list[Finding]) -> dict[str, Any]:
    base = ROOT / "data/raw/grid/pglib"
    files = sorted(base.glob("pglib_opf_case*.m"))
    required_sections = ["mpc.bus", "mpc.gen", "mpc.branch", "mpc.gencost"]
    key_cases = [
        "pglib_opf_case14_ieee.m",
        "pglib_opf_case24_ieee_rts.m",
        "pglib_opf_case30_ieee.m",
        "pglib_opf_case57_ieee.m",
        "pglib_opf_case118_ieee.m",
        "pglib_opf_case300_ieee.m",
    ]
    section_failures = []
    case_profiles = []

    function_pattern = re.compile(r"function\s+mpc\s*=\s*(\w+)")
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        missing = [section for section in required_sections if section not in text]
        function_match = function_pattern.search(text)
        case_profiles.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "function": function_match.group(1) if function_match else None,
                "missing_sections": missing,
            }
        )
        if missing or not function_match:
            section_failures.append({"file": path.name, "missing_sections": missing, "has_function": bool(function_match)})

    missing_key_cases = [name for name in key_cases if not (base / name).exists()]
    if missing_key_cases:
        findings.append(
            Finding(
                "High",
                "High",
                "grid",
                "PGLib key benchmark cases are missing",
                ", ".join(missing_key_cases),
                "The MVP grid-case selection cannot use standard IEEE baselines.",
                "Restore the PGLib-OPF clone or adjust the grid-case config.",
            )
        )
    if section_failures:
        findings.append(
            Finding(
                "High",
                "High",
                "grid",
                "PGLib MATPOWER case section validation failed",
                json.dumps(section_failures[:5], ensure_ascii=False),
                "Grid parser may fail or silently omit buses, branches, generators, or costs.",
                "Inspect the failed files and exclude/repair them before conversion to grid_case.json.",
            )
        )

    return {
        "case_file_count": len(files),
        "key_cases_present": sorted(set(key_cases).difference(missing_key_cases)),
        "missing_key_cases": missing_key_cases,
        "section_failure_count": len(section_failures),
        "case_profiles_sample": case_profiles[:10],
    }


def validate_downstream_targets(findings: list[Finding]) -> dict[str, Any]:
    targets = {
        "jobs_manifest": ROOT / "data/interim/jobs_manifest.parquet",
        "site_timeseries": ROOT / "data/interim/site_timeseries.parquet",
        "grid_case": ROOT / "data/interim/grid_case.json",
        "actions": ROOT / "data/processed/actions.parquet",
    }
    status = {
        name: {
            "path": path.relative_to(ROOT).as_posix(),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
        for name, path in targets.items()
    }
    missing = [name for name, item in status.items() if not item["exists"]]
    if missing:
        findings.append(
            Finding(
                "High",
                "High",
                "model_ready_tables",
                "MVP model-ready tables have not been generated",
                f"Missing: {', '.join(missing)}",
                "Raw data is partially validated, but the required jobs/actions/site/grid tables are not yet available for solver experiments.",
                "Implement and validate the cleaning/transform stage before running optimization or Ising experiments.",
            )
        )
    return status


def validate_git_repos(findings: list[Finding]) -> dict[str, Any]:
    repos = {
        "alibaba_clusterdata": ROOT / "data/raw/workload/alibaba_gpu",
        "google_cluster_data": ROOT / "data/raw/workload/google_cluster",
        "pglib_opf": ROOT / "data/raw/grid/pglib",
    }
    info = {name: git_info(path) for name, path in repos.items()}
    for name, item in info.items():
        if not item.get("is_git_repo"):
            findings.append(
                Finding(
                    "Medium",
                    "High",
                    "source_versioning",
                    f"{name} is not a Git repository",
                    str(repos[name].relative_to(ROOT)),
                    "Source version cannot be reproduced from commit hash.",
                    "Restore the shallow clone or record an equivalent immutable source archive checksum.",
                )
            )
        elif item.get("status_short"):
            severity = "Low"
            if name != "alibaba_clusterdata":
                severity = "Medium"
            findings.append(
                Finding(
                    severity,
                    "High",
                    "source_versioning",
                    f"{name} has untracked or modified files",
                    "; ".join(item["status_short"][:10]),
                    "Commit hash still identifies upstream code/data, but local raw files include additional material outside Git tracking.",
                    "Keep these files documented in download_manifest.json; consider storing external tarballs outside the cloned repo if clean Git status is required.",
                )
            )
    return info


def summarize_raw_tree() -> dict[str, Any]:
    raw = ROOT / "data/raw"
    files, bytes_total = count_files_bytes(raw)
    by_top = {}
    for child in sorted(raw.iterdir()):
        if child.is_dir():
            count, size = count_files_bytes(child)
            by_top[child.name] = {"files": count, "bytes": size}
    return {"files": files, "bytes": bytes_total, "by_top_level": by_top}


def overall_assessment(findings: list[Finding]) -> str:
    severities = Counter(finding.severity for finding in findings)
    if severities["Critical"] or severities["High"]:
        return "Needs revision for full MVP use"
    if severities["Medium"]:
        return "Share with caveats"
    return "Ready to share"


def write_outputs(results: dict[str, Any], findings: list[Finding]) -> None:
    results["findings"] = [asdict(finding) for finding in findings]
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    sorted_findings = sorted(findings, key=lambda item: (severity_order.get(item.severity, 9), item.area, item.title))
    lines = [
        "# Raw Data Validation Report",
        "",
        f"Generated: `{results['generated_at']}`",
        f"Overall Assessment: **{results['overall_assessment']}**",
        "",
        "## Dataset And Grain Summary",
        "",
        f"- Raw tree: {results['raw_tree']['files']} files, {results['raw_tree']['bytes']} bytes.",
        f"- Manifest entries: {results['manifest']['entry_count']} with statuses `{results['manifest']['status_counts']}`.",
        "- Intended downstream grain: job/action/site-hour/grid-case tables for the TAOLIN-DC-SIM MVP.",
        "",
        "## Checks Performed",
        "",
        "- Manifest/file reconciliation and source status checks.",
        "- Git source version checks for cloned repositories.",
        "- Alibaba v2020 tar.gz archive integrity and expected CSV member checks.",
        "- Alibaba v2023 pod/node CSV schema, row, duplicate-key, and basic domain checks.",
        "- NSRDB and Wind Toolkit hourly row count, timestamp uniqueness, null-like values, and range checks.",
        "- PGLib MATPOWER case presence and required-section checks.",
        "- Model-ready target table existence checks.",
        "",
        "## Findings",
        "",
    ]
    if sorted_findings:
        for idx, finding in enumerate(sorted_findings, 1):
            lines.extend(
                [
                    f"{idx}. [{finding.severity} / {finding.confidence}] {finding.title}",
                    f"   - Area: `{finding.area}`",
                    f"   - Evidence: {finding.evidence}",
                    f"   - Impact: {finding.impact}",
                    f"   - Remediation: {finding.remediation}",
                ]
            )
    else:
        lines.append("No material findings.")

    lines.extend(
        [
            "",
            "## Calculation Spot-Checks",
            "",
            f"- Alibaba v2020 archives: {results['alibaba_v2020']['archive_count']} expected archives checked; all present archives must be tar/gzip-readable.",
            f"- Alibaba v2023 CSVs: {results['alibaba_v2023']['csv_file_count']} files, {results['alibaba_v2023']['total_data_rows']} data rows.",
            f"- NSRDB: {results['nsrdb']['rows']} hourly data rows, {results['nsrdb']['unique_timestamp_keys']} unique timestamp keys, Feb 29 rows = {results['nsrdb']['feb_29_rows']}.",
            f"- Wind Toolkit: {results['wind_toolkit']['rows']} hourly data rows, {results['wind_toolkit']['unique_timestamp_keys']} unique timestamp keys, deprecated power N/A rows = {results['wind_toolkit']['power_na_count']}.",
            f"- PGLib: {results['pglib']['case_file_count']} MATPOWER case files; section failures = {results['pglib']['section_failure_count']}.",
            "",
            "## Required Caveats",
            "",
            "- The current local data is validated at raw-source level, not at the final `jobs_manifest`, `actions`, `site_timeseries`, or `grid_case` grain.",
            "- CAISO price/demand files are missing locally due DNS/network failure during download.",
            "- Alibaba v2026 bulk archives are not local because the upstream compressed payload is hundreds of GB.",
            "- Wind Toolkit sample has wind speed/direction but no usable direct power column; derive power through a documented curve if needed.",
            "",
            f"Detailed machine-readable evidence: `{RESULTS_PATH.relative_to(ROOT).as_posix()}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    findings: list[Finding] = []
    results: dict[str, Any] = {"generated_at": now_utc(), "root": str(ROOT)}
    results["raw_tree"] = summarize_raw_tree()
    results["manifest"] = validate_manifest(findings)
    results["git_repos"] = validate_git_repos(findings)
    results["alibaba_v2020"] = validate_alibaba_v2020(findings)
    results["alibaba_v2023"] = validate_alibaba_v2023(findings)
    results["nsrdb"] = validate_nsrdb(findings)
    results["wind_toolkit"] = validate_wind(findings)
    results["pglib"] = validate_pglib(findings)
    results["downstream_targets"] = validate_downstream_targets(findings)
    results["overall_assessment"] = overall_assessment(findings)
    write_outputs(results, findings)
    print(json.dumps({"report": str(REPORT_PATH), "results": str(RESULTS_PATH), "overall": results["overall_assessment"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
