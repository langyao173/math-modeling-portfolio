"""Validate cleaned MVP tables."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"


def finding(severity: str, title: str, evidence: str, remediation: str) -> dict[str, str]:
    return {
        "severity": severity,
        "title": title,
        "evidence": evidence,
        "remediation": remediation,
    }


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    jobs = pd.read_parquet(INTERIM_DIR / "jobs_manifest.parquet")
    site_ts = pd.read_parquet(INTERIM_DIR / "site_timeseries.parquet")
    grid = json.loads((INTERIM_DIR / "grid_case.json").read_text(encoding="utf-8"))
    actions = pd.read_parquet(PROCESSED_DIR / "actions.parquet")
    return jobs, site_ts, grid, actions


def validate() -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    required_paths = [
        INTERIM_DIR / "jobs_manifest.parquet",
        INTERIM_DIR / "site_timeseries.parquet",
        INTERIM_DIR / "grid_case.json",
        PROCESSED_DIR / "actions.parquet",
        INTERIM_DIR / "raw_long_jobs.parquet",
    ]
    missing = [path.relative_to(ROOT).as_posix() for path in required_paths if not path.exists()]
    if missing:
        findings.append(finding("Critical", "Required cleaned files are missing", ", ".join(missing), "Run src/02_clean/clean_mvp_data.py."))
        return {"findings": findings}

    jobs, site_ts, grid, actions = load_tables()
    horizon = 24

    job_required = {
        "job_id",
        "workload",
        "release_slot",
        "deadline_slot",
        "duration_slots",
        "gpu_req",
        "gpu_util_avg",
        "power_total_mw",
        "priority",
        "mode_set",
    }
    missing_job_cols = sorted(job_required.difference(jobs.columns))
    if missing_job_cols:
        findings.append(finding("Critical", "jobs_manifest missing required columns", str(missing_job_cols), "Fix clean_mvp_data.py schema."))
    if jobs["job_id"].duplicated().any():
        findings.append(finding("Critical", "Duplicate job_id values", str(int(jobs["job_id"].duplicated().sum())), "Deduplicate at source_job_id grain."))
    invalid_jobs = jobs[
        (jobs["duration_slots"] <= 0)
        | (jobs["gpu_req"] <= 0)
        | (jobs["release_slot"] < 1)
        | (jobs["release_slot"] > horizon)
        | (jobs["deadline_slot"] <= jobs["release_slot"])
        | (jobs["power_total_mw"] <= 0)
    ]
    if len(invalid_jobs):
        findings.append(finding("High", "jobs_manifest has invalid domain values", f"{len(invalid_jobs)} rows", "Inspect invalid rows and update filters."))

    site_required = {
        "site",
        "slot",
        "timestamp_utc",
        "delta_t_hours",
        "pv_cf",
        "wind_cf",
        "pv_mw",
        "wind_mw",
        "lmp_usd_mwh",
        "carbon_kgco2_mwh",
    }
    missing_site_cols = sorted(site_required.difference(site_ts.columns))
    if missing_site_cols:
        findings.append(finding("Critical", "site_timeseries missing required columns", str(missing_site_cols), "Fix clean_mvp_data.py schema."))
    if len(site_ts) != horizon or site_ts["slot"].nunique() != horizon:
        findings.append(finding("Critical", "site_timeseries does not cover exactly 24 unique slots", f"rows={len(site_ts)}, unique_slots={site_ts['slot'].nunique()}", "Regenerate with T=24."))
    cf_invalid = site_ts[(site_ts["pv_cf"] < 0) | (site_ts["pv_cf"] > 1) | (site_ts["wind_cf"] < 0) | (site_ts["wind_cf"] > 1)]
    if len(cf_invalid):
        findings.append(finding("High", "Capacity factors outside [0, 1]", f"{len(cf_invalid)} rows", "Clamp or repair renewable inputs."))
    if site_ts["is_synthetic_price"].any():
        findings.append(
            finding(
                "Medium",
                "Price/demand fields use synthetic fallback",
                f"{int(site_ts['is_synthetic_price'].sum())}/{len(site_ts)} rows are synthetic.",
                "Download CAISO and replace synthetic price/demand before market-facing claims.",
            )
        )

    for section in ["bus", "gen", "branch", "gencost", "dc_site_to_bus", "renewable_to_bus"]:
        if section not in grid or not grid[section]:
            findings.append(finding("Critical", "grid_case missing section", section, "Repair MATPOWER parser or choose another PGLib case."))

    action_required = {
        "action_id",
        "job_id",
        "site",
        "start_slot",
        "end_slot",
        "mode",
        "gpu_vector",
        "power_vector_mw",
        "action_cost",
        "is_dummy",
        "conflict_keys",
    }
    missing_action_cols = sorted(action_required.difference(actions.columns))
    if missing_action_cols:
        findings.append(finding("Critical", "actions missing required columns", str(missing_action_cols), "Fix action builder schema."))
    if actions["action_id"].duplicated().any():
        findings.append(finding("Critical", "Duplicate action_id values", str(int(actions["action_id"].duplicated().sum())), "Regenerate action IDs."))
    orphan_actions = actions.loc[~actions["job_id"].isin(set(jobs["job_id"]))]
    if len(orphan_actions):
        findings.append(finding("Critical", "actions contain orphan job_id values", str(len(orphan_actions)), "Join actions only from jobs_manifest."))
    dummy_counts = actions.loc[actions["is_dummy"]].groupby("job_id").size()
    missing_dummy_jobs = set(jobs["job_id"]).difference(set(dummy_counts.index))
    if missing_dummy_jobs:
        findings.append(finding("High", "Some jobs lack dummy action", str(len(missing_dummy_jobs)), "Ensure one dummy action per job."))
    bad_slots = actions.loc[
        (~actions["is_dummy"])
        & ((actions["start_slot"] < 1) | (actions["end_slot"] <= actions["start_slot"]) | (actions["end_slot"] > horizon + 1))
    ]
    if len(bad_slots):
        findings.append(finding("High", "Non-dummy actions have invalid slot ranges", str(len(bad_slots)), "Repair action interval generation."))

    bad_vectors = 0
    for _, row in actions.iterrows():
        try:
            gpu_vec = json.loads(row["gpu_vector"])
            power_vec = json.loads(row["power_vector_mw"])
            if len(gpu_vec) != horizon or len(power_vec) != horizon:
                bad_vectors += 1
        except json.JSONDecodeError:
            bad_vectors += 1
    if bad_vectors:
        findings.append(finding("High", "Action vectors are malformed or wrong length", str(bad_vectors), "Regenerate JSON vectors with length T."))

    dummy_violations = 0
    for job_id, group in actions.groupby("job_id"):
        non_dummy_max = group.loc[~group["is_dummy"], "action_cost"].max()
        dummy_min = group.loc[group["is_dummy"], "action_cost"].min()
        non_dummy_max = 0.0 if pd.isna(non_dummy_max) else float(non_dummy_max)
        if pd.isna(dummy_min) or float(dummy_min) <= non_dummy_max:
            dummy_violations += 1
    if dummy_violations:
        findings.append(finding("High", "Dummy cost is not greater than non-dummy cost", str(dummy_violations), "Raise dummy penalty."))

    severity_counts = Counter(item["severity"] for item in findings)
    if severity_counts["Critical"] or severity_counts["High"]:
        overall = "Needs revision"
    elif severity_counts["Medium"]:
        overall = "Ready for structural MVP tests with caveats"
    else:
        overall = "Ready for structural MVP tests"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall": overall,
        "row_counts": {
            "jobs_manifest": int(len(jobs)),
            "site_timeseries": int(len(site_ts)),
            "actions": int(len(actions)),
            "grid_buses": len(grid.get("bus", [])),
            "grid_branches": len(grid.get("branch", [])),
        },
        "action_mode_counts": actions["mode"].value_counts().to_dict(),
        "job_priority_counts": jobs["priority"].value_counts().to_dict(),
        "findings": findings,
    }


def write_report(results: dict[str, Any]) -> None:
    lines = [
        "# Cleaned Data Validation Report",
        "",
        f"Generated: `{results.get('generated_at')}`",
        f"Overall: **{results.get('overall')}**",
        "",
        "## Row Counts",
        "",
    ]
    for key, value in results.get("row_counts", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Findings", ""])
    if results.get("findings"):
        for idx, item in enumerate(results["findings"], 1):
            lines.append(f"{idx}. [{item['severity']}] {item['title']}")
            lines.append(f"   - Evidence: {item['evidence']}")
            lines.append(f"   - Remediation: {item['remediation']}")
    else:
        lines.append("No findings.")
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "Synthetic price/demand is acceptable only for structural QUBO/solver testing. Replace it with CAISO before claiming market realism.",
        ]
    )
    (REPORTS_DIR / "cleaned_data_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results = validate()
    (REPORTS_DIR / "cleaned_data_validation_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_report(results)
    print(json.dumps({"overall": results.get("overall"), "row_counts": results.get("row_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
