"""Clean raw sources into MVP-ready TAOLIN-DC-SIM tables.

The cleaning follows the project rules:
- one synthetic UTC calendar for relative workload traces;
- one-hour MVP slots, T=24 by default;
- MW/MWh/USD-per-MWh/kgCO2-per-MWh units;
- explicit source flags where a production source is missing.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "instances" / "config.yaml"
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"


@dataclass(frozen=True)
class CleanConfig:
    horizon_slots: int = 24
    delta_t_hours: float = 1.0
    synthetic_anchor_utc: str = "2024-01-01T00:00:00Z"
    workload_csv: str = "data/raw/workload/alibaba_gpu/cluster-trace-gpu-v2023/csv/openb_pod_list_default.csv"
    nsrdb_csv: str = "data/raw/energy/nsrdb/nsrdb_goes_aggregated_v4_0_0_2024_sf_bay_np15_sample.csv"
    wind_csv: str = "data/raw/energy/wind_toolkit/wind_toolkit_2014_sf_bay_np15_sample_100m.csv"
    pglib_case: str = "data/raw/grid/pglib/pglib_opf_case14_ieee.m"
    site_id: str = "site_np15_sample"
    pv_capacity_mw: float = 120.0
    wind_capacity_mw: float = 80.0
    pue: float = 1.2
    gpu_idle_w: float = 60.0
    gpu_peak_w: float = 300.0
    host_base_w: float = 50.0
    host_w_per_cpu_core: float = 20.0
    q995_cap_enabled: bool = True


def ensure_dirs() -> None:
    for path in [INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_project_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return {}


def parse_anchor(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    anchor = datetime.fromisoformat(value)
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    return anchor.astimezone(timezone.utc)


def utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def priority_from_qos(qos: str) -> str:
    text = str(qos).strip().lower()
    if text in {"guaranteed", "ls"}:
        return "high"
    if text in {"burstable"}:
        return "medium"
    return "low"


def qos_to_modes(qos: str) -> list[str]:
    priority = priority_from_qos(qos)
    if priority == "high":
        return ["normal", "delay", "migrate", "dummy"]
    if priority == "medium":
        return ["normal", "delay", "degrade", "migrate", "dummy"]
    return ["normal", "delay", "degrade", "dummy"]


def sla_delta_slots(priority: str, duration_slots: int) -> int:
    factor = {"high": 0.25, "medium": 0.50, "low": 1.00}[priority]
    return max(1, int(math.ceil(factor * duration_slots)))


def sla_penalty(priority: str) -> float:
    return {"high": 200.0, "medium": 100.0, "low": 50.0}[priority]


def migration_cost(priority: str, duration_slots: int) -> float:
    return {"high": 80.0, "medium": 50.0, "low": 30.0}[priority] * max(1, duration_slots)


def clamp(series: pd.Series, lo: float, hi: float) -> pd.Series:
    return series.astype(float).clip(lower=lo, upper=hi)


def clean_workload(cfg: CleanConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source_path = ROOT / cfg.workload_csv
    raw = pd.read_csv(source_path)
    raw["source_row_id"] = np.arange(len(raw))
    raw["source_file"] = source_path.relative_to(ROOT).as_posix()

    numeric_cols = [
        "cpu_milli",
        "memory_mib",
        "num_gpu",
        "gpu_milli",
        "creation_time",
        "deletion_time",
        "scheduled_time",
    ]
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    audit: dict[str, Any] = {
        "source_file": source_path.relative_to(ROOT).as_posix(),
        "raw_rows": int(len(raw)),
        "rules": {
            "duration_positive": "deletion_time > scheduled_time",
            "gpu_required": "num_gpu > 0 and gpu_milli > 0",
            "resource_required": "cpu_milli > 0 and memory_mib > 0",
            "timestamp_required": "scheduled_time, deletion_time, creation_time are non-null",
            "duplicate_job_id": "drop duplicate name after validation",
            "outlier_duration": "cap duration at q99.5 but preserve raw_long_jobs.parquet",
        },
    }

    valid = raw.copy()
    valid["drop_reason"] = ""
    invalid_masks = {
        "missing_timestamp": valid[["creation_time", "deletion_time", "scheduled_time"]].isna().any(axis=1),
        "gpu_missing_or_zero": (valid["num_gpu"].fillna(0) <= 0) | (valid["gpu_milli"].fillna(0) <= 0),
        "resource_missing_or_zero": (valid["cpu_milli"].fillna(0) <= 0) | (valid["memory_mib"].fillna(0) <= 0),
        "finish_not_after_submit": valid["deletion_time"].fillna(-1) <= valid["scheduled_time"].fillna(np.inf),
        "scheduled_before_creation": (valid["scheduled_time"].fillna(0) != 0)
        & (valid["scheduled_time"].fillna(np.inf) < valid["creation_time"].fillna(-np.inf)),
    }
    keep_mask = pd.Series(True, index=valid.index)
    reject_counts: dict[str, int] = {}
    for reason, mask in invalid_masks.items():
        reject_counts[reason] = int(mask.sum())
        valid.loc[mask & (valid["drop_reason"] == ""), "drop_reason"] = reason
        keep_mask &= ~mask
    duplicates = valid.loc[keep_mask, "name"].duplicated(keep="first")
    duplicate_index = duplicates[duplicates].index
    valid.loc[duplicate_index, "drop_reason"] = "duplicate_job_id"
    reject_counts["duplicate_job_id"] = int(len(duplicate_index))
    keep_mask.loc[duplicate_index] = False

    cleaned_all = valid.loc[keep_mask].copy()
    cleaned_all["raw_duration_seconds"] = cleaned_all["deletion_time"] - cleaned_all["scheduled_time"]
    cleaned_all["raw_duration_hours"] = cleaned_all["raw_duration_seconds"] / 3600.0
    cap_seconds = float(cleaned_all["raw_duration_seconds"].quantile(0.995))
    if cfg.q995_cap_enabled:
        cleaned_all["capped_duration_seconds"] = np.minimum(cleaned_all["raw_duration_seconds"], cap_seconds)
    else:
        cleaned_all["capped_duration_seconds"] = cleaned_all["raw_duration_seconds"]
    cleaned_all["duration_was_capped"] = cleaned_all["capped_duration_seconds"] < cleaned_all["raw_duration_seconds"]
    cleaned_all["duration_slots"] = np.ceil(cleaned_all["capped_duration_seconds"] / (cfg.delta_t_hours * 3600)).astype(int)
    cleaned_all["duration_slots"] = cleaned_all["duration_slots"].clip(lower=1)

    raw_long_jobs = cleaned_all.loc[
        cleaned_all["raw_duration_seconds"] > cap_seconds,
        [
            "name",
            "source_row_id",
            "cpu_milli",
            "memory_mib",
            "num_gpu",
            "gpu_milli",
            "qos",
            "pod_phase",
            "creation_time",
            "scheduled_time",
            "deletion_time",
            "raw_duration_seconds",
            "raw_duration_hours",
            "capped_duration_seconds",
            "duration_slots",
        ],
    ].copy()

    day_counts = (cleaned_all["scheduled_time"] // 86400).astype(int).value_counts().sort_values(ascending=False)
    selected_relative_day = int(day_counts.index[0])
    selected_day_start_s = selected_relative_day * 86400
    selected_day_end_s = selected_day_start_s + cfg.horizon_slots * cfg.delta_t_hours * 3600
    horizon = cleaned_all.loc[
        (cleaned_all["scheduled_time"] >= selected_day_start_s)
        & (cleaned_all["scheduled_time"] < selected_day_end_s)
    ].copy()
    horizon["release_slot"] = (
        np.floor((horizon["scheduled_time"] - selected_day_start_s) / (cfg.delta_t_hours * 3600)).astype(int) + 1
    )
    horizon["finish_slot_raw"] = horizon["release_slot"] + np.ceil(horizon["raw_duration_seconds"] / 3600).astype(int)
    horizon["finish_slot_capped"] = horizon["release_slot"] + horizon["duration_slots"]
    horizon["priority"] = horizon["qos"].map(priority_from_qos)
    horizon["deadline_delta_slots"] = [
        sla_delta_slots(priority, duration)
        for priority, duration in zip(horizon["priority"], horizon["duration_slots"])
    ]
    horizon["deadline_slot"] = horizon["release_slot"] + horizon["duration_slots"] + horizon["deadline_delta_slots"]
    horizon["sla_penalty_per_slot"] = horizon["priority"].map(sla_penalty)
    horizon["migration_cost"] = [
        migration_cost(priority, duration)
        for priority, duration in zip(horizon["priority"], horizon["duration_slots"])
    ]
    horizon["preemptible"] = horizon["priority"].isin(["medium", "low"])
    horizon["migratable"] = horizon["priority"].isin(["high", "medium"])
    horizon["mode_set"] = horizon["qos"].map(lambda value: ",".join(qos_to_modes(value)))
    horizon["gpu_util_avg"] = clamp(horizon["gpu_milli"] / 1000.0, 0.0, 1.0)
    horizon["gpu_req"] = horizon["num_gpu"].astype(int)
    host_w = cfg.host_base_w + cfg.host_w_per_cpu_core * (horizon["cpu_milli"] / 1000.0)
    gpu_it_w = horizon["gpu_req"] * (
        cfg.gpu_idle_w + horizon["gpu_util_avg"] * (cfg.gpu_peak_w - cfg.gpu_idle_w)
    )
    horizon["power_it_mw"] = (gpu_it_w + host_w) / 1_000_000.0
    horizon["pue"] = cfg.pue
    horizon["power_total_mw"] = horizon["pue"] * horizon["power_it_mw"]
    horizon["power_model"] = "trace_calibrated_gpu_power_proxy"
    horizon["utilization_source"] = "gpu_milli_request_proxy"

    anchor = parse_anchor(cfg.synthetic_anchor_utc)
    horizon["timestamp_utc"] = [
        utc_iso(anchor + timedelta(hours=int(slot - 1))) for slot in horizon["release_slot"]
    ]
    horizon["local_time"] = horizon["timestamp_utc"]
    horizon["workload"] = "alibaba_gpu_v2023_default"
    horizon["job_id"] = [f"job_{i:05d}" for i in range(len(horizon))]
    horizon["source_job_id"] = horizon["name"].astype(str)
    horizon["cleaning_status"] = np.where(horizon["duration_was_capped"], "valid_duration_capped", "valid")

    cols = [
        "job_id",
        "source_job_id",
        "workload",
        "source_file",
        "source_row_id",
        "timestamp_utc",
        "local_time",
        "release_slot",
        "deadline_slot",
        "deadline_delta_slots",
        "duration_slots",
        "finish_slot_raw",
        "finish_slot_capped",
        "raw_duration_seconds",
        "raw_duration_hours",
        "capped_duration_seconds",
        "duration_was_capped",
        "gpu_req",
        "gpu_milli",
        "gpu_util_avg",
        "cpu_milli",
        "memory_mib",
        "gpu_spec",
        "qos",
        "pod_phase",
        "priority",
        "preemptible",
        "migratable",
        "mode_set",
        "sla_penalty_per_slot",
        "migration_cost",
        "power_it_mw",
        "pue",
        "power_total_mw",
        "power_model",
        "utilization_source",
        "creation_time",
        "scheduled_time",
        "deletion_time",
        "cleaning_status",
    ]
    jobs = horizon[cols].sort_values(["release_slot", "job_id"]).reset_index(drop=True)

    audit.update(
        {
            "reject_counts": reject_counts,
            "valid_rows_after_rules": int(len(cleaned_all)),
            "q995_duration_seconds": cap_seconds,
            "raw_long_jobs": int(len(raw_long_jobs)),
            "selected_relative_day": selected_relative_day,
            "selected_day_start_seconds": selected_day_start_s,
            "selected_day_end_seconds": selected_day_end_s,
            "horizon_jobs": int(len(jobs)),
            "horizon_slots": cfg.horizon_slots,
            "delta_t_hours": cfg.delta_t_hours,
            "synthetic_anchor_utc": cfg.synthetic_anchor_utc,
            "priority_counts": jobs["priority"].value_counts().to_dict(),
            "release_slot_counts": jobs["release_slot"].value_counts().sort_index().to_dict(),
        }
    )
    return jobs, raw_long_jobs, audit


def read_nsrdb(cfg: CleanConfig) -> pd.DataFrame:
    path = ROOT / cfg.nsrdb_csv
    return pd.read_csv(path, skiprows=2)


def read_wind(cfg: CleanConfig) -> pd.DataFrame:
    path = ROOT / cfg.wind_csv
    return pd.read_csv(path, skiprows=1)


def clean_site_timeseries(cfg: CleanConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    nsrdb = read_nsrdb(cfg).head(cfg.horizon_slots).copy()
    wind = read_wind(cfg).head(cfg.horizon_slots).copy()
    anchor = parse_anchor(cfg.synthetic_anchor_utc)
    slot = np.arange(1, cfg.horizon_slots + 1)

    pv_cf = clamp(nsrdb["GHI"] / 1000.0, 0.0, 1.0)
    wind_speed = pd.to_numeric(wind["wind speed at 100m (m/s)"], errors="coerce")
    wind_cf = ((wind_speed - 3.0) / (12.0 - 3.0)).clip(lower=0.0, upper=1.0) ** 3
    pv_mw = cfg.pv_capacity_mw * pv_cf
    wind_mw = cfg.wind_capacity_mw * wind_cf
    renewable_mw = pv_mw + wind_mw
    renewable_cf = renewable_mw / (cfg.pv_capacity_mw + cfg.wind_capacity_mw)

    hour_angle = 2 * np.pi * (slot - 1) / cfg.horizon_slots
    demand_forecast_mw = 520.0 + 80.0 * np.sin(hour_angle - np.pi / 3) + 25.0 * np.cos(2 * hour_angle)
    demand_forecast_mw = np.maximum(demand_forecast_mw, 350.0)
    scarcity = 1.0 - renewable_cf
    lmp = 55.0 + 35.0 * scarcity + 8.0 * np.sin(hour_angle - np.pi / 2)
    lmp = np.maximum(lmp, 5.0)
    carbon = np.clip(420.0 - 180.0 * renewable_cf + 20.0 * np.cos(hour_angle), 120.0, 600.0)

    df = pd.DataFrame(
        {
            "site": cfg.site_id,
            "slot": slot,
            "timestamp_utc": [utc_iso(anchor + timedelta(hours=int(i - 1))) for i in slot],
            "local_time": [utc_iso(anchor + timedelta(hours=int(i - 1))) for i in slot],
            "delta_t_hours": cfg.delta_t_hours,
            "pv_cf": pv_cf.round(6),
            "wind_cf": wind_cf.round(6),
            "pv_capacity_mw": cfg.pv_capacity_mw,
            "wind_capacity_mw": cfg.wind_capacity_mw,
            "pv_mw": pv_mw.round(6),
            "wind_mw": wind_mw.round(6),
            "renewable_mw": renewable_mw.round(6),
            "demand_forecast_mw": demand_forecast_mw.round(6),
            "flex_call_prob": np.clip(0.15 + 0.35 * scarcity, 0.0, 1.0).round(6),
            "flex_price_cap_usd_mwh": (lmp + 40.0).round(6),
            "flex_price_cut_usd_mwh": np.maximum(lmp - 15.0, 0.0).round(6),
            "lmp_usd_mwh": lmp.round(6),
            "carbon_kgco2_mwh": carbon.round(6),
            "penalty_short_usd_mwh": 250.0,
            "source_year_nsrdb": nsrdb["Year"].astype(int).to_numpy(),
            "source_year_wind": wind["Year"].astype(int).to_numpy(),
            "source_time_policy": "synthetic_utc_slot_alignment",
            "price_source": "synthetic_fallback_caiso_missing",
            "is_synthetic_price": True,
            "power_unit": "MW",
            "energy_unit": "MWh",
            "price_unit": "USD/MWh",
            "carbon_unit": "kgCO2/MWh",
        }
    )
    audit = {
        "nsrdb_rows_used": int(len(nsrdb)),
        "wind_rows_used": int(len(wind)),
        "capacity_factor_rule": "clip to [0,1]; PV uses GHI/1000; wind uses cubic cut-in/rated proxy from 3 to 12 m/s",
        "price_rule": "synthetic fallback because CAISO raw files are absent; not valid for market claims",
        "timestamp_policy": "both weather sources are historical samples from different years, aligned only by synthetic UTC slot for MVP testing",
    }
    return df, audit


def parse_matpower_matrix(text: str, field: str) -> list[list[float]]:
    pattern = re.compile(rf"mpc\.{field}\s*=\s*\[(.*?)\];", re.DOTALL)
    match = pattern.search(text)
    if not match:
        return []
    rows = []
    for line in match.group(1).splitlines():
        line = line.split("%", 1)[0].strip()
        if not line:
            continue
        line = line.rstrip(";")
        if not line:
            continue
        values = [float(token) for token in line.split()]
        rows.append(values)
    return rows


def clean_grid_case(cfg: CleanConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    path = ROOT / cfg.pglib_case
    text = path.read_text(encoding="utf-8", errors="replace")
    function_match = re.search(r"function\s+mpc\s*=\s*(\w+)", text)
    base_mva_match = re.search(r"mpc\.baseMVA\s*=\s*([0-9.]+);", text)
    grid = {
        "case_name": function_match.group(1) if function_match else path.stem,
        "source_file": path.relative_to(ROOT).as_posix(),
        "format": "MATPOWER",
        "baseMVA": float(base_mva_match.group(1)) if base_mva_match else None,
        "bus_columns": ["bus_i", "type", "Pd", "Qd", "Gs", "Bs", "area", "Vm", "Va", "baseKV", "zone", "Vmax", "Vmin"],
        "gen_columns": ["bus", "Pg", "Qg", "Qmax", "Qmin", "Vg", "mBase", "status", "Pmax", "Pmin", "Pc1", "Pc2", "Qc1min", "Qc1max", "Qc2min", "Qc2max", "ramp_agc", "ramp_10", "ramp_30", "ramp_q", "apf"],
        "branch_columns": ["fbus", "tbus", "r", "x", "b", "rateA", "rateB", "rateC", "ratio", "angle", "status", "angmin", "angmax"],
        "gencost_columns": ["model", "startup", "shutdown", "n", "c2", "c1", "c0"],
        "bus": parse_matpower_matrix(text, "bus"),
        "gen": parse_matpower_matrix(text, "gen"),
        "branch": parse_matpower_matrix(text, "branch"),
        "gencost": parse_matpower_matrix(text, "gencost"),
        "dc_site_to_bus": {cfg.site_id: 2},
        "renewable_to_bus": {"pv": 3, "wind": 6},
        "notes": "MVP DC power-flow approximation input; not AC OPF. Bus mappings are deterministic placeholders for TAOLIN-DC-SIM experiments.",
    }
    audit = {
        "source_file": path.relative_to(ROOT).as_posix(),
        "bus_count": len(grid["bus"]),
        "gen_count": len(grid["gen"]),
        "branch_count": len(grid["branch"]),
        "gencost_count": len(grid["gencost"]),
        "case_name": grid["case_name"],
    }
    return grid, audit


def vector_for_interval(length: int, start_slot: int, end_slot: int, value: float) -> list[float]:
    vector = [0.0] * length
    for idx in range(max(1, start_slot), min(length, end_slot - 1) + 1):
        vector[idx - 1] = round(float(value), 9)
    return vector


def action_cost(
    job: pd.Series,
    site_ts: pd.DataFrame,
    start_slot: int,
    end_slot: int,
    power_mw: float,
    extra_cost: float = 0.0,
) -> tuple[float, float, float]:
    if start_slot <= 0 or end_slot <= start_slot:
        return 0.0, 0.0, extra_cost
    slots = list(range(start_slot, min(end_slot, len(site_ts) + 1)))
    if not slots:
        return 0.0, 0.0, extra_cost
    prices = site_ts.set_index("slot").loc[slots, "lmp_usd_mwh"].astype(float)
    energy_mwh = power_mw * len(slots)
    energy_cost = float((prices * power_mw).sum())
    completion_slot = end_slot
    sla_cost = float(job["sla_penalty_per_slot"] * max(0, completion_slot - int(job["deadline_slot"])))
    return energy_mwh, energy_cost, energy_cost + sla_cost + extra_cost


def build_actions(jobs: pd.DataFrame, site_ts: pd.DataFrame, cfg: CleanConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    max_non_dummy_by_job: dict[str, float] = {}
    site = cfg.site_id

    for _, job in jobs.iterrows():
        job_actions = []
        duration = int(job["duration_slots"])
        release = int(job["release_slot"])
        power = float(job["power_total_mw"])
        gpu_req = int(job["gpu_req"])
        modes = set(str(job["mode_set"]).split(","))

        candidates: list[tuple[str, int, int, float, float]] = []
        if "normal" in modes:
            candidates.append(("normal", release, release + duration, power, 0.0))
        if "delay" in modes:
            delayed_start = release + 1
            candidates.append(("delay", delayed_start, delayed_start + duration, power, 0.0))
        if "degrade" in modes:
            degraded_duration = int(math.ceil(duration * 1.25))
            degraded_power = power * 0.70
            candidates.append(("degrade", release, release + degraded_duration, degraded_power, 0.20 * float(job["sla_penalty_per_slot"]) * duration))

        for mode, start_slot, end_slot, mode_power, extra_cost in candidates:
            if start_slot < 1 or end_slot > cfg.horizon_slots + 1 or end_slot <= start_slot:
                continue
            energy_mwh, energy_cost_usd, total_cost_usd = action_cost(job, site_ts, start_slot, end_slot, mode_power, extra_cost)
            record = {
                "action_id": "",
                "job_id": job["job_id"],
                "site": site,
                "start_slot": start_slot,
                "end_slot": end_slot,
                "mode": mode,
                "gpu_req": gpu_req,
                "power_mw": round(mode_power, 9),
                "energy_mwh": round(energy_mwh, 9),
                "energy_cost_usd": round(energy_cost_usd, 6),
                "sla_cost_usd": round(max(0.0, total_cost_usd - energy_cost_usd - extra_cost), 6),
                "extra_mode_cost_usd": round(extra_cost, 6),
                "action_cost": round(total_cost_usd, 6),
                "is_dummy": False,
                "gpu_vector": json.dumps(vector_for_interval(cfg.horizon_slots, start_slot, end_slot, gpu_req)),
                "power_vector_mw": json.dumps(vector_for_interval(cfg.horizon_slots, start_slot, end_slot, mode_power)),
                "conflict_keys": json.dumps([f"job:{job['job_id']}"]),
                "source": "clean_mvp_data",
            }
            job_actions.append(record)

        if not job_actions and release <= cfg.horizon_slots:
            rescue_start = max(1, min(release, cfg.horizon_slots))
            rescue_end = cfg.horizon_slots + 1
            active_slots = max(0, rescue_end - rescue_start)
            if active_slots > 0:
                unfinished_slots = max(0, duration - active_slots)
                rescue_extra_cost = float(job["sla_penalty_per_slot"]) * (
                    unfinished_slots + 0.25 * max(1, duration)
                )
                energy_mwh, energy_cost_usd, total_cost_usd = action_cost(
                    job,
                    site_ts,
                    rescue_start,
                    rescue_end,
                    power,
                    rescue_extra_cost,
                )
                job_actions.append(
                    {
                        "action_id": "",
                        "job_id": job["job_id"],
                        "site": site,
                        "start_slot": rescue_start,
                        "end_slot": rescue_end,
                        "mode": "horizon_rescue",
                        "gpu_req": gpu_req,
                        "power_mw": round(power, 9),
                        "energy_mwh": round(energy_mwh, 9),
                        "energy_cost_usd": round(energy_cost_usd, 6),
                        "sla_cost_usd": round(max(0.0, total_cost_usd - energy_cost_usd - rescue_extra_cost), 6),
                        "extra_mode_cost_usd": round(rescue_extra_cost, 6),
                        "action_cost": round(total_cost_usd, 6),
                        "is_dummy": False,
                        "gpu_vector": json.dumps(vector_for_interval(cfg.horizon_slots, rescue_start, rescue_end, gpu_req)),
                        "power_vector_mw": json.dumps(vector_for_interval(cfg.horizon_slots, rescue_start, rescue_end, power)),
                        "conflict_keys": json.dumps([f"job:{job['job_id']}"]),
                        "source": "horizon_edge_partial_rescue",
                    }
                )

        max_cost = max([item["action_cost"] for item in job_actions], default=0.0)
        max_non_dummy_by_job[str(job["job_id"])] = max_cost
        dummy_cost = max(1_000_000.0, max_cost + 100_000.0 + float(job["sla_penalty_per_slot"]) * max(1, duration))
        dummy = {
            "action_id": "",
            "job_id": job["job_id"],
            "site": site,
            "start_slot": 0,
            "end_slot": 0,
            "mode": "dummy",
            "gpu_req": 0,
            "power_mw": 0.0,
            "energy_mwh": 0.0,
            "energy_cost_usd": 0.0,
            "sla_cost_usd": 0.0,
            "extra_mode_cost_usd": 0.0,
            "action_cost": round(dummy_cost, 6),
            "is_dummy": True,
            "gpu_vector": json.dumps([0.0] * cfg.horizon_slots),
            "power_vector_mw": json.dumps([0.0] * cfg.horizon_slots),
            "conflict_keys": json.dumps([f"job:{job['job_id']}"]),
            "source": "dummy_unserved_or_infeasible",
        }
        actions.extend(job_actions)
        actions.append(dummy)

    action_df = pd.DataFrame(actions)
    action_df["action_id"] = [f"act_{i:06d}" for i in range(len(action_df))]
    audit = {
        "actions": int(len(action_df)),
        "non_dummy_actions": int((~action_df["is_dummy"]).sum()),
        "dummy_actions": int(action_df["is_dummy"].sum()),
        "jobs_without_non_dummy_action": int(
            sum(max_non_dummy_by_job[job_id] == 0.0 for job_id in max_non_dummy_by_job)
        ),
        "mode_counts": action_df["mode"].value_counts().to_dict(),
        "dummy_cost_rule": "max(1e6, max_non_dummy_job_action_cost + 100000 + sla_penalty_per_slot * duration_slots)",
    }
    return action_df, audit


def write_report(audit: dict[str, Any]) -> None:
    lines = [
        "# Data Cleaning Report",
        "",
        f"Generated: `{audit['generated_at']}`",
        "",
        "## Cleaning Contract",
        "",
        "- Time: one-hour UTC slots, `t = 1..24`, synthetic anchor for relative workload traces.",
        "- Units: power MW, energy MWh, price USD/MWh, carbon kgCO2/MWh.",
        "- Workload: Alibaba v2023 default pod list for MVP; long jobs preserved in `raw_long_jobs.parquet`.",
        "- Power: trace-calibrated GPU proxy using requested GPU milli as utilization proxy; not real A100 telemetry.",
        "- Price: synthetic fallback because CAISO raw files are absent; not valid for market-price claims.",
        "",
        "## Outputs",
        "",
        f"- `data/interim/jobs_manifest.parquet`: {audit['outputs']['jobs_manifest_rows']} rows.",
        f"- `data/interim/site_timeseries.parquet`: {audit['outputs']['site_timeseries_rows']} rows.",
        f"- `data/interim/grid_case.json`: {audit['grid']['case_name']} with {audit['grid']['bus_count']} buses.",
        f"- `data/processed/actions.parquet`: {audit['outputs']['actions_rows']} rows.",
        f"- `data/interim/raw_long_jobs.parquet`: {audit['workload']['raw_long_jobs']} rows.",
        "",
        "## Key Cleaning Evidence",
        "",
        f"- Raw workload rows: {audit['workload']['raw_rows']}; valid after rules: {audit['workload']['valid_rows_after_rules']}; horizon jobs: {audit['workload']['horizon_jobs']}.",
        f"- Reject counts: `{audit['workload']['reject_counts']}`.",
        f"- Duration q99.5 cap: {audit['workload']['q995_duration_seconds']:.2f} seconds.",
        f"- Action mode counts: `{audit['actions']['mode_counts']}`.",
        "",
        "## Caveats",
        "",
        "- CAISO must be downloaded before using `lmp_usd_mwh` or demand/price conclusions as real market evidence.",
        "- Wind power is derived from wind speed through a simple proxy; replace with a documented turbine curve for production.",
        "- The single-site MVP has no real migration destination; migrate remains a mode-set option but action generation only emits feasible local actions plus dummy.",
        "- PGLib bus mappings for data center and renewables are deterministic placeholders.",
    ]
    (REPORTS_DIR / "cleaning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: CleanConfig) -> dict[str, Any]:
    ensure_dirs()
    jobs, raw_long_jobs, workload_audit = clean_workload(cfg)
    site_ts, site_audit = clean_site_timeseries(cfg)
    grid_case, grid_audit = clean_grid_case(cfg)
    actions, action_audit = build_actions(jobs, site_ts, cfg)

    jobs.to_parquet(INTERIM_DIR / "jobs_manifest.parquet", index=False)
    raw_long_jobs.to_parquet(INTERIM_DIR / "raw_long_jobs.parquet", index=False)
    site_ts.to_parquet(INTERIM_DIR / "site_timeseries.parquet", index=False)
    (INTERIM_DIR / "grid_case.json").write_text(json.dumps(grid_case, ensure_ascii=False, indent=2), encoding="utf-8")
    actions.to_parquet(PROCESSED_DIR / "actions.parquet", index=False)

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "project_config": load_project_config(),
        "workload": workload_audit,
        "site_timeseries": site_audit,
        "grid": grid_audit,
        "actions": action_audit,
        "outputs": {
            "jobs_manifest_rows": int(len(jobs)),
            "site_timeseries_rows": int(len(site_ts)),
            "actions_rows": int(len(actions)),
            "raw_long_jobs_rows": int(len(raw_long_jobs)),
        },
    }
    (REPORTS_DIR / "cleaning_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_report(audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-slots", type=int, default=24)
    parser.add_argument("--delta-t-hours", type=float, default=1.0)
    parser.add_argument("--anchor-utc", default="2024-01-01T00:00:00Z")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CleanConfig(
        horizon_slots=args.horizon_slots,
        delta_t_hours=args.delta_t_hours,
        synthetic_anchor_utc=args.anchor_utc,
    )
    audit = run(cfg)
    print(json.dumps({"outputs": audit["outputs"], "report": str(REPORTS_DIR / "cleaning_report.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
