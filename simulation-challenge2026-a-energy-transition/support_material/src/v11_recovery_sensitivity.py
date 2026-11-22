"""Explore recovery definitions on the five completed native trajectories.

AI-assisted with OpenAI Codex / GPT-6, 2026-10-19. This is a secondary
analysis of fixed V10 outputs: dispatch and the original acceptance rule stay fixed.
"""
from itertools import product
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLOUD = ROOT / "outputs/v10/anymath/cloud_final"
OUT = ROOT / "outputs/v11/recovery_sensitivity"
CASES = ("base", "mild", "contract", "power", "decentralized")
FUELS = ("coal", "oil", "gas")
PROTOCOL = {
    "analysis_class": "EXPLORATORY_SECONDARY_ANALYSIS_OF_FIXED_NATIVE_TRAJECTORIES",
    "stock_ratios": [.90, .95, .99],
    "service_shortfall_limits": [.005, .01, .02],
    "window_hours": [12, 24, 48],
    "cases": list(CASES),
    "original_rule": {"stock_ratio": .95, "service_limit": .01, "window": 24},
    "dispatch_changes": False,
    "event_normalization": "each case and event uses its own end-state immediately before that event",
    "boundary": "event 1 ends before event 2; event 2 ends at hour 360",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recovery_windows(frame, config, stock_ratio=.95, service_limit=.01, window=24):
    """Use completed post-shock windows, with all fuels and service checked."""
    starts = [int(config.event1_start_hour), int(config.event2_start_hour)]
    stocks = frame[[f + "_stock_mtce" for f in FUELS]].to_numpy(float)
    rows = []
    for event, start in enumerate(starts):
        stop = start + int(config.event_duration_hours)
        end = starts[event + 1] if event + 1 < len(starts) else int(config.hours)
        pre = stocks[start - 1]
        ratios = stocks / pre

        def ready(first):
            part = frame.iloc[first:first + window]
            service = part.unserved_gw.sum() <= service_limit * part.demand_gw.sum() + 1e-8
            service &= all(
                part[f + "_direct_unserved_mtce"].sum()
                <= service_limit * part[f + "_direct_demand_mtce"].sum() + 1e-8
                for f in FUELS
            )
            stock = bool(np.all(stocks[first:first + window] >= stock_ratio * pre - 1e-8))
            return bool(service), stock

        first_service = first_joint = None
        for first in range(stop, end - window + 1):
            service, stock = ready(first)
            if service and first_service is None:
                first_service = first
            if service and stock and first_joint is None:
                first_joint = first
        rebreach = None
        if first_joint is not None:
            rebreach = next((h for h in range(first_joint + window, end)
                             if np.any(stocks[h] < stock_ratio * pre - 1e-8)), None)
        final_service, final_stock = ready(end - window)
        limiting = np.unravel_index(np.argmin(ratios[end - window:end]), (window, len(FUELS)))
        rows.append({
            "event": event + 1, "shock_end_hour": stop, "observation_end_hour": end,
            "stock_ratio": stock_ratio, "service_limit": service_limit, "window_hours": window,
            "service_confirmation_hour": None if first_service is None else first_service + window - 1,
            "joint_confirmation_hour": None if first_joint is None else first_joint + window - 1,
            "service_right_censored": first_service is None, "joint_right_censored": first_joint is None,
            "stock_rebreach_hour_after_joint_confirmation": rebreach,
            "final_window_service_ready": final_service, "final_window_stock_ready": final_stock,
            "final_window_joint_ready": final_service and final_stock,
            "final_window_stock_floor_ratio": float(np.min(ratios[end - window:end])),
            "limiting_fuel": FUELS[limiting[1]], "limiting_hour": end - window + int(limiting[0]),
        })
    return rows


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2), "utf-8")
    receipt_path = ROOT / "outputs/v10/anymath/execution_receipt.json"
    receipt = json.loads(receipt_path.read_text("utf-8"))
    assert receipt["case_count"] == 5 and receipt["total_hours"] == 1800
    sources = {receipt_path.relative_to(ROOT).as_posix(): digest(receipt_path)}
    rows = []
    for name in CASES:
        paths = {suffix: CLOUD / f"{name}_{suffix}.csv" for suffix in ("actions_states", "config", "recovery")}
        for path in paths.values():
            key = path.relative_to(ROOT).as_posix()
            assert receipt["artifact_sha256"][key] == digest(path), key
            sources[key] = digest(path)
        frame = pd.read_csv(paths["actions_states"])
        config = pd.read_csv(paths["config"]).iloc[0]
        assert len(frame) == 360 and np.array_equal(frame.hour, np.arange(360))
        assert frame.plant_backend.eq("ANYMATH_NATIVE_BLOCK_FEEDBACK").all()
        expected = pd.read_csv(paths["recovery"])
        defaults = recovery_windows(frame, config)
        for actual, original in zip(defaults, expected.to_dict("records")):
            for field in ("event", "service_confirmation_hour", "joint_confirmation_hour",
                          "service_right_censored", "joint_right_censored",
                          "stock_rebreach_hour_after_joint_confirmation"):
                assert (pd.isna(original[field]) if actual[field] is None else actual[field] == original[field]), (name, field)
            assert actual["final_window_joint_ready"] == original["final_24h_joint_ready"]
        for stock, service, window in product(PROTOCOL["stock_ratios"], PROTOCOL["service_shortfall_limits"], PROTOCOL["window_hours"]):
            rows.extend(dict(case=name, **r) for r in recovery_windows(frame, config, stock, service, window))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "all_definitions.csv", index=False, float_format="%.15g")
    intervals = []
    for window in PROTOCOL["window_hours"]:
        for event in (1, 2):
            pair = result[(result.window_hours == window) & (result.event == event)
                          & (result.stock_ratio == .95) & (result.service_limit == .01)].set_index("case")
            intervals.append({"event": event, "window_hours": window,
                              "baseline_stock_floor": float(pair.loc["base", "final_window_stock_floor_ratio"]),
                              "contract_stock_floor": float(pair.loc["contract", "final_window_stock_floor_ratio"]),
                              "both_service_ready": bool(pair.loc[["base", "contract"], "final_window_service_ready"].all()),
                              "interval": "baseline floor < required stock ratio <= contract floor",
                              "limiting_fuel": pair.loc["base", "limiting_fuel"]})
    pd.DataFrame(intervals).to_csv(OUT / "contract_advantage_intervals.csv", index=False, float_format="%.15g")
    report = {"status": "PASS_ORIGINAL_RECOVERY_REPRODUCTION_AND_270_SECONDARY_EVENT_DEFINITIONS",
              "event_definition_count": len(result), "native_hours_reused": 1800, "new_dispatch_runs": 0,
              "protocol": PROTOCOL, "original_rule_matches_all_ten_events": True,
              "source_sha256": sources, "renderer_sha256": digest(__file__),
              "contract_advantage_intervals": intervals,
              "outputs": {p.name: digest(p) for p in (OUT / "all_definitions.csv", OUT / "contract_advantage_intervals.csv")}}
    (OUT / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
    print(json.dumps({k: report[k] for k in ("status", "event_definition_count", "contract_advantage_intervals")}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
