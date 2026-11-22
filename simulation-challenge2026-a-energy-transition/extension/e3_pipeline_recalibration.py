"""Extension E3 — what the 2025 renewable gap identifies, and how much it moves the paper's conclusions.

E1 found the simulated 2025 wind and solar fleets 55 GW and 121 GW below the official statistics while the
energy totals were within about 1 %.  In the model the 2025 fleet is the 2024 fleet plus the commissioning
slot of the *initial construction pipeline*, which the paper estimates from steady throughput: the mean net
addition of 2022-2024 (wind 64 GW/yr, solar 193 GW/yr).  New orders, and therefore the construction lag,
cannot reach 2025 (the shortest lead is one year).  E3 makes that identification explicit and then asks how
far the headline results move when the throughput estimate is recalibrated.

Throughput estimators for the wind and solar net additions (all other technologies unchanged):
  native            mean of 2022-2024 (the paper)
  last year         2024 net addition (wind 79 GW, solar 277 GW)
  linear trend      2022-2024 OLS trend evaluated at 2025 (wind 101 GW, solar 373 GW)
  observed 2025     the realised 2025 net addition (wind 119 GW, solar 315 GW) -- uses 2025 data, so it is a
                    re-anchoring, not a forecast
The estimate enters the initial pipeline and the investors' reference build path (both use the same
throughput in the paper); a pipeline-only variant separates the two channels.  The construction lag
itself is varied with the model's own delay multiplier (1, 1.5, 2) to show that it leaves 2025 unchanged.

Outputs per variant and scenario (baseline, policy): 2025 out-of-sample errors, 2030 wind + solar fleet,
2030 energy CO2, 2030 non-fossil share, cumulative unserved useful service, and 2060 remaining net CO2.
Run:  python extension/e3_pipeline_recalibration.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SM = ROOT / "support_material"
OUT = ROOT / "results" / "extension"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SM))
from src import v6_annual as model, v3_annual as frozen      # noqa: E402
from src.v6_annual_experiments import enrich                 # noqa: E402  (service losses as in the contest run)

WIND, SOLAR = 4, 5
QEND, QDEPTH = model.QEND, model.QDEPTH
ORIGINAL_INIT = model.initialize_state


def throughput(k, variant, bench):
    net = np.diff(k[-4:, [WIND, SOLAR]], axis=0)                       # 2022, 2023, 2024 net additions
    if variant == "native":
        return net.mean(axis=0)
    if variant == "last year":
        return net[-1]
    if variant == "linear trend":
        return np.array([np.polyval(np.polyfit([0, 1, 2], net[:, i], 1), 3) for i in range(2)])
    if variant == "observed 2025":
        return np.array([bench["wind_capacity_gw"] - k[-1, WIND], bench["solar_capacity_gw"] - k[-1, SOLAR]])
    raise ValueError(variant)


def patched_init(variant, channels, bench):
    """initialize_state with the wind/solar throughput replaced in the pipeline and/or the reference path."""
    def init(data, params, start_year=2024, *, mechanisms=None):
        x, p = ORIGINAL_INIT(data, params, start_year, mechanisms=mechanisms)
        if variant == "native":
            return x, p
        _, _, k, *_ = frozen._observations(data, start_year)
        net = throughput(k, variant, bench)
        e = data["config"]["engineering"]
        queue = x[20:QEND].reshape(12, QDEPTH)
        for i, j in enumerate((WIND, SOLAR)):
            ref, r, L = float(x[5 + j]), float(e["retirement_fraction"][j]), int(e["construction_years"][j])
            if "pipeline" in channels:                                  # same construction as the paper's estimate
                for slot in range(L):
                    h = slot + 1
                    queue[j, slot] = ref + net[i] * h - (1 - r) * (ref + net[i] * (h - 1))
            if "anchor" in channels:
                p["reference_net_increment"][j] = float(net[i])
        x[20:QEND] = queue.ravel()
        return x, p
    return init


GROUPS = {"wind + solar": ["wind", "solar"], "firm generation": ["coal", "gas", "nuclear", "hydro", "biomass"],
          "storage": ["battery", "pumped"], "fuel supply": ["fuel_coal", "fuel_oil", "fuel_gas"]}


def metrics(frame, bench, actions=None):
    f = frame.set_index("year")
    spend = {}
    if actions is not None:                                   # cumulative capital spend 2025-2060, trillion CNY
        by = actions.groupby("agent").capital_spend_million_cny.sum() / 1e6
        spend = {f"capex_{g.replace(' ', '_').replace('+', 'and')}_tn_cny": float(by.reindex(a).fillna(0).sum()) for g, a in GROUPS.items()}
    y25, y30, y60 = f.loc[2025], f.loc[2030], f.loc[2060]
    return {"wind_2025_gw": y25.wind_capacity_gw, "solar_2025_gw": y25.solar_capacity_gw,
            "wind_2025_err_pct": 100 * (y25.wind_capacity_gw / bench["wind_capacity_gw"] - 1),
            "solar_2025_err_pct": 100 * (y25.solar_capacity_gw / bench["solar_capacity_gw"] - 1),
            "primary_2025_err_pct": 100 * (y25.energy_total_mtce / bench["energy_total_mtce"] - 1),
            "nonfossil_share_2025_pp": 100 * (y25.nonfossil_primary_mtce / y25.energy_total_mtce - bench["nonfossil_share"]),
            "wind_solar_2030_gw": y30.wind_capacity_gw + y30.solar_capacity_gw,
            "energy_co2_2030_mt": y30.energy_co2_mt,
            "nonfossil_share_2030": y30.nonfossil_primary_mtce / y30.energy_total_mtce,
            "remaining_net_co2_2060_mt": y60.net_co2_mt,
            "cumulative_unserved_twh": y60.cumulative_useful_service_unserved_twh,
            "cumulative_electricity_unserved_twh": y60.cumulative_electricity_unserved_twh,
            "cumulative_direct_fuel_unserved_mtce": y60.cumulative_direct_fuel_unserved_mtce,
            "budget_binding_years": int((f.investment_spend_million_cny >= 0.999 * f.investment_budget_million_cny).sum()),
            "energy_co2_peak_year": int(f.energy_co2_mt.idxmax()), **spend}


def main():
    data = model.load_data(SM)
    protocol = json.loads((SM / "config/v6_annual_experiments.json").read_text("utf-8"))
    params = model.calibrate(data, protocol["calibration_end"])
    bench = json.loads((SM / "data/benchmarks_2025.json").read_text("utf-8"))
    stored = pd.read_csv(SM / "outputs/v6/annual/contest_required_metrics.csv").set_index("scenario")
    _, _, k, *_ = frozen._observations(data, 2024)
    rows = []
    cases = [("native", ("pipeline", "anchor")), ("last year", ("pipeline", "anchor")), ("linear trend", ("pipeline", "anchor")),
             ("observed 2025", ("pipeline", "anchor")), ("observed 2025", ("pipeline",))]
    for scen in ("baseline", "policy"):
        for variant, channels in cases:
            model.initialize_state = patched_init(variant, channels, bench)
            try:
                frame, actions = model.simulate(data, params, scen, start_year=protocol["initial_year"], end_year=protocol["final_year"])
                frame = enrich(frame, actions, data)
            finally:
                model.initialize_state = ORIGINAL_INIT
            m = metrics(frame, bench, actions)
            if variant == "native":                                    # regression check against the stored contest run
                assert abs(m["remaining_net_co2_2060_mt"] - stored.loc[scen, "remaining_net_co2_2060_mt"]) < 1e-6, scen
            net = throughput(k, variant, bench)
            rows.append({"scenario": scen, "throughput": variant, "channels": "+".join(channels), "construction_delay_multiplier": 1.0,
                         "wind_net_gw_per_yr": net[0], "solar_net_gw_per_yr": net[1], **m})
        for mult in (1.5, 2.0):                                         # the construction lag of new orders
            s = {"name": scen, "construction_delay_multiplier": mult}
            frame, actions = model.simulate(data, params, s, start_year=protocol["initial_year"], end_year=protocol["final_year"])
            frame = enrich(frame, actions, data)
            net = throughput(k, "native", bench)
            rows.append({"scenario": scen, "throughput": "native", "channels": "pipeline+anchor", "construction_delay_multiplier": mult,
                         "wind_net_gw_per_yr": net[0], "solar_net_gw_per_yr": net[1], **metrics(frame, bench, actions)})
    df = pd.DataFrame(rows); df.to_csv(OUT / "e3_pipeline_recalibration.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    base = df[(df.throughput == "native") & (df.construction_delay_multiplier == 1.0)].set_index("scenario")
    summ = {"net_additions_2022_2024_gw": {"wind": np.diff(k[-4:, WIND]).round(2).tolist(), "solar": np.diff(k[-4:, SOLAR]).round(2).tolist()},
            "lag_leaves_2025_unchanged": bool(np.allclose(df[df.throughput == "native"].groupby("scenario")[["wind_2025_gw", "solar_2025_gw"]].std().to_numpy(), 0)),
            "changes_vs_native": {}}
    for r in df.itertuples():
        if r.throughput == "native" and r.construction_delay_multiplier == 1.0:
            continue
        b = base.loc[r.scenario]
        key = f"{r.scenario} | {r.throughput} | {r.channels} | delay x{r.construction_delay_multiplier}"
        summ["changes_vs_native"][key] = {"wind_solar_2030_gw": float(r.wind_solar_2030_gw - b.wind_solar_2030_gw),
                                          "energy_co2_2030_mt": float(r.energy_co2_2030_mt - b.energy_co2_2030_mt),
                                          "nonfossil_share_2030_pp": float(100 * (r.nonfossil_share_2030 - b.nonfossil_share_2030)),
                                          "remaining_net_co2_2060_mt": float(r.remaining_net_co2_2060_mt - b.remaining_net_co2_2060_mt),
                                          "cumulative_unserved_twh": float(r.cumulative_unserved_twh - b.cumulative_unserved_twh),
                                          **{c: float(getattr(r, c) - b[c]) for c in df.columns if c.startswith("capex_")}}
    (OUT / "e3_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
