"""Extension E1 — out-of-sample check of the simulated 2025 state.

The multi-agent model is calibrated on the 2010-2024 yearbook vintage.  Its first
simulated year, 2025, is fixed by the 2024 initial state and the construction
queue (identical in every investment plan), so it can be compared with the 2025
statistics published after the model was built: NBS 2025 statistical
communique (28 Feb 2026), NBS energy-mix supplement (Jun 2026) and NEA power
statistics, as preserved in support_material/data/benchmarks_2025.json.
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SM = ROOT / "support_material"
OUT = ROOT / "results" / "extension"; OUT.mkdir(parents=True, exist_ok=True)
bench = json.loads((SM / "data" / "benchmarks_2025.json").read_text(encoding="utf-8"))
plans = sorted((SM / "outputs" / "v6" / "planning").glob("trajectory_plan_*.csv"))
states = pd.concat([pd.read_csv(p).query("year == 2025") for p in plans])
num = states.select_dtypes("number")
physical = ["energy_total_mtce", "electricity_demand_twh", "nonfossil_primary_mtce", "wind_capacity_gw", "solar_capacity_gw", "pumped_capacity_gw",
            "hydro_capacity_gw", "nuclear_capacity_gw", "coal_capacity_gw", "gas_capacity_gw", "biomass_capacity_gw"]
# new orders placed in 2025 differ by plan; the operating 2025 state does not
assert (num[physical].max() - num[physical].min()).abs().max() < 1e-6, "2025 operating state should not depend on the plan"
s = num.iloc[0]
rows = [
    ("Primary energy consumption (Mtce)", s.energy_total_mtce, bench["energy_total_mtce"]),
    ("Electricity demand vs. consumption (TWh)", s.electricity_demand_twh, bench["electricity_consumption_twh"]),
    ("Non-fossil share of primary energy", s.nonfossil_primary_mtce / s.energy_total_mtce, bench["nonfossil_share"]),
    ("Wind capacity (GW)", s.wind_capacity_gw, bench["wind_capacity_gw"]),
    ("Solar capacity (GW)", s.solar_capacity_gw, bench["solar_capacity_gw"]),
    ("Hydro incl. pumped storage (GW)", s.hydro_capacity_gw + s.pumped_capacity_gw, bench["hydro_capacity_gw"]),
    ("Nuclear capacity (GW)", s.nuclear_capacity_gw, bench["nuclear_capacity_gw"]),
    ("Coal + gas + biomass capacity vs. thermal (GW)", s.coal_capacity_gw + s.gas_capacity_gw + s.biomass_capacity_gw, bench["thermal_capacity_gw"]),
]
t = pd.DataFrame(rows, columns=["quantity", "model_2025", "observed_2025"])
t["relative_error_pct"] = 100 * (t.model_2025 / t.observed_2025 - 1)
t.to_csv(OUT / "e1_holdout_2025.csv", index=False)
print(f"{len(plans)} plans share the same 2025 state"); print(t.round(3).to_string(index=False))
