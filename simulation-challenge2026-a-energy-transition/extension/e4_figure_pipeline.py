"""Figure E2: what the 2025 renewable gap identifies, and what correcting it changes.

(a) 2025 error of the simulated wind and solar fleets for three throughput estimators of the initial
    construction pipeline and for two longer construction lags of new orders (which cannot reach 2025);
(b) change against the paper's run, by 2060, in remaining net CO2 and in cumulative unserved useful service.
Reads results/extension/e3_pipeline_recalibration.csv (e3_pipeline_recalibration.py).
Run:  python extension/e4_figure_pipeline.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "extension"; FIG.mkdir(parents=True, exist_ok=True); fs.apply()
d = pd.read_csv(ROOT / "results" / "extension" / "e3_pipeline_recalibration.csv")
b = d[d.scenario == "baseline"].copy()
b["label"] = np.select(
    [(b.throughput == "native") & (b.construction_delay_multiplier == 1),
     b.construction_delay_multiplier > 1, b.throughput == "observed 2025"],
    ["paper: mean 2022–2024 throughput",
     "construction lag × " + b.construction_delay_multiplier.map("{:g}".format),
     "observed 2025 throughput (" + b.channels.map({"pipeline+anchor": "pipeline + investor anchor", "pipeline": "pipeline only"}) + ")"],
    b.throughput + " throughput")

fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.DOUBLE, 0.36 * fs.DOUBLE), gridspec_kw=dict(width_ratios=[1, 1.05], wspace=0.75))
# (a) identification: 2025 errors
rows = b[b.throughput != "observed 2025"].drop_duplicates("label")
order = ["paper: mean 2022–2024 throughput", "construction lag × 1.5", "construction lag × 2", "last year throughput", "linear trend throughput"]
shown = {"last year throughput": "2024 throughput only", "linear trend throughput": "2022–2024 linear trend"}
rows = rows.set_index("label").loc[order].rename(index=shown)
y = np.arange(len(rows)); h = 0.36
ax.barh(y - h / 2, rows.wind_2025_err_pct, height=h, color=fs.OKABE_ITO[0], label="wind")
ax.barh(y + h / 2, rows.solar_2025_err_pct, height=h, color=fs.OKABE_ITO[4], label="solar")
for yi, (w, s) in enumerate(zip(rows.wind_2025_err_pct, rows.solar_2025_err_pct)):
    for v, off in ((w, -h / 2), (s, h / 2)):
        ax.text(v + (0.3 if v >= 0 else -0.3), yi + off, f"{v:+.1f}".replace("-", "−"), va="center", ha="left" if v >= 0 else "right", fontsize=5.5)
ax.axvline(0, color="k", lw=0.6)
ax.set_yticks(y); ax.set_yticklabels(rows.index); ax.invert_yaxis()
ax.set_xlim(-14.5, 9); ax.set_xlabel("2025 fleet, model minus observed (% of observed)")
ax.set_title("(a) the 2025 gap moves with throughput, not with the lag", loc="left")
ax.legend(loc="upper right", handlelength=1.0)
# (b) consequences by 2060 (baseline filled, policy open)
nat = d[(d.throughput == "native") & (d.construction_delay_multiplier == 1)].set_index("scenario")
pts = d[~((d.throughput == "native") & (d.construction_delay_multiplier == 1))].copy()
pts["dco2"] = [(r.remaining_net_co2_2060_mt - nat.loc[r.scenario, "remaining_net_co2_2060_mt"]) / 1e3 for r in pts.itertuples()]
pts["dun"] = [(r.cumulative_unserved_twh - nat.loc[r.scenario, "cumulative_unserved_twh"]) / 1e3 for r in pts.itertuples()]
style = {("last year", "pipeline+anchor", 1.0): (fs.OKABE_ITO[2], "2024 throughput"),
         ("linear trend", "pipeline+anchor", 1.0): (fs.OKABE_ITO[5], "2022–2024 linear trend"),
         ("observed 2025", "pipeline+anchor", 1.0): (fs.OKABE_ITO[1], "observed 2025,\npipeline + anchor"),
         ("observed 2025", "pipeline", 1.0): (fs.OKABE_ITO[3], "observed 2025, pipeline only"),
         ("native", "pipeline+anchor", 1.5): ("0.45", "lag × 1.5"), ("native", "pipeline+anchor", 2.0): ("0.15", "lag × 2")}
offsets = {"2024 throughput": (6, -2), "2022–2024 linear trend": (6, 2), "observed 2025,\npipeline + anchor": (6, -10),
           "observed 2025, pipeline only": (-8, -4), "lag × 1.5": (-6, 4), "lag × 2": (6, -2)}
bx.axhline(0, color="0.6", lw=0.5); bx.axvline(0, color="0.6", lw=0.5)
for key, (c, lab) in style.items():
    for scen, mk in (("baseline", "o"), ("policy", "s")):
        r = pts[(pts.throughput == key[0]) & (pts.channels == key[1]) & (pts.construction_delay_multiplier == key[2]) & (pts.scenario == scen)].iloc[0]
        bx.plot(r.dco2, r.dun, mk, ms=4.2, color=c, mfc=c if scen == "baseline" else "white", mew=0.9)
        if scen == "baseline":
            dx, dy = offsets[lab]
            bx.annotate(lab, (r.dco2, r.dun), xytext=(dx, dy), textcoords="offset points", fontsize=5.5,
                        ha="left" if dx > 0 else "right", va="center", color=c)
bx.set_xlabel("change in 2060 remaining net CO$_2$ (Gt)"); bx.set_ylabel("change in cumulative unserved\nuseful service 2025–2060 (PWh)")
bx.set_title("(b) faster renewables cut CO$_2$ but crowd out supply", loc="left")
bx.plot([], [], "o", color="0.3", ms=4, label="baseline"); bx.plot([], [], "s", color="0.3", mfc="white", ms=4, label="policy")
bx.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), handlelength=0.8)
lo, hi = bx.get_xlim(); bx.set_xlim(lo - 0.1, hi + 0.35); bx.set_ylim(bottom=-12)
fs.save(fig, FIG / "figE2_pipeline_recalibration")
