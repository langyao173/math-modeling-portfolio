"""Figure for E2 (density scaling, script e3_density_scaling.py): core size, device fit and classical solve time against demand density.

Reads results/extension/e3_density_runs.csv and e3_density_periods.csv.gz; with --partial it previews the
runs finished so far (results/extension/_e3_work/done/).
"""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results" / "extension", ROOT / "figures" / "extension"
FIG.mkdir(parents=True, exist_ok=True); fs.apply()

if "--partial" in sys.argv:
    recs = [json.loads(p.read_text()) for p in sorted((RES / "_e3_work" / "done").glob("*.json"))]
    runs = pd.DataFrame([r["run"] for r in recs]); per = pd.DataFrame([x for r in recs for x in r["periods"]])
else:
    runs = pd.read_csv(RES / "e3_density_runs.csv"); per = pd.read_csv(RES / "e3_density_periods.csv.gz")
per = per.merge(runs[["run", "density", "fleet", "epoch"]], on="run")
per["fleet_per_240"] = (per.fleet / per.density).round().astype(int)
CAP = 549                                                   # binaries that fit SPQC-550 with the reference spin
styles = {(16, 60): (fs.OKABE_ITO[0], "-", "o"), (16, 300): (fs.OKABE_ITO[0], "--", "s"),
          (32, 60): (fs.OKABE_ITO[1], "-", "o"), (32, 300): (fs.OKABE_ITO[1], "--", "s")}
label = {(16, 60): "16 veh./240 req., 60-s periods", (16, 300): "16 veh./240 req., 300-s periods",
         (32, 60): "32 veh./240 req., 60-s periods", (32, 300): "32 veh./240 req., 300-s periods"}

fig, axes = plt.subplots(1, 3, figsize=(fs.DOUBLE, 0.36 * fs.DOUBLE))
for (f240, ep), g in per.groupby(["fleet_per_240", "epoch"]):
    c, ls, mk = styles[(f240, ep)]
    agg = g.groupby("density").agg(med=("largest_component", "median"),
                                   p95=("largest_component", lambda x: np.quantile(x, .95)),
                                   fit=("fits_device_rho_ge_1", "mean"),
                                   mip=("mip_seconds", "max")).reset_index()
    ks = agg.density.to_numpy()
    axes[0].plot(ks, agg.p95, color=c, ls=ls, marker=mk, ms=3, lw=1, label=label[(f240, ep)])
    axes[0].plot(ks, agg.med, color=c, ls=ls, marker=mk, ms=2.2, lw=0.6, alpha=0.55, mfc="white")
    axes[1].plot(ks, 100 * agg.fit, color=c, ls=ls, marker=mk, ms=3, lw=1)
    axes[2].plot(ks, agg.mip, color=c, ls=ls, marker=mk, ms=3, lw=1)
axes[0].axhline(CAP, color="0.35", lw=0.8, ls=":", label="SPQC-550 capacity (549 binaries)")
axes[0].set_yscale("symlog", linthresh=1); axes[0].set_ylim(0, 3e4)
axes[0].set_ylabel("Largest component after reduction\n(binaries; bold = 95th pct., light = median)")
axes[1].set_ylabel(r"Periods that fit SPQC-550 at $\rho\geq 1$ (%)"); axes[1].set_ylim(-3, 103)
axes[2].set_yscale("log"); axes[2].set_ylabel("Longest exact MILP solve of a period (s)")
for i, ax in enumerate(axes):
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 12.5]); ax.set_xticklabels(["1", "2", "4", "8", "12.5"])
    ax.set_xlabel("Demand density (× the public trace)")
    ax.text(-0.2, 1.04, "abc"[i], transform=ax.transAxes, fontweight="bold", fontsize=8)
h, l = axes[0].get_legend_handles_labels()
fig.tight_layout(w_pad=1.6, rect=(0, 0.1, 1, 1))
fig.legend(h, l, loc="lower center", ncol=5, fontsize=5.8, handlelength=2.4, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
fs.save(fig, FIG / ("figE2_density_preview" if "--partial" in sys.argv else "figE2_density_scaling"))
print(runs[["run", "share_solved_by_reduction", "median_core", "max_component", "share_fit_device", "max_mip_s"]].to_string())
