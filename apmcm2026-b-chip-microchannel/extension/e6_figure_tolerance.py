"""Figure E3: tolerance spread at the recommended design against what the surrogate leaves unknown.

Reads results/extension/e5_tolerance_posterior_summary.json (e5_tolerance_posterior.py).  For each response:
the coefficient of variation caused by the manufacturing tolerances (paper's GPR plug-in; median and 95 %
interval over 1 000 UK posterior draws) and the posterior standard deviation of the nominal performance
itself (surrogate uncertainty), both in % of the nominal value.
Run:  python extension/e6_figure_tolerance.py
"""
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "extension"; FIG.mkdir(parents=True, exist_ok=True); fs.apply()
S = json.loads((ROOT / "results" / "extension" / "e5_tolerance_posterior_summary.json").read_text())
LAB = {"R": "thermal resistance $R^*$", "dP": "pressure drop $\\Delta P^*$", "phi": "non-uniformity $\\phi$"}
fig, ax = plt.subplots(figsize=(fs.SINGLE * 1.2, 0.62 * fs.SINGLE))
for i, t in enumerate(["R", "dP", "phi"]):
    v = S["targets"][t]; y = 2 - i
    lo, med, hi = v["posterior_2.5_50_97.5"]["cv_pct"]
    ax.errorbar(med, y + 0.12, xerr=[[med - lo], [hi - med]], fmt="o", color=fs.OKABE_ITO[0], ms=4, elinewidth=1.2, capsize=2,
                label="tolerance CV, UK posterior median (95 % band)" if i == 0 else None)
    ax.plot(S["paper_gpr_plugin"][t]["cv_pct"], y + 0.12, "x", color="k", ms=5, mew=1.1, label="tolerance CV, paper's GPR plug-in" if i == 0 else None)
    sd = v["nominal_posterior_sd_pct"]
    if sd < 1e-3:                                   # R*: the UK surface is the exact cubic, posterior s.d. ~1e-5 %
        ax.annotate("surrogate s.d. ≈ 0 (exact cubic)", (0.012, y - 0.14), fontsize=5.6, va="center", color=fs.OKABE_ITO[1])
    else:
        ax.plot(sd, y - 0.14, "s", color=fs.OKABE_ITO[1], ms=4, label="surrogate s.d. of the nominal value" if i == 1 else None)
ax.set_xscale("log"); ax.set_xlim(0.01, 3)
ax.set_xticks([0.01, 0.1, 1]); ax.set_xticklabels(["0.01", "0.1", "1"])
ax.set_yticks([2, 1, 0]); ax.set_yticklabels([LAB[t] for t in ["R", "dP", "phi"]])
ax.set_ylim(-0.6, 3.1); ax.set_xlabel("% of the nominal value at (0.22, 4.5, 4)")
ax.legend(loc="upper left", fontsize=5.6, handlelength=1.2)
fs.save(fig, FIG / "figE3_tolerance_vs_surrogate")
