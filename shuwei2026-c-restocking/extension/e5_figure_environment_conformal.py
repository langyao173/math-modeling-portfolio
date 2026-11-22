"""Figure E2: coverage of the calibrated 90 % prediction ball in a held-out environment, and its size.

Reads results/extension/e4_environment_conformal.csv (e4_environment_conformal.py).  (a) Coverage over all horizons
up to 3 s in each of the eight non-training environments when it is held out; (b) mean radius of the ball.
Run:  python extension/e5_figure_environment_conformal.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "extension"; FIG.mkdir(parents=True, exist_ok=True); fs.apply()
d = pd.read_csv(ROOT / "results" / "extension" / "e4_environment_conformal.csv")
RULES = [("windows pooled", "static, windows pooled (submission's rule)", fs.OKABE_ITO[1], "o"),
         ("worst environment", "static, worst calibration environment", "0.45", "s"),
         ("windows pooled + online adaptation (gamma=0.01)", "windows pooled + online adaptation (γ = 0.01)", fs.OKABE_ITO[0], "D")]
envs = d[d.rule == "windows pooled"].sort_values("coverage_all_horizons").environment_id.tolist()
short = {e: e.replace("_x_", " / ").replace("courier_station", "courier").replace("supermarket", "market").replace("restaurant", "restaurant")
         for e in envs}
fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.DOUBLE, 0.36 * fs.DOUBLE), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.08), sharey=True)
y = np.arange(len(envs))
for k, (rule, lab, c, mk) in enumerate(RULES):
    g = d[d.rule == rule].set_index("environment_id").loc[envs]
    off = (k - 1) * 0.18
    ax.plot(g.coverage_all_horizons, y + off, mk, color=c, ms=3.6, label=lab, mfc=c if k != 1 else "white", mew=0.9)
    bx.plot(g.mean_radius_mm, y + off, mk, color=c, ms=3.6, mfc=c if k != 1 else "white", mew=0.9)
ax.axvline(0.90, color="k", lw=0.6, ls="--"); ax.text(0.902, len(envs) - 0.45, "nominal 0.90", fontsize=5.6, va="top")
ax.axvspan(0.6, 0.85, color="0.94", lw=0)
ax.set_yticks(y); ax.set_yticklabels([short[e] for e in envs], fontsize=6)
ax.set_xlim(0.72, 1.005); ax.set_xlabel("coverage when the environment is held out")
ax.set_title("(a) coverage in a new environment", loc="left")
ax.legend(loc="upper left", fontsize=5.5, handlelength=1.0, borderaxespad=0.3)
bx.set_xlabel("mean radius of the ball (mm)"); bx.set_title("(b) size of the ball", loc="left")
bx.set_xlim(100, 175)
fs.save(fig, FIG / "figE2_environment_conformal")
