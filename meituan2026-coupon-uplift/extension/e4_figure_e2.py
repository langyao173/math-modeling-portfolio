"""Figure for E2: Qini coefficient of every learner on three randomised campaigns, and CATE calibration.

Reads results/extension/e2_<campaign>_scores.csv / _calibration.csv (written by e2_uplift_benchmark.py).
Error bars: mean over repeated 50/50 splits with the Nadeau-Bengio corrected 95 % interval.
"""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs
from e3_summarise_e2 import corrected_ci

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results" / "extension", ROOT / "figures" / "extension"
FIG.mkdir(parents=True, exist_ok=True); fs.apply()
CAMPAIGNS = [("hillstrom", "Hillstrom e-mail (64 000)"), ("x5", "X5 Retail SMS (200 039)"), ("lenta", "Lenta SMS (687 029)")]
CAMPAIGNS = [c for c in CAMPAIGNS if (RES / f"e2_{c[0]}_scores.csv").exists()]
ORDER = ["response", "S-learner", "T-learner", "transformed outcome", "X-learner", "DR-learner", "causal forest", "random"]
NAMES = {"response": "Response model (\"who will buy\")", "S-learner": "S-learner",
         "T-learner": "T-learner = proposal's ranking", "transformed outcome": "Transformed outcome",
         "X-learner": "X-learner", "DR-learner": "DR-learner", "causal forest": "Causal forest", "random": "Random"}
COL = {"response": fs.OKABE_ITO[1], "T-learner": fs.OKABE_ITO[0], "random": "0.65"}

fig, axes = plt.subplots(1, len(CAMPAIGNS) + 1, figsize=(fs.DOUBLE, 0.42 * fs.DOUBLE),
                         gridspec_kw={"width_ratios": [1] * len(CAMPAIGNS) + [1.2]})
y = np.arange(len(ORDER))[::-1]
for ax, (name, title) in zip(axes, CAMPAIGNS):
    s = pd.read_csv(RES / f"e2_{name}_scores.csv")
    q = s.pivot(index="rep", columns="learner", values="qini")
    for yi, lr in zip(y, ORDER):
        m, lo, hi = corrected_ci(q[lr].to_numpy())
        c = COL.get(lr, "0.2")
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.1)
        ax.plot(m, yi, "o", color=c, ms=3.6, mfc=c if lr in COL else "white", mew=0.9)
    ax.axvline(0, color="0.6", lw=0.5, ls=":")
    ax.set_title(title, fontsize=6.6)
    ax.set_yticks(y); ax.set_yticklabels([NAMES[l] for l in ORDER] if ax is axes[0] else [])
    ax.set_xlabel("Qini coefficient (held-out half)")
    ax.set_ylim(-0.6, len(ORDER) - 0.4)
# calibration of the CATE scale
axc = axes[-1]
CAL = [("T-learner", "T-learner, raw", fs.OKABE_ITO[0], "white"),
       ("T-learner + isotonic (proposal)", "T-learner + isotonic (proposal)", fs.OKABE_ITO[0], fs.OKABE_ITO[0]),
       ("X-learner", "X-learner", "0.2", "white"), ("DR-learner", "DR-learner", "0.2", "0.2"),
       ("causal forest", "Causal forest", "0.55", "0.55")]
for j, (name, title) in enumerate(CAMPAIGNS):
    c = pd.read_csv(RES / f"e2_{name}_calibration.csv").pivot(index="rep", columns="learner", values="calibration_error")
    for k, (lr, lab, col, face) in enumerate(CAL):
        xx = j + (k - 2) * 0.14
        m, lo, hi = corrected_ci(100 * c[lr].to_numpy())
        axc.plot([xx, xx], [max(lo, 0), hi], color=col, lw=0.9)
        axc.plot(xx, m, "o", color=col, mfc=face, ms=3.4, mew=0.9, label=lab if j == 0 else None)
axc.set_xticks(range(len(CAMPAIGNS))); axc.set_xticklabels([t.split(" ")[0] for _, t in CAMPAIGNS])
axc.set_ylabel("CATE calibration error (pp, decile bins)")
axc.set_ylim(0, 1.75 * axc.get_ylim()[1])
axc.legend(loc="upper right", ncol=1, fontsize=5.3, handletextpad=0.3, borderaxespad=0.2)
for i, ax in enumerate(axes):
    ax.text(-0.08 if i else -0.62, 1.05, "abcd"[i], transform=ax.transAxes, fontweight="bold", fontsize=8)
fig.tight_layout(w_pad=1.0)
fs.save(fig, FIG / "figE2_uplift_benchmark")
print("ok", [c[0] for c in CAMPAIGNS])
