"""Figure: share of the total incremental effect captured by treating the top 30 %, Hillstrom RCT.

Reads results/extension/e1_hillstrom_policy_scores.csv (e1_hillstrom_rct.py): 20 random train/test
halves.  Uplift with IPTW isotonic calibration ranks customers exactly as the T-learner (isotonic maps are
monotone), so it coincides with the uplift column.  Grey lines pair the two learned policies within a split.
Run:  python extension/e2_figure.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "extension"; FIG.mkdir(parents=True, exist_ok=True); fs.apply()
d = pd.read_csv(ROOT / "results" / "extension" / "e1_hillstrom_policy_scores.csv")
POL = [("random", "random", "0.55"), ("response model (who will buy)", "response model", fs.OKABE_ITO[0]),
       ("T-learner uplift (who buys because of it)", "uplift model", fs.OKABE_ITO[1])]
rng = np.random.default_rng(1)
fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE * 0.72, 0.30 * fs.DOUBLE), gridspec_kw=dict(wspace=0.3))
for ax, (out, title) in zip(axes, (("visit", "(a) site visit"), ("conversion", "(b) purchase"))):
    w = d[d.outcome == out].pivot(index="rep", columns="policy", values="share_of_total_increment_at_30pct") * 100
    jit = rng.uniform(-0.08, 0.08, len(w))
    for r in range(len(w)):
        ax.plot([1 + jit[r], 2 + jit[r]], [w.iloc[r][POL[1][0]], w.iloc[r][POL[2][0]]], color="0.8", lw=0.5, zorder=1)
    for i, (p, lab, c) in enumerate(POL):
        ax.scatter(i + jit, w[p], s=9, color=c, alpha=0.8, linewidths=0, zorder=2)
        ax.plot([i - 0.22, i + 0.22], [w[p].mean()] * 2, color=c, lw=1.6, zorder=3)
        ax.text(i + 0.25, w[p].mean(), f"{w[p].mean():.1f}", fontsize=6, va="center", color=c,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=4)
    ax.axhline(30, color="0.6", lw=0.5, ls=":")
    ax.set_xticks(range(3)); ax.set_xticklabels([p[1] for p in POL], fontsize=6)
    better = int((w[POL[2][0]] > w[POL[1][0]] + 1e-12).sum())
    ax.set_title(f"{title}: uplift ahead in {better} of {len(w)} splits", loc="left", fontsize=6.5)
    ax.set_xlim(-0.5, 2.6)
axes[0].set_ylabel("share of total increment captured\nby treating the top 30 % (%)")
fs.save(fig, FIG / "figE1_hillstrom_targeting")
