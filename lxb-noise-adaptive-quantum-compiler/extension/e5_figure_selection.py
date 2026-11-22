"""Figure E2: candidate pools against the selected candidate, under exact expected success.

Reads results/extension/e4_selection_headroom.csv and e4_candidate_pool.csv.gz (e4_selection_headroom.py).
(a) For each exactly scored circuit, the exact expected success of every candidate in the pool (grey) and of the
    candidate the submitted rule selects (coloured), circuits ordered by the best candidate;
(b) per-circuit Spearman correlation between the reliability proxy and exact success across the pool.
Run:  python extension/e5_figure_selection.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results" / "extension", ROOT / "figures" / "extension"
FIG.mkdir(parents=True, exist_ok=True); fs.apply()
d = pd.read_csv(RES / "e4_selection_headroom.csv"); d = d[d.scored].sort_values("exact_all").reset_index(drop=True)
c = pd.read_csv(RES / "e4_candidate_pool.csv.gz").dropna(subset=["exact"])
COL = {"public": fs.OKABE_ITO[3], "S1_unseen_sizes": fs.OKABE_ITO[0], "S2_qasmbench": fs.OKABE_ITO[1]}
LAB = {"public": "public Track A", "S1_unseen_sizes": "S1 unseen sizes", "S2_qasmbench": "S2 QASMBench"}
fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.DOUBLE, 0.34 * fs.DOUBLE), gridspec_kw=dict(width_ratios=[2.2, 1], wspace=0.28))
rho = []
for i, r in d.iterrows():
    g = c[(c.suite == r.suite) & (c.benchmark == r.benchmark)]
    ax.plot(np.full(len(g), i), g.exact, "o", color="0.75", ms=1.6, mew=0, zorder=1)
    ax.plot(i, r.submitted, "o", color=COL[r.suite], ms=3.0, zorder=3, label=LAB[r.suite])
    if g.exact.nunique() > 1:
        rho.append(spearmanr(g.log_rel_proxy, g.exact)[0])
h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h))
ax.plot([], [], "o", color="0.75", ms=2.5, label="other candidates in the pool")
h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h))
ax.legend(u.values(), u.keys(), loc="lower right", fontsize=5.8, handlelength=0.8)
ax.set_xlabel("circuit (ordered by its best candidate)"); ax.set_ylabel("exact expected success")
ax.set_title("(a) the selected candidate sits at the top of a wide pool", loc="left"); ax.set_xticks([])
bx.hist(rho, bins=np.linspace(0.8, 1.0, 21), color=fs.OKABE_ITO[0], alpha=0.8)
bx.axvline(np.median(rho), color="k", lw=0.7, ls="--"); bx.text(np.median(rho) - 0.004, bx.get_ylim()[1] * 0.92, f"median {np.median(rho):.2f}", ha="right", fontsize=6)
bx.set_xlabel("Spearman ρ, reliability proxy vs exact success"); bx.set_ylabel("circuits")
from matplotlib.ticker import MaxNLocator
bx.yaxis.set_major_locator(MaxNLocator(integer=True))
bx.set_title("(b) proxy ranking within each pool", loc="left")
fs.save(fig, FIG / "figE2_selection_headroom")
