"""Figure: macro R^2 of the fusion design and its comparators under the reference study's protocol.

Reads results/extension/e1_split_scores.csv (e1_published_protocol.py): split 0 is the reference
split (random_state = 0), splits 1-10 are further style-stratified 70/30 splits.  Ridge is omitted
(macro R^2 between -1.5 and -140 because of near-constant attributes; see README).
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
d = pd.read_csv(ROOT / "results" / "extension" / "e1_split_scores.csv")
ORDER = [("GBR (reference study's best model)", "GBR (re-run)", "0.45"),
         ("KRR-RBF, 50 independent outputs", "KRR, 50 outputs", fs.OKABE_ITO[5]),
         ("KRR-RBF, 16-d low-rank targets", "KRR, 16-d low rank", fs.OKABE_ITO[0]),
         ("Grouped convex fusion (contest design)", "fusion (contest design)", fs.OKABE_ITO[1]),
         ("Extra Trees", "Extra Trees", fs.OKABE_ITO[2])]
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(fs.SINGLE * 1.15, 0.62 * fs.SINGLE))
for i, (m, lab, c) in enumerate(ORDER):
    v = d[d.model == m].set_index("split_seed").macro_r2
    rep = v.drop(0)
    ax.scatter(rep.values, i + rng.uniform(-0.12, 0.12, rep.size), s=9, color=c, alpha=0.7, linewidths=0)
    ax.scatter(v.loc[0], i, s=26, facecolor="none", edgecolor=c, linewidths=1.0, zorder=3)
    ax.plot([rep.mean()] * 2, [i - 0.28, i + 0.28], color=c, lw=1.5)
ax.axvline(0.21, color="0.3", lw=0.7, ls=(0, (3, 2)))
ax.text(0.2108, len(ORDER) - 0.45, "published GBR, 0.21", fontsize=6, color="0.3", ha="left", va="center")
ax.scatter([], [], s=9, color="0.4", label="10 further splits"); ax.scatter([], [], s=26, facecolor="none", edgecolor="0.4", label="reference split")
ax.plot([], [], color="0.4", lw=1.5, label="mean of 10 splits")
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([o[1] for o in ORDER]); ax.set_ylim(-0.6, len(ORDER) - 0.2)
ax.set_xlabel("macro $R^2$ over 50 sensory attributes (trained panel)")
ax.legend(loc="lower right", fontsize=5.8, handletextpad=0.3)
fs.save(fig, FIG / "figE1_published_protocol")
