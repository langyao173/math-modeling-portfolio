"""Figure E2: (a) the fusion design with strong branches and the style prior, paired over 10 splits;
(b) the contest's key compound against what consumers write, within beer styles.

Reads results/extension/e2_split_scores.csv and e2_key_compound_consumers.csv (e2_strong_fusion.py).
Run:  python extension/e3_figure_e2.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results" / "extension", ROOT / "figures" / "extension"
FIG.mkdir(parents=True, exist_ok=True); fs.apply()
d = pd.read_csv(RES / "e2_split_scores.csv"); kc = pd.read_csv(RES / "e2_key_compound_consumers.csv")
ORDER = [("style prior", "style prior only", "0.55"), ("KRR, 16-d low rank", "KRR, 16-d low rank", fs.OKABE_ITO[5]),
         ("ET, 16-d low rank", "ET, 16-d low rank", fs.OKABE_ITO[0]), ("ET, 50 outputs", "ET, 50 outputs", fs.OKABE_ITO[2]),
         ("fusion of chemistry branches", "fusion, chemistry only", fs.OKABE_ITO[4]), ("fusion incl. style prior", "fusion incl. style prior", fs.OKABE_ITO[1])]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.DOUBLE, 0.36 * fs.DOUBLE), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.55))
rep = d[d.split_seed != 0].pivot(index="split_seed", columns="model", values="macro_r2")
for i, (m, lab, c) in enumerate(ORDER):
    v = rep[m]
    ax.scatter(v.values, np.full(len(v), i), s=8, color=c, alpha=0.6, linewidths=0)
    ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=c, lw=1.6)
    ref = d[(d.split_seed == 0) & (d.model == m)].macro_r2.iloc[0]
    ax.scatter(ref, i, s=24, facecolor="none", edgecolor=c, linewidths=0.9, zorder=3)
ax.axvline(0.21, color="0.3", lw=0.7, ls=(0, (3, 2))); ax.text(0.212, -0.55, "published GBR 0.21", fontsize=5.6, color="0.3", va="center")
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([o[1] for o in ORDER]); ax.set_ylim(-0.8, len(ORDER) - 0.4)
ax.set_xlabel("macro $R^2$, 50 trained-panel attributes"); ax.set_title("(a) 10 style-stratified splits (bar = mean)", loc="left")
TG = ["panel: 4VG note", "panel: clove", "consumers: clove aroma", "consumers: clove taste"]
CMP = [("4-vinylguaiacol", fs.OKABE_ITO[1], "o"), ("eugenol", fs.OKABE_ITO[0], "s"), ("4-ethylguaiacol", fs.OKABE_ITO[2], "^"), ("4-ethylphenol", "0.45", "D")]
for k, (cmp_, c, mk) in enumerate(CMP):
    for i, t in enumerate(TG):
        r = kc[(kc.target == t) & (kc.compound == cmp_)].iloc[0]
        y = i + (k - 1.5) * 0.17
        bx.plot([r.within_lo95, r.within_hi95], [y, y], color=c, lw=0.9)
        bx.plot(r.spearman_within_style, y, mk, color=c, ms=3.4, label=cmp_ if i == 0 else None)
bx.axvline(0, color="k", lw=0.6)
bx.set_yticks(range(len(TG))); bx.set_yticklabels(TG); bx.invert_yaxis()
bx.set_xlabel("within-style rank correlation with the compound (95 % bootstrap CI)")
bx.set_title("(b) phenolic compounds vs panel and consumers", loc="left")
bx.set_xlim(-0.3, 0.78)
bx.legend(loc="lower right", fontsize=5.6, handlelength=1.0)
fs.save(fig, FIG / "figE2_strong_fusion_key_compound")
