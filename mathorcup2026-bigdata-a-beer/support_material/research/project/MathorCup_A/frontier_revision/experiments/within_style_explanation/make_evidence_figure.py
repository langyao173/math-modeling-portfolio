"""Source-backed, editable SVG of the within-style sensitivity experiment."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mathorcup_mpl_cache")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.edgecolor": "#CBD4DD",
    "axes.labelcolor": "#273B51", "text.color": "#13243A",
    "xtick.color": "#5A6C7D", "ytick.color": "#273B51",
    "axes.titleweight": "bold", "savefig.facecolor": "white",
})
INK, TOTAL, WITHIN, MUTED = "#13243A", "#E58A56", "#138C83", "#ABB7C4"
f = pd.read_csv(HERE / "full_data_pair_evidence.csv").set_index(["target", "field"])
b = pd.read_csv(HERE / "fixed_alpha_pair_evidence.csv").set_index(["target", "field"])
p = pd.read_csv(HERE / "oof_paired_differences.csv")
p = p[(p.model == "within_linear") & (p.baseline == "style_mean_raw")].set_index("target")
keys = [
    ("lactic", "lactic_acid.mg.L.", "Lactic acid → Lactic"),
    ("acidity", "lactic_acid.mg.L.", "Lactic acid → Acidity"),
    ("X4vg", "X4VG", "4-VG → Phenolic"),
    ("alcohol", "ethanol..v.v.", "Ethanol → Alcohol"),
    ("A_malt_all", "X2.acetylfuran", "2-Acetylfuran → Malt aroma"),
    ("F_malt_all", "X2.6.dimethyl.pyrazine", "2,6-DMP → Malt flavour"),
    ("dms", "DMS", "DMS → DMS"),
    ("diacetyl", "X2.3.butanedione", "Diacetyl → Diacetyl"),
]
fig = plt.figure(figsize=(15.6, 7.3))
gs = fig.add_gridspec(1, 3, left=.205, right=.975, top=.70, bottom=.18,
                      width_ratios=[1.04, .98, 1.12], wspace=.38)
ax, bx, cx = [fig.add_subplot(gs[0, i]) for i in range(3)]
fig.text(.038, .932, "Which chemical signals survive within a beer style?",
         fontsize=22, fontweight="bold", color=INK)
fig.text(.039, .884, "250 beers  /  22 styles  /  50 sensory dimensions",
         fontsize=11.5, color="#5A6C7D")
fig.text(.039, .826, "Stable anchors", fontsize=11, color=WITHIN, fontweight="bold")
fig.text(.138, .826, "4-VG and ethanol retain within-style evidence", fontsize=11)
fig.text(.514, .826, "Re-ranked links", fontsize=11, color=TOTAL, fontweight="bold")
fig.text(.62, .826, "Lactic-acid Top-3 status falls after style centering", fontsize=11)
for panel, letter, title, subtitle in [
    (ax, "A", "Association", "Normal-score correlation"),
    (bx, "B", "Rank stability", "Top-3 frequency · original sensory scale"),
    (cx, "C", "Predictive increment", "ΔR² over a style-only predictor"),
]:
    panel.text(0, 1.16, letter, transform=panel.transAxes, fontsize=12,
               color="white", fontweight="bold",
               bbox=dict(boxstyle="round,pad=.32", fc=INK, ec="none"))
    panel.text(.12, 1.16, title, transform=panel.transAxes,
               fontsize=13.4, fontweight="bold", va="center")
    panel.text(0, 1.065, subtitle, transform=panel.transAxes,
               fontsize=9.3, color="#65778A")
    panel.tick_params(axis="y", length=0)
    panel.grid(axis="x", color="#E7ECF1", lw=.7, zorder=0)
    panel.set_axisbelow(True)
for y, (target, field, label) in enumerate(keys):
    row, boot = f.loc[(target, field)], b.loc[(target, field)]
    left, right = row.correlation_total, row.correlation_within
    ax.plot([left, right], [y, y], color=MUTED, lw=2, zorder=1)
    ax.scatter(left, y, s=60, c=TOTAL, edgecolors="white", lw=1, zorder=3)
    ax.scatter(right, y, s=60, c=WITHIN, edgecolors="white", lw=1, zorder=3)
    l, r = boot.total_linear_fixed_top3_frequency, boot.within_linear_fixed_top3_frequency
    bx.plot([l, r], [y, y], color=MUTED, lw=2, zorder=1)
    # Small vertical offset preserves coincident endpoints (e.g. the 4-VG anchor).
    bx.scatter(l, y-.06, s=60, c=TOTAL, edgecolors="white", lw=1, zorder=3)
    bx.scatter(r, y+.06, s=60, c=WITHIN, edgecolors="white", lw=1, zorder=3)
ax.set_yticks(range(len(keys)), [k[2] for k in keys])
bx.set_yticks(range(len(keys)), [""]*len(keys))
for panel in [ax, bx]:
    panel.set_ylim(len(keys)-.35, -.65)
ax.set_xlim(-.20, .95); ax.set_xticks([0, .25, .5, .75])
ax.axvline(0, color="#B7C3CF", lw=.8)
bx.set_xlim(-.07, 1.08); bx.set_xticks([0, .5, 1], ["0%", "50%", "100%"])
bx.axvline(.5, color="#D0D8E0", lw=.8, ls=(0, (3, 3)))
ax.scatter([], [], s=55, c=TOTAL, label="Total")
ax.scatter([], [], s=55, c=WITHIN, label="Within style")
ax.legend(loc="upper left", bbox_to_anchor=(-.03, -.12), frameon=False,
          ncol=2, handletextpad=.5, columnspacing=1.4)
target_keys = [("__macro__", "All 50"), ("acidity", "Acidity"), ("alcohol", "Alcohol"),
               ("lactic", "Lactic"), ("A_malt_all", "Malt aroma"),
               ("F_malt_all", "Malt flavour"), ("X4vg", "Phenolic"),
               ("diacetyl", "Diacetyl"), ("dms", "DMS"), ("t2n", "t2n")]
for y, (target, label) in enumerate(target_keys):
    row = p.loc[target]
    point, low, high = row.delta_R2, row.paired_bootstrap_025, row.paired_bootstrap_975
    c = INK if target == "__macro__" else WITHIN if low > 0 else "#91A3B4"
    cx.plot([low, high], [y, y], color=c, lw=2.1 if target == "__macro__" else 1.8)
    cx.scatter(point, y, s=65 if target == "__macro__" else 38, color=c,
               marker="D" if target == "__macro__" else "o", zorder=3)
cx.set_yticks(range(len(target_keys)), [t[1] for t in target_keys], fontsize=9.5)
cx.set_ylim(len(target_keys)-.35, -.65); cx.set_xlim(-.055, .125)
cx.set_xticks([0, .05, .10], ["0", "+.05", "+.10"])
cx.axvline(0, color="#B7C3CF", lw=1)
cx.text(.99, -.14, "+0.0418  [0.0322, 0.0525]", transform=cx.transAxes,
        ha="right", color=WITHIN, fontweight="bold", fontsize=10.4)
fig.text(.039, .065,
         "A: pooled correlations before/after training-style centering.  B: matched α = 3; 400 paired, style-stratified bootstrap fits.",
         fontsize=9.1, color="#607185")
fig.text(.039, .035,
         "C: five-fold out-of-fold predictions; 2,000 paired loss bootstraps conditional on the fitted folds. Associations are descriptive, not intervention effects.",
         fontsize=9.1, color="#607185")
fig.savefig(HERE / "within_style_evidence.svg")
fig.savefig(HERE / "within_style_evidence.pdf")
fig.savefig(HERE / "within_style_evidence.png", dpi=180)
print(HERE / "within_style_evidence.svg")
