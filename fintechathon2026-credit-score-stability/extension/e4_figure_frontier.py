"""Figure E2: stability-discrimination frontier of the stabilisers (reads results/extension/e3_stability_frontier.csv).

x: worst tail instability over the three unchanged-risk cohorts and two months (CVaR95 of the absolute score
change, pp; lower is more stable); y: AUC on the labelled states of both months.  Lines connect the radius
(clipping) or weight (smoothing) sweeps; filled markers use the reference-scale base model of the submission,
open markers the strong risk ensemble.
Run:  python extension/e4_figure_frontier.py
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "extension"; FIG.mkdir(parents=True, exist_ok=True); fs.apply()
d = pd.read_csv(ROOT / "results" / "extension" / "e3_stability_frontier.csv")
FAM = [("gated clip", "clip gated updates (radius 0.25–4 pp)", fs.OKABE_ITO[0]),
       ("clip all", "clip every update (radius 0.25–4 pp)", fs.OKABE_ITO[1]),
       ("smooth all", "smooth every update (weight 0.25–0.9)", fs.OKABE_ITO[2])]
fig, ax = plt.subplots(figsize=(fs.SINGLE * 1.25, 0.82 * fs.SINGLE))
for fam, lab, c in FAM:
    for base, mk, ls in (("base_reference", "o", "-"), ("strong_S", "o", "--")):
        g = d[(d.family == fam) & (d.base_model == base)]
        g = g.sort_values("radius" if fam != "smooth all" else "beta", ascending=fam == "smooth all")
        ax.plot(g.worst_cohort_cvar95_pp, g.auc_phase0_1, ls, color=c, lw=0.9, marker=mk, ms=3.2,
                mfc=c if base == "base_reference" else "white", mew=0.8, label=lab if base == "base_reference" else None)
for base, mfc in (("base_reference", "0.25"), ("strong_S", "white")):
    r = d[(d.family == "none") & (d.base_model == base)].iloc[0]
    ax.plot(r.worst_cohort_cvar95_pp, r.auc_phase0_1, "s", color="0.25", mfc=mfc, ms=4.2, mew=0.8)
    ax.annotate("no stabiliser" + ("\n(strong ensemble)" if base == "strong_S" else "\n(reference scale)"), (r.worst_cohort_cvar95_pp, r.auc_phase0_1),
                xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=5.6, color="0.25")
sub = d[(d.base_model == "base_reference") & (d.variant == "gated clip r=0.01")].iloc[0]
ax.plot(sub.worst_cohort_cvar95_pp, sub.auc_phase0_1, "*", color="k", ms=9, zorder=5)
ax.annotate("submitted (gate, ±1 pp)", (sub.worst_cohort_cvar95_pp, sub.auc_phase0_1), xytext=(-18, 26), textcoords="offset points",
            fontsize=6, ha="center", arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3"))
alt = d[(d.base_model == "base_reference") & (d.variant == "clip all r=0.02")].iloc[0]
ax.annotate("±2 pp", (alt.worst_cohort_cvar95_pp, alt.auc_phase0_1), xytext=(3, -9), textcoords="offset points",
            fontsize=6, ha="left", va="center", color=fs.OKABE_ITO[1])
ax.plot([], [], "o", color="0.3", ms=3.2, label="reference-scale base (filled)")
ax.plot([], [], "o--", color="0.3", mfc="white", ms=3.2, lw=0.9, label="strong ensemble base (open)")
ax.set_xlabel("worst-cohort tail instability, CVaR$_{95}$ of |score change| (pp)")
ax.set_ylabel("AUC, both months")
ax.set_xlim(1.5, 11.8); ax.set_ylim(0.776, 0.8015)
ax.legend(loc="lower right", fontsize=5.6, handlelength=1.6)
fs.save(fig, FIG / "figE2_stability_frontier")
