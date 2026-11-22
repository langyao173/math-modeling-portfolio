"""Figure E2: improved forecasters on the 232 prospective matches (2018, 2022, 2026).

(a), (b) log loss and RPS of each model relative to the submitted classifier, with 95 % paired bootstrap intervals;
(c) reliability of the outcome probabilities (all three outcomes pooled, deciles).
Run after e2_improved_forecaster.py:  python extension/e3_figure_e2.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results" / "extension", ROOT / "figures" / "extension"
FIG.mkdir(parents=True, exist_ok=True); fs.apply()
S = json.loads((RES / "e2_summary.json").read_text()); B = S["paired_bootstrap"]
cal = pd.read_csv(RES / "e2_calibration.csv")
M = [("contest features, refit", "contest features, refit\n(no class re-weighting)", fs.OKABE_ITO[2]),
     ("Elo (ordered logit)", "Elo benchmark", fs.OKABE_ITO[1]),
     ("equal-weight pool", "equal-weight pool", "0.5"),
     ("learned pool", "learned pool (w fitted 2002–2014)", fs.OKABE_ITO[5]),
     ("combined logit (contest features + Elo)", "combined logit\n(contest features + Elo)", fs.OKABE_ITO[0])]
fig, axes = plt.subplots(1, 3, figsize=(fs.DOUBLE, 0.33 * fs.DOUBLE), gridspec_kw=dict(width_ratios=[1, 1, 1.05], wspace=0.25))
for ax, k, title in zip(axes[:2], ("log_loss", "rps"), ("(a) log loss vs submitted", "(b) RPS vs submitted")):
    for i, (m, lab, c) in enumerate(M):
        mean, lo, hi = B[f"{m} minus contest logit (submitted) ({k})"]
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o", ms=3.5, color=c, lw=1.0, capsize=0)
    ax.axvline(0, color="k", lw=0.6)
    lo_, hi_ = ax.get_xlim(); ax.set_xlim(lo_, max(hi_, 0.08 * (hi_ - lo_)))   # keep the zero line inside the frame
    ax.set_yticks(range(len(M))); ax.set_yticklabels([m[1] for m in M] if ax is axes[0] else [], fontsize=6)
    ax.invert_yaxis(); ax.set_title(title, loc="left"); ax.set_xlabel("difference (lower is better)")
ax = axes[2]
for m, lab, c in [("contest logit (submitted)", "submitted", fs.OKABE_ITO[3]), ("contest features, refit", "refit", fs.OKABE_ITO[2]),
                  ("combined logit (contest features + Elo)", "combined", fs.OKABE_ITO[0])]:
    g = cal[(cal.model == m) & (cal.n > 0)]
    ece = float(cal[(cal.model == m) & (cal.n == -1)].ece.iloc[0])
    ax.plot(g.p_mean, g.obs_freq, "o-", ms=3, lw=0.9, color=c, label=f"{lab} (ECE {ece:.3f})")
ax.plot([0, 0.8], [0, 0.8], color="0.6", lw=0.6, ls="--")
ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8); ax.set_aspect("equal"); ax.set_anchor("N")   # top-aligned with (a), (b)
ax.set_xlabel("forecast probability"); ax.set_ylabel("observed frequency"); ax.set_title("(c) reliability, 232 matches", loc="left")
ax.legend(loc="lower right", fontsize=5.8, handlelength=1.2)
fs.save(fig, FIG / "figE2_improved_forecaster")
