"""Figure: the three admissible guard thresholds compared on the operating criteria (5 584 updates).

Reads results/extension/e1_optimal_guard_summary.json written by e1_optimal_guard.py.
Run:  python extension/e2_figure.py
"""
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "extension"; FIG.mkdir(parents=True, exist_ok=True); fs.apply()
S = json.loads((ROOT / "results" / "extension" / "e1_optimal_guard_summary.json").read_text())
RULES = [("fixed_threshold (submitted)", "midpoint (submitted)", fs.OKABE_ITO[0]),
         ("L2_optimal_threshold", "exact L2 minimum", fs.OKABE_ITO[1]),
         ("promise_optimal_threshold (extension)", "fewest broken promises", fs.OKABE_ITO[2])]
PANELS = [("rows_changed_by_guard", "scores touched", "{:.0f}"),
          ("gated_rows_with_excess_over_radius", "broken ±1 pp promises", "{:.0f}"),
          ("total_excess_over_radius_pp", "total excess over ±1 pp (pp)", "{:.1f}"),
          ("l2_change_from_stability_candidate", "L2 change from stable scores", "{:.3f}")]

fig, axes = plt.subplots(1, 4, figsize=(fs.DOUBLE, 0.24 * fs.DOUBLE), gridspec_kw=dict(wspace=0.12))
for ax, (key, title, fmt) in zip(axes, PANELS):
    vals = [S[r][key] for r, _, _ in RULES]
    best = min(vals)
    for i, ((r, lab, c), v) in enumerate(zip(RULES, vals)):
        ax.barh(i, v, color=c, height=0.62, alpha=1.0 if v == best else 0.55)
        ax.text(v, i, " " + fmt.format(v), va="center", ha="left", fontsize=6, fontweight="bold" if v == best else "normal")
    ax.set_xlim(0, max(vals) * 1.32); ax.set_title(title, loc="left", fontsize=6.5)
    ax.set_yticks(range(len(RULES))); ax.invert_yaxis(); ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_yticklabels([lab for _, lab, _ in RULES] if ax is axes[0] else [])
fig.text(0.0, -0.04, "Lower is better for every criterion; the best rule per criterion is shown in full colour.", fontsize=6, color="0.35")
fs.save(fig, FIG / "figE1_guard_threshold_tradeoff")
