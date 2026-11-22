"""Figure E2: what limits the annealing route (reads results/extension/e3_aligned_energy_*.{json,csv}).

(a) Gap of the decoded schedules to the certified optimum, by annealing energy (best and median of 8 seeds),
    without and with the submitted schedule's peak as a cap; the submitted schedule for reference.
(b) Exact objective along the best-first single-job path from the submitted schedule to the certified optimum
    (no cap): the climb above the running minimum is the barrier a descent method has to cross.
Run:  python extension/e4_figure_aligned_energy.py
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
S = json.loads((RES / "e3_aligned_energy_summary.json").read_text())
runs = pd.read_csv(RES / "e3_aligned_energy_runs.csv")
final = runs[runs["round"] == runs["round"].max()]
ref = S["references"]
opt = {"no peak cap": ref["certified optimum"]["F"], "peak capped at submitted": ref["certified optimum, peak capped"]["F"]}
EN = [("action cost", "action cost\n(base QUBO)"), ("linear recourse", "linear recourse\nsurrogate"),
      ("quadratic recourse", "quadratic recourse\nsurrogate"), ("exact objective (not QUBO)", "exact objective\n(not a QUBO)")]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.DOUBLE, 0.34 * fs.DOUBLE), gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.32))
for (cap, col, off) in (("no peak cap", fs.OKABE_ITO[0], -0.16), ("peak capped at submitted", fs.OKABE_ITO[1], 0.16)):
    for i, (en, _) in enumerate(EN):
        g = 100 * (final[(final.cap == cap) & (final.energy == en)].F - opt[cap]) / opt[cap]
        ax.plot(np.full(len(g), i + off), g, "o", color=col, ms=2.4, alpha=0.45, mew=0)
        ax.plot([i + off - 0.1, i + off + 0.1], [g.median()] * 2, color=col, lw=1.6)
    sub = 100 * (ref["submitted"]["F"] - opt[cap]) / opt[cap]
    ax.axhline(sub, color=col, lw=0.7, ls="--")
    ax.text(0.55, sub * (1.04 if cap == "no peak cap" else 0.96), f"submitted {sub:.1f} %", color=col, fontsize=5.6,
            va="bottom" if cap == "no peak cap" else "top", ha="left")
ax.plot([], [], "o", color=fs.OKABE_ITO[0], ms=3, label="no peak cap"); ax.plot([], [], "o", color=fs.OKABE_ITO[1], ms=3, label="peak capped at submitted")
ax.set_yscale("symlog", linthresh=1, linscale=0.6); ax.set_ylim(0, 40)
ax.set_yticks([0, 0.5, 1, 2, 5, 10, 20]); ax.set_yticklabels(["0", "0.5", "1", "2", "5", "10", "20"])
ax.set_xticks(range(len(EN))); ax.set_xticklabels([e[1] for e in EN], fontsize=5.8)
ax.set_ylabel("gap to certified optimum (%)"); ax.set_title("(a) decoded schedules, 8 seeds (bar = median)", loc="left")
ax.legend(loc="upper right", fontsize=5.8, handlelength=0.8)
b = S["barrier"]["certified optimum"]; path = np.array(b["path_F"]); run_min = np.minimum.accumulate(path)
k = np.arange(len(path))
bx.plot(k, path, color=fs.OKABE_ITO[0], lw=1.0, label="exact objective along the path")
bx.plot(k, run_min, color="0.55", lw=0.7, ls="--", label="running minimum")
k0, k1 = b["climb_start_step"], b["barrier_at_step"]
bx.annotate("", xy=(k1, path[k1]), xytext=(k1, run_min[k1]), arrowprops=dict(arrowstyle="<->", lw=0.7, color=fs.OKABE_ITO[1]))
bx.annotate(f"barrier {b['barrier_usd']:.2f} USD", xy=(k1, (path[k1] + run_min[k1]) / 2), xytext=(k1 - 70, run_min[k1] + 0.52),
            fontsize=6, color=fs.OKABE_ITO[1], ha="center", va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=fs.OKABE_ITO[1]))
sl = b["slots_kw"]["slot6"]
bx.text(k0, run_min[k0] - 0.12, f"slot 6 at {sl['climb_start']:.1f} kW\n(reference {sl['reference']:.1f})", fontsize=5.6, ha="center", va="top", color="0.3")
bx.text(k1, path[k1] + 0.1, f"slot 6 at {sl['barrier_top']:.1f} kW", fontsize=5.6, ha="center", va="bottom", color="0.3")
bx.axhline(ref["certified optimum"]["F"], color="k", lw=0.5, ls=":"); bx.text(2, ref["certified optimum"]["F"] + 0.03, "certified optimum", fontsize=5.6, va="bottom")
bx.set_xlabel("jobs switched from the submitted to the optimal action (best first)"); bx.set_ylabel("expected + CVaR$_{95}$ recourse (USD)")
bx.set_title("(b) the optimum lies behind a barrier for single-job moves", loc="left")
bx.legend(loc="upper right", fontsize=5.8, handlelength=1.6); bx.set_xlim(0, len(path) + 2)
lo, hi = bx.get_ylim(); bx.set_ylim(lo - 0.25, hi + 0.1)
fs.save(fig, FIG / "figE2_aligned_energy")
