"""Figure: submitted schedule against the certified optima of the full 588-job instance.

Reads results/extension/e1_full_exact.json written by e1_full_exact.py.
Run:  python extension/e2_figure.py
"""
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "extension"; FIG.mkdir(parents=True, exist_ok=True); fs.apply()
J = json.loads((ROOT / "results" / "extension" / "e1_full_exact.json").read_text())
obj = lambda m: m["expected_recourse_cost_usd"] + m["cvar95_recourse_cost_usd"]
rows = [("submitted\n(QUBO/Ising)", J["submitted_metrics"], fs.OKABE_ITO[0]),
        ("certified optimum,\nsame peak", J["peak_guarded_metrics"], fs.OKABE_ITO[2]),
        ("certified optimum,\nfree peak", J["metrics"], fs.OKABE_ITO[1])]

fig, (a, b) = plt.subplots(1, 2, figsize=(fs.DOUBLE * 0.78, 0.29 * fs.DOUBLE), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.38))
for i, (lab, m, c) in enumerate(rows):
    e, v = m["expected_recourse_cost_usd"], m["cvar95_recourse_cost_usd"]
    a.bar(i, e, color=c, width=0.6, alpha=0.55); a.bar(i, v, bottom=e, color=c, width=0.6)
    a.text(i, e + v + 1.2, f"{e + v:.2f}", ha="center", fontsize=6.5)
a.set_xticks(range(3)); a.set_xticklabels([r[0] for r in rows], fontsize=6)
a.set_ylabel("recourse cost (USD)"); a.set_ylim(0, 108)
a.legend(handles=[Patch(color="0.6", alpha=0.55, label="expected"), Patch(color="0.3", label="CVaR$_{95}$")],
         loc="upper center", ncol=2, fontsize=6)
a.set_title("(a) objective = expected + CVaR$_{95}$", loc="left")

for lab, m, c in rows:
    x, y = 1000 * m["peak_it_mw"], obj(m)
    b.scatter(x, y, s=30, color=c, zorder=3)
    b.annotate(lab.replace("\n", " "), (x, y), textcoords="offset points", xytext=(6, 3), fontsize=5.8, color=c)
xs, s_obj = 1000 * J["submitted_metrics"]["peak_it_mw"], obj(J["submitted_metrics"])
g, u, xu = obj(J["peak_guarded_metrics"]), obj(J["metrics"]), 1000 * J["metrics"]["peak_it_mw"]
b.annotate("", (xs, g), (xs, s_obj), arrowprops=dict(arrowstyle="->", lw=0.6, color="0.3"))
b.text(xs - 0.4, (g + s_obj) / 2, f"−{100 * (1 - g / s_obj):.1f} %", ha="right", va="center", fontsize=6, color="0.3")
b.annotate("", (xu, u), (xs, g), arrowprops=dict(arrowstyle="->", lw=0.6, color="0.3"))
b.text(35.6, 81.30, f"−{100 * (1 - u / g):.1f} % for\n+{100 * (xu / xs - 1):.0f} % peak", fontsize=6, color="0.3", ha="center", va="top")
b.set_xlabel("peak IT load (kW)"); b.set_ylabel("expected + CVaR$_{95}$ (USD)")
b.set_xlim(28, 50); b.set_ylim(80.3, 83.6); b.set_title("(b) risk cost against peak load", loc="left")
fs.save(fig, FIG / "figE1_exact_certificate")
