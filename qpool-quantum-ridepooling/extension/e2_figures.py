"""Figure for the precision audit (E1/E2)."""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results" / "extension", ROOT / "figures" / "extension"
FIG.mkdir(parents=True, exist_ok=True); fs.apply()
s = json.load(open(RES / "e1_precision_audit_summary.json"))["all_nondegenerate"]
groups = [("Worst returned tie", "worst_tie_loss_vehmin"), ("Best tie", "best_tie_loss_vehmin"),
          ("Worst tie + QPool polish", "after_polish_loss_vehmin")]
fig, ax = plt.subplots(figsize=(fs.SINGLE, 0.6 * fs.SINGLE))
x = np.arange(len(groups)); wd = 0.36
for j, (tag, lab, c) in enumerate([("ladder", r"Power-of-two $\rho$ (paper)", fs.OKABE_ITO[0]),
                                    ("fine", r"Finest passing $\rho$ (step 1/8)", fs.OKABE_ITO[1])]):
    v = [max(s[f"{k}_{tag}"], 0.5) for _, k in groups]
    b = ax.bar(x + (j - 0.5) * wd, v, wd, color=c, label=lab)
    for xi, vi in zip(x + (j - 0.5) * wd, v):
        ax.text(xi, vi * 1.12, f"{vi:.0f}", ha="center", va="bottom", fontsize=6)
ax.set_yscale("log"); ax.set_ylim(5, 5000)
ax.set_xticks(x); ax.set_xticklabels([g for g, _ in groups])
ax.set_ylabel("Planned vehicle-minutes lost\n(sum over 169 hardware-ready components)")
ax.legend(loc="upper right")
fs.save(fig, FIG / "figE1_quantization_loss")
print("ok")
