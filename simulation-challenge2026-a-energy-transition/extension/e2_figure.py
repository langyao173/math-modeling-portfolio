"""Figure: first simulated year (2025) against official 2025 statistics.

Reads results/extension/e1_holdout_2025.csv (e1_holdout_2025.py).  The thermal row is not shown: the
statistical "thermal" category includes oil-fired and waste-to-energy plants that the model does not contain.
The non-fossil share is shown as a relative error of the share (the README also gives it in percentage points).
Run:  python extension/e2_figure.py
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "extension"; FIG.mkdir(parents=True, exist_ok=True); fs.apply()
d = pd.read_csv(ROOT / "results" / "extension" / "e1_holdout_2025.csv")
d = d[~d.quantity.str.startswith("Coal + gas")]
LAB = {"Primary energy consumption (Mtce)": "primary energy consumption", "Electricity demand vs. consumption (TWh)": "electricity demand",
       "Non-fossil share of primary energy": "non-fossil share of primary energy", "Hydro incl. pumped storage (GW)": "hydro incl. pumped storage capacity",
       "Nuclear capacity (GW)": "nuclear capacity", "Wind capacity (GW)": "wind capacity", "Solar capacity (GW)": "solar capacity"}
d = d.assign(label=d.quantity.map(LAB)).sort_values("relative_error_pct")
fig, ax = plt.subplots(figsize=(fs.SINGLE * 1.2, 0.55 * fs.SINGLE))
ax.axvspan(-2, 2, color="0.93", lw=0)
for i, (_, r) in enumerate(d.iterrows()):
    c = fs.OKABE_ITO[1] if abs(r.relative_error_pct) > 2 else fs.OKABE_ITO[0]
    ax.barh(i, r.relative_error_pct, color=c, height=0.6)
    gap = r.model_2025 - r.observed_2025
    note = f"{r.relative_error_pct:+.1f} %".replace("-", "−")
    if r.quantity.endswith("(GW)") and abs(r.relative_error_pct) > 2:
        note += f"  ({gap:+.0f} GW)".replace("-", "−")
    x = r.relative_error_pct
    ax.text(x + (0.25 if x >= 0 else -0.25), i, note, va="center", ha="left" if x >= 0 else "right", fontsize=6)
ax.axvline(0, color="k", lw=0.6)
ax.set_yticks(range(len(d))); ax.set_yticklabels(d.label)
ax.set_xlim(-16.5, 5); ax.set_xlabel("model minus observed, 2025 (% of observed)")
ax.text(0, len(d) - 0.35, "±2 % band", ha="center", fontsize=5.5, color="0.4")
fs.save(fig, FIG / "figE1_holdout_2025")
