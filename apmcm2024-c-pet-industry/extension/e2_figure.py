"""Figure: the paper's Problem-1 forecasts against the populations published afterwards.

Observed urban cat and dog numbers (10^4 animals), China Pet Industry White Paper (Petdata):
2020 edition for 2020; 2025 edition for 2021-2024 (as reproduced by MobTech, Mar 2025); 2026 edition for
2025 (values also used in e1_forecast_check.py).  Forecasts: paper, Sec. 5 (combined model), 2025
interpolated geometrically as in e1_forecast_check.py.
Run:  python extension/e2_figure.py
"""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "extension"; FIG.mkdir(parents=True, exist_ok=True); fs.apply()
OBS = {"cat": {2020: 4862, 2021: 5806, 2022: 6536, 2023: 6980, 2024: 7153, 2025: 7289},
       "dog": {2020: 5222, 2021: 5429, 2022: 5119, 2023: 5175, 2024: 5258, 2025: 5343}}
FC = {"cat": {2024: 7297, 2026: 8386}, "dog": {2024: 5110, 2026: 5061}}
for k in FC:
    FC[k][2025] = (FC[k][2024] * FC[k][2026]) ** 0.5

fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE * 0.72, 0.30 * fs.DOUBLE), gridspec_kw=dict(wspace=0.32))
for ax, k, title in zip(axes, ("cat", "dog"), ("(a) pet cats", "(b) pet dogs")):
    yo = sorted(OBS[k]); yf = sorted(FC[k])
    ax.axvspan(2023.5, 2026.4, color="0.94", lw=0)
    ax.plot(yo, [OBS[k][y] / 100 for y in yo], "o-", color="k", ms=3, lw=1.0, label="observed (white paper)")
    ax.plot([2023] + yf, [OBS[k][2023] / 100] + [FC[k][y] / 100 for y in yf], "s--", color=fs.OKABE_ITO[1], ms=3, lw=1.0,
            label="paper forecast (made in 2024)")
    for y in (2024, 2025):
        e = 100 * (FC[k][y] / OBS[k][y] - 1); lo, hi = sorted((FC[k][y] / 100, OBS[k][y] / 100))
        ax.plot([y, y], [lo, hi], color=fs.OKABE_ITO[1], lw=0.6, ls=":")
        lab = f"{e:+.1f} %".replace("-", "\u2212"); span = max(OBS[k].values()) / 100 - min(OBS[k].values()) / 100
        if hi - lo < 0.12 * span:      # short gap: label below the pair instead of beside it
            ax.text(y, lo - 0.04 * span, lab, fontsize=6, color=fs.OKABE_ITO[1], ha="center", va="top")
        else:
            ax.text(y + 0.09, (lo + hi) / 2, lab, fontsize=6, color=fs.OKABE_ITO[1], va="center")
    ax.set_title(title, loc="left"); ax.set_xticks(range(2020, 2027)); ax.set_xlim(2019.6, 2026.4)
    ax.set_ylabel("urban population (million)")
    ax.text(2023.6, ax.get_ylim()[1] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "forecast period", fontsize=5.5, color="0.4", va="top")
axes[0].legend(loc="lower right", fontsize=6)
fs.save(fig, FIG / "figE1_forecast_vs_observed")
