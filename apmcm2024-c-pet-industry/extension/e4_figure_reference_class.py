"""Figure E2: interval coverage on the reference class, and calibrated forecasts for China's cats and dogs.

(a) Share of held-out windows (M4 Demographic, 5 annual training values) whose outcome falls inside the
    nominal 80 % and 95 % intervals: classical OLS intervals of the linear trend against class-calibrated
    intervals of the same forecasts.
(b, c) Observed urban cats and dogs; the linear-trend forecast from 2023 (the paper's information set) for the
    two check years with its class-calibrated 80 % / 95 % bands; the ARIMA forecast from 2025 with its bands (the
    most accurate method of the class for seven training values); the paper's forecasts.
Reads results/extension/e3_*.csv (e3_reference_class.py).   Run:  python extension/e4_figure_reference_class.py
"""
import json, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results" / "extension", ROOT / "figures" / "extension"
FIG.mkdir(parents=True, exist_ok=True); fs.apply()
S = json.loads((RES / "e3_summary.json").read_text())
cov = pd.read_csv(RES / "e3_interval_coverage_L5.csv")
c5 = pd.read_csv(RES / "e3_china_L5.csv"); c7 = pd.read_csv(RES / "e3_china_L7.csv")
sys.path.insert(0, str(ROOT / "extension"))
from e3_reference_class import CHINA, PAPER   # noqa: E402

fig, axes = plt.subplots(1, 3, figsize=(fs.DOUBLE, 0.31 * fs.DOUBLE), gridspec_kw=dict(width_ratios=[0.9, 1, 1], wspace=0.38))
ax = axes[0]
for lv, mk in ((0.80, "o"), (0.95, "s")):
    ols = [S["L5"]["ols_linear_coverage"][f"h{h}"][str(int(lv * 100))] for h in (1, 2, 3)]
    cal = cov[(cov.method == "linear") & (cov.level == lv)].sort_values("h").test_coverage.to_numpy()
    ax.plot([1, 2, 3], ols, mk + "--", color=fs.OKABE_ITO[1], ms=3.5, lw=0.9, label=f"OLS interval, {int(lv * 100)} %")
    ax.plot([1, 2, 3], cal, mk + "-", color=fs.OKABE_ITO[0], ms=3.5, lw=0.9, label=f"class-calibrated, {int(lv * 100)} %")
    ax.axhline(lv, color="0.7", lw=0.6, ls=":")
ax.set_xticks([1, 2, 3]); ax.set_xlabel("horizon (years)"); ax.set_ylabel("coverage on held-out series")
ax.set_ylim(0.15, 1.0); ax.set_title("(a) 5-point trend intervals", loc="left")
ax.legend(loc="lower left", fontsize=5.4, handlelength=1.8)
for ax, sp, title in ((axes[1], "cat", "(b) urban pet cats"), (axes[2], "dog", "(c) urban pet dogs")):
    yrs = sorted(CHINA[sp]); ax.plot(yrs, [CHINA[sp][y] / 100 for y in yrs], "o-", color="k", ms=3, lw=1.0, label="observed", zorder=5)
    for frame, method, origin, col, lab in ((c5, "linear", 2023, fs.OKABE_ITO[1], "linear trend from 2023"),
                                            (c7, "ARIMA", 2025, fs.OKABE_ITO[0], "ARIMA from 2025")):
        g = frame[(frame.species == sp) & (frame.method == method) & (frame.origin == origin)].sort_values("year")
        if origin == 2023:
            g = g[g.year <= 2025]                    # the check years; the 2025 fan starts there
        x = [origin] + g.year.tolist(); y0 = CHINA[sp][origin] / 100
        ax.fill_between(x, [y0] + (g.lo95 / 100).tolist(), [y0] + (g.hi95 / 100).tolist(), color=col, alpha=0.12, lw=0)
        ax.fill_between(x, [y0] + (g.lo80 / 100).tolist(), [y0] + (g.hi80 / 100).tolist(), color=col, alpha=0.25, lw=0)
        ax.plot(x, [y0] + (g.forecast / 100).tolist(), "--", color=col, lw=1.0, label=lab)
    py = sorted(PAPER[sp]); ax.plot(py, [PAPER[sp][y] / 100 for y in py], "x", color=fs.OKABE_ITO[3], ms=4, mew=1.0, label="paper's forecast")
    ax.set_title(title, loc="left"); ax.set_xticks(range(2019, 2029, 2)); ax.set_xlim(2018.6, 2028.4)
    ax.set_ylabel("million")
axes[1].legend(loc="upper left", fontsize=5.4, handlelength=1.6)
fs.save(fig, FIG / "figE2_reference_class")
