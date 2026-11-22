"""Figure: prospective ranked probability score by tournament (lower is better)."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results" / "extension", ROOT / "figures" / "extension"
FIG.mkdir(parents=True, exist_ok=True); fs.apply()
df = pd.read_csv(RES / "e1_match_level_predictions.csv")
rng = np.random.default_rng(7)
models = [("base_rate", "Base rate", fs.OKABE_ITO[7]), ("contest_logit_frozen2014", "Contest logit (frozen 2014)", fs.OKABE_ITO[0]),
          ("elo_ordered_logit", "Elo benchmark", fs.OKABE_ITO[1]), ("pool_logit_elo", "Equal-weight pool", fs.OKABE_ITO[2])]
groups = [2018, 2022, 2026, "All"]
fig, ax = plt.subplots(figsize=(fs.SINGLE, 0.62 * fs.SINGLE))
for j, (m, lab, c) in enumerate(models):
    for i, g in enumerate(groups):
        v = df[(df.model == m) & ((df.year == g) if g != "All" else True)].rps.to_numpy()
        bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        x = i + (j - 1.5) * 0.18
        ax.errorbar(x, v.mean(), yerr=[[v.mean() - np.percentile(bs, 2.5)], [np.percentile(bs, 97.5) - v.mean()]],
                    fmt="o", ms=3.2, color=c, elinewidth=0.8, capsize=0, label=lab if i == 0 else None)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([f"{g}\n(n={64 if g in (2018, 2022) else 104 if g == 2026 else 232})" for g in groups])
ax.set_ylabel("Ranked probability score (lower is better)")
ax.axvline(2.5, color="0.8", lw=0.6)
ax.legend(loc="upper left", bbox_to_anchor=(0, 1.22), ncol=2, handletextpad=0.2, columnspacing=0.8)
fs.save(fig, FIG / "figE1_prospective_rps")
