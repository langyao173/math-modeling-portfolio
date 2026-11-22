"""Figures for the extension analyses (E1-E3)."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results" / "extension", ROOT / "figures" / "extension"
FIG.mkdir(parents=True, exist_ok=True)
fs.apply()
LAB = {"R": r"$R^*$", "dP": r"$\Delta P^*$", "phi": r"$\phi$"}

# ---- Fig. E1: predictive-interval coverage under leave-one-level-out ----
e1 = pd.read_csv(RES / "e1_level_holdout.csv")
lv = {v: sorted(e1.loc[e1.held_out == v, "level"].unique()) for v in e1.held_out.unique()}
e1["kind"] = [("boundary" if l in (lv[h][0], lv[h][-1]) else "interior") for h, l in zip(e1.held_out, e1.level)]
cov = e1[e1.model.isin(["GPR-Matern", "UK-cubic+Matern"])]
fig, ax = plt.subplots(figsize=(fs.SINGLE, 0.62 * fs.SINGLE))
targets = ["R", "dP", "phi"]; kinds = ["interior", "boundary"]
models = {"GPR-Matern": (fs.OKABE_ITO[1], "o", "GPR (paper surrogate)"),
          "UK-cubic+Matern": (fs.OKABE_ITO[0], "s", "Universal kriging (cubic trend)")}
xpos = {(t, k): i * 2.9 + 1.15 * j for i, t in enumerate(targets) for j, k in enumerate(kinds)}
for m, (c, mk, lab) in models.items():
    sub = cov[cov.model == m]
    for (t, k), x in xpos.items():
        v = sub[(sub.target == t) & (sub.kind == k)].pi95_coverage.to_numpy()
        off = -0.17 if m == "GPR-Matern" else 0.17
        ax.scatter(np.full(v.size, x + off), v, s=7, color=c, marker=mk, alpha=0.45, linewidths=0)
        ax.plot([x + off - 0.13, x + off + 0.13], [v.mean()] * 2, color=c, lw=1.4)
    ax.scatter([], [], color=c, marker=mk, s=12, label=lab)
ax.axhline(0.95, color="0.35", lw=0.7, ls=(0, (3, 2)))
ax.text(4.42, 0.938, "nominal 95 %", va="top", ha="left", fontsize=5.5, color="0.35")
ax.set_xticks(list(xpos.values()))
ax.set_xticklabels([k for (t, k) in xpos], fontsize=6)
for t in targets:
    ax.text((xpos[(t, "interior")] + xpos[(t, "boundary")]) / 2, -0.2, LAB[t], transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5)
ax.set_ylabel("95% interval coverage on the held-out level")
ax.set_ylim(-0.03, 1.05); ax.set_xlim(-0.6, xpos[("phi", "boundary")] + 0.6)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.0)
fs.save(fig, FIG / "figE1_level_holdout_coverage")

# ---- Fig. E2: posterior Pareto-membership probability, N = 4 slice ----
g = pd.read_csv(RES / "e2_posterior_decision_grid_uk.csv")
nx = pd.read_csv(RES / "e3_next_simulations_uk.csv")
data = pd.read_csv(ROOT / "data" / "attachment2.tsv", sep="\t", skiprows=2,
                   names=["id", "beta", "gamma", "N", "R", "dP", "phi"]).dropna()
s = g[g.N == 4].pivot(index="gamma", columns="beta", values="P_pareto")
fig, ax = plt.subplots(figsize=(fs.SINGLE, 0.72 * fs.SINGLE))
im = ax.pcolormesh(s.columns, s.index, s.values, cmap="Greys", vmin=0, vmax=1, shading="nearest", rasterized=True)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label(r"$P(\mathbf{x}$ non-dominated$)$"); cb.outline.set_linewidth(0.5)
d4 = data[data.N == 4]
ax.scatter(d4.beta, d4.gamma, s=9, facecolors="white", edgecolors="black", linewidths=0.5, label="CFD sample (N = 4)", zorder=3)
ax.scatter([0.22], [4.5], marker="*", s=70, color=fs.OKABE_ITO[1], edgecolors="black", linewidths=0.4, zorder=4, clip_on=False,
           label="Recommended (0.22, 4.5, 4)")
grp = nx.groupby(["beta", "gamma"]).agg(Ns=("N", lambda v: ",".join(str(int(x)) for x in sorted(v))),
                                        first=("step", "min")).reset_index()
ax.scatter(grp.beta, grp.gamma, marker="D", s=16, color=fs.OKABE_ITO[2], edgecolors="black", linewidths=0.4, zorder=4,
           clip_on=False, label="Proposed next CFD run (label: N)")
for _, r in grp.iterrows():
    ax.annotate(f"N={r.Ns}", (r.beta, r.gamma), xytext=(4, -1), textcoords="offset points", fontsize=5.5,
                color=fs.OKABE_ITO[2], va="center")
ax.set_ylim(2.965, 4.535)
ax.set_xlabel(r"Pin-fin width ratio $\beta$"); ax.set_ylabel(r"Manifold depth ratio $\gamma$")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, handletextpad=0.3, columnspacing=0.8)
fs.save(fig, FIG / "figE2_posterior_pareto_N4")
print("figures written to", FIG)
