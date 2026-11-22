"""Figure E1: Problem 3 on 300 fresh seeds per split.

(a) mean objective U per policy (nominal mapping), 300 seeds (95 % CI, seed-clustered) and the submission's 30-seed value;
(b) FROZEN minus the prefix-feature prior on 5 000 random 30-seed pools of each split;
(c) gap prefix-feature prior minus family prior on the 60 emulated pools used by the selection rule,
    with the gaps measured on the real validation and test pools (300 seeds, nominal mapping).
Run after e1_seed_replication.py:  python extension/e2_figure.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
RES, OUT, FIG = ROOT / "results", ROOT / "results" / "extension", ROOT / "figures" / "extension"
FIG.mkdir(parents=True, exist_ok=True); fs.apply()
ALT = "RO-feat-learn"                                   # best configuration on validation in the submission
COL = {"validation": fs.OKABE_ITO[0], "test": fs.OKABE_ITO[1]}
LABEL = {"FROZEN": "FROZEN (family prior)", "RO-feat-learn": "prefix-feature prior, learning", "RO-feat-static": "prefix-feature prior",
         "RO-bma-static": "model-averaged prior", "EDD-median": "EDD baseline", "CLAIRVOYANT": "clairvoyant"}

lev = pd.read_csv(OUT / "e1_policy_levels.csv"); lev = lev[lev.scope == "nominal"]
summ = json.loads((OUT / "e1_summary.json").read_text())
big = pd.concat([pd.read_csv(OUT / f"e1_trials_{s}.csv.gz") for s in ("validation", "test")])
S = big[(big.alpha == 1.0) & (big.beta == 0.5)].groupby(["split", "policy", "seed"]).utility.mean().unstack("policy")

fig, ax = plt.subplots(1, 3, figsize=(fs.DOUBLE, 0.36 * fs.DOUBLE), gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.42))
order = ["CLAIRVOYANT", "FROZEN", "RO-feat-learn", "RO-feat-static", "RO-bma-static", "EDD-median"]
for k, split in enumerate(("validation", "test")):
    d = lev[lev.split == split].set_index("policy").loc[order]
    y = np.arange(len(order)) + (k - 0.5) * 0.28
    ax[0].errorbar(d.mean_300, y, xerr=[d.mean_300 - d.lo95, d.hi95 - d.mean_300], fmt="o", ms=3, lw=0.8, color=COL[split],
                   label=f"{split}, 300 seeds")
    ax[0].plot(d.submitted_30, y, "o", ms=4, mfc="none", mew=0.7, color=COL[split], label=f"{split}, submitted 30 seeds")
ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels([LABEL[p] for p in order]); ax[0].invert_yaxis()
ax[0].set_xlabel("objective $U$ (nominal mapping, 3 cases)"); ax[0].legend(loc="upper left", fontsize=5.5, handlelength=1)
ax[0].set_xlim(0.45, 0.97)
ax[0].set_title("(a) policy level", loc="left")

rng = np.random.default_rng(2026)
for split in ("validation", "test"):
    dif = (S.loc[split]["FROZEN"] - S.loc[split][ALT]).to_numpy()
    q = dif[np.array([rng.choice(len(dif), 30, replace=False) for _ in range(5000)])].mean(1)
    ax[1].hist(q, bins=50, color=COL[split], alpha=0.55, label=f"{split}: FROZEN ahead in {100 * (q > 0).mean():.0f} %")
    sub = summ["nominal"][split][ALT]["diff_submitted_30"]
    ax[1].axvline(sub, color=COL[split], lw=0.9, ls="--")
ax[1].axvline(0, color="k", lw=0.6)
ax[1].plot([], [], color="0.3", lw=0.9, ls="--", label="submitted 30 seeds")
ax[1].set_xlabel("FROZEN − prefix-feature prior, learning ($U$)"); ax[1].set_ylabel("random 30-seed pools (of 5 000)")
ax[1].legend(loc="upper left", fontsize=5.5, handlelength=1); ax[1].set_title("(b) ranking on 30-seed pools", loc="left")

pt = pd.read_csv(RES / "p3_pooltransfer_pools.csv"); emu = pt[f"U_{ALT}"] - pt["U_RO-family-static"]
rv = summ["reversal_nominal"][ALT]
ax[2].hist(emu, bins=15, color="0.75", label="60 emulated pools (30 trials each)")
for split, key in (("validation", "gap_vs_family_validation"), ("test", "gap_vs_family_test")):
    m, lo, hi = rv[key]
    ax[2].axvline(m, color=COL[split], lw=1.0); ax[2].axvspan(lo, hi, color=COL[split], alpha=0.2, lw=0)
    ax[2].text(m + 0.006, 12.3, f"{split}\npool", color=COL[split], va="top", ha="left", fontsize=5.5, linespacing=0.95)
ax[2].axvline(0, color="k", lw=0.6)
ax[2].set_xlabel("prefix-feature (learning) − family prior ($U$)"); ax[2].set_ylabel("pools"); ax[2].set_ylim(0, 12.6)
ax[2].set_xlim(min(emu.min(), rv["gap_vs_family_test"][1]) - 0.06, emu.max() + 0.02)
ax[2].legend(loc="upper right", fontsize=5.5, handlelength=1, bbox_to_anchor=(1, 0.80)); ax[2].set_title("(c) real pools vs emulated pools", loc="left")
fs.save(fig, FIG / "figE1_seed_replication")
