"""Figure: success probability of the submitted compiler against the RouteCQC baseline on unseen circuits.

Reads results/extension/e1_generalisation_samples.csv (run_suites.py / summarise.py).  One point per
circuit and seed; invalid samples (compile time over the limit) score zero and are marked x;
panel (b) shows valid samples only.
Run:  python extension/e3_figure.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "extension"; FIG.mkdir(parents=True, exist_ok=True); fs.apply()
d = pd.read_csv(ROOT / "results" / "extension" / "e1_generalisation_samples.csv")
d.loc[~d.valid.astype(bool), "success"] = 0.0
w = d.pivot_table(index=["suite", "benchmark", "seed"], columns="compiler", values=["success", "two_qubit"]).reset_index()
SUITE = {"S1_unseen_sizes": ("S1: unseen sizes (46 circuits)", fs.OKABE_ITO[0], "o"),
         "S2_qasmbench": ("S2: QASMBench (11 circuits)", fs.OKABE_ITO[1], "s")}

fig, (a, b) = plt.subplots(1, 2, figsize=(fs.DOUBLE * 0.78, 0.33 * fs.DOUBLE), gridspec_kw=dict(wspace=0.32))
for s, (lab, c, mk) in SUITE.items():
    x = w[w.suite == s]; ok = x[("success", "submission")] > 0
    a.scatter(x[("success", "baseline_cirq_cqc")][ok], x[("success", "submission")][ok], s=10, color=c, marker=mk, alpha=0.75,
              linewidths=0, label=lab)
    a.scatter(x[("success", "baseline_cirq_cqc")][~ok], x[("success", "submission")][~ok], s=16, color=c, marker="x", linewidths=0.9)
    b.scatter(x[("two_qubit", "baseline_cirq_cqc")][ok], x[("two_qubit", "submission")][ok], s=10, color=c, marker=mk, alpha=0.75,
              linewidths=0)
a.scatter([], [], s=16, color="0.3", marker="x", linewidths=0.9, label="compile timeout (scored 0)")
a.plot([0, 1], [0, 1], color="0.5", lw=0.6, ls="--")
a.set_xlim(0, 1.02); a.set_ylim(-0.03, 1.02); a.set_aspect("equal")
a.set_xlabel("success probability, RouteCQC baseline"); a.set_ylabel("success probability, submission")
a.legend(loc="lower right", fontsize=6, handletextpad=0.2); a.set_title("(a) success (above the diagonal = better)", loc="left")
m = max(w[("two_qubit", "baseline_cirq_cqc")].max(), w[("two_qubit", "submission")].max()) * 1.05
b.plot([1.5, m], [1.5, m], color="0.5", lw=0.6, ls="--"); b.set_xscale("log"); b.set_yscale("log")
b.set_xlim(1.5, m); b.set_ylim(1.5, m); b.set_aspect("equal")
b.set_xticks([2, 5, 10, 20, 50, 100]); b.set_xticklabels(["2", "5", "10", "20", "50", "100"])
b.set_yticks([2, 5, 10, 20, 50, 100]); b.set_yticklabels(["2", "5", "10", "20", "50", "100"])
b.minorticks_off()
b.set_xlabel("two-qubit gates, RouteCQC baseline"); b.set_ylabel("two-qubit gates, submission")
b.set_title("(b) compiled CX count, log scale (below = fewer)", loc="left")
fs.save(fig, FIG / "figE1_generalisation")
