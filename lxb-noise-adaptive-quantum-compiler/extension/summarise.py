"""Summarise S1/S2 generalisation runs: per-suite scores, paired per-circuit gaps, invalid counts."""
import json, sys, glob
from pathlib import Path
import pandas as pd
RES, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
rows = []
for f in sorted(RES.glob("*.json")):
    suite, comp, seed = f.stem.split("__"); s = json.loads(f.read_text())
    for r in s["sample_results"]:
        rows.append({"suite": suite, "compiler": comp, "seed": int(seed.replace("seed", "")), "benchmark": r["benchmark_name"],
                     "family": r["benchmark_name"].rsplit("_", 2)[0] if suite.startswith("S1") else r["benchmark_name"].replace("qasmbench_", ""),
                     "success": r["success_rate"], "valid": r["valid"], "two_qubit": r["two_qubit_gate_count"], "depth": r["depth"],
                     "compile_s": r["compile_time_sec"]})
df = pd.DataFrame(rows); df.to_csv(OUT / "e1_generalisation_samples.csv", index=False)
p = df.pivot_table(index=["suite", "benchmark", "family", "seed"], columns="compiler", values=["success", "two_qubit", "valid"]).reset_index()
p.columns = ["_".join(c).strip("_") for c in p.columns]
p["gap"] = p.success_submission - p.success_baseline_cirq_cqc
summ = {}
for suite, g in p.groupby("suite"):
    summ[suite] = {"circuits": int(g.benchmark.nunique()), "seeds": int(g.seed.nunique()),
                   "mean_success_submission": round(float(g.success_submission.mean()), 4),
                   "mean_success_baseline": round(float(g.success_baseline_cirq_cqc.mean()), 4),
                   "mean_gap": round(float(g.gap.mean()), 4),
                   "invalid_submission": int((g.valid_submission == 0).sum()), "invalid_baseline": int((g.valid_baseline_cirq_cqc == 0).sum()),
                   "circuit_seed_pairs_submission_better": int((g.gap > 0.02).sum()), "worse": int((g.gap < -0.02).sum()),
                   "mean_2q_submission": round(float(g.two_qubit_submission.mean()), 1), "mean_2q_baseline": round(float(g.two_qubit_baseline_cirq_cqc.mean()), 1)}
fam = p.groupby(["suite", "family"])[["success_submission", "success_baseline_cirq_cqc", "gap"]].mean().round(3)
fam.to_csv(OUT / "e1_generalisation_by_family.csv")
(OUT / "e1_generalisation_summary.json").write_text(json.dumps(summ, indent=2))
print(json.dumps(summ, indent=2)); print(fam.to_string())
