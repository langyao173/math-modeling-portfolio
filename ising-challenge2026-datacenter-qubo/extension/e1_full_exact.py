"""Extension E1 — certify the submitted schedule against the exact full-instance MILP.

The submission solved exact MILPs only for controlled neighbourhoods of 30, 60
and 100 jobs (all other jobs fixed to the incumbent).  Here the same exact
five-scenario recourse model (objective Expected + CVaR95, same action-cost
guard, no dummy actions) is built for ALL 588 jobs / 1 929 actions and solved
with HiGHS, giving the certified optimality gap of the submitted
Ising/QUBO-based schedule.
"""
from __future__ import annotations
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(sys.argv[1]).resolve(); OUT = Path(sys.argv[2]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
TL = float(sys.argv[3]) if len(sys.argv) > 3 else 1200.0
spec = importlib.util.spec_from_file_location("ecb", ROOT / "src/07_solve_isling/build_exact_controlled_baseline.py")
ecb = importlib.util.module_from_spec(spec); sys.modules["ecb"] = ecb; spec.loader.exec_module(ecb)
# job selection for neighbourhoods needs the interim site table; for the full instance
# no selection is made, so the packaged validation site table is substituted.
tmp = OUT / "_site_for_selection.parquet"
pd.read_csv(ROOT / "deliverables/final_solution_package/validation_data/site_timeseries_for_validation.csv",
            encoding="utf-8-sig").to_parquet(tmp)
ecb.SELECTION_SITE_PATH = tmp
data = ecb.load_frozen_data()
jobs = sorted(pd.read_csv(ecb.ACTIONS_PATH, encoding="utf-8-sig").job_id.astype(str).unique())
t0 = time.time()
res = ecb.solve_controlled_case(data, jobs, time_limit=TL, mip_rel_gap=0.0, lexicographic_tolerance=1e-6)
res["wall_seconds_total"] = time.time() - t0
cur = json.loads((ROOT / "deliverables/final_solution_package/FINAL_METRICS.json").read_text(encoding="utf-8"))
res["submitted_metrics"] = cur
(OUT / "e1_full_exact.json").write_text(json.dumps(res, indent=2, default=str))
m = res.get("metrics") or {}
print(json.dumps({k: res[k] for k in ["controlled_jobs", "candidate_actions", "variables", "integer_variables", "constraints"]}, indent=1))
print("primary", res["primary"]); print("metrics", {k: m.get(k) for k in ["expected_recourse_cost_usd", "cvar95_recourse_cost_usd", "audit_objective_usd", "peak_it_mw", "true_action_cost_usd"]})
