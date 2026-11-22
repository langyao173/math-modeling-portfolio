"""Extension E3 — which environment drives the validation-to-test reversal?

The held-out duration results of the submission (results/p3_duration_by_env.csv)
show that adding the 0.2-s prefix features lowers the log-duration error on the
validation environments but raises it on the test environments, and that the rise
comes mostly from one environment, logistics_x_courier_station_0041 (21 of the 62
test units; log-MAE 0.25 with the family prior, 0.70 with prefix features).  Two
checks relate that environment to the scheduling reversal found in E1:

  (i)  observational: for every 300-seed trial the hidden job sources are
       regenerated with the submission's reference sampler (`Scenario.sample_sources`,
       a function of case, seed and split only, hence identical for all policies),
       and the paired gap U(prefix-feature prior) - U(FROZEN) at the nominal mapping
       is regressed on the share of the trial's jobs drawn from each environment
       (case fixed effects, seed-clustered standard errors);
  (ii) interventional: the three priors are replayed on the same 300 seeds with
       that environment removed from the test pool (nominal mapping), using the
       submission's runner unchanged apart from the pool.

Policies never read the source pools (they see only the released prefixes), so
removing an environment changes the jobs, not the decision rules.
Requires the official task folder and interim/frames.npz (as e1_replay_large.py).  About 3 minutes on two cores.

Run:  python extension/e3_attribution.py [--procs 2]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import p3_core, p3_run                                       # noqa: E402  (task folder resolved in common.py)
from common import load_json                                 # noqa: E402

OUT = ROOT / "results" / "extension"
ENV = "logistics_x_courier_station_0041"
ALT = ["RO-feat-static", "RO-feat-learn"]
SEEDS = [2026130000 + i for i in range(300)]


def preflight():
    """Problem 3 replays need the official task folder and interim/frames.npz (built by step s02 from the raw archives)."""
    from common import TASK_DIR, INTERIM
    miss = [p for p in (Path(TASK_DIR) / "common_conditions.json", Path(INTERIM) / "frames.npz") if not p.exists()]
    if miss:
        sys.exit("missing %s - see SUPPORT_PACKAGE.md, step 1, then run `python code/run_all.py --only s02_build_dataset`"
                 % ", ".join(map(str, miss)))


class ScenarioWithout(p3_core.Scenario):
    """Official scenario with one environment removed from the test pools."""
    def __init__(self):
        super().__init__()
        for k in list(self.pools):
            if k[0] == "test":
                self.pools[k] = [r for r in self.pools[k] if r["environment_id"] != ENV]


p3_run.Scenario = ScenarioWithout   # module level, so that worker processes (fork or spawn) build the reduced scenario;
                                    # the observational part below uses the official p3_core.Scenario directly

def observational(D):
    tab = []
    for split in ("validation", "test"):
        d = D[D.split == split].copy()
        envs = sorted(c.split("::")[1] for c in d.columns if c.startswith("share::") and d[c].sum() > 0)
        for a in ALT:
            d["gap"] = d[a] - d["FROZEN"]
            for e in envs:
                d["s"] = d[f"share::{e}"]
                if d.groupby("case_id").s.std().max() < 1e-12:       # share fixed by the case design: not identified
                    continue
                fit = smf.ols("gap ~ s + C(case_id)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["seed"]})
                lo, hi = fit.conf_int().loc["s"]
                tab.append(dict(split=split, alt=a, environment=e, mean_share=d.s.mean(), slope=fit.params["s"], slope_lo=lo, slope_hi=hi,
                                gap=d.gap.mean(), gap_at_zero_share=fit.params["s"] * (0 - d.s.mean()) + d.gap.mean()))
    return pd.DataFrame(tab)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=2); a = ap.parse_args(); preflight()
    scen = p3_core.Scenario()
    big = pd.concat([pd.read_csv(OUT / f"e1_trials_{s}.csv.gz") for s in ("validation", "test")], ignore_index=True)
    nom = big[(big.alpha == 1.0) & (big.beta == 0.5)]
    U = nom.pivot_table(index=["split", "case_id", "seed"], columns="policy", values="utility").reset_index()
    rows = []
    for _, r in U[["split", "case_id", "seed"]].iterrows():
        env = pd.Series([s["environment_id"] for s in scen.sample_sources(r.case_id, int(r.seed), r.split)]).value_counts(normalize=True)
        rows.append(dict(split=r.split, case_id=r.case_id, seed=r.seed, **{f"share::{k}": v for k, v in env.items()}))
    D = U.merge(pd.DataFrame(rows).fillna(0.0), on=["split", "case_id", "seed"])
    tab = observational(D); tab.to_csv(OUT / "e3_env_attribution.csv", index=False, float_format="%.4f")

    # (ii) counterfactual replay without the environment
    fzp = load_json("frozen_p3_policy.json"); tag = "e3_test_without_env"
    cf = p3_run.run(["FROZEN"] + ALT, "test", SEEDS, [(1.0, 0.5)], tag, sel=dict(mode=fzp["mode"], learn=fzp["learn"]), procs=a.procs)
    for f in (ROOT / "results").glob(f"p3_*_{tag}.*"):
        f.unlink()
    cf.to_csv(OUT / "e3_trials_test_without_env.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    assert cf.invalid_actions.sum() == 0
    S = cf.groupby(["policy", "seed"]).utility.mean().unstack("policy")
    out = {"environment": ENV, "nominal_mapping": True, "seeds": len(SEEDS)}
    for alt in ALT:
        g = (S[alt] - S["FROZEN"]); se = g.std(ddof=1) / np.sqrt(len(g))
        t = tab[(tab.split == "test") & (tab.alt == alt) & (tab.environment == ENV)].iloc[0]
        out[alt] = dict(gap_full_test_pool=float(t.gap), slope_per_unit_share=[float(t.slope), float(t.slope_lo), float(t.slope_hi)],
                        mean_share_of_env=float(t.mean_share), gap_at_zero_share_by_regression=float(t.gap_at_zero_share),
                        gap_test_pool_without_env=[float(g.mean()), float(g.mean() - 1.96 * se), float(g.mean() + 1.96 * se)])
    out["FROZEN_level_without_env"] = float(S["FROZEN"].mean())
    (OUT / "e3_summary.json").write_text(json.dumps(out, indent=1))
    pd.set_option("display.width", 220); print(tab.round(4).to_string(index=False)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
