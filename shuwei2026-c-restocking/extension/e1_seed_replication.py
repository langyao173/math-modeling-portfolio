"""Extension E1, step 2 — what a 30-seed evaluation of Problem 3 can and cannot tell.

Problem 3 is scored by the organisers on 30 undisclosed seeds; the submission
reports 30 public validation seeds and 30 team-chosen test seeds.  This script
reads the 300-seed replays written by `e1_replay_large.py` and answers four
questions:

  Q1  How close are the submission's 30-seed means to the 300-seed means?
  Q2  Is the ranking of the frozen rule (FROZEN) against its alternatives stable
      across random 30-seed pools drawn from the same environment pool?
  Q3  Is the validation-to-test reversal between FROZEN (family prior) and the
      prefix-feature prior a seed artefact or a property of the environment pools,
      and where do the two real pools fall among the 60 emulated pools that the
      selection rule used (results/p3_pooltransfer_*)?
  Q4  How much margin does the frozen rule keep above the service floor
      (mean weighted on-time rate >= 0.50 per case, nominal mapping, 30 trials)?

Main scope is the nominal duration mapping (alpha = 1, beta = 0.5 s), on which the
paper's headline objective, the selection rule and the service floor are defined;
the mean over all five mappings is reported alongside as a robustness check.
Seed is the independent unit: all policies share the jobs drawn for a seed (common
random numbers), so paired differences are averaged within a seed first.

Run:  python extension/e1_seed_replication.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "results" / "extension"
N_POOLS, POOL = 5000, 30
ALT = ["EDD-median", "RO-feat-static", "RO-feat-learn", "RO-bma-static"]
SPLITS = ("validation", "test")


def ci(x):
    x = np.asarray(x, float); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x))
    return [float(m), float(m - 1.96 * se), float(m + 1.96 * se)], float(se)


def by_seed(d, col="utility"):
    return d.groupby(["split", "policy", "seed"])[col].mean().unstack("policy")


def main():
    rng = np.random.default_rng(2026)
    big = pd.concat([pd.read_csv(OUT / f"e1_trials_{s}.csv.gz") for s in SPLITS], ignore_index=True)
    sub = pd.concat([pd.read_csv(RES / f"p3_trials_{s}.csv") for s in SPLITS], ignore_index=True)
    sub = sub[sub.policy.isin(big.policy.unique())]
    assert big.invalid_actions.sum() == 0 and big.valid_protocol.all()
    nominal = lambda d: d[(d.alpha == 1.0) & (d.beta == 0.5)]
    scopes = {"nominal": (by_seed(nominal(big)), by_seed(nominal(sub))), "all_mappings": (by_seed(big), by_seed(sub))}
    pools = {sp: np.array([rng.choice(big.seed.nunique(), POOL, replace=False) for _ in range(N_POOLS)]) for sp in SPLITS}
    out = {"design": {"seeds_per_split": int(big.seed.nunique()), "seed_range": [int(big.seed.min()), int(big.seed.max())],
                      "trials_per_policy_and_split": int(len(big) // big.groupby(["split", "policy"]).ngroups),
                      "policies": sorted(big.policy.unique()), "random_30_seed_pools": N_POOLS, "invalid_actions": 0}}
    levels = []
    for scope, (S, T) in scopes.items():
        out[scope] = {}
        for sp in SPLITS:
            s, t = S.loc[sp], T.loc[sp]
            for p in s.columns:                                               # Q1: level of each policy
                c, se = ci(s[p]); q = s[p].to_numpy()[pools[sp]].mean(1)
                levels.append(dict(scope=scope, split=sp, policy=p, mean_300=c[0], lo95=c[1], hi95=c[2], sd_of_30_seed_pool=q.std(ddof=1),
                                   submitted_30=float(t[p].mean()), submitted_percentile=float((q <= t[p].mean()).mean())))
            res = {}
            for a in ALT:                                                     # Q2: ranking FROZEN vs alternative
                dif = (s["FROZEN"] - s[a]).to_numpy(); q = dif[pools[sp]].mean(1); c, se = ci(dif)
                res[a] = dict(diff_300=c, sd_of_30_seed_pool=float(q.std(ddof=1)), p_frozen_ahead_on_30_seed_pool=float((q > 0).mean()),
                              diff_submitted_30=float((t["FROZEN"] - t[a]).mean()))
            res["CLAIRVOYANT_minus_FROZEN"] = ci(s["CLAIRVOYANT"] - s["FROZEN"])[0]
            out[scope][sp] = res
    lev = pd.DataFrame(levels); lev.to_csv(OUT / "e1_policy_levels.csv", index=False, float_format="%.5f")
    # hindsight regret of each learnt prior relative to the best prior on each real pool (300 seeds; CLAIRVOYANT and EDD excluded)
    for scope in scopes:
        L = lev[(lev.scope == scope) & lev.policy.isin(["FROZEN", "RO-feat-static", "RO-feat-learn", "RO-bma-static"])]
        R = L.pivot(index="policy", columns="split", values="mean_300")
        reg = R.max() - R
        out[scope]["hindsight_regret"] = {p: dict(validation=float(reg.loc[p, "validation"]), test=float(reg.loc[p, "test"]),
                                                  max=float(reg.loc[p].max())) for p in reg.index}

    # Q3: family prior (FROZEN) vs prefix-feature priors, nominal mapping, against the 60 emulated pools
    S = scopes["nominal"][0]
    pt = pd.read_csv(RES / "p3_pooltransfer_pools.csv"); summ = pd.read_csv(RES / "p3_pooltransfer_summary.csv")
    rev = {}
    for a in ("RO-feat-static", "RO-feat-learn"):
        g = {sp: ci(S.loc[sp][a] - S.loc[sp]["FROZEN"]) for sp in SPLITS}
        emu = (pt[f"U_{a}"] - pt["U_RO-family-static"]).to_numpy()
        row = summ[(summ.a == a) & (summ.b == "RO-family-static")].iloc[0]
        rev[a] = dict(gap_vs_family_validation=g["validation"][0], gap_vs_family_test=g["test"][0],
                      swing_validation_minus_test=g["validation"][0][0] - g["test"][0][0],
                      swing_se=float(np.hypot(g["validation"][1], g["test"][1])),
                      emulated=dict(pools=int(len(emu)), mean=float(emu.mean()), sd=float(emu.std(ddof=1)),
                                    trial_se_within_pool=float(row.trial_se_within_pool),
                                    share_at_or_above_validation=float((emu >= g["validation"][0][0]).mean()),
                                    share_at_or_below_test=float((emu <= g["test"][0][0]).mean())))
    out["reversal_nominal"] = rev

    # Q4: service floor, nominal mapping, per case, on random 30-seed pools
    ot = nominal(big).groupby(["split", "policy", "case_id", "seed"]).weighted_on_time_rate.mean()
    floor = []
    for (sp, p, c), v in ot.groupby(level=[0, 1, 2]):
        v = v.to_numpy(); q = v[pools[sp]].mean(1)
        floor.append(dict(split=sp, policy=p, case=c, on_time_300=v.mean(), q01_of_30_seed_pool=float(np.quantile(q, 0.01)),
                          p_below_floor_on_30_seed_pool=float((q < 0.5).mean())))
    floor = pd.DataFrame(floor); floor.to_csv(OUT / "e1_service_floor.csv", index=False, float_format="%.4f")
    out["service_floor"] = {sp: {p: dict(min_case_on_time=float(f.on_time_300.min()), worst_case=f.loc[f.on_time_300.idxmin(), "case"],
                                         max_p_below_floor=float(f.p_below_floor_on_30_seed_pool.max()))
                                 for p, f in floor[floor.split == sp].groupby("policy")} for sp in SPLITS}
    (OUT / "e1_summary.json").write_text(json.dumps(out, indent=1))
    pd.set_option("display.width", 220)
    print(lev.round(4).to_string(index=False)); print(floor.round(3).to_string(index=False)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
