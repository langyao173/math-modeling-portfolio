"""Extension E1 — test the proposal's central thesis on a real randomised experiment.

Thesis: allocating incentives by estimated causal uplift ("who buys BECAUSE of
the coupon") beats allocating by predicted response ("who will buy").  The
proposal's quantitative results come from a calibrated 100k-user digital twin;
here the same comparison is run on the MineThatData E-Mail Analytics challenge
(Hillstrom, 2008): 64 000 customers randomly assigned to a men's e-mail, a
women's e-mail or no e-mail, with 2-week visit, conversion and spend.
Treatment = any e-mail (propensity 2/3 by design; IPTW weights are therefore
known constants).  Policies are learned on a random half and evaluated on the
other half with the Qini curve; 20 repetitions.
"""
from __future__ import annotations
import io, json, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "extension"; OUT.mkdir(parents=True, exist_ok=True)
URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
CACHE = ROOT / "data" / "hillstrom.csv"   # not redistributed; downloaded on first run


def load():
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(urllib.request.urlopen(URL, timeout=60).read())
    d = pd.read_csv(CACHE)
    X = pd.get_dummies(d[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel", "history_segment"]],
                       columns=["zip_code", "channel", "history_segment"], dtype=float)
    t = (d.segment != "No E-Mail").astype(int).to_numpy()
    return X.to_numpy(float), t, d


def qini(score, y, t, fracs):
    """Incremental outcomes (per customer in the population) when the top-q fraction is treated,
    estimated from the randomised holdout: n_q * (mean_y_treated - mean_y_control) within top-q."""
    o = np.argsort(-score); n = len(score); out = []
    for q in fracs:
        k = max(int(round(q * n)), 1); s = o[:k]
        yt, yc = y[s][t[s] == 1], y[s][t[s] == 0]
        out.append(k * (yt.mean() - yc.mean()) / n if len(yt) and len(yc) else 0.0)
    return np.array(out)


def gbm(): return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0)


def policies(Xtr, ytr, ttr, Xte):
    resp = gbm().fit(Xtr, ytr).predict_proba(Xte)[:, 1]                     # "who will buy" (pooled response)
    m1 = gbm().fit(Xtr[ttr == 1], ytr[ttr == 1]); m0 = gbm().fit(Xtr[ttr == 0], ytr[ttr == 0])
    cate_tr = m1.predict_proba(Xtr)[:, 1] - m0.predict_proba(Xtr)[:, 1]
    cate = m1.predict_proba(Xte)[:, 1] - m0.predict_proba(Xte)[:, 1]        # T-learner uplift
    # isotonic calibration of uplift on the training half with the transformed outcome
    # z = y (t - e) / (e (1 - e)), e = 2/3 known (IPTW), which is unbiased for the CATE
    e = 2 / 3; z = ytr * (ttr - e) / (e * (1 - e))
    iso = IsotonicRegression(out_of_bounds="clip").fit(cate_tr, z)
    return {"random": np.random.default_rng(0).random(len(Xte)), "response model (who will buy)": resp,
            "T-learner uplift (who buys because of it)": cate, "uplift + IPTW isotonic calibration": iso.predict(cate) + 1e-9 * cate}


def main():
    X, t, d = load(); fracs = np.linspace(0.05, 1.0, 20); rows = []
    for outcome in ["visit", "conversion"]:
        y = d[outcome].to_numpy()
        for rep in range(20):
            tr, te = train_test_split(np.arange(len(y)), test_size=0.5, random_state=rep, stratify=t * 2 + y)
            P = policies(X[tr], y[tr], t[tr], X[te])
            for name, s in P.items():
                q = qini(s, y[te], t[te], fracs)
                rows.append({"outcome": outcome, "rep": rep, "policy": name,
                             "qini_area": float(np.trapezoid(q - fracs * q[-1], fracs)),
                             "increment_at_30pct": float(np.interp(0.30, fracs, q)), "increment_at_100pct": float(q[-1]),
                             "share_of_total_increment_at_30pct": float(np.interp(0.30, fracs, q) / q[-1])})
    df = pd.DataFrame(rows); df.to_csv(OUT / "e1_hillstrom_policy_scores.csv", index=False)
    summ = df.groupby(["outcome", "policy"])[["qini_area", "increment_at_30pct", "share_of_total_increment_at_30pct"]].agg(["mean", "std"]).round(5)
    summ.to_csv(OUT / "e1_hillstrom_summary.csv")
    paired = {}
    for outcome in ["visit", "conversion"]:
        a = df[(df.outcome == outcome) & (df.policy == "uplift + IPTW isotonic calibration")].sort_values("rep")
        b = df[(df.outcome == outcome) & (df.policy == "response model (who will buy)")].sort_values("rep")
        dd = a.share_of_total_increment_at_30pct.to_numpy() - b.share_of_total_increment_at_30pct.to_numpy()
        paired[outcome] = {"uplift_minus_response_share_at_30pct_mean": float(dd.mean()), "uplift_better_in_reps": int((dd > 0).sum())}
    # structure behind the result: does a high baseline go with a high or a low uplift?
    from scipy.stats import spearmanr
    y = d["visit"].to_numpy(); tr, te = train_test_split(np.arange(len(y)), test_size=0.5, random_state=0, stratify=t * 2 + y)
    m1 = gbm().fit(X[tr][t[tr] == 1], y[tr][t[tr] == 1]); m0 = gbm().fit(X[tr][t[tr] == 0], y[tr][t[tr] == 0])
    base = m0.predict_proba(X[te])[:, 1]; up = m1.predict_proba(X[te])[:, 1] - base
    resp = gbm().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    paired["structure_visit"] = {"spearman_baseline_vs_uplift": float(spearmanr(base, up)[0]),
                                 "spearman_response_vs_uplift": float(spearmanr(resp, up)[0]),
                                 "top30_overlap_response_vs_uplift": float(len(set(np.argsort(-resp)[:int(.3 * len(te))]) & set(np.argsort(-up)[:int(.3 * len(te))])) / int(.3 * len(te)))}
    (OUT / "e1_hillstrom_paired.json").write_text(json.dumps(paired, indent=2))
    print(summ.to_string()); print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
