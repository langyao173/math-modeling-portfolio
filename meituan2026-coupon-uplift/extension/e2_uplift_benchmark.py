"""Extension E2 — causal targeting against response targeting on three real randomised campaigns.

E1 found that, on the Hillstrom e-mail experiment, the proposal's uplift model (IPTW T-learner with
isotonic calibration) ranks customers no better than a plain response model.  E2 asks whether that is a
property of the T-learner or of the data, by running the established uplift learners on three public
randomised campaigns that bracket the proposal's setting (a marketing contact with a purchase outcome):

  hillstrom  64 000 customers, e-mail vs none (propensity 2/3), outcome: visit within two weeks
             (Hillstrom 2008, MineThatData);
  x5         200 039 loyalty-card customers of X5 Retail Group, SMS offer vs none (propensity 1/2),
             outcome: purchase after the campaign; features from the pre-campaign purchase history
             (X5 RetailHero uplift challenge, 2019);
  lenta      687 029 customers of Lenta, SMS vs none (propensity 3/4), outcome: store visit
             (Lenta uplift dataset, 2020).

Learners (identical gradient-boosting base learners everywhere; treatment is randomised, so the
propensity is the known design constant e):
  response            P(y = 1 | x), pooled over arms ("who will buy")
  S-learner           one model on (x, t); tau = f(x, 1) - f(x, 0)
  T-learner           separate models per arm (the proposal's ranking; its isotonic calibration is
                      monotone and leaves the ranking unchanged)
  transformed outcome regression of y (t - e) / (e (1 - e)) on x (Athey & Imbens 2016)
  X-learner           Kunzel, Sekhon, Bickel & Yu (2019)
  DR-learner          Kennedy (2023), two-fold cross-fitting
  causal forest       generalised random forest via econml CausalForestDML (Athey, Tibshirani & Wager 2019)

Evaluation on the held-out half of repeated stratified 50/50 splits (Hillstrom 20, X5 10, Lenta 5):
Qini coefficient (area between the Qini curve and the random line, divided by the same area for the
best possible ordering; sklift.qini_auc_score), uplift at 30 %, share of the total incremental effect
captured by treating the top 30 %, and the CATE calibration of the T-learner before and after the
proposal's isotonic step (decile bins, observed uplift = difference in arm means within the bin).

Data are downloaded on first run into data/external/ (not redistributed). Lenta, the largest campaign, is held in
float32; finished splits are checkpointed in data/external/ so an interrupted run resumes.
Run:  python extension/e2_uplift_benchmark.py [hillstrom|x5|lenta ...]
"""
from __future__ import annotations
import gzip, json, sys, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklift.metrics import qini_auc_score, uplift_at_k

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data" / "external", ROOT / "results" / "extension"
DATA.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
S3 = "https://sklift.s3.eu-west-2.amazonaws.com/"
URLS = {"hillstrom.csv": "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
        "lenta_dataset.csv.gz": S3 + "lenta_dataset.csv.gz", "uplift_train.csv.gz": S3 + "uplift_train.csv.gz",
        "clients.csv.gz": S3 + "clients.csv.gz", "purchases.csv.gz": S3 + "purchases.csv.gz"}
REPS = {"hillstrom": 20, "x5": 10, "lenta": 5}
CF_MAX_TRAIN = 100_000          # causal-forest training rows per split (subsampled above this size)


def fetch(name):
    p = DATA / name
    if not p.exists():
        with urllib.request.urlopen(URLS[name], timeout=600) as r, open(p, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    return p


# ----------------------------------------------------------------------------- data
def load_hillstrom():
    d = pd.read_csv(fetch("hillstrom.csv"))
    X = pd.get_dummies(d[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]],
                       columns=["zip_code", "channel"], dtype=float)
    return X, (d.segment != "No E-Mail").astype(int).to_numpy(), d.visit.to_numpy(), 2 / 3


def x5_features():
    """Client-level features from the pre-campaign purchase history (45.8 M receipt lines).

    The receipt lines are first collapsed to transactions (header fields are constant within a
    transaction; line count and quantity are summed), then to clients.  Polars' streaming engine keeps
    the peak memory well below the size of the raw file.
    """
    cache = DATA / "x5_features.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import polars as pl
    from datetime import timedelta
    raw = DATA / "purchases.csv"
    if not raw.exists():                                   # polars streams only uncompressed CSV
        with gzip.open(fetch("purchases.csv.gz"), "rb") as fi, open(raw, "wb") as fo:
            while chunk := fi.read(1 << 24):
                fo.write(chunk)
    head = ["regular_points_received", "express_points_received", "regular_points_spent",
            "express_points_spent", "purchase_sum"]
    tr = (pl.scan_csv(raw, schema_overrides={c: pl.Float64 for c in head + ["product_quantity"]})
          .select(["client_id", "transaction_id", "transaction_datetime", "store_id", "product_quantity"] + head)
          .group_by(["client_id", "transaction_id"])
          .agg([pl.col(c).first() for c in ["transaction_datetime", "store_id"] + head]
               + [pl.len().alias("n_lines"), pl.col("product_quantity").sum().alias("quantity")])
          .with_columns(pl.col("transaction_datetime").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("ts"))
          .collect(engine="streaming"))
    end = tr["ts"].max()
    a = (tr.group_by("client_id").agg(
            n_trans=pl.len(), sum_total=pl.col("purchase_sum").sum(), sum_mean=pl.col("purchase_sum").mean(),
            sum_std=pl.col("purchase_sum").std(), sum_max=pl.col("purchase_sum").max(),
            lines_mean=pl.col("n_lines").mean(), qty_total=pl.col("quantity").sum(),
            n_stores=pl.col("store_id").n_unique(), reg_recv=pl.col("regular_points_received").sum(),
            exp_recv=pl.col("express_points_received").sum(), reg_spent=pl.col("regular_points_spent").sum(),
            exp_spent=pl.col("express_points_spent").sum(), first_ts=pl.col("ts").min(), last_ts=pl.col("ts").max(),
            n_recent30=(pl.col("ts") > end - timedelta(days=30)).sum())
         .to_pandas().set_index("client_id"))
    del tr
    end = pd.Timestamp(end)
    a["recency_d"] = (end - a.last_ts).dt.total_seconds() / 86400
    a["tenure_d"] = (a.last_ts - a.first_ts).dt.total_seconds() / 86400
    a["freq_per30d"] = a.n_trans / np.maximum(a.tenure_d, 1) * 30
    a["share_exp_spent"] = -a.exp_spent / np.maximum(a.sum_total, 1)
    a = a.drop(columns=["first_ts", "last_ts"])
    c = pd.read_csv(fetch("clients.csv.gz")).set_index("client_id")
    iss, red = pd.to_datetime(c.first_issue_date), pd.to_datetime(c.first_redeem_date)
    f = pd.DataFrame({"age": c.age.where(c.age.between(14, 100)), "gender_F": (c.gender == "F").astype(float),
                      "gender_M": (c.gender == "M").astype(float), "issue_age_d": (end - iss).dt.days,
                      "redeemed": red.notna().astype(float), "issue_to_redeem_d": (red - iss).dt.days}, index=c.index)
    f = f.join(a, how="left")
    f.to_parquet(cache)
    raw.unlink()                                           # keep only the compact feature cache
    return f


def load_x5():
    u = pd.read_csv(fetch("uplift_train.csv.gz")).set_index("client_id")
    X = x5_features().reindex(u.index)
    return X.reset_index(drop=True), u.treatment_flg.to_numpy(), u.target.to_numpy(), 0.5


def load_lenta():
    head = pd.read_csv(fetch("lenta_dataset.csv.gz"), nrows=5)
    num = {c: np.float32 for c in head.columns if c not in ("group", "gender")}
    d = pd.read_csv(fetch("lenta_dataset.csv.gz"), dtype=num)
    t = (d.group == "test").astype(int).to_numpy(); y = d.response_att.to_numpy()
    d = d.drop(columns=["group", "response_att", "response_sms", "response_viber"])
    d["gender"] = d.gender.map({"Ж": 1.0, "М": 0.0})
    return d.astype(np.float32), t, y, float(t.mean())          # float32: 687 k x 193 features in half the memory


LOADERS = {"hillstrom": load_hillstrom, "x5": load_x5, "lenta": load_lenta}


# ----------------------------------------------------------------------------- learners
def clf(): return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                                                 l2_regularization=1.0, random_state=0)
def reg(): return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                                l2_regularization=1.0, random_state=0)


def fit_scores(Xtr, ttr, ytr, Xte, e, seed):
    p1 = lambda m, X: m.predict_proba(X)[:, 1]
    s, tm = {}, {}
    t0 = time.time()
    s["response"] = p1(clf().fit(Xtr, ytr), Xte); tm["response"] = time.time() - t0
    t0 = time.time()
    dt = Xtr.dtype
    m = clf().fit(np.column_stack([Xtr, ttr.astype(dt)]), ytr)
    s["S-learner"] = (p1(m, np.column_stack([Xte, np.ones(len(Xte), dt)]))
                      - p1(m, np.column_stack([Xte, np.zeros(len(Xte), dt)])))
    tm["S-learner"] = time.time() - t0
    t0 = time.time()
    m1, m0 = clf().fit(Xtr[ttr == 1], ytr[ttr == 1]), clf().fit(Xtr[ttr == 0], ytr[ttr == 0])
    s["T-learner"] = p1(m1, Xte) - p1(m0, Xte); tm["T-learner"] = time.time() - t0
    z = ytr * (ttr - e) / (e * (1 - e))
    t0 = time.time()
    s["transformed outcome"] = reg().fit(Xtr, z).predict(Xte); tm["transformed outcome"] = time.time() - t0
    t0 = time.time()
    d1 = ytr[ttr == 1] - p1(m0, Xtr[ttr == 1]); d0 = p1(m1, Xtr[ttr == 0]) - ytr[ttr == 0]
    tau1, tau0 = reg().fit(Xtr[ttr == 1], d1), reg().fit(Xtr[ttr == 0], d0)
    s["X-learner"] = e * tau0.predict(Xte) + (1 - e) * tau1.predict(Xte); tm["X-learner"] = time.time() - t0 + tm["T-learner"]
    t0 = time.time()
    phi, tau_oof = np.zeros(len(ytr)), np.zeros(len(ytr))
    for a, b in StratifiedKFold(2, shuffle=True, random_state=seed).split(Xtr, ttr * 2 + ytr):
        q1 = clf().fit(Xtr[a][ttr[a] == 1], ytr[a][ttr[a] == 1]); q0 = clf().fit(Xtr[a][ttr[a] == 0], ytr[a][ttr[a] == 0])
        mu1, mu0 = p1(q1, Xtr[b]), p1(q0, Xtr[b])
        tau_oof[b] = mu1 - mu0
        phi[b] = mu1 - mu0 + ttr[b] * (ytr[b] - mu1) / e - (1 - ttr[b]) * (ytr[b] - mu0) / (1 - e)
    s["DR-learner"] = reg().fit(Xtr, phi).predict(Xte); tm["DR-learner"] = time.time() - t0
    # the proposal's calibration step: isotonic regression of the IPTW-transformed outcome on the
    # (cross-fitted) T-learner prediction; monotone, so the ranking is the T-learner's (ties broken by it)
    iso = IsotonicRegression(out_of_bounds="clip").fit(tau_oof, z)
    s["T-learner + isotonic (proposal)"] = iso.predict(s["T-learner"])
    t0 = time.time()
    from econml.dml import CausalForestDML
    from sklearn.dummy import DummyClassifier
    idx = np.arange(len(ytr))
    if len(idx) > CF_MAX_TRAIN:
        idx = np.random.default_rng(seed).choice(idx, CF_MAX_TRAIN, replace=False)
    cf = CausalForestDML(model_y=reg(), model_t=DummyClassifier(strategy="prior"), discrete_treatment=True, cv=2,
                         n_estimators=200, min_samples_leaf=50, max_samples=0.3, random_state=seed, n_jobs=2)
    Xc = np.nan_to_num(Xtr[idx], nan=-1.0)
    cf.fit(ytr[idx].astype(float), ttr[idx], X=Xc)
    s["causal forest"] = cf.effect(np.nan_to_num(Xte, nan=-1.0)).ravel(); tm["causal forest"] = time.time() - t0
    s["random"] = np.random.default_rng(seed).random(len(Xte))
    return s, tm


# ----------------------------------------------------------------------------- metrics
def captured_share(score, y, t, q=0.3):
    o = np.argsort(-score, kind="stable"); n = len(o); k = int(round(q * n))
    inc = lambda sel: len(sel) * (y[sel][t[sel] == 1].mean() - y[sel][t[sel] == 0].mean())
    return inc(o[:k]) / inc(o)


def calibration(pred, y, t, bins=10):
    """Weighted mean |predicted - observed| uplift over equal-count bins of the prediction."""
    o = np.argsort(pred, kind="stable"); err = 0.0
    for b in np.array_split(o, bins):
        obs = y[b][t[b] == 1].mean() - y[b][t[b] == 0].mean()
        err += len(b) / len(o) * abs(pred[b].mean() - obs)
    return err


def run(name):
    X, t, y, e = LOADERS[name]()
    X = X.to_numpy(np.float32 if name == "lenta" else float)
    rows, cal = [], []
    ckpt = DATA / f"_e2_{name}_splits.jsonl"                      # finished splits, so an interrupted run resumes
    done = set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            r = json.loads(line); rows += r["rows"]; cal += r["cal"]; done.add(r["rep"])
    for rep in range(REPS[name]):
        if rep in done:
            continue
        n0, c0 = len(rows), len(cal)
        tr, te = train_test_split(np.arange(len(y)), test_size=0.5, random_state=rep, stratify=t * 2 + y)
        S, TM = fit_scores(X[tr], t[tr], y[tr], X[te], e, rep)
        for k, v in S.items():
            if k == "T-learner + isotonic (proposal)":
                v = v + 1e-9 * S["T-learner"]                                   # same ranking as the T-learner
            rows.append(dict(dataset=name, rep=rep, learner=k, qini=qini_auc_score(y[te], v, t[te]),
                             uplift_at_30=uplift_at_k(y[te], v, t[te], strategy="overall", k=0.3),
                             share_top30=captured_share(v, y[te], t[te]), fit_seconds=TM.get(k, np.nan)))
        for k in ("T-learner", "T-learner + isotonic (proposal)", "X-learner", "DR-learner", "causal forest"):
            cal.append(dict(dataset=name, rep=rep, learner=k, calibration_error=calibration(S[k], y[te], t[te])))
        print(name, rep, {k: round(r["qini"], 4) for k, r in zip(S, rows[-len(S):])}, flush=True)
        with open(ckpt, "a") as fh:
            fh.write(json.dumps({"rep": rep, "rows": rows[n0:], "cal": cal[c0:]}, default=float) + "\n")
        del S
    info = dict(dataset=name, n=int(len(y)), features=int(X.shape[1]), propensity=float(e), ate=float(y[t == 1].mean() - y[t == 0].mean()),
                base_rate_control=float(y[t == 0].mean()), reps=REPS[name])
    return pd.DataFrame(rows), pd.DataFrame(cal), info


def main():
    names = sys.argv[1:] or list(LOADERS)
    for name in names:
        r, c, info = run(name)
        r = r.sort_values(["rep"], kind="stable"); c = c.sort_values(["rep"], kind="stable")
        r.to_csv(OUT / f"e2_{name}_scores.csv", index=False); c.to_csv(OUT / f"e2_{name}_calibration.csv", index=False)
        (OUT / f"e2_{name}_info.json").write_text(json.dumps(info, indent=1))
        (DATA / f"_e2_{name}_splits.jsonl").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
