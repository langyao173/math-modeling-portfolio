"""Extension E1 — prospective holdout on the 2018, 2022 and 2026 World Cups.

The contest model (official four-feature multinomial logit, team history frozen
at 2014) is applied unchanged to every match of the three tournaments played
after the data end.  It is compared with
  * a base-rate forecast (1930-2014 WDL frequencies),
  * an external Elo benchmark: World Football Elo ratings updated on all
    international matches (martj42/international_results, CC0) and mapped to
    W/D/L by an ordered logit fitted only on 1930-2014 World Cup matches,
  * an equal-weight linear pool of the two.
Outcomes are after extra time (shoot-outs count as draws), matching the
training data.  Scores: log loss, Brier, ranked probability score (RPS),
accuracy; paired match-level bootstrap for differences.
"""
from __future__ import annotations
import hashlib, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "contest_submission"
sys.path.insert(0, str(SUB))
from data_io import load_competition_data            # noqa: E402
from prediction_api import MatchResultPredictor      # noqa: E402

RES = ROOT / "data" / "external" / "international_results.csv"
OUT = ROOT / "results" / "extension"; OUT.mkdir(parents=True, exist_ok=True)
YEARS = (2018, 2022, 2026); RNG = np.random.default_rng(2026)
K_BY = {"FIFA World Cup": 60, "Friendly": 20}


def k_factor(t: str) -> float:
    if t in K_BY: return K_BY[t]
    t = t.lower()
    if "qualification" in t: return 40
    if any(s in t for s in ["euro", "copa am", "african cup of nations", "afc asian cup", "gold cup", "confederations"]): return 50
    return 30


def elo_ratings(res: pd.DataFrame) -> pd.DataFrame:
    """Pre-match Elo for every match (eloratings.net formulation)."""
    R, pre_h, pre_a = {}, [], []
    for h, a, hs, as_, t, neu in res[["home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]].itertuples(index=False):
        rh, ra = R.get(h, 1500.0), R.get(a, 1500.0)
        pre_h.append(rh); pre_a.append(ra)
        if np.isnan(hs) or np.isnan(as_): continue
        dr = rh - ra + (0 if neu else 100)
        we = 1 / (10 ** (-dr / 400) + 1)
        w = 1.0 if hs > as_ else 0.5 if hs == as_ else 0.0
        n = abs(hs - as_); g = 1 if n <= 1 else 1.5 if n == 2 else (11 + n) / 8
        d = k_factor(t) * g * (w - we)
        R[h], R[a] = rh + d, ra - d
    out = res.copy(); out["elo_home"], out["elo_away"] = pre_h, pre_a
    out["elo_diff"] = out.elo_home - out.elo_away + np.where(out.neutral, 0, 100)
    return out


def label(hs, as_):  # 0 home win, 1 draw, 2 away win (contest coding)
    return np.where(hs > as_, 0, np.where(hs == as_, 1, 2))


def stage_names(n_matches: int, year: int):
    groups = 72 if year == 2026 else 48
    ko = (["Round of 32"] * 16 if year == 2026 else []) + ["Round of 16"] * 8 + ["Quarter-finals"] * 4 + \
         ["Semi-finals"] * 2 + ["Match for third place", "Final"]
    s = ["Group"] * groups + ko
    assert len(s) == n_matches, (year, n_matches, len(s))
    return s


def scores(p, y):
    p = np.clip(p, 1e-12, 1); onehot = np.eye(3)[y]
    cum_p, cum_o = np.cumsum(p, 1), np.cumsum(onehot, 1)
    return {"log_loss": -np.log(p[np.arange(len(y)), y]), "brier": ((p - onehot) ** 2).sum(1),
            "rps": ((cum_p[:, :2] - cum_o[:, :2]) ** 2).sum(1) / 2, "accuracy": (p.argmax(1) == y).astype(float)}


def main():
    _, matches, _, _ = load_competition_data(SUB / "data")
    res = pd.read_csv(RES).sort_values("date", kind="stable").reset_index(drop=True)
    res["neutral"] = res.neutral.astype(str).str.upper().eq("TRUE")
    res = elo_ratings(res)
    wc = res[res.tournament == "FIFA World Cup"].copy(); wc["year"] = wc.date.str[:4].astype(int)

    train = wc[(wc.year <= 2014)].dropna(subset=["home_score"])
    y_tr = label(train.home_score.values, train.away_score.values)
    # ordered logit on Elo difference: classes ordered away win < draw < home win
    om = OrderedModel(2 - y_tr, train[["elo_diff"]].values / 100.0, distr="logit").fit(method="bfgs", disp=False)
    base = np.bincount(y_tr, minlength=3) / len(y_tr)

    rows = []
    for yr in YEARS:
        t = wc[wc.year == yr].reset_index(drop=True)
        q = pd.DataFrame({"Home Team Name": t.home_team, "Away Team Name": t.away_team, "Year": yr,
                          "Stage": stage_names(len(t), yr)})
        pr = MatchResultPredictor(matches, cutoff_year=yr).predict_dataframe(q)
        p_logit = pr[["主队胜概率", "平局概率", "客队胜概率"]].to_numpy(float)
        p_elo = om.model.predict(om.params, exog=t[["elo_diff"]].values / 100.0)[:, ::-1]
        y = label(t.home_score.values, t.away_score.values)
        unknown = pr["球队实体警告"].astype(str).str.len().gt(0).to_numpy()
        for name, p in [("base_rate", np.tile(base, (len(t), 1))), ("contest_logit_frozen2014", p_logit),
                        ("elo_ordered_logit", p_elo), ("pool_logit_elo", 0.5 * (p_logit + p_elo))]:
            s = scores(p, y)
            for i in range(len(t)):
                rows.append({"year": yr, "match": i, "home": t.home_team[i], "away": t.away_team[i], "y": int(y[i]),
                             "unknown_team": bool(unknown[i]), "model": name, "p_home": p[i, 0], "p_draw": p[i, 1],
                             "p_away": p[i, 2], **{k: float(v[i]) for k, v in s.items()}})
    df = pd.DataFrame(rows); df.to_csv(OUT / "e1_match_level_predictions.csv", index=False)

    metr = ["log_loss", "brier", "rps", "accuracy"]
    tab = df.groupby(["model", "year"])[metr].mean().unstack("year")
    allm = df.groupby("model")[metr].mean(); allm.columns = pd.MultiIndex.from_product([metr, ["all"]])
    table = pd.concat([tab, allm], axis=1).sort_index(axis=1)
    table.round(4).to_csv(OUT / "e1_scores_by_tournament.csv")

    # paired bootstrap: Elo and pool vs contest logit (match-level resampling, all 232 matches)
    piv = {m: df[df.model == m].sort_values(["year", "match"]) for m in df.model.unique()}
    boot = {}
    for m in ["elo_ordered_logit", "pool_logit_elo", "base_rate"]:
        for k in ["log_loss", "rps", "brier"]:
            d = piv[m][k].to_numpy() - piv["contest_logit_frozen2014"][k].to_numpy()
            bs = [RNG.choice(d, len(d)).mean() for _ in range(5000)]
            boot[f"{m}_minus_logit_{k}"] = {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    summ = {"n_matches": int(df[df.model == "base_rate"].shape[0]),
            "share_matches_with_unseen_team_for_logit": float(df[df.model == "base_rate"].unknown_team.mean()),
            "ordered_logit_params": om.params.round(4).tolist(),
            "results_csv_sha256": hashlib.sha256(RES.read_bytes()).hexdigest(),
            "paired_bootstrap_vs_contest_logit": boot}
    (OUT / "e1_summary.json").write_text(json.dumps(summ, indent=2))
    print(table.round(4).to_string()); print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
