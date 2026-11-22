"""Extension E2 — improving the contest forecaster without looking at the holdout.

E1 applied the submitted classifier unchanged to the 232 matches of 2018, 2022 and 2026: it beat the base
rate, an external Elo benchmark did better, and an equal-weight pool of the two had the best log loss.
E2 builds improved forecasters from information available before 2018 and freezes them before scoring
2018-2026 (training data: the contest's World Cup matches 1930-2014; international results for Elo).

Candidates
  contest logit (submitted)   the four contest features, multinomial logit with class_weight='balanced',
                              C = 0.001 (frozen 2014; as in E1)
  contest features, refit     the same four features without class re-weighting; C chosen by
                              leave-one-tournament-out log loss on 1982-2010 (the contest's own
                              selection window).  Class re-weighting optimises balanced accuracy, not
                              probabilities, so it inflates draw probabilities by construction.
  combined logit              the four contest features plus the pre-match Elo difference of the
                              international ratings (home advantage included), same selection protocol
  learned pool                linear pool w * contest + (1 - w) * Elo with w fitted on rolling-origin
                              forecasts of the 2002-2014 tournaments (each forecast uses only earlier data)
  Elo / equal pool            as in E1

All candidates are scored on the same 232 matches with log loss, ranked probability score (RPS), Brier
score and accuracy; paired match-level bootstrap (5 000 resamples) against the equal-weight pool and
against Elo; reliability (calibration) of the outcome probabilities in deciles.
Run:  python extension/e2_improved_forecaster.py
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_prospective_holdout as e1                                  # noqa: E402  (Elo, labels, scores, data paths)
from features.team_state import TeamStateEngine                     # noqa: E402
from models.registry import P3_FEATURE_NAMES, swap_prematch_feature_rows   # noqa: E402
from prediction_api import MatchResultPredictor, normalize_query_frame   # noqa: E402
from data_io import load_competition_data                           # noqa: E402

OUT = e1.OUT
RNG = np.random.default_rng(2026)
ALIAS = {"Germany FR": "Germany", "German DR": "East Germany", "Korea Republic": "South Korea", "Korea DPR": "North Korea",
         "IR Iran": "Iran", "Iran": "Iran", "USA": "United States", "China PR": "China", "Côte d'Ivoire": "Ivory Coast",
         "C�te d'Ivoire": "Ivory Coast", "Soviet Union": "Russia", "Czechoslovakia": "Czech Republic", "Zaire": "DR Congo",
         "Dutch East Indies": "Indonesia", "Serbia and Montenegro": "Serbia", "Bosnia and Herzegovina": "Bosnia and Herzegovina",
         "Republic of Ireland": "Republic of Ireland", "Northern Ireland": "Northern Ireland", "rn\">Republic of Ireland": "Republic of Ireland",
         "rn\">Bosnia and Herzegovina": "Bosnia and Herzegovina", "rn\">Serbia and Montenegro": "Serbia", "rn\">Trinidad and Tobago": "Trinidad and Tobago",
         "rn\">United Arab Emirates": "United Arab Emirates", "Yugoslavia": "Serbia"}


def attach_elo(m: pd.DataFrame, res: pd.DataFrame) -> np.ndarray:
    """Pre-match Elo difference (home - away, +100 if not neutral) of the international ratings for each contest match,
    matched on date and score (orientation checked); NaN if not found."""
    wc = res[res.tournament == "FIFA World Cup"].copy()
    key = {}
    for r in wc.itertuples():
        key.setdefault(r.date, []).append(r)
    out = []
    for dt, h0, a0, hs, as_ in zip(m.Datetime, m["Home Team Name"], m["Away Team Name"], m["Home Team Goals"], m["Away Team Goals"]):
        d = str(dt)[:10]; cand = key.get(d, [])
        h = ALIAS.get(h0, h0); a = ALIAS.get(a0, a0)
        val = np.nan
        for c in cand:
            if {c.home_team, c.away_team} == {h, a} or (c.home_score, c.away_score) in {(hs, as_), (as_, hs)}:
                same = (c.home_team == h) or (c.away_team == a) or ((c.home_score, c.away_score) == (hs, as_) and c.home_team != a)
                val = c.elo_diff if same else -c.elo_diff
                if {c.home_team, c.away_team} == {h, a}:
                    break
        out.append(val)
    return np.array(out, float)


def fit_logit(F: pd.DataFrame, y: np.ndarray, names, C, weighted=False):
    X = F[names].copy(); lab = y.copy()
    sw = swap_prematch_feature_rows(X, names) if all(n.endswith("Diff") for n in names) else -X
    X = pd.concat([X, sw], ignore_index=True); lab = np.concatenate([lab, np.where(lab == 0, 2, np.where(lab == 2, 0, 1))])
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, class_weight="balanced" if weighted else None, max_iter=2000).fit(sc.transform(X), lab)
    return lambda G: lr.predict_proba(sc.transform(G[names]))


def select_C(F, y, years, names, grid=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)):
    best = None
    for C in grid:
        ll = []
        for yr in years:
            tr, te = F.Year < yr, F.Year == yr
            p = np.clip(fit_logit(F[tr], y[tr.to_numpy()], names, C)(F[te]), 1e-12, 1)
            ll.extend(-np.log(p[np.arange(te.sum()), y[te.to_numpy()]]))
        best = min(best or (np.inf, C), (float(np.mean(ll)), C))
    return best[1], best[0]


def main():
    _, matches, _, _ = load_competition_data(e1.SUB / "data")
    res = pd.read_csv(e1.RES).sort_values("date", kind="stable").reset_index(drop=True)
    res["neutral"] = res.neutral.astype(str).str.upper().eq("TRUE")
    res = e1.elo_ratings(res)

    F = TeamStateEngine(form_decay=0.8, entity_policy="conservative", era_tournaments=5).transform_history_frozen(matches)
    F = F.reset_index(drop=True); F["Datetime"] = matches.Datetime.to_numpy()[: len(F)]
    F["EloIntl"] = attach_elo(matches.iloc[: len(F)].reset_index(drop=True), res) / 100.0
    y = F.Outcome.map({"Home Win": 0, "Draw": 1, "Away Win": 2}).to_numpy()
    matched = float(np.isfinite(F.EloIntl).mean())
    F["EloIntl"] = F.EloIntl.fillna(0.0)
    names4 = list(P3_FEATURE_NAMES); names5 = names4 + ["EloIntl"]
    sel_years = list(range(1982, 2011, 4))
    C4, ll4 = select_C(F, y, sel_years, names4); C5, ll5 = select_C(F, y, sel_years, names5)
    m4, m5 = fit_logit(F, y, names4, C4), fit_logit(F, y, names5, C5)

    # learned pool: rolling-origin forecasts 2002-2014 from both E1 models
    wc = res[res.tournament == "FIFA World Cup"].copy(); wc["year"] = wc.date.str[:4].astype(int)
    def elo_model(upto):
        tr = wc[wc.year < upto].dropna(subset=["home_score"])
        yt = e1.label(tr.home_score.values, tr.away_score.values)
        om = OrderedModel(2 - yt, tr[["elo_diff"]].values / 100.0, distr="logit").fit(method="bfgs", disp=False)
        return lambda t: om.model.predict(om.params, exog=t[["elo_diff"]].values / 100.0)[:, ::-1]
    pl, pe, yy = [], [], []
    for yr in (2002, 2006, 2010, 2014):
        t = wc[wc.year == yr].reset_index(drop=True)
        q = pd.DataFrame({"Home Team Name": t.home_team, "Away Team Name": t.away_team,   # resolved by the contest's
                          "Year": yr, "Stage": e1.stage_names(len(t), yr)})                      # own entity policy, as in E1
        pr = MatchResultPredictor(matches, cutoff_year=yr).predict_dataframe(q)
        pl.append(pr[["主队胜概率", "平局概率", "客队胜概率"]].to_numpy(float)); pe.append(elo_model(yr)(t))
        yy.append(e1.label(t.home_score.values, t.away_score.values))
    pl, pe, yy = np.vstack(pl), np.vstack(pe), np.concatenate(yy)
    nll = lambda w: -np.mean(np.log(np.clip((w * pl + (1 - w) * pe)[np.arange(len(yy)), yy], 1e-12, 1)))
    w_star = float(minimize_scalar(nll, bounds=(0, 1), method="bounded").x)

    # prospective scoring 2018-2026
    elo_final = elo_model(2015)
    rows = []
    for yr in e1.YEARS:
        t = wc[wc.year == yr].reset_index(drop=True)
        q = pd.DataFrame({"Home Team Name": t.home_team, "Away Team Name": t.away_team,
                          "Year": yr, "Stage": e1.stage_names(len(t), yr)})
        pred = MatchResultPredictor(matches, cutoff_year=yr)
        pr = pred.predict_dataframe(q)
        G, _ = pred._matrix_and_warnings(normalize_query_frame(q, "e2", entity_policy=pred.entity_policy))
        G = G.copy(); G["EloIntl"] = t.elo_diff.to_numpy() / 100.0
        p_sub = pr[["主队胜概率", "平局概率", "客队胜概率"]].to_numpy(float); p_elo = elo_final(t)
        y_t = e1.label(t.home_score.values, t.away_score.values)
        P = {"contest logit (submitted)": p_sub, "contest features, refit": m4(G), "combined logit (contest features + Elo)": m5(G),
             "Elo (ordered logit)": p_elo, "equal-weight pool": 0.5 * (p_sub + p_elo), "learned pool": w_star * p_sub + (1 - w_star) * p_elo}
        for name, p in P.items():
            s = e1.scores(p, y_t)
            for i in range(len(t)):
                rows.append({"year": yr, "match": i, "model": name, "y": int(y_t[i]), "p_home": p[i, 0], "p_draw": p[i, 1], "p_away": p[i, 2],
                             **{k: float(v[i]) for k, v in s.items()}})
    df = pd.DataFrame(rows); df.to_csv(OUT / "e2_match_level_predictions.csv", index=False)
    metr = ["log_loss", "rps", "brier", "accuracy"]
    tab = df.groupby("model")[metr].mean().sort_values("log_loss"); tab.round(4).to_csv(OUT / "e2_scores.csv")
    piv = {m: df[df.model == m].sort_values(["year", "match"]) for m in df.model.unique()}
    boot = {}
    for ref in ("equal-weight pool", "Elo (ordered logit)", "contest logit (submitted)"):
        for m in piv:
            if m == ref: continue
            for k in ("log_loss", "rps"):
                d = piv[m][k].to_numpy() - piv[ref][k].to_numpy()
                bs = [RNG.choice(d, len(d)).mean() for _ in range(5000)]
                boot[f"{m} minus {ref} ({k})"] = [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    cal = []
    for m, g in df.groupby("model"):
        P = g[["p_home", "p_draw", "p_away"]].to_numpy(); Y = np.eye(3)[g.y.to_numpy()]
        p, o = P.ravel(), Y.ravel(); idx = np.argsort(p)
        ece = 0.0
        for b in np.array_split(idx, 10):
            cal.append({"model": m, "p_mean": float(p[b].mean()), "obs_freq": float(o[b].mean()), "n": int(len(b))})
            ece += len(b) / len(p) * abs(p[b].mean() - o[b].mean())
        cal.append({"model": m, "p_mean": np.nan, "obs_freq": np.nan, "n": -1, "ece": float(ece),
                    "mean_p_draw": float(g.p_draw.mean()), "obs_draw": float((g.y == 1).mean())})
    pd.DataFrame(cal).to_csv(OUT / "e2_calibration.csv", index=False)
    # per-tournament gaps to Elo, and whether they differ across tournaments (permutation test on the
    # variance of the three tournament means of the match-level log-loss difference)
    by_year, rng = {}, np.random.default_rng(7)
    for m in ("contest features, refit", "combined logit (contest features + Elo)"):
        diffs = {}
        for yr in e1.YEARS:
            d = (piv[m][piv[m].year == yr].log_loss.to_numpy() - piv["Elo (ordered logit)"][piv["Elo (ordered logit)"].year == yr].log_loss.to_numpy())
            bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
            diffs[yr] = d; by_year[f"{m} minus Elo, {yr} (log_loss)"] = [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        allx, sizes = np.concatenate(list(diffs.values())), [len(v) for v in diffs.values()]
        obs = np.var([v.mean() for v in diffs.values()])
        perm = [np.var([x.mean() for x in np.split(rng.permutation(allx), np.cumsum(sizes)[:-1])]) for _ in range(10000)]
        by_year[f"{m} minus Elo, heterogeneity permutation p"] = float((1 + np.sum(np.array(perm) >= obs)) / 10001)
    # split by whether a team has no World Cup record up to 2014 (flag from E1, which must be run first)
    flag = pd.read_csv(OUT / "e1_match_level_predictions.csv").query("model == 'base_rate'")[["year", "match", "unknown_team"]]
    split = (df.merge(flag, on=["year", "match"]).groupby(["model", "year", "unknown_team"]).log_loss.agg(["mean", "size"]).reset_index())
    by_year["log_loss_by_tournament_and_unrecorded_team"] = [
        {"model": r.model, "year": int(r.year), "team_without_record_to_2014": bool(r.unknown_team), "log_loss": round(float(r["mean"]), 4),
         "matches": int(r["size"])} for _, r in split.iterrows() if r.model in ("contest features, refit", "Elo (ordered logit)")]
    corr = float(np.corrcoef(F.EloDiff, F.EloIntl)[0, 1])
    summ = {"training_matches": int(len(F)), "elo_matched_share": matched, "corr_contest_internal_elo_vs_international_elo": corr, "C_refit": C4, "C_combined": C5,
            "selection_logloss_refit": ll4, "selection_logloss_combined": ll5, "learned_pool_weight_on_contest": w_star,
            "scores": tab.round(4).to_dict(orient="index"), "paired_bootstrap": boot, "by_tournament_vs_elo": by_year,
            "log_loss_by_tournament": df.groupby(["model", "year"]).log_loss.mean().unstack().round(4).to_dict(orient="index"),
            "draw_rate_observed": float((df[df.model == "Elo (ordered logit)"].y == 1).mean()),
            "mean_p_draw": df.groupby("model").p_draw.mean().round(4).to_dict()}
    (OUT / "e2_summary.json").write_text(json.dumps(summ, indent=1))
    print(tab.round(4).to_string()); print(json.dumps({k: summ[k] for k in summ if k not in ("scores",)}, indent=1))


if __name__ == "__main__":
    main()
