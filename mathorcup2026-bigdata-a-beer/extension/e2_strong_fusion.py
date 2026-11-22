"""Extension E2 — the contest's fusion design with strong branches, the style baseline, and an
independent check of the key-compound claim, all on the reference study's public data.

E1 showed that the contest's grouped convex fusion helps a little when its branches are kernel ridge
models (+0.004 over the better branch) and that Extra Trees is the strongest single learner under the
reference protocol (Schreurs et al., Nat. Commun. 15, 2368, 2024; macro R^2 0.231 over 10 splits, against
0.21 published for their tuned GBR).  E2 asks three questions.

(a) Does the fusion design set a new best when it is fed strong, diverse branches?  Branches:
    Extra Trees on the 50 standardised attributes, Extra Trees on 16 principal components of the
    attributes (the contest's low-rank idea), kernel ridge on the same 16 components, and a style prior
    (training mean of each attribute within the beer's style).  Non-negative weights summing to one are
    chosen per attribute group on inner 3-fold out-of-fold predictions of the training part only.
(b) How much of what is predictable is beer style?  A style-only model is scored next to the chemistry
    models, and the fusion's weight on the style prior is reported.
(c) Does 4-vinylguaiacol (4-VG) — the compound the contest paper found to stay in the within-style Top 3
    for the panel's phenolic (4VG) note in 100 % of resamples — also explain what *consumers* write?
    The reference study's RateBeer descriptor frequencies (Supplementary File S7, a different population
    from the trained panel) are related to the measured X4VG concentration within styles, next to the
    other clove/smoky phenols in the dataset (eugenol, 4-ethylguaiacol, 4-ethylphenol).

Protocol as in E1: 70/30 split stratified by style (the reference split random_state = 0 and ten further
seeds), X standardised on the training part, macro R^2 over the 50 trained-panel attributes.
Run:  python extension/e2_strong_fusion.py      (about 25 minutes on two cores)
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "data" / "external" / "schreurs2024_supplementary.xlsx"
OUT = ROOT / "results" / "extension"; OUT.mkdir(parents=True, exist_ok=True)
SPLITS = [0] + list(range(100, 110))


def load():
    s1 = pd.read_excel(XLSX, sheet_name="Supplementary File S1").set_index("beer")
    s4 = pd.read_excel(XLSX, sheet_name="Supplementary File S4").set_index("beer")
    s7 = pd.read_excel(XLSX, sheet_name="Supplementary File S7").set_index("beer")
    d = s1.join(s4.drop(columns=["beer_id", "tasting_category_fine"]))
    X = d.loc[:, "acetaldehyde":"sulfur_sum"].astype(float)
    X = X.fillna(X.median())
    Y = d.loc[:, "A_malt_all":"overall"].astype(float)
    return X, Y, d["tasting_category_fine"].to_numpy(), s7.reindex(d.index)


def groups_of(cols):
    return np.array([c.split("_")[0] if "_" in c else "other" for c in cols])


# ----------------------------------------------------------------------------- branches
class ET50:
    def fit(self, X, Y, style):
        self.sy = StandardScaler().fit(Y)
        self.m = ExtraTreesRegressor(500, min_samples_leaf=2, max_features=0.3, random_state=0, n_jobs=-1).fit(X, self.sy.transform(Y))
        return self
    def predict(self, X, style): return self.sy.inverse_transform(self.m.predict(X))


class LowRank:
    def __init__(self, learner, k=16): self.learner, self.k = learner, k
    def fit(self, X, Y, style):
        self.sy = StandardScaler().fit(Y); Z = self.sy.transform(Y)
        self.pca = PCA(self.k, random_state=0).fit(Z)
        if self.learner == "et":
            self.m = ExtraTreesRegressor(500, min_samples_leaf=2, max_features=0.3, random_state=0, n_jobs=-1)
        else:
            self.m = GridSearchCV(KernelRidge(kernel="rbf"), {"alpha": [0.1, 0.3, 1, 3], "gamma": [1e-3, 3e-3, 1e-2]}, cv=3, scoring="r2")
        self.m.fit(X, self.pca.transform(Z)); return self
    def predict(self, X, style):
        return self.sy.inverse_transform(self.pca.inverse_transform(self.m.predict(X)))


class StylePrior:
    """Training mean of each attribute within the beer's style (overall mean for unseen styles)."""
    def fit(self, X, Y, style):
        self.mu = Y.mean(0); self.by = {s: Y[style == s].mean(0) for s in np.unique(style)}; return self
    def predict(self, X, style): return np.vstack([self.by.get(s, self.mu) for s in style])


BRANCHES = {"ET, 50 outputs": ET50, "ET, 16-d low rank": lambda: LowRank("et"), "KRR, 16-d low rank": lambda: LowRank("krr"),
            "style prior": StylePrior}


class SimplexFusion:
    """Per attribute group, non-negative weights summing to one over the branches, fitted by least squares on
    inner 3-fold out-of-fold predictions (training part only)."""
    def __init__(self, groups, names): self.groups, self.names = groups, names
    def fit(self, X, Y, style):
        oof = {n: np.zeros_like(Y) for n in self.names}
        for tr, va in KFold(3, shuffle=True, random_state=0).split(X):
            for n in self.names:
                oof[n][va] = BRANCHES[n]().fit(X[tr], Y[tr], style[tr]).predict(X[va], style[va])
        self.w = {}
        for g in np.unique(self.groups):
            j = self.groups == g
            A = np.column_stack([oof[n][:, j].ravel() for n in self.names]); b = Y[:, j].ravel()
            A1 = np.vstack([A, 1e3 * np.ones(len(self.names))]); b1 = np.append(b, 1e3)      # sum-to-one penalty
            w, _ = nnls(A1, b1); self.w[g] = w / w.sum()
        self.models = {n: BRANCHES[n]().fit(X, Y, style) for n in self.names}
        return self
    def predict(self, X, style):
        P = {n: m.predict(X, style) for n, m in self.models.items()}; out = np.zeros_like(next(iter(P.values())))
        for g, w in self.w.items():
            j = self.groups == g
            out[:, j] = sum(wi * P[n][:, j] for wi, n in zip(w, self.names))
        return out


def main():
    Xdf, Ydf, style, s7 = load(); X, Y, cols = Xdf.to_numpy(), Ydf.to_numpy(), list(Ydf.columns); groups = groups_of(cols)
    chem = ["ET, 50 outputs", "ET, 16-d low rank", "KRR, 16-d low rank"]
    rows, weights = [], []
    for seed in SPLITS:
        tr, te = train_test_split(np.arange(len(X)), test_size=0.30, random_state=seed, shuffle=True, stratify=style)
        sx = StandardScaler().fit(X[tr]); Xtr, Xte = sx.transform(X[tr]), sx.transform(X[te])
        preds = {n: BRANCHES[n]().fit(Xtr, Y[tr], style[tr]).predict(Xte, style[te]) for n in BRANCHES}
        f_chem = SimplexFusion(groups, chem).fit(Xtr, Y[tr], style[tr]); preds["fusion of chemistry branches"] = f_chem.predict(Xte, style[te])
        f_all = SimplexFusion(groups, chem + ["style prior"]).fit(Xtr, Y[tr], style[tr]); preds["fusion incl. style prior"] = f_all.predict(Xte, style[te])
        for n, p in preds.items():
            rows.append({"split_seed": seed, "model": n, "macro_r2": r2_score(Y[te], p),
                         **{f"r2::{c}": r2_score(Y[te][:, j], p[:, j]) for j, c in enumerate(cols)}})
        for g, w in f_all.w.items():
            weights.append({"split_seed": seed, "group": g, **dict(zip(chem + ["style prior"], w))})
        print(seed, {r["model"][:18]: round(r["macro_r2"], 3) for r in rows if r["split_seed"] == seed}, flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "e2_split_scores.csv", index=False)
    pd.DataFrame(weights).to_csv(OUT / "e2_fusion_weights.csv", index=False)
    rep = df[df.split_seed != 0].pivot(index="split_seed", columns="model", values="macro_r2")
    e1 = pd.read_csv(OUT / "e1_split_scores.csv")
    e1rep = e1[e1.split_seed != 0].pivot(index="split_seed", columns="model", values="macro_r2")
    best = "fusion incl. style prior"
    d_et = rep[best] - rep["ET, 50 outputs"]; d_gbr = rep[best] - e1rep["GBR (reference study's best model)"]
    summ = {"reference_split_R2": df[df.split_seed == 0].set_index("model").macro_r2.round(4).to_dict(),
            "repeated_mean_R2": rep.mean().round(4).to_dict(), "repeated_sd_R2": rep.std().round(4).to_dict(),
            "best_minus_ET_mean": float(d_et.mean()), "best_minus_ET_se": float(d_et.std(ddof=1) / np.sqrt(len(d_et))),
            "best_beats_ET_in_splits": int((d_et > 0).sum()),
            "best_minus_GBR_rerun_mean": float(d_gbr.mean()), "best_beats_GBR_rerun_in_splits": int((d_gbr > 0).sum()),
            "mean_weight_style_prior": float(pd.DataFrame(weights)["style prior"].mean())}
    # per-attribute: where does chemistry add to style?
    pa = df[df.split_seed != 0].groupby("model")[[f"r2::{c}" for c in cols]].mean().T
    pa.index = cols; pa.to_csv(OUT / "e2_per_attribute_r2.csv")
    gain = pa[best] - pa["style prior"]
    summ["attributes_where_chemistry_adds_over_style_gt_0.05"] = gain[gain > 0.05].round(3).sort_values(ascending=False).to_dict()
    summ["attributes_best_r2_top5"] = pa[best].sort_values(ascending=False).head(5).round(3).to_dict()

    # (c) key compound vs consumers' clove descriptors, within style
    cand = ["X4VG", "eugenol", "X4.ethyl.guaiacol", "X4.ethyl.phenol"]
    lab = {"X4VG": "4-vinylguaiacol", "eugenol": "eugenol", "X4.ethyl.guaiacol": "4-ethylguaiacol", "X4.ethyl.phenol": "4-ethylphenol"}
    targets = {"panel: 4VG note": Ydf["X4vg"], "panel: clove": Ydf["clove"],
               "consumers: clove aroma": s7["aroma_clove"], "consumers: clove taste": s7["taste_clove"]}
    rng = np.random.default_rng(0)
    def within(a, b):                                       # Spearman after removing style medians of the ranks
        ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
        df_ = pd.DataFrame({"a": ra, "b": rb, "s": style})
        df_["a"] -= df_.groupby("s").a.transform("mean"); df_["b"] -= df_.groupby("s").b.transform("mean")
        return float(np.corrcoef(df_.a, df_.b)[0, 1])
    kc = []
    for tname, tv in targets.items():
        for c in cand:
            a, b = Xdf[c].to_numpy(), tv.to_numpy(); ok = ~np.isnan(b)
            r_all = float(spearmanr(a[ok], b[ok])[0]); r_w = within(a[ok], b[ok]) if ok.all() else np.nan
            boots = []
            for _ in range(2000):
                i = rng.integers(0, ok.sum(), ok.sum())
                aa, bb, ss = a[ok][i], b[ok][i], style[ok][i]
                ra = pd.Series(aa).rank().to_numpy(); rb = pd.Series(bb).rank().to_numpy()
                t_ = pd.DataFrame({"a": ra, "b": rb, "s": ss}); t_["a"] -= t_.groupby("s").a.transform("mean"); t_["b"] -= t_.groupby("s").b.transform("mean")
                boots.append(np.corrcoef(t_.a, t_.b)[0, 1])
            kc.append({"target": tname, "compound": lab[c], "spearman_overall": r_all, "spearman_within_style": r_w,
                       "within_lo95": float(np.nanpercentile(boots, 2.5)), "within_hi95": float(np.nanpercentile(boots, 97.5))})
    kc = pd.DataFrame(kc); kc.to_csv(OUT / "e2_key_compound_consumers.csv", index=False)
    summ["key_compound_within_style"] = {f"{r.target} ~ {r.compound}": [round(r.spearman_within_style, 3), round(r.within_lo95, 3), round(r.within_hi95, 3)]
                                         for r in kc.itertuples()}
    (OUT / "e2_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
