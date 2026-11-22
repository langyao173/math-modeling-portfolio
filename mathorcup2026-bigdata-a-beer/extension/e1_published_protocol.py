"""Extension E1 — the dual-representation fusion idea under the protocol of the
published reference study (Schreurs et al., Nat. Commun. 15, 2368, 2024).

Data: Supplementary Files S1 (231 log-chemical features) and S4 (50 trained-panel
attributes) of the reference study, Zenodo 10653704, CC BY 4.0 — the same 250
beers that the contest attachments were drawn from.
Protocol (copied from the study's training script): 70/30 split stratified by
`tasting_category_fine`, X standardised on the training part, macro R^2 over the
50 attributes (sklearn r2_score, uniform average).  The study's best model,
multi-output gradient boosting, reports R^2 = 0.21 on its split.
Here every model is scored on the study's split (random_state=0) and on 10
further stratified splits.  The fusion is a foundation-model-free
re-implementation of the contest paper's design: a low-rank branch (50 targets ->
16 principal components -> predict -> decode) and an independent 50-output branch,
combined per attribute group with convex weights chosen by inner 3-fold
out-of-fold R^2 on the training part only.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "data" / "external" / "schreurs2024_supplementary.xlsx"
OUT = ROOT / "results" / "extension"; OUT.mkdir(parents=True, exist_ok=True)
SPLITS = [0] + list(range(100, 110))


def load():
    s1 = pd.read_excel(XLSX, sheet_name="Supplementary File S1").set_index("beer")
    s4 = pd.read_excel(XLSX, sheet_name="Supplementary File S4").set_index("beer")
    d = s1.join(s4.drop(columns=["beer_id", "tasting_category_fine"]))
    X = d.loc[:, "acetaldehyde":"sulfur_sum"].astype(float)
    X = X.fillna(X.median())
    Y = d.loc[:, "A_malt_all":"overall"].astype(float)
    return X.to_numpy(), Y.to_numpy(), d["tasting_category_fine"].to_numpy(), list(Y.columns)


def krr():
    return GridSearchCV(KernelRidge(kernel="rbf"), {"alpha": [0.1, 0.3, 1, 3], "gamma": [1e-3, 3e-3, 1e-2]},
                        cv=3, scoring="r2")


class LowRank:
    """Targets -> k principal components -> regressor -> decode."""
    def __init__(self, k=16): self.k = k
    def fit(self, X, Y):
        self.sy = StandardScaler().fit(Y); Z = self.sy.transform(Y)
        self.pca = PCA(self.k, random_state=0).fit(Z)
        self.m = krr().fit(X, self.pca.transform(Z)); return self
    def predict(self, X):
        return self.sy.inverse_transform(self.pca.inverse_transform(self.m.predict(X)))


class Independent:
    def fit(self, X, Y):
        self.sy = StandardScaler().fit(Y)
        self.m = krr().fit(X, self.sy.transform(Y)); return self
    def predict(self, X): return self.sy.inverse_transform(self.m.predict(X))


def group_of(cols):
    return np.array([c.split("_")[0] if "_" in c else "overall" for c in cols])


class GroupFusion:
    """Convex fusion of LowRank and Independent with one weight per attribute group,
    chosen by inner 3-fold OOF macro R^2 on the training data."""
    def __init__(self, groups): self.groups = groups
    def fit(self, X, Y):
        oof_a, oof_b = np.zeros_like(Y), np.zeros_like(Y)
        for tr, va in KFold(3, shuffle=True, random_state=0).split(X):
            oof_a[va] = LowRank().fit(X[tr], Y[tr]).predict(X[va])
            oof_b[va] = Independent().fit(X[tr], Y[tr]).predict(X[va])
        self.w = {}
        for g in np.unique(self.groups):
            j = self.groups == g
            scores = [(r2_score(Y[:, j], w * oof_a[:, j] + (1 - w) * oof_b[:, j]), w) for w in np.linspace(0, 1, 21)]
            self.w[g] = max(scores)[1]
        self.a, self.b = LowRank().fit(X, Y), Independent().fit(X, Y); return self
    def predict(self, X):
        pa, pb = self.a.predict(X), self.b.predict(X); out = np.empty_like(pa)
        for g, w in self.w.items():
            j = self.groups == g; out[:, j] = w * pa[:, j] + (1 - w) * pb[:, j]
        return out


def models(groups):
    return {
        "GBR (reference study's best model)": lambda: MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)),
        "Ridge": lambda: GridSearchCV(Ridge(), {"alpha": np.logspace(0, 5, 11)}, cv=5, scoring="r2"),
        "Extra Trees": lambda: ExtraTreesRegressor(500, min_samples_leaf=2, max_features=0.3, random_state=0, n_jobs=-1),
        "KRR-RBF, 50 independent outputs": Independent,
        "KRR-RBF, 16-d low-rank targets": LowRank,
        "Grouped convex fusion (contest design)": lambda: GroupFusion(groups),
    }


def main():
    X, Y, style, cols = load(); groups = group_of(cols)
    rows, per_attr = [], []
    for seed in SPLITS:
        tr, te = train_test_split(np.arange(len(X)), test_size=0.30, random_state=seed, shuffle=True, stratify=style)
        sx = StandardScaler().fit(X[tr]); Xtr, Xte = sx.transform(X[tr]), sx.transform(X[te])
        for name, make in models(groups).items():
            p = make().fit(Xtr, Y[tr]).predict(Xte)
            rows.append({"split_seed": seed, "model": name, "macro_r2": r2_score(Y[te], p)})
            if seed == 0:
                per_attr += [{"model": name, "attribute": c, "r2": r2_score(Y[te][:, j], p[:, j])} for j, c in enumerate(cols)]
        print(seed, {r["model"][:12]: round(r["macro_r2"], 3) for r in rows if r["split_seed"] == seed}, flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "e1_split_scores.csv", index=False)
    pd.DataFrame(per_attr).to_csv(OUT / "e1_per_attribute_reference_split.csv", index=False)
    ref = df[df.split_seed == 0].set_index("model").macro_r2
    rep = df[df.split_seed != 0].pivot(index="split_seed", columns="model", values="macro_r2")
    fus = "Grouped convex fusion (contest design)"; gbr = "GBR (reference study's best model)"
    d = rep[fus] - rep[gbr]
    summary = {"published_GBR_R2_reference_split": 0.21,
               "reference_split_R2": ref.round(4).to_dict(),
               "repeated_splits": int(len(rep)),
               "repeated_mean_R2": rep.mean().round(4).to_dict(),
               "repeated_sd_R2": rep.std().round(4).to_dict(),
               "fusion_minus_GBR_mean": float(d.mean()), "fusion_beats_GBR_in_splits": int((d > 0).sum()),
               "fusion_minus_best_single_branch_mean": float((rep[fus] - rep[["KRR-RBF, 50 independent outputs", "KRR-RBF, 16-d low-rank targets"]].max(axis=1)).mean())}
    (OUT / "e1_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
