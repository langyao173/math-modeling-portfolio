"""Extension E1 — leave-one-level-out (LOLO) validation of the surrogates.

LOOCV on a full-factorial-like design is optimistic: when one point is removed,
its neighbours on the same grid line remain in the training set.  The design
recommended in the paper, (beta, gamma, N) = (0.22, 4.5, 4), sits on the upper
gamma boundary and inside the 0.20-0.30 beta gap, so the relevant question is
how well the surrogates predict a *whole unseen level*.  We hold out every
level of beta, gamma and N in turn (the 4 pin-free base cases beta=N=0 always
stay in training) and refit RSM-3 and GPR-Matern on the remainder.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as Ck, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from uk import UniversalKriging

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "extension"
OUT.mkdir(parents=True, exist_ok=True)
INPUTS, TARGETS = ["beta", "gamma", "N"], ["R", "dP", "phi"]


def load() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "attachment2.tsv", sep="\t", skiprows=2,
                       names=["id", *INPUTS, *TARGETS], encoding="utf-8-sig").dropna()


def make_gpr(seed: int = 0) -> GaussianProcessRegressor:
    k = Ck(1.0, (1e-3, 1e3)) * Matern([1.0] * 3, (1e-2, 1e3), nu=2.5) + WhiteKernel(1e-6, (1e-10, 1e-2))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=seed)


def make_rsm3() -> Pipeline:
    return Pipeline([("s", StandardScaler()), ("p", PolynomialFeatures(3)), ("r", LinearRegression())])


def main() -> None:
    df = load()
    X, rows = df[INPUTS].to_numpy(float), []
    pinned = (df.beta > 0).to_numpy()
    for var in INPUTS:
        for lev in sorted(df.loc[pinned, var].unique()):
            test = pinned & np.isclose(df[var].to_numpy(), lev)
            train = ~test
            for t in TARGETS:
                y = df[t].to_numpy(float)
                sc = StandardScaler().fit(X[train])
                g = make_gpr().fit(sc.transform(X[train]), y[train])
                mu, sd = g.predict(sc.transform(X[test]), return_std=True)
                r = make_rsm3().fit(X[train], y[train]).predict(X[test])
                uk_mu, uk_sd = UniversalKriging().fit(X[train], y[train]).predict(X[test], return_std=True)
                yt = y[test]
                band = {"GPR-Matern": (mu - 1.96 * sd, mu + 1.96 * sd),
                        "UK-cubic+Matern": (uk_mu - 1.96 * uk_sd, uk_mu + 1.96 * uk_sd)}
                for name, p in [("GPR-Matern", mu), ("RSM-3", r), ("UK-cubic+Matern", uk_mu)]:
                    rows.append(dict(
                        held_out=var, level=lev, target=t, model=name, n_test=int(test.sum()),
                        rmse=float(np.sqrt(np.mean((p - yt) ** 2))),
                        max_rel_err_pct=float(100 * np.max(np.abs(p - yt) / np.abs(yt))),
                        rmse_over_range_pct=float(100 * np.sqrt(np.mean((p - yt) ** 2)) / np.ptp(y)),
                        pi95_coverage=(float(np.mean((yt >= band[name][0]) & (yt <= band[name][1])))
                                       if name in band else np.nan),
                    ))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "e1_level_holdout.csv", index=False)
    # Extrapolation (boundary levels) vs interpolation (interior levels)
    def kind(r):
        lv = sorted(df.loc[pinned, r.held_out].unique())
        return "boundary" if r.level in (lv[0], lv[-1]) else "interior"
    res["kind"] = res.apply(kind, axis=1)
    summ = (res.groupby(["model", "target", "kind"])[["rmse_over_range_pct", "max_rel_err_pct", "pi95_coverage"]]
            .mean().round(3))
    summ.to_csv(OUT / "e1_level_holdout_summary.csv")
    focus = res[((res.held_out == "gamma") & (res.level == 4.5)) | ((res.held_out == "beta") & (res.level.isin([0.2, 0.3])))]
    focus.to_csv(OUT / "e1_focus_recommended_region.csv", index=False)
    print(summ.to_string()); print(focus.round(4).to_string())


if __name__ == "__main__":
    main()
