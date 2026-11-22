"""Extension E2/E3 — decision stability under surrogate (epistemic) uncertainty,
and a variance-reduction plan for the next CFD runs.

The paper tests robustness to *preferences* (Problem 4) and to *manufacturing /
operating perturbations* (Problem 5).  A third source is left open: the
surrogate itself is fitted to 84 CFD samples, and the recommended design lies
on the gamma boundary and between sampled beta levels.  Here the GPR posterior
is treated as a distribution over response surfaces.  For each joint posterior
draw over a candidate grid we recompute the Pareto set and the three
compromise rules used in the paper (equal-weight ideal point, entropy-TOPSIS,
Chebyshev).  This yields
  * P_pareto(x): posterior probability that design x is non-dominated;
  * P_win(x):    posterior probability that x is selected by each rule.
E3 then picks K new simulation points that maximally reduce posterior variance
over the decision-relevant region (designs with P_pareto in (0.05, 0.95) or
P_win > 0); GP posterior variance does not depend on future observations, so
the plan is computable before any new CFD run.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as Ck, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from uk import UniversalKriging

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "extension"; OUT.mkdir(parents=True, exist_ok=True)
INPUTS, TARGETS = ["beta", "gamma", "N"], ["R", "dP", "phi"]
S, K_NEXT, SEED = 1000, 5, 20260612
MODEL = sys.argv[1] if len(sys.argv) > 1 else "uk"   # "uk" (main) or "gpr" (paper surrogate)
X_STAR = np.array([0.22, 4.5, 4.0])


def load():
    return pd.read_csv(ROOT / "data" / "attachment2.tsv", sep="\t", skiprows=2,
                       names=["id", *INPUTS, *TARGETS], encoding="utf-8-sig").dropna()


def pareto_mask(F):
    """Non-dominated mask (all objectives minimised); sort-and-sweep, O(n * |front|)."""
    order = np.lexsort(F.T[::-1])
    front = np.empty((0, F.shape[1])); keep = np.zeros(len(F), bool)
    for i in order:
        f = F[i]
        if front.size and ((front <= f).all(1) & (front < f).any(1)).any():
            continue
        keep[i] = True
        front = np.vstack([front, f])
    return keep


def entropy_weights(F):
    B = (F.max(0) - F) / (F.max(0) - F.min(0) + 1e-12)
    P = B / (B.sum(0) + 1e-12)
    e = -(P * np.log(np.where(P > 0, P, 1.0))).sum(0) / np.log(len(F))
    d = 1 - e
    return d / d.sum()


def rules(F_front):
    Z = (F_front - F_front.min(0)) / (np.ptp(F_front, 0) + 1e-12)
    w = entropy_weights(F_front)
    V = F_front / np.sqrt((F_front ** 2).sum(0)) * w
    dp = np.sqrt(((V - V.min(0)) ** 2).sum(1)); dn = np.sqrt(((V - V.max(0)) ** 2).sum(1))
    return {"ideal": int(np.argmin(np.sqrt((Z ** 2).sum(1)))),
            "topsis": int(np.argmax(dn / (dp + dn + 1e-12))),
            "chebyshev": int(np.argmin(Z.max(1)))}


def main():
    rng = np.random.default_rng(SEED)
    df = load(); X = df[INPUTS].to_numpy(float)
    betas = np.round(np.arange(0.10, 0.3001, 0.01), 3)
    gammas = np.round(np.arange(3.0, 4.5001, 0.05), 3)
    Ns = np.array([2, 4, 6, 8, 10.0])
    G = np.array([[b, g, n] for n in Ns for b in betas for g in gammas])
    istar = int(np.argmin(np.abs(G - X_STAR).sum(1)))
    sc = StandardScaler().fit(X); Xs, Gs = sc.transform(X), sc.transform(G)

    draws, covs, noise = [], {}, {}
    for t in TARGETS:
        y = df[t].to_numpy()
        if MODEL == "uk":
            m = UniversalKriging().fit(X, y)
            mu, cov = m.predict(G, return_cov=True)
            noise[t] = m.kernel.k2.noise_level * m.ys ** 2
        else:
            k = Ck(1.0, (1e-3, 1e3)) * Matern([1.0] * 3, (1e-2, 1e3), nu=2.5) + WhiteKernel(1e-6, (1e-10, 1e-2))
            g = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=5, random_state=0).fit(Xs, y)
            mu, cov = g.predict(Gs, return_cov=True)
            noise[t] = g.kernel_.k2.noise_level * g._y_train_std ** 2
        cov = 0.5 * (cov + cov.T)
        w, V = np.linalg.eigh(cov); w = np.clip(w, 0, None)
        draws.append(mu[None, :] + rng.standard_normal((S, len(G))) @ (V * np.sqrt(w)).T)
        covs[t] = cov
    D = np.stack(draws, -1)                               # (S, n, 3)

    pareto_cnt = np.zeros(len(G)); win = {r: np.zeros(len(G)) for r in ["ideal", "topsis", "chebyshev"]}
    for s in range(S):
        m = pareto_mask(D[s]); idx = np.flatnonzero(m)
        pareto_cnt[m] += 1
        for r, i in rules(D[s][idx]).items():
            win[r][idx[i]] += 1
    P_par = pareto_cnt / S; P_win = {r: v / S for r, v in win.items()}

    near = (np.abs(G[:, 0] - X_STAR[0]) <= 0.0101) & (np.abs(G[:, 1] - X_STAR[1]) <= 0.0501) & (G[:, 2] == 4)
    out = pd.DataFrame(G, columns=INPUTS)
    out["P_pareto"] = P_par
    for r in P_win: out[f"P_win_{r}"] = P_win[r]
    for j, t in enumerate(TARGETS):
        out[f"{t}_mean"] = D[:, :, j].mean(0); out[f"{t}_sd"] = D[:, :, j].std(0)
    out.to_csv(OUT / f"e2_posterior_decision_grid_{MODEL}.csv", index=False)

    summary = {"model": MODEL, "posterior_draws": S, "grid_size": len(G),
               "x_star": X_STAR.tolist(), "P_pareto_x_star": float(P_par[istar]),
               "P_win_x_star": {r: float(P_win[r][istar]) for r in P_win},
               "P_win_neighbourhood_x_star(|dbeta|<=0.01,|dgamma|<=0.05,N=4)": {r: float(P_win[r][near].sum()) for r in P_win},
               "P_win_by_N": {r: {int(n): float(P_win[r][G[:, 2] == n].sum()) for n in Ns} for r in P_win},
               "P_win_gamma_at_upper_bound": {r: float(P_win[r][G[:, 1] == 4.5].sum()) for r in P_win},
               "modal_winner": {r: dict(zip(INPUTS, G[int(np.argmax(P_win[r]))].tolist()), prob=float(P_win[r].max())) for r in P_win},
               "expected_pareto_set_size": float(P_par.sum())}

    # ---- E3: greedy variance-reduction plan over the decision-relevant region ----
    A = np.flatnonzero(((P_par > 0.05) & (P_par < 0.95)) | (sum(P_win.values()) > 0))
    C = {t: covs[t].copy() for t in TARGETS}
    base = {t: np.trace(C[t][np.ix_(A, A)]) for t in TARGETS}
    chosen, trace = [], []
    for _ in range(K_NEXT):
        val = np.zeros(len(G))
        for t in TARGETS:
            c = C[t]; tr = np.trace(c[np.ix_(A, A)])
            red = (c[A, :] ** 2).sum(0) / (np.diag(c) + noise[t])
            val += (tr - red) / base[t]
        val[chosen] = np.inf
        best = int(np.argmin(val)); best_val = float(val[best])
        chosen.append(best)
        for t in TARGETS:
            c = C[t]; kj = c[:, best].copy()
            C[t] = c - np.outer(kj, kj) / (c[best, best] + noise[t])
        trace.append(dict(step=len(chosen), beta=float(G[best, 0]), gamma=float(G[best, 1]), N=float(G[best, 2]),
                          remaining_variance_fraction=best_val / len(TARGETS)))
    plan = pd.DataFrame(trace); plan.to_csv(OUT / f"e3_next_simulations_{MODEL}.csv", index=False)
    summary["decision_region_size"] = int(len(A))
    summary["next_simulations"] = trace
    (OUT / f"e2_summary_{MODEL}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
