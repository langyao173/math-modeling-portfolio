"""Extension E5 — the Problem 5 tolerance analysis with the surrogate's own uncertainty included.

Problem 5 propagates manufacturing tolerances (conditional truncated normals inside the design box, sigma 2 % of
beta and 1.5 % of gamma, N fixed) through the paper's GPR surrogate at the recommended design (0.22, 4.5, 4) and
reports, e.g., a coefficient of variation of 1.01 % and a worst shift of 6.24 % for the pressure drop.  Those
figures treat the surrogate as the truth.  E1 showed that the GPR intervals do not transfer to unseen levels and
that a cubic-trend universal-kriging (UK) model is calibrated; E2 used its posterior for the decision.  E5 repeats
the tolerance analysis on 1 000 joint posterior draws of the UK surfaces:

  * for each draw, the tolerance statistics of the paper (mean shift, CV, 95th-percentile shift and worst shift
    relative to that draw's nominal value) on the same truncated-normal tolerance distribution, and the first-
    order / total Sobol indices of beta and gamma (Jansen estimators);
  * the spread of the nominal performance across draws, i.e. what the 84 samples leave unknown about the design
    itself, set against the spread caused by the tolerances.

Every draw is evaluated exactly at the Monte Carlo points (joint posterior over 4 001 inputs: 2 000 tolerance
samples used as the A/B matrices of the Sobol estimators, their 2 000 column-swapped copies, and the nominal
design); no interpolation.
Run:  python extension/e5_tolerance_posterior.py      (about 2 minutes)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import truncnorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uk import UniversalKriging                      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "extension"; OUT.mkdir(parents=True, exist_ok=True)
INPUTS, TARGETS = ["beta", "gamma", "N"], ["R", "dP", "phi"]
X_STAR = np.array([0.22, 4.5, 4.0])
SIG_B, SIG_G = 0.02, 0.015                           # relative tolerances of the paper (config.py)
BOX = dict(beta=(0.1, 0.3), gamma=(3.0, 4.5))
N_MC, DRAWS, SEED = 2000, 1000, 20260612
PAPER = {"R": dict(cv_pct=0.091, p95_shift_pct=0.303, worst_shift_pct=0.504),            # p5_results.json (GPR)
         "dP": dict(cv_pct=1.014, p95_shift_pct=3.018, worst_shift_pct=6.24),
         "phi": dict(cv_pct=0.09, p95_shift_pct=0.306, worst_shift_pct=0.495)}


def load():
    return pd.read_csv(ROOT / "data" / "attachment2.tsv", sep="\t", skiprows=2,
                       names=["id", *INPUTS, *TARGETS], encoding="utf-8-sig").dropna()


def tolerance_sample(n, rng):
    def tn(mu, sd, lo, hi):
        return truncnorm.rvs((lo - mu) / sd, (hi - mu) / sd, loc=mu, scale=sd, size=n, random_state=rng)
    b = tn(X_STAR[0], SIG_B * X_STAR[0], *BOX["beta"]); g = tn(X_STAR[1], SIG_G * X_STAR[1], *BOX["gamma"])
    return np.column_stack([b, g, np.full(n, X_STAR[2])])


def joint_draws(model, P, n_draws, rng):
    mu, cov = model.predict(P, return_cov=True)
    w, V = np.linalg.eigh(0.5 * (cov + cov.T))
    w = np.clip(w, 0, None)
    return mu[None, :] + (rng.standard_normal((n_draws, len(w))) * np.sqrt(w)) @ V.T, mu


def tol_stats(f_tol, f0):
    return dict(mean_shift_pct=100 * (f_tol.mean() - f0) / f0, cv_pct=100 * f_tol.std() / f_tol.mean(),
                p95_shift_pct=100 * (np.percentile(f_tol, 95) - f0) / f0, worst_shift_pct=100 * (f_tol.max() - f0) / f0)


def sobol(fA, fB, fAB):
    """Jansen first-order and total indices; fAB[j] = f at A with column j taken from B."""
    mu = 0.5 * (fA.mean() + fB.mean())                    # centring is essential: the variance is ~1e-7 of the mean
    fA, fB, fAB = fA - mu, fB - mu, [f - mu for f in fAB]
    V = np.var(np.r_[fA, fB], ddof=1)
    S1 = [np.mean(fB * (fAB[j] - fA)) / V for j in range(2)]
    ST = [0.5 * np.mean((fA - fAB[j]) ** 2) / V for j in range(2)]
    return S1, ST


def main():
    rng = np.random.default_rng(SEED)
    df = load(); X = df[INPUTS].to_numpy(float)
    T = tolerance_sample(N_MC, np.random.default_rng(0))
    A, B = T[: N_MC // 2], T[N_MC // 2:]
    AB = []
    for j in range(2):
        M = A.copy(); M[:, j] = B[:, j]; AB.append(M)
    P = np.vstack([A, B, AB[0], AB[1], X_STAR[None, :]])
    n2 = N_MC // 2
    rows, summ = [], {"draws": DRAWS, "mc_points": N_MC, "paper_gpr_plugin": PAPER, "targets": {}}
    for t in TARGETS:
        model = UniversalKriging().fit(X, df[t].to_numpy(float))
        D, mu = joint_draws(model, P, DRAWS, rng)
        mean_stats = tol_stats(mu[:N_MC], mu[-1])                          # UK posterior mean used as a plug-in surface
        S1m, STm = sobol(mu[:n2], mu[n2:N_MC], [mu[N_MC:N_MC + n2], mu[N_MC + n2:2 * N_MC]])
        per = []
        for d in D:
            st = tol_stats(d[:N_MC], d[-1])
            S1, ST = sobol(d[:n2], d[n2:N_MC], [d[N_MC:N_MC + n2], d[N_MC + n2:2 * N_MC]])
            st.update(nominal=d[-1], tol_sd=d[:N_MC].std(), S1_beta=S1[0], S1_gamma=S1[1], ST_beta=ST[0], ST_gamma=ST[1])
            per.append(st)
        per = pd.DataFrame(per); per.insert(0, "target", t); rows.append(per)
        q = lambda c: [float(np.percentile(per[c], 2.5)), float(np.percentile(per[c], 50)), float(np.percentile(per[c], 97.5))]
        nominal_sd = float(per.nominal.std())
        summ["targets"][t] = {
            "uk_plugin": {k: float(v) for k, v in mean_stats.items()} | {"S1_beta": float(S1m[0]), "S1_gamma": float(S1m[1])},
            "posterior_2.5_50_97.5": {c: q(c) for c in ("mean_shift_pct", "cv_pct", "p95_shift_pct", "worst_shift_pct", "S1_beta", "S1_gamma")},
            "nominal_posterior_mean": float(per.nominal.mean()), "nominal_posterior_sd": nominal_sd,
            "nominal_posterior_sd_pct": 100 * nominal_sd / float(per.nominal.mean()),
            "tolerance_sd_median": float(per.tol_sd.median()),
            "ratio_surrogate_sd_to_tolerance_sd": nominal_sd / float(per.tol_sd.median()),
            "share_draws_gamma_dominant": float(np.mean(per.S1_gamma > per.S1_beta)),
        }
        print(t, json.dumps(summ["targets"][t], indent=0)[:600], flush=True)
    pd.concat(rows).to_csv(OUT / "e5_tolerance_posterior_draws.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    (OUT / "e5_tolerance_posterior_summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
