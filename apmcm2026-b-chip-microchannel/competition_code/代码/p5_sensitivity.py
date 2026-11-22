# -*- coding: utf-8 -*-
"""
Problem 5 audit-fixed sensitivity and stability analysis.

Fixes:
- gamma tolerance at design_star=(0.22, 4.5, 4) uses a conditional truncated
  normal within the validated design domain [3.0, 4.5];
- no implicit clipping is used for the reported Monte Carlo distribution;
- Sobol indices are estimated with multiple seeds and confidence intervals.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from config import (
    CODE_DIR,
    INPUTS,
    N_levels,
    SIGMA_BETA_REL,
    SIGMA_GAMMA_REL,
    TARGETS,
    beta_max,
    beta_min,
    design_star,
    domain_clip,
    gamma_max,
    gamma_min,
    mc_n,
    seed_list,
    sobol_n_base,
)

M = joblib.load(CODE_DIR / "p2_gpr_models.pkl")
scaler = M["scalerX"]
x_star = np.array(design_star, dtype=float)


def predict(X, kind="gpr"):
    X = np.atleast_2d(np.asarray(X, float))
    if kind == "gpr":
        Xs = scaler.transform(X)
        return np.column_stack([M[t].predict(Xs) for t in TARGETS])
    return np.column_stack([M["rsm3_" + t].predict(X) for t in TARGETS])


def conditional_truncnorm(mu, sigma, lo, hi, n, rng):
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n, random_state=rng)


def symmetric_3sigma_wrong(mu, sigma, n, rng):
    x = rng.normal(mu, sigma, n)
    return np.clip(x, mu - 3 * sigma, mu + 3 * sigma)


def draw_domain_tolerance(x0, n, seed):
    rng = np.random.default_rng(seed)
    b = conditional_truncnorm(
        x0[0], SIGMA_BETA_REL * x0[0], beta_min, beta_max, n, rng
    )
    g = conditional_truncnorm(
        x0[1], SIGMA_GAMMA_REL * x0[1], gamma_min, gamma_max, n, rng
    )
    return np.column_stack([b, g, np.full(n, x0[2])])


def sample_domain_audit(Xs):
    outside = (
        (Xs[:, 0] < beta_min)
        | (Xs[:, 0] > beta_max)
        | (Xs[:, 1] < gamma_min)
        | (Xs[:, 1] > gamma_max)
        | ~np.isin(Xs[:, 2], N_levels)
    )
    return {
        "beta_min_sample": float(Xs[:, 0].min()),
        "beta_max_sample": float(Xs[:, 0].max()),
        "gamma_min_sample": float(Xs[:, 1].min()),
        "gamma_max_sample": float(Xs[:, 1].max()),
        "outside_domain_count": int(outside.sum()),
        "distribution_type": "conditional_truncated_normal",
        "truncation_is_asymmetric": bool(np.isclose(x_star[1], gamma_max)),
        "domain_clip": bool(domain_clip),
    }


def elasticity(x0, h=0.01):
    E = np.zeros((3, 3))
    f0 = predict(x0)[0]
    for j in range(3):
        dx = np.zeros(3)
        dx[j] = h * x0[j]
        xp, xm = x0 + dx, x0 - dx
        if j == 1 and xp[1] > gamma_max:
            fp, fm = predict(x0)[0], predict(xm)[0]
            E[:, j] = (fp - fm) / (h * x0[j]) * x0[j] / f0
        else:
            fp, fm = predict(xp)[0], predict(xm)[0]
            E[:, j] = (fp - fm) / (2 * h * x0[j]) * x0[j] / f0
    return E, f0


def sobol_once(sample_fn, n, seed):
    rng = np.random.default_rng(seed)
    A, B = sample_fn(n, rng), sample_fn(n, rng)
    fA, fB = predict(A), predict(B)
    mu = 0.5 * (fA.mean(0) + fB.mean(0))
    fA, fB = fA - mu, fB - mu
    V = fA.var(0, ddof=1)
    S1 = np.zeros((3, 3))
    ST = np.zeros((3, 3))
    for j in range(3):
        ABj = A.copy()
        ABj[:, j] = B[:, j]
        fABj = predict(ABj) - mu
        S1[:, j] = np.mean(fB * (fABj - fA), 0) / V
        ST[:, j] = 0.5 * np.mean((fA - fABj) ** 2, 0) / V
    return np.clip(S1, 0, None), ST


def sample_full(n, rng):
    b = rng.uniform(beta_min, beta_max, n)
    g = rng.uniform(gamma_min, gamma_max, n)
    N = rng.choice(N_levels, n).astype(float)
    return np.column_stack([b, g, N])


def sample_tol(n, rng):
    b = conditional_truncnorm(
        x_star[0], SIGMA_BETA_REL * x_star[0], beta_min, beta_max, n, rng
    )
    g = conditional_truncnorm(
        x_star[1], SIGMA_GAMMA_REL * x_star[1], gamma_min, gamma_max, n, rng
    )
    return np.column_stack([b, g, np.full(n, x_star[2])])


def sobol_replicates(sample_fn, domain_name):
    rows = []
    for seed in seed_list:
        S1, ST = sobol_once(sample_fn, sobol_n_base, seed)
        for i, target in enumerate(TARGETS):
            for j, var in enumerate(INPUTS):
                rows.append(
                    {
                        "domain": domain_name,
                        "seed": seed,
                        "target": target,
                        "variable": var,
                        "S1": float(S1[i, j]),
                        "ST": float(ST[i, j]),
                        "ST_minus_S1": float(ST[i, j] - S1[i, j]),
                    }
                )
        print(f"Sobol {domain_name} seed={seed} done")
    return pd.DataFrame(rows)


def summarize_sobol(df_rep):
    summary = {}
    table_rows = []
    for (domain, target, variable), g in df_rep.groupby(["domain", "target", "variable"]):
        rec = {}
        for metric in ["S1", "ST"]:
            vals = g[metric].to_numpy(float)
            mean = float(vals.mean())
            sd = float(vals.std(ddof=1))
            se = sd / np.sqrt(len(vals))
            ci = (mean - 1.96 * se, mean + 1.96 * se)
            rec[f"{metric}_mean"] = mean
            rec[f"{metric}_sd"] = sd
            rec[f"{metric}_ci95"] = [float(ci[0]), float(ci[1])]
        rec["violation_rate_ST_lt_S1"] = float(np.mean(g["ST"] < g["S1"]))
        rec["ST_display"] = max(rec["ST_mean"], rec["S1_mean"])
        rec["ST_display_adjusted_for_theory"] = bool(rec["ST_mean"] < rec["S1_mean"])
        summary.setdefault(domain, {}).setdefault(target, {})[variable] = rec
        table_rows.append(
            {
                "domain": domain,
                "target": target,
                "variable": variable,
                "S1_mean": rec["S1_mean"],
                "S1_ci95_low": rec["S1_ci95"][0],
                "S1_ci95_high": rec["S1_ci95"][1],
                "ST_mean_raw": rec["ST_mean"],
                "ST_display": rec["ST_display"],
                "ST_ci95_low": rec["ST_ci95"][0],
                "ST_ci95_high": rec["ST_ci95"][1],
                "violation_rate_ST_lt_S1": rec["violation_rate_ST_lt_S1"],
                "ST_display_adjusted_for_theory": rec[
                    "ST_display_adjusted_for_theory"
                ],
            }
        )
    return summary, pd.DataFrame(table_rows)


def mc_statistics(x0, n=mc_n, seed=0, kind="gpr"):
    Xmc = draw_domain_tolerance(np.asarray(x0, float), n, seed)
    F = predict(Xmc, kind)
    f0 = predict(x0, kind)[0]
    st = {}
    for k, target in enumerate(TARGETS):
        y = F[:, k]
        st[target] = {
            "nominal": round(float(f0[k]), 5),
            "mean": round(float(y.mean()), 5),
            "mean_shift_pct": round(100 * float((y.mean() - f0[k]) / f0[k]), 3),
            "std": round(float(y.std()), 5),
            "cv_pct": round(100 * float(y.std() / y.mean()), 3),
            "p95": round(float(np.percentile(y, 95)), 5),
            "p95_shift_pct": round(
                100 * float((np.percentile(y, 95) - f0[k]) / f0[k]), 3
            ),
            "worst_shift_pct": round(100 * float((y.max() - f0[k]) / f0[k]), 3),
        }
    return Xmc, F, st


def boundary_risk_compare():
    designs = {
        "original_wrong_symmetric_3sigma_at_4.5": np.array([0.22, 4.5, 4.0]),
        "domain_conditional_truncated_at_4.5": np.array([0.22, 4.5, 4.0]),
        "safe_interior_gamma_4.45": np.array([0.22, 4.45, 4.0]),
        "safe_interior_gamma_4.40": np.array([0.22, 4.40, 4.0]),
    }
    rows = []
    for name, x0 in designs.items():
        rng = np.random.default_rng(0)
        if name.startswith("original_wrong"):
            b = symmetric_3sigma_wrong(
                x0[0], SIGMA_BETA_REL * x0[0], mc_n, rng
            )
            g = symmetric_3sigma_wrong(
                x0[1], SIGMA_GAMMA_REL * x0[1], mc_n, rng
            )
            Xmc = np.column_stack([b, g, np.full(mc_n, x0[2])])
            distribution_type = "old_symmetric_3sigma_clip_around_mean"
        else:
            Xmc = draw_domain_tolerance(x0, mc_n, 0)
            distribution_type = "conditional_truncated_normal_in_design_domain"
        F = predict(Xmc)
        f0 = predict(x0)[0]
        outside = int(((Xmc[:, 1] < gamma_min) | (Xmc[:, 1] > gamma_max)).sum())
        row = {
            "case": name,
            "distribution_type": distribution_type,
            "nominal_beta": x0[0],
            "nominal_gamma": x0[1],
            "nominal_N": x0[2],
            "gamma_min_sample": float(Xmc[:, 1].min()),
            "gamma_max_sample": float(Xmc[:, 1].max()),
            "outside_domain_count": outside,
        }
        for k, target in enumerate(TARGETS):
            y = F[:, k]
            row[f"{target}_nominal"] = float(f0[k])
            row[f"{target}_mean"] = float(y.mean())
            row[f"{target}_cv_pct"] = 100 * float(y.std() / y.mean())
            row[f"{target}_p95_shift_pct"] = 100 * float(
                (np.percentile(y, 95) - f0[k]) / f0[k]
            )
            row[f"{target}_worst_shift_pct"] = 100 * float((y.max() - f0[k]) / f0[k])
        rows.append(row)
    return pd.DataFrame(rows)


E_star, f_star = elasticity(x_star)
print("=== Elasticity @ design_star ===")
print(pd.DataFrame(E_star, index=TARGETS, columns=INPUTS).round(4).to_string())

sobol_rep = pd.concat(
    [sobol_replicates(sample_full, "full_design_domain"),
     sobol_replicates(sample_tol, "domain_truncated_tolerance")],
    ignore_index=True,
)
sobol_rep.to_csv(CODE_DIR / "p5_sobol_replicates.csv", index=False, encoding="utf-8-sig")
sobol_ci, sobol_table = summarize_sobol(sobol_rep)
with open(CODE_DIR / "p5_sobol_ci.json", "w", encoding="utf-8") as f:
    json.dump(sobol_ci, f, ensure_ascii=False, indent=2)
sobol_table.to_csv(
    CODE_DIR / "p5_sobol_table_for_paper.csv", index=False, encoding="utf-8-sig"
)

X_robust, F_robust, robust_stats = mc_statistics(x_star, mc_n, seed=0, kind="gpr")
Fr_robust = predict(X_robust, "rsm3")
mc_domain_audit = sample_domain_audit(X_robust)
with open(CODE_DIR / "p5_mc_domain_audit.json", "w", encoding="utf-8") as f:
    json.dump(mc_domain_audit, f, ensure_ascii=False, indent=2)

designs = {
    "鲁棒方案(0.22,4.5,4)": [0.22, 4.5, 4],
    "TOPSIS解(0.2175,4.5,2)": [0.2175, 4.5, 2],
    "Chebyshev解(0.22,4.5,6)": [0.22, 4.5, 6],
    "R*单指标最优(0.227,3.03,10)": [0.227, 3.034, 10],
}
mc_stats = {}
for name, x0 in designs.items():
    _, _, stats = mc_statistics(x0, mc_n, seed=0, kind="gpr")
    mc_stats[name] = stats
print("=== Domain-safe MC robust design ===")
for target in TARGETS:
    print(" ", target, robust_stats[target])

risk_df = boundary_risk_compare()
risk_df.to_csv(
    CODE_DIR / "p5_boundary_risk_compare.csv", index=False, encoding="utf-8-sig"
)

# Operating-condition sweep with the existing calibrated mechanism model.
with open(CODE_DIR / "p1_fit_results.json", encoding="utf-8") as f:
    p1 = json.load(f)
rho, cp, kf, mu, ks = 998.2, 4182.0, 0.6, 1.003e-3, 200.0
Pr = mu * cp / kf
M_units, n_ch = 5, 19
w_c, w_w, h_mc, L_seg, w_mf, t_cap = (
    200e-6,
    200e-6,
    400e-6,
    800e-6,
    1.6e-3,
    200e-6,
)
n_seg = M_units * n_ch * 2
Dh = 2 * w_c * h_mc / (w_c + h_mc)
A_src = 36e-6
r1, r2 = np.sqrt(A_src / np.pi), np.sqrt(1e-4 / np.pi)
R_cond = t_cap / (ks * A_src) + (1 - r1 / r2) ** 1.5 / (
    np.pi**0.5 * ks * r1
)
Rf, Pf = p1["R_fit"], p1["dP_fit"]


def fRe_rect(a):
    return 4 * 24 * (
        1
        - 1.3553 * a
        + 1.9467 * a**2
        - 1.7012 * a**3
        + 0.9564 * a**4
        - 0.2537 * a**5
    )


def mech_RdP(beta, gamma, N, mdot):
    u_ch = (mdot / n_seg) / (rho * w_c * h_mc)
    Re = rho * u_ch * Dh / mu
    Gz = Dh * Re * Pr / L_seg
    Nu = 4.12 + 0.0668 * Gz / (1 + 0.04 * Gz ** (2 / 3))
    u_mf = (mdot / M_units) / (rho * w_mf * gamma * h_mc)
    h_w = (1 + Rf["c_imp"] * u_mf / u_ch) * Nu * kf / Dh
    m_f = np.sqrt(2 * h_w / (ks * w_w))
    eta = np.tanh(m_f * h_mc) / (m_f * h_mc)
    A_wall = (2 * h_mc * eta + 2 * w_c) * L_seg
    psi = 1 + (Rf["a1"] * beta + Rf["a2"] * beta**2) * (
        1 - np.exp(-((N / Rf["n0"]) ** Rf["q"]))
    )
    d = beta * w_c
    u_max = u_ch / (1 - beta)
    Re_d = rho * u_max * d / mu
    Nu_d = 1.04 * Re_d**0.4 * Pr**0.36
    cw = np.exp(-Rf["a_w"] * max((L_seg / N) / d - 1, 0))
    hA = psi * h_w * A_wall + Rf["eps0"] * (1 - cw) * Nu_d * kf / d * N * np.pi * d * h_mc
    R = R_cond + 1 / (2 * mdot * cp) + 1 / (n_seg * hA)
    Dh_mf = 2 * w_mf * gamma * h_mc / (w_mf + gamma * h_mc)
    a_mf = min(w_mf, gamma * h_mc) / max(w_mf, gamma * h_mc)
    dp = (
        2 * fRe_rect(a_mf) * mu * 0.5 * u_mf * 0.01 / (2 * Dh_mf**2)
        + Pf["K_turn"] * 0.5 * rho * u_mf**2
        + fRe_rect(w_c / h_mc) * mu * u_ch * L_seg / (2 * Dh**2)
        + 1.2 * 0.5 * rho * u_ch**2
        + N
        * Pf["cD_scale"]
        * (10 / Re_d**0.8 + 1)
        / (1 - beta) ** Pf["p_blk"]
        * 0.5
        * rho
        * u_max**2
        * beta
    )
    return R, dp


mm = np.linspace(0.8, 1.2, 41)
op_R, op_dP = [], []
R0, dp0 = mech_RdP(0.22, 4.5, 4, 1e-3)
for s in mm:
    R, dp = mech_RdP(0.22, 4.5, 4, s * 1e-3)
    op_R.append(R)
    op_dP.append(dp)
op_R, op_dP = np.array(op_R), np.array(op_dP)
i_m10, i_p10 = 8, 32

np.savez(
    CODE_DIR / "p5_mc_samples.npz",
    robust=F_robust,
    robust_rsm3=Fr_robust,
    robust_X=X_robust,
    mm=mm,
    op_R=op_R,
    op_dP=op_dP,
)

# Backward-compatible point estimates for old tables, now based on replicate means.
def nested_metric(domain, metric):
    key = f"{metric}_mean"
    if metric == "ST":
        key = "ST_display"
    return {
        target: {
            var: round(float(sobol_ci[domain][target][var][key]), 3)
            for var in INPUTS
        }
        for target in TARGETS
    }


out = {
    "design": x_star.tolist(),
    "tolerances": {
        "sigma_beta_rel": SIGMA_BETA_REL,
        "sigma_gamma_rel": SIGMA_GAMMA_REL,
        "distribution": "conditional_truncated_normal_within_design_domain",
        "N": "fixed integer",
    },
    "domain_audit": mc_domain_audit,
    "elasticity": {
        t: dict(zip(INPUTS, E_star[i].round(4).tolist()))
        for i, t in enumerate(TARGETS)
    },
    "sobol_full": {
        "S1": nested_metric("full_design_domain", "S1"),
        "ST": nested_metric("full_design_domain", "ST"),
    },
    "sobol_tol": {
        "S1": nested_metric("domain_truncated_tolerance", "S1"),
        "ST": nested_metric("domain_truncated_tolerance", "ST"),
    },
    "sobol_ci_file": "p5_sobol_ci.json",
    "mc_stats": mc_stats,
    "rsm3_check": {
        t: {
            "mean": round(float(Fr_robust[:, k].mean()), 5),
            "std": round(float(Fr_robust[:, k].std()), 5),
        }
        for k, t in enumerate(TARGETS)
    },
    "operating": {
        "R0_KW": round(float(R0), 4),
        "dP0_Pa": round(float(dp0), 2),
        "R_m10_pct": round(100 * float(op_R[i_m10] / R0 - 1), 2),
        "R_p10_pct": round(100 * float(op_R[i_p10] / R0 - 1), 2),
        "dP_m10_pct": round(100 * float(op_dP[i_m10] / dp0 - 1), 2),
        "dP_p10_pct": round(100 * float(op_dP[i_p10] / dp0 - 1), 2),
    },
}
with open(CODE_DIR / "p5_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("saved p5 audit-fixed results")
