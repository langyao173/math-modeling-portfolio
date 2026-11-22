# -*- coding: utf-8 -*-
"""Ablation and identifiability audit for the grey-box model."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

import p1_mechanism_model as p1
from config import CODE_DIR, seed_list

X = p1.X
yR, ydP, yphi = p1.yR, p1.ydP, p1.yphi
n = len(X)


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def info_metrics(y, p, k):
    resid = y - p
    sse = float(np.sum(resid**2))
    sigma = max(sse / len(y), 1e-30)
    return {
        "R2": float(r2(y, p)),
        "RMSE": float(np.sqrt(np.mean(resid**2))),
        "AIC": float(len(y) * np.log(sigma) + 2 * k),
        "BIC": float(len(y) * np.log(sigma) + k * np.log(len(y))),
        "Spearman": float(spearmanr(y, p)[0]),
    }


def has_interior_min(vals):
    vals = np.asarray(vals)
    idx = int(np.argmin(vals))
    return bool(0 < idx < len(vals) - 1)


def beta_main_effect(pred):
    df = pd.DataFrame(X, columns=["beta", "gamma", "N"])
    df["pred"] = pred
    sub = df[df["N"] > 0]
    return sub.groupby("beta")["pred"].mean().sort_index().values


def gamma_main_effect(pred):
    df = pd.DataFrame(X, columns=["beta", "gamma", "N"])
    df["pred"] = pred
    sub = df[df["N"] > 0]
    return sub.groupby("gamma")["pred"].mean().sort_index().values


def N_main_effect(pred):
    df = pd.DataFrame(X, columns=["beta", "gamma", "N"])
    df["pred"] = pred
    sub = df[df["N"] > 0]
    return sub.groupby("N")["pred"].mean().sort_index().values


def fit_R_variant(name):
    if name == "full_R":
        pred = p1.predR
        return pred, 8
    if name == "no_pin_direct_heat":
        def model(t):
            c_imp, a_w, a1, a2, n0, q, R_ref = t
            return p1.R_model(X, [c_imp, 0.0, a_w, a1, a2, n0, q, R_ref])
        bounds = ([0, 0.05, 0.0, -50.0, 0.5, 1.0, 0.3],
                  [3, 10, 20, 0.0, 20, 4, 3])
        x0 = [0.1, 3.0, 0.5, -1.0, 3.0, 2.0, 1.0]
        res = least_squares(lambda t: model(t) - yR, x0=x0, bounds=bounds)
        return model(res.x), len(res.x)
    if name == "no_beta2_blockage":
        def model(t):
            c_imp, eps0, a_w, a1, n0, q, R_ref = t
            return p1.R_model(X, [c_imp, eps0, a_w, a1, 0.0, n0, q, R_ref])
        bounds = ([0, 0, 0.05, 0.0, 0.5, 1.0, 0.3],
                  [3, 1, 10, 20, 20, 4, 3])
        x0 = [0.1, 0.02, 3.0, 0.5, 3.0, 2.0, 1.0]
        res = least_squares(lambda t: model(t) - yR, x0=x0, bounds=bounds)
        return model(res.x), len(res.x)
    if name == "no_N_saturation":
        def model(t):
            c_imp, eps0, a_w, a1, a2, R_ref = t
            return p1.R_model(X, [c_imp, eps0, a_w, a1, a2, 0.5, 1.0, R_ref])
        bounds = ([0, 0, 0.05, 0.0, -50.0, 0.3],
                  [3, 1, 10, 20, 0.0, 3])
        x0 = [0.1, 0.02, 3.0, 0.5, -1.0, 1.0]
        res = least_squares(lambda t: model(t) - yR, x0=x0, bounds=bounds)
        return model(res.x), len(res.x)
    raise KeyError(name)


def fit_phi_variant(name):
    if name == "full_phi":
        return p1.predPhi, 8
    if name == "no_N_linear_penalty":
        keep = [0, 1, 2, 3, 5, 6]
    elif name == "no_gamma_inverse2":
        keep = [0, 1, 2, 3, 4, 6]
    elif name == "no_gamma_linear":
        keep = [0, 1, 2, 3, 4, 5]
    elif name == "no_distribution_imbalance":
        keep = [0, 1, 2, 3, 4, 6]
    else:
        raise KeyError(name)
    best = None
    for tau in np.arange(0.3, 6.05, 0.05):
        F = p1.phi_features(X, tau)[:, keep]
        c, *_ = np.linalg.lstsq(F, yphi, rcond=None)
        pred = F @ c
        sse = float(np.sum((pred - yphi) ** 2))
        if best is None or sse < best[0]:
            best = (sse, pred, len(c) + 1)
    return best[1], best[2]


rows = []
for name in ["full_R", "no_pin_direct_heat", "no_beta2_blockage", "no_N_saturation"]:
    pred, k = fit_R_variant(name)
    rec = {"model": name, "target": "R", "n_params": k}
    rec.update(info_metrics(yR, pred, k))
    rec["reproduces_beta_nonmonotonic"] = has_interior_min(beta_main_effect(pred))
    rec["reproduces_N_nonmonotonic"] = has_interior_min(N_main_effect(pred))
    rec["reproduces_gamma_nonmonotonic"] = has_interior_min(gamma_main_effect(pred))
    rows.append(rec)

for name in [
    "full_phi",
    "no_N_linear_penalty",
    "no_gamma_inverse2",
    "no_gamma_linear",
    "no_distribution_imbalance",
]:
    pred, k = fit_phi_variant(name)
    rec = {"model": name, "target": "phi", "n_params": k}
    rec.update(info_metrics(yphi, pred, k))
    rec["reproduces_beta_nonmonotonic"] = has_interior_min(beta_main_effect(pred))
    rec["reproduces_N_nonmonotonic"] = has_interior_min(N_main_effect(pred))
    rec["reproduces_gamma_nonmonotonic"] = has_interior_min(gamma_main_effect(pred))
    rows.append(rec)

pd.DataFrame(rows).to_csv(
    CODE_DIR / "p1_ablation_identifiability.csv", index=False, encoding="utf-8-sig"
)

# Multi-start stability for nonlinear R and dP submodels.
rng = np.random.default_rng(0)
R_lower = np.array([0, 0.005, 0.05, 0.0, -50.0, 0.5, 1.0, 0.3])
R_upper = np.array([3, 1, 10, 20, 0.0, 20, 4, 3])
P_lower = np.array([0.1, 0.05, 0.0, 50])
P_upper = np.array([30, 20, 8, 5000])
stability_rows = []

for seed in range(30):
    x0 = R_lower + rng.random(len(R_lower)) * (R_upper - R_lower)
    res = least_squares(lambda t: p1.R_model(X, t) - yR, x0=x0, bounds=(R_lower, R_upper))
    for name, value, lo, hi in zip(
        ["c_imp", "eps0", "a_w", "a1", "a2", "n0", "q", "R_ref"],
        res.x,
        R_lower,
        R_upper,
    ):
        stability_rows.append(
            {
                "submodel": "R",
                "seed": seed,
                "parameter": name,
                "value": float(value),
                "at_lower_bound": bool(np.isclose(value, lo, rtol=0, atol=1e-5)),
                "at_upper_bound": bool(np.isclose(value, hi, rtol=0, atol=1e-5)),
                "cost": float(res.cost),
            }
        )
    x0p = P_lower + rng.random(len(P_lower)) * (P_upper - P_lower)
    resp = least_squares(lambda t: p1.dP_model(X, t) - ydP, x0=x0p, bounds=(P_lower, P_upper))
    for name, value, lo, hi in zip(
        ["K_turn", "cD_scale", "p_blk", "dP_ref"], resp.x, P_lower, P_upper
    ):
        stability_rows.append(
            {
                "submodel": "dP",
                "seed": seed,
                "parameter": name,
                "value": float(value),
                "at_lower_bound": bool(np.isclose(value, lo, rtol=0, atol=1e-5)),
                "at_upper_bound": bool(np.isclose(value, hi, rtol=0, atol=1e-5)),
                "cost": float(resp.cost),
            }
        )

stab = pd.DataFrame(stability_rows)
stab.to_csv(CODE_DIR / "p1_parameter_stability.csv", index=False, encoding="utf-8-sig")

summary = {}
for (submodel, parameter), g in stab.groupby(["submodel", "parameter"]):
    summary.setdefault(submodel, {})[parameter] = {
        "mean": float(g["value"].mean()),
        "std": float(g["value"].std(ddof=1)),
        "min": float(g["value"].min()),
        "max": float(g["value"].max()),
        "lower_bound_hit_rate": float(g["at_lower_bound"].mean()),
        "upper_bound_hit_rate": float(g["at_upper_bound"].mean()),
    }
with open(CODE_DIR / "p1_parameter_stability_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

wide = stab.pivot_table(
    index=["submodel", "seed"], columns="parameter", values="value", aggfunc="first"
)
corr_payload = {}
for submodel in wide.index.get_level_values(0).unique():
    block = wide.loc[submodel]
    corr_payload[submodel] = block.corr().round(6).fillna(0).to_dict()
with open(CODE_DIR / "p1_parameter_correlation.json", "w", encoding="utf-8") as f:
    json.dump(corr_payload, f, ensure_ascii=False, indent=2)

eps0_summary = summary["R"]["eps0"]
diagnosis = {
    "eps0_lower_bound_hit_rate": eps0_summary["lower_bound_hit_rate"],
    "eps0_mean": eps0_summary["mean"],
    "interpretation": (
        "If no_pin_direct_heat keeps comparable AIC/BIC and eps0 repeatedly hits "
        "the lower bound, direct pin-fin heat transfer should be treated as weakly "
        "identified rather than claimed as an independent strong mechanism."
    ),
}
with open(CODE_DIR / "p1_eps0_identifiability_note.json", "w", encoding="utf-8") as f:
    json.dump(diagnosis, f, ensure_ascii=False, indent=2)

print("saved p1 ablation and identifiability audit")
