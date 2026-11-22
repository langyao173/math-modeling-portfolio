# -*- coding: utf-8 -*-
"""
Problem 2 audit-fixed surrogate models.

Main protocol:
- strict outer LOOCV: each sample is predicted by models fitted only on the
  other 83 samples;
- fold-local preprocessing: every StandardScaler and PolynomialFeatures object
  is inside the outer training fold;
- nested tuning: KRR, SVR and MLP GridSearchCV are fitted only inside the
  current outer training fold;
- final dual proxies: RSM-3 precision proxy and GPR uncertainty proxy are both
  saved for downstream checks.
"""
from __future__ import annotations

import json
import shutil
import warnings
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as Ck
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from config import (
    CODE_DIR,
    DATA_FILE,
    INPUTS,
    TARGETS,
    cv_inner_splits,
    ensure_dirs,
    gpr_restarts_cv,
    gpr_restarts_final,
    random_seed,
    strict_cv,
)

warnings.filterwarnings("ignore")
ensure_dirs()

legacy_cv = CODE_DIR / "p2_cv_results.json"
legacy_backup = CODE_DIR / "p2_cv_results_legacy_reference.json"
if legacy_cv.exists() and not legacy_backup.exists():
    try:
        with open(legacy_cv, encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("protocol") != "strict_nested_loocv":
            shutil.copy2(legacy_cv, legacy_backup)
    except Exception:
        pass

df = pd.read_csv(
    DATA_FILE,
    sep="\t",
    skiprows=2,
    header=None,
    names=["id", "beta", "gamma", "N", "R", "dP", "phi"],
    encoding="utf-8-sig",
).dropna()
X = df[INPUTS].to_numpy(float)
n = len(df)


def make_gpr(n_restarts: int) -> GaussianProcessRegressor:
    kernel = (
        Ck(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=[1.0, 1.0, 1.0],
            length_scale_bounds=(1e-2, 1e3),
            nu=2.5,
        )
        + WhiteKernel(1e-6, (1e-10, 1e-2))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=random_seed,
    )


def make_pipeline_model(name: str):
    if name == "RSM-2次":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(2, include_bias=True)),
                ("regressor", LinearRegression()),
            ]
        ), None
    if name == "RSM-3次":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(3, include_bias=True)),
                ("regressor", LinearRegression()),
            ]
        ), None
    if name == "KRR-RBF":
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", KernelRidge(kernel="rbf")),
            ]
        )
        grid = {
            "regressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
            "regressor__gamma": [0.05, 0.1, 0.3, 1.0, 3.0],
        }
        return pipe, grid
    if name == "SVR-RBF":
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", SVR(kernel="rbf")),
            ]
        )
        grid = {
            "regressor__C": [1.0, 10.0, 100.0],
            "regressor__gamma": [0.05, 0.1, 0.3, 1.0],
            "regressor__epsilon": [1e-4, 1e-3, 1e-2],
        }
        return pipe, grid
    if name == "随机森林":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    RandomForestRegressor(n_estimators=500, random_state=random_seed),
                ),
            ]
        ), None
    if name == "MLP(16-8)":
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    MLPRegressor(
                        hidden_layer_sizes=(16, 8),
                        max_iter=8000,
                        random_state=random_seed,
                    ),
                ),
            ]
        )
        grid = {
            "regressor__alpha": [1e-3, 1e-2, 1e-1],
            "regressor__learning_rate_init": [1e-3, 3e-3],
        }
        return pipe, grid
    raise KeyError(name)


MODEL_NAMES = [
    "GPR-Matern",
    "RSM-2次",
    "RSM-3次",
    "KRR-RBF",
    "SVR-RBF",
    "随机森林",
    "MLP(16-8)",
]


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    return {
        "R2": round(float(r2), 4),
        "RMSE": round(float(np.sqrt(np.mean((y - p) ** 2))), 5),
        "MAE": round(float(np.mean(np.abs(y - p))), 5),
        "MaxRelErr_pct": round(float(np.max(np.abs(y - p) / y)) * 100, 2),
    }


def fit_predict_gpr_fold(X_train, y_train, X_test):
    scaler = StandardScaler().fit(X_train)
    gpr = make_gpr(gpr_restarts_cv).fit(scaler.transform(X_train), y_train)
    mu, sd = gpr.predict(scaler.transform(X_test), return_std=True)
    return float(mu[0]), float(sd[0]), {
        "scaler_fitted_on": int(len(X_train)),
        "kernel": str(gpr.kernel_),
    }


def fit_predict_pipeline_fold(name, X_train, y_train, X_test):
    pipe, grid = make_pipeline_model(name)
    if grid:
        inner_cv = KFold(
            n_splits=cv_inner_splits, shuffle=True, random_state=random_seed
        )
        estimator = GridSearchCV(
            pipe,
            grid,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        return float(pred[0]), {
            "pipeline_fitted_on": int(len(X_train)),
            "inner_cv_splits": cv_inner_splits,
            "best_params": deepcopy(estimator.best_params_),
        }
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return float(pred[0]), {"pipeline_fitted_on": int(len(X_train))}


pred_columns: dict[str, np.ndarray] = {}
sd_columns: dict[str, np.ndarray] = {}
results: dict[str, dict[str, dict]] = {}
hyperparams_by_fold: dict[str, dict[str, list]] = {}
fold_audit = []

for target in TARGETS:
    y = df[target].to_numpy(float)
    results[target] = {}
    hyperparams_by_fold[target] = {m: [] for m in MODEL_NAMES}
    preds = {m: np.zeros(n) for m in MODEL_NAMES}
    sds = np.full(n, np.nan)

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train = y[train_mask]

        if target == TARGETS[0]:
            fold_audit.append(
                {
                    "outer_fold": int(i),
                    "train_count": int(len(X_train)),
                    "test_count": int(len(X_test)),
                    "test_id": int(df.iloc[i]["id"]),
                    "fit_scope": "all preprocessing and tuning fitted within this outer training fold",
                }
            )

        mu, sd, audit = fit_predict_gpr_fold(X_train, y_train, X_test)
        preds["GPR-Matern"][i] = mu
        sds[i] = sd
        hyperparams_by_fold[target]["GPR-Matern"].append(audit)

        for name in MODEL_NAMES[1:]:
            pred, audit = fit_predict_pipeline_fold(name, X_train, y_train, X_test)
            preds[name][i] = pred
            hyperparams_by_fold[target][name].append(audit)

    for name in MODEL_NAMES:
        res = metrics(y, preds[name])
        if name == "GPR-Matern":
            res["CI95_coverage_pct"] = round(
                float(np.mean(np.abs(y - preds[name]) <= 1.96 * sds)) * 100, 1
            )
        results[target][name] = res
        pred_columns[f"{target}_{name}_pred"] = preds[name]
    sd_columns[f"{target}_GPR-Matern_sd"] = sds
    print("==", target, "nested LOOCV done ==")
    for name, res in results[target].items():
        print("  %-10s" % name, res)


# Final full-data proxies for downstream use.
final_scaler = StandardScaler().fit(X)
final = {"scalerX": final_scaler, "protocol": "strict_nested_loocv_for_reported_cv"}
gpr_only = {"scalerX": final_scaler}
rsm3_only = {}
learned_kernels = {}
for target in TARGETS:
    y = df[target].to_numpy(float)
    gpr = make_gpr(gpr_restarts_final).fit(final_scaler.transform(X), y)
    final[target] = gpr
    gpr_only[target] = gpr
    learned_kernels[target] = str(gpr.kernel_)

    rsm3 = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(3, include_bias=True)),
            ("regressor", LinearRegression()),
        ]
    ).fit(X, y)
    final[f"rsm3_{target}"] = rsm3
    rsm3_only[target] = rsm3

joblib.dump(final, CODE_DIR / "p2_gpr_models.pkl")
joblib.dump(gpr_only, CODE_DIR / "gpr_uncertainty_models.pkl")
joblib.dump(rsm3_only, CODE_DIR / "rsm3_precision_models.pkl")


def predict_final_gpr(X_query):
    X_query = np.atleast_2d(np.asarray(X_query, float))
    Xs = final_scaler.transform(X_query)
    return np.column_stack([final[t].predict(Xs) for t in TARGETS])


def predict_final_rsm3(X_query):
    X_query = np.atleast_2d(np.asarray(X_query, float))
    return np.column_stack([rsm3_only[t].predict(X_query) for t in TARGETS])


gpr_train = predict_final_gpr(X)
rsm3_train = predict_final_rsm3(X)
dual = df[["id", "beta", "gamma", "N"]].copy()
for k, target in enumerate(TARGETS):
    dual[f"{target}_gpr"] = gpr_train[:, k]
    dual[f"{target}_rsm3"] = rsm3_train[:, k]
    dual[f"{target}_abs_diff"] = np.abs(gpr_train[:, k] - rsm3_train[:, k])
dual["source"] = "training_samples"
dual.to_csv(CODE_DIR / "dual_proxy_check.csv", index=False, encoding="utf-8-sig")

loocv_all = df.copy()
for col, values in pred_columns.items():
    loocv_all[col] = values
for col, values in sd_columns.items():
    loocv_all[col] = values
loocv_all.to_csv(
    CODE_DIR / "p2_nested_loocv_all_models.csv", index=False, encoding="utf-8-sig"
)

gpr_out = df.copy()
for target in TARGETS:
    gpr_out[f"{target}_loocv"] = pred_columns[f"{target}_GPR-Matern_pred"]
    gpr_out[f"{target}_loocv_sd"] = sd_columns[f"{target}_GPR-Matern_sd"]
gpr_out.to_csv(CODE_DIR / "p2_loocv_gpr.csv", index=False, encoding="utf-8-sig")

payload = {
    "protocol": "strict_nested_loocv",
    "strict_cv": strict_cv,
    "loocv": results,
    "learned_kernels": learned_kernels,
    "reported_results_file": "p2_nested_cv_results.json",
}
with open(CODE_DIR / "p2_nested_cv_results.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
with open(CODE_DIR / "p2_cv_results.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

audit = {
    "protocol": "strict nested LOOCV",
    "outer_folds": fold_audit,
    "model_names": MODEL_NAMES,
    "pre_fit_before_outer_loop": False,
    "forbidden_full_data_fit_before_loocv": "not used for any reported LOOCV result",
    "pipeline_scope": "Pipeline/GridSearchCV are instantiated and fitted inside each outer training fold",
    "gpr_scope": "StandardScaler and GaussianProcessRegressor are fitted inside each outer training fold",
    "hyperparams_by_fold": hyperparams_by_fold,
}
with open(CODE_DIR / "p2_cv_protocol_audit.json", "w", encoding="utf-8") as f:
    json.dump(audit, f, ensure_ascii=False, indent=2)

print("saved strict nested CV results and dual proxy models")
