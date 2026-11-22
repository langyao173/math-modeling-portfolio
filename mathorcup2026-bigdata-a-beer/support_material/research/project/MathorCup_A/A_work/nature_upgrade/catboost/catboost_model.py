# Team review and independent work must be documented truthfully.
"""Joint 50-output CatBoost with preprocessing learned inside each fit.

The local dependency directory is optional. For independent reproduction,
install catboost==1.2.10 in an isolated environment before importing this file.
"""
from pathlib import Path
import json
import sys

_vendor = Path(__file__).resolve().parents[1] / "vendor"
if _vendor.exists():
    sys.path.insert(0, str(_vendor))

import numpy as np
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer

MODEL_VERSION = "catboost-multirmse-fold-local-v1"
SEED = 20261023
FIXED = dict(loss_function="MultiRMSE", learning_rate=0.05,
             l2_leaf_reg=10, border_count=32, thread_count=2,
             random_seed=SEED, boosting_type="Plain",
             allow_writing_files=False, verbose=False)


class CatboostMultiTarget:
    """One shared tree structure, with a vector-valued leaf for each target."""

    def __init__(self, depth=3, n_estimators=300):
        self.depth = int(depth)
        self.n_estimators = int(n_estimators)

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2 or not np.isfinite(Y).all():
            raise ValueError("Training targets must be a finite 2D matrix.")
        self.n_features_in_ = X.shape[1]
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        Z = self.imputer_.fit_transform(X)
        self.y_mean_ = Y.mean(axis=0)
        self.y_scale_ = np.maximum(Y.std(axis=0, ddof=1), 1e-8)
        self.model_ = CatBoostRegressor(depth=self.depth,
                                       iterations=self.n_estimators, **FIXED)
        self.model_.fit(Z, (Y - self.y_mean_) / self.y_scale_)
        return self

    def predict(self, X, n_estimators=None):
        n = self.n_estimators if n_estimators is None else int(n_estimators)
        if n < 1 or n > self.model_.tree_count_:
            raise ValueError("Requested prefix must be within the fitted tree count.")
        Z = self.imputer_.transform(np.asarray(X, dtype=float))
        P = self.model_.predict(Z, ntree_end=n, thread_count=FIXED["thread_count"])
        return np.asarray(P, dtype=float) * self.y_scale_ + self.y_mean_

    def save_native(self, folder, stem, features=None, targets=None,
                    selected_trees=None):
        """Save without pickle: native model plus complete numeric preprocessing."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.model_.save_model(str(folder / f"{stem}.cbm"))
        metadata = dict(algorithm_version=MODEL_VERSION, depth=self.depth,
                        fitted_trees=int(self.model_.tree_count_),
                        selected_trees=int(selected_trees or self.n_estimators),
                        n_features=self.n_features_in_,
                        imputer_statistics=self.imputer_.statistics_.tolist(),
                        target_mean=self.y_mean_.tolist(),
                        target_scale=self.y_scale_.tolist(),
                        features=features, targets=targets, fixed_parameters=FIXED)
        (folder / f"{stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load_native(cls, folder, stem):
        folder = Path(folder)
        m = json.loads((folder / f"{stem}.json").read_text(encoding="utf-8"))
        if m["algorithm_version"] != MODEL_VERSION:
            raise ValueError("Unknown preprocessing/model version.")
        obj = cls(m["depth"], m["selected_trees"])
        obj.n_features_in_ = m["n_features"]
        # Refit this deterministic transformer to its saved median row. This
        # restores sklearn's supported fitted attributes without unpickling.
        obj.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        obj.imputer_.fit(np.asarray(m["imputer_statistics"], dtype=float)[None, :])
        obj.y_mean_ = np.asarray(m["target_mean"], dtype=float)
        obj.y_scale_ = np.asarray(m["target_scale"], dtype=float)
        obj.model_ = CatBoostRegressor()
        obj.model_.load_model(str(folder / f"{stem}.cbm"))
        return obj
