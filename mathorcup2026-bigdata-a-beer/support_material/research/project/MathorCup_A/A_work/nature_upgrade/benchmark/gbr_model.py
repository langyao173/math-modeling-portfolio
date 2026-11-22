# Team review and independent work must be documented truthfully.
"""Portable, array-based inference for the classical GBR challengers."""
import numpy as np


class GBRTargetModel:
    def __init__(self, imputer, estimators, specs):
        self.imputer_ = imputer
        self.estimators_ = list(estimators)
        self.specs_ = list(specs)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Expected a two-dimensional array of chemical inputs.")
        Z = self.imputer_.transform(X)
        return np.column_stack([model.predict(Z) for model in self.estimators_])

