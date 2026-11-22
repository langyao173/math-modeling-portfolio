"""Universal kriging: cubic polynomial trend (flat prior) + Matern-5/2 GP residual.

Predictive covariance includes trend-coefficient uncertainty:
  Cov = K** - K*' Ki K* + R' (P' Ki P)^-1 R,   R = P* - P' Ki K*
Kernel hyper-parameters are estimated by maximum likelihood on the OLS residuals.
"""
from __future__ import annotations
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as Ck, Matern, WhiteKernel
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class UniversalKriging:
    def __init__(self, degree: int = 3, seed: int = 0):
        self.degree, self.seed = degree, seed

    def fit(self, X, y):
        self.sx = StandardScaler().fit(X)
        Xs = self.sx.transform(X)
        self.pf = PolynomialFeatures(self.degree).fit(Xs)
        P = self.pf.transform(Xs)
        self.ym, self.ys = y.mean(), y.std()
        z = (y - self.ym) / self.ys
        r = z - P @ np.linalg.lstsq(P, z, rcond=None)[0]
        k = (Ck(1e-2, (1e-8, 1e2)) * Matern([1.0] * X.shape[1], (1e-2, 1e3), nu=2.5)
             + WhiteKernel(1e-6, (1e-12, 1e-1)))
        g = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=3, random_state=self.seed).fit(Xs, r)
        self.kernel = g.kernel_
        K = self.kernel(Xs)
        self.L = np.linalg.cholesky(K + 1e-12 * np.eye(len(K)))
        Ki_P = self._solve(P); Ki_z = self._solve(z)
        self.A = P.T @ Ki_P
        self.Ainv = np.linalg.inv(self.A)
        self.beta = self.Ainv @ (P.T @ Ki_z)
        self.alpha = self._solve(z - P @ self.beta)
        self.Xs, self.P = Xs, P
        return self

    def _solve(self, B):
        return np.linalg.solve(self.L.T, np.linalg.solve(self.L, B))

    def predict(self, Xn, return_cov=False, return_std=False):
        Xs = self.sx.transform(Xn)
        Ps = self.pf.transform(Xs)
        k1 = self.kernel.k1  # signal part only (no nugget) for latent function
        Ks = k1(self.Xs, Xs)
        mu = Ps @ self.beta + Ks.T @ self.alpha
        Ki_Ks = self._solve(Ks)
        R = Ps.T - self.P.T @ Ki_Ks
        if return_cov:
            cov = k1(Xs) - Ks.T @ Ki_Ks + R.T @ self.Ainv @ R
            return mu * self.ys + self.ym, cov * self.ys ** 2
        var = np.diag(k1(Xs)) - np.einsum("ij,ij->j", Ks, Ki_Ks) + np.einsum("ij,ik,kj->j", R, self.Ainv, R)
        sd = np.sqrt(np.clip(var, 0, None)) * self.ys
        noise = np.sqrt(self.kernel.k2.noise_level) * self.ys
        if return_std:
            return mu * self.ys + self.ym, np.sqrt(sd ** 2 + noise ** 2)
        return mu * self.ys + self.ym
