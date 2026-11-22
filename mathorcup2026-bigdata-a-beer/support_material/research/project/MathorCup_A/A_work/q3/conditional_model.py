"""Conditional chemical-profile model for MathorCup 2026 problem A, question 3.

All reported numerical results are produced by executing this module on the
provided data. Quantile transforms and estimators are fitted on training rows.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri, ndtr
from sklearn.covariance import LedoitWolf


DERIVED = ["kcalperc", "esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum"]
TECHNICAL = ["X3.methyl.2.butenal"]


class EmpiricalNormalMap:
    """Invertible piecewise-linear rank map on observed training support.

    For tied measurements, average ranks are used. Outside the observed range
    values map to the endpoint; this is a support restriction, not a claim of
    chemical feasibility. Medians used for missing inputs are training-only.
    """

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.medians_ = np.nanmedian(x, axis=0)
        self.knots_ = []
        for j in range(x.shape[1]):
            v = x[np.isfinite(x[:, j]), j]
            values, counts = np.unique(v, return_counts=True)
            rank_mid = np.cumsum(counts) - (counts - 1) / 2
            scores = ndtri((rank_mid - .5) / len(v))
            if len(values) == 1:
                scores = np.zeros(1)
            self.knots_.append((values, scores))
        self.n_features_in_ = x.shape[1]
        self.lower_ = np.array([v[1][0] for v in self.knots_])
        self.upper_ = np.array([v[1][-1] for v in self.knots_])
        return self

    def impute(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(np.isfinite(x), x, self.medians_)

    def transform(self, x):
        x = self.impute(x)
        return np.column_stack([
            np.interp(x[:, j], values, scores)
            for j, (values, scores) in enumerate(self.knots_)
        ])

    def inverse_transform(self, z):
        z = np.asarray(z, dtype=float)
        return np.column_stack([
            np.interp(z[:, j], scores, values)
            for j, (values, scores) in enumerate(self.knots_)
        ])


class ConditionalCopula:
    """Style-conditioned Gaussian copula with pooled covariance shrinkage.

    Style mean: n_s/(n_s+kappa) * mean_s + kappa/(n_s+kappa) * mean.
    Style covariance: w_s S_s + (1-w_s) Sigma_pool,
    w_s=(n_s-1)/(n_s-1+nu). Sigma_pool is Ledoit-Wolf regularized
    covariance of style-centered residuals with degrees-of-freedom correction.
    """

    def __init__(self, kind="hierarchical", kappa=2., nu=20.):
        self.kind, self.kappa, self.nu = kind, float(kappa), float(nu)

    def fit(self, x, style):
        self.map_ = EmpiricalNormalMap().fit(x)
        self.x_ = self.map_.impute(x)
        self.z_ = self.map_.transform(x)
        self.style_ = np.asarray(style).astype(str)
        self.styles_ = np.unique(self.style_)
        n, p = self.z_.shape
        grand = self.z_.mean(axis=0)
        centers = {s: self.z_[self.style_ == s].mean(axis=0) for s in self.styles_}
        residual = np.vstack([z - centers[s] for z, s in zip(self.z_, self.style_)])
        residual *= np.sqrt(n / max(n - len(self.styles_), 1))
        lw = LedoitWolf(assume_centered=True).fit(residual)
        self.pooled_cov_ = lw.covariance_ + np.eye(p) * 1e-5
        self.pooled_shrinkage_ = float(lw.shrinkage_)
        self.mean_, self.cov_, self.chol_, self.n_ = {}, {}, {}, {}
        for s in self.styles_:
            zs = self.z_[self.style_ == s]
            ns = len(zs)
            self.n_[s] = ns
            self.mean_[s] = (ns * centers[s] + self.kappa * grand) / (ns + self.kappa)
            if ns >= 2:
                local = np.atleast_2d(np.cov(zs, rowvar=False))
            else:
                local = self.pooled_cov_.copy()
            w = (ns - 1) / (ns - 1 + self.nu) if np.isfinite(self.nu) else 0.
            cov = w * local + (1 - w) * self.pooled_cov_
            if self.kind == "pooled":
                cov = self.pooled_cov_.copy()
            elif self.kind == "independent":
                cov = np.diag(np.diag(cov))
            cov = (cov + cov.T) / 2 + np.eye(p) * 1e-5
            self.cov_[s] = cov
            self.chol_[s] = np.linalg.cholesky(cov)
        return self

    def sample(self, style, n, rng):
        if self.kind == "bootstrap":
            idx = np.flatnonzero(self.style_ == style)
            chosen = rng.choice(idx, int(n), replace=True)
            x = self.x_[chosen].copy()
            return x, self.map_.transform(x)
        z = self.mean_[style] + rng.standard_normal((int(n), self.z_.shape[1])) @ self.chol_[style].T
        x = self.map_.inverse_transform(z)
        return x, self.map_.transform(x)

    def prototype(self, style):
        return self.map_.inverse_transform(self.mean_[style][None])[0]


def energy_scores(y, samples):
    """Energy score in training-fitted, dimension-normalized rank space."""
    p = y.shape[1]
    term1 = np.linalg.norm(y[:, None, :] - samples[None, :, :], axis=2).mean(axis=1)
    # Independent disjoint draws give a linear-cost unbiased second term.
    n = len(samples) // 2
    term2 = np.linalg.norm(samples[:n] - samples[n:2*n], axis=1).mean()
    return (term1 - .5 * term2) / np.sqrt(p)


def variogram_scores(y, samples, pairs):
    a, b = pairs[:, 0], pairs[:, 1]
    expected = np.abs(samples[:, a] - samples[:, b]) ** .5
    observed = np.abs(y[:, a] - y[:, b]) ** .5
    return np.mean((observed - expected.mean(axis=0)) ** 2, axis=1)


class IntervalCopula(ConditionalCopula):
    """Copula with explicit probability intervals for repeated/floored values.

    A chemical observed repeatedly at its detection floor corresponds to an
    interval of latent normal values, not to a single normal score. Conditional
    first and second moments restore the diagonal latent uncertainty. Inversion
    uses the empirical generalized inverse, preserving the point masses.

    kind='interval_kernel' additionally retains style-specific multimodality
    through a smoothed empirical latent mixture. It is a synthetic-profile
    model, not a mechanism for estimating unobserved true chemical quantities.
    """

    def __init__(self, kind="interval_gaussian", kappa=2., nu=20., bandwidth=.6):
        super().__init__(kind=kind,kappa=kappa,nu=nu)
        self.bandwidth=float(bandwidth)

    def fit(self,x,style):
        self.map_=EmpiricalNormalMap().fit(x)
        self.x_=self.map_.impute(x)
        self.z_=self.map_.transform(x)  # common evaluation/prediction coordinate
        self.style_=np.asarray(style).astype(str)
        self.styles_=np.unique(self.style_)
        n,p=self.x_.shape
        latent=np.zeros_like(self.x_);conditional_var=np.zeros_like(self.x_)
        self.intervals_=[];self.row_low_=np.zeros_like(self.x_);self.row_high_=np.zeros_like(self.x_)
        for j in range(p):
            values,count=np.unique(self.x_[:,j],return_counts=True)
            hi=np.cumsum(count)/n;lo=np.r_[0.,hi[:-1]]
            a,b=ndtri(lo),ndtri(hi)
            phia=np.exp(-a*a/2)/np.sqrt(2*np.pi)
            phib=np.exp(-b*b/2)/np.sqrt(2*np.pi)
            ex=(phia-phib)/(hi-lo)
            ap=np.zeros_like(a);bp=np.zeros_like(b)
            np.multiply(a,phia,out=ap,where=np.isfinite(a))
            np.multiply(b,phib,out=bp,where=np.isfinite(b))
            var=np.maximum(1+(ap-bp)/(hi-lo)-ex*ex,1e-8)
            ix=np.searchsorted(values,self.x_[:,j])
            latent[:,j]=ex[ix];conditional_var[:,j]=var[ix]
            self.row_low_[:,j]=lo[ix];self.row_high_[:,j]=hi[ix]
            self.intervals_.append((values,lo,hi))
        self.latent_=latent
        grand=latent.mean(axis=0)
        centers={s:latent[self.style_==s].mean(axis=0) for s in self.styles_}
        residual=np.vstack([z-centers[s] for z,s in zip(latent,self.style_)])
        residual*=np.sqrt(n/max(n-len(self.styles_),1))
        lw=LedoitWolf(assume_centered=True).fit(residual)
        self.pooled_cov_=lw.covariance_+np.diag(conditional_var.mean(axis=0))+np.eye(p)*1e-5
        self.pooled_shrinkage_=float(lw.shrinkage_)
        self.mean_,self.cov_,self.chol_,self.n_,self.center_={},{},{},{},centers
        for s in self.styles_:
            mask=self.style_==s;zs=latent[mask];ns=len(zs);self.n_[s]=ns
            self.mean_[s]=(ns*centers[s]+self.kappa*grand)/(ns+self.kappa)
            local=np.atleast_2d(np.cov(zs,rowvar=False)) if ns>=2 else self.pooled_cov_.copy()
            local+=np.diag(conditional_var[mask].mean(axis=0))
            w=(ns-1)/(ns-1+self.nu)
            cov=w*local+(1-w)*self.pooled_cov_+np.eye(p)*1e-5
            self.cov_[s]=cov;self.chol_[s]=np.linalg.cholesky(cov)
        return self

    def inverse_latent(self,z):
        u=ndtr(z);x=np.zeros_like(z)
        for j,(values,lo,hi) in enumerate(self.intervals_):
            ix=np.minimum(np.searchsorted(hi,u[:,j],side="left"),len(values)-1)
            x[:,j]=values[ix]
        return x

    def sample(self,style,n,rng):
        eps=rng.standard_normal((int(n),self.x_.shape[1]))@self.chol_[style].T
        if self.kind=="interval_kernel":
            idx=rng.choice(np.flatnonzero(self.style_==style),int(n),replace=True)
            u=rng.uniform(self.row_low_[idx],self.row_high_[idx])
            donors=ndtri(np.clip(u,1e-9,1-1e-9))
            h=self.bandwidth
            z=self.mean_[style]+np.sqrt(1-h*h)*(donors-self.center_[style])+h*eps
        else:
            z=self.mean_[style]+eps
        x=self.inverse_latent(z)
        return x,self.map_.transform(x)

    def prototype(self,style):
        return self.inverse_latent(self.mean_[style][None])[0]
