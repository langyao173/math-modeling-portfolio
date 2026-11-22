"""A fixed-penalty Gaussian interval MCEM challenge to the reference generator.

The E step uses finite Gibbs sampling. The M step exactly minimizes a stated
expected penalized complete-data criterion for those sampled moments. It does
not certify monotonicity of the observed likelihood or Monte Carlo convergence.
The pooled prior moments are frozen at the training-only reference fit.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from scipy.special import ndtri, ndtr
from conditional_moments import IntervalCopula, gibbs_sweep


def expected_scatter(total, second, n, mu, prior_mean, prior_cov, kappa, nu):
    return (second - np.outer(total, mu) - np.outer(mu, total)
            + n * np.outer(mu, mu) + nu * prior_cov
            + kappa * np.outer(mu-prior_mean, mu-prior_mean))


def expected_penalized_loss(total, second, n, mu, cov, prior_mean, prior_cov, kappa, nu):
    sign, ld = np.linalg.slogdet(cov)
    if sign <= 0:
        return float('inf')
    scatter = expected_scatter(total, second, n, mu, prior_mean, prior_cov, kappa, nu)
    return float(.5 * ((n+nu)*ld + np.trace(np.linalg.solve(cov, scatter))))


def exact_m_step(total, second, n, prior_mean, prior_cov, kappa, nu):
    """Joint minimizer, including the mean-shrinkage scatter correction."""
    mu = (total + kappa*prior_mean)/(n+kappa)
    m = total/n
    within = second - n*np.outer(m, m)
    correction = n*kappa/(n+kappa)*np.outer(m-prior_mean, m-prior_mean)
    cov = (within + correction + nu*prior_cov)/(n+nu)
    cov = (cov+cov.T)/2
    np.linalg.cholesky(cov)
    return mu, cov


class PenalizedIntervalEM(IntervalCopula):
    def __init__(self, kappa=2., nu=20., iterations=8, burn_sweeps=8,
                 retained_sweeps=8, seed=2026102409):
        super().__init__(kind='interval_gaussian', kappa=kappa, nu=nu)
        self.iterations = iterations
        self.burn_sweeps = burn_sweeps
        self.retained_sweeps = retained_sweeps
        self.seed = seed

    def fit(self, x, style):
        super().fit(x, style)
        x = np.asarray(x, float)
        n, p = x.shape
        lower, upper = ndtri(self.row_low_), ndtri(self.row_high_)
        z = ndtri(np.clip((ndtr(lower)+ndtr(upper))/2, 1e-10, 1-1e-10))
        self.prior_mean_ = self.latent_.mean(0).copy()
        self.prior_cov_ = self.pooled_cov_.copy()
        labels = np.searchsorted(self.styles_, self.style_)
        masks = [np.flatnonzero(labels == k) for k in range(len(self.styles_))]
        rng = np.random.default_rng(self.seed)
        self.refinement_trace_ = []
        for iteration in range(self.iterations):
            old_mean = np.stack([self.mean_[s] for s in self.styles_])
            old_cov = np.stack([self.cov_[s] for s in self.styles_])
            precision = np.linalg.inv(old_cov)
            means = old_mean[labels]
            for _ in range(self.burn_sweeps):
                z = gibbs_sweep(z, means, precision, labels, lower, upper, rng)
            mean_z = np.zeros_like(z)
            second = [np.zeros((p, p)) for _ in masks]
            for _ in range(self.retained_sweeps):
                z = gibbs_sweep(z, means, precision, labels, lower, upper, rng)
                mean_z += z/self.retained_sweeps
                for k, idx in enumerate(masks):
                    second[k] += z[idx].T@z[idx]/self.retained_sweeps
            before, after, mean_eq, covariance_eq, records = [], [], [], [], []
            for k, (s, idx) in enumerate(zip(self.styles_, masks)):
                ns = len(idx)
                total = mean_z[idx].sum(0)
                mu, cov = exact_m_step(total, second[k], ns, self.prior_mean_, self.prior_cov_, self.kappa, self.nu)
                before.append(expected_penalized_loss(total, second[k], ns, old_mean[k], old_cov[k],
                    self.prior_mean_, self.prior_cov_, self.kappa, self.nu))
                after.append(expected_penalized_loss(total, second[k], ns, mu, cov,
                    self.prior_mean_, self.prior_cov_, self.kappa, self.nu))
                scatter = expected_scatter(total, second[k], ns, mu, self.prior_mean_, self.prior_cov_, self.kappa, self.nu)
                mean_eq.append(float(np.max(np.abs((ns+self.kappa)*mu-total-self.kappa*self.prior_mean_))))
                covariance_eq.append(float(np.max(np.abs((ns+self.nu)*cov-scatter))))
                self.mean_[s], self.cov_[s] = mu, cov
                self.chol_[s] = np.linalg.cholesky(cov)
                self.center_[s] = total/ns
                records.append(dict(style=s, n=ns, total=total.copy(), second=second[k].copy(),
                                    old_mean=old_mean[k].copy(), old_cov=old_cov[k].copy(),
                                    loss_before=before[-1], loss_after=after[-1]))
            drops = np.asarray(before)-np.asarray(after)
            assert drops.min() >= -1e-8, 'Closed-form M step increased its fixed-E criterion'
            new_cov = np.stack([self.cov_[s] for s in self.styles_])
            self.refinement_trace_.append(dict(iteration=iteration+1,
                expected_penalized_loss_before=float(np.sum(before)),
                expected_penalized_loss_after=float(np.sum(after)),
                minimum_style_loss_drop=float(drops.min()),
                mean_stationarity_max=float(max(mean_eq)),
                covariance_stationarity_max=float(max(covariance_eq)),
                relative_covariance_change=float(np.linalg.norm(new_cov-old_cov)/np.linalg.norm(old_cov)),
                mean_change_rms=float(np.sqrt(np.mean((np.stack([self.mean_[s] for s in self.styles_])-old_mean)**2))),
                minimum_style_covariance_eigenvalue=float(min(np.linalg.eigvalsh(v)[0] for v in new_cov)),
                maximum_interval_violation=float(max(0, np.max(lower-z), np.max(z-upper)))))
            self.latent_ = mean_z
            self.last_statistics_ = records
        return self
