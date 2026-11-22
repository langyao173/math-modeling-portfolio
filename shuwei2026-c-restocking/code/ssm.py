# -*- coding: utf-8 -*-
"""Linear-Gaussian state-space models for the representative hand point.

Per axis the state is (position, velocity).  Dynamics: integrated Ornstein-Uhlenbeck (IOU)
    dc = v dt,   dv = -theta v dt + sigma dW,     y_k = c(t_k) + eps_k,  eps ~ N(0, r^2)
theta -> 0 gives the constant-velocity (white-noise-acceleration) model; the random walk
(hold-last) baseline has state (position) only.  Parameters may depend on a discrete phase.
All three axes share parameters and are filtered independently.
"""
import math

import numpy as np

try:
    from numba import njit
except Exception:                                    # pragma: no cover - plain Python fallback
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (len(a) == 1 and callable(a[0])) else a[0]

LOG2PI = math.log(2.0 * math.pi)


@njit(cache=True)
def iou_FQ(theta, sigma, dt):
    """Exact discretisation of the IOU model over a step dt: returns (a, b, q11, q12, q22)."""
    if theta * dt < 1e-6:
        return dt, 1.0, sigma * sigma * dt ** 3 / 3.0, sigma * sigma * dt * dt / 2.0, sigma * sigma * dt
    b = math.exp(-theta * dt)
    a = (1.0 - b) / theta
    s2 = sigma * sigma
    q11 = s2 / (theta * theta) * (dt - 2.0 * a + (1.0 - b * b) / (2.0 * theta))
    q12 = s2 / (2.0 * theta * theta) * (1.0 - b) ** 2
    q22 = s2 / (2.0 * theta) * (1.0 - b * b)
    return a, b, q11, q12, q22


@njit(cache=True)
def kf_iou(y, dt, phase, theta, sigma, r, burn, want):
    """Kalman filter for one sequence.  y: (T,3); phase: (T,) int (phase used for the step k-1 -> k is phase[k-1]);
    theta, sigma: per-phase arrays.  Returns (nll, n, sse, pred, S, m_filt, P_filt) where pred/S are one-step
    predictive means/variances and m_filt (T,3,2), P_filt (T,3) [p11,p12,p22 identical for the 3 axes]."""
    T = y.shape[0]
    K = theta.shape[0]
    A = np.empty(K); B = np.empty(K); Q11 = np.empty(K); Q12 = np.empty(K); Q22 = np.empty(K)
    for k in range(K):
        A[k], B[k], Q11[k], Q12[k], Q22[k] = iou_FQ(theta[k], sigma[k], dt)
    r2 = r * r
    nll = 0.0; n = 0; sse = 0.0
    pred = np.empty((T, 3)) if want else np.empty((1, 3))
    Sout = np.empty(T) if want else np.empty(1)
    mf = np.empty((T, 3, 2)) if want else np.empty((1, 3, 2))
    Pf = np.empty((T, 3)) if want else np.empty((1, 3))
    m0 = np.empty(3); m1 = np.zeros(3)
    for d in range(3):
        m0[d] = y[0, d]
    p11 = r2; p12 = 0.0; p22 = 1.0            # velocity prior sd 1 m/s
    if want:
        for d in range(3):
            pred[0, d] = y[0, d]; mf[0, d, 0] = m0[d]; mf[0, d, 1] = 0.0
        Sout[0] = p11 + r2; Pf[0, 0] = p11; Pf[0, 1] = p12; Pf[0, 2] = p22
    for k in range(1, T):
        ph = phase[k - 1]
        a = A[ph]; b = B[ph]
        # predict
        n11 = p11 + 2.0 * a * p12 + a * a * p22 + Q11[ph]
        n12 = b * (p12 + a * p22) + Q12[ph]
        n22 = b * b * p22 + Q22[ph]
        S = n11 + r2
        k1 = n11 / S; k2 = n12 / S
        for d in range(3):
            pm0 = m0[d] + a * m1[d]
            pm1 = b * m1[d]
            e = y[k, d] - pm0
            if k >= burn:
                nll += 0.5 * (LOG2PI + math.log(S) + e * e / S)
                sse += e * e
                n += 1
            m0[d] = pm0 + k1 * e
            m1[d] = pm1 + k2 * e
            if want:
                pred[k, d] = pm0; mf[k, d, 0] = m0[d]; mf[k, d, 1] = m1[d]
        p11 = n11 - k1 * n11
        p12 = n12 - k1 * n12
        p22 = n22 - k2 * n12
        if want:
            Sout[k] = S; Pf[k, 0] = p11; Pf[k, 1] = p12; Pf[k, 2] = p22
    return nll, n, sse, pred, Sout, mf, Pf


@njit(cache=True)
def kf_rw(y, dt, q, r, burn):
    """Random-walk (hold-last) Kalman filter: returns (nll, n, sse)."""
    T = y.shape[0]
    r2 = r * r
    m = np.empty(3)
    for d in range(3):
        m[d] = y[0, d]
    p = r2
    nll = 0.0; n = 0; sse = 0.0
    for k in range(1, T):
        pp = p + q * q * dt
        S = pp + r2
        g = pp / S
        for d in range(3):
            e = y[k, d] - m[d]
            if k >= burn:
                nll += 0.5 * (LOG2PI + math.log(S) + e * e / S)
                sse += e * e
                n += 1
            m[d] += g * e
        p = pp - g * pp
    return nll, n, sse


def iou_predict(m, P, theta, sigma, tau):
    """Predictive mean/variance of the position at horizons tau (array) from filtered (m (3,2), P (3,)) ."""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    if theta < 1e-8:
        a = tau; b = np.ones_like(tau)
        q11 = sigma ** 2 * tau ** 3 / 3.0
    else:
        b = np.exp(-theta * tau); a = (1 - b) / theta
        q11 = sigma ** 2 / theta ** 2 * (tau - 2 * a + (1 - b * b) / (2 * theta))
    mean = m[None, :, 0] + a[:, None] * m[None, :, 1]
    var = P[0] + 2 * a * P[1] + a * a * P[2] + q11
    return mean, var


def iou_var(theta, sigma, tau):
    """Process-noise part of the position variance of the IOU model after a gap tau (per axis)."""
    tau = np.asarray(tau, dtype=float)
    if theta < 1e-8:
        return sigma ** 2 * tau ** 3 / 3.0
    b = np.exp(-theta * tau); a = (1 - b) / theta
    return sigma ** 2 / theta ** 2 * (tau - 2 * a + (1 - b * b) / (2 * theta))


# --------------------------------------------------------------------------- Gaussian HMM (phase model)
@njit(cache=True)
def _fb_core(b, A, pi):
    T, K = b.shape
    alpha = np.empty((T, K)); c = np.empty(T)
    s = 0.0
    for k in range(K):
        alpha[0, k] = pi[k] * b[0, k]; s += alpha[0, k]
    c[0] = s
    for k in range(K):
        alpha[0, k] /= s
    for t in range(1, T):
        s = 0.0
        for k in range(K):
            a = 0.0
            for i in range(K):
                a += alpha[t - 1, i] * A[i, k]
            a *= b[t, k]
            alpha[t, k] = a; s += a
        c[t] = s
        for k in range(K):
            alpha[t, k] /= s
    beta = np.ones((T, K))
    xi = np.zeros((K, K))
    for t in range(T - 2, -1, -1):
        for i in range(K):
            a = 0.0
            for k in range(K):
                a += A[i, k] * b[t + 1, k] * beta[t + 1, k]
            beta[t, i] = a / c[t + 1]
        for i in range(K):
            for k in range(K):
                xi[i, k] += alpha[t, i] * A[i, k] * b[t + 1, k] * beta[t + 1, k] / c[t + 1]
    gamma = alpha * beta
    for t in range(T):
        s = 0.0
        for k in range(K):
            s += gamma[t, k]
        for k in range(K):
            gamma[t, k] /= s
    return np.log(c).sum(), gamma, xi, alpha


def hmm_forward_backward(logb, logA, logpi):
    """Scaled forward-backward for one sequence.  logb: (T,K) emission log-densities.
    Returns (loglik, gamma (T,K), xi_sum (K,K), alpha_filtered (T,K))."""
    mx = logb.max(axis=1, keepdims=True)
    ll, gamma, xi, alpha = _fb_core(np.exp(logb - mx), np.exp(logA), np.exp(logpi))
    return ll + mx.sum(), gamma, xi, alpha


class GaussHMM(object):
    """K-state HMM with univariate Gaussian emissions fitted by EM on a list of sequences."""

    def __init__(self, K=3):
        self.K = K

    def _logb(self, x):
        return -0.5 * (LOG2PI + 2 * np.log(self.sd)[None, :] + ((x[:, None] - self.mu[None, :]) / self.sd[None, :]) ** 2)

    def fit(self, seqs, n_iter=200, tol=1e-5, stay=0.95, init=None):
        K = self.K
        allx = np.concatenate(seqs)
        if init is not None:                                       # warm start (used by the bootstrap)
            self.mu, self.sd = np.array(init['mu'], float), np.array(init['sd'], float)
            self.A, self.pi = np.array(init['A'], float) + 1e-12, np.array(init['pi'], float) + 1e-12
        else:
            self.mu = np.quantile(allx, (np.arange(K) + 0.5) / K)
            self.sd = np.full(K, allx.std() / K + 1e-3)
            self.A = np.full((K, K), (1 - stay) / (K - 1)); np.fill_diagonal(self.A, stay)
            self.pi = np.full(K, 1.0 / K)
        prev = -np.inf
        for it in range(n_iter):
            ll = 0.0
            g_sum = np.zeros(K); gx = np.zeros(K); gxx = np.zeros(K); xi = np.zeros((K, K)); pi = np.zeros(K)
            for x in seqs:
                l, g, x2, _ = hmm_forward_backward(self._logb(x), np.log(self.A), np.log(self.pi))
                ll += l; g_sum += g.sum(0); gx += g.T @ x; gxx += g.T @ (x * x); xi += x2; pi += g[0]
            self.mu = gx / g_sum
            self.sd = np.sqrt(np.maximum(gxx / g_sum - self.mu ** 2, 1e-6))
            self.A = xi / xi.sum(axis=1, keepdims=True)
            self.pi = pi / pi.sum()
            if ll - prev < tol * abs(ll):
                break
            prev = ll
        o = np.argsort(self.mu)                                   # order phases by mean speed
        self.mu, self.sd, self.pi = self.mu[o], self.sd[o], self.pi[o]
        self.A = self.A[o][:, o]
        self.loglik_ = ll
        self.n_iter_ = it + 1
        return self

    def loglik(self, seqs):
        return sum(hmm_forward_backward(self._logb(x), np.log(self.A), np.log(self.pi))[0] for x in seqs)

    def smooth(self, x):
        return hmm_forward_backward(self._logb(x), np.log(self.A), np.log(self.pi))[1]

    def filter(self, x):
        """Causal filtered phase probabilities P(z_t | x_1..t)."""
        return hmm_forward_backward(self._logb(x), np.log(self.A), np.log(self.pi))[3]

    def to_dict(self):
        return dict(K=self.K, mu=self.mu.tolist(), sd=self.sd.tolist(), A=self.A.tolist(), pi=self.pi.tolist())

    @classmethod
    def from_dict(cls, d):
        h = cls(d['K'])
        h.mu, h.sd, h.A, h.pi = np.array(d['mu']), np.array(d['sd']), np.array(d['A']), np.array(d['pi'])
        return h


# --------------------------------------------------------------------------- structural "level + damped spring" model
# Per axis:   xi_t = m_t + d_t + eps_t
#   dm = sigma_m dW_m                        (slowly drifting anchor / home position, estimated from history)
#   dd = u dt,  du = (-kappa d - theta u) dt + sigma_k dW   (damped, mean-reverting deviation; sigma depends on phase k)
def spring_FQ(kappa, theta, dt):
    """Exact discretisation of the 2x2 damped-spring block for unit sigma (Van Loan)."""
    from scipy.linalg import expm
    A = np.array([[0.0, 1.0], [-kappa, -theta]])
    G = np.array([[0.0, 0.0], [0.0, 1.0]])
    M = np.zeros((4, 4))
    M[:2, :2] = -A; M[:2, 2:] = G; M[2:, 2:] = A.T
    E = expm(M * dt)
    F = E[2:, 2:].T
    Q = F @ E[:2, 2:]
    return F, 0.5 * (Q + Q.T)


def build_struct(par, dt, A_phase, horizons):
    """Pre-compute the matrices used by kf_struct.  par: dict(kappa, theta, sigma (K,), sigma_m, r).
    Returns F (3,3), Q (K,3,3), Fh (nh,3,3), Qh (K,nh,3,3) with phase mixing along the Markov chain A_phase."""
    K = len(par['sigma'])
    F2, Q2 = spring_FQ(par['kappa'], par['theta'], dt)
    F = np.eye(3); F[1:, 1:] = F2
    Q = np.zeros((K, 3, 3))
    for k in range(K):
        Q[k, 0, 0] = par['sigma_m'] ** 2 * dt
        Q[k, 1:, 1:] = par['sigma'][k] ** 2 * Q2
    hmax = int(max(horizons)) if len(horizons) else 0
    Fh = np.zeros((len(horizons), 3, 3)); Qh = np.zeros((K, len(horizons), 3, 3))
    hs = {int(h): i for i, h in enumerate(horizons)}
    Fp = np.eye(3)
    Qacc = np.zeros((K, 3, 3))
    Ap = np.eye(K)                                   # phase distribution after i steps for each initial phase
    for i in range(1, hmax + 1):
        Qmix = np.einsum('kj,jab->kab', Ap, Q)       # expected one-step Q given initial phase
        for k in range(K):
            Qacc[k] = F @ Qacc[k] @ F.T + Qmix[k]
        Fp = F @ Fp
        Ap = Ap @ A_phase
        if i in hs:
            Fh[hs[i]] = Fp; Qh[:, hs[i]] = Qacc
    return F, Q, Fh, Qh


@njit(cache=True)
def kf_struct(y, phase, F, Q, r, Fh, Qh, hor, burn, m_init, P_init, want):
    """Kalman filter of the structural model for one sequence y (T,3) and multi-horizon predictive scores.
    phase[k] is the (causal) phase available at step k.  Returns
       nll_h (nh,), n_h (nh,), sse_h (nh,)  : Gaussian predictive NLL / squared error per horizon (all axes)
       z2 (T, nh)  : squared standardised 3-D error  ||e||^2 / s^2  (NaN where undefined)   [if want]
       err (T, nh) : 3-D error norm                                                    [if want]
       sd (T, nh)  : per-axis predictive sd                                              [if want]
       mf (T,3,3), Pf (T,3,3) : filtered means (axis, state) and covariance             [if want]"""
    T = y.shape[0]
    nh = hor.shape[0]
    r2 = r * r
    nll = np.zeros(nh); cnt = np.zeros(nh); sse = np.zeros(nh)
    if want:
        z2 = np.full((T, nh), np.nan); err = np.full((T, nh), np.nan); sdo = np.full((T, nh), np.nan)
        mf = np.zeros((T, 3, 3)); Pf = np.zeros((T, 3, 3))
    else:
        z2 = np.zeros((1, 1)); err = np.zeros((1, 1)); sdo = np.zeros((1, 1))
        mf = np.zeros((1, 3, 3)); Pf = np.zeros((1, 3, 3))
    m = np.zeros((3, 3))                    # m[axis, state]
    for d in range(3):
        m[d, 0] = y[0, d] if m_init[d] != m_init[d] else m_init[d]
        m[d, 1] = y[0, d] - m[d, 0]
    P = P_init.copy()
    Pn = np.zeros((3, 3)); FP = np.zeros((3, 3))
    for k in range(T):
        if k > 0:
            ph = phase[k - 1]
            # predict  P <- F P F' + Q
            for i in range(3):
                for j in range(3):
                    s = 0.0
                    for l in range(3):
                        s += F[i, l] * P[l, j]
                    FP[i, j] = s
            for i in range(3):
                for j in range(3):
                    s = 0.0
                    for l in range(3):
                        s += FP[i, l] * F[j, l]
                    Pn[i, j] = s + Q[ph, i, j]
            for d in range(3):
                a0 = m[d, 0]
                a1 = F[1, 1] * m[d, 1] + F[1, 2] * m[d, 2]
                a2 = F[2, 1] * m[d, 1] + F[2, 2] * m[d, 2]
                m[d, 0] = a0; m[d, 1] = a1; m[d, 2] = a2
        else:
            for i in range(3):
                for j in range(3):
                    Pn[i, j] = P[i, j]
        # update with H = [1, 1, 0]
        S = Pn[0, 0] + 2.0 * Pn[0, 1] + Pn[1, 1] + r2
        g0 = (Pn[0, 0] + Pn[0, 1]) / S; g1 = (Pn[1, 0] + Pn[1, 1]) / S; g2 = (Pn[2, 0] + Pn[2, 1]) / S
        for d in range(3):
            e = y[k, d] - m[d, 0] - m[d, 1]
            m[d, 0] += g0 * e; m[d, 1] += g1 * e; m[d, 2] += g2 * e
        h0 = Pn[0, 0] + Pn[1, 0]; h1 = Pn[0, 1] + Pn[1, 1]; h2 = Pn[0, 2] + Pn[1, 2]
        P[0, 0] = Pn[0, 0] - g0 * h0; P[0, 1] = Pn[0, 1] - g0 * h1; P[0, 2] = Pn[0, 2] - g0 * h2
        P[1, 0] = Pn[1, 0] - g1 * h0; P[1, 1] = Pn[1, 1] - g1 * h1; P[1, 2] = Pn[1, 2] - g1 * h2
        P[2, 0] = Pn[2, 0] - g2 * h0; P[2, 1] = Pn[2, 1] - g2 * h1; P[2, 2] = Pn[2, 2] - g2 * h2
        if want:
            for d in range(3):
                for i in range(3):
                    mf[k, d, i] = m[d, i]
            for i in range(3):
                for j in range(3):
                    Pf[k, i, j] = P[i, j]
        if k < burn:
            continue
        ph = phase[k]
        for ih in range(nh):
            h = hor[ih]
            if k + h >= T:
                continue
            # predictive variance of xi_{k+h}:  H (Fh P Fh' + Qh) H' + r^2
            v = r2
            for i in range(2):
                for j in range(2):
                    s = Qh[ph, ih, i, j]
                    for a in range(3):
                        for b in range(3):
                            s += Fh[ih, i, a] * P[a, b] * Fh[ih, j, b]
                    v += s
            e2 = 0.0
            for d in range(3):
                pm = m[d, 0] + Fh[ih, 1, 1] * m[d, 1] + Fh[ih, 1, 2] * m[d, 2]
                e = y[k + h, d] - pm
                e2 += e * e
            nll[ih] += 1.5 * (LOG2PI + math.log(v)) + 0.5 * e2 / v
            sse[ih] += e2
            cnt[ih] += 3.0
            if want:
                z2[k, ih] = e2 / v; err[k, ih] = math.sqrt(e2); sdo[k, ih] = math.sqrt(v)
    return nll, cnt, sse, z2, err, sdo, mf, Pf
