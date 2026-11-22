"""Fold-local fixed-effect ridge models for style-confounding sensitivity.

All transformations, imputations, and group means are fitted on training rows.
No causal interpretation is attached to the conditional coefficients.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy.special import ndtri
from scipy.linalg import eigh, cho_factor, cho_solve

ALPHAS = np.array([.001, .003, .01, .03, .1, .3, 1., 3., 10., 30., 100.])


class NormalScoreMap:
    """Median imputation and training empirical midpoint normal scores.

    Inverse predictions interpolate training knots and clip at observed ends.
    Ties have one midpoint score. Y has no missing values in this dataset.
    """
    def fit(self, x):
        x = np.asarray(x, float)
        self.median = np.nanmedian(x, axis=0)
        self.median = np.nan_to_num(self.median)
        filled = np.where(np.isfinite(x), x, self.median)
        self.sorted = np.sort(filled, axis=0)
        self.mean = np.zeros(x.shape[1])
        self.scale = np.ones(x.shape[1])
        z = self.transform(x)
        self.mean = z.mean(axis=0)
        self.scale = z.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.
        self.inverse_knots = []
        n = len(x)
        for j in range(x.shape[1]):
            values, count = np.unique(filled[:, j], return_counts=True)
            left = np.r_[0, np.cumsum(count)[:-1]]
            right = np.cumsum(count)
            score = ndtri((left+right+1.)/(2*(n+1)))
            score = (score-self.mean[j])/self.scale[j]
            self.inverse_knots.append((score, values))
        return self

    def transform(self, x):
        x = np.asarray(x, float)
        filled = np.where(np.isfinite(x), x, self.median)
        z = np.empty_like(filled)
        n = len(self.sorted)
        for j in range(x.shape[1]):
            left = np.searchsorted(self.sorted[:, j], filled[:, j], side="left")
            right = np.searchsorted(self.sorted[:, j], filled[:, j], side="right")
            z[:, j] = ndtri((left+right+1.)/(2*(n+1)))
        return (z-self.mean)/self.scale

    def inverse_transform(self, z):
        z = np.asarray(z, float)
        return np.column_stack([
            np.interp(z[:, j], score, values)
            for j, (score, values) in enumerate(self.inverse_knots)
        ])


class StandardMap:
    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        self.scale = np.std(x, axis=0)
        self.scale[self.scale < 1e-12] = 1.
        return self

    def transform(self, x):
        return (np.asarray(x)-self.mean)/self.scale

    def inverse_transform(self, z):
        return np.asarray(z)*self.scale+self.mean


def group_centers(values, styles):
    return {s:values[styles == s].mean(axis=0) for s in np.unique(styles)}


def lookup_centers(centers, styles, fallback):
    return np.stack([centers.get(s, fallback) for s in styles])


def prepare(x, y, styles, target_mode="rank"):
    styles = np.asarray(styles)
    xm = NormalScoreMap().fit(x)
    ym = (NormalScoreMap() if target_mode == "rank" else StandardMap()).fit(y)
    u, v = xm.transform(x), ym.transform(y)
    uc, vc = group_centers(u, styles), group_centers(v, styles)
    um, vm = u.mean(axis=0), v.mean(axis=0)
    u_style = lookup_centers(uc, styles, um)
    v_style = lookup_centers(vc, styles, vm)
    return dict(xmap=xm, ymap=ym, u=u, v=v, uc=uc, vc=vc, um=um, vm=vm,
        ur=u-u_style, vr=v-v_style, u_style=u_style, v_style=v_style,
        raw_y_centers=group_centers(y, styles), raw_y_mean=np.mean(y, axis=0),
        target_mode=target_mode, n=len(x))


def design(prep, within):
    return (prep["ur"], prep["vr"]) if within else (prep["u"]-prep["um"], prep["v"]-prep["vm"])


def ridge_path(prep, within, alphas=ALPHAS):
    u, v = design(prep, within)
    gram = u.T@u/len(u)
    xy = u.T@v/len(u)
    eigenvalues, q = eigh(gram, check_finite=False)
    eigenvalues = np.maximum(eigenvalues, 0.)
    qtxy = q.T@xy
    return np.stack([q@(qtxy/(eigenvalues[:, None]+a)) for a in alphas])


def ridge_fit(prep, within, alpha):
    u, v = design(prep, within)
    gram = u.T@u/len(u)+float(alpha)*np.eye(u.shape[1])
    return cho_solve(cho_factor(gram, lower=True, check_finite=False),
                     u.T@v/len(u), check_finite=False)


def predict_scores(prep, coefficient, x, styles, within):
    u = prep["xmap"].transform(x)
    if within:
        center_u = lookup_centers(prep["uc"], styles, prep["um"])
        offset = lookup_centers(prep["vc"], styles, prep["vm"])
        u = u-center_u
    else:
        u = u-prep["um"]
        offset = prep["vm"]
    if coefficient.ndim == 3:
        return np.einsum("np,apq->anq", u, coefficient, optimize=True)+offset
    return u@coefficient+offset


def predict(prep, coefficient, x, styles, within):
    score = predict_scores(prep, coefficient, x, styles, within)
    return prep["ymap"].inverse_transform(score), score


def grouped_style_folds(ids, styles, n_splits=3, seed=20261024):
    """Split unique beer IDs, approximately balancing each style.

    Repeated bootstrap copies of one beer are kept in one fold. A style with
    only one distinct sampled beer has no possible same-style training donor
    when that beer is held out; prediction then uses training grand means.
    """
    ids, styles = np.asarray(ids), np.asarray(styles)
    rng = np.random.default_rng(seed)
    assignments = {}
    for style in np.unique(styles):
        unique_ids = np.unique(ids[styles == style]).copy()
        rng.shuffle(unique_ids)
        offset = int(rng.integers(n_splits))
        for pos, beer in enumerate(unique_ids):
            assignments[int(beer)] = (offset+pos) % n_splits
    fold = np.array([assignments[int(beer)] for beer in ids])
    result = []
    for k in range(n_splits):
        tr, va = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        if not len(tr) or not len(va):
            raise ValueError("Empty grouped inner split")
        if set(ids[tr]) & set(ids[va]):
            raise AssertionError("A bootstrap beer leaked across inner folds")
        result.append((tr, va))
    return result


def choose_alphas(x, y, styles, ids, seed=20261024, target_mode="rank", alphas=ALPHAS):
    losses = np.zeros((2, len(alphas)))
    missing_style_validation_rows = 0
    folds = grouped_style_folds(ids, styles, seed=seed)
    total_validation_rows = sum(len(va) for _, va in folds)
    for tr, va in folds:
        prep = prepare(x[tr], y[tr], styles[tr], target_mode)
        truth = prep["ymap"].transform(y[va])
        missing_style_validation_rows += int(sum(s not in prep["vc"] for s in styles[va]))
        for k, within in enumerate((False, True)):
            coefficients = ridge_path(prep, within, alphas)
            prediction = predict_scores(prep, coefficients, x[va], styles[va], within)
            losses[k] += ((prediction-truth[None])**2).sum(axis=(1, 2))
    losses /= total_validation_rows*y.shape[1]
    index = losses.argmin(axis=1)
    return dict(total=float(alphas[index[0]]), within=float(alphas[index[1]])), losses, dict(
        inner_missing_style_validation_rows=missing_style_validation_rows,
        unique_training_beers=len(np.unique(ids)),
        inner_group_leakage_checks_passed=True,
    )


def stratified_bootstrap_indices(styles, rng):
    return np.concatenate([
        rng.choice(np.flatnonzero(styles == s), int(np.sum(styles == s)), replace=True)
        for s in np.unique(styles)
    ])


def top_indices(coefficient, eligible, k=3):
    return eligible[np.argsort(-np.abs(coefficient[eligible]), axis=0, kind="stable")[:k]]
