# -*- coding: utf-8 -*-
"""Problem 3 - duration model: estimation on the training environments, selection on validation, held-out error.

Model for the annotated duration d of a work unit of family f sampled from an unseen pool:
    log d = mu_f + shift_g + shift_f + sigma * e ,     e ~ empirical residual distribution (mean 0, variance 1)
    shift_g ~ N(0, tau_g^2), shift_f ~ N(0, tau_f^2)   pool-level shifts (unknown for a new pool, learned on line
                                                        from execution feedback inside one trial; reset every trial)
Optional prefix-feature variants add  beta_L' z(x_L)  with ridge-estimated coefficients for the prefix level L.

Outputs (results/): p3_duration_selection.csv, p3_duration_heldout.csv, p3_duration_online.csv, frozen_p3_duration.json
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PALM_IDX, RESULTS, TIP_IDX, load_json, quat_to_rot, save_json
from p3_core import PREFIX_LEVELS, PrefixStore, Scenario

warnings.filterwarnings('ignore')
FAMS = ('P', 'A', 'L')
FEATS = ['lsp_max', 'lsp_min', 'disp_max', 'reach_max', 'hz_min', 'hz_act', 'ap_act', 'ap_pas', 'hand_dist', 'lhead_sp',
         'lhead_ang', 'pitch']
QS = (0.25, 0.5, 0.75, 0.9)
RNG = np.random.default_rng(20261124)


def prefix_features(px):
    """Translation/rotation-invariant features of a prefix (egocentric state of Problem 1).  Absolute timestamps,
    world coordinates and sampling rate are deliberately not used."""
    t = px['t'] - px['t'][0]
    n = len(t)
    Rh = quat_to_rot(px['headq'].astype(float))
    head = px['head'].astype(float)
    f = {}
    sp, reach, hz, disp, ap = [], [], [], [], []
    xi = []
    for key in ('L', 'R'):
        kp = px[key].astype(float)
        c = kp[:, PALM_IDX].mean(axis=1)
        x = np.einsum('nji,nj->ni', Rh, c - head)
        xi.append(x)
        v = np.linalg.norm(np.diff(x, axis=0), axis=1) / np.diff(t) if n > 1 else np.zeros(1)
        sp.append(v.mean()); disp.append(np.linalg.norm(x[-1] - x[0])); reach.append(np.linalg.norm(x, axis=1).mean())
        hz.append(x[:, 2].mean()); ap.append(np.linalg.norm(kp[:, 4] - kp[:, 8], axis=1).mean())
    act = int(np.argmax(sp))
    f['lsp_max'] = np.log(max(sp) + 0.01); f['lsp_min'] = np.log(min(sp) + 0.01); f['disp_max'] = max(disp)
    f['reach_max'] = max(reach); f['hz_min'] = min(hz); f['hz_act'] = hz[act]; f['ap_act'] = ap[act]; f['ap_pas'] = ap[1 - act]
    f['hand_dist'] = np.linalg.norm(xi[0] - xi[1], axis=1).mean()
    if n > 1:
        hv = np.linalg.norm(np.diff(head, axis=0), axis=1) / np.diff(t)
        dR = np.einsum('nij,nkj->nik', Rh[1:], Rh[:-1])
        ang = np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1)) / np.diff(t)
        f['lhead_sp'] = np.log(hv.mean() + 0.01); f['lhead_ang'] = np.log(ang.mean() + 0.01)
    else:
        f['lhead_sp'] = np.log(0.01); f['lhead_ang'] = np.log(0.01)
    f['pitch'] = Rh[:, 2, 0].mean()
    f['flag_share'] = float((px['flag'] > 0).any(axis=1).mean())
    return f


def segment_table(scen, store):
    rows = []
    for u in scen.index:
        if u['eligible'] != '1':
            continue
        row = dict(unit_id=u['unit_id'], family=u['family_id'], split=u['split'], env=u['environment_id'], rec=u['episode_uuid'],
                   dataset=u['dataset'], d=float(u['duration_s']))
        for li, L in enumerate(PREFIX_LEVELS):
            f = prefix_features(store.get(u, L))
            for k, v in f.items():
                row['%s_%d' % (k, li)] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    df['ld'] = np.log(df['d'])
    return df


def family_means(df, shrink=True):
    """Family means of log d; with shrinkage towards the grand mean of family means (few-sample families)."""
    n = {f: int((df.family == f).sum()) for f in FAMS}
    have = [f for f in FAMS if n[f] > 0]
    mu = {f: df[df.family == f].ld.mean() for f in have}
    s2 = float(np.mean([(df[df.family == f].ld - mu[f]).var(ddof=1) for f in have if n[f] > 1]))
    grand = float(np.mean([mu[f] for f in have]))
    if shrink and len(have) > 1:
        tau2 = max(np.var([mu[f] for f in have], ddof=1) - np.mean([s2 / n[f] for f in have]), 1e-4)
        mu = {f: grand + tau2 / (tau2 + s2 / n[f]) * (mu[f] - grand) for f in have}
    for f in FAMS:
        mu.setdefault(f, grand)
    return mu, n


def ridge_fit(X, y, alpha):
    m, s = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - m) / s
    b = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ y)
    return dict(mean=m, scale=s, coef=b)


def ridge_pred(mod, X):
    return ((X - mod['mean']) / mod['scale']) @ mod['coef']


def fit_models(tr, alphas=(10, 30, 100, 300, 1000)):
    """Family model + ridge feature models per prefix level (alpha by environment-grouped CV on train)."""
    envs = tr.env.unique()
    folds = np.array_split(RNG.permutation(envs), 8)
    mu, n = family_means(tr)
    base = tr.family.map(mu).to_numpy()
    r0 = tr.ld.to_numpy() - base
    # out-of-fold residuals of the family model (family means re-estimated without the held-out environments)
    oof0 = np.zeros(len(tr))
    for fd in folds:
        te = tr.env.isin(fd).to_numpy()
        mu_k, _ = family_means(tr[~te])
        oof0[te] = tr.ld.to_numpy()[te] - tr[te].family.map(mu_k).to_numpy()
    out = dict(mu=mu, n=n, oof=dict(family=oof0), ridge={}, alpha={})
    for li in range(len(PREFIX_LEVELS)):
        X = tr[['%s_%d' % (f, li) for f in FEATS]].to_numpy()
        best = None
        for a in alphas:
            oof = np.zeros(len(tr))
            for fd in folds:
                te = tr.env.isin(fd).to_numpy()
                mu_k, _ = family_means(tr[~te])
                rk = tr.ld.to_numpy()[~te] - tr[~te].family.map(mu_k).to_numpy()
                mod = ridge_fit(X[~te], rk, a)
                oof[te] = tr.ld.to_numpy()[te] - tr[te].family.map(mu_k).to_numpy() - ridge_pred(mod, X[te])
            sc = np.mean(np.abs(oof))
            if best is None or sc < best[0]:
                best = (sc, a, oof)
        out['ridge'][li] = ridge_fit(X, r0, best[1]); out['alpha'][li] = best[1]; out['oof']['feat%d' % li] = best[2]
    return out


def predict_center(mods, df, kind):
    base = df.family.map(mods['mu']).to_numpy()
    if kind == 'pooled':
        return np.full(len(df), mods['pooled_mu'])
    if kind == 'family':
        return base
    li = int(kind[-1])
    X = df[['%s_%d' % (f, li) for f in FEATS]].to_numpy()
    return base + ridge_pred(mods['ridge'][li], X)


def pinball(y, q, tau):
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def heldout_metrics(mods, df, kind, shift=0.0):
    c = predict_center(mods, df, kind) + shift
    r = mods['oof']['family' if kind in ('family', 'pooled') else kind] if kind != 'pooled' else mods['oof_pooled']
    y = df.ld.to_numpy(); d = df.d.to_numpy()
    out = dict(n=len(df), mae_log=float(np.mean(np.abs(y - c))), bias_log=float(np.mean(y - c)),
               mae_s=float(np.mean(np.abs(d - np.exp(c + np.median(r))))), medae_s=float(np.median(np.abs(d - np.exp(c + np.median(r))))))
    for tau in QS:
        q = np.exp(c + np.quantile(r, tau))
        out['pin%02d_s' % round(100 * tau)] = pinball(d, q, tau); out['cov%02d' % round(100 * tau)] = float(np.mean(d <= q))
    out['pin_mean_s'] = float(np.mean([out['pin%02d_s' % round(100 * tau)] for tau in QS[:3]]))
    return out


def shift_variances(tr, n_pools=400, envs_per_pool=4):
    """Pool emulation on the training environments: hold out 4 environments, fit the family means on the others and
    estimate the variance components of the pool shift by maximum likelihood."""
    envs = tr.env.unique()
    pools = []
    for _ in range(n_pools):
        pick = RNG.choice(envs, size=envs_per_pool, replace=False)
        te = tr.env.isin(pick).to_numpy()
        if te.sum() < 8:
            continue
        mu_k, _ = family_means(tr[~te])
        r = tr.ld.to_numpy()[te] - tr[te].family.map(mu_k).to_numpy()
        fam = tr[te].family.to_numpy()
        pools.append((r, np.array([FAMS.index(f) for f in fam])))

    def nll(p):
        tg2, tf2, s2 = np.exp(p)
        tot = 0.0
        for r, fi in pools:
            n = len(r)
            C = s2 * np.eye(n) + tg2 * np.ones((n, n)) + tf2 * (fi[:, None] == fi[None, :])
            sign, ld = np.linalg.slogdet(C)
            tot += 0.5 * (ld + r @ np.linalg.solve(C, r))
        return tot / len(pools)
    res = minimize(nll, np.log([0.05, 0.05, 0.4]), method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-6, maxiter=600))
    tg2, tf2, s2 = np.exp(res.x)
    return float(np.sqrt(tg2)), float(np.sqrt(tf2)), float(np.sqrt(s2)), len(pools)


class OnlineDuration(object):
    """Predictive distribution of log d per job with on-line learning of the pool shift (reset for every trial)."""

    def __init__(self, fz, learn=True):
        self.mu = fz['mu']; self.sigma = fz['sigma']; self.resid = np.asarray(fz['resid_std'])
        self.learn = learn
        tg2, tf2 = fz['tau_g'] ** 2, fz['tau_f'] ** 2
        self.m = np.zeros(4)                                            # (shift_g, shift_P, shift_A, shift_L)
        self.V = np.diag([tg2, tf2, tf2, tf2])
        self.n_obs = 0

    def _h(self, fam):
        h = np.zeros(4); h[0] = 1.0; h[1 + FAMS.index(fam)] = 1.0
        return h

    def update(self, fam, log_d, center):
        """Kalman update of the shift vector with one realised duration (center = model centre without shift)."""
        self.n_obs += 1
        if not self.learn:
            return
        h = self._h(fam)
        S = h @ self.V @ h + self.sigma ** 2
        K = self.V @ h / S
        self.m = self.m + K * (log_d - center - h @ self.m)
        self.V = self.V - np.outer(K, h @ self.V)

    def shift(self, fam):
        h = self._h(fam)
        return float(h @ self.m), float(h @ self.V @ h)

    def sample(self, fams, centers, n_scen, rng):
        """Scenario matrix (n_scen, n_jobs) of log durations: shared pool-shift draw per scenario + bootstrap residuals."""
        Lc = np.linalg.cholesky(self.V + 1e-12 * np.eye(4))
        sh = self.m[None, :] + rng.standard_normal((n_scen, 4)) @ Lc.T
        H = np.stack([self._h(f) for f in fams])                        # (n_jobs, 4)
        e = self.resid[rng.integers(0, len(self.resid), size=(n_scen, len(fams)))]
        return np.asarray(centers)[None, :] + sh @ H.T + self.sigma * e


def main():
    scen = Scenario(); store = PrefixStore()
    seg = segment_table(scen, store)
    seg.to_csv(os.path.join(RESULTS, 'p3_segment_features.csv'), index=False)
    tr, va, te = (seg[seg.split == s].reset_index(drop=True) for s in ('train', 'validation', 'test'))
    mods = fit_models(tr)
    mods['pooled_mu'] = float(tr.ld.mean())
    mods['oof_pooled'] = tr.ld.to_numpy() - mods['pooled_mu']
    kinds = ['pooled', 'family', 'feat0', 'feat1', 'feat2']
    label = {'pooled': 'pooled constant', 'family': 'family constant', 'feat0': 'family + prefix 0.2 s', 'feat1': 'family + prefix 0.6 s',
             'feat2': 'family + prefix 1.0 s'}
    rows = []
    for k in kinds:
        r = mods['oof_pooled'] if k == 'pooled' else mods['oof']['family' if k == 'family' else k]
        row = dict(model=label[k], kind=k, n_par={'pooled': 1, 'family': 3}.get(k, 3 + len(FEATS)), cv_mae_log=float(np.mean(np.abs(r))),
                   cv_sd_log=float(np.std(r)))
        for s, d in (('val', va), ('test', te)):
            for kk, v in heldout_metrics(mods, d, k).items():
                row['%s_%s' % (s, kk)] = v
        rows.append(row)
    sel = pd.DataFrame(rows)
    # one-standard-error rule on validation (record-cluster bootstrap of the mean pinball loss), simplest model first
    def point_loss(k):
        c = predict_center(mods, va, k)
        r = mods['oof_pooled'] if k == 'pooled' else mods['oof']['family' if k == 'family' else k]
        d = va.d.to_numpy()
        return np.mean([np.maximum(t * (d - np.exp(c + np.quantile(r, t))), (t - 1) * (d - np.exp(c + np.quantile(r, t)))) for t in QS[:3]], axis=0)
    PL = {k: point_loss(k) for k in kinds}
    best = min(kinds, key=lambda k: PL[k].mean())
    recs = va.rec.to_numpy(); ur = np.unique(recs)
    se = {}
    for k in kinds:
        d = PL[k] - PL[best]
        sums = np.array([d[recs == r].sum() for r in ur]); cnt = np.array([(recs == r).sum() for r in ur])
        pick = RNG.integers(0, len(ur), size=(4000, len(ur)))
        bs = sums[pick].sum(axis=1) / cnt[pick].sum(axis=1)
        se[k] = (float(d.mean()), float(bs.std(ddof=1)))
    chosen = best
    for k in kinds:                                                   # ordered from simplest to most complex
        if se[k][0] <= se[k][1] + 1e-12:
            chosen = k
            break
    if chosen == 'pooled':
        chosen = 'family'          # the job family is a public attribute of every job and is always kept
    sel['val_diff_to_best'] = sel['kind'].map({k: v[0] for k, v in se.items()})
    sel['val_se_diff'] = sel['kind'].map({k: v[1] for k, v in se.items()})
    sel['selected'] = sel['kind'] == chosen
    sel.to_csv(os.path.join(RESULTS, 'p3_duration_selection.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
    print(sel.round(3).T.to_string())
    print('best on validation:', best, ' selected (one-SE rule):', chosen)

    # ---------------------------------------------------------------- pool-shift hyper-parameters (training environments only)
    tau_g, tau_f, sigma, npools = shift_variances(tr)
    r_fam = mods['oof']['family']
    resid_std = (r_fam - r_fam.mean()) / r_fam.std()
    print('pool emulation (%d pseudo-pools): tau_g=%.3f tau_f=%.3f sigma=%.3f ; oof sd of family model %.3f' % (npools, tau_g, tau_f, sigma, r_fam.std()))
    frozen = dict(selected=chosen, mu=mods['mu'], n_train=mods['n'], sigma=sigma, tau_g=tau_g, tau_f=tau_f, resid_std=resid_std.tolist(),
                  features=FEATS, ridge={str(li): dict(mean=mods['ridge'][li]['mean'].tolist(), scale=mods['ridge'][li]['scale'].tolist(),
                                                       coef=mods['ridge'][li]['coef'].tolist(), alpha=float(mods['alpha'][li]),
                                                       oof_sd=float(np.std(mods['oof']['feat%d' % li]))) for li in range(len(PREFIX_LEVELS))},
                  oof_sd_family=float(r_fam.std()), frozen_before_test=True)
    save_json(frozen, 'frozen_p3_duration.json')

    # ---------------------------------------------------------------- held-out error with and without on-line shift learning
    rows = []
    for s, d in (('validation', va), ('test', te)):
        for k in kinds:
            rows.append(dict(split=s, model=label[k], learning='none', **heldout_metrics(mods, d, k)))
        # sequential replay: jobs arrive in random order from the pool (with replacement), 16 per sequence
        y = d.ld.to_numpy(); fam = d.family.to_numpy(); cen = d.family.map(mods['mu']).to_numpy()
        err_static, err_online, cover = [[] for _ in range(16)], [[] for _ in range(16)], [[] for _ in range(16)]
        for rep in range(3000):
            od = OnlineDuration(frozen, learn=True)
            idx = RNG.integers(0, len(d), size=16)
            for i, ii in enumerate(idx):
                sh, vs = od.shift(fam[ii])
                err_static[i].append(abs(y[ii] - cen[ii])); err_online[i].append(abs(y[ii] - cen[ii] - sh))
                q80 = cen[ii] + sh + np.sqrt(vs + sigma ** 2) * np.quantile(resid_std, 0.8)
                cover[i].append(y[ii] <= q80)
                od.update(fam[ii], y[ii], cen[ii])
        for i in (0, 1, 3, 7, 11, 15):
            rows.append(dict(split=s, model='family constant', learning='on-line, after %d feedbacks' % i, n=3000,
                             mae_log=float(np.mean(err_online[i])), bias_log=np.nan, mae_static_log=float(np.mean(err_static[i])),
                             cov80=float(np.mean(cover[i]))))
    ho = pd.DataFrame(rows)
    ho.to_csv(os.path.join(RESULTS, 'p3_duration_heldout.csv'), index=False)
    print(ho.round(3).to_string())


if __name__ == '__main__':
    main()
