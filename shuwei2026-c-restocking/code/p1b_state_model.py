# -*- coding: utf-8 -*-
"""Problem 1, part B.  Operational-stage (phase) model and state-change model of the egocentric palm state.

Protocol: parameters are fitted on the 16 training environments, the model family and the number of
phases are selected on the 4 validation environments, the selected model is frozen
(results/frozen_p1_state.json) and only then evaluated on the 4 test environments.

Outputs (results/): p1_hmm_selection.csv, p1_model_selection.csv, p1_state_params.csv,
p1_state_metrics.csv, p1_state_by_env.csv, p1_state_failures.csv, frozen_p1_state.json
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, SPLITS, save_json
from features import Data
from ssm import GaussHMM
from statemodel import (BURN, DT, HOR, HOR_S, R_MEAS, assign_phases, par_from_vec, phase_feature, run_list,
                        score_runs)

LEVELS = (0.5, 0.8, 0.9, 0.95)
RNG = np.random.default_rng(20261120)


def metrics(err, sd):
    """err: 3-D error norms (m); sd: per-axis predictive sd (m, scalar or array)."""
    sd = np.broadcast_to(np.asarray(sd, dtype=float), err.shape)
    z2 = err ** 2 / sd ** 2
    out = dict(n=int(err.size), mean_mm=1e3 * err.mean(), median_mm=1e3 * np.median(err), q90_mm=1e3 * np.quantile(err, 0.9),
               rmse_mm=1e3 * np.sqrt((err ** 2).mean()),
               nll=float(np.mean(0.5 * (np.log(2 * np.pi) + 2 * np.log(sd)) + z2 / 6.0)))
    for p in LEVELS:
        out['cov%d' % round(100 * p)] = float((z2 <= chi2.ppf(p, 3)).mean())
    return out


def baseline_errors(D, runs):
    """Hold-last and constant-velocity errors on exactly the evaluation points used by the Kalman models."""
    hold = [[] for _ in HOR]; cv = [[] for _ in HOR]; meta = [[] for _ in HOR]
    for (j, r, a, b) in runs:
        x = D.xi[j][a:b]; T = b - a
        u = np.zeros_like(x); u[3:] = (x[3:] - x[:-3]) / (3 * DT)
        for ih, h in enumerate(HOR):
            k = np.arange(BURN, T - h)
            if len(k) == 0:
                continue
            d = x[k + h] - x[k]
            hold[ih].append(np.linalg.norm(d, axis=1))
            cv[ih].append(np.linalg.norm(d - u[k] * h * DT, axis=1))
            meta[ih].append(np.stack([np.full(len(k), r), np.full(len(k), j), a + k], axis=1))
    cat = lambda L: [np.concatenate(v) if len(v) else np.zeros(0) for v in L]
    return cat(hold), cat(cv), [np.concatenate(v) if len(v) else np.zeros((0, 3), dtype=int) for v in meta]


def model_errors(extra):
    """Concatenate per-run (z2, err, sd, ...) outputs of kf_struct into per-horizon arrays."""
    err = [[] for _ in HOR]; sd = [[] for _ in HOR]
    for (z2, e, s, mf, Pf) in extra:
        for ih in range(len(HOR)):
            m = ~np.isnan(e[:, ih])
            err[ih].append(e[m, ih]); sd[ih].append(s[m, ih])
    return [np.concatenate(v) for v in err], [np.concatenate(v) for v in sd]


def fit_model(D, runs, phase, K, kind, A_phase, p0):
    cnt = score_runs(D, runs, phase, par_from_vec(p0, K, kind), K, kind, A_phase)[1].sum(axis=0)

    def obj_runs(p):
        nll = score_runs(D, runs, phase, par_from_vec(p, K, kind), K, kind, A_phase)[0]
        return (nll / cnt[None, :]).sum(axis=1)

    def obj(p):
        if np.any(np.abs(p) > 12):
            return 1e6
        try:
            return float(obj_runs(p).sum())
        except Exception:
            return 1e6
    best = None
    for start in (p0, p0 + 0.4 * RNG.standard_normal(len(p0))):
        res = minimize(obj, start, method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-8, maxiter=4000, maxfev=4000))
        res = minimize(obj, res.x, method='Powell', options=dict(xtol=1e-4, ftol=1e-9))
        if best is None or res.fun < best.fun:
            best = res
    return best.x, best.fun, obj_runs


def sandwich(obj_runs, p_hat, clusters, eps=2e-3):
    """Cluster-robust (Godambe) covariance of the composite-likelihood estimator in the log-parameter scale."""
    p = len(p_hat)

    def grad_runs(q):
        g = np.zeros((len(clusters), p))
        for i in range(p):
            e = np.zeros(p); e[i] = eps
            g[:, i] = (obj_runs(q + e) - obj_runs(q - e)) / (2 * eps)
        return g
    g0 = grad_runs(p_hat)
    H = np.zeros((p, p))
    for j in range(p):
        e = np.zeros(p); e[j] = eps
        H[:, j] = (grad_runs(p_hat + e).sum(axis=0) - grad_runs(p_hat - e).sum(axis=0)) / (2 * eps)
    H = 0.5 * (H + H.T)
    uniq = np.unique(clusters)
    G = np.stack([g0[clusters == c].sum(axis=0) for c in uniq])
    J = G.T @ G * len(uniq) / (len(uniq) - 1)
    Hi = np.linalg.pinv(H)
    return Hi @ J @ Hi, len(uniq)


def main():
    t_start = time.time()
    D = Data()
    runs = run_list(D)
    n_frames = {s: int(sum(b - a for _, _, a, b in runs[s])) for s in SPLITS}
    print('runs / frames per split:', {s: (len(runs[s]), n_frames[s]) for s in SPLITS}, flush=True)
    seqs = {s: [phase_feature(D, j, a, b)[3:] for j, r, a, b in runs[s]] for s in SPLITS}
    env_tr = np.array([D.env_of_rid[r] for _, r, _, _ in runs['train']])

    # ------------------------------------------------------------------ 1. phase HMM: K = 2, 3, 4
    hmms, rows = {}, []
    for K in (2, 3, 4):
        h = GaussHMM(K).fit(seqs['train'], n_iter=200)
        hmms[K] = h
        npar = K * 2 + K * (K - 1) + (K - 1)
        ntr = sum(map(len, seqs['train']))
        rows.append(dict(K=K, n_par=npar, train_ll_per_frame=h.loglik_ / ntr, bic_train=-2 * h.loglik_ + npar * np.log(ntr),
                         val_ll_per_frame=h.loglik(seqs['validation']) / sum(map(len, seqs['validation'])),
                         mean_speed_mps=' / '.join('%.3f' % (10 ** m - 0.01) for m in h.mu),
                         dwell_s=' / '.join('%.2f' % (DT / (1 - a)) for a in np.diag(h.A))))
        print('HMM K=%d fitted (%d iterations)' % (K, h.n_iter_), flush=True)
    hmm_sel = pd.DataFrame(rows)

    # ------------------------------------------------------------------ 2. baselines (identical evaluation points)
    base = {s: baseline_errors(D, runs[s]) for s in SPLITS}
    sd_hold = [np.sqrt((e ** 2).mean() / 3.0) for e in base['train'][0]]       # per-axis sd, estimated on train
    sd_cv = [np.sqrt((e ** 2).mean() / 3.0) for e in base['train'][1]]

    # ------------------------------------------------------------------ 3. Kalman models fitted on train
    cand = [('DV', 1, 'DV'), ('MR', 1, 'MR'), ('MR-2ph', 2, 'MR'), ('MR-3ph', 3, 'MR'), ('MR-4ph', 4, 'MR')]
    fits = {}
    for name, K, kind in cand:
        t0 = time.time()
        ph, _ = assign_phases(D, runs, hmms[K]) if K > 1 else ([np.zeros(D.N, dtype=np.int64)] * 2, None)
        A = hmms[K].A if K > 1 else np.ones((1, 1))
        if kind == 'DV':
            p0 = np.log([4.0] + [0.35] * K)
        else:
            p0 = np.log([6.0, 6.0] + list(np.linspace(0.22, 0.5, K)) + [0.003]) if K > 1 else np.log([6.0, 6.0, 0.35, 0.003])
        x, f, obj_runs = fit_model(D, runs['train'], ph, K, kind, A, p0)
        fits[name] = dict(K=K, kind=kind, x=x, fun=f, phase=ph, A=A, obj_runs=obj_runs)
        print('%-7s fitted in %4.0fs  objective %.4f  par %s' % (name, time.time() - t0, f, np.round(np.exp(x), 4)), flush=True)

    # ------------------------------------------------------------------ 4. validation: model selection
    def evaluate(split, names):
        out = []
        h_e, c_e, _ = base[split]
        for ih, h in enumerate(HOR):
            out.append(dict(split=split, model='Hold', horizon_s=HOR_S[ih], **metrics(h_e[ih], sd_hold[ih])))
            out.append(dict(split=split, model='CV', horizon_s=HOR_S[ih], **metrics(c_e[ih], sd_cv[ih])))
        store = {}
        for name in names:
            ft = fits[name]
            ex = score_runs(D, runs[split], ft['phase'], par_from_vec(ft['x'], ft['K'], ft['kind']), ft['K'], ft['kind'], ft['A'], want=True)[3]
            err, sd = model_errors(ex)
            store[name] = (err, sd, ex)
            for ih in range(len(HOR)):
                assert len(err[ih]) == len(h_e[ih])
                out.append(dict(split=split, model=name, horizon_s=HOR_S[ih], **metrics(err[ih], sd[ih])))
        return pd.DataFrame(out), store

    val, vstore = evaluate('validation', [c[0] for c in cand])
    sel = val.groupby('model').agg(val_nll=('nll', 'mean'), val_mean_mm=('mean_mm', 'mean'), val_cov90=('cov90', 'mean')).reset_index()
    sel['n_par'] = sel['model'].map({'Hold': len(HOR), 'CV': len(HOR), **{n: len(fits[n]['x']) for n in fits}})
    # One-standard-error rule (fixed before looking at any test result): among the Kalman models, take the
    # simplest one whose validation score is within one standard error of the best; the standard error of the
    # paired score difference is obtained by a record-cluster bootstrap over the validation records.
    meta_v = base['validation'][2]

    def point_nll(name):
        err, sd, _ = vstore[name]
        return [0.5 * (np.log(2 * np.pi) + 2 * np.log(s_)) + e_ ** 2 / (6.0 * s_ ** 2) for e_, s_ in zip(err, sd)]
    pn = {n: point_nll(n) for n in fits}
    order_simple = ['DV', 'MR', 'MR-2ph', 'MR-3ph', 'MR-4ph']
    best_name = min(fits, key=lambda n: float(sel[sel.model == n]['val_nll'].iloc[0]))
    recs_v = np.unique(np.concatenate([m_[:, 0] for m_ in meta_v]))
    diff_se = {}
    for n in order_simple:
        per_rec = np.zeros((len(recs_v), len(HOR), 2))            # sums of (difference, count) per record and horizon
        for ih in range(len(HOR)):
            d_ = pn[n][ih] - pn[best_name][ih]
            for ir, rr in enumerate(recs_v):
                m_ = meta_v[ih][:, 0] == rr
                per_rec[ir, ih, 0] = d_[m_].sum(); per_rec[ir, ih, 1] = m_.sum()
        pick = RNG.integers(0, len(recs_v), size=(2000, len(recs_v)))
        tot = per_rec[pick].sum(axis=1)                              # (B, nh, 2)
        bs = (tot[:, :, 0] / np.maximum(tot[:, :, 1], 1)).mean(axis=1)
        full = (per_rec[:, :, 0].sum(axis=0) / per_rec[:, :, 1].sum(axis=0)).mean()
        diff_se[n] = (float(full), float(bs.std(ddof=1)))
    chosen = best_name
    for n in order_simple:
        if diff_se[n][0] <= diff_se[n][1] + 1e-12:
            chosen = n
            break
    sel['diff_to_best'] = sel['model'].map({n: v[0] for n, v in diff_se.items()})
    sel['se_diff'] = sel['model'].map({n: v[1] for n, v in diff_se.items()})
    sel = sel.sort_values('val_nll').reset_index(drop=True)
    sel['selected'] = sel['model'] == chosen
    print(sel.round(4).to_string(), '\nbest:', best_name, ' selected by the one-SE rule:', chosen, flush=True)

    # ------------------------------------------------------------------ 5. freeze
    ft = fits[chosen]
    par = par_from_vec(ft['x'], ft['K'], ft['kind'])
    cov, n_cl = sandwich(ft['obj_runs'], ft['x'], env_tr)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    names = ['kappa', 'theta'] + ['sigma_%d' % (k + 1) for k in range(ft['K'])] + ['sigma_m'] if ft['kind'] == 'MR' else \
            ['theta'] + ['sigma_%d' % (k + 1) for k in range(ft['K'])]
    units = {'kappa': '1/s^2', 'theta': '1/s', 'sigma_m': 'm/s^0.5'}
    prow = [dict(parameter=n, unit=units.get(n, 'm/s^1.5'), estimate=float(np.exp(v)), ci_low=float(np.exp(v - 1.96 * s)),
                 ci_high=float(np.exp(v + 1.96 * s)), se_log=float(s)) for n, v, s in zip(names, ft['x'], se)]
    # derived quantities: relaxation rates and stationary sd of the deviation
    kap, th = par['kappa'], par['theta']
    disc = th * th - 4 * kap
    rates = [(th - np.sqrt(disc)) / 2, (th + np.sqrt(disc)) / 2] if disc >= 0 else [th / 2, th / 2]
    hsel = hmms[ft['K']] if ft['K'] > 1 else None
    stat_pi = None
    if hsel is not None:
        w, v = np.linalg.eig(hsel.A.T)
        stat_pi = np.real(v[:, np.argmax(np.real(w))]); stat_pi = stat_pi / stat_pi.sum()
        sig2 = float((stat_pi * par['sigma'] ** 2).sum())
    else:
        sig2 = float(par['sigma'][0] ** 2)
    derived = dict(relaxation_rates_per_s=[float(x) for x in rates], damping_ratio=float(th / (2 * np.sqrt(kap))),
                   stationary_sd_axis_mm=float(1e3 * np.sqrt(sig2 / (2 * th * kap))),
                   stationary_phase_probabilities=None if stat_pi is None else [float(x) for x in stat_pi], n_clusters=int(n_cl))
    # HMM uncertainty: environment-cluster bootstrap with warm start
    boot = []
    if hsel is not None:
        envs = np.unique(env_tr)
        for bidx in range(100):
            pick = RNG.choice(envs, size=len(envs), replace=True)
            ss = [s for e in pick for s, ee in zip(seqs['train'], env_tr) if ee == e]
            hb = GaussHMM(hsel.K).fit(ss, n_iter=100, init=hsel.to_dict())
            boot.append(np.concatenate([10 ** hb.mu - 0.01, np.diag(hb.A)]))
        boot = np.array(boot)
    frozen = dict(model=chosen, kind=ft['kind'], K=int(ft['K']), params={k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in par.items()},
                  hmm=None if hsel is None else hsel.to_dict(), horizons_frames=HOR.tolist(), burn=BURN, r_meas=R_MEAS,
                  sd_hold=[float(x) for x in sd_hold], sd_cv=[float(x) for x in sd_cv], derived=derived,
                  hmm_boot_ci=None if hsel is None else dict(speed_lo=np.quantile(boot[:, :hsel.K], 0.025, axis=0).tolist(),
                                                             speed_hi=np.quantile(boot[:, :hsel.K], 0.975, axis=0).tolist(),
                                                             stay_lo=np.quantile(boot[:, hsel.K:], 0.025, axis=0).tolist(),
                                                             stay_hi=np.quantile(boot[:, hsel.K:], 0.975, axis=0).tolist()),
                  candidates={n: dict(kind=fits[n]['kind'], K=int(fits[n]['K']),
                                      params={k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
                                              for k, v in par_from_vec(fits[n]['x'], fits[n]['K'], fits[n]['kind']).items()},
                                      hmm=None if fits[n]['K'] == 1 else hmms[fits[n]['K']].to_dict()) for n in fits},
                  frozen_before_test=True)
    save_json(frozen, 'frozen_p1_state.json')
    pd.DataFrame(prow).to_csv(os.path.join(RESULTS, 'p1_state_params.csv'), index=False)
    hmm_sel.to_csv(os.path.join(RESULTS, 'p1_hmm_selection.csv'), index=False)
    sel.to_csv(os.path.join(RESULTS, 'p1_model_selection.csv'), index=False)
    print(pd.DataFrame(prow).round(4).to_string()); print(derived, flush=True)

    # ------------------------------------------------------------------ 6. test evaluation of the frozen model
    report_models = sorted({chosen, 'DV', 'MR'}, key=lambda n: ['DV', 'MR', 'MR-2ph', 'MR-3ph', 'MR-4ph'].index(n))
    tr, _ = evaluate('train', report_models)
    val2 = val[val.model.isin(['Hold', 'CV'] + report_models)]
    te, store = evaluate('test', report_models)
    allm = pd.concat([tr, val2, te], ignore_index=True)
    allm.to_csv(os.path.join(RESULTS, 'p1_state_metrics.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(te.round(3).to_string(), flush=True)
    # per-environment aggregation on validation + test (frozen model vs baselines)
    rows = []
    for split in ('validation', 'test'):
        h_e, c_e, meta = base[split]
        ex = score_runs(D, runs[split], ft['phase'], par, ft['K'], ft['kind'], ft['A'], want=True)[3]
        err, sd = model_errors(ex)
        for ih in range(len(HOR)):
            env = D.env_of_rid[meta[ih][:, 0]]
            for e in np.unique(env):
                m = env == e
                rows.append(dict(split=split, environment_id=e, horizon_s=HOR_S[ih], n=int(m.sum()), hold_mm=1e3 * h_e[ih][m].mean(),
                                 cv_mm=1e3 * c_e[ih][m].mean(), model_mm=1e3 * err[ih][m].mean(),
                                 cov90=float((err[ih][m] ** 2 / sd[ih][m] ** 2 <= chi2.ppf(0.9, 3)).mean())))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'p1_state_by_env.csv'), index=False)
    # failure cases: largest 0.5-s errors of the frozen model on the test split, with context
    ih = int(np.flatnonzero(HOR == 15)[0])
    h_e, c_e, meta = base['test']
    err, sd = store[chosen][0], store[chosen][1]
    order = np.argsort(-err[ih])
    fr, used = [], set()
    for o in order:
        r, j, gi = meta[ih][o]
        key = (r, j, gi // 30)                     # at most one case per second of record and hand
        if key in used:
            continue
        used.add(key)
        fr.append(dict(rid=int(r), hand='LR'[j], t_s=float(D.t[gi]), err_mm=1e3 * err[ih][o], hold_mm=1e3 * h_e[ih][o],
                       pred_sd_mm=1e3 * sd[ih][o], z=float(err[ih][o] / (np.sqrt(3) * sd[ih][o])), phase_at_origin=int(ft['phase'][j][gi]) + 1,
                       speed_at_origin=float(D.lagspeed[j][gi]), speed_after=float(D.lagspeed[j][gi + 15]),
                       phase_after=int(ft['phase'][j][gi + 15]) + 1, task=D.rec.task[r], dataset=D.rec.dataset[r]))
        if len(fr) >= 25:
            break
    pd.DataFrame(fr).to_csv(os.path.join(RESULTS, 'p1_state_failures.csv'), index=False)
    # error of the frozen model by phase at the prediction origin and by phase change within the horizon (test)
    gi = meta[ih][:, 2]; jj = meta[ih][:, 1]
    ph0 = np.array([ft['phase'][j][g] for j, g in zip(jj, gi)]); ph1 = np.array([ft['phase'][j][g + 15] for j, g in zip(jj, gi)])
    byph = [dict(phase_origin=int(k) + 1, phase_after=int(k2) + 1, n=int(((ph0 == k) & (ph1 == k2)).sum()),
                 err_mm=float(1e3 * err[ih][(ph0 == k) & (ph1 == k2)].mean()) if ((ph0 == k) & (ph1 == k2)).any() else np.nan,
                 hold_mm=float(1e3 * h_e[ih][(ph0 == k) & (ph1 == k2)].mean()) if ((ph0 == k) & (ph1 == k2)).any() else np.nan)
            for k in range(ft['K']) for k2 in range(ft['K'])]
    pd.DataFrame(byph).to_csv(os.path.join(RESULTS, 'p1_state_by_phase.csv'), index=False)
    print(pd.DataFrame(byph).round(1).to_string())
    print('done in %.0fs' % (time.time() - t_start))


if __name__ == '__main__':
    main()
