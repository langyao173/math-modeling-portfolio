# -*- coding: utf-8 -*-
"""Problem 2, part A.  Artificial masking (open loop): error growth, calibrated uncertainty, admissible horizons.

Experimental conditions (fixed in advance, identical for every compared method)
    * state: egocentric palm centre xi (frozen definition of Problem 1); reference = recorded xi on reference-grade frames
    * a mask hides ALL samples of the target hand after the last visible frame k0 (positions, orientations and every
      feature derived from them); information available to the predictors = samples of that hand up to k0
    * window onsets every 0.5 s (15 frames) after at least 1 s of visible history, inside reference runs of 30-Hz records
    * missing durations 0.2, 0.5, 1, 2, 3 s; horizons evaluated at every frame up to 3 s
    * models and all tuning use train; conformal calibration uses validation; test is used for evaluation only, after freezing

Outputs (results/): p2_growth.csv, p2_masking_summary.csv, p2_coverage.csv, p2_radius_curves.csv, p2_admissible.csv, p2_history.csv,
p2_by_env.csv, frozen_p2_calibration.json
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, SPLITS, save_json
from features import Data
from ssm import GaussHMM, build_struct, kf_struct
from statemodel import DT, P_INIT, SPEED_EPS, assign_phases, load_frozen_state, p_init, run_list

HMAX = 90
HOR2 = np.arange(1, HMAX + 1, dtype=np.int64)
ONSET_STEP = 15
HIST_MIN = 30
WINDOWS = {0.2: 6, 0.5: 15, 1.0: 30, 2.0: 60, 3.0: 90}
LEVELS = (0.5, 0.8, 0.9, 0.95)


def collect(D, runs, phase, hmm, cand, hist=None):
    """Open-loop predictions at the onset grid.  Returns dict of arrays with one row per (onset) and HMAX columns.
    hist: None = full visible history, otherwise the filter (and the phase filter) are restarted hist frames before k0."""
    out = dict(hold=[], cv=[], err=[], sd=[], meta=[], phase=[])
    K = cand['K']; par = dict(cand['params']); par['sigma'] = np.array(par['sigma'])
    A = np.array(hmm.A) if K > 1 else np.ones((1, 1))
    F, Q, Fh, Qh = build_struct(par, DT, A, HOR2)
    Pi = p_init(cand['kind'])
    for (j, r, a, b) in runs:
        x = D.xi[j][a:b]; T = b - a
        k0s = np.arange(HIST_MIN, T - 1, ONSET_STEP)
        if len(k0s) == 0:
            continue
        ph_full = phase[j][a:b] if K > 1 else np.zeros(T, dtype=np.int64)
        if hist is None:
            o = kf_struct(x, ph_full, F, Q, par['r'], Fh, Qh, HOR2, HIST_MIN, np.full(3, np.nan), Pi, True)
            err, sd = o[4][k0s], o[5][k0s]
            ph0 = ph_full[k0s]
        else:
            err = np.full((len(k0s), HMAX), np.nan); sd = np.full((len(k0s), HMAX), np.nan); ph0 = np.zeros(len(k0s), dtype=np.int64)
            for i, k0 in enumerate(k0s):
                s0 = k0 - hist
                xs = x[s0:min(k0 + HMAX + 1, T)]
                if K > 1:
                    lag = min(3, hist)
                    sp = np.zeros(len(xs))
                    sp[lag:] = np.linalg.norm(xs[lag:] - xs[:-lag], axis=1) / (lag * DT)
                    sp[:lag] = sp[lag]
                    phs = hmm.filter(np.log10(sp + SPEED_EPS)).argmax(axis=1)
                    # frames after k0 are hidden: the phase array is only read at index <= hist by the filter
                else:
                    phs = np.zeros(len(xs), dtype=np.int64)
                o = kf_struct(xs, phs, F, Q, par['r'], Fh, Qh, HOR2, hist, np.full(3, np.nan), Pi, True)
                err[i, :], sd[i, :] = o[4][hist], o[5][hist]
                ph0[i] = phs[hist]
        u = np.zeros_like(x); u[3:] = (x[3:] - x[:-3]) / (3 * DT)
        hold = np.full((len(k0s), HMAX), np.nan); cv = np.full((len(k0s), HMAX), np.nan)
        for i, k0 in enumerate(k0s):
            n = min(HMAX, T - 1 - k0)
            d = x[k0 + 1:k0 + 1 + n] - x[k0]
            hold[i, :n] = np.linalg.norm(d, axis=1)
            cv[i, :n] = np.linalg.norm(d - u[k0][None, :] * (np.arange(1, n + 1) * DT)[:, None], axis=1)
        out['hold'].append(hold); out['cv'].append(cv); out['err'].append(err); out['sd'].append(sd)
        out['meta'].append(np.stack([np.full(len(k0s), r), np.full(len(k0s), j), a + k0s], axis=1)); out['phase'].append(ph0)
    return {k: np.concatenate(v) for k, v in out.items()}


def summarise(e):
    m = ~np.isnan(e)
    return dict(n=int(m.sum()), mean_mm=1e3 * np.nanmean(e), median_mm=1e3 * np.nanmedian(e), q90_mm=1e3 * np.nanquantile(e, 0.9),
                rmse_mm=1e3 * np.sqrt(np.nanmean(e ** 2)), q25_mm=1e3 * np.nanquantile(e, 0.25), q75_mm=1e3 * np.nanquantile(e, 0.75))


def main():
    t0 = time.time()
    D = Data()
    fz, hmm, par = load_frozen_state()
    runs = run_list(D)
    phase, _ = assign_phases(D, runs, hmm)
    main_c = fz['candidates'][fz['model']]
    K = main_c['K']
    res = {s: collect(D, runs[s], phase, hmm, main_c) for s in SPLITS}
    alt = {}
    for name in ('DV', 'MR'):
        alt[name] = {s: collect(D, runs[s], phase, hmm, fz['candidates'][name]) for s in SPLITS}
    print('onsets:', {s: len(res[s]['meta']) for s in SPLITS}, ' (%.0fs)' % (time.time() - t0), flush=True)

    # ------------------------------------------------------------------ error growth by horizon / phase
    grow = []
    for s in SPLITS:
        R = res[s]
        for name, E in (('Hold', R['hold']), ('CV', R['cv']), ('DV', alt['DV'][s]['err']), ('MR', alt['MR'][s]['err']),
                        (fz['model'], R['err'])):
            for h in range(HMAX):
                grow.append(dict(split=s, method=name, phase='all', horizon_s=(h + 1) * DT, **summarise(E[:, h])))
            if name in ('Hold', fz['model']):
                for k in range(K):
                    m = R['phase'] == k
                    for h in range(HMAX):
                        grow.append(dict(split=s, method=name, phase=str(k + 1), horizon_s=(h + 1) * DT, **summarise(E[m, h])))
        for h in range(HMAX):                                   # model-implied mean error (Gaussian): sd * E[chi_3]
            grow.append(dict(split=s, method='model-implied', phase='all', horizon_s=(h + 1) * DT, n=int((~np.isnan(R['sd'][:, h])).sum()),
                             mean_mm=1e3 * np.nanmean(R['sd'][:, h]) * 2 * np.sqrt(2 / np.pi), median_mm=np.nan, q90_mm=np.nan,
                             rmse_mm=1e3 * np.sqrt(3 * np.nanmean(R['sd'][:, h] ** 2))))
    grow = pd.DataFrame(grow)
    grow.to_csv(os.path.join(RESULTS, 'p2_growth.csv'), index=False)

    # ------------------------------------------------------------------ masking windows of fixed duration
    summ = []
    for s in SPLITS:
        R = res[s]
        for wname, W in WINDOWS.items():
            full = ~np.isnan(R['hold'][:, W - 1])                      # windows that fit entirely inside the run
            for name, E in (('Hold', R['hold']), ('CV', R['cv']), ('DV', alt['DV'][s]['err']), ('MR', alt['MR'][s]['err']),
                            (fz['model'], R['err'])):
                w = E[full, :W]
                summ.append(dict(split=s, window_s=wname, method=name, n_windows=int(full.sum()), mean_over_window_mm=1e3 * w.mean(),
                                 end_of_window_mm=1e3 * w[:, -1].mean(), end_q90_mm=1e3 * np.quantile(w[:, -1], 0.9),
                                 worst_in_window_mm=1e3 * w.max(axis=1).mean()))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RESULTS, 'p2_masking_summary.csv'), index=False)

    # ------------------------------------------------------------------ conformal calibration on validation
    V = res['validation']
    zv = V['err'] / V['sd']
    cal = {}
    for p in LEVELS:
        q = np.array([np.nanquantile(zv[:, h], p) for h in range(HMAX)])
        qs = np.array([np.median(q[max(0, h - 3):h + 4]) for h in range(HMAX)])          # light smoothing over +-0.1 s
        cal[p] = qs
    emp = {name: {p: np.array([np.nanquantile(E[:, h], p) for h in range(HMAX)]) for p in LEVELS}
           for name, E in (('Hold', V['hold']), ('CV', V['cv']))}
    frozen = dict(horizon_frames=HOR2.tolist(), levels=list(LEVELS), multiplier={str(p): cal[p].tolist() for p in LEVELS},
                  gaussian_multiplier={str(p): float(np.sqrt(chi2.ppf(p, 3))) for p in LEVELS},
                  hold_radius_m={str(p): emp['Hold'][p].tolist() for p in LEVELS}, cv_radius_m={str(p): emp['CV'][p].tolist() for p in LEVELS},
                  calibrated_on='validation', frozen_before_test=True)
    save_json(frozen, 'frozen_p2_calibration.json')

    cov = []
    for s in ('validation', 'test'):
        R = res[s]
        for p in LEVELS:
            for h in (0, 2, 5, 14, 29, 59, 89):
                e = R['err'][:, h]; sd = R['sd'][:, h]; m = ~np.isnan(e)
                rg = np.sqrt(chi2.ppf(p, 3)) * sd[m]; rc = cal[p][h] * sd[m]
                cov.append(dict(split=s, nominal=p, horizon_s=(h + 1) * DT, n=int(m.sum()),
                                gauss_cov=float((e[m] <= rg).mean()), gauss_radius_mm=1e3 * rg.mean(),
                                conf_cov=float((e[m] <= rc).mean()), conf_radius_mm=1e3 * rc.mean(),
                                hold_cov=float((R['hold'][m, h] <= emp['Hold'][p][h]).mean()), hold_radius_mm=1e3 * emp['Hold'][p][h],
                                cv_cov=float((R['cv'][m, h] <= emp['CV'][p][h]).mean()), cv_radius_mm=1e3 * emp['CV'][p][h]))
            # pooled over all horizons
            m = ~np.isnan(R['err'])
            rc = cal[p][None, :] * R['sd']; rg = np.sqrt(chi2.ppf(p, 3)) * R['sd']
            rh = np.broadcast_to(emp['Hold'][p][None, :], R['err'].shape); rv = np.broadcast_to(emp['CV'][p][None, :], R['err'].shape)
            cov.append(dict(split=s, nominal=p, horizon_s=-1.0, n=int(m.sum()), gauss_cov=float((R['err'][m] <= rg[m]).mean()),
                            gauss_radius_mm=1e3 * rg[m].mean(), conf_cov=float((R['err'][m] <= rc[m]).mean()), conf_radius_mm=1e3 * rc[m].mean(),
                            hold_cov=float((R['hold'][m] <= rh[m]).mean()), hold_radius_mm=1e3 * rh[m].mean(),
                            cv_cov=float((R['cv'][m] <= rv[m]).mean()), cv_radius_mm=1e3 * rv[m].mean()))
    cov = pd.DataFrame(cov)
    cov.to_csv(os.path.join(RESULTS, 'p2_coverage.csv'), index=False)

    # ------------------------------------------------------------------ calibrated 90 % radius of a typical validation onset against the
    # missing duration, by phase at the onset (the curve behind tau*; drawn in the mechanism figures)
    rcv = []
    for k in list(range(K)) + [-1]:
        mv = (V['phase'] == k) if k >= 0 else np.ones(len(V['phase']), dtype=bool)
        rad = cal[0.9] * np.nanmedian(V['sd'][mv], axis=0)
        for h in range(HMAX):
            rcv.append(dict(phase='all' if k < 0 else str(k + 1), horizon_s=(h + 1) * DT, radius90_mm=1e3 * rad[h],
                            hold_radius90_mm=1e3 * emp['Hold'][0.9][h]))
    pd.DataFrame(rcv).to_csv(os.path.join(RESULTS, 'p2_radius_curves.csv'), index=False)

    # ------------------------------------------------------------------ admissible prediction horizon tau*(eps, phase)
    adm = []
    p = 0.9
    for eps in (0.02, 0.03, 0.05, 0.08, 0.10, 0.15):
        for k in list(range(K)) + [-1]:
            mv = (V['phase'] == k) if k >= 0 else np.ones(len(V['phase']), dtype=bool)
            # calibrated 90 % radius of a typical validation onset of this phase, per horizon
            rad = cal[p] * np.nanmedian(V['sd'][mv], axis=0)
            ok = rad <= eps
            hstar = int(np.argmax(~ok)) if (~ok).any() else HMAX          # first horizon index violating the tolerance
            row = dict(tolerance_mm=1e3 * eps, phase='all' if k < 0 else str(k + 1), tau_star_s=hstar * DT, censored_at_3s=bool(hstar == HMAX))
            for s in ('validation', 'test'):
                R = res[s]
                ms = (R['phase'] == k) if k >= 0 else np.ones(len(R['phase']), dtype=bool)
                if hstar >= 1:
                    e = R['err'][ms, hstar - 1]; e = e[~np.isnan(e)]
                    row['%s_within_tol_at_tau' % s] = float((e <= eps).mean()) if len(e) else np.nan
                    eh = R['hold'][ms, hstar - 1]; eh = eh[~np.isnan(eh)]
                    row['%s_hold_within_tol_at_tau' % s] = float((eh <= eps).mean()) if len(eh) else np.nan
                else:
                    row['%s_within_tol_at_tau' % s] = np.nan; row['%s_hold_within_tol_at_tau' % s] = np.nan
            # same question for the hold-last baseline (radius = empirical validation quantile, phase-blind)
            okh = emp['Hold'][p] <= eps
            row['tau_star_hold_s'] = (int(np.argmax(~okh)) if (~okh).any() else HMAX) * DT
            adm.append(row)
    adm = pd.DataFrame(adm)
    adm.to_csv(os.path.join(RESULTS, 'p2_admissible.csv'), index=False)

    # ------------------------------------------------------------------ effect of the visible history length
    hist_rows = []
    for hn in (3, 9, 30, 90):
        for s in ('validation', 'test'):
            runs_s = [(j, r, a, b) for (j, r, a, b) in runs[s] if b - a > 90 + 2]
            Rh = collect_hist(D, runs_s, hmm, main_c, hn)
            Rf = collect_hist(D, runs_s, hmm, main_c, None, phase)
            for h in (2, 14, 29, 59, 89):
                e = Rh['err'][:, h]; ef = Rf['err'][:, h]; m = ~np.isnan(e) & ~np.isnan(ef)
                hist_rows.append(dict(split=s, history_s=hn * DT, horizon_s=(h + 1) * DT, n=int(m.sum()), err_mm=1e3 * e[m].mean(),
                                      full_history_err_mm=1e3 * ef[m].mean(), hold_mm=1e3 * Rh['hold'][m, h].mean()))
    hist_rows = pd.DataFrame(hist_rows)
    hist_rows.to_csv(os.path.join(RESULTS, 'p2_history.csv'), index=False)

    # ------------------------------------------------------------------ aggregation by environment (test + validation)
    env_rows = []
    for s in ('validation', 'test'):
        R = res[s]
        env = D.env_of_rid[R['meta'][:, 0]]
        for e in np.unique(env):
            m = env == e
            for wname, W in WINDOWS.items():
                full = m & ~np.isnan(R['hold'][:, W - 1])
                if full.sum() == 0:
                    continue
                rc = cal[0.9][None, :W] * R['sd'][full, :W]
                env_rows.append(dict(split=s, environment_id=e, window_s=wname, n_windows=int(full.sum()),
                                     hold_mm=1e3 * R['hold'][full, :W].mean(), cv_mm=1e3 * R['cv'][full, :W].mean(),
                                     model_mm=1e3 * R['err'][full, :W].mean(), conf90_cov=float((R['err'][full, :W] <= rc).mean()),
                                     conf90_radius_mm=1e3 * rc.mean()))
    env_rows = pd.DataFrame(env_rows)
    env_rows.to_csv(os.path.join(RESULTS, 'p2_by_env.csv'), index=False)

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 200)
    print(summ[summ.split != 'train'].round(2).to_string())
    print(cov[(cov.split == 'test')].round(3).to_string())
    print(adm.round(3).to_string())
    print(hist_rows.round(2).to_string())
    print(env_rows[env_rows.window_s.isin([0.5, 2.0])].round(2).to_string())
    print('done in %.0fs' % (time.time() - t0))


def collect_hist(D, runs, hmm, cand, hist, phase=None):
    """collect() restricted to onsets that have at least 90 frames of visible history (for the history experiment)."""
    global HIST_MIN
    old = HIST_MIN
    HIST_MIN = 90
    try:
        if hist is None:
            return collect(D, runs, phase, hmm, cand, None)
        return collect(D, runs, [np.zeros(D.N, dtype=np.int64)] * 2, hmm, cand, hist)
    finally:
        HIST_MIN = old


if __name__ == '__main__':
    main()
