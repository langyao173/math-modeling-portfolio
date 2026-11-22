# -*- coding: utf-8 -*-
"""Problem 1, part C.  Observation-reliability model for the original hand-quality flags.

Per hand, the flag z_t in {0 (usable), 1 (flagged: occlusion / near edge)} is modelled as a two-state Markov
chain whose transition probabilities depend on covariates computed from the previous frame only:
    onset        P(z_t = 1 | z_{t-1} = 0, x_{t-1}) = logistic(b0 + b' x_{t-1})
    persistence  P(z_t = 1 | z_{t-1} = 1, a_{t-1}) = logistic(g0 + g1 log(age))     (age = frames since onset)
Covariate groups (geometry, motion, hand interaction, history, task family, source) are tested with
environment-clustered likelihood-ratio / Wald statistics on the training split and by held-out log loss.

Outputs (results/): p1_quality_selection.csv, p1_quality_coef.csv, p1_quality_metrics.csv,
p1_quality_calibration.csv, p1_quality_capture.csv, frozen_p1_quality.json
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, SPLITS, save_json
from features import Data

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(20261121)

GROUPS = {
    'geometry': ['edge', 'near', 'far'],
    'motion': ['lspeed'],
    'interaction': ['overlap_behind', 'other_flag'],
    'history': ['recent'],
    'task': ['fam_A', 'fam_L'],
    'source': ['ds_pro', 'ds_std', 'mimic'],
}
ORDER = ['geometry', 'motion', 'interaction', 'history', 'task', 'source']


def build_table(D):
    fr = D.frame_table()
    fr = fr.sort_values(['hand', 'gi']).reset_index(drop=True)
    first = np.zeros(len(fr), dtype=bool)
    first[np.flatnonzero(np.diff(fr['rid'].to_numpy(), prepend=-1) != 0)] = True
    first |= np.diff(fr['hand'].to_numpy(), prepend=-1) != 0
    z = (fr['flag'].to_numpy() > 0).astype(int)
    lag = lambda a: np.where(first, np.nan, np.roll(a, 1))
    fr['z'] = z
    fr['z_prev'] = lag(z.astype(float))
    # covariates from the previous frame (t-1)
    km = fr['kmargin'].to_numpy(); dep = fr['depth'].to_numpy(); uvd = fr['uv_dist'].to_numpy()
    edge = np.clip((0.25 - km) / 0.25, 0, 3)                       # proximity of the worst keypoint to the image border
    near = np.clip((0.30 - dep) / 0.10, 0, 3)                      # hand very close to the camera
    far = np.clip((dep - 0.45) / 0.10, 0, 3)                       # hand reaching far from the camera
    lspeed = np.log10(fr['lagspeed'].to_numpy() + 0.01)
    # hand-hand occlusion proxy: image-plane overlap while this hand is farther from the camera than the other
    j = fr['hand'].to_numpy()
    dep_other = np.where(j == 0, D.pc[1][fr['gi'].to_numpy(), 2], D.pc[0][fr['gi'].to_numpy(), 2])
    overlap_behind = np.clip((0.18 - uvd) / 0.18, 0, 1) * (dep > dep_other)
    other_flag = (fr['other_flag'].to_numpy() > 0).astype(float)
    # history: any flag of the same hand in the preceding second (frames t-31 .. t-2)
    recent = np.zeros(len(fr)); age = np.zeros(len(fr))
    zz = z.astype(float)
    start_idx = np.flatnonzero(first)
    bounds = list(start_idx) + [len(fr)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        c = np.concatenate([[0.0], np.cumsum(zz[a:b])])
        n = b - a
        k = np.arange(n)
        lo = np.clip(k - 31, 0, n); hi = np.clip(k - 1, 0, n)      # window [t-31, t-2] in 0-based local index
        recent[a:b] = (c[hi] - c[lo]) > 0
        run = 0
        for i in range(n):                                          # frames since onset of the current flagged run
            run = run + 1 if zz[a + i] > 0 else 0
            age[a + i] = run
    for name, v in (('edge', edge), ('near', near), ('far', far), ('lspeed', lspeed), ('overlap_behind', overlap_behind),
                    ('other_flag', other_flag)):
        fr[name] = lag(v)
    fr['recent'] = recent
    fr['age_prev'] = lag(age)
    fam = fr['family'].to_numpy()
    fr['fam_A'] = (fam == 'A').astype(float); fr['fam_L'] = (fam == 'L').astype(float)
    fr['ds_pro'] = (fr['dataset'] == 'EgoPro').astype(float); fr['ds_std'] = (fr['dataset'] == 'EgoStandard').astype(float)
    fr['mimic'] = (fr['paradigm'] == 'robot_gripper_mimic').astype(float)
    fr['reason_prev'] = lag(fr['flag'].to_numpy().astype(float))
    return fr[~first].reset_index(drop=True)


def fit_logit(df, cols):
    X = sm.add_constant(df[cols].to_numpy(dtype=float), has_constant='add') if cols else np.ones((len(df), 1))
    m = sm.GLM(df['z'].to_numpy(dtype=float), X, family=sm.families.Binomial())
    res = m.fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(df['environment_id'])[0]})
    return res


def predict(res, df, cols):
    X = sm.add_constant(df[cols].to_numpy(dtype=float), has_constant='add') if cols else np.ones((len(df), 1))
    return np.clip(res.predict(X), 1e-7, 1 - 1e-7)


def logloss(y, p):
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def boot_ci(df, fn, B=1000):
    """Record-cluster bootstrap CI of a statistic fn(frame subset)."""
    rids = df['rid'].unique()
    idx = {r: np.flatnonzero(df['rid'].to_numpy() == r) for r in rids}
    vals = []
    for _ in range(B):
        pick = RNG.choice(rids, size=len(rids), replace=True)
        ii = np.concatenate([idx[r] for r in pick])
        try:
            vals.append(fn(ii))
        except Exception:
            pass
    return (float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))) if vals else (np.nan, np.nan)


def main():
    D = Data()
    tab = build_table(D)
    onset = tab[tab['z_prev'] == 0].reset_index(drop=True)
    pers = tab[tab['z_prev'] == 1].reset_index(drop=True)
    pers['lage'] = np.log(pers['age_prev'].clip(lower=1))
    pers['edge_reason'] = (pers['reason_prev'] == 2).astype(float)
    tr_o, va_o, te_o = (onset[onset.split == s].reset_index(drop=True) for s in SPLITS)
    tr_p, va_p, te_p = (pers[pers.split == s].reset_index(drop=True) for s in SPLITS)
    print('onset rows / events:', {s: (len(d), int(d.z.sum())) for s, d in zip(SPLITS, (tr_o, va_o, te_o))})
    print('persistence rows / stays:', {s: (len(d), int(d.z.sum())) for s, d in zip(SPLITS, (tr_p, va_p, te_p))})

    # ---------------------------------------------------------------- onset model: forward selection of covariate groups
    rows, fits = [], {}
    cols = []
    res0 = fit_logit(tr_o, [])
    fits['constant'] = (res0, [])
    rows.append(dict(model='constant hazard', groups='-', k=1, train_ll=res0.llf, val_logloss=logloss(va_o.z.to_numpy(), predict(res0, va_o, [])),
                     val_auc=np.nan, lr_stat=np.nan, lr_df=np.nan, lr_p=np.nan, wald_p_cluster=np.nan))
    prev = res0
    for g in ORDER:
        cols = cols + GROUPS[g]
        res = fit_logit(tr_o, cols)
        lr = 2 * (res.llf - prev.llf); df_ = len(GROUPS[g])
        # cluster-robust Wald test of the newly added group
        k = len(cols)
        Rm = np.zeros((df_, k + 1)); Rm[np.arange(df_), k + 1 - df_ + np.arange(df_)] = 1
        try:
            wp = float(res.wald_test(Rm, scalar=True).pvalue)
        except Exception:
            wp = np.nan
        p = predict(res, va_o, cols)
        name = '+' + g
        fits[name] = (res, list(cols))
        rows.append(dict(model=name, groups=' + '.join(ORDER[:ORDER.index(g) + 1]), k=k + 1, train_ll=res.llf,
                         val_logloss=logloss(va_o.z.to_numpy(), p), val_auc=roc_auc_score(va_o.z, p), lr_stat=lr, lr_df=df_,
                         lr_p=float(chi2.sf(lr, df_)), wald_p_cluster=wp))
        prev = res
    sel = pd.DataFrame(rows)
    best = sel.loc[sel['val_logloss'].idxmin(), 'model']
    best_name = 'constant' if best == 'constant hazard' else best
    sel['selected'] = sel['model'] == best
    print(sel.round(5).to_string())
    res_on, cols_on = fits[best_name]

    # ---------------------------------------------------------------- persistence model
    prow = []
    pfits = {}
    for name, pc in (('constant', []), ('log age', ['lage']), ('log age + reason', ['lage', 'edge_reason'])):
        r = fit_logit(tr_p, pc)
        pfits[name] = (r, pc)
        prow.append(dict(model=name, k=len(pc) + 1, train_ll=r.llf, val_logloss=logloss(va_p.z.to_numpy(), predict(r, va_p, pc))))
    psel = pd.DataFrame(prow)
    pbest = psel.loc[psel['val_logloss'].idxmin(), 'model']
    res_pe, cols_pe = pfits[pbest]
    print(psel.round(5).to_string(), '\npersistence model selected:', pbest)

    # ---------------------------------------------------------------- frozen coefficient table
    coef = []
    ci = res_on.conf_int()
    for i, n in enumerate(['intercept'] + cols_on):
        coef.append(dict(part='onset', term=n, coef=res_on.params[i], se_cluster=res_on.bse[i], ci_low=ci[i, 0], ci_high=ci[i, 1],
                         odds_ratio=np.exp(res_on.params[i]), p=res_on.pvalues[i]))
    ci = res_pe.conf_int()
    for i, n in enumerate(['intercept'] + cols_pe):
        coef.append(dict(part='persistence', term=n, coef=res_pe.params[i], se_cluster=res_pe.bse[i], ci_low=ci[i, 0], ci_high=ci[i, 1],
                         odds_ratio=np.exp(res_pe.params[i]), p=res_pe.pvalues[i]))
    coef = pd.DataFrame(coef)
    coef.to_csv(os.path.join(RESULTS, 'p1_quality_coef.csv'), index=False)
    print(coef.round(4).to_string())
    pi_tr = float(tab[tab.split == 'train'].z.mean())
    h_tr = float(tr_o.z.mean()); s_tr = float(tr_p.z.mean())
    frozen = dict(onset_terms=['intercept'] + cols_on, onset_coef=res_on.params.tolist(), persistence_terms=['intercept'] + cols_pe,
                  persistence_coef=res_pe.params.tolist(), baseline_rate=pi_tr, markov_onset=h_tr, markov_stay=s_tr,
                  feature_definition=dict(edge='clip((0.25-kmargin)/0.25,0,3)', near='clip((0.30-depth)/0.10,0,3)',
                                          far='clip((depth-0.45)/0.10,0,3)', lspeed='log10(lag-0.1s egocentric speed + 0.01)',
                                          overlap_behind='clip((0.18-uv_dist)/0.18,0,1)*[depth>depth_other]',
                                          recent='any flag of the hand in the preceding 1 s', lage='log(frames since onset)'),
                  selected_onset_model=best, selected_persistence_model=pbest, frozen_before_test=True)
    save_json(frozen, 'frozen_p1_quality.json')
    sel.to_csv(os.path.join(RESULTS, 'p1_quality_selection.csv'), index=False)
    psel.to_csv(os.path.join(RESULTS, 'p1_quality_persistence_selection.csv'), index=False)

    # ---------------------------------------------------------------- held-out evaluation (frame-level, one step ahead)
    def frame_probs(split):
        d = tab[tab.split == split].reset_index(drop=True)
        d['lage'] = np.log(d['age_prev'].clip(lower=1)); d['edge_reason'] = (d['reason_prev'] == 2).astype(float)
        on = d['z_prev'].to_numpy() == 0
        p_full = np.where(on, predict(res_on, d, cols_on), predict(res_pe, d, cols_pe))
        p_mark = np.where(on, h_tr, s_tr)
        p_const = np.full(len(d), pi_tr)
        return d, on, dict(constant=p_const, markov=p_mark, full=p_full)

    mrows, crow, lrow = [], [], []
    for split in SPLITS:
        d, on, P = frame_probs(split)
        y = d['z'].to_numpy()
        for name, p in P.items():
            row = dict(split=split, model=name, n=len(y), events=int(y.sum()), logloss=logloss(y, p), brier=float(np.mean((p - y) ** 2)))
            row['auc'] = roc_auc_score(y, p) if name != 'constant' else 0.5
            if split != 'train':
                lo, hi = boot_ci(d, lambda ii: logloss(y[ii], p[ii]))
                row['logloss_lo'], row['logloss_hi'] = lo, hi
            mrows.append(row)
        # onset-only discrimination among at-risk frames
        yo = y[on]; po = P['full'][on]
        row = dict(split=split, model='onset (at-risk frames)', n=int(on.sum()), events=int(yo.sum()), logloss=logloss(yo, po),
                   brier=float(np.mean((po - yo) ** 2)), auc=roc_auc_score(yo, po), ap=average_precision_score(yo, po),
                   base_rate=float(yo.mean()), logloss_const=logloss(yo, np.full(len(yo), h_tr)))
        if split != 'train':
            do = d[on].reset_index(drop=True)
            lo, hi = boot_ci(do, lambda ii: roc_auc_score(yo[ii], po[ii]))
            row['auc_lo'], row['auc_hi'] = lo, hi
        mrows.append(row)
        # capture curve: share of the onsets found among the x % of at-risk frames with the highest predicted hazard
        order = np.argsort(-po, kind='stable'); cum = np.cumsum(yo[order]) / max(yo.sum(), 1)
        for x in np.unique(np.round(np.concatenate([np.logspace(-3, 0, 121), np.linspace(0.01, 1, 100)]), 6)):
            k = max(int(round(x * len(po))), 1)
            lrow.append(dict(split=split, frame_share=k / len(po), onset_share=float(cum[k - 1]), n_frames=int(len(po)), n_onsets=int(yo.sum())))
        # calibration of the onset hazard in quantile bins of the prediction (per 1000 frames)
        qs = np.quantile(po, [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0])
        for a, b in zip(qs[:-1], qs[1:]):
            m = (po >= a) & (po <= b)
            crow.append(dict(split=split, p_lo=a, p_hi=b, n=int(m.sum()), predicted_per_1000=1e3 * po[m].mean(), observed_per_1000=1e3 * yo[m].mean()))
    met = pd.DataFrame(mrows)
    met.to_csv(os.path.join(RESULTS, 'p1_quality_metrics.csv'), index=False)
    pd.DataFrame(crow).to_csv(os.path.join(RESULTS, 'p1_quality_calibration.csv'), index=False)
    pd.DataFrame(lrow).to_csv(os.path.join(RESULTS, 'p1_quality_capture.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(met.round(5).to_string()); print(pd.DataFrame(crow).round(3).to_string())
    # implied stationary quantities of the plain Markov chain
    print('train: onset hazard per frame %.5f, stay prob %.4f, mean run %.2f frames, stationary flagged share %.4f (observed %.4f)' % (
        h_tr, s_tr, 1 / (1 - s_tr), h_tr / (h_tr + 1 - s_tr), pi_tr))


if __name__ == '__main__':
    main()
