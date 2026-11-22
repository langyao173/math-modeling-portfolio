# -*- coding: utf-8 -*-
"""Problem 1, part D.  Influencing factors: task family, source/environment, stage inside a segment, history.

All inference is clustered by environment (24 clusters); descriptive use of all three splits is allowed here
because no predictive model is selected or tuned in this script.

Outputs (results/): p1_segment_table.csv, p1_factor_mixedlm.csv, p1_factor_groups.csv,
p1_stage_profile.csv, p1_factor_tests.json
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, save_json
from features import Q_FLAG, Q_REF, Data
from statemodel import load_frozen_state, phase_feature

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(20261122)


def env_category(e):
    for k in ('supermarket', 'convenience', 'apartment', 'courier', 'office', 'restaurant'):
        if k in e:
            return {'supermarket': 'retail', 'convenience': 'retail', 'apartment': 'home', 'courier': 'logistics',
                    'office': 'office', 'restaurant': 'hospitality'}[k]
    return 'other'


def cluster_boot(df, value, weight, group, B=2000):
    """Environment-cluster bootstrap of a weighted mean per level of `group`."""
    out = {}
    envs = df['environment_id'].unique()
    for lv, d in df.groupby(group):
        ev = d['environment_id'].unique()
        num = d.groupby('environment_id').apply(lambda x: (x[value] * x[weight]).sum()).reindex(ev).to_numpy()
        den = d.groupby('environment_id')[weight].sum().reindex(ev).to_numpy()
        est = num.sum() / den.sum()
        if len(ev) > 1:
            pick = RNG.integers(0, len(ev), size=(B, len(ev)))
            bs = num[pick].sum(axis=1) / den[pick].sum(axis=1)
            lo, hi = np.quantile(bs, [0.025, 0.975])
        else:
            lo = hi = np.nan
        out[lv] = (est, lo, hi, len(ev), int(den.sum()))
    return out


def main():
    D = Data()
    fz, hmm, par = load_frozen_state()
    K = hmm.K
    # causal phase for all frames of 30-Hz records (filter restarted at every reference run)
    phase = [np.full(D.N, -1, dtype=int) for _ in range(2)]
    for j in range(2):
        for r, a, b in D.reference_runs(j, 0.2, only_30hz=True):
            phase[j][a:b] = hmm.filter(phase_feature(D, j, a, b)).argmax(axis=1)
    rows = []
    for k, s in enumerate(D.seg.itertuples()):
        m = D.seg_id == k
        for j in range(2):
            ref = m & (D.q[j] == Q_REF)
            if ref.sum() < 10:
                continue
            row = dict(seg=k, rid=s.rid, hand='LR'[j], family=s.family_id, dataset=s.dataset, split=s.split, task=s.task,
                       environment_id=s.environment_id, env_cat=env_category(s.environment_id), duration_s=s.duration_s,
                       paradigm=D.rec.paradigm[s.rid] if isinstance(D.rec.paradigm[s.rid], str) else 'unknown',
                       n_ref=int(ref.sum()), n=int(m.sum()), ego_speed=float(D.espeed[j][ref].mean()),
                       world_speed=float(D.speed[j][ref].mean()), head_speed=float(D.hspeed[ref].mean()),
                       reach=float(D.reach[j][ref].mean()), flagged_share=float((D.q[j][m] == Q_FLAG).mean()),
                       flagged_frames=int((D.q[j][m] == Q_FLAG).sum()))
            ph = phase[j][ref]
            for kk in range(K):
                row['phase%d' % (kk + 1)] = float((ph == kk).sum() / max((ph >= 0).sum(), 1))
            rows.append(row)
    seg = pd.DataFrame(rows)
    seg['log_speed'] = np.log(seg['ego_speed'])
    seg['ldur'] = np.log(seg['duration_s'])
    seg.to_csv(os.path.join(RESULTS, 'p1_segment_table.csv'), index=False)

    tests = {}
    # ---------------------------------------------------------------- mixed model: log mean egocentric speed
    def mixed(formula):
        return smf.mixedlm(formula, seg, groups=seg['environment_id']).fit(reml=False)
    full = mixed('log_speed ~ C(family) + C(dataset) + C(hand) + ldur')
    res = []
    for name, drop in (('task family', 'log_speed ~ C(dataset) + C(hand) + ldur'),
                       ('data source', 'log_speed ~ C(family) + C(hand) + ldur'),
                       ('hand', 'log_speed ~ C(family) + C(dataset) + ldur'),
                       ('segment duration', 'log_speed ~ C(family) + C(dataset) + C(hand)')):
        red = mixed(drop)
        lr = 2 * (full.llf - red.llf); dfree = len(full.fe_params) - len(red.fe_params)
        res.append(dict(factor=name, lr_stat=lr, df=dfree, p=float(chi2.sf(lr, dfree))))
    var_env = float(full.cov_re.iloc[0, 0]); var_res = float(full.scale)
    tests['speed_mixed_model'] = dict(icc_environment=var_env / (var_env + var_res), var_env=var_env, var_resid=var_res,
                                      n_segments_hands=len(seg), n_env=int(seg.environment_id.nunique()),
                                      fixed_effects={k: float(v) for k, v in full.fe_params.items()},
                                      fixed_se={k: float(v) for k, v in full.bse_fe.items()}, lr_tests=res)
    pd.DataFrame(res).to_csv(os.path.join(RESULTS, 'p1_factor_mixedlm.csv'), index=False)
    # paradigm effect inside the two sources that record it
    sub = seg[seg.paradigm != 'unknown']
    if sub.paradigm.nunique() > 1:
        f1 = smf.mixedlm('log_speed ~ C(paradigm) + C(family) + C(hand)', sub, groups=sub['environment_id']).fit(reml=False)
        f0 = smf.mixedlm('log_speed ~ C(family) + C(hand)', sub, groups=sub['environment_id']).fit(reml=False)
        lr = 2 * (f1.llf - f0.llf)
        tests['paradigm_speed'] = dict(coef_mimic=float(f1.fe_params.get('C(paradigm)[T.robot_gripper_mimic]', np.nan)),
                                       se=float(f1.bse_fe.get('C(paradigm)[T.robot_gripper_mimic]', np.nan)), lr_stat=float(lr),
                                       p=float(chi2.sf(lr, 1)), n=len(sub))

    # ---------------------------------------------------------------- group summaries with cluster-bootstrap CIs
    grp = []
    for g in ('family', 'dataset', 'env_cat', 'paradigm', 'split'):
        for val, lab in (('ego_speed', 'egocentric speed (m/s)'), ('world_speed', 'world speed (m/s)'),
                         ('head_speed', 'head speed (m/s)')) + tuple(('phase%d' % (k + 1), 'share of phase %d' % (k + 1)) for k in range(K)):
            for lv, (est, lo, hi, ne, nf) in cluster_boot(seg, val, 'n_ref', g).items():
                grp.append(dict(group=g, level=lv, quantity=lab, estimate=est, ci_low=lo, ci_high=hi, n_env=ne, n_frames=nf))
        for lv, (est, lo, hi, ne, nf) in cluster_boot(seg, 'flagged_share', 'n', g).items():
            grp.append(dict(group=g, level=lv, quantity='flagged share', estimate=est, ci_low=lo, ci_high=hi, n_env=ne, n_frames=nf))
    grp = pd.DataFrame(grp)
    grp.to_csv(os.path.join(RESULTS, 'p1_factor_groups.csv'), index=False)

    # ---------------------------------------------------------------- stage inside the annotated segment
    prof = []
    nb = 10
    fam_of_seg = D.seg['family_id'].to_numpy()
    env_of_seg = D.seg['environment_id'].to_numpy()
    for fam in ('P', 'A', 'L', 'ALL'):
        for b in range(nb):
            sp, fl, n = [], 0, 0
            per_env = {}
            for j in range(2):
                m = (D.seg_id >= 0) & (D.seg_u >= b / nb) & (D.seg_u < (b + 1) / nb)
                if fam != 'ALL':
                    m &= fam_of_seg[np.clip(D.seg_id, 0, None)] == fam
                ref = m & (D.q[j] == Q_REF)
                e = env_of_seg[D.seg_id[ref]]
                for ee in np.unique(e):
                    v = D.espeed[j][ref][e == ee]
                    a = per_env.setdefault(ee, [0.0, 0])
                    a[0] += v.sum(); a[1] += len(v)
                fl += int((D.q[j][m] == Q_FLAG).sum()); n += int(m.sum())
            num = np.array([v[0] for v in per_env.values()]); den = np.array([v[1] for v in per_env.values()])
            pick = RNG.integers(0, len(num), size=(2000, len(num)))
            bs = num[pick].sum(axis=1) / den[pick].sum(axis=1)
            prof.append(dict(family=fam, bin=b, u_mid=(b + 0.5) / nb, ego_speed=num.sum() / den.sum(), ci_low=np.quantile(bs, 0.025),
                             ci_high=np.quantile(bs, 0.975), flagged_per_1000=1e3 * fl / max(n, 1), n=n))
    prof = pd.DataFrame(prof)
    prof.to_csv(os.path.join(RESULTS, 'p1_stage_profile.csv'), index=False)
    # paired test: first 20 % vs middle 60 % vs last 20 % of each segment (environment-level sign-flip permutation)
    st = []
    for k, s in enumerate(D.seg.itertuples()):
        for j in range(2):
            m = (D.seg_id == k) & (D.q[j] == Q_REF)
            u = D.seg_u[m]; v = D.espeed[j][m]
            if (u < 0.2).sum() > 3 and ((u >= 0.2) & (u < 0.8)).sum() > 3 and (u >= 0.8).sum() > 3:
                st.append(dict(env=s.environment_id, early=v[u < 0.2].mean(), mid=v[(u >= 0.2) & (u < 0.8)].mean(), late=v[u >= 0.8].mean()))
    st = pd.DataFrame(st)
    ev = st.groupby('env').mean()
    out = {}
    for a, b in (('early', 'mid'), ('late', 'mid'), ('early', 'late')):
        d = np.log(ev[a]) - np.log(ev[b])
        signs = RNG.choice([-1, 1], size=(20000, len(d)))
        pval = float((np.abs((signs * d.to_numpy()).mean(axis=1)) >= abs(d.mean()) - 1e-12).mean())
        out['%s_vs_%s' % (a, b)] = dict(ratio=float(np.exp(d.mean())), p_signflip=pval, n_env=int(len(d)))
    tests['stage_within_segment'] = out
    # ---------------------------------------------------------------- history: Markov dependence of the quality flag
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(D.start, D.stop):
        for j in range(2):
            z = (D.flag[a:b, j] > 0).astype(int)
            n00 += int(((z[:-1] == 0) & (z[1:] == 0)).sum()); n01 += int(((z[:-1] == 0) & (z[1:] == 1)).sum())
            n10 += int(((z[:-1] == 1) & (z[1:] == 0)).sum()); n11 += int(((z[:-1] == 1) & (z[1:] == 1)).sum())
    p = (n01 + n11) / (n00 + n01 + n10 + n11); p01 = n01 / (n00 + n01); p11 = n11 / (n10 + n11)
    ll0 = (n01 + n11) * np.log(p) + (n00 + n10) * np.log(1 - p)
    ll1 = n01 * np.log(p01) + n00 * np.log(1 - p01) + n11 * np.log(p11) + n10 * np.log(1 - p11)
    tests['flag_markov_vs_iid'] = dict(n00=n00, n01=n01, n10=n10, n11=n11, p_onset=p01, p_stay=p11, lr_stat=float(2 * (ll1 - ll0)),
                                       p=float(chi2.sf(2 * (ll1 - ll0), 1)))
    # velocity memory of the egocentric state: lag autocorrelation of the velocity (reference runs, train)
    acf = {}
    for lag in (1, 3, 6, 15, 30):
        num = den = 0.0
        for j in range(2):
            for r, a, b in D.reference_runs(j, only_30hz=True):
                if D.split_of_rid[r] != 'train' or b - a <= lag + 2:
                    continue
                u = D.u[j][a + 1:b]
                num += float((u[lag:] * u[:-lag]).sum()); den += float((u * u).sum())
        acf['%.2fs' % (lag / 30)] = num / den
    tests['velocity_autocorrelation_train'] = acf
    save_json(tests, 'p1_factor_tests.json')
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 300)
    print(pd.DataFrame(res).round(4).to_string())
    print(grp[grp.quantity.isin(['egocentric speed (m/s)', 'flagged share', 'world speed (m/s)'])].round(4).to_string())
    print(prof[prof.family == 'ALL'].round(4).to_string())
    import json
    print(json.dumps(tests, indent=1, default=float))


if __name__ == '__main__':
    main()
