# -*- coding: utf-8 -*-
"""Problem 2, part C.  Natural occlusion: occurrence, duration, recovery, and contrast with artificial masking.

No position error is computed inside flagged (low-quality) intervals.  The only state comparison made is the
"bridging discrepancy": the frozen model predicts across a natural episode from the last reference-grade frame
before it to the first reference-grade frame after it, and the prediction is compared with that qualified frame.

Outputs (results/): p2_natural_summary.json, p2_natural_runs.csv, p2_natural_duration.csv, p2_natural_bridge.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, load_json, runs_of, save_json
from features import GUARD, Q_FLAG, Q_REF, Data
from ssm import build_struct, kf_struct
from statemodel import DT, SPEED_EPS, assign_phases, load_frozen_state, p_init, phase_feature, run_list
import p1c_quality_model as qm
import p2a_masking as pa

RNG = np.random.default_rng(20261123)
HIST = 15


def main():
    D = Data()
    fz, hmm, par = load_frozen_state()
    cal = load_json('frozen_p2_calibration.json')
    c90 = np.array(cal['multiplier']['0.9'])
    cand = fz['candidates'][fz['model']]
    K = cand['K']
    F, Q, Fh, Qh = build_struct(par, DT, np.array(hmm.A), pa.HOR2)
    # ---------------------------------------------------------------- 1. episodes (maximal flagged runs per hand)
    rows = []
    for r, (a, b) in enumerate(zip(D.start, D.stop)):
        for j in range(2):
            fl = D.flag[a:b, j]
            for u, w in runs_of(fl > 0):
                codes = np.unique(fl[u:w])
                rows.append(dict(rid=r, hand=j, u=int(u), w=int(w), n=int(w - u), dur_s=(w - u) * D.dt_rec[r],
                                 reason={1: 'occlusion', 2: 'near edge'}.get(int(codes[0]), 'mixed') if len(codes) == 1 else 'mixed',
                                 censored=bool(u == 0 or w == b - a), dataset=D.rec.dataset[r], split=D.rec.split[r],
                                 environment_id=D.rec.environment_id[r], t_on=float(D.t[a + u]),
                                 family=D.seg.family_id[D.seg_id[a + u]] if D.seg_id[a + u] >= 0 else '',
                                 seg_u=float(D.seg_u[a + u]), both_hands=bool((D.flag[a + u:a + w, 1 - j] > 0).any())))
    ep = pd.DataFrame(rows)
    minutes = D.rec['duration_s'].sum() / 60.0
    summ = dict(episodes=len(ep), records_with_episodes=int(ep.rid.nunique()), records=len(D.rec),
                hand_minutes=float(2 * minutes), episodes_per_hand_minute=float(len(ep) / (2 * minutes)),
                flagged_share_of_hand_frames=float((D.flag > 0).mean()),
                by_reason=ep.groupby('reason').agg(n=('n', 'size'), median_s=('dur_s', 'median'), mean_s=('dur_s', 'mean'),
                                                   max_s=('dur_s', 'max')).round(4).to_dict('index'),
                duration_quantiles_s={q: float(ep.dur_s.quantile(float(q))) for q in ('0.25', '0.5', '0.75', '0.9', '0.95', '0.99')},
                duration_mean_s=float(ep.dur_s.mean()), duration_max_s=float(ep.dur_s.max()), censored=int(ep.censored.sum()),
                share_single_frame=float((ep.n == 1).mean()), share_le_0p2s=float((ep.dur_s <= 0.2 + 1e-9).mean()),
                share_gt_0p5s=float((ep.dur_s > 0.5).mean()), both_hands_share=float(ep.both_hands.mean()))
    # rates per hand-minute by dataset / family with environment-cluster bootstrap
    def rate_ci(key):
        out = {}
        for lv in sorted(ep[key].unique()):
            if lv == '':
                continue
            if key == 'family':
                mins = {e: 0.0 for e in D.rec.environment_id.unique()}
                for s in D.seg.itertuples():
                    if s.family_id == lv:
                        mins[s.environment_id] += 2 * s.duration_s / 60.0
            else:
                mins = (2 * D.rec[D.rec[key] == lv].groupby('environment_id')['duration_s'].sum() / 60.0).to_dict()
            envs = [e for e, m in mins.items() if m > 0]
            cnt = np.array([((ep[key] == lv) & (ep.environment_id == e)).sum() for e in envs], dtype=float)
            mm = np.array([mins[e] for e in envs])
            pick = RNG.integers(0, len(envs), size=(2000, len(envs)))
            bs = cnt[pick].sum(axis=1) / mm[pick].sum(axis=1)
            out[lv] = dict(rate_per_hand_min=float(cnt.sum() / mm.sum()), ci_low=float(np.quantile(bs, 0.025)),
                           ci_high=float(np.quantile(bs, 0.975)), episodes=int(cnt.sum()), hand_minutes=float(mm.sum()), n_env=len(envs))
        return out
    summ['rate_by_dataset'] = rate_ci('dataset'); summ['rate_by_family'] = rate_ci('family'); summ['rate_by_split'] = rate_ci('split')
    # ---------------------------------------------------------------- 2. duration law: geometric (memoryless) vs empirical
    unc = ep[~ep.censored]
    p_stay = 1 - len(unc) / unc.n.sum()                                 # MLE of the geometric stay probability
    kmax = int(unc.n.max())
    dur = pd.DataFrame(dict(frames=np.arange(1, kmax + 1)))
    dur['empirical_survival'] = [(unc.n >= k).mean() for k in dur.frames]
    dur['geometric_survival'] = p_stay ** (dur.frames - 1)
    haz = []
    for k in dur.frames:
        at_risk = (unc.n >= k).sum(); end = (unc.n == k).sum()
        haz.append(end / at_risk if at_risk > 0 else np.nan)
    dur['empirical_recovery_hazard'] = haz
    dur['at_risk'] = [(unc.n >= k).sum() for k in dur.frames]
    dur.to_csv(os.path.join(RESULTS, 'p2_natural_duration.csv'), index=False)
    # recovery hazard in the first 3 frames vs later (duration dependence), episode-bootstrap CI
    def rec_haz(n, lo, hi):
        ks = np.arange(lo, hi + 1)
        risk = sum((n >= k).sum() for k in ks); end = sum((n == k).sum() for k in ks)
        return end / max(risk, 1)
    nn = unc.n.to_numpy()
    bs = np.array([[rec_haz(x, 1, 3), rec_haz(x, 4, 10), rec_haz(x, 11, 60)] for x in (RNG.choice(nn, len(nn)) for _ in range(2000))])
    summ['recovery'] = dict(p_stay_geometric=float(p_stay), mean_duration_frames=float(unc.n.mean()),
                            hazard_frames_1_3=float(rec_haz(nn, 1, 3)), hazard_frames_4_10=float(rec_haz(nn, 4, 10)),
                            hazard_frames_11_plus=float(rec_haz(nn, 11, 60)),
                            hazard_ci=dict(f1_3=[float(x) for x in np.quantile(bs[:, 0], [0.025, 0.975])],
                                           f4_10=[float(x) for x in np.quantile(bs[:, 1], [0.025, 0.975])],
                                           f11=[float(x) for x in np.quantile(bs[:, 2], [0.025, 0.975])]),
                            half_recovered_s=float(unc.dur_s.median()), recovered_90pct_s=float(unc.dur_s.quantile(0.9)))
    # re-occurrence after recovery: probability of a new onset of the same hand within 1 s, versus the base rate
    re = []
    for r in ep.rid.unique():
        for j in range(2):
            e = ep[(ep.rid == r) & (ep.hand == j)].sort_values('u')
            n_fr = D.stop[r] - D.start[r]
            for i in range(len(e)):
                wend = e.w.iloc[i]
                if wend + 30 > n_fr:
                    continue
                nxt = e.u.iloc[i + 1] if i + 1 < len(e) else 10 ** 9
                re.append(nxt - wend <= 30)
    h1 = 1 - (1 - load_json('frozen_p1_quality.json')['markov_onset']) ** 30
    summ['reoccurrence'] = dict(p_new_onset_within_1s_after_recovery=float(np.mean(re)), n=len(re), base_rate_any_1s_window=float(h1),
                                ratio=float(np.mean(re) / h1))
    # ---------------------------------------------------------------- 3. bridging discrepancy across natural episodes
    br = []
    for e in ep.itertuples():
        r, j = e.rid, e.hand
        if not D.ok_rate[r]:
            continue
        a, b = D.start[r], D.stop[r]
        q = D.q[j][a:b]
        k0 = e.u - 1
        while k0 >= 0 and q[k0] != Q_REF:
            k0 -= 1
        k1 = e.w
        while k1 < b - a and q[k1] != Q_REF:
            k1 += 1
        if k0 < HIST or k1 >= b - a or k1 - k0 > pa.HMAX:
            continue
        s0 = k0
        while s0 > 0 and q[s0 - 1] == Q_REF:
            s0 -= 1
        if k0 - s0 < HIST:
            continue
        x = D.xi[j][a + s0:a + k0 + 1]
        ph = hmm.filter(np.log10(D.lagspeed[j][a + s0:a + k0 + 1] + SPEED_EPS)).argmax(axis=1).astype(np.int64)
        g = k1 - k0
        o = kf_struct(np.vstack([x, D.xi[j][a + k1][None, :].repeat(g, axis=0)]), np.concatenate([ph, np.zeros(g, dtype=np.int64)]),
                      F, Q, par['r'], Fh[g - 1:g], Qh[:, g - 1:g], np.array([g], dtype=np.int64), len(x) - 1, np.full(3, np.nan), p_init(cand['kind']), True)
        err = o[4][len(x) - 1, 0]; sd = o[5][len(x) - 1, 0]
        hold = float(np.linalg.norm(D.xi[j][a + k1] - D.xi[j][a + k0]))
        br.append(dict(rid=r, hand=j, split=e.split, reason=e.reason, flagged_frames=e.n, gap_frames=g, gap_s=g * DT, model_mm=1e3 * err,
                       hold_mm=1e3 * hold, radius90_mm=1e3 * c90[g - 1] * sd, inside90=bool(err <= c90[g - 1] * sd),
                       phase_before=int(ph[-1]) + 1, speed_before=float(D.lagspeed[j][a + k0]), gi0=int(a + k0)))
    br = pd.DataFrame(br)
    # matched artificial masks: every artificial onset of all 30-Hz reference runs, same gap lengths
    runs = run_list(D)
    phase, _ = assign_phases(D, runs, hmm)
    art = {s: pa.collect(D, runs[s], phase, hmm, cand) for s in runs}
    A_err = np.concatenate([art[s]['err'] for s in art]); A_hold = np.concatenate([art[s]['hold'] for s in art])
    A_sd = np.concatenate([art[s]['sd'] for s in art]); A_meta = np.concatenate([art[s]['meta'] for s in art])
    # onset propensity of every artificial onset (frozen Problem-1 reliability model, covariates of the last visible frame)
    qz = load_json('frozen_p1_quality.json')
    tab = qm.build_table(D)
    terms = qz['onset_terms'][1:]
    key = {(int(h), int(g)): i for i, (h, g) in enumerate(zip(tab['hand'].to_numpy(), tab['gi'].to_numpy()))}
    X = tab[terms].to_numpy(dtype=float)
    beta = np.array(qz['onset_coef'])
    lin = beta[0] + X @ beta[1:]
    haz_tab = 1 / (1 + np.exp(-lin))
    idx = np.array([key.get((int(jj), int(gi) + 1), -1) for (_, jj, gi) in A_meta])
    w = np.where(idx >= 0, haz_tab[np.clip(idx, 0, None)], np.nan)
    w = w / np.nanmean(w)
    br['art_model_mm'] = [1e3 * np.nanmean(A_err[:, g - 1]) for g in br.gap_frames]
    br['art_hold_mm'] = [1e3 * np.nanmean(A_hold[:, g - 1]) for g in br.gap_frames]
    def wmean(v, ww):
        m = ~np.isnan(v) & ~np.isnan(ww)
        return float((v[m] * ww[m]).sum() / ww[m].sum())
    br['ipw_model_mm'] = [1e3 * wmean(A_err[:, g - 1], w) for g in br.gap_frames]
    br['ipw_hold_mm'] = [1e3 * wmean(A_hold[:, g - 1], w) for g in br.gap_frames]
    br['art_inside90'] = [float(np.nanmean(A_err[:, g - 1] <= c90[g - 1] * A_sd[:, g - 1])) for g in br.gap_frames]
    br.to_csv(os.path.join(RESULTS, 'p2_natural_bridge.csv'), index=False)
    def boot_ratio(num, den):
        n = len(num); pick = RNG.integers(0, n, size=(2000, n))
        bs = num[pick].sum(axis=1) / den[pick].sum(axis=1)
        return [float(num.sum() / den.sum()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    summ['bridging'] = dict(
        episodes_used=len(br), gap_s_median=float(br.gap_s.median()), natural_model_mm=float(br.model_mm.mean()),
        natural_hold_mm=float(br.hold_mm.mean()), matched_artificial_model_mm=float(br.art_model_mm.mean()),
        matched_artificial_hold_mm=float(br.art_hold_mm.mean()), propensity_weighted_model_mm=float(br.ipw_model_mm.mean()),
        propensity_weighted_hold_mm=float(br.ipw_hold_mm.mean()),
        ratio_natural_to_artificial_model=boot_ratio(br.model_mm.to_numpy(), br.art_model_mm.to_numpy()),
        ratio_natural_to_ipw_model=boot_ratio(br.model_mm.to_numpy(), br.ipw_model_mm.to_numpy()),
        ratio_natural_to_artificial_hold=boot_ratio(br.hold_mm.to_numpy(), br.art_hold_mm.to_numpy()),
        natural_inside_calibrated_90=float(br.inside90.mean()), artificial_inside_calibrated_90=float(br.art_inside90.mean()),
        natural_inside_twice_radius=float((br.model_mm <= 2.0 * br.radius90_mm).mean()),      # design rule: doubled radius after a natural gap
        model_vs_hold_natural_reduction=float(1 - br.model_mm.mean() / br.hold_mm.mean()),
        by_split={s: dict(n=int((br.split == s).sum()), natural_model_mm=float(br[br.split == s].model_mm.mean()),
                          artificial_model_mm=float(br[br.split == s].art_model_mm.mean()),
                          inside90=float(br[br.split == s].inside90.mean())) for s in ('train', 'validation', 'test')})
    # train/test boundary of the design rule: the factor is estimated without the test records and the doubled ball is checked on them
    fit, tst = br[br.split != 'test'], br[br.split == 'test']
    summ['bridging']['correction_rule'] = dict(
        estimated_on='train+validation', n_fit=int(len(fit)), factor=2.0,
        ratio_natural_to_artificial_model=boot_ratio(fit.model_mm.to_numpy(), fit.art_model_mm.to_numpy()),
        fit_inside_calibrated_90=float(fit.inside90.mean()), fit_inside_twice_radius=float((fit.model_mm <= 2.0 * fit.radius90_mm).mean()),
        n_test=int(len(tst)), test_inside_calibrated_90=float(tst.inside90.mean()),
        test_inside_twice_radius=float((tst.model_mm <= 2.0 * tst.radius90_mm).mean()),
        test_inside_twice_radius_count=int((tst.model_mm <= 2.0 * tst.radius90_mm).sum()))
    # MNAR evidence: state just before natural onsets versus all reference frames
    sp_on = br.speed_before.to_numpy()
    sp_all = np.concatenate([D.lagspeed[j][D.good[j]] for j in range(2)])
    summ['selection'] = dict(median_speed_before_onset=float(np.median(sp_on)), median_speed_all_reference=float(np.median(sp_all)),
                             share_onsets_in_fastest_phase=float((br.phase_before == K).mean()),
                             propensity_weight_cv=float(np.nanstd(w) / np.nanmean(w)),
                             propensity_weight_p99_over_median=float(np.nanquantile(w, 0.99) / np.nanmedian(w)))
    ep.to_csv(os.path.join(RESULTS, 'p2_natural_runs.csv'), index=False)
    save_json(summ, 'p2_natural_summary.json')
    import json
    print(json.dumps(summ, indent=1, default=float))
    print(dur.head(16).round(3).to_string())


if __name__ == '__main__':
    main()
