# -*- coding: utf-8 -*-
"""Problem 2, part B.  Closed-loop replay of re-observation policies inside artificial missing windows.

Experimental conditions (fixed in advance, shared by every policy)
    window          3.0 s (90 frames) of one hand hidden; >= 1 s of visible history; onsets every 1 s
    request         returns the 4 recorded frames (0.1 s exposure) ending at the request time
    return delay    delta = 0.1 s (nominal) or 0.3 s (sensitivity); information is unusable before it has returned
    cost            1 unit per request; hard cap of 30 requests per window
    tolerances      30 / 50 / 80 mm, reliability level 90 %
Policies
    none                                   pure prediction
    periodic(n)   + Hold | CV | Model      fixed-interval observation, n requests equally spaced
    trigger(eps)  + Model                  request when the calibrated 90 % radius at the earliest possible return
                                           of a later request would exceed eps (stage-aware through the phase belief)
    dp(B)         + Model                  exact finite-horizon dynamic programme: state = (time, information age,
                                           phase at the newest snapshot, remaining budget, pending request);
                                           stage cost = expected error f_k(age) estimated on the training split;
                                           solved with and without the stage (phase) information
Two designs are reported: (i) fixed budget B per window shared by all methods; (ii) minimum cost subject to a
tolerance (exceedance <= 10 %), where the number of requests is the outcome.
Outputs (results/): p2_policy_frontier.csv, p2_policy_budget.csv, p2_policy_trials.csv, p2_policy_constrained.csv,
p2_wait_table.csv, p2_policy_meta.json, p2_request_raster.csv, p2_raster_error_mm.csv, p2_raster_radius_mm.csv
"""
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, load_json, save_json
from features import Data
from ssm import GaussHMM, build_struct, kf_struct, njit
from statemodel import DT, SPEED_EPS, assign_phases, load_frozen_state, p_init, phase_feature, run_list

W = 90
HIST_MIN = 30
ONSET_STEP = 30
BURST = 4
AMAX = 130
BUDGET_CAP = 30
TOLS = (0.03, 0.05, 0.08)
GMIN = 3                      # minimum spacing between two requests (frames)


@njit(cache=True)
def _var_at(age, P, Fh, Qh, ph, r2):
    v = r2
    for i in range(2):
        for j in range(2):
            s = Qh[ph, age - 1, i, j]
            for a in range(3):
                for b in range(3):
                    s += Fh[age - 1, i, a] * P[a, b] * Fh[age - 1, j, b]
            v += s
    return v


@njit(cache=True)
def sim_trial(y, k0, m0, P0, pi0, F, Fh, Qh, Apow, mu, sdv, r, cmult, base_rad, predictor, pol, param, delta, budget, dec):
    """Replay one missing window.  y: (T,3) recorded states of the run; k0: last visible frame.
    predictor: 0 hold, 1 constant velocity, 2 Kalman model.
    pol: 0 none, 1 periodic (param = n), 2 trigger (param = eps), 3 dynamic programme (decision table dec).
    Returns err (W,), rad (W,), n_req, pending_steps, sum_age, req_t (frames at which the requests were issued)."""
    K = pi0.shape[0]
    r2 = r * r
    err = np.zeros(W); rad = np.zeros(W)
    m = m0.copy(); P = P0.copy(); pi = pi0.copy()
    s = 0                                   # window-relative index of the newest usable sample (0 = frame k0)
    last = y[k0].copy()
    vel = (y[k0] - y[k0 - 3]) / (3.0 * DT)
    req_t = np.zeros(budget + 1, dtype=np.int64); n_req = 0; n_done = 0
    last_req = -1000
    pend_steps = 0; sum_age = 0.0
    n_per = int(param) if pol == 1 else 0
    Pn = np.zeros((3, 3)); FP = np.zeros((3, 3)); pi2 = np.zeros(K)
    for i in range(1, W + 1):
        # ---- 1. returned information
        while n_done < n_req and req_t[n_done] + delta <= i:
            rq = req_t[n_done]; n_done += 1
            f0 = max(rq - (BURST - 1), s + 1)
            if predictor == 2:
                for f in range(f0, rq + 1):
                    g = f - s
                    ph = 0
                    for k in range(1, K):
                        if pi[k] > pi[ph]:
                            ph = k
                    # predict over g frames (phase mixing inside Qh), then update with the returned sample
                    for a in range(3):
                        for b in range(3):
                            acc = 0.0
                            for c in range(3):
                                acc += Fh[g - 1, a, c] * P[c, b]
                            FP[a, b] = acc
                    for a in range(3):
                        for b in range(3):
                            acc = 0.0
                            for c in range(3):
                                acc += FP[a, c] * Fh[g - 1, b, c]
                            Pn[a, b] = acc + Qh[ph, g - 1, a, b]
                    S = Pn[0, 0] + 2.0 * Pn[0, 1] + Pn[1, 1] + r2
                    g0 = (Pn[0, 0] + Pn[0, 1]) / S; g1 = (Pn[1, 0] + Pn[1, 1]) / S; g2 = (Pn[2, 0] + Pn[2, 1]) / S
                    for d in range(3):
                        a1 = Fh[g - 1, 1, 1] * m[d, 1] + Fh[g - 1, 1, 2] * m[d, 2]
                        a2 = Fh[g - 1, 2, 1] * m[d, 1] + Fh[g - 1, 2, 2] * m[d, 2]
                        e = y[k0 + f, d] - m[d, 0] - a1
                        m[d, 0] += g0 * e; m[d, 1] = a1 + g1 * e; m[d, 2] = a2 + g2 * e
                    h0 = Pn[0, 0] + Pn[1, 0]; h1 = Pn[0, 1] + Pn[1, 1]; h2 = Pn[0, 2] + Pn[1, 2]
                    P[0, 0] = Pn[0, 0] - g0 * h0; P[0, 1] = Pn[0, 1] - g0 * h1; P[0, 2] = Pn[0, 2] - g0 * h2
                    P[1, 0] = Pn[1, 0] - g1 * h0; P[1, 1] = Pn[1, 1] - g1 * h1; P[1, 2] = Pn[1, 2] - g1 * h2
                    P[2, 0] = Pn[2, 0] - g2 * h0; P[2, 1] = Pn[2, 1] - g2 * h1; P[2, 2] = Pn[2, 2] - g2 * h2
                    # phase belief: propagate g frames
                    for k in range(K):
                        acc = 0.0
                        for c in range(K):
                            acc += pi[c] * Apow[g, c, k]
                        pi2[k] = acc
                    for k in range(K):
                        pi[k] = pi2[k]
                    s = f
                # phase belief: emission update with the 0.1-s speed measured by the burst
                if K > 1 and rq >= s:
                    dx = 0.0
                    for d in range(3):
                        dd = y[k0 + rq, d] - y[k0 + rq - 3, d]
                        dx += dd * dd
                    x = math.log10(math.sqrt(dx) / (3.0 * DT) + SPEED_EPS)
                    tot = 0.0
                    for k in range(K):
                        pi[k] = pi[k] * math.exp(-0.5 * ((x - mu[k]) / sdv[k]) ** 2) / sdv[k] + 1e-300
                        tot += pi[k]
                    for k in range(K):
                        pi[k] /= tot
            if rq > s or predictor != 2:
                s = max(s, rq)
            for d in range(3):
                last[d] = y[k0 + rq, d]
                vel[d] = (y[k0 + rq, d] - y[k0 + rq - 3, d]) / (3.0 * DT)
        # ---- 2. prediction of frame i from the newest usable information
        age = i - s
        ph = 0
        for k in range(1, K):
            if pi[k] > pi[ph]:
                ph = k
        e2 = 0.0
        if predictor == 2:
            v = _var_at(age, P, Fh, Qh, ph, r2)
            for d in range(3):
                pm = m[d, 0] + Fh[age - 1, 1, 1] * m[d, 1] + Fh[age - 1, 1, 2] * m[d, 2]
                e2 += (y[k0 + i, d] - pm) ** 2
            rad[i - 1] = cmult[min(age, cmult.shape[0]) - 1] * math.sqrt(v)
        else:
            for d in range(3):
                pm = last[d] + (vel[d] * age * DT if predictor == 1 else 0.0)
                e2 += (y[k0 + i, d] - pm) ** 2
            rad[i - 1] = base_rad[min(age, base_rad.shape[0]) - 1]
        err[i - 1] = math.sqrt(e2)
        sum_age += age * DT
        if n_done < n_req:
            pend_steps += 1
        # ---- 3. decision: request the observation of frame i ?
        if n_req < budget and (i - last_req >= GMIN or pol == 3) and i + delta <= W:
            ask = False
            if pol == 1:
                for k in range(1, n_per + 1):
                    if i == int(round(k * W / (n_per + 1.0))):
                        ask = True
            elif pol == 2:
                s_new = s if last_req < s else last_req          # newest snapshot already requested
                a_next = i + 1 + delta - s_new                    # information age when a request made one step later returns
                if a_next < 1:
                    a_next = 1
                if a_next > AMAX:
                    a_next = AMAX
                vv = _var_at(a_next, P, Fh, Qh, ph, r2)
                if cmult[min(a_next, cmult.shape[0]) - 1] * math.sqrt(vv) > param:
                    ask = True
            elif pol == 3:
                if n_done == n_req:                               # the programme allows one pending request at a time
                    aa = age if age < dec.shape[1] else dec.shape[1] - 1
                    kk = ph if ph < dec.shape[2] else 0
                    if dec[i, aa, kk, budget - n_req] == 1:
                        ask = True
            if ask:
                req_t[n_req] = i; n_req += 1; last_req = i
    return err, rad, n_req, pend_steps, sum_age, req_t


@njit(cache=True)
def dp_policy(f, Apow, Wn, B, delta):
    """Backward induction for the observation-timing MDP.  f: (K, AM) expected error by phase at the newest
    snapshot and information age (frames, 1-based).  Returns dec (Wn+1, AM+1, K, B+1) with 1 = request now,
    and the optimal expected cumulative error for the initial state (age 1) per phase and budget."""
    K, AM = f.shape
    Vn = np.zeros((AM + 2, K, B + 1, delta + 1))          # layer i+1
    Vc = np.zeros((AM + 2, K, B + 1, delta + 1))          # layer i
    dec = np.zeros((Wn + 1, AM + 1, K, B + 1), dtype=np.uint8)
    for i in range(Wn, 0, -1):
        amax = min(i, AM)
        for a in range(1, amax + 1):
            for k in range(K):
                c = f[k, a - 1]
                for b in range(B + 1):
                    for p in range(delta):
                        if p > a - 1 and p > 0:
                            continue
                        nxt = 0.0
                        if i < Wn:
                            if p > 0:
                                if p + 1 == delta:
                                    acc = 0.0
                                    for k2 in range(K):
                                        acc += Apow[a - p, k, k2] * Vn[delta, k2, b, 0]
                                    nxt = acc
                                else:
                                    nxt = Vn[min(a + 1, AM), k, b, p + 1]
                            else:
                                no = Vn[min(a + 1, AM), k, b, 0]
                                nxt = no
                                if b > 0 and i + delta <= Wn:
                                    if delta == 1:
                                        yes = 0.0
                                        for k2 in range(K):
                                            yes += Apow[a, k, k2] * Vn[1, k2, b - 1, 0]
                                    else:
                                        yes = Vn[min(a + 1, AM), k, b - 1, 1]
                                    if yes < no - 1e-12:
                                        nxt = yes
                                        dec[i, a, k, b] = 1
                        Vc[a, k, b, p] = c + nxt
        for a in range(AM + 2):
            for k in range(K):
                for b in range(B + 1):
                    for p in range(delta + 1):
                        Vn[a, k, b, p] = Vc[a, k, b, p]
    v0 = np.zeros((K, B + 1))
    for k in range(K):
        for b in range(B + 1):
            v0[k, b] = Vn[1, k, b, 0]
    return dec, v0


def prepare(D, runs, phase_prob, hmm, cand):
    """History filtering for every trial: returns list of (j, r, a, b, k0, m0, P0, pi0)."""
    K = cand['K']; par = dict(cand['params']); par['sigma'] = np.array(par['sigma'])
    A = np.array(hmm.A) if K > 1 else np.ones((1, 1))
    F, Q, Fh, Qh = build_struct(par, DT, A, np.arange(1, AMAX + 1, dtype=np.int64))
    Apow = np.zeros((AMAX + 1, K, K)); Apow[0] = np.eye(K)
    for g in range(1, AMAX + 1):
        Apow[g] = Apow[g - 1] @ A
    trials = []
    for (j, r, a, b) in runs:
        T = b - a
        k0s = np.arange(HIST_MIN, T - W - 1, ONSET_STEP)
        if len(k0s) == 0:
            continue
        x = D.xi[j][a:b]
        pr = phase_prob[j][a:b] if K > 1 else np.ones((T, 1))
        ph = pr.argmax(axis=1).astype(np.int64)
        o = kf_struct(x, ph, F, Q, par['r'], Fh[:1], Qh[:, :1], np.array([1], dtype=np.int64), HIST_MIN, np.full(3, np.nan), p_init(cand['kind']), True)
        mf, Pf = o[6], o[7]
        for k0 in k0s:
            trials.append((j, r, a, b, int(k0), mf[k0].copy(), Pf[k0].copy(), pr[k0].copy()))
    return trials, dict(F=F, Fh=Fh, Qh=Qh, Apow=Apow, r=par['r'])


NO_DEC = np.zeros((1, 1, 1, 1), dtype=np.uint8)


def run_policy(D, trials, mats, hmm_mu, hmm_sd, cmult, base_rad, predictor, pol, param, delta, budget=BUDGET_CAP, dec=NO_DEC, requests=None):
    """Replays every trial.  When `requests` is a list, the request frames of each trial are appended to it."""
    n = len(trials)
    E = np.zeros((n, W)); Rr = np.zeros((n, W)); NR = np.zeros(n); PS = np.zeros(n); AG = np.zeros(n)
    for t, (j, r, a, b, k0, m0, P0, pi0) in enumerate(trials):
        e, rad, nr, ps, ag, rq = sim_trial(D.xi[j][a:b], k0, m0, P0, pi0, mats['F'], mats['Fh'], mats['Qh'], mats['Apow'], hmm_mu, hmm_sd,
                                           mats['r'], cmult, base_rad, predictor, pol, float(param), int(delta), int(budget), dec)
        E[t], Rr[t], NR[t], PS[t], AG[t] = e, rad, nr, ps, ag
        if requests is not None:
            requests.append(rq[:nr].copy())
    return E, Rr, NR, PS, AG


def summarise(E, Rr, NR, PS, AG, env):
    row = dict(n_trials=len(NR), mean_err_mm=1e3 * E.mean(), q90_err_mm=1e3 * np.quantile(E, 0.9), max_err_mm=1e3 * E.max(axis=1).mean(),
               requests=NR.mean(), requests_per_s=NR.mean() / (W * DT), cost=NR.mean(), pending_s=PS.mean() * DT,
               mean_info_age_s=AG.mean() / W, cov90=float((E <= Rr).mean()), radius90_mm=1e3 * Rr.mean())
    for tol in TOLS:
        row['exceed_%dmm' % round(1e3 * tol)] = float((E > tol).mean())
    # environment-level mean and spread of the trial-mean error
    tm = E.mean(axis=1)
    em = np.array([tm[env == e].mean() for e in np.unique(env)])
    row['env_mean_err_mm'] = 1e3 * em.mean(); row['env_sd_err_mm'] = 1e3 * em.std(ddof=1) if len(em) > 1 else np.nan
    return row


def paired_ci(d, rec, B=2000, seed=7):
    """Record-cluster bootstrap CI of the mean of paired differences d (one value per trial)."""
    rng = np.random.default_rng(seed)
    recs = np.unique(rec)
    sums = np.array([d[rec == r].sum() for r in recs]); cnts = np.array([(rec == r).sum() for r in recs])
    pick = rng.integers(0, len(recs), size=(B, len(recs)))
    bs = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main():
    t0 = time.time()
    D = Data()
    fz, hmm, par = load_frozen_state()
    cal = load_json('frozen_p2_calibration.json')
    c90 = np.array(cal['multiplier']['0.9']); hold90 = np.array(cal['hold_radius_m']['0.9']); cv90 = np.array(cal['cv_radius_m']['0.9'])
    runs = run_list(D)
    phase, prob = assign_phases(D, runs, hmm)
    main_c = fz['candidates'][fz['model']]
    mu, sdv = np.array(hmm.mu), np.array(hmm.sd)
    K = len(mu)
    T, M = {}, {}
    for s in ('validation', 'test'):
        T[s], M[s] = prepare(D, runs[s], prob, hmm, main_c)
    print('trials:', {s: len(T[s]) for s in T}, flush=True)
    env = {s: np.array([D.env_of_rid[t[1]] for t in T[s]]) for s in T}
    rec = {s: np.array([t[1] for t in T[s]]) for s in T}

    # ---------------------------------------------------------------- stage cost of the timing MDP (training split only)
    import p2a_masking as pa
    Rtr = pa.collect(D, runs['train'], phase, hmm, main_c)
    AM = W + 1
    f_stage = np.zeros((K, AM)); f_blind = np.zeros((1, AM))
    for a in range(AM):
        h = min(a, pa.HMAX - 1)
        f_blind[0, a] = np.nanmean(Rtr['err'][:, h])
        for k in range(K):
            f_stage[k, a] = np.nanmean(Rtr['err'][Rtr['phase'] == k, h])
    for arr in (f_stage, f_blind):                                  # light smoothing along the age axis
        for k in range(arr.shape[0]):
            arr[k] = np.convolve(np.pad(arr[k], 2, mode='edge'), np.ones(5) / 5.0, mode='valid')
    Apow = M['test']['Apow']
    Apow1 = np.ones((AMAX + 1, 1, 1))

    rows, trial_rows, budget_rows = [], [], []
    periodic_n = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 26, 30]
    eps_grid = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15]
    timing = {}
    for delta in (3, 9):
        for s in ('validation', 'test'):
            cfgs = [('none', 'Hold', 0, 0, 0), ('none', 'CV', 1, 0, 0), ('none', 'Model', 2, 0, 0)]
            for n in periodic_n:
                cfgs += [('periodic', 'Hold', 0, 1, n), ('periodic', 'CV', 1, 1, n), ('periodic', 'Model', 2, 1, n)]
            for e in eps_grid:
                cfgs += [('trigger', 'Model', 2, 2, e)]
            keep = {}
            for (pol_name, pred_name, pred, pol, param) in cfgs:
                tt = time.perf_counter()
                out = run_policy(D, T[s], M[s], mu, sdv, c90, hold90 if pred == 0 else cv90, pred, pol, param, delta)
                timing[(pol_name, pred_name)] = 1e6 * (time.perf_counter() - tt) / (len(T[s]) * W)
                rows.append(dict(split=s, delay_s=delta * DT, policy=pol_name, predictor=pred_name, param=param, **summarise(*out, env[s])))
                keep[(pol_name, pred_name, param)] = out
            # ---- design (i): fixed budget B per window, shared by all methods
            for B in (1, 2, 3, 4, 6, 8):
                tt = time.perf_counter()
                dec_s, v_s = dp_policy(f_stage, Apow, W, B, delta)
                dec_b, v_b = dp_policy(f_blind, Apow1, W, B, delta)
                t_dp = time.perf_counter() - tt
                o_s = run_policy(D, T[s], M[s], mu, sdv, c90, cv90, 2, 3, 0.0, delta, budget=B, dec=dec_s)
                o_b = run_policy(D, T[s], M[s], mu, sdv, c90, cv90, 2, 3, 0.0, delta, budget=B, dec=dec_b)
                ref = keep[('periodic', 'Model', B)]
                for name, o in (('periodic + Hold', keep[('periodic', 'Hold', B)]), ('periodic + CV', keep[('periodic', 'CV', B)]),
                                ('periodic + Model', ref), ('DP (no stage) + Model', o_b), ('DP (stage-aware) + Model', o_s)):
                    d, lo, hi = paired_ci(1e3 * (o[0].mean(axis=1) - ref[0].mean(axis=1)), rec[s])
                    budget_rows.append(dict(split=s, delay_s=delta * DT, budget=B, method=name, **summarise(*o, env[s]),
                                            diff_vs_periodic_model_mm=d, diff_lo=lo, diff_hi=hi,
                                            dp_expected_mm=(1e3 * float(v_s.mean(axis=0)[B]) / W if 'stage-aware' in name else
                                                            (1e3 * float(v_b[0, B]) / W if 'no stage' in name else np.nan))))
                if s == 'test' and delta == 3 and B in (2, 4, 6):
                    for name, o in (('none + Hold', keep[('none', 'Hold', 0)]), ('none + Model', keep[('none', 'Model', 0)]),
                                    ('periodic + Hold', keep[('periodic', 'Hold', B)]), ('periodic + CV', keep[('periodic', 'CV', B)]),
                                    ('periodic + Model', ref), ('DP (stage-aware) + Model', o_s)):
                        for t, tr in enumerate(T[s]):
                            trial_rows.append(dict(delay_s=delta * DT, budget=B, method=name, trial=t, rid=tr[1], hand='LR'[tr[0]],
                                                   onset_t=float(D.t[tr[2] + tr[4]]), environment_id=env[s][t], mean_err_mm=1e3 * o[0][t].mean(),
                                                   max_err_mm=1e3 * o[0][t].max(), requests=int(o[2][t]), pending_s=o[3][t] * DT,
                                                   exceed_50mm=float((o[0][t] > 0.05).mean()), cov90=float((o[0][t] <= o[1][t]).mean())))
                timing[('dp', 'solve_ms_B%d_d%d' % (B, delta))] = 1e3 * t_dp
        print('delay %d frames done (%.0fs)' % (delta, time.time() - t0), flush=True)
    fr = pd.DataFrame(rows)
    fr.to_csv(os.path.join(RESULTS, 'p2_policy_frontier.csv'), index=False)
    bud = pd.DataFrame(budget_rows)
    bud.to_csv(os.path.join(RESULTS, 'p2_policy_budget.csv'), index=False)
    pd.DataFrame(trial_rows).to_csv(os.path.join(RESULTS, 'p2_policy_trials.csv'), index=False)

    # ---------------------------------------------------------------- design (ii): minimum cost subject to a tolerance
    cons = []
    for delta in (3, 9):
        for tol in TOLS:
            col = 'exceed_%dmm' % round(1e3 * tol)
            for (pol, pred) in (('periodic', 'Hold'), ('periodic', 'CV'), ('periodic', 'Model'), ('trigger', 'Model')):
                v = fr[(fr.split == 'validation') & (fr.delay_s == delta * DT) & (fr.policy == pol) & (fr.predictor == pred)]
                ok = v[v[col] <= 0.10].sort_values('requests')
                if len(ok) == 0:
                    cons.append(dict(delay_s=delta * DT, tolerance_mm=1e3 * tol, policy=pol, predictor=pred, feasible_on_validation=False,
                                     best_val_exceed=float(v[col].min())))
                    continue
                best = ok.iloc[0]
                te = fr[(fr.split == 'test') & (fr.delay_s == delta * DT) & (fr.policy == pol) & (fr.predictor == pred) & (fr.param == best.param)].iloc[0]
                cons.append(dict(delay_s=delta * DT, tolerance_mm=1e3 * tol, policy=pol, predictor=pred, feasible_on_validation=True,
                                 param=best.param, val_requests=best.requests, val_exceed=best[col], test_requests=te.requests,
                                 test_exceed=te[col], test_mean_err_mm=te.mean_err_mm, test_pending_s=te.pending_s,
                                 test_info_age_s=te.mean_info_age_s, test_cov90=te.cov90))
    cons = pd.DataFrame(cons)
    cons.to_csv(os.path.join(RESULTS, 'p2_policy_constrained.csv'), index=False)

    # ---------------------------------------------------------------- request raster of the selected trigger policy (test windows)
    # tolerance 50 mm, return delay 0.1 s: when does the policy ask, and what error does the planner carry between two views?
    sel = cons[(cons.delay_s == 3 * DT) & (cons.tolerance_mm == 50.0) & (cons.policy == 'trigger')].iloc[0]
    rq = []
    E, Rr, NR, PS, AG = run_policy(D, T['test'], M['test'], mu, sdv, c90, cv90, 2, 2, float(sel.param), 3, requests=rq)
    Eh = run_policy(D, T['test'], M['test'], mu, sdv, c90, hold90, 0, 0, 0.0, 3)[0]
    ras = pd.DataFrame(dict(trial=np.arange(len(NR)), rid=rec['test'], hand=['LR'[t[0]] for t in T['test']], environment_id=env['test'],
                            onset_phase=[int(np.argmax(t[7])) + 1 for t in T['test']], requests=NR.astype(int),
                            request_times_s=[';'.join('%.4f' % (f * DT) for f in r_) for r_ in rq],
                            mean_err_mm=1e3 * E.mean(axis=1), max_err_mm=1e3 * E.max(axis=1), mean_radius_mm=1e3 * Rr.mean(axis=1),
                            unobserved_mean_err_mm=1e3 * Eh.mean(axis=1)))
    ras.to_csv(os.path.join(RESULTS, 'p2_request_raster.csv'), index=False)
    cols = ['f%02d' % (i + 1) for i in range(W)]
    # nine significant digits: enough for the figures and stable across machines (no rounding flips in the comparison of runs)
    pd.DataFrame(1e3 * E, columns=cols).to_csv(os.path.join(RESULTS, 'p2_raster_error_mm.csv'), index_label='trial', float_format='%.9g')
    pd.DataFrame(1e3 * Rr, columns=cols).to_csv(os.path.join(RESULTS, 'p2_raster_radius_mm.csv'), index_label='trial', float_format='%.9g')

    # ---------------------------------------------------------------- waiting table: how long to wait after a returned observation
    wait = []
    Pm = np.median(np.stack([t[6] for t in T['validation']]), axis=0)
    for tol in TOLS + (0.10,):
        for k in range(K):
            rad = np.array([c90[min(a, len(c90)) - 1] * math.sqrt(_var_at(a, Pm, M['validation']['Fh'], M['validation']['Qh'], k, par['r'] ** 2))
                            for a in range(1, AMAX + 1)])
            over = np.flatnonzero(rad > tol)
            tau = (over[0] if len(over) else AMAX) * DT
            wait.append(dict(tolerance_mm=1e3 * tol, phase=k + 1, tau_star_s=tau, wait_delta_0p1_s=max(tau - 0.1 - DT, 0.0),
                             wait_delta_0p3_s=max(tau - 0.3 - DT, 0.0), radius_at_delay_0p1_mm=1e3 * rad[3], radius_at_delay_0p3_mm=1e3 * rad[9]))
    wait = pd.DataFrame(wait)
    wait.to_csv(os.path.join(RESULTS, 'p2_wait_table.csv'), index=False)
    save_json(dict(us_per_frame={'%s|%s' % k: v for k, v in timing.items()}, n_trials={s: len(T[s]) for s in T}, window_s=W * DT,
                   burst_frames=BURST, budget_cap=BUDGET_CAP, gmin_frames=GMIN, stage_cost_mm=(1e3 * f_stage[:, [2, 5, 14, 29, 59, 89]]).tolist(),
                   blind_cost_mm=(1e3 * f_blind[0, [2, 5, 14, 29, 59, 89]]).tolist()), 'p2_policy_meta.json')
    pd.set_option('display.width', 280); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 400)
    show = ['split', 'delay_s', 'policy', 'predictor', 'param', 'mean_err_mm', 'q90_err_mm', 'requests', 'pending_s', 'mean_info_age_s',
            'cov90', 'radius90_mm', 'exceed_30mm', 'exceed_50mm', 'exceed_80mm']
    print(fr[(fr.split == 'test') & (fr.delay_s < 0.2) & (fr.policy == 'trigger')][show].round(3).to_string())
    showb = ['split', 'delay_s', 'budget', 'method', 'mean_err_mm', 'requests', 'exceed_50mm', 'exceed_80mm', 'pending_s', 'cov90',
             'diff_vs_periodic_model_mm', 'diff_lo', 'diff_hi', 'dp_expected_mm', 'env_mean_err_mm', 'env_sd_err_mm']
    print(bud[bud.split == 'test'][showb].round(3).to_string())
    print(cons.round(3).to_string())
    print(wait.round(3).to_string())
    print('timing:', {str(k): round(v, 3) for k, v in timing.items()})
    print('done in %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
