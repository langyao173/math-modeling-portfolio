# -*- coding: utf-8 -*-
"""Problem 3 - duration beliefs and scheduling policies (decision rules frozen in results/frozen_p3_policy.json).

Belief      mixture of two experts for log d of every pending job
              E1  family constant                     centre = mu_f
              E2  family constant + 0.2-s prefix      centre = mu_f + beta' z(x)        (egocentric state features of Problem 1)
            each expert carries its own Kalman-filtered pool shift; expert weights follow the sequential predictive
            likelihood of the realised durations returned by executed jobs (Bayesian model averaging, reset every trial)
Decision    one-step rollout (lookahead) over sampled duration scenarios with a dynamic-priority base rule;
            the reservation is the quantile of the predictive service time that maximises the expected objective
Objective   U = weighted on-time rate - tardiness share - rho * (overrun + reserved idle) / H      (rho = 0.3, p3_run.RHO)
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_core import EPS, PREFIX_LEVELS
from p3_duration import FAMS, prefix_features
from ssm import njit

Q_GRID = np.array([0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90])


# =============================================================================================== belief
class Belief(object):
    def __init__(self, fz, mode='bma', learn=True):
        self.fz = fz; self.mode = mode; self.learn = learn
        tg2, tf2 = fz['tau_g'] ** 2, fz['tau_f'] ** 2
        self.sig = [fz['sigma'], fz['sigma'] * fz['ridge']['0']['oof_sd'] / fz['oof_sd_family']]
        self.m = [np.zeros(4), np.zeros(4)]
        self.V = [np.diag([tg2, tf2, tf2, tf2]), np.diag([tg2, tf2, tf2, tf2])]
        self.logw = np.log(np.array({'bma': [0.5, 0.5], 'family': [1.0, 1e-300], 'feat': [1e-300, 1.0]}[mode]))
        self.resid = np.asarray(fz['resid_std'])
        self.feat_center = {}                      # job_id -> (level, beta'z)
        self.n_obs = 0

    @staticmethod
    def _h(fam):
        h = np.zeros(4); h[0] = 1.0; h[1 + FAMS.index(fam)] = 1.0
        return h

    def weights(self):
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def set_prefix(self, job_id, level, px):
        """Compute the feature contribution of expert E2 from the numerical prefix (level = index of PREFIX_LEVELS)."""
        if job_id in self.feat_center and self.feat_center[job_id][0] >= level:
            return
        r = self.fz['ridge'][str(level)]
        f = prefix_features(px)
        x = np.array([f[k] for k in self.fz['features']])
        self.feat_center[job_id] = (level, float(((x - np.array(r['mean'])) / np.array(r['scale'])) @ np.array(r['coef'])))

    def centers(self, job_id, fam):
        c1 = self.fz['mu'][fam]
        c2 = c1 + (self.feat_center[job_id][1] if job_id in self.feat_center else 0.0)
        return c1, c2

    def update(self, job_id, fam, log_d):
        """Execution feedback: update expert weights (predictive likelihood) and the pool shift of each expert."""
        self.n_obs += 1
        if not self.learn:
            return
        h = self._h(fam)
        cs = self.centers(job_id, fam)
        for k in range(2):
            S = h @ self.V[k] @ h + self.sig[k] ** 2
            e = log_d - cs[k] - h @ self.m[k]
            self.logw[k] += -0.5 * (math.log(2 * math.pi * S) + e * e / S)
            K = self.V[k] @ h / S
            self.m[k] = self.m[k] + K * e
            self.V[k] = self.V[k] - np.outer(K, h @ self.V[k])

    def sample(self, jobs, n_scen, rng):
        """Scenario matrix (n_scen, len(jobs)) of annotated durations d; jobs = list of (job_id, family)."""
        w = self.weights()
        which = rng.random(n_scen) < w[1]                       # True -> expert E2
        out = np.empty((n_scen, len(jobs)))
        H = np.stack([self._h(f) for _, f in jobs])
        for k in range(2):
            idx = np.flatnonzero(which == (k == 1))
            if len(idx) == 0:
                continue
            Lc = np.linalg.cholesky(self.V[k] + 1e-12 * np.eye(4))
            sh = self.m[k][None, :] + rng.standard_normal((len(idx), 4)) @ Lc.T
            cen = np.array([self.centers(j, f)[k] for j, f in jobs])
            e = self.resid[rng.integers(0, len(self.resid), size=(len(idx), len(jobs)))]
            out[idx] = np.exp(cen[None, :] + sh @ H.T + self.sig[k] * e)
        return out


# =============================================================================================== fast model simulator
@njit(cache=True)
def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@njit(cache=True)
def base_rollout(P, rel, due, w, zx, zy, pmask, status0, t0, cx0, cy0, cost0, H, B, W, wt_max, rho, b0, mean_p, mu_d, s_d, mad,
                 alpha, beta, k_atc, lam_late, p_min, wo0, tard_done0, over0, idle0, act, act_job, act_b, act_wait,
                 fee_fixed, fee_res, fee_dist, changeover):
    """Mean objective over the scenario matrix P (N, n) of service durations when the first action is `act`
    (0 none, 1 execute act_job with reservation act_b, 2 wait act_wait seconds, 3 stop) and the base rule follows."""
    N, n = P.shape
    total = 0.0
    status = np.empty(n, dtype=np.int64)
    comp = np.empty(n)
    for sc in range(N):
        for j in range(n):
            status[j] = status0[j]; comp[j] = -1.0
        t = t0; cx = cx0; cy = cy0; cost = cost0
        wo = wo0; over = over0; idle = idle0; tard_future = 0.0
        alive = True
        first = act
        while alive:
            jsel = -1; bsel = 0.0
            if first == 3:
                break
            if first == 2:
                t = min(t + act_wait, H); first = 0
                continue
            if first == 1:
                jsel = act_job; bsel = act_b; first = 0
            else:
                best = -1.0
                mbar = 0.0; cnt = 0
                for j in range(n):
                    if status[j] == 0:
                        mbar += mean_p[j]; cnt += 1
                if cnt == 0:
                    break
                mbar /= cnt
                nxt_rel = 1e18
                for j in range(n):
                    if status[j] != 0:
                        continue
                    if rel[j] > t + 1e-9:
                        if rel[j] < nxt_rel:
                            nxt_rel = rel[j]
                        continue
                    ok = True
                    for q in range(n):
                        if (pmask[j] >> q) & 1 and status[q] != 1:
                            ok = False
                    if not ok:
                        continue
                    D = math.sqrt((zx[j] - cx) ** 2 + (zy[j] - cy) ** 2)
                    tau = D + changeover
                    fee = fee_fixed + fee_res * b0[j] + fee_dist * D
                    if cost + fee > B + 1e-9 or t + tau + b0[j] > H + 1e-9:
                        continue
                    # probability of finishing before the horizon / before the due time (shifted log-normal approximation)
                    xh = (H - t - tau - beta) / alpha
                    ph = _ncdf((math.log(xh) - mu_d[j]) / s_d[j]) if xh > 1e-9 else 0.0
                    if ph < p_min:
                        continue
                    xd = (due[j] - t - tau - beta) / alpha
                    pot = _ncdf((math.log(xd) - mu_d[j]) / s_d[j]) if xd > 1e-9 else 0.0
                    cexp = t + tau + mean_p[j]
                    gain = w[j] / W * pot + w[j] / wt_max * max(H - max(cexp, due[j]), 0.0) * ph
                    if gain <= rho * mad[j] / H:
                        continue
                    slack = due[j] - cexp
                    idx = w[j] * (pot + lam_late) * math.exp(-max(slack, 0.0) / (k_atc * mbar)) / (tau + mean_p[j])
                    if idx > best:
                        best = idx; jsel = j
                if jsel < 0:
                    if nxt_rel < H - 1e-9:
                        t = nxt_rel
                        continue
                    break
                bsel = b0[jsel]
            # execute jsel with reservation bsel
            D = math.sqrt((zx[jsel] - cx) ** 2 + (zy[jsel] - cy) ** 2)
            s = t + D + changeover
            fee = fee_fixed + fee_res * bsel + fee_dist * D
            if cost + fee > B + 1e-9 or s + bsel > H + 1e-9 or rel[jsel] > t + 1e-9:
                total += -10.0          # infeasible first action (never proposed by the policy; guards the search)
                alive = False
                first = -1
                break
            cost += fee
            p = P[sc, jsel]
            cx = zx[jsel]; cy = zy[jsel]
            if s + p > H + 1e-9:
                status[jsel] = 2
                over += max(min(p, H - s) - bsel, 0.0)
                t = H
                break
            status[jsel] = 1; comp[jsel] = s + p
            if s + p <= due[jsel] + 1e-9:
                wo += w[jsel]
            over += max(p - bsel, 0.0); idle += max(bsel - p, 0.0)
            t = s + max(p, bsel)
            if t >= H - 1e-9:
                break
        if first == -1:
            continue
        tard = tard_done0
        for j in range(n):
            if status0[j] != 0:
                continue                                   # already accounted for in tard_done0
            c = comp[j] if status[j] == 1 else H
            tard += w[j] * max(c - due[j], 0.0)
        total += wo / W - tard / wt_max - rho * (over + idle) / H
    return total / N


# =============================================================================================== policies
class PolicyBase(object):
    name = 'base'

    def __init__(self, scen, fz, params):
        self.S = scen; self.fz = fz; self.par = dict(params)

    # ---- helpers shared by all policies
    def _static(self, env):
        jobs = env.public['jobs']
        n = len(jobs)
        idx = {j['job_id']: i for i, j in enumerate(jobs)}
        zx = np.array([env.public['zones'][j['zone']][0] for j in jobs]); zy = np.array([env.public['zones'][j['zone']][1] for j in jobs])
        pmask = np.zeros(n, dtype=np.int64)
        for i, j in enumerate(jobs):
            for p in j['preds']:
                pmask[i] |= (1 << idx[p])
        return dict(n=n, idx=idx, rel=np.array([j['release'] for j in jobs]), due=np.array([j['due'] for j in jobs]),
                    w=np.array([j['weight'] for j in jobs]), zx=zx, zy=zy, pmask=pmask, W=float(sum(j['weight'] for j in jobs)),
                    wt_max=float(sum(j['weight'] * max(env.public['H'] - j['due'], 0.0) for j in jobs)), fam=[j['family'] for j in jobs],
                    ids=[j['job_id'] for j in jobs])

    def feasible(self, env, st, i, b):
        j = env.public['jobs'][i]
        if env.status[i] != 'pending' or env.now < j['release'] - EPS:
            return False
        if any(env.status[st['idx'][p]] != 'completed' for p in j['preds']):
            return False
        D = self.S.dist(env.zone, j['zone'])
        fee = self.S.fee_fixed + self.S.fee_reserve * b + self.S.fee_dist * D
        return env.cost + fee <= env.B + EPS and env.now + D + self.S.changeover + b <= env.H + EPS

    def next_release(self, env):
        r = [j['release'] for i, j in enumerate(env.public['jobs']) if env.status[i] == 'pending' and j['release'] > env.now + EPS]
        return min(r) if r else None


class SimpleRule(PolicyBase):
    """Transparent baselines: FIFO / EDD order, fixed family-level reservation, no learning, no review."""

    def __init__(self, scen, fz, params, order='edd', reserve='family_median'):
        PolicyBase.__init__(self, scen, fz, params)
        self.order = order; self.reserve = reserve
        self.name = '%s|%s' % (order.upper(), reserve)

    def reservation(self, env, fam):
        if self.reserve == 'min':
            return self.S.reserve_min
        med = math.exp(self.fz['mu'][fam])                                   # training median of the annotated duration
        mean = math.exp(self.fz['mu'][fam] + 0.5 * self.fz['oof_sd_family'] ** 2)
        d = med if self.reserve == 'family_median' else mean
        return max(self.S.reserve_min, env.alpha * d + env.beta)

    def run(self, env, rng):
        st = self._static(env)
        jobs = env.public['jobs']
        while not env.done:
            cand = [i for i in range(st['n']) if self.feasible(env, st, i, self.reservation(env, jobs[i]['family']))]
            if cand:
                key = (lambda i: (jobs[i]['release'], jobs[i]['job_id'])) if self.order == 'fifo' else (lambda i: (jobs[i]['due'], jobs[i]['job_id']))
                i = min(cand, key=key)
                env.execute(jobs[i]['job_id'], self.reservation(env, jobs[i]['family']))
            else:
                nr = self.next_release(env)
                if nr is not None and nr < env.H - EPS:
                    env.wait(nr - env.now)
                else:
                    env.stop()


class Lookahead(PolicyBase):
    """Belief-based policy: dynamic-priority base rule, optional one-step rollout, optional paid reviews."""

    def __init__(self, scen, fz, params, mode='bma', learn=True, rollout=True, review='none', rho=0.3, q_min=0.5, clairvoyant=False,
                 name=None):
        PolicyBase.__init__(self, scen, fz, params)
        self.mode, self.learn, self.rollout, self.review, self.rho, self.clair = mode, learn, rollout, review, rho, clairvoyant
        self.q_min = q_min                       # lowest admissible predictive quantile for a reservation (0 = minimum 0.2 s)
        self.name = name or 'LA|%s|learn=%d|rollout=%d|review=%s|rho=%g' % (mode, learn, rollout, review, rho)
        self.trace = None                        # set to a list to record the candidate values of every decision epoch (figures only)

    def run(self, env, rng):
        st = self._static(env)
        par = self.par
        jobs = env.public['jobs']; n = st['n']
        bel = Belief(self.fz, self.mode, self.learn)
        seen_fb = set()
        H, B = env.H, env.B
        truth = np.array(env._p) if self.clair else None            # clairvoyant benchmark only (uses hidden durations)
        self.voi_log = []
        while not env.done:
            # ---- information that has become available
            for i, j in enumerate(jobs):
                if env.status[i] == 'pending' and env.released(i) and self.mode != 'family':
                    lv = env.level[i]
                    if j['job_id'] not in bel.feat_center or bel.feat_center[j['job_id']][0] < lv:
                        bel.set_prefix(j['job_id'], lv, env.prefix(j['job_id']))
            for jid, p in env.feedback.items():
                if jid not in seen_fb:
                    seen_fb.add(jid)
                    d = max((p - env.beta) / env.alpha, 1e-3)
                    bel.update(jid, jobs[st['idx'][jid]]['family'], math.log(d))
            pend = [i for i in range(n) if env.status[i] == 'pending']
            if not pend:
                env.stop(); break
            # ---- scenarios of service durations for the pending jobs
            Ns = 1 if self.clair else par['n_scen']
            P = np.zeros((Ns, n))
            if self.clair:
                P[0, :] = truth
            else:
                Dm = bel.sample([(jobs[i]['job_id'], jobs[i]['family']) for i in pend], Ns, rng)
                P[:, pend] = env.alpha * Dm + env.beta
            b_q = np.zeros((n, len(Q_GRID))); b0 = np.full(n, self.S.reserve_min); mean_p = np.ones(n); mu_d = np.zeros(n); s_d = np.ones(n); mad = np.zeros(n)
            for i in pend:
                col = P[:, i]
                if self.clair:
                    b_q[i, :] = col[0]; b0[i] = max(self.S.reserve_min, col[0]); mean_p[i] = col[0]
                    mu_d[i] = math.log(max((col[0] - env.beta) / env.alpha, 1e-3)); s_d[i] = 1e-3; mad[i] = 0.0
                else:
                    if self.q_min <= 0:
                        b_q[i, :] = self.S.reserve_min; b0[i] = self.S.reserve_min
                    else:
                        qg = np.maximum(Q_GRID, self.q_min)
                        b_q[i, :] = np.maximum(self.S.reserve_min, np.quantile(col, qg))
                        b0[i] = max(self.S.reserve_min, float(np.quantile(col, max(par['q0'], self.q_min))))
                    mean_p[i] = col.mean()
                    ld = np.log(np.maximum((col - env.beta) / env.alpha, 1e-3))
                    mu_d[i] = ld.mean(); s_d[i] = max(ld.std(), 1e-3); mad[i] = float(np.mean(np.abs(col - b0[i])))
            status0 = np.array([0 if env.status[i] == 'pending' else (1 if env.status[i] == 'completed' else 2) for i in range(n)], dtype=np.int64)
            wo0 = sum(jobs[i]['weight'] for i in range(n) if env.status[i] == 'completed' and env.completion[i] <= jobs[i]['due'] + EPS)
            tard0 = sum(jobs[i]['weight'] * max((env.completion[i] if env.status[i] == 'completed' else H) - jobs[i]['due'], 0.0)
                        for i in range(n) if env.status[i] != 'pending')
            cx, cy = env.public['zones'][env.zone]

            def value(act, job=0, b=0.0, wait=0.0):
                return base_rollout(P, st['rel'], st['due'], st['w'], st['zx'], st['zy'], st['pmask'], status0, env.now, cx, cy, env.cost, H, B,
                                    st['W'], st['wt_max'], self.rho, b0, mean_p, mu_d, s_d, mad, env.alpha, env.beta, par['k_atc'],
                                    par['lam_late'], par['p_min'], wo0, tard0, env.overrun, env.idle, act, job, b, wait,
                                    self.S.fee_fixed, self.S.fee_reserve, self.S.fee_dist, self.S.changeover)
            # ---- optional paid review (rule-based variants used for the information ablation)
            if self.review in ('once', 'twice') and env.reviews_used < env.K and env.cost + 1.0 <= B - 2.5 and env.now + 1.0 <= H - 5.0:
                cand_r = [i for i in pend if env.released(i) and env.n_reviews[i] < (1 if self.review == 'once' else 2)]
                if cand_r:
                    i = max(cand_r, key=lambda i: (jobs[i]['weight'], -jobs[i]['due']))
                    env.review(jobs[i]['job_id'], note=dict(rule=self.review))
                    continue
            # ---- candidate first actions
            def best_feasible_b(i):
                """Base reservation of job i, reduced to the largest feasible grid value if budget or horizon bind."""
                opts = sorted(set([float(b0[i])] + [float(x) for x in b_q[i] if x <= b0[i] + 1e-12]), reverse=True)
                for b in opts:
                    if self.feasible(env, st, i, b):
                        return b
                return None
            cand = {i: best_feasible_b(i) for i in pend}
            cand = {i: b for i, b in cand.items() if b is not None}
            nr = self.next_release(env)
            can_wait = nr is not None and nr < H - EPS
            if self.rollout:
                best_v, best = value(3), ('stop', None, None)
                tr = dict(t=env.now, zone=env.zone, cost=env.cost, values={'stop': best_v}, reserve={}) if self.trace is not None else None
                if can_wait:
                    v = value(2, wait=nr - env.now)
                    if tr is not None:
                        tr['values']['wait'] = v
                    if v > best_v + 1e-12:
                        best_v, best = v, ('wait', None, nr - env.now)
                for i, b in cand.items():
                    v = value(1, i, b)
                    if tr is not None:
                        tr['values'][jobs[i]['job_id']] = v
                    if v > best_v + 1e-12:
                        best_v, best = v, ('exec', i, b)
                if best[0] == 'exec' and self.q_min > 0 and not self.clair:         # second stage: reservation of the chosen job
                    i = best[1]
                    if tr is not None:
                        tr['reserve'][float(best[2])] = best_v
                    for b in np.unique(b_q[i]):
                        if abs(b - best[2]) > 1e-9 and self.feasible(env, st, i, float(b)):
                            v = value(1, i, float(b))
                            if tr is not None:
                                tr['reserve'][float(b)] = v
                            if v > best_v + 1e-12:
                                best_v, best = v, ('exec', i, float(b))
                if tr is not None:
                    tr.update(chosen=best[0], job=(jobs[best[1]]['job_id'] if best[0] == 'exec' else None), b=best[2],
                              samples=(P[:, best[1]].copy() if best[0] == 'exec' else None), pending=[jobs[q]['job_id'] for q in pend])
                    self.trace.append(tr)
            else:                                                    # base rule alone (no lookahead)
                pick, pv = None, -1e18
                for i, b in cand.items():
                    vi = self._index(env, st, i, b0, mean_p, mu_d, s_d, mad, par)
                    if vi is not None and vi > pv:
                        pv, pick = vi, i
                if pick is not None:
                    best = ('exec', pick, cand[pick])
                elif can_wait:
                    best = ('wait', None, nr - env.now)
                else:
                    best = ('stop', None, None)
            # review variant 'idle': buy a review only while the equipment would otherwise wait for a release (no time lost)
            if self.review == 'idle' and best[0] == 'wait' and best[2] >= 1.0 and env.reviews_used < env.K and env.cost + 1.0 <= B - 2.5:
                cand_r = [i for i in pend if env.released(i) and env.n_reviews[i] < 2]
                if cand_r:
                    i = max(cand_r, key=lambda i: (jobs[i]['weight'], -jobs[i]['due']))
                    env.review(jobs[i]['job_id'], note=dict(rule='idle'))
                    continue
            wts = bel.weights()
            if best[0] == 'exec':
                i = best[1]
                col = P[:, i]
                env.execute(jobs[i]['job_id'], float(best[2]), note=dict(
                    pred_median_s=round(float(np.median(col)), 3), pred_q10_s=round(float(np.quantile(col, 0.1)), 3),
                    pred_q90_s=round(float(np.quantile(col, 0.9)), 3), reserve_quantile=round(float(np.mean(col <= best[2])), 3),
                    w_prefix_expert=round(float(wts[1]), 3), n_feedback=bel.n_obs, n_candidates=len(cand)))
            elif best[0] == 'wait':
                env.wait(float(best[2]), note=dict(reason='wait for next release', n_candidates=len(cand)))
            else:
                env.stop(note=dict(reason='no action with positive expected gain', n_candidates=len(cand)))
        self.final_weights = bel.weights().tolist()

    def _index(self, env, st, i, b0, mean_p, mu_d, s_d, mad, par):
        """Priority index of the base rule for job i at the current state (None if the rule would skip the job)."""
        j = env.public['jobs'][i]
        D = self.S.dist(env.zone, j['zone']); tau = D + self.S.changeover
        t = env.now; H = env.H
        pend = [q for q in range(st['n']) if env.status[q] == 'pending']
        mbar = float(np.mean([mean_p[q] for q in pend]))
        xh = (H - t - tau - env.beta) / env.alpha
        ph = _ncdf((math.log(xh) - mu_d[i]) / s_d[i]) if xh > 1e-9 else 0.0
        if ph < par['p_min']:
            return None
        xd = (j['due'] - t - tau - env.beta) / env.alpha
        pot = _ncdf((math.log(xd) - mu_d[i]) / s_d[i]) if xd > 1e-9 else 0.0
        cexp = t + tau + mean_p[i]
        gain = j['weight'] / st['W'] * pot + j['weight'] / st['wt_max'] * max(H - max(cexp, j['due']), 0.0) * ph
        if gain <= self.rho * mad[i] / H:
            return None
        slack = j['due'] - cexp
        return j['weight'] * (pot + par['lam_late']) * math.exp(-max(slack, 0.0) / (par['k_atc'] * mbar)) / (tau + mean_p[i])
