# -*- coding: utf-8 -*-
"""Problem 3 - common replay environment (official protocol of Problem_3_Requirements.pdf).

The environment owns every hidden quantity (source identity, annotated duration, later samples).  A policy talks to
it only through
    env.public            public task table of the case and the numerical conditions
    env.now / zone / cost / reviews_used
    env.prefix(job_id)    numerical prefix currently visible for a released job (0.2 / 0.6 / 1.0 source seconds)
    env.feedback          realised service durations of jobs whose execution has finished
    env.execute(job, b) / env.review(job) / env.wait(dt) / env.stop()
so that compliance with the information rules can be checked from the process log.
"""
import csv
import json
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INTERIM, TASK_DIR

EPS = 1e-9
PREFIX_LEVELS = (0.2, 0.6, 1.0)
MAPPINGS = [(1.0, 0.5), (0.75, 0.5), (1.5, 0.5), (1.0, 0.0), (1.0, 1.0)]       # nominal first


def _read_csv(name):
    with open(os.path.join(TASK_DIR, name), encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


class Scenario(object):
    """Static task data shared by all trials."""

    def __init__(self):
        with open(os.path.join(TASK_DIR, 'common_conditions.json'), encoding='utf-8') as fh:
            self.cond = json.load(fh)
        self.cases = {c['case_id']: c for c in self.cond['cases']}
        self.zones = {z['zone_id']: (float(z['x']), float(z['y'])) for z in _read_csv('service_zones.csv')}
        self.jobs = {}
        for row in _read_csv('tasks.csv'):
            self.jobs.setdefault(row['case_id'], []).append(dict(
                job_id=row['job_id'], family=row['family_id'], release=float(row['release_s']), due=float(row['due_s']),
                weight=float(row['priority_weight']), zone=row['zone_id'],
                preds=[p for p in row['predecessors'].split('|') if p]))
        for c in self.jobs:
            self.jobs[c].sort(key=lambda j: j['job_id'])
        self.index = _read_csv('source_segment_index.csv')            # row order of the source index is preserved
        self.pools = {}
        for row in self.index:
            if row['eligible'] == '1':
                self.pools.setdefault((row['split'], row['family_id']), []).append(row)
        self.seeds = [int(s) for s in self.cond['validation_seeds']]
        o = self.cond['observation']; s = self.cond['service']; m = self.cond['movement']; f = self.cond['fees']
        self.review_delay, self.review_cost, self.review_max = float(o['delay_s']), float(o['cost']), int(o['max_per_job'])
        self.reserve_min = float(s['reserve_min_s'])
        self.speed, self.changeover = float(m['speed']), float(m['changeover_s'])
        self.fee_fixed, self.fee_reserve, self.fee_dist = float(f['service_fixed']), float(f['reserve_per_second']), float(f['distance_per_unit'])
        self.max_actions = int(self.cond['max_actions'])
        self.initial_zone = self.cond['initial_zone']

    def dist(self, a, b):
        (x1, y1), (x2, y2) = self.zones[a], self.zones[b]
        return math.hypot(x1 - x2, y1 - y2)

    def sample_sources(self, case_id, seed, split):
        """Reference sampler: random.Random(int(seed)), one choice per job in sorted job order."""
        rng = random.Random(int(seed))
        return [rng.choice(self.pools[(split, j['family'])]) for j in self.jobs[case_id]]


class PrefixStore(object):
    """Serves the numerical prefix of a source interval (graded release of the recorded demonstration)."""

    def __init__(self):
        Z = np.load(os.path.join(INTERIM, 'frames.npz'))
        self.Z = {k: Z[k] for k in ('t', 'L', 'R', 'Lq', 'Rq', 'head', 'headq', 'camL', 'camLq', 'camR', 'camRq', 'flag')}
        import pandas as pd
        rec = pd.read_csv(os.path.join(INTERIM, 'records.csv'))
        self.span = {u: (int(a), int(b)) for u, a, b in zip(rec['episode_uuid'], rec['start'], rec['stop'])}

    def get(self, unit, seconds):
        a, b = self.span[unit['episode_uuid']]
        t = self.Z['t'][a:b]
        t0 = float(unit['start_s'])
        m = np.flatnonzero((t >= t0 - EPS) & (t < t0 + seconds - EPS))
        g = a + m
        out = {k: self.Z[k][g].copy() for k in self.Z}          # original timestamps (record time, s) are part of the prefix
        out['seconds'] = seconds
        return out


class Trial(object):
    """One replay trial.  All accounting follows the attached rules; infeasible commands terminate the trial."""

    def __init__(self, scen, store, case_id, seed, split, alpha, beta, trial_no=0, log=True, dH=0.0, dB=0.0):
        self.S = scen; self.store = store
        self.case_id, self.seed, self.split, self.alpha, self.beta, self.trial_no = case_id, int(seed), split, float(alpha), float(beta), trial_no
        c = scen.cases[case_id]
        # dH / dB perturb the horizon and the budget; they are used only for the shadow-price analysis (default 0)
        self.H, self.B, self.K = float(c['horizon_s']) + dH, float(c['cost_budget']) + dB, int(c['observation_limit'])
        self.jobs = [dict(j) for j in scen.jobs[case_id]]
        self.idx = {j['job_id']: i for i, j in enumerate(self.jobs)}
        self._src = scen.sample_sources(case_id, seed, split)             # hidden
        self._p = [self.alpha * float(u['duration_s']) + self.beta for u in self._src]   # hidden realised service durations
        n = len(self.jobs)
        self.now, self.zone, self.cost, self.reviews_used = 0.0, scen.initial_zone, 0.0, 0
        self.status = ['pending'] * n                    # pending | completed | censored
        self.level = [0] * n                             # index into PREFIX_LEVELS
        self.n_reviews = [0] * n
        self.completion = [None] * n
        self.start = [None] * n
        self.reserved = [None] * n
        self.feedback = {}                               # job_id -> realised service duration (after execution)
        self.done = False
        self.invalid = 0
        self.n_actions = 0
        self.end_time = None
        self.overrun = 0.0; self.idle = 0.0; self.censored = 0
        self.log = [] if log else None
        self._note = None
        self.public = dict(case_id=case_id, H=self.H, B=self.B, K=self.K, alpha=self.alpha, beta=self.beta, jobs=self.jobs,
                           zones=scen.zones, initial_zone=scen.initial_zone)

    # ------------------------------------------------------------------ information access
    def released(self, i):
        return self.now >= self.jobs[i]['release'] - EPS

    def prefix(self, job_id):
        i = self.idx[job_id]
        if not self.released(i):
            raise PermissionError('prefix of an unreleased job requested')
        return self.store.get(self._src[i], PREFIX_LEVELS[self.level[i]])

    def info_state(self):
        return {j['job_id']: (PREFIX_LEVELS[self.level[i]] if self.released(i) else 0.0) for i, j in enumerate(self.jobs)}

    # ------------------------------------------------------------------ helpers
    def _entry(self, action, job=None, **kw):
        if self.log is None:
            return None
        e = dict(case_id=self.case_id, split=self.split, trial=self.trial_no, seed=self.seed, alpha=self.alpha, beta=self.beta,
                 step=self.n_actions, t_decision=round(self.now, 6), action=action, job_id=job, zone_before=self.zone,
                 cost_before=round(self.cost, 6), reviews_before=self.reviews_used,
                 info_s={k: v for k, v in self.info_state().items() if v > 0})
        e.update(kw)
        if getattr(self, '_note', None):
            e['policy_note'] = self._note
        self.log.append(e)
        return e

    def _finish(self, t):
        self.done = True
        self.end_time = min(t, self.H)

    def _violate(self, action, job, reason, **kw):
        self.invalid += 1
        e = self._entry(action, job, valid=False, reason=reason, **kw)
        self._finish(self.now)
        return False

    def _count(self):
        self.n_actions += 1
        if self.n_actions > self.S.max_actions:
            self.invalid += 1
            self._entry('limit', None, valid=False, reason='more than %d actions' % self.S.max_actions)
            self._finish(self.now)
            return False
        return True

    # ------------------------------------------------------------------ actions
    def execute(self, job_id, b, note=None):
        self._note = note
        if self.done or not self._count():
            return False
        if job_id not in self.idx:
            return self._violate('execute', job_id, 'unknown job', reserved_s=b)
        i = self.idx[job_id]; j = self.jobs[i]
        if not (isinstance(b, (int, float)) and math.isfinite(b)):
            return self._violate('execute', job_id, 'non-finite reservation', reserved_s=str(b))
        if not self.released(i):
            return self._violate('execute', job_id, 'job not released', reserved_s=b)
        if self.status[i] != 'pending':
            return self._violate('execute', job_id, 'job already attempted', reserved_s=b)
        if any(self.status[self.idx[p]] != 'completed' for p in j['preds']):
            return self._violate('execute', job_id, 'unmet predecessor', reserved_s=b)
        if b < self.S.reserve_min - EPS:
            return self._violate('execute', job_id, 'reservation below minimum', reserved_s=b)
        D = self.S.dist(self.zone, j['zone'])
        s = self.now + D / self.S.speed + self.S.changeover
        fee = self.S.fee_fixed + self.S.fee_reserve * b + self.S.fee_dist * D
        if self.cost + fee > self.B + EPS:
            return self._violate('execute', job_id, 'cost budget exceeded', reserved_s=b, fee=fee)
        if s + b > self.H + EPS:
            return self._violate('execute', job_id, 'reserved interval crosses the horizon', reserved_s=b, service_start=s)
        p = self._p[i]
        e = self._entry('execute', job_id, valid=True, reserved_s=round(b, 6), distance=round(D, 6), fee=round(fee, 6),
                        service_start=round(s, 6))
        self.cost += fee
        self.start[i], self.reserved[i] = s, b
        zone_after = j['zone']
        if s + p > self.H + EPS:                                   # service crosses the horizon: censored, trial ends at H
            self.status[i] = 'censored'; self.censored += 1
            over = max(min(p, self.H - s) - b, 0.0)
            self.overrun += over
            self.zone = zone_after
            if e is not None:
                e.update(completion=None, censored=True, overrun_s=round(over, 6), idle_s=0.0, t_after=self.H, zone_after=zone_after,
                         cost_after=round(self.cost, 6), source_ref=self._src[i]['unit_id'])
            self.now = self.H
            self._finish(self.H)
            return True
        C = s + p
        over, idle = max(p - b, 0.0), max(b - p, 0.0)
        self.status[i] = 'completed'; self.completion[i] = C
        self.overrun += over; self.idle += idle
        self.now = s + max(p, b); self.zone = zone_after
        self.feedback[job_id] = p
        if e is not None:
            e.update(completion=round(C, 6), censored=False, overrun_s=round(over, 6), idle_s=round(idle, 6), t_after=round(self.now, 6),
                     zone_after=zone_after, cost_after=round(self.cost, 6), realised_service_s=round(p, 6), source_ref=self._src[i]['unit_id'])
        if self.now >= self.H - EPS:
            self._finish(self.H)
        return True

    def review(self, job_id, note=None):
        self._note = note
        if self.done or not self._count():
            return False
        if job_id not in self.idx:
            return self._violate('review', job_id, 'unknown job')
        i = self.idx[job_id]
        if not self.released(i):
            return self._violate('review', job_id, 'job not released')
        if self.status[i] != 'pending':
            return self._violate('review', job_id, 'job already attempted')
        if self.n_reviews[i] >= self.S.review_max:
            return self._violate('review', job_id, 'more than two reviews of one job')
        if self.reviews_used >= self.K:
            return self._violate('review', job_id, 'review limit exceeded')
        if self.cost + self.S.review_cost > self.B + EPS:
            return self._violate('review', job_id, 'cost budget exceeded')
        if self.now + self.S.review_delay > self.H + EPS:
            return self._violate('review', job_id, 'review return time crosses the horizon')
        e = self._entry('review', job_id, valid=True, fee=self.S.review_cost, return_time=round(self.now + self.S.review_delay, 6))
        self.cost += self.S.review_cost; self.reviews_used += 1; self.n_reviews[i] += 1
        self.now += self.S.review_delay
        self.level[i] += 1                                          # the longer prefix becomes visible after the delay
        if e is not None:
            e.update(prefix_after_s=PREFIX_LEVELS[self.level[i]], t_after=round(self.now, 6), zone_after=self.zone, cost_after=round(self.cost, 6))
        if self.now >= self.H - EPS:
            self._finish(self.H)
        return True

    def wait(self, dt, note=None):
        self._note = note
        if self.done or not self._count():
            return False
        if not (isinstance(dt, (int, float)) and math.isfinite(dt)) or dt <= 0:
            return self._violate('wait', None, 'non-positive or non-finite waiting time', wait_s=str(dt))
        if self.now + dt > self.H + EPS:
            return self._violate('wait', None, 'waiting crosses the horizon', wait_s=dt)
        e = self._entry('wait', None, valid=True, wait_s=round(dt, 6))
        self.now += dt
        if e is not None:
            e.update(t_after=round(self.now, 6), zone_after=self.zone, cost_after=round(self.cost, 6))
        if self.now >= self.H - EPS:
            self._finish(self.H)
        return True

    def stop(self, note=None):
        self._note = note
        if self.done:
            return False
        self.n_actions += 1
        self._entry('stop', None, valid=True, t_after=round(self.now, 6), zone_after=self.zone, cost_after=round(self.cost, 6))
        self._finish(self.now)
        return True

    # ------------------------------------------------------------------ common metrics
    def metrics(self):
        if not self.done:
            self._finish(self.now)
        W = sum(j['weight'] for j in self.jobs)
        wc = wo = tard = 0.0
        for i, j in enumerate(self.jobs):
            if self.status[i] == 'completed':
                C = self.completion[i]
                wc += j['weight']
                if C <= j['due'] + EPS:
                    wo += j['weight']
            else:
                C = self.H
            tard += j['weight'] * max(C - j['due'], 0.0)
        return dict(case_id=self.case_id, split=self.split, trial=self.trial_no, seed=self.seed, alpha=self.alpha, beta=self.beta,
                    weighted_completion_rate=wc / W, weighted_on_time_rate=wo / W, weighted_tardiness_s=tard, cost=self.cost,
                    observations=self.reviews_used, elapsed_s=self.end_time, observed_overrun_s=self.overrun, reserved_idle_s=self.idle,
                    censored_units=self.censored, invalid_actions=self.invalid, valid_protocol=int(self.invalid == 0),
                    executed_jobs=sum(s != 'pending' for s in self.status), completed_jobs=sum(s == 'completed' for s in self.status),
                    n_actions=self.n_actions)

    def max_tardiness(self):
        """Weighted tardiness if nothing were completed (normalising constant of the custom objective)."""
        return sum(j['weight'] * max(self.H - j['due'], 0.0) for j in self.jobs)


def utility(m, H, wt_max, rho=1.0):
    """Custom objective of this report (not comparable across teams):
       U = weighted on-time rate - tardiness share - rho * (overrun + reserved idle) / H."""
    return m['weighted_on_time_rate'] - m['weighted_tardiness_s'] / wt_max - rho * (m['observed_overrun_s'] + m['reserved_idle_s']) / H
