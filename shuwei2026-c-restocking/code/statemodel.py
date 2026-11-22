# -*- coding: utf-8 -*-
"""State-model utilities shared by Problems 1 and 2: phase assignment, run lists, model scoring."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SPLITS, load_json
from ssm import GaussHMM, build_struct, kf_struct

HOR = np.array([1, 3, 6, 15, 30, 60, 90], dtype=np.int64)       # prediction horizons in frames (30 Hz)
HOR_S = HOR / 30.0
BURN = 15                                                        # frames of history before predictions are scored
R_MEAS = 5e-4                                                    # measurement-noise sd (m); ML estimate is ~0, fixed
P_INIT = np.diag([0.10 ** 2, 0.05 ** 2, 0.5 ** 2])               # prior covariance of (level, deviation, velocity)
DT = 1.0 / 30.0
SPEED_EPS = 0.01                                                 # m/s, floor inside the log of the phase feature


def run_list(D, min_s=1.0):
    """Reference runs (hand, rid, a, b) of the 30-Hz records, grouped by split."""
    runs = {s: [] for s in SPLITS}
    for j in range(2):
        for r, a, b in D.reference_runs(j, min_s, only_30hz=True):
            runs[D.split_of_rid[r]].append((j, r, a, b))
    return runs


def phase_feature(D, j, a, b):
    return np.log10(D.lagspeed[j][a:b] + SPEED_EPS)


def assign_phases(D, runs, hmm):
    """Causal phase (argmax of the forward-filtered probability) for every frame of every run."""
    ph = [np.zeros(D.N, dtype=np.int64) for _ in range(2)]
    pr = [np.zeros((D.N, hmm.K)) for _ in range(2)]
    for s in runs:
        for j, r, a, b in runs[s]:
            al = hmm.filter(phase_feature(D, j, a, b))
            pr[j][a:b] = al
            ph[j][a:b] = al.argmax(axis=1)
    return ph, pr


def par_from_vec(p, K, kind):
    """Natural-scale parameter dict from the log-parameter vector."""
    e = np.exp(p)
    if kind == 'DV':                       # damped velocity, no mean reversion, anchor frozen at the first sample
        return dict(kappa=1e-9, theta=e[0], sigma=e[1:1 + K], sigma_m=0.0, r=R_MEAS)
    return dict(kappa=e[0], theta=e[1], sigma=e[2:2 + K], sigma_m=e[2 + K], r=R_MEAS)


def p_init(kind):
    P = P_INIT.copy()
    if kind == 'DV':
        P[0, 0] = 0.0
    return P


def score_runs(D, runs, phase, par, K, kind, A_phase, hor=HOR, want=False, burn=BURN):
    """Multi-horizon predictive scores of the structural model on a list of runs.
    Returns per-run arrays nll (R,nh), cnt (R,nh), sse (R,nh) and, if want, the per-frame outputs."""
    F, Q, Fh, Qh = build_struct(par, DT, A_phase if K > 1 else np.ones((1, 1)), hor)
    Pi = p_init(kind)
    nll = np.zeros((len(runs), len(hor))); cnt = np.zeros_like(nll); sse = np.zeros_like(nll)
    extra = []
    zeros = None
    for i, (j, r, a, b) in enumerate(runs):
        ph = phase[j][a:b] if K > 1 else np.zeros(b - a, dtype=np.int64)
        o = kf_struct(D.xi[j][a:b], ph, F, Q, par['r'], Fh, Qh, hor, burn, np.full(3, np.nan), Pi, want)
        nll[i], cnt[i], sse[i] = o[0], o[1], o[2]
        if want:
            extra.append(o[3:])
    return nll, cnt, sse, extra


def load_frozen_state():
    fz = load_json('frozen_p1_state.json')
    hmm = GaussHMM.from_dict(fz['hmm'])
    par = dict(fz['params'])
    par['sigma'] = np.array(par['sigma'])
    return fz, hmm, par
