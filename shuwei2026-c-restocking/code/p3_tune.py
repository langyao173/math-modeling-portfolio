# -*- coding: utf-8 -*-
"""Problem 3 - sensitivity grid for the three parameters of the dynamic-priority base rule, evaluated on
TRAINING-pool trials only (60 seeds x 3 cases, nominal mapping, family-constant belief, reservation at the
predictive median, no rollout).  The grid is a check of the values fixed in p3_run.BASE_PAR
(k_atc = 4, lam_late = 0.15, p_min = 0.5): they lie on a flat plateau (0.004 utility units below the grid
optimum, far inside the spread across cases), so they were kept unchanged.  No validation or test trial is
used here.
Output: results/p3_tuning.csv"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, load_json
from p3_core import PrefixStore, Scenario, Trial, utility
from p3_policy import Lookahead

RHO = 0.3
SEEDS = list(range(1000, 1060))


def main():
    scen = Scenario(); store = PrefixStore(); fz = load_json('frozen_p3_duration.json')
    rows = []
    for k_atc, lam, pmin in itertools.product((1.0, 2.0, 4.0, 8.0, 16.0), (0.05, 0.15, 0.4), (0.3, 0.5, 0.7)):
        par = dict(n_scen=100, q0=0.4, k_atc=k_atc, lam_late=lam, p_min=pmin)
        pol = Lookahead(scen, fz, par, mode='family', learn=False, rollout=False, rho=RHO, q_min=0.5)
        r = dict(k_atc=k_atc, lam_late=lam, p_min=pmin)
        us = []
        for case in ('S1', 'S2', 'S3'):
            u, w = [], []
            for s in SEEDS:
                env = Trial(scen, store, case, s, 'train', 1.0, 0.5, log=False)
                pol.run(env, np.random.default_rng(20261120))
                m = env.metrics()
                u.append(utility(m, env.H, env.max_tardiness(), RHO)); w.append(m['weighted_on_time_rate'])
            r['U_' + case] = float(np.mean(u)); r['WOT_' + case] = float(np.mean(w)); us.append(np.mean(u))
        r['U'] = float(np.mean(us)); r['U_se'] = float(np.std(us) / np.sqrt(3))
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows).sort_values('U', ascending=False)
    df.to_csv(os.path.join(RESULTS, 'p3_tuning.csv'), index=False)
    print(df.head(10).round(4).to_string())


if __name__ == '__main__':
    main()
