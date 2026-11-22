# -*- coding: utf-8 -*-
"""Problem 3 - experiment runner: common replay of every policy on identical sampled jobs.

Usage
    python p3_run.py validation1           # 30 public seeds x 3 cases x 5 mappings: baselines and candidate belief configurations
    (python p3_pooltransfer.py             # selects the final configuration and writes results/frozen_p3_policy.json)
    python p3_run.py validation2           # final policy (with process logs), ablations and sensitivity variants built around it
    python p3_run.py test                  # final policy, baselines and candidates on the test pool with the report's own 30 seeds
    python p3_run.py custom --split test --seeds 1,2,3 --policies FROZEN,EDD-median --tag final
Outputs (results/): p3_trials_<tag>.csv (one row per policy x trial with every common metric),
                    p3_summary_<tag>.csv, logs/p3_process_log_<policy>_<tag>.jsonl.gz
FROZEN is the final decision rule of results/frozen_p3_policy.json.
The policy's internal Monte-Carlo stream is seeded by a constant, never by the trial seed.
"""
import argparse
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LOGS, RESULTS, load_json, save_json
from p3_core import MAPPINGS, PrefixStore, Scenario, Trial, utility
from p3_policy import Lookahead, SimpleRule

POLICY_SEED = 20261120
RHO = 0.3                                      # weight of the capacity-planning error, anchored on the scenario fees (0.1 H / B)
BASE_PAR = dict(n_scen=100, q0=0.4, k_atc=4.0, lam_late=0.15, p_min=0.5)     # fixed before the replay; plateau check on training-pool trials: p3_tune.py
TEST_SEEDS = list(range(2026112400, 2026112430))   # seeds of this report for the test pool (the organisers' final-test seeds are not published)
CASES = ('S1', 'S2', 'S3')

_G = {}


def registry(scen, fz):
    R = {}
    R['FIFO-mean'] = lambda: SimpleRule(scen, fz, BASE_PAR, 'fifo', 'family_mean')
    R['EDD-median'] = lambda: SimpleRule(scen, fz, BASE_PAR, 'edd', 'family_median')
    R['EDD-min'] = lambda: SimpleRule(scen, fz, BASE_PAR, 'edd', 'min')
    R['Index-static'] = lambda: Lookahead(scen, fz, BASE_PAR, mode='family', learn=False, rollout=False, rho=RHO, q_min=0.5, name='Index-static')
    for mode in ('family', 'feat', 'bma'):
        for learn in (0, 1):
            nm = 'RO-%s-%s' % (mode, 'learn' if learn else 'static')
            R[nm] = (lambda mode=mode, learn=learn, nm=nm: Lookahead(scen, fz, BASE_PAR, mode=mode, learn=bool(learn), rollout=True, rho=RHO,
                                                                     q_min=0.5, name=nm))
    return R


def add_variants(R, scen, fz, sel):
    """Ablations and sensitivity variants built around the selected configuration `sel` = dict(mode, learn)."""
    mk = lambda **kw: Lookahead(scen, fz, BASE_PAR, **{**dict(mode=sel['mode'], learn=sel['learn'], rollout=True, rho=RHO, q_min=0.5), **kw})
    R['FROZEN'] = lambda: mk(name='FROZEN')
    R['ABL-no-rollout'] = lambda: mk(rollout=False, name='ABL-no-rollout')
    R['ABL-review-once'] = lambda: mk(review='once', mode='feat' if sel['mode'] == 'family' else sel['mode'], name='ABL-review-once')
    R['ABL-review-twice'] = lambda: mk(review='twice', mode='feat' if sel['mode'] == 'family' else sel['mode'], name='ABL-review-twice')
    R['ABL-review-idle'] = lambda: mk(review='idle', mode='feat' if sel['mode'] == 'family' else sel['mode'], name='ABL-review-idle')
    R['RES-min'] = lambda: mk(q_min=0.0, rho=0.0, name='RES-min')
    R['RES-q25'] = lambda: mk(q_min=0.25, name='RES-q25')
    R['RES-q75'] = lambda: mk(q_min=0.75, name='RES-q75')
    R['RES-q90'] = lambda: mk(q_min=0.90, name='RES-q90')
    R['RHO-0.1'] = lambda: mk(rho=0.1, name='RHO-0.1')
    R['RHO-1.0'] = lambda: mk(rho=1.0, name='RHO-1.0')
    R['CLAIRVOYANT'] = lambda: mk(clairvoyant=True, name='CLAIRVOYANT')
    return R


def _init(sel):
    _G['scen'] = Scenario(); _G['store'] = PrefixStore(); _G['fz'] = load_json('frozen_p3_duration.json')
    R = registry(_G['scen'], _G['fz'])
    if sel is not None:
        add_variants(R, _G['scen'], _G['fz'], sel)
    _G['R'] = R


def _work(job):
    pname, split, tasks, want_log = job
    scen, store = _G['scen'], _G['store']
    pol = _G['R'][pname]()
    rows, logs = [], []
    t0 = time.perf_counter()
    for (case, tno, seed, alpha, beta) in tasks:
        env = Trial(scen, store, case, seed, split, alpha, beta, trial_no=tno, log=want_log)
        pol.run(env, np.random.default_rng(POLICY_SEED))
        m = env.metrics()
        m['policy'] = pname
        m['utility'] = utility(m, env.H, env.max_tardiness(), RHO)
        m['wt_max'] = env.max_tardiness(); m['H'] = env.H; m['B'] = env.B
        rows.append(m)
        if want_log:
            for e in env.log:
                e['policy'] = pname
            logs.extend(env.log)
    return pname, rows, logs, time.perf_counter() - t0


def run(policies, split, seeds, mappings, tag, sel=None, log_policies=(), procs=2):
    tasks = [(case, tno, seed, a, b) for (a, b) in mappings for case in CASES for tno, seed in enumerate(seeds)]
    jobs = []
    for p in policies:
        # split each policy's trials into two chunks so that both workers stay busy
        half = len(tasks) // 2
        jobs.append((p, split, tasks[:half], p in log_policies)); jobs.append((p, split, tasks[half:], p in log_policies))
    rows, logs, secs = [], {}, {}
    with Pool(procs, initializer=_init, initargs=(sel,)) as pool:
        for pname, r, lg, dt in pool.imap_unordered(_work, jobs):
            rows.extend(r); secs[pname] = secs.get(pname, 0.0) + dt
            if lg:
                logs.setdefault(pname, []).extend(lg)
            print('  %-18s chunk done (%.0fs)' % (pname, dt), flush=True)
    df = pd.DataFrame(rows)
    order = ['policy', 'case_id', 'split', 'trial', 'seed', 'alpha', 'beta', 'weighted_completion_rate', 'weighted_on_time_rate',
             'weighted_tardiness_s', 'cost', 'observations', 'elapsed_s', 'observed_overrun_s', 'reserved_idle_s', 'censored_units',
             'invalid_actions', 'valid_protocol', 'executed_jobs', 'completed_jobs', 'n_actions', 'utility', 'wt_max', 'H', 'B']
    df = df[order].sort_values(['policy', 'alpha', 'beta', 'case_id', 'trial']).reset_index(drop=True)
    df.to_csv(os.path.join(RESULTS, 'p3_trials_%s.csv' % tag), index=False)
    for pname, lg in logs.items():
        lg.sort(key=lambda e: (e['alpha'], e['beta'], e['case_id'], e['trial'], e['step']))
        with gzip.open(os.path.join(LOGS, 'p3_process_log_%s_%s.jsonl.gz' % (pname, tag)), 'wt', encoding='utf-8') as fh:
            for e in lg:
                fh.write(json.dumps(e) + '\n')
    summarise(df, tag)
    save_json({k: round(v, 1) for k, v in secs.items()}, 'p3_runtime_%s.json' % tag)
    return df


METRICS = ['utility', 'weighted_completion_rate', 'weighted_on_time_rate', 'weighted_tardiness_s', 'cost', 'observations', 'elapsed_s',
           'observed_overrun_s', 'reserved_idle_s', 'censored_units', 'invalid_actions', 'valid_protocol', 'completed_jobs']


def summarise(df, tag):
    g = df.groupby(['policy', 'alpha', 'beta', 'case_id'])
    s = g[METRICS].agg(['mean', 'std'])
    s.columns = ['%s_%s' % c for c in s.columns]
    s['n'] = g.size()
    s = s.reset_index()
    s.to_csv(os.path.join(RESULTS, 'p3_summary_%s.csv' % tag), index=False)
    return s


def paired(df, a, b, metric='utility', nominal=True):
    """Paired differences a - b per case (same trials): mean, standard error, 95 % CI."""
    d = df[(df.alpha == 1.0) & (df.beta == 0.5)] if nominal else df
    out = []
    for case in list(CASES) + ['ALL']:
        da = d[(d.policy == a)] if case == 'ALL' else d[(d.policy == a) & (d.case_id == case)]
        db = d[(d.policy == b)] if case == 'ALL' else d[(d.policy == b) & (d.case_id == case)]
        m = da.merge(db, on=['case_id', 'trial', 'seed', 'alpha', 'beta'], suffixes=('_a', '_b'))
        x = (m[metric + '_a'] - m[metric + '_b']).to_numpy()
        se = x.std(ddof=1) / np.sqrt(len(x))
        out.append(dict(a=a, b=b, metric=metric, case=case, n=len(x), mean_a=float(m[metric + '_a'].mean()), mean_b=float(m[metric + '_b'].mean()),
                        diff=float(x.mean()), se=float(se), lo=float(x.mean() - 1.96 * se), hi=float(x.mean() + 1.96 * se),
                        win_share=float((x > 1e-12).mean())))
    return out


def select(df):
    """Selection of the belief configuration on the validation trials (nominal mapping): one-standard-error rule,
    candidates ordered from the simplest to the most complex."""
    order = ['RO-family-static', 'RO-family-learn', 'RO-feat-static', 'RO-feat-learn', 'RO-bma-static', 'RO-bma-learn']
    d = df[(df.alpha == 1.0) & (df.beta == 0.5)]
    means = {p: d[d.policy == p].utility.mean() for p in order}
    best = max(order, key=lambda p: means[p])
    rows = []
    chosen = best
    found = False
    for p in order:
        pr = paired(df, best, p)[-1]
        rows.append(dict(policy=p, mean_utility=means[p], diff_best_minus=pr['diff'], se=pr['se']))
        if not found and pr['diff'] <= pr['se'] + 1e-12:
            chosen = p; found = True
    sel = pd.DataFrame(rows)
    sel['best'] = sel.policy == best; sel['selected'] = sel.policy == chosen
    return sel, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['validation1', 'validation2', 'test', 'custom'])
    ap.add_argument('--split', default='validation'); ap.add_argument('--seeds', default=''); ap.add_argument('--policies', default='')
    ap.add_argument('--tag', default='custom'); ap.add_argument('--procs', type=int, default=2)
    a = ap.parse_args()
    scen = Scenario()
    t0 = time.time()
    if a.what == 'validation1':
        # stage 1: baselines and candidate belief configurations (all five mappings, 450 trials each).  The final
        # configuration is chosen afterwards by p3_pooltransfer.py (one-standard-error rule with the pool-level error).
        R = registry(scen, load_json('frozen_p3_duration.json'))
        df1 = run(list(R.keys()), 'validation', scen.seeds, MAPPINGS, 'validation_stage1', procs=a.procs)
        sel, chosen = select(df1)
        sel = sel.rename(columns={'se': 'trial_se', 'selected': 'selected_with_trial_se'})
        sel.to_csv(os.path.join(RESULTS, 'p3_policy_selection.csv'), index=False)
        print(sel.round(4).to_string(), '\nwith the trial-level standard error the rule would select:', chosen, flush=True)
    elif a.what == 'validation2':
        # stage 2: final policy with process logs + ablations / sensitivity variants built around it
        fzp = load_json('frozen_p3_policy.json')
        df1 = pd.read_csv(os.path.join(RESULTS, 'p3_trials_validation_stage1.csv'))
        extra = ['FROZEN', 'ABL-review-once', 'ABL-review-twice', 'ABL-review-idle', 'RES-min', 'RES-q25', 'RES-q75', 'RES-q90', 'RHO-0.1',
                 'RHO-1.0', 'CLAIRVOYANT']
        df2 = run(extra, 'validation', scen.seeds, MAPPINGS, 'validation_stage2', sel=dict(mode=fzp['mode'], learn=fzp['learn']),
                  log_policies=('FROZEN',), procs=a.procs)
        df = pd.concat([df1, df2], ignore_index=True)
        df.to_csv(os.path.join(RESULTS, 'p3_trials_validation.csv'), index=False)
        summarise(df, 'validation')
        rows = []
        for b in ('FIFO-mean', 'EDD-median', 'EDD-min', 'Index-static', 'ABL-review-once', 'ABL-review-twice', 'ABL-review-idle', 'RES-min',
                  'RES-q25', 'RES-q75', 'RES-q90', 'RHO-0.1', 'RHO-1.0', 'RO-family-static', 'RO-family-learn', 'RO-feat-static', 'RO-feat-learn',
                  'RO-bma-static', 'RO-bma-learn', 'CLAIRVOYANT'):
            for metric in ('utility', 'weighted_on_time_rate', 'weighted_completion_rate', 'weighted_tardiness_s', 'observed_overrun_s',
                           'reserved_idle_s', 'cost'):
                rows += paired(df, 'FROZEN', b, metric)
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'p3_paired_validation.csv'), index=False)
    elif a.what == 'test':
        fzp = load_json('frozen_p3_policy.json')
        sel = dict(mode=fzp['mode'], learn=fzp['learn'])
        pols = ['FROZEN', 'FIFO-mean', 'EDD-median', 'EDD-min', 'Index-static', 'RO-family-static', 'RO-family-learn', 'RO-feat-static',
                'RO-feat-learn', 'RO-bma-static', 'RO-bma-learn', 'ABL-review-once', 'ABL-review-idle', 'RES-min', 'RES-q75', 'CLAIRVOYANT']
        df = run(pols, 'test', TEST_SEEDS, MAPPINGS, 'test', sel=sel, log_policies=('FROZEN',), procs=a.procs)
        rows = []
        for b in pols[1:]:
            for metric in ('utility', 'weighted_on_time_rate', 'weighted_completion_rate', 'weighted_tardiness_s', 'observed_overrun_s',
                           'reserved_idle_s', 'cost'):
                rows += paired(df, 'FROZEN', b, metric)
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'p3_paired_test.csv'), index=False)
    else:
        fzp = load_json('frozen_p3_policy.json')
        seeds = [int(x) for x in a.seeds.split(',') if x]
        pols = [p for p in a.policies.split(',') if p] or ['FROZEN']
        run(pols, a.split, seeds, MAPPINGS, a.tag, sel=dict(mode=fzp['mode'], learn=fzp['learn']), log_policies=tuple(pols), procs=a.procs)
    print('done in %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
