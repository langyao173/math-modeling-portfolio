# -*- coding: utf-8 -*-
"""Problem 3 - selection of the belief configuration with a pool-level error (training and validation data only).

All trials of one pool sample the same few environments, so the trial-level standard error of a policy difference says
nothing about a NEW pool.  Here new pools are emulated inside the training split: four training environments are held
out, the duration model (family means, ridge coefficients, variance components, residual distribution) is re-estimated
on the other twelve, and the policies are replayed on jobs sampled from the held-out pool with the public seeds.  The
spread of a paired policy difference across emulated pools is its pool-level standard deviation, i.e. the standard
error that applies to the single validation pool.  The final belief configuration is the simplest candidate whose
validation gap to the best candidate lies within that pool-level standard deviation (one-standard-error rule).

Outputs (results/): p3_pooltransfer_pools.csv   one row per emulated pool (composition, duration error, mean objective per policy)
                    p3_pooltransfer_summary.csv paired differences: mean, pool-level sd, share of pools with a gain
                    p3_pooltransfer.json        pool-level one-standard-error rule applied to the validation differences
                    frozen_p3_policy.json       final decision rule (configuration selected by that rule)
                    p3_duration_by_env.csv      duration error of the family and the prefix model per held-out environment
                                                (training environments left out in turn; validation / test with the frozen model)
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, load_json, save_json
import p3_duration as pdur
from p3_core import PrefixStore, Scenario, Trial, utility
from p3_run import CASES, POLICY_SEED, RHO, registry

N_POOLS = 60
ENVS_PER_POOL = 4
N_SEEDS = 10                      # first ten public seeds: 30 trials per pool and policy (nominal mapping)
POLICIES = ['EDD-median', 'Index-static', 'RO-family-static', 'RO-family-learn', 'RO-feat-static', 'RO-feat-learn', 'RO-bma-static', 'RO-bma-learn']
PAIRS = [('RO-feat-static', 'RO-family-static'), ('RO-feat-learn', 'RO-family-static'), ('RO-bma-static', 'RO-family-static'),
         ('RO-bma-learn', 'RO-family-static'), ('RO-family-learn', 'RO-family-static'), ('RO-family-static', 'Index-static'),
         ('RO-family-static', 'EDD-median'), ('Index-static', 'EDD-median'), ('RO-feat-static', 'EDD-median')]
_G = {}


def fit_fold(tr):
    """Duration model of the main pipeline (p3_duration) re-estimated on a subset of the training environments."""
    mods = pdur.fit_models(tr)
    tau_g, tau_f, sigma, _ = pdur.shift_variances(tr, n_pools=150)
    r_fam = mods['oof']['family']
    fz = dict(selected='feat0', mu=mods['mu'], n_train=mods['n'], sigma=sigma, tau_g=tau_g, tau_f=tau_f,
              resid_std=((r_fam - r_fam.mean()) / r_fam.std()).tolist(), features=pdur.FEATS,
              ridge={str(li): dict(mean=mods['ridge'][li]['mean'].tolist(), scale=mods['ridge'][li]['scale'].tolist(),
                                   coef=mods['ridge'][li]['coef'].tolist(), alpha=float(mods['alpha'][li]),
                                   oof_sd=float(np.std(mods['oof']['feat%d' % li]))) for li in range(3)},
              oof_sd_family=float(r_fam.std()))
    return mods, fz


def by_environment(seg):
    """Held-out duration error per environment: every training environment left out in turn; frozen model elsewhere."""
    def row(split, e, te, mods):
        y = te.ld.to_numpy()
        cf = pdur.predict_center(mods, te, 'family'); cp = pdur.predict_center(mods, te, 'feat0')
        return dict(split=split, environment_id=e, n_units=len(te), families=''.join(sorted(te.family.unique())),
                    mae_family=float(np.mean(np.abs(y - cf))), mae_prefix=float(np.mean(np.abs(y - cp))),
                    bias_family=float(np.mean(y - cf)), bias_prefix=float(np.mean(y - cp)))
    tr = seg[seg.split == 'train'].reset_index(drop=True)
    pdur.RNG = np.random.default_rng(20261127)
    rows = []
    for e in sorted(tr.env.unique()):
        rows.append(row('train (left out)', e, tr[tr.env == e], pdur.fit_models(tr[tr.env != e].reset_index(drop=True))))
    fz = load_json('frozen_p3_duration.json')
    frozen = dict(mu=fz['mu'], ridge={0: {k: np.array(fz['ridge']['0'][k]) for k in ('mean', 'scale', 'coef')}})
    for sp in ('validation', 'test'):
        d = seg[seg.split == sp]
        for e in sorted(d.env.unique()):
            rows.append(row(sp, e, d[d.env == e], frozen))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, 'p3_duration_by_env.csv'), index=False)
    return out


def _init():
    _G['scen'] = Scenario(); _G['store'] = PrefixStore()
    seg = pd.read_csv(os.path.join(RESULTS, 'p3_segment_features.csv'))
    _G['tr'] = seg[seg.split == 'train'].reset_index(drop=True)


def _work(job):
    """One emulated pool: re-estimate the duration model without the pool, then replay every policy on the pool."""
    pid, pick, seeds = job
    scen, store, tr = _G['scen'], _G['store'], _G['tr']
    pdur.RNG = np.random.default_rng(20261125 + pid)                   # own random stream per pool (folds, nested pools)
    te = tr[tr.env.isin(pick)]; sub = tr[~tr.env.isin(pick)].reset_index(drop=True)
    mods, fz = fit_fold(sub)
    y = te.ld.to_numpy()
    c_fam = pdur.predict_center(mods, te, 'family'); c_ft = pdur.predict_center(mods, te, 'feat0')
    info = dict(pool=pid, environments='|'.join(pick), n_units=len(te), **{'n_' + f: int((te.family == f).sum()) for f in pdur.FAMS},
                mae_family=float(np.mean(np.abs(y - c_fam))), mae_prefix=float(np.mean(np.abs(y - c_ft))),
                bias_family=float(np.mean(y - c_fam)),
                # variance components are reported to six decimals: a component estimated at the zero boundary is numerically arbitrary below that
                tau_g=round(fz['tau_g'], 6), tau_f=round(fz['tau_f'], 6), sigma=round(fz['sigma'], 6))
    split = 'cv%03d' % pid
    es = set(pick)
    for fam in pdur.FAMS:                                              # held-out pool in the row order of the source index
        scen.pools[(split, fam)] = [r for r in scen.index if r['eligible'] == '1' and r['split'] == 'train' and r['family_id'] == fam
                                    and r['environment_id'] in es]
    R = registry(scen, fz)
    rows = []
    for pname in POLICIES:
        pol = R[pname]()
        for case in CASES:
            for tno, seed in enumerate(seeds):
                env = Trial(scen, store, case, seed, split, 1.0, 0.5, trial_no=tno, log=False)
                pol.run(env, np.random.default_rng(POLICY_SEED))
                m = env.metrics()
                rows.append(dict(pool=pid, policy=pname, case_id=case, seed=seed, utility=utility(m, env.H, env.max_tardiness(), RHO),
                                 on_time=m['weighted_on_time_rate'], completion=m['weighted_completion_rate'], invalid=m['invalid_actions']))
    return info, rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--procs', type=int, default=2); a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(20261126)
    scen = Scenario()
    seg = pd.read_csv(os.path.join(RESULTS, 'p3_segment_features.csv'))
    tr = seg[seg.split == 'train'].reset_index(drop=True)
    envs = np.array(sorted(tr.env.unique()))
    benv = by_environment(seg)
    print('duration error by held-out environment written (%d environments, %.0fs)' % (len(benv), time.time() - t0), flush=True)
    # emulated pools: random sets of four training environments that contain every job family
    pools, seen = [], set()
    while len(pools) < N_POOLS:
        pick = tuple(sorted(rng.choice(envs, size=ENVS_PER_POOL, replace=False)))
        if pick in seen:
            continue
        seen.add(pick)
        te = tr[tr.env.isin(pick)]
        if all((te.family == f).sum() >= 2 for f in pdur.FAMS):
            pools.append(pick)
    seeds = scen.seeds[:N_SEEDS]
    jobs = [(pid, pick, seeds) for pid, pick in enumerate(pools)]
    rows, info = [], []
    with Pool(a.procs, initializer=_init) as pool:
        for k, (inf, r) in enumerate(pool.imap_unordered(_work, jobs)):
            info.append(inf); rows.extend(r)
            if (k + 1) % 10 == 0:
                print('  %d / %d emulated pools (%.0fs)' % (k + 1, len(jobs), time.time() - t0), flush=True)
    info.sort(key=lambda z: z['pool'])
    df = pd.DataFrame(rows).sort_values(['pool', 'policy', 'case_id', 'seed']).reset_index(drop=True)
    assert int(df.invalid.sum()) == 0
    U = df.pivot_table(index='pool', columns='policy', values='utility', aggfunc='mean')
    T = df.pivot_table(index='pool', columns='policy', values='on_time', aggfunc='mean')
    pl = pd.DataFrame(info).set_index('pool')
    for p in POLICIES:
        pl['U_' + p] = U[p]
    for p in ('EDD-median', 'RO-family-static', 'RO-feat-static'):
        pl['on_time_' + p] = T[p]
    pl.reset_index().to_csv(os.path.join(RESULTS, 'p3_pooltransfer_pools.csv'), index=False)

    # paired differences: pool level (mean, sd across pools) and the trial-level standard error inside one pool
    val = pd.read_csv(os.path.join(RESULTS, 'p3_trials_validation_stage1.csv'))
    val = val[(val.alpha == 1.0) & (val.beta == 0.5)]
    out = []
    for a_, b_ in PAIRS:
        d = (U[a_] - U[b_]).to_numpy()
        w = df[df.policy == a_].merge(df[df.policy == b_], on=['pool', 'case_id', 'seed'], suffixes=('_a', '_b'))
        w['d'] = w.utility_a - w.utility_b
        se_trial = float(np.sqrt(np.mean(w.groupby('pool').d.var(ddof=1) / w.groupby('pool').d.count())))
        va = val[val.policy == a_].merge(val[val.policy == b_], on=['case_id', 'trial', 'seed'], suffixes=('_a', '_b'))
        dv = (va.utility_a - va.utility_b).to_numpy()
        out.append(dict(a=a_, b=b_, pools=len(d), mean_diff=float(d.mean()), pool_sd=float(d.std(ddof=1)), q10=float(np.quantile(d, 0.1)),
                        q90=float(np.quantile(d, 0.9)), share_positive=float((d > 0).mean()), trial_se_within_pool=se_trial,
                        validation_diff=float(dv.mean()), validation_trial_se=float(dv.std(ddof=1) / np.sqrt(len(dv))),
                        validation_z_pool=float(dv.mean() / d.std(ddof=1))))
    sm = pd.DataFrame(out)
    sm.to_csv(os.path.join(RESULTS, 'p3_pooltransfer_summary.csv'), index=False)

    # pool-level one-standard-error rule on the validation pool: simplest configuration whose validation gap to the best
    # configuration does not exceed the pool-level standard deviation of that difference
    order = ['RO-family-static', 'RO-family-learn', 'RO-feat-static', 'RO-feat-learn', 'RO-bma-static', 'RO-bma-learn']
    vm = {p: float(val[val.policy == p].utility.mean()) for p in order}
    best = max(order, key=lambda p: vm[p])
    rule, chosen = [], None
    for p in order:
        sd = float((U[best] - U[p]).std(ddof=1)) if p != best else 0.0
        gap = vm[best] - vm[p]
        rule.append(dict(policy=p, validation_mean=vm[p], gap_to_best=gap, pool_sd_of_gap=sd, within_one_pool_se=bool(gap <= sd + 1e-12)))
        if chosen is None and gap <= sd + 1e-12:
            chosen = p
    save_json(dict(n_pools=len(pools), envs_per_pool=ENVS_PER_POOL, trials_per_pool=3 * N_SEEDS, seeds=seeds, best_on_validation=best,
                   selected_pool_level_rule=chosen, rule=rule,
                   mean_objective={p: float(U[p].mean()) for p in POLICIES}, sd_objective={p: float(U[p].std(ddof=1)) for p in POLICIES},
                   duration_mae_family=float(pl.mae_family.mean()), duration_mae_prefix=float(pl.mae_prefix.mean()),
                   share_pools_prefix_predicts_better=float((pl.mae_prefix < pl.mae_family).mean())), 'p3_pooltransfer.json')
    # final decision rule: configuration selected by the pool-level rule, no paid review (value of information below its price)
    from p3_run import BASE_PAR
    save_json(dict(selected_candidate=chosen, mode=chosen.split('-')[1], learn=chosen.endswith('learn'), rollout=True, rho=RHO, q_min=0.5,
                   base_rule=BASE_PAR, policy_seed=POLICY_SEED, review_rule='none (value of information below its price)',
                   duration_model='results/frozen_p3_duration.json',
                   selection='one-standard-error rule on the validation pool; the standard error is the pool-to-pool standard deviation of '
                             'the paired difference, measured on %d emulated pools of training environments' % len(pools),
                   inputs_of_the_rule='validation trials of the six belief configurations and the emulated pools of training environments',
                   replay='python code/p3_run.py custom --split <split> --seeds <s1,s2,...> --policies FROZEN --tag <tag>'),
              'frozen_p3_policy.json')
    pd.set_option('display.width', 250)
    print(sm.round(4).to_string())
    print(pd.DataFrame(rule).round(4).to_string())
    print('best on validation: %s ; pool-level one-SE rule selects: %s ; done in %.0fs' % (best, chosen, time.time() - t0))


if __name__ == '__main__':
    main()
