# -*- coding: utf-8 -*-
"""Problem 3 - value-of-information analysis, schedules of the final policy and summary tables for the report.

Outputs (results/): p3_voi.json, p3_table_validation_nominal.csv, p3_table_mappings.csv, p3_table_test_nominal.csv,
p3_frontier.csv, p3_service_floor_check.csv, p3_compliance.json,
p3_schedule_by_job.csv, p3_schedule_by_case.csv, p3_gantt_trials.json (direct answers for the cases S1-S3),
p3_decision_trace.json (candidate values of every decision epoch of one replayed trial per case)
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LOGS, RESULTS, load_json, save_json
from p3_core import PrefixStore, Scenario, Trial, utility
from p3_policy import Lookahead
from p3_run import BASE_PAR, CASES, METRICS, POLICY_SEED, RHO


def frozen_policy(scen, fz, fzp, **kw):
    a = dict(mode=fzp['mode'], learn=fzp['learn'], rollout=True, rho=fzp['rho'], q_min=fzp['q_min'])
    a.update(kw)
    return Lookahead(scen, fz, BASE_PAR, **a)


def read_log(tag, policy='FROZEN'):
    rows = []
    with gzip.open(os.path.join(LOGS, 'p3_process_log_%s_%s.jsonl.gz' % (policy, tag)), 'rt', encoding='utf-8') as fh:
        for line in fh:
            e = json.loads(line)
            if e['alpha'] == 1.0 and e['beta'] == 0.5:
                rows.append(e)
    return rows


def schedule_answers(scen, val, tst):
    """What the final policy actually does in the cases S1-S3 (nominal mapping): order, reservations, waiting, stopping."""
    by_job, by_case, gantt = [], [], {}
    for pool, tag, df in (('validation', 'validation_stage2', val), ('test', 'test', tst)):
        log = read_log(tag)
        nomdf = df[(df.policy == 'FROZEN') & (df.alpha == 1.0) & (df.beta == 0.5)]
        for case in CASES:
            jobs = scen.jobs[case]
            ev = [e for e in log if e['case_id'] == case]
            trials = sorted(set(e['trial'] for e in ev))
            n = len(trials)
            seqs, waits, stops, nexec, quant, occ, trav = [], [], [], [], [], [], []
            per = {j['job_id']: dict(rank=[], start=[], res=[], on_time=0, done=0, over=[], idle=[]) for j in jobs}
            for t in trials:
                et = sorted((e for e in ev if e['trial'] == t), key=lambda e: e['step'])
                ex = [e for e in et if e['action'] == 'execute']
                seqs.append(tuple(e['job_id'] for e in ex))
                waits.append(sum(e.get('wait_s', 0.0) for e in et if e['action'] == 'wait'))
                stops.append(et[-1]['t_after'])
                nexec.append(len(ex))
                occ.append(sum(e['t_after'] - e['service_start'] for e in ex))        # service or reservation, max(p, b)
                trav.append(sum(e['service_start'] - e['t_decision'] for e in ex))   # travel and changeover
                for r, e in enumerate(ex):
                    p = per[e['job_id']]
                    due = next(j['due'] for j in jobs if j['job_id'] == e['job_id'])
                    p['rank'].append(r + 1); p['start'].append(e['service_start']); p['res'].append(e['reserved_s'])
                    p['over'].append(e['overrun_s']); p['idle'].append(e['idle_s'])
                    quant.append(e['policy_note']['reserve_quantile'])
                    if not e['censored']:
                        p['done'] += 1
                        p['on_time'] += int(e['completion'] <= due + 1e-9)
            for j in jobs:
                p = per[j['job_id']]
                k = len(p['rank'])
                by_job.append(dict(pool=pool, case=case, job=j['job_id'], family=j['family'], release=j['release'], due=j['due'],
                                   weight=j['weight'], zone=j['zone'], predecessors=';'.join(j['preds']), trials=n, executed_share=k / n,
                                   completed_share=p['done'] / n, on_time_share=p['on_time'] / n,
                                   mean_rank=float(np.mean(p['rank'])) if k else np.nan,
                                   median_start_s=float(np.median(p['start'])) if k else np.nan,
                                   median_reserved_s=float(np.median(p['res'])) if k else np.nan,
                                   mean_overrun_s=float(np.mean(p['over'])) if k else np.nan,
                                   mean_idle_s=float(np.mean(p['idle'])) if k else np.nan))
            # consensus order: jobs executed in at least half of the trials, sorted by their mean position
            cons = sorted((b for b in by_job if b['pool'] == pool and b['case'] == case and b['executed_share'] >= 0.5),
                          key=lambda b: b['mean_rank'])
            dropped = [b for b in by_job if b['pool'] == pool and b['case'] == case and b['executed_share'] < 0.5]
            vals, cnt = np.unique(np.array([' '.join(q) for q in seqs]), return_counts=True)
            first = {}
            for q in seqs:
                first[q[0]] = first.get(q[0], 0) + 1
            res_fam = {f: float(np.median([e['reserved_s'] for e in ev if e['action'] == 'execute'
                                           and next(j['family'] for j in jobs if j['job_id'] == e['job_id']) == f]))
                       for f in ('P', 'A', 'L')}
            qs, qc = np.unique(np.array(quant), return_counts=True)
            by_case.append(dict(pool=pool, case=case, trials=n, jobs=len(jobs), mean_executed=float(np.mean(nexec)),
                                consensus_order=' '.join(b['job'].split('_')[1] for b in cons),
                                rarely_executed=' '.join('%s(%.0f%%)' % (b['job'].split('_')[1], 100 * b['executed_share']) for b in dropped),
                                distinct_sequences=int(len(vals)), modal_sequence_share=float(cnt.max() / n),
                                first_job=max(first, key=first.get).split('_')[1], first_job_share=max(first.values()) / n,
                                mean_wait_s=float(np.mean(waits)), mean_stop_time_s=float(np.mean(stops)),
                                mean_occupied_s=float(np.mean(occ)), mean_travel_s=float(np.mean(trav)),
                                occupancy_share_of_H=float(np.mean(occ) / scen.cases[case]['horizon_s']),
                                median_reserved_P_s=res_fam['P'], median_reserved_A_s=res_fam['A'], median_reserved_L_s=res_fam['L'],
                                reserve_quantile_shares=';'.join('%.2f:%.2f' % (a_, b_ / qc.sum()) for a_, b_ in zip(qs, qc))))
            # representative trial for the Gantt chart: objective closest to the case median (first such trial)
            u = nomdf[nomdf.case_id == case].sort_values('trial')
            k = int((u.utility - u.utility.median()).abs().to_numpy().argmin())
            gantt['%s_%s' % (pool, case)] = dict(trial=int(u.trial.iloc[k]), seed=int(u.seed.iloc[k]), utility=float(u.utility.iloc[k]),
                                                 median_utility=float(u.utility.median()))
    pd.DataFrame(by_job).to_csv(os.path.join(RESULTS, 'p3_schedule_by_job.csv'), index=False)
    bc = pd.DataFrame(by_case)
    bc.to_csv(os.path.join(RESULTS, 'p3_schedule_by_case.csv'), index=False)
    save_json(gantt, 'p3_gantt_trials.json')
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_colwidth', 90)
    print(bc.round(2).to_string())


def decision_trace(scen, store, fz, fzp):
    """Replays the trials shown in the schedule figure once more with the trace of the look-ahead policy switched on and
    stores, for every decision epoch, the rollout value of each candidate first action and of each reservation length.
    The decisions are identical to the logged ones (the trace only records what the policy computes anyway)."""
    gantt = load_json('p3_gantt_trials.json')
    out = {}
    for case in CASES:
        g = gantt['validation_%s' % case]
        env = Trial(scen, store, case, g['seed'], 'validation', 1.0, 0.5, trial_no=g['trial'], log=True)
        pol = frozen_policy(scen, fz, fzp)
        pol.trace = []
        pol.run(env, np.random.default_rng(POLICY_SEED))
        m = env.metrics()
        u = utility(m, env.H, env.max_tardiness(), RHO)
        assert abs(u - g['utility']) < 1e-9, (case, u, g['utility'])
        jobs = env.jobs
        epochs = []
        for tr in pol.trace:
            rel = [j['job_id'] for j in jobs if j['job_id'] in tr['pending'] and j['release'] <= tr['t'] + 1e-9]
            edd = min(rel, key=lambda q: jobs[env.idx[q]]['due']) if rel else None
            e = dict(t=float(tr['t']), zone=tr['zone'], cost=float(tr['cost']), chosen=tr['chosen'], job=tr['job'],
                     b=(None if tr['b'] is None else float(tr['b'])), values={k: float(v) for k, v in tr['values'].items()},
                     reserve={('%.4f' % k): float(v) for k, v in tr['reserve'].items()}, pending=tr['pending'], released=rel, edd_choice=edd)
            if tr['samples'] is not None:
                sm = np.asarray(tr['samples'], float)
                e['samples'] = [round(float(x), 3) for x in sm]
                e['realised_s'] = float(env._p[env.idx[tr['job']]])
            epochs.append(e)
        out[case] = dict(trial=g['trial'], seed=g['seed'], utility=float(u), H=env.H, B=env.B, alpha=1.0, beta=0.5,
                         zones={k: list(v) for k, v in scen.zones.items()}, initial_zone=scen.initial_zone,
                         jobs=[dict(job_id=j['job_id'], family=j['family'], release=j['release'], due=j['due'], weight=j['weight'], zone=j['zone'])
                               for j in jobs],
                         completion={j['job_id']: (None if env.completion[i] is None else float(env.completion[i])) for i, j in enumerate(jobs)},
                         epochs=epochs)
    save_json(out, 'p3_decision_trace.json')
    return out


def main():
    scen = Scenario(); store = PrefixStore()
    fz = load_json('frozen_p3_duration.json'); fzp = load_json('frozen_p3_policy.json')
    val = pd.read_csv(os.path.join(RESULTS, 'p3_trials_validation.csv'))
    tst = pd.read_csv(os.path.join(RESULTS, 'p3_trials_test.csv'))
    nom = val[(val.alpha == 1.0) & (val.beta == 0.5)]
    voi = {}
    # ---------------------------------------------------------------- 1. shadow prices of one second and one cost unit
    pol = frozen_policy(scen, fz, fzp)
    base = {}
    rows = []
    for dH, dB in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
        for case in CASES:
            for tno, seed in enumerate(scen.seeds):
                env = Trial(scen, store, case, seed, 'validation', 1.0, 0.5, trial_no=tno, log=False, dH=dH, dB=dB)
                pol.run(env, np.random.default_rng(POLICY_SEED))
                m = env.metrics()
                # the objective is always normalised with the unperturbed horizon so that values are comparable
                H0 = scen.cases[case]['horizon_s']
                wt0 = sum(j['weight'] * max(H0 - j['due'], 0.0) for j in scen.jobs[case])
                u = m['weighted_on_time_rate'] - (m['weighted_tardiness_s'] + sum(j['weight'] for i, j in enumerate(scen.jobs[case])
                                                                                    if env.status[i] != 'completed') * (H0 - env.H)) / wt0 \
                    - RHO * (m['observed_overrun_s'] + m['reserved_idle_s']) / H0
                rows.append(dict(dH=dH, dB=dB, case=case, trial=tno, u=u))
    sp = pd.DataFrame(rows)
    piv = sp.pivot_table(index=['case', 'trial'], columns=['dH', 'dB'], values='u')
    def stat(x):
        return dict(mean=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))))
    voi['shadow_price'] = {}
    for case in list(CASES) + ['ALL']:
        pv = piv if case == 'ALL' else piv.loc[case]
        voi['shadow_price'][case] = dict(one_second=stat(pv[(0, 0)] - pv[(-1, 0)]), one_cost_unit=stat(pv[(0, 0)] - pv[(0, -1)]),
                                         one_review_price=stat(pv[(0, 0)] - pv[(-1, -1)]))
    # ---------------------------------------------------------------- 2. value of perfect information per executed job
    def paired(a, b, df):
        m = df[df.policy == a].merge(df[df.policy == b], on=['case_id', 'trial', 'seed', 'alpha', 'beta'], suffixes=('_a', '_b'))
        return m
    m = paired('CLAIRVOYANT', 'FROZEN', nom)
    d = m.utility_a - m.utility_b
    voi['evpi'] = dict(total=stat(d), per_executed_job=float(d.mean() / m.executed_jobs_b.mean()),
                       by_case={c: stat(d[m.case_id == c]) for c in CASES}, executed_jobs_mean=float(m.executed_jobs_b.mean()))
    # ---------------------------------------------------------------- 3. what a review actually buys (duration model, held-out)
    sel = pd.read_csv(os.path.join(RESULTS, 'p3_duration_selection.csv')).set_index('kind')
    voi['prefix_levels'] = {k: dict(cv_sd_log=float(sel.loc[k, 'cv_sd_log']), val_pinball_s=float(sel.loc[k, 'val_pin_mean_s']),
                                    test_pinball_s=float(sel.loc[k, 'test_pin_mean_s']), val_mae_log=float(sel.loc[k, 'val_mae_log']),
                                    test_mae_log=float(sel.loc[k, 'test_mae_log'])) for k in ('family', 'feat0', 'feat1', 'feat2')}
    v0, v1, v2 = (sel.loc[k, 'cv_sd_log'] for k in ('feat0', 'feat1', 'feat2'))
    voi['variance_reduction_cv'] = dict(first_review=float(1 - (v1 / v0) ** 2), second_review=float(1 - (v2 / v1) ** 2))
    # expected benefit of a review under the model: planning-error reduction of one job, in objective units
    mad = float(nom[nom.policy == 'FROZEN'].eval('(observed_overrun_s + reserved_idle_s) / executed_jobs').mean())
    Hbar = float(np.mean([scen.cases[c]['horizon_s'] for c in CASES]))
    voi['model_based_benefit_per_review'] = float(RHO * mad * max(1 - v1 / v0, 0.0) / Hbar)
    voi['planning_error_per_job_s'] = mad
    # empirical effect of forced reviews; the review variants use the prefix belief (the only belief that can exploit a
    # longer prefix), so they are paired with the prefix-belief look-ahead that buys no review
    review_ref = 'RO-feat-static' if fzp['mode'] == 'family' else 'FROZEN'
    voi['review_reference_policy'] = review_ref
    for tag, df in (('validation', nom), ('test', tst[(tst.alpha == 1.0) & (tst.beta == 0.5)])):
        for pol_name in ('ABL-review-once', 'ABL-review-twice', 'ABL-review-idle'):
            if pol_name not in set(df.policy):
                continue
            m = paired(pol_name, review_ref, df)
            d = m.utility_a - m.utility_b
            voi['forced_%s_%s' % (pol_name, tag)] = dict(delta_utility=stat(d), reviews=float(m.observations_a.mean()),
                                                         per_review=float(d.mean() / m.observations_a.mean()),
                                                         delta_on_time=float((m.weighted_on_time_rate_a - m.weighted_on_time_rate_b).mean()),
                                                         delta_overrun_idle_s=float((m.observed_overrun_s_a + m.reserved_idle_s_a
                                                                                    - m.observed_overrun_s_b - m.reserved_idle_s_b).mean()))
    save_json(voi, 'p3_voi.json')

    # ---------------------------------------------------------------- report tables
    def table(df, pols, by_case=True):
        d = df[df.policy.isin(pols)]
        keys = ['policy', 'case_id'] if by_case else ['policy']
        g = d.groupby(keys)
        t = g[METRICS].mean()
        t['utility_se'] = g['utility'].std(ddof=1) / np.sqrt(g.size())
        t['on_time_se'] = g['weighted_on_time_rate'].std(ddof=1) / np.sqrt(g.size())
        t['n'] = g.size()
        return t.reset_index()
    main_pols = ['FIFO-mean', 'EDD-median', 'EDD-min', 'Index-static', 'RO-family-static', 'RO-family-learn', 'RO-feat-static',
                 'RO-feat-learn', 'RO-bma-static', 'RO-bma-learn', 'FROZEN', 'ABL-review-once', 'ABL-review-twice', 'ABL-review-idle',
                 'RES-min', 'RES-q25', 'RES-q75', 'RES-q90', 'RHO-0.1', 'RHO-1.0', 'CLAIRVOYANT']
    t1 = pd.concat([table(nom, main_pols), table(nom, main_pols, False).assign(case_id='ALL')], ignore_index=True)
    t1.to_csv(os.path.join(RESULTS, 'p3_table_validation_nominal.csv'), index=False)
    tn = tst[(tst.alpha == 1.0) & (tst.beta == 0.5)]
    t2 = pd.concat([table(tn, list(tn.policy.unique())), table(tn, list(tn.policy.unique()), False).assign(case_id='ALL')], ignore_index=True)
    t2.to_csv(os.path.join(RESULTS, 'p3_table_test_nominal.csv'), index=False)
    rows = []
    for tag, df in (('validation', val), ('test', tst)):
        g = df.groupby(['policy', 'alpha', 'beta', 'case_id'])
        t = g[METRICS].mean(); t['utility_se'] = g['utility'].std(ddof=1) / np.sqrt(g.size()); t['n'] = g.size()
        t = t.reset_index(); t['pool'] = tag
        rows.append(t)
        g = df.groupby(['policy', 'alpha', 'beta'])
        t = g[METRICS].mean(); t['utility_se'] = g['utility'].std(ddof=1) / np.sqrt(g.size()); t['n'] = g.size()
        t = t.reset_index(); t['pool'] = tag; t['case_id'] = 'ALL'
        rows.append(t)
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(RESULTS, 'p3_table_mappings.csv'), index=False)
    fr = table(nom, ['RES-min', 'RES-q25', 'FROZEN', 'RES-q75', 'RES-q90'], False)
    fr['reserve_floor'] = fr.policy.map({'RES-min': 0.0, 'RES-q25': 0.25, 'FROZEN': 0.5, 'RES-q75': 0.75, 'RES-q90': 0.9})
    fr.sort_values('reserve_floor').to_csv(os.path.join(RESULTS, 'p3_frontier.csv'), index=False)
    # service floor of the problem statement: mean weighted on-time rate >= 0.50 per case under the nominal mapping
    chk = nom.groupby(['policy', 'case_id']).weighted_on_time_rate.mean().unstack()
    chk['meets_0.50_all_cases'] = (chk[list(CASES)] >= 0.5).all(axis=1)
    chk.reset_index().to_csv(os.path.join(RESULTS, 'p3_service_floor_check.csv'), index=False)
    # ---------------------------------------------------------------- schedules of the final policy (direct answers for S1-S3)
    schedule_answers(scen, val, tst)
    decision_trace(scen, store, fz, fzp)
    # protocol compliance of every trial of every policy
    comp = dict(validation_trials=int(len(val)), validation_invalid=int(val.invalid_actions.sum()), test_trials=int(len(tst)),
                test_invalid=int(tst.invalid_actions.sum()), validation_policies=int(val.policy.nunique()))
    save_json(comp, 'p3_compliance.json')
    print(json.dumps(voi, indent=1)); print(comp)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(chk.round(3).to_string())
    print(fr.sort_values('reserve_floor')[['policy', 'reserve_floor', 'utility', 'weighted_on_time_rate', 'weighted_completion_rate', 'cost',
                                          'observed_overrun_s', 'reserved_idle_s']].round(3).to_string())


if __name__ == '__main__':
    main()
