# -*- coding: utf-8 -*-
"""Write the LaTeX tables of the report directly from the result files (results/*.csv|json -> paper/tables/*.tex),
so that every number printed in a table is traceable to a result file and cannot drift from it."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PROJ, RESULTS

OUT = os.path.join(PROJ, 'paper', 'tables') if os.path.isdir(os.path.join(PROJ, 'paper')) else os.path.join(PROJ, 'tables')
os.makedirs(OUT, exist_ok=True)
R = lambda f: os.path.join(RESULTS, f)


def w(name, lines):
    with open(os.path.join(OUT, name), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('wrote', name)


def f(x, nd=1):
    return ('%.' + str(nd) + 'f') % x


def thousands(x):
    return '{:,}'.format(int(round(x))).replace(',', '\\,')


# ------------------------------------------------------------------------------------------- data retention
ret = pd.read_csv(R('p1_retention.csv'))
rows = []
for ds in ('EgoDemo', 'EgoPro', 'EgoStandard'):
    d = ret[ret.dataset == ds]
    rows.append((ds, d.environments.sum(), d.records.sum(), d.frames.sum(), d.hand_frames.sum(), d.ref.sum(), d.flagged.sum(), d.suspect.sum(),
                 d.guard.sum(), d.runs_ge1s.sum(), d.run_frames.sum()))
tot = ret[ret.dataset == 'ALL']
L = ['\\begin{ctab}{lrrrrrrrrr}', '\\toprule',
     ' & records & frames & hand-frames & reference & flagged & suspect & guard & runs $\\ge$1 s & frames in runs\\\\', '\\midrule']
for (ds, ne, nr, nf, hf, rf, fl, su, gu, ru, rfm) in rows:
    L.append('%s & %d & %s & %s & %s (%s\\%%) & %d & %d & %d & %d & %s\\\\' % (ds, nr, thousands(nf), thousands(hf), thousands(rf), f(100 * rf / hf, 1),
                                                                             fl, su, gu, ru, thousands(rfm)))
L.append('\\midrule')
for sp, nm in (('train', 'Training (16 env.)'), ('validation', 'Validation (4 env.)'), ('test', 'Test (4 env.)')):
    d = tot[tot.split == sp].iloc[0]
    L.append('%s & %d & %s & %s & %s (%s\\%%) & %d & %d & %d & %d & %s\\\\' % (nm, d.records, thousands(d.frames), thousands(d.hand_frames), thousands(d.ref),
                                                                             f(100 * d.ref / d.hand_frames, 1), d.flagged, d.suspect, d.guard, d.runs_ge1s,
                                                                             thousands(d.run_frames)))
L.append('\\midrule')
L.append('All (24 env.) & %d & %s & %s & %s (%s\\%%) & %d & %d & %d & %d & %s\\\\' % (
    tot.records.sum(), thousands(tot.frames.sum()), thousands(tot.hand_frames.sum()), thousands(tot.ref.sum()),
    f(100 * tot.ref.sum() / tot.hand_frames.sum(), 1), tot.flagged.sum(), tot.suspect.sum(), tot.guard.sum(), tot.runs_ge1s.sum(), thousands(tot.run_frames.sum())))
L[-1] = L[-1].replace('& 1000 &', '& 1\\,000 &')
L += ['\\bottomrule', '\\end{ctab}']
w('tab_retention.tex', L)

# ------------------------------------------------------------------------------------------- state model comparison
sel = pd.read_csv(R('p1_model_selection.csv')).set_index('model')
met = pd.read_csv(R('p1_state_metrics.csv'))
te = met[met.split == 'test']
names = [('Hold', 'Hold-last (baseline)'), ('CV', 'Constant velocity (baseline)'), ('DV', 'Damped velocity'), ('MR', 'Mean-reverting, 1 phase'),
         ('MR-2ph', 'Mean-reverting, 2 phases'), ('MR-3ph', 'Mean-reverting, 3 phases'), ('MR-4ph', '\\textbf{Mean-reverting, 4 phases}')]
hz = [1 / 30, 0.1, 0.5, 1.0, 3.0]
L = ['\\begin{ctab}{lcc|ccccc|c}', '\\toprule',
     ' & & validation & \\multicolumn{5}{c|}{test: mean error (mm) at horizon} & test cov.\\\\',
     'model & par. & NLL & 0.03 s & 0.1 s & 0.5 s & 1 s & 3 s & 90\\,\\% (1 s)\\\\', '\\midrule']
for key, nm in names:
    r = sel.loc[key]
    d = te[te.model == key]
    if len(d):
        vals = [f(float(d[np.isclose(d.horizon_s, h, atol=1e-3)].mean_mm.iloc[0]), 1 if h > 0.05 else 2) for h in hz]
        cov = f(float(d[np.isclose(d.horizon_s, 1.0)].cov90.iloc[0]), 3)
    else:
        vals = ['--'] * 5; cov = '--'
    L.append('%s & %d & $%s$ & %s & %s\\\\' % (nm, r.n_par, f(r.val_nll, 3), ' & '.join(vals), cov))
L += ['\\bottomrule', '\\end{ctab}']
w('tab_state.tex', L)

# ------------------------------------------------------------------------------------------- parameters
par = pd.read_csv(R('p1_state_params.csv')).set_index('parameter')
lab = {'kappa': ('$\\kappa$', 's$^{-2}$', 2), 'theta': ('$\\theta$', 's$^{-1}$', 2), 'sigma_1': ('$\\sigma_1$', 'm\\,s$^{-3/2}$', 3),
       'sigma_2': ('$\\sigma_2$', 'm\\,s$^{-3/2}$', 3), 'sigma_3': ('$\\sigma_3$', 'm\\,s$^{-3/2}$', 3), 'sigma_4': ('$\\sigma_4$', 'm\\,s$^{-3/2}$', 3),
       'sigma_m': ('$\\sigma_m$', 'm\\,s$^{-1/2}$', 4)}
cells = []
for k, (sym, unit, nd) in lab.items():
    r = par.loc[k]
    cells.append('%s (%s) & %s & [%s, %s]' % (sym, unit, f(r.estimate, nd), f(r.ci_low, nd), f(r.ci_high, nd)))
L = ['\\begin{ctab}{lcc|lcc}', '\\toprule', 'parameter & estimate & 95\\,\\% interval & parameter & estimate & 95\\,\\% interval\\\\', '\\midrule']
half = (len(cells) + 1) // 2
for i in range(half):
    right = cells[i + half] if i + half < len(cells) else ' & & '
    L.append('%s & %s\\\\' % (cells[i], right))
L += ['\\bottomrule', '\\end{ctab}']
w('tab_params.tex', L)

# ------------------------------------------------------------------------------------------- motion phases
fzs = json.load(open(R('frozen_p1_state.json')))
A = np.array(fzs['hmm']['A'])
spd = 10 ** np.array(fzs['hmm']['mu']) - 0.01                       # typical 0.1-s speed of each phase (m/s)
share = fzs['derived']['stationary_phase_probabilities']
ci = fzs['hmm_boot_ci']
fg = pd.read_csv(R('p1_factor_groups.csv'))
fam = {(r.level, int(r.quantity[-1])): r.estimate for r in fg[(fg.group == 'family') & fg.quantity.str.startswith('share of phase')].itertuples()}
reading = ['hold, fine placement, labelling', 'slow adjustment and arrangement', 'reach and transfer', 'fast transport']
L = ['\\begin{ctab}{c|cccc|ccc|l}', '\\toprule',
     ' & typical speed & share of & mean stay & forcing $\\sigma_k$ (m\\,s$^{-3/2}$) & \\multicolumn{3}{c|}{share of time, family} & reading of the phase\\\\',
     'phase $k$ & (m/s) [95\\,\\%] & time & (s) & [95\\,\\%] & P & A & L & (interpretation, not a label)\\\\', '\\midrule']
for k in range(4):
    r = par.loc['sigma_%d' % (k + 1)]
    L.append('%d & %s [%s, %s] & %s & %s & %s [%s, %s] & %s & %s & %s & %s\\\\' % (
        k + 1, f(spd[k], 3), f(ci['speed_lo'][k], 3), f(ci['speed_hi'][k], 3), f(share[k], 2), f((1 / 30.0) / (1 - A[k, k]), 2),
        f(r.estimate, 3), f(r.ci_low, 3), f(r.ci_high, 3), f(fam[('P', k + 1)], 2), f(fam[('A', k + 1)], 2), f(fam[('L', k + 1)], 2), reading[k]))
L += ['\\bottomrule', '\\end{ctab}']
w('tab_phases.tex', L)

# ------------------------------------------------------------------------------------------- reliability model
co = pd.read_csv(R('p1_quality_coef.csv'))
on = co[(co.part == 'onset') & (co.term != 'intercept')]
nice = {'edge': 'worst keypoint near image border', 'near': 'hand very close to camera', 'far': 'far reach', 'lspeed': 'speed (per decade)',
        'overlap_behind': 'behind the other hand', 'other_flag': 'other hand flagged', 'recent': 'flag in the last second',
        'fam_A': 'family A (vs P)', 'fam_L': 'family L (vs P)'}
L = ['\\begin{ctab}{lccc|lccc}', '\\toprule', 'covariate at $t-1$ & OR & 95\\,\\% interval & $p$ & covariate at $t-1$ & OR & 95\\,\\% interval & $p$\\\\', '\\midrule']
cells = ['%s & %s & [%s, %s] & %s' % (nice[r.term], f(np.exp(r.coef), 2), f(np.exp(r.ci_low), 2), f(np.exp(r.ci_high), 1 if np.exp(r.ci_high) > 100 else 2),
                                      ('$<$0.001' if r.p < 0.001 else f(r.p, 3))) for r in on.itertuples()]
half = (len(cells) + 1) // 2
for i in range(half):
    right = cells[i + half] if i + half < len(cells) else ' & & & '
    L.append('%s & %s\\\\' % (cells[i], right))
L += ['\\bottomrule', '\\end{ctab}']
w('tab_quality.tex', L)

# ------------------------------------------------------------------------------------------- masking summary
ms = pd.read_csv(R('p2_masking_summary.csv'))
ms = ms[ms.split == 'test']
cov = pd.read_csv(R('p2_coverage.csv'))
wins = [0.2, 0.5, 1.0, 2.0, 3.0]
L = ['\\begin{ctab}{lccccc}', '\\toprule', ' & \\multicolumn{5}{c}{length of the missing window}\\\\',
     'predictor & 0.2 s & 0.5 s & 1 s & 2 s & 3 s\\\\', '\\midrule']
for key, nm in (('Hold', 'Hold-last'), ('CV', 'Constant velocity'), ('DV', 'Damped velocity'), ('MR', 'Mean-reverting, 1 phase'),
                ('MR-4ph', '\\textbf{Mean-reverting, 4 phases}')):
    d = ms[ms.method == key].set_index('window_s')
    L.append('%s & %s\\\\' % (nm, ' & '.join(f(d.loc[x, 'mean_over_window_mm'], 1) for x in wins)))
n_w = ms[ms.method == 'Hold'].set_index('window_s')
L.append('\\midrule')
L.append('windows evaluated & %s\\\\' % ' & '.join('%d' % n_w.loc[x, 'n_windows'] for x in wins))
ct = cov[(cov.split == 'test') & np.isclose(cov.nominal, 0.9)]
hmap = {0.2: 0.2, 0.5: 0.5, 1.0: 1.0, 2.0: 2.0, 3.0: 3.0}
row = lambda col, nd: ' & '.join(f(float(ct[np.isclose(ct.horizon_s, h)][col].iloc[0]), nd) for h in wins)
L.append('coverage at window end, Gaussian 90\\,\\%% & %s\\\\' % row('gauss_cov', 3))
L.append('coverage at window end, calibrated 90\\,\\%% & %s\\\\' % row('conf_cov', 3))
L.append('calibrated 90\\,\\%% radius at window end (mm) & %s\\\\' % row('conf_radius_mm', 0))
L += ['\\bottomrule', '\\end{ctab}']
w('tab_masking.tex', L)

# ------------------------------------------------------------------------------------------- admissible horizon and waiting
adm = pd.read_csv(R('p2_admissible.csv'), dtype={'phase': str})
wt = pd.read_csv(R('p2_wait_table.csv'))
L = ['\\begin{ctab}{c|cccc|cc|cccc}', '\\toprule',
     'tolerance & \\multicolumn{4}{c|}{$\\tau^\\star$ (s), model, phase at onset} & \\multicolumn{2}{c|}{$\\tau^\\star$ (s), all onsets} & '
     '\\multicolumn{4}{c}{wait after a return (s), $\\delta=0.1$ s}\\\\',
     '(mm) & 1 & 2 & 3 & 4 & model & hold-last & 1 & 2 & 3 & 4\\\\', '\\midrule']
for tol in (20.0, 30.0, 50.0, 80.0, 100.0, 150.0):
    a = adm[np.isclose(adm.tolerance_mm, tol)].set_index('phase')
    cell = lambda k: ('$\\ge$3' if bool(a.loc[k, 'censored_at_3s']) else f(a.loc[k, 'tau_star_s'], 2))
    ww = wt[np.isclose(wt.tolerance_mm, tol)].set_index('phase')
    waits = [f(ww.loc[k, 'wait_delta_0p1_s'], 2) if len(ww) else '--' for k in (1, 2, 3, 4)]
    L.append('%d & %s & %s & %s & %s\\\\' % (tol, ' & '.join(cell(str(k)) for k in (1, 2, 3, 4)), cell('all'), f(a.loc['all', 'tau_star_hold_s'], 2),
                                             ' & '.join(waits)))
L += ['\\bottomrule', '\\end{ctab}']
w('tab_admissible.tex', L)

# ------------------------------------------------------------------------------------------- re-observation policies (test)
fr = pd.read_csv(R('p2_policy_frontier.csv'))
bu = pd.read_csv(R('p2_policy_budget.csv'))
L = ['\\begin{ctab}{c|ccc|cc|c|ccc}', '\\toprule',
     ' & \\multicolumn{6}{c|}{return delay $\\delta = 0.1$ s} & \\multicolumn{3}{c}{$\\delta = 0.3$ s}\\\\',
     'requests & hold & const. & model & model: share & model: info. & exact DP $-$ periodic & hold & model & exact DP $-$ periodic\\\\',
     'per window & -last & vel. & & $>$50 mm & age (s) & (mm, 95\\,\\% interval) & -last & & (mm, 95\\,\\% interval)\\\\', '\\midrule']
for n in (0, 1, 2, 4, 8):
    cells = []
    for dl in (0.1, 0.3):
        d = fr[(fr.split == 'test') & np.isclose(fr.delay_s, dl) & np.isclose(fr.param, n) & fr.policy.isin(['none', 'periodic'])].set_index('predictor')
        b = bu[(bu.split == 'test') & np.isclose(bu.delay_s, dl) & (bu.budget == n) & (bu.method == 'DP (stage-aware) + Model')]
        dp = ('%s [%s, %s]' % (f(b.diff_vs_periodic_model_mm.iloc[0], 2), f(b.diff_lo.iloc[0], 2), f(b.diff_hi.iloc[0], 2))).replace('-', '$-$') if len(b) else '--'
        if dl == 0.1:
            cells += [f(d.loc['Hold', 'mean_err_mm'], 1), f(d.loc['CV', 'mean_err_mm'], 1), f(d.loc['Model', 'mean_err_mm'], 1),
                      f(d.loc['Model', 'exceed_50mm'], 2), f(d.loc['Model', 'mean_info_age_s'], 2), dp]
        else:
            cells += [f(d.loc['Hold', 'mean_err_mm'], 1), f(d.loc['Model', 'mean_err_mm'], 1), dp]
    L.append('%d & %s\\\\' % (n, ' & '.join(cells)))
L += ['\\bottomrule', '\\end{ctab}']
w('tab_policy.tex', L)

# ------------------------------------------------------------------------------------------- duration model
ds = pd.read_csv(R('p3_duration_selection.csv'))
nm = {'pooled constant': 'Pooled constant (baseline)', 'family constant': '\\textbf{Family constant (final policy)}', 'family + prefix 0.2 s': 'Family + 0.2-s prefix',
      'family + prefix 0.6 s': 'Family + 0.6-s prefix', 'family + prefix 1.0 s': 'Family + 1.0-s prefix'}
L = ['\\begin{ctab}{lc|c|ccc|cccc}', '\\toprule',
     ' & & train CV & \\multicolumn{3}{c|}{validation pool (35 units)} & \\multicolumn{4}{c}{test pool (62 units)}\\\\',
     'duration model & par. & MAE$_{\\log}$ & MAE$_{\\log}$ & bias$_{\\log}$ & pinball (s) & MAE$_{\\log}$ & bias$_{\\log}$ & pinball (s) & cov. 50/90\\,\\%\\\\',
     '\\midrule']
for r in ds.itertuples():
    L.append('%s & %d & %s & %s & %s & %s & %s & %s & %s & %s / %s\\\\' % (
        nm[r.model], r.n_par, f(r.cv_mae_log, 3), f(r.val_mae_log, 3), ('%+.3f' % r.val_bias_log).replace('-', '$-$').replace('+', '$+$'), f(r.val_pin_mean_s, 3),
        f(r.test_mae_log, 3), ('%+.3f' % r.test_bias_log).replace('-', '$-$').replace('+', '$+$'), f(r.test_pin_mean_s, 3), f(r.test_cov50, 2), f(r.test_cov90, 2)))
L += ['\\bottomrule', '\\end{ctab}']
w('tab_duration.tex', L)

# ------------------------------------------------------------------------------------------- scheduling results
COLS = [('utility', 3), ('weighted_completion_rate', 3), ('weighted_on_time_rate', 3), ('weighted_tardiness_s', 1), ('cost', 2), ('observations', 2),
        ('elapsed_s', 1), ('observed_overrun_s', 1), ('reserved_idle_s', 1), ('censored_units', 2), ('invalid_actions', 0)]
HEAD = 'policy & $U$ & compl. & on-time & tard. (s) & cost & reviews & elapsed (s) & overrun (s) & idle (s) & unfin. & infeas.\\\\'
# FROZEN is the final decision rule (look-ahead with the family belief); the identical candidate RO-family-static is not repeated
PN = {'FIFO-mean': 'FIFO, family-mean reservation', 'EDD-min': 'EDD, minimum reservation', 'EDD-median': 'EDD, family-median reservation',
      'Index-static': 'Priority index (no look-ahead)', 'FROZEN': '\\textbf{Look-ahead policy (final)}',
      'RO-family-learn': 'Look-ahead, family belief + learning', 'RO-feat-static': 'Look-ahead, prefix belief',
      'RO-feat-learn': 'Look-ahead, prefix belief + learning', 'RO-bma-static': 'Look-ahead, averaged belief',
      'RO-bma-learn': 'Look-ahead, averaged belief + learning',
      'ABL-review-once': 'Prefix belief + 1 paid review per job', 'ABL-review-twice': 'Prefix belief + 2 paid reviews per job',
      'ABL-review-idle': 'Prefix belief + reviews while waiting',
      'RES-min': 'Final, reservation floor: minimum', 'RES-q25': 'Final, reservation floor: 0.25', 'RES-q75': 'Final, reservation floor: 0.75',
      'RES-q90': 'Final, reservation floor: 0.90', 'RHO-0.1': 'Final, $\\rho = 0.1$', 'RHO-1.0': 'Final, $\\rho = 1.0$',
      'CLAIRVOYANT': 'Known durations (reference)'}
plain = lambda p: PN[p].replace('\\textbf{', '').replace('}', '') if p != 'FROZEN' else PN[p]


def agg(tab, p, case=None):
    d = tab[(tab.policy == p) & (tab.case_id == ('ALL' if case is None else case))]
    return [f(d[c].iloc[0], nd) for c, nd in COLS]


vn = pd.read_csv(R('p3_table_validation_nominal.csv'))
tn = pd.read_csv(R('p3_table_test_nominal.csv'))
pv = pd.read_csv(R('p3_paired_validation.csv')); pt = pd.read_csv(R('p3_paired_test.csv'))
sg = lambda x, nd=3: ('%+.*f' % (nd, x)).replace('-', '$-$').replace('+', '$+$')
main = ['FIFO-mean', 'EDD-min', 'EDD-median', 'Index-static', 'FROZEN', 'CLAIRVOYANT']
L = ['\\begin{ctab}{l|ccc|c|cc|c|cc|cc}', '\\toprule', HEAD, '\\midrule',
     '\\multicolumn{12}{l}{\\textit{Validation pool: 30 public seeds $\\times$ 3 cases, nominal mapping}}\\\\']
for p in main:
    L.append('%s & %s\\\\' % (PN[p], ' & '.join(agg(vn, p))))
L += ['\\midrule', '\\multicolumn{12}{l}{\\textit{Test pool: 30 seeds of our own $\\times$ 3 cases, nominal mapping}}\\\\']
for p in main:
    L.append('%s & %s\\\\' % (PN[p], ' & '.join(agg(tn, p))))
L += ['\\bottomrule', '\\end{ctab}']
w('tab_p3_main.tex', L)

# final policy by case (direct answer for S1-S3): every common metric, executed jobs, paired difference to EDD
from common import TASK_DIR  # same task-folder resolution as the pipeline
cc = json.load(open(os.path.join(TASK_DIR, 'common_conditions.json'), encoding='utf-8'))
CASEINFO = {c['case_id']: c for c in cc['cases']}
sch = pd.read_csv(R('p3_schedule_by_case.csv'))
L = ['\\begin{ctab}{ll|c|ccc|c|c|c|cc|c|c}', '\\toprule',
     'pool & case & executed & $U$ & compl. & on-time & tard. (s) & cost / budget & elapsed (s) & overrun (s) & idle (s) & unfin. '
     '& $\\Delta U$ to EDD\\\\', '\\midrule']
for pool, tab, pr in (('validation', vn, pv), ('test', tn, pt)):
    for case in ('S1', 'S2', 'S3'):
        d = tab[(tab.policy == 'FROZEN') & (tab.case_id == case)].iloc[0]
        q = pr[(pr.a == 'FROZEN') & (pr.b == 'EDD-median') & (pr.metric == 'utility') & (pr.case == case)].iloc[0]
        ne = sch[(sch.pool == pool) & (sch.case == case)].mean_executed.iloc[0]
        assert d.observations == 0 and d.invalid_actions == 0          # stated in the caption of the table
        L.append('%s & %s & %s of %d & %s & %s & %s & %s & %s / %d & %s & %s & %s & %s & %s\\,(%s)\\\\' % (
            pool if case == 'S1' else '', case, f(ne, 1), CASEINFO[case]['jobs'], f(d.utility, 3), f(d.weighted_completion_rate, 3),
            f(d.weighted_on_time_rate, 3), f(d.weighted_tardiness_s, 1), f(d.cost, 1), CASEINFO[case]['cost_budget'],
            f(d.elapsed_s, 1), f(d.observed_overrun_s, 1), f(d.reserved_idle_s, 1), f(d.censored_units, 2), sg(q['diff']), f(q.se, 3)))
    if pool == 'validation':
        L.append('\\midrule')
L += ['\\bottomrule', '\\end{ctab}']
w('tab_p3_cases.tex', L)

# typical schedule of the final policy in each case (validation pool, nominal mapping)
num = lambda txt: ', '.join(str(int(t[1:])) for t in txt.split())
L = ['\\begin{ctab}{l|p{5.6cm}|l|c|c|c}', '\\toprule',
     'case & typical order of execution (job numbers) & seldom executed & median reservation (s) & equipment time (s): service and & stop\\\\',
     ' & & (share of trials) & P / A / L & reservation / travel / waiting & time (s)\\\\', '\\midrule']
for case in ('S1', 'S2', 'S3'):
    d = sch[(sch.pool == 'validation') & (sch.case == case)].iloc[0]
    rare = '--' if not isinstance(d.rarely_executed, str) else ', '.join('J%s' % t[1:].replace('(', ' (').replace('%', '\\,\\%') for t in d.rarely_executed.split())
    L.append('%s & %s & %s & %s / %s / %s & %s / %s / %s & %s\\\\' % (case, num(d.consensus_order), rare, f(d.median_reserved_P_s, 1), f(d.median_reserved_A_s, 1),
                                                                    f(d.median_reserved_L_s, 1), f(d.mean_occupied_s, 1), f(d.mean_travel_s, 1), f(d.mean_wait_s, 1),
                                                                    f(d.mean_stop_time_s, 1)))
L += ['\\bottomrule', '\\end{ctab}']
w('tab_p3_schedule.tex', L)

# per-case tables for the appendix (every policy, every common metric)
for tab, tag in ((vn, 'val'), (tn, 'test')):
    L = ['\\begin{ctab}{ll|ccc|c|cc|c|cc|cc}', '\\toprule', 'case & ' + HEAD, '\\midrule']
    pols = [p for p in PN if p in set(tab.policy)]
    for case in ('S1', 'S2', 'S3'):
        for p in pols:
            L.append('%s & %s & %s\\\\' % (case, plain(p), ' & '.join(agg(tab, p, case))))
        if case != 'S3':
            L.append('\\midrule')
    L += ['\\bottomrule', '\\end{ctab}']
    w('tab_p3_cases_%s.tex' % tag, L)

# paired differences (appendix): final policy minus comparator
L = ['\\begin{ctab}{l|ccc|c}', '\\toprule', 'comparator & S1 & S2 & S3 & all cases\\\\']


def pd_cell(tab, b, case):
    d = tab[(tab.a == 'FROZEN') & (tab.b == b) & (tab.metric == 'utility') & (tab.case == case)]
    if len(d) == 0:
        return '--'
    return sg(d['diff'].iloc[0]) + '\\,(' + f(d.se.iloc[0], 3) + ')'


for tab, cap in ((pv, 'validation pool'), (pt, 'test pool')):
    L += ['\\midrule', '\\multicolumn{5}{l}{\\emph{%s: $\\Delta U$ = final policy $-$ comparator}}\\\\' % cap]
    for b in ('FIFO-mean', 'EDD-min', 'EDD-median', 'Index-static', 'RO-family-learn', 'RO-feat-static', 'RO-bma-static', 'RES-min', 'RES-q25', 'RES-q75',
              'RHO-0.1', 'RHO-1.0', 'CLAIRVOYANT'):
        cells = [pd_cell(tab, b, c) for c in ('S1', 'S2', 'S3', 'ALL')]
        if all(c == '--' for c in cells):
            continue                                   # comparator not replayed on this pool
        L.append('%s & %s\\\\' % (plain(b), ' & '.join(cells)))
L += ['\\bottomrule', '\\end{ctab}']
w('tab_p3_paired.tex', L)

# belief configurations: validation pool, emulated pools (training environments only) and test pool
ps_ = pd.read_csv(R('p3_pooltransfer_summary.csv')); pl_ = pd.read_csv(R('p3_pooltransfer_pools.csv'))
tv_ = pd.read_csv(R('p3_trials_validation.csv')); tt_ = pd.read_csv(R('p3_trials_test.csv'))
tv_ = tv_[(tv_.alpha == 1.0) & (tv_.beta == 0.5)]; tt_ = tt_[(tt_.alpha == 1.0) & (tt_.beta == 0.5)]
ptj = json.load(open(R('p3_pooltransfer.json')))
rule = {r['policy']: r for r in ptj['rule']}
L = ['\\begin{ctab}{llc|ccc|ccc|c}', '\\toprule',
     ' & & & \\multicolumn{3}{c|}{validation pool} & \\multicolumn{3}{c|}{60 emulated pools: $\\Delta U$ to first row} & test\\\\',
     'belief & learning & par. & $U$ & gap to best & pool-level sd & mean & sd & gain in & $U$\\\\', '\\midrule']
for p, nm, lr, npar in (('RO-family-static', '\\textbf{family (final)}', 'no', 3), ('RO-family-learn', 'family', 'yes', 3),
                        ('RO-feat-static', 'family + prefix', 'no', 15), ('RO-feat-learn', 'family + prefix', 'yes', 15),
                        ('RO-bma-static', 'average of both', 'no', 15), ('RO-bma-learn', 'average of both', 'yes', 15)):
    if p == 'RO-family-static':
        em = ['--', '--', '--']
    else:
        q = ps_[(ps_.a == p) & (ps_.b == 'RO-family-static')].iloc[0]
        em = [sg(q.mean_diff), f(q.pool_sd, 3), '%d\\,\\%%' % round(100 * q.share_positive)]
    r = rule[p]
    gap = '--' if r['gap_to_best'] == 0 else f(r['gap_to_best'], 3)
    sd = '--' if r['gap_to_best'] == 0 else f(r['pool_sd_of_gap'], 3)
    L.append('%s & %s & %d & %s & %s & %s & %s & %s\\\\' % (nm, lr, npar, f(tv_[tv_.policy == p].utility.mean(), 3), gap, sd, ' & '.join(em),
                                                         f(tt_[tt_.policy == p].utility.mean(), 3)))
L += ['\\bottomrule', '\\end{ctab}']
w('tab_p3_beliefs.tex', L)

# pool-level differences (appendix): validation pool, emulated pools, test pool
L = ['\\begin{ctab}{l|c|cccc|c}', '\\toprule',
     ' & validation & \\multicolumn{4}{c|}{60 emulated pools (training environments)} & test\\\\',
     'difference of $U$ & pool & mean & sd & 10--90\\,\\% range & gain in & pool\\\\', '\\midrule']
for a_, b_, nm in (('Index-static', 'EDD-median', 'priority index $-$ EDD'), ('RO-family-static', 'EDD-median', 'look-ahead (final) $-$ EDD'),
                   ('RO-family-static', 'Index-static', 'look-ahead (final) $-$ priority index'),
                   ('RO-feat-static', 'RO-family-static', 'prefix belief $-$ family belief'),
                   ('RO-family-learn', 'RO-family-static', 'learning $-$ no learning (family belief)'),
                   ('RO-bma-learn', 'RO-family-static', 'averaged belief with learning $-$ family belief')):
    q = ps_[(ps_.a == a_) & (ps_.b == b_)].iloc[0]
    dv = tv_[tv_.policy == a_].utility.mean() - tv_[tv_.policy == b_].utility.mean()
    dt = tt_[tt_.policy == a_].utility.mean() - tt_[tt_.policy == b_].utility.mean()
    L.append('%s & %s & %s & %s & %s to %s & %d\\,\\%% & %s\\\\' % (nm, sg(dv), sg(q.mean_diff), f(q.pool_sd, 3), sg(q.q10), sg(q.q90),
                                                              round(100 * q.share_positive), sg(dt)))
L += ['\\bottomrule', '\\end{ctab}']
w('tab_p3_pools.tex', L)

# mapping sensitivity (appendix)
mp = pd.read_csv(R('p3_table_mappings.csv'))
mp = mp[mp.case_id == 'ALL']
maps = [(0.75, 0.5), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (1.5, 0.5)]
L = ['\\begin{ctab}{ll|ccccc|ccccc}', '\\toprule',
     ' & & \\multicolumn{5}{c|}{custom objective $U$ at $(\\alpha,\\beta)$} & \\multicolumn{5}{c}{weighted on-time rate at $(\\alpha,\\beta)$}\\\\',
     'pool & policy & ' + ' & '.join('(%g, %g)' % m for m in maps) + ' & ' + ' & '.join('(%g, %g)' % m for m in maps) + '\\\\', '\\midrule']
for pool in ('validation', 'test'):
    for p in ('FIFO-mean', 'EDD-median', 'Index-static', 'FROZEN', 'CLAIRVOYANT'):
        d = mp[(mp.pool == pool) & (mp.policy == p)]
        if len(d) == 0:
            continue
        u = [f(d[np.isclose(d.alpha, a) & np.isclose(d.beta, b)].utility.iloc[0], 3) for a, b in maps]
        o = [f(d[np.isclose(d.alpha, a) & np.isclose(d.beta, b)].weighted_on_time_rate.iloc[0], 3) for a, b in maps]
        L.append('%s & %s & %s & %s\\\\' % (pool, plain(p), ' & '.join(u), ' & '.join(o)))
    if pool == 'validation':
        L.append('\\midrule')
L += ['\\bottomrule', '\\end{ctab}']
w('tab_p3_mappings.tex', L)

# ------------------------------------------------------------------------------------------- audit checklist (appendix)
au = json.load(open(R('p1_audit.json'), encoding='utf-8'))
t, c, du, cp, fl, th = au['time'], au['coordinates'], au['duplicates'], au['completeness'], au['flags'], au['thresholds']
em = c['frac_any_keypoint_outside_image']
rows = [
    ('Non-increasing timestamps / gaps longer than $1.5\\Delta$', '%d / %d' % (t['non_increasing_timestamps'], t['gaps_gt_1p5dt'])),
    ('Largest within-record variation of the frame interval', '%.1f $\\mu$s' % t['within_record_dt_jitter_us_max']),
    ('Sampling rate', '%d records at 30 Hz; %d training records at 25 Hz' % (t['records_30hz'], len(t['records_other_rate']))),
    ('Pose, camera and video-frame time stamps of the MCAP sessions', 'largest difference %g ns' % max(t['mcap_stream_max_abs_dt_ns'], t['video_vs_pose_max_abs_dt_ns'])),
    ('Semantic segments against the official index', '%d of %d identical; %d gaps between segments; unannotated tail $\\le$ %.2f s'
     % (t['segments'], t['segments'], t['segment_gaps'], t['unannotated_tail_s_max'])),
    ('Unit of length (stereo baseline of the head cameras)', '$%.2f \\pm %.2f$ mm' % (c['stereo_baseline_mm_mean'], c['stereo_baseline_mm_sd'])),
    ('World frame', 'right-handed, vertical axis up; head height from $%.2f$ to %.2f m (record-specific origin)' % tuple(c['head_height_range_m'])),
    ('Camera frame / head frame', '$x$ right, $y$ down, $z$ forward / $x$ forward, $y$ left, $z$ up'),
    ('Hands in front of the camera / left hand to the left of the right hand', '%.0f\\,\\%% / %.0f\\,\\%% of frames'
     % (100 * c['hands_in_front_of_camera_frac'], 100 * c['left_hand_left_of_right_frac'])),
    ('Variation of palm bone lengths within a record', 'coefficient of variation $<10^{-5}$'),
    ('Duplicated \\texttt{episode\\_uuid} / duplicated content hash', '%d / %d' % (du['duplicate_episode_uuid'], du['duplicate_content_hash'])),
    ('Environments / scenes / operators', '%d / %d / %d (at most %d records per environment)'
     % (du['environments'], du['scenes'], du['operators'], du['records_per_environment_max'])),
    ('Scenes shared across splits; operators shared train--test / train--validation', '%d; %d / %d'
     % (du['scenes_shared_across_splits'], du['operators_shared_train_test'], du['operators_shared_train_val'])),
    ('Missing values / all-zero hand frames / exactly repeated hand frames', '%d / %d / %d'
     % (cp['nan_values'], cp['all_zero_hand_frames'], cp['exact_repeated_hand_frames'])),
    ('Flagged hand-frames (occlusion / near edge)', '%s (%d / %d); %d runs in %d records'
     % (thousands(fl['flagged_hand_frames']), fl['hand_frames_occlusion'], fl['hand_frames_near_edge'], fl['flagged_runs'], fl['records_with_flags'])),
    ('Length of flagged runs: median / mean / maximum', '%.2f / %.2f / %.1f s' % (fl['run_median_s'], fl['run_mean_s'], fl['run_max_s'])),
    ('Share of frames with a keypoint outside the image: unflagged / occlusion / near edge', '%.2f\\,\\%% / %.1f\\,\\%% / %.0f\\,\\%%'
     % (100 * em['good'], 100 * em['occlusion'], 100 * em['near_edge'])),
    ('Plausibility thresholds (hand speed, acceleration, head speed); guard band; shortest run', '%.1f m/s, %.0f m/s$^2$, %.1f m/s; $\\pm$%d frames; %.0f s'
     % (th['V_MAX'], th['A_MAX'], th['VH_MAX'], th['GUARD'], th['MIN_RUN_S'])),
]
L = ['\\begin{ctab}{p{8.6cm}p{7.6cm}}', '\\toprule', 'check & result\\\\', '\\midrule'] + ['%s & %s\\\\' % r for r in rows] + ['\\bottomrule', '\\end{ctab}']
w('tab_audit.tex', L)

# reproduction report (appendix), if a comparison file is present
rp = os.path.join(RESULTS, 'repro_report.json')
if os.path.isfile(rp):
    rep = json.load(open(rp, encoding='utf-8'))
    s = rep['summary']
    bad = {k: v for k, v in rep['files'].items() if v.get('judged') and not v.get('identical_within_tol')}
    L = ['\\begin{ctab}{lr}', '\\toprule', 'quantity & value\\\\', '\\midrule',
         'result files compared (CSV / JSON) & %d\\\\' % s['files_compared'],
         'numeric cells compared & %s\\\\' % thousands(s['numeric_cells']),
         'files identical within relative tolerance $10^{-6}$ & %d\\\\' % s['files_within_tolerance'],
         'largest relative difference in the remaining files & $%.1f\\times10^{-5}$\\\\' % (s['max_rel_diff'] * 1e5),
         'text cells that differ & %d\\\\' % sum(v.get('text_diffs', 0) for v in rep['files'].values() if v.get('judged')),
         '\\bottomrule', '\\end{ctab}']
    w('tab_repro.tex', L)
