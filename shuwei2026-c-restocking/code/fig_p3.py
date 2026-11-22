# -*- coding: utf-8 -*-
"""Schedules of the cases S1-S3 under the final policy (one replayed trial per case) and the pool-level check of the
candidate beliefs (appendix).  Reads results/p3_*.csv|json, the public task table and the process log of the final policy."""
# 1. imports, paths, style ---------------------------------------------------------------------------------
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LOGS, RESULTS, TASK_DIR, load_json
import plotstyle as ps
from plotstyle import C, W_FULL, label, plt, save
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ps.setup()
plt.rcParams.update({'axes.labelsize': 7.2, 'xtick.labelsize': 6.4, 'ytick.labelsize': 6.4})
R = lambda f: os.path.join(RESULTS, f)
FAM_COL = {'P': C['blue'], 'A': C['orange'], 'L': C['green']}

# 2. read data --------------------------------------------------------------------------------------------
tasks = pd.read_csv(os.path.join(TASK_DIR, 'tasks.csv'), encoding='utf-8-sig')
FINAL = 'FROZEN'                                   # final decision rule of results/frozen_p3_policy.json

# =============================================================================================== Figure: schedules of the three cases
# One replay per case of the final policy (validation pool, nominal mapping).  The trial is chosen by a rule written in
# p3_analysis.py: the trial whose objective is closest to the case median.
gt = load_json('p3_gantt_trials.json')
vtr = pd.read_csv(R('p3_trials_validation.csv'))
vtr = vtr[(vtr.policy == FINAL) & (vtr.alpha == 1.0) & (vtr.beta == 0.5)]
ev = []
with gzip.open(os.path.join(LOGS, 'p3_process_log_%s_validation_stage2.jsonl.gz' % FINAL), 'rt', encoding='utf-8') as fh:
    for line in fh:
        e = json.loads(line)
        if abs(e['alpha'] - 1.0) < 1e-9 and abs(e['beta'] - 0.5) < 1e-9 and e['trial'] == gt['validation_' + e['case_id']]['trial']:
            ev.append(e)
HZ = {'S1': 100.0, 'S2': 100.0, 'S3': 125.0}
fig, axs = plt.subplots(1, 3, figsize=(W_FULL, 2.55), gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.22]))
for ax, case in zip(axs, ('S1', 'S2', 'S3')):
    tk = tasks[tasks.case_id == case].reset_index(drop=True)
    ex = {e['job_id']: e for e in ev if e['case_id'] == case and e['action'] == 'execute'}
    order = {e['job_id']: k + 1 for k, e in enumerate(sorted(ex.values(), key=lambda e: e['step']))}
    stop = max(e['t_after'] for e in ev if e['case_id'] == case)
    for i, j in tk.iterrows():
        col = FAM_COL[j.family_id]
        ax.add_patch(Rectangle((j.release_s, i - 0.12), j.due_s - j.release_s, 0.24, color='#E4E4E4', lw=0, zorder=1))
        ax.plot([j.due_s, j.due_s], [i - 0.42, i + 0.42], color=C['black'], lw=1.0, zorder=5)
        if j.job_id not in ex:
            ax.text(j.release_s + 1.5, i, 'not executed', fontsize=4.8, va='center', ha='left', color=C['grey'], style='italic', zorder=6)
            continue
        e = ex[j.job_id]
        end = e['completion'] if not e['censored'] else HZ[case]
        late = e['censored'] or e['completion'] > j.due_s + 1e-9
        ax.add_patch(Rectangle((e['service_start'], i - 0.36), end - e['service_start'], 0.72, facecolor=col, alpha=0.9, lw=0, zorder=3,
                               hatch='////' if late else None, edgecolor='white'))
        ax.add_patch(Rectangle((e['service_start'], i - 0.36), e['reserved_s'], 0.72, fill=False, edgecolor=C['black'], lw=0.8, zorder=4))
        if e['service_start'] < 0.07 * HZ[case]:           # no room on the left of the first bars: number on the right
            ax.text(max(end, e['service_start'] + e['reserved_s']) + 1.2, i, '%d' % order[j.job_id], fontsize=5.2, va='center', ha='left', zorder=6)
        else:
            ax.text(e['service_start'] - 1.2, i, '%d' % order[j.job_id], fontsize=5.2, va='center', ha='right', zorder=6)
    ax.axvline(stop, color=C['grey'], lw=0.8, ls=':', zorder=2)
    m = vtr[(vtr.case_id == case) & (vtr.trial == gt['validation_' + case]['trial'])].iloc[0]
    ax.set_title('%s: %d jobs, $H$ = %d s\n$U$ = %.2f, on time %.2f, cost %.1f' % (case, len(tk), HZ[case], m.utility, m.weighted_on_time_rate, m.cost),
                 fontsize=6.6, pad=2)
    ax.set_ylim(len(tk) - 0.45, -0.55); ax.set_xlim(0, HZ[case]); ax.grid(axis='y', visible=False)
    ax.set_yticks(range(len(tk)))
    ax.set_yticklabels(['%s  %s  w%d' % (j.job_id.split('_')[1], j.family_id, j.priority_weight) for _, j in tk.iterrows()],
                       fontsize=5.6 if len(tk) <= 12 else 5.0)
    ax.set_xlabel('time (s)')
h = [Rectangle((0, 0), 1, 1, color=FAM_COL['P'], alpha=0.9, label='service, family P'),
     Rectangle((0, 0), 1, 1, color=FAM_COL['A'], alpha=0.9, label='family A'), Rectangle((0, 0), 1, 1, color=FAM_COL['L'], alpha=0.9, label='family L'),
     Rectangle((0, 0), 1, 1, facecolor=C['light'], hatch='////', edgecolor='white', label='completed late'),
     Rectangle((0, 0), 1, 1, fill=False, edgecolor=C['black'], lw=0.8, label='reserved window'),
     Rectangle((0, 0), 1, 1, color='#E4E4E4', label='release to due'),
     Line2D([], [], color=C['black'], lw=1.0, marker='|', ms=6, ls='', label='due time'),
     Line2D([], [], color=C['grey'], lw=0.8, ls=':', label='stop')]
fig.legend(handles=h, loc='outside lower center', ncol=8, fontsize=5.9, handlelength=1.2, handletextpad=0.3, columnspacing=0.9)
save(fig, 'fig10_schedules')

# =============================================================================================== Figure: pool-level transfer
pl = pd.read_csv(R('p3_pooltransfer_pools.csv'))
tv = pd.read_csv(R('p3_trials_validation.csv')); tt = pd.read_csv(R('p3_trials_test.csv'))
tv = tv[(tv.alpha == 1.0) & (tv.beta == 0.5)]; tt = tt[(tt.alpha == 1.0) & (tt.beta == 0.5)]


def pool_diff(tab, a, b):
    return float(tab[tab.policy == a].utility.mean() - tab[tab.policy == b].utility.mean())


PAIRS = [('Index-static', 'EDD-median', 'priority index $-$ EDD'), ('RO-family-static', 'EDD-median', 'look-ahead $-$ EDD'),
         ('RO-family-static', 'Index-static', 'look-ahead $-$ priority index'), ('RO-feat-static', 'RO-family-static', 'prefix belief $-$ family belief'),
         ('RO-family-learn', 'RO-family-static', 'learning $-$ no learning')]
fig, axs = plt.subplots(1, 2, figsize=(W_FULL, 2.12), gridspec_kw=dict(width_ratios=[1.9, 1]))
ax = axs[0]
rng = np.random.default_rng(3)
for i, (a, b, nm) in enumerate(PAIRS):
    yy = len(PAIRS) - 1 - i
    d = (pl['U_' + a] - pl['U_' + b]).to_numpy()
    ax.scatter(d, yy + rng.uniform(-0.22, 0.22, len(d)), s=5, color=C['light'], linewidths=0, zorder=2)
    ax.plot([d.mean() - d.std(ddof=1), d.mean() + d.std(ddof=1)], [yy, yy], '-', color=C['black'], lw=1.6, zorder=3)
    ax.plot([d.mean()], [yy], 'o', color=C['black'], ms=4.2, zorder=4)
    ax.plot([pool_diff(tv, a, b)], [yy], 's', color=C['orange'], ms=5, mec='white', mew=0.5, zorder=5)
    ax.plot([pool_diff(tt, a, b)], [yy], 'D', color=C['red'], ms=4.6, mec='white', mew=0.5, zorder=5)
    ax.text(0.47, yy, '%d %%' % round(100 * (d > 0).mean()), fontsize=6.6, ha='right', va='center')
ax.axvline(0, color=C['black'], lw=0.7, zorder=1)
ax.set_yticks(range(len(PAIRS))); ax.set_yticklabels([p[2] for p in PAIRS][::-1], fontsize=7)
ax.set_xlim(-0.24, 0.48); ax.set_ylim(-0.6, len(PAIRS) - 0.25); ax.grid(axis='y', visible=False)
ax.set_xlabel('difference of the objective $U$ on one pool')
ax.text(0.47, len(PAIRS) - 0.42, 'pools with a gain', fontsize=6.2, ha='right', va='center', color=C['grey'])
ax.legend(handles=[Line2D([], [], color=C['light'], marker='o', ls='', ms=3, label='emulated pool'),
                   Line2D([], [], color=C['black'], marker='o', ms=4.2, lw=1.6, label='mean $\\pm$ sd over pools'),
                   Line2D([], [], color=C['orange'], marker='s', ls='', ms=5, label='validation pool'),
                   Line2D([], [], color=C['red'], marker='D', ls='', ms=4.6, label='test pool')],
          loc='lower center', bbox_to_anchor=(0.36, 1.0), ncol=4, fontsize=6.4, handletextpad=0.25, columnspacing=0.9, borderaxespad=0.1)
label(ax, 'a', -0.56, 1.03)
# (b) duration error per held-out environment: family model against prefix model
ax = axs[1]
be = pd.read_csv(R('p3_duration_by_env.csv'))
ax.plot([0, 1.9], [0, 1.9], '-', color=C['black'], lw=0.7, zorder=1)
for sp, col, mk, nm in (('train (left out)', C['grey'], 'o', 'training, left out'), ('validation', C['orange'], 's', 'validation'), ('test', C['red'], 'D', 'test')):
    d = be[be.split == sp]
    ax.scatter(d.mae_family, d.mae_prefix, s=6 + 1.7 * d.n_units, color=col, marker=mk, alpha=0.85, edgecolor='white', linewidth=0.5, zorder=3, label=nm)
w = be[be.split == 'test'].assign(loss=lambda z: z.mae_prefix - z.mae_family).sort_values('loss').iloc[-1]
ax.annotate('%d of %d\ntest units' % (w.n_units, be[be.split == 'test'].n_units.sum()), xy=(w.mae_family, w.mae_prefix), xytext=(0.74, 1.30),
            fontsize=6.4, ha='center', va='center', arrowprops=dict(arrowstyle='-', color=C['red'], lw=0.6, shrinkA=1, shrinkB=3), color=C['red'])
ax.text(1.42, 0.50, 'prefix model\nbetter', fontsize=6.4, ha='center', va='center', color=C['grey'])
ax.text(0.42, 1.68, 'prefix model worse', fontsize=6.4, ha='center', va='center', color=C['grey'])
ax.set_xlim(0, 1.9); ax.set_ylim(0, 1.9); ax.set_aspect('equal', adjustable='box')
ax.set_xticks([0, 0.5, 1.0, 1.5]); ax.set_yticks([0, 0.5, 1.0, 1.5])
ax.set_xlabel('error of $\\log d$, family model'); ax.set_ylabel('error, family + prefix')
lg = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.4, handletextpad=0.1, columnspacing=0.7, borderaxespad=0.1, markerscale=0.8)
for hd in lg.legend_handles:
    hd.set_sizes([16])
label(ax, 'b', -0.36, 1.03)
save(fig, 'figA1_pooltransfer')
