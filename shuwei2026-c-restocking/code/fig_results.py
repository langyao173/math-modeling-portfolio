# -*- coding: utf-8 -*-
"""Result figures of the report: reliability (Problem 1), admissible horizon, request policy and natural occlusion
(Problem 2), duration model, one decision of the look-ahead policy and the comparison of policies (Problem 3), and the
sensitivity summary.  Reads results/*.csv|json (and the frame table for the flag timeline).

    python code/fig_results.py                 all figures
    python code/fig_results.py horizon policy  a subset (names: reliability horizon policy natural duration decision
                                               comparison sensitivity)
"""
# 1. imports, paths, style ---------------------------------------------------------------------------------
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, load_json
import plotstyle as ps
from plotstyle import C, PHASE_COL, P_COL, W_FULL, plt, save
import schem as S
from schem import INK, MUTE
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle, Wedge

ps.setup()
plt.rcParams.update({'axes.labelsize': 6.8, 'xtick.labelsize': 6.2, 'ytick.labelsize': 6.2})      # compact panels
R = lambda f: os.path.join(RESULTS, f)
GRID = '#E3E3E3'


def new_fig(h, w=W_FULL):
    fig = plt.figure(figsize=(w, h))
    fig.set_layout_engine('none')                    # every axes is placed by hand (inches)
    return fig


def axi(fig, x, y, w, h, **kw):
    """Axes placed in inches from the lower-left corner of the figure."""
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def tag(fig, x, y, s, title=None, fs=6.6):
    """Panel letter (and a short panel title) at (x, y) inches."""
    W, H = fig.get_size_inches()
    fig.text(x / W, y / H, '(%s)' % s, fontsize=8.5, fontweight='bold', va='center', ha='left', color='#111111')
    if title:
        fig.text((x + 0.25) / W, y / H, title, fontsize=fs, va='center', ha='left', color=INK)


def bare(ax, left=False, bottom=True):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_visible(left); ax.spines['bottom'].set_visible(bottom)


def tint(col, a):
    """Colour `col` mixed with white (a = 1 keeps the colour)."""
    from matplotlib.colors import to_rgb
    c = np.array(to_rgb(col))
    return tuple(1 - a * (1 - c))


# =============================================================================================== Figure: reliability (Problem 1)
def fig_reliability():
    from features import Data
    D = Data()
    runs = pd.read_csv(R('p2_natural_runs.csv'))
    nat = load_json('p2_natural_summary.json')
    coef = pd.read_csv(R('p1_quality_coef.csv'))
    cap = pd.read_csv(R('p1_quality_capture.csv'))
    met = pd.read_csv(R('p1_quality_metrics.csv'))
    prof = pd.read_csv(R('p1_stage_profile.csv'))
    grp = pd.read_csv(R('p1_factor_groups.csv'))
    c1, t1 = P_COL[1]
    RC = {'occlusion': C['red'], 'near edge': c1, 'mixed': C['purple']}
    fig = new_fig(3.42)

    # ---- (a) timeline of the flagged runs: one row per hand-record, the 16 hand-records with the most runs
    ax = axi(fig, 0.30, 2.22, 4.02, 0.86)
    cnt = runs.groupby(['rid', 'hand']).size().sort_values(ascending=False, kind='stable')
    rows = list(cnt.index[:16])
    tmax = 0.0
    for i, (rid, hand) in enumerate(rows):
        a, b = int(D.start[rid]), int(D.stop[rid]); T = float(D.t[b - 1] - D.t[a]); tmax = max(tmax, T)
        ax.plot([0, T], [i, i], color='#E4E6E9', lw=2.6, solid_capstyle='butt', zorder=1)
        d = runs[(runs.rid == rid) & (runs.hand == hand)]
        for _, r_ in d.iterrows():
            ax.plot([r_.t_on, r_.t_on], [i - 0.42, i + 0.42], color=RC[r_.reason], lw=0.55 + 2.4 * np.sqrt(r_.dur_s), solid_capstyle='butt', zorder=3)
    ax.set_ylim(len(rows) - 0.3, -0.7); ax.set_xlim(-0.5, tmax + 0.5)
    ax.set_yticks([]); ax.grid(False); bare(ax)
    ax.set_xlabel('record time (s)', labelpad=1.5)
    ax.text(-0.012, 0.5, '16 hand-records', transform=ax.transAxes, rotation=90, fontsize=6.2, ha='right', va='center', color=INK)
    h = [Line2D([], [], color=RC[k], lw=2.2, label=k) for k in ('occlusion', 'near edge', 'mixed')]
    ax.legend(handles=h, loc='lower right', bbox_to_anchor=(1.0, 0.98), ncol=3, fontsize=5.8, handlelength=0.9, columnspacing=0.9, handletextpad=0.4,
              borderaxespad=0.0)
    tag(fig, 0.06, 3.28, 'a', 'flagged runs come in bursts: %d runs, a new one within 1 s after %.0f %% of the recoveries'
        % (nat['episodes'], 100 * nat['reoccurrence']['p_new_onset_within_1s_after_recovery']))

    # ---- (b) capture curve of the onset model
    ax = axi(fig, 5.05, 2.22, 1.42, 0.86)
    for sp, col, ls, lb in (('validation', MUTE, (0, (3, 1.5)), 'validation'), ('test', c1, '-', 'test')):
        d = cap[cap.split == sp].sort_values('frame_share')
        ax.plot(100 * d.frame_share, 100 * d.onset_share, color=col, ls=ls, lw=1.5 if sp == 'test' else 1.0, label=lb, drawstyle='steps-post')
    ax.plot([0, 100], [0, 100], color='#9AA0A6', lw=0.7, ls=':')
    d = cap[cap.split == 'test']; r20 = d.iloc[(d.frame_share - 0.20).abs().argmin()]
    ax.plot([100 * r20.frame_share], [100 * r20.onset_share], 'o', color=c1, ms=3.6, mec='white', mew=0.6, zorder=5)
    ax.annotate('%.0f %% of the onsets\nin 20 %% of the frames' % (100 * r20.onset_share), xy=(100 * r20.frame_share, 100 * r20.onset_share),
                xytext=(26, 60), fontsize=5.6, color=c1, ha='left', va='center', arrowprops=dict(arrowstyle='-', color=c1, lw=0.6, shrinkA=1, shrinkB=2))
    auc = met[(met.split == 'test') & (met.model == 'onset (at-risk frames)')].auc.iloc[0]
    ax.text(58, 5, 'AUC %.2f' % auc, fontsize=6.2, color=c1, ha='right', va='bottom', fontweight='bold')
    ax.set_xlim(0, 60); ax.set_ylim(0, 102); ax.set_xticks([0, 20, 40, 60]); ax.set_yticks([0, 50, 100])
    ax.set_xlabel('riskiest frames watched (%)', labelpad=1.5); ax.set_ylabel('onsets found (%)', labelpad=1.5)
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.16), fontsize=5.6, handlelength=1.5, borderaxespad=0.2, labelspacing=0.2)
    tag(fig, 4.55, 3.28, 'b', 'the model ranks the risk')

    # ---- (c) odds ratios of the onset hazard as lollipops
    ax = axi(fig, 1.20, 0.40, 1.50, 1.22)
    on = coef[(coef.part == 'onset') & (coef.term != 'intercept')].copy()
    nice = {'edge': 'near image border', 'near': 'very close to camera', 'far': 'far reach', 'lspeed': 'speed, tenfold',
            'overlap_behind': 'behind other hand', 'other_flag': 'other hand flagged', 'recent': 'flag in the last second', 'fam_A': 'family A (vs P)',
            'fam_L': 'family L (vs P)'}
    on = on.sort_values('coef').reset_index(drop=True)
    for i, r_ in on.iterrows():
        orr, lo, hi = np.exp(r_.coef), np.exp(r_.ci_low), np.exp(r_.ci_high)
        col = C['red'] if lo > 1 else (c1 if hi < 1 else '#8A9099')
        ax.plot([lo, hi], [i, i], color=tint(col, 0.35), lw=2.4, solid_capstyle='round', zorder=2)
        ax.plot([1, orr], [i, i], color=col, lw=1.1, zorder=3)
        ax.plot([orr], [i], 'o', color=col, ms=4.0, mec='white', mew=0.6, zorder=4)
        ax.text(max(hi, orr) * 1.35, i, ('%.1f' % orr) if orr < 10 else ('%.0f' % orr), fontsize=5.4, color=col, va='center', ha='left')
    ax.axvline(1, color=INK, lw=0.7)
    ax.set_xscale('log'); ax.set_xlim(0.02, 2500); ax.set_xticks([0.1, 1, 10, 100]); ax.set_xticklabels(['0.1', '1', '10', '100']); ax.minorticks_off()
    ax.set_yticks(range(len(on))); ax.set_yticklabels([nice[t] for t in on.term], fontsize=6.0); ax.set_ylim(-0.6, len(on) - 0.4)
    ax.tick_params(axis='y', length=0); ax.grid(axis='y', visible=False); bare(ax)
    ax.set_xlabel('odds ratio of a flag onset', labelpad=1.5)
    tag(fig, 0.06, 1.79, 'c', 'what raises the risk')

    # ---- (d) position inside the annotated segment: speed (line) and flag rate (bars)
    ax = axi(fig, 3.38, 0.40, 1.22, 1.22)
    d = prof[prof.family == 'ALL']
    axb = ax.twinx()
    axb.bar(d.u_mid, d.flagged_per_1000, width=0.085, color=tint(C['red'], 0.30), zorder=1)
    axb.set_ylim(0, 60); axb.set_yticks([0, 10]); axb.tick_params(axis='y', labelsize=5.6, colors=C['red'], length=2)
    axb.spines['top'].set_visible(False); axb.spines['right'].set_color(tint(C['red'], 0.6)); axb.grid(False)
    axb.set_ylabel('flags per 1000 frames', fontsize=5.4, color=C['red'], loc='bottom', labelpad=1.5)
    ax.set_zorder(axb.get_zorder() + 1); ax.patch.set_visible(False)
    ax.fill_between(d.u_mid, d.ci_low, d.ci_high, color=tint(INK, 0.16), lw=0)
    for fam, col in (('P', C['blue']), ('A', C['orange']), ('L', C['green'])):
        q = prof[prof.family == fam]
        ax.plot(q.u_mid, q.ego_speed, color=col, lw=0.9)
        ax.text(q.u_mid.iloc[-1] + 0.03, q.ego_speed.iloc[-1], fam, fontsize=5.8, color=col, va='center', fontweight='bold')
    ax.plot(d.u_mid, d.ego_speed, color=INK, lw=1.6, marker='o', ms=2.6)
    ax.set_xlim(0, 1.08); ax.set_ylim(0.05, 0.24); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(['start', 'middle', 'end'])
    ax.set_yticks([0.10, 0.15, 0.20]); ax.set_ylabel('head-frame speed (m/s)', labelpad=1.5); ax.set_xlabel('position in the task segment', labelpad=1.5)
    tag(fig, 2.92, 1.79, 'd', 'along a task segment')

    # ---- (e) groups: two aligned columns with environment-cluster bootstrap intervals
    rows = [('family', 'P', 'family P'), ('family', 'A', 'family A'), ('family', 'L', 'family L'), ('dataset', 'EgoDemo', 'EgoDemo'),
            ('dataset', 'EgoPro', 'EgoPro'), ('dataset', 'EgoStandard', 'EgoStandard'), ('paradigm', 'natural_human_hand', 'bare hand'),
            ('paradigm', 'robot_gripper_mimic', 'gripper mimic')]
    y = np.arange(len(rows))
    for j, (q, col, sc, xl, lim, tk) in enumerate((('egocentric speed (m/s)', INK, 1.0, 'speed (m/s)', (0.07, 0.21), [0.1, 0.2]),
                                                 ('flagged share', C['red'], 100.0, 'flagged (%)', (0, 2.3), [0, 1, 2]))):
        ax = axi(fig, 5.42 + 0.56 * j, 0.40, 0.49, 1.22)
        for k in (0, 1, 2):                                                      # bands: family | dataset | hand type
            a_, b_ = ((-0.5, 2.5), (2.5, 5.5), (5.5, 7.5))[k]
            ax.axhspan(a_, b_, color='#F4F5F7' if k % 2 == 0 else 'white', lw=0, zorder=0)
        for i, (g_, lv, _) in enumerate(rows):
            r_ = grp[(grp.group == g_) & (grp.level == lv) & (grp.quantity == q)].iloc[0]
            ax.plot([r_.ci_low * sc, r_.ci_high * sc], [i, i], color=tint(col, 0.40), lw=2.2, solid_capstyle='round', zorder=2)
            ax.plot([r_.estimate * sc], [i], 'o', color=col, ms=3.4, mec='white', mew=0.5, zorder=3)
        ax.set_ylim(len(rows) - 0.5, -0.5); ax.set_xlim(*lim); ax.set_xticks(tk); ax.tick_params(axis='x', labelsize=6.0)
        ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0); bare(ax)
        ax.set_xlabel(xl, labelpad=1.5, fontsize=6.4)
        if j == 0:
            ax.set_yticks(y); ax.set_yticklabels([r_[2] for r_ in rows], fontsize=6.0)
        else:
            ax.set_yticks([])
    tag(fig, 4.80, 1.79, 'e', 'families, datasets, hand type')
    save(fig, 'fig04_reliability')


# =============================================================================================== Figure: admissible horizon (Problem 2)
def fig_horizon():
    growth = pd.read_csv(R('p2_growth.csv'), dtype={'phase': str})
    cov = pd.read_csv(R('p2_coverage.csv'))
    adm = pd.read_csv(R('p2_admissible.csv'), dtype={'phase': str})
    rad = pd.read_csv(R('p2_radius_curves.csv'), dtype={'phase': str})
    ras = pd.read_csv(R('p2_request_raster.csv'))
    E = pd.read_csv(R('p2_raster_error_mm.csv'), index_col=0).to_numpy()
    Rr = pd.read_csv(R('p2_raster_radius_mm.csv'), index_col=0).to_numpy()
    MODEL = load_json('frozen_p1_state.json')['model']
    c2, t2 = P_COL[2]
    fig = new_fig(2.62)

    # ---- (a) fan chart of the discrepancy after the last view + calibrated radius; coverage strip above
    ax = axi(fig, 0.47, 0.42, 2.02, 1.42)
    g = growth[(growth.split == 'test') & (growth.phase == 'all')]
    m = g[g.method == MODEL].sort_values('horizon_s'); hd = g[g.method == 'Hold'].sort_values('horizon_s')
    ra = rad[rad.phase == 'all'].sort_values('horizon_s')
    ax.fill_between(m.horizon_s, m.q25_mm, m.q75_mm, color=tint(INK, 0.22), lw=0, zorder=2)
    ax.plot(m.horizon_s, m.median_mm, color=INK, lw=1.3, zorder=3)
    ax.plot(hd.horizon_s, hd.median_mm, color='#9AA0A6', lw=1.0, ls=(0, (4, 2)), zorder=2)
    ax.plot(ra.horizon_s, ra.radius90_mm, color=c2, lw=2.0, zorder=4)
    tau = adm[adm.phase == 'all'].set_index('tolerance_mm').tau_star_s
    for tol in (30.0, 50.0, 80.0):
        t_ = float(tau.loc[tol])
        ax.plot([0.03, t_], [tol, tol], color=c2, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.plot([t_, t_], [0, tol], color=c2, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.plot([t_], [tol], 'o', color='white', mec=c2, mew=1.0, ms=3.6, zorder=6)
        ax.text(0.034, tol + 1.5, '%d mm' % tol, fontsize=5.5, color=c2, ha='left', va='bottom', fontweight='bold')
        ax.text(t_ * 1.05, 3.5, '%.2f s' % t_, fontsize=5.3, color=c2, ha='left', va='bottom', rotation=90, fontweight='bold',
                bbox=dict(fc='white', ec='none', pad=0.4, alpha=0.85), zorder=7)
    ax.set_xscale('log'); ax.set_xlim(0.031, 3.2); ax.set_ylim(0, 150)
    ax.set_xticks([1 / 30.0, 0.1, 0.3, 1, 3]); ax.set_xticklabels(['0.03', '0.1', '0.3', '1', '3']); ax.minorticks_off()
    ax.set_xlabel('time since the last view (s)', labelpad=1.5); ax.set_ylabel('discrepancy (mm)', labelpad=1.5)
    ax.text(0.62, 128, 'calibrated\n90 % radius', fontsize=5.6, color=c2, ha='right', va='center', fontweight='bold', linespacing=1.0)
    ax.text(2.9, 54, 'model: median\nand middle half', fontsize=5.4, color=INK, ha='right', va='top', linespacing=1.0)
    ax.text(2.9, 111, 'hold-last median', fontsize=5.4, color='#7E848B', ha='right', va='center')
    axc = axi(fig, 0.47, 1.93, 2.02, 0.36)
    c = cov[(cov.nominal == 0.9) & (cov.horizon_s > 0) & (cov.split == 'test')].sort_values('horizon_s')
    axc.axhline(90, color=INK, lw=0.6, ls=':')
    axc.plot(c.horizon_s, 100 * c.gauss_cov, color='#9AA0A6', lw=0.9, marker='v', ms=2.4)
    axc.plot(c.horizon_s, 100 * c.conf_cov, color=c2, lw=1.2, marker='o', ms=2.6)
    axc.set_xscale('log'); axc.set_xlim(0.031, 3.2); axc.set_ylim(66, 100); axc.set_yticks([70, 80, 90]); axc.minorticks_off()
    axc.set_xticks([1 / 30.0, 0.1, 0.3, 1, 3]); axc.set_xticklabels([]); axc.tick_params(axis='y', labelsize=5.6)
    axc.set_ylabel('inside (%)', fontsize=6.0, labelpad=1.5)
    axc.text(0.036, 71, 'Gaussian radius', fontsize=5.2, color='#7E848B', ha='left', va='center')
    axc.text(2.9, 96.5, 'calibrated radius', fontsize=5.2, color=c2, ha='right', va='center', fontweight='bold')
    tag(fig, 0.06, 2.47, 'a', 'discrepancy after the last view (test)')

    # ---- (b) admissible horizon by motion phase and tolerance (bars), hold-last as a tick
    ax = axi(fig, 3.02, 0.42, 1.06, 1.85)
    yy, lab, ypos = 0.0, [], []
    for tol in (30.0, 50.0, 80.0):
        for k in range(4):
            r_ = adm[(adm.tolerance_mm == tol) & (adm.phase == str(k + 1))].iloc[0]
            ax.barh(yy, r_.tau_star_s, height=0.78, color=PHASE_COL[k], zorder=2)
            ax.plot([r_.tau_star_hold_s] * 2, [yy - 0.39, yy + 0.39], color=INK, lw=1.1, zorder=3)
            ax.text(r_.tau_star_s + 0.012, yy, '%.2f' % r_.tau_star_s, fontsize=5.0, va='center', ha='left', color=INK)
            if tol == 30.0:
                ax.text(-0.012, yy, 'phase %d' % (k + 1), fontsize=5.0, va='center', ha='right', color=PHASE_COL[k], fontweight='bold')
            yy += 1
        ypos.append(yy - 2.5); lab.append('%d mm' % tol); yy += 0.9
    ax.set_ylim(yy - 0.6, -0.7); ax.set_xlim(0, 0.78); ax.set_xticks([0, 0.3, 0.6]); ax.set_yticks([])
    for yv, lb in zip(ypos, lab):
        ax.text(0.76, yv + 1.2, lb, fontsize=6.2, ha='right', va='center', color=c2, fontweight='bold')
    ax.grid(axis='y', visible=False); bare(ax)
    ax.set_xlabel('admissible horizon (s)', labelpad=1.5)
    ax.plot([0.47, 0.47], [0.2, 1.0], color=INK, lw=1.1); ax.text(0.50, 0.6, 'hold-last', fontsize=5.0, va='center', ha='left', color=INK)
    tag(fig, 2.62, 2.47, 'b', 'how long a view stays valid')

    # ---- (c) one held-out window under the trigger rule: calibrated radius, discrepancy and the requests
    ax = axi(fig, 4.60, 0.42, 1.72, 1.85)
    med = int(ras.requests.median())
    cand = ras[ras.requests == med].copy()
    gap_sd = cand.request_times_s.apply(lambda s: float(np.std(np.diff([float(x) for x in s.split(';')]))))
    tr = int(cand.loc[gap_sd.idxmax()].trial)                      # median number of requests, most uneven spacing (chosen by rule)
    t = (np.arange(E.shape[1]) + 1) / 30.0
    rq = np.array([float(x) for x in ras.loc[ras.trial == tr, 'request_times_s'].iloc[0].split(';')])
    free = np.interp(t, ra.horizon_s, ra.radius90_mm)
    ax.plot(t, free, color='#9AA0A6', lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.fill_between(t, 0, Rr[tr], color=t2, lw=0, zorder=2)
    ax.plot(t, Rr[tr], color=c2, lw=1.3, zorder=4)
    ax.plot(t, E[tr], color=INK, lw=0.9, zorder=5)
    ax.axhline(50, color=c2, lw=0.6, ls=(0, (3, 2)), zorder=3)
    ax.text(3.06, 50, '50 mm', fontsize=5.4, color=c2, ha='left', va='center', clip_on=False, fontweight='bold')
    for x in rq:
        ax.plot([x, x], [-17, -5], color=INK, lw=0.9, clip_on=False, zorder=6)
        ax.plot([x + 0.1], [-11], marker='|', color=c2, ms=3.2, mew=0.9, clip_on=False, zorder=6)
    ax.text(-0.07, -11, 'requests', fontsize=5.2, color=INK, ha='right', va='center', clip_on=False)
    ax.text(2.95, 131, 'without re-observation', fontsize=5.4, color='#7E848B', ha='right', va='center')
    ax.text(2.95, 119, 'radius carried by the planner', fontsize=5.4, color=c2, ha='right', va='center', fontweight='bold')
    ax.text(2.95, 107, 'realised discrepancy', fontsize=5.4, color=INK, ha='right', va='center')
    ax.set_xlim(0, 3.0); ax.set_ylim(-20, 150); ax.set_yticks([0, 50, 100, 150]); ax.set_xticks([0, 1, 2, 3])
    ax.spines['left'].set_bounds(0, 150)
    ax.set_xlabel('time inside the hidden window (s)', labelpad=1.5); ax.set_ylabel('mm', labelpad=1.0)
    tag(fig, 4.20, 2.47, 'c', 'one held-out window, %d requests' % len(rq))
    save(fig, 'fig05_horizon')


# =============================================================================================== Figure: request policy (Problem 2)
def fig_policy():
    front = pd.read_csv(R('p2_policy_frontier.csv'))
    cons = pd.read_csv(R('p2_policy_constrained.csv'))
    budget = pd.read_csv(R('p2_policy_budget.csv'))
    ras = pd.read_csv(R('p2_request_raster.csv'))
    E = pd.read_csv(R('p2_raster_error_mm.csv'), index_col=0).to_numpy()
    c2, t2 = P_COL[2]
    fig = new_fig(3.02)
    cmap = LinearSegmentedColormap.from_list('err', ['#FFFFFF', '#FBE3CC', '#F2A65A', '#D4660A', '#7A2E05'])

    # ---- (a) raster: one row per held-out window (every sixth window after sorting by the number of requests)
    order = np.argsort(ras.requests.to_numpy(), kind='stable')
    pick = order[::6]
    n = len(pick)
    ax = axi(fig, 0.42, 0.42, 2.72, 1.80)
    im = ax.imshow(E[pick], aspect='auto', cmap=cmap, vmin=0, vmax=80, extent=(0, 3.0, n, 0), interpolation='nearest', zorder=1)
    for i, tr in enumerate(pick):
        rq = [float(x) for x in ras.request_times_s.iloc[tr].split(';')] if isinstance(ras.request_times_s.iloc[tr], str) else []
        ax.plot(rq, [i + 0.5] * len(rq), '|', color='#101418', ms=2.3, mew=0.55, zorder=3)
    ax.set_xlim(0, 3.0); ax.set_ylim(n, 0); ax.set_yticks([]); ax.grid(False)
    ax.set_xlabel('time inside the hidden window (s)', labelpad=1.5); ax.set_xticks([0, 1, 2, 3])
    ax.text(-0.02, 0.5, '%d held-out windows, sorted by requests' % n, transform=ax.transAxes, rotation=90, fontsize=6.0, ha='right', va='center', color=INK)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color('#9AA0A6')
    # request rate over the window (all held-out windows), aligned with the time axis of the raster
    axt = axi(fig, 0.42, 2.27, 2.72, 0.30)
    allr = np.concatenate([[float(x) for x in s_.split(';')] for s_ in ras.request_times_s if isinstance(s_, str)])
    hh, edges = np.histogram(allr, bins=np.arange(0, 3.0001, 0.1))
    axt.bar(edges[:-1] + 0.05, hh / len(ras) / 0.1, width=0.085, color=c2, lw=0)
    axt.set_xlim(0, 3.0); axt.set_xticks([]); axt.set_yticks([0, 5]); axt.tick_params(axis='y', labelsize=5.4); axt.set_ylim(0, 6.5); axt.grid(False); bare(axt, left=True)
    axt.text(0.03, 6.3, 'requests per second, all %d windows' % len(ras), fontsize=5.2, color=c2, ha='left', va='center')
    # requests per window (same row order)
    axr = axi(fig, 3.18, 0.42, 0.34, 1.80)
    axr.barh(np.arange(n) + 0.5, ras.requests.to_numpy()[pick], height=1.0, color=tint(c2, 0.55), lw=0)
    axr.set_ylim(n, 0); axr.set_yticks([]); axr.set_xlim(0, 22); axr.set_xticks([10, 20]); axr.tick_params(axis='x', labelsize=5.4); axr.grid(False); bare(axr)
    axr.set_xlabel('requests', fontsize=5.8, labelpad=1.5)
    cax = axi(fig, 3.60, 1.32, 0.055, 0.90)
    cb = fig.colorbar(im, cax=cax, orientation='vertical'); cb.set_ticks([0, 40, 80]); cb.ax.tick_params(labelsize=5.4, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    fig.text(3.60 / W_FULL, 1.20 / 3.02, 'discrepancy\n(mm)', fontsize=5.2, color=INK, va='top', ha='left', linespacing=1.0)
    tag(fig, 0.06, 2.88, 'a', 'when the trigger rule asks for a new view (tolerance 50 mm, delay 0.1 s)')

    # ---- (b) cost against accuracy on the held-out windows
    ax = axi(fig, 4.55, 1.88, 1.92, 0.80)
    f = front[(front.split == 'test') & np.isclose(front.delay_s, 0.1)]
    ser = [('periodic', 'Hold', '#9AA0A6', 's', (0, (4, 2)), 'fixed interval, hold-last', 1.0), ('periodic', 'Model', tint(c2, 0.55), 'o', '-', 'fixed interval, model', 1.0),
           ('trigger', 'Model', c2, 'D', '-', 'trigger rule, model', 1.5)]
    for pol, pred, col, mk, ls, lb, lw in ser:
        d = f[(f.policy == pol) & (f.predictor == pred)].sort_values('requests')
        d = d[d.requests <= 26]
        ax.plot(d.requests, d.mean_err_mm, color=col, marker=mk, ms=2.4, lw=lw, ls=ls, label=lb, zorder=4 if pol == 'trigger' else 3)
    b = budget[(budget.split == 'test') & np.isclose(budget.delay_s, 0.1) & budget.method.str.contains('stage-aware')]
    ax.plot(b.requests, b.mean_err_mm, '*', color=INK, ms=4.2, mew=0, label='exact programme', zorder=5)
    ax.set_xlim(-0.5, 26.5); ax.set_ylim(10, 95); ax.set_xticks([0, 5, 10, 15, 20, 25]); ax.set_yticks([25, 50, 75])
    ax.set_xlabel('requests per 3-s window', labelpad=1.5); ax.set_ylabel('mean discrepancy\n(mm)', labelpad=1.5, fontsize=6.2, linespacing=1.0)
    ax.legend(loc='upper right', fontsize=5.3, handlelength=1.6, labelspacing=0.25, borderaxespad=0.1)
    tag(fig, 4.08, 2.88, 'b', 'cost against accuracy (held-out windows)')

    # ---- (c) requests needed to keep 90 % of the frames inside the tolerance
    ax = axi(fig, 5.02, 0.24, 1.46, 0.86)
    rowsm = [('periodic', 'Hold', 'fixed, hold-last'), ('periodic', 'CV', 'fixed, const. velocity'), ('periodic', 'Model', 'fixed, model'), ('trigger', 'Model', 'trigger, model')]
    cols = [(0.1, 30.0), (0.1, 50.0), (0.1, 80.0), (0.3, 30.0), (0.3, 50.0), (0.3, 80.0)]
    cm2 = LinearSegmentedColormap.from_list('req', ['#D4660A', '#F2A65A', '#FBE3CC'])
    for i, (pol, pred, lb) in enumerate(rowsm):
        for j, (dl, tol) in enumerate(cols):
            r_ = cons[(cons.policy == pol) & (cons.predictor == pred) & np.isclose(cons.delay_s, dl) & (cons.tolerance_mm == tol)].iloc[0]
            x0 = j + (0.18 if j >= 3 else 0.0)
            if bool(r_.feasible_on_validation):
                v = float(r_.test_requests)
                ax.add_patch(Rectangle((x0, i), 0.94, 0.90, fc=cm2(min(v, 26) / 26.0), ec='none', zorder=2))
                ax.text(x0 + 0.47, i + 0.45, '%.0f' % v, fontsize=6.2, ha='center', va='center', color=INK, fontweight='bold' if i >= 2 else 'normal', zorder=3)
            else:
                ax.add_patch(Rectangle((x0, i), 0.94, 0.90, fc='#F1F2F4', ec='#C9CDD2', lw=0.3, hatch='////', zorder=2))
        ax.text(-0.12, i + 0.45, lb, fontsize=5.8, ha='right', va='center', color=INK, fontweight='bold' if i >= 2 else 'normal')
    for j, (dl, tol) in enumerate(cols):
        ax.text(j + (0.18 if j >= 3 else 0.0) + 0.47, -0.16, '%d' % tol, fontsize=5.8, ha='center', va='bottom', color=INK)
    ax.text(1.41, -0.75, 'delay 0.1 s', fontsize=5.8, ha='center', va='bottom', color=INK); ax.text(4.59, -0.75, 'delay 0.3 s', fontsize=5.8, ha='center', va='bottom', color=INK)
    ax.text(-0.12, -0.16, 'tolerance (mm)', fontsize=5.6, ha='right', va='bottom', color=MUTE)
    ax.set_xlim(0, 6.2); ax.set_ylim(4.0, -0.2); ax.axis('off')
    ax.add_patch(Rectangle((3.18, 4.14), 0.5, 0.32, fc='#F1F2F4', ec='#C9CDD2', lw=0.3, hatch='////', clip_on=False))
    ax.text(3.80, 4.30, 'not attainable', fontsize=5.4, ha='left', va='center', color=MUTE, clip_on=False)
    tag(fig, 4.08, 1.47, 'c', 'requests needed for 90 % of frames in tolerance')
    save(fig, 'fig06_policy')


# =============================================================================================== Figure: natural occlusion (Problem 2)
def fig_natural():
    dur = pd.read_csv(R('p2_natural_duration.csv'))
    bridge = pd.read_csv(R('p2_natural_bridge.csv'))
    growth = pd.read_csv(R('p2_growth.csv'), dtype={'phase': str})
    nat = load_json('p2_natural_summary.json')
    c2, t2 = P_COL[2]
    fig = new_fig(2.02)

    # ---- (a) how long a flagged run lasts
    ax = axi(fig, 0.50, 0.42, 1.85, 1.28)
    d = dur[dur.frames <= 30]
    ax.fill_between(d.frames / 30.0, 0.004, d.empirical_survival, step='post', color=t2, lw=0, zorder=1)
    ax.step(d.frames / 30.0, d.empirical_survival, where='post', color=c2, lw=1.4, zorder=3)
    ax.plot(d.frames / 30.0, d.geometric_survival, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.set_yscale('log'); ax.set_ylim(0.004, 1.3); ax.set_xlim(0, 1.0); ax.set_yticks([1, 0.1, 0.01]); ax.set_yticklabels(['100 %', '10 %', '1 %'])
    ax.minorticks_off()
    ax.set_xlabel('duration of the flagged run (s)', labelpad=1.5); ax.set_ylabel('runs lasting longer', labelpad=1.5)
    ax.text(0.97, 0.9, 'median %.2f s' % nat['duration_quantiles_s']['0.5'], fontsize=5.8, color=c2, ha='right', va='center', fontweight='bold')
    ax.text(0.97, 0.52, '%.0f %% end within 0.2 s' % (100 * nat['share_le_0p2s']), fontsize=5.8, color=c2, ha='right', va='center')
    ax.text(0.97, 0.30, 'recovery %.2f per frame' % (1 - nat['recovery']['p_stay_geometric']), fontsize=5.8, color=INK, ha='right', va='center')
    ax.text(0.60, 0.012, 'geometric law', fontsize=5.4, color=INK, ha='left', va='center', rotation=-33)
    tag(fig, 0.06, 1.88, 'a', 'a flagged run is short')

    # ---- (b) natural gaps against artificial masks of the same length
    bg = nat['bridging']
    for j, (lb, a_, b_, lim, fmt, unit) in enumerate((('bridging discrepancy', bg['matched_artificial_model_mm'], bg['natural_model_mm'], (0, 95), '%.0f', 'mm'),
                                                    ('inside the calibrated radius', 100 * bg['artificial_inside_calibrated_90'], 100 * bg['natural_inside_calibrated_90'],
                                                     (60, 100), '%.0f', '%'))):
        ax = axi(fig, 3.00, 1.16 - 0.62 * j, 1.50, 0.30)
        ax.plot([a_, b_], [0, 0], color=tint(c2, 0.45), lw=3.0, solid_capstyle='round', zorder=2)
        ax.plot([a_], [0], 'o', color='white', mec=INK, mew=1.0, ms=5.0, zorder=3)
        ax.plot([b_], [0], 'o', color=c2, mec='white', mew=0.6, ms=5.6, zorder=3)
        ax.text(a_, 0.58, (fmt % a_) + ' ' + unit, fontsize=6.0, ha='center', va='bottom', color=INK)
        ax.text(b_, 0.58, (fmt % b_) + ' ' + unit, fontsize=6.0, ha='center', va='bottom', color=c2, fontweight='bold')
        ax.set_xlim(*lim); ax.set_ylim(-1, 1.6); ax.set_yticks([]); ax.grid(False); bare(ax)
        ax.tick_params(axis='x', labelsize=5.6); ax.set_xlabel(lb + (' (mm)' if unit == 'mm' else ' (%)'), fontsize=6.2, labelpad=1.0)
    fig.text(3.00 / W_FULL, 1.68 / 2.02, 'open: artificial mask, same length', fontsize=5.4, color=INK, va='center'); fig.text(4.62 / W_FULL, 1.68 / 2.02, 'filled: natural', fontsize=5.4, color=c2, va='center', ha='left', fontweight='bold')
    fig.text(4.56 / W_FULL, 1.31 / 2.02, r'$\times$%.2f' % bg['ratio_natural_to_artificial_model'][0], fontsize=7.0, color=c2, va='center', ha='left', fontweight='bold')
    S_ax = axi(fig, 2.62, 0.02, 2.4, 0.30); S_ax.set_xlim(0, 2.4); S_ax.set_ylim(0, 0.30); S_ax.axis('off')
    cr = bg['correction_rule']
    S.chip(S_ax, 1.13, 0.14, 'twice the radius: %d of %d test gaps inside, %.0f %% of all' % (cr['test_inside_twice_radius_count'], cr['n_test'],
                                                                                           100 * bg['natural_inside_twice_radius']), c2, fs=5.6, h=0.17,
           color=c2, weight='bold')
    tag(fig, 2.62, 1.88, 'b', 'natural gaps are harder')

    # ---- (c) motion phase just before the onset
    ax = axi(fig, 5.35, 0.42, 1.12, 1.28)
    pn = bridge.phase_before.value_counts(normalize=True).reindex([1, 2, 3, 4]).fillna(0).to_numpy()
    g0 = growth[(growth.method == 'Hold') & (growth.phase != 'all')]
    g0 = g0[np.isclose(g0.horizon_s, g0.horizon_s.min())].groupby('phase').n.sum()
    pa = (g0 / g0.sum()).reindex(['1', '2', '3', '4']).to_numpy()
    for x, p in ((0, pa), (1, pn)):
        bot = 0.0
        for k in range(4):
            ax.bar(x, p[k], bottom=bot, width=0.62, color=PHASE_COL[k], ec='white', lw=0.6)
            if p[k] > 0.10:
                ax.text(x, bot + p[k] / 2, '%.0f %%' % (100 * p[k]), fontsize=5.6, ha='center', va='center', color='white', fontweight='bold')
            bot += p[k]
    for k in range(4):                                                            # ribbons between the two bars
        y0a, y1a = pa[:k].sum(), pa[:k + 1].sum(); y0n, y1n = pn[:k].sum(), pn[:k + 1].sum()
        ax.add_patch(Polygon([[0.31, y0a], [0.69, y0n], [0.69, y1n], [0.31, y1a]], closed=True, fc=tint(PHASE_COL[k], 0.28), ec='none', zorder=1))
        ax.text(1.36, (y0n + y1n) / 2, 'phase %d' % (k + 1), fontsize=5.4, color=PHASE_COL[k], va='center', ha='left', fontweight='bold')
    ax.set_xlim(-0.4, 1.95); ax.set_ylim(0, 1); ax.set_xticks([0, 1]); ax.set_xticklabels(['artificial\nmasks', 'natural\nonsets'], fontsize=6.0)
    ax.set_yticks([]); ax.grid(False); bare(ax, bottom=False); ax.tick_params(axis='x', length=0)
    tag(fig, 5.05, 1.88, 'c', 'they start in fast motion')
    save(fig, 'fig07_natural')


# =============================================================================================== Figure: duration and reservation (Problem 3)
FAM_COL = {'P': C['blue'], 'A': C['orange'], 'L': C['green']}
FAM_NAME = {'P': 'placement', 'A': 'arrangement', 'L': 'inspection'}


def _kde(x, grid, bw):
    """Gaussian kernel density of x on `grid` (NumPy only)."""
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(axis=1) / (len(x) * bw * np.sqrt(2 * np.pi))


def fig_duration():
    from p3_policy import Belief
    seg = pd.read_csv(R('p3_segment_features.csv'), usecols=['unit_id', 'family', 'split', 'd'])
    fz = load_json('frozen_p3_duration.json')
    fr = pd.read_csv(R('p3_frontier.csv')).sort_values('reserve_floor')
    c3, t3 = P_COL[3]
    fig = new_fig(2.42)

    # ---- (a) raincloud of the annotated durations by job family
    ax = axi(fig, 1.02, 0.42, 1.50, 1.66)
    rng = np.random.default_rng(3)
    grid = np.linspace(np.log(0.8), np.log(150), 300)
    for i, fam in enumerate(('P', 'A', 'L')):
        d = seg[seg.family == fam]
        ld = np.log(d.d.to_numpy())
        dens = _kde(ld, grid, 0.9 * ld.std() * len(ld) ** -0.2)
        ax.fill_between(np.exp(grid), i, i - 0.44 * dens / dens.max(), color=tint(FAM_COL[fam], 0.55), lw=0, zorder=2)
        ax.plot(np.exp(grid), i - 0.44 * dens / dens.max(), color=FAM_COL[fam], lw=0.9, zorder=3)
        tr = (d.split == 'train').to_numpy()
        jit = i + 0.10 + 0.26 * rng.random(len(d))
        ax.scatter(d.d[tr], jit[tr], s=2.2, color='#9AA0A6', linewidths=0, zorder=2)
        ax.scatter(d.d[~tr], jit[~tr], s=3.6, color=FAM_COL[fam], linewidths=0, zorder=3)
        q1, q2, q3 = np.quantile(d.d, [0.25, 0.5, 0.75])
        ax.plot([q1, q3], [i + 0.045, i + 0.045], color=INK, lw=1.6, solid_capstyle='butt', zorder=4)
        ax.plot([q2], [i + 0.045], 'o', color='white', mec=INK, mew=0.8, ms=3.0, zorder=5)
        ax.text(0.62, i - 0.14, '%s  %s' % (fam, FAM_NAME[fam]), fontsize=6.0, color=FAM_COL[fam], ha='right', va='center', fontweight='bold', clip_on=False)
        ax.text(0.62, i + 0.14, '%d units, median %.1f s' % (len(d), q2), fontsize=5.2, color=INK, ha='right', va='center', clip_on=False)
    ax.set_xscale('log'); ax.set_xlim(0.8, 150); ax.set_xticks([1, 3, 10, 30, 100]); ax.set_xticklabels(['1', '3', '10', '30', '100']); ax.minorticks_off()
    ax.set_ylim(2.5, -0.55); ax.set_yticks([]); ax.grid(axis='y', visible=False); bare(ax)
    ax.set_xlabel('annotated duration (s)', labelpad=1.5)
    ax.text(150, -0.52, 'grey: training\ncolour: held-out\nenvironments', fontsize=5.0, color=MUTE, ha='right', va='top', linespacing=1.1)
    tag(fig, 0.06, 2.27, 'a', 'durations of the recorded work units')

    # ---- (b) where the uncertainty of a new store comes from (variance shares of the log duration)
    ax = axi(fig, 2.80, 0.92, 0.98, 0.98); ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    v = np.array([fz['tau_g'] ** 2, fz['tau_f'] ** 2, fz['sigma'] ** 2]); v = v / v.sum()
    cols3 = [tint(c3, 0.45), tint(c3, 0.80), '#C9CED4']
    names3 = ('between stores', 'store and family', 'unit to unit')
    a0 = 90.0
    for sh, col in zip(v, cols3):
        a1 = a0 - 360.0 * sh
        ax.add_patch(Wedge((0, 0), 1.0, a1, a0, width=0.44, fc=col, ec='white', lw=1.0))
        am = np.deg2rad((a0 + a1) / 2)
        ax.text(0.78 * np.cos(am), 0.78 * np.sin(am), '%.0f' % (100 * sh), fontsize=5.8, ha='center', va='center', color=INK, fontweight='bold')
        a0 = a1
    ax.text(0, 0.10, 'variance of', fontsize=5.2, ha='center', va='center', color=MUTE); ax.text(0, -0.12, 'log duration', fontsize=5.2, ha='center', va='center', color=MUTE)
    for k, (col, nm, sh) in enumerate(zip(cols3, names3, v)):
        fig.patches.append(Rectangle(((2.78) / W_FULL, (0.70 - 0.13 * k) / 2.42), 0.09 / W_FULL, 0.075 / 2.42, fc=col, ec='none', transform=fig.transFigure))
        fig.text(2.91 / W_FULL, (0.737 - 0.13 * k) / 2.42, '%s  %.0f %%' % (nm, 100 * sh), fontsize=5.5, color=INK, va='center', ha='left')
    tag(fig, 2.66, 2.27, 'b', 'what is unknown')

    # ---- (c) predictive distribution of one service time and the candidate reservations
    ax = axi(fig, 4.08, 0.42, 1.12, 1.56)
    bel = Belief(fz, 'family', False)
    p = bel.sample([('x', 'P')], 20000, np.random.default_rng(11))[:, 0] + 0.5          # nominal mapping: service = duration + 0.5 s
    g = np.linspace(np.log(0.8), np.log(60), 300)
    dens = _kde(np.log(p), g, 0.12)
    x = np.exp(g)
    qs = [0.5, 0.6, 0.75, 0.9]; bq = np.quantile(p, qs)
    ax.fill_between(x, 0, dens, where=x <= bq[0], color=tint(c3, 0.30), lw=0, zorder=1)
    ax.fill_between(x, 0, dens, where=x >= bq[0], color=tint(C['red'], 0.28), lw=0, zorder=1)
    ax.plot(x, dens, color=INK, lw=1.0, zorder=3)
    for q_, b_ in zip(qs, bq):
        yt = float(np.interp(np.log(b_), g, dens))
        hot = q_ == 0.5
        ax.plot([b_, b_], [0, yt], color=c3 if hot else '#6F757C', lw=1.5 if hot else 0.7, ls='-' if hot else (0, (2, 1.5)), zorder=4)
        ax.text(b_ * (0.93 if hot else 1.05), yt + 0.012 * dens.max() + (0.05 * dens.max() if hot else 0.0), ('median' if hot else '%g' % q_), fontsize=5.0,
                ha='right' if hot else 'left', va='bottom', color=c3 if hot else '#6F757C', fontweight='bold' if hot else 'normal')
    ax.text(bq[0] * 0.56, 0.26 * dens.max(), 'early:\nidle', fontsize=5.2, ha='center', va='center', color=c3, linespacing=1.0, fontweight='bold')
    ax.text(bq[0] * 1.75, 0.10 * dens.max(), 'overrun', fontsize=5.2, ha='left', va='center', color=C['red'], fontweight='bold')
    ax.set_xscale('log'); ax.set_xlim(0.9, 50); ax.set_ylim(0, dens.max() * 1.22); ax.set_yticks([]); ax.grid(False); bare(ax)
    ax.set_xticks([1, 3, 10, 30]); ax.set_xticklabels(['1', '3', '10', '30']); ax.minorticks_off()
    ax.set_xlabel('service time, placement job (s)', labelpad=1.5, fontsize=6.2)
    tag(fig, 3.98, 2.27, 'c', 'how long to reserve')

    # ---- (d) overrun against idle time when the reservation may not go below a quantile; objective at the right
    ax = axi(fig, 5.66, 0.42, 0.60, 1.44)
    labs = ['min', '0.25', 'median', '0.75', '0.9']
    for i, (_, r_) in enumerate(fr.iterrows()):
        hot = abs(r_.reserve_floor - 0.5) < 1e-9
        ax.barh(i, -r_.observed_overrun_s, height=0.66, color=tint(C['red'], 0.75 if hot else 0.40), lw=0, zorder=2)
        ax.barh(i, r_.reserved_idle_s, height=0.66, color=tint(c3, 0.85 if hot else 0.45), lw=0, zorder=2)
        ax.text(57, i, '%.2f' % r_.utility, fontsize=5.8, ha='left', va='center', color=INK, fontweight='bold' if hot else 'normal', clip_on=False)
        ax.text(-86, i, labs[i], fontsize=5.6, ha='right', va='center', color=INK, fontweight='bold' if hot else 'normal', clip_on=False)
    ax.axvline(0, color=INK, lw=0.7, zorder=3)
    ax.set_xlim(-82, 52); ax.set_ylim(4.6, -0.6); ax.set_yticks([]); ax.set_xticks([-60, 0, 40]); ax.set_xticklabels(['60', '0', '40'])
    ax.grid(axis='y', visible=False); bare(ax)
    ax.set_xlabel('seconds per trial', labelpad=1.0, fontsize=6.2)
    ax.text(-40, -0.95, 'overrun', fontsize=5.4, ha='center', va='center', color=C['red'], fontweight='bold', clip_on=False)
    ax.text(26, -0.95, 'idle', fontsize=5.4, ha='center', va='center', color=c3, fontweight='bold', clip_on=False)
    ax.text(57, -0.95, '$U$', fontsize=6.0, ha='left', va='center', color=INK, clip_on=False)
    tag(fig, 5.30, 2.27, 'd', 'what it costs')
    fig.text(5.30 / W_FULL, 2.12 / 2.42, 'lowest reservation quantile allowed', fontsize=5.0, color=MUTE, va='center', ha='left')
    save(fig, 'fig08_duration')


# =============================================================================================== Figure: one decision of the look-ahead policy
def fig_decision():
    tr = load_json('p3_decision_trace.json')['S2']
    K = 3                                                           # fourth decision of the replayed S2 trial of the schedule figure
    e = tr['epochs'][K]
    jobs = {j['job_id']: j for j in tr['jobs']}
    short = lambda q: 'J' + q.split('_J')[1]
    c3, t3 = P_COL[3]
    W, H = W_FULL, 2.92
    fig, ax = S.canvas(W, H)

    # ---- header: the decisions of the trial on a time line
    y0 = 2.52
    x_of = lambda t: 1.05 + (W - 1.45) * t / tr['H']
    ax.plot([x_of(0), x_of(tr['H'])], [y0, y0], color='#C9CED4', lw=1.0, zorder=1)
    for t_ in (0, 25, 50, 75, 100):
        ax.plot([x_of(t_)] * 2, [y0 - 0.03, y0 + 0.03], color='#9AA0A6', lw=0.6)
        ax.text(x_of(t_), y0 - 0.10, '%d s' % t_, fontsize=5.0, ha='center', va='center', color=MUTE)
    for i, ep in enumerate(tr['epochs']):
        if ep['chosen'] != 'exec':
            ax.plot([x_of(ep['t'])], [y0], 's', color=INK, ms=3.2, zorder=3)
            ax.text(x_of(ep['t']), y0 + 0.11, 'stop', fontsize=5.0, ha='center', va='center', color=INK)
            continue
        fam = jobs[ep['job']]['family']
        hot = i == K
        ax.add_patch(Circle((x_of(ep['t']), y0), 0.062 if hot else 0.042, fc=FAM_COL[fam], ec=INK if hot else 'white', lw=1.0 if hot else 0.6, zorder=4))
        ax.text(x_of(ep['t']), y0 + (0.14 if hot else 0.11), short(ep['job']), fontsize=5.6 if hot else 5.0, ha='center', va='center',
                color=INK, fontweight='bold' if hot else 'normal')
    ax.text(0.10, y0 + 0.02, 'decisions of', fontsize=5.6, ha='left', va='center', color=INK); ax.text(0.10, y0 - 0.09, 'one S2 trial', fontsize=5.6, ha='left', va='center', color=INK)
    ax.plot([x_of(e['t']), x_of(e['t'])], [y0 - 0.07, 2.27], color=INK, lw=0.7, ls=(0, (2, 1.5)), zorder=2)

    # ---- (a) the store at the decision
    S.card(ax, 0.06, 0.08, 3.16, 2.12, fc='#FBFCFD', ec='#D4D8DD', lw=0.6, r=0.08, shadow=False, z=0.5)
    ax.text(0.14, 2.07, '(a)', fontsize=8.5, fontweight='bold', va='center', color='#111111')
    ax.text(0.39, 2.07, 'the store at $t$ = %.1f s: six jobs can start' % e['t'], fontsize=6.4, va='center', color=INK)
    zx = lambda z: 0.40 + 1.02 * tr['zones'][z][0]
    zy = lambda z: 0.66 + 0.88 * tr['zones'][z][1]
    done = [q for q in jobs if tr['completion'][q] is not None and tr['completion'][q] <= e['t'] + 1e-9]
    for z in tr['zones']:
        if z == tr['initial_zone']:
            ax.add_patch(Circle((zx(z), zy(z)), 0.10, fc='white', ec='#9AA0A6', lw=0.7, zorder=3))
            ax.text(zx(z), zy(z), 'start', fontsize=5.0, ha='center', va='center', color=MUTE, zorder=4)
        else:
            S.shelf(ax, zx(z), zy(z), w=0.22, h=0.19, col='#6F757C', lw=0.7, z=3)
            ax.text(zx(z), zy(z) - 0.15, z, fontsize=5.4, ha='center', va='center', color='#6F757C', fontweight='bold')
    S.robot(ax, zx(e['zone']) + 0.36, zy(e['zone']) - 0.04, s=0.13, col=INK, z=7)
    ax.text(zx(e['zone']) + 0.36, zy(e['zone']) - 0.21, 'robot', fontsize=5.0, ha='center', va='center', color=INK)
    # job chips next to their zone: released (solid), not yet released (dashed), finished (grey)
    slots = {}
    for q, j in jobs.items():
        slots.setdefault(j['zone'], []).append(q)
    for z, qs in slots.items():
        qs = sorted(qs, key=lambda q: (q in done, q not in e['released'], jobs[q]['due']))
        for k, q in enumerate(qs):
            j = jobs[q]
            if z == e['zone']:                                      # zone of the robot: chips stacked above the shelf
                cx = zx(z) - 0.30; cy = zy(z) + 0.20 + 0.125 * (len(qs) - 1 - k)
            else:
                cx = zx(z) + 0.16; cy = zy(z) + 0.105 - 0.125 * k
            col = FAM_COL[j['family']]
            if q in done:
                ax.text(cx + 0.04, cy - 0.003, short(q) + '  done', fontsize=5.0, ha='left', va='center', color='#A3A9B0')
                continue
            rel = q in e['released']
            hot = q == e['job']
            lab = '%s  w%d  due %d' % (short(q), j['weight'], j['due'])
            w_ = S.text_w(ax, lab, 5.0, 'bold' if hot else 'normal') + 0.08
            ax.add_patch(FancyBboxPatch((cx, cy - 0.052), w_, 0.104, boxstyle='round,pad=0,rounding_size=0.05', fc=tint(col, 0.22) if rel else 'white',
                                        ec=col if rel else '#B8BDC4', lw=1.3 if hot else 0.6, ls='-' if rel else (0, (2, 1.5)), zorder=5))
            ax.text(cx + 0.04, cy - 0.003, lab, fontsize=5.0, ha='left', va='center', color=INK if rel else '#9AA0A6',
                    fontweight='bold' if hot else 'normal', zorder=6)
            if q == e['edd_choice']:
                ax.text(cx + w_ / 2, cy + 0.10, 'earliest due date', fontsize=5.0, ha='center', va='center', color=C['red'], style='italic')
            if hot:
                ax.text(cx + w_ / 2, cy + 0.10, 'chosen', fontsize=5.0, ha='center', va='center', color=c3, fontweight='bold')
    for k, (fam, nm) in enumerate((('P', 'placement'), ('A', 'arrangement'), ('L', 'inspection'))):
        ax.add_patch(Circle((0.20 + 0.72 * k, 0.30), 0.032, fc=FAM_COL[fam], ec='none', zorder=5))
        ax.text(0.26 + 0.72 * k, 0.298, nm, fontsize=5.0, ha='left', va='center', color=INK)
    ax.text(0.17, 0.17, 'w: weight   due: due time (s)   dashed outline: not released yet   grey: finished', fontsize=5.0, ha='left', va='center', color=MUTE)

    # ---- (b) value of each candidate first action
    axb = fig.add_axes([4.12 / W, 0.42 / H, 0.86 / W, 1.44 / H])
    vals = sorted(e['values'].items(), key=lambda kv: -kv[1])
    lo = 0.40
    for i, (k, v) in enumerate(vals):
        hot = k == e['job']
        col = c3 if hot else ('#9AA0A6' if k in ('wait', 'stop') else FAM_COL[jobs[k]['family']])
        vv = max(v, lo)
        axb.plot([lo, vv], [i, i], color=tint(col, 0.55) if not hot else col, lw=2.6 if hot else 1.6, solid_capstyle='round', zorder=2)
        axb.plot([vv], [i], 'o', color=col, ms=4.6 if hot else 3.4, mec='white', mew=0.6, zorder=3)
        lab = {'wait': 'wait for a release', 'stop': 'stop now'}.get(k, None) or ('%s  %s' % (short(k), FAM_NAME[jobs[k]['family']]))
        axb.text(lo - 0.008, i, lab, fontsize=5.6, ha='right', va='center', color=INK, fontweight='bold' if hot else 'normal')
        axb.text(vv + 0.012, i, ('%.3f' % v) if v >= lo else ('%.2f' % v), fontsize=5.4, ha='left', va='center', color=INK, fontweight='bold' if hot else 'normal')
        if k == e['edd_choice']:
            axb.text(lo + 0.004, i - 0.36, 'earliest due date', fontsize=5.0, ha='left', va='center', color=C['red'], style='italic')
    axb.set_xlim(lo, 0.66); axb.set_ylim(len(vals) - 0.5, -0.6); axb.set_yticks([]); axb.set_xticks([0.4, 0.5, 0.6]); axb.grid(axis='y', visible=False)
    for sp in ('top', 'right', 'left'):
        axb.spines[sp].set_visible(False)
    axb.set_xlabel('expected objective', labelpad=1.5, fontsize=6.2)
    ax.text(3.34, 2.07, '(b)', fontsize=8.5, fontweight='bold', va='center', color='#111111')
    ax.text(3.59, 2.07, 'each first action is rolled out', fontsize=6.4, va='center', color=INK)
    ax.text(3.59, 1.955, 'over 100 duration scenarios', fontsize=5.4, va='center', color=MUTE)

    # ---- (c) reservation of the chosen job
    axc = fig.add_axes([5.52 / W, 0.42 / H, 0.98 / W, 1.44 / H])
    sm = np.array(e['samples'])
    bins = np.exp(np.linspace(np.log(1.0), np.log(50), 22))
    hh, _ = np.histogram(sm, bins=bins)
    axc.bar(bins[:-1], hh, width=np.diff(bins), align='edge', color=tint(FAM_COL[jobs[e['job']]['family']], 0.40), ec='white', lw=0.4, zorder=2)
    res = sorted(((float(k), v) for k, v in e['reserve'].items()))
    top = hh.max()
    for i, (b_, v) in enumerate(res):
        hot = abs(b_ - e['b']) < 1e-3
        yl = top * (1.20 + 0.19 * i)
        axc.plot([b_, b_], [0, yl], color=c3 if hot else '#6F757C', lw=1.5 if hot else 0.7, ls='-' if hot else (0, (2, 1.5)), zorder=4)
        axc.text(b_ / 1.07, yl, '%.1f s: %.3f' % (b_, v), fontsize=5.2, ha='right', va='center', color=c3 if hot else '#6F757C',
                 fontweight='bold' if hot else 'normal', clip_on=False)
    axc.annotate('realised\n%.1f s' % e['realised_s'], xy=(e['realised_s'], 0.0), xytext=(22, top * 0.62), fontsize=5.0, color=INK, ha='center', va='center',
                 linespacing=1.0, arrowprops=dict(arrowstyle='-|>,head_length=0.35,head_width=0.18', color=INK, lw=0.6, shrinkA=1, shrinkB=0), zorder=7)
    axc.set_xscale('log'); axc.set_xlim(1.0, 50); axc.set_ylim(0, top * 1.90); axc.set_yticks([]); axc.grid(False)
    axc.set_xticks([1, 3, 10, 30]); axc.set_xticklabels(['1', '3', '10', '30']); axc.minorticks_off()
    for sp in ('top', 'right', 'left'):
        axc.spines[sp].set_visible(False)
    axc.set_xlabel('service time of %s (s)' % short(e['job']), labelpad=1.5, fontsize=6.2)
    ax.text(5.14, 2.07, '(c)', fontsize=8.5, fontweight='bold', va='center', color='#111111')
    ax.text(5.39, 2.07, 'then its reservation', fontsize=6.4, va='center', color=INK)
    ax.text(5.39, 1.955, 'length: expected objective', fontsize=5.4, va='center', color=MUTE)
    save(fig, 'fig09_decision')


# =============================================================================================== Figure: comparison of the policies (Problem 3)
POL = [('FIFO-mean', 'first come,\nfirst served', 'FIFO'), ('EDD-median', 'earliest\ndue date', 'EDD'), ('Index-static', 'priority\nindex', 'index'),
       ('FROZEN', 'look-ahead\n(ours)', 'look-ahead'), ('CLAIRVOYANT', 'known\ndurations', 'known dur.')]


def fig_comparison():
    vnom = pd.read_csv(R('p3_table_validation_nominal.csv'))
    tnom = pd.read_csv(R('p3_table_test_nominal.csv'))
    mp = pd.read_csv(R('p3_table_mappings.csv'))
    tv = pd.read_csv(R('p3_trials_validation.csv'))
    c3, t3 = P_COL[3]
    fig = new_fig(2.66)
    uv = vnom.groupby('policy').utility.mean(); ut = tnom.groupby('policy').utility.mean()

    # ---- (a) staircase of the objective: each step adds one ingredient
    ax = axi(fig, 0.45, 0.62, 2.12, 1.52)
    prev = 0.0
    for i, (p, lb, _) in enumerate(POL):
        v = float(uv[p]); hot = p == 'FROZEN'; ref = p == 'CLAIRVOYANT'
        col = c3 if hot else ('#FFFFFF' if ref else '#B4BAC1')
        ax.bar(i, v - prev, bottom=prev, width=0.62, color=col, ec='#6F757C' if ref else 'none', lw=0.7, hatch='////' if ref else None, zorder=2)
        if i:
            ax.plot([i - 1 + 0.31, i - 0.31], [prev, prev], color='#6F757C', lw=0.6, zorder=3)
        txt = ('+%.2f' % (v - prev)) if i else ('%.2f' % v)
        if v - prev > 0.10:
            ax.text(i - 0.07, (v + prev) / 2, txt, fontsize=5.8, ha='center', va='center', color=INK, zorder=6,
                    bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.85) if ref else None)
        else:
            ax.text(i, v + 0.035, txt, fontsize=5.8, ha='center', va='bottom', color=c3 if hot else INK, fontweight='bold', zorder=6)
        ax.plot([i + 0.20], [float(ut[p])], 'D', color='white', mec=INK, mew=0.8, ms=3.4, zorder=5)
        prev = v
    ax.plot([], [], 'D', color='white', mec=INK, mew=0.8, ms=3.4, label='test environments')
    ax.bar([0], [0], color='#B4BAC1', label='validation environments')
    ax.set_xticks(range(len(POL))); ax.set_xticklabels([lb for _, lb, _ in POL], fontsize=5.6, linespacing=1.0)
    ax.get_xticklabels()[3].set_color(c3); ax.get_xticklabels()[3].set_fontweight('bold')
    ax.set_ylim(0, 1.06); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.grid(axis='x', visible=False); ax.tick_params(axis='x', length=0)
    ax.set_ylabel('objective $U$', labelpad=1.5)
    ax.legend(loc='upper left', fontsize=5.4, handlelength=1.0, borderaxespad=0.1, labelspacing=0.25)
    tag(fig, 0.06, 2.50, 'a', 'what each ingredient adds')

    # ---- (b) the common metrics on one radar (validation environments, nominal mapping)
    nom = tv[(tv.alpha == 1.0) & (tv.beta == 0.5)].copy()
    nom['low_tard'] = 1.0 - nom.weighted_tardiness_s / nom.wt_max
    nom['capacity'] = 1.0 - (nom.observed_overrun_s + nom.reserved_idle_s) / nom.H
    keys = [('weighted_on_time_rate', 'on time'), ('weighted_completion_rate', 'completed'), ('low_tard', 'low\ntardiness'),
            ('capacity', 'capacity\nplan kept'), ('utility', 'objective')]
    M = nom.groupby('policy')[[k for k, _ in keys]].mean()
    axp = axi(fig, 3.16, 0.52, 1.30, 1.30, projection='polar')
    axp.set_theta_offset(np.pi / 2); axp.set_theta_direction(-1)
    th = np.linspace(0, 2 * np.pi, len(keys), endpoint=False)
    close = lambda v: np.concatenate([v, v[:1]])
    for p, col, lw, ls, fill in (('FIFO-mean', '#B4BAC1', 0.9, '-', 0.0), ('EDD-median', '#6F757C', 0.9, (0, (3, 1.5)), 0.0), ('CLAIRVOYANT', INK, 0.7, (0, (1, 1.5)), 0.0),
                                 ('FROZEN', c3, 1.5, '-', 0.22)):
        v = M.loc[p].to_numpy()
        axp.plot(close(th), close(v), color=col, lw=lw, ls=ls, zorder=4 if p == 'FROZEN' else 3)
        if fill:
            axp.fill(close(th), close(v), color=col, alpha=fill, lw=0, zorder=2)
    axp.set_ylim(0.2, 1.0); axp.set_yticks([0.4, 0.6, 0.8, 1.0]); axp.set_yticklabels(['', '0.6', '', '1'], fontsize=5.0, color=MUTE)
    axp.set_rlabel_position(36); axp.set_xticks(th); axp.set_xticklabels([lb for _, lb in keys], fontsize=5.6, linespacing=1.0)
    axp.tick_params(axis='x', pad=-1.0); axp.spines['polar'].set_linewidth(0.5); axp.spines['polar'].set_color('#9AA0A6'); axp.grid(lw=0.35, color='#D5D8DC')
    h = [Line2D([], [], color=c3, lw=1.5, label='look-ahead (ours)'), Line2D([], [], color='#6F757C', lw=0.9, ls=(0, (3, 1.5)), label='earliest due date'),
         Line2D([], [], color='#B4BAC1', lw=0.9, label='first come, first served'), Line2D([], [], color=INK, lw=0.7, ls=(0, (1, 1.5)), label='known durations')]
    fig.legend(handles=h, loc='lower left', bbox_to_anchor=(2.84 / W_FULL, 0.03 / 2.66), ncol=2, fontsize=5.2, handlelength=1.6, columnspacing=0.9, labelspacing=0.25,
               borderaxespad=0.0)
    tag(fig, 2.86, 2.50, 'b', 'all common metrics at once')

    # ---- (c) objective under the five duration mappings, both pools
    ax = axi(fig, 5.22, 0.12, 1.26, 2.20)
    maps = [(0.75, 0.5), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (1.5, 0.5)]
    cm3 = LinearSegmentedColormap.from_list('u', ['#FFFFFF', '#CDEBE1', '#57B99A', '#0B8A66'])
    yy = 0.0
    for pool in ('validation', 'test'):
        ax.text(0.0, yy - 0.34, pool + ' environments', fontsize=5.4, ha='left', va='center', color=MUTE)
        for (p, _, sh) in POL:
            d = mp[(mp.pool == pool) & (mp.policy == p)]
            hot = p == 'FROZEN'
            for j, (al, be) in enumerate(maps):
                v = float(d[np.isclose(d.alpha, al) & np.isclose(d.beta, be)].utility.mean())
                ax.add_patch(Rectangle((j, yy), 0.95, 0.92, fc=cm3(float(np.clip(v, 0, 1))), ec='none', zorder=2))
                ax.text(j + 0.475, yy + 0.47, '%.2f' % v, fontsize=5.2, ha='center', va='center', color='white' if v > 0.72 else INK,
                        fontweight='bold' if hot else 'normal', zorder=3)
            if hot:
                ax.add_patch(Rectangle((-0.03, yy - 0.03), 5.01, 0.98, fc='none', ec=INK, lw=0.9, zorder=4))
            ax.text(-0.14, yy + 0.47, sh, fontsize=5.4, ha='right', va='center', color=c3 if hot else INK, fontweight='bold' if hot else 'normal')
            yy += 1.0
        yy += 0.85
    for j, (al, be) in enumerate(maps):
        ax.text(j + 0.475, -1.85, '%g' % al, fontsize=5.4, ha='center', va='center', color=INK); ax.text(j + 0.475, -1.15, '%g' % be, fontsize=5.4, ha='center', va='center', color=INK)
    ax.text(-0.14, -1.85, r'scale $\alpha$', fontsize=5.4, ha='right', va='center', color=INK); ax.text(-0.14, -1.15, r'offset $\beta$ (s)', fontsize=5.4, ha='right', va='center', color=INK)
    ax.set_xlim(0, 5.0); ax.set_ylim(yy - 0.85, -2.3); ax.axis('off')
    tag(fig, 4.72, 2.50, 'c', 'duration mappings')
    save(fig, 'fig11_comparison')


# =============================================================================================== Figure: sensitivity summary
def fig_sensitivity():
    cons = pd.read_csv(R('p2_policy_constrained.csv'))
    vnom = pd.read_csv(R('p3_table_validation_nominal.csv'))
    mp = pd.read_csv(R('p3_table_mappings.csv'))
    fr = pd.read_csv(R('p3_frontier.csv')).sort_values('reserve_floor')
    (c2, t2), (c3, t3) = P_COL[2], P_COL[3]
    fig = new_fig(2.30)
    uv = vnom.groupby('policy').utility.mean()

    def req(pol, pred, dl, tol):
        r_ = cons[(cons.policy == pol) & (cons.predictor == pred) & np.isclose(cons.delay_s, dl) & (cons.tolerance_mm == tol)].iloc[0]
        return float(r_.test_requests)

    def umap(al, be):
        d = mp[(mp.pool == 'validation') & (mp.policy == 'FROZEN')]
        return float(d[np.isclose(d.alpha, al) & np.isclose(d.beta, be)].utility.mean())

    def panel(ax, rows, col, tcol, xlim, xlabel, fmt):
        """rows: (name, [(setting label, value), ...] ordered along the setting, value of the report's setting)."""
        for i, (nm, pts, base) in enumerate(rows):
            xs = [v for _, v in pts]
            ax.plot([min(xs), max(xs)], [i, i], color=tcol, lw=5.0, solid_capstyle='round', zorder=2)
            for (lb, v) in pts[1:-1]:
                ax.plot([v], [i], 'o', color='white', mec=col, mew=0.7, ms=2.8, zorder=4)
            for (lb, v), end in ((pts[0], 0), (pts[-1], 1)):
                ax.plot([v], [i], 'o', color=col, mec='white', mew=0.5, ms=4.2, zorder=5)
                left = v <= (min(xs) + max(xs)) / 2
                if abs(max(xs) - min(xs)) < 0.04 * (xlim[1] - xlim[0]):
                    left = end == (0 if pts[0][1] <= pts[-1][1] else 1)
                off = 0.022 * (xlim[1] - xlim[0])
                ax.text(v - off if left else v + off, i, '%s: %s' % (lb, fmt % v), fontsize=5.2, ha='right' if left else 'left', va='center', color=INK)
            ax.plot([base, base], [i - 0.30, i + 0.30], color=INK, lw=1.1, zorder=6)
            ax.text(xlim[0], i - 0.44, nm, fontsize=5.6, ha='left', va='center', color=col, fontweight='bold')
        ax.set_xlim(*xlim); ax.set_ylim(len(rows) - 0.45, -0.75); ax.set_yticks([]); ax.grid(axis='y', visible=False); bare(ax)
        ax.set_xlabel(xlabel, labelpad=1.5)

    # ---- (a) Problem 2: requests per 3-s window on the held-out windows
    ax = axi(fig, 0.30, 0.42, 2.78, 1.58)
    base = req('trigger', 'Model', 0.1, 50.0)
    rows = [('position tolerance (trigger rule)', [('80 mm', req('trigger', 'Model', 0.1, 80.0)), ('50 mm', base), ('30 mm', req('trigger', 'Model', 0.1, 30.0))], base),
            ('return delay (tolerance 80 mm)', [('0.1 s', req('trigger', 'Model', 0.1, 80.0)), ('0.3 s', req('trigger', 'Model', 0.3, 80.0))], req('trigger', 'Model', 0.1, 80.0)),
            ('predictor (fixed interval, 50 mm)', [('our model', req('periodic', 'Model', 0.1, 50.0)), ('hold-last', req('periodic', 'Hold', 0.1, 50.0))],
             req('periodic', 'Model', 0.1, 50.0)),
            ('timing rule (model, 50 mm)', [('trigger', base), ('fixed interval', req('periodic', 'Model', 0.1, 50.0))], base)]
    panel(ax, rows, c2, tint(c2, 0.30), (-6, 36), 'requests per 3-s window', '%.0f')
    ax.set_xticks([0, 10, 20, 30])
    tag(fig, 0.06, 2.15, 'a', 'Problem 2: observation cost')

    # ---- (b) Problem 3: objective of the look-ahead policy on the validation environments
    ax = axi(fig, 3.62, 0.42, 2.86, 1.58)
    b0 = float(uv['FROZEN'])
    fl = {round(float(r_.reserve_floor), 2): float(r_.utility) for _, r_ in fr.iterrows()}
    rows = [(r'duration scale $\alpha$', [('1.5', umap(1.5, 0.5)), ('1', umap(1.0, 0.5)), ('0.75', umap(0.75, 0.5))], b0),
            ('lowest reservation quantile', [('0.9', fl[0.9]), ('0.75', fl[0.75]), ('min', fl[0.0]), ('median', fl[0.5]), ('0.25', fl[0.25])], b0),
            (r'duration offset $\beta$', [('1 s', umap(1.0, 1.0)), ('0.5 s', b0), ('0 s', umap(1.0, 0.0))], b0),
            ('paid reviews (prefix belief)', [('one per job', float(uv['ABL-review-once'])), ('none', float(uv['RO-feat-static']))], float(uv['RO-feat-static'])),
            ('duration belief', [('family', b0), ('family + prefix', float(uv['RO-feat-static']))], b0),
            (r'capacity weight $\rho$ in the rule', [('1.0', float(uv['RHO-1.0'])), ('0.1', float(uv['RHO-0.1']))], b0)]
    panel(ax, rows, c3, tint(c3, 0.30), (0.0, 1.12), 'objective $U$', '%.2f')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    tag(fig, 3.38, 2.15, 'b', 'Problem 3: objective of the look-ahead policy')
    fig.text(6.48 / W_FULL, 2.15 / 2.30, 'black tick: setting of the report', fontsize=5.0, color=MUTE, ha='right', va='center')
    save(fig, 'fig12_sensitivity')


FIGS = {'reliability': fig_reliability, 'horizon': fig_horizon, 'policy': fig_policy, 'natural': fig_natural, 'duration': fig_duration,
        'decision': fig_decision, 'comparison': fig_comparison, 'sensitivity': fig_sensitivity}

if __name__ == '__main__':
    which = sys.argv[1:] or list(FIGS)
    for k in which:
        FIGS[k]()
