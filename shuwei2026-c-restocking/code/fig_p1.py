# -*- coding: utf-8 -*-
"""Figure of the state definition (Problem 1): why the head frame, motion phase, geometry of the quality flags and the
representative point.  Reads results/p1_*.csv|json and the frame table."""
# 1. imports, paths, style ---------------------------------------------------------------------------------
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, load_json
import plotstyle as ps
from plotstyle import C, PHASE_COL, W_FULL, label, plt, save

ps.setup()
plt.rcParams.update({'axes.labelsize': 7.2, 'xtick.labelsize': 6.4, 'ytick.labelsize': 6.4})
R = lambda f: os.path.join(RESULTS, f)

# 2. read data --------------------------------------------------------------------------------------------
from features import Q_FLAG, Q_REF, Data
from statemodel import load_frozen_state, phase_feature

D = Data()
fz, hmm, par = load_frozen_state()
rep = pd.read_csv(R('p1_rep_point.csv'))
wve = pd.read_csv(R('p1_world_vs_ego.csv'))

# =============================================================================================== Figure
# State definition: (a-c) why the head frame, (d) motion phase, (e) geometry of the quality flags, (f) representative point
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from common import quat_to_rot, runs_of

fig = plt.figure(figsize=(W_FULL, 3.4))
gs = fig.add_gridspec(2, 3, width_ratios=[1.12, 1.0, 1.0], height_ratios=[1.0, 0.92])

# record with the largest head travel among the fully usable 30-Hz records (chosen by rule, not by eye)
best, R0 = -1.0, None
for r in np.flatnonzero(D.ok_rate):
    a, b = int(D.start[r]), int(D.stop[r])
    if not (35.0 < D.t[b - 1] - D.t[a] < 95.0) or ((D.q[0][a:b] != Q_REF) | (D.q[1][a:b] != Q_REF)).mean() > 0.02:
        continue
    ext = float(np.ptp(D.head[a:b, :2], axis=0).max())
    if ext > best:
        best, R0 = ext, int(r)
a, b = int(D.start[R0]), int(D.stop[R0]); J0 = 1
cw = D.c[J0][a:b]; hw = D.head[a:b]; xe = D.xi[J0][a:b]
o = hw[:, :2].mean(axis=0)

# (a) one record seen from above: world frame, with the head-frame path at the same scale
ax = fig.add_subplot(gs[0, 0])
ax.plot(cw[:, 0] - o[0], cw[:, 1] - o[1], '-', color=C['grey'], lw=0.6, alpha=0.9)
ax.plot(hw[:, 0] - o[0], hw[:, 1] - o[1], '-', color=C['black'], lw=0.9)
xl = np.array([min(hw[:, 0].min(), cw[:, 0].min()), max(hw[:, 0].max(), cw[:, 0].max())]) - o[0]
yl = np.array([min(hw[:, 1].min(), cw[:, 1].min()), max(hw[:, 1].max(), cw[:, 1].max())]) - o[1]
# head-frame path of the same palm (forward, left), drawn at the same scale in a corner
ox, oy = xl[1] + 0.20, yl[0] + 0.05
ax.plot(ox + (xe[:, 0] - xe[:, 0].min()), oy + (xe[:, 1] - xe[:, 1].min()), '-', color=C['blue'], lw=0.6)
ax.text(ox + 0.5 * np.ptp(xe[:, 0]), oy + np.ptp(xe[:, 1]) + 0.06, 'same palm,\nhead frame', color=C['blue'], fontsize=6.6, ha='center', va='bottom',
        linespacing=0.95)
ax.text(0.03, 0.80, 'head path', transform=ax.transAxes, color=C['black'], fontsize=6.6, ha='left', va='center')
ax.text(0.33, 0.085, 'right palm, world frame', transform=ax.transAxes, color='#666666', fontsize=6.6, ha='left', va='center')
ax.set_title('one record, seen from above', fontsize=6.8, fontstyle='italic', pad=3)
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlim(xl[0] - 0.08, ox + np.ptp(xe[:, 0]) + 0.10); ax.set_ylim(yl[0] - 0.30, yl[1] + 0.12)
ax.plot([xl[0], xl[0] + 0.5], [yl[0] - 0.22, yl[0] - 0.22], '-', color=C['black'], lw=1.4)
ax.text(xl[0] + 0.25, yl[0] - 0.19, '0.5 m', ha='center', va='bottom', fontsize=6.5)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
label(ax, 'a', -0.06, 1.06)

# (b) the same path in the head frame, coloured by the filtered phase
ax = fig.add_subplot(gs[0, 1])
ph = np.full(b - a, -1)
for (rr, u, w) in D.reference_runs(J0, 0.2, only_30hz=True):
    if rr == R0:
        ph[u - a:w - a] = hmm.filter(phase_feature(D, J0, u, w)).argmax(axis=1)
X, Y = -xe[:, 1], xe[:, 0]                                   # right, forward
seg = np.stack([np.c_[X[:-1], Y[:-1]], np.c_[X[1:], Y[1:]]], axis=1)
cols = [PHASE_COL[k] if k >= 0 else C['light'] for k in ph[1:]]
ax.add_collection(LineCollection(seg, colors=cols, linewidths=0.6, alpha=0.9))
ax.plot([0], [0], marker='^', ms=6, color=C['black'], ls='', clip_on=False, zorder=6)
ax.text(0.045, 0.012, 'head', ha='left', va='center', fontsize=6.6)
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlim(min(X.min(), -0.12) - 0.04, max(X.max(), 0.12) + 0.04); ax.set_ylim(-0.03, Y.max() + 0.05)
ax.set_xlabel('to the right (m)'); ax.set_ylabel('forward (m)')
ax.xaxis.set_major_locator(plt.MultipleLocator(0.2)); ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.legend(handles=[Line2D([], [], color=PHASE_COL[k], lw=1.6, label='phase %d' % (k + 1)) for k in range(4)], loc='lower center',
          bbox_to_anchor=(0.5, 0.97), ncol=4, fontsize=6.4, handlelength=1.0, columnspacing=0.7, handletextpad=0.3, borderaxespad=0.1)
label(ax, 'b', -0.30, 1.06)

# (c) hold-last discrepancy in the two frames
ax = fig.add_subplot(gs[0, 2])
for split, ls in (('train', '-'), ('test', '--')):
    d = wve[wve.split == split]
    ax.plot(d.horizon_s, d.world_hold_mm, ls, color=C['grey'], marker='o')
    ax.plot(d.horizon_s, d.ego_hold_mm, ls, color=C['blue'], marker='s')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('horizon (s)'); ax.set_ylabel('hold-last discrepancy (mm)')
ax.set_xticks([0.1, 0.5, 1, 3]); ax.set_xticklabels(['0.1', '0.5', '1', '3'])
ax.set_yticks([10, 30, 100, 300, 1000]); ax.set_yticklabels(['10', '30', '100', '300', '1000']); ax.minorticks_off(); ax.set_ylim(8, 1500)
ax.text(0.34, 300, 'world frame', color=C['grey'], fontsize=7, rotation=33, ha='center', va='center')
ax.text(1.0, 32, 'head frame', color=C['blue'], fontsize=7, ha='center', va='center')
ax.legend(handles=[Line2D([], [], color=C['black'], ls='-', lw=1.0, label='training'), Line2D([], [], color=C['black'], ls='--', lw=1.0, label='test')],
          loc='lower center', bbox_to_anchor=(0.5, 0.97), fontsize=6.4, handlelength=1.8, ncol=2, columnspacing=0.9, borderaxespad=0.1)
label(ax, 'c', -0.34, 1.06)

# (d) phase along one record: head-frame speed over 0.1 s, filtered phase, flagged run
ax = fig.add_subplot(gs[1, 0])
r = 39; j = 1
a, b = D.start[r], D.stop[r]
t = D.t[a:b]; sel = (t >= 20) & (t <= 32)
sp = D.lagspeed[j][a:b]
ph = np.full(b - a, -1)
for (rr, u, w) in D.reference_runs(j, 0.2, only_30hz=True):
    if rr == r:
        ph[u - a:w - a] = hmm.filter(phase_feature(D, j, u, w)).argmax(axis=1)
idx = np.flatnonzero(sel)
seg = np.stack([np.c_[t[idx[:-1]], sp[idx[:-1]]], np.c_[t[idx[1:]], sp[idx[1:]]]], axis=1)
cols = [PHASE_COL[k] if k >= 0 else C['light'] for k in ph[idx[1:]]]
ax.add_collection(LineCollection(seg, colors=cols, linewidths=1.0))
flg = D.flag[a:b, j] > 0
for u, w in runs_of(flg & sel):
    ax.axvspan(t[u] - 0.017, t[w - 1] + 0.017, color=C['red'], alpha=0.28, lw=0)
    ax.text(t[u] - 0.25, 1.75, 'flagged run', fontsize=6.4, color=C['red'], ha='right', va='center')
ax.set_yscale('log'); ax.set_ylim(0.015, 2.6); ax.set_xlim(20, 32)
ax.set_yticks([0.03, 0.1, 0.3, 1.0]); ax.set_yticklabels(['0.03', '0.1', '0.3', '1']); ax.minorticks_off()
ax.set_xlabel('record time (s)'); ax.set_ylabel('head-frame speed (m/s)')
ax.set_xticks([20, 24, 28, 32])
label(ax, 'd', -0.30, 1.06)

# (e) palm centre in the image of the left head camera: unflagged density and flagged frames by reason
ax = fig.add_subplot(gs[1, 1])
uv = np.vstack(D.uv); fl = np.concatenate([D.flag[:, 0], D.flag[:, 1]])
g = fl == 0
ax.hexbin(uv[g, 0], uv[g, 1], gridsize=26, extent=(-0.25, 1.25, -0.25, 1.25), cmap='Greys', mincnt=3, linewidths=0, vmax=None, bins='log', alpha=0.75,
          rasterized=True)
ax.scatter(uv[fl == 1, 0], uv[fl == 1, 1], s=2.0, c=C['blue'], linewidths=0, label='occlusion', rasterized=True, alpha=0.8)
ax.scatter(uv[fl == 2, 0], uv[fl == 2, 1], s=9, c=C['red'], marker='x', linewidths=0.8, label='near edge')
ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=C['black'], lw=0.9))
ax.set_xlim(-0.25, 1.25); ax.set_ylim(1.25, -0.25); ax.set_aspect('equal', adjustable='box'); ax.grid(False)
ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
ax.set_xlabel('image width'); ax.set_ylabel('image height')
ax.text(0.5, -0.045, 'image border', ha='center', va='bottom', fontsize=6.2)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.985), ncol=2, columnspacing=0.8, handletextpad=0.1, fontsize=6.4, markerscale=1.6, borderaxespad=0.1)
label(ax, 'e', -0.52, 1.06)

# (f) candidate representative points: stability against the number of keypoints averaged
ax = fig.add_subplot(gs[1, 2])
names = {'wrist': 'wrist', 'palm centre (wrist + 4 MCP)': 'palm centre', 'centroid of 21 keypoints': '21-point centroid',
         'index fingertip': 'index fingertip', 'thumb-index pinch midpoint': 'pinch midpoint'}
off = {'wrist': (0.006, 0.10), 'palm centre (wrist + 4 MCP)': (0.008, -0.02), 'centroid of 21 keypoints': (0.008, -0.04),
       'index fingertip': (-0.008, -0.03), 'thumb-index pinch midpoint': (-0.008, 0.06)}
for _, q in rep.iterrows():
    main = q['point'].startswith('palm')
    ax.scatter([q.jitter_rms_mm], [q.hold_1f_mm], s=14 + 2.2 * q.n_keypoints, color=C['red'] if main else C['grey'], edgecolor='white', linewidth=0.5,
               zorder=3)
    dx, dy = off[q['point']]
    ax.text(q.jitter_rms_mm + dx, q.hold_1f_mm + dy, names[q['point']], fontsize=6.6, ha='left' if dx > 0 else 'right', va='center',
            color=C['red'] if main else C['black'], fontweight='bold' if main else 'normal')
ax.set_xlim(0.79, 0.985); ax.set_ylim(4.05, 6.2)
ax.set_xlabel('high-frequency jitter (mm)'); ax.set_ylabel('one-frame change (mm)')
ax.text(0.97, 0.05, 'marker area: number\nof keypoints averaged', transform=ax.transAxes, fontsize=6.0, ha='right', va='bottom', color=C['grey'])
label(ax, 'f', -0.34, 1.06)
save(fig, 'fig02_data_state')
