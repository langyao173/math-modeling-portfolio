# -*- coding: utf-8 -*-
"""Schematic (mechanism) figures of the report, drawn with plain Matplotlib from the records and the result files.

    fig01_architecture   overall architecture of the solution (graphical abstract)
    fig03_state_mech     Problem 1: anchor-spring state model in the head frame, motion-phase chain, flag chain, held-out error

    python code/fig_mechanism.py            both figures
    python code/fig_mechanism.py arch       one of them (arch | state)

Every number and every curve in these figures is read from results/ or recomputed from the frozen models; the hand
skeleton is the frame chosen by the rule written in pick_hand_frame(), not by eye.
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LOGS, RESULTS, TASK_DIR, load_json, quat_to_rot
from plotstyle import C, PHASE_COL, P_COL, W_FULL, plt, save, setup
import schem as S
from schem import INK, MUTE
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle, Wedge
from matplotlib.lines import Line2D

setup()
R = lambda f: os.path.join(RESULTS, f)
# task folder resolved in common.py (env TASK_DIR, ./tasks, or the official folder next to the package)
FAM_COL = {'P': C['blue'], 'A': C['orange'], 'L': C['green']}

# ------------------------------------------------------------------------------------------------ shared inputs
from features import Data
D = Data()
ret = pd.read_csv(R('p1_retention.csv'))
audit = load_json('p1_audit.json')
fz1 = load_json('frozen_p1_state.json')
wt = pd.read_csv(R('p2_wait_table.csv'))
adm = pd.read_csv(R('p2_admissible.csv'))
zones = pd.read_csv(os.path.join(TASK_DIR, 'service_zones.csv'))
tasks = pd.read_csv(os.path.join(TASK_DIR, 'tasks.csv'))
vnom = pd.read_csv(R('p3_table_validation_nominal.csv'))
tnom = pd.read_csv(R('p3_table_test_nominal.csv'))
gt = load_json('p3_gantt_trials.json')


def hand_in_head_frame(h, i):
    """21 keypoints of hand h at frame i in the head frame (x forward, y left, z up), metres."""
    Rh = quat_to_rot(D.headq[i:i + 1])[0]
    return (D.kp[h][i] - D.head[i]) @ Rh


def pick_hand_frame():
    """Rule: among reference frames of the right hand in 30-Hz training records, the frame whose hand is most open
    (largest mean fingertip distance from the palm centre) among those within 5 mm of the median reach."""
    h = 1
    ok = (D.q[h] == 0) & D.ok_rate[D.rid] & (D.split_of_rid[D.rid] == 'train')
    idx = np.flatnonzero(ok)[::7]
    reach = np.linalg.norm(D.xi[h][idx], axis=1)
    near = idx[np.abs(reach - np.median(reach)) < 0.005]
    tips = D.kp[h][near][:, [4, 8, 12, 16, 20]]
    spread = np.linalg.norm(tips - D.c[h][near][:, None, :], axis=2).mean(axis=1)
    return h, int(near[np.argmax(spread)])


HAND, IFR = pick_hand_frame()
KP = hand_in_head_frame(HAND, IFR)                     # (21, 3)
PALM = KP[[0, 5, 9, 13, 17]].mean(axis=0)


def draw_hand(ax, x, y, size, ang=30.0, col='#51606F', lw=0.8, dot=0.05, ms=1.2):
    """Pictogram of the recorded hand: its 21 keypoints projected on their own principal plane (so that the open hand is
    seen flat), wrist-to-middle-fingertip length `size` inches, fingers pointing along `ang` degrees, palm centre at (x, y)."""
    q = KP - PALM
    u, sv, vt = np.linalg.svd(q - q.mean(axis=0), full_matrices=False)
    p = q @ vt[:2].T
    d = p[12] - p[0]
    a = np.deg2rad(ang) - np.arctan2(d[1], d[0])
    p = p @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    p = p * size / np.linalg.norm(d) + [x, y]
    S.hand_skeleton(ax, p, col=col, lw=lw, ms=ms)
    ax.add_patch(Circle((x, y), dot, fc='#FFD23F', ec=INK, lw=0.7, zorder=6))


# ================================================================================================ Figure 1: architecture
def fig_architecture():
    W, H = W_FULL, 3.80
    fig, ax = S.canvas(W, H)
    (c1, t1), (c2, t2), (c3, t3), (c0, t0) = P_COL[1], P_COL[2], P_COL[3], P_COL[0]

    # ---------------------------------------------------------------- top band: evidence
    yb, hb = 3.04, 0.70
    S.card(ax, 0.06, yb, W - 0.12, hb, fc='#FAFAFB', ec='#D4D8DD', lw=0.7, r=0.10)
    ax.text(0.17, yb + hb - 0.115, 'EVIDENCE: RAW EGOCENTRIC RECORDS OF HUMAN OPERATION', fontsize=5.4, fontweight='bold', color=MUTE, va='center')
    rec = D.rec
    for nm, y in (('EgoDemo', yb + 0.43), ('EgoPro', yb + 0.275), ('EgoStandard', yb + 0.12)):
        n = int((rec.dataset == nm).sum())
        ax.add_patch(FancyBboxPatch((0.17, y - 0.062), 1.04, 0.124, boxstyle='round,pad=0,rounding_size=0.062', fc='white', ec='#B8BDC4', lw=0.6, zorder=3))
        ax.add_patch(Circle((0.25, y), 0.03, fc=c0, ec='none', zorder=4))
        ax.text(0.325, y - 0.004, nm, fontsize=5.8, va='center', color=INK, zorder=4)
        ax.text(1.165, y - 0.004, '%d records' % n, fontsize=5.0, va='center', ha='right', color=MUTE, zorder=4)
    S.flow(ax, (1.25, yb + 0.275), (1.50, yb + 0.275), c0, lw=1.3, head=6)
    S.card(ax, 1.54, yb + 0.075, 1.14, 0.42, fc='white', ec='#B8BDC4', lw=0.6, r=0.06, shadow=False, z=3)
    ax.text(1.62, yb + 0.40, 'harmonise and audit', fontsize=5.9, fontweight='bold', color=INK, va='center', zorder=4)
    for k, w_ in enumerate(('time', 'frames', 'duplicates', 'flags')):
        xx = 1.635 + (k % 2) * 0.47; yy = yb + 0.27 - (k // 2) * 0.115
        ax.plot([xx, xx + 0.022, xx + 0.06], [yy, yy - 0.024, yy + 0.03], color=c3, lw=1.0, zorder=4, solid_capstyle='round', solid_joinstyle='round')
        ax.text(xx + 0.085, yy, w_, fontsize=5.3, va='center', color=INK, zorder=4)
    S.flow(ax, (2.72, yb + 0.275), (2.97, yb + 0.275), c0, lw=1.3, head=6)
    tot = audit['retention_total']
    x0, wbar = 3.02, 1.66
    ax.text(x0, yb + 0.44, '{:,} hand-frames'.format(tot['hand_frames']).replace(',', ' '), fontsize=5.9, fontweight='bold', color=INK, va='center')
    share = tot['ref'] / tot['hand_frames']
    ax.add_patch(FancyBboxPatch((x0, yb + 0.215), wbar, 0.13, boxstyle='round,pad=0,rounding_size=0.03', fc='#8FAEC7', ec='none', zorder=3))
    ax.add_patch(Rectangle((x0 + wbar * 0.925, yb + 0.215), wbar * 0.03, 0.13, fc='#E8C36A', ec='none', zorder=3.1))
    ax.add_patch(FancyBboxPatch((x0 + wbar * 0.955, yb + 0.215), wbar * 0.045, 0.13, boxstyle='round,pad=0,rounding_size=0.03', fc=C['red'], ec='none', zorder=3.1))
    ax.add_patch(Rectangle((x0 + wbar * 0.955, yb + 0.215), wbar * 0.02, 0.13, fc=C['red'], ec='none', zorder=3.1))
    ax.text(x0 + wbar * 0.46, yb + 0.277, '%.1f %% reference frames' % (100 * share), fontsize=5.6, color='white', fontweight='bold', va='center', ha='center', zorder=4)
    ax.text(x0, yb + 0.11, 'fit and evaluate', fontsize=5.0, color=MUTE, va='center')
    ax.text(x0 + wbar, yb + 0.11, '{:,} flagged: reliability model'.format(tot['flagged']).replace(',', ' '), fontsize=5.0, color=C['red'], va='center', ha='right')
    S.flow(ax, (x0 + wbar + 0.05, yb + 0.275), (x0 + wbar + 0.30, yb + 0.275), c0, lw=1.3, head=6)
    xs = 5.05
    ax.text(xs, yb + 0.44, '24 environments, split once', fontsize=5.9, fontweight='bold', color=INK, va='center')
    xk = []
    for k in range(24):
        col = '#8B939C' if k < 16 else (C['orange'] if k < 20 else C['red'])
        xx = xs + k * 0.055 + (0.035 if k >= 16 else 0) + (0.035 if k >= 20 else 0)
        xk.append(xx)
        ax.add_patch(FancyBboxPatch((xx, yb + 0.228), 0.043, 0.105, boxstyle='round,pad=0,rounding_size=0.012', fc=col, ec='none', zorder=3))
    ax.text((xk[0] + xk[15] + 0.043) / 2, yb + 0.11, '16 to fit', fontsize=5.2, color='#5F6770', va='center', ha='center')
    ax.text((xk[16] + xk[19] + 0.043) / 2 - 0.07, yb + 0.11, '4 to select', fontsize=5.2, color='#B37400', va='center', ha='center')
    ax.text((xk[20] + xk[23] + 0.043) / 2 + 0.05, yb + 0.11, '4 to test', fontsize=5.2, color=C['red'], va='center', ha='center')

    # ---------------------------------------------------------------- three problem cards
    yc, hc, wc = 0.82, 2.06, 2.00
    xc = [0.06, 2.30, 4.54]
    heads = [('1', 'State and reliability', 'Where is the hand?  Can the observation be trusted?', c1, t1),
             ('2', 'Prediction and re-observation', 'How long can we predict?  When to look again?', c2, t2),
             ('3', 'Scheduling under uncertainty', 'Which job next?  How long to reserve?', c3, t3)]
    for x, (num, title, sub, cc, tt) in zip(xc, heads):
        S.card(ax, x, yc, wc, hc, fc='white', ec=cc, lw=1.0, r=0.11)
        S.gradient_band(ax, x + 0.012, yc + hc - 0.47, wc - 0.024, 0.458, tt, 'white', z=2.2, r=0.10)
        S.badge(ax, x + 0.19, yc + hc - 0.19, num, cc)
        ax.text(x + 0.355, yc + hc - 0.178, title, fontsize=7.3, fontweight='bold', color=cc, va='center', zorder=5)
        ax.text(x + 0.10, yc + hc - 0.40, sub, fontsize=5.4, style='italic', color=INK, va='center', zorder=5)
    for x in xc:
        S.flow(ax, (x + wc / 2, yb - 0.02), (x + wc / 2, yc + hc + 0.025), '#B8BDC4', lw=1.2, head=6, z=1.5)
    ym = 1.80
    for k, (cc, lab) in enumerate(((c1, r'$\xi_t,\ k_t,\ s^2$'), (c2, r'$z(x),\ q_p$'))):
        xa, xb_ = xc[k] + wc, xc[k + 1]
        S.flow(ax, (xa - 0.04, ym), (xb_ + 0.045, ym), cc, lw=2.2, head=8.5, z=6)
        S.chip(ax, (xa + xb_) / 2, ym + 0.17, lab, cc, fs=5.6, h=0.165, pad=0.05, z=7, color=cc)
    METH_Y, CHIP_Y = yc + 0.385, yc + 0.20

    # ---- card 1: anchor-spring state in the head frame
    x = xc[0]
    hx, hy = x + 0.30, 1.80
    S.head_top(ax, hx, hy, r=0.13, ang=10, fov=30, fov_r=1.60, fov_col=t1)
    ax.add_patch(Ellipse((x + 1.17, 1.97), 0.98, 0.52, angle=10, fc='none', ec=c1, lw=0.7, ls=(0, (3, 2)), zorder=3))
    am = (x + 0.90, 1.90); pm = (x + 1.40, 2.06)
    S.flow(ax, (hx + 0.17, hy - 0.02), (pm[0] - 0.035, pm[1] - 0.075), c1, lw=0.7, head=4.5, z=3.5, rad=0.30)
    ax.text(x + 0.99, 1.585, r'state $\xi_t = R_h^{\top}(c_t - p_h)$', fontsize=5.2, color=c1, ha='center', va='center')
    ax.add_patch(Circle(am, 0.046, fc='white', ec=INK, lw=0.9, zorder=6))
    ax.add_patch(Circle(am, 0.015, fc=INK, ec='none', zorder=6.1))
    S.spring(ax, am, pm, n=7, amp=0.032, lead=0.16, color=INK, lw=0.9, zorder=5)
    draw_hand(ax, pm[0], pm[1], 0.42, ang=35, col='#5B6978', lw=0.85, dot=0.05, ms=1.4)
    ax.text(am[0] - 0.03, am[1] + 0.115, 'anchor $m_t$', fontsize=5.2, color=INK, ha='center', va='center')
    ax.text(x + 1.52, 1.725, 'working region', fontsize=5.0, color=c1, ha='center', va='center', rotation=10)
    S.wavy(ax, (pm[0] - 0.36, pm[1] + 0.245), (pm[0] - 0.08, pm[1] + 0.07), c2, n=2, amp=0.016, lw=0.8)
    ax.text(pm[0] - 0.40, pm[1] + 0.30, 'phase-driven forcing', fontsize=5.0, color=c2, ha='right', va='center')
    for k in range(4):
        ax.add_patch(Circle((x + 0.22 + k * 0.165, 1.385), 0.05, fc=PHASE_COL[k], ec='white', lw=0.6, zorder=5))
        if k < 3:
            S.flow(ax, (x + 0.275 + k * 0.165, 1.385), (x + 0.33 + k * 0.165, 1.385), MUTE, lw=0.6, head=3.2, z=4)
    ax.text(x + 0.88, 1.385, 'motion phase $k_t$\nhidden Markov chain', fontsize=5.0, color=INK, va='center', linespacing=1.1)
    ax.text(x + 0.10, METH_Y, 'structural state-space model, seven parameters', fontsize=5.1, color=MUTE, va='center')
    xx = x + 0.10
    for tx in ('9 mm at 0.1 s', '84 mm at 3 s', 'flag AUC 0.89'):
        xx += S.chip(ax, xx, CHIP_Y, tx, c1, fs=5.5, ha='left', weight='bold', color=c1, pad=0.05) + 0.045

    # ---- card 2: growth of the calibrated radius and the request cycle (the curve is the real calibrated radius)
    x = xc[1]
    rad = radius_curve(2)
    eps, delta = 50.0, 0.1
    tau = float(wt[(wt.tolerance_mm == 50) & (wt.phase == 2)].tau_star_s.iloc[0])
    xs0, xs1, y0, y1 = x + 0.27, x + wc - 0.20, 1.535, 2.255
    T_SHOW = 0.72
    sx = (xs1 - xs0) / T_SHOW; sy = (y1 - y0) / 60.0
    seg_t = [np.linspace(0, tau, 30)]; seg_r = [np.interp(seg_t[0], RAD_T, rad)]
    marks, t_now = [], tau
    for _ in range(4):
        marks.append((t_now - delta, t_now))
        tt = np.linspace(delta, tau, 20)
        seg_t.append(t_now + tt - delta); seg_r.append(np.interp(tt, RAD_T, rad))
        t_now += tau - delta
    tt_ = np.concatenate(seg_t); rr_ = np.concatenate(seg_r)
    keep = tt_ <= T_SHOW
    X = xs0 + tt_[keep] * sx; Y = y0 + rr_[keep] * sy
    ax.fill_between(X, y0, Y, color=t2, zorder=3, lw=0)
    ax.plot(X, Y, color=c2, lw=1.3, zorder=4, solid_joinstyle='round')
    ax.plot([xs0, xs1 + 0.02], [y0, y0], color=INK, lw=0.7, zorder=4)
    S.flow(ax, (xs1, y0), (xs1 + 0.09, y0), INK, lw=0.7, head=4)
    ax.text(xs1 + 0.10, y0 - 0.075, 'time', fontsize=5.0, color=MUTE, ha='right', va='center')
    ax.plot([xs0, xs1 + 0.05], [y0 + eps * sy] * 2, color=INK, lw=0.7, ls=(0, (4, 2)), zorder=4)
    ax.text(xs0 - 0.035, y0 + eps * sy, r'$\varepsilon$', fontsize=6.5, color=INK, ha='right', va='center')
    ax.text(xs0 + 0.03, y0 + eps * sy + 0.065, 'tolerance', fontsize=5.0, color=MUTE, va='center')
    ax.text(xs1 + 0.05, y0 + eps * sy + 0.065, 'calibrated 90 % radius', fontsize=5.0, color=c2, va='center', ha='right')
    r_floor = float(np.interp(delta, RAD_T, rad))
    first = True
    for q, (tr, tb) in enumerate(marks):
        if tb <= T_SHOW:
            xr, xb_ = xs0 + tr * sx, xs0 + tb * sx
            S.eye(ax, xr, y0 - 0.095, s=0.05, col=c2)
            ax.plot([xr, xr], [y0, y0 + 0.055], color=c2, lw=0.6, zorder=5)
            if first:
                ax.annotate('', xy=(xb_, y0 + 0.12), xytext=(xr, y0 + 0.12), arrowprops=dict(arrowstyle='<->', lw=0.5, color=INK, shrinkA=0, shrinkB=0,
                                                                                     mutation_scale=4), zorder=6)
                ax.text((xr + xb_) / 2, y0 + 0.19, r'$\delta$', fontsize=5.6, color=INK, ha='center', va='center', zorder=6)
                first = False
    xl = xs0 + marks[0][1] * sx
    ax.plot([xl, xs1 + 0.05], [y0 + r_floor * sy] * 2, color=INK, lw=0.5, ls=(0, (1, 2)), zorder=3.5)
    ax.text(xs1 + 0.05, y0 + r_floor * sy - 0.06, 'latency floor', fontsize=5.0, color=INK, ha='right', va='center', zorder=5)
    S.eye(ax, xs0 - 0.01, y0 - 0.095, s=0.05, col=INK)
    yl = y0 - 0.215
    S.eye(ax, x + 0.20, yl, s=0.04, col=INK); ax.text(x + 0.27, yl, 'last view', fontsize=5.0, color=MUTE, va='center')
    S.eye(ax, x + 0.76, yl, s=0.04, col=c2); ax.text(x + 0.83, yl, 'request', fontsize=5.0, color=MUTE, va='center')
    ax.text(x + 1.24, yl, r'$\delta$', fontsize=5.4, color=INK, va='center'); ax.text(x + 1.32, yl, 'return delay', fontsize=5.0, color=MUTE, va='center')
    ax.text(x + 0.10, METH_Y, 'conformal radius, trigger rule, exact dynamic programme', fontsize=5.1, color=MUTE, va='center')
    xx = x + 0.10
    for tx in ('50 mm for 0.20 s', '12 requests instead of 22'):
        xx += S.chip(ax, xx, CHIP_Y, tx, c2, fs=5.5, ha='left', weight='bold', color=c2, pad=0.05) + 0.045

    # ---- card 3: store map with one replayed route, and the reservations of the same replay
    x = xc[2]
    case = 'S2'
    tr = gt['validation_' + case]['trial']
    ev = []
    with gzip.open(os.path.join(LOGS, 'p3_process_log_FROZEN_validation_stage2.jsonl.gz'), 'rt', encoding='utf-8') as fh:
        for line in fh:
            e = json.loads(line)
            if e['case_id'] == case and e['trial'] == tr and abs(e['alpha'] - 1) < 1e-9 and abs(e['beta'] - 0.5) < 1e-9 and e['action'] == 'execute':
                ev.append(e)
    ev.sort(key=lambda e: e['step'])
    tk = tasks[tasks.case_id == case].set_index('job_id')
    zxy = {r.zone_id: (r.x, r.y) for r in zones.itertuples()}
    mx0, my0, msx, msy = x + 0.21, 1.50, 0.335, 0.52
    Pz = lambda z: (mx0 + zxy[z][0] * msx, my0 + zxy[z][1] * msy)
    route = ['Z0'] + [tk.loc[e['job_id']].zone_id for e in ev]
    seen = {}
    for a_, b_ in zip(route[:-1], route[1:]):
        if a_ != b_:
            k = seen.get((a_, b_), 0); seen[(a_, b_)] = k + 1
            S.flow(ax, Pz(a_), Pz(b_), '#6FBFA6', lw=0.9, rad=0.20 + 0.12 * k, head=4.5, z=3, shrink=6.5)
    for z in zxy:
        px, py = Pz(z)
        if z == 'Z0':
            ax.add_patch(Circle((px, py), 0.07, fc=t3, ec=c3, lw=0.9, zorder=5))
            ax.text(px, py - 0.004, 'start', fontsize=4.6, color=c3, ha='center', va='center', zorder=6)
        else:
            S.shelf(ax, px, py, w=0.17, h=0.14, col=c3)
            ax.text(px, py + (0.115 if zxy[z][1] > 0.5 else -0.115), z, fontsize=5.0, color=c3, ha='center', va='center', zorder=6, fontweight='bold')
    ax.text(x + 0.10, 2.30, 'zones and route', fontsize=5.0, color=MUTE, va='center')
    gx0, gx1, gy1 = x + 1.17, x + wc - 0.11, 2.21
    nshow = min(len(ev), 9)
    rowh = 0.092
    sc = (gx1 - gx0) / 100.0
    for i, e in enumerate(ev[:nshow]):
        yy = gy1 - (i + 0.5) * rowh
        fam = tk.loc[e['job_id']].family_id
        end = e['completion'] if not e['censored'] else 100.0
        ax.add_patch(Rectangle((gx0 + e['service_start'] * sc, yy - 0.034), (end - e['service_start']) * sc, 0.068, fc=FAM_COL[fam], ec='none', zorder=4))
        ax.add_patch(Rectangle((gx0 + e['service_start'] * sc, yy - 0.034), e['reserved_s'] * sc, 0.068, fc='none', ec=INK, lw=0.6, zorder=5))
        ax.plot([gx0 + tk.loc[e['job_id']].due_s * sc] * 2, [yy - 0.04, yy + 0.04], color=INK, lw=0.7, zorder=5)
    yax = gy1 - nshow * rowh - 0.015
    ax.plot([gx0, gx1], [yax, yax], color=INK, lw=0.6)
    ax.text(gx0, 2.30, 'reservation and service', fontsize=5.0, color=MUTE, va='center')
    ax.text(gx1, yax - 0.065, '100 s', fontsize=5.0, color=MUTE, ha='right', va='center')
    ax.text(gx0, yax - 0.065, '0', fontsize=5.0, color=MUTE, ha='left', va='center')
    u_la = float(vnom[(vnom.policy == 'FROZEN') & (vnom.case_id != 'ALL')].utility.mean())
    u_ed = float(vnom[(vnom.policy == 'EDD-median') & (vnom.case_id != 'ALL')].utility.mean())
    ot = [float(vnom[(vnom.policy == 'FROZEN') & (vnom.case_id == c_)].weighted_on_time_rate.iloc[0]) for c_ in ('S1', 'S2', 'S3')]
    ax.text(x + 0.10, METH_Y, 'look-ahead (rollout) policy on a priority index', fontsize=5.1, color=MUTE, va='center')
    xx = x + 0.10
    for tx in ('on time %.2f / %.2f / %.2f' % tuple(ot), '$U$ %.2f vs %.2f' % (u_la, u_ed)):
        xx += S.chip(ax, xx, CHIP_Y, tx, c3, fs=5.5, ha='left', weight='bold', color=c3, pad=0.05) + 0.045

    # ---------------------------------------------------------------- bottom ribbon: one currency of uncertainty
    yr, hr = 0.10, 0.52
    S.gradient_band(ax, 0.06, yr, W - 0.12, hr, t1, t2, t3, z=1, r=0.12)
    ax.add_patch(FancyBboxPatch((0.06, yr), W - 0.12, hr, boxstyle='round,pad=0,rounding_size=0.12', fc='none', ec='#D4D8DD', lw=0.7, zorder=1.2))
    ax.text(0.20, yr + hr - 0.125, 'UNCERTAINTY IS CARRIED FORWARD IN THE UNIT OF EACH DECISION', fontsize=5.4, fontweight='bold', color=INK, va='center')
    toks = [(r'predictive variance $s^2(\tau)$', c1), (r'calibrated radius $r_{0.9}(\tau)$ (mm)', c2), ('horizon and waiting time (s)', c2),
            ('requests per window', c2), ('reservation quantile (s)', c3)]
    ws = [S.text_w(ax, tx, 5.4) + 0.12 for tx, _ in toks]
    gap = (W - 0.40 - sum(ws)) / (len(toks) - 1)
    xx = 0.20
    for k, ((tx, cc), w_) in enumerate(zip(toks, ws)):
        S.chip(ax, xx, yr + 0.165, tx, cc, fs=5.4, ha='left', h=0.19, w=w_)
        if k < len(toks) - 1:
            S.flow(ax, (xx + w_ + 0.02, yr + 0.165), (xx + w_ + gap - 0.02, yr + 0.165), INK, lw=0.8, head=4.5, z=5)
        xx += w_ + gap
    save(fig, 'fig01_architecture')


# ================================================================================================ Figure 3: state-model mechanism
def fig_state_mech():
    W, H = W_FULL, 2.86
    fig, ax = S.canvas(W, H)
    (c1, t1), (c2, t2) = P_COL[1], P_COL[2]
    par = pd.read_csv(R('p1_state_params.csv')).set_index('parameter')
    hm = fz1['hmm']; A = np.array(hm['A']); share = np.array(fz1['derived']['stationary_phase_probabilities'])
    spd = 10 ** np.array(hm['mu']) - 0.01
    qc = pd.read_csv(R('p1_quality_coef.csv')).set_index('term')
    p_stay = 1.0 / (1.0 + np.exp(-qc.loc['intercept'].iloc[-1].coef)) if isinstance(qc.loc['intercept'], pd.DataFrame) else None

    # ---------------------------------------------------------------- (a) scene seen from above
    cp = S.card(ax, 0.05, 1.04, 3.96, 1.77, fc='#FBFCFD', ec='#D4D8DD', lw=0.6, r=0.08, shadow=False, z=0.5)
    ax.text(0.13, 2.70, '(a)', fontsize=8, fontweight='bold', va='center'); ax.text(0.36, 2.70, 'state model in the head frame (seen from above)', fontsize=6.2, color=INK, va='center')
    hx, hy = 0.47, 1.72
    wd = S.head_top(ax, hx, hy, r=0.15, ang=6, fov=27, fov_r=3.9, fov_col=t1)
    wd.set_clip_path(cp)
    ax.add_patch(Ellipse((2.22, 1.90), 2.34, 1.18, angle=6, fc='none', ec=c1, lw=0.7, ls=(0, (3, 2)), zorder=3))
    ax.text(3.20, 1.43, 'working region', fontsize=5.4, color=c1, ha='center', va='center', rotation=14)
    o = np.array([0.20, 1.17]); ca, sa = np.cos(np.deg2rad(6)), np.sin(np.deg2rad(6))
    S.flow(ax, tuple(o), tuple(o + 0.30 * np.array([ca, sa])), INK, lw=0.6, head=4); ax.text(*(o + 0.34 * np.array([ca, sa])), '$x$ forward', fontsize=5.0, va='center', color=INK)
    S.flow(ax, tuple(o), tuple(o + 0.24 * np.array([-sa, ca])), INK, lw=0.6, head=4); ax.text(*(o + 0.29 * np.array([-sa, ca]) + [0.03, 0]), '$y$ left', fontsize=5.0, va='center', color=INK)
    am = np.array([1.62, 1.80]); pm = np.array([2.72, 2.16])
    # slow drift of the anchor: a short random-walk trail (seeded, decorative only)
    tt = np.linspace(0, 1, 60)
    ax.plot(am[0] - 0.34 * (1 - tt), am[1] - 0.015 - 0.11 * (1 - tt) + 0.022 * np.sin(9 * tt) * (1 - tt), color=MUTE, lw=0.6, ls=(0, (1, 1.5)), zorder=3)
    u = (pm - am) / np.linalg.norm(pm - am); nrm = np.array([-u[1], u[0]])
    S.spring(ax, am + nrm * 0.075, pm + nrm * 0.075 - u * 0.02, n=9, amp=0.04, lead=0.13, color=INK, lw=0.9, zorder=5)
    S.damper(ax, am - nrm * 0.075, pm - nrm * 0.075 - u * 0.02, wd=0.045, col=INK, lw=0.85, z=5)
    ax.plot(*zip(am + nrm * 0.075, am - nrm * 0.075), color=INK, lw=0.9, zorder=5); ax.plot(*zip(pm + nrm * 0.075 - u * 0.02, pm - nrm * 0.075 - u * 0.02), color=INK, lw=0.9, zorder=5)
    ax.add_patch(Circle(am, 0.055, fc='white', ec=INK, lw=1.0, zorder=6)); ax.add_patch(Circle(am, 0.018, fc=INK, ec='none', zorder=6.1))
    draw_hand(ax, pm[0], pm[1], 0.60, ang=28, col='#5B6978', lw=0.95, dot=0.06, ms=1.7)
    mid = (am + pm) / 2
    ax.text(*(mid + nrm * 0.20), r'stiffness $\kappa$ = %.1f s$^{-2}$' % par.loc['kappa'].estimate, fontsize=5.4, ha='center', va='center', rotation=np.degrees(np.arctan2(u[1], u[0])))
    ax.text(*(mid - nrm * 0.20), r'damping $\theta$ = %.1f s$^{-1}$' % par.loc['theta'].estimate, fontsize=5.4, ha='center', va='center', rotation=np.degrees(np.arctan2(u[1], u[0])))
    ax.text(am[0] - 0.02, am[1] - 0.25, 'anchor $m_t$', fontsize=5.6, ha='center', va='center', fontweight='bold')
    ax.text(am[0] - 0.02, am[1] - 0.36, r'slow drift, $\sigma_m$ = %.1f mm s$^{-1/2}$' % (1e3 * par.loc['sigma_m'].estimate), fontsize=5.0, ha='center', va='center', color=MUTE)
    S.flow(ax, (hx + 0.21, hy + 0.02), tuple(am - np.array([0.075, 0.0])), c1, lw=0.8, head=4.5, z=3.6)
    ax.text((hx + 0.21 + am[0]) / 2, (hy + am[1]) / 2 + 0.085, '$m_t$', fontsize=6.0, color=c1, ha='center', va='center')
    S.flow(ax, tuple(am + u * 0.07), tuple(pm - u * 0.085), c1, lw=0.8, head=4.5, z=5.5)
    td = ax.text(*(am + u * 0.30), '$d_t$', fontsize=6.0, color=c1, ha='center', va='center', zorder=6)
    td.set_bbox(dict(fc='#FBFCFD', ec='none', pad=0.6))
    ax.add_patch(Circle((0.66, 2.575), 0.045, fc='#FFD23F', ec=INK, lw=0.6, zorder=6))
    ax.text(0.76, 2.575, r'palm centre in the head frame:  $\xi_t = m_t + d_t$', fontsize=5.8, color=c1, ha='left', va='center', fontweight='bold')
    # random forcing: one arrow per phase, length proportional to sigma_k
    sig = [par.loc['sigma_%d' % (k + 1)].estimate for k in range(4)]
    base = pm + np.array([-0.02, 0.085])
    for k in range(4):
        a_ = np.deg2rad(140 - 28 * k)
        tip = base + 0.95 * sig[k] * np.array([np.cos(a_), np.sin(a_)])
        S.flow(ax, tuple(base), tuple(tip), PHASE_COL[k], lw=1.3, head=5.5, z=6.5)
        ax.text(*(tip + 0.065 * np.array([np.cos(a_), np.sin(a_)])), '%d' % (k + 1), fontsize=5.0, color=PHASE_COL[k], ha='center', va='center', fontweight='bold')
    ax.text(3.22, 2.60, 'random forcing $\\sigma_k$,\none level per motion phase $k$', fontsize=5.0, color=INK, ha='left', va='center', linespacing=1.15)
    S.chip(ax, 2.72, 1.165, 'spread 40 mm per axis', c1, fs=5.0, h=0.15, color=c1)
    S.chip(ax, 1.62, 1.165, 'relaxation 1.3 and 4.7 s$^{-1}$', c1, fs=5.0, h=0.15, color=c1)

    # ---------------------------------------------------------------- (b) motion-phase chain
    ax.text(0.13, 0.90, '(b)', fontsize=8, fontweight='bold', va='center'); ax.text(0.36, 0.90, 'motion-phase chain (probability per frame)', fontsize=6.2, color=INK, va='center')
    xn = [0.42, 0.95, 1.48, 2.01]; yn = 0.50
    rn = 0.24 * np.sqrt(share)
    for k in range(4):
        ax.add_patch(Circle((xn[k], yn), rn[k], fc=PHASE_COL[k], ec='white', lw=0.8, zorder=5))
        ax.text(xn[k], yn, '%d' % (k + 1), fontsize=6.5, color='white', fontweight='bold', ha='center', va='center', zorder=6)
        ax.text(xn[k], yn - 0.235, '%.2f m/s' % spd[k], fontsize=5.0, ha='center', va='center', color=INK)
        ax.text(xn[k], yn - 0.335, 'stays %.2f s' % ((1 / 30.0) / (1 - A[k, k])), fontsize=5.0, ha='center', va='center', color=MUTE)
        if k < 3:
            S.flow(ax, (xn[k] + rn[k] * 0.8, yn + rn[k] * 0.62), (xn[k + 1] - rn[k + 1] * 0.8, yn + rn[k + 1] * 0.62), INK, lw=0.5 + 9 * A[k, k + 1], rad=-0.35, head=4.5, z=4)
            S.flow(ax, (xn[k + 1] - rn[k + 1] * 0.8, yn - rn[k + 1] * 0.62), (xn[k] + rn[k] * 0.8, yn - rn[k] * 0.62), INK, lw=0.5 + 9 * A[k + 1, k], rad=-0.35, head=4.5, z=4)
            ax.text((xn[k] + xn[k + 1]) / 2, yn + 0.235, '%.2f' % A[k, k + 1], fontsize=5.0, ha='center', va='center', color=MUTE)
    # ---------------------------------------------------------------- (c) reliability chain
    ax.text(2.42, 0.90, '(c)', fontsize=8, fontweight='bold', va='center'); ax.text(2.65, 0.90, 'observation-reliability chain', fontsize=6.2, color=INK, va='center')
    xa, xb_, yq = 2.72, 3.70, 0.50
    ax.add_patch(Circle((xa, yq), 0.15, fc='white', ec=INK, lw=0.9, zorder=5)); ax.text(xa, yq, 'usable', fontsize=5.0, ha='center', va='center', zorder=6)
    ax.add_patch(Circle((xb_, yq), 0.15, fc=C['red'], ec='white', lw=0.9, zorder=5)); ax.text(xb_, yq, 'flagged', fontsize=5.0, ha='center', va='center', color='white', zorder=6, fontweight='bold')
    S.flow(ax, (xa + 0.13, yq + 0.09), (xb_ - 0.13, yq + 0.09), C['red'], lw=1.1, rad=-0.30, head=5, z=4)
    S.flow(ax, (xb_ - 0.13, yq - 0.09), (xa + 0.13, yq - 0.09), INK, lw=0.8, rad=-0.30, head=5, z=4)
    ax.text((xa + xb_) / 2, yq + 0.265, 'onset: logistic hazard', fontsize=5.0, ha='center', va='center', color=C['red'])
    stay = 1.0 / (1.0 + np.exp(-1.442)) if p_stay is None else p_stay
    ax.text((xa + xb_) / 2, yq - 0.265, 'recovery %.2f per frame' % (1 - load_json('frozen_p1_quality.json')['markov_stay']), fontsize=5.0, ha='center', va='center', color=MUTE)
    orr = {r: float(qc.loc[r].odds_ratio) for r in ('recent', 'other_flag', 'lspeed')}
    for i, tx in enumerate(('flag in the last second  x%.1f' % orr['recent'], 'other hand flagged  x%.1f' % orr['other_flag'], 'tenfold speed  x%.1f' % orr['lspeed'])):
        ax.text(xb_ + 0.24, yq + 0.17 - 0.17 * i, tx.replace('x', r'$\times$'), fontsize=5.0, ha='left', va='center', color=INK)
    ax.plot([xb_ + 0.20, xb_ + 0.20], [yq - 0.26, yq + 0.26], color=C['red'], lw=0.8)
    ax.text(xb_ + 0.24, yq + 0.33, 'odds of an onset', fontsize=5.0, ha='left', va='center', color=C['red'])

    # ---------------------------------------------------------------- (d) held-out discrepancy against the horizon
    m = pd.read_csv(R('p1_state_metrics.csv')); m = m[m.split == 'test']
    axd = fig.add_axes([5.02 / W, 0.40 / H, 1.50 / W, 2.10 / H])
    hz = np.sort(m.horizon_s.unique())
    get = lambda nm: np.array([m[(m.model == nm) & np.isclose(m.horizon_s, h)].mean_mm.iloc[0] for h in hz])
    yh, ym_, ycv = get('Hold'), get('MR-4ph'), get('CV')
    axd.fill_between(hz, ym_, yh, color=t1, lw=0, zorder=1)
    axd.plot(hz, ycv, color=C['light'], lw=1.0, marker='^', ms=2.6, label='constant velocity', zorder=2)
    axd.plot(hz, yh, color=C['grey'], lw=1.2, marker='s', ms=2.8, label='hold-last', zorder=3)
    axd.plot(hz, ym_, color=c1, lw=1.6, marker='o', ms=3.2, label='our model', zorder=4)
    for h_, fx, dy, ha_ in ((1 / 30.0, 0.94, 11, 'left'), (0.1, 0.86, 18, 'center'), (1.0, 1.22, -14, 'center'), (3.0, 0.90, -14, 'center')):
        i = int(np.argmin(np.abs(hz - h_)))
        axd.text(hz[i] * fx, ym_[i] + dy, u'\u2212%d %%' % round(100 * (1 - ym_[i] / yh[i])), fontsize=5.4, color=c1, ha=ha_, va='center', fontweight='bold')
    axd.text(3.5, 5, 'shaded: gain over hold-last', fontsize=5.0, color=MUTE, ha='right', va='center')
    axd.set_xscale('log'); axd.set_xticks([1 / 30.0, 0.1, 0.3, 1, 3]); axd.set_xticklabels(['0.03', '0.1', '0.3', '1', '3'])
    axd.minorticks_off(); axd.set_ylim(0, 128); axd.set_xlim(0.027, 3.7)
    axd.set_xlabel('prediction horizon (s)', fontsize=6.8, labelpad=1.5); axd.set_ylabel('discrepancy on test (mm)', fontsize=6.8, labelpad=1.5)
    axd.tick_params(labelsize=6.2)
    axd.legend(loc='upper left', fontsize=5.6, handlelength=1.6, borderaxespad=0.2)
    fig.text(4.42 / W, 2.70 / H, '(d)', fontsize=8, fontweight='bold', va='center'); fig.text(4.65 / W, 2.70 / H, 'error on unseen environments', fontsize=6.2, color=INK, va='center')
    save(fig, 'fig03_state_mech')


# ------------------------------------------------------------------------------------------------ calibrated radius curves
_rc = None
RAD_T = None


def radius_curve(k):
    """Calibrated 90 % radius (mm) of a typical onset in phase k against the information age (s); results/p2_radius_curves.csv."""
    global _rc, RAD_T
    if _rc is None:
        _rc = pd.read_csv(R('p2_radius_curves.csv'))
        _rc['phase'] = _rc.phase.astype(str)
        RAD_T = np.r_[0.0, _rc[_rc.phase == 'all'].horizon_s.to_numpy()]
    d = _rc[_rc.phase == str(k)]
    return np.r_[0.0, d.radius90_mm.to_numpy()]


if __name__ == '__main__':
    which = sys.argv[1:] or ['arch', 'state']
    if 'arch' in which:
        fig_architecture()
    if 'state' in which:
        fig_state_mech()
