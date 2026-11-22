# -*- coding: utf-8 -*-
"""Derived per-frame quantities shared by Problems 1-3.

State definition (kept unchanged in Problems 1 and 2)
    representative hand point  c_t  = mean of {wrist, index/middle/ring/little MCP}  (palm centre, world frame, m)
    egocentric state           xi_t = R_head(t)^T (c_t - p_head(t))   (head frame: x forward, y left, z up, m)
                               u_t  = d xi / dt  (causal backward difference, m/s)
    The Euclidean error of xi equals the world-frame error of c when the head pose is known.
Quality classes per hand and frame
    Q0 reference   : unflagged, kinematically plausible, outside the guard band
    Q1 flagged     : original quality flag (occlusion / near edge)      -> low-quality analysis only
    Q2 suspect     : unflagged but kinematically implausible (glitch)   -> excluded
    Q3 guard       : within GUARD frames of a Q1/Q2 frame                -> excluded from references
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INTERIM, PALM_IDX, TIP_IDX, quat_to_rot, runs_of

# plausibility thresholds (fixed before any model was fitted; see paper Section 4.2)
V_MAX = 2.5        # m/s   palm-centre speed
A_MAX = 50.0       # m/s^2 palm-centre acceleration (second difference)
VH_MAX = 2.0       # m/s   head speed (world-frame relocalisation jump)
GUARD = 2          # frames on each side of a flagged / suspect frame
MIN_RUN_S = 1.0    # minimum length of a reference interval used for fitting / evaluation

Q_REF, Q_FLAG, Q_SUSPECT, Q_GUARD = 0, 1, 2, 3


class Data(object):
    """Loads the harmonised frame table and derives states, features and quality classes."""

    def __init__(self):
        Z = np.load(os.path.join(INTERIM, 'frames.npz'))
        self.rec = pd.read_csv(os.path.join(INTERIM, 'records.csv'))
        self.seg = pd.read_csv(os.path.join(INTERIM, 'segments.csv'))
        self.t = Z['t']
        self.rid = Z['rid'].astype(int)
        self.flag = Z['flag']
        self.N = len(self.t)
        self.kp = [Z['L'].astype(np.float64), Z['R'].astype(np.float64)]       # (N,21,3) per hand
        self.wq = [Z['Lq'][:, 0].astype(np.float64), Z['Rq'][:, 0].astype(np.float64)]  # wrist quaternion
        self.head = Z['head'].astype(np.float64)
        self.headq = Z['headq'].astype(np.float64)
        self.cam = Z['camL'].astype(np.float64)
        self.camq = Z['camLq'].astype(np.float64)
        self.c = [k[:, PALM_IDX].mean(axis=1) for k in self.kp]               # palm centre (N,3)
        self.start = self.rec['start'].to_numpy()
        self.stop = self.rec['stop'].to_numpy()
        self.split_of_rid = self.rec['split'].to_numpy()
        self.env_of_rid = self.rec['environment_id'].to_numpy()
        self.dt_rec = np.array([np.median(np.diff(self.t[a:b])) for a, b in zip(self.start, self.stop)])
        self.dt = self.dt_rec[self.rid]
        self.ok_rate = np.abs(self.dt_rec * 30.0 - 1.0) < 0.02      # records sampled at (nominally) 30 Hz
        self._kinematics()
        self._ego()
        self._quality()
        self._context()

    # ------------------------------------------------------------------ kinematics
    def _diff(self, x):
        """Causal backward difference within records (first frame copies the second)."""
        v = np.zeros_like(x)
        v[1:] = (x[1:] - x[:-1]) / (self.t[1:] - self.t[:-1])[:, None]
        for a in self.start:
            v[a] = v[a + 1] if a + 1 < self.N else 0.0
        return v

    def _kinematics(self):
        self.v = [self._diff(c) for c in self.c]                 # velocity (N,3)
        self.speed = [np.linalg.norm(v, axis=1) for v in self.v]
        self.vhead = self._diff(self.head)
        self.hspeed = np.linalg.norm(self.vhead, axis=1)
        self.acc = []
        for v in self.v:
            a = self._diff(v)
            for s in self.start:                                   # second difference undefined on first two frames
                a[s:s + 2] = 0.0
            self.acc.append(np.linalg.norm(a, axis=1))

    def _ego(self):
        Rh = quat_to_rot(self.headq)                                 # head-to-world rotation
        self.xi = [np.einsum('nji,nj->ni', Rh, c - self.head) for c in self.c]
        self.u = [self._diff(x) for x in self.xi]
        self.espeed = [np.linalg.norm(u, axis=1) for u in self.u]
        self.eacc = []
        for u in self.u:
            a = self._diff(u)
            for s in self.start:
                a[s:s + 2] = 0.0
            self.eacc.append(np.linalg.norm(a, axis=1))
        # causal speed over a 0.1 s lag (3 frames): input of the phase model
        self.lagspeed = []
        for x in self.xi:
            s = np.zeros(self.N)
            s[3:] = np.linalg.norm(x[3:] - x[:-3], axis=1) / (self.t[3:] - self.t[:-3])
            for a in self.start:
                s[a:a + 3] = s[min(a + 3, self.N - 1)]
            self.lagspeed.append(s)

    # ------------------------------------------------------------------ quality classes
    def _quality(self):
        self.q = []
        for j in range(2):
            q = np.zeros(self.N, dtype=np.int8)
            flagged = self.flag[:, j] > 0
            suspect = (~flagged) & ((self.speed[j] > V_MAX) | (self.acc[j] > A_MAX) | (self.hspeed > VH_MAX) |
                                    (self.espeed[j] > V_MAX) | (self.eacc[j] > A_MAX))
            # a speed spike involves two consecutive frames: mark the previous frame as well
            sus2 = suspect.copy()
            sus2[:-1] |= suspect[1:] & (self.rid[:-1] == self.rid[1:]) & (~flagged[:-1])
            bad = flagged | sus2
            guard = np.zeros(self.N, dtype=bool)
            idx = np.flatnonzero(bad)
            for g in range(1, GUARD + 1):
                for sh in (-g, g):
                    k = idx + sh
                    ok = (k >= 0) & (k < self.N)
                    k, src = k[ok], idx[ok]
                    same = self.rid[k] == self.rid[src]
                    guard[k[same]] = True
            q[guard & ~bad] = Q_GUARD
            q[sus2] = Q_SUSPECT
            q[flagged] = Q_FLAG
            self.q.append(q)
        self.good = [q == Q_REF for q in self.q]

    def reference_runs(self, j, min_s=MIN_RUN_S, only_30hz=False):
        """Maximal runs of reference-grade frames of hand j with duration >= min_s: (rid, a, b) global indices."""
        out = []
        for r, (a, b) in enumerate(zip(self.start, self.stop)):
            if only_30hz and not self.ok_rate[r]:
                continue
            for u, w in runs_of(self.good[j][a:b]):
                if (w - u - 1) * self.dt_rec[r] >= min_s - 1e-9:
                    out.append((r, a + u, a + w))
        return out

    # ------------------------------------------------------------------ context features
    def _context(self):
        Rc = quat_to_rot(self.camq)                                  # camera-to-world rotation
        Rh = quat_to_rot(self.headq)
        fx = self.rec['fx'].to_numpy()[self.rid]; fy = self.rec['fy'].to_numpy()[self.rid]
        cx = self.rec['cx'].to_numpy()[self.rid]; cy = self.rec['cy'].to_numpy()[self.rid]
        W = self.rec['img_w'].to_numpy()[self.rid]; Hh = self.rec['img_h'].to_numpy()[self.rid]
        self.pc, self.uv, self.margin, self.reach, self.rel_head = [], [], [], [], []
        self.tipmargin = []
        for j in range(2):
            pc = np.einsum('nji,nj->ni', Rc, self.c[j] - self.cam)   # camera frame: x right, y down, z forward
            z = np.where(np.abs(pc[:, 2]) < 1e-3, 1e-3, pc[:, 2])
            u = fx * pc[:, 0] / z + cx
            v = fy * pc[:, 1] / z + cy
            # signed normalised distance to the nearest image border (>0 inside, 0 on the border); behind camera -> -1
            m = np.minimum.reduce([u / W, 1 - u / W, v / Hh, 1 - v / Hh]) * 2.0
            m = np.where(pc[:, 2] <= 0.02, -1.0, np.clip(m, -1.0, 1.0))
            self.pc.append(pc); self.uv.append(np.stack([u / W, v / Hh], axis=1)); self.margin.append(m)
            self.reach.append(np.linalg.norm(self.c[j] - self.head, axis=1))
            self.rel_head.append(self.xi[j])
            # margin of the worst keypoint (fingers leave the image before the palm)
            kc = np.einsum('nji,nkj->nki', Rc, self.kp[j] - self.cam[:, None, :])
            kz = np.where(np.abs(kc[..., 2]) < 1e-3, 1e-3, kc[..., 2])
            ku = (fx[:, None] * kc[..., 0] / kz + cx[:, None]) / W[:, None]
            kv = (fy[:, None] * kc[..., 1] / kz + cy[:, None]) / Hh[:, None]
            km = np.minimum.reduce([ku, 1 - ku, kv, 1 - kv]) * 2.0
            km = np.where(kc[..., 2] <= 0.02, -1.0, np.clip(km, -1.0, 1.0))
            self.tipmargin.append(km.min(axis=1))
        self.hand_dist = np.linalg.norm(self.c[0] - self.c[1], axis=1)
        # image-plane distance between the two palms (normalised) and depth ordering (hand-hand occlusion proxy)
        self.uv_dist = np.linalg.norm(self.uv[0] - self.uv[1], axis=1)
        # segment index / normalised time inside the annotated segment for each frame
        self.seg_id = np.full(self.N, -1, dtype=int)
        self.seg_u = np.full(self.N, np.nan)
        for k, s in enumerate(self.seg.itertuples()):
            a, b = self.start[s.rid], self.stop[s.rid]
            tt = self.t[a:b]
            m = (tt >= s.start_s - 1e-9) & (tt < s.end_s - 1e-9)
            self.seg_id[a:b][m] = k
            self.seg_u[a:b][m] = (tt[m] - s.start_s) / max(s.end_s - s.start_s, 1e-9)

    def frame_table(self):
        """Long table (one row per hand-frame) with the main per-frame quantities."""
        rows = []
        for j in range(2):
            rows.append(pd.DataFrame(dict(
                gi=np.arange(self.N), rid=self.rid, hand=j, t=self.t, q=self.q[j], flag=self.flag[:, j],
                speed=self.speed[j], acc=self.acc[j], hspeed=self.hspeed, reach=self.reach[j],
                espeed=self.espeed[j], lagspeed=self.lagspeed[j],
                depth=self.pc[j][:, 2], u=self.uv[j][:, 0], v=self.uv[j][:, 1], margin=self.margin[j],
                kmargin=self.tipmargin[j], hand_dist=self.hand_dist, uv_dist=self.uv_dist,
                other_speed=self.speed[1 - j], other_flag=self.flag[:, 1 - j],
                relx=self.rel_head[j][:, 0], rely=self.rel_head[j][:, 1], relz=self.rel_head[j][:, 2],
                seg_id=self.seg_id, seg_u=self.seg_u)))
        df = pd.concat(rows, ignore_index=True)
        r = self.rec.set_index('rid')
        for col in ('dataset', 'split', 'environment_id', 'task', 'paradigm', 'operator_id'):
            df[col] = r[col].to_numpy()[df['rid'].to_numpy()]
        fam = self.seg['family_id'].to_numpy()
        df['family'] = np.where(df['seg_id'] >= 0, fam[np.clip(df['seg_id'].to_numpy(), 0, None)], '')
        return df


if __name__ == '__main__':
    D = Data()
    for j, h in enumerate('LR'):
        print(h, 'quality class counts (ref, flagged, suspect, guard):', np.bincount(D.q[j], minlength=4).tolist(),
              ' reference runs >=%.1fs: %d' % (MIN_RUN_S, len(D.reference_runs(j))))
