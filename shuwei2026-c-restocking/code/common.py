# -*- coding: utf-8 -*-
"""Shared paths, constants and helpers for the Problem C pipeline.

Directory layout (all paths are relative to the project root = parent of code/):
    code/      programs                    interim/   harmonised arrays (rebuildable)
    results/   result tables / json        figures/   paper figures
    logs/      run logs and process logs   tasks/     copy of the official task files
Raw data are read, in this order, from (1) data_raw/<Dataset>/ (extracted tables),
(2) <TABLE_DIR>/<Dataset>_tables.zip (table-only archives made by s01; default interim/),
(3) the official archives <RAW_DIR>/<Dataset>.zip.  RAW_DIR / TABLE_DIR / TASK_DIR can be
overridden through environment variables.
"""
import io
import json
import os
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
INTERIM = os.path.join(PROJ, 'interim')
RESULTS = os.path.join(PROJ, 'results')
FIGURES = os.path.join(PROJ, 'figures')
LOGS = os.path.join(PROJ, 'logs')
for _d in (INTERIM, RESULTS, FIGURES, LOGS):
    os.makedirs(_d, exist_ok=True)

_OFFICIAL = os.path.join(os.path.dirname(PROJ), 'Restocking_Problem_and_Data_EN')
RAW_DIR = os.environ.get('RAW_DIR', os.path.join(_OFFICIAL, 'Raw_Data'))
TASK_DIR = os.environ.get('TASK_DIR', os.path.join(PROJ, 'tasks'))
if not os.path.isdir(TASK_DIR):
    TASK_DIR = os.path.join(_OFFICIAL, 'Restocking_Tasks_and_Constraints')

DATASETS = ('EgoDemo', 'EgoPro', 'EgoStandard')
SPLITS = ('train', 'validation', 'test')
FPS = 30.0
DT = 1.0 / FPS

# 21 keypoints, identical order in the three sources (checked in s02)
KP_NAMES = ['Wrist', 'ThumbMCP', 'ThumbPIP', 'ThumbDIP', 'ThumbTip',
            'IndexMCP', 'IndexPIP', 'IndexDIP', 'IndexTip',
            'MiddleMCP', 'MiddlePIP', 'MiddleDIP', 'MiddleTip',
            'RingMCP', 'RingPIP', 'RingDIP', 'RingTip',
            'LittleMCP', 'LittlePIP', 'LittleDIP', 'LittleTip']
# representative hand point: centroid of the wrist and the four finger knuckles (palm centre)
PALM_IDX = [0, 5, 9, 13, 17]
TIP_IDX = [4, 8, 12, 16, 20]
# "bones" used for the rigid-palm consistency check (pairs of keypoint indices)
PALM_BONES = [(0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13), (13, 17), (5, 17)]

FLAG_GOOD, FLAG_OCC, FLAG_EDGE, FLAG_BOTH = 0, 1, 2, 3     # per-hand quality code


# --------------------------------------------------------------------------- raw access
class RawSource(object):
    """Uniform member access for one dataset (directory, table zip or official zip)."""

    def __init__(self, ds):
        self.ds = ds
        self.mode = None
        d = os.path.join(PROJ, 'data_raw', ds)
        z1 = os.path.join(os.environ.get('TABLE_DIR', INTERIM), ds + '_tables.zip')
        z2 = os.path.join(RAW_DIR, ds + '.zip')
        if os.path.isdir(os.path.join(d, ds)):
            self.mode, self.root = 'dir', d
        elif os.path.isfile(z1):
            self.mode, self.zf = 'zip', zipfile.ZipFile(z1)
        elif os.path.isfile(z2):
            self.mode, self.zf = 'zip', zipfile.ZipFile(z2)
        else:
            raise FileNotFoundError('no raw source for %s (looked in %s, %s, %s)' % (ds, d, z1, z2))

    def members(self):
        if self.mode == 'dir':
            out = []
            for r, _, fs in os.walk(self.root):
                for f in fs:
                    out.append(os.path.relpath(os.path.join(r, f), self.root).replace(os.sep, '/'))
            return sorted(out)
        return sorted(n for n in self.zf.namelist() if not n.endswith('/') and not n.lower().endswith('.mp4'))

    def read(self, member):
        if self.mode == 'dir':
            with open(os.path.join(self.root, member), 'rb') as fh:
                return fh.read()
        return self.zf.read(member)

    def open(self, member):
        return io.BytesIO(self.read(member))


# --------------------------------------------------------------------------- small maths
def quat_to_rot(q):
    """Quaternion(s) (w, x, y, z) -> rotation matrix/matrices (..., 3, 3)."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    w, x, y, z = np.moveaxis(q / n, -1, 0)
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def runs_of(mask):
    """Start/stop (exclusive) indices of maximal True runs of a boolean vector."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return np.zeros((0, 2), dtype=int)
    d = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)], axis=1)


def save_json(obj, name):
    path = name if os.path.isabs(name) else os.path.join(RESULTS, name)

    def _c(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=1, default=_c)
    return path


def load_json(name):
    path = name if os.path.isabs(name) else os.path.join(RESULTS, name)
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)
