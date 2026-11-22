# -*- coding: utf-8 -*-
"""Step 02. Harmonise the three raw sources into one frame table.

Outputs
    interim/frames.npz    concatenated per-frame arrays for all 69 records
    interim/records.csv   one row per record (metadata, counts, time checks)
    interim/segments.csv  one row per annotated semantic segment (349 rows)
    results/p0_build_log.json  cross-checks against the official index files
No value is filled, interpolated or corrected here; the only transformations are
unit harmonisation (ns -> s), a common quaternion order (w, x, y, z) and a common
per-hand quality code.
"""
import hashlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATASETS, FLAG_BOTH, FLAG_EDGE, FLAG_GOOD, FLAG_OCC, INTERIM, RawSource, TASK_DIR,
                    save_json)

REASON_CODE = {'occlusion': FLAG_OCC, 'near edge': FLAG_EDGE, 'near_edge': FLAG_EDGE}


def _code(old, reason):
    c = REASON_CODE[reason.strip().lower()]
    return c if old == FLAG_GOOD or old == c else FLAG_BOTH


def read_egodemo(src, sample):
    import pyarrow.parquet as pq
    base = 'EgoDemo/%s/' % sample
    t = pq.read_table(src.open(base + 'data/chunk-000/file-000.parquet')).to_pandas()
    ep = pq.read_table(src.open(base + 'meta/episodes/chunk-000/file-000.parquet')).to_pandas().iloc[0]
    info = json.loads(src.read(base + 'meta/info.json').decode('utf-8'))
    ann = json.loads(src.read(base + 'annotation.json').decode('utf-8'))
    bad = json.loads(src.read(base + 'bad_frame_ratio.json').decode('utf-8'))
    n = len(t)

    def arr(col, shape):
        return np.stack([np.stack(v) if getattr(v, 'dtype', None) == object else np.asarray(v) for v in t[col]]
                        ).reshape((n,) + shape).astype(np.float64)
    rec = {}
    rec['t'] = t['timestamp'].to_numpy(dtype=np.float64)
    rec['frame'] = t['frame_index'].to_numpy(dtype=np.int64)
    rec['L'] = arr('observation.state.hand_left_world', (21, 3))
    rec['R'] = arr('observation.state.hand_right_world', (21, 3))
    rec['Lq'] = arr('observation.state.hand_left_world_rotation', (21, 4))
    rec['Rq'] = arr('observation.state.hand_right_world_rotation', (21, 4))
    rec['head'] = arr('observation.state.head_world', (3,))
    rec['headq'] = arr('observation.state.head_world_rotation', (4,))
    rec['camL'] = arr('observation.state.head_left_camera_position', (3,))
    rec['camLq'] = arr('observation.state.head_left_camera_rotation', (4,))
    rec['camR'] = arr('observation.state.head_right_camera_position', (3,))
    rec['camRq'] = arr('observation.state.head_right_camera_rotation', (4,))
    fl = np.zeros((n, 2), dtype=np.int8)
    for d in bad.get('hand_bad_frame_details', []):
        f = int(d['frame'])
        if d.get('left_bad_reason'):
            fl[f, 0] = _code(fl[f, 0], d['left_bad_reason'])
        if d.get('right_bad_reason'):
            fl[f, 1] = _code(fl[f, 1], d['right_bad_reason'])
    rec['flag'] = fl
    segs = [(float(a['start_time']), float(a['end_time']), a['description']) for a in ann['subtask_annotations']]
    kin = np.asarray(ep['camera_intrinsics/head_left'], dtype=float)
    kin_r = np.asarray(ep['camera_intrinsics/head_right'], dtype=float)
    meta = dict(dataset='EgoDemo', sample=sample, episode_uuid=str(ep['episode_uuid']),
                environment_id=str(ep['environment_id']), scene_id=str(ep['scene_id']),
                operator_id=str(ep['operator_id']), task=str(ep['tasks'][0]),
                episode_description=ann.get('episode_description', ''), fps_meta=float(info['fps']),
                n_video_views=sum(k.startswith('observation.images') for k in info['features']),
                paradigm='', op_height=np.nan, op_arm_span=np.nan,
                fx=kin[0], fy=kin[1], cx=kin[2], cy=kin[3], fx_r=kin_r[0], cx_r=kin_r[2], cy_r=kin_r[3],
                img_w=float(info['features']['observation.images.head_left']['shape'][1]),
                img_h=float(info['features']['observation.images.head_left']['shape'][0]),
                total_frames_meta=int(bad.get('total_frames', n)),
                video_from=float(ep['videos/observation.images.head_left/from_timestamp']),
                video_to=float(ep['videos/observation.images.head_left/to_timestamp']),
                time_base='float32 seconds from record start')
    return rec, meta, segs


def _pose_table(df, k):
    pos = np.stack([df['transforms[%d].pos.%s' % (k, a)].to_numpy(float) for a in 'xyz'], axis=1)
    quat = np.stack([df['transforms[%d].quat.%s' % (k, a)].to_numpy(float) for a in 'wxyz'], axis=1)
    return pos, quat


def read_mcap_csv(src, ds, sample):
    base = '%s/%s/' % (ds, sample)
    rd = lambda name: pd.read_csv(src.open(base + name))
    md = pd.read_csv(src.open(base + 'session_metadata.csv'), dtype=str, keep_default_na=False)
    md = dict(zip(md['field'], md['value']))
    lh, rh = rd('tables/pose_left_hand.csv'), rd('tables/pose_right_hand.csv')
    hd, cl, cr = rd('tables/pose_head.csv'), rd('tables/pose_headcam.csv'), rd('tables/pose_right_eye_cam.csv')
    bf, sg = rd('tables/annotation_bad_frame_pose_hand.csv'), rd('tables/annotation_semantic_segments.csv')
    kl, kr = rd('tables/sensor_camera_head_left_intrinsic.csv'), rd('tables/sensor_camera_head_right_intrinsic.csv')
    el = rd('tables/sensor_camera_head_left_extrinsic.csv')
    vf = rd('videos/head_left_frames.csv')
    n = len(lh)
    tns = lh['_log_time_ns'].to_numpy(np.int64)
    # --- time-alignment checks between message streams of the same record
    same = dict(n_left=n, n_right=len(rh), n_head=len(hd), n_camL=len(cl), n_camR=len(cr), n_flag=len(bf),
                n_video=len(vf))
    dmax = 0
    for other in (rh, hd, cl, cr, bf):
        if len(other) == n:
            dmax = max(dmax, int(np.abs(other['_log_time_ns'].to_numpy(np.int64) - tns).max()))
        else:
            dmax = -1
    vmax = int(np.abs(vf['video_timestamp_ns'].to_numpy(np.int64) - tns).max()) if len(vf) == n else -1
    # extrinsic stream (x, y, z, w order) must reproduce pose_headcam (w, x, y, z order)
    ext_pos = np.stack([el['transforms[0].translation.%s' % a].to_numpy(float) for a in 'xyz'], axis=1)
    ext_q = np.stack([el['transforms[0].rotation.%s' % a].to_numpy(float) for a in 'wxyz'], axis=1)
    rec = {}
    rec['t'] = (tns - tns[0]) / 1e9
    rec['frame'] = lh['_message_index'].to_numpy(np.int64)
    L = np.empty((n, 21, 3)); R = np.empty((n, 21, 3)); Lq = np.empty((n, 21, 4)); Rq = np.empty((n, 21, 4))
    for k in range(21):
        L[:, k], Lq[:, k] = _pose_table(lh, k)
        R[:, k], Rq[:, k] = _pose_table(rh, k)
    rec.update(L=L, R=R, Lq=Lq, Rq=Rq)
    rec['head'], rec['headq'] = _pose_table(hd, 0)
    rec['camL'], rec['camLq'] = _pose_table(cl, 0)
    rec['camR'], rec['camRq'] = _pose_table(cr, 0)
    ext_err = float(max(np.abs(ext_pos - rec['camL']).max(),
                        np.minimum(np.abs(ext_q - rec['camLq']), np.abs(ext_q + rec['camLq'])).max())) if len(el) == n else np.nan
    fl = np.zeros((n, 2), dtype=np.int8)
    pcols = [c for c in bf.columns if c.startswith('problem_type[')]
    unflagged_bad = 0
    for i in np.flatnonzero(bf['is_bad'].to_numpy(bool)):
        got = False
        for c in pcols:
            v = bf[c].iat[i]
            if isinstance(v, str) and ':' in v:
                side, reason = v.split(':', 1)
                j = 0 if side.strip().lower() == 'left' else 1
                fl[i, j] = _code(fl[i, j], reason)
                got = True
        unflagged_bad += (not got)
    rec['flag'] = fl
    segs = [(float(a), float(b), str(c)) for a, b, c in
            zip(sg['segment.start_time'], sg['segment.end_time'], sg['segment.subtask_description'])]
    meta = dict(dataset=ds, sample=sample, episode_uuid=md['task_info.episode_uuid'],
                environment_id=md['task_info.environment_id'], scene_id=md['task_info.scene_id'],
                operator_id=md['operator.operator_id'], task=md['task_info.task_name'],
                episode_description=str(sg['task_description'].iat[0]) if len(sg) else '',
                fps_meta=float(md.get('devices[0].fps', 'nan')), n_video_views=4 if ds == 'EgoPro' else 2,
                paradigm=md.get('operator.collection_method.paradigm', ''),
                op_height=float(md.get('operator.operator_info.height', 'nan') or 'nan'),
                op_arm_span=float(md.get('operator.operator_info.arm_span', 'nan') or 'nan'),
                fx=float(kl['K[0]'].iat[0]), fy=float(kl['K[4]'].iat[0]), cx=float(kl['K[2]'].iat[0]),
                cy=float(kl['K[5]'].iat[0]), fx_r=float(kr['K[0]'].iat[0]), cx_r=float(kr['K[2]'].iat[0]),
                cy_r=float(kr['K[5]'].iat[0]), img_w=float(kl['width'].iat[0]), img_h=float(kl['height'].iat[0]),
                total_frames_meta=n, video_from=0.0, video_to=float(rec['t'][-1]),
                time_base='int64 ns log time; converted to seconds from first message',
                stream_counts=json.dumps(same), stream_max_dt_ns=dmax, video_max_dt_ns=vmax,
                extrinsic_vs_headcam_maxabs=ext_err, bad_without_type=int(unflagged_bad),
                intrinsic_constant=bool((kl['K[0]'].nunique() == 1) and (kl['K[2]'].nunique() == 1)),
                unix_start_ms=md.get('devices[0].unix_start_time', ''))
    return rec, meta, segs


def main():
    split = pd.read_csv(os.path.join(TASK_DIR, 'record_split.csv'), encoding='utf-8-sig')
    index = pd.read_csv(os.path.join(TASK_DIR, 'source_segment_index.csv'), encoding='utf-8-sig')
    recs, metas, seg_rows = [], [], []
    for ds in DATASETS:
        src = RawSource(ds)
        samples = sorted({m.split('/')[1] for m in src.members() if m.startswith(ds + '/')})
        for s in samples:
            rec, meta, segs = read_egodemo(src, s) if ds == 'EgoDemo' else read_mcap_csv(src, ds, s)
            recs.append(rec); metas.append(meta)
            for k, (a, b, desc) in enumerate(segs):
                seg_rows.append(dict(episode_uuid=meta['episode_uuid'], dataset=ds, sample=s, seg=k,
                                     start_s=a, end_s=b, description=desc))
            print('%-11s %-36s n=%5d  T=%7.2fs' % (ds, s, len(rec['t']), rec['t'][-1]), flush=True)
    # ---- order records as in the official split file (stable record ids)
    order = {u: i for i, u in enumerate(split['episode_uuid'])}
    assert len(set(m['episode_uuid'] for m in metas)) == len(metas), 'duplicate episode_uuid'
    idx = sorted(range(len(metas)), key=lambda i: order[metas[i]['episode_uuid']])
    recs = [recs[i] for i in idx]; metas = [metas[i] for i in idx]
    sp = split.set_index('episode_uuid')
    off = 0
    out = {k: [] for k in recs[0]}
    rid = []
    for i, (rec, meta) in enumerate(zip(recs, metas)):
        n = len(rec['t'])
        row = sp.loc[meta['episode_uuid']]
        # the official split file must agree with the metadata embedded in the raw record
        for k in ('dataset', 'environment_id', 'scene_id', 'operator_id'):
            assert str(row[k]) == str(meta[k]), (meta['episode_uuid'], k, row[k], meta[k])
        meta.update(rid=i, split=row['split'], n=n, start=off, stop=off + n, duration_s=float(rec['t'][-1]))
        # content fingerprint for duplicate detection (hands only, rounded to 0.1 mm)
        h = hashlib.sha1(np.round(np.concatenate([rec['L'].ravel(), rec['R'].ravel()]) * 1e4).astype(np.int64).tobytes())
        meta['content_sha1'] = h.hexdigest()
        for k in out:
            out[k].append(rec[k])
        rid.append(np.full(n, i, dtype=np.int16))
        off += n
    arrays = {k: np.concatenate(v, axis=0) for k, v in out.items()}
    arrays['rid'] = np.concatenate(rid)
    for k in ('L', 'R', 'Lq', 'Rq', 'head', 'headq', 'camL', 'camLq', 'camR', 'camRq'):
        arrays[k] = arrays[k].astype(np.float32)
    np.savez_compressed(os.path.join(INTERIM, 'frames.npz'), **arrays)
    rec_df = pd.DataFrame(metas)
    first = ['rid', 'dataset', 'sample', 'episode_uuid', 'split', 'environment_id', 'scene_id', 'operator_id', 'task',
             'n', 'start', 'stop', 'duration_s', 'fps_meta']
    rec_df = rec_df[first + [c for c in rec_df.columns if c not in first]]
    rec_df.to_csv(os.path.join(INTERIM, 'records.csv'), index=False)
    # ---- segments: join with the official index (family, eligibility, unit ids)
    seg = pd.DataFrame(seg_rows)
    seg['duration_s'] = seg['end_s'] - seg['start_s']
    m = seg.merge(index[['episode_uuid', 'source_segment_index', 'unit_id', 'family_id', 'split', 'environment_id',
                         'task', 'start_s', 'end_s', 'duration_s', 'eligible']],
                  left_on=['episode_uuid', 'seg'], right_on=['episode_uuid', 'source_segment_index'], how='outer',
                  suffixes=('', '_idx'), indicator=True)
    chk = dict(n_records=len(metas), n_frames=int(off), n_segments_raw=len(seg), n_segments_index=len(index),
               merge_both=int((m['_merge'] == 'both').sum()),
               max_abs_start_diff=float((m['start_s'] - m['start_s_idx']).abs().max()),
               max_abs_end_diff=float((m['end_s'] - m['end_s_idx']).abs().max()),
               frames_by_dataset=rec_df.groupby('dataset')['n'].sum().to_dict(),
               records_by_dataset=rec_df.groupby('dataset').size().to_dict(),
               environments=int(rec_df['environment_id'].nunique()),
               env_by_split=rec_df.groupby('split')['environment_id'].nunique().to_dict(),
               records_by_split=rec_df.groupby('split').size().to_dict(),
               frames_by_split=rec_df.groupby('split')['n'].sum().to_dict(),
               duplicate_content=int(rec_df['content_sha1'].duplicated().sum()),
               duplicate_uuid=int(rec_df['episode_uuid'].duplicated().sum()))
    rid_of = dict(zip(rec_df['episode_uuid'], rec_df['rid']))
    m['rid'] = m['episode_uuid'].map(rid_of)
    m = m.drop(columns=['_merge', 'source_segment_index', 'start_s_idx', 'end_s_idx', 'duration_s_idx'])
    m = m.sort_values(['rid', 'seg']).reset_index(drop=True)
    m.to_csv(os.path.join(INTERIM, 'segments.csv'), index=False)
    save_json(chk, 'p0_build_log.json')
    print(json.dumps(chk, indent=1))


if __name__ == '__main__':
    main()
