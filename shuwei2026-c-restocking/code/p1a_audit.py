# -*- coding: utf-8 -*-
"""Problem 1, part A.  Data audit, retention tables and choice of the representative hand point.

Outputs (results/):
    p1_audit.json              time / coordinate / duplicate / flag checks
    p1_retention.csv           retention by split and dataset (records, frames, quality classes, usable runs)
    p1_record_usage.csv        per-record usage (fit / select / evaluate / low-quality analysis / exclusions)
    p1_rep_point.csv           comparison of candidate representative points
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PALM_IDX, RESULTS, SPLITS, quat_to_rot, runs_of, save_json
from features import A_MAX, GUARD, MIN_RUN_S, Q_FLAG, Q_GUARD, Q_REF, Q_SUSPECT, V_MAX, VH_MAX, Data


def main():
    D = Data()
    rec = D.rec
    audit = {}
    # ------------------------------------------------------------------ time
    dts = D.dt_rec
    gaps = dups = 0
    for a, b, dt in zip(D.start, D.stop, dts):
        d = np.diff(D.t[a:b])
        gaps += int((d > 1.5 * dt).sum()); dups += int((d <= 0).sum())
    seg_end = D.seg.groupby('rid')['end_s'].max().to_numpy()
    tail = rec['duration_s'].to_numpy() - seg_end
    audit['time'] = dict(
        records=len(rec), frames=int(D.N), gaps_gt_1p5dt=gaps, non_increasing_timestamps=dups,
        records_30hz=int(D.ok_rate.sum()), records_other_rate={int(r): round(float(1 / dts[r]), 3) for r in np.flatnonzero(~D.ok_rate)},
        dt_ms_min=float(dts.min() * 1e3), dt_ms_max=float(dts.max() * 1e3),
        within_record_dt_jitter_us_max=float(max(np.ptp(np.diff(D.t[a:b])) for a, b in zip(D.start, D.stop)) * 1e6),
        mcap_stream_max_abs_dt_ns=float(np.nanmax(rec['stream_max_dt_ns'])), video_vs_pose_max_abs_dt_ns=float(np.nanmax(rec['video_max_dt_ns'])),
        segments=len(D.seg), frames_inside_segments=int((D.seg_id >= 0).sum()), unannotated_tail_s_max=float(tail.max()),
        unannotated_tail_s_mean=float(tail.mean()), segment_gaps=0)
    # ------------------------------------------------------------------ coordinates
    Z = np.load(os.path.join(os.path.dirname(RESULTS), 'interim', 'frames.npz'))
    camR = Z['camR'].astype(float); camL = Z['camL'].astype(float)
    base = camR - camL
    Rc = quat_to_rot(D.camq)
    up_cam = Rc[:, 2, :]                                   # world z expressed in camera axes (R^T e_z)
    base_cam = np.einsum('nji,nj->ni', Rc, base)
    hand_sign = np.cross(base, 0.5 * (D.c[0] + D.c[1]) - D.head)[:, 2]
    fr = D.frame_table()
    edge = fr[fr['flag'] == 2]; occ = fr[fr['flag'] == 1]; good = fr[fr['flag'] == 0]
    audit['coordinates'] = dict(
        stereo_baseline_mm_mean=float(np.linalg.norm(base, axis=1).mean() * 1e3),
        stereo_baseline_mm_sd=float(np.linalg.norm(base, axis=1).std() * 1e3),
        baseline_vertical_component_mm_meanabs=float(np.abs(base[:, 2]).mean() * 1e3),
        baseline_in_camera_axes_mm=[float(x) for x in base_cam.mean(axis=0) * 1e3],
        world_up_in_camera_y_mean=float(up_cam[:, 1].mean()), frac_frames_up_maps_to_minus_y=float((up_cam[:, 1] < 0).mean()),
        frac_frames_right_handed=float((hand_sign > 0).mean()),
        hands_in_front_of_camera_frac=float(np.mean([(D.pc[j][:, 2] > 0).mean() for j in range(2)])),
        left_hand_left_of_right_frac=float((D.pc[0][:, 0] < D.pc[1][:, 0]).mean()),
        head_height_range_m=[float(D.head[:, 2].min()), float(D.head[:, 2].max())],
        reach_m_median=float(np.median(np.concatenate(D.reach))),
        palm_bone_cv_max=float(max(np.std(np.linalg.norm(D.kp[j][a:b, 0] - D.kp[j][a:b, 9], axis=1)) /
                                   np.mean(np.linalg.norm(D.kp[j][a:b, 0] - D.kp[j][a:b, 9], axis=1))
                                   for j in range(2) for a, b in zip(D.start, D.stop))),
        edge_margin_median=dict(good=float(good['kmargin'].median()), occlusion=float(occ['kmargin'].median()),
                                near_edge=float(edge['kmargin'].median())),
        frac_any_keypoint_outside_image=dict(good=float((good['kmargin'] < 0).mean()), occlusion=float((occ['kmargin'] < 0).mean()),
                                             near_edge=float((edge['kmargin'] < 0).mean())))
    # ------------------------------------------------------------------ duplicates / completeness / flags
    audit['duplicates'] = dict(duplicate_episode_uuid=int(rec['episode_uuid'].duplicated().sum()),
                               duplicate_content_hash=int(rec['content_sha1'].duplicated().sum()),
                               environments=int(rec['environment_id'].nunique()), scenes=int(rec['scene_id'].nunique()),
                               operators=int(rec['operator_id'].nunique()),
                               records_per_environment_max=int(rec.groupby('environment_id').size().max()),
                               operators_shared_train_test=int(len(set(rec[rec.split == 'train'].operator_id) & set(rec[rec.split == 'test'].operator_id))),
                               operators_shared_train_val=int(len(set(rec[rec.split == 'train'].operator_id) & set(rec[rec.split == 'validation'].operator_id))),
                               scenes_shared_across_splits=int(len(set(rec[rec.split == 'train'].scene_id) & set(rec[rec.split != 'train'].scene_id))))
    audit['completeness'] = dict(nan_values=int(sum(np.isnan(Z[k]).sum() for k in ('L', 'R', 'Lq', 'Rq', 'head', 'headq', 'camL', 'camLq'))),
                                 all_zero_hand_frames=int(sum((np.abs(D.kp[j]).sum(axis=(1, 2)) == 0).sum() for j in range(2))),
                                 exact_repeated_hand_frames=int(sum((np.abs(np.diff(D.c[j], axis=0)).sum(axis=1) == 0).sum() for j in range(2))))
    fl = D.flag
    run_rows = []
    for r, (a, b) in enumerate(zip(D.start, D.stop)):
        for j in range(2):
            for u, w in runs_of(fl[a:b, j] > 0):
                codes = np.unique(fl[a + u:a + w, j])
                run_rows.append(dict(rid=r, hand=j, n=w - u, dur_s=(w - u) * D.dt_rec[r], reason=int(codes[0]) if len(codes) == 1 else 3,
                                     censored=bool(u == 0 or w == b - a)))
    runs = pd.DataFrame(run_rows)
    audit['flags'] = dict(flagged_frames_any_hand=int((fl > 0).any(axis=1).sum()), flagged_hand_frames=int((fl > 0).sum()),
                          hand_frames_occlusion=int((fl == 1).sum()), hand_frames_near_edge=int((fl == 2).sum()),
                          both_hands_flagged_frames=int((fl > 0).all(axis=1).sum()), flagged_runs=len(runs),
                          records_with_flags=int(runs['rid'].nunique()), run_median_s=float(runs['dur_s'].median()),
                          run_mean_s=float(runs['dur_s'].mean()), run_max_s=float(runs['dur_s'].max()),
                          bad_without_reason=int(np.nansum(rec['bad_without_type'])))
    audit['thresholds'] = dict(V_MAX=V_MAX, A_MAX=A_MAX, VH_MAX=VH_MAX, GUARD=GUARD, MIN_RUN_S=MIN_RUN_S)
    # ------------------------------------------------------------------ retention tables
    rows = []
    usage = []
    for r in range(len(rec)):
        a, b = D.start[r], D.stop[r]
        row = dict(rid=r, dataset=rec.dataset[r], split=rec.split[r], environment_id=rec.environment_id[r], frames=b - a,
                   rate_hz=round(1 / D.dt_rec[r], 2))
        for name, code in (('ref', Q_REF), ('flagged', Q_FLAG), ('suspect', Q_SUSPECT), ('guard', Q_GUARD)):
            row[name] = int(sum((D.q[j][a:b] == code).sum() for j in range(2)))
        rr = [(u, w) for j in range(2) for (r2, u, w) in D.reference_runs(j) if r2 == r]
        row['runs_ge1s'] = len(rr); row['run_frames'] = int(sum(w - u for u, w in rr))
        row['short_fragment_frames'] = row['ref'] - row['run_frames']
        role = {'train': 'fit', 'validation': 'select', 'test': 'evaluate'}[rec.split[r]]
        row['dynamics_role'] = role if D.ok_rate[r] else 'excluded (25 Hz)'
        row['quality_role'] = role
        row['low_quality_analysis'] = bool(row['flagged'] > 0)
        usage.append(row)
    usage = pd.DataFrame(usage)
    usage.to_csv(os.path.join(RESULTS, 'p1_record_usage.csv'), index=False)
    g = usage.groupby(['split', 'dataset'])
    ret = g.agg(records=('rid', 'size'), environments=('environment_id', 'nunique'), frames=('frames', 'sum'), ref=('ref', 'sum'),
                flagged=('flagged', 'sum'), suspect=('suspect', 'sum'), guard=('guard', 'sum'), runs_ge1s=('runs_ge1s', 'sum'),
                run_frames=('run_frames', 'sum'), short_fragment_frames=('short_fragment_frames', 'sum')).reset_index()
    dyn = usage[usage.dynamics_role.isin(['fit', 'select', 'evaluate'])].groupby(['split', 'dataset'])['run_frames'].sum().rename('dyn_frames').reset_index()
    ret = ret.merge(dyn, how='left').fillna({'dyn_frames': 0})
    tot = usage.groupby('split').agg(records=('rid', 'size'), environments=('environment_id', 'nunique'), frames=('frames', 'sum'), ref=('ref', 'sum'),
                                     flagged=('flagged', 'sum'), suspect=('suspect', 'sum'), guard=('guard', 'sum'), runs_ge1s=('runs_ge1s', 'sum'),
                                     run_frames=('run_frames', 'sum'), short_fragment_frames=('short_fragment_frames', 'sum')).reset_index()
    tot['dataset'] = 'ALL'
    tot['dyn_frames'] = [usage[(usage.split == s) & usage.dynamics_role.isin(['fit', 'select', 'evaluate'])]['run_frames'].sum() for s in tot['split']]
    ret = pd.concat([ret, tot], ignore_index=True)
    ret['hand_frames'] = 2 * ret['frames']
    ret['ref_pct'] = 100 * ret['ref'] / ret['hand_frames']
    ret['run_pct'] = 100 * ret['run_frames'] / ret['hand_frames']
    ret.to_csv(os.path.join(RESULTS, 'p1_retention.csv'), index=False)
    audit['retention_total'] = dict(hand_frames=int(2 * D.N), ref=int(usage.ref.sum()), flagged=int(usage.flagged.sum()),
                                    suspect=int(usage.suspect.sum()), guard=int(usage.guard.sum()),
                                    run_frames=int(usage.run_frames.sum()), runs=int(usage.runs_ge1s.sum()),
                                    short_fragment_frames=int(usage.short_fragment_frames.sum()),
                                    dyn_frames=int(usage[usage.dynamics_role != 'excluded (25 Hz)'].run_frames.sum()))
    # ------------------------------------------------------------------ representative point
    Rh = quat_to_rot(D.headq)
    cands = {'wrist': [0], 'palm centre (wrist + 4 MCP)': PALM_IDX, 'centroid of 21 keypoints': list(range(21)),
             'index fingertip': [8], 'thumb-index pinch midpoint': [4, 8]}
    rep = []
    for name, idx in cands.items():
        jit, e1, e15, cv1, sdw, ratio = [], [], [], [], [], []
        for j in range(2):
            p = np.einsum('nji,nj->ni', Rh, D.kp[j][:, idx].mean(axis=1) - D.head)       # egocentric candidate point
            spd = np.zeros(D.N); spd[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.diff(D.t)
            fl_j = (D.q[j] == Q_FLAG); rf_j = (D.q[j] == Q_REF)
            ratio.append(np.median(spd[fl_j]) / np.median(spd[rf_j]))
            for (r, a, b) in D.reference_runs(j, only_30hz=True):
                if D.split_of_rid[r] != 'train':
                    continue
                x = p[a:b]
                sm = savgol_filter(x, 7, 2, axis=0)
                jit.append(np.linalg.norm(x - sm, axis=1)[3:-3])
                e1.append(np.linalg.norm(x[1:] - x[:-1], axis=1))
                cv1.append(np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=1))
                e15.append(np.linalg.norm(x[15:] - x[:-15], axis=1))
                sdw.append(x - x.mean(axis=0))
        rep.append(dict(point=name, n_keypoints=len(idx), jitter_rms_mm=1e3 * np.sqrt(np.mean(np.concatenate(jit) ** 2)),
                        hold_1f_mm=1e3 * np.concatenate(e1).mean(), cv_1f_mm=1e3 * np.concatenate(cv1).mean(),
                        hold_0p5s_mm=1e3 * np.concatenate(e15).mean(),
                        within_run_sd_mm=1e3 * np.sqrt((np.concatenate(sdw) ** 2).sum(axis=1).mean()),
                        flagged_to_ref_speed_ratio=float(np.mean(ratio))))
    rep = pd.DataFrame(rep)
    rep.to_csv(os.path.join(RESULTS, 'p1_rep_point.csv'), index=False)
    # world-frame versus egocentric hold error of the palm centre (train reference runs)
    hz = [3, 15, 30, 60, 90]
    cmp_rows = []
    for split in SPLITS:
        for h in hz:
            ew, ee = [], []
            for j in range(2):
                for (r, a, b) in D.reference_runs(j, only_30hz=True):
                    if D.split_of_rid[r] != split or b - a <= h + 1:
                        continue
                    ew.append(np.linalg.norm(D.c[j][a + h:b] - D.c[j][a:b - h], axis=1))
                    ee.append(np.linalg.norm(D.xi[j][a + h:b] - D.xi[j][a:b - h], axis=1))
            cmp_rows.append(dict(split=split, horizon_s=h / 30.0, world_hold_mm=1e3 * np.concatenate(ew).mean(),
                                 ego_hold_mm=1e3 * np.concatenate(ee).mean()))
    pd.DataFrame(cmp_rows).to_csv(os.path.join(RESULTS, 'p1_world_vs_ego.csv'), index=False)
    # size of the hand in the records (all frames, both hands): used to express the position tolerances of Problem 2
    # in units of the hand itself (keypoints 5 and 17: index and little MCP; 0: wrist; 9-12: middle finger)
    span, palm, length = [], [], []
    for k in D.kp:
        span.append(np.linalg.norm(k[:, 5] - k[:, 17], axis=1)); palm.append(np.linalg.norm(k[:, 0] - k[:, 9], axis=1))
        length.append(palm[-1] + sum(np.linalg.norm(k[:, i] - k[:, i + 1], axis=1) for i in (9, 10, 11)))
    audit['hand_scale_mm'] = dict(knuckle_span_index_little_median=float(1e3 * np.median(np.concatenate(span))),
                                  palm_length_wrist_middle_mcp_median=float(1e3 * np.median(np.concatenate(palm))),
                                  hand_length_wrist_middle_tip_median=float(1e3 * np.median(np.concatenate(length))))
    save_json(audit, 'p1_audit.json')
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(ret.round(2).to_string()); print(rep.round(3).to_string()); print(pd.DataFrame(cmp_rows).round(1).to_string())
    import json
    print(json.dumps(audit, indent=1, default=float))


if __name__ == '__main__':
    main()
