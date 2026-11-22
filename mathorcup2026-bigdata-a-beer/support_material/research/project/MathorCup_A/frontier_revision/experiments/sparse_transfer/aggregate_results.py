"""Audit persisted witnesses, combine compatible bounds, and assemble results.

This operation changes no optimizer or frozen scientific model. A lower bound
is used only for its recorded policy, distance radius, and classifier margin.
Every upper bound comes from a decoded chemical row rechecked against the full
sensory ensemble. Strict-policy witnesses may also enter the relaxed-policy
pool, but the converse is forbidden.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import argparse
import json
import math
import shutil
import numpy as np
import pandas as pd
from control_problem import FixedProblem, AW, file_sha
from conditional_model import DERIVED, TECHNICAL
from transfer_profiles import DEFECTS

HERE = Path(__file__).resolve().parent
OUT = HERE / 'final_summary'
RHOS = [.001, .1, .25, .4]
ATOL = 1e-8


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def truth(x):
    if x.dtype == bool:
        return x
    return x.astype(str).str.lower().eq('true')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def collect_policy(policy, require_perspective=True):
    problem = FixedProblem(policy, style_epsilon=1e-5)
    frames, expected, files, bounds = [], [], [], []
    for beer in sorted(problem.contexts):
        base = HERE / policy / f'beer_{beer}'
        if not (base / 'DONE.json').exists():
            raise RuntimeError(f'Primary source unfinished: {base}')
        folders = [(base, 'joint_sparse')]
        perspective = HERE / 'perspective' / policy / f'beer_{beer}'
        if require_perspective and not (perspective / 'DONE.json').exists():
            raise RuntimeError(f'Perspective source unfinished: {perspective}')
        if (perspective / 'DONE.json').exists():
            folders.append((perspective, 'perspective_lambda_min'))
        # Optional stronger diagonal split: only completed, auditable cases.
        for other in sorted(HERE.glob(f'perspective_sdp*/{policy}/beer_{beer}')):
            if (other / 'DONE.json').exists():
                folders.append((other, 'perspective_sdp'))
        for folder, family in folders:
            pf, cf = folder / 'profiles.csv', folder / 'checks.csv'
            frame, checked = read(pf), read(cf)
            assert len(frame) == len(checked)
            assert np.array_equal(frame.beer, checked.beer)
            assert np.array_equal(frame.candidate_id, checked.candidate_id)
            uid = [f'{policy}:{beer}:{family}:{int(c)}' for c in frame.candidate_id]
            frame['candidate_uid'] = uid
            frame['search_policy'] = policy
            frame['search_family'] = family
            frame['source_file'] = str(pf.relative_to(HERE))
            checked['candidate_uid'] = uid
            frames.append(frame)
            expected.append(checked)
            files.extend([dict(path=str(x.relative_to(HERE)), sha256=file_sha(x)) for x in [pf, cf]])
            bound = json.loads((folder / 'solver_bound.json').read_text())
            assert bound['policy'] == policy and int(bound['beer']) == beer
            assert abs(float(bound['rho']) - .1) < 1e-14
            assert abs(float(bound['style_epsilon']) - 1e-5) < 1e-14
            assert float(bound['margin']) == 0
            bound.update(bound_source=str((folder / 'solver_bound.json').relative_to(HERE)), family=family)
            bounds.append(bound)
    frame = pd.concat(frames, ignore_index=True)
    stored = pd.concat(expected, ignore_index=True)
    actual, sensory = problem.check_rows(frame.to_dict('records'))
    comparable = ['target_distance_reduction', 'quadratic_objective', 'ensemble_stale_delta',
                  'linear_stale_delta', 'min_style_margin', 'chemical_violation', 'box_violation']
    errors = {col: float(np.max(np.abs(stored[col] - actual[col]))) for col in comparable}
    assert max(errors.values()) < 2e-9, errors
    assert np.array_equal(truth(stored.accepted), actual.accepted)
    assert np.array_equal(stored.raw_fields_changed, actual.raw_fields_changed)
    actual.insert(0, 'candidate_uid', frame.candidate_uid)
    actual['search_policy'] = policy
    actual['search_family'] = frame.search_family
    actual['source_file'] = frame.source_file
    sensory.insert(0, 'candidate_uid', frame.candidate_uid)
    return problem, frame, actual, sensory, bounds, files, errors


def canonical_rank_artifacts(problem, policy):
    """Preserve optimizer proposals and add exactly decoded rank witnesses.

    Early strict runs retained pre-polish solver proposals in rank_candidates.
    They are useful optimizer records, but cannot stand in for the chemical
    row that was actually checked. A new unambiguous file is written, and the
    old file remains unchanged with its original checksum.
    """
    rows = []
    for beer in sorted(problem.contexts):
        folder = HERE / policy / f'beer_{beer}'
        path = folder / 'rank_candidates.npz'
        frame = read(folder / 'profiles.csv')
        old = np.load(path, allow_pickle=False)
        vals = frame[problem.features].to_numpy(float)
        ctx = problem.contexts[beer]
        decoded = problem.control.map_.transform(vals) - ctx['z0']
        same = (vals == ctx['x0']) | (np.isnan(vals) & np.isnan(ctx['x0']))
        proposed = old['delta']
        discrepancy = np.max(np.abs(decoded - proposed), axis=1)
        np.savez_compressed(folder / 'decoded_rank_witnesses.npz', delta=decoded,
                            actual_raw_support=~same,
                            proposed_support=old['support'],
                            candidate_id=frame.candidate_id.to_numpy(int),
                            beer=np.full(len(frame), beer),
                            features=np.asarray(problem.features))
        rows.append(dict(policy=policy, beer=beer,
            original_proposal_file=str(path.relative_to(HERE)), original_sha256=file_sha(path),
            decoded_file=str((folder / 'decoded_rank_witnesses.npz').relative_to(HERE)),
            decoded_sha256=file_sha(folder / 'decoded_rank_witnesses.npz'),
            max_proposal_vs_decoded=float(discrepancy.max()),
            rows_differing_above_1e_8=int(np.sum(discrepancy > 1e-8)),
            interpretation='Original optimization proposals preserved; only decoded chemical rows are feasibility witnesses.'))
    return rows


def export_edit_tables(problem, selected_profiles, selections):
    """Translate every chosen witness into field-level, unit-audited records."""
    encoding = read(AW / 'nature_upgrade/data_claim_audit/paired_70_field_encodings.csv').set_index('field')
    names = read(AW / 'q2/chemical_dictionary.csv').set_index('field')
    calibration = read(AW / 'audit/calibrated_chemistry_source.csv').set_index('beer')
    profiles = selected_profiles.set_index(['evaluation_policy', 'candidate_uid'])
    changed, auxiliary, defects, invariants = [], [], [], []
    def units(field, before, after, beer):
        if field not in encoding.index:
            return dict(encoding='unresolved', unit='unresolved',
                        source_decoded_value=np.nan, candidate_decoded_value=np.nan,
                        source_original_calibration_value=np.nan,
                        decoded_value_interpretation='Supplied input scale only; no physical-unit conversion asserted.')
        info = encoding.loc[field]
        transform = str(info.empirical_encoding)
        assert transform in ('log10', 'identity', 'raw', 'raw_identity')
        inv = (lambda v: 10.**v) if transform == 'log10' else (lambda v: v)
        raw_before = float(calibration.loc[beer, field])
        return dict(encoding=transform, unit=str(info.unit),
                    source_decoded_value=float(inv(before)), candidate_decoded_value=float(inv(after)),
                    source_original_calibration_value=raw_before,
                    decoded_value_interpretation='Decoded calibrated-scale target; author replacement floors are representatives, not known instrument detection limits or measured intervention outcomes.')
    for case in selections[selections.candidate_exists].itertuples():
        row = profiles.loc[(case.policy, case.candidate_uid)]
        ctx = problem.contexts[int(case.beer)]
        src = ctx['src']
        shared = dict(policy=case.policy, beer=int(case.beer), rho=float(case.rho), candidate_uid=case.candidate_uid)
        decoded_z = problem.control.map_.transform(row[problem.features].to_numpy(float)[None])[0]
        same_count = 0
        for j, field in enumerate(problem.features):
            before, after = float(src[field]), float(row[field])
            same = before == after or (np.isnan(before) and np.isnan(after))
            same_count += int(same)
            if same:
                continue
            item = dict(shared, field=field, chemical_name=str(names.loc[field, 'graph_name']),
                        source_input_value=before, candidate_input_value=after,
                        change_kind='observed_value_change' if np.isfinite(before) else 'missing_value_completion',
                        counted_in_edit_budget=True,
                        rank_coordinate_change=float(decoded_z[j]-ctx['z0'][j]),
                        **units(field, before, after, int(case.beer)))
            changed.append(item)
        tech_ok = True
        for field in DERIVED + TECHNICAL:
            before, after = float(src[field]), float(row[field])
            same = before == after or (np.isnan(before) and np.isnan(after))
            if field in TECHNICAL:
                tech_ok = tech_ok and same
            auxiliary.append(dict(shared, field=field, source_input_value=before, candidate_input_value=after,
                unchanged=bool(same), counted_in_edit_budget=False,
                rule='technical field kept exactly' if field in TECHNICAL else
                     ('source-verified caloric identity' if field == 'kcalperc' else 'source-anchored auxiliary prediction, clipped to observed range'),
                **units(field, before, after, int(case.beer))))
        assert tech_ok
        assert problem.p - same_count == int(case.integer_upper_bound)
        invariants.append(dict(shared, direct_fields=problem.p, unchanged_direct_fields=same_count,
            changed_direct_fields=int(case.integer_upper_bound), observed_fields_changed=int(case.observed_fields_changed),
            missing_fields_completed=int(case.unknown_fields_completed), technical_field_exactly_preserved=tech_ok))
        for field, info in DEFECTS.items():
            before, after = float(src[field]), float(row[field])
            inv = (lambda v: 10.**v) if info['transform'] == 'log10' else (lambda v:v)
            c0, cp = float(inv(before)), float(inv(after))
            threshold = float(info['threshold'])
            cap = (c0 if case.policy == 'strict' else max(c0, threshold)) if np.isfinite(c0) else threshold
            assert cp <= cap + 1e-7
            defects.append(dict(shared, field=field, unit=info['unit'], threshold=threshold,
                threshold_source=info['source'], source_observed=bool(np.isfinite(before)),
                source_decoded_concentration=c0, candidate_decoded_concentration=cp,
                decoded_concentration_cap=cap,
                source_normalized_exceedance=max(c0/threshold-1, 0) if np.isfinite(c0) else np.nan,
                candidate_normalized_exceedance=max(cp/threshold-1, 0),
                source_original_calibration_value=float(calibration.loc[int(case.beer), field]),
                physical_intervention_measured=False))
    pd.DataFrame(changed).to_csv(OUT / 'changed_direct_fields.csv', index=False)
    pd.DataFrame(auxiliary).to_csv(OUT / 'reconstructed_auxiliary_fields.csv', index=False)
    pd.DataFrame(invariants).to_csv(OUT / 'unchanged_field_audit.csv', index=False)
    pd.DataFrame(defects).to_csv(OUT / 'defect_certificates.csv', index=False)


def summarize(require_perspective=True):
    OUT.mkdir(exist_ok=True)
    (OUT / 'COMPLETE.json').unlink(missing_ok=True)
    sdp_root = HERE / 'perspective_sdp'
    if require_perspective and (sdp_root / 'diagonal_sdp.npz').exists():
        if not (sdp_root / 'DONE.json').exists():
            raise RuntimeError('The registered SDP pilot/conditional extension has not finished.')
        sdp_done = json.loads((sdp_root / 'DONE.json').read_text())
        expected = 22 if sdp_done['expanded'] else 1
        assert int(sdp_done['cases_completed']) == expected
        assert len(list(sdp_root.glob('*/beer_*/DONE.json'))) == expected
    collected, sources, errors, audits = {}, [], {}, []
    frozen_before = {str(p.relative_to(AW)):file_sha(p) for p in [
        AW / 'q1/sensory_ensemble.joblib', AW / 'q3/transfer_and_augmentation_models.joblib',
        AW / 'competition_revision/style_feasibility/frozen_style_classifier.joblib']}
    for policy in ['strict', 'exceedance']:
        problem, frame, checked, sensory, bounds, files, err = collect_policy(policy, require_perspective)
        collected[policy] = dict(problem=problem, frame=frame, checks=checked, sensory=sensory, bounds=bounds)
        sources += files
        errors[policy] = err
        audits += canonical_rank_artifacts(problem, policy)
    # Relaxing concentration caps cannot invalidate a strict feasible witness.
    strict = collected['strict']
    relaxed = collected['exceedance']
    keep = strict['checks'].accepted.to_numpy()
    strict_rows = strict['frame'].loc[keep].copy()
    cross, cross_sensory = relaxed['problem'].check_rows(strict_rows.to_dict('records'))
    assert cross.accepted.all()
    cross.insert(0, 'candidate_uid', strict_rows.candidate_uid.to_numpy())
    cross['search_policy'] = 'strict'
    cross['search_family'] = strict_rows.search_family.to_numpy()
    cross['source_file'] = strict_rows.source_file.to_numpy()
    cross_sensory.insert(0, 'candidate_uid', strict_rows.candidate_uid.to_numpy())
    for key, extra in [('frame', strict_rows), ('checks', cross), ('sensory', cross_sensory)]:
        relaxed[key] = pd.concat([relaxed[key], extra], ignore_index=True)

    chosen_frames, chosen_sensory, chosen_checks, selection, bounds_all = [], [], [], [], []
    for policy, data in collected.items():
        problem, frame, checked = data['problem'], data['frame'], data['checks']
        frame = frame.set_index('candidate_uid', drop=False)
        sensory = data['sensory'].set_index('candidate_uid', drop=False)
        checked.to_csv(OUT / f'{policy}_audited_pool_checks.csv', index=False)
        # Full pool chemical rows already have lossless source_file+row identity;
        # retain selected rows once, then map all rho cases to their identities.
        for beer in sorted(problem.contexts):
            eligible = checked[checked.accepted & checked.beer.eq(beer)]
            records = [b for b in data['bounds'] if int(b['beer']) == beer]
            valid_bounds = [b for b in records if np.isfinite(float(b['dual_bound']))]
            best_bound = max(valid_bounds, key=lambda x: float(x['dual_bound']))
            dual = float(best_bound['dual_bound'])
            integer_lower = max(0, int(math.ceil(dual - 1e-6)))
            bounds_all.extend(records)
            for rho in RHOS:
                good = eligible[eligible.target_distance_reduction >= rho - ATOL].sort_values(
                    ['raw_fields_changed', 'quadratic_objective', 'candidate_uid'], kind='stable')
                rec = dict(policy=policy, beer=beer, rho=rho, eligible_candidates=len(good),
                           candidate_exists=bool(len(good)), lower_bound_rho=.1,
                           raw_solver_lower_bound=dual,
                           integer_lower_bound=integer_lower if rho >= .1 else np.nan,
                           lower_bound_source=best_bound['bound_source'])
                if len(good):
                    row = good.iloc[0]
                    uid = row.candidate_uid
                    witness = frame.loc[uid].copy()
                    vals = witness[problem.features].to_numpy(float)
                    old = problem.contexts[beer]['x0']
                    observed_changes = np.isfinite(old) & (vals != old)
                    imputed_unknowns = np.isnan(old) & np.isfinite(vals)
                    assert int(observed_changes.sum() + imputed_unknowns.sum()) == int(row.raw_fields_changed)
                    rec.update(row.to_dict())
                    rec.update(policy=policy, rho=rho, integer_upper_bound=int(row.raw_fields_changed),
                               observed_fields_changed=int(observed_changes.sum()),
                               unknown_fields_completed=int(imputed_unknowns.sum()))
                    if rho >= .1:
                        assert integer_lower <= int(row.raw_fields_changed)
                    witness['evaluation_policy'] = policy
                    chosen_frames.append(witness.to_dict())
                    chosen_sensory.append(dict(sensory.loc[uid], evaluation_policy=policy))
                    chosen_checks.append(dict(row, evaluation_policy=policy))
                selection.append(rec)
    sel = pd.DataFrame(selection)
    sel.to_csv(OUT / 'selection_cases.csv', index=False)
    chosen_frame = pd.DataFrame(chosen_frames).drop_duplicates(['evaluation_policy', 'candidate_uid'])
    chosen_frame.to_csv(OUT / 'selected_profiles.csv', index=False)
    pd.DataFrame(chosen_sensory).drop_duplicates(['evaluation_policy', 'candidate_uid']).to_csv(OUT / 'selected_predicted_sensory.csv', index=False)
    pd.DataFrame(chosen_checks).drop_duplicates(['evaluation_policy', 'candidate_uid']).to_csv(OUT / 'selected_full_checks.csv', index=False)
    export_edit_tables(strict['problem'], chosen_frame, sel)
    bound_frame = pd.DataFrame(bounds_all)
    # Nested dictionaries are left as JSON in source files, omitted from CSV.
    bound_frame.drop(columns=['diagonal_split'], errors='ignore').to_csv(OUT / 'all_solver_bounds.csv', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'rank_witness_audit.csv', index=False)

    main = []
    legacy = {policy:read(HERE / 'matched_grid_reference' / f'{policy}_minimum_found.csv').set_index('beer')
              for policy in collected}
    for beer in sorted(strict['problem'].contexts):
        rec = dict(beer=beer)
        for policy in collected:
            z = sel[(sel.policy == policy) & (sel.beer == beer) & (sel.rho == .1)].iloc[0]
            old = legacy[policy].loc[beer]
            assert bool(z.candidate_exists)
            rec.update({f'{policy}_lower':int(z.integer_lower_bound),
                        f'{policy}_upper':int(z.integer_upper_bound),
                        f'{policy}_distance':float(z.target_distance_reduction),
                        f'{policy}_stale':float(z.ensemble_stale_delta),
                        f'{policy}_candidate_uid':z.candidate_uid,
                        f'{policy}_observed_fields_changed':int(z.observed_fields_changed),
                        f'{policy}_unknown_fields_completed':int(z.unknown_fields_completed),
                        f'legacy_{policy}_upper':int(old.raw_fields_changed),
                        f'legacy_{policy}_distance':float(old.target_distance_reduction),
                        f'{policy}_cost_reduction':1-float(z.integer_upper_bound)/float(old.raw_fields_changed)})
        assert rec['exceedance_upper'] <= rec['strict_upper']
        main.append(rec)
    main = pd.DataFrame(main)
    main.to_csv(OUT / 'main_comparison.csv', index=False)
    summaries = []
    for (policy, rho), group in sel.groupby(['policy', 'rho'], sort=False):
        ok = group[group.candidate_exists]
        summaries.append(dict(policy=policy, rho=rho, accepted=len(ok),
            median_fields=float(ok.integer_upper_bound.median()) if len(ok) else np.nan,
            max_fields=int(ok.integer_upper_bound.max()) if len(ok) else None,
            all11_mean_distance_reduction=float(ok.target_distance_reduction.mean()) if len(ok)==11 else None))
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(OUT / 'policy_summary.csv', index=False)
    mlo, mhi = int(main.strict_upper.min()), int(main.strict_upper.max())
    med = float(main.strict_upper.median())
    oldmed = float(main.legacy_strict_upper.median())
    reduction = 1-main.strict_upper.sum()/main.legacy_strict_upper.sum()
    nclosed = int((main.strict_lower == main.strict_upper).sum())
    abstract = (f'在全模型缺陷约束及至少10%风格距离改善下，11款源样本均获得可行迁移，'
                f'实际编辑数为{mlo}—{mhi}个，中位数由相同条件下旧支持策略的{oldmed:g}个降至{med:g}个，'
                f'总编辑数减少{100*reduction:.1f}%；整数求解器同时给出兼容下界。')
    interpretation = (f'同一固定分类器、感官集成、化学边界、$10^{{-5}}$分类间隔和$\\rho=0.1$下，'
        f'11款均通过完整复核；总编辑数由{int(main.legacy_strict_upper.sum())}降至{int(main.strict_upper.sum())}。'
        f'旧支持比较重新求解并逐点复核，未将“80字段首次全通过”的旧网格结论当作新的10%改善基准。'
        f'当前{nclosed}/11款的整数上下界闭合，其余报告$[L,U]$和实际可行谱，不把时限内最佳解写成已证明的全局最少。'
        f'编辑成本计入缺失字段的补全，并在明细中与已有观测值的改变分开。'
        f'放松缺陷政策的候选池纳入全部严格政策可行点，因此其上界遵循可行集包含关系。')
    frozen_after = {p:file_sha(AW / p) for p in frozen_before}
    assert frozen_after == frozen_before
    metadata = dict(abstract=abstract, interpretation=interpretation,
        require_all_perspective=require_perspective,
        strict_min_fields=mlo, strict_max_fields=mhi, strict_median_fields=med,
        matched_legacy_median_fields=oldmed, strict_total_edit_reduction=reduction,
        closed_strict_intervals=nclosed, all_sources=11,
        proof_scope='Numerical solver dual bounds for the frozen decision model containing BOTH the explicit linear stale guard and complete nonlinear frozen-sensory constraint. The relaxation removes only the nonlinear condition. Selected decoded chemical rows pass both. These primary lower bounds do not automatically apply after removing the linear guard.',
        lower_bound_ceil_tolerance=1e-6, feasibility_tolerance=ATOL,
        matched_comparison='Identical scientific constraints; search algorithms have different computational budgets, reported in their own logs.',
        selection='Minimum actual raw-field edit count, then quadratic objective, then stable candidate identity; bootstrap stability never enters selection.',
        source_files=sources, recheck_max_errors=errors,
        frozen_models_before=frozen_before, frozen_models_after=frozen_after)
    write_json(OUT / 'summary.json', metadata)
    write_json(OUT / 'COMPLETE.json', dict(complete=True, policies=2, sources=11,
        selections=len(sel), feasible_selections=int(sel.candidate_exists.sum()),
        all_perspective=require_perspective,
        sdp_finished=(sdp_root / 'DONE.json').exists(),
        source_pool_sha256=file_sha(OUT / 'selection_cases.csv')))
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['source_files','frozen_models_before','frozen_models_after']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-partial-perspective', action='store_true')
    args = parser.parse_args()
    summarize(require_perspective=not args.allow_partial_perspective)
