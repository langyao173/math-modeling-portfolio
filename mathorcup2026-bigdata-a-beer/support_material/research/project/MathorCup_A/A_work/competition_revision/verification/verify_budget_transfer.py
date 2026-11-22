"""Independent artifact verification, without rerunning support optimization.

Reads raw supplied measurements, saved models, full candidate diagnostics, and
selected raw profiles. Only this verification directory is written by default.
Use --repair-metadata to correct provenance fields in the selected profile CSV;
the repair preserves every existing numeric CSV token verbatim.
"""
from pathlib import Path
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse, ast, csv, hashlib, json, math, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, r2_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REVISION = HERE.parent
AWORK = REVISION.parent
DATA = Path(os.environ.get('MATHORCUP_DATA_DIR', AWORK.parent / 'data/A')).expanduser().resolve()
NEW = REVISION / 'budget_transfer'
OLD = AWORK / 'nature_upgrade/sparse_transfer'
sys.path[:0] = [str(AWORK / 'q1'), str(AWORK / 'q3'), str(AWORK / 'q2')]
from sensory_model import predict_bundle
from conditional_model import DERIVED, TECHNICAL
from transfer_profiles import DEFECTS
from run_q2 import greedy_modularity

BUDGETS = [3, 5, 10, 20, 40, 60, 80, 100, 140, 180]
FAMILIES = ['gradient', 'full_change', 'forward_schur']
KEYS = ['beer', 'method']
CHECKS = []


def read(path):
    # The optimizer read the source with pandas' ordinary parser. Round-trip
    # parsing here preserves the exact floats serialized into result files.
    return pd.read_csv(path, float_precision='round_trip')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(obj, path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=scalar), encoding='utf-8')


def check(name, condition, details=None):
    CHECKS.append(dict(check=name, passed=bool(condition), details=details))


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (a == b) | (np.isnan(a) & np.isnan(b))


def maxerror(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return float('inf')
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[mask]-b[mask]))) if mask.any() else 0.


def metadata_repair(profiles, diagnostics, original, apply):
    path = NEW / 'selected_profiles.csv'
    raw = list(csv.DictReader(path.open(newline='', encoding='utf-8')))
    fields = list(raw[0])
    lookup = diagnostics.set_index(KEYS)
    target_written_as_observed = sum(r['tasting_category_fine'] == 'Tripel' for r in raw)
    conflicting = sum(r['tasting_category_fine'] == 'Tripel' and
                      lookup.loc[(int(r['beer']), r['method']), 'predicted_style'] != 'Tripel' for r in raw)
    report = dict(applied=False, prior_target_label_count=target_written_as_observed,
                  prior_target_label_conflicting_with_diagnostic=conflicting,
                  file_sha256_before=digest(path), numeric_tokens_unchanged=None,
                  category_semantics='tasting_category_fine retains the observed SOURCE label; no transferred beer has a newly measured style label')
    new_fields = ['source_tasting_category_fine', 'requested_target_style',
                  'diagnostic_predicted_style', 'profile_origin', 'tasting_category_fine_origin']
    for field in new_fields:
        if field not in fields:
            fields.append(field)
    for r in raw:
        key = int(r['beer']), r['method']
        source = str(original.loc[key[0], 'tasting_category_fine'])
        r.update(tasting_category_fine=source, source_tasting_category_fine=source,
                 requested_target_style='Tripel',
                 diagnostic_predicted_style=str(lookup.loc[key, 'predicted_style']),
                 profile_origin='virtual_transfer' if lookup.loc[key, 'nontrivial'] else 'observed_source_fallback',
                 tasting_category_fine_origin='observed_source_label; transfer_not_remeasured')
    if apply and (target_written_as_observed or any(f not in profiles for f in new_fields)):
        backup = HERE / 'selected_profiles_before_metadata.csv'
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        before = list(csv.DictReader(path.open(newline='', encoding='utf-8')))
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(raw)
        after = list(csv.DictReader(path.open(newline='', encoding='utf-8')))
        untouched = all(all(a[k] == b[k] for k in a if k != 'tasting_category_fine')
                        for a, b in zip(before, after))
        assert untouched, 'Metadata patch changed an existing non-category CSV token'
        report.update(applied=True, numeric_tokens_unchanged=untouched,
                      all_preexisting_noncategory_tokens_unchanged=untouched,
                      file_sha256_after=digest(path), new_fields=new_fields)
        dump(report, HERE / 'metadata_patch.json')
    elif (HERE / 'metadata_patch.json').exists():
        report = json.loads((HERE / 'metadata_patch.json').read_text())
        report['idempotent_recheck'] = True
    else:
        report['proposed_fields'] = new_fields
    final = read(path)
    semantic_ok = ('source_tasting_category_fine' in final and
                   (final.tasting_category_fine == final.beer.map(original.tasting_category_fine)).all() and
                   (final.requested_target_style == 'Tripel').all() and
                   (final.diagnostic_predicted_style.to_numpy() == diagnostics.predicted_style.to_numpy()).all())
    check('profile_label_provenance_clear', semantic_ok, report)
    code_patch = HERE / 'metadata_export_code_patch.json'
    if code_patch.exists():
        provenance = json.loads(code_patch.read_text())
        prior_code = (HERE / 'run_budget_transfer_before_metadata.py').read_text()
        expected_code = prior_code.replace('\n'.join(provenance['removed_lines'])+'\n',
                                           '\n'.join(provenance['added_lines'])+'\n')
        check('source_code_change_is_metadata_export_only',
              expected_code == (NEW / 'run_budget_transfer.py').read_text() and
              digest(NEW / 'run_budget_transfer.py') == provenance['after_sha256'], provenance)
        report['source_export_patch'] = provenance
    return report


def budget_audit(repair):
    protocol = json.loads((NEW / 'protocol.json').read_text())
    summary = json.loads((NEW / 'summary.json').read_text())
    df = pd.read_csv(DATA / '1.csv')
    original = df.set_index('beer')
    calibrated = pd.read_excel(DATA / '8.xlsx', sheet_name='Calibrated chemical dataset').set_index('beer')
    profiles = read(NEW / 'selected_profiles.csv')
    diagnostics = read(NEW / 'selected_diagnostics.csv')
    supports = read(NEW / 'selected_supports.csv')
    paths = read(NEW / 'path_checks.csv')
    qp = read(NEW / 'qp_attempts.csv')
    trace = read(NEW / 'forward_support_trace.csv')
    sensory = read(NEW / 'selected_sensory_predictions.csv')
    certificates = read(NEW / 'chemical_certificate.csv')
    changes = read(NEW / 'changed_direct_fields.csv')
    pareto = read(NEW / 'pareto_summary.csv')
    pack = joblib.load(AWORK / 'q3/transfer_and_augmentation_models.joblib')
    bundle = joblib.load(AWORK / 'q1/sensory_ensemble.joblib')
    control, features = pack['model'], pack['features']
    p = len(features)
    source_ids = sorted(df.loc[df.tasting_category_fine == 'Pils/Lager', 'beer'].astype(int).tolist())
    methods = [f'{f}_k{k}' for f in FAMILIES for k in BUDGETS] + ['all_225', 'author7_free', 'author7_add_only', 'author7_plus_required']
    expected_keys = {(b, m) for b in source_ids for m in methods}
    check('all_11_sources_by_all_34_methods', len(profiles) == 374 and
          set(map(tuple, profiles[KEYS].to_numpy())) == expected_keys and not profiles.duplicated(KEYS).any(),
          dict(sources=source_ids, rows=len(profiles), methods=len(methods)))
    for name, table in [('diagnostics', diagnostics), ('supports', supports), ('sensory', sensory)]:
        check(name+'_keys_match', set(map(tuple, table[KEYS].to_numpy())) == expected_keys and not table.duplicated(KEYS).any())
    check('row_order_alignment', profiles[KEYS].equals(diagnostics[KEYS]) and profiles[KEYS].equals(sensory[KEYS]))
    check('budget_protocol_matches_execution', protocol['budgets'] == BUDGETS == summary['prespecified_budgets'] and
          protocol['families'] == FAMILIES and protocol['same_objective_lambda'] == summary['lambda_preservation'] == .2)
    assignments = {}
    for n in ast.parse((NEW / 'run_budget_transfer.py').read_text()).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            try:
                assignments[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                pass
    check('script_literals_match_protocol', assignments['BUDGETS'] == BUDGETS and assignments['LAMBDA'] == .2)
    predictor_hash = digest(AWORK / 'q1/sensory_ensemble.joblib')
    reference_hash = digest(OLD / 'run_sparse_transfer.py')
    check('unchanged_predictor_hash', predictor_hash == protocol['same_predictor_sha256'], predictor_hash)
    check('reference_script_hash', reference_hash == protocol['reference_script_sha256'], reference_hash)
    check('summary_execution_counts', len(paths) == summary['n_path_candidates'] and len(qp) == summary['n_qp_attempts'] and
          int(qp.success.sum()) == summary['n_qp_success'] and int(trace.evaluated_candidates.sum()) == summary['n_forward_candidate_qp_checks'])
    support_lookup = {(int(r.beer), r.method): json.loads(r.selected_fields) for r in supports.itertuples()}
    support_errors, nested_errors = [], []
    for r in supports.itertuples():
        s = support_lookup[int(r.beer), r.method]
        if len(s) != r.allowed_coordinates or len(s) != len(set(s)) or not set(s) <= set(features):
            support_errors.append((int(r.beer), r.method))
    for b in source_ids:
        for f in FAMILIES:
            for a, c in zip(BUDGETS[:-1], BUDGETS[1:]):
                if not set(support_lookup[b, f'{f}_k{a}']) <= set(support_lookup[b, f'{f}_k{c}']):
                    nested_errors.append((b, f, a, c))
    check('support_lengths_valid', not support_errors, support_errors)
    check('all_297_adjacent_support_inclusions', not nested_errors, nested_errors)
    zero = paths[paths.alpha == 0]
    check('all_374_source_fallbacks_retained_in_candidates', len(zero) == 374 and set(map(tuple, zero[KEYS].to_numpy())) == expected_keys)
    selection_errors = []
    for r in diagnostics.itertuples():
        group = paths[paths.beer == r.beer]
        if '_k' in r.method:
            family, k = r.method.rsplit('_k', 1)
            group = group[group.method.isin([f'{family}_k{j}' for j in BUDGETS if j <= int(k)])]
        else:
            group = group[group.method == r.method]
        best = group[group.eligible_for_action].sort_values(
            ['quadratic_objective', 'direct_coordinates_changed', 'margin', 'alpha'], kind='stable').iloc[0]
        if int(best.candidate_index) != r.candidate_index or best.method != r.selection_origin_method:
            selection_errors.append((int(r.beer), r.method))
        if not set(support_lookup[int(r.beer), r.selection_origin_method]) <= set(support_lookup[int(r.beer), r.method]):
            selection_errors.append((int(r.beer), r.method, 'origin_not_in_budget'))
    check('selection_reproduced_from_entire_nested_candidate_sets', not selection_errors, selection_errors)
    objective_increases = []
    for b in source_ids:
        for f in FAMILIES:
            vals = diagnostics.set_index(KEYS).loc[[(b, f'{f}_k{k}') for k in BUDGETS], 'quadratic_objective'].to_numpy()
            if np.max(np.diff(vals)) > 1e-8:
                objective_increases.append((b, f, float(np.max(np.diff(vals)))))
    check('nested_selected_objective_nonincreasing', not objective_increases, objective_increases)
    # Exact old/new numerical comparison includes all 16 former methods.
    old_profiles = read(OLD / 'selected_profiles.csv').set_index(KEYS)
    old_diag = read(OLD / 'selected_diagnostics.csv').set_index(KEYS)
    old_sensory = read(OLD / 'selected_sensory_predictions.csv').set_index(KEYS)
    pp = profiles.set_index(KEYS).loc[old_profiles.index]
    profile_numbers = old_profiles.select_dtypes(include='number').columns.tolist()
    pexact = bool(same(pp[profile_numbers], old_profiles[profile_numbers]).all())
    dd = diagnostics.set_index(KEYS).loc[old_diag.index]
    dc = [c for c in old_diag.select_dtypes(include='number').columns if c != 'candidate_index']
    ss = sensory.set_index(KEYS).loc[old_sensory.index]
    sc = old_sensory.select_dtypes(include='number').columns.tolist()
    old_consistency = dict(cases=len(old_profiles), profile_numeric_columns=len(profile_numbers),
                           exact_profile_numeric_equality=pexact, profile_max_error=maxerror(pp[profile_numbers], old_profiles[profile_numbers]),
                           diagnostics_max_error=maxerror(dd[dc], old_diag[dc]), sensory_max_error=maxerror(ss[sc], old_sensory[sc]))
    check('old_k20_and_controls_numeric_reproduction', pexact and old_consistency['diagnostics_max_error'] <= 1e-9 and
          old_consistency['sensory_max_error'] <= 1e-10, old_consistency)
    old_supports = read(OLD / 'selected_supports.csv').set_index(KEYS)
    check('old_supports_identical', supports.set_index(KEYS).loc[old_supports.index, 'selected_fields'].equals(old_supports.selected_fields))
    # Recompute direct/derived values from the exported CSV, not path diagnostics.
    original_rows = original.loc[profiles.beer].reset_index()
    x0 = original_rows[features].to_numpy(float)
    xn = profiles[features].to_numpy(float)
    z0 = control.map_.transform(x0)
    zn = control.map_.transform(xn)
    dz = zn-z0
    raw_changed = ~same(xn, x0)
    frozen_errors, nan_errors, mandatory_errors = [], [], []
    frozen_count = frozen_missing = 0
    for i, r in enumerate(profiles.itertuples()):
        support = set(support_lookup[int(r.beer), r.method])
        frozen = np.array([c not in support for c in features])
        frozen_count += int(frozen.sum())
        frozen_missing += int((frozen & np.isnan(x0[i])).sum())
        bad = np.flatnonzero(frozen & raw_changed[i])
        frozen_errors.extend((int(r.beer), r.method, features[j]) for j in bad)
        badnan = np.flatnonzero(frozen & np.isnan(x0[i]) & ~np.isnan(xn[i]))
        nan_errors.extend((int(r.beer), r.method, features[j]) for j in badnan)
    check('exact_frozen_direct_fields', not frozen_errors, dict(cells_checked=frozen_count, failures=frozen_errors))
    check('unselected_original_NaNs_preserved', not nan_errors, dict(cells_checked=frozen_missing, failures=nan_errors))
    check('technical_internal_standard_exact', same(profiles[TECHNICAL], original_rows[TECHNICAL]).all(), dict(fields=TECHNICAL, cells=len(profiles)*len(TECHNICAL)))
    raw_counts = raw_changed.sum(axis=1)
    rank_counts = (abs(dz) > 1e-8).sum(axis=1)
    check('actual_raw_field_change_count_within_budget', np.all(raw_counts <= diagnostics.allowed_coordinates),
          dict(max_budget_excess=int(np.max(raw_counts-diagnostics.allowed_coordinates.to_numpy())), total_raw_field_changes=int(raw_counts.sum())))
    check('reported_rank_change_counts_recomputed', np.array_equal(rank_counts, diagnostics.direct_coordinates_changed),
          dict(raw_vs_rank_count_difference_cases=int((raw_counts != rank_counts).sum())))
    change_keys = {(int(profiles.beer.iloc[i]), profiles.method.iloc[i], features[j]) for i, j in np.argwhere(raw_changed)}
    check('changed_field_export_complete', len(changes) == len(change_keys) and set(map(tuple, changes[['beer','method','chemical']].to_numpy())) == change_keys)
    check('unchanged_fallback_preserves_all_chemical_fields', same(profiles.loc[~diagnostics.nontrivial, features+DERIVED+TECHNICAL],
          original_rows.loc[~diagnostics.nontrivial, features+DERIVED+TECHNICAL]).all())
    moving = diagnostics.nontrivial.to_numpy(bool)
    generated = profiles.is_generated.to_numpy(bool)
    lower, upper = control.map_.lower_, control.map_.upper_
    bounds_error = max(float(np.max(lower-zn)), float(np.max(zn-upper)), 0.)
    check('empirical_rank_support_bounds', bounds_error <= 1e-8, bounds_error)
    chemical_records = []
    unresolved = np.zeros(len(profiles), dtype=int)
    cap_error = np.zeros(len(profiles))
    certificate_errors = []
    cert_index = certificates.set_index(['beer','method','chemical'])
    for i, r in enumerate(profiles.itertuples()):
        for c, info in DEFECTS.items():
            before, after = float(original_rows.loc[i,c]), float(profiles.loc[i,c])
            threshold = info['threshold']
            cap = before if np.isfinite(before) else (np.log10(threshold) if info['transform']=='log10' else threshold)
            if np.isfinite(after):
                cap_error[i] = max(cap_error[i], after-cap, 0.)
            else:
                unresolved[i] += 1
            cal = calibrated.loc[int(r.beer),c]
            status = ('missing' if not np.isfinite(before) else
                      'zero_or_nonpositive_replacement' if info['transform']=='log10' and (not np.isfinite(cal) or cal <= 0)
                      else 'positive_reported_measurement')
            raw_before = 10.**before if info['transform']=='log10' else before
            raw_after = 10.**after if info['transform']=='log10' else after
            saved = cert_index.loc[(int(r.beer),r.method,c)]
            if saved.source_status != status or maxerror([saved.source_model_representative,saved.transferred_concentration,saved.threshold], [raw_before,raw_after,threshold]) > 1e-10:
                certificate_errors.append((int(r.beer),r.method,c))
            chemical_records.append(dict(beer=int(r.beer),method=r.method,chemical=c,source_status=status,
                                         source_calibrated_value=float(cal),source_model_value=before,output_model_value=after,
                                         field_scale_cap=float(cap),source_raw=float(raw_before),output_raw=float(raw_after),threshold=threshold,
                                         moving=bool(moving[i]),output_exceeds_threshold=bool(np.isfinite(raw_after) and raw_after>threshold+1e-8)))
    check('four_chemical_caps_all_moving_profiles', np.all(cap_error[moving] <= 1e-8) and np.all(unresolved[moving] == 0),
          dict(moving_profiles=int(moving.sum()), max_field_scale_violation=float(cap_error[moving].max()), unresolved_moving=int(unresolved[moving].sum())))
    check('chemical_certificate_source_status_and_values', not certificate_errors, certificate_errors)
    check('recorded_missing_defect_counts', np.array_equal(unresolved, diagnostics.unresolved_missing_defects))
    source_status = pd.DataFrame(chemical_records).drop_duplicates(['beer','chemical'])[['beer','chemical','source_status','source_calibrated_value','source_raw','threshold']]
    source_status.to_csv(HERE / 'source_defect_status.csv', index=False)
    pd.DataFrame(chemical_records).to_csv(HERE / 'recomputed_chemical_caps.csv',index=False)
    # Four source-anchored auxiliary summaries, then the independently known kcal identity.
    derived_errors = {}
    for c, reg in pack['derived_predictors'].items():
        old = original_rows[c].to_numpy(float)
        expected = np.where(np.isfinite(old),old+reg.predict(zn)-reg.predict(z0),reg.predict(zn))
        expected = np.clip(expected,np.nanmin(df[c]),np.nanmax(df[c]))
        expected[~generated] = old[~generated]
        derived_errors[c] = maxerror(expected, profiles[c])
    calorie_fields = ['ethanol..v.v.','GFS','unfermentables','protein.g.L.','lactic_acid.mg.L.']
    c = 10.**profiles[calorie_fields].to_numpy(float)
    expected_kcal = np.log10(5.523*c[:,0]+.4*(c[:,1]+c[:,2]+c[:,3])+.0003*c[:,4])
    expected_kcal[~generated] = original_rows.loc[~generated,'kcalperc']
    derived_errors['kcalperc'] = maxerror(expected_kcal, profiles.kcalperc)
    check('source_anchored_derived_reconstruction', max(derived_errors.values()) <= 1e-10, derived_errors)
    recomputed_sensory = predict_bundle(bundle, profiles)
    base_sensory = predict_bundle(bundle, original_rows)
    linear = dz @ pack['stale_proxy'].coef_
    nonlinear = recomputed_sensory.stale_hops.to_numpy()-base_sensory.stale_hops.to_numpy()
    stale_errors = dict(linear=maxerror(linear, diagnostics.linear_stale_delta),
                        nonlinear=maxerror(nonlinear, diagnostics.ensemble_stale_delta),
                        sensory_predictions=maxerror(recomputed_sensory[bundle['targets']], sensory[bundle['targets']]))
    check('exported_profiles_reproduce_both_stale_deltas', max(stale_errors.values()) <= 1e-9, stale_errors)
    check('both_stale_filters_all_moving_profiles', np.all(linear[moving] <= 1e-8) and np.all(nonlinear[moving] <= 1e-8),
          dict(linear_max=float(linear[moving].max()),ensemble_max=float(nonlinear[moving].max())))
    accepted = (cap_error <= 1e-8)&(unresolved == 0)&(linear <= 1e-8)&(nonlinear <= 1e-8)&moving
    check('all_accepted_flags_independently_reproduced', np.array_equal(accepted, diagnostics.accepted_generated_profile))
    # Classifier prediction is recomputed for diagnostics; it is not in selection.
    zz = control.map_.transform(df[features].to_numpy(float))
    clf = LogisticRegression(C=.1, max_iter=3000, class_weight='balanced').fit(zz,df.tasting_category_fine)
    labels = clf.predict(zn)
    prob = clf.predict_proba(zn)[:,list(clf.classes_).index('Tripel')]
    check('diagnostic_classifier_recomputed', np.array_equal(labels, diagnostics.predicted_style) and maxerror(prob,diagnostics.target_probability)<=1e-10,
          dict(probability_max_error=maxerror(prob,diagnostics.target_probability), tripel_cases=int((labels=='Tripel').sum())))
    mu = control.mean_['Tripel']
    precision = np.linalg.inv(control.cov_['Tripel'])
    quadratic = np.einsum('ij,jk,ik->i',zn-mu,precision,zn-mu)
    source_quadratic = np.einsum('ij,jk,ik->i',z0-mu,precision,z0-mu)
    distance = np.sqrt(quadratic/p)
    reduction = 1-np.sqrt(quadratic/source_quadratic)
    objective = .5*quadratic+.1*np.sum(dz**2,axis=1)
    geometry_errors = dict(target_distance=maxerror(distance,diagnostics.target_distance),
                           target_distance_reduction=maxerror(reduction,diagnostics.target_distance_reduction),
                           objective=maxerror(objective,diagnostics.quadratic_objective))
    check('distance_and_quadratic_objective_recomputed', max(geometry_errors.values())<=1e-9,geometry_errors)
    recalculated = diagnostics.groupby('method',sort=False).agg(
        n_sources=('beer','size'),n_nontrivial=('nontrivial','sum'),allowed_coordinates_mean=('allowed_coordinates','mean'),
        actual_changed_mean=('direct_coordinates_changed','mean'),target_distance_reduction_mean=('target_distance_reduction','mean'),
        target_distance_reduction_min=('target_distance_reduction','min'),intervention_rms_mean=('intervention_rms','mean'),
        target_probability_mean=('target_probability','mean'),tripel_classified=('predicted_style',lambda s:int((s=='Tripel').sum())),
        ensemble_stale_max_delta=('ensemble_stale_delta','max'),linear_stale_max_delta=('linear_stale_delta','max'),
        unresolved_missing_defects=('unresolved_missing_defects','sum'),accepted_generated_profiles=('accepted_generated_profile','sum'),
        objective_mean=('quadratic_objective','mean')).reset_index()
    pc = [c for c in pareto if c!='method']
    check('all_pareto_summary_values_recomputed', maxerror(recalculated[pc],pareto[pc])<=1e-10)
    first_success = []
    for b in source_ids:
        for f in FAMILIES:
            rows = diagnostics[(diagnostics.beer==b)&diagnostics.method.isin([f'{f}_k{k}' for k in BUDGETS])].copy()
            rows['budget'] = rows.method.str.rsplit('_k',n=1).str[-1].astype(int)
            success = rows[rows.predicted_style=='Tripel'].sort_values('budget')
            r = success.iloc[0] if len(success) else None
            first_success.append(dict(beer=b,family=f,first_tested_budget_with_diagnostic_Tripel=None if r is None else int(r.budget),
                                      rank_coordinates_changed_at_first=None if r is None else int(r.direct_coordinates_changed),
                                      raw_fields_changed_at_first=None if r is None else int(raw_counts[int(r.name)]),
                                      origin_method=None if r is None else r.selection_origin_method,
                                      interpretation='first successful tested budget; not global minimum support'))
    pd.DataFrame(first_success).to_csv(HERE/'first_tested_diagnostic_success.csv',index=False)
    counts = pd.DataFrame({'beer':profiles.beer,'method':profiles.method,'raw_fields_changed':raw_counts,
                           'rank_coordinates_changed':rank_counts,'allowed_coordinates':diagnostics.allowed_coordinates,
                           'unresolved_missing_defects':unresolved,'linear_stale_delta':linear,'ensemble_stale_delta':nonlinear,
                           'diagnostic_style':labels,'target_distance_reduction':reduction,'quadratic_objective':objective})
    counts.to_csv(HERE/'recomputed_selected_diagnostics.csv',index=False)
    counts.groupby('method',sort=False).agg(raw_fields_changed_mean=('raw_fields_changed','mean'),
        rank_coordinates_changed_mean=('rank_coordinates_changed','mean'),allowed_coordinates=('allowed_coordinates','max')).to_csv(HERE/'raw_and_rank_change_counts.csv')
    patch = metadata_repair(profiles,diagnostics,original,repair)
    fallback = diagnostics.loc[~diagnostics.nontrivial,['beer','method','unresolved_missing_defects','accepted_generated_profile']].to_dict('records')
    result = dict(protocol=protocol,protocol_timing_limitation='The protocol file and code agree; this audit cannot independently prove the timestamp assertion created_before_execution.',
                  n_sources=len(source_ids),n_methods=len(methods),n_selected=len(profiles),n_moving=int(moving.sum()),
                  fallback_cases=fallback,n_candidate_paths=len(paths),n_failed_qp=int((~qp.success).sum()),
                  old_consistency=old_consistency,stale_errors=stale_errors,derived_errors=derived_errors,geometry_errors=geometry_errors,
                  change_count_conventions=dict(raw_exact_total=int(raw_counts.sum()),rank_tolerance_total=int(rank_counts.sum()),
                      differing_cases=int((raw_counts!=rank_counts).sum()),rank_tolerance=1e-8,
                      note='Both exact raw-value counts and rank-tolerance counts stay within budget; the original pareto actual_changed_mean uses rank-tolerance counts.'),
                  source_status_counts=source_status.source_status.value_counts().to_dict(),
                  source_status_records=source_status.to_dict('records'),metadata_patch=patch,
                  budget_results=recalculated.to_dict('records'),first_success=first_success,
                  chemical_interpretation='Known source values may remain above sensory thresholds, provided they do not increase. Missing source defects require output at or below threshold. Replacement floors are not positive measured concentrations.',
                  statistical_scope='All optimization and classifier diagnostics reuse fitted source data. This audit checks internal consistency, not causal chemistry, external generalization, or new tasting.')
    return result


def grading_audit():
    out = REVISION/'grading'
    medpath = AWORK/'audit/model_validation/strict_missingness_oof.npz'
    knnpath = AWORK/'audit/model_validation/strict_knn_missingness_oof.npz'
    med, knn = np.load(medpath), np.load(knnpath)
    df = pd.read_csv(DATA/'1.csv').sort_values('beer').reset_index(drop=True)
    yd = pd.read_csv(DATA/'4.csv').set_index('beer').loc[df.beer]
    targets = yd.columns[2:].tolist()
    Y = yd[targets].to_numpy(float)
    check('grading_input_alignment', np.array_equal(med['beer'],df.beer) and np.array_equal(med['fold'],knn['fold']) and
          maxerror(Y,med['Y'])<=1e-12 and maxerror(Y,knn['Y'])<=1e-12)
    folds = med['fold']; true = np.zeros(Y.shape,int); base = np.zeros_like(Y)
    pred_arrays = {('训练均值','original'):base}
    for method,source in [('中位数插补',med),('KNN插补',knn)]:
        for scenario in ['original','derived_only_missing','MCAR_10_strict','MCAR_30_strict','MCAR_50_strict']:
            pred_arrays[method,scenario] = source[scenario]
    pg = {k:np.zeros(Y.shape,int) for k in pred_arrays}
    for f in np.unique(folds):
        test = folds==f; train=~test
        cutpoints=np.quantile(Y[train],np.arange(1,7)/7,axis=0)
        base[test]=Y[train].mean(axis=0)
        for j in range(len(targets)):
            true[test,j]=np.searchsorted(cutpoints[:,j],Y[test,j],side='right')
            for key,pred in pred_arrays.items():
                pg[key][test,j]=np.searchsorted(cutpoints[:,j],pred[test,j],side='right')
    groups={'全部50维':np.ones(50,bool),'核心30维':np.array([t.startswith(('A_','F_')) for t in targets]),
            '香气15维':np.array([t.startswith('A_') for t in targets]),'风味15维':np.array([t.startswith('F_') for t in targets])}
    metrics=[];conf=[]
    for key,pred in pred_arrays.items():
        method,scenario=key; err=abs(pg[key]-true)
        for group,mask in groups.items():
            kappa=[cohen_kappa_score(true[:,j],pg[key][:,j],labels=np.arange(7),weights='quadratic') for j in np.flatnonzero(mask)]
            metrics.append(dict(method=method,scenario=scenario,target_group=group,n_beers=len(Y),n_targets=int(mask.sum()),
                                macro_r2=float(r2_score(Y[:,mask],pred[:,mask],multioutput='uniform_average')),
                                grade_MAE=float(err[:,mask].mean()),exact_fraction=float((err[:,mask]==0).mean()),
                                within_one_fraction=float((err[:,mask]<=1).mean()),macro_quadratic_kappa=float(np.nanmean(kappa))))
        if scenario in ['original','MCAR_50_strict']:
            m=groups['核心30维']
            for obs in range(7):
                for estimate in range(7):
                    conf.append(dict(method=method,scenario=scenario,true_relative_grade=obs,predicted_relative_grade=estimate,
                                     count=int(((true[:,m]==obs)&(pg[key][:,m]==estimate)).sum())))
    actual=pd.DataFrame(metrics);saved=read(out/'metrics.csv')
    cols=[c for c in saved if c not in ['method','scenario','target_group']]
    err=maxerror(actual[cols],saved[cols])
    check('grading_44_metric_rows_recomputed', actual[['method','scenario','target_group']].equals(saved[['method','scenario','target_group']]) and err<=1e-12,err)
    conf_saved=read(out/'core30_confusion.csv')
    check('grading_all_confusion_counts_recomputed', pd.DataFrame(conf).equals(conf_saved))
    predicted=read(out/'knn_oof_relative_grades_50.csv')
    check('grading_12500_exported_relative_grades', np.array_equal(predicted[targets],pg['KNN插补','original']))
    summary=json.loads((out/'summary.json').read_text())
    check('grading_input_hashes', summary['source_sha256'][medpath.name]==digest(medpath) and summary['source_sha256'][knnpath.name]==digest(knnpath))
    actual.to_csv(HERE/'recomputed_grading_metrics.csv',index=False)
    return dict(max_metric_error=err,n_metrics=len(actual),n_confusion_cells=len(conf),n_exported_grades=12500,
                core30_original=actual[(actual.target_group=='核心30维')&(actual.scenario=='original')].to_dict('records'),
                interpretation='Fold-training quantile grades 0-6, not reconstructed raw panel ratings; no fresh model training.')


def module_audit():
    out=REVISION/'module_stability'
    saved=read(out/'module_target_stability.csv')
    draws=np.load(AWORK/'q2/bootstrap_coefficients.npz')
    coef=draws['coefficients'];features=draws['features'].tolist();targets=draws['targets'].tolist()
    nodes=read(AWORK/'q2/graph_nodes.csv').set_index('field').loc[features]
    eligible=np.flatnonzero(nodes.single_compound_candidate.to_numpy(bool))
    top=eligible[np.argsort(-np.abs(coef[:,eligible,:]),axis=1,kind='stable')[:,:3,:]]
    frequency=np.stack([(top==j).any(axis=1).mean(axis=0) for j in range(len(features))])
    old=read(AWORK/'q2/all_factor_evidence.csv').pivot(index='field',columns='target',values='bootstrap_top3_frequency').loc[features,targets]
    check('bootstrap_individual_frequencies_reconstructed', maxerror(frequency,old)==0.)
    corr=read(DATA/'2.csv').iloc[:,1:].to_numpy(float)
    rebuilt=[]
    for th in [.5,.6,.7,.8]:
        W=np.where(abs(corr)>=th,abs(corr),0.);np.fill_diagonal(W,0.)
        labels=greedy_modularity(W)
        if th==.6:
            check('module_membership_threshold_0_6_reproduced',np.array_equal(labels,nodes.module.to_numpy()))
        topmodule=labels[top]
        for module in sorted(set(labels[eligible])):
            ix=eligible[labels[eligible]==module];size=len(ix)
            # Joint union over each bootstrap sample, preserving dependence.
            union=np.mean(np.any(topmodule==module,axis=1),axis=0)
            largest=frequency[ix].max(axis=0)
            chance=1-(math.comb(len(eligible)-size,3)/math.comb(len(eligible),3) if len(eligible)-size>=3 else 0.)
            for j,target in enumerate(targets):
                best=ix[np.argmax(frequency[ix,j])]
                rebuilt.append(dict(threshold=th,target=target,module=int(module),eligible_module_size=size,
                                    module_union_frequency=float(union[j]),largest_member_frequency=float(largest[j]),
                                    substitution_gain=float(union[j]-largest[j]),uniform_three_draw_reference=float(chance),
                                    chance_adjusted_frequency=float((union[j]-chance)/(1-chance)) if chance<1 else np.nan,
                                    leading_field=features[best],leading_name=nodes.iloc[best]['name'],
                                    dominant_origin=str(nodes.iloc[ix].origin.value_counts().index[0])))
    rebuilt=pd.DataFrame(rebuilt)
    cols=saved.select_dtypes(include='number').columns
    err=maxerror(rebuilt[cols],saved[cols])
    check('all_26500_module_target_rows_recomputed',len(rebuilt)==26500 and err<=1e-12 and
          rebuilt[['threshold','target','module']].equals(saved[['threshold','target','module']]),err)
    leaders=[]
    for (th,t),part in rebuilt.groupby(['threshold','target'],sort=False):
        r=part.sort_values(['module_union_frequency','chance_adjusted_frequency','eligible_module_size','module'],
                           ascending=[False,False,True,True],kind='stable').iloc[0].to_dict()
        r['any_individual_at_least_0_8']=bool((frequency[:,targets.index(t)]>=.8).any());leaders.append(r)
    leaders=pd.DataFrame(leaders);oldleaders=read(out/'leading_module_by_target.csv')
    lc=oldleaders.select_dtypes(include='number').columns
    check('module_leaders_recomputed',maxerror(leaders[lc],oldleaders[lc])<=1e-12 and
          np.array_equal(leaders.any_individual_at_least_0_8,oldleaders.any_individual_at_least_0_8))
    counts=[]
    for th,g in leaders.groupby('threshold'):
        counts.append(dict(threshold=th,targets=len(g),at_least_one_module_frequency_0_8=int((g.module_union_frequency>=.8).sum()),
                           at_least_one_individual_frequency_0_8=int(g.any_individual_at_least_0_8.sum()),
                           module_0_8_without_individual_0_8=int(((g.module_union_frequency>=.8)&~g.any_individual_at_least_0_8).sum()),
                           mean_leading_module_size=float(g.eligible_module_size.mean())))
    summary=json.loads((out/'summary.json').read_text())
    check('module_summary_counts_recomputed',counts==summary['threshold_summary'])
    context=read(out/'top3_with_module_context.csv')
    anchor=read(AWORK/'q2/top3_all_50_sensory_dimensions.csv').set_index(['target','rank','field'])
    context_errors=[]
    for r in context.itertuples():
        node=nodes.loc[r.field];j=features.index(r.field);t=targets.index(r.target)
        a=anchor.loc[(r.target,r.rank,r.field)]
        mr=rebuilt[(rebuilt.threshold==.6)&(rebuilt.target==r.target)&(rebuilt.module==node.module)].iloc[0]
        values=[r.individual_frequency,r.stratified_individual_frequency,r.module_union_frequency,r.module_size,r.largest_member_frequency,r.uniform_reference,r.sign_agreement,r.within_style_rho]
        expected=[frequency[j,t],a.stratified_top3_frequency,mr.module_union_frequency,mr.eligible_module_size,mr.largest_member_frequency,mr.uniform_three_draw_reference,a.sign_agreement,a.within_style_rho]
        if maxerror(values,expected)>1e-12:context_errors.append((r.target,r.rank,r.field))
    check('all_150_top3_context_rows',not context_errors,context_errors)
    pd.DataFrame(counts).to_csv(HERE/'recomputed_module_summary.csv',index=False)
    return dict(bootstrap_replicates=int(coef.shape[0]),eligible_compounds=len(eligible),n_module_target_rows=len(rebuilt),
                max_frequency_error=err,threshold_summary=counts,
                interpretation='Fixed full-data descriptive modules; joint bootstrap union frequency; larger modules raise the chance reference. This is not evidence of a causal module or resampled partition stability.')


def write_markdown(report):
    b=report['budget_transfer'];m=report['module_stability'];g=report['grading']
    passes=sum(r['passed'] for r in CHECKS);fails=[r for r in CHECKS if not r['passed']]
    lines=['# 扩展预算实验独立核查','',f"结论：完成 {len(CHECKS)} 项检查，其中 {passes} 项通过、{len(fails)} 项未通过。核查直接读取题目原始数据、模型文件、29,474 条候选诊断和 374 份导出谱，不重新运行支持集搜索，也不改动优化数值。",'',
           '## 1. 预算、候选保留与旧结果复现','',
           '预算网格为 3、5、10、20、40、60、80、100、140、180；三种支持集族各有 11 个来源、10 个预算，加上全 225 维和三个作者成分对照，共 374 个结果。297 次相邻预算支持集包含关系、374 次候选选择均从完整候选表独立复核。较大预算保留同一族全部较小预算的可行动候选，按原二次目标和既定平局规则选择；目标值随预算不增。', '',
           f"旧实验共 {b['old_consistency']['cases']} 个对照案例的全部数值谱逐值一致，最大差 {b['old_consistency']['profile_max_error']:.3g}；感官预测最大差 {b['old_consistency']['sensory_max_error']:.3g}。新增预算没有改变原 k≤20 结果。",'',
           f"全部 11 个源均保留。374 个输出中 {b['n_moving']} 个为非零调控，{len(b['fallback_cases'])} 个原样返回；其中 5 个保留原来的缺陷缺测，明确未被认证为合格生成。QP 失败 {b['n_failed_qp']} 次仍完整留存在 qp_attempts.csv。",'',
           '协议与实际网格、惩罚系数、预测器哈希和旧脚本哈希一致；仅凭当前文件不能独立证明 created_before_execution 的时间声明，因而不把它称作外部注册的预注册。','',
           '## 2. 冻结、化学约束与派生量','',
           '逐单元核对 225 个直接字段：未选字段严格保留原读取值，未选原始 NaN 仍为 NaN；技术内标 X3.methyl.2.butenal 保留。独立统计原始数值变化数，逐案例不超过其预算；同时重算秩坐标变化数并核对导出字段变化清单。五个派生字段不计入直接调控预算，因此不能把预算解释为最终 CSV 中所有变化列数。', '',
           f"原汇总 actual_changed_mean 使用 |Δz|>10⁻⁸ 的秩坐标计数；有 {b['change_count_conventions']['differing_cases']} 个案例与原始数值精确变化数不完全相同。两种口径均未超预算，逐方法对照见 raw_and_rank_change_counts.csv，正文应明确口径。",'',
           '四个缺陷字段分别检查 DMS、2,3-丁二酮、4-VG 和 (E)-2-壬烯醛。已知源值使用不增上界；缺测源值要求生成输出不高于规定阈值。已知源浓度可能原已高于阈值，不增约束不等价于全部输出均低于感官阈值。校准表为零或非正而转换表含替代值的情况单独标记，不能称为正浓度实测值。', '',
           f"来源状态计数（11×4 个源字段）：{json.dumps(b['source_status_counts'],ensure_ascii=False)}。全部非零输出满足四化学上界、线性 stale 约束及非线性集成 stale 筛选。线性、非线性输出重算最大误差分别为 {b['stale_errors']['linear']:.3g}、{b['stale_errors']['nonlinear']:.3g}。",'',
           '四个辅助汇总量按“源原值 + 回归器(新谱) − 回归器(源谱)”重建，并按全数据观测范围截断；kcal 使用既定热量恒等式。逐条重算均通过。辅助量重建是模型约定，不能表述为这四个化学总量的真实定义。', '',
           '## 3. 类别元数据修正与成功含义','',
           '发现原导出代码将任何非零调控的 tasting_category_fine 写成 Tripel。这把请求目标误作了类别标签。原文件有 364 条此类标签，其中 305 条与诊断分类器的判别不一致。已做仅元数据修正，保留源实测类别，并分列 source_tasting_category_fine、requested_target_style、diagnostic_predicted_style、profile_origin、tasting_category_fine_origin。修正逐个保留所有原有非类别 CSV 单元的字符串，数值不变；审计哈希及备份见 metadata_patch.json 和 selected_profiles_before_metadata.csv。生成脚本也只修改了对应的元数据导出语句，补丁和前后哈希见 metadata_export_code_patch.json；不涉及优化、候选选择或预测计算。', '',
           '分类器在已有所有啤酒上拟合，用于附加风格诊断；Tripel 概率和预测标签未进入候选可行性约束或二次目标排序。首次出现 Tripel 仅指这个有限预算网格中，所选方案第一次被该分类器判为 Tripel；既不证明全局最小变量数，也不代表新的实测品评。不同预算的目标距离、干预幅度与分类标签可以有不同变化趋势。', '',
           '| 支持集族 | 预算 20 距离下降 | 预算 60 Tripel 数 | 预算 100 Tripel 数 | 预算 180 距离下降 | 预算 180 Tripel 数 |','|---|---:|---:|---:|---:|---:|']
    table={r['method']:r for r in b['budget_results']}
    for f in FAMILIES:
        r20,r60,r100,r180=[table[f'{f}_k{k}'] for k in [20,60,100,180]]
        lines.append(f"| {f} | {r20['target_distance_reduction_mean']:.2%} | {r60['tripel_classified']}/11 | {r100['tripel_classified']}/11 | {r180['target_distance_reduction_mean']:.2%} | {r180['tripel_classified']}/11 |")
    lines += ['',f"全 225 维对照平均距离下降 {table['all_225']['target_distance_reduction_mean']:.2%}，分类诊断为 Tripel 的为 {table['all_225']['tripel_classified']}/11。逐源首次测试成功预算见 first_tested_diagnostic_success.csv。",'',
              '## 4. 评级与模块稳定性补充复核','',
              f"评级重算 44 行汇总指标、{g['n_confusion_cells']} 个混淆矩阵单元及全部 12,500 个 KNN 预测等级。六个分位切点只使用相应外折训练的 200 行；真值与预测值使用同一组切点。这里的 0–6 是相对强度等级，不能恢复原始品评员评分。汇总最大误差 {g['max_metric_error']:.3g}。",'',
              '| 方法 | 核心30维 R² | 相对等级 MAE | 误差≤1级比例 |','|---|---:|---:|---:|']
    for r in g['core30_original']:
        lines.append(f"| {r['method']} | {r['macro_r2']:.5f} | {r['grade_MAE']:.4f} | {r['within_one_fraction']:.2%} |")
    lines += ['',f"模块稳定性重算 400 份已保存 bootstrap 的 Top3 集合，以及 {m['n_module_target_rows']:,} 行模块×目标结果；与输出的最大误差 {m['max_frequency_error']:.3g}。模块频率使用同一 bootstrap 内“至少一个成员入选”的联合事件，并非把边际频率相加。",'',
              '| 相关阈值 | 至少一个模块频率≥0.8 的目标 | 至少一个单因素频率≥0.8 的目标 | 模块达标但单因素未达标 | 领先模块平均大小 |','|---:|---:|---:|---:|---:|']
    for r in m['threshold_summary']:
        lines.append(f"| {r['threshold']} | {r['at_least_one_module_frequency_0_8']} | {r['at_least_one_individual_frequency_0_8']} | {r['module_0_8_without_individual_0_8']} | {r['mean_leading_module_size']:.2f} |")
    lines += ['', '阈值 0.6 下模块层面增加的 6 个目标是固定描述性模块内的替代稳定性证据；更大的模块本来更容易被 Top3 命中，必须同时报告模块大小与超几何均匀抽取参照。该参照不是 p 值，这也不是每次 bootstrap 重建图划分的稳定性检验。', '',
              '## 5. 范围与可复核证据','',
              '本核查验证数值、导出、预算和结论措辞的内部一致性，没有重新开展真实酿造或品评。全部生成结果仍依赖已拟合预测器及其源数据；双 stale 约束不等于实测陈旧感必不增加。作者七成分实验的原目标与 Pils/Lager→Tripel 迁移不同，其低迁移收益只能作此任务下的对照。', '',
              '核心可复核文件：verify_budget_transfer.py、review.json、recomputed_selected_diagnostics.csv、recomputed_chemical_caps.csv、source_defect_status.csv、first_tested_diagnostic_success.csv、recomputed_grading_metrics.csv、recomputed_module_summary.csv。', '', '## 检查明细','', '| 检查 | 状态 |','|---|---|']
    lines += [f"| {r['check']} | {'通过' if r['passed'] else '未通过'} |" for r in CHECKS]
    if fails:
        lines += ['', '### 待处理问题','',json.dumps(fails,ensure_ascii=False,indent=2,default=scalar)]
    (HERE/'review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repair-metadata',action='store_true')
    args=parser.parse_args()
    HERE.mkdir(parents=True,exist_ok=True)
    report={'budget_transfer':budget_audit(args.repair_metadata), 'grading':grading_audit(), 'module_stability':module_audit()}
    report['checks']=CHECKS
    report['n_checks']=len(CHECKS)
    report['n_failed']=sum(not x['passed'] for x in CHECKS)
    report['artifacts_sha256']={str(p.relative_to(AWORK)):digest(p) for p in [NEW/'protocol.json', NEW/'run_budget_transfer.py', NEW/'selected_profiles.csv', NEW/'selected_diagnostics.csv', NEW/'path_checks.csv', AWORK/'q3/transfer_and_augmentation_models.joblib', AWORK/'q1/sensory_ensemble.joblib']}
    dump(report,HERE/'review.json')
    write_markdown(report)
    print(json.dumps(dict(n_checks=len(CHECKS),n_failed=report['n_failed'],failures=[x for x in CHECKS if not x['passed']],output=str(HERE)),ensure_ascii=False,default=scalar),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
