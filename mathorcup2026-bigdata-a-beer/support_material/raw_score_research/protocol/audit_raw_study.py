"""Independent read-only targets, leakage contracts, metrics and paired intervals.

No predictive learner or study implementation is imported. Targets are rebuilt
from original records; all metric arrays are recomputed directly from stored
out-of-fold predictions. This audit neither fits nor selects a new model.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def canonical_targets(raw, mapping, beer, targets):
    frame = pd.read_csv(raw, encoding='cp1252').rename(columns={
        'A_ester_fruity': 'A_esters_fruity', 'F_ester_fruity': 'F_esters_fruity'})
    links = pd.read_csv(mapping).set_index('raw_beer_id')['beer']
    frame['mapped'] = frame['beer_name'].map(links)
    assert frame['mapped'].notna().all()
    rating = frame[list(targets)].to_numpy(dtype=int)
    assert ((rating >= 0) & (rating <= 6)).all()
    q = np.empty((len(beer), len(targets), 7))
    qt = np.empty_like(q)
    records, tasters = [], []
    for row, identifier in enumerate(beer):
        subset = frame.loc[frame['mapped'] == identifier]
        values = subset[list(targets)].to_numpy(dtype=int)
        q[row] = (values[:, :, None] == np.arange(7)).mean(axis=0)
        individual = []
        for _, part in subset.groupby('taster', sort=True):
            values = part[list(targets)].to_numpy(dtype=int)
            individual.append((values[:, :, None] == np.arange(7)).mean(axis=0))
        qt[row] = np.mean(individual, axis=0)
        records.append(len(subset))
        tasters.append(len(individual))
    return frame, q, qt, np.array(records), np.array(tasters)


def point_statistics(y, prediction):
    square = (prediction-y)**2
    variance = y.var(axis=0)
    return dict(mean_mae=float(np.abs(prediction-y).mean()),
                pooled_mean_rmse=float(np.sqrt(square.mean())),
                macro_mean_rmse=float(np.sqrt(square.mean(axis=0)).mean()),
                macro_r2=float(np.mean(1-square.mean(axis=0)/variance)))


def distribution_statistics(q, p):
    f = np.cumsum(p, axis=-1)[..., :-1]
    g = np.cumsum(q, axis=-1)[..., :-1]
    # Direct expansion of expected individual-rating RPS, independently of the
    # study's distance-plus-constant implementation.
    rps_row_target = (f*f-2*f*g+g).mean(axis=-1)
    cdf_error = ((f-g)**2).mean()
    panel_term = (g*(1-g)).mean()
    np.testing.assert_allclose(rps_row_target.mean(), cdf_error+panel_term,
                               rtol=0, atol=1e-14)
    p_log = np.maximum(p, 1e-6)
    p_log /= p_log.sum(axis=-1, keepdims=True)
    median = np.argmax(np.cumsum(p, axis=-1) >= .5, axis=-1)
    distance = np.abs(np.arange(7)-median[..., None])
    brier = (np.sum(p*p-2*p*q, axis=-1)+1).mean()
    return dict(rps=float(rps_row_target.mean()), cdf_mse=float(cdf_error),
                irreducible_rps=float(panel_term),
                brier=float(brier),
                nll=float(-(q*np.log(p_log)).sum(axis=-1).mean()),
                individual_grade_mae=float((q*distance).sum(axis=-1).mean()),
                individual_grade_exact=float((q*(distance == 0)).sum(axis=-1).mean()),
                individual_grade_within_one=float((q*(distance <= 1)).sum(axis=-1).mean()))


def assert_columns(actual, expected, keys, counters):
    columns = [name for name in expected.columns if name not in keys and name in actual.columns]
    merged = actual.merge(expected, on=keys, suffixes=('_stored', '_recomputed'), validate='one_to_one')
    assert len(merged) == len(actual) == len(expected)
    maximum = 0.
    for name in columns:
        left = merged[name+'_stored'].to_numpy(float)
        right = merged[name+'_recomputed'].to_numpy(float)
        assert np.array_equal(np.isnan(left), np.isnan(right)), name
        valid = np.isfinite(left) & np.isfinite(right)
        if valid.any():
            error = float(np.max(np.abs(left[valid]-right[valid])))
            maximum = max(maximum, error)
            np.testing.assert_allclose(left[valid], right[valid], rtol=0, atol=1e-12, err_msg=name)
            counters['numeric_values'] += int(valid.sum())
    return maximum


def paired_interval(vector, indices):
    boot = vector[indices].mean(axis=1)
    return float(vector.mean()), *np.quantile(boot, [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--official-x', type=Path, required=True)
    parser.add_argument('--official-y', type=Path, required=True)
    parser.add_argument('--legacy-research', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    input_files = [args.inputs, args.raw, args.mapping, args.official_x, args.official_y]
    input_files += sorted(path for path in args.results.iterdir() if path.is_file())
    before = {str(path): sha(path) for path in input_files}
    a = load(args.inputs)
    predictions = load(args.results/'predictions.npz')
    beer, targets, fold = a['beer'], a['targets'], a['fold']
    assert a['X'].shape == (250, 231)
    assert len(set(beer.tolist())) == 250 and np.array_equal(beer, np.sort(beer))
    assert np.array_equal(predictions['beer'], beer)
    assert np.array_equal(predictions['targets'], targets)
    assert np.array_equal(predictions['fold'], fold)
    raw, q, qt, counts, taster_counts = canonical_targets(args.raw, args.mapping, beer, targets)
    np.testing.assert_array_equal(q, a['raw_probability'])
    np.testing.assert_allclose(qt, a['taster_probability'], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(q @ np.arange(7), a['raw_mean'])
    np.testing.assert_allclose(qt @ np.arange(7), a['taster_mean'], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(counts, a['record_count'])
    np.testing.assert_array_equal(taster_counts, a['taster_count'])
    np.testing.assert_array_equal(raw['mapped'], a['records_beer'])
    np.testing.assert_array_equal(raw['taster'], a['records_taster'])
    np.testing.assert_array_equal(raw[list(targets)], a['records_scores'])
    official_x = pd.read_csv(args.official_x).set_index('beer').loc[beer]
    official_y = pd.read_csv(args.official_y).set_index('beer').loc[beer]
    np.testing.assert_allclose(official_x[list(a['features'])], a['X'], rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(official_y[list(targets)], a['Yz'], rtol=0, atol=0)
    outer = load(args.legacy_research/'prediction/fusion_replay_v1/outer_predictions.npz')
    np.testing.assert_array_equal(beer, outer['beer'])
    np.testing.assert_array_equal(fold, outer['fold'])
    np.testing.assert_allclose(a['frozen_outer_z'], outer['fixed_half_y16_native50'], rtol=0, atol=1e-14)
    np.testing.assert_allclose(a['frozen_groupwise_z'], outer['inner_groupwise_y16_native50'], rtol=0, atol=1e-14)
    nested = args.legacy_research/'checkpoint/unpacked/nested_selection'
    for outer_id in range(5):
        train, test = np.flatnonzero(fold != outer_id), np.flatnonzero(fold == outer_id)
        assert len(train) == 200 and len(test) == 50
        cache = load(nested/f'outer_{outer_id}/inner_oof.npz')
        np.testing.assert_array_equal(beer[train], cache['beer'])
        np.testing.assert_allclose(a['frozen_inner_z'][outer_id, train],
                                  .5*(cache['raw231_y16_no_residual']+cache['raw231_y50_native']),
                                  rtol=0, atol=1e-14)
        assert np.isnan(a['frozen_inner_z'][outer_id, test]).all()
        assert (a['inner_fold'][outer_id, test] == -1).all()
        for inner in range(3):
            component = load(nested/f'outer_{outer_id}/inner_{inner}/raw231_y16_complete.npz')
            expected = np.flatnonzero(a['inner_fold'][outer_id] == inner)
            np.testing.assert_array_equal(expected, component['test_indices'])
            assert not np.intersect1d(component['train_indices'], test).size
            assert not np.intersect1d(component['test_indices'], test).size
    # Every recorded base fit contains exactly its expected training beer and
    # raw-record IDs, not merely a count of records.
    fits = json.loads((args.results/'fit_log.json').read_text())
    for fit in fits:
        outer_id, inner_id = fit['outer_fold'], fit['inner_fold']
        expected = np.flatnonzero(fold != outer_id)
        if inner_id != -1:
            expected = expected[a['inner_fold'][outer_id, expected] != inner_id]
        np.testing.assert_array_equal(fit['training_beers'], beer[expected])
        raw_rows = np.flatnonzero(np.isin(a['records_beer'], beer[expected]))
        np.testing.assert_array_equal(fit['training_records'], raw_rows)
        assert not np.intersect1d(raw_rows, np.flatnonzero(np.isin(a['records_beer'], beer[fold == outer_id]))).size
    masks = load(args.results/'test_masks.npz')
    derived = [list(a['features']).index(name) for name in
               ('kcalperc', 'esters_sum', 'aroma_hops_sum', 'acids_sum', 'sulfur_sum')]
    for outer_id in range(5):
        test = np.flatnonzero(fold == outer_id)
        original = a['X'][test]
        np.testing.assert_array_equal(masks[f'fold{outer_id}_complete'], np.isnan(original))
        expected = np.isnan(original).copy()
        expected[:, derived] = True
        np.testing.assert_array_equal(masks[f'fold{outer_id}_derived_only_missing'], expected)
        for level in (10, 30, 50):
            random = np.random.default_rng(20261023+1000*outer_id+level).random(original.shape)
            expected = np.isnan(original) | (random < level/100)
            expected[:, derived] = True
            np.testing.assert_array_equal(masks[f'fold{outer_id}_MCAR_{level}_strict'], expected)
    # Closed-form calibration replay; calibration parameters are reconstructed
    # only for checking stored values, not for a new study or method selection.
    affine_error = 0.
    for outer_id in range(5):
        train, test = np.flatnonzero(fold != outer_id), np.flatnonzero(fold == outer_id)
        saved = load(args.results/f'calibration_fold{outer_id}.npz')
        np.testing.assert_array_equal(saved['train_beer'], beer[train])
        np.testing.assert_array_equal(saved['test_beer'], beer[test])
        x, y = a['frozen_inner_z'][outer_id, train], a['raw_mean'][train]
        np.testing.assert_allclose(saved['inner_z'], x, rtol=0, atol=0)
        np.testing.assert_allclose(saved['outer_z'], a['frozen_outer_z'][test], rtol=0, atol=0)
        xc, yc = x-x.mean(axis=0), y-y.mean(axis=0)
        denominator = np.sum(xc*xc, axis=0)
        slope = np.maximum(np.sum(xc*yc, axis=0)/denominator, 0)
        intercept = y.mean(axis=0)-slope*x.mean(axis=0)
        estimate = np.clip(intercept+slope*a['frozen_outer_z'][test], 0, 6)
        np.testing.assert_allclose(saved['affine_slope'], slope, rtol=0, atol=1e-13)
        np.testing.assert_allclose(saved['affine_intercept'], intercept, rtol=0, atol=1e-13)
        error = float(np.max(np.abs(predictions['point__z_affine__complete'][test]-estimate)))
        affine_error = max(affine_error, error)
        assert error < 1e-12
    selected = json.loads((args.results/'selected_parameters.json').read_text())
    inner_scores = pd.read_csv(args.results/'inner_scores.csv')
    for choice in selected:
        grid = inner_scores.loc[(inner_scores.branch == choice['branch']) &
                                (inner_scores.outer_fold == choice['outer_fold'])]
        averages = grid.groupby('parameter').apply(
            lambda block: np.average(block.loss, weights=block.validation_rows), include_groups=False)
        assert float(averages.idxmin()) == choice['selected_parameter']
        for parameter, loss in choice['inner_losses'].items():
            np.testing.assert_allclose(averages.loc[float(parameter)], loss, rtol=0, atol=1e-12)
    groups = {'all50': np.arange(50), 'core30': np.arange(30), 'other20': np.arange(30, 50)}
    point = {tuple(key.split('__')[1:]): value for key, value in predictions.items() if key.startswith('point__')}
    probability = {tuple(key.split('__')[1:]): value for key, value in predictions.items() if key.startswith('probability__')}
    official = {tuple(key.split('__')[1:]): value for key, value in predictions.items() if key.startswith('official__')}
    rows, dimensions, zrows = [], [], []
    probability_min, probability_sum_error = 1., 0.
    for name, p in probability.items():
        assert np.isfinite(p).all()
        probability_min = min(probability_min, float(p.min()))
        probability_sum_error = max(probability_sum_error, float(np.max(np.abs(p.sum(axis=-1)-1))))
        assert p.min() >= -1e-14 and np.max(np.abs(p.sum(axis=-1)-1)) < 1e-13
        np.testing.assert_allclose(point[name], p @ np.arange(7), rtol=0, atol=1e-14)
    for weighting, truth_q in [('record_equal', q), ('taster_equal_sensitivity', qt)]:
        truth_m = truth_q @ np.arange(7)
        for (method, scenario), estimate in point.items():
            assert np.isfinite(estimate).all() and estimate.min() >= 0 and estimate.max() <= 6
            for group, js in groups.items():
                row = dict(method=method, scenario=scenario, target_weighting=weighting, targets=group,
                           **point_statistics(truth_m[:, js], estimate[:, js]))
                if (method, scenario) in probability:
                    row.update(distribution_statistics(truth_q[:, js], probability[method, scenario][:, js]))
                rows.append(row)
            if weighting == 'record_equal' and scenario == 'complete':
                for j, target in enumerate(targets):
                    row = dict(method=method, target=str(target),
                               **point_statistics(truth_m[:, j:j+1], estimate[:, j:j+1]))
                    if (method, scenario) in probability:
                        row.update(distribution_statistics(truth_q[:, j:j+1], probability[method, scenario][:, j:j+1]))
                    dimensions.append(row)
    for (method, scenario), estimate in official.items():
        for group, js in groups.items():
            zrows.append(dict(method=method, scenario=scenario, targets=group,
                              **point_statistics(a['Yz'][:, js], estimate[:, js])))
    counter = {'numeric_values': 0}
    error = max(assert_columns(pd.read_csv(args.results/'raw_metrics.csv'), pd.DataFrame(rows),
                               ['method', 'scenario', 'target_weighting', 'targets'], counter),
                assert_columns(pd.read_csv(args.results/'raw_per_target_metrics.csv'), pd.DataFrame(dimensions),
                               ['method', 'target'], counter),
                assert_columns(pd.read_csv(args.results/'official_metrics.csv'), pd.DataFrame(zrows),
                               ['method', 'scenario', 'targets'], counter))
    pd.DataFrame(rows).to_csv(out/'raw_metrics_independent.csv', index=False)
    pd.DataFrame(dimensions).to_csv(out/'raw_per_target_independent.csv', index=False)
    pd.DataFrame(zrows).to_csv(out/'official_metrics_independent.csv', index=False)
    # All complete-input pairs are reported, rather than choosing contrasts
    # after their signs or confidence intervals become known.
    rng = np.random.default_rng(2026102527)
    indices = rng.integers(0, len(beer), size=(2000, len(beer)))
    np.savez_compressed(out/'paired_bootstrap_indices.npz', beer=beer, indices=indices)
    comparisons = []
    for weighting, truth_q in [('record_equal', q), ('taster_equal_sensitivity', qt)]:
        truth_m = truth_q @ np.arange(7)
        for group, js in groups.items():
            means = {method: ((prediction[:, js]-truth_m[:, js])**2).mean(axis=1)
                     for (method, scenario), prediction in point.items() if scenario == 'complete'}
            for left, right in itertools.combinations(sorted(means), 2):
                vl, vr = means[left], means[right]
                boots = np.sqrt(vl[indices].mean(axis=1))-np.sqrt(vr[indices].mean(axis=1))
                estimate = float(np.sqrt(vl.mean())-np.sqrt(vr.mean()))
                lower, upper = np.quantile(boots, [.025, .975])
                comparisons.append(dict(target_weighting=weighting, targets=group, statistic='pooled_mean_rmse',
                                        left=left, right=right, estimate=estimate, lower95=lower, upper95=upper))
            scores = {}
            for (method, scenario), p in probability.items():
                if scenario != 'complete':
                    continue
                f, g = np.cumsum(p[:, js], axis=-1)[..., :-1], np.cumsum(truth_q[:, js], axis=-1)[..., :-1]
                scores[method] = (f*f-2*f*g+g).mean(axis=(1, 2))
            for left, right in itertools.combinations(sorted(scores), 2):
                estimate, lower, upper = paired_interval(scores[left]-scores[right], indices)
                comparisons.append(dict(target_weighting=weighting, targets=group, statistic='rps',
                                        left=left, right=right, estimate=estimate, lower95=lower, upper95=upper))
    paired = pd.DataFrame(comparisons)
    paired['scenario'] = 'complete'
    paired['n_beer'] = len(beer)
    paired['resamples'] = 2000
    paired['seed'] = 2026102527
    paired['interval_type'] = 'pointwise conditional paired beer bootstrap; no multiplicity adjustment'
    paired.to_csv(out/'paired_comparisons.csv', index=False)
    assert all(sha(path) == before[str(path)] for path in input_files)
    report = dict(status='PASS', raw_targets_independently_rebuilt=True,
                  official_and_frozen_prediction_identity=True,
                  cached_inner_training_scope_verified=True, base_fits_scope_verified=len(fits),
                  original_and_corruption_masks_verified=len(masks),
                  affine_closed_form_replay_max_error=affine_error,
                  metrics_numeric_values_recomputed=counter['numeric_values'], metrics_max_abs_error=error,
                  probability_minimum=probability_min, probability_sum_max_error=probability_sum_error,
                  paired_comparisons=len(paired), bootstrap_resamples=2000, bootstrap_seed=2026102527,
                  bootstrap_scope='All complete-input point and distribution method pairs; pointwise intervals conditional on stored predictions.',
                  new_predictive_model_fits=0, new_weight_searches=0, new_neural_inferences=0,
                  scoring='Expected individual-rating RPS from direct quadratic expansion; CDF part and panel term separately retained.',
                  input_sha256=before, source_files_unchanged=True, audit_script_sha256=sha(Path(__file__)))
    (out/'AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'input_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
