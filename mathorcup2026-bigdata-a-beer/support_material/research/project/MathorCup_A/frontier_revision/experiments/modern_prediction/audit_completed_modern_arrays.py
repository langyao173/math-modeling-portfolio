"""Read-only, relocation-safe audit of completed modern prediction artifacts.

This never imports a predictor, refits a model, overwrites a protocol, or writes
an output file. Historical absolute paths remain intact; their recorded hashes
are checked against the corresponding files in this copy's relative layout.
Training runners deliberately retain their stricter frozen-spec comparisons.
"""
from pathlib import Path
import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import hashlib
import json
import platform
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def locate_project():
    """Support the development layout and the final BASE/frontier_revision tree."""
    required = ('data/A/1.csv', 'data/A/4.csv', 'A_work/q1/fold_assignments.csv')
    for ancestor in (HERE, *HERE.parents):
        for candidate in (ancestor / 'project' / 'MathorCup_A', ancestor):
            if all((candidate / name).is_file() for name in required):
                return candidate
    raise FileNotFoundError('No ancestor contest project with chemistry, sensory data and frozen folds.')


BASE = locate_project()
SCENARIOS = ('derived_only_missing', 'MCAR_10_strict', 'MCAR_30_strict', 'MCAR_50_strict')
DERIVED = ('kcalperc', 'esters_sum', 'aroma_hops_sum', 'acids_sum', 'sulfur_sum')
SEED = 20261023


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def assert_hash(path, expected):
    actual = sha(path)
    assert actual == expected, f'Hash mismatch: {path}'
    return actual


def array_equal(a, b, label):
    assert np.array_equal(a, b, equal_nan=True), f'Array mismatch: {label}'


def inputs():
    x = pd.read_csv(BASE / 'data/A/1.csv').sort_values('beer').reset_index(drop=True)
    y = pd.read_csv(BASE / 'data/A/4.csv').set_index('beer').loc[x.beer].reset_index()
    folds = pd.read_csv(BASE / 'A_work/q1/fold_assignments.csv').set_index('beer').loc[x.beer, 'fold'].to_numpy()
    X, Y = x.iloc[:, 3:].to_numpy(float), y.iloc[:, 3:].to_numpy(float)
    assert X.shape == (250, 231) and Y.shape == (250, 50)
    assert np.isfinite(Y).all() and set(folds) == set(range(5))
    assert all(np.count_nonzero(folds == f) == 50 for f in range(5))
    return x, y, X, Y, folds


def common_spec(directory, weight):
    spec = read_json(directory / 'experiment_spec.json')
    evidence = {'spec_sha256': sha(directory / 'experiment_spec.json')}
    for filename, expected in spec['data_sha256'].items():
        assert_hash(BASE / 'data/A' / filename, expected)
    expected_fold = spec.get('fold_sha256', spec.get('folds_sha256'))
    assert_hash(BASE / 'A_work/q1/fold_assignments.csv', expected_fold)
    evidence['weight_sha256'] = assert_hash(weight, spec['checkpoint_sha256'])
    for filename, expected in spec.get('implementation_sha256', {}).items():
        assert_hash(HERE / filename, expected)
    evidence['historical_platform'] = spec.get('platform')
    evidence['historical_cpu_count'] = spec.get('cpu_count')
    return spec, evidence


def check_indices(checkpoint, fold, folds):
    tr, te = checkpoint['train_indices'], checkpoint['test_indices']
    array_equal(tr, np.flatnonzero(folds != fold), 'train indices')
    array_equal(te, np.flatnonzero(folds == fold), 'test indices')
    assert not np.intersect1d(tr, te).size
    return tr, te


def check_oof_identity(archive, x, y, Y, folds):
    array_equal(archive['Y'], Y, 'OOF Y')
    array_equal(archive['beer'], x.beer.to_numpy(), 'OOF beers')
    array_equal(archive['fold'], folds, 'OOF folds')
    assert archive['targets'].tolist() == y.columns[3:].tolist()


def score(Y, prediction):
    assert prediction.shape == Y.shape and np.isfinite(prediction).all()
    error = np.square(Y - prediction)
    r2 = 1 - error.sum(0) / np.square(Y - Y.mean(0)).sum(0)
    return dict(macro_r2=float(r2.mean()), core30_r2=float(r2[:30].mean()),
                aroma15_r2=float(r2[:15].mean()), flavour15_r2=float(r2[15:30].mean()),
                other20_r2=float(r2[30:].mean()), macro_rmse=float(np.sqrt(error.mean(0)).mean()),
                positive_r2=int(np.count_nonzero(r2 > 0)))


def metric_row(frame, name, calculated):
    rows = frame.loc[frame.model == name]
    assert len(rows) == 1
    row = rows.iloc[0]
    for key, value in calculated.items():
        assert np.isclose(float(row[key]), value, rtol=1e-11, atol=1e-12), (name, key)


def pca_coordinates(checkpoint, Ytrain):
    mean, scale = checkpoint['target_mean'], checkpoint['target_scale']
    np.testing.assert_allclose(mean, Ytrain.mean(0), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(scale, Ytrain.std(0, ddof=1), rtol=1e-12, atol=1e-13)
    standardized = (Ytrain - mean) / scale
    V = checkpoint['target_components'].T
    assert V.shape == (50, 16)
    np.testing.assert_allclose(V.T @ V, np.eye(16), rtol=1e-10, atol=1e-10)
    centered = standardized - standardized.mean(0)
    covariance = centered.T @ centered / (len(Ytrain) - 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, -16:]
    np.testing.assert_allclose(V @ V.T, principal @ principal.T, rtol=1e-8, atol=1e-9)
    return mean, scale, V, standardized.mean(0)


def tabicl(data):
    x, y, X, Y, folds = data
    spec, evidence = common_spec(HERE, HERE / 'models/tabicl-regressor-v2-20260212.ckpt')
    status = read_json(HERE / 'status.json')
    assert status['completed_model_folds'] == status['total_model_folds'] == 25
    oof = np.load(HERE / 'oof_arrays.npz', allow_pickle=False)
    check_oof_identity(oof, x, y, Y, folds)
    metrics = pd.read_csv(HERE / 'model_comparison.csv')
    candidates, checkpoints = {}, []
    for view in ('rank_pca64_y16', 'raw231_y16', 'raw231_y50'):
        for f in range(5):
            path = HERE / 'checkpoints' / f'{view}_fold{f}_complete.npz'
            a = np.load(path, allow_pickle=False)
            tr, te = check_indices(a, f, folds)
            if view.endswith('y16'):
                mean, scale, V, center = pca_coordinates(a, Y[tr])
                reconstructed = (a['latent_prediction'] @ V.T + center) * scale + mean
                np.testing.assert_allclose(a['prediction_no_residual'], reconstructed, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(a['prediction_ridge_residual'], reconstructed + a['residual_prediction'], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose((a['residual_prediction'] / scale) @ V, 0., rtol=0, atol=1e-10)
                outputs = {f'{view}_no_residual': a['prediction_no_residual'],
                           f'{view}_ridge_residual': a['prediction_ridge_residual']}
            else:
                outputs = {'raw231_y50_native': a['prediction_native']}
            for name, prediction in outputs.items():
                assert prediction.shape == (50, 50) and np.isfinite(prediction).all()
                array_equal(oof[name][te], prediction, f'{name} fold {f} OOF assembly')
                candidates[name] = True
            assert np.isfinite(a['timing']).all()
            checkpoints.append(dict(file=str(path.relative_to(HERE)), sha256=sha(path)))
    assert len(candidates) == 5
    scores = {}
    for name in candidates:
        scores[name] = score(Y, oof[name])
        metric_row(metrics, name, scores[name])
    evidence.update(complete=True, checkpoint_count=15, model_folds=25, scores=scores,
                    oof_sha256=sha(HERE / 'oof_arrays.npz'), checkpoints=checkpoints)
    return evidence


def tabdpt(data):
    x, y, X, Y, folds = data
    directory = HERE / 'tabdpt'
    spec, evidence = common_spec(directory, directory / 'models/tabdpt1_3.safetensors')
    assert read_json(directory / 'status.json')['completed_folds'] == 5
    oof = np.load(directory / 'oof_arrays.npz', allow_pickle=False)
    check_oof_identity(oof, x, y, Y, folds)
    checkpoints = []
    for f in range(5):
        path = directory / 'checkpoints' / f'fold{f}_complete.npz'
        a = np.load(path, allow_pickle=False)
        tr, te = check_indices(a, f, folds)
        assert a['prediction'].shape == (50, 50) and np.isfinite(a['prediction']).all()
        assert a['input_projection'].shape == (231, 128) and np.isfinite(a['input_projection']).all()
        assert a['timing'].shape == (50, 4) and np.isfinite(a['timing']).all()
        array_equal(a['timing'][:, 0], np.arange(50), 'TabDPT target coverage')
        array_equal(oof['prediction'][te], a['prediction'], f'TabDPT fold {f} OOF assembly')
        checkpoints.append(dict(file=str(path.relative_to(HERE)), sha256=sha(path)))
    calculated = score(Y, oof['prediction'])
    metric_row(pd.read_csv(directory / 'model_comparison.csv'), spec['model'], calculated)
    evidence.update(complete=True, checkpoint_count=5, scalar_target_folds=250, scores=calculated,
                    oof_sha256=sha(directory / 'oof_arrays.npz'), checkpoints=checkpoints,
                    boundary='Array reconstruction and provenance audit; no new neural prediction or assertion of pretraining non-overlap.')
    return evidence


def missingness(data):
    x, y, X, Y, folds = data
    directory = HERE / 'missingness'
    spec, evidence = common_spec(directory, HERE / 'models/tabicl-regressor-v2-20260212.ckpt')
    assert read_json(directory / 'status.json').get('complete') is True
    references = {
        'run_mask_augmentation.py': HERE.parent / 'missingness_augmentation/run_mask_augmentation.py',
        'strict_knn_missingness.py': BASE / 'A_work/audit/model_validation/strict_knn_missingness.py',
        'strict_missingness.py': BASE / 'A_work/audit/model_validation/strict_missingness.py',
    }
    for historical, expected in spec['mask_reference_sha256'].items():
        assert_hash(references[Path(historical).name], expected)
    oof = np.load(directory / 'oof_predictions.npz', allow_pickle=False)
    check_oof_identity(oof, x, y, Y, folds)
    hidden = [x.columns[3:].tolist().index(name) for name in DERIVED]
    checkpoints, max_error = [], 0.
    for f in range(5):
        path = directory / 'checkpoints' / f'fold{f}_complete.npz'
        a = np.load(path, allow_pickle=False)
        tr, te = check_indices(a, f, folds)
        masks = np.load(directory / 'checkpoints' / f'fold{f}_masks.npz', allow_pickle=False)
        check_indices(masks, f, folds)
        mean, scale, V, center = pca_coordinates(a, Y[tr])
        np.testing.assert_allclose(a['target_pca_mean'], center, rtol=1e-12, atol=1e-12)
        original = np.load(HERE / 'checkpoints' / f'raw231_y16_fold{f}_complete.npz', allow_pickle=False)
        array_equal(a['target_components'], original['target_components'], 'same complete-input target encoding')
        assert a['latent_prediction'].shape == (4, 50, 16) and np.isfinite(a['latent_prediction']).all()
        for i, scenario in enumerate(SCENARIOS):
            mask = np.isnan(X[te]).copy()
            if scenario.startswith('MCAR_'):
                level = int(scenario.split('_')[1])
                mask |= np.random.default_rng(SEED + 1000*f + level).random((50, 231)) < level/100
            mask[:, hidden] = True
            array_equal(masks[scenario], mask, f'fold {f} {scenario} mask')
            reconstructed = (a['latent_prediction'][i] @ V.T + center) * scale + mean
            error = float(np.max(np.abs(a[scenario] - reconstructed)))
            max_error = max(max_error, error)
            np.testing.assert_allclose(a[scenario], reconstructed, rtol=1e-12, atol=1e-12)
            array_equal(oof[f'tabicl_raw231_y16__{scenario}'][te], a[scenario], 'missing OOF assembly')
        checkpoints.append(dict(file=str(path.relative_to(HERE)), sha256=sha(path)))
    baseline_dir = BASE / 'A_work/audit/model_validation'
    for name, filename in (('old_knn', 'strict_knn_missingness_oof.npz'), ('old_median', 'strict_missingness_oof.npz')):
        baseline = np.load(baseline_dir / filename, allow_pickle=False)
        for scenario in ('original',) + SCENARIOS:
            array_equal(oof[f'{name}__{scenario}'], baseline[scenario], f'{name} {scenario} baseline')
    main_oof = np.load(HERE / 'oof_arrays.npz', allow_pickle=False)
    array_equal(oof['tabicl_raw231_y16__original'], main_oof['raw231_y16_no_residual'], 'original reference')
    comparison = pd.read_csv(directory / 'comparison.csv')
    truth = np.empty_like(Y, dtype=int)
    cuts = {}
    for f in range(5):
        cuts[f] = np.quantile(Y[folds != f], np.arange(1, 7)/7, axis=0)
        truth[folds == f] = np.stack([np.searchsorted(cuts[f][:, j], Y[folds == f, j], side='right') for j in range(50)], axis=1)
    for name in ('tabicl_raw231_y16', 'old_knn', 'old_median'):
        for scenario in ('original',) + SCENARIOS:
            prediction = oof[f'{name}__{scenario}']
            calculated = score(Y, prediction)
            grade = np.empty_like(Y, dtype=int)
            for f in range(5):
                grade[folds == f] = np.stack([np.searchsorted(cuts[f][:, j], prediction[folds == f, j], side='right') for j in range(50)], axis=1)
            error = np.abs(grade - truth)
            for group, ix, metric_name in (('all50', slice(None), 'macro_r2'), ('core30', slice(30), 'core30_r2')):
                rows = comparison.loc[(comparison.model == name) & (comparison.scenario == scenario) & (comparison.targets == group)]
                assert len(rows) == 1
                row = rows.iloc[0]
                values = dict(r2=calculated[metric_name], grade_mae=error[:, ix].mean(),
                              grade_exact=(error[:, ix] == 0).mean(), grade_within_one=(error[:, ix] <= 1).mean())
                for key, value in values.items():
                    assert np.isclose(row[key], value, rtol=1e-11, atol=1e-12), (name, scenario, group, key)
    evidence.update(complete=True, checkpoint_count=5, scenarios=4, masks_verified=20,
                    reconstruction_max_abs_error=max_error,
                    oof_sha256=sha(directory / 'oof_predictions.npz'), checkpoints=checkpoints)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('tabicl', 'tabdpt', 'missingness', 'all'), default='all')
    parser.add_argument('--require-final-ready', action='store_true')
    args = parser.parse_args()
    data = inputs()
    phases = ('tabicl', 'tabdpt', 'missingness') if args.phase == 'all' else (args.phase,)
    report = dict(read_only=True, no_predictor_import_or_model_refit=True,
                  historical_protocols_unchanged=True, current_platform=platform.platform(),
                  resolved_project=str(BASE), phases={})
    auditors = dict(tabicl=tabicl, tabdpt=tabdpt, missingness=missingness)
    for phase in phases:
        report['phases'][phase] = auditors[phase](data)
    ready_path = HERE / 'FINAL_READY.json'
    if args.require_final_ready or ready_path.exists():
        ready = read_json(ready_path)
        assert ready['state'] == 'all_registered_modern_prediction_work_complete'
        for relative, expected in ready['result_sha256'].items():
            assert_hash(HERE / relative, expected)
        manifest = read_json(HERE / 'final_artifact/manifest.json')
        assert_hash(HERE / 'final_artifact/model_context.npz', manifest['context_sha256'])
        assert_hash(HERE / 'predict_tabicl_csv.py', manifest['inference_script_sha256'])
        assert_hash(HERE / 'final_artifact' / manifest['checkpoint']['relative_path'], manifest['checkpoint']['sha256'])
        report['final_ready_verified'] = True
        report['final_ready_sha256'] = sha(ready_path)
    else:
        report['final_ready_verified'] = False
    report['complete'] = True
    report['scope'] = ('Read-only reuse audit of completed arrays and their recorded provenance, using the current relative project layout. '
                       'It neither changes nor substitutes for the frozen training protocols and does not certify unrecorded source history.')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
