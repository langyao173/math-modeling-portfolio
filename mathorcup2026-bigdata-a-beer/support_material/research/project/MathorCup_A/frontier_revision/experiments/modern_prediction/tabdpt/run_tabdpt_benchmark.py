"""Frozen, same-fold TabDPT 1.3.1 comparator; public local weights only.


The --pilot mode measures one fixed target without calculating a validation score.
All 50 targets and five folds are required for the main benchmark.
"""
from __future__ import annotations

import os
THREADS = int(os.environ.get('BEER_TABDPT_THREADS', '6'))
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = str(THREADS)

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from tabdpt import TabDPTRegressor

HERE = Path(__file__).resolve().parent
MODERN = HERE.parent
BASE = HERE.parents[2] / 'project' / 'MathorCup_A'
WEIGHT = HERE / 'models' / 'tabdpt1_3.safetensors'
SEED = 20261023
MODEL_NAME = 'tabdpt13_raw231_pca128_y50'


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def atomic_npz(path, **arrays):
    temporary = path.with_name(path.stem + '_temporary.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_data():
    x = pd.read_csv(BASE / 'data/A/1.csv').sort_values('beer').reset_index(drop=True)
    y = pd.read_csv(BASE / 'data/A/4.csv').set_index('beer').loc[x.beer].reset_index()
    assert np.array_equal(x.beer, y.beer)
    folds = pd.read_csv(BASE / 'A_work/q1/fold_assignments.csv').set_index('beer').loc[x.beer, 'fold'].to_numpy()
    X = x.iloc[:, 3:].to_numpy(float)
    Y = y.iloc[:, 3:].to_numpy(float)
    assert X.shape == (250, 231) and Y.shape == (250, 50)
    for k, (_, te) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(X)):
        assert np.array_equal(np.flatnonzero(folds == k), te)
    previous = np.load(BASE / 'A_work/q1/oof_arrays.npz')
    assert np.allclose(Y, previous['Y'])
    return x, y, X, Y, folds


def assess(y, p):
    r = r2_score(y, p, multioutput='raw_values')
    return {
        'macro_r2': float(r.mean()), 'core30_r2': float(r[:30].mean()),
        'aroma15_r2': float(r[:15].mean()), 'flavour15_r2': float(r[15:30].mean()),
        'other20_r2': float(r[30:].mean()),
        'macro_rmse': float(np.sqrt(((y - p) ** 2).mean(axis=0)).mean()),
        'positive_r2': int((r > 0).sum()),
    }


def frozen_spec():
    source = Path(inspect.getfile(TabDPTRegressor)).resolve()
    spec = {
        'frozen_date': '2026-10-24', 'model': MODEL_NAME,
        'outer_fold_source': str(BASE / 'A_work/q1/fold_assignments.csv'),
        'outer_seed': SEED, 'model_rng_seed': '20261023 + outer_fold; reset before every target constructor',
        'target_modeling': '50 independent scalar regressions in the original attachment sensory scale',
        'selection': 'None: one fixed official default comparator, all five folds and 50 targets reported.',
        'input': 'Raw 231 numeric chemistry columns; official train-only mean imputation, StandardScaler and randomized PCA to 128 network inputs.',
        'constructor_parameters': {
            'normalizer': 'standard', 'missing_indicators': False, 'clip_sigma': 8.0,
            'feature_reduction': 'pca', 'context_reduction': 'subsample', 'faiss_metric': 'l2',
            'device': 'cpu', 'use_flash': False, 'compile': False, 'verbose': False,
            'model_weight_path': str(WEIGHT),
        },
        'predict_parameters': {'n_ensembles': 8, 'context_size': None, 'batch_size': None,
                               'seed': '20261023 + outer_fold', 'output_type': 'mean'},
        'context': 'All 200 training beers; no context subsampling or retrieval when context_size=None.',
        'threads': THREADS, 'checkpoint_sha256': digest(WEIGHT),
        'checkpoint_repository': 'Layer6/TabDPT',
        'checkpoint_revision': 'a5ca6e01c0fa09ec68c73e958e5199d1932abb3a',
        'fold_sha256': digest(BASE / 'A_work/q1/fold_assignments.csv'),
        'data_sha256': {n: digest(BASE / 'data/A' / n) for n in ('1.csv', '4.csv')},
        'versions': {p: importlib.metadata.version(p) for p in (
            'tabdpt', 'torch', 'numpy', 'pandas', 'scikit-learn', 'scipy',
            'safetensors', 'huggingface-hub', 'omegaconf', 'faiss-cpu')},
        'source_sha256': {p.name: digest(p) for p in (source, source.parent / 'estimator.py', source.parent / 'model.py')},
        'platform': platform.platform(), 'cpu_count': os.cpu_count(),
        'pretraining_boundary': 'The actual v1.3 checkpoint points to extended_openml_filtered_deduped.csv, absent from the public training repository inspected on 2026-10-24. The newer 1445-table real-data corpus cannot be certified disjoint from all beer-table variants. This is not an uncontaminated external SOTA estimate.',
        'development_boundary': 'Internal same-fold comparison on a previously studied dataset; no external validation and no outer-fold hyperparameter tuning.',
    }
    target = HERE / 'experiment_spec.json'
    if target.exists():
        assert json.loads(target.read_text()) == spec, 'Frozen specification changed; use a separate directory.'
    else:
        atomic_json(target, spec)
    return spec


def predict_one(X, Y, tr, te, fold, target):
    rng_seed = SEED + fold
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    begin = time.perf_counter()
    estimator = TabDPTRegressor(
        normalizer='standard', missing_indicators=False, clip_sigma=8.0,
        feature_reduction='pca', context_reduction='subsample', faiss_metric='l2',
        device='cpu', use_flash=False, compile=False, verbose=False,
        model_weight_path=str(WEIGHT),
    )
    init_s = time.perf_counter() - begin
    begin = time.perf_counter()
    estimator.fit(X[tr], Y[tr, target])
    fit_s = time.perf_counter() - begin
    begin = time.perf_counter()
    prediction = np.asarray(estimator.predict(
        X[te], n_ensembles=8, context_size=None, batch_size=None,
        seed=rng_seed, output_type='mean',
    ), dtype=float)
    predict_s = time.perf_counter() - begin
    assert prediction.shape == (len(te),) and np.isfinite(prediction).all()
    assert estimator.max_features == 128
    projection = estimator.projection.detach().cpu().numpy().copy()
    diagnostics = {
        'fold': fold, 'target_done': target + 1, 'target_total': 50,
        'init_s': init_s, 'fit_s': fit_s, 'predict_s': predict_s,
        'total_s': init_s + fit_s + predict_s,
        'network_features': int(estimator.max_features),
        'train_context': int(estimator.n_instances), 'threads': THREADS,
        'peak_rss_MB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    del estimator
    return prediction, projection, diagnostics


def summarize(x, y, Y, folds):
    predictions = np.full_like(Y, np.nan)
    records = []
    timings = []
    for fold in range(5):
        path = HERE / 'checkpoints' / f'fold{fold}_complete.npz'
        if not path.exists():
            continue
        checkpoint = np.load(path)
        te = checkpoint['test_indices']
        assert np.array_equal(te, np.flatnonzero(folds == fold))
        predictions[te] = checkpoint['prediction']
        records.append({'model': MODEL_NAME, 'fold': fold, **assess(Y[te], checkpoint['prediction'])})
        for j, init_s, fit_s, infer_s in checkpoint['timing']:
            timings.append({'fold': fold, 'target': int(j), 'init_s': init_s, 'fit_s': fit_s,
                            'predict_s': infer_s, 'total_s': init_s + fit_s + infer_s})
    pd.DataFrame(records).to_csv(HERE / 'fold_metrics.csv', index=False)
    pd.DataFrame(timings).to_csv(HERE / 'timing.csv', index=False)
    status = {'completed_folds': len(records), 'required_folds': 5,
              'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if np.isfinite(predictions).all():
        score = assess(Y, predictions)
        status['metrics'] = score
        atomic_npz(HERE / 'oof_arrays.npz', Y=Y, prediction=predictions,
                   beer=x.beer.to_numpy(), targets=y.columns[3:].to_numpy(str), fold=folds)
        pd.DataFrame([{'model': MODEL_NAME, **score}]).to_csv(HERE / 'model_comparison.csv', index=False)
        dimensions = r2_score(Y, predictions, multioutput='raw_values')
        rmse = np.sqrt(((Y - predictions) ** 2).mean(axis=0))
        pd.DataFrame({'model': MODEL_NAME, 'target': y.columns[3:], 'core30': np.arange(50) < 30,
                      'r2': dimensions, 'rmse': rmse}).to_csv(HERE / 'dimension_metrics.csv', index=False)
        old = np.load(BASE / 'A_work/q1/oof_arrays.npz')['nested_ensemble']
        paired_uncertainty(Y, predictions, old)
        if (MODERN / 'model_comparison.csv').exists():
            previous = pd.read_csv(MODERN / 'model_comparison.csv')
            pd.concat([previous, pd.DataFrame([{'model': MODEL_NAME, **score}])], ignore_index=True).to_csv(
                HERE / 'all_modern_comparison.csv', index=False)
    atomic_json(HERE / 'status.json', status)
    return status


def paired_uncertainty(Y, prediction, reference, repetitions=2000):
    rng = np.random.default_rng(SEED + 2026)
    differences = []
    for _ in range(repetitions):
        ids = rng.integers(0, len(Y), len(Y))
        observed = Y[ids]
        denominator = ((observed - observed.mean(axis=0)) ** 2).sum(axis=0)
        improvement = (((observed - reference[ids]) ** 2).sum(axis=0)
                       - ((observed - prediction[ids]) ** 2).sum(axis=0)) / denominator
        differences.append([improvement.mean(), improvement[:30].mean()])
    differences = np.asarray(differences)
    point = r2_score(Y, prediction, multioutput='raw_values') - r2_score(Y, reference, multioutput='raw_values')
    records = []
    for k, group in enumerate(('all50', 'core30')):
        records.append({
            'model': MODEL_NAME, 'reference': 'old_nested_ensemble', 'targets': group,
            'delta_r2': float(point.mean() if k == 0 else point[:30].mean()),
            'ci025': float(np.quantile(differences[:, k], .025)),
            'ci975': float(np.quantile(differences[:, k], .975)),
            'repetitions': repetitions,
            'interpretation': 'Descriptive paired beer bootstrap of fixed OOF predictions; no refit or multiplicity correction.',
        })
    pd.DataFrame(records).to_csv(HERE / 'paired_bootstrap.csv', index=False)


def main(args):
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    (HERE / 'checkpoints').mkdir(exist_ok=True)
    x, y, X, Y, folds = load_data()
    frozen_spec()
    if args.pilot:
        tr, te = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
        prediction, projection, timing = predict_one(X, Y, tr, te, 0, 0)
        timing['mode'] = 'Resource-only pilot; no validation score calculated.'
        atomic_npz(HERE / 'pilot_prediction.npz', prediction=prediction, projection=projection,
                   train_indices=tr, test_indices=te)
        atomic_json(HERE / 'pilot_timing.json', timing)
        print('PILOT ' + json.dumps(timing), flush=True)
        return
    for fold in range(5):
        complete = HERE / 'checkpoints' / f'fold{fold}_complete.npz'
        if complete.exists():
            print(f'SKIP complete fold={fold}', flush=True)
            continue
        tr, te = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        partial = HERE / 'checkpoints' / f'fold{fold}_partial.npz'
        prediction = np.full((len(te), 50), np.nan)
        timing = []
        fold_projection = None
        if partial.exists():
            previous = np.load(partial)
            assert np.array_equal(previous['train_indices'], tr)
            assert np.array_equal(previous['test_indices'], te)
            prediction = previous['prediction']
            timing = previous['timing'].tolist()
            fold_projection = previous['input_projection']
        for j in range(50):
            if np.isfinite(prediction[:, j]).all():
                continue
            result, projection, diagnostic = predict_one(X, Y, tr, te, fold, j)
            if fold_projection is None:
                fold_projection = projection
            else:
                assert np.allclose(fold_projection, projection, atol=1e-7, rtol=1e-7), 'Target-independent PCA changed.'
            prediction[:, j] = result
            timing.append([j, diagnostic['init_s'], diagnostic['fit_s'], diagnostic['predict_s']])
            atomic_npz(partial, prediction=prediction, train_indices=tr, test_indices=te,
                       timing=np.asarray(timing), input_projection=fold_projection)
            print(json.dumps(diagnostic), flush=True)
            if fold == 0 and j == 0 and (HERE / 'pilot_prediction.npz').exists():
                pilot = np.load(HERE / 'pilot_prediction.npz')
                difference = float(np.max(np.abs(pilot['prediction'] - result)))
                atomic_json(HERE / 'pilot_reproducibility.json', {
                    'max_abs_prediction_difference': difference,
                    'passed_atol_1e-6': bool(difference < 1e-6),
                })
                assert difference < 1e-6
        atomic_npz(complete, prediction=prediction, train_indices=tr, test_indices=te,
                   timing=np.asarray(timing), input_projection=fold_projection)
        print('FOLD_METRIC ' + json.dumps({'model': MODEL_NAME, 'fold': fold, **assess(Y[te], prediction)}), flush=True)
        summarize(x, y, Y, folds)
    final = summarize(x, y, Y, folds)
    assert final['completed_folds'] == 5
    print('COMPLETE ' + json.dumps(final), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    main(parser.parse_args())
