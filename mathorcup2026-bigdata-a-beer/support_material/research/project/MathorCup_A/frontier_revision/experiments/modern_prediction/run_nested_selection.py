# Reconstructed after the workspace file loss on 2026-10-24. This is not a
# byte-for-byte recovery of the lost script or of any lost partial results.
"""Training-only selection among five frozen TabICL configurations.

Keep the original five outer folds. Within each 200-row outer training set,
fit the three original views in three shuffled inner folds, producing five
candidate predictions. Select by pooled inner core-30 R2, then all-50 R2,
then ascending candidate name. Reuse the selected original outer complete
prediction, checking both indices and equality to the original OOF archive.

This program never updates the deployed fixed Y16 predictor. Completed inner
views and every scalar partial checkpoint can be resumed. Aggregation refuses
missing or incompatible views. The default execution is four subprocesses,
each using two CPU threads, with separate split directories.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Set numerical thread counts before importing numerical libraries.
_early = argparse.ArgumentParser(add_help=False)
_early.add_argument('--threads', type=int, default=2)
_boot, _ = _early.parse_known_args()
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = str(_boot.threads)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
SEED = 20261023
VIEWS = ('rank_pca64_y16', 'raw231_y16', 'raw231_y50')
CANDIDATES = (
    'rank_pca64_y16_no_residual',
    'rank_pca64_y16_ridge_residual',
    'raw231_y16_no_residual',
    'raw231_y16_ridge_residual',
    'raw231_y50_native',
)
VIEW_KEYS = {
    'rank_pca64_y16': ('prediction_no_residual', 'prediction_ridge_residual'),
    'raw231_y16': ('prediction_no_residual', 'prediction_ridge_residual'),
    'raw231_y50': ('prediction_native',),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def require_equal(actual, expected, label):
    if not np.array_equal(actual, expected, equal_nan=True):
        raise ValueError(f'Array identity mismatch: {label}')


def assess(Y, prediction):
    if prediction.shape != Y.shape or not np.isfinite(prediction).all():
        raise ValueError('Incomplete or non-finite prediction cannot be scored.')
    denominator = np.square(Y - Y.mean(axis=0)).sum(axis=0)
    if np.any(denominator <= 0):
        raise ValueError('A constant target prevents the registered R2 metric.')
    r2 = 1.0 - np.square(Y - prediction).sum(axis=0) / denominator
    return {'core30_r2': float(r2[:30].mean()),
            'macro_r2': float(r2.mean()),
            'aroma15_r2': float(r2[:15].mean()),
            'flavour15_r2': float(r2[15:30].mean()),
            'other20_r2': float(r2[30:].mean())}


def default_project():
    """Resolve the flat replay tree first, then the packaged project tree."""
    flat = HERE.parents[1] / 'project/MathorCup_A'
    candidates = [flat]
    for ancestor in (HERE, *HERE.parents):
        candidates.extend((ancestor, ancestor / 'project/MathorCup_A'))
    for candidate in candidates:
        if ((candidate / 'data/A/1.csv').is_file()
                and (candidate / 'A_work/q1/fold_assignments.csv').is_file()):
            return candidate.resolve()
    raise FileNotFoundError('No default project found; supply --project-dir.')


def input_data(args):
    base = args.project_dir.resolve()
    default_modern = (HERE if (HERE / 'oof_arrays.npz').is_file() else
                      base / 'frontier_revision/experiments/modern_prediction')
    modern = (args.modern_dir or default_modern).resolve()
    x = pd.read_csv(base / 'data/A/1.csv').sort_values('beer').reset_index(drop=True)
    original_y = pd.read_csv(base / 'data/A/4.csv')
    assignment = pd.read_csv(base / 'A_work/q1/fold_assignments.csv')
    for frame, name in ((x, 'chemistry'), (original_y, 'sensory'),
                        (assignment, 'fold assignments')):
        if len(frame) != 250 or frame.beer.isna().any() or not frame.beer.is_unique:
            raise ValueError(f'Invalid beer identities in {name}.')
        if set(frame.beer) != set(x.beer):
            raise ValueError(f'Beer identity set mismatch in {name}.')
    y = original_y.set_index('beer').loc[x.beer].reset_index()
    folds = assignment.set_index('beer').loc[x.beer, 'fold'].to_numpy()
    X, Y = x.iloc[:, 3:].to_numpy(float), y.iloc[:, 3:].to_numpy(float)
    if X.shape != (250, 231) or Y.shape != (250, 50):
        raise ValueError('The fixed input contract requires 250 x 231 and 250 x 50.')
    if not np.isfinite(Y).all() or np.isinf(X).any():
        raise ValueError('Invalid raw numerical data.')
    for fold, (_, test) in enumerate(KFold(5, shuffle=True,
                                         random_state=SEED).split(X)):
        require_equal(np.flatnonzero(folds == fold), test, f'outer fold {fold}')
    oof = load_npz(modern / 'oof_arrays.npz')
    for key, expected in (('Y', Y), ('beer', x.beer.to_numpy()), ('fold', folds)):
        require_equal(oof[key], expected, f'original OOF {key}')
    if oof['targets'].tolist() != y.columns[3:].tolist():
        raise ValueError('Original OOF target names/order differ from attachment 4.')
    outer_hashes = {}
    for outer in range(5):
        train, test = np.flatnonzero(folds != outer), np.flatnonzero(folds == outer)
        for view in VIEWS:
            relative = f'checkpoints/{view}_fold{outer}_complete.npz'
            path = modern / relative
            saved = load_npz(path)
            require_equal(saved['train_indices'], train, relative + ' train')
            require_equal(saved['test_indices'], test, relative + ' test')
            for key in VIEW_KEYS[view]:
                candidate = view + '_' + key.removeprefix('prediction_')
                prediction = saved[key]
                if prediction.shape != (50, 50) or not np.isfinite(prediction).all():
                    raise ValueError(f'Invalid original outer prediction: {relative}/{key}')
                require_equal(prediction, oof[candidate][test], candidate + ' outer OOF')
            outer_hashes[relative] = sha(path)
    weight = modern / 'models/tabicl-regressor-v2-20260212.ckpt'
    baseline_spec = json.loads((modern / 'experiment_spec.json').read_text())
    if sha(weight) != baseline_spec['checkpoint_sha256']:
        raise ValueError('TabICL weight hash differs from the original benchmark.')
    for name, expected in baseline_spec['data_sha256'].items():
        if sha(base / 'data/A' / name) != expected:
            raise ValueError(f'Original data hash mismatch: {name}')
    if sha(base / 'A_work/q1/fold_assignments.csv') != baseline_spec['fold_sha256']:
        raise ValueError('Original fold assignment hash mismatch.')
    return {'base': base, 'modern': modern, 'x': x, 'y': y,
            'X': X, 'Y': Y, 'fold': folds, 'oof': oof, 'weight': weight,
            'outer_hashes': outer_hashes}


def frozen_spec(args, data):
    base, modern = data['base'], data['modern']
    return {
        'reconstruction': {
            'date': '2026-10-24',
            'reason': 'Workspace regular-file loss; original completed benchmark restored from saved research package.',
            'lost_nested_partials_reused': False,
            'lost_script_byte_identity_claimed': False,
            'last_pre_failure_reviewed_source_sha256':
                'e45a5ede0e0464bd23ef12258276d2027105e0221c5afe5bb882488f81bf6646',
        },
        'protocol': {
            'outer_folds': 5, 'outer_seed': SEED,
            'inner_folds': 3, 'inner_seed_rule': '20261023 + 1000 + outer_fold (0..4)',
            'shuffle_inner': True, 'outer_training_rows': 200,
            'candidates': list(CANDIDATES), 'base_views': list(VIEWS),
            'scalar_components_per_inner_split': 82,
            'total_inner_splits': 15, 'total_scalar_components': 1230,
            'selection': ['largest pooled-inner core30_r2',
                          'largest pooled-inner all50 macro_r2',
                          'ascending candidate name for an exact tie'],
            'candidate_metric_pool': 'Exactly the 200 outer-training beers, each predicted once by an inner model not trained on that beer.',
            'outer_labels_used_in_candidate_selection': False,
            'input_rank_preprocessing': 'ChemicalPreprocessor(mode=rank), fit on the inner training rows only',
            'input_pca_rank': 64, 'input_pca_solver': 'full',
            'target_mean_and_scale': 'Inner-training mean and sample standard deviation (ddof=1), scale floored at 1e-12',
            'target_pca_rank': 16, 'target_pca_solver': 'full',
            'residual_ridge_alpha': 100.0,
            'residual_coordinates': 'Rank-transformed full chemistry for both y16 views; orthogonal target complement before and after Ridge prediction',
            'native50_target_transform': 'None; train on original 50 target columns',
            'tabicl': {'n_estimators': 8, 'device': 'cpu', 'use_amp': False,
                       'allow_auto_download': False, 'random_state': SEED,
                       'kv_cache': False, 'predict_call': 'predict(inner_validation_matrix), original default query batching'},
            'outer_prediction': 'Reuse original complete NPZ prediction only; verify train/test indices and exact equality to original OOF archive',
            'fixed_deployment_y16_is_changed': False,
        },
        'inputs_sha256': {
            '1.csv': sha(base / 'data/A/1.csv'),
            '4.csv': sha(base / 'data/A/4.csv'),
            'fold_assignments.csv': sha(base / 'A_work/q1/fold_assignments.csv'),
            'original_oof_arrays.npz': sha(modern / 'oof_arrays.npz'),
            'original_experiment_spec.json': sha(modern / 'experiment_spec.json'),
            'public_checkpoint': sha(data['weight']),
            'original_outer_complete_files': data['outer_hashes'],
        },
        'implementation_sha256': {
            'this_script': sha(Path(__file__)),
            'ChemicalPreprocessor_source': sha(base / 'A_work/q1/sensory_model.py'),
            'original_run_tabicl_benchmark.py': sha(modern / 'run_tabicl_benchmark.py'),
        },
        'execution_schedule': {'workers': args.workers, 'threads_per_worker': args.threads,
                               'unit': 'one outer/inner split',
                               'checkpoint_unit': 'one scalar target prediction'},
        'versions': {name: importlib.metadata.version(name) for name in
                     ('tabicl', 'torch', 'numpy', 'pandas', 'scikit-learn', 'scipy')},
        'interpretation': 'Evaluation of a train-only selection procedure within the original random five-fold partition; not hidden-test or leave-style-out performance.',
    }


def initialize(args):
    data = input_data(args)
    spec = frozen_spec(args, data)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'experiment_spec.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != spec:
            raise ValueError('Frozen specification changed. Do not mix checkpoints; use a separate output or audit an explicit amendment.')
    else:
        atomic_json(path, spec)
    data['spec_id'] = json_digest(spec)
    return data


def split_indices(data, outer, inner):
    outer_train = np.flatnonzero(data['fold'] != outer)
    outer_test = np.flatnonzero(data['fold'] == outer)
    split = list(KFold(3, shuffle=True, random_state=SEED + 1000 + outer).split(outer_train))
    local_train, local_test = split[inner]
    train, test = outer_train[local_train], outer_train[local_test]
    if np.intersect1d(train, test).size or np.intersect1d(outer_test, train).size or np.intersect1d(outer_test, test).size:
        raise ValueError('Train/validation/outer-test intersection detected.')
    require_equal(np.sort(np.r_[train, test]), outer_train, 'inner split coverage')
    return {'train_indices': train, 'test_indices': test,
            'outer_train_indices': outer_train, 'outer_test_indices': outer_test,
            'inner_train_positions': local_train, 'inner_test_positions': local_test}


def checkpoint_identity(saved, indices, spec_id, outer, inner, view):
    for key, expected in indices.items():
        require_equal(saved[key], expected, f'{view} {outer}/{inner} {key}')
    if str(saved['spec_id'].item()) != spec_id:
        raise ValueError('Checkpoint belongs to a different frozen specification.')
    if str(saved['view'].item()) != view or int(saved['outer_fold']) != outer or int(saved['inner_fold']) != inner:
        raise ValueError('Checkpoint view or split metadata mismatch.')


def validate_complete(saved, indices, spec_id, outer, inner, view):
    checkpoint_identity(saved, indices, spec_id, outer, inner, view)
    n, width = len(indices['test_indices']), (50 if view.endswith('y50') else 16)
    latent = saved['latent_prediction']
    if latent.shape != (n, width) or not np.isfinite(latent).all():
        raise ValueError('Incomplete latent predictions in complete file.')
    timing = saved['timing']
    if timing.shape != (width, 3) or not np.isfinite(timing).all():
        raise ValueError('Incomplete per-target timing coverage.')
    require_equal(np.sort(timing[:, 0].astype(int)), np.arange(width), 'scalar target coverage')
    for key in VIEW_KEYS[view]:
        if saved[key].shape != (n, 50) or not np.isfinite(saved[key]).all():
            raise ValueError('Incomplete candidate prediction in complete file.')
    if width == 16:
        reconstructed = ((latent @ saved['target_components'] + saved['target_pca_mean'])
                         * saved['target_scale'] + saved['target_mean'])
        np.testing.assert_allclose(reconstructed, saved['prediction_no_residual'], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(saved['prediction_ridge_residual'],
                                   reconstructed + saved['residual_prediction'], rtol=1e-12, atol=1e-12)
    else:
        require_equal(saved['prediction_native'], latent, 'native target prediction')


def run_inner(args, data, outer, inner):
    import torch
    from tabicl import TabICLRegressor
    sys.path.insert(0, str(data['base'] / 'A_work/q1'))
    from sensory_model import ChemicalPreprocessor

    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    indices = split_indices(data, outer, inner)
    tr, te = indices['train_indices'], indices['test_indices']
    directory = args.output / f'outer_{outer}' / f'inner_{inner}'
    directory.mkdir(parents=True, exist_ok=True)
    common = {**indices, 'spec_id': np.asarray(data['spec_id']),
              'outer_fold': np.asarray(outer), 'inner_fold': np.asarray(inner)}
    X, Y = data['X'], data['Y']
    with threadpool_limits(limits=args.threads):
        # Every fitted transform below sees only this inner training subset.
        prep = ChemicalPreprocessor(mode='rank').fit(X[tr])
        xr, xt = prep.transform(X[tr]), prep.transform(X[te])
        mean = Y[tr].mean(axis=0)
        scale = np.maximum(Y[tr].std(axis=0, ddof=1), 1e-12)
        standardized = (Y[tr] - mean) / scale
        ypca = PCA(n_components=16, svd_solver='full').fit(standardized)
        latent_target = ypca.transform(standardized)
        projector = np.eye(50) - ypca.components_.T @ ypca.components_
        residual_target = (standardized - ypca.mean_) @ projector
        ridge = Ridge(alpha=100.0).fit(xr, residual_target)
        residual_prediction = ridge.predict(xt) @ projector
        for view in VIEWS:
            complete = directory / f'{view}_complete.npz'
            if complete.exists():
                validate_complete(load_npz(complete), indices, data['spec_id'], outer, inner, view)
                print(f'SKIP complete outer={outer} inner={inner} view={view}', flush=True)
                continue
            if view.startswith('rank_pca'):
                xpca = PCA(n_components=64, svd_solver='full').fit(xr)
                a, b = xpca.transform(xr), xpca.transform(xt)
            else:
                a, b = X[tr], X[te]
            is_latent = view.endswith('y16')
            target = latent_target if is_latent else Y[tr]
            partial = directory / f'{view}_partial.npz'
            prediction = np.full((len(te), target.shape[1]), np.nan)
            timing = []
            if partial.exists():
                saved = load_npz(partial)
                checkpoint_identity(saved, indices, data['spec_id'], outer, inner, view)
                prediction = saved['prediction']
                timing = saved['timing'].tolist()
                if prediction.shape != (len(te), target.shape[1]):
                    raise ValueError('Partial checkpoint prediction shape differs.')
                done = np.isfinite(prediction).all(axis=0)
                empty = np.isnan(prediction).all(axis=0)
                if not np.all(done | empty):
                    raise ValueError('A partial scalar column is neither complete nor wholly missing.')
                logged = sorted(int(row[0]) for row in timing)
                if logged != np.flatnonzero(done).tolist():
                    raise ValueError('Partial timing and completed scalar columns differ.')
            for column in range(target.shape[1]):
                if np.isfinite(prediction[:, column]).all():
                    continue
                model = TabICLRegressor(
                    n_estimators=8, n_jobs=args.threads, device='cpu', use_amp=False,
                    model_path=data['weight'], allow_auto_download=False,
                    random_state=SEED, kv_cache=False)
                start = time.perf_counter()
                model.fit(a, target[:, column])
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                prediction[:, column] = model.predict(b)
                infer_seconds = time.perf_counter() - start
                if not np.isfinite(prediction[:, column]).all():
                    raise ValueError('Non-finite TabICL scalar predictions.')
                timing.append([column, fit_seconds, infer_seconds])
                atomic_npz(partial, **common, view=np.asarray(view),
                           prediction=prediction, timing=np.asarray(timing, dtype=float))
                print(json.dumps({'outer': outer, 'inner': inner, 'view': view,
                                  'target_done': int(np.isfinite(prediction).all(axis=0).sum()),
                                  'target_total': target.shape[1], 'fit_s': fit_seconds,
                                  'predict_s': infer_seconds}, ensure_ascii=False), flush=True)
                del model
            saved = {**common, 'view': np.asarray(view),
                     'latent_prediction': prediction, 'timing': np.asarray(timing, dtype=float)}
            if is_latent:
                reconstructed = ypca.inverse_transform(prediction)
                saved.update(
                    prediction_no_residual=reconstructed * scale + mean,
                    prediction_ridge_residual=(reconstructed + residual_prediction) * scale + mean,
                    target_mean=mean, target_scale=scale,
                    target_components=ypca.components_, target_pca_mean=ypca.mean_,
                    target_explained_variance_ratio=ypca.explained_variance_ratio_,
                    residual_prediction=residual_prediction * scale)
                if view.startswith('rank_pca'):
                    saved.update(input_components=xpca.components_, input_mean=xpca.mean_,
                                 input_explained_variance_ratio=xpca.explained_variance_ratio_)
            else:
                saved['prediction_native'] = prediction
            validate_complete(saved, indices, data['spec_id'], outer, inner, view)
            atomic_npz(complete, **saved)
    atomic_json(directory / 'split_complete.json', {
        'outer_fold': outer, 'inner_fold': inner, 'spec_id': data['spec_id'],
        'complete': True, 'scalar_components': 82,
        'files_sha256': {f'{view}_complete.npz': sha(directory / f'{view}_complete.npz') for view in VIEWS},
    })


def aggregate_outer(args, data, outer):
    outer_train = np.flatnonzero(data['fold'] != outer)
    outer_test = np.flatnonzero(data['fold'] == outer)
    pool = {name: np.full((len(outer_train), 50), np.nan) for name in CANDIDATES}
    coverage = np.zeros(len(outer_train), dtype=int)
    hashes = {}
    for inner in range(3):
        indices = split_indices(data, outer, inner)
        positions = indices['inner_test_positions']
        coverage[positions] += 1
        directory = args.output / f'outer_{outer}' / f'inner_{inner}'
        for view in VIEWS:
            path = directory / f'{view}_complete.npz'
            if not path.is_file():
                raise FileNotFoundError(f'Cannot select from incomplete inner predictions: {path}')
            saved = load_npz(path)
            validate_complete(saved, indices, data['spec_id'], outer, inner, view)
            for key in VIEW_KEYS[view]:
                name = view + '_' + key.removeprefix('prediction_')
                pool[name][positions] = saved[key]
            hashes[str(path.relative_to(args.output))] = sha(path)
    require_equal(coverage, np.ones(len(outer_train), dtype=int), 'pooled inner validation coverage')
    scores = [{'model': name, **assess(data['Y'][outer_train], pool[name])} for name in CANDIDATES]
    selected = sorted(scores, key=lambda row: (-row['core30_r2'], -row['macro_r2'], row['model']))[0]['model']
    directory = args.output / f'outer_{outer}'
    atomic_npz(directory / 'inner_oof.npz', Y=data['Y'][outer_train],
               beer=data['x'].beer.to_numpy()[outer_train], train_indices=outer_train,
               targets=data['y'].columns[3:].to_numpy(str), spec_id=np.asarray(data['spec_id']), **pool)
    selection = {'outer_fold': outer, 'spec_id': data['spec_id'], 'inner_complete': True,
                 'training_beers': 200, 'selected_model': selected, 'candidate_scores': scores,
                 'selection_targets': 'Only Y[outer_train]; no outer validation score used',
                 'inner_files_sha256': hashes}
    atomic_json(directory / 'selection.json', selection)
    # Selection is fixed before any outer validation metric is computed here.
    view = next(v for v in VIEWS if selected.startswith(v + '_'))
    key = 'prediction_' + selected.removeprefix(view + '_')
    original = load_npz(data['modern'] / 'checkpoints' / f'{view}_fold{outer}_complete.npz')
    require_equal(original['train_indices'], outer_train, 'selected original outer train')
    require_equal(original['test_indices'], outer_test, 'selected original outer test')
    prediction = original[key]
    require_equal(prediction, data['oof'][selected][outer_test], 'selected original OOF prediction')
    atomic_npz(directory / 'selected_outer.npz', prediction=prediction,
               train_indices=outer_train, test_indices=outer_test,
               selected_model=np.asarray(selected), spec_id=np.asarray(data['spec_id']))
    return {**selection, 'outer_metrics': assess(data['Y'][outer_test], prediction)}, prediction


def aggregate_all(args, data):
    prediction = np.full_like(data['Y'], np.nan)
    selected_by_row = np.full(len(prediction), '', dtype='U80')
    records = []
    for outer in range(5):
        record, value = aggregate_outer(args, data, outer)
        mask = data['fold'] == outer
        prediction[mask] = value
        selected_by_row[mask] = record['selected_model']
        records.append(record)
    metrics = assess(data['Y'], prediction)
    atomic_npz(args.output / 'nested_oof.npz', Y=data['Y'], prediction=prediction,
               beer=data['x'].beer.to_numpy(), fold=data['fold'],
               targets=data['y'].columns[3:].to_numpy(str),
               selected_candidate=selected_by_row, spec_id=np.asarray(data['spec_id']))
    pd.DataFrame([{'outer_fold': row['outer_fold'], **candidate}
                  for row in records for candidate in row['candidate_scores']]).to_csv(
                      args.output / 'inner_candidate_scores.csv', index=False)
    result = {'complete': True, 'spec_id': data['spec_id'], 'outer_folds': 5,
              'inner_splits_completed': 15, 'scalar_components_completed': 1230,
              'metrics': metrics, 'selections': records,
              'nested_oof_sha256': sha(args.output / 'nested_oof.npz'),
              'interpretation': 'Random-CV evaluation of training-only candidate selection; no external test and no leave-style-out claim.',
              'fixed_raw231_y16_deployment_changed': False}
    atomic_json(args.output / 'results.json', result)
    print('COMPLETE ' + json.dumps({'metrics': metrics,
                                  'selected_models': [r['selected_model'] for r in records]}), flush=True)


def launch_split(args, outer, inner):
    command = [sys.executable, str(Path(__file__).resolve()),
               '--project-dir', str(args.project_dir), '--output', str(args.output),
               '--outer-fold', str(outer), '--inner-fold', str(inner),
               '--threads', str(args.threads), '--workers', str(args.workers)]
    if args.modern_dir is not None:
        command += ['--modern-dir', str(args.modern_dir)]
    log = args.output / f'outer_{outer}_inner_{inner}.log'
    with log.open('a', encoding='utf-8') as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'Inner split {outer}/{inner} failed; see {log}')
    print(f'SPLIT_COMPLETE outer={outer} inner={inner}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path)
    parser.add_argument('--modern-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--outer-fold', type=int, choices=range(5))
    parser.add_argument('--inner-fold', type=int, choices=range(3))
    parser.add_argument('--initialize-only', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    args.project_dir = (args.project_dir or default_project()).resolve()
    args.output = (args.output or HERE / 'nested_selection').resolve()
    if args.modern_dir is not None:
        args.modern_dir = args.modern_dir.resolve()
    if args.threads < 1 or args.workers < 1:
        parser.error('Positive thread and worker counts are required.')
    if args.inner_fold is not None and args.outer_fold is None:
        parser.error('--inner-fold requires --outer-fold.')
    if args.initialize_only and (args.aggregate_only or args.outer_fold is not None):
        parser.error('--initialize-only is a separate read/registration operation.')
    if args.aggregate_only and args.inner_fold is not None:
        parser.error('An individual inner split has no selection to aggregate.')
    data = initialize(args)
    if args.initialize_only:
        print('INITIALIZED ' + data['spec_id'], flush=True)
        return
    if args.aggregate_only:
        if args.outer_fold is None:
            aggregate_all(args, data)
        else:
            aggregate_outer(args, data, args.outer_fold)
        return
    if args.outer_fold is not None:
        for inner in ([args.inner_fold] if args.inner_fold is not None else range(3)):
            run_inner(args, data, args.outer_fold, inner)
        # A scheduled single-inner task must not select from only one split.
        if args.inner_fold is None:
            aggregate_outer(args, data, args.outer_fold)
        return
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(launch_split, args, outer, inner): (outer, inner)
                   for outer in range(5) for inner in range(3)}
        for future in as_completed(futures):
            future.result()
    aggregate_all(args, data)


if __name__ == '__main__':
    main()
