"""Audit existing TabICL predictions without fitting or selecting a model.

Checks index coverage, target-standardization provenance, reconstruction,
orthogonality, and the exact standardized squared-error decomposition.

"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / 'project' / 'MathorCup_A'
x = pd.read_csv(BASE / 'data/A/1.csv').sort_values('beer').reset_index(drop=True)
y = pd.read_csv(BASE / 'data/A/4.csv').set_index('beer').loc[x.beer].reset_index()
Y = y.iloc[:, 3:].to_numpy(float)
folds = pd.read_csv(BASE / 'A_work/q1/fold_assignments.csv').set_index('beer').loc[x.beer, 'fold'].to_numpy()
records = []
coverage = {}
for path in sorted((HERE / 'checkpoints').glob('*_complete.npz')):
    checkpoint = np.load(path)
    view, foldstr, _ = path.stem.rsplit('_', 2)
    f = int(foldstr.removeprefix('fold'))
    tr, te = checkpoint['train_indices'], checkpoint['test_indices']
    assert np.array_equal(tr, np.flatnonzero(folds != f))
    assert np.array_equal(te, np.flatnonzero(folds == f))
    assert not np.intersect1d(tr, te).size
    assert len(tr) == 200 and len(te) == 50
    coverage.setdefault(view, []).extend(te.tolist())
    record = {'view': view, 'fold': f, 'train_count': len(tr), 'test_count': len(te)}
    for key in checkpoint.files:
        if key.startswith('prediction_'):
            assert checkpoint[key].shape == (50, 50)
            assert np.isfinite(checkpoint[key]).all()
    if 'target_components' in checkpoint:
        mean, scale = checkpoint['target_mean'], checkpoint['target_scale']
        assert np.allclose(mean, Y[tr].mean(axis=0), atol=1e-14)
        assert np.allclose(scale, Y[tr].std(axis=0, ddof=1), atol=1e-14)
        T = (Y[tr] - mean) / scale
        test = (Y[te] - mean) / scale
        pca = PCA(n_components=16, svd_solver='full').fit(T)
        V = checkpoint['target_components'].T
        assert np.allclose(pca.components_, V.T, atol=1e-12)
        residual = checkpoint['residual_prediction'] / scale
        latent = checkpoint['latent_prediction']
        no_residual = (checkpoint['prediction_no_residual'] - mean) / scale
        full = (checkpoint['prediction_ridge_residual'] - mean) / scale
        reconstruction = latent @ V.T + pca.mean_
        np.testing.assert_allclose(no_residual, reconstruction, atol=1e-12)
        np.testing.assert_allclose(full, reconstruction + residual, atol=1e-12)
        identity_error = float(np.max(np.abs(V.T @ V - np.eye(16))))
        orthogonality_error = float(np.max(np.abs(residual @ V)))
        assert identity_error < 1e-10 and orthogonality_error < 1e-10
        true_residual = (test - pca.mean_) @ (np.eye(50) - V @ V.T)
        latent_mse = float(np.square((test - pca.mean_) @ V - latent).sum(axis=1).mean() / 50)
        residual_zero_mse = float(np.square(true_residual).mean())
        residual_ridge_mse = float(np.square(true_residual - residual).mean())
        total_mse = float(np.square(test - full).mean())
        total_zero_mse = float(np.square(test - no_residual).mean())
        decomposition_error = abs(total_mse - latent_mse - residual_ridge_mse)
        assert decomposition_error < 1e-12
        assert abs(total_zero_mse - latent_mse - residual_zero_mse) < 1e-12
        record.update({
            'target_pca_variance_retained': float(checkpoint['target_explained_variance_ratio'].sum()),
            'orthonormality_max_error': identity_error,
            'residual_orthogonality_max_error': orthogonality_error,
            'decomposition_max_error': decomposition_error,
            'latent_standardized_mse': latent_mse,
            'residual_zero_standardized_mse': residual_zero_mse,
            'residual_ridge_standardized_mse': residual_ridge_mse,
            'residual_added_standardized_mse': residual_ridge_mse - residual_zero_mse,
            'total_no_residual_standardized_mse': total_zero_mse,
            'total_ridge_residual_standardized_mse': total_mse,
        })
        if 'input_explained_variance_ratio' in checkpoint:
            record['input_pca_variance_retained'] = float(checkpoint['input_explained_variance_ratio'].sum())
    records.append(record)
for view, indices in coverage.items():
    assert len(indices) == len(set(indices)), f'Duplicate test predictions for {view}'
    if len(indices) == 250:
        assert sorted(indices) == list(range(250))
pd.DataFrame(records).to_csv(HERE / 'reconstruction_audit.csv', index=False)
summary = {
    'completed_checkpoint_count': len(records),
    'test_rows_by_view': {k: len(v) for k, v in coverage.items()},
    'all_checked_invariants_passed': True,
    'interpretation': 'The decomposition uses training-fold standardized coordinates. It does not equate this MSE with macro R2 or establish causal target predictability.',
}
(HERE / 'reconstruction_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False))
