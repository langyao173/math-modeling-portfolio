"""Export an all-data TabICL deployment context after the full CV finishes.

No in-sample predictions are computed. TabICL is an in-context model: the
training context and target encoding are the task-specific fitted artifact.
The deployment candidate is selected by internal CV, not external validation.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
for _k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_k] = '1'
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]/'project'/'MathorCup_A'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def export(output):
    status = json.loads((HERE/'status.json').read_text())
    if status['completed_model_folds'] != status['total_model_folds']:
        raise RuntimeError('All preregistered TabICL candidates must finish before deployment selection.')
    comparison = pd.read_csv(HERE/'model_comparison.csv')
    modern = comparison[~comparison.model.str.startswith('old_')].copy()
    if len(modern) != 5:
        raise RuntimeError('Expected all five registered TabICL candidates.')
    choice = modern.sort_values(['core30_r2','macro_r2','model'], ascending=[False,False,True]).iloc[0]
    if choice.model not in ('raw231_y16_no_residual','raw231_y50_native'):
        raise RuntimeError('Winning route requires an additional explicit input/residual export implementation.')
    x = pd.read_csv(BASE/'data/A/1.csv').sort_values('beer').reset_index(drop=True)
    y = pd.read_csv(BASE/'data/A/4.csv').set_index('beer').loc[x.beer].reset_index()
    X = x.iloc[:,3:].to_numpy(float)
    Y = y.iloc[:,3:].to_numpy(float)
    assert X.shape == (250,231) and Y.shape == (250,50)
    assert np.array_equal(x.beer, y.beer) and np.isfinite(Y).all()
    output.mkdir(parents=True, exist_ok=True)
    arrays = dict(X=X, Y=Y, beer=x.beer.to_numpy(), features=x.columns[3:].to_numpy(str),
                  targets=y.columns[3:].to_numpy(str),
                  relative_grade_cuts=np.quantile(Y,np.arange(1,7)/7,axis=0))
    encoding = {'kind':'identity', 'outputs':50}
    if choice.model == 'raw231_y16_no_residual':
        mean = Y.mean(axis=0)
        scale = np.maximum(Y.std(axis=0, ddof=1), 1e-12)
        standardized = (Y-mean)/scale
        pca = PCA(n_components=16, svd_solver='full').fit(standardized)
        latent = pca.transform(standardized)
        arrays.update(context_targets=latent, target_mean=mean, target_scale=scale,
                      pca_components=pca.components_, pca_mean=pca.mean_,
                      pca_explained_variance_ratio=pca.explained_variance_ratio_)
        encoding = dict(kind='standardize_pca', outputs=16, rank=16, standard_deviation_ddof=1,
                        variance_retained=float(pca.explained_variance_ratio_.sum()),
                        note='Refitted on all 250 examples for deployment only; CV encodings remain train-fold-specific.')
        projected = latent@pca.components_+pca.mean_
        residual = standardized-projected
        assert np.max(np.abs(residual@pca.components_.T)) < 1e-10
    else:
        arrays['context_targets'] = Y.copy()
    np.savez_compressed(output/'model_context.npz', **arrays)
    spec = json.loads((HERE/'experiment_spec.json').read_text())
    manifest = dict(format_version=1, model_family='TabICLv2', chosen_cv_candidate=str(choice.model),
        selection=dict(rule='Largest complete internal fivefold core30 R2 across all five registered TabICL candidates; macro50 R2 breaks ties.',
                       rationale='The primary contest outputs are 15 aroma and 15 flavour descriptors; the additional 20 outputs are an extended evaluation.',
                       criterion_declared='Task-aligned deployment criterion explicitly confirmed by project lead on 2026-10-24, before all native50 folds completed; candidate comparisons are internal and all reported.',
                       basis='Internal cross-validation on previously analyzed data; no external validation',
                       metrics=choice.to_dict(), comparison_sha256=sha(HERE/'model_comparison.csv'),
                       all_candidates=modern.to_dict(orient='records')),
        task=dict(inputs=231, outputs=50, context_rows=250,
                  input_units='Exactly the transformed feature columns of contest attachment 1.csv; preserve existing log10 transforms.',
                  output_units='Standardized sensory values of contest attachment 4.csv, not raw 0-6 panel ratings.'),
        target_encoding=encoding, input_encoding='Official TabICL preprocessing fitted to the saved context at inference time',
        relative_grades=dict(range=[0,6], quantiles=(np.arange(1,7)/7).tolist(), side='right',
                             reference='All 250 training-context sensory rows',
                             interpretation='Within-study training-reference relative grades, not recovered raw panel scores.'),
        context_file='model_context.npz', context_sha256=sha(output/'model_context.npz'),
        checkpoint=dict(relative_path='../models/tabicl-regressor-v2-20260212.ckpt',
                        sha256=spec['checkpoint_sha256'], license='BSD-3-Clause',
                        repository='https://huggingface.co/jingang/TabICL',
                        revision='4dcd344ece2c00be9e831fdd35bed57b5ad83e19'),
        parameters=dict(n_estimators=8, random_state=20261023, device='cpu', n_jobs=6,
                        use_amp=False, allow_auto_download=False, norm_methods=None,
                        batch_size=8, kv_cache=False, outlier_threshold=4.0,
                        feat_shuffle_method='latin'),
        versions={p:importlib.metadata.version(p) for p in ('numpy','pandas','scikit-learn')},
        required_tabicl_version=spec['versions']['tabicl'], required_torch_version=spec['versions']['torch'],
        data_sha256=spec['data_sha256'], source_fold_sha256=spec['fold_sha256'],
        exporter_sha256=sha(__file__), inference_script_sha256=sha(HERE/'predict_tabicl_csv.py'),
        fit_predictions_generated=False)
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(dict(model=choice.model, context_rows=250, target_encoding=encoding,
                          output=str(output), context_sha256=manifest['context_sha256']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE/'final_artifact')
    args = parser.parse_args()
    export(args.output)
