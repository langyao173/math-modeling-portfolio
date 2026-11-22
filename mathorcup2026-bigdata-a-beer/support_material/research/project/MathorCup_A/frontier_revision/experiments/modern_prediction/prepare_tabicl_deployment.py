"""Prepare the fixed raw231-Y16 deployment context from attachments 1 and 4.

This explicitly fixed deployment choice comes from the completed study's
comparison of five configurations by pooled outer-fold core30 R2. It is not
a new model selection run or a claim of nested inner-fold selection. The
export refits only the target encoding on all 250 supplied training rows.
It makes no sensory predictions and requires no saved CV outputs or models.
"""
from pathlib import Path
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key] = '1'

HERE = Path(__file__).resolve().parent
INPUT_SHA256 = {
    '1.csv':'419849227a02e0c792cd405e6865035b7f58af1a92efa362dbb68a7ca2882f69',
    '4.csv':'43a7b38682ed81ce64d97ebc1874c09dffac023ade4877f34e3c83ef7d42dc1c',
}
VERSIONS = {'numpy':'2.3.5','pandas':'2.2.3','scikit-learn':'1.8.0','scipy':'1.17.0'}
CHECKPOINT_NAME = 'tabicl-regressor-v2-20260212.ckpt'
CHECKPOINT_SHA256 = '0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a'
CONTEXT_REFERENCE_SHA256 = 'cd11dd5cb7632a82eb54557007b3edda6691b37163e004be74204142e8445030'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1048576),b''):
            digest.update(block)
    return digest.hexdigest()


def check_header(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        try:
            columns = next(csv.reader(handle))
        except StopIteration:
            raise ValueError(f'Empty training CSV: {path.name}') from None
    duplicates = [name for name,count in Counter(columns).items() if count>1]
    if duplicates or any(not name.strip() for name in columns):
        raise ValueError(f'Duplicate or blank original columns in {path.name}: {duplicates}')
    if columns[:3] != ['beer','beer_id','tasting_category_fine']:
        raise ValueError(f'Unexpected identifier/metadata columns in {path.name}.')


def load_inputs(data):
    import numpy as np
    import pandas as pd
    for name in INPUT_SHA256:
        check_header(data/name)
    x = pd.read_csv(data/'1.csv').sort_values('beer').reset_index(drop=True)
    original_y = pd.read_csv(data/'4.csv')
    for name,frame in [('1.csv',x),('4.csv',original_y)]:
        ids = pd.to_numeric(frame['beer'],errors='raise').to_numpy(float)
        if len(frame)!=250 or not np.isfinite(ids).all() or not np.equal(ids,np.floor(ids)).all():
            raise ValueError(f'{name} requires 250 rows and finite integer BeerIDs.')
        if frame['beer'].duplicated().any():
            raise ValueError(f'{name} contains duplicate BeerIDs.')
    if set(x.beer)!=set(original_y.beer):
        raise ValueError('Training chemistry and sensory BeerIDs do not match.')
    y = original_y.set_index('beer').loc[x.beer].reset_index()
    if not np.array_equal(x.beer,y.beer):
        raise ValueError('Training row alignment failed.')
    # Attachment 4 rounds the barcode-like beer_id field. Only the integer
    # beer key is the training join identity; style is a separate alignment check.
    for column in ['tasting_category_fine']:
        if not np.array_equal(x[column].to_numpy(),y[column].to_numpy()):
            raise ValueError(f'Aligned metadata differs: {column}')
    X = x.iloc[:,3:].to_numpy(float)
    Y = y.iloc[:,3:].to_numpy(float)
    if X.shape!=(250,231) or Y.shape!=(250,50):
        raise ValueError('Expected 250x231 chemistry and 250x50 sensory arrays.')
    if np.isinf(X).any() or np.all(np.isnan(X),axis=1).any():
        raise ValueError('Training chemistry contains infinite or entirely missing rows.')
    if not np.isfinite(Y).all() or not (Y.std(axis=0,ddof=1)>0).all():
        raise ValueError('Training sensory targets must be finite and nonconstant.')
    identities = {name:sha(data/name) for name in INPUT_SHA256}
    if identities!=INPUT_SHA256:
        raise ValueError('Training attachments differ from the analyzed official files; supplied feature scales must be preserved.')
    report = dict(valid=True, rows=250, chemical_features=231, sensory_targets=50,
        training_ids_unique=True, training_ids_aligned=True, style_metadata_aligned=True,
        identity_column='beer',barcode_column_used_for_alignment=False,
        source_sha256=identities, chemistry_missing_cells=int(np.isnan(X).sum()),
        input_scale='Original attachment-1 supplied scales, including existing log10 transforms.',
        target_scale='Original attachment-4 standardized sensory values, not raw panel grades.')
    return x,y,X,Y,report


def prepare(args):
    import numpy as np
    from sklearn.decomposition import PCA
    x,y,X,Y,validation = load_inputs(args.data_dir.expanduser().resolve())
    if args.check_input:
        print(json.dumps(dict(validation, validation_only=True, context_exported=False),ensure_ascii=False))
        return
    if args.output is None:
        raise ValueError('--output is required unless --check-input is used.')
    actual_versions = {name:importlib.metadata.version(name) for name in VERSIONS}
    if actual_versions!=VERSIONS:
        raise RuntimeError(f'Encoding environment differs from the validated versions: {actual_versions}; required {VERSIONS}')
    checkpoint = (args.checkpoint or HERE/'models'/CHECKPOINT_NAME).expanduser().resolve()
    if sha(checkpoint)!=CHECKPOINT_SHA256:
        raise ValueError('Public weight SHA256 mismatch; use the pinned fetch_public_weights.py command.')
    inference = (args.inference_script or HERE/'predict_tabicl_csv.py').expanduser().resolve()
    if not inference.is_file():
        raise FileNotFoundError(inference)
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty deployment artifact directory; existing evidence is not overwritten.')
    output.mkdir(parents=True,exist_ok=True)
    arrays = dict(X=X, Y=Y, beer=x.beer.to_numpy(), features=x.columns[3:].to_numpy(str),
                  targets=y.columns[3:].to_numpy(str),
                  relative_grade_cuts=np.quantile(Y,np.arange(1,7)/7,axis=0))
    mean = Y.mean(axis=0)
    scale = np.maximum(Y.std(axis=0, ddof=1), 1e-12)
    standardized = (Y-mean)/scale
    pca = PCA(n_components=16, svd_solver='full').fit(standardized)
    latent = pca.transform(standardized)
    arrays.update(context_targets=latent, target_mean=mean, target_scale=scale,
                  pca_components=pca.components_, pca_mean=pca.mean_,
                  pca_explained_variance_ratio=pca.explained_variance_ratio_)
    projected = latent@pca.components_+pca.mean_
    residual = standardized-projected
    if np.max(np.abs(residual@pca.components_.T))>=1e-10:
        raise RuntimeError('Target subspace orthogonality check failed.')
    np.savez_compressed(output/'model_context.npz', **arrays)
    encoding = dict(kind='standardize_pca', outputs=16, rank=16, standard_deviation_ddof=1,
        variance_retained=float(pca.explained_variance_ratio_.sum()),
        note='Refitted on all 250 official examples for deployment only; no validation or query labels used.')
    manifest = dict(format_version=1,model_family='TabICLv2',chosen_cv_candidate='raw231_y16_no_residual',
        selection=dict(mode='fixed_configuration_from_completed_study',
            rule='Fixed raw231-Y16 without residual; no configuration selection occurs in this script.',
            basis='The prior five-configuration study selected this deployment route by pooled outer-fold core30 R2, with macro50 R2 as tie-breaker. This is not nested inner-fold selection.',
            new_training_or_query_selection=False, prior_comparison_reexecuted=False),
        task=dict(inputs=231,outputs=50,context_rows=250,
            input_units='Exactly the transformed feature columns of contest attachment 1.csv; preserve existing log10 transforms.',
            output_units='Standardized sensory values of contest attachment 4.csv, not raw 0-6 panel ratings.'),
        target_encoding=encoding,input_encoding='Official TabICL preprocessing fitted to the saved context at inference time',
        relative_grades=dict(range=[0,6],quantiles=(np.arange(1,7)/7).tolist(),side='right',
            reference='All 250 training-context sensory rows',
            interpretation='Within-study training-reference relative grades, not recovered raw panel scores.'),
        context_file='model_context.npz',context_sha256=sha(output/'model_context.npz'),
        checkpoint=dict(relative_path=os.path.relpath(checkpoint,output),sha256=CHECKPOINT_SHA256,
            license='BSD-3-Clause',repository='https://huggingface.co/jingang/TabICL',
            revision='4dcd344ece2c00be9e831fdd35bed57b5ad83e19'),
        parameters=dict(n_estimators=8,random_state=20261023,device='cpu',n_jobs=6,
            use_amp=False,allow_auto_download=False,norm_methods=None,batch_size=8,
            kv_cache=False,outlier_threshold=4.0,feat_shuffle_method='latin'),
        versions=actual_versions,required_tabicl_version='2.2.0',required_torch_version='2.14.1+cpu',
        data_sha256=validation['source_sha256'],input_validation=validation,
        exporter_sha256=sha(__file__),inference_script_sha256=sha(inference),
        fit_predictions_generated=False,reference_context_sha256=CONTEXT_REFERENCE_SHA256,
        context_bytes_match_frozen_reference=sha(output/'model_context.npz')==CONTEXT_REFERENCE_SHA256)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dict(model=manifest['chosen_cv_candidate'],context_rows=250,target_encoding=encoding,
        output=str(output),context_sha256=manifest['context_sha256'],
        context_bytes_match_frozen_reference=manifest['context_bytes_match_frozen_reference']),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True,help='Directory containing the original 1.csv and 4.csv.')
    parser.add_argument('--output',type=Path,help='Separate empty artifact directory.')
    parser.add_argument('--checkpoint',type=Path,help='Pinned public TabICLv2 checkpoint; defaults to sibling models directory.')
    parser.add_argument('--inference-script',type=Path,help='CSV inference source recorded in the manifest.')
    parser.add_argument('--check-input',action='store_true',help='Validate only the two official training inputs; no model or artifact output.')
    prepare(parser.parse_args())
