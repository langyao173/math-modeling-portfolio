"""Predict 50 standardized sensory outputs directly from a chemistry CSV.

Requires the exported model_context.npz/manifest.json and pinned public
TabICLv2 weights. This script does not require the original contest project.
All model fitting at inference time only binds the saved training context;
new rows are never used to choose a target encoding or tune hyperparameters.
"""
from pathlib import Path
import argparse
import csv
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def chemistry_input(path, features, targets, missing_policy, id_column):
    """Read chemistry by exact names; validate headers before pandas mangling.

    
    Explicit identifiers, or the default beer column, must be nonblank and
    unique. Their exact CSV text is preserved, including leading zeros and NA.
    With no identifier column, zero-based input_row records source order.
    """
    import numpy as np
    import pandas as pd
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError('Input CSV is empty.') from None
    duplicate = [name for name, count in Counter(header).items() if count > 1]
    if duplicate:
        raise ValueError(f'Duplicate original CSV column names: {duplicate}')
    if any(not name.strip() for name in header):
        raise ValueError('CSV column names must be nonblank.')
    identifier = id_column if id_column is not None else ('beer' if 'beer' in header else None)
    if identifier is not None and identifier not in header:
        raise ValueError(f'Identifier column does not exist: {identifier}')
    if identifier is not None and identifier in set(features) | set(targets):
        raise ValueError('Identifier column must not be a chemistry or sensory target name.')
    converters = {identifier: lambda value: value} if identifier is not None else None
    frame = pd.read_csv(path, encoding='utf-8-sig', converters=converters)
    if len(frame) == 0:
        raise ValueError('Input CSV has no data rows.')
    if identifier is None:
        identifiers = np.arange(len(frame))
        output_identifier = 'input_row'
    else:
        identifiers = frame[identifier].to_numpy()
        if any(not isinstance(value, str) or not value.strip() for value in identifiers):
            raise ValueError('Identifier values must be nonblank text.')
        if len(set(identifiers.tolist())) != len(identifiers):
            raise ValueError('Identifier values must be unique.')
        output_identifier = identifier
    missing = [name for name in features if name not in frame.columns]
    if missing and missing_policy == 'error':
        raise ValueError(f'Missing {len(missing)} required chemistry columns: {missing}')
    for name in missing:
        frame[name] = np.nan
    X = frame[features].apply(pd.to_numeric, errors='raise').to_numpy(float)
    if np.isinf(X).any():
        raise ValueError('Infinite chemistry values are not permitted; use NaN for unavailable measurements.')
    if np.all(np.isnan(X), axis=1).any():
        raise ValueError('At least one input row has no observed chemistry values.')
    extra = [name for name in header if name not in features and name != identifier]
    return frame, X, output_identifier, identifiers, missing, extra


def check_context(a, manifest):
    """Validate the published 231-input/50-output artifact contract."""
    import numpy as np
    features, targets = a['features'].tolist(), a['targets'].tolist()
    if len(features) != 231 or len(set(features)) != 231:
        raise ValueError('Artifact must contain 231 distinct chemical feature names.')
    if len(targets) != 50 or len(set(targets)) != 50:
        raise ValueError('Artifact must contain 50 distinct sensory target names.')
    n = int(manifest['task']['context_rows'])
    if a['X'].shape != (n,231) or a['Y'].shape != (n,50):
        raise ValueError('Training context arrays have incompatible dimensions.')
    if np.isinf(a['X']).any() or not np.isfinite(a['Y']).all():
        raise ValueError('Invalid training context values.')
    outputs = int(manifest['target_encoding']['outputs'])
    if a['context_targets'].shape != (n,outputs) or not np.isfinite(a['context_targets']).all():
        raise ValueError('Encoded target context is invalid.')
    cuts = a['relative_grade_cuts']
    if cuts.shape != (6,50) or not np.isfinite(cuts).all() or (np.diff(cuts,axis=0)<0).any():
        raise ValueError('Relative-grade cuts must be six ordered finite thresholds per target.')
    if manifest['target_encoding']['kind'] == 'standardize_pca':
        if a['pca_components'].shape != (outputs,50) or a['pca_mean'].shape != (50,):
            raise ValueError('PCA output decoding dimensions are invalid.')
        if a['target_scale'].shape != (50,) or a['target_mean'].shape != (50,):
            raise ValueError('Target scaling dimensions are invalid.')
        if not np.isfinite(a['target_scale']).all() or not (a['target_scale']>0).all():
            raise ValueError('Target scales must be positive and finite.')
    elif manifest['target_encoding']['kind'] != 'identity' or outputs != 50:
        raise ValueError('Unsupported target encoding.')
    return features, targets


def predict(args):
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
        os.environ[key] = str(args.threads)
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits

    manifest = json.loads((args.artifact/'manifest.json').read_text())
    context_path = args.artifact/manifest['context_file']
    if sha(context_path) != manifest['context_sha256']:
        raise ValueError('Training context hash mismatch.')
    a = np.load(context_path, allow_pickle=False)
    features, targets = check_context(a, manifest)
    frame, X, identifier, identifiers, missing, extra = chemistry_input(
        args.input, features, targets, args.missing_columns, args.id_column)
    validation = dict(valid=True, rows=len(X), chemical_features=len(features), sensory_targets=len(targets),
        input_identifier=identifier, missing_columns_filled=missing, ignored_extra_columns=extra,
        input_missing_fraction=float(np.isnan(X).mean()), artifact_manifest_sha256=sha(args.artifact/'manifest.json'),
        input_sha256=sha(args.input), chosen_cv_candidate=manifest['chosen_cv_candidate'],
        contract='Columns aligned by exact feature name; no sensory labels required; supplied attachment-1 scales preserved.')
    if args.validate_only:
        print(json.dumps(dict(validation, validation_only=True, neural_inference_executed=False), ensure_ascii=False))
        return
    checkpoint = args.checkpoint or (args.artifact/manifest['checkpoint']['relative_path']).resolve()
    if sha(checkpoint) != manifest['checkpoint']['sha256']:
        raise ValueError('Public weight hash mismatch; use fetch_public_weights.py.')
    for package, required in [('tabicl',manifest['required_tabicl_version']),
                               ('torch',manifest['required_torch_version'])]:
        actual = importlib.metadata.version(package)
        if actual != required:
            raise RuntimeError(f'{package} version {actual} differs from validated {required}.')
    import torch
    from tabicl import TabICLRegressor
    torch.set_num_threads(args.threads)
    torch.manual_seed(manifest['parameters']['random_state'])
    context_targets = a['context_targets']
    latent = np.empty((len(X), context_targets.shape[1]))
    timings = []
    with threadpool_limits(limits=args.threads):
        for j in range(context_targets.shape[1]):
            model = TabICLRegressor(n_estimators=manifest['parameters']['n_estimators'],
                n_jobs=args.threads, device='cpu', use_amp=False,
                model_path=checkpoint, allow_auto_download=False,
                random_state=manifest['parameters']['random_state'])
            start = time.perf_counter()
            model.fit(a['X'], context_targets[:,j])
            fit_seconds = time.perf_counter()-start
            start = time.perf_counter()
            latent[:,j] = model.predict(X)
            predict_seconds = time.perf_counter()-start
            timings.append(dict(target=j, fit_seconds=fit_seconds, predict_seconds=predict_seconds))
            print(json.dumps(dict(target_done=j+1, target_total=context_targets.shape[1],
                                  rows=len(X), **timings[-1])), file=sys.stderr, flush=True)
    if manifest['target_encoding']['kind'] == 'standardize_pca':
        predictions = (latent@a['pca_components']+a['pca_mean'])*a['target_scale']+a['target_mean']
    else:
        predictions = latent
    if predictions.shape != (len(X),50) or not np.isfinite(predictions).all():
        raise RuntimeError('Invalid sensory output shape or nonfinite predictions.')
    result = pd.DataFrame(predictions, columns=a['targets'].tolist())
    result.insert(0,identifier,identifiers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    grades = np.stack([np.searchsorted(a['relative_grade_cuts'][:,j],predictions[:,j],side='right')
                       for j in range(50)], axis=1)
    grade_frame = pd.DataFrame(grades, columns=a['targets'].tolist())
    grade_frame.insert(0,result.columns[0],result.iloc[:,0].to_numpy())
    grade_path = args.output.with_name(args.output.stem+'.relative_grades_0_6.csv')
    grade_frame.to_csv(grade_path, index=False)
    audit = dict(artifact_manifest_sha256=sha(args.artifact/'manifest.json'),
                 input_sha256=sha(args.input), output_sha256=sha(args.output),
                 checkpoint_sha256=manifest['checkpoint']['sha256'],
                 chosen_cv_candidate=manifest['chosen_cv_candidate'], rows=len(X), threads=args.threads,
                 missing_columns_filled=missing, input_missing_fraction=float(np.isnan(X).mean()),
                 input_identifier=identifier, ignored_extra_columns=extra,
                 timings=timings, output_units=manifest['task']['output_units'],
                 relative_grades_file=str(grade_path), relative_grades_sha256=sha(grade_path),
                 relative_grades_interpretation=manifest['relative_grades']['interpretation'])
    args.output.with_suffix(args.output.suffix+'.audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(f'Wrote {len(result)} rows and 50 sensory predictions to {args.output}', file=sys.stderr)
    print(f'Wrote training-reference relative grades to {grade_path}; these are not raw panel scores.', file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--artifact', type=Path, default=Path(__file__).resolve().parent/'final_artifact')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--id-column', help='Explicit identifier column; default beer if present, otherwise input_row.')
    parser.add_argument('--validate-only', action='store_true', help='Check CSV/context without model inference or prediction outputs.')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--missing-columns', choices=['error','nan'], default='error',
                        help='Explicitly permit unavailable entire chemistry columns as NaN.')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    predict(args)
