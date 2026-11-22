"""Paired raw-score auxiliary response experiment on official standardized targets.

All preprocessing and tuning are training-side. The raw-score weight tau is
prespecified and methods are evaluated separately. Shared reduced rank is the
only coupling mechanism; unrestricted ridge is checked for exact separation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_name] = '1'

import numpy as np
import pandas as pd
from scipy.special import ndtri

SEED = 20261023
TAUS = (0., .5, 1.)
RANKS = (4, 8, 16, 32)
ALPHAS = (1., 10., 100., 1000.)
GROUPS = {'core30': np.arange(30), 'all50': np.arange(50), 'other20': np.arange(30, 50)}
DERIVED = ('kcalperc', 'esters_sum', 'aroma_hops_sum', 'acids_sum', 'sulfur_sum')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.writing')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def standardize(x):
    mean = x.mean(0)
    std = x.std(0)
    std = np.where(std > 1e-12, std, 1.)
    return (x-mean)/std, mean, std


def rank_features(filled, ordered):
    n = ordered.shape[0]
    out = np.empty_like(filled)
    for j in range(filled.shape[1]):
        left = np.searchsorted(ordered[:, j], filled[:, j], side='left')
        right = np.searchsorted(ordered[:, j], filled[:, j], side='right')
        probability = np.clip((left+right)/(2*n), .5/n, 1-.5/n)
        out[:, j] = ndtri(probability)
    return out


def prepare(x, z, raw, input_transform):
    median = np.nanmedian(x, axis=0)
    if not np.isfinite(median).all():
        raise ValueError('A training feature has no observed value')
    filled = np.where(np.isfinite(x), x, median)
    ordered = np.sort(filled, axis=0) if input_transform == 'normal_rank' else None
    if ordered is not None:
        filled = rank_features(filled, ordered)
    xx, xm, xs = standardize(filled)
    zz, zm, zs = standardize(z)
    rr, rm, rs = standardize(raw)
    u, singular, vt = np.linalg.svd(xx, full_matrices=False)
    return dict(median=median, xm=xm, xs=xs, zm=zm, zs=zs, rm=rm, rs=rs,
                X=xx, Z=zz, raw=rr, U=u, singular=singular, Vt=vt,
                input_transform=input_transform, ordered=ordered)


def transform_x(x, state):
    filled = np.where(np.isfinite(x), x, state['median'])
    if state['ordered'] is not None:
        filled = rank_features(filled, state['ordered'])
    return (filled-state['xm'])/state['xs']


def candidates(state, alpha, tau):
    y = state['Z'] if tau == 0 else np.column_stack((state['Z'], np.sqrt(tau)*state['raw']))
    uy = state['U'].T@y
    s = state['singular']
    ridge = (state['Vt'].T*(s/(s*s+alpha)))@uy
    h = (uy.T*(s*s/(s*s+alpha)))@uy
    values, vectors = np.linalg.eigh((h+h.T)/2)
    order = np.argsort(values)[::-1]
    vectors = vectors[:, order]
    return ridge, vectors


def predict(x, state, ridge, vectors, rank):
    v = vectors[:, :rank]
    coefficient = (ridge@v)@v.T
    return (transform_x(x, state)@coefficient)[:, :50]*state['zs'] + state['zm']


def metrics(y, prediction):
    per_target = 1.-np.mean((prediction-y)**2, axis=0)/np.var(y, axis=0)
    return {group: float(per_target[js].mean()) for group, js in GROUPS.items()}


def input_array(data, *keys):
    for key in keys:
        if key in data:
            return data[key]
    raise ValueError('Shared input lacks any of: '+','.join(keys))


def main(args):
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing auxiliary result directory')
    args.output.mkdir(parents=True)
    with np.load(args.input, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    X = input_array(data, 'X')
    Z = input_array(data, 'Z', 'Y_z', 'Yz', 'official_z')
    raw = input_array(data, 'raw_mean', 'R', 'mean_raw', 'raw_means')
    beer = input_array(data, 'beer')
    fold = input_array(data, 'fold')
    inner_fold = input_array(data, 'inner_fold')
    features = input_array(data, 'features')
    targets = input_array(data, 'targets')
    assert X.shape == (250, 231) and Z.shape == raw.shape == (250, 50)
    assert np.unique(beer).size == 250 and set(fold) == set(range(5))
    assert inner_fold.shape == (5, 250)
    assert np.isfinite(Z).all() and np.isfinite(raw).all()
    derived = np.array([list(features).index(name) for name in DERIVED])
    direct = np.setdiff1d(np.arange(X.shape[1]), derived)
    scenarios = ('original', 'derived_hidden', 'MCAR10', 'MCAR30', 'MCAR50')
    protocol = dict(primary_comparison='tau1 minus tau0', sensitivity='tau0.5; all prespecified methods retained',
                    taus=list(TAUS), tau_interpretation='raw standardized squared-loss weight; concatenate sqrt(tau)*R',
                    ranks=list(RANKS), alphas=list(ALPHAS), outer_folds='existing frozen fold vector',
                    inner_folds=3, inner_fold_source='frozen shared input inner_fold array, reused without changes',
                    selection='mean squared official-Z prediction error on core30 normalized by each inner-training target variance',
                    preprocessing='training-side median imputation, feature centering/scaling, separate target centering/scaling',
                    input_transform=args.input_transform,
                    response_coupling='joint reduced-rank ridge; full-rank ridge is a separation sanity check only',
                    missingness='test-time only; derived summaries hidden; paired MCAR on remaining226; frozen full-input tuning',
                    input_sha256=sha(args.input), script_sha256=sha(Path(__file__)),
                    source_archive=str(args.input.resolve()), no_outer_raw_used_for_training=True,
                    external_independent_validation=False, no_neural_fits=True, no_paid_calls=True)
    atomic_json(args.output/'PROTOCOL_LOCK.json', protocol)
    predictions = {(tau, scene): np.full_like(Z, np.nan) for tau in TAUS for scene in scenarios}
    selected, inner_rows, trace = [], [], []
    unrestricted_error = 0.
    start = time.monotonic()
    for f in range(5):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        losses = {(tau, alpha, rank): [] for tau in TAUS for alpha in ALPHAS for rank in RANKS}
        assert np.all(inner_fold[f, te] == -1) and set(inner_fold[f, tr]) == {0, 1, 2}
        for inner in range(3):
            itr, iva = tr[inner_fold[f, tr] != inner], tr[inner_fold[f, tr] == inner]
            assert np.intersect1d(itr, te).size == 0 and np.intersect1d(iva, te).size == 0
            state = prepare(X[itr], Z[itr], raw[itr], args.input_transform)
            for tau in TAUS:
                for alpha in ALPHAS:
                    ridge, vectors = candidates(state, alpha, tau)
                    for rank in RANKS:
                        p = predict(X[iva], state, ridge, vectors, rank)
                        value = ((p-Z[iva])**2/state['zs']**2)[:, :30].sum()
                        losses[tau, alpha, rank].append((float(value), len(iva)*30))
            trace.append(dict(outer_fold=f, inner_fold=inner,
                              training_beers=beer[itr].tolist(), validation_beers=beer[iva].tolist(),
                              outer_test_beers=beer[te].tolist(), training_rows=len(itr), validation_rows=len(iva)))
        state = prepare(X[tr], Z[tr], raw[tr], args.input_transform)
        # Shared masks depend only on fixed seeds and dimensions, never outcomes.
        inputs = {'original': X[te].copy(), 'derived_hidden': X[te].copy()}
        inputs['derived_hidden'][:, derived] = np.nan
        for level in (10, 30, 50):
            masked = X[te].copy()
            random = np.random.default_rng(SEED+1000*f+level)
            mask = random.random((len(te), len(direct))) < level/100.
            part = masked[:, direct].copy()
            part[mask] = np.nan
            masked[:, direct] = part
            masked[:, derived] = np.nan
            inputs['MCAR'+str(level)] = masked
        for tau in TAUS:
            scores = []
            for alpha in ALPHAS:
                for rank in RANKS:
                    sums, counts = zip(*losses[tau, alpha, rank])
                    score = float(sum(sums)/sum(counts))
                    row = dict(outer_fold=f, tau=tau, alpha=alpha, rank=rank, inner_loss=score)
                    scores.append(row)
                    inner_rows.append(row)
            choice = min(scores, key=lambda row: (row['inner_loss'], row['rank'], row['alpha']))
            selected.append(choice)
            ridge, vectors = candidates(state, choice['alpha'], tau)
            for scene, xx in inputs.items():
                predictions[tau, scene][te] = predict(xx, state, ridge, vectors, choice['rank'])
        # Adding response columns to unrestricted ridge cannot change the first50
        # coefficients; this checks the exact distinction from shared reduced rank.
        for alpha in ALPHAS:
            ridge0, _ = candidates(state, alpha, 0.)
            ridge1, _ = candidates(state, alpha, 1.)
            unrestricted_error = max(unrestricted_error, float(np.max(np.abs(ridge0-ridge1[:, :50]))))
        print(json.dumps(dict(outer_fold=f, completed=True, elapsed_seconds=round(time.monotonic()-start, 2))), flush=True)
    assert unrestricted_error < 1e-10
    rows = []
    dimension = []
    saved = {'Y': Z, 'raw_mean': raw, 'beer': beer, 'fold': fold, 'features': features, 'targets': targets}
    for (tau, scene), pred in predictions.items():
        assert np.isfinite(pred).all()
        label = 'tau'+str(tau).replace('.', 'p')+'_'+scene
        saved[label] = pred
        rows.append(dict(method='reduced_rank_ridge', tau=tau, scenario=scene, **metrics(Z, pred)))
        r2 = 1.-np.mean((pred-Z)**2, axis=0)/np.var(Z, axis=0)
        dimension += [dict(tau=tau, scenario=scene, target=str(target), r2=float(r2[j]))
                      for j, target in enumerate(targets)]
    pd.DataFrame(rows).to_csv(args.output/'pooled_metrics.csv', index=False, float_format='%.17g')
    pd.DataFrame(dimension).to_csv(args.output/'target_metrics.csv', index=False, float_format='%.17g')
    pd.DataFrame(selected).to_csv(args.output/'selected_parameters.csv', index=False, float_format='%.17g')
    pd.DataFrame(inner_rows).to_csv(args.output/'inner_selection_grid.csv', index=False, float_format='%.17g')
    np.savez_compressed(args.output/'outer_predictions.npz', **saved)
    atomic_json(args.output/'training_identity_trace.json', trace)
    differences = []
    variance = Z.var(0)
    rng = np.random.default_rng(args.bootstrap_seed)
    indices = rng.integers(0, 250, size=(2000, 250))
    for scene in scenarios:
        base_loss = (predictions[0., scene]-Z)**2/variance
        for tau in (.5, 1.):
            diff = base_loss-(predictions[tau, scene]-Z)**2/variance
            for group, js in GROUPS.items():
                d = diff[:, js].mean(1)
                bootstrap = d[indices].mean(1)
                differences.append(dict(tau=tau, reference_tau=0., scenario=scene, targets=group,
                                        delta_macro_r2=float(d.mean()), conditional_ci025=float(np.quantile(bootstrap, .025)),
                                        conditional_ci975=float(np.quantile(bootstrap, .975)),
                                        bootstrap_unit='beer', refits=False, variance='frozen pooled250 target variance'))
    pd.DataFrame(differences).to_csv(args.output/'paired_differences.csv', index=False, float_format='%.17g')
    record = dict(status='PASS', input_sha256=sha(args.input), input_unchanged=sha(args.input)==protocol['input_sha256'],
                  implementation_sha256=sha(Path(__file__)), unrestricted_ridge_separation_max_error=unrestricted_error,
                  outer_folds=5, inner_folds_per_outer=3, parameter_grid_per_tau=len(RANKS)*len(ALPHAS),
                  tau_strategies=len(TAUS), theoretical_candidate_fits=5*len(TAUS)*(3*len(RANKS)*len(ALPHAS)+1),
                  elapsed_seconds=time.monotonic()-start, new_neural_fits=0, new_paid_calls=0,
                  input_transform=args.input_transform, bootstrap_seed=args.bootstrap_seed,
                  all_prespecified_strategies_reported=True, no_method_selected_on_outer_scores=True,
                  source_manifest={p.name:sha(p) for p in args.output.iterdir() if p.is_file()})
    atomic_json(args.output/'run_summary.json', record)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Canonical identity-locked shared inputs NPZ')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-transform', choices=['standard', 'normal_rank'], default='standard')
    parser.add_argument('--bootstrap-seed', type=int, default=2026102527)
    main(parser.parse_args())
