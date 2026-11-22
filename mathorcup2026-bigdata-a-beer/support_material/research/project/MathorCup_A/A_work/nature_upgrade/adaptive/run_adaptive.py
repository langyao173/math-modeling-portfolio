# Team review and independent work must be documented truthfully.
"""Predeclared 64-candidate nested test; all choices use inner OOF only."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA_HELP, resolve_data_dir
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
from adaptive_models import AdaptiveChemicalRegressor, specifications, MODEL_VERSION

SEED = 20261023


def metrics(y, pred):
    r = r2_score(y, pred, multioutput='raw_values')
    return dict(macro_r2=float(np.mean(r)), core30_r2=float(np.mean(r[:30])),
                mean_rmse=float(np.mean(np.sqrt(np.mean((y-pred)**2, axis=0)))),
                positive_r2=int(np.sum(r > 0)))


def main(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    original = pd.read_csv(data/'1.csv')
    x = original.sort_values('beer').reset_index(drop=True)
    y = pd.read_csv(data/'4.csv').set_index('beer').loc[x.beer]
    w = pd.read_csv(data/'7.csv').set_index('beer').loc[x.beer]
    features = x.columns[3:].tolist(); targets = y.columns[2:].tolist()
    X = x[features].to_numpy(float); Y = y[targets].to_numpy(float)
    W = w.iloc[:, 2:].to_numpy(float)
    assert X.shape == (250, 231) and Y.shape == (250, 50)
    specs = specifications()
    families = ['sparse_ridge', 'sparse_rbf', 'sparse_svr', 'auxiliary_rrr']
    (out/'experiment_spec.json').write_text(json.dumps(dict(seed=SEED, specs=specs,
        features=features, targets=targets,
        auxiliary='7.csv TF-IDF used only in training folds; deployment requires chemistry only.',
        primary_selection='per-family global and per-target candidate selection by inner normalized MSE',
        interpretation='Development comparison on an already explored dataset, not a new external test.'), indent=2))
    oof = {f'{f}_{mode}': np.empty_like(Y) for f in families for mode in ['global', 'target']}
    selected = []; start = time.time()
    splits = [(f'outer_{i}', tr, te, 3, SEED+i) for i, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(X))]
    position = {v: i for i, v in enumerate(x.beer)}
    tr0, te0 = train_test_split(np.arange(len(original)), test_size=.3, random_state=0,
                               shuffle=True, stratify=original.tasting_category_fine)
    tra = np.array([position[b] for b in original.iloc[tr0].beer])
    tea = np.array([position[b] for b in original.iloc[te0].beer])
    splits.append(('author_holdout', tra, tea, 5, None))
    for name, train, test, n_inner, seed in splits:
        dest = out/f'{name}_candidates.npz'
        valid_cache = dest.exists() and 'algorithm_version' in np.load(dest).files and str(np.load(dest)['algorithm_version']) == MODEL_VERSION
        if valid_cache:
            cache = np.load(dest); inner = cache['inner_predictions']; pred = cache['test_predictions']
            print(name, 'loaded cache', flush=True)
        else:
            cv = list(KFold(n_inner, shuffle=seed is not None, random_state=seed).split(train))
            inner = np.empty((len(specs), len(train), 50))
            pred = np.empty((len(specs), len(test), 50))
            print(name, 'start', round(time.time()-start, 1), flush=True)
            for c, spec in enumerate(specs):
                for a, b in cv:
                    m = AdaptiveChemicalRegressor(spec).fit(X[train[a]], Y[train[a]], W[train[a]])
                    inner[c, b] = m.predict(X[train[b]])
                m = AdaptiveChemicalRegressor(spec).fit(X[train], Y[train], W[train])
                pred[c] = m.predict(X[test])
                if c % 10 == 9 or c == len(specs)-1:
                    print(name, 'candidate', c+1, '/', len(specs), 'elapsed', round(time.time()-start, 1), flush=True)
            np.savez_compressed(dest, algorithm_version=MODEL_VERSION, inner_predictions=inner, test_predictions=pred,
                                train_indices=train, test_indices=test)
        loss = np.mean(((inner-Y[train]) / np.maximum(Y[train].std(axis=0, ddof=1), 1e-8))**2, axis=1)
        rows = []
        chosen = {}
        for family in families:
            ix = np.array([i for i, s in enumerate(specs) if s['family'] == family])
            global_best = int(ix[np.argmin(loss[ix].mean(axis=1))])
            target_best = ix[np.argmin(loss[ix], axis=0)]
            for mode, sel in [('global', np.full(50, global_best)), ('target', target_best)]:
                key = f'{family}_{mode}'
                p = np.column_stack([pred[k, :, j] for j, k in enumerate(sel)])
                chosen[key] = dict(indices=sel.tolist(), metrics=metrics(Y[test], p))
                rows.append(dict(model=key, **metrics(Y[test], p)))
                if name.startswith('outer_'):
                    oof[key][test] = p
                else:
                    pd.DataFrame(p, columns=targets).assign(beer=x.iloc[test].beer.to_numpy()).to_csv(out/f'{name}_{key}.csv', index=False)
        (out/f'{name}_selection.json').write_text(json.dumps(chosen, indent=2))
        pd.DataFrame(rows).to_csv(out/f'{name}_metrics.csv', index=False)
        selected.append(dict(split=name, selected=chosen))
        print(name, 'done', round(time.time()-start, 1), flush=True)
    summary = [dict(model=k, **metrics(Y, p)) for k, p in oof.items()]
    pd.DataFrame(summary).to_csv(out/'model_comparison.csv', index=False)
    np.savez_compressed(out/'oof_arrays.npz', Y=Y, **oof)
    pd.DataFrame([dict(model=k, target=t, r2=float(v)) for k, p in oof.items()
                  for t, v in zip(targets, r2_score(Y, p, multioutput='raw_values'))]).to_csv(out/'dimension_metrics.csv', index=False)
    (out/'summary.json').write_text(json.dumps(dict(models=summary, elapsed_seconds=time.time()-start), indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(resolve_data_dir()), help=DATA_HELP)
    p.add_argument('--output', default=str(Path(__file__).resolve().parent))
    main(p.parse_args())
