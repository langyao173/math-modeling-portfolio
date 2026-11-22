"""Auditable fusion of old, adaptive, GBR and CatBoost predictions.

All choices use the enclosing training partition. Retrospective development
evaluation is not an independent blind validation or a preregistered study.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
UP = HERE.parent
WORK = UP.parent
for path in [UP/'adaptive', WORK/'q1', UP/'benchmark', UP/'catboost']:
    sys.path.insert(0, str(path))
from prepare_fusion import dataset, choose_adaptive, shrink_weights, metric, NEW_FAMILIES
from adaptive_models import MODEL_VERSION, specifications, PerTargetSelection


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def choose_trees(cache, Y, targetwise=True):
    ip = cache['inner_predictions']
    tp = cache['test_predictions'] if 'test_predictions' in cache else None
    scale = np.maximum(Y.std(axis=0, ddof=1), 1e-8)
    loss = np.mean(((ip-Y)/scale)**2, axis=1)
    gi = int(np.argmin(loss.mean(axis=1)))
    ti = np.argmin(loss, axis=0)
    inp, out = [ip[gi]], ([] if tp is None else [tp[gi]])
    if targetwise:
        inp.append(np.column_stack([ip[k, :, j] for j, k in enumerate(ti)]))
        if tp is not None:
            out.append(np.column_stack([tp[k, :, j] for j, k in enumerate(ti)]))
    return inp, out, dict(global_index=gi, target_indices=ti.tolist())


def partition(tag, nbase, x, Y):
    a = np.load(UP/'adaptive'/f'{tag}_candidates.npz')
    assert str(a['algorithm_version']) == MODEL_VERSION
    full = tag == 'full_data'
    tr = np.arange(len(Y)) if full else a['train_indices']
    te = None if full else a['test_indices']
    selected = {}
    if full:
        old = np.load(UP/'adaptive/full_data_old_cache.npz')
        np.testing.assert_allclose(old['Y'], Y, rtol=0, atol=0)
        inner, test = [old['inner_predictions']], []
    elif tag == 'author_holdout':
        oi = np.load(UP/'author_protocol_baseline/inner_selection_arrays.npz')
        ot = np.load(UP/'author_protocol_baseline/holdout_arrays.npz')
        imap = pd.Series(np.arange(len(oi['beer'])), index=oi['beer'])
        tmap = pd.Series(np.arange(len(ot['beer'])), index=ot['beer'])
        ii = imap.loc[x.iloc[tr].beer].to_numpy()
        tt = tmap.loc[x.iloc[te].beer].to_numpy()
        np.testing.assert_allclose(oi['Y'][ii], Y[tr], rtol=0, atol=1e-14)
        np.testing.assert_allclose(ot['Y'][tt], Y[te], rtol=0, atol=1e-14)
        inner, test = [oi['ensemble'][ii]], [ot['ensemble'][tt]]
    else:
        old = np.load(UP/'adaptive'/f'{tag}_old_cache.npz')
        assert np.array_equal(tr, old['train_indices'])
        assert np.array_equal(te, old['test_indices'])
        inner, test = [old['inner_predictions']], [old['test_predictions']]
    ai, ap, selected['adaptive'] = choose_adaptive(
        a['inner_predictions'], None if full else a['test_predictions'], Y[tr])
    inner += ai
    test += ap
    names = ['old_ensemble'] + NEW_FAMILIES
    for folder, targetwise, extra_names in [
            ('benchmark', True, ['gbr_global', 'gbr_target']),
            ('catboost', False, ['catboost_global'])]:
        if (folder == 'benchmark' and nbase < 7) or (folder == 'catboost' and nbase < 8):
            continue
        c = np.load(UP/folder/f'{tag}_candidates.npz')
        if not full:
            assert np.array_equal(tr, c['train_indices'])
            assert np.array_equal(te, c['test_indices'])
        if 'Y' in c:
            np.testing.assert_allclose(Y[tr], c['Y'], rtol=0, atol=1e-14)
        if 'y_train' in c:
            np.testing.assert_allclose(Y[tr], c['y_train'], rtol=0, atol=1e-14)
        if 'features' in c:
            assert np.array_equal(c['features'], x.columns[3:].to_numpy())
        ci, cp, selected[folder] = choose_trees(c, Y[tr], targetwise)
        inner += ci
        if not full:
            test += cp
        names += extra_names
    P = np.stack(inner)
    assert len(names) == nbase and np.isfinite(P).all()
    gw, tw = shrink_weights(P, Y[tr], eta=.20)
    assert np.all(gw >= -1e-10) and np.all(tw >= -1e-10)
    np.testing.assert_allclose(gw.sum(), 1, atol=1e-10)
    np.testing.assert_allclose(tw.sum(axis=0), 1, atol=1e-10)
    out = dict(inner_predictions=P, train_indices=tr, global_weights=gw,
               target_weights=tw, base_names=np.asarray(names))
    if not full:
        T = np.stack(test)
        assert T.shape == (nbase, len(te), 50) and np.isfinite(T).all()
        out.update(test_indices=te, test_predictions=T,
                   global_prediction=np.einsum('m,mij->ij', gw, T),
                   target_prediction=np.einsum('mj,mij->ij', tw, T))
    np.savez_compressed(HERE/f'{tag}_{nbase}base.npz', **out)
    save_json(HERE/f'{tag}_{nbase}base_selection.json',
              dict(bases=names, eta=.20, selection=selected,
                   training_rows=tr.tolist(), test_rows=None if full else te.tolist(),
                   global_weights=gw.tolist(), target_weights=tw.tolist()))
    return out


def paired_bootstrap(Y, a, b, seed=20261023, nboot=3000):
    """Paired row bootstrap of fixed predictions; does not refit the models."""
    rng = np.random.default_rng(seed)
    differr = (Y-b)**2 - (Y-a)**2
    diffs = []
    for start in range(0, nboot, 100):
        ix = rng.integers(0, len(Y), size=(min(100, nboot-start), len(Y)))
        yy = Y[ix]
        sst = ((yy-yy.mean(axis=1, keepdims=True))**2).sum(axis=1)
        diffs.extend(np.mean(differr[ix].sum(axis=1)/np.maximum(sst, 1e-15), axis=1))
    return dict(delta_macro_r2=float(metric(Y, a)['macro_r2']-metric(Y, b)['macro_r2']),
                paired_row_bootstrap_95ci=np.quantile(diffs, [.025, .975]).tolist(),
                bootstrap_replicates=nboot,
                scope='Fixed predictions; conditional uncertainty only. No refitting or model-selection uncertainty.')


def evaluate(variants):
    x, X, Y, W, targets = dataset()
    old = np.load(WORK/'q1/oof_arrays.npz')
    np.testing.assert_allclose(old['Y'], Y, rtol=0, atol=1e-14)
    preds = {'old_ensemble': old['nested_ensemble']}
    for key in ['extra_trees', 'random_forest']:
        if key in old:
            preds['old_'+key] = old[key]
    rows, dims, author_rows, comparisons = [], [], [], {}
    author_preds, author_Y = {}, None
    for nbase in variants:
        gp, tp = np.full_like(Y, np.nan), np.full_like(Y, np.nan)
        fold = np.empty(len(Y), int)
        for k in range(5):
            p = partition(f'outer_{k}', nbase, x, Y)
            te = p['test_indices']
            gp[te], tp[te], fold[te] = p['global_prediction'], p['target_prediction'], k
        assert np.isfinite(gp).all() and np.isfinite(tp).all()
        np.savez_compressed(HERE/f'{nbase}base_oof.npz', Y=Y, beer=x.beer.to_numpy(),
                            targets=np.asarray(targets), outer_fold=fold,
                            global_prediction=gp, target_prediction=tp)
        preds[f'fusion{nbase}_global'] = gp
        preds[f'fusion{nbase}_shrink'] = tp
        ap = partition('author_holdout', nbase, x, Y)
        author_Y = Y[ap['test_indices']]
        author_preds[f'fusion{nbase}_global'] = ap['global_prediction']
        author_preds[f'fusion{nbase}_shrink'] = ap['target_prediction']
        author_preds['old_ensemble'] = ap['test_predictions'][0]
    for label, file, keys in [
            ('gbr', UP/'benchmark/gbr_oof_predictions.npz', ['gbr_global','gbr_targetwise','gbr_shrunk_selection']),
            ('catboost', UP/'catboost/oof_arrays.npz', ['catboost_global'])]:
        if file.exists():
            c = np.load(file)
            np.testing.assert_allclose(c['Y'], Y, rtol=0, atol=1e-14)
            for key in keys:
                preds[key] = c[key]
    for name, p in preds.items():
        rows.append(dict(model=name, **metric(Y, p)))
        r = r2_score(Y, p, multioutput='raw_values')
        dims.extend(dict(model=name, target=t, r2=float(v)) for t, v in zip(targets, r))
    for name, p in author_preds.items():
        author_rows.append(dict(model=name, **metric(author_Y, p)))
    for nbase in variants:
        for rule in ['global', 'shrink']:
            key = f'fusion{nbase}_{rule}'
            comparisons[key+'_vs_old_oof'] = paired_bootstrap(Y, preds[key], preds['old_ensemble'])
            comparisons[key+'_vs_old_author'] = paired_bootstrap(author_Y, author_preds[key], author_preds['old_ensemble'])
    if 8 in variants and 7 in variants:
        comparisons['fusion8_vs_fusion7_oof'] = paired_bootstrap(Y, preds['fusion8_shrink'], preds['fusion7_shrink'])
    pd.DataFrame(rows).to_csv(HERE/'model_comparison.csv', index=False)
    pd.DataFrame(dims).to_csv(HERE/'dimension_metrics.csv', index=False)
    pd.DataFrame(author_rows).to_csv(HERE/'author_holdout_metrics.csv', index=False)
    save_json(HERE/'paired_bootstrap.json', comparisons)
    save_json(HERE/'summary.json', dict(nested=rows, author=author_rows, comparisons=comparisons))
    print(json.dumps(dict(nested=rows, author=author_rows), indent=2), flush=True)


def fit_final(nbase=8):
    import joblib
    from fusion_model import load_fusion, make_old_bundle_reference
    import shutil
    x, X, Y, W, targets = dataset()
    p = partition('full_data', nbase, x, Y)
    selection = json.loads((HERE/f'full_data_{nbase}base_selection.json').read_text())['selection']
    old_spec = json.loads((UP/'adaptive/full_data_old_selection.json').read_text())
    old_path = WORK/'q1/sensory_ensemble.joblib'
    old_bundle = joblib.load(old_path)
    assert old_bundle['features'] == x.columns[3:].tolist() and old_bundle['targets'] == targets
    assert [m.spec for m in old_bundle['models']] == [old_spec['specs'][f] for f in old_spec['families']]
    np.testing.assert_array_equal(old_bundle['weights'], np.asarray(old_spec['weights']))
    np.testing.assert_array_equal(old_bundle['reference_y'], Y)
    old_reference = make_old_bundle_reference(HERE, old_path, old_bundle)
    adaptive = [PerTargetSelection(specifications(), selection['adaptive'][family]).fit(X, Y, W).compact_for_inference()
                for family in NEW_FAMILIES]
    gbr_models = []
    if nbase >= 7:
        saved_gbr_selection = json.loads((UP/'benchmark/full_data_selection.json').read_text())
        assert selection['benchmark']['global_index'] == saved_gbr_selection['global_index']
        assert selection['benchmark']['target_indices'] == saved_gbr_selection['targetwise_indices']
        gb = joblib.load(UP/'benchmark/final_gbr_models.joblib')
        assert gb['features'] == x.columns[3:].tolist() and gb['targets'] == targets
        assert gb['global_model'].specs_ == [saved_gbr_selection['global_spec']] * len(targets)
        assert gb['targetwise_model'].specs_ == [saved_gbr_selection['targetwise_specs'][t] for t in targets]
        gbr_models = [gb['global_model'], gb['targetwise_model']]
    final = dict(features=x.columns[3:].tolist(), targets=targets, reference_y=Y,
                 old_bundle_reference=old_reference, adaptive=adaptive,
                 gbr_models=gbr_models,
                 base_names=p['base_names'].tolist(), global_weights=p['global_weights'],
                 target_weights=p['target_weights'], model_version='research_fusion_exact_rrr_v1',
                 inference_storage='selected_outputs_with_shared_baseline_v1',
                 source_protocol='five outer / three inner, seed 20261023; full refit seed+99',
                 scope='Research predictor; Q3 constraints in this package use the separately frozen original predictor.')
    joblib.dump(final, HERE/'final_fusion_models.joblib', compress=3)
    if nbase == 8:
        cb_selection = json.loads((UP/'catboost/full_data_selection.json').read_text())
        assert selection['catboost']['global_index'] == cb_selection['global_index']
        for ext in ['cbm', 'json']:
            shutil.copyfile(UP/'catboost'/f'final_global_model.{ext}', HERE/f'catboost_branch.{ext}')
    loaded = load_fusion(HERE)
    probe = X[:7].copy()
    probe[:, ::2] = np.nan
    q = loaded.predict(probe)
    assert q.shape == (7, 50) and np.isfinite(q).all()
    full_pred = loaded.predict(X)
    pd.concat([x.iloc[:, :3], pd.DataFrame(full_pred, columns=targets)], axis=1).to_csv(
        HERE/'full_fit_predictions_not_validation.csv', index=False)
    save_json(HERE/'final_model_interface_check.json',
              dict(probe_rows=7, missing_columns=116, shape=list(q.shape), all_finite=True,
                   diagnostic_only='Saved model loads and handles missing input; not a validation score.'))
    save_json(HERE/'final_manifest.json',
              dict(model_version=final['model_version'], base_names=final['base_names'],
                   inference_storage=final['inference_storage'], old_bundle_reference=old_reference,
                   primary_prediction='target_shrink', global_weights=p['global_weights'].tolist(),
                   n_training_rows=250, features=final['features'], targets=targets,
                   catboost_version='1.2.10' if nbase == 8 else None,
                   q3_predictor='Original A_work/q1 model, frozen separately; do not transfer its constraint certificate.'))
    print('Final fusion fitted, serialized and reloaded.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', default='5,7,8')
    parser.add_argument('--fit-final', action='store_true')
    args = parser.parse_args()
    variants = [int(v) for v in args.variants.split(',')]
    evaluate(variants)
    if args.fit_final:
        fit_final(max(variants))
