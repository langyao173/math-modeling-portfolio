"""Training-only semantic controls that preserve, or isolate, beer style.


This post-diagnostic experiment does not replace any selected predictor.
Five independent, prespecified within-style permutations are evaluated
separately; they are not averaged into a privileged ensemble control.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from run_semantic_rrr import (HERE, PROJECT, AW, SEED, SPECS, SemanticRRR,
                              grid_predictions, paired_ci, core_indices)

OUT = HERE / 'style_controls'
PERMUTATION_SEEDS = [2026102411, 2026102412, 2026102413, 2026102414, 2026102415]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auxiliary(S, styles, kind, seed):
    """Construct labels using only the actual fitting partition."""
    if kind == 'style_only':
        levels = np.unique(styles)
        out = (styles[:, None] == levels[None, :]).astype(float)
        return out, dict(n=len(styles), active_styles=len(levels),
                         moved_rows=0, singleton_rows=0)
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(styles))
    singleton_rows = 0
    for label in np.unique(styles):
        ids = np.flatnonzero(styles == label)
        permutation[ids] = rng.permutation(ids)
        singleton_rows += int(len(ids) == 1)
    assert np.array_equal(styles[permutation], styles)
    assert np.array_equal(np.sort(permutation), np.arange(len(styles)))
    out = S[permutation].copy()
    return out, dict(n=len(styles), active_styles=len(np.unique(styles)),
                     moved_rows=int(np.sum(permutation != np.arange(len(styles)))),
                     singleton_rows=singleton_rows,
                     permutation_sha256=hashlib.sha256(permutation.tobytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wait-for-concentration', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.wait_for_concentration:
        gate = HERE.parent / 'sparse_transfer/concentration_space/COMPLETE.json'
        while not gate.exists() or not json.loads(gate.read_text()).get('complete'):
            (OUT / 'status.json').write_text(json.dumps({'stage': 'waiting_for_concentration'}))
            time.sleep(10)
    xx = pd.read_csv(PROJECT / 'data/A/1.csv').sort_values('beer').reset_index(drop=True)
    yy = pd.read_csv(PROJECT / 'data/A/4.csv').set_index('beer').loc[xx.beer]
    ss = pd.read_csv(PROJECT / 'data/A/7.csv').set_index('beer').loc[xx.beer]
    targets = yy.columns[2:].tolist()
    X = xx.iloc[:, 3:].to_numpy(float)
    Y = yy[targets].to_numpy(float)
    S = ss.iloc[:, 2:].to_numpy(float)
    styles = xx['tasting_category_fine'].astype(str).to_numpy()
    core = core_indices(targets)
    assert len(core) == 30 and len(xx) == 250
    variants = [('style_only', 2026102410)] + [
        (f'within_style_seed_{s}', s) for s in PERMUTATION_SEEDS]
    protocol = dict(
        purpose='Distinguish style-level auxiliary information from within-style review alignment',
        status='Post-diagnostic falsification control; no final-model selection or tuning from its outer scores',
        seed=SEED, outer_folds=5, inner_folds=3, variants=variants,
        permutation_scope='Each actual inner-training or outer-training partition separately; no validation labels or reviews',
        specifications=SPECS, selection_metrics=['all50', 'core30'],
        blend_regularization=.02, bootstrap_replicates=2000,
        multiple_comparisons='Exploratory conditional paired intervals, no familywise significance claim',
        controls_not_ensembled=True, inference_inputs='231 chemical columns only',
        source_hashes={str(i): sha(PROJECT / f'data/A/{i}.csv') for i in (1, 4, 7)},
        code_sha256=sha(__file__), reference_code_sha256=sha(HERE / 'run_semantic_rrr.py'))
    protocol_path = OUT / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, 'Protocol changed; use a new output directory.'
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    old_oof = np.load(AW / 'q1/oof_arrays.npz')['nested_ensemble']
    reference = np.load(HERE / 'oof_predictions.npz')
    reference_blend = np.load(HERE / 'blend_oof.npz')
    family_names = [f'{name}_{metric}' for name, _ in variants for metric in ('all50', 'core30')]
    pred = {name: np.empty_like(Y) for name in family_names}
    blend = {name: np.empty_like(Y) for name in family_names}
    folds = np.empty(len(Y), int)
    selections, permutation_records, grid_records, weights = [], [], [], []
    choices = np.array([i for i, spec in enumerate(SPECS) if spec['tau'] > 0])
    t0 = time.perf_counter()
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(X)):
        old = np.load(AW / f'nature_upgrade/adaptive/outer_{fold}_old_cache.npz')
        np.testing.assert_array_equal(old['train_indices'], tr)
        np.testing.assert_array_equal(old['test_indices'], te)
        np.testing.assert_allclose(old['test_predictions'], old_oof[te], atol=1e-13, rtol=0)
        folds[te] = fold
        sd = np.maximum(Y[tr].std(0, ddof=1), 1e-10)
        for name, seed in variants:
            kind = 'style_only' if name == 'style_only' else 'within_style'
            pp = np.empty((len(SPECS), len(tr), Y.shape[1]))
            for inner, (itr, ite) in enumerate(KFold(3, shuffle=True, random_state=SEED + fold).split(tr)):
                aux, rec = auxiliary(S[tr[itr]], styles[tr[itr]], kind,
                                     seed + 1009 * fold + 137 * inner)
                permutation_records.append(dict(variant=name, fold=fold, inner=inner, stage='inner', **rec))
                pp[:, ite] = grid_predictions(X[tr[itr]], Y[tr[itr]], aux, X[tr[ite]])
            outer_aux, rec = auxiliary(S[tr], styles[tr], kind, seed + 100003 + fold)
            permutation_records.append(dict(variant=name, fold=fold, inner=-1, stage='outer', **rec))
            selected_inner = {}
            for metric, idx in [('all50', np.arange(50)), ('core30', core)]:
                scores = np.mean(((pp[:, :, idx] - Y[tr][None, :, idx]) / sd[idx]) ** 2, axis=(1, 2))
                best = int(choices[np.argmin(scores[choices])])
                spec = SPECS[best]
                family = f'{name}_{metric}'
                model = SemanticRRR(spec).fit(X[tr], Y[tr], outer_aux)
                pred[family][te] = model.predict(X[te])
                joblib.dump(model, OUT / f'fold_{fold}_{family}.joblib')
                selected_inner[metric] = pp[best]
                residual = (old['inner_predictions'][:, idx] - Y[tr][:, idx]) / sd[idx]
                change = (pp[best][:, idx] - old['inner_predictions'][:, idx]) / sd[idx]
                weight = float(np.clip(-np.mean(residual * change) / (np.mean(change ** 2) + .02), 0, 1))
                blend[family][te] = (1 - weight) * old['test_predictions'] + weight * pred[family][te]
                weights.append(dict(fold=fold, model=family, weight=weight))
                selections.append(dict(fold=fold, family=family, spec=spec, inner_nmse=float(scores[best])))
                grid_records.extend(dict(fold=fold, variant=name, selection=metric, spec_id=i,
                                         inner_nmse=float(value), eligible=i in choices)
                                    for i, value in enumerate(scores))
            np.savez_compressed(OUT / f'fold_{fold}_{name}_selected_inner.npz',
                                train_indices=tr, test_indices=te, **selected_inner)
            (OUT / 'status.json').write_text(json.dumps(dict(stage='running', fold=fold,
                                                            variant=name, elapsed=time.perf_counter()-t0)))
        print(json.dumps(dict(fold=fold, elapsed_seconds=time.perf_counter()-t0)), flush=True)
    np.testing.assert_array_equal(folds, reference['fold'])
    np.savez_compressed(OUT / 'oof_predictions.npz', Y=Y, beer=xx.beer, fold=folds, **pred)
    np.savez_compressed(OUT / 'blend_oof.npz', Y=Y, beer=xx.beer, fold=folds, **blend)
    pd.DataFrame(weights).to_csv(OUT / 'blend_weights.csv', index=False)
    pd.DataFrame(grid_records).to_csv(OUT / 'inner_scores.csv', index=False)
    pd.DataFrame(permutation_records).to_csv(OUT / 'permutation_diagnostics.csv', index=False)
    (OUT / 'nested_selection.json').write_text(json.dumps(selections, indent=2))
    rows, contrasts = [], []
    for stage, predictions, ref in [('rrr', pred, reference), ('blend', blend, reference_blend)]:
        for family, p in predictions.items():
            r = r2_score(Y, p, multioutput='raw_values')
            metric = family.rsplit('_', 1)[-1]
            rows.append(dict(stage=stage, model=family, all50_r2=float(r.mean()), core30_r2=float(r[core].mean())))
            q = ref[f'aligned_aux_{metric}']
            contrasts.append(dict(stage=stage, control=family,
                contrast='aligned minus control',
                all50_delta=float(r2_score(Y, q) - r.mean()),
                all50_ci=paired_ci(Y, q, p, np.arange(50)),
                core30_delta=float(r2_score(Y[:, core], q[:, core]) - r[core].mean()),
                core30_ci=paired_ci(Y, q, p, core)))
    pd.DataFrame(rows).to_csv(OUT / 'model_comparison.csv', index=False)
    (OUT / 'contrasts.json').write_text(json.dumps(contrasts, indent=2))
    complete = dict(complete=True, variants=6, permutation_replicates=5, outer_folds=5,
                    fitted_outer_models=len(selections), elapsed_seconds=time.perf_counter()-t0,
                    final_selected_predictors_unchanged=True, protocol_sha256=sha(protocol_path),
                    prediction_sha256=sha(OUT / 'oof_predictions.npz'))
    (OUT / 'COMPLETE.json').write_text(json.dumps(complete, indent=2))
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
