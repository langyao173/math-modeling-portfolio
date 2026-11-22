"""Explain saved OOF errors in the fixed training-fold sensory subspace.

This is a post-hoc diagnostic, never a model-selection or tuning procedure.
The exact Pythagorean decomposition applies to the full 50-dimensional
training-standardized Euclidean loss, not macro R2 or a 30-target subset.
No model is fitted and no new prediction is generated.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def diagnose(Y, fold, predictions, grades, truth_grades, out):
    lowrank = 'raw231_y16_no_residual'
    native = 'raw231_y50_native'
    if lowrank not in predictions or native not in predictions:
        return {'complete': False, 'reason': 'Await both complete fivefold OOF candidates.'}
    n, d = Y.shape
    assert (n, d) == (250, 50)
    components = {name: np.empty((n, 3)) for name in predictions}
    truncation = np.empty(n)
    fold_records = []
    provenance = []
    maximum_error = 0.
    for f in range(5):
        path = HERE/'checkpoints'/f'raw231_y16_fold{f}_complete.npz'
        a = np.load(path, allow_pickle=False)
        tr, te = a['train_indices'], a['test_indices']
        assert np.array_equal(te, np.flatnonzero(fold == f))
        assert np.array_equal(tr, np.flatnonzero(fold != f))
        mean, scale = a['target_mean'], a['target_scale']
        V = a['target_components'].T
        # The original benchmark stores the fitted components and target
        # standardizer; PCA's center is exactly the mean of this training Z.
        center = ((Y[tr]-mean)/scale).mean(0)
        assert np.max(np.abs(V.T@V-np.eye(16))) < 1e-10
        assert np.allclose(mean, Y[tr].mean(0), atol=1e-13, rtol=1e-13)
        assert np.allclose(scale, Y[tr].std(0, ddof=1), atol=1e-13, rtol=1e-13)
        true_z = (Y[te]-mean)/scale-center
        true_orthogonal = true_z-(true_z@V)@V.T
        truncation[te] = np.square(true_orthogonal).sum(1)/d
        for name, pred in predictions.items():
            pred_z = (pred[te]-mean)/scale-center
            error = true_z-pred_z
            projection = error@V
            orthogonal = error-projection@V.T
            inside = np.square(projection).sum(1)/d
            outside = np.square(orthogonal).sum(1)/d
            total = np.square(error).sum(1)/d
            residual = float(np.max(np.abs(total-inside-outside)))
            maximum_error = max(maximum_error, residual)
            assert residual < 1e-12
            components[name][te] = np.stack([inside, outside, total], axis=1)
            if name == lowrank:
                assert np.max(np.abs(pred_z-(pred_z@V)@V.T)) < 1e-10
                assert np.allclose(outside, truncation[te], atol=1e-12, rtol=1e-12)
            fold_records.append(dict(model=name, fold=f, rows=len(te),
                subspace_mse=float(inside.mean()), orthogonal_mse=float(outside.mean()),
                total_standardized_mse=float(total.mean()),
                zero_orthogonal_prediction_mse=float(truncation[te].mean()),
                orthogonal_error_reduction_vs_zero=float((truncation[te]-outside).mean()),
                exact_decomposition_max_error=residual,
                pca_training_variance_retained=float(a['target_explained_variance_ratio'].sum())))
        provenance.append(dict(fold=f, checkpoint=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pd.DataFrame(fold_records).to_csv(out/'subspace_error_by_fold.csv', index=False)
    rows = [dict(model=name, rows=n,
        subspace_mse=float(v[:,0].mean()), orthogonal_mse=float(v[:,1].mean()),
        total_standardized_mse=float(v[:,2].mean()),
        zero_orthogonal_prediction_mse=float(truncation.mean()),
        orthogonal_error_reduction_vs_zero=float((truncation-v[:,1]).mean()))
        for name, v in components.items()]
    pd.DataFrame(rows).to_csv(out/'subspace_error_budget.csv', index=False)
    np.savez_compressed(out/'subspace_error_by_beer.npz', fold=fold,
        zero_orthogonal_prediction_mse=truncation,
        component_names=np.asarray(['subspace_mse','orthogonal_mse','total_standardized_mse']),
        **components)

    # Direct pairing addresses the native-output versus low-rank tradeoff.
    # This added comparison is declared post hoc; deployment still uses the
    # already declared core30-first internal criterion.
    rng = np.random.default_rng(20261023+91219)
    ids = rng.integers(0, n, size=(2000, n))
    error_low = np.square(Y-predictions[lowrank])
    error_native = np.square(Y-predictions[native])
    sst = np.square(Y-Y.mean(0)).sum(0)
    delta_r = (error_low.sum(0)-error_native.sum(0))/sst
    samples = []
    for batch in ids:
        variance = np.square(Y[batch]-Y[batch].mean(0)).sum(0)
        delta = (error_low[batch].sum(0)-error_native[batch].sum(0))/variance
        samples.append([delta.mean(), delta[:30].mean()])
    samples = np.asarray(samples)
    grade_error_native = np.abs(grades[native]-truth_grades)
    grade_error_low = np.abs(grades[lowrank]-truth_grades)
    comparisons = []
    for gi, (group, index) in enumerate([('all50',np.arange(50)), ('core30',np.arange(30))]):
        comparisons.append(dict(model=native, reference=lowrank, targets=group,
            metric='r2', delta=float(delta_r[index].mean()),
            ci025=float(np.quantile(samples[:,gi],.025)), ci975=float(np.quantile(samples[:,gi],.975))))
        a, b = grade_error_native[:,index], grade_error_low[:,index]
        for metric, differences in [('grade_mae',a.mean(1)-b.mean(1)),
            ('grade_exact',(a==0).mean(1)-(b==0).mean(1)),
            ('grade_within_one',(a<=1).mean(1)-(b<=1).mean(1))]:
            draws = differences[ids].mean(1)
            comparisons.append(dict(model=native, reference=lowrank, targets=group,
                metric=metric, delta=float(differences.mean()),
                ci025=float(np.quantile(draws,.025)), ci975=float(np.quantile(draws,.975))))
    pd.DataFrame(comparisons).to_csv(out/'native_vs_lowrank_paired_bootstrap.csv', index=False)
    protocol = dict(complete=True, diagnostic_type='Post-hoc explanation of fixed complete OOF predictions; not used for tuning.',
        coordinate_system='For each outer fold, (Y-training_mean)/training_sample_sd minus the training PCA center.',
        identity='||e||^2/50 = ||e V||^2/50 + ||e (I-V V^T)||^2/50, where V has the 16 frozen training-fold principal directions.',
        lowrank_identity='The low-rank predictor has zero orthogonal component, so its orthogonal error equals the zero-orthogonal-prediction truncation loss.',
        limitation='This decomposition is full-50 training-standardized Euclidean MSE. It is not a decomposition of macro R2 or the core30 subset; PCA projection mixes target coordinates.',
        oracle_interpretation='The zero-orthogonal-prediction loss is the truncation cost of this chosen subspace, not intrinsic irreducible noise or a universal prediction lower bound.',
        bootstrap='2,000 paired beer resamples of fixed OOF predictions; native minus lowrank; no refit or simultaneous/multiplicity adjustment.',
        seed=20261023+91219, exact_decomposition_max_error=maximum_error,
        source_checkpoints=provenance, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'subspace_error_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    return protocol
