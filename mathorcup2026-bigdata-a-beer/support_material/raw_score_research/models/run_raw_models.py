"""Prespecified beer-grouped raw-score prediction and official-score controls.

The supplied protocol and identity map are frozen before this script is run.
No mapping is inferred, no external-test raw score enters a fit, and cached
neural predictions are used only with their original outer/inner roles.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import os
import time
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261025
ALPHAS = (1., 10., 100., 1000.)
LEAVES = (2, 5)
NLL_EPSILON = 1e-6
DERIVED = ('kcalperc', 'esters_sum', 'aroma_hops_sum', 'acids_sum', 'sulfur_sum')
FIT_COUNTS = Counter()
FIT_LOG = []
CV_LOG = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pava(values):
    """Euclidean projection onto nondecreasing six-threshold CDFs in [0,1]."""
    a = np.asarray(values, float)
    assert a.shape[-1] == 6 and np.isfinite(a).all()
    flat = a.reshape(-1, 6)
    out = np.empty_like(flat)
    for row, source in enumerate(flat):
        level, count = [], []
        for value in source:
            level.append(float(value))
            count.append(1)
            while len(level) > 1 and level[-2] > level[-1]:
                total = count[-2] + count[-1]
                average = (level[-2]*count[-2] + level[-1]*count[-1])/total
                level[-2:] = [average]
                count[-2:] = [total]
        out[row] = np.clip(np.repeat(level, count), 0., 1.)
    return out.reshape(a.shape)


def probabilities(cdf):
    f = pava(cdf)
    return np.diff(np.concatenate([np.zeros_like(f[..., :1]), f, np.ones_like(f[..., :1])], axis=-1), axis=-1)


def cdf_of(probability):
    return np.cumsum(probability, axis=-1)[..., :6]


def make_model(family, parameter, seed):
    if family == 'ridge':
        regressor = Ridge(alpha=float(parameter), solver='cholesky')
    else:
        regressor = ExtraTreesRegressor(n_estimators=240, min_samples_leaf=int(parameter),
                                        max_features=.7, random_state=seed, n_jobs=1)
    return Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('scale', StandardScaler()), ('regressor', regressor)])


def record_weights(counts, indices):
    n = counts[indices].astype(float)
    weight = n/(n + np.median(n))
    return weight/weight.mean()


def fitted_model(data, train, y, family, parameter, outer, inner, branch, weighted=False):
    model = make_model(family, parameter, SEED+1000*outer+100*(inner+1))
    kwargs = {'regressor__sample_weight': record_weights(data['record_count'], train)} if weighted else {}
    model.fit(data['X'][train], y[train], **kwargs)
    FIT_COUNTS[family] += 1
    FIT_LOG.append(dict(branch=branch, outer_fold=int(outer), inner_fold=int(inner),
                        family=family, parameter=float(parameter), weighted=bool(weighted),
                        training_beers=data['beer'][train].tolist(),
                        training_records=np.flatnonzero(np.isin(data['records_beer'], data['beer'][train])).tolist()))
    return model


def select_parameter(data, train, y, family, outer, branch, objective, weighted=False):
    candidates = ALPHAS if family == 'ridge' else LEAVES
    losses = []
    for parameter in candidates:
        total, rows = 0., 0
        for inner in range(3):
            validation = train[data['inner_fold'][outer, train] == inner]
            fitting = train[data['inner_fold'][outer, train] != inner]
            assert len(validation) and len(fitting)
            model = fitted_model(data, fitting, y, family, parameter, outer, inner, branch, weighted)
            prediction = model.predict(data['X'][validation])
            if objective == 'cdf':
                prediction = pava(prediction.reshape(len(validation), 50, 6))
                error = (prediction-y[validation].reshape(len(validation), 50, 6))**2
            elif objective == 'official':
                variance = np.maximum(y[fitting].var(axis=0), 1e-12)
                error = (prediction-y[validation])**2/variance
            else:
                prediction = np.clip(prediction, 0., 6.)
                error = (prediction-y[validation])**2
            # The protocol fixes Aroma+Flavour (the first 30 targets) as the
            # inner selection objective. Remaining targets are still fitted
            # and reported, but never select a hyperparameter here.
            loss = float(error[:, :30].mean())
            total += len(validation)*loss
            rows += len(validation)
            CV_LOG.append(dict(branch=branch, outer_fold=int(outer), inner_fold=inner,
                               family=family, parameter=float(parameter), objective=objective,
                               validation_rows=len(validation), loss=loss))
        losses.append(total/rows)
    chosen = candidates[int(np.argmin(losses))]
    return chosen, dict(zip(map(str, candidates), losses))


def cases(data, test, outer):
    x = data['X'][test]
    derived = [list(data['features']).index(field) for field in DERIVED]
    result = {'complete': x.copy(), 'derived_only_missing': x.copy()}
    result['derived_only_missing'][:, derived] = np.nan
    for level in (10, 30, 50):
        masked = x.copy()
        rng = np.random.default_rng(20261023 + 1000*outer + level)
        masked[rng.random(masked.shape) < level/100] = np.nan
        masked[:, derived] = np.nan
        result[f'MCAR_{level}_strict'] = masked
    return result


def affine_prediction(training_z, training_y, testing_z):
    centered_z = training_z-training_z.mean(axis=0)
    centered_y = training_y-training_y.mean(axis=0)
    denominator = np.sum(centered_z**2, axis=0)
    slope = np.maximum(np.divide(np.sum(centered_z*centered_y, axis=0), denominator,
                                out=np.zeros_like(denominator), where=denominator > 1e-15), 0.)
    intercept = training_y.mean(axis=0)-slope*training_z.mean(axis=0)
    return np.clip(intercept+slope*testing_z, 0., 6.), slope, intercept


def isotonic_prediction(training_z, training_y, testing_z, cdf=False):
    q = training_z.shape[1]
    out = np.empty((len(testing_z), q, 6)) if cdf else np.empty_like(testing_z)
    for target in range(q):
        if cdf:
            for threshold in range(6):
                model = IsotonicRegression(increasing=False, y_min=0., y_max=1., out_of_bounds='clip')
                out[:, target, threshold] = model.fit(training_z[:, target], training_y[:, target, threshold]).predict(testing_z[:, target])
                FIT_COUNTS['cdf_isotonic'] += 1
        else:
            model = IsotonicRegression(increasing=True, y_min=0., y_max=6., out_of_bounds='clip')
            out[:, target] = model.fit(training_z[:, target], training_y[:, target]).predict(testing_z[:, target])
            FIT_COUNTS['mean_isotonic'] += 1
    return probabilities(out) if cdf else out


def point_metrics(truth, prediction):
    error = prediction-truth
    mse = np.mean(error**2, axis=0)
    variance = truth.var(axis=0)
    r2 = 1.-np.divide(mse, variance, out=np.full_like(mse, np.nan), where=variance > 1e-15)
    return dict(mean_mae=float(np.mean(np.abs(error))), pooled_mean_rmse=float(np.sqrt(mse.mean())),
                macro_mean_rmse=float(np.sqrt(mse).mean()), macro_r2=float(np.nanmean(r2)))


def distribution_metrics(truth, prediction):
    true_cdf, predicted_cdf = cdf_of(truth), cdf_of(prediction)
    cdf_mse = np.mean((predicted_cdf-true_cdf)**2)
    irreducible = np.mean(true_cdf*(1-true_cdf))
    scoring_copy = np.maximum(prediction, NLL_EPSILON)
    scoring_copy /= scoring_copy.sum(axis=-1, keepdims=True)
    nll = -np.mean(np.sum(truth*np.log(scoring_copy), axis=-1))
    median = np.argmax(np.cumsum(prediction, axis=-1) >= .5, axis=-1)
    absolute_grade_error = np.abs(np.arange(7)[None, None, :]-median[..., None])
    return dict(rps=float(cdf_mse+irreducible), cdf_mse=float(cdf_mse),
                irreducible_rps=float(irreducible), nll=float(nll),
                individual_grade_mae=float(np.mean(np.sum(truth*absolute_grade_error, axis=-1))),
                individual_grade_exact=float(np.mean(np.sum(truth*(absolute_grade_error == 0), axis=-1))),
                individual_grade_within_one=float(np.mean(np.sum(truth*(absolute_grade_error <= 1), axis=-1))))


def summarize(data, point, probability, official, output):
    groups = {'all50': np.arange(50), 'core30': np.arange(30), 'other20': np.arange(30, 50)}
    rows, dimensions = [], []
    for target_name in ('record_equal', 'taster_equal_sensitivity'):
        mean_truth = data['raw_mean'] if target_name == 'record_equal' else data['taster_mean']
        probability_truth = data['raw_probability'] if target_name == 'record_equal' else data['taster_probability']
        for (method, scenario), prediction in point.items():
            for group, targets in groups.items():
                row = dict(method=method, scenario=scenario, target_weighting=target_name, targets=group,
                           **point_metrics(mean_truth[:, targets], prediction[:, targets]))
                if (method, scenario) in probability:
                    row.update(distribution_metrics(probability_truth[:, targets], probability[(method, scenario)][:, targets]))
                rows.append(row)
            if target_name == 'record_equal' and scenario == 'complete':
                for j, target in enumerate(data['targets']):
                    row = dict(method=method, target=str(target), **point_metrics(mean_truth[:, j:j+1], prediction[:, j:j+1]))
                    if (method, scenario) in probability:
                        row.update(distribution_metrics(probability_truth[:, j:j+1], probability[(method, scenario)][:, j:j+1]))
                    dimensions.append(row)
    pd.DataFrame(rows).to_csv(output/'raw_metrics.csv', index=False)
    pd.DataFrame(dimensions).to_csv(output/'raw_per_target_metrics.csv', index=False)
    zrows = []
    for (method, scenario), prediction in official.items():
        for group, targets in groups.items():
            zrows.append(dict(method=method, scenario=scenario, targets=group,
                              **point_metrics(data['Yz'][:, targets], prediction[:, targets])))
    pd.DataFrame(zrows).to_csv(output/'official_metrics.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--protocol-lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.protocol_lock.is_file()
    protocol = json.loads(args.protocol_lock.read_text())
    assert not args.output.exists(), 'Use a new output directory; prior frozen results are immutable.'
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    with np.load(args.inputs, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    assert data['X'].shape == (250, 231) and data['raw_probability'].shape == (250, 50, 7)
    assert np.array_equal(data['raw_mean'], data['raw_probability'] @ np.arange(7))
    source_hash = sha(args.inputs)
    (args.output/'INPUTS_FROZEN.json').write_text(json.dumps(dict(inputs_sha256=source_hash,
        protocol_sha256=sha(args.protocol_lock), implementation_sha256=sha(Path(__file__)),
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        input_beer=data['beer'].tolist(), outer_fold=data['fold'].tolist(), protocol=protocol),
        ensure_ascii=False, indent=2)+'\n')
    point, probability, official, selection, fold_log, masks = {}, {}, {}, [], [], {}
    def put(store, key, indices, values):
        if key not in store:
            store[key] = np.full((250,)+values.shape[1:], np.nan)
        store[key][indices] = values
    raw_mean = data['raw_mean']
    empirical_cdf = cdf_of(data['raw_probability'])
    flat_cdf = empirical_cdf.reshape(250, -1)
    for outer in range(5):
        train, test = np.flatnonzero(data['fold'] != outer), np.flatnonzero(data['fold'] == outer)
        assert len(train) == 200 and len(test) == 50
        for inner in range(3):
            assert np.intersect1d(test, np.flatnonzero(data['inner_fold'][outer] == inner)).size == 0
        train_records = np.flatnonzero(np.isin(data['records_beer'], data['beer'][train]))
        test_records = np.flatnonzero(np.isin(data['records_beer'], data['beer'][test]))
        assert np.intersect1d(train_records, test_records).size == 0
        fold_log.append(dict(outer_fold=outer, training_beers=data['beer'][train].tolist(),
                             testing_beers=data['beer'][test].tolist(), training_records=train_records.tolist(),
                             testing_records=test_records.tolist()))
        views = cases(data, test, outer)
        for name, view in views.items():
            masks[f'fold{outer}_{name}'] = np.isnan(view)
            base_probability = np.broadcast_to(data['raw_probability'][train].mean(axis=0), (len(test),50,7)).copy()
            put(probability, ('training_distribution',name), test, base_probability)
            put(point, ('training_distribution',name), test, base_probability @ np.arange(7))
        for branch, family, target, objective, weighted in [
            ('direct_ridge','ridge',raw_mean,'mean',False),
            ('direct_et','et',raw_mean,'mean',False),
            ('cdf_ridge','ridge',flat_cdf,'cdf',False),
            ('cdf_et','et',flat_cdf,'cdf',False),
            ('official_ridge','ridge',data['Yz'],'official',False),
            ('official_count_weighted_ridge','ridge',data['Yz'],'official',True)]:
            parameter, losses = select_parameter(data, train, target, family, outer, branch, objective, weighted)
            selection.append(dict(branch=branch, outer_fold=outer, selected_parameter=float(parameter), inner_losses=losses))
            model = fitted_model(data, train, target, family, parameter, outer, -1, branch, weighted)
            for scenario, view in views.items():
                prediction = model.predict(view)
                if objective == 'cdf':
                    p = probabilities(prediction.reshape(len(test),50,6))
                    put(probability,(branch,scenario),test,p)
                    put(point,(branch,scenario),test,p @ np.arange(7))
                elif objective == 'official':
                    put(official,(branch,scenario),test,prediction)
                else:
                    put(point,(branch,scenario),test,np.clip(prediction,0.,6.))
        inner_z = data['frozen_inner_z'][outer,train]
        outer_z = data['frozen_outer_z'][test]
        assert np.isfinite(inner_z).all() and np.isnan(data['frozen_inner_z'][outer,test]).all()
        calibrated,slope,intercept = affine_prediction(inner_z,raw_mean[train],outer_z)
        FIT_COUNTS['affine_targets'] += 50
        put(point,('z_affine','complete'),test,calibrated)
        put(point,('z_isotonic','complete'),test,isotonic_prediction(inner_z,raw_mean[train],outer_z))
        cdf_probability = isotonic_prediction(inner_z,empirical_cdf[train],outer_z,cdf=True)
        put(probability,('z_cdf_isotonic','complete'),test,cdf_probability)
        put(point,('z_cdf_isotonic','complete'),test,cdf_probability @ np.arange(7))
        np.savez_compressed(args.output/f'calibration_fold{outer}.npz',train_beer=data['beer'][train],
                            test_beer=data['beer'][test],inner_z=inner_z,outer_z=outer_z,
                            affine_slope=slope,affine_intercept=intercept,
                            affine_prediction=calibrated)
        print(json.dumps(dict(fold_complete=outer+1,seconds=round(time.perf_counter()-start,2),fits=dict(FIT_COUNTS))),flush=True)
    official[('frozen_half','complete')] = data['frozen_outer_z']
    official[('frozen_groupwise','complete')] = data['frozen_groupwise_z']
    for store in (point,probability,official):
        assert all(np.isfinite(values).all() for values in store.values())
    for values in probability.values():
        assert values.min() >= -1e-15
        np.testing.assert_allclose(values.sum(axis=-1),1.,rtol=0,atol=1e-14)
    np.savez_compressed(args.output/'predictions.npz',beer=data['beer'],targets=data['targets'],fold=data['fold'],
                        **{'point__'+m+'__'+s:p for (m,s),p in point.items()},
                        **{'probability__'+m+'__'+s:p for (m,s),p in probability.items()},
                        **{'official__'+m+'__'+s:p for (m,s),p in official.items()})
    np.savez_compressed(args.output/'test_masks.npz',**masks)
    summarize(data,point,probability,official,args.output)
    (args.output/'selected_parameters.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
    (args.output/'fold_record_masks.json').write_text(json.dumps(fold_log,ensure_ascii=False,indent=2)+'\n')
    (args.output/'fit_log.json').write_text(json.dumps(FIT_LOG,ensure_ascii=False,indent=2)+'\n')
    pd.DataFrame(CV_LOG).to_csv(args.output/'inner_scores.csv',index=False)
    assert sha(args.inputs) == source_hash
    summary = dict(status='PASS',new_fits=dict(FIT_COUNTS),new_base_model_fits=sum(FIT_COUNTS[f]for f in ['ridge','et']),
                   new_neural_inferences=0,new_paid_calls=0,seconds=time.perf_counter()-start,
                   calibration_base='pre-fixed equal mixture of Y16 and Native50',
                   calibration_training='each outer training set own internal OOF only',
                   test_target_used_to_fit=False,identity_mapping_estimated_here=False,
                   inputs_sha256=source_hash,protocol_sha256=sha(args.protocol_lock),
                   implementation_sha256=sha(Path(__file__)),
                   primary_rps='observed-rating expectation, beer then target equally weighted',
                   nll_probability_floor=NLL_EPSILON,
                   output_sha256={p.name:sha(p)for p in args.output.iterdir()if p.is_file()})
    (args.output/'RUN_COMPLETE.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
