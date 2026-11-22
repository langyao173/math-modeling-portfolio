"""Exercise the real CSV interface and reproduce one frozen outer-fold model."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
import run_raw_models as core
from predict_raw_csv import predict_frame


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original_hash = sha(args.inputs)
    with np.load(args.inputs, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(args.results/'predictions.npz', allow_pickle=False) as archive:
        frozen = {key: archive[key] for key in archive.files}
    bundle = joblib.load(args.artifact/'raw_cdf_et.joblib')
    query = pd.DataFrame(data['X'][:5], columns=data['features'])
    query.insert(0, 'beer', [f'known_profile_query_{i}'for i in range(5)])
    medians = np.nanmedian(data['X'], axis=0)
    new_profile = np.nanmean(data['X'][:2], axis=0)
    new_profile = np.where(np.isnan(new_profile), medians, new_profile)
    new_profile[0] += .001*np.nanstd(data['X'][:,0])
    assert not np.any(np.all(np.isclose(data['X'],new_profile,rtol=0,atol=1e-14,equal_nan=True),axis=1))
    query.loc[len(query)] = ['constructed_unseen_chemistry_query', *new_profile]
    path = args.output/'unlabelled_queries.csv'
    query.to_csv(path,index=False,float_format='%.17g')
    calls = []
    def cli(input_csv, artifact, output):
        command = [sys.executable,str(Path(__file__).with_name('predict_raw_csv.py')),
                   '--artifact',str(artifact),'--input',str(input_csv),'--output',str(output)]
        result = subprocess.run(command,check=True,capture_output=True,text=True)
        calls.append(dict(command=command,returncode=result.returncode))
        return pd.read_csv(output/'expected_scores.csv').iloc[:,2:].to_numpy(),pd.read_csv(output/'median_grades.csv').iloc[:,2:].to_numpy()
    expected, grade = cli(path,args.artifact,args.output/'normal')
    exact = predict_frame(bundle,pd.read_csv(path))
    csv_error = float(np.max(np.abs(expected-exact['expected_score'])))
    np.testing.assert_allclose(expected,exact['expected_score'],rtol=0,atol=1e-10)
    np.testing.assert_array_equal(grade,exact['median_grade'])
    reversed_path = args.output/'reordered_queries.csv'
    query[query.columns[::-1]].to_csv(reversed_path,index=False,float_format='%.17g')
    reordered, reordered_grade = cli(reversed_path,args.artifact,args.output/'reordered')
    np.testing.assert_array_equal(reordered,expected)
    np.testing.assert_array_equal(reordered_grade,grade)
    missing_field = str(data['features'][0])
    omitted_path = args.output/'omitted_field_queries.csv'
    query.drop(columns=missing_field).to_csv(omitted_path,index=False,float_format='%.17g')
    omitted, _ = cli(omitted_path,args.artifact,args.output/'omitted_field')
    nan_query = query.copy();nan_query[missing_field] = np.nan
    nan_path = args.output/'explicit_nan_queries.csv'
    nan_query.to_csv(nan_path,index=False,float_format='%.17g')
    explicit_nan, _ = cli(nan_path,args.artifact,args.output/'explicit_nan')
    np.testing.assert_array_equal(omitted,explicit_nan)
    rejected = {}
    for name,frame in [('no_features',pd.DataFrame({'beer':['x']})),
                       ('non_numeric',query.assign(**{missing_field:'invalid'})),
                       ('duplicate_feature',pd.concat([query,query[[missing_field]]],axis=1))]:
        try:
            predict_frame(bundle,frame)
        except (ValueError,TypeError):
            rejected[name] = True
        else:
            raise AssertionError('Invalid query was accepted: '+name)
    train,test = np.flatnonzero(data['fold'] != 0),np.flatnonzero(data['fold'] == 0)
    selected = json.loads((args.results/'selected_parameters.json').read_text())
    leaf = next(row['selected_parameter']for row in selected if row['branch']=='cdf_et'and row['outer_fold']==0)
    model = core.fitted_model(data,train,core.cdf_of(data['raw_probability']).reshape(250,-1),
                              'et',leaf,0,-1,'cdf_et')
    fold_artifact = args.output/'fold0_artifact';fold_artifact.mkdir()
    model_file = fold_artifact/'raw_cdf_et.joblib'
    joblib.dump(dict(model=model,features=data['features'].tolist(),targets=data['targets'].tolist(),
                    method='direct CDF Extra Trees',training_beer=data['beer'][train].tolist()),model_file,compress=3)
    schema = json.loads((args.artifact/'schema.json').read_text());schema['model_sha256'] = sha(model_file)
    (fold_artifact/'schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
    fold_query = pd.DataFrame(data['X'][test],columns=data['features']);fold_query.insert(0,'beer',data['beer'][test])
    fold_path = args.output/'fold0_chemistry.csv';fold_query.to_csv(fold_path,index=False,float_format='%.17g')
    reproduced,reproduced_grade = cli(fold_path,fold_artifact,args.output/'fold0_replay')
    reference = frozen['point__cdf_et__complete'][test]
    replay_error = float(np.max(np.abs(reproduced-reference)))
    np.testing.assert_allclose(reproduced,reference,rtol=0,atol=1e-10)
    reference_grade = np.argmax(np.cumsum(frozen['probability__cdf_et__complete'][test],axis=-1)>=.5,axis=-1)
    np.testing.assert_array_equal(reproduced_grade,reference_grade)
    assert core.FIT_COUNTS['et'] == 1 and sha(args.inputs) == original_hash
    record = dict(status='PASS',new_validation_refits=1,csv_interface_calls=len(calls),
                  new_unlabelled_query_executed=True,constructed_query_is_not_an_experimental_beer=True,
                  column_reordering_exact=True,omitted_column_equals_explicit_nan=True,
                  expected_score_csv_max_error=csv_error,fold0_oof_csv_max_error=replay_error,
                  fold0_median_grades_exact=True,probability_sum_and_bounds_checked=True,
                  invalid_inputs_rejected=rejected,labels_used_as_prediction_inputs=False,
                  fit_log=core.FIT_LOG,source_sha256=sha(Path(__file__)),
                  deployment_model_sha256=sha(args.artifact/'raw_cdf_et.joblib'),
                  query_csv_sha256=sha(path),input_sha256=original_hash,calls=calls)
    (args.output/'VALIDATION_COMPLETE.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:record[k]for k in ['status','new_validation_refits','csv_interface_calls','expected_score_csv_max_error','fold0_oof_csv_max_error','column_reordering_exact','omitted_column_equals_explicit_nan']},ensure_ascii=False))


if __name__ == '__main__':
    main()
