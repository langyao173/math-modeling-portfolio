"""Predict original-panel raw-score distributions for arbitrary chemistry rows."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from run_raw_models import probabilities


def predict_frame(bundle, frame):
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate CSV field names are not accepted.')
    fields, targets = bundle['features'], bundle['targets']
    present = [name for name in fields if name in frame]
    if not present:
        raise ValueError('No recognized chemistry fields were supplied.')
    missing = [name for name in fields if name not in frame]
    x = frame.reindex(columns=fields).apply(pd.to_numeric, errors='raise').to_numpy(float)
    if np.isinf(x).any():
        raise ValueError('Infinite chemistry values are not accepted.')
    if len(frame) == 0:
        raise ValueError('No chemistry rows were supplied.')
    cdf = bundle['model'].predict(x).reshape(len(frame), len(targets), 6)
    p = probabilities(cdf)
    expected = p @ np.arange(7)
    grade = np.argmax(np.cumsum(p, axis=-1) >= .5, axis=-1)
    assert np.isfinite(p).all() and p.min() >= -1e-15
    np.testing.assert_allclose(p.sum(axis=-1), 1., rtol=0, atol=1e-14)
    assert expected.min() >= -1e-12 and expected.max() <= 6+1e-12
    assert grade.min() >= 0 and grade.max() <= 6
    identifiers = frame['beer'].tolist() if 'beer' in frame else [f'query_{i+1}'for i in range(len(frame))]
    return dict(probability=p, expected_score=expected, median_grade=grade,
                identifiers=identifiers, missing_columns=missing, targets=targets)


def write_predictions(result, output):
    output.mkdir(parents=True, exist_ok=True)
    for key, filename in [('expected_score','expected_scores.csv'), ('median_grade','median_grades.csv')]:
        frame = pd.DataFrame(result[key], columns=result['targets'])
        frame.insert(0, 'beer', result['identifiers'])
        frame.insert(0, 'query_row', np.arange(1,len(frame)+1))
        frame.to_csv(output/filename, index=False, float_format='%.12g')
    rows = []
    for i, identity in enumerate(result['identifiers']):
        for j, target in enumerate(result['targets']):
            rows.append(dict(query_row=i+1, beer=identity, target=target,
                             expected_score=float(result['expected_score'][i,j]),
                             median_grade=int(result['median_grade'][i,j]),
                             **{f'p_{k}':float(result['probability'][i,j,k])for k in range(7)}))
    pd.DataFrame(rows).to_csv(output/'score_probabilities.csv', index=False, float_format='%.12g')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, default=Path(__file__).resolve().parent/'deployment')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding='utf-8-sig', newline='') as stream:
        names = next(csv.reader(stream))
    if len(names) != len(set(names)):
        raise ValueError('Duplicate CSV field names are not accepted.')
    model_file = args.artifact/'raw_cdf_et.joblib'
    schema = json.loads((args.artifact/'schema.json').read_text())
    assert hashlib.sha256(model_file.read_bytes()).hexdigest() == schema['model_sha256']
    bundle = joblib.load(model_file)
    assert bundle['features'] == schema['features'] and bundle['targets'] == schema['targets']
    result = predict_frame(bundle, pd.read_csv(args.input))
    write_predictions(result, args.output)
    record = dict(input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  model_sha256=schema['model_sha256'], query_rows=len(result['identifiers']),
                  missing_columns=result['missing_columns'], score_scale='original panel 0–6',
                  model_refits=0, new_neural_inferences=0, method=schema['method'])
    (args.output/'prediction_context.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False))


if __name__ == '__main__':
    main()
