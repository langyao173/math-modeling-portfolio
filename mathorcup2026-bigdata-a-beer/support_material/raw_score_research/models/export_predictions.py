"""Export frozen OOF results without fitting, selection, rounding of raw means,
or replacement of the official standardized labels.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.inputs, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(args.predictions, allow_pickle=False) as archive:
        predictions = {key: archive[key] for key in archive.files}
    np.testing.assert_array_equal(data['beer'], predictions['beer'])
    np.testing.assert_array_equal(data['targets'], predictions['targets'])
    np.testing.assert_array_equal(data['fold'], predictions['fold'])
    args.output.mkdir(parents=True, exist_ok=True)
    files = []
    distributions = []
    for key, values in predictions.items():
        if key.startswith('point__') and key.endswith('__complete'):
            method = key.split('__')[1]
            path = args.output/f'raw_expected_score_{method}.csv'
            frame = pd.DataFrame(values, index=data['beer'], columns=data['targets'])
            frame.index.name = 'beer'
            frame.insert(0, 'outer_fold', data['fold'])
            frame.to_csv(path, float_format='%.12g')
            files.append(path)
        if key.startswith('probability__') and key.endswith('__complete'):
            method = key.split('__')[1]
            cdf = values.cumsum(axis=-1)
            median = np.argmax(cdf >= .5, axis=-1)
            grade_path = args.output/f'raw_median_grade_{method}.csv'
            frame = pd.DataFrame(median, index=data['beer'], columns=data['targets'])
            frame.index.name = 'beer'
            frame.insert(0, 'outer_fold', data['fold'])
            frame.to_csv(grade_path)
            files.append(grade_path)
            for row, beer in enumerate(data['beer']):
                for j, target in enumerate(data['targets']):
                    distributions.append(dict(method=method, beer=int(beer), outer_fold=int(data['fold'][row]),
                        target=str(target), expected_score=float(values[row,j] @ np.arange(7)),
                        median_grade=int(median[row,j]), **{f'p_{k}':float(values[row,j,k]) for k in range(7)}))
    path = args.output/'raw_score_distributions_long.csv'
    pd.DataFrame(distributions).to_csv(path, index=False, float_format='%.12g')
    files.append(path)
    long = pd.read_csv(path)
    np.testing.assert_allclose(long[[f'p_{k}'for k in range(7)]].sum(axis=1),1.,rtol=0,atol=5e-12)
    assert long.median_grade.between(0,6).all()
    manifest = dict(inputs_sha256=sha(args.inputs), predictions_sha256=sha(args.predictions),
                    script_sha256=sha(Path(__file__)), new_fits=0, new_selection=0,
                    score_definition='expectation of raw 0–6 panel score; continuous values are intentional',
                    grade_definition='smallest k with model CDF(k)>=0.5; an absolute-error Bayes decision',
                    predicted_objects='original 250 beers with each row predicted by its frozen outer heldout fold',
                    source_role='public raw sensory records supplement the contest data after identity audit',
                    scores_are_not_official_z_or_training_quantile_grades=True,
                    method_specific_files=True,
                    files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p))for p in files])
    (args.output/'EXPORT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(exported=len(files),distribution_rows=len(long),new_fits=0),ensure_ascii=False))


if __name__ == '__main__':
    main()
