"""Refit the direct CDF-ET on all 250 training beers for future chemistry rows.

This is deployment training, not an additional heldout performance estimate.
The fixed two-value leaf grid is reselected by training-only three-fold CV.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import joblib
import numpy as np
from sklearn.model_selection import KFold
import run_raw_models as core


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--protocol-lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'deployment')
    args = parser.parse_args()
    assert not args.output.exists(), 'Use a distinct deployment directory.'
    with np.load(args.inputs, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    assert args.protocol_lock.is_file()
    fold = np.empty((1, len(data['beer'])), int)
    for index, (_, validation) in enumerate(KFold(3, shuffle=True, random_state=core.SEED).split(data['X'])):
        fold[0, validation] = index
    local = dict(data)
    local['inner_fold'] = fold
    y = core.cdf_of(data['raw_probability']).reshape(len(data['beer']), -1)
    train = np.arange(len(data['beer']))
    start = time.perf_counter()
    parameter, losses = core.select_parameter(local, train, y, 'et', 0, 'deployment_cdf_et', 'cdf')
    model = core.fitted_model(local, train, y, 'et', parameter, 0, -1, 'deployment_cdf_et')
    assert sum(core.FIT_COUNTS.values()) == 7
    args.output.mkdir(parents=True)
    artifact = args.output/'raw_cdf_et.joblib'
    joblib.dump(dict(model=model, features=data['features'].tolist(), targets=data['targets'].tolist(),
                    method='direct CDF Extra Trees', training_beer=data['beer'].tolist(),
                    selected_min_samples_leaf=int(parameter), input_scale='competition attachment 1 given scales'),
                artifact, compress=3)
    schema = dict(method='direct CDF Extra Trees', features=data['features'].tolist(),
                  targets=data['targets'].tolist(), expected_feature_count=231, target_count=50,
                  scores=[0,1,2,3,4,5,6], input_units='unchanged competition attachment 1 field representation',
                  missing_columns='filled with NaN and imputed by stored training medians',
                  extra_columns='metadata ignored', output_expected_score='sum(k*p_k), continuous in [0,6]',
                  output_grade='minimum k with cumulative probability >=0.5',
                  output_probability='PAVA-projected CDF differences; no NLL floor applied',
                  model_sha256=sha(artifact))
    (args.output/'schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n')
    record = dict(new_deployment_base_fits=7, inner_grid=[2,5], selection_targets='core30',
                  selected_min_samples_leaf=int(parameter), inner_losses=losses,
                  inner_fold=fold[0].tolist(), fit_log=core.FIT_LOG,
                  all_training_beer=data['beer'].tolist(), training_rows=len(train),
                  seconds=time.perf_counter()-start, inputs_sha256=sha(args.inputs),
                  protocol_sha256=sha(args.protocol_lock), core_source_sha256=sha(Path(core.__file__)),
                  source_sha256=sha(Path(__file__)), model_sha256=sha(artifact),
                  assessment='full-data training artifact; never used as heldout CV evidence')
    (args.output/'deployment_fit_log.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:record[k]for k in ['new_deployment_base_fits','selected_min_samples_leaf','training_rows','seconds','model_sha256']},ensure_ascii=False))


if __name__ == '__main__':
    main()
