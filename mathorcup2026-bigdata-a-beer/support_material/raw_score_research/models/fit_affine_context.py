"""Fit a full-training calibration context for a genuine fixed-half predictor.

This artifact is a deployment component. Its 250-beer training inputs are the
original outer-OOF fixed-half predictions. It is never used to score those
same 250 beers as heldout calibrated predictions; the evaluation uses each
outer fold's own 200-beer inner-OOF calibration context instead.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from run_raw_models import affine_prediction


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    with np.load(args.inputs, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    _, slope, intercept = affine_prediction(data['frozen_outer_z'],data['raw_mean'],
                                            np.empty((0,len(data['targets']))))
    assert np.isfinite(slope).all() and np.isfinite(intercept).all() and (slope >= 0).all()
    args.output.mkdir(parents=True)
    path = args.output/'fixed_half_raw_affine.npz'
    np.savez_compressed(path, targets=data['targets'], slope=slope, intercept=intercept,
                        training_beer=data['beer'])
    record = dict(method='nonnegative-affine raw-panel-mean calibration',training_beers=250,
                  new_affine_target_fits=50,new_neural_inferences=0,additional_base_model_fits=0,
                  expected_input_method='fixed_half_y16_native50',
                  expected_input_role='genuine chemically predicted standardized scores',
                  formula='clip(intercept_j + slope_j * predicted_z_j, 0, 6)',
                  training_feature_role='frozen 250-row outer-OOF standardized predictions',
                  performance_role='deployment training component, not additional heldout evaluation',
                  assessment_contexts='results_v2/calibration_fold0.npz through calibration_fold4.npz',
                  inputs_sha256=sha(args.inputs),source_sha256=sha(Path(__file__)),
                  calibration_sha256=sha(path))
    (args.output/'calibration_context.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False))


if __name__ == '__main__':
    main()
