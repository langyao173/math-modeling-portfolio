"""Independent whitened-SVD check and saved-prediction audit of the auxiliary study."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import pandas as pd
from scipy.special import ndtri


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    with np.load(args.input, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(args.results/'outer_predictions.npz', allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    lock = json.loads((args.results/'PROTOCOL_LOCK.json').read_text())
    assert sha(args.input) == lock['input_sha256']
    np.testing.assert_array_equal(data['Yz'], result['Y'])
    np.testing.assert_array_equal(data['raw_mean'], result['raw_mean'])
    np.testing.assert_array_equal(data['beer'], result['beer'])
    np.testing.assert_array_equal(data['fold'], result['fold'])
    trace = json.loads((args.results/'training_identity_trace.json').read_text())
    for row in trace:
        train, val, test = [set(row[key]) for key in ('training_beers', 'validation_beers', 'outer_test_beers')]
        assert not train & val and not train & test and not val & test
        assert len(train | val | test) == 250
    # The following construction does not use the implementation's U^T Y
    # eigenproblem: it whitens the feature-side regularized normal matrix and
    # applies Eckart-Young directly to the whitened response coefficient.
    parameters = pd.read_csv(args.results/'selected_parameters.csv')
    checks = []
    for fold in range(5):
        tr, te = np.flatnonzero(data['fold'] != fold), np.flatnonzero(data['fold'] == fold)
        x, z, raw = data['X'][tr], data['Yz'][tr], data['raw_mean'][tr]
        median = np.nanmedian(x, axis=0)
        filled = np.where(np.isfinite(x), x, median)
        test = np.where(np.isfinite(data['X'][te]), data['X'][te], median)
        if lock.get('input_transform', 'standard') == 'normal_rank':
            ordered = np.sort(filled, axis=0)
            n = len(filled)
            for j in range(filled.shape[1]):
                # Both transform arrays use only this outer training column.
                for matrix in (filled, test):
                    lo = np.searchsorted(ordered[:, j], matrix[:, j], side='left')
                    hi = np.searchsorted(ordered[:, j], matrix[:, j], side='right')
                    matrix[:, j] = ndtri(np.clip((lo+hi)/(2*n), .5/n, 1-.5/n))
        xm, xs = filled.mean(0), filled.std(0)
        xs[xs <= 1e-12] = 1.
        xx = (filled-xm)/xs
        zm, zs = z.mean(0), z.std(0)
        zs[zs <= 1e-12] = 1.
        zz = (z-zm)/zs
        rm, rs = raw.mean(0), raw.std(0)
        rs[rs <= 1e-12] = 1.
        rr = (raw-rm)/rs
        test = (test-xm)/xs
        for row in parameters[parameters.outer_fold == fold].itertuples():
            y = zz if row.tau == 0 else np.column_stack((zz, np.sqrt(row.tau)*rr))
            c = xx.T@xx + row.alpha*np.eye(xx.shape[1])
            eigenvalue, eigenvector = np.linalg.eigh(c)
            c_inverse_root = (eigenvector/np.sqrt(eigenvalue))@eigenvector.T
            matrix = c_inverse_root@xx.T@y
            u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
            rank = int(row.rank)
            coefficient = c_inverse_root@((u[:, :rank]*singular[:rank])@vt[:rank])
            pred = (test@coefficient)[:, :50]*zs+zm
            key = 'tau'+str(float(row.tau)).replace('.', 'p')+'_original'
            difference = float(np.max(np.abs(pred-result[key][te])))
            assert difference < 1e-9
            checks.append(dict(fold=fold, tau=row.tau, independent_whitened_svd_max_error=difference))
    variance = result['Y'].var(0)
    summary = pd.read_csv(args.results/'pooled_metrics.csv')
    errors = []
    for row in summary.itertuples():
        key = 'tau'+str(float(row.tau)).replace('.', 'p')+'_'+row.scenario
        scores = 1.-np.mean((result[key]-result['Y'])**2, axis=0)/variance
        errors.extend([abs(scores[:30].mean()-row.core30), abs(scores.mean()-row.all50),
                       abs(scores[30:].mean()-row.other20)])
    assert max(errors) < 1e-12
    base, aux = result['tau0p0_original'], result['tau1p0_original']
    difference = ((base-result['Y'])**2-(aux-result['Y'])**2)/variance
    row_gain = difference[:, :30].mean(1)
    index = int(np.argmax(np.abs(row_gain)))
    record = dict(status='PASS', archive_input_hash_unchanged=True, all_folds_and_targets_identical=True,
                  no_outer_test_beers_in_inner_training_or_validation=True,
                  independent_whitened_svd_checks=checks,
                  independent_whitened_svd_max_error=max(row['independent_whitened_svd_max_error'] for row in checks),
                  pooled_metric_recalculation_max_error=max(errors),
                  primary_all250_core_r2_delta=float(row_gain.mean()),
                  influential_beer_diagnostic=dict(beer=int(result['beer'][index]), paired_loss_gain=float(row_gain[index]),
                                                  share_of_signed_gain_sum=float(row_gain[index]/row_gain.sum()),
                                                  excluded_beer_delta_DIAGNOSTIC_ONLY=float(np.delete(row_gain, index).mean())),
                  interpretation='Every primary score retains all250 beers; the exclusion is an influence diagnostic, not a new model score.',
                  no_new_method_or_hyperparameter_selected=True,
                  input_sha256=sha(args.input), results_archive_sha256=sha(args.results/'outer_predictions.npz'),
                  input_transform=lock.get('input_transform', 'standard'),
                  implementation_sha256=sha(Path(__file__)))
    (args.results/'independent_audit.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    main(parser.parse_args())
