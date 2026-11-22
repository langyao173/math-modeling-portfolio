"""Independent NumPy-loop verification of the frozen mapping/reconstruction.

Does not import or execute audit_raw_identity.py and fits no predictor. Uses
only raw-score means/SDs to calculate the reconstruction; official scores are
read afterwards only for numerical comparison.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--workbook', type=Path, required=True)
    parser.add_argument('--identity-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.identity_dir
    result = json.loads((output/'identity_result.json').read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(args.raw) == result['input_sha256']['raw']
    assert sha(args.workbook) == result['input_sha256']['workbook']
    for name, expected in result['outputs'].items():
        assert sha(output/name) == expected, name
    raw = pd.read_csv(args.raw, encoding='cp1252').rename(columns={
        'A_ester_fruity': 'A_esters_fruity', 'F_ester_fruity': 'F_esters_fruity'})
    mapping = pd.read_csv(output/'raw_to_competition_mapping.csv')
    s4 = pd.read_excel(args.workbook, sheet_name='Supplementary File S4')
    cols = s4.columns[3:].tolist()
    # The mapping is fixed structurally, never reconstructed by score matching.
    assert mapping.raw_beer_id.tolist() == list(range(1, 251))
    assert mapping.beer.tolist() == s4.beer.tolist()
    ids, tasters = raw.beer_name.to_numpy(), raw.taster.to_numpy()
    unique_tasters = sorted(np.unique(tasters))
    reconstructed = np.empty((250, len(cols)))
    all_zero = True
    max_csv_error = 0.
    for j, col in enumerate(cols):
        values = raw[col].to_numpy(float)
        means = np.array([values[tasters == a].mean() for a in unique_tasters])
        sds = np.array([values[tasters == a].std(ddof=1) for a in unique_tasters])
        pooled_mean, pooled_sd = means.mean(), sds.mean()
        scaled = np.empty(len(raw))
        for i, a in enumerate(unique_tasters):
            ix = tasters == a
            if sds[i] > 0:
                scaled[ix] = (values[ix]-means[i])/sds[i]
            else:
                all_zero = all_zero and bool(np.all(values[ix] == 0))
                scaled[ix] = (values[ix]-pooled_mean)/pooled_sd
        reconstructed[:, j] = [scaled[ids == k].mean() for k in range(1, 251)]
    assert all_zero
    saved = pd.read_csv(output/'standardized_sensory_reconstruction.csv', float_precision='round_trip')
    max_csv_error = float(np.max(np.abs(saved[cols].to_numpy()-reconstructed)))
    official_error = np.abs(reconstructed-s4[cols].to_numpy())
    assert max_csv_error < 1e-13
    assert official_error.max() < 3e-8
    repeated_raw = raw.groupby('beer_name').date.nunique().loc[lambda x: x > 1].index
    repeated_mapped = sorted(mapping.set_index('raw_beer_id').loc[repeated_raw, 'beer'].tolist())
    s2 = pd.read_excel(args.workbook, sheet_name='Supplementary Table S2')
    published = sorted(int(re.fullmatch(r'Beer (\d+)', col).group(1)) for col in s2.columns[1:])
    assert repeated_mapped == published
    report = dict(status='PASS', predictor_fits=0, mapping_searches=0,
                  no_import_or_execution_of_main_auditor=True,
                  raw_only_loop_reconstruction=True,
                  max_difference_from_saved_reconstruction=max_csv_error,
                  max_difference_from_published_s4=float(official_error.max()),
                  rmse_from_published_s4=float(np.sqrt(np.mean(official_error**2))),
                  all_12500_cells_within_3e_8=True,
                  zero_sd_full_data_cells_all_have_raw_score_zero=all_zero,
                  repeated_beer_identity_anchor_set_exact=True,
                  all_manifested_output_hashes_equal=True,
                  source_codebook_published=False,
                  scope='Independent arithmetic and fixed-rule consistency audit; not an independent clinical/physical or out-of-sample prediction validation.')
    (output/'independent_identity_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
