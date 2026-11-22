"""Independently audit the saved CSV deployment challenge without model calls.

Only reads files and prints JSON. Recomputes row identities, chemistry mapping,
relative grades, target-subspace membership, differences, and event counts.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'deployment_transfer_challenge'
SPARSE = HERE.parent / 'sparse_transfer'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(name):
    return pd.read_csv(OUT / name, float_precision='round_trip')


def main():
    done = read_json(OUT / 'COMPLETE.json')
    assert done['complete'] and done['all50'] and done['source_count'] == 11
    for name, expected in done['result_sha256'].items():
        assert sha(OUT / name) == expected, name
    for name, expected in done['source_summary_sha256'].items():
        assert sha(SPARSE / 'final_summary' / name) == expected, name
        assert sha(OUT / ('frozen_' + name)) == expected, name
    assert sha(SPARSE / 'post_aggregation_integrity/verification.json') == done['post_aggregation_verification_sha256']
    verification = read_json(SPARSE / 'post_aggregation_integrity/verification.json')
    assert verification['complete'] and verification['passed']
    assert verification['selection_cases_sha256'] == done['candidate_pool_sha256']
    assert sha(HERE / 'predict_tabicl_csv.py') == done['csv_interface_sha256']
    assert sha(HERE / 'final_artifact/manifest.json') == done['artifact_manifest_sha256']
    assert sha(HERE / 'run_deployment_transfer_challenge.py') == done['challenge_source_sha256']
    manifest = read_json(HERE / 'final_artifact/manifest.json')
    context_path = HERE / 'final_artifact' / manifest['context_file']
    assert sha(context_path) == manifest['context_sha256']
    context = np.load(context_path, allow_pickle=False)
    features, targets = context['features'].tolist(), context['targets'].tolist()
    assert len(features) == 231 and len(targets) == 50

    chemistry = read_csv('source_and_candidate_chemistry.csv').set_index('beer')
    predictions = read_csv('all50_sensory_predictions.csv').set_index('beer')
    grades = read_csv('all50_sensory_predictions.relative_grades_0_6.csv').set_index('beer')
    identities = read_csv('input_identity.csv').set_index('input_id')
    assert sha(OUT / 'source_and_candidate_chemistry.csv') == done['input_sha256']
    assert list(chemistry.columns) == features
    assert list(predictions.columns) == list(grades.columns) == targets
    assert chemistry.index.equals(predictions.index) and chemistry.index.equals(grades.index)
    assert chemistry.index.equals(identities.index) and chemistry.index.is_unique
    assert len(chemistry) == done['input_rows']
    assert np.isfinite(predictions.to_numpy()).all()
    assert not np.isinf(chemistry.to_numpy()).any()
    assert identities.kind.eq('source').sum() == 11
    assert identities.kind.eq('candidate').sum() == done['unique_candidate_profiles']
    source_ids = []
    for identifier in identities.index[identities.kind.eq('source')]:
        beer = int(identities.loc[identifier, 'source_beer'])
        position = np.flatnonzero(context['beer'] == beer).item()
        assert identifier == f'source_{beer}'
        assert np.array_equal(chemistry.loc[identifier].to_numpy(), context['X'][position], equal_nan=True)
        source_ids.append(beer)
    assert len(set(source_ids)) == 11

    # Counting cutpoints at or below a prediction independently reproduces
    # the documented right-side empirical grade, including tied cutpoints.
    expected_grades = (predictions.to_numpy()[:, None, :] >= context['relative_grade_cuts'][None, :, :]).sum(axis=1)
    assert np.array_equal(expected_grades, grades.to_numpy())
    assert expected_grades.min() >= 0 and expected_grades.max() <= 6
    subspace_error = None
    if manifest['target_encoding']['kind'] == 'standardize_pca':
        components = context['pca_components']
        assert np.allclose(components @ components.T, np.eye(len(components)), atol=2e-12, rtol=0.)
        centered = (predictions.to_numpy() - context['target_mean']) / context['target_scale'] - context['pca_mean']
        remainder = centered - (centered @ components.T) @ components
        subspace_error = float(np.max(np.abs(remainder)))
        assert subspace_error < 2e-12

    cases = read_csv('cross_model_case_checks.csv')
    original_mapping = read_csv('case_input_mapping.csv')
    selected = read_csv('frozen_selection_cases.csv')
    profiles = read_csv('frozen_selected_profiles.csv').set_index(['evaluation_policy', 'candidate_uid'])
    assert len(cases) == len(original_mapping) == len(selected) == 88
    assert not cases.duplicated(['policy', 'beer', 'rho']).any()
    assert not selected.duplicated(['policy', 'beer', 'rho']).any()
    assert cases.groupby(['policy', 'rho']).size().eq(11).all()
    selection_index = selected.set_index(['policy', 'beer', 'rho'])
    for row in cases.itertuples():
        original = selection_index.loc[(row.policy, row.beer, row.rho)]
        assert row.candidate_uid == original.candidate_uid
        assert row.raw_fields_changed == original.integer_upper_bound
        assert bool(row.fstar_full_accepted) and bool(original.accepted)
        assert row.source_input_id == f'source_{int(row.beer)}'
        profile = profiles.loc[(row.policy, row.candidate_uid), features].to_numpy(float)
        assert np.array_equal(profile, chemistry.loc[row.candidate_input_id].to_numpy(), equal_nan=True)
        assert abs(row.fstar_stale_delta - original.ensemble_stale_delta) < 1e-12
    new_differences = np.asarray([predictions.loc[c, done['target']] - predictions.loc[s, done['target']]
                                 for c, s in zip(cases.candidate_input_id, cases.source_input_id)])
    difference_error = float(np.max(np.abs(new_differences - cases.fplus_stale_delta.to_numpy())))
    assert difference_error < 1e-12
    assert np.array_equal(new_differences <= 0., cases.fplus_nonincrease_exact.to_numpy())
    assert np.array_equal(new_differences <= 1e-8, cases.fplus_nonincrease_at_1e_8.to_numpy())
    unique = cases.drop_duplicates(['beer', 'candidate_input_id'])
    assert len(unique) == done['unique_source_candidate_pairs']
    assert int(unique.fplus_nonincrease_exact.sum()) == done['unique_pair_nonincrease_exact']
    assert int((~unique.fplus_nonincrease_exact).sum()) == done['unique_pair_increase']
    for row in done['policy_rho_summary']:
        group = cases[cases.policy.eq(row['policy']) & cases.rho.eq(row['rho'])]
        assert len(group) == row['cases'] == 11
        assert int(group.fplus_nonincrease_exact.sum()) == row['fplus_nonincrease_exact']
        assert int((~group.fplus_nonincrease_exact).sum()) == row['fplus_increase']
        for suffix, value in [('mean', group.fplus_stale_delta.mean()),
                              ('min', group.fplus_stale_delta.min()), ('max', group.fplus_stale_delta.max())]:
            assert abs(value - row['fplus_stale_delta_' + suffix]) < 1e-12
    batching = read_json(OUT / 'query_batching_audit.json')
    assert batching['input_sha256'] == done['input_sha256']
    assert batching['all_rows_exact_equal'] and batching['rows'] == len(chemistry)
    assert sum(row['shape'][0] for row in batching['records']) == 8
    print(json.dumps(dict(complete=True, passed=True, new_inference=False, writes_files=False,
        chemistry_rows=len(chemistry), source_count=11, unique_candidate_profiles=done['unique_candidate_profiles'],
        outputs_per_row=50, relative_grade_cells_verified=int(expected_grades.size),
        all_source_and_candidate_chemistry_exact=True, all_frozen_hashes_unchanged=True,
        selected_case_mappings_verified=len(cases), stale_difference_maximum_error=difference_error,
        target_subspace_maximum_absolute_residual=subspace_error,
        unique_pair_nonincrease_exact=int(unique.fplus_nonincrease_exact.sum()),
        unique_pair_increase=int((~unique.fplus_nonincrease_exact).sum()),
        actual_query_preprocessing_batch_invariance=True,
        candidate_pool_sha256=done['candidate_pool_sha256'],
        complete_sha256=sha(OUT / 'COMPLETE.json'), implementation_sha256=sha(__file__),
        scope='Independent array, row-mapping, grade and immutable-evidence checks; neither a new model run nor biological validation.'),
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
