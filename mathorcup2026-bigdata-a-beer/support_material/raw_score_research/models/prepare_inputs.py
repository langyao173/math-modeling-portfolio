"""Build beer-level targets without estimating the anonymous identity mapping.

This script consumes an externally frozen identity map. It never searches for
an alignment from sensory values. Anonymous-record identity uncertainty is
retained in the output provenance rather than silently converted to certainty.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def empirical_targets(records, beers, targets):
    n, q = len(beers), len(targets)
    probability = np.empty((n, q, 7))
    equal_taster = np.empty_like(probability)
    counts = np.empty(n, int)
    tasters = np.empty(n, int)
    for i, beer in enumerate(beers):
        subset = records[records.official_beer == beer]
        y = subset[targets].to_numpy()
        assert len(y) > 0
        probability[i] = (y[:, :, None] == np.arange(7)).mean(axis=0)
        taster_distributions = []
        for _, panelist in subset.groupby('taster'):
            scores = panelist[targets].to_numpy()
            taster_distributions.append((scores[:, :, None] == np.arange(7)).mean(axis=0))
        equal_taster[i] = np.mean(taster_distributions, axis=0)
        counts[i], tasters[i] = len(y), len(taster_distributions)
    np.testing.assert_allclose(probability.sum(axis=2), 1, atol=1e-15)
    np.testing.assert_allclose(equal_taster.sum(axis=2), 1, atol=1e-15)
    return probability, equal_taster, counts, tasters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chemistry', type=Path, required=True)
    parser.add_argument('--official-targets', type=Path, required=True)
    parser.add_argument('--raw-scores', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--mapping-record', type=Path, required=True)
    parser.add_argument('--research-cache', type=Path, required=True)
    parser.add_argument('--raw-key', default='raw_beer_id')
    parser.add_argument('--official-key', default='beer')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    chemistry = pd.read_csv(args.chemistry).sort_values('beer').reset_index(drop=True)
    official = pd.read_csv(args.official_targets).set_index('beer').loc[chemistry.beer]
    targets = [c for c in official.columns if c not in ('beer_id', 'tasting_category_fine')]
    features = [c for c in chemistry if c not in ('beer', 'beer_id', 'tasting_category_fine')]
    assert len(targets) == 50 and len(features) == 231
    raw = pd.read_csv(args.raw_scores, encoding='cp1252').rename(columns={
        'A_ester_fruity': 'A_esters_fruity', 'F_ester_fruity': 'F_esters_fruity'})
    mapping = pd.read_csv(args.mapping)
    assert mapping[args.raw_key].is_unique and mapping[args.official_key].is_unique
    assert set(mapping[args.raw_key]) == set(raw.beer_name)
    assert set(mapping[args.official_key]) == set(chemistry.beer)
    dictionary = mapping.set_index(args.raw_key)[args.official_key].to_dict()
    raw['official_beer'] = raw.beer_name.map(dictionary)
    values = raw[targets].to_numpy()
    assert np.isfinite(values).all() and np.array_equal(values, np.rint(values))
    assert values.min() >= 0 and values.max() <= 6
    beers = chemistry.beer.to_numpy()
    probability, taster_probability, n, t = empirical_targets(raw, beers, targets)
    mean = probability @ np.arange(7)
    taster_mean = taster_probability @ np.arange(7)
    cache_files = []
    outer_path = args.research_cache/'prediction/fusion_replay_v1/outer_predictions.npz'
    cache_files.append(outer_path)
    with np.load(outer_path, allow_pickle=False) as archive:
        outer = {key: archive[key] for key in archive.files}
    np.testing.assert_array_equal(outer['beer'], beers)
    np.testing.assert_array_equal(outer['targets'], targets)
    np.testing.assert_allclose(outer['Y'], official[targets].to_numpy(), rtol=0, atol=1e-12)
    fold = outer['fold']
    frozen_outer = .5*(outer['raw231_y16_no_residual']+outer['raw231_y50_native'])
    np.testing.assert_allclose(frozen_outer, outer['fixed_half_y16_native50'], rtol=0, atol=1e-12)
    frozen_inner = np.full((5, len(beers), len(targets)), np.nan)
    inner_fold = np.full((5, len(beers)), -1, int)
    for outer_id in range(5):
        train = np.flatnonzero(fold != outer_id)
        test = np.flatnonzero(fold == outer_id)
        directory = args.research_cache/'checkpoint/unpacked/nested_selection'/f'outer_{outer_id}'
        path = directory/'inner_oof.npz'
        cache_files.append(path)
        with np.load(path, allow_pickle=False) as archive:
            inner = {key: archive[key] for key in archive.files}
        np.testing.assert_array_equal(inner['train_indices'], train)
        np.testing.assert_array_equal(inner['beer'], beers[train])
        np.testing.assert_array_equal(inner['targets'], targets)
        np.testing.assert_allclose(inner['Y'], outer['Y'][train], rtol=0, atol=1e-12)
        frozen_inner[outer_id, train] = .5*(inner['raw231_y16_no_residual']+inner['raw231_y50_native'])
        for inner_id in range(3):
            source = directory/f'inner_{inner_id}'/'raw231_y16_complete.npz'
            cache_files.append(source)
            with np.load(source, allow_pickle=False) as archive:
                inner_train, inner_test = archive['train_indices'], archive['test_indices']
                np.testing.assert_array_equal(archive['outer_train_indices'], train)
                np.testing.assert_array_equal(archive['outer_test_indices'], test)
            assert np.intersect1d(inner_train, inner_test).size == 0
            assert np.intersect1d(inner_test, test).size == 0
            np.testing.assert_array_equal(np.sort(np.r_[inner_train, inner_test]), train)
            assert (inner_fold[outer_id, inner_test] == -1).all()
            inner_fold[outer_id, inner_test] = inner_id
        assert (inner_fold[outer_id, train] >= 0).all()
        assert np.isnan(frozen_inner[outer_id, test]).all()
    args.output.mkdir(parents=True, exist_ok=True)
    target_file = args.output/'inputs.npz'
    np.savez_compressed(target_file, beer=beers, targets=np.asarray(targets), features=np.asarray(features),
                        X=chemistry[features].to_numpy(), Yz=official[targets].to_numpy(),
                        raw_probability=probability, raw_mean=mean,
                        taster_probability=taster_probability, taster_mean=taster_mean,
                        record_count=n, taster_count=t,
                        style=np.asarray(chemistry.tasting_category_fine.astype(str).tolist()),
                        fold=fold, inner_fold=inner_fold, frozen_outer_z=frozen_outer,
                        frozen_inner_z=frozen_inner, frozen_groupwise_z=outer['inner_groupwise_y16_native50'],
                        records_beer=raw.official_beer.to_numpy(), records_taster=raw.taster.to_numpy(),
                        records_scores=values)
    pd.DataFrame(mean, index=beers, columns=targets).rename_axis('beer').to_csv(args.output/'raw_panel_means.csv')
    pd.DataFrame({'beer': beers, 'records': n, 'distinct_tasters': t}).to_csv(args.output/'panel_counts.csv', index=False)
    record = dict(input_files={str(path): sha(path) for path in [args.chemistry, args.official_targets,
                      args.raw_scores, args.mapping, args.mapping_record]},
                  frozen_cache_files={str(path): sha(path) for path in cache_files},
                  identity_record=json.loads(args.mapping_record.read_text()),
                  no_identity_search=True, label_dependent_mapping=False,
                  primary_target='empirical distribution with observed records equally weighted within each beer',
                  sensitivity_target='each taster equally weighted within beer; repeats averaged within taster',
                  evaluation_weighting='beers equally weighted, then targets equally weighted',
                  beers=len(beers), targets=len(targets), raw_records=len(raw),
                  original_score_range=[int(values.min()), int(values.max())],
                  output_sha256=sha(target_file), script_sha256=sha(Path(__file__)))
    (args.output/'target_provenance.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['beers','targets','raw_records','original_score_range','no_identity_search']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
