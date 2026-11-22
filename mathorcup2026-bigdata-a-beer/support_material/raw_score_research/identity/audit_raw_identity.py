"""Freeze a structural raw-to-contest mapping and audit sensory processing.

No pairing optimization or prediction model is fitted. The candidate structural
rule is raw beer_name k -> one-based row k of published Supplementary File S4.
Its evidence consists of independently published repeated-beer anchors and the
full fixed-rule numerical reconstruction, rather than searching pairings to
maximize agreement. The source deposit does not publish its anonymization
codebook or the S4 reconstruction program.

The zero-SD fallback below is computed from RAW SCORES ONLY; no official outcome
is used to estimate a fallback value. All full-data parameters in this audit
are descriptive provenance checks and must not be reused as train-fold fitted
parameters in a prediction experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

ALIASES = {'A_ester_fruity': 'A_esters_fruity',
           'F_ester_fruity': 'F_esters_fruity'}
STATUS = 'fixed_structural_rule_with_repeat_anchors_and_full_numerical_reproduction'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.writing')
    with temp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def csv(path: Path, frame: pd.DataFrame) -> None:
    atomic(path, frame.to_csv(index=False, float_format='%.17g').encode('utf-8'))


def metrics(prediction: pd.DataFrame, truth: pd.DataFrame) -> dict:
    diff = prediction.to_numpy() - truth.to_numpy()
    return dict(cells=int(diff.size), rmse=float(np.sqrt(np.mean(diff**2))),
                max_absolute_difference=float(np.abs(diff).max()),
                median_absolute_difference=float(np.median(np.abs(diff))),
                cells_within_1e_8=int((np.abs(diff) < 1e-8).sum()),
                cells_within_3e_8=int((np.abs(diff) < 3e-8).sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--workbook', type=Path, required=True)
    parser.add_argument('--official-chemistry', type=Path, required=True)
    parser.add_argument('--official-sensory', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = dict(raw=args.raw.resolve(), workbook=args.workbook.resolve(),
                  official_chemistry=args.official_chemistry.resolve(),
                  official_sensory=args.official_sensory.resolve())
    before = {name: sha(path) for name, path in inputs.items()}
    raw = pd.read_csv(args.raw, encoding='cp1252').rename(columns=ALIASES)
    source = pd.read_excel(args.workbook, sheet_name='Supplementary File S4')
    public_x = pd.read_excel(args.workbook, sheet_name='Supplementary File S1')
    official_x = pd.read_csv(args.official_chemistry)
    official_y = pd.read_csv(args.official_sensory)
    targets = source.columns[3:].tolist()
    assert len(targets) == 50
    assert len(source) == len(public_x) == len(official_x) == len(official_y) == 250
    assert set(targets).issubset(raw)
    assert set(raw.beer_name) == set(range(1, 251))
    assert source.beer.is_unique and official_x.beer.is_unique and official_y.beer.is_unique
    for other in [public_x, official_x, official_y]:
        assert set(other.beer) == set(source.beer)

    source_indexed = source.set_index('beer').sort_index()
    official_y_indexed = official_y.set_index('beer').sort_index()
    official_x_indexed = official_x.set_index('beer').sort_index()
    public_x_indexed = public_x.set_index('beer').sort_index()
    chemistry_columns = public_x.columns[3:].tolist()
    chemistry_error = np.abs(official_x_indexed[chemistry_columns].to_numpy()
                             - public_x_indexed[chemistry_columns].to_numpy())
    sensory_error = np.abs(official_y_indexed[targets].to_numpy()
                          - source_indexed[targets].to_numpy())
    assert np.array_equal(official_x_indexed[chemistry_columns].isna(),
                          public_x_indexed[chemistry_columns].isna())
    assert np.nanmax(chemistry_error) < 1e-12 and sensory_error.max() < 1e-12
    assert np.array_equal(official_x_indexed.beer_id, public_x_indexed.beer_id)

    # FIXED RULE: no outcome distance, nearest neighbor, Hungarian matching, or
    # pair search enters the mapping construction.
    counts = raw.groupby('beer_name').size()
    taster_counts = raw.groupby('beer_name').taster.nunique()
    date_counts = raw.groupby('beer_name').date.nunique()
    coarse = raw.groupby('beer_name').beer_category.first()
    mapping = pd.DataFrame({
        'raw_beer_id': np.arange(1, 251),
        'public_s4_row_one_based': np.arange(1, 251),
        'beer': source.beer.to_numpy(),
    })
    mapping['official_chemistry_beer_id'] = mapping.beer.map(official_x_indexed.beer_id)
    mapping['published_s4_beer_id'] = source.beer_id.to_numpy()
    mapping['fine_style'] = source.tasting_category_fine.to_numpy()
    mapping['raw_tasting_category'] = mapping.raw_beer_id.map(coarse)
    mapping['raw_records'] = mapping.raw_beer_id.map(counts)
    mapping['raw_unique_tasters'] = mapping.raw_beer_id.map(taster_counts)
    mapping['raw_dates'] = mapping.raw_beer_id.map(date_counts)
    mapping['repeated_date_anchor'] = mapping.raw_dates > 1
    mapping['mapping_status'] = STATUS
    mapping['author_codebook_available'] = False
    csv(out/'raw_to_competition_mapping.csv', mapping)

    repeat_table = pd.read_excel(args.workbook, sheet_name='Supplementary Table S2')
    published_repeats = sorted(int(re.fullmatch(r'Beer (\d+)', name).group(1))
                               for name in repeat_table.columns[1:])
    mapped_repeats = sorted(mapping.loc[mapping.repeated_date_anchor, 'beer'].tolist())
    assert mapped_repeats == published_repeats
    raw_repeat_ids = sorted(mapping.loc[mapping.repeated_date_anchor, 'raw_beer_id'].tolist())
    assert raw_repeat_ids != published_repeats

    groups = raw.groupby('taster')[targets]
    means = groups.mean()
    sds = groups.std(ddof=1)
    mu = groups.transform('mean')
    sd = groups.transform('std')
    z_drop = (raw[targets] - mu)/sd
    fallback = -means.mean(axis=0)/sds.mean(axis=0)
    assert np.isfinite(fallback).all()
    z = z_drop.fillna(fallback)
    assert np.isfinite(z.to_numpy()).all()

    def aggregate(values: pd.DataFrame) -> pd.DataFrame:
        frame = values.copy()
        frame['raw_beer_id'] = raw.beer_name
        return frame.groupby('raw_beer_id')[targets].mean().reindex(range(1, 251)).reset_index(drop=True)

    y_source = source[targets].reset_index(drop=True)
    baseline = aggregate(z_drop)
    zero_filled = aggregate(z_drop.fillna(0.))
    reconstructed = aggregate(z)
    main_check = metrics(reconstructed, y_source)
    assert main_check['max_absolute_difference'] < 3e-8
    assert main_check['rmse'] < 1e-9
    reconstruction_output = reconstructed.copy()
    reconstruction_output.insert(0, 'beer', source.beer.to_numpy())
    reconstruction_output.insert(0, 'raw_beer_id', np.arange(1, 251))
    csv(out/'standardized_sensory_reconstruction.csv', reconstruction_output)
    raw_mean = aggregate(raw[targets])
    raw_mean.insert(0, 'beer', source.beer.to_numpy())
    raw_mean.insert(0, 'raw_beer_id', np.arange(1, 251))
    csv(out/'raw_mean_scores.csv', raw_mean)

    # Keep the complete disagreement record, including the two tiny values
    # stored by the published workbook in shortened scientific notation.
    residuals = []
    for i in range(250):
        for j, name in enumerate(targets):
            residuals.append(dict(raw_beer_id=i+1, beer=int(source.beer.iloc[i]), target=name,
                                  reconstructed=float(reconstructed.iloc[i, j]),
                                  published=float(y_source.iloc[i, j]),
                                  residual=float(reconstructed.iloc[i, j]-y_source.iloc[i, j]),
                                  naive_drop_nan_residual=float(baseline.iloc[i, j]-y_source.iloc[i, j])))
    residual_frame = pd.DataFrame(residuals)
    csv(out/'all_reconstruction_residuals.csv', residual_frame)
    csv(out/'reconstruction_conflicts.csv', residual_frame[residual_frame.residual.abs() > 1e-8])
    target_stats = []
    parameters = []
    for name in targets:
        target_stats.append(dict(target=name, raw_zero_fraction=float(raw[name].eq(0).mean()),
                                 zero_sd_tasters=int(sds[name].eq(0).sum()),
                                 affected_beers=int(raw.loc[z_drop[name].isna(), 'beer_name'].nunique()),
                                 raw_only_zero_sd_fallback=float(fallback[name]),
                                 max_absolute_reconstruction_error=float((reconstructed[name]-y_source[name]).abs().max()),
                                 naive_max_absolute_error=float((baseline[name]-y_source[name]).abs().max())))
        for taster in means.index:
            parameters.append(dict(taster=int(taster), target=name, mean=float(means.loc[taster, name]),
                                   sample_sd=float(sds.loc[taster, name]),
                                   n=int(groups.size().loc[taster]),
                                   zero_variance=bool(sds.loc[taster, name] == 0),
                                   audit_only_fallback=float(fallback[name])))
    csv(out/'target_reconstruction_summary.csv', pd.DataFrame(target_stats))
    csv(out/'full_data_audit_taster_parameters.csv', pd.DataFrame(parameters))
    csv(out/'zero_variance_taster_targets.csv', pd.DataFrame(parameters).query('zero_variance'))
    csv(out/'column_aliases.csv', pd.DataFrame([dict(raw=old, canonical=new) for old, new in ALIASES.items()]))

    raw_values = raw[targets].to_numpy()
    stats = dict(records=len(raw), beer_objects=raw.beer_name.nunique(), tasters=raw.taster.nunique(),
                 sessions=raw.date.nunique(), score_min=int(raw_values.min()), score_max=int(raw_values.max()),
                 integer_scores=bool(np.all(raw_values == np.round(raw_values))),
                 missing_scores=int(np.isnan(raw_values).sum()), zero_scores=int((raw_values == 0).sum()),
                 score_cells=int(raw_values.size), zero_fraction=float((raw_values == 0).mean()),
                 records_per_beer_min=int(counts.min()), records_per_beer_max=int(counts.max()),
                 unique_tasters_per_beer_min=int(taster_counts.min()),
                 unique_tasters_per_beer_max=int(taster_counts.max()))
    assert all(sha(inputs[name]) == value for name, value in before.items())
    result = dict(
        status='PASS', mapping_status=STATUS,
        mapping_formula='raw beer_name k -> one-based row k of published S4 -> its beer key -> official chemistry/sensory beer key',
        mapping_searches=0, predictive_model_fits=0, labels_used_to_select_pairs=False,
        source_declared_anonymization_codebook_available=False,
        author_confirmation_obtained_by_this_team=False,
        evidence_scope=dict(structural_rule_fixed_before_pair_search=True,
                            published_repeated_beer_ids=published_repeats,
                            raw_repeated_ids=raw_repeat_ids,
                            fixed_rule_repeat_anchor_set_exact=True,
                            whole_250_numerical_reproduction=main_check,
                            missing_source_evidence='Original anonymization generator and original raw-to-S4 preprocessing code are not in the public deposit.'),
        raw_only_processing=dict(
            within_taster_center='mean per taster and sensory attribute over all raw records for this descriptive audit',
            within_taster_scale='sample standard deviation, ddof=1',
            zero_sd_fallback='c_j = -mean_a(mu_aj)/mean_a(sigma_aj); only raw scores enter these means',
            aggregation='arithmetic mean of all included rating records within raw beer ID',
            predictive_use='Recompute all learned means, SDs and fallback values using TRAIN BEERS ONLY; the full-data audit parameters must not be used as fold-fitted parameters.'),
        comparisons=dict(naive_drop_zero_sd=metrics(baseline, y_source),
                         zero_sd_fill_zero=metrics(zero_filled, y_source),
                         raw_only_pooled_zero_sd=main_check),
        official_public_match=dict(chemistry_max_absolute_difference=float(np.nanmax(chemistry_error)),
                                   sensory_max_absolute_difference=float(sensory_error.max()),
                                   chemistry_stable_barcode_equal=True,
                                   official_sensory_barcode_warning='Rounded scientific-notation identifiers in official 4.csv are not used as the linking key.'),
        raw_data_summary=stats, input_paths={name: str(path) for name, path in inputs.items()},
        input_sha256=before, source_files_unchanged=True,
        script_sha256=sha(Path(__file__)),
        outputs={name: sha(out/name) for name in [
            'raw_to_competition_mapping.csv', 'standardized_sensory_reconstruction.csv',
            'raw_mean_scores.csv', 'all_reconstruction_residuals.csv',
            'reconstruction_conflicts.csv', 'target_reconstruction_summary.csv',
            'full_data_audit_taster_parameters.csv', 'zero_variance_taster_targets.csv',
            'column_aliases.csv']})
    atomic(out/'identity_result.json', (json.dumps(result, ensure_ascii=False, indent=2)+'\n').encode())
    print(json.dumps({k: result[k] for k in ['status', 'mapping_status', 'comparisons', 'raw_data_summary']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
