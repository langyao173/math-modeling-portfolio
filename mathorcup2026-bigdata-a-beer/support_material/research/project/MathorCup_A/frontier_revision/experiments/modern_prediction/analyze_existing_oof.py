# Reconstructed after the 2026-10-24 workspace file loss. This is a new
# implementation of the recorded analysis protocol, not the lost source bytes.
"""Compute style-group diagnostics from existing, identity-checked OOF arrays.

No fitting, prediction, data augmentation, or bootstrap is performed. Each
target is scaled by its variance over the same original 250 observed beers.
The global mean normalized squared error therefore equals one minus macro R2.
Group NMSE uses that same fixed scale; small groups are marked, not discarded.

Optional --extra-oof accepts a completed nested-selection archive with the
same Y, beer, fold, and target-name metadata. It remains a separate procedure
from the fixed raw231/PCA16 predictor used in prior downstream experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
METADATA = {'Y', 'beer', 'fold', 'targets'}
REFERENCE = 'old_nested_ensemble'
FIXED_Y16 = 'raw231_y16_no_residual'
NATIVE_Y50 = 'raw231_y50_native'
TARGET_SETS = {'core30': np.arange(30), 'all50': np.arange(50)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def locate_project():
    for ancestor in (HERE, *HERE.parents):
        for candidate in (ancestor, ancestor / 'project/MathorCup_A'):
            if (candidate / 'data/A/1.csv').is_file() and (candidate / 'A_work/q1/fold_assignments.csv').is_file():
                return candidate
    raise FileNotFoundError('Supply --project-dir containing data/A and A_work/q1.')


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def exact(actual, expected, description):
    if np.asarray(actual).dtype.kind in 'OUS' or np.asarray(expected).dtype.kind in 'OUS':
        same = np.array_equal(actual, expected)
    else:
        same = np.array_equal(actual, expected, equal_nan=True)
    if not same:
        raise ValueError(f'Identity or order mismatch: {description}')


def load_reference(project):
    chemistry = pd.read_csv(project / 'data/A/1.csv').sort_values('beer').reset_index(drop=True)
    sensory = pd.read_csv(project / 'data/A/4.csv')
    fold_table = pd.read_csv(project / 'A_work/q1/fold_assignments.csv')
    for frame, label in ((chemistry, 'chemistry'), (sensory, 'sensory'), (fold_table, 'fold table')):
        if len(frame) != 250 or not frame.beer.is_unique or frame.beer.isna().any():
            raise ValueError(f'Invalid original beer identifiers in {label}.')
        if set(frame.beer) != set(chemistry.beer):
            raise ValueError(f'Beer set mismatch in {label}.')
    sensory = sensory.set_index('beer').loc[chemistry.beer].reset_index()
    fold = fold_table.set_index('beer').loc[chemistry.beer, 'fold'].to_numpy()
    if set(fold) != set(range(5)) or any(np.count_nonzero(fold == k) != 50 for k in range(5)):
        raise ValueError('The registered five folds must contain 50 original beers each.')
    Y = sensory.iloc[:, 3:].to_numpy(float)
    if Y.shape != (250, 50) or not np.isfinite(Y).all():
        raise ValueError('Attachment 4 must provide 250 x 50 finite observed values.')
    if chemistry['tasting_category_fine'].isna().any():
        raise ValueError('Missing original fine-style labels.')
    style = chemistry['tasting_category_fine'].astype(str).to_numpy()
    if len(np.unique(style)) != 22:
        raise ValueError('The registered original dataset has 22 fine styles.')
    return {'Y': Y, 'beer': chemistry.beer.to_numpy(), 'fold': fold,
            'targets': sensory.columns[3:].to_numpy(str), 'style': style}


def validate_archive(archive, reference, label):
    for key in METADATA:
        if key not in archive:
            raise ValueError(f'{label} lacks required alignment metadata {key!r}.')
        exact(archive[key], reference[key], f'{label}/{key}')


def validate_prediction(prediction, Y, label):
    if prediction.shape != Y.shape or not np.isfinite(prediction).all():
        raise ValueError(f'Incomplete, differently shaped, or non-finite prediction: {label}')


def normalized_losses(Y, prediction, target_variance, columns):
    return (np.square(Y[:, columns] - prediction[:, columns]) / target_variance[columns]).mean(axis=1)


def macro_r2(Y, prediction, columns):
    a, b = Y[:, columns], prediction[:, columns]
    denominator = np.square(a - a.mean(axis=0)).sum(axis=0)
    if np.any(denominator <= 0):
        return float('nan')
    return float((1 - np.square(a - b).sum(axis=0) / denominator).mean())


def write_frame(output, filename, records):
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    frame.to_csv(output / filename, index=False)
    return frame


def summarize_models(reference, predictions, output, minimum_group_n):
    Y, style = reference['Y'], reference['style']
    variance = Y.var(axis=0, ddof=0)
    if np.any(variance <= 0) or not np.isfinite(variance).all():
        raise ValueError('Target-normalization variance must be positive and finite.')
    unique_styles = sorted(np.unique(style).tolist())
    summaries, groups, per_beer, folds, omissions, dimensions = [], [], [], [], [], []
    for name, prediction in predictions.items():
        validate_prediction(prediction, Y, name)
        dimension_r2 = 1 - np.square(Y - prediction).mean(axis=0) / variance
        for j, target in enumerate(reference['targets']):
            dimensions.append({'model': name, 'target': target, 'core30': j < 30,
                               'r2': float(dimension_r2[j]),
                               'rmse': float(np.sqrt(np.square(Y[:, j] - prediction[:, j]).mean()))})
        for target_set, columns in TARGET_SETS.items():
            loss = normalized_losses(Y, prediction, variance, columns)
            global_r2 = macro_r2(Y, prediction, columns)
            np.testing.assert_allclose(loss.mean(), 1 - global_r2, rtol=1e-12, atol=1e-12)
            tail_n = int(math.ceil(0.10 * len(loss)))
            tail_ids = np.argsort(-loss, kind='stable')[:tail_n]
            in_tail = np.zeros(len(loss), dtype=bool)
            in_tail[tail_ids] = True
            model_groups = []
            for group in unique_styles:
                mask = style == group
                values = loss[mask]
                row = {'model': name, 'targets': target_set, 'style': group,
                       'n': int(mask.sum()), 'small_n': bool(mask.sum() < minimum_group_n),
                       'nmse': float(values.mean()), 'median_individual_nmse': float(np.median(values)),
                       'p90_individual_nmse': float(np.quantile(values, .90)),
                       'max_individual_nmse': float(values.max())}
                groups.append(row)
                model_groups.append(row)
                remain = ~mask
                omissions.append({'model': name, 'targets': target_set, 'omitted_style': group,
                                  'remaining_n': int(remain.sum()),
                                  'nmse_fixed_all250_reference': float(loss[remain].mean()),
                                  'macro_r2_recomputed_on_remaining_rows': macro_r2(Y[remain], prediction[remain], columns),
                                  'model_refitted': False,
                                  'interpretation': 'Evaluation-row omission only; not leave-style-out training.'})
            worst = max(model_groups, key=lambda row: row['nmse'])
            supported = [row for row in model_groups if row['n'] >= minimum_group_n]
            worst_supported = max(supported, key=lambda row: row['nmse'])
            summaries.append({'model': name, 'targets': target_set, 'n': len(Y),
                              'macro_r2': global_r2, 'nmse': float(loss.mean()),
                              'style_equal_nmse': float(np.mean([row['nmse'] for row in model_groups])),
                              'median_individual_nmse': float(np.median(loss)),
                              'p90_individual_nmse': float(np.quantile(loss, .90)),
                              'p95_individual_nmse': float(np.quantile(loss, .95)),
                              'own_worst10pct_n': tail_n,
                              'own_worst10pct_mean_nmse': float(loss[tail_ids].mean()),
                              'worst_style': worst['style'], 'worst_style_n': worst['n'],
                              'worst_style_nmse': worst['nmse'],
                              'worst_style_n_ge_threshold': worst_supported['style'],
                              'worst_supported_style_nmse': worst_supported['nmse']})
            for i, value in enumerate(loss):
                per_beer.append({'model': name, 'targets': target_set,
                                 'beer': reference['beer'][i], 'fold': int(reference['fold'][i]),
                                 'style': style[i], 'nmse': float(value),
                                 'in_model_own_worst10pct': bool(in_tail[i])})
            for fold in range(5):
                mask = reference['fold'] == fold
                folds.append({'model': name, 'targets': target_set, 'fold': fold,
                              'n': int(mask.sum()), 'nmse_fixed_all250_reference': float(loss[mask].mean()),
                              'fold_macro_r2': macro_r2(Y[mask], prediction[mask], columns)})
    write_frame(output, 'summary_metrics.csv', summaries)
    group_frame = write_frame(output, 'group_metrics.csv', groups)
    write_frame(output, 'per_beer_errors.csv', per_beer)
    write_frame(output, 'fold_metrics.csv', folds)
    write_frame(output, 'omit_style_evaluation_sensitivity.csv', omissions)
    write_frame(output, 'dimension_metrics.csv', dimensions)
    write_frame(output, 'target_evaluation_scales.csv',
                [{'target': t, 'core30': j < 30, 'mean_all250': float(Y[:, j].mean()),
                  'variance_all250_ddof0': float(variance[j])} for j, t in enumerate(reference['targets'])])
    coverage = []
    for group in unique_styles:
        group_mask = style == group
        for fold in range(5):
            n_test = int(np.count_nonzero(group_mask & (reference['fold'] == fold)))
            n_train = int(np.count_nonzero(group_mask & (reference['fold'] != fold)))
            coverage.append({'style': group, 'fold': fold, 'style_total_n': int(group_mask.sum()),
                             'outer_test_n': n_test, 'same_style_outer_training_n': n_train,
                             'evaluated_without_same_style_training': bool(n_test > 0 and n_train == 0)})
    coverage_frame = write_frame(output, 'style_fold_coverage.csv', coverage)
    return variance, group_frame, coverage_frame


def make_a2(group_frame, output, nested_name):
    core = group_frame.loc[group_frame.targets == 'core30']
    count = core[['style', 'n', 'small_n']].drop_duplicates('style').sort_values('style')
    table = count.copy()
    for model, column in ((REFERENCE, 'six_model_nmse'), (FIXED_Y16, 'fixed_y16_nmse'),
                          (nested_name, 'nested_selection_nmse')):
        values = core.loc[core.model == model].set_index('style')['nmse']
        table[column] = table['style'].map(values)
    table.to_csv(output / 'table_A2_core30_nmse.csv', index=False)
    lines = ['# 表 A2：核心 30 维折外误差的风格分组诊断', '',
             '| 风格 | n | 六模型 NMSE | 固定 Y16 NMSE | 内层选模流程 NMSE |',
             '|---|---:|---:|---:|---:|']
    for row in table.itertuples(index=False):
        values = [row.six_model_nmse, row.fixed_y16_nmse, row.nested_selection_nmse]
        cells = ['—' if pd.isna(value) else f'{value:.6f}' for value in values]
        name = str(row.style).replace('|', '\\|') + ('*' if row.small_n else '')
        lines.append(f"| {name} | {row.n} | {' | '.join(cells)} |")
    lines += ['', '* 表示 n<10，仅作小样本描述。NMSE 使用全体 250 款原始啤酒的逐目标方差统一归一化；',
              '各行仍来自原随机五折划分，不能解释为未见风格测试。未提供完整嵌套预测时该列留空，',
              '不以原先固定配置的分数代替。', '']
    (output / 'table_A2_core30_nmse.md').write_text('\n'.join(lines), encoding='utf-8')


def missingness_diagnostics(path, reference, variance, output, minimum_group_n):
    archive = load_npz(path)
    validate_archive(archive, reference, 'missingness OOF')
    rows, counts = [], []
    scenarios = ('original', 'derived_only_missing', 'MCAR_10_strict', 'MCAR_30_strict', 'MCAR_50_strict')
    for scenario in scenarios:
        target = archive[f'tabicl_raw231_y16__{scenario}']
        validate_prediction(target, reference['Y'], scenario)
        for baseline in ('old_knn', 'old_median'):
            comparator = archive[f'{baseline}__{scenario}']
            validate_prediction(comparator, reference['Y'], baseline + scenario)
            for target_set, columns in TARGET_SETS.items():
                modern_loss = normalized_losses(reference['Y'], target, variance, columns)
                baseline_loss = normalized_losses(reference['Y'], comparator, variance, columns)
                group_rows = []
                for style in sorted(np.unique(reference['style'])):
                    mask = reference['style'] == style
                    a, b = float(modern_loss[mask].mean()), float(baseline_loss[mask].mean())
                    row = {'scenario': scenario, 'reference': baseline, 'targets': target_set,
                           'style': style, 'n': int(mask.sum()), 'small_n': bool(mask.sum() < minimum_group_n),
                           'fixed_y16_nmse': a, 'reference_nmse': b,
                           'delta_nmse_y16_minus_reference': a - b, 'y16_lower_error': a < b}
                    rows.append(row)
                    group_rows.append(row)
                supported = [row for row in group_rows if not row['small_n']]
                counts.append({'scenario': scenario, 'reference': baseline, 'targets': target_set,
                               'groups': len(group_rows), 'y16_lower_error_groups': sum(row['y16_lower_error'] for row in group_rows),
                               'groups_n_ge_threshold': len(supported),
                               'y16_lower_error_supported_groups': sum(row['y16_lower_error'] for row in supported)})
    write_frame(output, 'missingness_by_style.csv', rows)
    write_frame(output, 'missingness_group_counts.csv', counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path)
    parser.add_argument('--experiment-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--extra-oof', type=Path)
    parser.add_argument('--extra-key', default='prediction')
    parser.add_argument('--extra-name', default='tabicl_nested_selection')
    parser.add_argument('--minimum-group-n', type=int, default=10)
    parser.add_argument('--bootstrap-repetitions', type=int, default=0,
                        help='Compatibility option; only 0 is permitted. No repeated bootstrap is performed.')
    args = parser.parse_args()
    if args.bootstrap_repetitions != 0:
        parser.error('This point-diagnostic program requires --bootstrap-repetitions 0.')
    project = (args.project_dir or locate_project()).resolve()
    default_experiments = (HERE.parent if (HERE / 'oof_arrays.npz').is_file() else
                           project / 'frontier_revision/experiments')
    experiments = (args.experiment_dir or default_experiments).resolve()
    output = (args.output or HERE / 'generalization_groups').resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new or empty output directory so earlier evidence is preserved.')
    reference = load_reference(project)
    modern_path = experiments / 'modern_prediction/oof_arrays.npz'
    modern = load_npz(modern_path)
    validate_archive(modern, reference, 'modern OOF')
    predictions = {key: value for key, value in modern.items()
                   if key not in METADATA and value.shape == reference['Y'].shape}
    for key in (REFERENCE, FIXED_Y16, NATIVE_Y50):
        if key not in predictions:
            raise KeyError(f'Missing registered fixed-model prediction {key}.')
    legacy_path = project / 'A_work/q1/oof_arrays.npz'
    legacy = load_npz(legacy_path)
    exact(legacy['Y'], reference['Y'], 'legacy Y in original canonical beer order')
    exact(legacy['nested_ensemble'], predictions[REFERENCE], 'legacy and modern six-model OOF copies')
    input_paths = {'chemistry': project / 'data/A/1.csv', 'sensory': project / 'data/A/4.csv',
                   'folds': project / 'A_work/q1/fold_assignments.csv',
                   'modern_oof': modern_path, 'legacy_oof': legacy_path}
    tabdpt_path = experiments / 'modern_prediction/tabdpt/oof_arrays.npz'
    if tabdpt_path.is_file():
        tabdpt = load_npz(tabdpt_path)
        validate_archive(tabdpt, reference, 'TabDPT OOF')
        tabdpt_spec_path = tabdpt_path.parent / 'experiment_spec.json'
        tabdpt_name = json.loads(tabdpt_spec_path.read_text())['model']
        if tabdpt_name in predictions:
            raise ValueError('TabDPT model name collides with an existing model.')
        predictions[tabdpt_name] = tabdpt['prediction']
        input_paths['tabdpt_oof'] = tabdpt_path
        input_paths['tabdpt_spec'] = tabdpt_spec_path
    if args.extra_oof is not None:
        if args.extra_name in predictions:
            raise ValueError('Additional model name collides with an existing model.')
        extra = load_npz(args.extra_oof)
        validate_archive(extra, reference, 'additional/nested OOF')
        predictions[args.extra_name] = extra[args.extra_key]
        validate_prediction(predictions[args.extra_name], reference['Y'], args.extra_name)
        input_paths['additional_oof'] = args.extra_oof.resolve()
    output.mkdir(parents=True, exist_ok=True)
    variance, groups, coverage = summarize_models(reference, predictions, output, args.minimum_group_n)
    make_a2(groups, output, args.extra_name)
    missing_path = experiments / 'modern_prediction/missingness/oof_predictions.npz'
    missing_included = missing_path.is_file()
    if missing_included:
        missingness_diagnostics(missing_path, reference, variance, output, args.minimum_group_n)
        input_paths['missingness_oof'] = missing_path
    protocol = {
        'implementation_status': 'Reconstructed after workspace loss; not recovered original source bytes.',
        'script_sha256': sha(Path(__file__)), 'original_beers': 250, 'fine_styles': 22,
        'target_order': reference['targets'].tolist(), 'core_targets': reference['targets'][:30].tolist(),
        'alignment': 'Original chemistry sorted by beer; sensory and fold records joined by beer; modern, extra, and missingness archives checked exactly for Y, beer, fold, targets. Barcodes are not join keys.',
        'nmse_definition': 'v_j=(1/250) sum_i (Y_ij-mean_i Y_ij)^2; loss_i=(1/|T|) sum_j in T (Y_ij-pred_ij)^2/v_j; group NMSE=mean of loss_i within the group.',
        'variance_use': 'Evaluation normalization only; no model training or selection uses these group diagnostics.',
        'global_identity': 'Overall NMSE equals 1 minus macro target R2 on the same 250 rows.',
        'tail_definition': 'Each model separately takes its own highest ceil(0.1*250)=25 individual losses; groups need not contain the same beers. This is not a causal same-subgroup improvement comparison.',
        'style_equal_nmse': 'Unweighted arithmetic mean of the 22 style NMSE values.',
        'small_group_threshold': args.minimum_group_n,
        'random_cv_not_leave_style_out': True,
        'same_style_training_coverage': coverage.to_dict(orient='records'),
        'evaluated_style_fold_cells_without_same_style_training': int(coverage.evaluated_without_same_style_training.sum()),
        'bootstrap_repetitions': 0, 'independent_new_samples_claimed': False,
        'fixed_y16_and_nested_selection_kept_separate': True,
        'nested_column_populated': args.extra_oof is not None,
        'tabdpt_included': tabdpt_path.is_file(),
        'missingness_included': missing_included,
        'input_files': {name: {'path': str(path), 'sha256': sha(path)} for name, path in input_paths.items()},
    }
    (output / 'protocol_and_alignment.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'models': list(predictions), 'nested_included': args.extra_oof is not None,
                      'missingness_included': missing_included, 'fitted_models': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
