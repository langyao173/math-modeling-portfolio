"""Summarize the completed frozen missingness comparison; no inference."""
from pathlib import Path
import hashlib
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'missingness'
SCENARIOS = ('derived_only_missing', 'MCAR_10_strict', 'MCAR_30_strict', 'MCAR_50_strict')
LABELS = {'derived_only_missing': '仅屏蔽派生量', 'MCAR_10_strict': 'MCAR 10%',
          'MCAR_30_strict': 'MCAR 30%', 'MCAR_50_strict': 'MCAR 50%'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    status = json.loads((OUT / 'status.json').read_text())
    assert status['complete'] is True and status['completed_folds'] == 5
    table = pd.read_csv(OUT / 'comparison.csv')
    ci = pd.read_csv(OUT / 'paired_ci_vs_knn.csv')
    table_index = table.set_index(['model', 'scenario', 'targets'])
    ci_index = ci.set_index(['scenario', 'targets', 'metric'])
    records = []
    for scenario in SCENARIOS:
        for group in ('all50', 'core30'):
            modern = table_index.loc[('tabicl_raw231_y16', scenario, group)]
            knn = table_index.loc[('old_knn', scenario, group)]
            median = table_index.loc[('old_median', scenario, group)]
            rci = ci_index.loc[(scenario, group, 'r2')]
            assert abs(float(modern.r2 - knn.r2) - float(rci.delta)) < 1e-12
            row = dict(scenario=scenario, targets=group, tabicl_r2=float(modern.r2),
                knn_r2=float(knn.r2), median_r2=float(median.r2),
                delta_r2=float(rci.delta), delta_r2_ci025=float(rci.ci025), delta_r2_ci975=float(rci.ci975))
            for metric in ('grade_mae', 'grade_exact', 'grade_within_one'):
                gci = ci_index.loc[(scenario, group, metric)]
                assert abs(float(modern[metric] - knn[metric]) - float(gci.delta)) < 1e-12
                row[f'tabicl_{metric}'] = float(modern[metric])
                row[f'knn_{metric}'] = float(knn[metric])
                row[f'delta_{metric}'] = float(gci.delta)
                row[f'delta_{metric}_ci025'] = float(gci.ci025)
                row[f'delta_{metric}_ci975'] = float(gci.ci975)
            records.append(row)
    compact = pd.DataFrame(records)
    compact.to_csv(OUT / 'manuscript_table.csv', index=False)
    probe = json.loads((OUT / 'batch_cache_probe.json').read_text())
    summary = dict(complete=True, model='raw231_y16_no_residual', outer_folds=5, scenarios=4,
        shared_test_masks=True, paired_reference='old_knn', rows=records,
        bootstrap_repetitions=2000,
        bootstrap_scope='Paired beer bootstrap of fixed OOF predictions, without refitting or multiplicity adjustment.',
        selection_scope='This missingness follow-up was added after complete-input Y-PCA16 results; all four scenarios, same-fold masks and model settings were fixed before these stress outcomes. No deployment threshold or model mixture is selected from these outer scores.',
        grades='Training-fold seven-quantile relative intensity grades, not recovered raw panel ratings.',
        batch_probe=probe,
        source_sha256={name:sha(OUT / name) for name in ['comparison.csv', 'paired_ci_vs_knn.csv',
            'oof_predictions.npz', 'experiment_spec.json', 'mask_audit.csv']},
        implementation_sha256=sha(__file__))
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    lines = ['# TabICLv2目标低秩适配：严格缺测的完整五折证据', '',
        '固定 raw231→Y-PCA16，训练折保持原数据，测试折采用与既有中位数/KNN六模型完全相同的缺测规则。每个场景覆盖全部250个外层留出预测；下面的R²按合并OOF计算，不是五个折内R²的均值。', '',
        '## 四场景与同掩码KNN的配对比较', '',
        '| 场景 | 目标组 | TabICL R² | KNN R² | 中位数 R² | ΔR² | 配对95%区间 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for row in compact.itertuples():
        lines.append(f'| {LABELS[row.scenario]} | {row.targets} | {row.tabicl_r2:.6f} | {row.knn_r2:.6f} | '
                     f'{row.median_r2:.6f} | {row.delta_r2:+.6f} | [{row.delta_r2_ci025:.6f}, {row.delta_r2_ci975:.6f}] |')
    lines += ['', '## 核心30维的相对等级', '',
        '| 场景 | TabICL / KNN 等级MAE | ΔMAE及配对95%区间 | TabICL / KNN 完全相符率 | TabICL / KNN 误差≤1比例 |',
        '| --- | --- | --- | --- | --- |']
    for row in compact[compact.targets.eq('core30')].itertuples():
        lines.append(f'| {LABELS[row.scenario]} | {row.tabicl_grade_mae:.6f} / {row.knn_grade_mae:.6f} | '
                     f'{row.delta_grade_mae:+.6f} [{row.delta_grade_mae_ci025:.6f}, {row.delta_grade_mae_ci975:.6f}] | '
                     f'{row.tabicl_grade_exact:.6f} / {row.knn_grade_exact:.6f} | '
                     f'{row.tabicl_grade_within_one:.6f} / {row.knn_grade_within_one:.6f} |')
    core = compact[compact.targets.eq('core30')].set_index('scenario')
    sentences = []
    for scenario in SCENARIOS:
        row = core.loc[scenario]
        direction = '提高' if row.delta_r2 > 0 else '降低'
        sentences.append(f'{LABELS[scenario]}下，核心30维R²由KNN的{row.knn_r2:.6f}变为TabICL的{row.tabicl_r2:.6f}，'
                         f'{direction}{abs(row.delta_r2):.6f}（差值95%区间[{row.delta_r2_ci025:.6f}, {row.delta_r2_ci975:.6f}]）。')
    lines += ['', '## 可直接核对的结果表述', '', *sentences, '',
        '区间来自2,000次配对啤酒bootstrap，使用已固定的OOF预测，不重新训练，亦未作多场景或多指标同时推断校正。跨0区间须保留不确定性；负结果与正结果同等报告。完整输入中的性能改善不能直接替代缺测压力测试的结论。', '',
        '这是在完整输入Y-PCA16结果出现后追加、明确披露的后续研究；四个缺测场景均完成后才汇总，不从这些外层结果再学习缺失率切换阈值或混合权重。', '',
        '## 掩码与执行等价性', '',
        '测试先按 seed=20261023+1000×fold+level 对原50×231形状抽取MCAR掩码，再始终屏蔽 kcalperc、esters_sum、aroma_hops_sum、acids_sum、sulfur_sum 五个派生字段。derived-only不抽取额外MCAR；保留原有NaN。训练折不做缺测增强，目标标准化与PCA16均只使用对应200训练样本。', '',
        f'全五折官方特征预处理的四批与合批结果逐元素一致；神经探针合批差异{probe["concat_max_abs_difference"]:.3g}，KV缓存差异{probe["cache_max_abs_difference"]:.3g}，原完整输入复现差异{probe["original_benchmark_max_abs_difference"]:.3g}。实际执行选项为{probe["resolved"]}，按预先固定2e−5数值容差决定，未使用预测成绩选择。', '',
        '相对等级以各外折200训练响应的六个七分位点定义，真值与预测使用同一切点；这些0–6等级不反演原始评审员评分。全部50维、核心30、逐维指标和三种等级指标的配对区间均在原始CSV中保留。', '',
        '- comparison.csv / paired_ci_vs_knn.csv：完整输入与四个缺测场景的全部指标、差值与区间。',
        '- manuscript_table.csv / summary.json：四场景的精确数值与可供排版读取的结果。',
        '- oof_predictions.npz / checkpoints/：逐啤酒预测、训练/测试索引、目标编码与潜变量。',
        '- mask_audit.csv / checkpoints/fold*_masks.npz：20个准确掩码及SHA256。',
        '- *_relative_grades_0_6.csv：每场景250×50的相对等级输出。',
    ]
    (OUT / 'manuscript_results.md').write_text('\n'.join(lines) + '\n')
    print(compact[['scenario', 'targets', 'tabicl_r2', 'knn_r2', 'delta_r2', 'delta_r2_ci025', 'delta_r2_ci975']].to_string(index=False))


if __name__ == '__main__':
    main()
