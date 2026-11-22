"""Write a source-backed completion report only after all registered work."""
from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).resolve().parent


def table(frame,columns):
    labels=list(columns.values());out=['| '+' | '.join(labels)+' |',
                                      '| '+' | '.join(['---']*len(labels))+' |']
    for row in frame[list(columns)].to_dict(orient='records'):
        values=[]
        for key in columns:
            value=row[key]
            values.append(f'{value:.6f}' if isinstance(value,float) else str(value))
        out.append('| '+' | '.join(values)+' |')
    return '\n'.join(out)


def main():
    status=json.loads((HERE/'status.json').read_text())
    assert status['completed_model_folds']==status['total_model_folds']
    tabdpt=json.loads((HERE/'tabdpt/status.json').read_text())
    missing=json.loads((HERE/'missingness/status.json').read_text())
    assert tabdpt['completed_folds']==5 and missing['complete']
    manifest=json.loads((HERE/'final_artifact/manifest.json').read_text())
    selected=manifest['chosen_cv_candidate']
    scores=pd.concat([pd.read_csv(HERE/'model_comparison.csv'),
                      pd.read_csv(HERE/'tabdpt/model_comparison.csv')],ignore_index=True)
    keep=['old_nested_ensemble','old_eight_global','old_eight_targetwise']
    complete=scores[(~scores.model.str.startswith('old_'))|scores.model.isin(keep)]
    grade=pd.read_csv(HERE/'diagnostics/relative_grade_metrics.csv')
    grade=grade[grade.model.isin(complete.model)]
    ci=pd.concat([pd.read_csv(HERE/'paired_bootstrap.csv'),
                  pd.read_csv(HERE/'tabdpt/paired_bootstrap.csv')],ignore_index=True)
    grade_ci=pd.read_csv(HERE/'diagnostics/relative_grade_paired_ci.csv')
    grade_ci=grade_ci[grade_ci.model.eq(selected)]
    stress=pd.read_csv(HERE/'missingness/comparison.csv')
    subspace=pd.read_csv(HERE/'diagnostics/subspace_error_budget.csv')
    subspace=subspace[subspace.model.isin(complete.model)]
    native_pair=pd.read_csv(HERE/'diagnostics/native_vs_lowrank_paired_bootstrap.csv')
    probe=json.loads((HERE/'missingness/batch_cache_probe.json').read_text())
    audit=json.loads((HERE/'reconstruction_audit.json').read_text())
    assert audit['all_checked_invariants_passed'] and audit['completed_checkpoint_count']==15
    assert all(count==250 for count in audit['test_rows_by_view'].values())
    body=[
        '# 现代预测模型完整实验报告',
        '',
        f'全部 5 个预注册 TabICL 变体、独立 TabDPT 对照及 4 个严格缺测场景均已完成固定五折。TabICL 部署方案为 **{selected}**；在五个 TabICL 候选中按核心 30 个 aroma/flavour 输出的内部 OOF 宏平均 R²选择，宏 50 仅作为同分时的次级准则。TabDPT 是单独的现代方法对照，不属于这个部署选择集合。',
        '',
        '## 1. 评价问题与协议',
        '',
        '250 个啤酒，231 个化学输入，50 个标准化感官响应；前 30 个为主要 aroma/flavour 任务。完全复用旧版 KFold(5, shuffle=True, random_state=20261023)：每折 200 个训练、50 个测试，最终每个啤酒恰有一次留出预测。输入处理、目标标准化与 PCA 均仅使用相应训练折。目标 PCA 的秩固定为 16；正交残差分支的 Ridge alpha 固定为 100。全部候选统一报告，没有根据某个外折表现调整超参数。',
        '',
        '这是已经研究过的数据上的内部方法升级。部署模型的选择基于完整内部 CV；这些结果不等同于新的独立外部验证。',
        '',
        '## 2. 完整输入：主模型与必要消融',
        '',
        table(complete,dict(model='模型',macro_r2='宏 50 R²',core30_r2='核心 30 R²',
                            aroma15_r2='Aroma 15 R²',flavour15_r2='Flavour 15 R²',macro_rmse='宏 RMSE')),
        '',
        '输入 rank+PCA64 是一个组合消融，不能将其与 raw231 的差值单独归因于“秩变换”或“压缩”。正交 Ridge 分支的负面结果只否定这个固定分支在本协议中的收益，不证明所有正交残差信息都不可预测。50 个原始目标分别调用基础模型的 native 分支用于检验输出低秩建模的取舍。',
        '',
        '### 感官子空间中的误差预算',
        '',
        table(subspace,dict(model='模型',subspace_mse='子空间内 MSE',orthogonal_mse='正交 MSE',
                            total_standardized_mse='总标准化 MSE',orthogonal_error_reduction_vs_zero='相对零正交预测的减误差')),
        '',
        '各折使用同一个训练折 PCA16 正交基。总误差精确等于子空间内误差加正交误差；PCA16 预测器的正交分量为零，其正交误差就是该子空间的截断代价。其他预测器可以同时改变两部分。该事后诊断解释已冻结预测，不参与模型选择或调参。这里的量是全50维训练标准化欧氏 MSE；投影混合感官坐标，故不能将此恒等式移植为核心30子集或宏 R² 的分解。截断代价也不是不可约噪声。',
        '',
        '### Native50 相对 PCA16 的直接配对取舍',
        '',
        table(native_pair,dict(targets='目标组',metric='指标',delta='Native50−PCA16',ci025='95% 下界',ci975='95% 上界')),
        '',
        '### 配对不确定性',
        '',
        table(ci,dict(model='模型',targets='目标组',delta_r2='相对旧六模型 ΔR²',ci025='95% 下界',ci975='95% 上界')),
        '',
        '区间由 2,000 次配对啤酒 bootstrap 对固定 OOF 预测计算；每次同时重采样真值与各模型预测。它反映这些已生成 OOF 误差的样本波动，没有重新训练模型，也未作多候选/多指标的同时推断校正。跨越 0 的区间不能表述为显著优越。',
        '',
        '## 3. 对齐竞赛的 0–6 相对分级',
        '',
        '每折对训练 200 个感官响应计算 1/7,…,6/7 六个分位点，再以同一阈值将该折真值与预测映为 0–6 相对等级；阈值相等时采用 searchsorted(side=right)。旧六模型的等级矩阵被精确重现。由于附件只提供标准化感官数据，这些等级是训练参照的相对分级，不是原始评审员 0–6 评分的逆变换。',
        '',
        table(grade,dict(model='模型',targets='目标组',grade_mae='等级 MAE',
                         grade_exact='完全相符率',grade_within_one='误差≤1 比例')),
        '',
        '### 部署候选相对旧六模型的配对等级差异',
        '',
        table(grade_ci,dict(targets='目标组',metric='指标',delta='差值',ci025='95% 下界',ci975='95% 上界')),
        '',
        '全部 250×50 的真值等级、各现代候选等级及逐维差异都保存在 diagnostics。每个感官维度的改进和退步均保留，没有只挑改善指标。',
        '',
        '## 4. 缺测可靠性',
        '',
        '固定 raw231→Y-PCA16 方案，训练数据不做额外遮蔽。对测试折先按旧协议生成 MCAR 10/30/50 掩码，再始终隐藏 kcalperc、esters_sum、aroma_hops_sum、acids_sum、sulfur_sum 五个派生字段；derived-only 场景只额外隐藏这五项。掩码 seed 为 20261023+1000×fold+level，与旧 KNN/median 六模型结果逐啤酒、逐特征对齐。选择该候选追加缺测试验发生在完整输入 PCA16 结果出现后，作为明确披露的后续研究。',
        '',
        table(stress,dict(model='模型',scenario='场景',targets='目标组',r2='R²',
                          grade_mae='等级 MAE',grade_exact='完全相符率',grade_within_one='误差≤1 比例')),
        '',
        '缺测比较没有用测试标签来填补输入或调参数。完整逐维结果、相对 KNN 的配对区间、准确掩码张量与哈希位于 missingness。',
        '',
        f'执行等价性门禁结果：全五折预处理逐元素一致；拼批最大差 {probe["concat_max_abs_difference"]:.9g}，缓存最大差 {probe["cache_max_abs_difference"]:.9g}，原完整输入基准重现差 {probe["original_benchmark_max_abs_difference"]:.9g}。实际启用选项为 {json.dumps(probe["resolved"],ensure_ascii=False)}。这些选项仅按数值等价性确定，未按预测分数选择。',
        '',
        '## 5. 模型来源、许可与重现',
        '',
        'TabICL 使用公开 BSD-3-Clause 权重 tabicl-regressor-v2-20260212.ckpt，固定 Hugging Face revision 4dcd344ece2c00be9e831fdd35bed57b5ad83e19，SHA256 0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a。安装 tabicl==2.2.0、torch==2.14.1+cpu，CPU 6 线程、8 ensemble、AMP 关闭。',
        '',
        'TabDPT 使用公开 Apache-2.0 权重 tabdpt1_3.safetensors，revision a5ca6e01c0fa09ec68c73e958e5199d1932abb3a，SHA256 97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb。独立环境 tabdpt==1.3.1；按官方 train-only 预处理将 231 列 PCA 到 128 网络输入，使用所有 200 个训练上下文、8 ensemble，没有上下文抽样或检索。',
        '',
        'TabDPT 的实际权重头指向 extended_openml_filtered_deduped.csv，但该新版训练清单未包含在所核查的公开训练仓库。没有发现直接重合证据，也不能证明所有啤酒表变体均未进入真实数据预训练。该结果应作为有训练来源边界的内部现代对照，不能标为无污染外部 SOTA。完整核查见 tabdpt/pretraining_audit.md。',
        '',
        'TabICLv2 的论文预训练任务规模与本题 200 行×231 列的外折规模不完全一致；本数据的实测结果是推荐依据，不依赖范围外性能保证。',
        '',
        '## 6. 可运行的最终推断',
        '',
        f'final_artifact/model_context.npz 保存全部 250 个训练化学向量、响应上下文及 {manifest["target_encoding"]["kind"]} 编码；没有生成无必要的全训练集拟合预测。manifest.json 保存选择准则、所有候选分数、数据与代码哈希、权重身份和输入输出尺度。',
        '',
        'predict_tabicl_csv.py 接受列名与附件 1 相同、尺度一致的化学 CSV，输出 50 维标准化感官预测和训练参照 0–6 等级，并保存输入、输出、模型哈希与计算时间。缺失单元格可用 NaN；整列缺失须显式指定 --missing-columns nan。执行脚本不依赖旧项目目录，也不会自动下载模型或调用收费 API。',
        '',
        '## 7. 交付索引',
        '',
        '- model_comparison.csv / paired_bootstrap.csv：TabICL 完整比较和区间。',
        '- tabdpt/model_comparison.csv / paired_bootstrap.csv：独立现代对照。',
        '- diagnostics/relative_grade_metrics.csv / relative_grade_paired_ci.csv：分级指标与区间。',
        '- diagnostics/paired_dimension_differences.csv：逐感官维度的收益与代价。',
        '- missingness/comparison.csv / paired_ci_vs_knn.csv：严格缺测比较。',
        '- reconstruction_audit.json：训练折目标编码、重构与正交误差分解审计。',
        '- experiment_spec.json、tabdpt/experiment_spec.json、missingness/experiment_spec.json：完整参数、版本和来源。',
        '- final_artifact/manifest.json、model_context.npz 与 predict_tabicl_csv.py：部署上下文与推断入口。',
        '',
        '原始论文、官方仓库、模型卡、访问日期及阅读范围见 ../../literature/method_frontier.md 和 .json；各实验 README 提供准确重现命令。',
        '',
        '## 8. 主要方法来源',
        '',
        '- Qu et al., TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model. [原论文](https://arxiv.org/abs/2602.11139)、[官方代码](https://github.com/soda-inria/tabicl)、[本次权重仓库](https://huggingface.co/jingang/TabICL)。',
        '- TabDPT: Scaling Tabular Foundation Models. [原论文](https://arxiv.org/abs/2410.18164)、[官方推断代码](https://github.com/layer6ai-labs/TabDPT-inference)、[公开训练代码](https://github.com/layer6ai-labs/TabDPT-training)。',
        '- TabDPT-Turbo 后续工作：[论文](https://arxiv.org/abs/2608.01400)。实际实验严格使用本报告标识的 1.3 权重；不将论文中其他权重版本的结果移植为本实验结果。',
    ]
    (HERE/'RESULTS.md').write_text('\n'.join(body)+'\n')
    print('Wrote '+str(HERE/'RESULTS.md'))


if __name__=='__main__':main()
