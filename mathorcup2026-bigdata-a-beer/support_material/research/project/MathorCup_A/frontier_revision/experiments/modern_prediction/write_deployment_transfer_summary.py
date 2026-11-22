"""Render the completed deployment challenge into manuscript-ready evidence.

Run after run_deployment_transfer_challenge.py. No model call or reselection.
"""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name] = '1'
import hashlib
import inspect
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE/'deployment_transfer_challenge'


def audit_query_batching():
    """Check the real full-context feature transforms, without new inference."""
    from tabicl._sklearn.preprocessing import TransformToNumerical, EnsembleGenerator
    context = np.load(HERE/'final_artifact/model_context.npz',allow_pickle=False)
    manifest = json.loads((HERE/'final_artifact/manifest.json').read_text())
    inputs = pd.read_csv(OUT/'source_and_candidate_chemistry.csv')
    query = inputs[context['features'].tolist()].to_numpy(float)
    encoder = TransformToNumerical()
    train = encoder.fit_transform(context['X'])
    generator = EnsembleGenerator(classification=False,n_estimators=8,
        norm_methods=['none','power'],feat_shuffle_method='latin',outlier_threshold=4.,
        random_state=manifest['parameters']['random_state'])
    generator.fit(train,np.zeros(len(train)))
    joined = generator.transform(encoder.transform(query),mode='test')
    rowwise = [generator.transform(encoder.transform(row[None]),mode='test') for row in query]
    records = []
    for method,(values,) in joined.items():
        reference = np.concatenate([item[method][0] for item in rowwise],axis=1)
        records.append(dict(normalization=method,shape=list(values.shape),
            maximum_absolute_difference=float(np.max(np.abs(values-reference),initial=0.)),
            exact_equal=bool(np.array_equal(values,reference,equal_nan=True)),
            equal_at_2e_5=bool(np.allclose(values,reference,rtol=2e-5,atol=2e-5,equal_nan=True))))
    source = Path(inspect.getfile(EnsembleGenerator))
    audit = dict(all_rows_exact_equal=all(r['exact_equal'] for r in records),
        all_rows_equal_at_2e_5=all(r['equal_at_2e_5'] for r in records),rows=len(query),
        context_rows=len(train),records=records,
        comparison='All source/candidate rows in one batch versus each row separately after the exact official feature transforms and 8 shuffles.',
        heldout_scores_computed=False,new_model_inference=False,
        reason='The official power-transform exception fallback can act on the whole query batch; checking the actual queries rules out that source of dependence here.',
        limitation='This audits feature transforms for this exact exported context and query set. The numerical neural batching equivalence and train-only attention semantics are separately recorded in the missingness probe.',
        input_sha256=hashlib.sha256((OUT/'source_and_candidate_chemistry.csv').read_bytes()).hexdigest(),
        preprocessing_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (OUT/'query_batching_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    return audit


def main():
    done = json.loads((OUT/'COMPLETE.json').read_text())
    assert done['complete'] and done['all50'] and done['source_count'] == 11
    for name, expected in done['result_sha256'].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest() == expected
    with threadpool_limits(limits=1):
        batching = audit_query_batching()
    assert batching['all_rows_equal_at_2e_5'], 'Inspect query batch dependence before writing a pointwise interpretation.'
    frame = pd.read_csv(OUT/'policy_rho_summary.csv')
    primary = frame[frame.rho.eq(.1)].copy()
    assert len(primary) == 2
    policies = primary.set_index('policy')
    strict = int(policies.loc['strict','fplus_nonincrease_exact'])
    exceedance = int(policies.loc['exceedance','fplus_nonincrease_exact'])
    compact = dict(complete=True,all50=True,source_count=11,
        selected_model=done['model'],target=done['target'],
        selected_model_stale_oof_r2=done['selected_model_stale_oof_r2'],
        unique_candidate_profiles=done['unique_candidate_profiles'],
        unique_source_candidate_pairs=done['unique_source_candidate_pairs'],
        unique_pair_nonincrease_exact=done['unique_pair_nonincrease_exact'],
        unique_pair_increase=done['unique_pair_increase'],
        primary_rho=.1,primary_strict_nonincrease=strict,primary_exceedance_nonincrease=exceedance,
        primary_cases_per_policy=11,primary_policy_rows=primary.to_dict(orient='records'),
        all_policy_rho_rows=frame.to_dict(orient='records'),
        candidate_pool_sha256=done['candidate_pool_sha256'],
        original_fstar_pool_unchanged=done['original_fstar_pool_unchanged'],
        actual_query_preprocessing_batch_invariance=batching['all_rows_exact_equal'],
        interpretation='Cross-predictor sensitivity on an internally fitted all-data context, not independent experimental validation or a new selection criterion.',
        complete_file_sha256=hashlib.sha256((OUT/'COMPLETE.json').read_bytes()).hexdigest())
    (OUT/'summary.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2))
    lines = [
        '# 最终化学 CSV 部署挑战：固定迁移谱的跨预测器复核',
        '',
        '## 可直接用于正文的结果段落',
        '',
        f'进一步将最终固定候选与11款源啤酒一同输入全数据部署预测器 f⁺（{done["model"]}），通过实际化学 CSV 接口取得全部50维感官预测。在ρ=0.10的主情形下，严格化学政策的 {strict}/11 个候选、阈值超标不增加政策的 {exceedance}/11 个候选在 f⁺ 下仍满足 stale_hops 非增；两组在冻结主预测器 f★ 下均为11/11通过。该复核发生在候选池冻结之后，未以 f⁺ 结果重新选择候选或改变主问题的可行性标准。',
        '',
        f'f⁺ 的 stale_hops 内部 OOF R² 为 {done["selected_model_stale_oof_r2"]:.6f}，其全数据上下文还包含源样本。因此，这里的同向或反向事件反映跨预测器敏感性，不能转写为独立实验验证或真实酿造成功概率。',
        '',
        '## 全部政策与距离改善阈值',
        '',
        '| 政策 | ρ | f★完整通过 | f⁺非增（≤0） | f⁺增加 | Δstale均值 | Δstale最小 | Δstale最大 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for r in frame.itertuples():
        label = '严格不增加' if r.policy == 'strict' else '阈值超标不增加'
        lines.append(f'| {label} | {r.rho:g} | {int(r.fstar_full_accepted)}/{int(r.cases)} | '
                     f'{int(r.fplus_nonincrease_exact)}/{int(r.cases)} | {int(r.fplus_increase)} | '
                     f'{r.fplus_stale_delta_mean:.6f} | {r.fplus_stale_delta_min:.6f} | {r.fplus_stale_delta_max:.6f} |')
    lines += [
        '',
        '## 审计与文件',
        '',
        f'两个政策×四个ρ共88个选择，经231维化学向量精确去重后为 {done["unique_candidate_profiles"]} 个候选输入；加入11个源输入。以“源啤酒、候选化学谱”为单位共有 {done["unique_source_candidate_pairs"]} 个唯一比较，其中 {done["unique_pair_nonincrease_exact"]} 个保持非增，{done["unique_pair_increase"]} 个转为增加。',
        '',
        f'冻结候选池 selection_cases.csv 的 SHA256：{done["candidate_pool_sha256"]}。推断前后完全一致。全部50维推断、源/候选映射、原f★结论和新f⁺差值均保存；同时保留≤1e−8的数值容差判定，正文采用严格≤0的事件口径。',
        '',
        f'针对这批实际输入，另将官方预处理的整批计算与逐行计算逐元素核对，8个集成成员、两种归一化分支的精确一致结果为 {batching["all_rows_exact_equal"]}。该检查不额外调用预测模型，用于排除官方 power-transform 异常回退引起的批次依赖。',
        '',
        '- all50_sensory_predictions.csv：全部50维感官预测；相邻文件保存训练参照相对等级。',
        '- cross_model_case_checks.csv：88个政策/ρ选择的原主判定、新预测差值和事件关系。',
        '- cross_model_unique_checks.csv：按源啤酒与化学谱去重的跨模型复核。',
        '- policy_rho_summary.csv / summary.json：正文和程序可直接读取的汇总。',
        '- protocol.json / COMPLETE.json：候选池、上下文、推断代码与结果哈希，适用范围及完成门禁。',
        '- query_batching_audit.json：实际250行部署上下文及这批源/候选输入的逐行/整批预处理核对。',
        '',
        '上述推断没有改变原候选数据、整数成本、几何证书或 f★ 通过状态。',
    ]
    (OUT/'manuscript_results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(compact,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
