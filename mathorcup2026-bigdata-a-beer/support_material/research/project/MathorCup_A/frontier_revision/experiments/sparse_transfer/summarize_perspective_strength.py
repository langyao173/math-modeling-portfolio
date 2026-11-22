"""Summarize all 22 independently audited perspective comparisons.

This does not select final chemistry profiles or replace aggregate_results.py.
It reports matched-case numerical bounds and preserves negative outcomes.
"""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    for directory in ('perspective','perspective_sdp'):
        gate = json.loads((HERE/directory/'INTEGRITY_DONE.json').read_text())
        assert gate['state'] == 'complete_independently_verified'
        assert gate['completed_cases'] == 22 and gate['all_full_witnesses_pass']
    lm = pd.read_csv(HERE/'perspective/comparison.csv')
    sdp = pd.read_csv(HERE/'perspective_sdp/comparison.csv')
    assert len(lm) == len(sdp) == 22
    lm = lm.set_index(['policy','beer'])
    sdp = sdp.set_index(['policy','beer'])
    assert set(lm.index) == set(sdp.index)
    tolerance = 1e-6  # same conservative integerization as final aggregation
    records = []
    for (policy,beer), a in lm.sort_index().iterrows():
        b = sdp.loc[(policy,beer)]
        assert abs(a.old_dual_bound-b.old_dual_bound) < 1e-12
        assert a.old_full_upper_bound == b.old_full_upper_bound
        raw = dict(original=float(a.old_dual_bound),lambda_min=float(a.new_dual_bound),sdp=float(b.new_dual_bound))
        integer = {name:max(0,math.ceil(value-tolerance)) for name,value in raw.items()}
        upper = dict(original=int(a.old_full_upper_bound),lambda_min=int(a.new_full_upper_bound),sdp=int(b.new_full_upper_bound))
        prior_lb = max(integer['original'],integer['lambda_min'])
        prior_ub = min(upper['original'],upper['lambda_min'])
        records.append(dict(policy=policy,beer=int(beer),
            **{f'{name}_raw_dual':value for name,value in raw.items()},
            **{f'{name}_integer_lower':value for name,value in integer.items()},
            **{f'{name}_full_upper':value for name,value in upper.items()},
            best_prior_lower=prior_lb,best_prior_upper=prior_ub,
            pooled_lower=max(integer.values()),same_policy_pooled_upper=min(upper.values()),
            lambda_lower_improved=integer['lambda_min']>integer['original'],
            sdp_lower_improved_over_prior=integer['sdp']>prior_lb,
            sdp_full_upper_improved_over_prior=upper['sdp']<prior_ub,
            lambda_solver_seconds=float(a.solver_seconds),sdp_solver_seconds=float(b.solver_seconds)))
    frame = pd.DataFrame(records)
    frame.to_csv(HERE/'perspective_strength_comparison.csv',index=False)
    matrix = json.loads((HERE/'perspective_sdp/diagonal_sdp_audit.json').read_text())
    all_duals=frame[['original_raw_dual','lambda_min_raw_dual','sdp_raw_dual']].to_numpy()
    rounding_sensitivity={str(epsilon):int(np.max(np.abs(np.ceil(all_duals-tolerance)-np.ceil(all_duals-epsilon))))
                          for epsilon in (1e-7,1e-6,1e-5,1e-4,1e-3)}
    summary = dict(complete=True,cases=22,policies=2,sources_per_policy=11,
        integerization_tolerance=tolerance,
        maximum_integer_bound_change_under_rounding_tolerance=rounding_sensitivity,
        lambda_lower_improved_cases=int(frame.lambda_lower_improved.sum()),
        lambda_lower_worse_cases=int((frame.lambda_min_integer_lower<frame.original_integer_lower).sum()),
        lambda_full_upper_improved_cases=int((frame.lambda_min_full_upper<frame.original_full_upper).sum()),
        sdp_lower_improved_over_best_prior_cases=int(frame.sdp_lower_improved_over_prior.sum()),
        sdp_lower_worse_than_best_prior_cases=int((frame.sdp_integer_lower<frame.best_prior_lower).sum()),
        sdp_full_upper_improved_over_best_prior_cases=int(frame.sdp_full_upper_improved_over_prior.sum()),
        pooled_lower_improved_over_original_cases=int((frame.pooled_lower>frame.original_integer_lower).sum()),
        pooled_lower_total_gain=int((frame.pooled_lower-frame.original_integer_lower).sum()),
        same_policy_pooled_full_upper_total_reduction=int((frame.original_full_upper-frame.same_policy_pooled_upper).sum()),
        diagonal_sdp_primal_trace=matrix['primal_trace'],
        diagonal_sdp_dual_trace_upper_bound=matrix['dual_trace_upper_bound'],
        interpretation='Per-case maximum of compatible numerical solver lower bounds and minimum of fully rechecked upper bounds. This report pools within each policy; final aggregation may further improve relaxed-policy upper bounds by including strict-policy witnesses.',
        mathematical_scope='The primary frozen problem explicitly includes both the linear stale guard and complete nonlinear f-star nonincrease; relaxing only the nonlinear constraint gives the recorded lower bound.',
        timing_scope='180-second one-thread SCIP wall-time budgets on shared CPU resources; differing concurrent loads and the original accepted warm start mean this is not a dedicated-hardware algorithm speed ranking.',
        provenance='Established perspective and max-trace diagonal SDP constructions; no claim that the general methods are novel.')
    (HERE/'perspective_strength_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    lines = ['# 对角 perspective 加强：完整22例结果','',
        f'两种分解均完成11源×两政策的180秒求解，全部通过独立矩阵、来源与完整解码审计。λmin方案相对原问题在 {summary["lambda_lower_improved_cases"]}/22 例提高整数下界；最大迹SDP相对原方法与λmin的最佳已有证据，又在 {summary["sdp_lower_improved_over_best_prior_cases"]}/22 例提高下界。所有下降实例同样保留。',
        '',
        '| 政策 | Beer | 原LB | λmin LB | SDP LB | 汇总LB | 原完整UB | λmin完整UB | SDP完整UB | 同政策汇总UB |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in frame.itertuples():
        lines.append(f'| {r.policy} | {r.beer} | {r.original_integer_lower} | {r.lambda_min_integer_lower} | '
                     f'{r.sdp_integer_lower} | {r.pooled_lower} | {r.original_full_upper} | '
                     f'{r.lambda_min_full_upper} | {r.sdp_full_upper} | {r.same_policy_pooled_upper} |')
    lines += ['',
        '整数下界统一按 ceil(raw_dual−1e−6) 汇总，原始浮点对偶界仍保留在CSV；与每例原记录1e−7的容差并列说明。完整上界取解码后通过全部冻结约束的实际原始字段改动数，不取松弛模型的primal bound。',
        '',
        f'取整容差敏感性（容差→相对主口径的最大整数差）为 {json.dumps(rounding_sensitivity,ensure_ascii=False)}。该检查只衡量最后取整步骤的稳定性，不把它当作求解器全部浮点误差的严格上界。',
        '',
        '最大迹分解不逐坐标支配λmin分解，有限时限的搜索路径亦不同。提高迹不能代替逐例证据；最终采用兼容证据的最大下界和完整可行见证的最小上界。SDP矩阵目标的原始/对偶差距说明矩阵分解解得多准，不能当作基数优化gap。',
        '',
        '这份表不含“严格政策可行点转入放宽政策”的额外上界收益；最终论文的政策比较应以 final_summary/selection_cases.csv 为准。',
        '',
        '本次主问题显式包含线性 stale 守卫与非线性 f★ 守卫。删去线性守卫会扩大可行集，不能自动复用本表下界；对应补充结果单列于 certificate_without_proxy。',
        '',
        '求解使用共享CPU上的180秒墙钟配额，结果比较有效界，不宣传独占硬件的算法速度优势。证据由双精度代数检查、SCIP浮点全局界和完整原始谱复核构成；不是区间算术证书。',
        '',
        '方法来源及推导见 perspective_method.md、perspective_sdp_method.md 和 perspective_sources.json；一般 perspective、对角SDP与ADMM技术均明确引用原作者。',
    ]
    (HERE/'perspective_strength_results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
