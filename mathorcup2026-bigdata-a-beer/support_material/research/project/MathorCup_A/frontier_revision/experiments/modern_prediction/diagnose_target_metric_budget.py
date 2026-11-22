"""Exact fixed-OOF R2 loss decomposition with the required cross term.

This is an algebraic diagnostic of saved predictions, not a new model or
selection rule. It uses the same train-fold PCA16 basis for every predictor.
"""
from pathlib import Path
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'diagnostics'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    archive = np.load(HERE / 'oof_arrays.npz', allow_pickle=False)
    Y, fold = archive['Y'], archive['fold']
    predictions = {name: archive[name] for name in archive.files
                   if name not in ('Y', 'fold', 'beer', 'targets')}
    dpt = np.load(HERE / 'tabdpt/oof_arrays.npz', allow_pickle=False)
    assert np.array_equal(dpt['Y'], Y) and np.array_equal(dpt['fold'], fold)
    predictions['tabdpt13_raw231_pca128_y50'] = dpt['prediction']
    sst = np.square(Y - Y.mean(0)).sum(0)
    assert (sst > 0).all()
    groups = {'all50': np.arange(50), 'core30': np.arange(30)}
    records, sources, maximum_error = [], {}, 0.
    for f in range(5):
        path = HERE / 'checkpoints' / f'raw231_y16_fold{f}_complete.npz'
        checkpoint = np.load(path, allow_pickle=False)
        V, sigma = checkpoint['target_components'].T, checkpoint['target_scale']
        np.testing.assert_allclose(V.T @ V, np.eye(16), rtol=0., atol=1e-10)
        test = fold == f
        assert np.array_equal(checkpoint['test_indices'], np.flatnonzero(test))
        sources[str(path.relative_to(HERE))] = sha(path)
        for name, prediction in predictions.items():
            error = (prediction[test] - Y[test]) / sigma
            parallel = (error @ V) @ V.T
            perpendicular = error - parallel
            for group, indices in groups.items():
                weights = np.square(sigma[indices]) / (len(indices) * sst[indices])
                inside = float(np.sum(np.square(parallel[:, indices]) * weights))
                outside = float(np.sum(np.square(perpendicular[:, indices]) * weights))
                cross = float(np.sum(2 * parallel[:, indices] * perpendicular[:, indices] * weights))
                direct = float(np.sum(np.square(prediction[test][:, indices] - Y[test][:, indices]) /
                                      (len(indices) * sst[indices])))
                identity_error = abs(inside + outside + cross - direct)
                maximum_error = max(maximum_error, identity_error)
                assert identity_error < 1e-12
                records.append(dict(model=name, targets=group, fold=f,
                    projected_error_term=inside, orthogonal_error_term=outside,
                    weighted_cross_term=cross, r2_loss_contribution=direct,
                    identity_error=identity_error))
    by_fold = pd.DataFrame(records)
    total = by_fold.groupby(['model', 'targets'], as_index=False)[
        ['projected_error_term', 'orthogonal_error_term', 'weighted_cross_term', 'r2_loss_contribution']].sum()
    total['macro_r2'] = 1 - total.r2_loss_contribution
    for row in total.itertuples():
        indices = groups[row.targets]
        actual = (1 - np.square(predictions[row.model] - Y).sum(0) / sst)[indices].mean()
        assert abs(row.macro_r2 - actual) < 1e-12
    index = total.set_index(['model', 'targets'])
    contrasts = []
    for left, right in [('raw231_y50_native', 'raw231_y16_no_residual'),
                        ('raw231_y16_no_residual', 'old_nested_ensemble'),
                        ('raw231_y16_ridge_residual', 'raw231_y16_no_residual')]:
        for group in groups:
            delta = index.loc[(left, group)] - index.loc[(right, group)]
            reconstructed = -float(delta[['projected_error_term', 'orthogonal_error_term', 'weighted_cross_term']].sum())
            assert abs(reconstructed - delta.macro_r2) < 1e-12
            contrasts.append(dict(model=left, reference=right, targets=group,
                delta_projected_error=float(delta.projected_error_term),
                delta_orthogonal_error=float(delta.orthogonal_error_term),
                delta_weighted_cross=float(delta.weighted_cross_term),
                delta_macro_r2=float(delta.macro_r2), reconstructed_delta_macro_r2=reconstructed))
    contrasts = pd.DataFrame(contrasts)
    by_fold.to_csv(OUT / 'target_metric_budget_by_fold.csv', index=False)
    total.to_csv(OUT / 'target_metric_budget.csv', index=False)
    contrasts.to_csv(OUT / 'target_metric_budget_contrasts.csv', index=False)
    protocol = dict(method='Exact weighted error expansion of fixed saved OOF predictions',
        models=list(predictions), groups={key:value.tolist() for key,value in groups.items()},
        projector='The common training-fold standardized-response PCA16 projector, separately fitted on each 200-row training set in the completed benchmark.',
        scale='Training-fold sample standard deviations of each response.',
        evaluation_denominator='Global OOF truth SST for each target, exactly as in the reported pooled macro R2; this is a scoring denominator, never a fitted predictor input.',
        formula='1 - macro_R2_G = sum_f sum_i_in_f sum_j_in_G sigma_fj^2 (p_ij^2 + q_ij^2 + 2 p_ij q_ij) / (|G| SST_j), with p=E P_f, q=E(I-P_f), E=(prediction-truth)/sigma_f.',
        maximum_fold_identity_error=maximum_error,
        model_fit_or_inference=False, selection_feedback=False,
        distinction='Orthogonality cancels the cross term only in the full unweighted training-standardized Euclidean metric. Target selection and unequal macro-R2 weights generally reintroduce it.',
        interpretation='An exact error accounting identity, not a causal decomposition, independent validation, or claim of irreducible noise.',
        sources_sha256={'oof_arrays.npz':sha(HERE / 'oof_arrays.npz'),
                        'tabdpt/oof_arrays.npz':sha(HERE / 'tabdpt/oof_arrays.npz'), **sources},
        implementation_sha256=sha(__file__))
    (OUT / 'target_metric_budget_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    keep = ['old_nested_ensemble', 'raw231_y16_no_residual', 'raw231_y16_ridge_residual',
            'raw231_y50_native', 'tabdpt13_raw231_pca128_y50']
    lines = ['# 精确评分误差预算：解释50维与核心30维的取舍', '',
        '本诊断使用已冻结的全体 OOF 预测，不重新拟合、不追加候选、不改变部署选择。', '',
        '令第f折的训练目标尺度为D_f=diag(σ_f)，训练PCA16投影为P_f=V_fV_fᵀ。对该折留出样本定义标准化误差E_f=(Ŷ_f−Y_f)D_f⁻¹，p_f=E_fP_f，q_f=E_f(I−P_f)。对目标组G，记SST_j=Σ_i(Y_ij−Ȳ_j)²。则', '',
        '$$1-\\overline{R^2}_G=\\sum_f\\sum_{i\\in f}\\sum_{j\\in G}\\frac{\\sigma_{fj}^2}{|G|\\,SST_j}\\left(p_{ij}^2+q_{ij}^2+2p_{ij}q_{ij}\\right).$$', '',
        '三个分量依次为投影误差项、正交误差项与加权交叉项。原全50维训练标准化欧氏范数下，正交性令交叉项消失；宏R²具有逐维权重，核心30还截取坐标，所以不能省去交叉项。评分用的SST包含完整OOF真值，作用仅是重现评价指标，不进入模型拟合。', '',
        '| 模型 | 目标组 | 投影误差项 | 正交误差项 | 加权交叉项 | 1−R² | R² |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for name in keep:
        for row in total[total.model.eq(name)].itertuples():
            lines.append(f'| {row.model} | {row.targets} | {row.projected_error_term:.6f} | '
                         f'{row.orthogonal_error_term:.6f} | {row.weighted_cross_term:.6f} | '
                         f'{row.r2_loss_contribution:.6f} | {row.macro_r2:.6f} |')
    lines += ['', '## Native50相对Y-PCA16：主指标改变来自哪里', '',
        '| 目标组 | Δ投影误差 | Δ正交误差 | Δ交叉项 | ΔR² |',
        '| --- | --- | --- | --- | --- |']
    pair = contrasts[(contrasts.model == 'raw231_y50_native') & (contrasts.reference == 'raw231_y16_no_residual')]
    for row in pair.itertuples():
        lines.append(f'| {row.targets} | {row.delta_projected_error:.6f} | {row.delta_orthogonal_error:.6f} | '
                     f'{row.delta_weighted_cross:.6f} | {row.delta_macro_r2:.6f} |')
    lines += ['',
        'Native50在两组目标的投影误差项都更低，但核心30的正交项与交叉项恶化超过这一收益。因此宏50提高与核心30降低可以同时成立，不能用一个平均分替代任务权衡。此分解是精确代数记账，不把相关误差项当作可独立干预的因果机制。', '',
        f'逐折恒等式最大绝对误差为 {maximum_error:.3g}；各模型两组汇总均独立重现报告R²到1e−12内。全部模型而不只是正文示例行均保存在CSV；ΔR²等于三个误差项差值之和取负。', '',
        '该诊断与 subspace_error_budget.csv 的无权全50维欧氏分解互补；两份表的度量定义不同，数值不能直接相减。',
    ]
    (OUT / 'target_metric_budget.md').write_text('\n'.join(lines) + '\n')
    print(pair.to_string(index=False))
    print(json.dumps(dict(models=len(predictions), maximum_identity_error=maximum_error)))


if __name__ == '__main__':
    main()
