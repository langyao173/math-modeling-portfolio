"""Independently inspect completed penalized-MCEM checkpoints; never fit models."""
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import hashlib
import json
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtri
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXP = HERE.parent/"experiments/generation_moments"
OUT = EXP/"penalized_em"
sys.path.insert(0, str(EXP))
import penalized_interval_em  # Resolve the saved class only.
import conditional_moments


def loss(total, second, n, mu, covariance, mean0, covariance0, kappa, nu):
    """Direct objective, independent of the tested helper functions."""
    scatter = second-np.outer(total, mu)-np.outer(mu, total)+n*np.outer(mu, mu)
    scatter += nu*covariance0+kappa*np.outer(mu-mean0, mu-mean0)
    lower = np.linalg.cholesky(covariance)
    ld = 2*np.log(np.diag(lower)).sum()
    precision = np.linalg.solve(lower.T, np.linalg.solve(lower, np.eye(len(mu))))
    return .5*((n+nu)*ld+np.sum(precision*scatter.T)), scatter


def write_result_report(answer, result_frame):
    scores = pd.read_csv(OUT/"summary.csv").set_index("model")
    reference = pd.read_csv(EXP/"scores_by_beer.csv")
    reference_energy = reference.loc[reference.model.eq("interval_reference"), "energy"].mean()
    lines = ["# 显式惩罚区间MCEM：实际检查点独立审计", "",
        f"6个已完成模型及132个末轮风格更新均通过独立复算。固定参考均值与协方差逐值对应各外折的初始区间模型；重新计算保存的旧参数、新参数和同一组完整条件矩，确认其联合闭式M步。均值最大误差为{answer['largest_mean_error']:.3g}，协方差最大误差为{answer['largest_covariance_error']:.3g}，正规方程最大误差为{answer['largest_normal_equation_error']:.3g}。最小更新协方差特征值为{answer['minimum_covariance_eigenvalue']:.6f}，所有保存区间检查均通过。", "",
        "## 外折评分决定是否保留", "",
        f"固定区间参照的250款按beer等权能量评分为{reference_energy:.9f}；较低更好。所有配对差均相对于同一参照、相同三个外折和评价抽样。", "",
        "| 预算 | 能量评分 | 配对差 | 条件配对区间 | 变差评分配对差 |", "|---|---:|---:|---|---:|"]
    for item in sorted(answer["comparisons"], key=lambda row:row["model"], reverse=True):
        name = item["model"]
        interval = scores.loc[name]
        lines.append(f"| {name} | {item['beer_weighted_energy']:.9f} | {item['paired_energy_delta']:+.9f} | [{interval.energy_delta_ci_lower:.9f}, {interval.energy_delta_ci_upper:.9f}] | {item['paired_variogram_delta']:+.9f} |")
    lines += ["", "两组显式惩罚MCEM在该内部比较中都没有改善固定区间参照；它们的能量差区间均位于更差的一侧，变差评分也更高。较大预算没有扭转这一结论。应将其保留为定义明确、实现可验证的挑战模型，主模型选择继续依据实际分布评分。", "",
              "## 迭代诊断不能替代收敛证明", "",
              "| 预算 | 三折末轮协方差相对变化范围 | 三折末轮均值RMS变化范围 |", "|---|---|---|"]
    for name, group in result_frame.groupby("model", sort=False):
        lines.append(f"| {name} | {100*group.last_relative_covariance_change.min():.3f}%–{100*group.last_relative_covariance_change.max():.3f}% | {group.last_mean_change_rms.min():.6f}–{group.last_mean_change_rms.max():.6f} |")
    lines += ["", "更多保留扫描及更多轮数下的末轮变动较小，但这里同时改变了两个预算因素，也未估计链的有效样本量。每轮固定E统计量上的下降不等于跨轮观测似然单调，末轮差异也不证明MCEM收敛。精确E的标准EM保证只对应既定的带惩罚观测目标，不能推广为未惩罚似然或能量评分的改进。", "",
              "该候选同时固定了参考矩、明确了正则目标并保留均值收缩交叉散布项，因此不是只切换一个矩项的单因素消融。配对区间条件于既有外折模型与固定评价抽样，未覆盖持续方法探索及完整Monte Carlo误差。", "",
              "科学命名建议：固定参考惩罚的区间高斯MCEM；用“联合闭式M步”描述代数性质，用“有限Gibbs条件矩”描述E步。避免称为已收敛EM、精确E步或经验证的生成质量提升。", "",
              "证据：penalized_em_saved_model_audit.json、penalized_em_saved_model_checks.csv、penalized_em_style_mstep_checks.csv，以及experiments/generation_moments/penalized_em中的保存模型、迭代轨迹和逐beer评分。"]
    (HERE/"penalized_em_saved_model_review.md").write_text("\n".join(lines)+"\n")


def main():
    assert json.loads((OUT/"COMPLETE.json").read_text())["completed"]
    results, styles = [], []
    for fold in range(3):
        reference = joblib.load(EXP/f"fold_{fold}_interval_reference.joblib")
        for name, expected_iterations, expected_retained in [
            ("penalized_em_8x8", 8, 8), ("penalized_em_24x32", 24, 32)]:
            checkpoint = OUT/f"fold_{fold}_{name}.joblib"
            model = joblib.load(checkpoint)
            assert model.iterations == expected_iterations and model.retained_sweeps == expected_retained
            assert model.burn_sweeps == 8 and model.kappa == 2. and model.nu == 20.
            assert len(model.refinement_trace_) == expected_iterations
            assert np.array_equal(model.styles_, reference.styles_)
            prior_mean_error = float(np.max(abs(model.prior_mean_-reference.latent_.mean(axis=0))))
            prior_covariance_error = float(np.max(abs(model.prior_cov_-reference.pooled_cov_)))
            assert prior_mean_error < 1e-12 and prior_covariance_error < 1e-12
            trace = pd.DataFrame(model.refinement_trace_)
            disk_trace = pd.read_csv(OUT/f"fold_{fold}_{name}_trace.csv", float_precision="round_trip")
            np.testing.assert_allclose(trace.to_numpy(), disk_trace[trace.columns].to_numpy(), atol=1e-12, rtol=0)
            assert (trace.minimum_style_loss_drop >= -1e-8).all()
            assert (trace.maximum_interval_violation <= 1e-8).all()
            assert (trace.minimum_style_covariance_eigenvalue > 0).all()
            before_all, after_all = [], []
            for record in model.last_statistics_:
                s, n = record["style"], record["n"]
                total, second = record["total"], record["second"]
                mu, covariance = model.mean_[s], model.cov_[s]
                before, _ = loss(total, second, n, record["old_mean"], record["old_cov"],
                    model.prior_mean_, model.prior_cov_, model.kappa, model.nu)
                after, scatter = loss(total, second, n, mu, covariance,
                    model.prior_mean_, model.prior_cov_, model.kappa, model.nu)
                m = total/n
                within = second-n*np.outer(m, m)
                correction = n*model.kappa/(n+model.kappa)*np.outer(m-model.prior_mean_, m-model.prior_mean_)
                expected_mean = (total+model.kappa*model.prior_mean_)/(n+model.kappa)
                expected_covariance = (within+correction+model.nu*model.prior_cov_)/(n+model.nu)
                mean_error = float(np.max(abs(mu-expected_mean)))
                covariance_error = float(np.max(abs(covariance-expected_covariance)))
                equation_error = float(np.max(abs((n+model.nu)*covariance-scatter)))
                prior_inclusion_floor = model.nu/(n+model.nu)*np.linalg.eigvalsh(model.prior_cov_)[0]
                eigenvalue = float(np.linalg.eigvalsh(covariance)[0])
                assert before-after >= -1e-8
                assert abs(before-record["loss_before"]) < 1e-7
                assert abs(after-record["loss_after"]) < 1e-7
                assert mean_error < 1e-12 and covariance_error < 1e-11 and equation_error < 1e-8
                assert eigenvalue+1e-10 >= prior_inclusion_floor
                assert np.linalg.eigvalsh((within+within.T)/2)[0] >= -1e-7
                before_all.append(before);after_all.append(after)
                styles.append(dict(model=name, fold=fold, style=s, n=n,
                    mean_max_error=mean_error, covariance_max_error=covariance_error,
                    covariance_normal_equation_max_error=equation_error,
                    before_loss_replay_error=float(abs(before-record["loss_before"])),
                    after_loss_replay_error=float(abs(after-record["loss_after"])),
                    last_fixed_E_loss_drop=float(before-after),
                    minimum_covariance_eigenvalue=eigenvalue,
                    prior_positive_definiteness_lower_bound=float(prior_inclusion_floor)))
            last = trace.iloc[-1]
            assert abs(sum(before_all)-last.expected_penalized_loss_before) < 1e-6
            assert abs(sum(after_all)-last.expected_penalized_loss_after) < 1e-6
            lower, upper = ndtri(model.row_low_), ndtri(model.row_high_)
            violation = float(max(0, np.max(lower-model.latent_), np.max(model.latent_-upper)))
            assert violation < 1e-8
            results.append(dict(model=name, fold=fold, iterations=expected_iterations,
                retained_sweeps=expected_retained, prior_mean_max_error=prior_mean_error,
                prior_covariance_max_error=prior_covariance_error,
                minimum_logged_fixed_E_loss_drop=float(trace.minimum_style_loss_drop.min()),
                last_relative_covariance_change=float(last.relative_covariance_change),
                last_mean_change_rms=float(last.mean_change_rms),
                final_posterior_mean_interval_violation=violation,
                saved_statistics_styles=len(model.last_statistics_),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
            del model
    result_frame, style_frame = pd.DataFrame(results), pd.DataFrame(styles)
    result_frame.to_csv(HERE/"penalized_em_saved_model_checks.csv", index=False)
    style_frame.to_csv(HERE/"penalized_em_style_mstep_checks.csv", index=False)
    ss = pd.read_csv(OUT/"scores_by_beer.csv")
    ref = pd.read_csv(EXP/"scores_by_beer.csv")
    ref = ref[ref.model.eq("interval_reference")].sort_values("beer")
    table = pd.read_csv(OUT/"summary.csv").set_index("model")
    comparisons = []
    for name, group in ss.groupby("model"):
        group = group.sort_values("beer")
        assert len(group) == 250 and group.beer.is_unique and np.array_equal(group.beer, ref.beer)
        delta = float(np.mean(group.energy.to_numpy()-ref.energy.to_numpy()))
        assert abs(delta-table.loc[name, "paired_energy_delta"]) < 1e-13
        assert abs(group.energy.mean()-table.loc[name, "beer_weighted_energy"]) < 1e-13
        comparisons.append(dict(model=name, beer_weighted_energy=float(group.energy.mean()),
            paired_energy_delta=delta, paired_variogram_delta=float(np.mean(group.variogram.to_numpy()-ref.variogram.to_numpy())),
            confidence_scope="Fixed-fold models and fixed evaluation draws; interval excludes the full adaptive-research/MC uncertainty."))
    answer = dict(completed=True, model_checkpoints=len(results), style_mstep_checks=len(styles),
        all_fixed_prior_checks_pass=True, all_last_M_steps_replayed=True,
        finite_E_not_observed_likelihood=True, no_new_training=True, comparisons=comparisons,
        largest_mean_error=float(style_frame.mean_max_error.max()),
        largest_covariance_error=float(style_frame.covariance_max_error.max()),
        largest_normal_equation_error=float(style_frame.covariance_normal_equation_max_error.max()),
        largest_before_loss_replay_error=float(style_frame.before_loss_replay_error.max()),
        largest_after_loss_replay_error=float(style_frame.after_loss_replay_error.max()),
        minimum_covariance_eigenvalue=float(style_frame.minimum_covariance_eigenvalue.min()))
    (HERE/"penalized_em_saved_model_audit.json").write_text(json.dumps(answer, ensure_ascii=False, indent=2))
    write_result_report(answer, result_frame)
    print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
