"""Create the completed challenge report and its editable, source-backed figure."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mathorcup_mpl_cache")
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent


def manuscript_section(stability, cases, support, run, support_protocol):
    """Derive the paper section from completed evidence, without manual numbers."""
    main = cases[np.isclose(cases.rho, .1)].copy()
    assert len(main) == 22 and not main.duplicated(["policy", "beer"]).any()
    main = main.merge(support[["profile_id", "pca10_within_empirical_threshold",
                              "pca10_distance_ratio", "rank225_within_empirical_threshold",
                              "pca_residual_within_empirical_threshold"]], on="profile_id", how="left",
                      validate="many_to_one")
    lines = [
        f"冻结最终候选身份后，我们在预先固定的200组风格分层重抽样上，重拟合原分类器和完整50目标的六个感官模型，保持模型规格与集成权重不变。检验覆盖{run['selected_cases']}个政策—源啤酒—距离要求组合，包含{run['n_new_candidate_uids']}个不同候选UID；原源样本和旧k=80方案作为配对参照。候选仍按完整主模型合格性、实际字段变化数和二次目标选择，以下稳定性结果没有反馈给寻优。",
        "",
        "表9　至少10%目标距离改善时的条件模型稳定性。频率按11款源啤酒等权汇总，支持数量也按这11个已选方案计算。",
        "",
        "| 缺陷政策 | 有候选/11 | Tripel分类 | 陈旧味非增 | 两事件同时成立 | PCA10支持内 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    comparisons = []
    for policy, label in [("strict", "严格浓度不增"), ("exceedance", "超阈负担不增")]:
        group = main[main.policy.eq(policy)]
        selected = group[group.candidate_exists]
        assert len(group) == 11
        supported = int(selected.pca10_within_empirical_threshold.fillna(False).sum())
        lines.append(f"| {label} | {len(selected)}/11 | {100*selected.Tripel_frequency.mean():.1f}% | {100*selected.stale_nonincrease_frequency.mean():.1f}% | {100*selected.joint_frequency.mean():.1f}% | {supported}/{len(selected)} |")
        both = selected[selected.old80_meets_required_rho]
        comparisons.append(
            f"{label}下，{len(both)}款源的双方方案都满足10%距离要求；这一共同集合的联合频率为新方案{100*both.joint_frequency.mean():.1f}%、旧方案{100*both.old80_joint_frequency.mean():.1f}%，配对差{100*both.joint_frequency_minus_old80.mean():+.1f}个百分点"
        )
    lines += [
        "",
        "；".join(comparisons) + "。旧k=80是旧支持预算标记，其实际修改字段数单列。未达到同一距离要求的旧方案不进入上述共同集合；完整结果同时保留全部11款的参照频率。",
        "",
        f"两政策的平均联合频率仅为{100*main.groupby('policy').joint_frequency.mean().min():.1f}%–{100*main.groupby('policy').joint_frequency.mean().max():.1f}%；风格判定一致性明显低于陈旧味不增的一致性。固定模型下的可行和较低编辑成本，尚未转化为重拟合下的稳定迁移。",
        "",
        f"PCA10在原250款啤酒的固定秩坐标中拟合，保留{100*support_protocol['total_explained_variance']:.2f}%变异。支持阈值由28款真实Tripel排除自身后的3近邻距离取95%经验分位数得到；候选使用同一投影与阈值，另以完整225维近邻距离和重构残差补充检查。每个方案的真实近邻ID均可追溯。",
        "",
        f"上述22个主要求方案中，{int(main.rank225_within_empirical_threshold.sum())}个通过225维近邻检查，{int(main.pca_residual_within_empirical_threshold.sum())}个通过重构残差检查；这些邻域结果与较低的模型稳定性同时出现，表明经验支持诊断不能代替目标判据的重拟合检验。",
        "",
        "![条件模型稳定性与经验支持](../experiments/transfer_robustness/transfer_robustness_paper.svg){width=16.1cm}",
        "",
        "图12　固定候选在200次重拟合中的联合判据频率及其经验支持。上图保留各源和距离要求；下图显示联合频率与PCA10近邻距离比。橙色小方块标记超过经验阈值的方案。",
        "",
        "这里的联合频率直接计算“分类为Tripel且陈旧味预测不增”的交集。源样本相对自身的陈旧味差恒为零，其联合频率只反映分类波动。200次抽样给出0.5个百分点的数值分辨率，频率0.5时的条件Monte Carlo标准误约3.54个百分点。频率描述固定模型流程对经验重抽样的稳定程度；经验95%阈值也只是邻域诊断，两者都不是实际配制成功概率或分布无关覆盖保证。",
    ]
    (HERE / "manuscript_results.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    names = ["run_summary.json", "conditional_model_stability.csv", "case_comparisons.csv",
             "support_diagnostics.csv", "support_protocol.json"]
    provenance = {
        "generated_by": Path(__file__).name,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names},
        "rule": "All numerical paper statements are computed from completed outputs; no manually inserted result values.",
        "main_distance_requirement": .1,
        "unit": "One preselected feasible candidate per policy and source beer; paired comparison conditions on both new and old meeting the same rho.",
    }
    (HERE / "manuscript_results_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def vector_heatmap(ax, values, cmap):
    """Keep every cell as a vector path, with imshow's original coordinates."""
    rows, columns = values.shape
    mesh = ax.pcolormesh(np.arange(columns + 1)-.5, np.arange(rows + 1)-.5,
                         values, shading="flat", cmap=cmap, vmin=0, vmax=1,
                         rasterized=False)
    ax.set_xlim(-.5, columns-.5)
    ax.set_ylim(rows-.5, -.5)
    return mesh


def figure(stability, cases, support):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
        "svg.fonttype":"none", "pdf.fonttype":42,
        "axes.spines.top":False, "axes.spines.right":False,
        "axes.labelcolor":"#31465C", "text.color":"#13243A",
        "xtick.color":"#52677C", "ytick.color":"#52677C"})
    beers = sorted(stability.beer.unique())
    rhos = [.001, .1, .25, .4]
    s = stability.set_index("profile_id")
    support = support.set_index("profile_id")
    values = np.full((len(beers), 10), np.nan)
    outside = np.zeros_like(values, dtype=bool)
    for i, beer in enumerate(beers):
        ids = [f"source_b{beer}", f"old80_b{beer}"]
        for policy in ["strict", "exceedance"]:
            for rho in rhos:
                case = cases[cases.policy.eq(policy) & cases.beer.eq(beer) & np.isclose(cases.rho, rho)].iloc[0]
                ids.append(case.profile_id if case.candidate_exists else None)
        for j, uid in enumerate(ids):
            if uid is not None and pd.notna(uid):
                values[i, j] = s.loc[uid, "joint_frequency"]
                outside[i, j] = not bool(support.loc[uid, "pca10_within_empirical_threshold"])
    fig = plt.figure(figsize=(15.0, 7.7))
    gs = fig.add_gridspec(1, 2, left=.06, right=.97, bottom=.18, top=.75,
                          width_ratios=[2.15, 1], wspace=.24)
    ax, bx = [fig.add_subplot(gs[0, j]) for j in range(2)]
    cmap = LinearSegmentedColormap.from_list("stability", ["#F2F5F8", "#A6D7D1", "#087F79"])
    cmap.set_bad("#E4E9EF")
    im = vector_heatmap(ax, values, cmap)
    for i in range(len(beers)):
        for j in range(10):
            text = "—" if np.isnan(values[i, j]) else f"{100*values[i, j]:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if values[i, j] > .70 else "#17324B")
            if outside[i, j]:
                ax.scatter(j+.34, i-.31, s=11, c="#DB794B", marker="s", edgecolors="none")
    ax.set_xticks(np.arange(10), ["Source", "Old k=80", "0.1%", "10%", "25%", "40%", "0.1%", "10%", "25%", "40%"], fontsize=9)
    ax.set_yticks(np.arange(len(beers)), [str(b) for b in beers])
    ax.set_ylabel("Source beer", labelpad=10)
    ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(beers), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    for boundary in [1.5, 5.5]:ax.axvline(boundary, color="white", lw=7)
    ax.text(.10, 1.06, "Comparators", transform=ax.transAxes, ha="center", fontweight="bold", fontsize=10.5)
    ax.text(.40, 1.06, "Strict concentration cap", transform=ax.transAxes, ha="center", fontweight="bold", fontsize=10.5)
    ax.text(.80, 1.06, "Exceedance-preserving cap", transform=ax.transAxes, ha="center", fontweight="bold", fontsize=10.5)
    ax.text(.5, -.105, "Required reduction in target distance", transform=ax.transAxes, ha="center", fontsize=10)
    cax = fig.add_axes([.065, .080, .25, .015])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, .5, 1])
    cb.solids.set_rasterized(False)
    cb.ax.set_xticklabels(["0%", "50%", "100%"]);cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9)
    plot = stability.merge(support[["pca10_distance_ratio"]], left_on="profile_id", right_index=True, validate="one_to_one")
    for kind, color, label in [("old80", "#334F6C", "Old k=80"), ("strict", "#E68B57", "Strict"), ("exceedance", "#138C83", "Exceedance")]:
        group = plot[plot.kind.eq(kind) if kind == "old80" else plot.policy.eq(kind)]
        bx.scatter(group.pca10_distance_ratio, group.joint_frequency*100, s=42,
                   c=color, edgecolors="white", linewidths=.6, alpha=.84, label=label)
    bx.axvline(1, color="#8396A8", ls=(0, (4, 3)), lw=1)
    bx.set_xlabel("PCA10 3NN distance / empirical threshold")
    bx.set_ylabel("Joint model-stability frequency (%)")
    bx.set_ylim(-4, 104);bx.set_yticks([0, 25, 50, 75, 100])
    bx.grid(axis="y", color="#E6ECF1", linewidth=.7)
    bx.set_axisbelow(True)
    bx.legend(frameon=False, loc="lower left", bbox_to_anchor=(-.03, 1.04), ncol=3,
              fontsize=9, handletextpad=.3, columnspacing=.6)
    fig.text(.055, .93, "Do fixed transfer profiles survive a change in fitted models?",
             fontsize=21, fontweight="bold")
    fig.text(.056, .879, "200 paired, style-stratified refits  /  six sensory families × 50 targets  /  candidate identities fixed in advance",
             fontsize=10.8, color="#5C7185")
    fig.text(.056, .824, "A   Tripel classification AND no increase in predicted stale-hops intensity", fontsize=11.3, fontweight="bold")
    fig.text(.706, .824, "B   Stability versus observed support", fontsize=11.3, fontweight="bold")
    fig.text(.35, .084, "Orange square: outside the empirical PCA10 support threshold.  —: no selected feasible profile.", fontsize=8.9, color="#5C7185")
    fig.text(.056, .038, "Frequencies measure conditional model stability. The support threshold is descriptive; neither quantity is a probability of physical or sensory intervention success.",
             fontsize=9.0, color="#5C7185")
    fig.savefig(HERE / "transfer_robustness_evidence.svg")
    fig.savefig(HERE / "transfer_robustness_evidence.pdf")
    fig.savefig(HERE / "transfer_robustness_evidence.png", dpi=180)
    plt.close(fig)

    # A separate column-width version keeps labels readable at the paper's
    # final 16.1 cm width; shrinking the wide screen figure would not do so.
    fig = plt.figure(figsize=(6.34, 7.45))
    ax = fig.add_axes([.095, .535, .88, .345])
    bx = fig.add_axes([.14, .105, .81, .265])
    im = vector_heatmap(ax, values, cmap)
    for i in range(len(beers)):
        for j in range(10):
            value = values[i, j]
            ax.text(j, i, "—" if np.isnan(value) else f"{100*value:.1f}",
                    ha="center", va="center", fontsize=7.4,
                    color="white" if value > .70 else "#17324B")
            if outside[i, j]:
                ax.scatter(j+.36, i-.32, s=6, c="#DB794B", marker="s", edgecolors="none")
    ax.set_xticks(np.arange(10), ["Source", "Old 80", "0.1%", "10%", "25%", "40%", "0.1%", "10%", "25%", "40%"], fontsize=7.0)
    ax.set_yticks(np.arange(len(beers)), [str(b) for b in beers], fontsize=7.5)
    ax.set_ylabel("Source beer", labelpad=5, fontsize=8)
    ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(beers), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    for boundary in [1.5, 5.5]:ax.axvline(boundary, color="white", lw=4)
    for x, text in [(.10, "Comparators"), (.40, "Strict cap"), (.80, "Exceedance cap")]:
        ax.text(x, 1.04, text, transform=ax.transAxes, ha="center", fontsize=8, fontweight="bold")
    ax.text(.5, -.10, "Required reduction in target distance", transform=ax.transAxes, ha="center", fontsize=7.8)
    cax = fig.add_axes([.10, .471, .25, .012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, .5, 1])
    cb.solids.set_rasterized(False)
    cb.ax.set_xticklabels(["0%", "50%", "100%"])
    cb.ax.tick_params(length=0, labelsize=7);cb.outline.set_visible(False)
    fig.text(.40, .468, "Orange square: outside PCA10 support", fontsize=7.1, color="#5C7185")
    for kind, color, label in [("old80", "#334F6C", "Old k=80"), ("strict", "#E68B57", "Strict"), ("exceedance", "#138C83", "Exceedance")]:
        group = plot[plot.kind.eq(kind) if kind == "old80" else plot.policy.eq(kind)]
        bx.scatter(group.pca10_distance_ratio, group.joint_frequency*100, s=27,
                   c=color, edgecolors="white", linewidths=.4, alpha=.84, label=label)
    bx.axvline(1, color="#8396A8", ls=(0, (4, 3)), lw=.8)
    bx.set_xlabel("PCA10 3NN distance / empirical threshold", fontsize=8)
    bx.set_ylabel("Joint model stability (%)", fontsize=8)
    bx.set_ylim(-4, 104);bx.set_yticks([0, 25, 50, 75, 100])
    bx.tick_params(labelsize=7.5)
    bx.grid(axis="y", color="#E6ECF1", linewidth=.6);bx.set_axisbelow(True)
    bx.legend(frameon=False, loc="lower left", bbox_to_anchor=(-.03, 1.01), ncol=3,
              fontsize=7.7, handletextpad=.3, columnspacing=.9)
    fig.text(.065, .965, "Fixed profiles under 200 paired model refits", fontsize=11, fontweight="bold")
    fig.text(.065, .934, "A   Tripel classification AND no increase in predicted stale hops", fontsize=8.4, fontweight="bold")
    fig.text(.065, .420, "B   Conditional stability and empirical support", fontsize=8.4, fontweight="bold")
    fig.text(.065, .025, "Model-stability frequencies and empirical support do not quantify physical intervention success.",
             fontsize=6.8, color="#5C7185")
    fig.savefig(HERE / "transfer_robustness_paper.svg")
    fig.savefig(HERE / "transfer_robustness_paper.pdf")
    fig.savefig(HERE / "transfer_robustness_paper.png", dpi=220)
    plt.close(fig)


def main():
    run = json.loads((HERE / "run_summary.json").read_text())
    assert run["completed"] and run["replicates"] == 200
    protocol = json.loads((HERE / "protocol.json").read_text())
    support_protocol = json.loads((HERE / "support_protocol.json").read_text())
    stability = pd.read_csv(HERE / "conditional_model_stability.csv")
    support = pd.read_csv(HERE / "support_diagnostics.csv")
    cases = pd.read_csv(HERE / "case_comparisons.csv")
    aggregate = pd.read_csv(HERE / "aggregate_stability.csv")
    joined = stability.merge(support[["profile_id", "pca10_distance_ratio", "pca10_within_empirical_threshold", "rank225_distance_ratio", "rank225_within_empirical_threshold", "pca_residual_within_empirical_threshold"]], on="profile_id", validate="one_to_one")
    joined.to_csv(HERE / "stability_and_support.csv", index=False)
    figure(stability, cases, support)
    manuscript_section(stability, cases, support, run, support_protocol)
    lines = ["# 稀疏迁移方案的条件模型稳定性与经验支持审计", "",
        f"本次审计固定了 {run['selected_cases']}/{run['selection_cases']} 个可选政策—源啤酒—距离要求组合，对应 {run['n_new_profiles']} 个政策—候选身份，其中有 {run['n_new_candidate_uids']} 个不同 candidate UID；加上原源样本与旧 k=80 对照，共 {run['n_profiles']} 个评估身份。同一化学配置可在两种政策下分别合格，两种政策身份不被误计为两次独立实验证据。200 组重抽样在候选冻结前确定。所有新方案只按主优化中的完整冻结模型合格性、最少记录字段变化数、同字段数下最小二次目标选择，稳定性结果没有反馈给寻优。", "",
        "## 1. 实验直接检验什么", "",
        "对每次按 22 种风格分层的啤酒重抽样，固定原 control rank map，重拟合 C=0.1、class_weight=balanced 的多项逻辑回归；同时按原 bundle 的六组 specifications 和固定 ensemble weights，重新拟合完整 50 目标感官模型。两类树各有 240 棵树且共同拟合全部 50 个目标，避免只拟合 stale_hops 改变分裂结构。化学预处理与目标均值/标准差在各次重抽样中重估。", "",
        "逐候选报告：Tripel argmax 频率、候选减源样本的 stale_hops 预测差不大于 10⁻⁸ 的频率，以及两个事件同时满足的联合频率。源样本相对自身的 stale 差按定义为零，源样本联合频率只反映分类波动。200 次结果的分辨率为 0.5 个百分点；这不是同等大小的统计精度，条件 Monte Carlo 标准误在频率 0.5 时约为 3.54 个百分点。", "",
        "这些频率刻画冻结模型流程在经验分层重抽样下的稳定性。它们没有包含新的酿造试验、外部感官验证或模型家族之外的误差，也没有改变候选是在原全部数据上优化产生这一事实。", "",
        "## 2. 按目标要求的公平比较", "",
        "| 政策 | 最低距离改善 | 有候选/11 | 平均变化字段数 | Tripel频率 | stale非增频率 | 联合频率 | 旧k=80同源联合频率 | 配对联合频率差 | 旧k=80满足该距离/11 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in aggregate.itertuples():
        label = "严格浓度不增" if r.policy == "strict" else "超标程度不增"
        lines.append(f"| {label} | {100*r.rho:g}% | {int(r.selected_sources)}/{int(r.total_sources)} | {r.mean_edits:.1f} | {100*r.mean_Tripel_frequency:.1f}% | {100*r.mean_stale_nonincrease_frequency:.1f}% | {100*r.mean_joint_frequency:.1f}% | {100*r.mean_old80_joint_frequency_same_sources:.1f}% | {100*r.mean_paired_joint_frequency_delta_old80:+.1f} pp | {int(r.old80_sources_meeting_required_rho)}/11 |")
    lines += ["", "所有新候选的均值只以确有合格候选的源样本为分母；旧 k=80 比较列使用相同源样本集合，避免缺失候选导致两列样本组成不同。完整表也保存全部 11 个源样本的旧方案基线。旧 k=80 表示旧预算方案，不保证每个入选旧配置恰好改变 80 个字段；其实际字段数另行保存。它也未必满足当前设定的最低距离改善，因此表中单列满足数量，完整 CSV 另给只比较双方均满足相同距离要求的配对均值。", "",
        "![迁移稳定性与支持审计](transfer_robustness_evidence.png)", "",
        "## 3. 逐配置稳定性与数据邻域", "",
        "| 配置 | beer | 实际字段数 | Tripel频率 | stale非增频率 | 联合频率 | PCA10支持距离比 | 225维支持距离比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in joined.sort_values(["kind", "policy", "beer", "candidate_id"]).itertuples():
        lines.append(f"| {r.profile_id} | {r.beer} | {r.raw_fields_changed} | {100*r.Tripel_frequency:.1f}% | {100*r.stale_nonincrease_frequency:.1f}% | {100*r.joint_frequency:.1f}% | {r.pca10_distance_ratio:.3f} | {r.rank225_distance_ratio:.3f} |")
    new = joined[joined.kind.eq("new")]
    old = joined[joined.kind.eq("old80")]
    lines += ["", f"不同政策—候选身份的联合频率范围为 {100*new.joint_frequency.min():.1f}%–{100*new.joint_frequency.max():.1f}%；旧 k=80 范围为 {100*old.joint_frequency.min():.1f}%–{100*old.joint_frequency.max():.1f}%。这里同一政策内相同候选只列一次，避免其满足多个距离要求时重复计入；跨政策复用同一化学谱不构成独立实验。", "",
        "## 4. 经验支持阈值的定义", "",
        f"在原 control 正态分数坐标上，以 250 个观测样本拟合不白化的 PCA10，保留总变异的 {100*support_protocol['total_explained_variance']:.2f}%。对 28 个真实 Tripel，排除自身后计算三个最近 Tripel 的平均欧氏距离，并除以 √10。固定阈值为这 28 个分数的 95% 经验分位数（higher），值为 {support_protocol['thresholds']['pca10']:.6f}。候选相对于全部 28 个参考的相同距离除以该阈值，得到表中的距离比；比值不大于 1 仅表示通过该项经验邻域检查。", "",
        f"固定 PCA 投影中的留自身邻居分数不等于重新拟合投影的留一预测，阈值没有概率覆盖保证。新配置中 {int(new.pca10_within_empirical_threshold.sum())}/{len(new)} 个位于该阈值内，旧 k=80 为 {int(old.pca10_within_empirical_threshold.sum())}/{len(old)}。另保存完整 225 维的同类近邻分数和 PCA 重构残差，识别低维投影隐藏的偏离；它们都没有用于重新挑选方案。每个配置在两种空间中的三个最近真实 beer ID 均可追溯。", "",
        "## 5. 源样本自身影响与保存证据", "",
        f"在源啤酒没有被抽入训练集的重抽样中，另行计算 OOB 条件频率。每个源对应 {int(stability.source_oob_replicates.min())}–{int(stability.source_oob_replicates.max())} 次此类重抽样。这个切片用于观察单个源样本的训练影响；原候选的产生已经使用过全部数据，因此它仍属于事后模型稳定性诊断。", "",
        "`bootstrap_indices.npz/.csv` 保存全部 200×250 抽样索引；`bootstrap_draws.npz` 保存每次、每个候选、每个模型家族的全部 50 维预测差，以及源样本预测、固定权重、分类概率和 logit 余量。`bootstrap_prediction_differences.csv` 提供逐次 stale 与分类事件的易读表；`conditional_model_stability.csv`、`case_comparisons.csv` 和 `stability_and_support.csv` 汇总配置、政策目标及邻域证据。`bootstrap_fit_records.csv` 保存全部 1,200 次感官拟合与 200 次分类拟合的执行记录。", "",
        "复现命令：", "", "```bash", "python run_transfer_robustness.py --prepare", "python run_transfer_robustness.py", "python summarize_transfer_robustness.py", "```", "",
        "论文中的角色：主优化证明的是冻结约束下的可行性与稀疏性界；本审计检验这些固定配置在替换拟合样本后是否仍保持预测判据，以及它们距离已观测目标风格有多远。两层结果共同报告，便于进一步选择应进入真实添加/去除与感官验证的配置。"]
    (HERE / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(HERE / "report.md")
    print(HERE / "manuscript_results.md")


if __name__ == "__main__":
    main()
