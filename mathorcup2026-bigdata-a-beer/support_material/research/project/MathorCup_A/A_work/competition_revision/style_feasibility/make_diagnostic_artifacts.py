from pathlib import Path
import json
import numpy as np
import pandas as pd
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mathorcup_style_feasibility_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
HERE=Path(__file__).resolve().parent
s=pd.read_csv(HERE/'joint_budget_summary.csv')
f=pd.read_csv(HERE/'joint_first_tested_budgets.csv')
cv=json.loads((HERE/'classifier_cv_summary.json').read_text())
audit=json.loads((HERE/'audit_summary.json').read_text())
raw_input_checks=audit['input_hash_checks']
if audit['old_input_files_unchanged']:
 input_audit_text=f"{len(raw_input_checks)} 项旧输入原始哈希全部一致。"
elif audit.get('input_scientific_content_unchanged',False):
 unchanged_count=sum(raw_input_checks.values())
 changed_count=len(raw_input_checks)-unchanged_count
 input_audit_text=f"输入审计保留了 {changed_count} 项原始哈希不一致：{unchanged_count} 项原哈希一致，另外 {changed_count} 项经完整核验为已记录的类别元数据修订；除这些修订外，科学数值与模型保持不变。"
else:
 input_audit_text="输入文件的科学内容一致性核查尚未全部通过，详见 audit_summary.json 中的原始哈希结果和逐项审计证据。"
summary=json.loads((HERE/'summary.json').read_text())
joint=json.loads((HERE/'joint_summary.json').read_text())
figdir=HERE/'figures';figdir.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
x=np.arange(len(s));labels=s.budget.astype(str).tolist()
fig,(a,b)=plt.subplots(1,2,figsize=(12.4,4.6),gridspec_kw={'width_ratios':[1.15,1.0]})
a.plot(x,s.old_distance_objective_mean_distance_reduction,'o-',color='#546477',lw=1.6,ms=4,label='Original distance objective')
a.plot(x,s.class_only_all_11_mean_distance_reduction,'s--',color='#bb5b3b',lw=1.5,ms=4,label='Classifier constraint')
a.plot(x,s.all_11_mean_distance_reduction,'D-',color='#006d77',lw=2.1,ms=4,label='Joint criterion')
a.axhline(0,color='#777777',lw=.7)
a.set_xticks(x,labels);a.set_xlabel('Permitted direct fields: tested budgets')
a.set_ylabel('Mean target-distance reduction, all 11 sources')
a.yaxis.set_major_formatter(PercentFormatter(1));a.set_ylim(-.12,.85)
a.grid(axis='y',alpha=.2);a.set_title('(a) Distance improvement and its cost',loc='left',fontweight='bold',fontsize=11)
a.legend(loc='upper left',fontsize=8.5,frameon=False)
a.text(.045,.50,'At 40 fields, classifier-only\nmean distance increases 3.55%',transform=a.transAxes,fontsize=8.5,color='#8f432a')
mat=np.vstack([s.old_distance_objective_tripel,s.linear_feasible_m0,s.class_only_accepted,s.joint_accepted]).astype(int)
b.imshow(mat,vmin=0,vmax=11,cmap='Blues',aspect='auto')
b.set_xticks(x,labels);b.set_yticks(np.arange(4),['Original: Tripel','Linear feasible','Class + checks','Joint criterion'])
b.tick_params(axis='both',length=0);b.set_xlabel('Permitted direct fields: tested budgets')
b.set_title('(b) Qualifying source count, out of 11',loc='left',fontweight='bold',fontsize=11)
for iy in range(mat.shape[0]):
 for ix in range(mat.shape[1]):b.text(ix,iy,str(mat[iy,ix]),ha='center',va='center',color='white' if mat[iy,ix]>=7 else '#22313d',fontsize=9)
for spine in b.spines.values():spine.set_visible(False)
fig.text(.02,.025,'Joint criterion adds positive distance improvement (> 1e-8) after observing the classifier-only conflict. Unqualified sources remain unchanged.',fontsize=8.5,color='#45515b')
fig.subplots_adjust(left=.065,right=.99,bottom=.19,top=.88,wspace=.40)
for ext in ['png','svg','pdf']:fig.savefig(figdir/f'distance_classification_joint.{ext}',dpi=240,bbox_inches='tight')
plt.close(fig)

def link(name):return f'[{name}](sandbox:{HERE/name})'
def pct(v):return '—' if pd.isna(v) else f'{v*100:.2f}%'
def table(data,cols,heads,formats=None):
 out=['|'+'|'.join(heads)+'|','|'+'|'.join(['---']*len(cols))+'|']
 for _,r in data.iterrows():
  vals=[]
  for c in cols:
   v=r[c];vals.append((formats[c](v) if formats and c in formats else ('—' if pd.isna(v) else str(int(v)) if isinstance(v,(int,np.integer,float,np.floating)) and float(v).is_integer() else str(v))))
  out.append('|'+'|'.join(vals)+'|')
 return '\n'.join(out)
comparison_table=table(s,['budget','linear_feasible_m0','class_only_accepted','class_only_all_11_mean_distance_reduction','joint_accepted','all_11_mean_distance_reduction','accepted_conditional_mean_distance_reduction','old_distance_objective_tripel','old_distance_objective_mean_distance_reduction'],['预算k','线性可行','分类+复核通过','分类条件：全11改善','联合合格','联合：全11改善','联合：条件均值','原方案Tripel数','原方案全11改善'],formats={c:pct for c in ['class_only_all_11_mean_distance_reduction','all_11_mean_distance_reduction','accepted_conditional_mean_distance_reduction','old_distance_objective_mean_distance_reduction']})
first_table=table(f,['beer','first_tested_linear_feasible_budget','first_tested_complete_acceptance_budget','first_tested_joint_budget','joint_first_distance_reduction','joint_first_actual_raw_fields','joint_first_target_probability'],['beer','首次线性可行k','首次分类+复核k','首次联合k','联合首次距离改善','实际字段变化数','Tripel概率输出'],formats={'joint_first_distance_reduction':pct,'joint_first_target_probability':pct})
readme=rf'''# 固定风格分类区域与目标接近：新增探索诊断

## 1. 已得到的实质结论

**目标距离变小、被固定分类器判为 Tripel、以及满足全部代理约束是不同事件。** 在原先前向支持集上添加 Tripel 分类半空间后，40 个允许直接字段即可让 11 个源样本全部通过分类和约束复核，但平均目标距离反而增加 3.55%。因此不能把“分类成功 11/11”单独写成风格迁移成功。

观察到这一冲突后，我们对**已经生成并检查过的全部路径**增加 `target_distance_reduction > 1e-8` 的联合筛选。没有增加模型搜索、调参、QP 或路径搜索。联合结果为：k=20 时 1/11 合格，全 11 平均改善 1.33%；k=40 时 7/11、14.37%；k=60 时 10/11、30.19%；k=80 首次达到 11/11，平均改善 43.09%。这里的“首次”仅指所检验的固定嵌套支持集及预算网格。

原先所有结果与模型保持不变。分类条件结果保留在无 `joint_` 前缀的文件中，联合筛选保存在 `joint_` 文件中。该诊断的作用是区分优化目标、判别条件和候选适用范围。

## 2. 固定模型与线性可行域

沿用 225 维原控制层秩映射、Tripel 均值与精度矩阵、线性 stale 代理、四项化学上界、五项辅助字段重建规则，以及冻结的 Q1 感官预测器。直接字段预算不包括四个估计汇总字段、能量恒等式字段和内部标准；未选直接字段及内标逐值保持，包括原缺失值。

固定分类器使用 `LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)`，在同一 225 维原始秩空间、全部原观测风格上拟合一次。其在原预算输出上的全部分类一致，Tripel 概率的最大差异为 {summary['classifier_max_probability_difference']:.3g}。这只是确认本诊断和原预算诊断使用相同分类规则。

设源样本秩向量为 $z_0$，所选支持集为 $S_k$，调整为 $\delta$，Tripel 为 $T$。原目标保持为

$$F(\delta)=\frac12(z_0+\delta-\mu_T)^\top P_T(z_0+\delta-\mu_T)+\frac{{0.2}}2\|\delta\|_2^2.$$

原线性条件为

$$\delta_{{S_k^c}}=0,\quad \ell-z_0\le\delta\le u-z_0,\quad a^\top\delta\le-m,$$

其中 $u$ 已含四项化学上界：原来可观测的字段不增加，原来缺失者上界设为对应阈值。按原脚本，$m\in\{{0,0.025,0.05,0.1,0.2,0.4,0.8\}}$。

多项逻辑分类的 Tripel 区域精确表示为 21 个半空间：

$$ (w_T-w_j)^\top(z_0+\delta)+(b_T-b_j)\ge10^{{-7}},\qquad j\ne T.$$

$10^{{-7}}$ 用于避免并列最大值，并不意味着分类稳定或高置信。后续接受检查允许 $10^{{-8}}$ 的数值误差；实际接受候选的最小两类 logit 差仍为正。

### LP、QP 与完整候选检查

对 11 个样本、11 个预算（3、5、10、20、40、60、80、100、140、180、225）及 7 个余量共求解 847 个线性可行性问题。这里 225 为全维参照，其余均为原 `selected_supports.csv` 的前向支持集，不重新选字段。

首先用 `linprog` 检查半空间与箱约束是否相交。对不可行项，另求最小统一松弛量，并保存非负乘子 $\lambda$。若

$$\min_{{\ell_S-z_{{0,S}}\le d\le u_S-z_{{0,S}}}}\lambda^\top(Ad-b)>0,\quad\lambda\ge0,$$

则原半空间不存在可行点；左侧可按各坐标斜率符号在箱端点直接计算。327 项不可行结果均有正的数值分离下界，最小为 {audit['minimum_independent_separation_lower_bound']:.9f}，独立重算通过。它们是给定有限精度模型的数值证据，不是符号算术证明。

520 个可行问题随后以 LP 可行点初始化原凸 QP，使用原目标、`SLSQP(maxiter=600, ftol=1e-10)`，全部通过残差检查。沿原 22 个 $\alpha$ 网格点形成 $z_0+\alpha\delta$；对回译前越界点记载失败，合格点回到化学量纲、恢复未选字段、执行源值锚定的辅助重建及能量恒等式。再调用冻结 Q1，逐一检查化学约束、线性 stale 非增、非线性 stale 非增与 Tripel 判别条件。QP 的 $m$ 是搜索余量；路径候选最终按原规则检查非增，而不是声称每个缩放候选仍有完整的 $m$ 余量。

共得到 10,152 个回译路径，另保存 1,288 个回译前越界跳过记录；939 条路径通过分类与完整约束检查。全部失败和残差保留在对应 CSV 中。

## 3. 分类条件、联合条件与原方案的比较

归一化目标距离及相对改善定义为

$$D_T(z)=\sqrt{{(z-\mu_T)^\top P_T(z-\mu_T)/225}},\qquad R(z)=1-\frac{{D_T(z)}}{{D_T(z_0)}}.$$

新增联合条件是在已有完整通过集合中要求 $R(z)>10^{{-8}}$。每个预算允许继承所有更小嵌套支持集的合格候选，然后按原 $F$ 最小选择；没有合格候选时原样退回，$R=0$。因此下表的“全11改善”始终以 11 个源样本为分母；“条件均值”只在合格样本中计算，两者不能混用。

{comparison_table}

![距离、分类与联合合格对照](sandbox:{figdir/'distance_classification_joint.png'})

40 个字段下，分类条件的 11/11 包括四个目标距离恶化样本，其中 beer 124 的距离增加 94.20%。这解释了为什么分类数字好看仍不足以构成有用迁移。80 个字段下，联合方案才在当前网格中首次使全部 11 个样本同时进入判别区、满足两种 stale 代理非增并更接近目标中心；最弱样本的距离改善为 9.89%。

k=20 时，7 个样本线性可行，只有 5 个通过完整检查。beer 90 在可行路径上的冻结 Q1 stale 最低仍增加约 0.03582；beer 139 最低仍增加约 0.00942。它们在这批有限候选中被拒绝。此结果不证明其整个非线性可行域为空。

k=140 与 180 的联合距离均值略高于原无分类条件的最终方案，来源于约束改变后产生的有限路径及非线性筛选差异。不能由此推导“增加约束改善凸问题的全局最优值”：这里比较的是通过冻结非线性复核的有限候选集。

## 4. 逐样本首次预算与候选

{first_table}

预算是允许修改的直接字段上界；“实际字段变化数”按原值和输出值比较，包含原缺失字段变为候选数值，另保存秩坐标变化数量。较大预算复用较小预算方案时，实际变化数可以小于预算。所有“首次”均为本次已测网格中的首次，不是任意字段集合下的全局最小值，也不是实际酿造工艺的最少操作。

## 5. 固定 C=0.1 分类器的独立误差参照

为避免借用其他分类器结果，我们对本诊断使用的参数补做一次 3 折、22 风格分层交叉验证（最小风格样本量 3，随机种子 20261023）。每折只用训练行重拟合原 `ConditionalCopula` 和经验秩映射、中位数，再训练相同 C=0.1 分类器。无超参数搜索；全样本冻结分类器未改动。

|指标|汇总 OOF 结果|计数解释|
|---|---:|---|
|Accuracy|{cv['accuracy']:.2%}|136 / 250|
|Balanced accuracy|{cv['balanced_accuracy']:.4%}|22 类召回率的等权平均|
|Tripel precision|{cv['Tripel_precision']:.4%}|17 / 29 个预测 Tripel 正确|
|Tripel recall|{cv['Tripel_recall']:.4%}|17 / 28 个真实 Tripel 被识别|
|Tripel 假阳性|12|其他风格被判为 Tripel|
|Tripel 假阴性|11|真实 Tripel 被判为其他风格|

这表示该分类器在当前原观测样本中的区分能力有限。进入全样本模型的 Tripel 区域是一个明确、可计算的模型条件，不能认证真实风格；上述 OOF precision 也不能直接当作经过优化的合成样本真实属于 Tripel 的概率。小幅正 logit 边际、原数据中的误分及缺少新感官试验，共同限定了这个诊断的解释范围。

## 6. 独立核查与交付索引

847 个 LP 可行见证或分离证据全部独立重算；另以 `highs-ipm` 重新检查所有 121 个 $m=0$ 问题，判定完全一致。121 个分类条件最终输出重新加载后，逐项核对化学字段、秩边界、分类分数、冻结 Q1 stale、辅助字段重建、能量恒等式和冻结字段，全部通过。联合候选从上述已审计候选中选取，最小目标值筛选逐行核查。{input_audit_text}

- 可直接插入正文的精简段落：{link('manuscript_section.md')}。
- 完整比较：{link('joint_budget_summary.csv')}、{link('budget_summary.csv')}。
- 逐样本首次预算：{link('joint_first_tested_budgets.csv')}。
- 各预算联合最终化学候选：{link('joint_selected_profiles.csv')}；感官预测：{link('joint_selected_sensory_predictions.csv')}；化学证书：{link('joint_chemical_certificate.csv')}。
- 逐样本首次联合候选：{link('joint_first_accepted_profiles.csv')} 与 {link('joint_first_accepted_diagnostics.csv')}。
- 分类条件反例与原方案对照：{link('selected_profiles.csv')}、{link('comparison_with_distance_objective.csv')}。
- LP/失败/残差：{link('linear_feasibility.csv')}、{link('linear_witnesses_and_certificates.npz')}、{link('qp_attempts.csv')}、{link('path_grid_skips.csv')}、{link('candidate_checks.csv')}。
- 独立核查：{link('audit_summary.json')}、{link('audit_lp_witnesses.csv')}、{link('audit_independent_lp_method.csv')}、{link('audit_selected_candidates.csv')}、{link('joint_selection_audit.csv')}。
- 分类器参照：{link('classifier_cv_summary.json')}、{link('classifier_cv_confusion.csv')}、{link('classifier_cv_Tripel_confusion.csv')}、{link('classifier_cv_oof_predictions.csv')}、{link('classifier_cv_splits.csv')}。
- 事前固定的本次诊断协议：{link('protocol.json')}；看到分类与距离冲突后新增的联合筛选记录：{link('joint_protocol.json')}；分类器 CV 固定协议：{link('classifier_cv_protocol.json')}。

本诊断不增加新的风格生成模型，不替换旧结果，不证明化学因果、实际酿造可行性或真实感官风格转换。它补足的是问题四中“接近目标”与“进入判别区”之间的决策证据。
'''
(HERE/'diagnostic_report.md').write_text(readme)
selected=s[s.budget.isin([10,20,40,60,80,140,225])]
short_table=table(selected,['budget','class_only_accepted','class_only_all_11_mean_distance_reduction','joint_accepted','all_11_mean_distance_reduction','accepted_conditional_mean_distance_reduction'],['预算k','分类条件合格数','分类条件全11改善','联合合格数','联合全11改善','联合条件均值'],formats={c:pct for c in ['class_only_all_11_mean_distance_reduction','all_11_mean_distance_reduction','accepted_conditional_mean_distance_reduction']})
section=rf'''## 目标接近与风格判别的联合检验

前述目标距离优化衡量候选化学轮廓对 Tripel 中心的接近程度。为检验“接近中心”是否同时意味着“进入风格判别区”，我们保持 225 维秩映射、目标均值与协方差、已选前向支持集、四化学上界以及冻结的感官预测器不变，增加固定分类器约束。这是对原预算结果的新增探索诊断。

设源向量为 $z_0$，支持集为 $S_k$，$P_T=\Sigma_T^{{-1}}$。仍最小化

$$F(\delta)=\tfrac12(z_0+\delta-\mu_T)^\top P_T(z_0+\delta-\mu_T)+0.1\|\delta\|_2^2,$$

并保持 $\delta_{{S_k^c}}=0$、原秩边界与四化学上界、$a^\top\delta\le-m$。对全样本拟合的多项逻辑分类器（$C=0.1$、类别平衡权重），Tripel 为 $T$，其判别条件可准确写为

$$ (w_T-w_j)^\top(z_0+\delta)+(b_T-b_j)\ge10^{{-7}},\qquad j\ne T.$$

因此可先通过线性规划判定固定支持集的可行性，再求原凸二次目标。对 11 个样本、11 个预算和原 7 个余量共得到 847 个 LP：520 个可行，327 个不可行；不可行项均保存正的数值分离下界。可行 QP 沿原步长网格回译，未选化学字段及内标保持原值，辅助字段按源值锚定重建，再由冻结 Q1 检查 stale 非增。所有候选保留成功与失败记录。

分类条件单独成立仍可能使目标距离增加。例如 $k=40$ 时，11 个样本均通过分类及约束复核，但平均目标距离增加 **3.55%**。观察到该冲突后，我们在已经检查过的路径中增加联合条件

$$R(z)=1-\frac{{D_T(z)}}{{D_T(z_0)}}>10^{{-8}},\qquad D_T(z)=\sqrt{{(z-\mu_T)^\top P_T(z-\mu_T)/225}}.$$

联合合格需同时满足四化学约束、线性及冻结 Q1 的 stale 非增、Tripel 判别与正距离改善。此条件是**看到冲突后增加的探索性筛选**，没有新训练或搜索。各预算继承更小支持集的合格候选并按原 $F$ 最小选择；无合格者原样退回，改善计为零。

{short_table}

表中全11均值始终以全部源样本为分母，条件均值仅使用合格样本。当前网格中，$k=80$ 首次使全部 11 个样本联合合格，平均距离改善 **43.09%**，最弱样本改善 **9.89%**；原距离目标在同预算下平均改善 46.12%，但只有 1 个样本被判为 Tripel。这体现了目标接近与判别条件之间的取舍。分类条件反例和全部回退均保留，未把 11/11 分类直接视为迁移成功。

为给分类硬约束提供同模型的误差参照，另做一次 3 折风格分层验证，每折仅以训练行重拟合原控制层秩映射并训练相同参数分类器。汇总 250 个 OOF 预测的 accuracy 为 **54.40%**，balanced accuracy 为 **50.66%**；Tripel precision 为 **58.62%（17/29）**、recall 为 **60.71%（17/28）**。全样本冻结分类器未改变。这说明“进入 Tripel 区域”只是一项有误差的模型条件，不能认证实际感官风格；OOF precision 也不能转用为优化合成样本的真实风格概率。

全部 LP 见证/分离证据已独立重算，121 个零余量 LP 用另一算法复解一致，最终候选的字段冻结、化学上界、两种 stale 代理和重建公式均经复核。“首次预算”仅针对已测嵌套支持集与有限路径，不是全局最少字段证明；真正的风格与感官非恶化仍需新样品试验验证。

<!-- 数据来源：style_feasibility/joint_budget_summary.csv、classifier_cv_summary.json、audit_summary.json。作图：figures/distance_classification_joint.svg。 -->
'''
(HERE/'manuscript_section.md').write_text(section)
print(json.dumps({'report_characters':len(readme),'manuscript_characters':len(section),'figure':str(figdir/'distance_classification_joint.png'),'joint_first_all_budget':joint['first_tested_budget_with_all_11']},ensure_ascii=False))
