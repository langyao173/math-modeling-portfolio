"""Generate every current manuscript table from row-aligned frozen results."""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from v18_figures import ROOT,E,OUT,C,S,LABEL,COL,save,plt
P=ROOT/'paper'; U=ROOT/'evidence/capacity_update'
LABEL.update(base_reference='同底模静态',old_bound='原1pp更新',guard_six='六容量保护')
NAMES=['incumbent_R',S,'previous_tail','previous_FX','previous_E','base_reference','old_bound',C,'guard_six']
risk=pd.read_csv(E/'risk_extended.csv');tail=pd.read_csv(E/'joint/tail.csv');cap=pd.read_csv(E/'joint/capacity.csv');ci=pd.read_csv(E/'tail_intervals.csv')

def table(name,title,cols,header,rows,note=''):
    (P/name).write_text('\n'.join([r'\begin{table}[H]\centering\caption{'+title+'}',r'\fontsize{9bp}{12bp}\selectfont',r'\begin{tabularx}{\linewidth}{@{}'+cols+r'@{}}\toprule',header+r'\\\midrule',*rows,r'\bottomrule\end{tabularx}',r'\tabnote{'+note+'}' if note else '',r'\end{table}']))
def row(vals):return ' & '.join(map(str,vals))+r'\\'
def nm(n):return r'\textbf{本文方法}' if n==C else LABEL[n]
rows=[]
for n in NAMES[:-1]:
 z=risk[risk.candidate.eq(n)].set_index('phase');vals=[str(int(z.loc[p,k])) if k=='hits10' else f'{z.loc[p,k]:.4f}' for p in [0,1] for k in ['auc','ap','hits10']]
 rows.append(row([nm(n)]+([r'\textbf{'+v+'}' for v in vals] if n==C else vals)))
table('main_table.tex',r'10\%复核容量下的风险识别与事件覆盖','Yrrrrrr',r' & \multicolumn{3}{c}{提额月：2,907人 / 99事件} & \multicolumn{3}{c}{相邻月：2,677人 / 88事件}\\方法 & AUC & AP & 覆盖 & AUC & AP & 覆盖',rows,r'名额分别为291、268人；未知结局占用名额。粗体标识本文方法。同底模静态、原1pp更新与本文共用特征、融合及校准。')
groups=[('stable_05','账务金额近稳'),('repayment_state','还款与交易状态'),('flow_structure','支付与交易结构')]
rows=[]
for g,label in groups:
 for p in [0,1]:
  z=tail[tail.group.eq(g)&tail.phase.eq(p)].set_index('candidate')
  rows.append(row([label if p==0 else '',['提额月','相邻月'][p],int(z.loc[C,'n'])]+[f'{z.loc[n,"cvar95_pp"]:.3f}' for n in [C,S,'previous_tail']]))
 if g!='flow_structure':rows.append(r'\addlinespace[2bp]')
table('business_tail_table.tex',r'业务相近客户的极端评分波动（CVaR95）','Ylrrrr',r'业务口径 & 观察期 & 配对记录 & 本文 & 风险集成 S & 尾部优化 T',rows,r'单位：风险概率百分点。各业务组可重叠；完整分组定义、门控重合及配对区间见附录A.9。')
rows=[]
for n in NAMES:
 z=tail[tail.candidate.eq(n)&tail.group.eq('stable_05')].set_index('phase')
 rows.append(row([nm(n)]+[f'{z.loc[p,k]:.3f}' for p in [0,1] for k in ['p95_pp','cvar95_pp']]))
table('tail_table.tex','账务近稳客户的评分波动','Yrrrr',r' & \multicolumn{2}{c}{提额月：582对} & \multicolumn{2}{c}{相邻月：439对}\\方法 & P95 & CVaR95 & P95 & CVaR95',rows,r'单位为概率百分点。近稳评价使用余额、支付、支用、最低应还四个字段；更新门控还包含三项应收金额。')
rows=[]
for p in [0,1]:
 for n in NAMES:
  z=risk[risk.candidate.eq(n)&risk.phase.eq(p)].iloc[0]
  rows.append(row([['提额月','相邻月'][p]+' / '+LABEL[n]]+[f'{z[k]:.4f}' for k in ['ks','gini']]+[f'{z[k]:.5f}' for k in ['brier','logloss']]+[f'{z[k]:.3f}' for k in ['cal_intercept','cal_slope']]))
table('risk_details.tex','完整风险指标与概率校准诊断','Yrrrrrr',r'阶段 / 方法 & KS & Gini & Brier & Logloss & 截距 & 斜率',rows,'有效样本分别为2,841、2,623人。截距与斜率在已知结局上作描述性拟合，理想值为0、1，诊断结果不回写评分。')
rows=[]
for p in [0,1]:
 for n in NAMES:
  z=cap[cap.candidate.eq(n)&cap.phase.eq(p)].set_index('capacity')
  rows.append(row([['提额月','相邻月'][p]+' / '+LABEL[n]]+[int(z.loc[c,'hits']) for c in [.05,.08,.1,.12,.15,.2]]))
table('capacity_details.tex','两期六档固定容量的已观察事件覆盖','Yrrrrrr',r'阶段 / 方法 & 5\% & 8\% & 10\% & 12\% & 15\% & 20\%',rows,r'各容量名额向上取整。主版本按10\%边界生成一次评分，再评价六档容量；六容量版本一次保护全部六个边界。')
d=pd.read_parquet(E/'joint/predictions.parquet');edges=pd.read_parquet(E/'joint/edges.parquet')
def cv(v):
 v=np.sort(v)[::-1];m=.05*len(v);k=int(m);return (v[:k].sum()+(m-k)*(v[k] if k<len(v) else 0))/m
rows=[]
for p in [0,1]:
 for g,label in [('stable_01','1'),('stable_05','5'),('stable_10','10')]:
  q=edges[edges.phase_after.eq(p)&edges[g]];v=100*abs(d.loc[q.row_after,C].to_numpy()-d.loc[q.row_before,C].to_numpy())
  rows.append(row([['提额月','相邻月'][p]+' / '+label+r'\%',len(v),f'{v.mean():.3f}',f'{np.quantile(v,.95):.3f}',f'{cv(v):.3f}',f'{100*(v>2).mean():.2f}']))
table('sensitivity_table.tex','不同四字段近稳门槛下的表现','Yrrrrr',r'阶段 / 阈值 & 对数 & 均值 & P95 & CVaR95 & 超过2 pp\%',rows,'幅度均为概率百分点；仅改变评价分组，最终评分及更新规则保持冻结。')
rows=[]
for g,label in groups:
 z=tail[tail.candidate.eq(C)&tail.group.eq(g)].set_index('phase')
 rows.append(row([label]+[v for p in [0,1] for v in [int(z.loc[p,'n']),int(z.loc[p,'seven_n']),f'{100*z.loc[p,"seven_overlap"]:.2f}'+r'\%']]))
table('business_groups.tex','业务分组与七字段更新条件的重合','Yrrrrrr',r' & \multicolumn{3}{c}{提额月} & \multicolumn{3}{c}{相邻月}\\口径 & 对数 & 满足条件 & 重合率 & 对数 & 满足条件 & 重合率',rows,'满足条件且当期预测超出允许区间时才发生候选分修正；容量保护再检查入选边界。')
for g,label in groups:
 rows=[]
 for p in [0,1]:
  for n in ['incumbent_R',S,'previous_tail','previous_FX','previous_E','base_reference','old_bound']:
   z=ci[ci.group.eq(g)&ci.phase.eq(p)&ci.comparator.eq(n)].iloc[0]
   rows.append(row([['提额月','相邻月'][p]+' / '+LABEL[n],f'{z.difference:.3f}',f'[{z.ci_low:.3f}, {z.ci_high:.3f}]',f'[{z.sim_low:.3f}, {z.sim_high:.3f}]']))
 table('intervals_'+g+'.tex',label+'的CVaR95配对差与区间','Yrrr',r'阶段 / 对手 & 本文减对手 & 逐项95\%区间 & 同时95\%区间',rows,r'单位为概率百分点。2,000次客户共同重采样；同时区间覆盖三客群、两期、七对手共42项，临界值3.8839。')
cov=pd.read_csv(E/'coverage_intervals.csv');front=pd.read_csv(E/'operating/joint_frontier.csv');rows=[]
for n in NAMES:
 z=front[front.model.eq(n)&front.capacity.eq(.1)].iloc[0]
 rows.append(row([nm(n)]+[int(z[k]) for k in ['unique_clients','unique_events','repeat_clients','new_second']]))
table('continuity_details.tex','两期559个名额的客户接续与事件覆盖','Yrrrr',r'方法 & 独立客户 & 独立事件客户 & 连续入选 & 相邻月新增',rows,'连续入选客户需要核查新账务；原材料、处理意见及责任人随原案接续。')
rows=[row([LABEL[z.comparator],int(z.difference),f'[{z.ci_low:.0f}, {z.ci_high:.0f}]']) for z in cov.itertuples()]
table('coverage_intervals.tex','两期事件窗口覆盖差的配对区间','Yrr',r'对手 & 本文减对手 & 逐项95\%区间',rows,r'共559个名额，本文覆盖79个事件窗口。名单固定，按客户贡献重采样2,000次；同一客户在两期共享权重。窗口数可重复，独立事件客户另表列示。')
cond=pd.read_csv(U/'conditional_metrics.csv');rows=[]
for n in ['base_reference','old_bound',C]:
 z=cond[cond.group.eq('original_clipped_106')&cond.method.eq(n)].iloc[0]
 rows.append(row([nm(n),f'{z.mean_prediction*100:.2f}'+r'\%',f'{z.auc:.4f}',f'{z.ap:.4f}',f'{z.brier:.5f}',f'{z.logloss:.5f}',f'{z.offset_intercept:.3f}',f'{z.slope:.3f}']))
table('conditional_metrics.tex','原106次受限更新的条件评价','Yrrrrrrr',r'方法 & 均值 & AUC & AP & Brier & Logloss & 偏移截距 & 斜率',rows,'106次更新、98名客户；105次结果可判、12个事件。偏移截距将预测logit作为offset，斜率来自截距与斜率共同拟合。')
z=pd.read_csv(U/'conditional_intervals.csv');rows=[]
for r in z.itertuples():
 if r.metric in ['brier','logloss','auc','ap']:
  rows.append(row([LABEL[r.comparator]+' / '+r.metric,f'{r.difference:.5f}',f'[{r.ci95_low:.5f}, {r.ci95_high:.5f}]']))
table('conditional_intervals.tex','固定受限子组的配对变化','Yrr',r'对手 / 指标 & 本文减对手 & 逐项95\%区间',rows,'以原方案的106条记录固定子组，客户成组重采样2,000次；这些区间属于条件比较，不与尾部42项共同校正。')
sev=pd.read_csv(U/'serious_endpoints.csv');rows=[]
for r in sev.itertuples():
 rows.append(row([str(r.horizon)+'个月 / '+LABEL[r.method],r.known,r.observed_event_windows,r.event_customers,r.selected_observed_events,r.selected_event_customers]))
table('serious_endpoints.tex','延长随访的DPD30+结果','Yrrrrr',r'期限 / 方法 & 结局已知 & 事件窗口 & 事件客户 & 入选窗口 & 入选客户',rows,'沿用原559个名额和冻结评分；遇到DPD30+即记事件，未观察到事件且不足全窗口者记未知。延长终点不重新解释为三个月PD。')
sup=pd.read_parquet(ROOT/'research_extension/results/historical/support_per_row.parquet').merge(d[['sample_id','y',C,S]],on='sample_id',validate='one_to_one');rows=[];out=[]
for p in [0,1]:
 q=sup[sup.phase.eq(p)]
 for g,label,mask in [('balance_gt_ref','余额超过参考额度',q.balance_over_reference.gt(1)),('high_excursion','高越界比例',q.high_excursion),('other','其余输入',~q.high_excursion)]:
  z=q[mask];k=z[z.y.notna()];a=roc_auc_score(k.y,k[C]);b=roc_auc_score(k.y,k[S]);rows.append(row([['提额月','相邻月'][p]+' / '+label,len(z),len(k),int(k.y.sum()),f'{a:.4f}',f'{b:.4f}']));out.append(dict(phase=p,group=g,n=len(z),known=len(k),events=int(k.y.sum()),guard10_auc=a,strong_S_auc=b))
table('support_metrics.tex','参考尺度输入检查与分层风险识别','Yrrrrr',r'阶段 / 分组 & 人数 & 已知 & 事件 & 本文AUC & S的AUC',rows,r'396个可变特征逐维1\%--99\%区间；高越界比例以训练行P95为阈值。“其余输入”为高越界组补集；余额超过参考额度可与后二组重合。')
pd.DataFrame(out).to_csv(E/'support_metrics.csv',index=False)
dev=pd.read_csv(U/'development/comparison.csv');rows=[]
for r in dev.itertuples():
 label='六档同时保护' if r.candidate=='guard_six' else str(int(r.candidate[5:]))+r'\%边界'
 rows.append(row([label,f'{r.mean_auc:.4f}',f'{r.mean_ap:.4f}',int(r.hits10),f'{r.worst_cvar:.3f}',f'{r.worst_p95:.3f}']))
table('development_comparison.tex','开发集上的容量保护方式','Yrrrrr',r'保护方式 & 平均AUC & 平均AP & 10\%覆盖 & 较差期CVaR95 & 较差期P95',rows,r'提额月6,411条、相邻月6,308条交叉拟合更新。各方案保留自身指定边界；主业务容量预定为10\%。幅度单位为概率百分点。')
# Reliability curve follows the original restrained layout.
rel=pd.read_csv(E/'reliability.csv');f,axs=plt.subplots(1,2,figsize=(6.25,2.65));f.subplots_adjust(left=.1,right=.96,bottom=.25,top=.84,wspace=.35)
for p,ax in enumerate(axs):
 ax.plot([0,.28],[0,.28],ls='--',lw=.7,color='#B3BBAF')
 for n,m in [(S,'s'),('previous_FX','^'),(C,'o')]:
  z=rel[rel.model.eq(n)&rel.phase.eq(p)];ax.plot(z.mean_p,z.event_rate,color=COL[n],lw=1.3 if n==C else .8,marker=m,ms=3.4,label=LABEL[n])
  if n==C:ax.errorbar(z.mean_p,z.event_rate,yerr=[z.event_rate-z.ci_low,z.ci_high-z.event_rate],fmt='none',ecolor=COL[n],elinewidth=.6,capsize=1.7,alpha=.65)
 ax.set(xlim=(0,.25),ylim=(0,.3),xticks=[0,.1,.2],yticks=[0,.1,.2,.3],xlabel='分组平均风险评分',ylabel='已观察事件率' if p==0 else '');ax.set_title('a  提额月' if p==0 else 'b  相邻月',loc='left',weight='bold');ax.grid(alpha=.7);ax.set_axisbelow(True)
axs[0].legend(frameon=False,fontsize=7.5,loc='upper left');f.text(.07,.05,'按评分排序分成10个等人数组；本文误差线为组内事件率的Wilson 95%区间。',fontsize=7.8,color='#758073');save(f,'v18_reliability')
print('All current tables and reliability figure generated.')
