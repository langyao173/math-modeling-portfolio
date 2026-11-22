"""Publication figures read executed CSV results; no simulated performance data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
OUT=HERE/'figures'
OUT.mkdir(exist_ok=True)
fm.fontManager.addfont(str(WORK.parent/'assets/fonts/NotoSansCJKsc-Regular.otf'))
plt.rcParams.update({'font.family':'Noto Sans CJK SC','font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#263746',
    'xtick.color':'#374754','ytick.color':'#374754','axes.unicode_minus':False,
    'savefig.facecolor':'white','svg.fonttype':'none'})
BLUE='#245c8e';TEAL='#187b72';ORANGE='#c27a27';GRAY='#8b98a1';PURPLE='#80608e'
names=json.loads((WORK/'figures/sensory_names_zh.json').read_text())

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=240,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)

def framework():
    fig,ax=plt.subplots(figsize=(9,4.35));ax.set(xlim=(0,9),ylim=(0,4.5));ax.axis('off')
    boxes=[(.2,2.85,2.05,1.15,'数据与尺度','250 款 · 231 字段\n按 beer 对齐 · 保留缺失'),
           (3.05,3.0,2.35,1.1,'问题一  感官预测','折内处理与集成\n连续值 + 相对 0–6 档'),
           (6.25,3.0,2.45,1.1,'问题二  关联解释','231 节点共变图\nTop 3 + 重采样 + 模块'),
           (3.05,.65,2.35,1.25,'问题三  条件生成','保留重复值的 Copula\n22 风格 × 100 虚拟谱'),
           (6.25,.65,2.45,1.25,'问题三  风格迁移','目标距离 + 变量预算\n化学与感官代理约束')]
    for x,y,w,h,title,body in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.07',fc='#f1f5f8',ec='#6c8899',lw=1))
        ax.text(x+w/2,y+h-.28,title,ha='center',va='center',fontsize=11,weight='bold',color=BLUE)
        ax.text(x+w/2,y+.42,body,ha='center',va='center',fontsize=10,linespacing=1.5)
    for a,b in [((2.3,3.45),(3,3.45)),((5.45,3.55),(6.2,3.55)),((1.2,2.8),(3,1.3)),((5.45,1.3),(6.2,1.3)),((7.45,2.95),(7.45,1.95)),((4.2,2.95),(4.2,1.95))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#647b8c',lw=1.2))
    ax.text(.2,.15,'检验：折外误差与缺失压力  ·  稳定性与化学锚点  ·  分布诊断与逐样本约束复核',fontsize=10,color='#4d606e')
    save(fig,'01_framework')

def prediction():
    base=pd.read_csv(WORK/'q1/dimension_metrics.csv')
    rows=[]
    for model,label in [('nested_ensemble','六模型集成'),('extra_trees','Extra Trees')]:
        d=base[base.model==model]
        rows.append((label,d.r2.mean(),d[d.target.str.startswith(('A_','F_'))].r2.mean()))
    fusion=pd.read_csv(WORK/'nature_upgrade/fusion/dimension_metrics.csv')
    # The exact model-key is obtained from the persisted fusion table.
    if 'model' in fusion:
        candidates=fusion.model.unique().tolist()
        key='fusion8_shrink'
        assert key in candidates
        fusion=fusion[fusion.model==key]
    rows.append(('八分支研究对照',fusion.r2.mean(),fusion[fusion.target.str.startswith(('A_','F_'))].r2.mean()))
    fig,ax=plt.subplots(figsize=(8.3,3.1));y=np.arange(len(rows))
    for off,c,ix,label in [(-.16,BLUE,1,'全部 50 维'),(.16,TEAL,2,'香气风味 30 维')]:
        vals=[r[ix] for r in rows];ax.barh(y+off,vals,height=.27,color=c,label=label)
        for yy,v in zip(y+off,vals):ax.text(v+.003,yy,f'{v:.4f}',va='center',fontsize=10)
    ax.set(yticks=y,yticklabels=[r[0] for r in rows],xlim=(0,.315),xlabel='五折汇总 R²')
    ax.invert_yaxis();ax.legend(frameon=False,ncol=2,loc='upper right',bbox_to_anchor=(1,1.2));ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    save(fig,'02_prediction_selection')

def missingness():
    d=pd.read_csv(HERE/'grading/metrics.csv');d=d[d.target_group=='核心30维']
    scenarios=['derived_only_missing','MCAR_10_strict','MCAR_30_strict','MCAR_50_strict']
    x=[0,10,30,50]
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.6))
    for method,color,marker in [('中位数插补',GRAY,'s'),('KNN插补',TEAL,'o')]:
        z=d[d.method==method].set_index('scenario').loc[scenarios]
        axs[0].plot(x,z.macro_r2,'-',marker=marker,color=color,lw=2,label=method)
        axs[1].plot(x,100*z.within_one_fraction,'-',marker=marker,color=color,lw=2,label=method)
        for ax,vals,fmt in [(axs[0],z.macro_r2,'.3f'),(axs[1],100*z.within_one_fraction,'.1f')]:
            ax.annotate(format(vals.iloc[-1],fmt),(50,vals.iloc[-1]),xytext=(-4,8),textcoords='offset points',ha='right',color=color)
    axs[0].set(title='A  连续目标',ylabel='核心 30 维 R²',ylim=(.10,.25))
    axs[1].set(title='B  相对等级',ylabel='误差不超过 1 档的比例 / %',ylim=(46,59))
    for ax in axs:
        ax.set(xlabel='验证输入随机缺失率 / %',xticks=x);ax.grid(alpha=.17);ax.set_axisbelow(True)
    axs[0].legend(frameon=False,loc='lower left');fig.tight_layout(w_pad=2)
    save(fig,'03_missingness_grades')

def modules():
    d=pd.read_csv(HERE/'module_stability/leading_module_by_target.csv')
    z=d[(d.threshold==.6)&(~d.any_individual_at_least_0_8)&(d.module_union_frequency>=.8)].sort_values('module_union_frequency')
    fig,axs=plt.subplots(1,2,figsize=(9.5,4.1),gridspec_kw={'width_ratios':[1.45,1]})
    yy=np.arange(len(z))
    axs[0].hlines(yy,z.largest_member_frequency,z.module_union_frequency,color='#adb9bf',lw=3)
    axs[0].scatter(z.largest_member_frequency,yy,c=ORANGE,s=45,label='最稳定单成员',zorder=3)
    axs[0].scatter(z.module_union_frequency,yy,c=TEAL,s=45,label='模块任一成员',zorder=3)
    axs[0].scatter(z.uniform_three_draw_reference,yy,c=GRAY,marker='x',s=35,label='均匀抽取大小参照')
    axs[0].axvline(.8,color='#a9b4ba',ls='--',lw=1)
    axs[0].set(yticks=yy,yticklabels=[names[t]+f'  ({s}项)' for t,s in zip(z.target,z.eligible_module_size)],xlim=(0,1.04),xlabel='400 次 bootstrap 入选频率',title='A  单分子轮换与模块稳定')
    ss=json.loads((HERE/'module_stability/summary.json').read_text())['threshold_summary']
    xx=np.arange(4)
    axs[1].bar(xx-.16,[s['at_least_one_individual_frequency_0_8'] for s in ss],width=.3,color=ORANGE,label='单化合物')
    vals=[s['at_least_one_module_frequency_0_8'] for s in ss]
    axs[1].bar(xx+.16,vals,width=.3,color=TEAL,label='模块')
    for x,y in zip(xx+.16,vals):axs[1].text(x,y+.5,str(y),ha='center',fontsize=10)
    axs[1].set(xticks=xx,xticklabels=['.50','.60','.70','.80'],ylim=(0,31),xlabel='共变图 |ρ| 阈值',ylabel='至少有一项频率 ≥ 80% 的感官数',title='B  模块结论随分辨率变化')
    axs[0].legend(frameon=False,fontsize=8.5,loc='lower left',bbox_to_anchor=(0,-.42),ncol=2)
    axs[1].legend(frameon=False,fontsize=9,loc='upper right');fig.tight_layout(w_pad=1.9)
    save(fig,'04_module_stability')

def budgets():
    p=HERE/'budget_transfer/pareto_summary.csv'
    if not p.exists():return
    d=pd.read_csv(p);fig,axs=plt.subplots(1,2,figsize=(9.4,3.8))
    for family,label,color,marker in [('gradient','静态梯度',GRAY,'s'),('full_change','全维解幅度',ORANGE,'^'),('forward_schur','前向 Schur',TEAL,'o')]:
        z=d[d.method.str.startswith(family+'_k')].copy();z['k']=z.method.str.extract(r'_k(\d+)').astype(int);z=z.sort_values('k')
        axs[0].plot(z.k,100*z.target_distance_reduction_mean,color=color,marker=marker,lw=1.8,label=label)
        axs[1].plot(z.k,z.tripel_classified,color=color,marker=marker,lw=1.8,label=label)
    full=d[d.method=='all_225'].iloc[0]
    axs[0].scatter([225],[100*full.target_distance_reduction_mean],marker='D',color=PURPLE,s=48,label='全 225 维参照')
    axs[1].scatter([225],[full.tripel_classified],marker='D',color=PURPLE,s=48)
    axs[0].set(title='A  靠近程度',ylabel='11 款平均目标距离下降 / %',ylim=(0,90))
    axs[1].set(title='B  附加风格诊断',ylabel='判为 Tripel 的源啤酒数 / 11',ylim=(-.3,11.7),yticks=[0,3,6,9,11])
    for ax in axs:
        ax.set(xlabel='允许改变的直接字段数 k',xticks=[0,20,60,100,140,180,225]);ax.grid(alpha=.17);ax.set_axisbelow(True)
    axs[0].legend(frameon=False,fontsize=8.5,ncol=2,loc='upper left');fig.tight_layout(w_pad=2)
    save(fig,'05_budget_frontier')

if __name__=='__main__':
    framework();prediction();missingness();modules();budgets()
    print('Saved figures to',OUT)
