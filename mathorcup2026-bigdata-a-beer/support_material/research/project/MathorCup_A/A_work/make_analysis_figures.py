"""Source-backed paper figures. 

Figures use executed out-of-fold predictions and stored audit results, never
invented or schematic numerical data. All metrics remain on file 4's scale.
"""
from pathlib import Path
import os, json
os.environ.setdefault('MPLCONFIGDIR','/tmp/mathorcup_matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE=Path(__file__).parent
OUT=HERE/'figures'; OUT.mkdir(exist_ok=True)
DATA=Path(os.environ.get('MATHORCUP_DATA_DIR','/workspace/scratch/a7d3fe1f9c53/contest_2026/A/附件'))
FONT=Path(os.environ.get('MATHORCUP_CJK_FONT','/tmp/mathorcup_fonts/NotoSansCJKsc-Regular.otf'))
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    FAMILY=font_manager.FontProperties(fname=str(FONT)).get_name()
else:
    FAMILY='DejaVu Sans'
plt.rcParams.update({'font.family':FAMILY,'font.size':9,'axes.titlesize':11,'axes.labelsize':9,
    'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,
    'svg.fonttype':'path','savefig.facecolor':'white','figure.facecolor':'white',
    'axes.edgecolor':'#8d9594','grid.color':'#e1e5e3','xtick.color':'#394543','ytick.color':'#394543'})
GREEN='#1d685b';AMBER='#bc7a2d';PLUM='#796180';GRAY='#8e9995';INK='#243c36'

ZH={'A_malt_all':'麦芽总香','A_malt_grain':'谷物香','A_malt_bread':'面包香','A_malt_cara':'焦糖香','A_malt_burn':'焦烤香',
    'A_hops_all':'酒花总香','A_hops_citrus':'柑橘酒花香','A_hops_tropical':'热带酒花香','A_hops_noble':'传统酒花香','A_hops_woody':'木质酒花香',
    'A_esters_all':'酯类总香','A_esters_ethac':'乙酸乙酯香','A_esters_isoaa':'香蕉酯香','A_esters_flower':'花香','A_esters_fruity':'果香',
    'F_malt_all':'麦芽总味','F_malt_grain':'谷物味','F_malt_bread':'面包味','F_malt_cara':'焦糖味','F_malt_burn':'焦烤味',
    'F_hops_all':'酒花总味','F_hops_citrus':'柑橘酒花味','F_hops_tropical':'热带酒花味','F_hops_noble':'传统酒花味','F_hops_woody':'木质酒花味',
    'F_esters_all':'酯类总味','F_esters_ethac':'乙酸乙酯味','F_esters_isoaa':'香蕉酯味','F_esters_flower':'花味','F_esters_fruity':'果味',
    'acidity':'酸感','bitternes':'苦感','sweetness':'甜感','X4vg':'4VG感','diacetyl':'双乙酰感','dms':'DMS感','metallic':'金属感',
    'stale_hops':'陈旧酒花味','t2n':'t2n感','orange':'橙味','coriander':'芫荽味','clove':'丁香味','lactic':'乳酸感','acetic':'乙酸感',
    'barnyard':'畜棚味','alcohol':'酒精感','aftertaste':'余味','body':'酒体','co2':'碳酸感','overall':'总体欣赏度'}
(OUT/'sensory_names_zh.json').write_text(json.dumps(ZH,ensure_ascii=False,indent=2))

def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=220,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)

def models():
    s=json.loads((HERE/'q1/summary.json').read_text())
    order=['nested_ensemble','extra_trees','random_forest','rbf','ridge_rank','pls','style_mean','training_mean']
    names=['折内学习权重的集成','Extra Trees','随机森林','RBF 核岭','秩变换岭回归','偏最小二乘','风格均值（已知风格）','训练均值']
    values=[s['models'][m]['macro_r2'] for m in order]
    lo=[s['bootstrap_ci'][m]['lower'] for m in order]
    hi=[s['bootstrap_ci'][m]['upper'] for m in order]
    fig,ax=plt.subplots(figsize=(7.1,3.8));pos=np.arange(len(order))
    ax.barh(pos,values,color=[GREEN]+[GRAY]*5+[AMBER,GRAY],height=.58,zorder=2)
    # Bootstrap mean-baseline CI can lie wholly below its fixed-OOF point.
    ax.hlines(pos,lo,hi,color=INK,linewidth=1.1,zorder=3)
    ax.scatter(values,pos,color=INK,s=12,zorder=4)
    for i,v in enumerate(values):ax.text(max(v,hi[i])+.006,i,f'{v:.3f}',va='center',fontsize=9)
    ax.set_yticks(pos,names);ax.invert_yaxis();ax.set_xlim(-.03,.32)
    ax.axvline(0,color='#b9bfbc',lw=.8);ax.xaxis.grid(True,zorder=0)
    ax.set_xlabel('50 个维度的宏平均 R²（五折外层留出预测）')
    ax.set_title('感官预测：集成带来有限、可重复检查的增益',loc='left',pad=12)
    fig.text(.01,-.025,'线段为固定 OOF 预测按啤酒重采样的描述性 95% 区间；不包含重训练和选模不确定性。\n原尺度岭回归 R² = −0.538，见正文表；独立 HGB 对照 R² = 0.195。',fontsize=8,color='#596862')
    fig.tight_layout();save(fig,'01_q1_model_comparison')

def dimension_heatmap():
    d=pd.read_csv(HERE/'q1/dimension_metrics.csv');d=d[d.model=='nested_ensemble'].set_index('target')
    targets=list(ZH);fig,ax=plt.subplots(figsize=(7.1,5.4))
    values=np.array([d.loc[t,'r2'] for t in targets]).reshape(5,10)
    labels=np.array([ZH[t] for t in targets]).reshape(5,10)
    im=ax.imshow(values,cmap='YlGn',vmin=0,vmax=.75,aspect='auto')
    for i in range(5):
        for j in range(10):
            val=values[i,j];ax.text(j,i-.13,labels[i,j],ha='center',va='center',fontsize=8,color='white' if val>.51 else INK)
            ax.text(j,i+.17,f'{val:.3f}',ha='center',va='center',fontsize=9,color='white' if val>.51 else INK,fontweight='bold')
    ax.set_xticks(np.arange(-.5,10,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
    for sp in ax.spines.values():sp.set_visible(False)
    ax.set_title('50 维留出 R²：基础感受强，细分香气与部分缺陷味弱',loc='left',pad=12)
    fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.055,pad=.06,label='R²（花味的 −0.001 以颜色下界显示）')
    fig.tight_layout();save(fig,'02_q1_dimension_heatmap')

def scatters():
    a=np.load(HERE/'q1/oof_arrays.npz');meta=pd.read_csv(HERE/'audit/sensory_metadata.csv')
    ts=meta.field.tolist();fig,axes=plt.subplots(2,2,figsize=(7.1,5.8))
    for ax,t in zip(axes.flat,['alcohol','acidity','F_hops_all','overall']):
        j=ts.index(t);y=a['Y'][:,j];p=a['nested_ensemble'][:,j]
        low=min(y.min(),p.min());high=max(y.max(),p.max());gap=(high-low)*.06
        ax.scatter(y,p,s=12,alpha=.6,edgecolors='none',color=GREEN)
        ax.plot([low-gap,high+gap],[low-gap,high+gap],color=AMBER,lw=1,ls='--')
        ax.set_xlim(low-gap,high+gap);ax.set_ylim(low-gap,high+gap)
        r=1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)
        ax.set_title(f'{ZH[t]}  ·  R² = {r:.3f}',loc='left',fontsize=10)
        ax.set_xlabel('附件 4 观测分数');ax.set_ylabel('外层留出预测');ax.grid(alpha=.5)
    fig.tight_layout(h_pad=1.6,w_pad=1.5);save(fig,'03_q1_oof_scatter')

def missing():
    d=pd.read_csv(HERE/'audit/model_validation/strict_missingness_metrics.csv').set_index('scenario')
    normal=[d.loc['original','macro_r2']]+[d.loc[f'MCAR_{r}_original','macro_r2'] for r in [10,30,50]]
    strict=[d.loc['derived_only_missing','macro_r2']]+[d.loc[f'MCAR_{r}_strict','macro_r2'] for r in [10,30,50]]
    kn=pd.read_csv(HERE/'audit/model_validation/strict_knn_missingness_metrics.csv')
    kn=kn[kn.all_five_derived_hidden]
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.8),gridspec_kw={'width_ratios':[1.15,1]})
    ax=axes[0];ax.plot([0,10,30,50],normal,'o-',color=GRAY,label='普通随机遮蔽',lw=1.5,ms=4)
    ax.plot([0,10,30,50],strict,'s-',color=GREEN,label='额外屏蔽全部派生量',lw=1.8,ms=4)
    ax.plot([0,10,30,50],kn.macro_r2,'^--',color=PLUM,label='KNN（全部派生量隐藏）',lw=1.5,ms=4)
    ax.set_xlabel('对输入单元格施加的随机缺失率 (%)');ax.set_ylabel('宏平均 R²');ax.set_ylim(.08,.285)
    ax.set_xticks([0,10,30,50]);ax.grid(alpha=.7);ax.legend(frameon=False,fontsize=7.5,loc='lower left')
    ax.set_title('A  随机缺失',loc='left',fontsize=10)
    ax=axes[1];sc=['first37_strict','gcms_strict','high_value_strict']
    # Actual key has an explicit name in the audit file.
    sc[-1]=next(k for k in d.index if 'strict' in k and ('high' in k))
    vals=[d.loc[k,'macro_r2'] for k in sc]
    labels=['前 37 项整组缺失','GC–MS 组缺失','高值优先缺失']
    ax.barh(np.arange(3),vals,color=[GREEN,AMBER,PLUM],height=.5)
    for i,v in enumerate(vals):ax.text(v+.006,i,f'{v:.3f}',va='center',fontsize=9)
    ax.set_yticks(np.arange(3),labels,fontsize=8);ax.invert_yaxis();ax.set_xlim(0,.24)
    ax.set_xlabel('宏平均 R²');ax.xaxis.grid(True);ax.set_axisbelow(True)
    ax.set_title('B  分组 / 非随机缺失',loc='left',fontsize=10)
    fig.tight_layout(w_pad=1.8)
    fig.text(.01,-.055,'严格场景均额外屏蔽 kcalperc 与 4 个汇总指标；“前 37 项”不是单一仪器组。\n这些是输入受损压力测试；不替代新增缺测机制下的外部验证。',fontsize=8,color='#596862')
    save(fig,'04_q1_missingness')

def generation():
    initial=pd.read_csv(HERE/'q3/locked_test_comparison.csv')
    ext=pd.read_csv(HERE/'q3/extension_outer_comparison.csv')
    chosen=ext[ext.selected];smooth=ext[ext.model=='smooth_pooled'];ig=ext[ext.model=='interval_gaussian']
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.7))
    x=np.arange(3);labels=['平滑模型','间隔模型','内层选择流程']
    for ax,key,title in zip(axes,['energy','c2st_auc','nn_ratio'],['能量评分（低为优）','真假分类 AUC','生成 / 真实近邻距离']):
        groups=[smooth,ig,chosen];vals=[g[key].mean() for g in groups]
        ax.bar(x,vals,color=[GRAY,AMBER,GREEN],width=.6,zorder=2)
        for i,g in enumerate(groups):
            ax.scatter(np.full(len(g),i)+np.linspace(-.11,.11,len(g)),g[key],s=13,color=INK,zorder=4)
            ax.text(i,max(g[key])+(.007 if key=='energy' else .025),f'{vals[i]:.3f}',ha='center',fontsize=8)
        if key=='energy':ax.set_ylim(.57,.68)
        elif key=='c2st_auc':ax.set_ylim(.45,1.07);ax.axhline(.5,ls='--',color=PLUM,lw=1)
        else:ax.set_ylim(0,1.3);ax.axhline(1,ls='--',color=PLUM,lw=1)
        ax.set_xticks(x,labels,rotation=25,ha='right',fontsize=8);ax.set_title(title,fontsize=9,loc='left')
        ax.yaxis.grid(alpha=.6);ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1.3)
    fig.text(.01,-.11,'图中为诊断后 3×2 折内部验证；点表示外层折。AUC 接近 0.5 只表示该判别器难以区分。\n原先 63 款留出测试中，平滑模型 AUC = 0.959；该发现已用于修改模型，不能再次称为独立测试。',fontsize=8,color='#596862')
    save(fig,'05_q3_generation_validation')

def sensory_fingerprints():
    d=pd.read_csv(HERE/'q3/generated_sensory_fingerprint_summary.csv')
    obs=pd.read_csv(DATA/'4.csv')
    targets=['alcohol','acidity','bitternes','sweetness','F_malt_all','F_hops_all','F_esters_all','body','stale_hops','overall']
    fig,axes=plt.subplots(1,3,figsize=(7.2,4.4),sharey=True,sharex=True)
    for ax,style,color in zip(axes,['Tripel','Kriek','Lambic'],[GREEN,AMBER,PLUM]):
        z=d[d['style']==style].set_index('target').loc[targets]
        yy=np.arange(len(targets))
        ax.hlines(yy,z.predicted_q10,z.predicted_q90,color=color,lw=2)
        ax.scatter(z.predicted_median,yy,color=color,s=22,zorder=3,label='生成预测中位数')
        means=obs[obs.tasting_category_fine==style][targets].mean()
        ax.scatter(means,yy,marker='D',facecolors='white',edgecolors=GRAY,s=20,zorder=4,label='真实同风格观测均值')
        ax.axvline(0,color='#b9bfbc',lw=.8,ls='--');ax.set_yticks(yy,[ZH[t] for t in targets])
        ax.set_title(style,loc='left',color=color);ax.set_xlim(-1.2,2.5);ax.grid(axis='x',alpha=.6)
        ax.set_xlabel('附件 4 标签尺度')
    axes[0].invert_yaxis()
    handles,labels=axes[-1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=8,bbox_to_anchor=(.5,-.035))
    fig.tight_layout(w_pad=1.1)
    fig.text(.01,-.12,'线段为各风格 100 个生成样本预测值的 10%–90% 范围；不包含品评误差和模型不确定性。\n空心菱形仅作现有同风格数据参照，不是新样本实测。',fontsize=8,color='#596862')
    save(fig,'06_q3_sensory_fingerprints')

def final_transfer():
    d=pd.read_csv(HERE/'q3/transfer_final_diagnostics.csv').sort_values('beer').set_index('beer')
    cf=pd.read_csv(HERE/'audit/transfer_oof_proxy_diagnostic.csv').set_index('beer').loc[d.index]
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.7),sharey=True,gridspec_kw={'width_ratios':[.85,1.35]})
    yy=np.arange(len(d));labels=[str(i)+('*' if d.loc[i,'method']=='cross_model_margin_qp' else '') for i in d.index]
    colors=[AMBER if d.loc[i,'predicted_style']!='Tripel' else GREEN for i in d.index]
    axes[0].barh(yy,d.target_distance_reduction*100,color=colors,height=.55)
    for y,v in zip(yy,d.target_distance_reduction*100):axes[0].text(v+1,y,f'{v:.1f}',va='center',fontsize=8)
    axes[0].set_xlim(0,99);axes[0].set_yticks(yy,labels);axes[0].set_xlabel('目标距离下降 (%)')
    axes[0].set_ylabel('源啤酒编号');axes[0].set_title('A  目标靠近',loc='left',fontsize=10)
    axes[0].grid(axis='x',alpha=.5);axes[0].set_axisbelow(True)
    ax=axes[1]
    for key,label,color,marker in [('linear_stale_delta','线性约束代理',GRAY,'s'),('ensemble_stale_delta','完整感官集成',AMBER,'o')]:
        ax.scatter(d[key],yy,color=color,marker=marker,s=24,label=label,zorder=3)
    ax.scatter(cf.stale_delta_oof_model,yy,color=GREEN,marker='D',s=24,label='源样本留出模型',zorder=4)
    ax.axvline(0,color=PLUM,lw=1);ax.set_xlim(-.77,.025);ax.set_xlabel('陈旧酒花味预测差（候选 − 源）')
    ax.set_title('B  三种代理均未预测上升',loc='left',fontsize=10);ax.grid(axis='x',alpha=.5)
    axes[0].invert_yaxis();fig.tight_layout(w_pad=1.8)
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(.55,-.035))
    fig.text(.01,-.125,'* 为经有限余量搜索修复的 3 款；橙色目标条为仍判作 Blond 的 Beer 124。\n三类代理共享原始数据；一致下降不是实际感官非劣性证明。',fontsize=8,color='#596862')
    save(fig,'07_q3_final_transfer')

if __name__=='__main__':
    models();dimension_heatmap();scatters();missing();generation();sensory_fingerprints();final_transfer()
    print('Wrote seven source-backed PNG/SVG figure pairs to',OUT)
