"""Publication figures from frozen, row-aligned credit evaluations."""
from pathlib import Path
from figure_io import save_figure
import json,hashlib,importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'figures'
E=ROOT/'evidence/current'; C='guard10'; S='strong_S'
ORDER=['incumbent_R',S,'previous_tail','previous_FX','previous_E',C]
LABEL=dict(zip(ORDER,['目标加权 R','风险集成 S','尾部优化 T','参考混合 FX','参考集成 E','本文方法']))
COL=dict(zip(ORDER,['#8B969D','#A38699','#7D8580','#BE9575','#ADB49F','#376B56']))
STYLE=dict(zip(ORDER,[':',(0,(5,2)),(0,(3,2)),(0,(6,1,1,1)),(0,(1,2)),'-']))
FONT=ROOT/'paper/fonts/ReportChartCJK.ttf'
font_manager.fontManager.addfont(str(FONT)); family=font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.family':['DejaVu Sans',family],'font.size':8.4,'axes.labelsize':8.4,'axes.titlesize':9.2,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,'text.color':'#2D3C33','axes.labelcolor':'#35453C','xtick.color':'#67756C','ytick.color':'#455249','grid.color':'#E6EBE6','grid.linewidth':.5})

def save(fig,name):
    for ext in ['pdf','svg','png']:save_figure(fig,OUT/(name+'.'+ext),dpi=380,facecolor='white')
    plt.close(fig)

def radial():
    import radial_capacity as r
    r.ROOT=ROOT;r.OUT=OUT
    r.METHODS=[('A',S,'风险集成 S',COL[S]),('B','previous_tail','尾部优化 T',COL['previous_tail']),('C','incumbent_R','目标加权 R',COL['incumbent_R']),('D',C,'本文方法',COL[C])]
    r.CAPACITIES=[.05,.1,.15,.2]
    z=pd.read_csv(E/'joint/capacity.csv');z=z[z.phase.eq(0)&z.candidate.isin([a[1] for a in r.METHODS])&z.capacity.isin(r.CAPACITIES)]
    records=[]
    for x in z.itertuples():
        records.append(dict(model=x.candidate,n=x.n,k=x.slots,hits=x.hits,events=99,capacity=x.capacity,unknown_selected=x.unknown,observed_yield_percent=100*x.hits/x.slots))
    r.scene(records)
    for o in r.SCENE:
        if o['kind']!='text':continue
        if o['value']=='目标加权覆盖 48 / 99 例事件':o['value']='本文方法覆盖 46 / 99 例事件'
        if o['value']=='事件覆盖率 48.5%    已观察命中率 16.5%':o['value']='事件覆盖率 46.5%    名单命中率 15.8%'
    p=OUT/'v18_primary_capacity';r.render(p);r.svg_and_drawio(p)
    pd.DataFrame(records).to_csv(OUT/'v18_primary_capacity_data.csv',index=False)

def tail():
    d=pd.read_parquet(E/'joint/predictions.parquet');edges=pd.read_parquet(E/'joint/edges.parquet')
    e=edges[edges.phase_after.eq(0)&edges.stable_05]
    a=e.row_before.to_numpy(int);b=e.row_after.to_numpy(int)
    ci=pd.read_csv(E/'joint/paired_intervals.csv')
    fig=plt.figure(figsize=(6.25,2.85))
    ax=fig.add_axes([.095,.28,.38,.57]);frozen=[]
    for n in [S,'incumbent_R','previous_tail','previous_FX',C]:
        v=100*abs(d.loc[b,n].to_numpy()-d.loc[a,n].to_numpy())
        xx=np.unique(np.r_[0,v]);yy=np.array([(v>=x).mean()*100 for x in xx]);m=yy>0
        ax.step(xx[m],yy[m],where='post',color=COL[n],ls=STYLE[n],lw=1.65 if n==C else 1.1,label=LABEL[n])
        frozen.extend(dict(model=n,drift_pp=x,exceed_percent=y) for x,y in zip(xx,yy))
    ax.set_xscale('symlog',linthresh=2,linscale=.75)
    ax.set(xlim=(0,max(x['drift_pp'] for x in frozen)*1.08),ylim=(.1,100),yscale='log',xticks=[0,1,2,5,10,20,50],xticklabels=['0','1','2','5','10','20','50'],yticks=[.1,1,10,100],yticklabels=['0.1','1','10','100'],xlabel='绝对评分变化（百分点）',ylabel='变化 ≥ 横轴值的客户（%）')
    ax.grid(which='major',alpha=.85);ax.set_axisbelow(True)
    ax.axvline(1,color='#A7B7A8',lw=.7,ls='--');ax.text(1.35,55,'1 pp',fontsize=7.8,color=COL[C])
    ax.legend(loc='upper right',frameon=False,fontsize=7.3,handlelength=2.2,labelspacing=.35)
    fig.text(.035,.94,'a   真实相邻月的尾部波动',fontsize=10,weight='bold')
    ax=fig.add_axes([.68,.28,.265,.57]);comp=['incumbent_R',S,'previous_tail','previous_FX','previous_E']
    ax.axvline(0,color='#89958B',ls='--',lw=.85)
    for i,n in enumerate(comp):
        z=ci[(ci.comparator==n)&(ci.phase==0)&(ci.metric=='cvar95_pp')].iloc[0]
        ax.hlines(i,z.sim_low,z.sim_high,color='#B3BDB4',lw=1.1)
        ax.hlines(i,z.ci_low,z.ci_high,color=COL[n],lw=2.6)
        ax.plot(z.difference,i,'o',ms=4.4,color=COL[C],zorder=3)
    ax.set(ylim=(4.6,-.6),xlim=(-20,2),xticks=[-20,-10,0],yticks=range(5),yticklabels=[LABEL[n] for n in comp],xlabel='本文 − 对手：CVaR95（pp）')
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0);ax.grid(axis='x')
    fig.text(.555,.94,'b   尾部改善的配对比较',fontsize=10,weight='bold')
    fig.text(.035,.045,'a 横轴0–2 pp为线性，其后为对数。   b 粗线：逐项95%区间；细线：同时95%区间。',fontsize=7.6,color='#758073')
    save(fig,'v18_tail_evidence');pd.DataFrame(frozen).to_csv(OUT/'v18_tail_distribution_data.csv',index=False)

def operating():
    d=pd.read_csv(E/'operating/joint_frontier.csv');f=plt.figure(figsize=(6.25,2.65))
    a=f.add_axes([.09,.24,.36,.60])
    for n in ORDER:
        z=d[d.model.eq(n)].sort_values('capacity')
        a.plot(z.capacity*100,z.unique_events,color=COL[n],ls=STYLE[n],lw=1.7 if n==C else .85,marker='o' if n==C else None,ms=3)
    a.set(xlim=(4.5,20.5),ylim=(25,90),xticks=[5,8,10,12,15,20],yticks=[30,50,70,90],xlabel='各期复核容量（%）',ylabel='两期覆盖的独立事件客户（人）');a.grid(alpha=.8);a.set_axisbelow(True)
    figlegend=f.legend([plt.Line2D([0],[0],color=COL[n],ls=STYLE[n],lw=1.6) for n in ORDER],[LABEL[n] for n in ORDER],loc='lower left',bbox_to_anchor=(.035,-.005),ncol=6,frameon=False,fontsize=7.1,handlelength=1.5,columnspacing=.75)
    f.text(.03,.95,'a   固定容量下的连续风险覆盖',fontsize=10,weight='bold')
    z=d[d.capacity.eq(.1)].set_index('model')
    for x,w,col,title,lim,xtick in [(.635,.14,'unique_clients','接触客户',[315,365],[320,340,360]),(.835,.13,'unique_events','覆盖事件',[53,63],[54,58,62])]:
        ax=f.add_axes([x,.24,w,.60])
        for y,n in enumerate(ORDER):
            v=z.loc[n,col];ax.plot(v,y,'o',ms=4.5,color=COL[n]);ax.text(v+.8,y-.12,str(int(v)),fontsize=7.8,color=COL[n],va='center')
        ax.set(ylim=(5.6,-.8),xlim=lim,xticks=xtick,yticks=range(6),yticklabels=[LABEL[n] for n in ORDER] if col=='unique_clients' else [],xlabel=title+'（人）')
        ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);ax.grid(axis='x');ax.set_axisbelow(True)
    f.text(.535,.95,'b   同为559个复核席位',fontsize=10,weight='bold')
    save(f,'v18_continuity_capacity')

if __name__=='__main__':
    radial();tail();operating()
