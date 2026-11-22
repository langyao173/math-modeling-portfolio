"""Paper figures and tables from frozen row-level research results."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from figure_io import save_figure

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'evidence/current/joint'
OUT=ROOT/'figures'
FONT=ROOT/'assets/fonts/ReportChartCJK.ttf'
fm.fontManager.addfont(str(FONT))
FAMILY=fm.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.family':['DejaVu Sans',FAMILY], 'font.size':8.2,
 'axes.labelsize':8.1,'axes.titlesize':9,'axes.linewidth':.5,
 'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,
 'text.color':'#2D3C33','axes.labelcolor':'#35453C','xtick.color':'#67756C',
 'ytick.color':'#455249','grid.color':'#E7EBE6','grid.linewidth':.5,
 'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
GREEN='#376B56'; PURPLE='#A38699'; GRAY='#7D8580'; CLAY='#BE9575'; INK='#2D3C33'
METHODS=['guard10','strong_S','previous_tail','base_reference','incumbent_R']
LABELS=['本文方法','风险集成 S','尾部优化 T','同底模静态','目标加权 R']
COLORS=[GREEN,PURPLE,GRAY,CLAY,'#8B969D']
STYLES=['-',(0,(5,2)),(0,(3,2)),(0,(1,2)),(0,(5,1,1,1))]

def save(fig,name):
    for ext in ['pdf','svg','png']:
        save_figure(fig,OUT/f'{name}.{ext}',dpi=400,facecolor='white')
    plt.close(fig)

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2.2,width=.5,pad=2.5)
    ax.grid(axis='y');ax.set_axisbelow(True)

def capacity():
    d=pd.read_csv(DATA/'capacity.csv')
    f=plt.figure(figsize=(6.25,2.75))
    f.text(.02,.956,'a   提额月 · 99个事件窗口',fontsize=9.4,weight='bold')
    f.text(.55,.956,'b   相邻月 · 88个事件窗口',fontsize=9.4,weight='bold')
    f.legend([plt.Line2D([0],[0],color=c,ls=s,lw=1.7 if i==0 else 1.0) for i,(c,s) in enumerate(zip(COLORS,STYLES))],
             LABELS,loc='upper center',bbox_to_anchor=(.51,.927),ncol=5,frameon=False,
             fontsize=7.5,handlelength=1.8,columnspacing=1.1)
    cmap=LinearSegmentedColormap.from_list('clay_white_sage',['#D5B5A0','#FCFCFA','#A8C1AE'])
    records=[]
    for phase,left in [(0,.078),(1,.601)]:
        g=d[d.phase.eq(phase)].pivot(index='candidate',columns='capacity',values='hits').sort_index(axis=1)
        cap=g.columns.to_numpy()*100;xx=cap
        ax=f.add_axes([left,.42,.37,.40]);clean(ax)
        for i,(name,c,style) in enumerate(zip(METHODS,COLORS,STYLES)):
            values=g.loc[name].to_numpy()
            ax.plot(xx,values,color=c,ls=style,lw=1.6 if i==0 else 1.0,
                    marker='o' if i==0 else None,ms=3.5,zorder=5 if i==0 else 2)
            if i==0:
                for x,y in zip(xx,values):
                    ax.annotate(str(int(y)),(x,y),xytext=(0,6),textcoords='offset points',ha='center',
                                fontsize=7.7,color=GREEN,weight='bold')
        ax.set(xlim=(3.7,21.3),ylim=(0,74),yticks=[0,20,40,60],xticks=xx)
        ax.tick_params(axis='x',labelbottom=False,length=0)
        if phase==0:ax.set_ylabel('覆盖事件（例）',labelpad=4)
        f.text(left,.348,'相同名额下，本文多覆盖 / 少覆盖',fontsize=7.6,color='#617264')
        h=f.add_axes([left,.16,.37,.145])
        mat=g.loc['guard10'].to_numpy()[None,:]-g.loc[['strong_S','previous_tail']].to_numpy()
        norm=TwoSlopeNorm(0,-5,5)
        for i in range(2):
            for j,x in enumerate(xx):
                h.add_patch(Rectangle((x-.90,i-.46),1.8,.92,facecolor=cmap(norm(mat[i,j])),edgecolor='none'))
                h.text(x,i,f'{int(mat[i,j]):+d}' if mat[i,j] else '0',ha='center',va='center',fontsize=8.2,color=INK)
        h.set(xlim=(3.7,21.3),ylim=(1.5,-.5),xticks=xx,xticklabels=[f'{v:g}%' for v in cap],yticks=[0,1],yticklabels=['对 S','对 T'])
        h.tick_params(length=0,pad=4,labelsize=7.9)
        for spine in h.spines.values():spine.set_visible(False)
        f.text(left+.135,.035,'每期复核容量',fontsize=8.1)
        for name in METHODS:
            for c,v in g.loc[name].items():records.append(dict(phase=phase,candidate=name,capacity=c,hits=v))
    pd.DataFrame(records).to_csv(OUT/'v21_capacity_range_data.csv',index=False)
    save(f,'v21_capacity_range')

def tail():
    d=pd.read_csv(ROOT/'evidence/current/tail_intervals.csv')
    rows=[(g,p) for g in ['repayment_state','flow_structure'] for p in [0,1]]
    y=np.array([0,1,2.65,3.65])
    f=plt.figure(figsize=(6.25,2.58))
    f.text(.02,.945,'a   真实业务客群的尾部幅度',fontsize=9.4,weight='bold')
    f.text(.555,.945,'b   配对比较及同时区间',fontsize=9.4,weight='bold')
    f.legend([plt.Line2D([0],[0],marker=m,ls='',color=c,ms=4.5) for m,c in [('o',GREEN),('D',PURPLE),('^',GRAY)]],
             ['本文方法','风险集成 S','尾部优化 T'],loc='upper left',bbox_to_anchor=(.155,.897),
             frameon=False,ncol=3,fontsize=7.6,columnspacing=1.1,handletextpad=.4)
    a=f.add_axes([.205,.22,.274,.53]);b=f.add_axes([.600,.22,.355,.53])
    for k,(g,p) in enumerate(rows):
        s=d[d.group.eq(g)&d.phase.eq(p)&d.comparator.eq('strong_S')].iloc[0]
        t=d[d.group.eq(g)&d.phase.eq(p)&d.comparator.eq('previous_tail')].iloc[0]
        a.hlines(y[k],s.own_cvar,s.other_cvar,color='#CBD4CB',lw=1.5,zorder=1)
        for val,m,c in [(s.own_cvar,'o',GREEN),(s.other_cvar,'D',PURPLE),(t.other_cvar,'^',GRAY)]:
            a.plot(val,y[k],m,color=c,ms=4.4,zorder=3)
        for j,r in enumerate([s,t]):
            yy=y[k]+[-.13,.13][j];color=[PURPLE,GRAY][j]
            b.hlines(yy,r.sim_low,r.sim_high,color=color,lw=1.2)
            b.plot(r.difference,yy,['D','^'][j],ms=4,color=color,zorder=3)
    labels=['状态组 · 提额月','状态组 · 相邻月','结构组 · 提额月','结构组 · 相邻月']
    for ax in [a,b]:
        ax.set(ylim=(4.3,-.65));ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0,pad=5);ax.tick_params(axis='x',length=2.5,pad=4)
        ax.grid(axis='x');ax.set_axisbelow(True)
        ax.axhline(1.82,color='#E0E6DF',lw=.6)
    a.set(xlim=(0,12),xticks=[0,4,8,12],yticks=y,yticklabels=labels,xlabel='CVaR95（pp）')
    b.axvline(0,color='#758277',ls='--',lw=.8)
    b.set(xlim=(-13.2,3.0),xticks=[-12,-8,-4,0],yticks=[],xlabel='本文 − 对手：CVaR95（pp）')
    f.text(.205,.025,'状态组：支付覆盖状态、交易笔数档位相同；结构组：支付及交易比例相近。',fontsize=7.1,color='#71806F')
    d.to_csv(OUT/'v21_business_tail_data.csv',index=False)
    save(f,'v21_business_tail')

if __name__=='__main__':
    capacity();tail()
