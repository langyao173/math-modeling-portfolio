"""A dominant conditional joint distribution with an empirical-atom side study.
All coordinates are frozen observations or saved generated spectra. No refit,
new simulation, or reconstructed latent covariance is used.
"""
from repair_common import *
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import squareform

DIVERGE=LinearSegmentedColormap.from_list('rank_correlation',[BLUE,'#FFFFFF',RED])

def export_supp(fig,stem):
    p=OUT/stem;fig.savefig(p.with_suffix('.svg'),facecolor='white')
    subprocess.run(['inkscape','--app-id-tag=copulas8',str(p.with_suffix('.svg')),'--export-type=pdf','--export-area-page','--export-filename='+str(p.with_suffix('.pdf'))],capture_output=True,check=True)
    with fitz.open(p.with_suffix('.pdf'))as d:d[0].get_pixmap(dpi=220).save(p.with_suffix('.png'))
    plt.close(fig)

def supplement():
    path=DATA/'fig08_observed_normal_score_correlation.csv';corr=pd.read_csv(path,index_col=0)
    distance=np.clip(1-np.abs(corr.to_numpy()),0,1);np.fill_diagonal(distance,0)
    dis=squareform(distance,checks=False);tree=linkage(dis,method='average');order=leaves_list(optimal_leaf_ordering(tree,dis));a=corr.to_numpy()[np.ix_(order,order)]
    pd.DataFrame({'display_index':np.arange(1,226),'original_index':order+1,'field':corr.index[order]}).to_csv(DATA/'fig08_supplement_correlation_display_order.csv',index=False)
    f=plt.figure(figsize=(6.7,6.4));ax=f.add_axes([.12,.13,.76,.79]);im=ax.imshow(a,cmap=DIVERGE,vmin=-1,vmax=1,interpolation='nearest',aspect='equal',rasterized=False)
    ax.set(xticks=[0,49,99,149,199,224],xticklabels=[1,50,100,150,200,225],yticks=[0,49,99,149,199,224],yticklabels=[1,50,100,150,200,225],xlabel='按相关结构排序的字段',ylabel='按相关结构排序的字段');ax.tick_params(length=2,width=.4)
    for s in ax.spines.values():s.set_visible(True);s.set_lw(.45);s.set_color('#85919B')
    cb=f.colorbar(im,cax=f.add_axes([.91,.30,.015,.42]),ticks=[-1,0,1]);cb.outline.set_visible(False);cb.set_label('正态秩相关',fontsize=7)
    f.text(.12,.963,'补图 S8  实测化学字段的相关结构',fontsize=9,color=INK)
    f.text(.12,.055,'225字段 · 250款实测；平均链接按 1 − |r| 排序，仅改变显示顺序。',fontsize=6.6,color=SECONDARY)
    f.text(.12,.026,'描述性正态秩相关；不表示区间Copula拟合的潜协方差。完整字段顺序见配套底表。',fontsize=6.6,color=SECONDARY)
    export_supp(f,'supplement_S8_descriptive_correlation')
    return order


def main():
    setup()
    pa=DATA/'fig08_empirical_4VG_atoms.csv';pr=DATA/'fig08_observed_4VG_rug.csv';pj=DATA/'fig08_saved_joint_chemistry.csv';pk=DATA/'fig08_descriptive_log_kde.npz'
    atoms=pd.read_csv(pa);rug=pd.read_csv(pr);joint=pd.read_csv(pj);kd=np.load(pk)
    obs=joint[joint.origin.eq('observed')];con=joint[joint.origin.eq('conditioned')]
    assert len(obs)==250 and len(con)==2200 and joint['style'].nunique()==22
    assert len(atoms)==83 and atoms['count'].iloc[0]==168 and atoms['count'].sum()==250
    assert (con.DMS_ug_L<=19+1e-10).all() and (con.diacetyl_mg_L<=.09+1e-12).all()
    fig=figure(8);fig.set_size_inches(6.7,3.6)
    # One common top edge; the left study is visually subordinate to the joint plot.
    fig.text(.07,.953,'a  4VG经验原子',fontsize=7.4,color=INK)
    fig.text(.422,.953,'b  条件化学谱的联合分布',fontsize=7.4,color=INK)
    fig.text(.07,.816,'67.2%',fontsize=19,color=TEAL,fontweight='normal')
    fig.text(.245,.827,'168 / 250',fontsize=6.6,color=SECONDARY,ha='left')
    fig.text(.245,.789,'重复下限',fontsize=6.3,color=SECONDARY,ha='left')
    a=fig.add_axes([.08,.22,.245,.45]);v=atoms.concentration_mg_L.to_numpy();cdf=atoms.cdf.to_numpy();atom=v[0]
    a.step(np.r_[atom*.92,v],np.r_[0,cdf],where='post',c=BLUE,lw=.9,zorder=3)
    a.vlines(atom,0,cdf[0],color=TEAL,lw=1.8,zorder=4)
    a.scatter([atom],[cdf[0]],s=13,c=TEAL,ec='white',lw=.35,zorder=5)
    a.hlines(cdf[0],atom,v[-1],color=TEAL,lw=.45,linestyle=(0,(2,2)),alpha=.5)
    a.vlines(rug.X4VG_mg_L,-.045,-.011,color=BLUE,lw=.35,alpha=.27)
    a.set(xscale='log',xlim=(atom*.86,v[-1]*1.1),ylim=(-.055,1.035),xticks=[.01,.1,.5],xticklabels=['0.01','0.1','0.5'],yticks=[0,.5,1],yticklabels=['0','0.5','1'],xlabel='4VG / mg·L⁻¹',ylabel='经验累计比例')
    a.xaxis.set_minor_locator(NullLocator());clean(a,'y');a.spines['left'].set_bounds(0,1);a.spines['bottom'].set_position(('data',-.055))
    # The atom and all 250 rugs are retained; no smoothed atom approximation.
    fig.text(.08,.065,'实测浓度 · 对数刻度',fontsize=6.1,color=SECONDARY)
    # Dominant, tightly coupled joint panel with identical marginal endpoints.
    x0,y0,w,h=.422,.22,.454,.535
    ax=fig.add_axes([x0,y0,w,h]);top=fig.add_axes([x0,.775,w,.115],sharex=ax);side=fig.add_axes([.891,y0,.083,h],sharey=ax)
    ox=obs.DMS_ug_L.to_numpy();oy=obs.diacetyl_mg_L.to_numpy();cx=con.DMS_ug_L.to_numpy();cy=con.diacetyl_mg_L.to_numpy()
    xmin,xmax=min(joint.DMS_ug_L)*.92,max(joint.DMS_ug_L)*1.08;ymin,ymax=min(joint.diacetyl_mg_L)*.72,max(joint.diacetyl_mg_L)*1.3
    ax.set(xscale='log',yscale='log',xlim=(xmin,xmax),ylim=(ymin,ymax))
    # Quality event remains an actual rectangle on the physical concentration axes.
    ax.add_patch(Rectangle((xmin,ymin),19-xmin,.09-ymin,facecolor='#F1F8F7',edgecolor='none',zorder=0))
    gx,gy=kd['log10_dms_grid'],kd['log10_diacetyl_grid'];X,Y=np.meshgrid(10.**gx,10.**gy);density=kd['density']
    levels=np.array([.25,.5,.75])*density.max()
    ax.contour(X,Y,density,levels=levels,colors=[TEAL],linewidths=[.45,.6,.75],alpha=.5,zorder=1)
    ax.scatter(ox,oy,s=6,facecolors='white',edgecolors='#8D9BA8',lw=.38,alpha=.78,zorder=2)
    ax.scatter(cx,cy,s=2.0,color=TEAL,edgecolor='none',alpha=.21,zorder=3)
    ax.axvline(19,c=RED,lw=.7,ls=(0,(3,2)),zorder=4);ax.axhline(.09,c=RED,lw=.7,ls=(0,(3,2)),zorder=4)
    ax.set(xticks=[1,19,100,500],xticklabels=['1','19','100','500'],yticks=[1e-7,1e-4,.09,100],yticklabels=['10⁻⁷','10⁻⁴','0.09','10²'],xlabel='DMS / μg·L⁻¹',ylabel='双乙酰 / mg·L⁻¹')
    ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_locator(NullLocator());clean(ax)
    for spine in ax.spines.values():spine.set_visible(True);spine.set_linewidth(.45);spine.set_color('#85919B')
    ax.text(.025,.045,'合格域 E',transform=ax.transAxes,color=TEAL,fontsize=6.2)
    ax.get_xticklabels()[1].set_color(RED);ax.get_yticklabels()[2].set_color(RED)
    # Equal-width log bins with fractions: each series sums to one despite unequal n.
    bx=np.geomspace(xmin,xmax,27);by=np.geomspace(ymin,ymax,32)
    for xx,yy,col,fill,alpha in [(ox,oy,'#8D9BA8',False,.9),(cx,cy,TEAL,True,.43)]:
        ht,_=np.histogram(xx,bins=bx);hr,_=np.histogram(yy,bins=by)
        top.stairs(ht/len(xx),bx,color=col,lw=.65,fill=fill,alpha=alpha)
        side.stairs(hr/len(yy),by,color=col,lw=.65,fill=fill,alpha=alpha,orientation='horizontal')
    for m in [top,side]:
        m.tick_params(axis='both',which='both',bottom=False,labelbottom=False,left=False,labelleft=False)
        for spine in m.spines.values():spine.set_visible(False)
    top.set_ylim(bottom=0);side.set_xlim(left=0)
    # A compact stable legend outside the observed tails.
    fig.legend(handles=[Line2D([],[],marker='o',mfc='white',mec='#8D9BA8',mew=.5,ls='',ms=3.5,label='实测 250'),Line2D([],[],marker='o',mfc=TEAL,mec=TEAL,ls='',ms=3,label='条件谱 2200'),Line2D([],[],c=RED,ls=(0,(3,2)),lw=.7,label='质量上界')],loc='lower left',bbox_to_anchor=(.422,.057),frameon=False,ncol=3,handlelength=1.3,handletextpad=.4,columnspacing=1.1,fontsize=6.1,borderaxespad=0)
    order=supplement()
    caption=('经验原子与二阈值条件谱。a为250款实测4VG的经验累计分布及逐款底部短线，最小浓度0.009702 mg/L处168款重合（67.2%）。'
             'b以DMS—双乙酰联合分布为主体，显示250款实测与已有2200份条件化学谱（22风格、每风格100份）；两轴均为原浓度单位的对数刻度。'
             '虚线分别为19 μg/L与0.09 mg/L上界，左下浅色区域为合格域E；边际为等宽对数分箱的样本比例，每组各自归一化；等高线仅描述已存条件谱在对数坐标中的密度。'
             '225维实测正态秩相关矩阵移至补图S8，按平均链接的1−|r|显示排序，非区间Copula的拟合潜协方差。所有极值与重复点均保留，不作独立外部检验。')
    save(fig,8,caption,[pa,pr,pj,pk,DATA/'fig08_observed_normal_score_correlation.csv'],dict(observed_n=250,generated_n=2200,styles=22,atom_count=168,atom_fraction=.672,main_visual='physical_joint_distribution',supplement='supplement_S8_descriptive_correlation.pdf',correlation_dimensions=225,display_only_clustering=True,joint_axes='original physical concentration units; logarithmic',new_point_simulations=0))
    audit=dict(status='PASS',observed_n=250,generated_n=2200,styles=22,unique_4VG_values=len(atoms),atom_count=168,atom_mass=.672,all_2200_pass_two_ceilings=True,all_points_retained=True,correlation_matrix_shape=[225,225],clustering_for_display_only=True,correlation_original_order_recoverable=bool(len(np.unique(order))==225),supplement='supplement_S8_descriptive_correlation.pdf',new_model_fits=0,new_bootstrap_resamples=0,new_points=0)
    (BASE/'fig08_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
