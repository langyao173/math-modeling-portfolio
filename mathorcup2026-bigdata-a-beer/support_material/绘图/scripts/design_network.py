from repair_common import *
import pandas as pd
from matplotlib.patches import PathPatch,Arc,Rectangle
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
import shutil

P=DATA

def components(nodes,edges):
    adj={f:set()for f in nodes.field}
    for r in edges.itertuples():adj[r.source].add(r.target);adj[r.target].add(r.source)
    seen=set();cs=[]
    for f in adj:
        if f in seen:continue
        q=[f];c=set()
        while q:
            a=q.pop()
            if a in seen:continue
            seen.add(a);c.add(a);q.extend(adj[a]-seen)
        cs.append(c)
    return sorted(cs,key=lambda c:(-len(c),sorted(c)[0]))

def figure6():
    pn=P/'06_network_positions.csv';pe=P/'06_network_edges.csv';pf=P/'06_module_substitution.csv'
    n=pd.read_csv(pn);e=pd.read_csv(pe);fr=pd.read_csv(pf);cs=components(n,e)
    main=cs[0];nd=n[n.field.isin(main)].copy();ee=e[e.source.isin(main)]
    assert len(n)==130 and len(e)==266 and len(main)==99 and len(cs)==13
    palette={1:BLUE,2:TEAL,3:'#84BBB4',4:PURPLE,5:RED,6:'#A9B4BE',10:'#C4CBD1'}
    f=figure(6);f.set_size_inches(6.7,4.0)
    ax=f.add_axes([.015,.045,.61,.91]);ax.set_aspect('equal');ax.axis('off');ax.set(xlim=(-1.55,1.55),ylim=(-1.53,1.53))
    # A common circular index links module arcs, degree track, nodes, and edges.
    xy={};angles={};group_spans={};records=[];cursor=87
    modorder=[1,2,3,4,5,6,10];gap=5.8;step=(360-len(modorder)*gap)/len(nd)
    for mod in modorder:
        group=nd[nd.module.eq(mod)].copy();group['angle_old']=np.arctan2(group.layout_y,group.layout_x)
        group=group.sort_values('angle_old',kind='stable');start=cursor
        for row in group.itertuples():
            ang=np.deg2rad(cursor-step/2);xy[row.field]=np.array([np.cos(ang),np.sin(ang)]);angles[row.field]=ang
            records.append(dict(field=row.field,module=mod,angle_degrees=float(np.degrees(ang)),display_x=float(np.cos(ang)),display_y=float(np.sin(ang)),weighted_degree=row.weighted_degree));cursor-=step
        group_spans[mod]=(cursor,start);cursor-=gap
    for r in ee.itertuples():
        a,b=xy[r.source],xy[r.target];ma=int(nd.set_index('field').loc[r.source,'module']);mb=int(nd.set_index('field').loc[r.target,'module'])
        c=RED if r.rho<0 else palette[ma]if ma==mb else '#8B9AA7'
        bend=.61 if ma==mb else .20
        path=MPath([a,a*bend,b*bend,b],[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4])
        ax.add_patch(PathPatch(path,fill=False,ec=c,lw=.35 if r.rho>=0 else .65,alpha=.37 if r.rho>=0 else .85,ls=(0,(2,1.8))if r.rho<0 else'solid',zorder=1))
    mx=nd.weighted_degree.max()
    for r in nd.itertuples():
        a=angles[r.field];col=palette[int(r.module)];v=xy[r.field]
        ax.plot([1.075*v[0],(1.075+.145*r.weighted_degree/mx)*v[0]],[1.075*v[1],(1.075+.145*r.weighted_degree/mx)*v[1]],c=col,lw=1.5,solid_capstyle='butt',zorder=2)
        ax.scatter(*v,s=4.5+.8*r.weighted_degree,fc=col,ec='white',lw=.28,zorder=3)
    for mod,(a,b)in group_spans.items():
        ax.add_patch(Arc((0,0),2.59,2.59,theta1=a,theta2=b,lw=4.0,color=palette[mod],capstyle='butt'))
        mid=np.deg2rad((a+b)/2);rot=np.degrees(mid)-90
        rot=(rot+180)%360-180
        if rot>90:rot-=180
        if rot<-90:rot+=180
        label=f'M{mod} · {int(nd.module.eq(mod).sum())}'
        ax.text(1.43*np.cos(mid),1.43*np.sin(mid),label,fontsize=6.2,color=palette[mod],ha='center',va='center',rotation=rot,rotation_mode='anchor')
    labels=[('X2.acetylfuran','2-AF',(.31,.62)),('X2.6.dimethyl.pyrazine','2,6-DMP',(.42,.27)),('nerol.oxide','Nerol oxide',(.30,-.43)),('ethanol..v.v.','EtOH',(-.26,-.27)),('ethyl_acetate','Ethyl acetate',(-.06,-.70)),('ethyl.phenylacetate','Ethyl phenylacetate',(-.31,.24))]
    for field,label,pos in labels:
        col=palette[int(nd.set_index('field').loc[field,'module'])]
        ax.scatter(*xy[field],s=24,fc='white',ec=col,lw=.7,zorder=5)
        ax.annotate(label,xy[field],xytext=pos,ha='center',va='center',fontsize=5.9,color=INK,
            bbox=dict(fc='white',ec='none',alpha=.92,pad=.5),
            arrowprops=dict(arrowstyle='-',lw=.4,color='#9BA9B4',shrinkA=2,shrinkB=3),zorder=7)
    f.text(.045,.973,'a  共变结构',fontsize=7.5)
    f.text(.61,.973,'b  模块稳定性',fontsize=7.5)
    # Shared module colours trace six paired pieces of substitution evidence.
    bx=f.add_axes([.78,.31,.165,.535]);yy=np.arange(6)[::-1];labels=['麦芽味','酯类总香','乳酸感','乙酸乙酯味','余味','香蕉酯香']
    bx.axvline(.8,color='#9DAAB5',lw=.5,ls=(0,(2,2)),zorder=0)
    for y,r,lab in zip(yy,fr.itertuples(),labels):
        col=palette[int(r.module)];bx.plot([r.largest_member_frequency,r.module_union_frequency],[y,y],c=col,lw=1.6,alpha=.7,zorder=2)
        bx.scatter(r.largest_member_frequency,y,s=17,fc='white',ec=col,lw=.7,zorder=3);bx.scatter(r.module_union_frequency,y,s=17,c=col,ec='white',lw=.25,zorder=4)
        bx.scatter(r.uniform_three_draw_reference,y,s=11,marker='|',c='#AAB5BE',lw=.65,zorder=3)
        bx.text(-.08,y,lab,transform=bx.get_yaxis_transform(),ha='right',va='center',fontsize=6.3,color=INK)
        bx.text(1.04,y,f'{100*r.module_union_frequency:.1f}',transform=bx.get_yaxis_transform(),ha='left',va='center',fontsize=6.0,color=col)
        bx.text(-.10,y+.29,f'M{r.module}',transform=bx.get_yaxis_transform(),ha='right',va='center',fontsize=5.5,color=col)
    bx.set(xlim=(0,1.02),ylim=(-.55,5.6),xticks=[0,.5,1],xticklabels=['0','50','100'],yticks=[],xlabel='Top 3 频率 (%)');clean(bx)
    for side in ['left']:bx.spines[side].set_visible(False)
    f.legend(handles=[Line2D([],[],marker='o',ls='',mfc='white',mec=SECONDARY,ms=3,label='单成员'),Line2D([],[],marker='o',ls='',mfc=TEAL,mec='white',ms=3,label='模块并集'),Line2D([],[],marker='|',ls='',c=GRAY,ms=4,label='随机集合')],loc='upper right',bbox_to_anchor=(.97,.924),frameon=False,ncol=1,fontsize=5.6,handlelength=.7,labelspacing=.35)
    # Compact bookkeeping replaces 101 decorative dots and a fragmented graph row.
    cx=f.add_axes([.635,.105,.32,.09]);cx.axis('off');values=[99,31,101];cols=[BLUE,'#B8C3CC','#E4E9ED'];x=0
    for val,col in zip(values,cols):cx.add_patch(Rectangle((x,0.48),val/231,.24,fc=col,ec='white',lw=.6));x+=val/231
    for x,t,c in [(.0,'99 主连通',BLUE),(.42,'31 其他连通',SECONDARY),(.0,'101 零度字段',SECONDARY)]:cx.text(x,.19 if x else .19,t,transform=cx.transAxes,fontsize=5.6,color=c,va='center')
    # Move third census label to separate line, never overlay.
    cx.texts[-1].set_position((0,-.25))
    f.text(.645,.228,'231 字段 · 266 边',fontsize=6.4,color=INK)
    f.legend(handles=[Line2D([],[],c='#9DAAB5',lw=.6,label='正相关'),Line2D([],[],c=RED,lw=.6,ls='--',label='负相关'),Line2D([],[],marker='o',ls='',c=GRAY,ms=3,label='面积：加权度')],loc='lower left',bbox_to_anchor=(.035,.006),frameon=False,ncol=3,fontsize=5.8,handlelength=1.2,columnspacing=1.0)
    pd.DataFrame(records).to_csv(DATA/'fig06_circular_node_index.csv',index=False)
    caption=('化学共变结构与模块稳定性。a详细显示99字段主连通分量：同心分组弧标示冻结模块，外轨为节点加权度，节点面积随加权度增加；圆周排序与边的曲线只用于展示，不表示化学距离或系统发育。白芯圈与细引线标注6个真实关键化合物；2-AF为2-乙酰呋喃，2,6-DMP为2,6-二甲基吡嗪，EtOH为乙醇。另12个连通分量共31字段、101个零度字段以清点条汇总，完整130有边字段、266边和3条负边保留于底表。b为6组已存同模块最稳定单成员、模块并集及同大小随机集合的Top3频率，虚线为80%，右列是模块并集百分数；模块色与a一致。附录B的150项候选频率完整保留于补图S6，不扩造模块配对数据。')
    save(f,6,caption,[pn,pe,pf],dict(main_component_nodes=99,total_nodes=231,connected_nodes=130,other_components=12,other_connected_nodes=31,isolated_nodes=101,total_edges=266,main_component_edges=len(ee),negative_edges=3,shared_module_index=True,new_inference=False))
    return dict(main_nodes=99,total_connected=130,total_edges=266,negative_edges=3,module_pairs=6)

def figure7():
    ps=P/'fig07_all250_style_scatter.csv';pp=P/'11_within_style_chemical_evidence.csv';pa=P/'fig07_descriptive_association_audit.csv'
    raw=pd.read_csv(ps);d=pd.read_csv(pp);audit=pd.read_csv(pa)
    assert len(raw)==500 and len(d)==8
    np.testing.assert_allclose(d.covariance_total,d.covariance_within+d.covariance_between,atol=2e-15)
    f=figure(7);f.set_size_inches(6.7,3.8)
    f.text(.065,.967,'a  风格分层的实测关联',fontsize=7.5)
    # Two identical units: all observations are retained; no overlapping ellipse clouds.
    for j,(field,title)in enumerate([('lactic_acid.mg.L.','乳酸 → 乳酸感'),('X4VG','4-VG → 酚香')]):
        ax=f.add_axes([.075,.56-j*.425,.31,.30]);v=raw[raw.field.eq(field)];r=audit[audit.field.eq(field)].iloc[0]
        for s,c,al in [('__others__','#BDC6CE',.65),('Lambic',RED,.85),('Kriek',PURPLE,.85)]:
            g=v[~v['style'].isin(['Lambic','Kriek'])]if s=='__others__'else v[v['style'].eq(s)]
            ax.scatter(g.x_normal,g.y_normal,s=5.5 if s=='__others__'else 8.5,c=c,alpha=al,ec='white',lw=.15,zorder=2 if s=='__others__'else 3)
        xx=np.array([-2.7,2.7]);ax.plot(xx,r.slope_total*xx,c=INK,lw=.8,ls=(0,(3,2)),zorder=4);ax.plot(xx,r.slope_within*xx,c=TEAL,lw=1.2,zorder=5)
        ax.set(xlim=(-2.8,2.8),ylim=(-2.8,2.8),xticks=[-2,0,2],yticks=[-2,0,2],ylabel='感官正态分数',xlabel='化学正态分数'if j else'');clean(ax)
        ax.text(.02,.95,title,transform=ax.transAxes,va='top',fontsize=6.6,bbox=dict(fc='white',ec='none',alpha=.8,pad=.4))
        ax.text(.98,.03,f'{r.r_total:.3f} → {r.r_within:.3f}',transform=ax.transAxes,ha='right',va='bottom',fontsize=6.2,color=TEAL,bbox=dict(fc='white',ec='none',alpha=.8,pad=.3))
    f.legend(handles=[Line2D([],[],marker='o',ls='',c=RED,ms=3,label='Lambic'),Line2D([],[],marker='o',ls='',c=PURPLE,ms=3,label='Kriek'),Line2D([],[],marker='o',ls='',c='#BDC6CE',ms=3,label='其余20风格')],loc='upper left',bbox_to_anchor=(.061,.929),frameon=False,ncol=3,fontsize=5.5,handletextpad=.3,columnspacing=.7)
    # Eight shared-index rows connect correlation and selection frequency evidence.
    y=np.arange(8)[::-1];lab=['乳酸→乳酸感','乳酸→酸味','4-VG→酚香','乙醇→酒精感','2-AF→麦芽香','2,6-DMP→麦芽味','DMS→DMS感','双乙酰→双乙酰感']
    f.text(.455,.967,'b  同一配对的关联与入选证据',fontsize=7.5)
    axs=[]
    for k,(c1,c2,title,xmin,xmax,ticks)in enumerate([('correlation_total','correlation_within','相关系数',-.16,.9,[0,.4,.8]),('total_top3','within_top3','Top 3 频率',-.04,1.04,[0,.5,1])]):
        ax=f.add_axes([.65+.19*k,.16,.145,.66]);axs.append(ax)
        for i in range(8):
            if i%2==0:ax.axhspan(y[i]-.47,y[i]+.47,color='#F4F7F8',lw=0,zorder=0)
            ax.plot([d.iloc[i][c1],d.iloc[i][c2]],[y[i],y[i]],c='#BCC7CF',lw=1,zorder=2)
            ax.scatter(d.iloc[i][c1],y[i],s=17,fc='white',ec=BLUE,lw=.65,zorder=3)
            ax.scatter(d.iloc[i][c2],y[i],s=17,fc=TEAL,ec='white',lw=.25,zorder=4)
        ax.axvline(0,c='#D0D8DE',lw=.4,zorder=1)
        if k:ax.axvline(.8,c=GRAY,lw=.45,ls=(0,(2,2)),zorder=1)
        ax.set(xlim=(xmin,xmax),ylim=(-.65,7.65),xticks=ticks,yticks=[],xlabel=title);clean(ax)
        ax.spines['left'].set_visible(False)
        if k:ax.set_xticklabels(['0','50','100']);ax.set_xlabel(title+' (%)')
        if not k:
            for yi,txt in zip(y,lab):ax.text(-.10,yi,txt,transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=5.85,color=INK)
    f.legend(handles=[Line2D([],[],marker='o',ls='',mfc='white',mec=BLUE,ms=3,label='总体'),Line2D([],[],marker='o',ls='',mfc=TEAL,mec='white',ms=3,label='风格内')],loc='upper right',bbox_to_anchor=(.975,.922),frameon=False,ncol=2,fontsize=5.8,handletextpad=.4,columnspacing=1.2)
    f.legend(handles=[Line2D([],[],c=INK,lw=.8,ls='--',label='总体斜率'),Line2D([],[],c=TEAL,lw=1.2,label='风格内斜率')],loc='lower left',bbox_to_anchor=(.061,.005),frameon=False,ncol=2,fontsize=5.7,handlelength=1.5,columnspacing=.9)
    f.text(.455,.055,'250款啤酒 · 22风格',fontsize=6.1,color=SECONDARY)
    caption=('风格混杂与风格内关联。a两组严格同尺度散点均展示250款啤酒，Lambic和Kriek着色，其余20风格用灰色。虚线为总体描述性斜率，实线为去风格均值后的合并描述性斜率；图内数字依次为总体与风格内正态分数相关，两种斜率均在图示标准化坐标中通过原点，不是因果效应。b八组固定化学—感官配对共用行索引：左轨比较总体/风格内相关，右轨比较相同400组风格分层重采样、固定α=3的Top3入选频率，虚线为80%。空心表示总体、实心表示风格内。完整协方差分解和全化学模型折外增量及原2000次配对抽样95%条件区间保留于补图S7。')
    save(f,7,caption,[ps,pp,pa],dict(n_beers=250,n_styles=22,scatter_points=500,chemical_pairs=8,ellipse_count=0,shared_pair_index=True,covariance_identity_max_error=float(np.max(np.abs(d.covariance_total-d.covariance_within-d.covariance_between))),new_inference=False))
    # The existing complete evidence is preserved as supplementary material.
    source=OLD/'figures/supplement_S7_original_evidence.pdf'
    if source.exists():shutil.copyfile(source,OUT/source.name)
    source=OLD/'figures/supplement_S6_reported_top3.pdf'
    if source.exists():shutil.copyfile(source,OUT/source.name)
    return dict(scatter_points=500,styles=22,paired_rows=8,ellipse_count=0)

if __name__=='__main__':
    setup();a=figure6();b=figure7();(BASE/'design_network_audit.json').write_text(json.dumps(dict(status='PASS',fig6=a,fig7=b,new_model_fits=0,new_resampling=0),indent=2));print('Designed figures 6 and 7.')
