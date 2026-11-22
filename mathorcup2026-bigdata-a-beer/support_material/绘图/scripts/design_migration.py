"""Shared-index compositions for frozen migration results. No fits or resampling."""
from repair_common import *
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm,ListedColormap
from matplotlib.patches import Polygon,Rectangle
from matplotlib.lines import Line2D

DIVERGE=LinearSegmentedColormap.from_list('migration_signed',['#476F96','#A8C7D4','#FAFBFC','#EAC0B4','#BD7069'])
RATE=LinearSegmentedColormap.from_list('migration_rate',['#F2F6F5','#CCE0DC','#70A9A1','#2F716F'])
BEERS=np.array([28,81,90,96,119,124,139,185,209,218,230])

def geometry(ax):
    a=np.deg2rad(20);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]);t=np.linspace(0,2*np.pi,361)
    ell=(R@np.vstack([1.9*np.cos(t),1.15*np.sin(t)])).T;inside=.9*ell
    def clipped(poly,n,c):
        out=[]
        for p,q in zip(poly,poly[1:]+poly[:1]):
            u=np.dot(n,p)-c;v=np.dot(n,q)-c
            if u<=0:out.append(p)
            if (u<=0)!=(v<=0):out.append((np.array(p)+u/(u-v)*(np.array(q)-np.array(p))).tolist())
        return out
    feasible=clipped(inside.tolist(),[-1,0],.15);feasible=clipped(feasible,[-.45,-1],.40)
    ax.plot(*ell.T,c=GRAY,lw=.55,ls=(0,(2,2)))
    ax.plot(*inside.T,c=BLUE,lw=.65)
    ax.add_patch(Polygon(feasible,fc='#DCECE7',ec=TEAL,lw=.35,hatch='///'))
    xx=np.array([-2.1,2.1]);ax.plot(xx,-.45*xx-.4,c=RED,lw=.5,ls=(0,(3,2)));ax.axvline(-.15,c=TEAL,lw=.4)
    origin=R@np.array([1.9*np.cos(np.deg2rad(218)),1.15*np.sin(np.deg2rad(218))]);good=np.array([.83,.18]);bad=np.array([1.77,1.2])
    for pt,col,ls in [(good,BLUE,'-'),(bad,RED,'--')]:
        ax.annotate('',pt,origin,arrowprops=dict(arrowstyle='-|>',lw=.6,color=col,linestyle=ls,mutation_scale=5))
    ax.scatter(*origin,s=20,fc='white',ec=BLUE,lw=.6,zorder=5);ax.scatter(*origin,s=5,fc=BLUE,ec='none',zorder=6)
    ax.scatter(*good,s=13,c=BLUE,marker='D',ec='white',lw=.3,zorder=6);ax.scatter(*bad,s=15,c=RED,marker='x',lw=.7,zorder=6)
    ax.text(-1.9,.80,r'$D_0$',fontsize=6,color=SECONDARY);ax.text(-1.9,.46,r'$0.9D_0$',fontsize=6,color=BLUE)
    ax.text(origin[0],origin[1]-.35,'源谱',fontsize=6,ha='center')
    ax.text(.90,.44,'联合',fontsize=5.8,color=BLUE);ax.text(1.15,1.57,'仅分类',fontsize=5.8,color=RED)
    ax.set(xlim=(-2.15,2.2),ylim=(-1.65,1.9));ax.set_aspect('equal');ax.axis('off')

def design10():
    p=DATA/'fig10_actual_selected_adjustments.csv';meta=DATA/'fig10_field_selection.csv';q=pd.read_csv(p);fields=pd.read_csv(meta)
    groups=[('酯类',['ethyl_acetate','isopentyl_acetate','isobutyl_acetate','ethyl.butyrate','vinyl.caprylate','ethyl..E..cinnamate'],TEAL),('理化指标',['protein.g.L.','lactic_acid.mg.L.','ethanol..v.v.','glycerol','color.EBC.'],PURPLE),('其他字段',['SO2.mg.L.','n.hexanol','styrene','germacrene.D'],RED)]
    order=[x for _,xs,_ in groups for x in xs]
    labels=['乙酸乙酯','乙酸异戊酯','乙酸异丁酯','丁酸乙酯','辛酸乙烯酯','肉桂酸乙酯','蛋白质','乳酸','乙醇','甘油','色度','SO₂','正己醇','苯乙烯','Germacrene D']
    values=q.pivot(index='beer',columns='field',values='delta_control_z').loc[BEERS,order].to_numpy();limit=3.5
    # Values are exactly the frozen fixed normal-map differences.
    assert values.shape==(11,15) and np.max(np.abs(values))<limit and np.isfinite(values).all()
    full=np.load(DATA/'fig10_all225_delta_control_z.npz',allow_pickle=False)
    assert np.array_equal(full['beer'],BEERS)
    ix=[full['features'].tolist().index(field)for field in order]
    np.testing.assert_allclose(values,full['delta'][:,ix],rtol=0,atol=1e-14)
    f=figure(10);f.set_size_inches(6.7,3.7)
    f.text(.035,.94,'a  联合约束',fontsize=7.2)
    ax=f.add_axes([.025,.45,.245,.40]);geometry(ax)
    f.text(.052,.37,'B = 100',fontsize=10,color=BLUE)
    f.text(.052,.315,'η = 2   ·   11款来源',fontsize=6.7,color=SECONDARY)
    f.text(.052,.255,'225个直接字段',fontsize=6.5,color=SECONDARY)
    f.text(.052,.205,'15个高频改动字段 →',fontsize=6.5,color=SECONDARY)
    f.text(.052,.105,'几何机制示意',fontsize=5.9,color=SECONDARY)
    ax=f.add_axes([.345,.25,.61,.57]);im=ax.pcolormesh(np.arange(16)-.5,np.arange(12)-.5,values,cmap=DIVERGE,norm=TwoSlopeNorm(0,-limit,limit),edgecolors='white',linewidth=.45,shading='flat')
    ax.set(xlim=(-.5,14.5),ylim=(10.5,-.5),xticks=np.arange(15),xticklabels=labels,yticks=np.arange(11),yticklabels=BEERS)
    ax.tick_params(length=0,pad=3,labelsize=5.7);plt.setp(ax.get_xticklabels(),rotation=48,ha='right',rotation_mode='anchor')
    ax.set_ylabel('源啤酒编号',fontsize=6.3,labelpad=7)
    for s in ax.spines.values():s.set_visible(True);s.set_color('#9EA9B1');s.set_linewidth(.4)
    # Major effects are localized; all exact values remain in the accompanying table.
    for i,j in zip(*np.where(np.abs(values)>=2.5)):
        ax.text(j,i,f'{values[i,j]:+.1f}',ha='center',va='center',fontsize=5.2,color='white'if abs(values[i,j])>2.5 else '#222222')
    for split in [5.5,10.5]:ax.axvline(split,c='white',lw=2)
    top=f.add_axes([.345,.84,.61,.047]);top.set(xlim=(-.5,14.5),ylim=(0,1));top.axis('off')
    start=0
    for label,fs,col in groups:
        end=start+len(fs);top.add_patch(Rectangle((start-.48,.12),len(fs)-.04,.10,fc=col,ec='none'));top.text((start+end-1)/2,.56,label,ha='center',va='center',fontsize=6.1,color=col);start=end
    f.text(.345,.94,'b  冻结候选的控制坐标变化',fontsize=7.2)
    cb=f.colorbar(im,cax=f.add_axes([.47,.065,.36,.018]),orientation='horizontal',ticks=[-3,-1.5,0,1.5,3]);cb.ax.tick_params(labelsize=5.6,length=1.5,pad=1.5);cb.outline.set_linewidth(.35);cb.set_label('Δz',fontsize=6.2,labelpad=1)
    caption='联合约束机制与冻结候选的真实调整。a为二维几何机制示意，非真实可行域投影。b以同一来源顺序展开B=100、η=2的11份冻结候选，在15个高频改动字段上显示固定经验正态控制坐标Δz；按酯类、理化指标与其他字段分区仅作阅读组织。正值暖色、负值冷色，零值近白；仅对|Δz|≥2.5处标注数值。字段仍为原来按改动频次、平均绝对Δz选出的15项，完整225维与精确数值保留于底表。Δz不是原生单位增量或物理干预效应。'
    save(f,10,caption,[p,meta,DATA/'fig10_all225_delta_control_z.npz'],dict(source_count=11,direct_fields=225,display_fields=15,display_order_fields=order,source_order=BEERS.tolist(),color_limits=[-limit,limit],geometry_is_schematic=True))
    return dict(shape=list(values.shape),max_abs_delta=float(abs(values).max()),frozen_values_equal=True)

def design11():
    p=DATA/'fig11_certificate_phase_space.csv';d=pd.read_csv(p).set_index('beer').loc[BEERS].reset_index();x=100*d.unavoidable_distance_increase_lower_bound.to_numpy();y=d.ellipsoid_single_halfspace_min_upper_margin.to_numpy();k=d.support_size.to_numpy()
    assert (x>0).all()and(y<0).all()
    f=figure(11);f.set_size_inches(6.7,3.45)
    f.text(.035,.94,'a  固定支持的两类排除证书',fontsize=7.2)
    a=f.add_axes([.045,.19,.27,.65]);b=f.add_axes([.405,.19,.27,.65],sharey=a)
    rows=np.arange(11)
    for ax in [a,b]:
        ax.set(ylim=(10.5,-.5),yticks=rows,yticklabels=[]);ax.tick_params(axis='y',length=0);clean(ax,'x')
        for i in range(11):
            if i%2==0:ax.axhspan(i-.48,i+.48,color='#F6F8F9',zorder=0,lw=0)
        ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color('#A6B0B8');ax.tick_params(axis='y',length=0)
    a.barh(rows,x,height=.34,color=RED,alpha=.8,lw=0,zorder=3);a.scatter(x,rows,s=9,c=RED,ec='white',lw=.3,zorder=4)
    b.barh(rows,y,height=.34,color=TEAL,alpha=.8,lw=0,zorder=3);b.scatter(y,rows,s=9,c=TEAL,ec='white',lw=.3,zorder=4)
    for i in range(11):
        a.annotate(f'{x[i]:.1f}',(x[i],i),xytext=(-4,0),textcoords='offset points',ha='right',va='center',fontsize=5.8,color=INK)
        b.annotate(f'{y[i]:.2f}',(y[i],i),xytext=(4,0),textcoords='offset points',ha='left',va='center',fontsize=5.8,color=INK)
    a.set(xlim=(100,0),xticks=[100,50,0],xlabel='距离增幅下界 (%)')
    b.set(xlim=(0,-6.7),xticks=[0,-2,-4,-6],xlabel='分类余量上界')
    a.axvline(0,c=RED,lw=.6);b.axvline(0,c=RED,lw=.6)
    center=f.add_axes([.315,.19,.09,.65]);center.set(xlim=(0,1),ylim=(10.5,-.5));center.axis('off')
    for i,r in enumerate(d.itertuples()):
        center.text(.38,i,str(int(r.beer)),ha='center',va='center',fontsize=6.1,color=INK);center.text(.86,i,str(int(r.support_size)),ha='center',va='center',fontsize=5.7,color=SECONDARY)
    f.text(.349,.865,'来源',ha='center',fontsize=6.2);f.text(.392,.865,'k',ha='center',fontsize=6.2,color=SECONDARY)
    f.text(.765,.94,'b  证书摘要',fontsize=7.2)
    f.text(.765,.81,'11 / 11',fontsize=16,color=BLUE)
    f.text(.765,.745,'返回支持内均被排除',fontsize=6.3,color=SECONDARY)
    c=f.add_axes([.77,.22,.205,.40]);c.axvline(0,color=RED,lw=.5,ls=(0,(2,2)));c.axhline(0,color=RED,lw=.5,ls=(0,(2,2)))
    c.scatter(x,y,s=9+1.7*k,c=BLUE,alpha=.78,ec='white',lw=.35,zorder=4)
    for beer,offset in [(28,(5,2)),(124,(-4,4))]:
        i=np.where(BEERS==beer)[0][0];c.annotate(str(beer),(x[i],y[i]),xytext=offset,textcoords='offset points',ha='left'if beer==28 else'right',fontsize=5.6,color=BLUE)
    c.set(xlim=(-4,94),ylim=(-6.2,.55),xticks=[0,40,80],yticks=[-6,-3,0],xlabel='下界 (%)',ylabel='上界');c.tick_params(labelsize=5.4);clean(c)
    f.text(.765,.105,'点面积 ∝ 支持基数 k',fontsize=5.8,color=SECONDARY)
    caption='分类最少编辑返回支持的双证书。a以相同11款来源顺序对齐两类边界：左为满足分类条件时不可避免的目标距离增幅下界，右为源距离不增椭球内分类余量的上界；中央列示支持基数k。所有距离增幅下界为正、余量上界为负，分别排除对应约束集合内的可行连续值。b将同一对界放入摘要坐标，点面积随k增大；两轴来自不同约束集合，不构成实际联合可行域投影或排除锥。结论限于求解器返回的固定支持，非穷举全部同大小支持。数值不是置信区间。'
    save(f,11,caption,[p],dict(certificate_cases=11,source_order=BEERS.tolist(),all_distance_lower_bounds_positive=True,all_margin_upper_bounds_negative=True,bounds_are_not_ci=True,no_exclusion_cones=True))
    return dict(cases=11,min_lb_percent=float(x.min()),max_ub=float(y.max()))

def design12():
    p=DATA/'fig12_real_refit_rasters.npz';z=np.load(p,allow_pickle=False);cube=z['joint_pass'];beers=z['beer'];assert np.array_equal(beers,BEERS) and cube.shape==(4,11,100)
    rates=cube.mean(axis=2)*100;means=cube.mean(axis=(1,2))*100
    source=OLD/'plot_data/fig12_frozen_means_recomputed.csv';frozen=pd.read_csv(source)
    np.testing.assert_allclose(means,frozen.joint_all11.to_numpy()*100,atol=1e-12)
    f=figure(12);f.set_size_inches(6.7,3.65)
    # One source index runs through the four repeated blocks and the four summary tracks.
    left=.075;bottom=.225;height=.555;w=.1575;gap=.008
    colors=[BLUE,TEAL,BLUE,TEAL];labels=['低余量','η = 1','低余量','η = 2']
    for j in range(4):
        x=left+j*(w+gap);ax=f.add_axes([x,bottom,w,height])
        # Merge contiguous passing cells into vector runs to avoid PDF stitching seams.
        ax.add_patch(Rectangle((.5,-.5),100,11,fc='#F3F6F7',ec='none'))
        for ii in range(11):
            edges=np.diff(np.r_[False,cube[j,ii],False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
            for lo,hi in zip(starts,ends):ax.add_patch(Rectangle((lo+.5,ii-.5),hi-lo,1,fc='#428D88',ec='none'))
        ax.grid(False,which='both')
        ax.set(xlim=(.5,100.5),ylim=(10.5,-.5),xticks=[1,50,100],yticks=range(11),yticklabels=BEERS if j==0 else['']*11)
        ax.tick_params(length=0,pad=3,labelsize=5.9)
        ax.get_xticklabels()[0].set_ha('left');ax.get_xticklabels()[-1].set_ha('right')
        for h in np.arange(.5,11,1):ax.axhline(h,c='white',lw=.65)
        for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#B4BEC5');spine.set_linewidth(.35)
        if j==0:ax.set_ylabel('源啤酒编号',fontsize=6.4,labelpad=5)
        f.text(x+w/2,.805,labels[j],ha='center',fontsize=6.6,color=colors[j])
        f.text(x+w/2,.855,f'{means[j]:.2f}%',ha='center',fontsize=9.2,color=colors[j])
        f.add_artist(Line2D([x,x+w],[.885,.885],transform=f.transFigure,lw=1.2,c=colors[j]))
    f.text(left+(2*w+gap)/2,.94,'a  B = 60',ha='center',fontsize=7.2)
    f.text(left+2*(w+gap)+(2*w+gap)/2,.94,'b  B = 100',ha='center',fontsize=7.2)
    x=.765;ra=f.add_axes([x,bottom,.20,height]);im=ra.pcolormesh(np.arange(5)-.5,np.arange(12)-.5,rates.T,cmap=RATE,vmin=0,vmax=100,edgecolors='white',linewidth=.65)
    for i in range(11):
        for j in range(4):ra.text(j,i,f'{rates[j,i]:.0f}',ha='center',va='center',fontsize=5.9,color='white'if rates[j,i]>=70 else INK)
    ra.set(xlim=(-.5,3.5),ylim=(10.5,-.5),xticks=range(4),xticklabels=['60\n低','60\nη1','100\n低','100\nη2'],yticks=[]);ra.tick_params(length=0,pad=3,labelsize=5.3);plt.setp(ra.get_xticklabels(),ha='center')
    for spine in ra.spines.values():spine.set_visible(True);spine.set_color('#B4BEC5');spine.set_linewidth(.35)
    f.text(x+.10,.94,'c  来源通过率',ha='center',fontsize=7.2)
    f.text(x+.10,.855,'100次 · %',ha='center',fontsize=6.8,color=SECONDARY)
    f.text(left+(4*w+3*gap)/2,.13,'共同重拟合场景 · 原序号',ha='center',fontsize=6.3,color=SECONDARY)
    f.add_artist(Rectangle((left,.065),.014,.02,transform=f.transFigure,fc='#428D88',ec='none'));f.text(left+.022,.067,'联合通过',fontsize=6,color=SECONDARY)
    f.add_artist(Rectangle((left+.14,.065),.014,.02,transform=f.transFigure,fc='#F3F6F7',ec='#B4BEC5',lw=.35));f.text(left+.162,.067,'未通过',fontsize=6,color=SECONDARY)
    f.text(.965,.067,'11来源 × 100场景 × 4策略',ha='right',fontsize=6,color=SECONDARY)
    caption='同预算迁移的逐场景稳定性。四个二元光栅以共同的11来源行索引对齐，每块保留100次共同重拟合场景的原始顺序。a为B=60的低余量与η=1策略；b为B=100的低余量与η=2策略。深青色为同时满足风格分类与感官改善的联合通过，浅色为未通过；上方百分数为全部11×100结果的等权均值。c为对应来源的100次通过率，列顺序与四光栅相同。所有4400个二元结果均来自冻结候选的同一验证阶段，不是实品干预成功率；较早独立200次全预算试验保留于补图S12，未混入本图。逐场景汇总400值另存底表，未新增置信带。'
    save(f,12,caption,[p,source,DATA/'fig12_per_refit_source_mean.csv'],dict(source_count=11,refits=100,strategies=4,binary_cells=4400,means_percent=means.tolist(),source_order=BEERS.tolist(),refit_order_preserved=True,no_new_ci=True))
    return dict(binary_cells=4400,mean_percent=means.tolist(),all_binary_equal=True)

if __name__=='__main__':
    setup();result=dict(fig10=design10(),fig11=design11(),fig12=design12(),status='PASS',new_model_fits=0,new_bootstrap_resamples=0)
    (BASE/'migration_design_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
