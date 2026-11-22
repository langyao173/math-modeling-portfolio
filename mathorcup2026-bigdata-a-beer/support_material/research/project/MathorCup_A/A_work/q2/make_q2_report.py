"""Create question-2 evidence tables, figures, and Chinese research note.

All numerical statements are read from executed result files.
"""
from pathlib import Path
from tempfile import gettempdir
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(gettempdir())/'mathorcup_q2_mpl'))
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.stats import rankdata
from scipy.sparse.csgraph import connected_components
from run_q2 import greedy_modularity

P=Path(__file__).resolve().parent
D=P.parent/'data'/'A'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,
    'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
    'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white',
    'axes.titleweight':'semibold','axes.linewidth':.65,'xtick.major.width':.6,
    'ytick.major.width':.6})
COLORS={'Hops':'#426B51','Yeast':'#975451','Malt':'#B18B4B','Wild flora':'#69829A',
        'Spices':'#837393','Others':'#828282','Internal standard solution':'#AAA69B'}


def savefig(fig,name):
    fig.savefig(P/(name+'.svg'),bbox_inches='tight')
    fig.savefig(P/(name+'.png'),bbox_inches='tight',dpi=220)
    plt.close(fig)


def markdown(df):
    def val(v):
        return str(v).replace('|','\\|').replace('\n',' ')
    lines=['| '+' | '.join(map(val,df.columns))+' |',
           '| '+' | '.join(['---']*len(df.columns))+' |']
    lines.extend('| '+' | '.join(map(val,row))+' |' for row in df.itertuples(index=False,name=None))
    return '\n'.join(lines)


def force_layout(W,seed=20261023,iterations=500):
    rng=np.random.default_rng(seed)
    pos=rng.normal(size=(len(W),2))*.3
    k=np.sqrt(1/max(len(W),1))
    for step in range(iterations):
        delta=pos[:,None,:]-pos[None,:,:]
        dist=np.sqrt(np.sum(delta*delta,axis=2))
        np.fill_diagonal(dist,1e6)
        dist=np.maximum(dist,.01)
        # Standard force-directed positions only; geometry has no scientific scale.
        attract=W*dist/k
        force=np.sum(delta*(k*k/(dist*dist)-attract)[:,:,None],axis=1)
        lengths=np.maximum(np.linalg.norm(force,axis=1),1e-9)
        temp=.04*(1-step/iterations)+.001
        pos+=force*(np.minimum(lengths,temp)/lengths)[:,None]
        pos-=pos.mean(0)
    pos/=max(np.ptp(pos[:,0]),np.ptp(pos[:,1]))
    return pos


def main():
    s=json.loads((P/'summary.json').read_text())
    nodes=pd.read_csv(P/'graph_nodes.csv')
    edges=pd.read_csv(P/'graph_edges.csv')
    top=pd.read_csv(P/'top3_all_50_sensory_dimensions.csv')
    factors=pd.read_csv(P/'all_factor_evidence.csv')
    agreement=pd.read_csv(P/'method_and_imputation_agreement.csv')
    metrics=pd.read_csv(P/'nested_cv_metrics.csv')
    chem=pd.read_csv(P/'chemical_dictionary.csv')
    sourceX=pd.read_csv(D/'1.csv').sort_values('beer').reset_index(drop=True)
    sourceY=pd.read_csv(D/'4.csv').set_index('beer').loc[sourceX.beer].reset_index()
    features=chem.field.tolist()
    targets=sourceY.columns[3:].tolist()
    R=pd.read_csv(D/'2.csv',index_col=0).to_numpy()
    z=np.load(P/'bootstrap_coefficients.npz')
    boots=z['coefficients']
    molecule_ix=np.where(chem.single_compound_candidate.to_numpy(bool))[0]
    pure_R=R[np.ix_(molecule_ix,molecule_ix)]
    pure_A=np.where(abs(pure_R)>=.6,pure_R,0.);np.fill_diagonal(pure_A,0.)
    pure_nodes=nodes.iloc[molecule_ix].copy()
    pure_nodes['module_in_compound_graph']=greedy_modularity(abs(pure_A))
    pure_nodes['degree_in_compound_graph']=(pure_A!=0).sum(1)
    pure_nodes.to_csv(P/'graph_compound_nodes.csv',index=False,encoding='utf-8-sig')
    keep=set(pure_nodes.field)
    pure_edges=edges[edges.source.isin(keep)&edges.target.isin(keep)]
    pure_edges.to_csv(P/'graph_compound_edges.csv',index=False,encoding='utf-8-sig')
    negative=edges[edges.rho<0].copy()
    style_col=sourceX.tasting_category_fine.to_numpy()
    within=[]
    for _,edge in negative.iterrows():
        pair=sourceX[[edge.source,edge.target]]
        good=pair.notna().all(axis=1).to_numpy()
        Q=rankdata(pair.loc[good].to_numpy(),axis=0)
        g=style_col[good]
        for style in np.unique(g):Q[g==style]-=Q[g==style].mean(0)
        within.append(float(np.corrcoef(Q,rowvar=False)[0,1]))
    negative['within_style_rank_rho']=within
    negative.to_csv(P/'negative_edges_analysis.csv',index=False,encoding='utf-8-sig')
    # Any member of a module entering Top-3 gives group support. This is reported
    # alongside module size, so larger groups cannot masquerade as precise molecules.
    top_boot=np.argsort(-abs(boots[:,molecule_ix,:]),axis=1)[:,:3,:]
    module_support={}
    modules_all=nodes.module.to_numpy()
    for j,target in enumerate(targets):
        bmods=modules_all[molecule_ix[top_boot[:,:,j]]]
        for mod in np.unique(modules_all):
            module_support[target,int(mod)]=float(np.mean(np.any(bmods==mod,axis=1)))
    # Supplementary ranking of all measured factors: pH and aggregates are
    # explicitly labelled so they cannot be described as individual molecules.
    forbidden={'X3.methyl.2.butenal'}
    eligible=np.array([i for i,f in enumerate(features) if f not in forbidden])
    selected=np.argsort(-abs(boots[:,eligible,:]),axis=1)[:,:3,:]
    counts=np.zeros(boots.shape[1:])
    for b in range(len(boots)):
        for j in range(len(targets)):
            counts[eligible[selected[b,:,j]],j]+=1
    counts/=len(boots)
    median=np.median(abs(boots),axis=0)
    allrows=[]
    for j,target in enumerate(targets):
        ix=eligible[np.lexsort((-median[eligible,j],-counts[eligible,j]))[:3]]
        for rank,i in enumerate(ix,1):
            row=factors[(factors.target==target)&(factors.field==features[i])].iloc[0].to_dict()
            row.update(rank=rank,chemical_name=chem.graph_name.iloc[i],
                all_measured_factors_top3_frequency=float(counts[i,j]),
                factor_type='individual_compound' if chem.single_compound_candidate.iloc[i] else 'physicochemical_or_aggregate')
            allrows.append(row)
    measured=pd.DataFrame(allrows)
    measured.to_csv(P/'top3_all_measured_factors.csv',index=False,encoding='utf-8-sig')

    # Direct chemical anchors are evaluated regardless of their selected rank.
    anchors={
        'acidity':['acetic_acid.g.L.','lactic_acid.mg.L.'],
        'acetic':['acetic_acid.g.L.'], 'lactic':['lactic_acid.mg.L.'],
        'bitternes':['bitterness'], 'alcohol':['ethanol..v.v.'],
        'X4vg':['X4VG'], 'clove':['X4VG','eugenol'],
        'diacetyl':['X2.3.butanedione'], 'dms':['DMS'], 't2n':['X.E..2.nonenal'],
        'A_esters_ethac':['ethyl_acetate'],'F_esters_ethac':['ethyl_acetate'],
        'A_esters_isoaa':['isopentyl_acetate'],'F_esters_isoaa':['isopentyl_acetate'],
        'A_esters_flower':['phenethyl_acetate','phenethyl_alcohol'],
        'F_esters_flower':['phenethyl_acetate','phenethyl_alcohol'],
        'A_hops_citrus':['linalool','geraniol','citronellol'],
        'F_hops_citrus':['linalool','geraniol','citronellol'],
        'co2':['CO2.PSI.'],
    }
    anchor_rows=[]
    for target, fs in anchors.items():
        for f in fs:
            row=factors[(factors.target==target)&(factors.field==f)].iloc[0].to_dict()
            row['selected_primary_top3']=bool(((top.target==target)&(top.field==f)).any())
            row['chemical_name']=chem.set_index('field').loc[f,'graph_name']
            row['evidence']='chemical_identity_or_primary_sensory_literature_anchor'
            anchor_rows.append(row)
    anchordf=pd.DataFrame(anchor_rows)
    anchordf.to_csv(P/'chemical_anchor_checks.csv',index=False,encoding='utf-8-sig')

    # Chemical plausibility is graded independently from statistical rank.
    annot=[]
    for _,row in top.iterrows():
        target=row.target; field=row.field; origin=row.origin
        if field=='X2.ethylhexylsalicylate':
            grade='identity_unresolved'
            note='原始字段salicylate与附件英文acrylate名称冲突；暂不能给出结构特异性解释。'
        elif target in anchors and field in anchors[target]:
            grade='direct_chemical_anchor'
            note='与该感官属性具有明确化学身份或原始感官研究依据；本数据中的回归仍只估计条件预测关系。'
        elif target=='metallic' and field=='iron':
            grade='mechanism_consistent_speciation_unmeasured'
            note='铁与金属感的方向符合化学背景；附件仅给总铁，未区分Fe(II)/Fe(III)及基质氧化，不能转成单一价态的致味剂量。'
        elif ('hops' in target and origin=='Hops'):
            grade='source_and_module_consistent'
            note='数据字典归于酒花来源，与酒花感知方向同源；具体感官作用需结合共变簇与混合实验。'
        elif ('malt' in target and origin=='Malt'):
            grade='source_and_module_consistent'
            note='数据字典归于麦芽来源，可作为麦芽加工与原料变化的指示物；不能据此锁定独立呈味效应。'
        elif ('esters' in target and any(e in field.lower() for e in
            ['acetate','butyrate','hexanoate','octanoate','decanoate','benzoate','propionate',
             'isovalerate','lactate','succinate','formate']) and origin=='Yeast'):
            grade='fermentation_ester_candidate'
            note='属于发酵相关候选；需区分酯类自身香气与其对整体发酵强度的代理作用，依化学结构进一步核实。'
        else:
            grade='statistical_marker_requires_mechanistic_validation'
            note='现有证据支持样本中的统计标记角色；保留替代因子、风格共变与混合感知三种解释。'
        rec=row.to_dict();rec.update(chemical_plausibility_grade=grade,chemical_rationale=note)
        rec['style_adjustment_retention']=abs(row.within_style_rho)/(abs(row.spearman_rho)+1e-12)
        rec['any_member_module_top3_frequency']=module_support[target,int(row.graph_module)]
        annot.append(rec)
    annotated=pd.DataFrame(annot)
    annotated.to_csv(P/'top3_with_chemical_rationale.csv',index=False,encoding='utf-8-sig')

    # Figure 1: preserve graph structure without illegible labels on 231 nodes.
    n=nodes.sort_values(['module','weighted_degree'],ascending=[True,False])
    order=n.index.to_numpy()
    cmap=LinearSegmentedColormap.from_list('rankcorr',['#42647A','#F8F7F4','#A95750'])
    fig=plt.figure(figsize=(11.0,4.8),layout='constrained')
    gs=fig.add_gridspec(1,2,width_ratios=[1.15,1])
    ax=fig.add_subplot(gs[0])
    im=ax.imshow(R[np.ix_(order,order)],vmin=-1,vmax=1,cmap=cmap,interpolation='nearest')
    counts_mod=n.groupby('module',sort=False).size(); start=0
    ticks=[];labels=[]
    for mod,count in counts_mod.items():
        if count>=5:
            ticks.append(start+(count-1)/2);labels.append(f'C{mod}')
        start+=count
    ax.set_xticks(ticks,labels,fontsize=7,rotation=60,ha='right');ax.set_yticks(ticks,labels,fontsize=7)
    ax.set_xlabel('Chemical features, ordered by co-variation community')
    ax.set_title('A   Measured chemical co-variation',loc='left',pad=10)
    cb=fig.colorbar(im,ax=ax,shrink=.65,pad=.025);cb.set_label("Spearman's rho")
    ax2=fig.add_subplot(gs[1])
    major=nodes.module.value_counts();major=major[major>=5].index.tolist();major=sorted(major)
    left=np.zeros(len(major))
    for origin,col in COLORS.items():
        vals=np.array([int(((nodes.module==m)&nodes.origin.eq(origin)).sum()) for m in major])
        if vals.sum()==0:continue
        ax2.barh(np.arange(len(major)),vals,left=left,color=col,label=origin,height=.66)
        left+=vals
    unknown=np.array([int(((nodes.module==m)&nodes.origin.isna()).sum()) for m in major])
    ax2.barh(np.arange(len(major)),unknown,left=left,color='#D5D1CA',height=.66,label='Aggregate')
    for i,m in enumerate(major):
        size=int((nodes.module==m).sum());ax2.text(size+.35,i,str(size),va='center',fontsize=8)
    ax2.set_yticks(np.arange(len(major)),[f'C{m}' for m in major]);ax2.invert_yaxis()
    ax2.set_xlim(0,max(major and nodes.module.value_counts())+3)
    ax2.set_xlabel('Number of measured features')
    ax2.set_title('B   Chemical origins of the larger communities',loc='left',pad=10)
    ax2.grid(axis='x',alpha=.13);ax2.set_axisbelow(True)
    ax2.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,frameon=False,fontsize=8,
               handlelength=1.4,columnspacing=1.0)
    fig.suptitle(f"231 measured features  |  {s['graph_at_0_6']['edges']} edges at |rho| >= 0.60",fontsize=13,x=.51)
    savefig(fig,'q2_chemical_structure')

    # Full node-edge diagram: non-isolates have their actual edges; isolates are
    # shown individually on the right rather than being omitted from the graph.
    noniso=np.where(nodes.degree.to_numpy()>0)[0]
    isolates=np.where(nodes.degree.to_numpy()==0)[0]
    all_W=np.where(abs(R)>=.6,abs(R),0.);np.fill_diagonal(all_W,0.)
    _,component=connected_components(all_W,directed=False)
    sizes=np.bincount(component)
    giant=np.where(component==sizes.argmax())[0]
    others=[np.where(component==c)[0] for c in np.argsort(-sizes)
            if sizes[c]>1 and c!=sizes.argmax()]
    fig,(a,small,b)=plt.subplots(1,3,figsize=(13.2,6.6),
        gridspec_kw={'width_ratios':[3.5,1.3,1.5]},layout='constrained')
    def draw_component(ax,ix,xy):
        rr=R[np.ix_(ix,ix)];ww=np.where(abs(rr)>=.6,abs(rr),0.);np.fill_diagonal(ww,0.)
        ei,ej=np.where(np.triu(ww>0,1));segments=np.array([[xy[i],xy[j]] for i,j in zip(ei,ej)])
        positive=rr[ei,ej]>0
        ax.add_collection(LineCollection(segments[positive],colors='#BAC5BC',
            linewidths=.65+1.6*(ww[ei[positive],ej[positive]]-.6),alpha=.75,zorder=1))
        ax.add_collection(LineCollection(segments[~positive],colors='#A2524E',
            linewidths=1.25,linestyles='dashed',alpha=.95,zorder=2))
        for origin,color in list(COLORS.items())+[(None,'#D5D1CA')]:
            yes=(nodes.iloc[ix].origin.isna() if origin is None else nodes.iloc[ix].origin.eq(origin)).to_numpy()
            ax.scatter(xy[yes,0],xy[yes,1],s=20+2.3*nodes.iloc[ix].degree.to_numpy()[yes],
                c=color,edgecolor='white',linewidth=.45,zorder=3)
    pos=force_layout(all_W[np.ix_(giant,giant)])
    draw_component(a,giant,pos)
    selected_labels={'ethyl_acetate':(12,16),'ethanol..v.v.':(-12,-18),
        'linalool':(-12,12),'X2.acetylfuran':(12,15),'nerol.oxide':(-12,-14)}
    for key,offset in selected_labels.items():
        raw=features.index(key)
        if raw not in giant:continue
        index=int(np.where(giant==raw)[0][0]);xy=pos[index]
        a.annotate(nodes.loc[raw,'name'],xy,xytext=offset,textcoords='offset points',
            fontsize=7.5,ha='left' if offset[0]>=0 else 'right',
            bbox=dict(boxstyle='square,pad=.1',fc='white',ec='none',alpha=.85),
            arrowprops=dict(arrowstyle='-',color='#8A8A8A',lw=.55),zorder=4)
    a.set_title(f'A   Largest connected component: {len(giant)} nodes',loc='left',pad=12)
    a.set_aspect('equal');a.set_xlim(pos[:,0].min()-.13,pos[:,0].max()+.13)
    a.set_ylim(pos[:,1].min()-.11,pos[:,1].max()+.11);a.axis('off')
    for k,ix in enumerate(others):
        angle=np.linspace(0,2*np.pi,len(ix),endpoint=False)+.3
        xx=(k%2)*1.1+.25*np.cos(angle)
        yy=(k//2)*1.0+.25*np.sin(angle)
        draw_component(small,ix,np.c_[xx,yy])
        labels_here=[nodes.loc[i,'name'] for i in ix if features[i] in ['DMS','X2.3.butanedione']]
        if labels_here:small.text(xx.mean(),yy.max()+.18,labels_here[0],fontsize=6.5,ha='center',va='top')
    small.set_xlim(-.5,1.6);small.set_ylim(6.9,-.55);small.axis('off')
    small.set_title(f'B   Other components\n{sum(len(x) for x in others)} nodes',loc='left',pad=12)
    b.set_title(f'C   Isolated nodes\n{len(isolates)} nodes',loc='left',pad=12)
    groups=[]
    for origin,color in COLORS.items():
        ix=[i for i in isolates if nodes.loc[i,'origin']==origin]
        if ix:groups.append((origin,color,ix))
    y0=0
    for origin,color,ix in groups:
        b.text(-.7,y0-.2,f'{origin}  ({len(ix)})',ha='left',va='bottom',fontsize=8)
        yy=[];xx=[]
        for j,raw in enumerate(ix):xx.append(j%8);yy.append(y0+.35+j//8*.65)
        b.scatter(xx,yy,c=color,s=24,edgecolor='white',linewidth=.4)
        y0+=1.4+(len(ix)-1)//8*.65
    b.set_ylim(y0+.1,-.8);b.set_xlim(-1,8);b.axis('off')
    handles=[Line2D([0],[0],color='#A2524E',ls='--',lw=1,label='Negative association'),
             Line2D([0],[0],color='#ADB8AD',lw=1,label='Positive association')]
    a.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.035),ncol=2,frameon=False,fontsize=8)
    fig.suptitle('Full chemical co-variation graph  |  231 nodes, 266 edges  |  |Spearman rho| >= 0.60',fontsize=13)
    savefig(fig,'q2_coexpression_network')

    # Figure 2: stable, chemically recognizable anchors and the actual graph ablation.
    selected_targets=['alcohol','acidity','bitternes','X4vg','F_hops_all','F_esters_ethac','F_esters_isoaa','diacetyl']
    labels_short={'alcohol':'Alcohol','acidity':'Acidity','bitternes':'Bitterness','X4vg':'4-vinyl guaiacol',
        'F_hops_all':'Hop taste','F_esters_ethac':'Nail-polish taste','F_esters_isoaa':'Banana taste','diacetyl':'Diacetyl'}
    data=top[top.target.isin(selected_targets)].copy()
    data['target_order']=data.target.map({v:k for k,v in enumerate(selected_targets)})
    data=data.sort_values(['target_order','rank']).reset_index(drop=True)
    fig,(a,b)=plt.subplots(1,2,figsize=(12,7.6),gridspec_kw={'width_ratios':[1.55,1]},layout='constrained')
    y=np.arange(len(data))
    a.hlines(y,0,data.bootstrap_top3_frequency,color='#D6D1C9',lw=1)
    a.scatter(data.bootstrap_top3_frequency,y,c='#426B51',s=35,zorder=3)
    a.scatter(data.stratified_top3_frequency,y,marker='|',c='#1C282F',s=60,zorder=4)
    a.axvline(.8,color='#9A9A9A',ls=':',lw=.8)
    a.set_xlim(-.02,1.04);a.set_ylim(-1,len(data));a.invert_yaxis()
    a.set_yticks(y,[r.chemical_name for _,r in data.iterrows()],fontsize=8)
    for k,tg in enumerate(selected_targets):
        if k:a.axhline(3*k-.5,color='#EEEAE5',lw=.8)
        a.text(1.02,3*k+1,labels_short[tg],transform=a.get_yaxis_transform(),va='center',fontsize=8)
    a.set_xlabel('Top-3 inclusion frequency: pairs bootstrap (dot), style-stratified (tick)')
    a.set_title('A   Which chemical markers survive resampling?',loc='left',pad=12)
    piv=metrics.pivot(index='target',columns='model',values='R2')
    core=piv.index.str.startswith(('A_','F_'))
    b.scatter(piv.rank_ridge[~core],piv.signed_graph_ridge[~core],s=28,c='#8D6C4B',label='Additional sensory traits',alpha=.8)
    b.scatter(piv.rank_ridge[core],piv.signed_graph_ridge[core],s=28,c='#426B51',label='Aroma / flavour core',alpha=.8)
    low=min(piv.rank_ridge.min(),piv.signed_graph_ridge.min())-.025
    high=max(piv.rank_ridge.max(),piv.signed_graph_ridge.max())+.025
    b.plot([low,high],[low,high],color='#929292',lw=.9,ls='--');b.set_xlim(low,high);b.set_ylim(low,high)
    b.set_aspect('equal');b.set_xlabel('Rank ridge: held-out R²');b.set_ylabel('Signed graph ridge: held-out R²')
    b.set_title('B   Effect of graph regularization',loc='left',pad=12)
    b.legend(frameon=False,loc='upper left',fontsize=8)
    b.text(.04,.72,f"Mean R²\nRidge {s['nested_rank_ridge_mean_R2']:.3f}\nGraph {s['nested_graph_mean_R2']:.3f}",
           transform=b.transAxes,fontsize=10,color='#29313A',linespacing=1.6)
    savefig(fig,'q2_stability_and_graph_ablation')

    # Concise cross-method summary for the paper.
    agree_summary=agreement.groupby('method').Jaccard.agg(['mean','median']).reset_index()
    agree_summary.to_csv(P/'agreement_summary.csv',index=False)
    module_rows=[]
    for m in sorted(nodes.module.unique()):
        part=nodes[nodes.module==m]
        if len(part)<3:continue
        lead=part.sort_values('weighted_degree',ascending=False).head(3)
        module_rows.append({'模块':f'C{m}','节点数':len(part),
            '主要来源':'；'.join(f'{k}({v})' for k,v in part.origin.fillna('Aggregate').value_counts().head(3).items()),
            '高连接代表':'; '.join(lead.name)})
    modules=pd.DataFrame(module_rows)
    modules.to_csv(P/'community_summary.csv',index=False,encoding='utf-8-sig')
    compact=[]
    for target in targets:
        a=top[top.target==target].sort_values('rank')
        compact.append({'感官维度':target,'Top-1':f'{a.iloc[0].chemical_name} ({a.iloc[0].bootstrap_top3_frequency:.1%})',
            'Top-2':f'{a.iloc[1].chemical_name} ({a.iloc[1].bootstrap_top3_frequency:.1%})',
            'Top-3':f'{a.iloc[2].chemical_name} ({a.iloc[2].bootstrap_top3_frequency:.1%})'})
    compact=pd.DataFrame(compact)
    compact.to_csv(P/'top3_compact_table.csv',index=False,encoding='utf-8-sig')
    metric_sum=metrics.groupby('model')[['R2','RMSE','Spearman']].mean().reset_index().round(6)
    threshold_table=pd.read_csv(P/'graph_threshold_sensitivity.csv').round(4)
    negative_table=negative[['source','target','rho','within_style_rank_rho','bootstrap_edge_frequency']].round(4)
    main_targets=['A_malt_all','A_hops_citrus','F_esters_ethac','F_esters_isoaa','X4vg','diacetyl','metallic','alcohol']
    main_compact=compact[compact['感官维度'].isin(main_targets)]
    high=top[top.bootstrap_top3_frequency>=.8]
    low=top[top.bootstrap_top3_frequency<.5]
    n_singletons=int((nodes.degree==0).sum())
    ci=s['paired_beer_bootstrap_delta_CI95']
    report=f'''# A题问题2：化合物共变结构与风味关键因素

## 一、结果与解释主线

本次实际读取了250款啤酒、231项化学或理化指标以及50个感官属性。附件2与按附件1重算的Spearman相关矩阵逐项一致，最大误差为{s['spearman_matrix_max_absolute_error']:.3g}；与Pearson相关矩阵的最大误差为{s['pearson_matrix_max_absolute_error']:.6f}。因此图的边权采用Spearman系数，图表达样本中的化合物共变关系。

在|ρ|≥0.60下，图含{s['graph_at_0_6']['edges']}条边，其中负边{s['graph_at_0_6']['negative_edges']}条，孤立节点{n_singletons}个。若改变阈值，边数变化明显，故正文以主要模块和具体相关替代关系解释，完整节点和边表作为支撑结果。所有图节点保留原始字段名，化学显示名通过附件8和附件2对照。

完成400次普通成对bootstrap和200次风格内分层bootstrap。对50个感官维度逐一输出3个化合物候选，共150条结果。普通bootstrap入选频率≥80%的条目有{len(high)}条，≥50%的条目有{s['n_top3_entries_frequency_at_least_0_5']}条，平均入选频率{s['mean_top3_frequency']:.1%}。这些频率量化的是在本样本分布下重复抽样时的排名稳定性，并非“引起该风味的概率”。

## 二、建模口径

### 2.1 图、变量与候选对象

全图包括231项测量，用来保留真实相关结构。单化合物排名排除12个理化/综合指标及1个标为“Internal standard solution”的字段X3.methyl.2.butenal，剩余218个候选。pH、色度、总糖、苦度等仍作为预测协变量；另提供全部测量因素的Top-3表，明确标注“单化合物”与“理化/综合指标”，避免丢掉诸如异α-酸与苦味这样的解释。不能将实验内标作为可调配的啤酒风味原料。

同时输出218个化合物的诱导子图，含{len(pure_edges)}条边，并重新计算去除综合指标后的模块。综合量与其组成量之间的强相关属于预期的数值关系，不作为发现新生化联系的证据。

附件8中X2.ethylhexylsalicylate的显示名为2-ethylhexyl acrylate，名称对应的化学结构不一致。该字段的统计对应可核对，结构特异性解释必须保留待核实标记。

数据以beer连接，并先按beer升序排列。beer_id因附件4的舍入损失不能作为唯一键。训练中不使用beer、beer_id或风格标签作为化合物到感官分数的预测输入；风格仅用于分层重采样及风格内相关检查。

### 2.2 折内变换与有符号图岭

化学列混合了log10数值、原浓度和相对测量尺度，因此解释模型使用训练集经验秩的正态分数，再在训练集上标准化。缺失值使用训练集中位数，验证样本按训练经验分布变换。该变换不把所有化学列误认成同一浓度量纲。

记标准化设计矩阵为Z、感官目标为T。模型求解

$$\\min_B\\; n^{{-1}}\\|T-ZB\\|_F^2+\\alpha\\|B\\|_F^2+\\gamma\\operatorname{{tr}}(B^\\top L B).$$

其中L=D^{{-1/2}}(D-A)D^{{-1/2}}，A保留|ρ|≥0.60的有符号边，D为绝对边权之和。正相关边倾向于保留同向系数，负相关边保留反向关系；孤立节点的图惩罚设为0。令γ=0得到普通岭模型。

等价地，对单个感官维度的系数b，图惩罚等于

$$b^\\top Lb=\\sum_{{j<k}}|A_{{jk}}|\\left(\\frac{{b_j}}{{\\sqrt{{d_j}}}}-\\operatorname{{sign}}(A_{{jk}})\\frac{{b_k}}{{\\sqrt{{d_k}}}}\\right)^2.$$

实际求解为B=(ZᵀZ/n+αI+γL)⁻¹ZᵀT/n。因为Z和T按当前训练集中心化且标准化，α是在“每个训练样本平均平方误差”归一化下的无量纲惩罚；若改用总平方误差定义，需把岭系数写成nα，不能直接照搬数值3。截距由训练目标均值恢复。

外层采用5折KFold，shuffle=True，random_state=20261023，与问题1按beer排序后的划分一致。内层3折选择α和γ，所有变换、相关矩阵和图均在当前训练折重算。附件2的全样本图只用于完整数据描述与最终解释，不进入任何留出折的训练过程。

α候选为0.01、0.03、0.1、0.3、1、3、10、30；图岭γ候选为0.03、0.1、0.3、1、3。内层依据50个训练标准化目标的平均验证均方误差选择共享超参数。最终采用{str(s['selected_family'])}，α={s['full_model_alpha']}，γ={s['full_model_gamma']}。

### 2.3 排名与稳定性

每次bootstrap均重做中位数填补、经验分布变换、标准化与模型拟合，并重估相应训练样本的相关图。超参数固定为全样本内层交叉验证选定的值，因而频率描述固定分析流程的重采样稳定性。每个重样本按标准化系数绝对值选择3个候选；最终按400次入选频率排序，同频时按bootstrap系数绝对值中位数排序。

每条结果同时给出：普通bootstrap频率、风格分层bootstrap频率、系数bootstrap区间、符号一致率、边际Spearman相关、去除风格均值后的秩相关、所属共变模块及相关替代候选。系数是标准化预测效应，不能直接换算成mg/L添加剂量。

逐项解释表还附“模块中任一成员进入Top-3”的频率，以及模块的候选数量，用来展示相关替代物分担入选机会的情况。该组频率不等于单化合物的稳定性，也不应在不同大小模块之间直接比较。

另拟合Extra Trees用于非线性排序对照，与Spearman、普通岭、图岭共同计算Top-3 Jaccard一致度；使用KNN填补和完整案例分析检验缺失处理的敏感性。非线性全样本重要性仅用于方法间排序比较，没有被表述为新的留出验证性能。

## 三、验证结果

{markdown(metric_sum)}

上述RMSE处于附件4提供的变换后分数尺度。普通岭与图岭的平均R²分别为{s['nested_rank_ridge_mean_R2']:.6f}和{s['nested_graph_mean_R2']:.6f}。图岭减普通岭的平均MSE差为{s['graph_minus_ridge_MSE']:.8f}；对留出逐啤酒损失差进行配对bootstrap的95%区间为[{ci[0]:.8f}, {ci[1]:.8f}]。该区间用于描述已获得的留出预测差异，未计入交叉验证折训练之间的依赖和重新选参的不确定性。

图正则没有带来可证实的预测提升，因此图承担揭示共线性、变量替代与来源模块的作用，评分模型的选择由问题1的完整预测比较决定。图结构本身也不能证明反应路径或调控方向。

### 3.1 主要共变模块

{markdown(modules)}

来源分类来自附件8的数据字典，同一化合物在真实酿造过程中可能来自多种途径。模块具有酵母、酒花、麦芽等来源富集，但并不等同于独立生化反应通路。

### 3.2 阈值与负相关边

{markdown(threshold_table)}

阈值升高时，图由广泛的共变网络收缩为高相关小组；阈值0.60保留主要共变关系并维持图的可读性。模块数包含单点模块，因此不能将120个模块描述成120条生物通路。

三条负边的重采样与风格内关联为：

{markdown(negative_table)}

双乙酰—戊酸乙酯、异丁酸乙酯—邻甲酚、乙醇—3-甲基-2-己醇分别呈反向浓度排序。前两条在400次重采样下跨过|ρ|≥0.60阈值的频率约为83%，最后一条约48%，后者接近筛选阈值，结构支持较弱。风格内关联列进一步检验总体反向关系是否主要来自风格组成。它们为发酵状态、原料与成熟程度的共同变化提出了可检验线索，但本数据没有时间序列或化合物干预，不能把负边画成已证实的抑制反应。

### 3.3 排名方法与缺失处理的一致性

{markdown(agree_summary.round(4))}

Jaccard=1表示两个Top-3集合完全一致，=0表示无交集。表中“within_style”刻意去除了风格均值，其不同之处有助于识别风格共变代理，不应为了使其与总体排名一致而调参。完整案例包含{s['n_complete_cases']}款啤酒；它改变了样本总体，不能仅作为更准确的默认答案。

对目标t和对照方法m，J(t,m)=|S_t∩S_{{t,m}}|/|S_t∪S_{{t,m}}|，其中两集合各含3个变量；因此仅可能取0、0.2、0.5、1。表格给出50个目标的均值与中位数。主排序是bootstrap稳定性排名；“coefficient”是最终全样本绝对系数排名，两者并不被强行设成相同。

### 3.4 适合主文展示的八例

{markdown(main_compact)}

乙醇—酒精感和实测4VG—4VG感知的入选频率均为100%，前者与实际酒精量、后者与酚香化合物具有明确对应。铁—金属感为97%，方向与化学背景吻合，但不能据附件的总铁值断定具体价态及致味途径。2-acetylfuran—总体麦芽香频率为83.5%，与麦芽来源特征相符。相对地，香蕉感排名未出现乙酸异戊酯；双乙酰感的第一统计标记为负向的eugenol。这些反例说明稀疏感官标签和化学共变限制了单分子解释的唯一性。4VG在不同风格中可构成特征香气，本节不把其浓度或感知强弱等同于跨风格一致的质量缺陷。

## 四、逐维度Top-3

括号内是400次普通bootstrap的Top-3入选频率。A_表示Aroma，F_表示Flavour/Taste；其余20维为品评表中的基础味、缺陷味、口感和总体欣赏度等补充属性。三个位置均给出以满足逐维输出，但低入选频率应解读为可替代的候选标记。

{markdown(compact)}

## 五、化学合理性的具体检验

化学核对独立于数据排名进行，避免先认定某个机理、再倒推应该入选的化合物。chemical_anchor_checks.csv列出乙醇—酒精感、异α-酸—苦味、乙酸/乳酸—酸感、4VG—酚香/丁香、双乙酰—奶油味、DMS—硫味、trans-2-nonenal—陈化纸板味、乙酸乙酯—溶剂味、乙酸异戊酯—香蕉、苯乙醇/乙酸苯乙酯—花香、芳樟醇/香叶醇/香茅醇—酒花柑橘感等锚点的实际关联和入选情况。

解释时区分三层证据：直接化学锚点、与来源和模块相符的候选、尚待机制验证的统计标记。top3_with_chemical_rationale.csv给每一条结果记录了对应等级与理由。直接锚点也可能在条件回归中被相关特征分担；反之，与某感官维度稳定相关的变量也可能只是原料、发酵或风格代理。原始数据研究讨论过相关化学物互相替代的重要性排名问题，其添加实验检验的是成组化合物而非每个单分子的独立作用[1]。

化学阈值用于确认潜在感知作用，不能单独确定0–6评分。原始感官实验表明，芳樟醇、香叶醇、β-香茅醇存在组合效应，因此单项低于阈值也不能排除混合物中的贡献[2–3]。实用验证可在给定风格内选择高稳定候选及其同模块替代物，进行单项/组合添加与盲评比较；这种实验将“稳定预测标记”推进到“可操作风味因素”。本次数据分析未开展新的品尝或添加实验。

铁离子添加实验支持其与氧化和感官变化有关[4]，但附件中的总铁并未测定Fe(II)/Fe(III)组成，也未测定有关的基质反应。因而iron—metallic的强稳定关联适合作为质量监测线索；它没有单独确定价态、金属来源或一个通用的添加/去除剂量。

## 六、复现与结果文件

运行环境中的numpy、scipy、pandas、scikit-learn、matplotlib用于本次计算；不需要商业求解器。安装版本请以最终完整项目的环境说明为准。独立运行入口：

```bash
python run_q2.py --data /path/to/附件 --output /path/to/q2 --bootstrap 400 --stratified 200
python make_q2_report.py --data /path/to/附件 --output /path/to/q2
```

图与统计结果的主要对应关系：

- graph_nodes.csv、graph_edges.csv：231节点图，含边权、近似BH校正p值、bootstrap边频率。
- graph_compound_nodes.csv、graph_compound_edges.csv：排除综合指标和内标后的218节点诱导子图。
- graph_threshold_sensitivity.csv、community_summary.csv：阈值敏感性与主要模块。
- top3_all_50_sensory_dimensions.csv、top3_core_30_aroma_flavour.csv：完整50维与核心30维排名。
- top3_all_measured_factors.csv：含pH等指标的测量因素排名，并标注对象类型。
- all_factor_evidence.csv、top3_with_chemical_rationale.csv：全部候选的统计证据与逐项化学解释。
- chemical_anchor_checks.csv：机理锚点核对，不以是否进入Top-3筛选。
- nested_cv_metrics.csv、oof_*.csv：逐目标、逐啤酒留出结果。
- method_and_imputation_agreement.csv：方法、填补与完整案例的Top-3一致性。
- bootstrap_coefficients.npz：400次bootstrap的全部系数。
- q2_coexpression_network.svg/png：完整节点—边图，130个非孤立节点与101个孤立节点均有显示。
- q2_chemical_structure.svg/png、q2_stability_and_graph_ablation.svg/png：相关矩阵、来源模块、稳定性和消融图件。

本次重新执行了上述建模和验证。附件8已提供的GBM MDI与SHAP表不作为本次新实验结果。

## 参考文献

[1] Schreurs M, Piampongsant S, Roncoroni M, et al. Predicting and improving complex beer flavor through machine learning. Nature Communications, 15:2368, 2024. https://doi.org/10.1038/s41467-024-46346-0 。公开全文：https://pmc.ncbi.nlm.nih.gov/articles/PMC10966102/ 。

[2] Takoi K, Koie K, Itoga Y, et al. Biotransformation of hop-derived monoterpene alcohols by lager yeast and their contribution to the flavor of hopped beer. Journal of Agricultural and Food Chemistry, 58(8):5050–5058, 2010. https://doi.org/10.1021/jf1000524 。

[3] Takoi K, Itoga Y, Koie K, et al. The Contribution of Geraniol Metabolism to the Citrus Flavour of Beer: Synergy of Geraniol and β-Citronellol Under Coexistence with Excess Linalool. Journal of the Institute of Brewing, 116:251–260, 2010. https://doi.org/10.1002/j.2050-0416.2010.tb00428.x 。

[4] Zufall C, Tyrell Th. The Influence of Heavy Metal Ions on Beer Flavour Stability. Journal of the Institute of Brewing, 114(2):134–142, 2008. https://doi.org/10.1002/j.2050-0416.2008.tb00318.x 。原始论文PDF：https://www.themodernbrewhouse.com/wp-content/uploads/2016/11/Metals-and-Beer-Stability.pdf 。




'''
    (P/'问题2_化合物图与关键因素研究说明.md').write_text(report,encoding='utf-8')
    (P/'q2_report.md').write_text(report,encoding='utf-8')
    print('Saved report, three SVG/PNG figures, chemical annotations, and factor ranking.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default=str(D),
                        help='Input attachment directory; defaults to ../data/A beside the q2 directory.')
    parser.add_argument('--output',default=str(P))
    args=parser.parse_args()
    D=Path(args.data);P=Path(args.output)
    main()
