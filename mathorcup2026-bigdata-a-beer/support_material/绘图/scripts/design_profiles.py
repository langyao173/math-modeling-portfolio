"""A shared-index sensory atlas with three compact validation sidecars.

All values are read from frozen, previously exported tables. The circular
geometry is a categorical index; radius is not a quantitative deviation axis.
"""
from repair_common import *
import pandas as pd
from matplotlib.patches import Wedge, Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

STYLES = ['Tripel', 'Kriek', 'Lambic']


def main():
    paths = [DATA / ('fig09_' + stem + '.csv') for stem in [
        'all50_sensory_profiles', 'thresholds', 'energy_intervals', 'sampling_efficiency']]
    profiles, threshold, ci, effort = [pd.read_csv(p) for p in paths]
    targets = profiles.loc[profiles['style'].eq('Tripel'), 'target'].tolist()
    labels = profiles.loc[profiles['style'].eq('Tripel'), 'label'].tolist()
    assert len(targets) == 50 and len(profiles) == 150
    vals = np.vstack([profiles[profiles['style'].eq(s)].set_index('target')
                      .loc[targets, 'representative_centroid_deviation_sd'].to_numpy()
                      for s in STYLES])
    assert np.isfinite(vals).all() and np.abs(vals).max() < 1
    assert (threshold.new_ratio <= 1).all()

    fig = figure(9)
    fig.set_size_inches(6.7, 4.35)
    # The shared categorical skeleton carries all three profiles. Short gaps
    # separate the A/F/M blocks without sorting away their original identity.
    atlas = fig.add_axes([.002, .075, .69, .855])
    atlas.set_aspect('equal')
    atlas.set(xlim=(-1.43, 1.43), ylim=(-1.43, 1.43))
    atlas.axis('off')
    cmap = LinearSegmentedColormap.from_list('sensory_cold_warm',
                                             ['#4F789B', '#FCFDFD', '#CA7771'], N=257)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    group_defs = [(0, 15, '香气 · 15', TEAL),
                  (15, 30, '风味 · 15', PURPLE),
                  (30, 50, '口感及扩展 · 20', RED)]
    gap = 3.0
    cell = (360 - 3*gap) / 50
    start = 90 - gap/2
    angles = []
    ring_edges = [(.415, .58), (.59, .755), (.765, .93)]
    for lo, hi, group, col in group_defs:
        first = start
        for j in range(lo, hi):
            a1, a0 = start, start-cell
            theta = (a0+a1)/2
            angles.append(theta)
            for k, (r0, r1) in enumerate(ring_edges):
                atlas.add_patch(Wedge((0,0), r1, a0, a1, width=r1-r0,
                    facecolor=cmap(norm(vals[k,j])), edgecolor='white', lw=.4))
            # The outer classifier is a single, continuous narrow track.
            atlas.add_patch(Wedge((0,0), .977, a0, a1, width=.022,
                facecolor=col, edgecolor='white', lw=.3, alpha=.9))
            th = np.deg2rad(theta)
            left = np.cos(th) < 0
            rot = theta+180 if left else theta
            atlas.text(1.025*np.cos(th), 1.025*np.sin(th), labels[j],
                fontsize=6.0, rotation=rot, rotation_mode='anchor',
                ha='right' if left else 'left', va='center', color=INK)
            start -= cell
        last = start
        start -= gap
    atlas.add_patch(Circle((0,0), .375, facecolor='white', edgecolor=GRID, lw=.55))
    atlas.text(0, .17, '50', ha='center', va='center', fontsize=19, color=INK)
    atlas.text(0, .045, '感官属性', ha='center', va='center', fontsize=7.1, color=INK)
    for y, text in zip([-.08,-.18,-.28], ['内圈  Tripel', '中圈  Kriek', '外圈  Lambic']):
        atlas.text(0,y,text,ha='center',va='center',fontsize=6.1,color=SECONDARY)
    fig.text(.035,.975,'d  条件生成的感官画像',ha='left',va='top',fontsize=7.5,color=INK)
    for x, text, color in [(.045,'香气 15',TEAL),(.225,'风味 15',PURPLE),(.405,'口感及扩展 20',RED)]:
        fig.text(x,.934,text,ha='left',va='center',fontsize=6.1,color=color)
    cbax = fig.add_axes([.20,.050,.28,.018])
    cb = fig.colorbar(ScalarMappable(norm=norm,cmap=cmap), cax=cbax,
                     orientation='horizontal', ticks=[-1,0,1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.7,length=1.5,width=.35,pad=1)
    fig.text(.34,.007,'相对风格质心的偏差 / SD',ha='center',va='bottom',fontsize=6.0,color=SECONDARY)

    # Sidecar a: all six before/after normalized concentration comparisons.
    x0, width = .745, .225
    ax = fig.add_axes([x0,.705,width,.205])
    for k,s in enumerate(STYLES):
        rows = threshold[threshold['style'].eq(s)]
        for q,compound in enumerate(['DMS','双乙酰']):
            row = rows[rows.compound.eq(compound)].iloc[0]
            y = 5-2*k-q
            ax.plot([row.old_ratio,row.new_ratio],[y,y],color=GRAY,lw=.75,zorder=1)
            ax.scatter(row.old_ratio,y,s=12,facecolors='white',edgecolors=SECONDARY,lw=.55,zorder=3)
            ax.scatter(row.new_ratio,y,s=13,color=TEAL,edgecolors='white',lw=.3,zorder=4)
    ax.axvspan(0,1,color=TEAL,alpha=.055,lw=0)
    ax.axvline(1,color=SECONDARY,lw=.5,ls=(0,(2,2)))
    ax.set(xlim=(-.1,5.1),ylim=(-.5,5.7),xticks=[0,1,3,5],
           yticks=[4.5,2.5,.5],yticklabels=STYLES,xlabel='浓度 / 上界')
    ax.tick_params(axis='y',length=0,labelsize=5.7,pad=3)
    ax.tick_params(axis='x',labelsize=5.7)
    clean(ax)
    ax.spines['left'].set_visible(False)
    fig.text(x0,.975,'a  质量约束',ha='left',va='top',fontsize=7.3)
    fig.text(x0,.915,'每组：DMS、双乙酰',ha='left',va='center',fontsize=5.6,color=SECONDARY)
    fig.legend(handles=[Line2D([],[],marker='o',linestyle='none',ms=3,mfc='white',mec=SECONDARY,mew=.55,label='原代表'),
                       Line2D([],[],marker='o',linestyle='none',ms=3,mfc=TEAL,mec=TEAL,label='条件代表')],
              frameon=False,ncol=2,loc='upper left',bbox_to_anchor=(x0,.953),
              borderpad=0,borderaxespad=0,handletextpad=.2,columnspacing=.65,fontsize=5.6)

    # Sidecar b: frozen paired energy intervals, retaining the original n=29.
    bx = fig.add_axes([x0,.365,width,.15])
    for y,ref in [(1,'unconditional'),(0,'coordinate_clipping')]:
        row = ci[ci.reference.eq(ref)].iloc[0]
        bx.plot([row.ci95_low,row.ci95_high],[y,y],color=BLUE,lw=1.05)
        bx.scatter(row.energy_delta,y,s=17,color=BLUE,zorder=3)
        bx.text(row.energy_delta,y+.23,f'{row.energy_delta:+.4f}',
                ha='center',va='bottom',fontsize=5.8,color=BLUE)
    bx.axvline(0,color=SECONDARY,lw=.5,ls=(0,(2,2)))
    bx.set(xlim=(-.019,.0012),ylim=(-.5,1.65),yticks=[1,0],
           yticklabels=['无条件','坐标截断'],xticks=[-.02,-.01,0],xlabel='能量评分差')
    bx.tick_params(axis='y',length=0,labelsize=5.8,pad=3)
    bx.tick_params(axis='x',labelsize=5.7)
    clean(bx);bx.spines['left'].set_visible(False)
    fig.text(x0,.585,'b  条件留出检验',ha='left',va='top',fontsize=7.3)
    fig.text(x0,.550,'n = 29 · 95%配对区间',ha='left',va='center',fontsize=5.8,color=SECONDARY)

    # Sidecar c: count comparisons share one numerical scale. No connecting
    # line implies an ordering between the three categorical beer styles.
    cx = fig.add_axes([x0,.090,width,.13])
    counts=[]
    for k,s in enumerate(STYLES):
        q = effort[effort['style'].eq(s)].set_index('method')
        old=float(q.loc['full225_rejection','median_full225_vectors'])
        new=float(q.loc['marginal2_conditional','median_full225_vectors'])
        y=2-k
        cx.barh(y+.11,old,height=.18,color=GRAY,edgecolor='none')
        cx.barh(y-.11,new,height=.18,color=TEAL,edgecolor='none')
        cx.text(old+85,y+.11,str(int(old)),ha='left',va='center',fontsize=5.5,color=SECONDARY)
        cx.text(230,y-.11,str(int(new)),ha='left',va='center',fontsize=5.5,color=TEAL)
        counts.append(dict(style=s,full_vectors=old,conditional_vectors=new,ratio=old/new))
    cx.set(xlim=(0,4900),ylim=(-.45,2.45),yticks=[2,1,0],yticklabels=STYLES,
           xticks=[],xlabel='生成100谱所需完整向量数')
    cx.tick_params(axis='y',length=0,labelsize=5.7,pad=3)
    for spine in cx.spines.values():spine.set_visible(False)
    cx.xaxis.label.set_size(5.8)
    fig.text(x0,.285,'c  采样计算量',ha='left',va='top',fontsize=7.3)
    fig.text(x0,.249,'灰：完整拒绝  绿：条件补全',ha='left',va='center',fontsize=5.6,color=SECONDARY)

    caption=('二阈值条件生成的画像与验收。d为50属性共享分类索引的三同心轨，'
        '由内到外对应Tripel、Kriek、Lambic；每格仅以共同冷暖色标表示条件代表预测相对'
        '同风格实测样本模型预测质心的偏差，除以响应标准差，半径不编码偏差。'
        'a中各风格上、下点分别为DMS和双乙酰，空心/实心为原代表/条件代表，虚线为指定上界；'
        'b为29款条件留出样本的能量评分差及既有95%条件配对区间，负值有利于条件生成；'
        'c为生成100谱所计算的完整225维向量数（3次基准的中位数）。'
        '该感官画像为冻结模型预测，不是新实品测量；完整150格数值保留于底表，10%—90%生成跨度另见补图S9。')
    audit=dict(status='PASS',dimensions=50,styles=STYLES,cells=int(vals.size),
        value_range=[float(vals.min()),float(vals.max())],shared_color_limits=[-1,1],
        unchanged_tables=True,quality_checks_passed=int((threshold.new_ratio<=1).sum()),
        paired_energy_rows=len(ci),sampling_counts=counts,
        new_model_fits=0,new_bootstrap_resamples=0)
    save(fig,9,caption,paths,dict(categorical_index='50 original attributes, three contiguous domains',
        inner_to_outer_styles=STYLES,all_sensory_dimensions=50,all_style_target_cells=150,
        count_comparison=counts,full_value_table='fig09_all50_sensory_profiles.csv',spread_supplement='supplement_S9_generation_spread.pdf'))
    (OUT/'fig09_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
