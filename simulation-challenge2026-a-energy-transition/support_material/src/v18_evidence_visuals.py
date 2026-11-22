"""Redraw three V18 evidence figures using frozen V14/V17 values only.

No solver, simulator, native model, or prior-version artifact is modified.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.text import Text
import numpy as np
import pandas as pd

from src.v17_visuals import style as v17_style

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/v18/revisions'
FIG = ROOT / 'figures/V18/revisions'
INK, MUTED, GRID = '#243B4A', '#74828C', '#E5EAED'
BLUE, TEAL, ORANGE, PURPLE = '#2474AA', '#16877B', '#CE7D35', '#80639F'
COLORS = [BLUE, ORANGE, TEAL, PURPLE]
INPUTS: dict[str, str] = {}
INDEX: dict[str, dict] = {}
CHECKS: dict[str, dict] = {}
LAYOUT: dict[str, dict] = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(relative):
    path = ROOT / relative
    INPUTS[path.relative_to(ROOT).as_posix()] = sha(path)
    return path


def read(relative):
    return pd.read_csv(bind(relative), float_precision='round_trip')


def keep(name, frame):
    frame.to_csv(OUT / ('evidence_' + name + '.csv'), index=False, float_format='%.17g')


def style():
    v17_style()
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10,
                         'axes.titlesize': 11, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10, 'legend.fontsize': 10})


def clean(ax, axis='x'):
    ax.grid(axis=axis, color=GRID, lw=.55)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.8, width=.65)
    for spine in ax.spines.values():
        spine.set_linewidth(.65)


def title(ax, text, pad=8):
    ax.set_title(text, loc='left', fontweight='bold', pad=pad)


def finish(fig, stem, caption, notes, input_names):
    fig.canvas.draw()
    renderer, canvas = fig.canvas.get_renderer(), fig.bbox
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    outside = []
    for text in texts:
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > canvas.width+1 or box.y1 > canvas.height+1:
            outside.append(text.get_text())
    assert not outside, (stem, outside)
    assert min(t.get_fontsize() for t in texts) >= 10
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(FIG / f'{stem}.{extension}', dpi=300)
    width, height = np.array(fig.get_size_inches()) * 25.4
    plt.close(fig)
    with fitz.open(FIG / f'{stem}.pdf') as doc:
        page = doc[0]
        spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b
                 for line in b['lines'] for s in line['spans'] if s['text'].strip()]
        minimum = min(s['size'] for s in spans)
        assert minimum > 9.99
        page.get_pixmap(matrix=fitz.Matrix(160/72, 160/72)).save(OUT / f'evidence_{stem}__actual_pdf.png')
    INDEX[stem] = {ext: (FIG/f'{stem}.{ext}').relative_to(ROOT).as_posix() for ext in ('pdf', 'svg', 'png')}
    INDEX[stem].update(width_mm=float(width), height_mm=float(height), full_caption=caption,
                       caption_notes=notes,
                       input_sha256={name: INPUTS[name] for name in sorted(input_names)})
    LAYOUT[stem] = dict(minimum_font_pt=minimum, out_of_canvas_text=outside,
                        width_mm=float(width), height_mm=float(height), png_dpi=300)


def deadline():
    start = set(INPUTS)
    metrics = read('outputs/v14/deadline_investment_figures/case_metrics_recomputed.csv')
    pairs = read('outputs/v14/deadline_investment_figures/all_144_development_paired_paths.csv')
    groups = read('outputs/v14/deadline_investment_figures/complete_subgroup_counts_recomputed.csv')
    designs = read('outputs/v14/deadline_investment/development_design_frozen.csv')
    annual = read('outputs/v14/deadline_investment_figures/all_7056_yearly_key_values.csv')
    dev = metrics.loc[metrics.cohort.eq('development')]
    changed = pairs.loc[pairs.policy.eq('P2') & pairs.new_minus_P0_oil_2030_shortfall_fraction.abs().gt(1e-8), 'design_id'].tolist()
    changed = sorted(changed)
    assert changed == ['joint_000','joint_007','joint_018','joint_032','joint_038','joint_039']
    assert len(pairs) == 144 and len(designs) == 48 and len(metrics) == 196 and len(annual) == 7056
    fig = plt.figure(figsize=(160/25.4, 181/25.4))
    a = fig.add_axes([.115, .725, .34, .205])
    b = fig.add_axes([.66, .725, .30, .205])
    c = fig.add_axes([.115, .385, .34, .235])
    d = fig.add_axes([.685, .395, .275, .205])
    for j, lead in enumerate((4,5,6)):
        vals = [int(groups.loc[groups.policy.eq(p) & groups.group.eq('oil_lead_'+str(lead)), 'reliable_count'].item()) for p in ('P0','P1','P2','P3')]
        ys = np.arange(4)+(j-1)*.22
        a.barh(ys, vals, height=.19, color=COLORS[j], label=f'{lead}年')
        for y, value in zip(ys, vals):
            a.text(value+.20, y, str(value), va='center')
    a.set(yticks=range(4), yticklabels=['P0','P1','P2','P3'], xlim=(0,13.6), xticks=[0,5,10], xlabel='可靠设计数 / 每组16')
    a.invert_yaxis(); clean(a)
    title(a, 'a  全48设计：按油工期分组')
    a.legend(ncol=3, frameon=False, bbox_to_anchor=(-.03, 1.14), loc='lower left',
             handlelength=.8, columnspacing=.8, borderaxespad=0)
    selected=[]
    for i, case in enumerate(changed):
        r0, r2 = (dev.loc[dev.design_id.eq(case) & dev.policy.eq(policy)].iloc[0] for policy in ('P0','P2'))
        delta = (r2.oil_2030_shortfall_fraction-r0.oil_2030_shortfall_fraction)*100
        b.plot([delta,0],[i,i], color=BLUE, lw=1.8)
        b.scatter(0,i, facecolors='white',edgecolors=BLUE,s=30,lw=1.1,zorder=3)
        b.scatter(delta,i, color=BLUE,s=25,zorder=4)
        b.text(.065,i,f'{delta:.4f}',va='center')
        selected.append(dict(design_id=case,P0_percent=r0.oil_2030_shortfall_fraction*100,
                             P2_percent=r2.oil_2030_shortfall_fraction*100,P2_minus_P0_percentage_point=delta))
    b.set(yticks=range(6), yticklabels=['#'+name[-3:] for name in changed], xlim=(-.86,.58),
          xticks=[-.8,-.4,0], xlabel='油缺口变化 / 百分点')
    b.invert_yaxis();clean(b);title(b,'b  P2−P0：6个非零设计')
    for policy, color, marker in zip(('P1','P2','P3'),COLORS,('o','s','^')):
        part=pairs.loc[pairs.policy.eq(policy)]
        c.scatter(part.new_minus_P0_discounted_capital_trillion_cny,
                  part.new_minus_P0_net_emissions_2060_mt/1000,
                  color=color,marker=marker,s=23,alpha=.70,label=policy,linewidths=.4)
    c.axvline(0,color=MUTED,lw=.7);c.axhline(0,color=MUTED,lw=.7)
    c.set(xlim=(-.1,3.6),ylim=(-.12,4.1),xticks=[0,1,2,3],yticks=[0,1,2,3,4],
          xlabel='较P0折现资本增量 / 万亿元',ylabel='2060净排增量 / Gt')
    clean(c,'y');title(c,'c  全48设计的资本与净排')
    c.legend(frameon=False,ncol=3,loc='upper left',bbox_to_anchor=(-.04,1.03),
             handlelength=.7,columnspacing=.65,borderaxespad=0)
    flip=pairs.loc[pairs.old_reliable & ~pairs.new_reliable]
    assert len(flip)==1 and flip.iloc[0].design_id=='joint_021' and flip.iloc[0].policy=='P3'
    flip=flip.iloc[0]
    cx,cy=flip.new_minus_P0_discounted_capital_trillion_cny,flip.new_minus_P0_net_emissions_2060_mt/1000
    c.scatter(cx,cy,s=96,facecolors='none',edgecolors=INK,lw=1.1)
    c.annotate('#21',xy=(cx,cy),xytext=(.45,3.53),arrowprops={'arrowstyle':'-','lw':.8,'color':INK},fontweight='bold')
    r=designs.loc[designs['case'].eq('joint_021')].iloc[0]
    parameter_columns=['gdp_growth_multiplier','annual_budget_multiplier','import_multiplier','construction_delay_multiplier']
    values=r[parameter_columns].astype(float).to_numpy()
    d.axvline(1,color=MUTED,lw=.9,ls='--')
    for i,value in enumerate(values):
        color=BLUE if value>=1 else ORANGE
        d.plot([1,value],[i,i],color=color,lw=2)
        d.scatter(value,i,s=31,color=color,zorder=3)
        d.text(value,i-.14,f'{value:.3f}',ha='center',va='bottom',color=color)
    d.set(yticks=range(4),yticklabels=['GDP增速','年度预算','进口可用','建设工期'],xlim=(.54,1.58),xticks=[.6,1,1.4],xlabel='相对名义参数 / 倍')
    d.set_ylim(3.45,-.6);clean(d);title(d,'d  #21的四项扰动')
    fig.text(.115,.294,'e  #21：相同设计下的 P0 → P3',fontsize=11,fontweight='bold')
    endpoints=annual.loc[annual.design_id.eq('joint_021') & annual.year.eq(2060)].set_index('policy')
    r0,r3=(dev.loc[dev.design_id.eq('joint_021') & dev.policy.eq(policy)].iloc[0] for policy in ('P0','P3'))
    metrics3=[('折现资本 / 万亿元',[r0.discounted_capital_trillion_cny,r3.discounted_capital_trillion_cny],78,[0,40,80]),
              ('2060净排 / Gt',[r0.net_emissions_2060_mt/1000,r3.net_emissions_2060_mt/1000],16,[0,8,16]),
              ('2060电力缺供 / %',[(endpoints.loc[p,'electricity_unserved_twh']/endpoints.loc[p,'electricity_demand_twh']*100) for p in ('P0','P3')],1.35,[0,.5,1])]
    comparisons=[]
    for j,(label,vals,ymax,ticks) in enumerate(metrics3):
        ax=fig.add_axes([.115+j*.295,.085,.235,.155])
        ax.bar([0,1],vals,width=.48,color=[MUTED,BLUE],zorder=3)
        for x,value in enumerate(vals):
            ax.text(x,value+ymax*.035,f'{value:.3f}',ha='center',va='bottom',color=[MUTED,BLUE][x])
            comparisons.append(dict(design_id='joint_021',policy=['P0','P3'][x],metric=label,value=float(value)))
        if j==2:
            ax.axhline(1,color=ORANGE,lw=1,ls='--')
            ax.text(.025,1.04,'1%门槛',color=ORANGE,transform=ax.get_yaxis_transform(),va='bottom')
        ax.set(xlim=(-.6,1.6),ylim=(0,ymax),xticks=[0,1],xticklabels=['P0','P3'],yticks=ticks,xlabel=label)
        clean(ax,'y')
    keep('deadline_all_144_paired_paths',pairs);keep('deadline_all_196_case_metrics',metrics)
    keep('deadline_all_48_designs',designs);keep('deadline_all_7056_year_values',annual)
    keep('deadline_subgroup_counts',groups);keep('deadline_p2_six_changed',pd.DataFrame(selected))
    keep('deadline_design21_comparisons',pd.DataFrame(comparisons))
    CHECKS['deadline']={'paired_rows':144,'development_designs':48,'case_metrics':196,'annual_rows':7056,
                        'P2_nonzero_designs':changed,'P2_numerical_zero_designs':42,
                        'design21_parameters':dict(zip(parameter_columns,map(float,values))),
                        'design21_P0_P3_values':comparisons,'failed_comparison':'P3 minus P0',
                        'annual_reliability_gate_fraction':.01,'new_LP_calls':0}
    finish(fig,'figP_deadline_investment_mechanisms',
           '建设时点机制的完整开发域比较。a按油工期分组，每组16个设计；全48设计可靠数为20/20/20/19。b显示P2相对P0油缺口的6个非零差，其余42组该指标为数值零。c保留全部48设计的资本—净排变化。d以名义值1标记第21号设计的四项扰动；e分别以完整坐标比较该设计P0与P3的资本、净排及电力缺供，1%虚线为可靠门槛。资本按4%、2024原点折现；净排和电力缺供取2060年度。',
           ['P0为保供修复原控制；P1仅在有效投产窗口优先支付；P2另补窗口内油气建设上限；P3在全期补上限并优先。各臂48个固定开发设计加1条名义路径。',
            '逐年五服务缺供≤1%判为年度可靠。同一设计的预算金额相同；P3相对P0的#21失效不单独辨识资金挤出因果。',
            'b空心为差值0、实心为P2−P0，三位索引保持原始设计身份。所有196条指标、144对比及7056条年度原值保留。'],set(INPUTS)-start)


def ribbon(ax,x0,x1,y0a,y0b,y1a,y1b,color):
    mid=(x0+x1)/2
    vertices=[(x0,y0a),(mid,y0a),(mid,y1a),(x1,y1a),
              (x1,y1b),(mid,y1b),(mid,y0b),(x0,y0b),(x0,y0a)]
    codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,
           MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices,codes),facecolor=color,edgecolor='none',alpha=.24))


def replenishment():
    start=set(INPUTS)
    contrasts=read('outputs/v14/replenishment_figures/contract_minus_inherited.csv')
    points=read('outputs/v14/replenishment_figures/window_floor_points.csv')
    fuels=read('outputs/v14/replenishment_figures/fuel_resource_and_cash.csv')
    assert len(contrasts)==4 and len(points)==48 and points.plan.nunique()==4
    shared=['base_procurement_million_cny','premium_million_cny','procurement_cash_million_cny',
            'physical_resource_delta_million_cny','additional_terminal_fuel_asset_value_million_cny']
    assert contrasts[shared].max().sub(contrasts[shared].min()).abs().max()<=1e-6
    row=contrasts.iloc[0]
    base,premium,cash,net=[float(row[col])/100 for col in shared[:4]]
    assert abs(base+premium-cash)<1e-8 and abs(net-premium)<1e-6
    assert abs(base-row.additional_terminal_fuel_asset_value_million_cny/100)<1e-6
    fig=plt.figure(figsize=(160/25.4,153/25.4))
    a=fig.add_axes([.025,.56,.95,.36]);a.set(xlim=(0,100),ylim=(0,100));a.set_axis_off();a.set_xticks([]);a.set_yticks([])
    title(a,'a  新增采购现金的去向 / 亿元',pad=7)
    total_height=46.;asset_height=total_height*base/cash;fee_height=total_height*premium/cash
    source_bottom=25.;source_split=source_bottom+fee_height
    ribbon(a,22,72,source_split,source_bottom+total_height,58,58+asset_height,BLUE)
    ribbon(a,22,72,source_bottom,source_split,16,16+fee_height,ORANGE)
    a.add_patch(Rectangle((2,source_bottom),20,total_height,facecolor=INK,edgecolor='none'))
    a.text(12,source_bottom+total_height/2,'采购现金\n'+f'{cash:.3f}',ha='center',va='center',color='white',fontweight='bold',linespacing=1.6)
    a.add_patch(Rectangle((72,58),26,asset_height,facecolor=BLUE,edgecolor='none'))
    a.text(85,58+asset_height/2,'末库存资产\n'+f'{base:.3f}',ha='center',va='center',color='white',fontweight='bold',linespacing=1.4)
    a.add_patch(Rectangle((72,16),26,fee_height,facecolor=ORANGE,edgecolor='none'))
    a.text(85,16+fee_height/2,'资源净增费\n'+f'{net:.3f}',ha='center',va='center',color='white',fontweight='bold',linespacing=1.2)
    a.text(46,80,f'基价 {base/cash*100:.1f}%',color=BLUE,ha='center')
    a.text(46,13,f'溢价 {premium/cash*100:.1f}%',color=ORANGE,ha='center')
    labels=['原折中','方案Ⅰ','方案Ⅱ','方案Ⅲ']
    plans=['knee','repair_005','repair_006','repair_007']
    plotrows=[]
    titles=['b  A：事件前基准','c  B：期初基准','d  C：无事件目标']
    for j,(reference,paneltitle) in enumerate(zip(('A','B','C'),titles)):
        ax=fig.add_axes([.12+j*.294,.115,.252,.345])
        ax.axvspan(85,95,color='#FBF4EB',zorder=0)
        ax.axvline(95,color=ORANGE,lw=.95,ls='--')
        for i,plan in enumerate(plans):
            part=points.loc[points.plan.eq(plan)&points.event.eq(2)&points.reference.eq(reference)].set_index('supply_policy')
            old,new=[float(part.loc[policy,'floor_ratio'])*100 for policy in ('inherited','contract')]
            y=3-i
            ax.plot([old,new],[y,y],color=COLORS[i],lw=1.7,zorder=2)
            ax.scatter(old,y,s=32,facecolors='white',edgecolors=COLORS[i],lw=1.25,zorder=4)
            ax.scatter(new,y,s=32,color=COLORS[i],zorder=4)
            for value,offset in [(old,9),(new,-13)]:
                ax.annotate(f'{value:.3f}',(value,y),xytext=(0,offset),textcoords='offset points',
                            ha='center',va='center',color=COLORS[i])
            plotrows.append(dict(plan=plan,display_plan=labels[i],reference=reference,event=2,
                                 inherited_percent=old,contract_percent=new,contract_minus_inherited_percentage_point=new-old))
        ax.set(xlim=(85,101.5),ylim=(-.65,3.65),xticks=[85,90,95,100],yticks=range(4),
               yticklabels=list(reversed(labels)) if j==0 else [],xlabel='最低库存比 / %')
        clean(ax);title(ax,paneltitle)
    fig.legend(handles=[Line2D([],[],marker='o',color=MUTED,markerfacecolor='white',linestyle='none',label='原方案'),
                        Line2D([],[],marker='o',color=MUTED,linestyle='none',label='补库合同')],
               loc='lower center',bbox_to_anchor=(.57,.51),ncol=2,frameon=False,handlelength=1,columnspacing=1.7)
    keep('replenishment_all_four_cash_contrasts',contrasts);keep('replenishment_all_48_reference_points',points)
    keep('replenishment_all_fuel_resource_cash',fuels);keep('replenishment_visible_12_pairs',pd.DataFrame(plotrows))
    CHECKS['replenishment']={'contrast_rows':4,'reference_points_all_windows':48,'visible_paired_points':12,
          'cash_yi_cny':cash,'base_asset_yi_cny':base,'premium_yi_cny':premium,'resource_net_increment_yi_cny':net,
          'ribbon_heights_proportional_to_cash':True,'common_cash_spread_max_million_cny':float(contrasts[shared].max().sub(contrasts[shared].min()).abs().max()),
          'window_half_open':[336,360],'first_joint_confirmation_hours':[119,263],
          'plans':dict(zip(plans,labels)),'new_LP_calls':0}
    finish(fig,'figN_replenishment_cash_and_readiness',
           '补库的现金分解与资源准备。a按实际金额比例将1666.523亿元采购现金分为1111.016亿元新增末库存资产和555.508亿元溢价，后者等于物理资源净增费；基价不重复计入资源净耗费。b—d显示h336—359逐燃料、逐小时库存比最低值，空心原方案→实心合同。A、B、C分别以自身事件前库存、自身期初库存及相同资产下原无事件资源路径为参照；95%虚线为相应参照值。',
           ['四方案实际新增采购账相同；原始完整精度现金、资源账和48个双事件参照点保留。',
            '四对非库存动作、ENS、峰值与CO₂相同，首次联合确认均为119/263h。只有A及服务条件进入原联合恢复判据，B/C为资源诊断。',
            '合同按预定日历在事件结束12h后供货；总进口最高可达年能力折小时的150%，需额外通道。本采购账不含新增运输通道资本。'],set(INPUTS)-start)


def network():
    start=set(INPUTS)
    bind('simulation/anymath_v10/v10_dispatch.jl')
    bind('simulation/anymath_v10/v10_native_plant.jl')
    fig=plt.figure(figsize=(160/25.4,98/25.4))
    ax=fig.add_axes([.015,.015,.97,.97]);ax.set(xlim=(0,100),ylim=(0,100));ax.set_axis_off();ax.set_xticks([]);ax.set_yticks([])
    def box(x,y,w,h,title_,detail='',color='#F2F6F8',edge='#A8BAC3'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.35,rounding_size=1.2',
                                   facecolor=color,edgecolor=edge,lw=.8))
        ax.text(x+w/2,y+h-5,title_,ha='center',va='top',fontweight='bold',fontsize=11)
        if detail:ax.text(x+w/2,y+5,detail,ha='center',va='bottom',linespacing=1.4)
    def arrow(points,color=BLUE,dashed=False):
        xs,ys=zip(*points)
        ax.plot(xs[:-1],ys[:-1],color=color,lw=1.1,ls='--'if dashed else '-')
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=10,
                                     color=color,lw=1.1,linestyle='--'if dashed else '-'))
    box(2,77,27,21,'年度快照','初态 · 资产 · 通道',color='#F8F2E9')
    box(36,77,27,21,'共同24h前瞻','负荷 · 到货 · 可用出力',color='#F8F2E9')
    box(71,77,27,21,'当期外部输入','到货₃ · 直供₃ · 电力₁',color='#F8F2E9')
    box(2,31,22,30,'参数干预','强度 · 持续期\n合同 · 功率 · 规则')
    box(34,31,29,30,'24h滚动优化','首小时执行 · 18维',color='#EDF5F7',edge=BLUE)
    for i in range(24):
        ax.add_patch(Rectangle((36+i*1.04,43),.70,4.8,facecolor=BLUE if i==0 else '#DAE7EC',edgecolor='none'))
    box(75,31,23,30,'原生守恒网络','',color='#EEF7F5',edge=TEAL)
    for x,y,text_ in [(76.5,44,'燃料 × 3'),(87.5,44,'SOC × 2'),(76.5,34,'出力 × 2'),(87.5,34,'债务 × 1')]:
        ax.text(x+4.7,y,text_,ha='center',va='bottom',fontsize=10,color=TEAL)
    arrow([(15.5,76.7),(15.5,70),(42,70),(42,61.5)],MUTED)
    ax.text(22,70.5,'单向初始化',va='bottom',color=MUTED)
    arrow([(49.5,76.7),(49.5,61.5)])
    arrow([(84.5,76.7),(84.5,61.5)])
    ax.text(86,69,'7维',color=MUTED)
    arrow([(24.5,46),(33.5,46)])
    arrow([(63.5,46),(74.5,46)])
    ax.text(69,50,'18维',ha='center',color=BLUE)
    arrow([(86.5,30.5),(86.5,17),(48.5,17),(48.5,30.5)],TEAL,True)
    ax.text(68,10.7,'8维实际状态 → 下一小时',ha='center',color=TEAL)
    ax.text(2,4,'连接示意',color=MUTED)
    ax.text(98,4,'Δt = 1h',ha='right',color=MUTED)
    CHECKS['network']={'source_contract':['simulation/anymath_v10/v10_dispatch.jl','simulation/anymath_v10/v10_native_plant.jl'],'rolling_horizon_hours':24,
         'executed_horizon_hours':1,'executed_action_dimensions':18,'external_input_dimensions':7,
         'returned_state_dimensions':8,'state_components':[3,2,2,1],
         'annual_to_hourly':'one-way initial state and capacity; no annual feedback',
         'diagram_type':'connection schematic, not software screenshot','new_native_calls':0}
    finish(fig,'native_closed_loop',
           '已实现小时系统的分层连接。年度快照单向提供初态、资产和通道能力，优化器与原生网络使用同源初态；共同24小时前瞻进入滚动优化，只执行首小时18维动作。7维当期外部输入与动作进入原生守恒网络，三燃料库存、两类SOC、两类热机出力记忆及一项移峰债务共8维实际状态反馈至下一小时，步长1h。此图为连接示意；箭头宽度不表示流量。',
           ['外部输入为三种到货、三种直接燃料需求和一维电力负荷。状态单位依次为Mtce、GWh、GW及GWh。',
            '滚动优化使用Julia/JuMP/HiGHS，在共同前瞻下处理资源、功率、爬坡与债务约束；图中24格中仅首格着色表示仅执行首小时。会话把同源年度初态和各小时实际返回状态传给原生记忆。',
            '干预项为冲击强度与持续期、到货合同、储能功率及协调/局部规则。图未重建或修饰原生软件界面。'],set(INPUTS)-start)


def render_all():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    INPUTS.clear();INDEX.clear();CHECKS.clear();LAYOUT.clear();style()
    bind(Path(__file__).relative_to(ROOT))
    deadline();replenishment();network()
    drift=[name for name,value in INPUTS.items() if sha(ROOT/name)!=value]
    assert not drift,drift
    (OUT/'evidence_figure_index.json').write_text(json.dumps(INDEX,ensure_ascii=False,indent=2),'utf-8')
    (OUT/'evidence_visuals_audit.json').write_text(json.dumps({
        'status':'PASS_FROZEN_VALUES_AND_LAYOUT_PENDING_VISUAL_REVIEW',
        'completed_utc':datetime.now(timezone.utc).isoformat(),
        'actual_LP_calls':0,'new_simulation_calls':0,'new_native_calls':0,
        'source_input_sha256':INPUTS,'source_drift':drift,'data_checks':CHECKS,'figure_layouts':LAYOUT,
        'output_sha256':{p.relative_to(ROOT).as_posix():sha(p) for folder in (FIG,OUT) for p in sorted(folder.iterdir())
                         if p.is_file() and (p.name.startswith('evidence_') or p.stem in INDEX)
                         and p.name!='evidence_visuals_audit.json'},
        'scope':'Presentation only; all full frozen source tables copied without selecting scientific runs.',
        'minimum_font_exception':'10pt numeric labels/ticks at 160mm width; larger 11pt panel titles. Data-dense multi-panel figure is 181mm high; the connection schematic is 98mm high.'},ensure_ascii=False,indent=2),'utf-8')
    return INDEX


if __name__=='__main__':
    print(json.dumps({'status':'BUILT','figures':list(render_all())},ensure_ascii=True))





