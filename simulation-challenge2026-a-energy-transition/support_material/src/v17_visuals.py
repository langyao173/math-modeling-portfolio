"""V17 figure presentation from frozen, already executed evidence.

AI-assisted: OpenAI Codex, 2026-10-19. No solver or model is imported/run.
Native screenshots and model exports remain unchanged; Chinese explanations
are placed outside the original image. New network graphics are labelled as
connection schematics, never software captures.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/v17/visuals'
FIG = ROOT / 'figures/V17/visuals'
OLD = ROOT / 'paper/V16成果稿/figures'
FONT = Path('C:/Windows/Fonts/msyh.ttc')
COLORS = ['#247BA0', '#E09F3E', '#688C52', '#8C6BB1']
INK, GRAY, LIGHT = '#243B4A', '#7B8790', '#E5EAED'
INPUTS, INDEX, LAYOUT, DATA_CHECKS = {}, {}, {}, {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    p = ROOT / path
    INPUTS[p.relative_to(ROOT).as_posix()] = sha(p)
    return p


def read(path):
    return pd.read_csv(bind(path))


def keep_table(name, frame):
    frame.to_csv(OUT / name, index=False, float_format='%.17g')


def style():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({'font.family': [font_manager.FontProperties(fname=FONT).get_name(), 'DejaVu Sans'],
        'font.size': 8.8, 'axes.labelsize': 8.8, 'axes.titlesize': 9.4,
        'xtick.labelsize': 8.8, 'ytick.labelsize': 8.8, 'legend.fontsize': 8.8,
        'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': GRAY,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.unicode_minus': False, 'grid.color': LIGHT, 'grid.linewidth': .5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def clean(ax):
    ax.grid(axis='y'); ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=.6)
    for spine in ax.spines.values(): spine.set_linewidth(.6)


def register(name, caption_notes, height):
    INDEX[name] = {ext: (FIG / f'{name}.{ext}').relative_to(ROOT).as_posix() for ext in ('pdf', 'svg', 'png')}
    INDEX[name].update(caption_notes=caption_notes, width_mm=160, height_mm=height)


def inspect_pdf(name):
    with fitz.open(FIG / f'{name}.pdf') as d:
        p = d[0]
        p.get_pixmap(matrix=fitz.Matrix(180 / 72, 180 / 72)).save(OUT / f'{name}__actual_pdf.png')
        spans = [s for b in p.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
        return {'actual_pdf_page_count': len(d), 'pdf_page_mm': [p.rect.width * 25.4 / 72, p.rect.height * 25.4 / 72],
                'minimum_decoded_vector_text_pt': min((s['size'] for s in spans), default=None)}


def save(fig, name, notes):
    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); canvas = fig.bbox
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    outside = []
    for t in texts:
        bb = t.get_window_extent(renderer)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > canvas.width + 1 or bb.y1 > canvas.height + 1:
            outside.append(t.get_text())
    assert not outside, (name, outside)
    assert min(t.get_fontsize() for t in texts) >= 8.8
    for ext in ('pdf', 'svg', 'png'): fig.savefig(FIG / f'{name}.{ext}', dpi=300)
    height = fig.get_figheight() * 25.4
    plt.close(fig)
    register(name, notes, height)
    LAYOUT[name] = dict(minimum_matplotlib_font_pt=8.8, out_of_canvas_text=outside, **inspect_pdf(name))
    assert LAYOUT[name]['minimum_decoded_vector_text_pt'] >= 8.79


def investment():
    annual = {s: read(f'figures/competition_v7/figB_investment_transmission__annual_{s}.csv') for s in ('full', 'profit_off')}
    years = annual['full'].year.to_numpy()
    assert np.array_equal(years, annual['profit_off'].year) and len(years) == 36
    full_cap = annual['full'].loc[annual['full'].market_price_cny_mwh.ge(1200 - 1e-8), 'year'].tolist()
    off_cap = annual['profit_off'].loc[annual['profit_off'].market_price_cny_mwh.ge(1200 - 1e-8), 'year'].tolist()
    assert full_cap == list(range(2028, 2048))
    assert off_cap == list(range(2028, 2049)) + [2060]
    fig, axes = plt.subplots(4, 1, figsize=(160 / 25.4, 156 / 25.4), sharex=True,
                             gridspec_kw={'left': .14, 'right': .965, 'top': .91, 'bottom': .09, 'hspace': .55})
    rows = []
    for j, (tech, label) in enumerate([('wind', '风电'), ('solar', '光伏'), ('gas', '气电'), ('nuclear', '核电')]):
        full, off = (read(f'figures/competition_v7/figB_investment_transmission__{tech}_{s}.csv') for s in ('full', 'profit_off'))
        assert np.array_equal(full.year, years) and np.array_equal(off.year, years)
        axes[0].plot(years, full.normalized_profit_signal, color=COLORS[j], label=label, lw=1.35)
        for k, (field, title) in enumerate([('order', '实际订单'), ('pending', '期末在建'), ('commissioned', '当年投产')], 1):
            a = annual['full'][tech + '_pending'].to_numpy() if field == 'pending' else full[field].to_numpy()
            b = annual['profit_off'][tech + '_pending'].to_numpy() if field == 'pending' else off[field].to_numpy()
            delta = a - b
            axes[k].plot(years, delta, color=COLORS[j], lw=1.3)
            axes[k].fill_between(years, 0, delta, color=COLORS[j], alpha=.08)
            for y, av, bv, dv in zip(years, a, b, delta):
                rows.append(dict(technology=tech, year=int(y), metric=field, full=float(av), profit_off=float(bv), full_minus_profit_off=float(dv)))
        for y, a, b in zip(years, full.normalized_profit_signal, off.normalized_profit_signal):
            rows.append(dict(technology=tech, year=int(y), metric='normalized_profit_signal', full=float(a), profit_off=float(b), full_minus_profit_off=float(a-b)))
    titles = ['a  完整反馈下的标准化收益', 'b  收益响应带来的实际订单增量', 'c  收益响应带来的全部在建增量', 'd  收益响应带来的投产增量']
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], loc='left', pad=5, fontweight='bold')
        ax.axvspan(2027.5, 2047.5, color='#F6E7CE', alpha=.48, zorder=-5)
        if i: ax.axhline(0, color=GRAY, lw=.6)
        ax.set_ylabel('无量纲' if i == 0 else ('Δ GW' if i == 2 else 'Δ GW/年'))
        ax.set_xlim(2025, 2060); ax.set_xticks([2025, 2030, 2040, 2050, 2060]); clean(ax)
    axes[0].legend(ncol=4, loc='lower center', bbox_to_anchor=(.5, 1.25), frameon=False, handlelength=1.5, columnspacing=1.2)
    axes[-1].set_xlabel('年份')
    keep_table('investment_all_576_values.csv', pd.DataFrame(rows))
    keep_table('investment_all_72_market_prices.csv', pd.concat([v[['year', 'market_price_cny_mwh']].assign(arm=s) for s, v in annual.items()], ignore_index=True))
    DATA_CHECKS['investment'] = {'complete_comparison_rows': len(rows), 'full_cap_years': full_cap, 'profit_off_cap_years': off_cap,
        'difference_definition': 'full minus investor_profit_response_off', 'shading': 'full arm price=1200 CNY/MWh, 2028..2047 only'}
    save(fig, 'figB_investment_transmission', [
        'a为完整反馈的标准化收益；b—d为完整反馈减关闭投资收益响应的差值，订单/投产单位GW/年、期末在建单位GW。',
        '在建为全部期末工程，包含既有在建；不是单批2025订单。四技术同年对齐，完整两臂原值保留。',
        '浅色区仅表示完整反馈电价达到1200元/MWh上限的2028—2047年；关闭响应为2028—2048年及2060年。该色区不单独识别价格因果。'])


def shocks():
    names = ['inherited_coordinated', 'inherited_decentralized', 'contract_coordinated']
    tables = {n: read(f'figures/competition_v7/figD_continuous_shocks__{n}.csv') for n in names}
    assert all(len(q) == 2160 and np.array_equal(q.hour, np.arange(2160)) for q in tables.values())
    q, dec, contract = (tables[n] for n in names)
    denom = q.loc[q.hour.eq(119), ['coal_stock_mtce', 'oil_stock_mtce', 'gas_stock_mtce']].iloc[0]
    assert np.array_equal(denom.to_numpy(), contract.loc[contract.hour.eq(119), denom.index].iloc[0].to_numpy())
    fig = plt.figure(figsize=(160 / 25.4, 156 / 25.4))
    a = fig.add_axes([.14, .78, .82, .14]); b = fig.add_axes([.14, .57, .82, .14])
    c = fig.add_axes([.14, .32, .34, .16]); d = fig.add_axes([.64, .32, .32, .16])
    e = fig.add_axes([.14, .075, .82, .145])
    daily = {n: t.assign(day=t.hour // 24).groupby('day').mean(numeric_only=True) for n, t in tables.items()}
    x = daily[names[0]].index.to_numpy() + .5
    a.plot(x, daily[names[0]].demand_gw / 1000, color=GRAY, label='需求')
    a.plot(x, daily[names[0]].served_power_gw / 1000, color=COLORS[0], label='协调交付')
    b.plot(x, daily[names[1]].unserved_gw, color=GRAY, ls='--', label='分散')
    b.plot(x, daily[names[0]].unserved_gw, color=COLORS[0], label='协调')
    zoom = q.hour.between(55 * 24, 57 * 24 - 1)
    for t, co, ls in [(dec, GRAY, '--'), (q, COLORS[0], '-')]: c.step(t.hour[zoom] / 24, t.unserved_gw[zoom], color=co, ls=ls, lw=1.2, where='post')
    for field, co, label in [('battery_soc_gwh', COLORS[0], '电池'), ('pumped_soc_gwh', COLORS[2], '抽蓄')]:
        d.step(q.hour[zoom] / 24, q.loc[zoom, field] / 1000, color=co, label=label, where='post', lw=1.2)
    raw = []
    for j, (fuel, label) in enumerate([('coal', '煤'), ('oil', '油'), ('gas', '气')]):
        for arm, t, ls in [('继承供应', q, '-'), ('付费补库', contract, '--')]:
            values = t[fuel + '_stock_mtce'].to_numpy() / denom[fuel + '_stock_mtce'] * 100
            e.plot(t.hour / 24, values, color=COLORS[j], ls=ls, lw=1.15)
            raw.extend(dict(arm=arm, fuel=fuel, hour=int(h), stock_mtce=float(s), first_pre_event_stock_mtce=float(denom[fuel+'_stock_mtce']), retained_percent=float(v)) for h, s, v in zip(t.hour, t[fuel+'_stock_mtce'], values))
    e.axhline(95, color=GRAY, lw=.8, ls=':')
    handles = [Line2D([], [], color=COLORS[j], label=label) for j, label in enumerate(['煤', '油', '气'])]
    e.legend(handles=handles, ncol=3, loc='upper left', frameon=False, handlelength=1.3, columnspacing=.8, borderaxespad=.15)
    for ax in (a, b, e):
        for start, end in [(5, 25), (50, 70)]: ax.axvspan(start, end, color='#F2EADF', alpha=.65, zorder=-5)
        ax.set_xlim(0, 90); ax.set_xticks([0, 25, 50, 70, 90]); clean(ax)
    for ax in (a, b):
        ax.axvspan(55, 57, color=COLORS[0], alpha=.15); ax.tick_params(labelbottom=False)
    for ax in (c, d): ax.set_xlim(55, 57); ax.set_xticks([55, 56, 57]); ax.set_xlabel('实验时间 / 天'); clean(ax)
    for ax, title in zip((a,b,c,d,e), ['a  90天供需概览（日均）', 'b  90天缺供概览（日均）', 'c  55—57天：实际小时缺供', 'd  同期实际小时SOC', 'e  三燃料库存的共同参照']):
        ax.set_title(title, loc='left', fontweight='bold', pad=5)
    a.set_ylabel('功率 / TW'); b.set_ylabel('缺供 / GW'); c.set_ylabel('缺供 / GW'); d.set_ylabel('SOC / TWh'); e.set_ylabel('库存保有率 / %'); e.set_xlabel('连续实验时间 / 天')
    a.legend(ncol=2, loc='upper right', frameon=False, borderaxespad=0, handlelength=1.3)
    b.legend(ncol=2, loc='upper right', frameon=False, borderaxespad=0, handlelength=1.3)
    d.legend(ncol=2, loc='upper right', frameon=False, borderaxespad=0, handlelength=1)
    keep_table('continuous_all_12960_stock_ratios.csv', pd.DataFrame(raw))
    for name, table in tables.items(): keep_table('continuous_' + name + '_all_2160.csv', table)
    DATA_CHECKS['continuous'] = {'saved_paths': 3, 'saved_hours_per_path': 2160, 'stock_ratio_rows': len(raw), 'reference_hour': 119,
        'reference_values_mtce': denom.to_dict(), 'reference_line_percent':95, 'reference_is_joint_recovery_gate':False,
        'shade_actual_stress_hours': [[120,600], [1200,1680]], 'zoom_hours_half_open':[1320,1368]}
    save(fig, 'figD_continuous_shocks', [
        '年度折中2060快照；主窗90天/2160小时，a—b为日均、c—d为55—57天实际小时值，浅色标识同一两段冲击。',
        'e的三燃料、两供给方案统一各燃料首次冲击前h119库存为分母；实线继承、虚线付费补库。95%仅是该固定分母的参照线，不是每次事件前归一的连续24小时联合恢复判据。',
        '三燃料比率可同轴比较；各燃料原值及完整3×2160小时轨迹另表，所有单位分别标注。'])


def deadline():
    metrics = read('outputs/v14/deadline_investment_figures/case_metrics_recomputed.csv')
    pairs = read('outputs/v14/deadline_investment_figures/all_144_development_paired_paths.csv')
    groups = read('outputs/v14/deadline_investment_figures/complete_subgroup_counts_recomputed.csv')
    designs = read('outputs/v14/deadline_investment/development_design_frozen.csv')
    annual = read('outputs/v14/deadline_investment_figures/all_7056_yearly_key_values.csv')
    dev = metrics[metrics.cohort.eq('development')]
    p2 = pairs[pairs.policy.eq('P2')].set_index('design_id')
    changed = sorted(p2.index[p2.new_minus_P0_oil_2030_shortfall_fraction.abs().gt(1e-8)])
    assert changed == ['joint_000', 'joint_007', 'joint_018', 'joint_032', 'joint_038', 'joint_039']
    fig = plt.figure(figsize=(160 / 25.4, 164 / 25.4))
    a = fig.add_axes([.13, .68, .34, .23]); b = fig.add_axes([.65, .68, .31, .23])
    c = fig.add_axes([.13, .12, .34, .32]); d = fig.add_axes([.65, .12, .32, .32])
    ys = np.arange(4)
    for i, lead in enumerate((4, 5, 6)):
        vals = [int(groups.loc[groups.policy.eq(p) & groups.group.eq('oil_lead_'+str(lead)), 'reliable_count'].iloc[0]) for p in ['P0','P1','P2','P3']]
        a.barh(ys + (i-1)*.21, vals, height=.19, color=COLORS[i], label=f'油工期{lead}年')
        for yy, v in zip(ys + (i-1)*.21, vals): a.text(v+.15, yy, str(v), va='center', fontsize=8.8)
    a.set(yticks=ys, yticklabels=['P0','P1','P2','P3'], xlim=(0,14), xticks=[0,5,10], xlabel='年度可靠设计数 / 每组16')
    a.invert_yaxis(); a.grid(axis='x'); a.set_axisbelow(True)
    fig.legend(handles=[Rectangle((0,0),1,1,color=COLORS[i],label=f'{lead}年')for i,lead in enumerate((4,5,6))],
               ncol=3,loc='upper left',bbox_to_anchor=(.13,.995),frameon=False,handlelength=.8,columnspacing=.8)
    selected = []
    for i, case in enumerate(changed):
        row0 = dev.loc[dev.design_id.eq(case)&dev.policy.eq('P0')].iloc[0]
        row2 = dev.loc[dev.design_id.eq(case)&dev.policy.eq('P2')].iloc[0]
        old, new = row0.oil_2030_shortfall_fraction*100, row2.oil_2030_shortfall_fraction*100
        delta=new-old
        b.plot([0,delta],[i,i],color=COLORS[0],lw=1.5)
        b.scatter(0,i,facecolors='white',edgecolors=COLORS[0],s=23,zorder=3)
        b.scatter(delta,i,color=COLORS[0],s=18,zorder=3)
        b.text(.07,i,f'{delta:.4f}',va='center',fontsize=8.8)
        selected.append(dict(design_id=case,P0_percent=old,P2_percent=new,P2_minus_P0_percentage_point=new-old))
    b.set(yticks=range(6), yticklabels=['#'+t[-3:] for t in changed], xlabel='P2−P0油缺口 / 百分点')
    b.set_xlim(-.85,.58); b.set_xticks([-.8,-.4,0])
    b.invert_yaxis(); b.grid(axis='x'); b.set_axisbelow(True)
    for i,p in enumerate(['P0','P1','P2','P3']):
        part = pairs.loc[pairs.policy.eq(p)] if p!='P0' else None
        if part is not None:
            c.scatter(part.new_minus_P0_discounted_capital_trillion_cny, part.new_minus_P0_net_emissions_2060_mt/1000,
                      marker=['o','s','^'][i-1],color=COLORS[i-1],label=p,s=17,alpha=.65,linewidths=.4)
    c.axhline(0,color=GRAY,lw=.6); c.axvline(0,color=GRAY,lw=.6)
    c.set(xlabel='较P0折现资本增量 / 万亿元', ylabel='2060净排增量 / Gt', xlim=(-.1,3.6), ylim=(-.2,4.2))
    c.legend(ncol=3,loc='upper right',frameon=False,handlelength=.7,columnspacing=.5,borderaxespad=.1);clean(c)
    flip = pairs.loc[pairs.old_reliable & ~pairs.new_reliable]
    assert len(flip)==1 and flip.iloc[0].design_id=='joint_021' and flip.iloc[0].policy=='P3'
    flip = flip.iloc[0]
    c.scatter(flip.new_minus_P0_discounted_capital_trillion_cny, flip.new_minus_P0_net_emissions_2060_mt/1000,s=75,facecolors='none',edgecolors=INK,lw=.9)
    c.annotate('#21：P3失效',xy=(flip.new_minus_P0_discounted_capital_trillion_cny, flip.new_minus_P0_net_emissions_2060_mt/1000),xytext=(.3,3.55),arrowprops={'arrowstyle':'-','lw':.7,'color':INK})
    r = designs.loc[designs['case'].eq('joint_021')].iloc[0]
    r0, r3 = (dev.loc[dev.design_id.eq('joint_021')&dev.policy.eq(p)].iloc[0] for p in ('P0','P3'))
    endpoint = annual.loc[annual.design_id.eq('joint_021')&annual.policy.eq('P3')&annual.year.eq(2060)].iloc[0]
    failure = endpoint.electricity_unserved_twh/endpoint.electricity_demand_twh*100
    assert 1.025 < failure < 1.026
    d.set_axis_off()
    d.text(0,1,'第21号扰动设计',fontweight='bold',va='top')
    lines = [f'GDP增速倍率  {r.gdp_growth_multiplier:.3f}',f'年度预算倍率  {r.annual_budget_multiplier:.3f}',
             f'进口可用倍率  {r.import_multiplier:.3f}',f'新订单工期倍率  {r.construction_delay_multiplier:.3f}',
             '', 'P0 → P3（同一设计）', f'资本  {r0.discounted_capital_trillion_cny:.3f} → {r3.discounted_capital_trillion_cny:.3f} 万亿元',
             f'净排  {r0.net_emissions_2060_mt/1000:.3f} → {r3.net_emissions_2060_mt/1000:.3f} Gt',
             f'2060电力缺供 {failure:.3f}% > 1%']
    for i,line in enumerate(lines):d.text(0,.88-i*.103,line,va='top',color=COLORS[0] if i==5 else INK)
    for ax,t in zip((a,b,c,d), ['a  完整48组：按油工期分组','b  P2油缺口改变的6组','c  全48组：资金与净排','d  第21号设计的实际边界']):
        ax.set_title(t,loc='left',pad=5 if ax==a else 9,fontweight='bold')
    fig.text(.13,.57,'总达标数：P0/P1/P2/P3 = 20/20/20/19（均为48组）',fontsize=8.8)
    fig.text(.13,.53,'b：空心0 → 实心P2−P0；其余42组油缺口差为数值零。',fontsize=8.8)
    keep_table('deadline_all_144_paired_paths.csv',pairs);keep_table('deadline_all_196_case_metrics.csv',metrics)
    keep_table('deadline_all_48_development_designs.csv',designs);keep_table('deadline_p2_six_changed_oil_designs.csv',pd.DataFrame(selected))
    keep_table('deadline_all_subgroup_counts.csv',groups)
    DATA_CHECKS['deadline']={'development_design_count':48,'policies':4,'total_paths_including_nominal':196,
        'P2_nonzero_oil_2030':changed,'P2_numerical_zero_oil_2030':42,'numerical_zero_threshold_fraction':1e-8,
        'full_policy_nonzero_oil_2030':{p:int(x.new_minus_P0_oil_2030_shortfall_fraction.abs().gt(1e-8).sum())for p,x in pairs.groupby('policy')},
        'failure_comparison':'P3 minus P0','failed_design_id':'joint_021','failed_electricity_2060_percent':float(failure),'design_021_parameters':r.to_dict()}
    save(fig,'figP_deadline_investment_mechanisms',[
        'P0为保供修复原控制；P1仅在有效投产窗口优先支付；P2另补窗口内油气建设上限；P3在全期补上限并优先。各臂48个固定开发设计加1名义路径，逐年五服务缺供≤1%为可靠门槛。',
        'b只显示P2相对P0的6个非零油缺口设计；其余42组为该指标的数值零，不能将零差统一归因于建设窗口。P1/P3完整144配对原值同时保留，P3还含小幅正差。设计#000等沿用三位代码索引。',
        '各臂同一设计的逐年预算金额相同；折现资本按4%、2024原点计，净排取2060年度。第21号设计的失效及69.737→71.075万亿元、11.447→14.471Gt均是P3相对P0，不能替换为P3相对P2，也不单独辨识资金挤出因果。'])


def events():
    values = pd.read_csv(bind('outputs/v13/event_design_figures/panel_values.csv'),dtype={'cell_label':str})
    keep_table('event_all_128_panel_raw_values.csv',values)
    keep_table('event_all_cell_contrasts.csv',read('outputs/v13/event_design_figures/all_cell_contrasts.csv'))
    phases, durations = list(range(0,24,3)), [12,24,48,72]
    fig=plt.figure(figsize=(160/25.4,142/25.4))
    cmap=LinearSegmentedColormap.from_list('signed_v17',['#BA5A2A','#F7F1EB','#E8EDF0','#A6CDE3','#00649C'])
    specs=[('a','局部−协调：累计缺供','GWh',400),('b','局部−协调：资源费','十亿元',8),('c','扩功率：峰值下降','GW',80),('d','扩功率：累计缺供下降','GWh',600)]
    for i,(letter,title,unit,bound)in enumerate(specs):
        x=.095 if i%2==0 else .575; y=.64 if i<2 else .19
        ax=fig.add_axes([x,y,.36,.22]);part=values.loc[values.panel.eq(letter)]
        assert len(part)==32
        matrix=part.pivot(index='duration_hours',columns='phase_hours',values='display_difference').reindex(index=durations,columns=phases).to_numpy()
        image=ax.imshow(matrix,cmap=cmap,norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto',interpolation='nearest')
        for r in part.itertuples():
            color='white' if abs(r.display_difference)/bound>.58 else INK
            ax.text(phases.index(r.phase_hours),durations.index(r.duration_hours),str(r.cell_label),ha='center',va='center',fontsize=8.8,color=color)
        ax.set_xticks(range(8),phases);ax.set_yticks(range(4),durations);ax.tick_params(length=0)
        ax.set_xlabel('事件相位 / h');ax.set_ylabel('持续期 / h')
        ax.set_title(letter+'  '+title,loc='left',pad=19,fontweight='bold')
        ax.text(0,1.055,unit+'；32/32格均有值',transform=ax.transAxes,fontsize=8.8)
        ax.set_xticks(np.arange(-.5,8,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=.8);ax.tick_params(which='minor',bottom=False,left=False);ax.set_frame_on(False)
        cax=fig.add_axes([x,y-.105,.36,.013]); cb=fig.colorbar(image,cax=cax,orientation='horizontal',ticks=[-bound,0,bound]);cb.outline.set_visible(False)
        if letter=='d':
            nonzero=part.loc[part.classification.eq('positive')]
            assert sorted(zip(nonzero.phase_hours,nonzero.duration_hours))==[(9,12),(21,12),(21,24),(21,48),(21,72)]
            ax.add_patch(Rectangle((2.5,-.5),1,1,fill=False,edgecolor='#AA4A15',lw=1.5,ls='--'))
            ax.add_patch(Rectangle((6.5,-.5),1,4,fill=False,edgecolor='#AA4A15',lw=1.5,ls='--'))
            ax.text(.5,1.35,'虚线框：累计缺供改善5格',transform=ax.transAxes,ha='center',fontsize=8.8,color='#904218')
    DATA_CHECKS['events']={'panels':4,'cells_per_panel':32,'complete_raw_rows':len(values),
        'positive_energy_cells':5,'numerical_zero_energy_cells':27,'all_zero_are_valid':True,
        'nonzero_boundaries':[{'phase_hours':9,'duration_hours':[12]},{'phase_hours':21,'duration_hours':[12,24,48,72]}],
        'zero_threshold_in_each_panel_display_unit':1e-8,'no_claim_all_27_achieve_night_bound':True}
    save(fig,'figM_event_advantage_domain',[
        'a、b为原功率下可达全电源余量局部减协调；c、d为协调原功率减1.5倍功率，储能能量不变。每面板完整32格，零是有效数值，不是缺失。',
        '虚线仅框出实际5个累计缺供改善格：相位9h/持续12h及相位21h四种持续期；其余27格只判为本试验零净能量改善，不统一声称已到夜间能量下界。',
        '主评价窗h48—359；两事件起点48+相位与192+相位。|差值|≤10^-8按各面板单位显示为0；ENS取整、峰值1位、资源费2位，完整有符号原值保留。',
        '资源费含燃料库存与SOC净领用计价；各臂末SOC可能不同，共同满电终点检验另列。色阶为有符号差值，蓝正橙负。'])


def network():
    bind('src/v10_native_diagram.py')
    fig=plt.figure(figsize=(160/25.4,112/25.4));ax=fig.add_axes([.02,.02,.96,.96]);ax.set_axis_off();ax.set(xlim=(0,100),ylim=(0,100))
    def box(x,y,w,h,title,body,color='#EEF4F7'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.4,rounding_size=1',facecolor=color,edgecolor='#A8BAC3',lw=.8))
        ax.text(x+2,y+h-4,title,fontweight='bold',va='top')
        ax.text(x+2,y+h-12,body,va='top',linespacing=1.6)
    def arrow(a,b,dashed=False):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=1,color=COLORS[0],ls='--'if dashed else '-'))
    box(1,65,26,25,'a  参数干预','强度与持续期\n到货合同、储能功率\n协调或局部规则')
    box(36,65,28,25,'b  24h滚动重优化','Julia / JuMP / HiGHS\n资源、功率、爬坡与债务\n只执行首小时18维动作')
    box(73,50,26,40,'c  原生守恒网络','三燃料库存 / Mtce\n两类SOC / GWh\n煤气出力记忆 / GW\n移峰债务 / GWh')
    box(1,24,26,27,'年度状态与可用能力','折中2060快照\n初态、资产与通道能力\n单向传入小时层','#F8F1E6')
    box(36,24,28,27,'共同24h前瞻','负荷、到货与直接燃料\n各电源可用出力\n当期外部输入7维','#F8F1E6')
    arrow((27,77),(35,77));arrow((64,77),(72,77));arrow((14,51),(14,64));arrow((50,51),(50,64));arrow((64,38),(73,60))
    arrow((86,49),(86,13),True);arrow((86,13),(32,13),True);arrow((32,13),(32,60),True)
    arrow((32,60),(50,60),True);arrow((50,60),(50,64),True)
    ax.text(64,7,'8维实际新状态 → 下一小时优化',ha='center')
    ax.text(1,97,'已实现系统的连接示意（非软件截图）',fontweight='bold')
    ax.text(1,15,'Δt = 1h；实线：执行输入\n虚线：实际状态反馈',va='top',color=GRAY)
    DATA_CHECKS['network']={'source_contract':'src/v10_native_diagram.py','diagram_type':'Chinese explanatory connection schematic, not screenshot',
        'first_action_dimensions':18,'external_input_dimensions':7,'returned_state_dimensions':8,'rolling_horizon_hours':24,'actual_executed_horizon_hours':1}
    save(fig,'native_closed_loop',[
        '本图为已实现小时系统的中文连接示意，不是软件截图，箭头宽度无量化含义；原生软件全景与守恒算子分别展示。',
        '共同24h前瞻下只执行首小时18维动作，当期7维外部输入与实际上期状态进入原生网络，8维新状态返回下一次优化。年度状态单向传入，小时实验由外层会话按1h推进。'])


def pdf_text(page, point, text, size=8.8, color=(.14,.23,.29)):
    page.insert_text(point,text,fontname='cjk',fontsize=size,color=color)


def native_exports():
    name='native_v9_resource_conservation'; w=160/25.4*72; h=170/25.4*72
    doc=fitz.open();p=doc.new_page(width=w,height=h);p.insert_font(fontname='cjk',fontfile=str(FONT))
    src=bind('outputs/v6/native/model_views/v6_annual_native_model.png');pix=fitz.Pixmap(src)
    assert pix.width==2120 and pix.height==760
    # The lower explanatory paragraphs are not native topology; retain the complete upper native graph.
    original=fitz.open();op=original.new_page(width=pix.width,height=pix.height);op.insert_image(op.rect,filename=str(src))
    pdf_text(p,(5,14),'a  实际年度模型顶层：217维状态经主体决策与观测链演化',9.4)
    p.show_pdf_page(fitz.Rect(4,22,w-4,126),original,0,clip=fitz.Rect(0,0,2120,505));original.close()
    panels=[]
    for source,title,target in [
        ('outputs/v9/native/model_coal_conservation.pdf','b  实际煤库存网络：到货 − 直接交付 − 发电耗煤 + 上期库存',(7,156,w-7,294)),
        ('outputs/v9/native/model_storage_conservation.pdf','c  实际储能网络：充电乘效率 − 放电除效率 + 上期SOC',(7,330,w-7,467))]:
        d=fitz.open(bind(source));sp=d[0]
        bounds=[fitz.Rect(item[1])for item in sp.get_bboxlog() if item[0] in ('fill-text','stroke-text','stroke-path')]
        clip=fitz.Rect(bounds[0])
        for box in bounds[1:]:clip|=box
        clip &= sp.rect;clip=fitz.Rect(max(0,clip.x0-3),max(0,clip.y0-3),min(sp.rect.width,clip.x1+3),min(sp.rect.height,clip.y1+3))
        # Move native English formula notes into the Chinese formal caption;
        # keeping the topology alone permits genuinely legible native ports.
        clip.y1=409 if 'coal_'in source else 384
        pdf_text(p,(7,target[1]-10),title,9.4);p.show_pdf_page(fitz.Rect(target),d,0,clip=clip)
        panels.append({'source':source,'clip_points':list(clip),'target_points':list(target)});d.close()
    doc.subset_fonts();doc.save(FIG/f'{name}.pdf',deflate=True,garbage=4);doc.close()
    register(name,[
        'a为真实年度Simulink顶层导出，保留原生封装与观测连接；217为内生状态维度，非217个智能体。具体政府、消费、12资产×15槽队列均在决策函数内实现，未伪装成独立可见子系统。',
        'b、c为实际Simulink煤库存和储能子系统的原生矢量输出；中文标注位于原图外，原图内部英文端口未改写。原英文公式说明移入本图注：煤发电耗煤G/(8141η) Mtce/1h，SOC按ηc·充电−放电/ηd更新，Unit Delay保存上一小时状态。',
        '全景用于识别原生封装上下文，局部矢量用于阅读守恒连接；完整原始高分辨率图及可运行模型随支撑材料保留。'],170)
    finish_wrapped(name)
    LAYOUT[name].update(native_source_pixels=[2120,760],annual_clip_pixels=[0,0,2120,505],native_vector_panels=panels,
        immutable_native_image_scope='Original pixel content and native vector content, external Chinese labels only; full-context raster text may be smaller than 8.8pt')
    name='native_anymath_v10_live';h=170/25.4*72;doc=fitz.open();p=doc.new_page(width=w,height=h);p.insert_font(fontname='cjk',fontfile=str(FONT))
    source=bind('outputs/v10/anymath/ui_tests/anymath_1600_shock.png');pix=fitz.Pixmap(source)
    original=fitz.open();op=original.new_page(width=pix.width,height=pix.height);op.insert_image(op.rect,filename=str(source))
    pdf_text(p,(5,13),'a  实际AnyMath实验台全景：h66、已执行67次原生优化与反馈',9.4)
    p.show_pdf_page(fitz.Rect(4,22,w-4,319),original,0)
    pdf_text(p,(5,330),'b  同一原图局部：运行时钟与本次求解状态',9.4)
    crop1=fitz.Rect(235,130,1055,198)
    p.show_pdf_page(fitz.Rect(7,340,w-7,377),original,0,clip=crop1)
    pdf_text(p,(5,394),'c  同一原图局部：实际交付与缺供',9.4)
    crop2=fitz.Rect(810,272,1018,414)
    p.show_pdf_page(fitz.Rect(275,403,w-7,h-4),original,0,clip=crop2)
    pdf_text(p,(7,419),'同一h66实际状态：')
    pdf_text(p,(7,436),'原始负荷3432.2GW，实际交付2938.0GW。')
    pdf_text(p,(7,453),'未供电494.1GW；全景原图可单独放大。')
    original.close();doc.subset_fonts();doc.save(FIG/f'{name}.pdf',deflate=True,garbage=4);doc.close()
    register(name,[
        '完整截图来自已执行的同一AnyMath闭环实验：h66/359、67次优化与原生反馈、当前LP OPTIMAL；能流、库存、SOC和时钟均属同一小时。b、c仅放大同一原图，不是不同小时拼接。',
        '界面提供数值参数框和下拉选择，并非滑块控件；无新增软件UI或新运行。局部显示未供电494.1GW，完整原图保留实际参数、所有六视图和恢复表。'],170)
    finish_wrapped(name)
    LAYOUT[name].update(native_source=source.relative_to(ROOT).as_posix(),native_source_sha256=sha(source),native_source_pixels=[pix.width,pix.height],
        detail_crops_pixels=[list(crop1),list(crop2)],same_original_screenshot_hour=66,native_runs_started=0,
        immutable_native_image_scope='Unaltered existing same-hour screenshot plus explicit outside Chinese labels; small full-context raster labels not retyped')


def finish_wrapped(name):
    with fitz.open(FIG/f'{name}.pdf') as d:
        p=d[0];(FIG/f'{name}.svg').write_text(p.get_svg_image(text_as_path=False),'utf-8')
        p.get_pixmap(matrix=fitz.Matrix(300/72,300/72)).save(FIG/f'{name}.png')
    LAYOUT[name]=inspect_pdf(name)


def compact_saved_figures():
    # y1 cutoffs are in the immutable V16 PDF's page coordinates. All panels,
    # units and legends are retained; the removed domain notes are caption data.
    crops={
        'fig15_service_and_stocks':(216,[
            '六组比较中五组协调与分散缺供持平，折中2060少缺供29.635TWh。',
            '库存分母为各策略、各次事件前库存；首次后在h1200边界前、再次后在h2160边界前评估。95%是对应参照门槛，与连续轨迹的首次事件前固定分母不同。']),
        'figL_annual_operations_choice':(419,[
            '2030四控制两策略主窗ENS均为数值零（|ENS|≤10^-8GWh）；主窗h48—359，完整360h含预运行。h66为共同时间，各案需求依年度状态。',
            '库存取h168—191与h336—359逐燃料逐小时最低比，分母为各方案各事件前一小时库存，95%为库存门。图内005/006/007为保供修复、非化石优先及混合方案的原索引。']),
        'figN_replenishment_cash_and_readiness':(414,[
            '库存取h336—359逐燃料逐小时比率最低值；空心原方案→实心合同。A对自身事件前库存，B对自身期初库存，C对原无事件资源目标。',
            '四对非库存动作、ENS、峰值与CO2相同；首次联合确认均119/263h。合同在预定日历的事件结束12h后供货，总进口最高为年能力折小时的150%，需额外通道。',
            '新增采购现金1666.523325亿元=基价1111.015550+溢价555.507775亿元；基价全转为新增末库存资产，不是净资源耗用费。']),
        'figO_common_terminal_storage':(396,[
            'P1/P1.5仅储能功率不同；容量、初始50%电量、24h信息窗相同。ENS与峰值取h48—359，费用和CO2取完整360h；储能净取电为充电减放电。',
            '物理资源费含0.32百万元/GWh的净末电量估值，不含缺供罚金或碳价。共同满电约束进入h336—359，末燃料库存、热机出力仍可不同。']),
        'figR_energy_conservation_floor':(393,[
            '同一原相位48h事件；[a,b)为全局小时a至b-1，每窗允许从满电开始，最多交付ΣηdE，窗末为日末债务截止点。',
            '截止债务归零、期初非负，移峰不能净移出窗口；四窗不重叠，可加总下界。功率+50%保持E不变；4h/2.3h按基准电池功率确定E。',
            '原相位四窗合计5169.910GWh；电池由4h减至2.3h，每窗少交付99.163GWh，四窗下界增396.654GWh；范围限既定容量和日末偿还规则，完整14路径/56窗另表。']),
        'figS_joint_dispatch_constraints':(430,[
            'a—c较同条件滚动参照，主窗h48—359；固定实际前48h，以后312h真实输入一次优化；各方案保留原最小ENS+10^-6GWh界，碳和峰值上限均来自滚动参照。',
            '物理资源费含储能净领用估值与库存持有费，不含碳价和缺供罚金；≤10元不作收益。d固定各自参考最优解的3燃料库存、2SOC、2热机末出力，不是统一滚动末态。',
            '八条件、四约束费用/排放/峰值原值全部保留；同末态16条件全有保峰等费见证。表内“原参考”表示直接检验既有排程而未新增求解；原32次算法尝试及状态完整保留。'])}
    for name,(cut,notes)in crops.items():
        source=bind((OLD/f'{name}.pdf').relative_to(ROOT));old=fitz.open(source);page=old[0]
        d=fitz.open();p=d.new_page(width=page.rect.width,height=cut)
        if name=='figS_joint_dispatch_constraints':
            # Preserve the three full numeric panels and redraw the complete
            # 16-domain witness table from its frozen machine-readable values.
            p.show_pdf_page(fitz.Rect(0,0,page.rect.width,288),old,0,clip=fitz.Rect(0,0,page.rect.width,288))
            witnesses=read('outputs/v16/joint_dispatch_figures/all_16_endpoint_witnesses.csv')
            keep_table('joint_all_16_endpoint_witnesses.csv',witnesses)
            keep_table('joint_all_32_primary_comparisons.csv',read('outputs/v16/joint_dispatch_figures/all_32_primary_comparisons.csv'))
            assert len(witnesses)==16 and witnesses.condition_id.nunique()==8
            p.insert_font(fontname='cjk',fontfile=str(FONT))
            pdf_text(p,(7,305),'d  同末态保峰见证：全部8条件 × 2种参考最优域',9.4)
            for x,text in [(8,'相位·功率·末SOC'),(168,'无碳约束参考末态'),(318,'碳约束参考末态')]:pdf_text(p,(x,323),text)
            for i,(condition,part)in enumerate(witnesses.groupby('condition_id',sort=False)):
                bits=condition.split('_');label=f'{int(bits[0][5:])}h · {int(bits[1][1:])/100:g}× · '+('自由'if bits[2]=='free'else'满电')
                y=338+i*12
                if i%2==0:p.draw_rect(fitz.Rect(6,y-10,page.rect.width-7,y+3),color=None,fill=(.94,.96,.97),overlay=False)
                pdf_text(p,(8,y),label)
                for x,family in [(168,'unconstrained_resource_minimum'),(318,'carbon_capped_resource_minimum')]:
                    value=part.loc[part.reference_family.eq(family),'witness_source'].item()
                    pdf_text(p,(x,y),'已存原参考'if value=='原参考（0 LP）'else value)
            DATA_CHECKS['joint_table']={'complete_conditions':8,'reference_families':2,'witness_rows':16,'data_changed':False,'label_translation':'原参考（0 LP） to 已存原参考'}
        else:
            p.show_pdf_page(p.rect,old,0,clip=fitz.Rect(0,0,page.rect.width,cut))
        d.subset_fonts();d.save(FIG/f'{name}.pdf',deflate=True,garbage=4);d.close();old.close()
        register(name,notes,cut*25.4/72);finish_wrapped(name)
        LAYOUT[name].update(source_pdf=source.relative_to(ROOT).as_posix(),source_sha256=sha(source),retained_original_pdf_clip=[0,0,160/25.4*72,cut],
            content_operation='Lossless vector crop of footer; S preserves all three numeric panels and redraws complete 16-domain table from original CSV')
    # The two figures below have no multi-line bottom audit footer: preserve
    # panels, legend and useful top domain cue instead of cutting actual data.
    for name in ['figC2_investment_translation','figK_fixed_control_safety_envelope']:
        source=bind((OLD/f'{name}.pdf').relative_to(ROOT));shutil.copyfile(source,FIG/f'{name}.pdf')
        with fitz.open(source)as d:height=d[0].rect.height*25.4/72
        register(name,['原图实际已无多行底部审计附注；为保留全部面板、数值、单位和完整图例，V17直接保留原生矢量图。'],height)
        finish_wrapped(name);LAYOUT[name].update(source_pdf=source.relative_to(ROOT).as_posix(),byte_identical_pdf=True)


def build():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True);style()
    bind(Path(__file__).relative_to(ROOT))
    investment();shocks();deadline();events();network();native_exports();compact_saved_figures()
    assert len(INDEX)==15
    captions={
        'figB_investment_transmission':'投资收益响应的传导链。a为完整反馈的标准化收益；b—d为完整反馈减关闭收益响应的订单、全部在建及当年投产差，单位分别为GW/年、GW、GW/年。浅色区为完整反馈电价达到1200元/MWh的2028—2047年；两臂价格、四技术全部原值随表保留。',
        'figD_continuous_shocks':'连续90天的运行协调与资源准备。a—b为日均功率，c—d放大55—57天实际小时缺供与SOC；浅色为两次冲击。e按三燃料各自h119库存统一归一，实线为继承供应、虚线为付费补库；95%仅为首次事件前库存参照，不表示每次事件前归一的连续24小时联合恢复门。',
        'figP_deadline_investment_mechanisms':'建设时点机制的完整开发域比较。a为按油工期分组的可靠数，分母每组16；全48组可靠数为20/20/20/19。b仅放大P2相对P0油缺口改变的6组，空心0至实心为百分点差，其余42组该指标数值相同。c保留全部48组资本—净排变化，d显示第21号实际设计及P3相对P0失效；同设计预算相同，资本按4%折现，净排取2060年度。',
        'figM_event_advantage_domain':'32格事件条件下的完整优势域。a—b为原功率下可达全电源余量局部减协调；c—d为协调原功率减1.5倍功率，能量容量不变。虚线框仅标5个累计缺供改善格，27格为有效数值零，不能统一解释为已达夜间守恒下界。主窗h48—359；各面板单位内绝对差≤10^-8显示0，原值128格完整保留，费用含净库存与SOC领用估值。',
        'native_closed_loop':'已实现小时系统的中文连接示意。共同24小时前瞻只执行首小时18维动作，7维当期外部输入驱动原生守恒网络，8维实际状态反馈到下一次优化；年度状态单向传入。此图不是软件截图，箭头宽度无量化含义。',
        'native_v9_resource_conservation':'实际Simulink工程的全景与守恒局部。a为年度217维状态模型的真实顶层封装，主体与12类资产15槽队列在决策函数内实现；b—c为煤库存与储能的原生矢量导出，分别按G/(8141η) Mtce/小时和ηc·充电−放电/ηd更新。中文说明位于原图外，Unit Delay保存上一小时状态。',
        'native_anymath_v10_live':'同一h66的AnyMath真实全景与局部放大。参数控制、能流、库存/SOC、六视图和恢复表来自同一原图；b显示h66/359及已执行67次优化与原生反馈，c显示实际交付2938.0GW和缺供494.1GW。全景未拼接不同时刻，原始高清图随支撑材料提供。',
        'fig15_service_and_stocks':'服务恢复与资源准备的分离。颜色表示供给方式，形状表示冲击次数，配对线连接h1200与h2160观察边界的库存。分母为各策略各次事件前库存，95%为对应恢复参照；与连续轨迹的首次事件前固定分母不同。',
        'figL_annual_operations_choice':'年度组合在小时层的实际交付。a为2025—2060折现资本与2060净排，b为h48—359缺供，c在共同h66核对供需；d为两次边界前24小时三燃料最低库存比，分母为各方案各事件前库存。2030八条路径主窗缺供均数值为零（绝对值≤10^-8GWh），完整360小时含预运行。',
        'figN_replenishment_cash_and_readiness':'补库的现金、资产与资源准备。新增采购现金1666.523亿元中1111.016亿元转为末库存资产、555.508亿元为溢价。空心原方案→实心合同，库存比取h336—359逐燃料逐小时最低值，A/B/C分别对自身事件前、自身期初、原无事件目标归一。四对非库存动作、缺供、峰值和排放相同；合同在事件结束12小时后按预定日历到货，需支持年进口能力折小时150%的额外通道。',
        'figO_common_terminal_storage':'共同满电终点下的实际补电、收益与计价。ENS和峰值取h48—359，费用及排放取完整360小时；共同满电约束进入h336—359。两功率仅P不同，能量容量、初始50%电量和24小时前瞻相同；费用含0.32百万元/GWh末电量净领用估值，不含缺供罚金和碳价，末燃料与热机出力仍可不同。',
        'figR_energy_conservation_floor':'原相位四段夜间的能量守恒下界。四个不重叠7小时窗均结束于日末偿债点，每窗允许满电开始，最大储能交付ΣηdE；移峰不能净移出窗口。下界合计5169.910GWh，功率增加50%而E不变时下界不变；电池由4小时减至2.3小时使四窗交付减少396.654GWh。结论限该原相位、既定容量与日末偿还规则，完整14路径56窗另表。',
        'figS_joint_dispatch_constraints':'同一缺供边界上的四类最优排程。a—c按全部八条件相对对应滚动路径比较费用、排放、峰值，主窗h48—359；实际前48小时固定，后312小时一次优化。ENS容差10^-6GWh，物理费含净SOC领用与持有费，不含碳价和缺供罚金。d保留全部16个同末态保峰见证，固定各自最优解的三燃料、两SOC与两热机末出力；“已存原参考”表示直接验证既有排程，≤10元不作收益。',
        'figC2_investment_translation':'资金投向、供给结构与冲击暴露的组合比较。a按6/10/20年分段汇总折现资本；b为折中减低资本的资金差，五组合计13.663万亿元；c为2060绝对发电量；d分列同年储能功率与连续事件h120—2159缺供。资金技术分解为完整路径的描述性分解。',
        'figK_fixed_control_safety_envelope':'固定控制的进口—工期安全域。格点为实际测试组合，绿色满足2030—2060逐年五服务缺供≤1%，橙色为最大缺失率；星号是单列名义锚点。下图仅连接每种工期最小的已测可行进口倍率，含工期1名义点；叉号表示已测试进口值中未找到可行点。'}
    assert set(captions)==set(INDEX)
    for name,caption in captions.items():INDEX[name]['full_caption']=caption
    drift=[name for name,value in INPUTS.items() if sha(ROOT/name)!=value]
    assert not drift,drift
    (OUT/'figure_index.json').write_text(json.dumps(INDEX,ensure_ascii=False,indent=2),'utf-8')
    (OUT/'visuals_audit.json').write_text(json.dumps({
        'status':'PASS_NUMERIC_IDENTITIES_AND_RENDERED_PDFS_PENDING_ACTUAL_VISUAL_AND_INDEPENDENT_REVIEW',
        'completed_utc':datetime.now(timezone.utc).isoformat(), 'actual_LP_calls':0,'new_simulation_calls':0,'new_physical_hours':0,
        'source_input_sha256':INPUTS,'source_drift':drift,'data_checks':DATA_CHECKS,'figure_layouts':LAYOUT,
        'output_sha256':{p.relative_to(ROOT).as_posix():sha(p)for folder in (FIG,OUT)for p in sorted(folder.iterdir())if p.is_file()and p.name not in ('visuals_audit.json','code_review.json')},
        'visual_inspection':{'completed':False,'actual_pdf_renders':'outputs/v17/visuals/*__actual_pdf.png'},
        'scope':'Presentation changes only; frozen V16 and earlier scientific outputs and native original pixels remain unchanged.'},ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({'status':'BUILT_PENDING_VISUAL_REVIEW','figure_count':len(INDEX),'figures':{n:round(v['height_mm'],2)for n,v in INDEX.items()},'actual_LP_calls':0,'simulation_calls':0},ensure_ascii=False))


if __name__=='__main__':build()
