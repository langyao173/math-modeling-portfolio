"""Three V18 visual revisions from the frozen observed/validation/search tables.

Only presentation and summaries of saved values are produced here. No model,
optimizer, or native simulation is imported or executed.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures/V18/revisions"
OUT = ROOT / "outputs/v18/revisions"
WIDTH_MM, MIN_FONT = 160, 10
INK, GRAY, LIGHT = "#26343D", "#82909A", "#E5E9EC"
BLUE, TEAL, ORANGE, PURPLE = "#2474AA", "#16877B", "#CE7D35", "#80639F"
FONT = Path("C:/Windows/Fonts/msyh.ttc")
TARGETS = {
    "electricity_demand_twh": "电力需求", "coal_capacity_gw": "煤电",
    "gas_capacity_gw": "气电", "nuclear_capacity_gw": "核电",
    "hydro_capacity_gw": "水电", "wind_capacity_gw": "风电",
    "solar_capacity_gw": "光伏", "biomass_capacity_gw": "生物质",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({
        "font.family": [font_manager.FontProperties(fname=FONT).get_name(), "DejaVu Sans"],
        "font.size": MIN_FONT, "axes.labelsize": MIN_FONT,
        "axes.titlesize": 11, "xtick.labelsize": MIN_FONT,
        "ytick.labelsize": MIN_FONT, "legend.fontsize": MIN_FONT,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRAY,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "axes.unicode_minus": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
        "grid.color": LIGHT, "grid.linewidth": .5,
        "xtick.major.size": 2.5, "ytick.major.size": 0,
        "legend.frameon": False,
    })


def figure(height_mm):
    return plt.figure(figsize=(WIDTH_MM / 25.4, height_mm / 25.4))


def clean(ax, axis="y"):
    ax.grid(axis=axis)
    ax.set_axisbelow(True)


def inputs(*names):
    return {name: sha(ROOT / name) for name in names}


def save(fig, stem, caption_notes, source_hash, tables, calculations):
    """Save editable plots, complete numerical tables, and their input bindings."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    drawn_ticks, all_ticks = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                all_ticks.update((id(tick.label1), id(tick.label2)))
            if ax.axison:
                for tick in axis._update_ticks():
                    drawn_ticks.update((id(tick.label1), id(tick.label2)))
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()
             and (id(t) not in all_ticks or id(t) in drawn_ticks)]
    outside = []
    for text in texts:
        bb = text.get_window_extent(renderer)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.width + 1 or bb.y1 > fig.bbox.height + 1:
            outside.append(text.get_text())
    assert not outside, (stem, outside)
    assert min(t.get_fontsize() for t in texts) >= MIN_FONT
    output = {}
    for ext in ("pdf", "svg", "png"):
        p = FIG / f"{stem}.{ext}"
        fig.savefig(p, dpi=300)
        output[ext] = p.relative_to(ROOT).as_posix()
    table_paths = {}
    for label, frame in tables.items():
        p = OUT / f"{stem}__{label}.csv"
        frame.to_csv(p, index=False, float_format="%.17g")
        table_paths[p.relative_to(ROOT).as_posix()] = sha(p)
    plt.close(fig)
    result = {
        "stem": stem, "paths": output, "caption_notes": caption_notes,
        "sourcehash": {**source_hash, "src/v18_main_visuals.py": sha(__file__)},
        "width_mm": WIDTH_MM, "height_mm": fig.get_figheight() * 25.4,
        "minimum_font_pt": min(t.get_fontsize() for t in texts),
        "outside_canvas_text": outside, "data_tables_sha256": table_paths,
        "calculations": calculations, "output_sha256": {
            p: sha(ROOT / p) for p in output.values()},
        "actual_LP_calls": 0, "actual_simulation_calls": 0,
    }
    (OUT / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    return result


def fuel_supply():
    source = "data/fuel_supply_trade.csv"
    hashes = inputs(source)
    d = pd.read_csv(ROOT / source)
    fig = figure(119)
    panels = [("coal_balance", "a  煤 / Mt"),
              ("petroleum_balance", "b  石油平衡 / Mt"),
              ("natural_gas_volume", "c  天然气 / bcm")]
    gs = fig.add_gridspec(1, 3, left=.095, right=.978, bottom=.44, top=.845, wspace=.46)
    tables = {}
    for j, (fuel, title) in enumerate(panels):
        q = d.loc[d.fuel_class.eq(fuel) & d.year.ge(2010)].sort_values("year")
        tables[fuel] = q
        ax = fig.add_subplot(gs[0, j])
        for metric, name, color, marker in [("production", "生产", BLUE, "o"),
                                           ("consumption", "消费", TEAL, "D"),
                                           ("imports", "进口", ORANGE, "s")]:
            v = q.dropna(subset=[metric])
            if fuel == "natural_gas_volume" and metric != "production":
                v = v.loc[v.year.eq(2024)] if metric == "consumption" else v.iloc[:0]
            if len(v):
                ax.plot(v.year, v[metric], color=color, marker=marker, ms=3.2, lw=1.25, label=name)
        ax.set(xlim=(2009, 2025), ylim=(0, None), xticks=[2010, 2020, 2024])
        ax.set_title(title, loc="left", fontweight="bold", pad=7)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        clean(ax)
    cp = d.loc[d.fuel_class.eq("raw_coal_output") & d.year.eq(2024), "production"].item()
    ci = d.loc[d.fuel_class.eq("coal_and_lignite_trade") & d.year.eq(2024), "imports"].item()
    oil = d.loc[d.fuel_class.eq("crude_oil") & d.year.eq(2024)].iloc[0]
    gas = d.loc[d.fuel_class.eq("natural_gas_volume") & d.year.eq(2024)].iloc[0]
    rows = pd.DataFrame([
        dict(fuel="煤", denominator_label="生产＋进口", domestic=cp, exposed=ci,
             denominator=cp + ci, exposure_name="进口", unit="Mt"),
        dict(fuel="原油", denominator_label="生产＋进口", domestic=oil.production,
             exposed=oil.imports, denominator=oil.production + oil.imports,
             exposure_name="进口", unit="Mt"),
        dict(fuel="气", denominator_label="表观消费", domestic=gas.production,
             exposed=gas.consumption - gas.production, denominator=gas.consumption,
             exposure_name="表观缺口", unit="bcm"),
    ])
    rows["domestic_percent"] = rows.domestic / rows.denominator * 100
    rows["exposed_percent"] = rows.exposed / rows.denominator * 100
    assert np.allclose(rows.domestic_percent + rows.exposed_percent, 100, atol=1e-12)
    tables["exposure_2024"] = rows
    ax = fig.add_axes([.292, .105, .686, .205])
    y = np.arange(3)
    ax.barh(y, rows.domestic_percent, color=BLUE, height=.54)
    ax.barh(y, rows.exposed_percent, left=rows.domestic_percent, color=ORANGE, height=.54,
            hatch=[None, None, "///"], edgecolor="white", linewidth=.5)
    for i, r in rows.iterrows():
        ax.text(r.domestic_percent / 2, i, f"{r.domestic_percent:.1f}%", va="center", ha="center", color="white")
        ax.text(r.domestic_percent + r.exposed_percent / 2, i, f"{r.exposed_percent:.1f}%", va="center", ha="center", color="white",
                bbox=dict(facecolor=ORANGE, edgecolor="none", pad=.7) if i == 2 else None)
    ax.set(yticks=y, yticklabels=["煤", "原油", "天然气"],
           xlim=(0, 100), ylim=(2.5, -.5), xticks=[0, 25, 50, 75, 100], xlabel="各自供给口径中的构成 / %")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", pad=8)
    fig.text(.095, .344, "d  2024 供给结构", fontweight="bold")
    fig.legend(handles=[Line2D([], [], color=c, marker=m, lw=1.2, ms=3, label=n)
                        for n, c, m in [("生产", BLUE, "o"), ("消费", TEAL, "D"), ("进口", ORANGE, "s")]],
               ncol=3, loc="upper center", bbox_to_anchor=(.54, .995), columnspacing=1.7)
    ax.legend(handles=[Patch(facecolor=BLUE, label="国内生产"),
                       Patch(facecolor=ORANGE, label="进口 / 表观缺口")],
              loc="lower right", bbox_to_anchor=(1.01, 1.14), ncol=2, handlelength=1,
              columnspacing=1.0, borderaxespad=0)
    return save(fig, "fig00c_fuel_balance_exposure", [
        "上排为原始历史观测，煤和石油平衡仅连接选定观测年；天然气消费仅有2024年观测点。煤、油采用Mt，天然气采用bcm，未混合质量与体积。",
        "下排为100%组成条：煤、原油的分母各为国内生产＋进口；气的分母为表观消费，斜纹橙段是（表观消费－生产）/表观消费，不是观测到的总进口率。",
        "煤使用2024原煤产量与煤及褐煤进口，原油使用原油生产与进口，区别于上排石油平衡含成品油的贸易口径。",
    ], hashes, tables, {"exposure_2024": rows.to_dict("records"),
                       "bar_scope": "Each row normalized by its explicitly stated denominator; gas gap is not gross imports."})


def validation():
    source = "outputs/v6/annual/historical_rolling.csv"
    hashes = inputs(source)
    d = pd.read_csv(ROOT / source)
    summary = []
    paired = d.pivot(index=["origin_year", "horizon", "target_year", "target"],
                     columns="method", values="predicted")
    paired["dynamic_minus_recent_prediction"] = paired.dynamic_abm - paired.recent_increment3
    zero_checks = {}
    for h in (1, 3):
        q = (d.loc[d.horizon.eq(h)].groupby(["target", "method"]).ape_percent.mean()
             .unstack().loc[list(TARGETS)].copy())
        q["delta_mape_pp"] = q.dynamic_abm - q.recent_increment3
        q["horizon"] = h
        q["target_label"] = list(TARGETS.values())
        summary.append(q.reset_index())
        zero_targets = q.index[q.delta_mape_pp.abs().le(1e-12)]
        zero_checks[str(h)] = {name: float(paired.loc[(slice(None), h, slice(None), name),
                                                       "dynamic_minus_recent_prediction"].abs().max())
                               for name in zero_targets}
    combined = pd.concat(summary, ignore_index=True)
    assert len(combined) == 16
    assert len(zero_checks["1"]) == 7 and len(zero_checks["3"]) == 2
    assert all(value < 1e-10 for checks in zero_checks.values() for value in checks.values())
    fig = figure(118)
    axes = [fig.add_axes([.15, .155, .29, .665]), fig.add_axes([.60, .155, .29, .665])]
    y = np.arange(8)
    for ax, h, letter in zip(axes, (1, 3), "ab"):
        q = combined.loc[combined.horizon.eq(h)].set_index("target").loc[list(TARGETS)]
        ax.barh(y - .15, q.dynamic_abm, height=.25, color=BLUE, label="动态模块", zorder=3)
        ax.barh(y + .15, q.recent_increment3, height=.25, color="#B9C5CC", label="近期净增", zorder=2)
        ax.set(xlim=(0, 26), ylim=(7.65, -.65), yticks=y,
               yticklabels=list(TARGETS.values()) if h == 1 else [],
               xticks=[0, 10, 20], xlabel="MAPE / %")
        ax.spines["left"].set_visible(False)
        clean(ax, "x")
        ax.set_title(f"{letter}  {h}年距 · {7 if h == 1 else 5}起点", loc="left", fontweight="bold", pad=9)
        for j, delta in enumerate(q.delta_mape_pp):
            equal = abs(delta) <= 1e-12
            shown = "0" if equal else f"{delta:+.3f}"
            ax.text(1.045, j, shown, transform=ax.get_yaxis_transform(), ha="left", va="center",
                    color=GRAY if equal else TEAL if delta < 0 else ORANGE)
        ax.text(1.045, 1.025, "Δ", transform=ax.transAxes, ha="left", color=GRAY)
    fig.legend(handles=[Patch(facecolor=BLUE, label="动态模块"),
                        Patch(facecolor="#B9C5CC", label="近期三年净增基线")],
               ncol=2, loc="upper center", bbox_to_anchor=(.53, .995),
               columnspacing=1.8, handlelength=1.1)
    return save(fig, "fig03_validation", [
        "横条给出动态模块与近期三年净增基线的绝对MAPE，两年距采用同一0—26%尺度；右侧为动态减基线的ΔMAPE，负值表示动态模块误差较低，正值表示基线较优。",
        "一年距7个起点、三年距5个起点，8目标、16项比较完整保留；Δ在图中保留3位小数，全部未舍入MAPE、基线及逐起点预测随数据表保留。",
        "一年距7项容量以及三年距核电、水电，两臂预测逐起点在数值精度内相等（最大绝对差不超过5.685×10^-14）；|ΔMAPE|≤10^-12按数值零显示。验证针对共享预期/资产组件，条件给定已实现GDP。",
    ], hashes, {"mape_summary": combined, "all_original_rolling_values": d,
                "paired_predictions": paired.reset_index()},
        {"difference": "dynamic_abm minus recent_increment3 MAPE", "zero_tolerance_pp": 1e-12,
         "zero_prediction_max_abs_diff": zero_checks, "absolute_mape_axis_percent": [0, 26],
         "comparison_count": 16, "actual_validation_origins": {"1": 7, "3": 5}})


def capital_frontier():
    source = "outputs/v6/planning/candidate_evaluations.csv"
    selected = "outputs/v6/planning/selected_cases.csv"
    hashes = inputs(source, selected)
    d, s = pd.read_csv(ROOT / source), pd.read_csv(ROOT / selected)
    yes = d.annual_reliability_feasible.astype(str).str.lower().eq("true")
    front = d.loc[d.pareto_annual_reliable.astype(str).str.lower().eq("true")].sort_values("discounted_capital_trillion_cny")
    assert (len(d), int(yes.sum()), len(front)) == (2460, 1306, 28)
    role = lambda name: s.loc[s.roles.fillna("").str.split("|").map(lambda roles: name in roles)].iloc[0]
    stats = {}
    for name in ("minimum_capital", "frontier_knee", "minimum_net_emissions"):
        q = role(name)
        stats[name] = {"case_id": q.case_id, "capital_trillion_cny": float(q.discounted_capital_trillion_cny),
                       "net_emissions_gt": float(q.net_emissions_2060_mt / 1000)}
    low, knee, net = [stats[name] for name in ("minimum_capital", "frontier_knee", "minimum_net_emissions")]
    extra = knee["capital_trillion_cny"] - low["capital_trillion_cny"]
    reduction = low["net_emissions_gt"] - knee["net_emissions_gt"]
    floor = float(d.structural_net_floor_mt.iloc[0] / 1000)
    fig = figure(112)
    ax = fig.add_axes([.105, .16, .87, .75])
    ax.scatter(d.loc[yes, "discounted_capital_trillion_cny"], d.loc[yes, "net_emissions_2060_mt"] / 1000,
               s=6, color=GRAY, alpha=.23, lw=0, rasterized=False)
    ax.plot(front.discounted_capital_trillion_cny, front.net_emissions_2060_mt / 1000,
            color=ORANGE, lw=1.8, zorder=4)
    ax.set(xlim=(30, 94), ylim=(3.8, 17.15), xticks=[30, 40, 50, 60, 70, 80, 90],
           yticks=[4, 6, 8, 10, 12, 14, 16],
           xlabel="2025—2060 折现已付工程资本 / 万亿元", ylabel="2060 净CO₂ / Gt·年⁻¹")
    clean(ax)
    # Orthogonal components describe the measured difference between the two
    # saved candidates; they do not depict another feasible trajectory.
    x0, y0 = low["capital_trillion_cny"], low["net_emissions_gt"]
    x1, y1 = knee["capital_trillion_cny"], knee["net_emissions_gt"]
    arrow_style = dict(arrowstyle="-|>", color=PURPLE, lw=1.0, linestyle=(0, (3, 2)), mutation_scale=8)
    ax.annotate("", (x0, y1), (x0, y0), arrowprops=arrow_style, zorder=5)
    ax.annotate("", (x1, y1), (x0, y1), arrowprops=arrow_style, zorder=5)
    ax.text(x0 - 1.35, (y0 + y1) / 2, f"−{reduction:.3f} Gt", rotation=90, color=PURPLE,
            va="center", ha="right", bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.text((x0 + x1) / 2, y1 - .72, f"+{extra:.3f} 万亿元", color=PURPLE,
            va="center", ha="center", bbox=dict(facecolor="white", edgecolor="none", pad=1))
    labels = [
        ("minimum_capital", "低资本", BLUE, "o", (37.4, 16.25)),
        ("frontier_knee", "折中", PURPLE, "D", (51.4, 10.55)),
        ("minimum_net_emissions", "低净排", TEAL, "s", (74.0, 6.9)),
    ]
    for name, title, color, marker, loc in labels:
        q = stats[name]
        point = (q["capital_trillion_cny"], q["net_emissions_gt"])
        ax.scatter(*point, s=50, marker=marker, color=color, edgecolor="white", lw=.9, zorder=8)
        ax.annotate(f"{title}\n{point[0]:.3f} 万亿元\n{point[1]:.3f} Gt/年", point,
                    xytext=loc, textcoords="data", color=color, ha="left", va="top",
                    linespacing=1.4, fontweight="bold", zorder=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.92, pad=2),
                    arrowprops=dict(arrowstyle="-", color=color, lw=.7, shrinkA=5, shrinkB=5))
    fig.text(.105, .955, "2460个评价候选  /  1306个年度可靠  /  28个非支配", fontweight="bold")
    return save(fig, "figC_capital_frontier", [
        "灰点为2460个已评价控制中通过年度可靠性的1306个候选，橙线连接28个已评价且可靠的非支配点；直接标出最低观察可靠资本、折中、最低观察净排的资本及2060净排坐标。",
        "紫色虚线箭头仅分解折中相对低资本的坐标差：资本增加13.662884440688643万亿元，2060净排减少7.520345251521137 Gt；折线路径不是新增可行方案或动态转型轨迹。",
        "资本为2025—2060已付工程资本按4%折现至2024年，不是社会总成本。0.17534575974905597 Gt是独立于候选前沿的技术域推导下界；5次固定预算搜索均达到迭代上限且未收敛。",
    ], hashes, {"all_candidates": d, "selected": s, "frontier": front},
        {"candidate_count": len(d), "annual_reliable_count": int(yes.sum()),
         "frontier_count": len(front), "selected": stats,
         "knee_delta_capital": extra, "knee_emissions_reduction_gt": reduction,
         "structural_floor_gt": floor, "global_convergence": False})


def render_all():
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    results = [fuel_supply(), validation(), capital_frontier()]
    assert all(sha(ROOT / name) == value for r in results for name, value in r["sourcehash"].items())
    (OUT / "main_visuals_index.json").write_text(json.dumps({
        "completed_utc": datetime.now(timezone.utc).isoformat(), "figures": results,
        "source_drift": [], "new_scientific_runs": 0,
    }, ensure_ascii=False, indent=2), "utf-8")
    captions = {
        "fig00c_fuel_balance_exposure": "三燃料历史供需与2024年供给结构。a—c保持Mt与bcm原始量纲，煤、油仅连接选定观测年，气消费仅显示2024年点。d的煤、原油比例分别以国内生产加进口为分母，气以表观消费为分母，斜纹段表示消费减生产的表观缺口。上排石油平衡含成品油贸易，下排使用原油生产与进口；气表观缺口不等同总进口。",
        "fig03_validation": "历史滚动验证的绝对误差与相对改进。两列同尺度横条分别为动态模块和近期三年净增基线的MAPE，右列Δ为动态减基线，单位百分点；负值表示动态误差较低。全部8目标、16项比较保留；一年距7个、三年距5个起点。一年距7项容量及三年距核电、水电的两臂逐起点预测在数值精度内相等，|Δ|≤10^-12显示0，图中其余差值保留3位小数。验证针对共享预期/资产组件，条件给定已实现GDP。",
        "figC_capital_frontier": "年度可靠性约束下的资本—净排取舍。2460个评价候选中1306个年度可靠，橙线连接其中28个非支配点，三个代表点直接标注资本与净排坐标。紫色虚线箭头分解折中相对低资本的差值：资本增加13.663万亿元，2060年净排减少7.520 Gt。资本为2025—2060已付工程资本按4%折现至2024年；技术域独立推导下界为0.175 Gt，五次固定预算搜索均未收敛。",
    }
    return {r["stem"]: {**r["paths"], "full_caption": captions[r["stem"]],
                         "caption_notes": r["caption_notes"], "input_sha256": r["sourcehash"]}
            for r in results}


if __name__ == "__main__":
    print(json.dumps(render_all(), ensure_ascii=False, indent=2))
