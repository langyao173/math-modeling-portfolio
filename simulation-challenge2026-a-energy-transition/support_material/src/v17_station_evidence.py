"""Read the five official station lists and draw their stated spatial evidence.

Run: python -B -m src.v17_station_evidence
No national calibration, simulation, or optimizer is imported or executed.
AI-assisted implementation: OpenAI Codex, 2026-10-19.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "data/v17/official_attachments"
OUT = ROOT / "outputs/v17/stations"
FIG = ROOT / "figures/V17/stations"
SHEETS = (
    ("附件1.xlsx", "火力发电厂", "火电", "thermal"),
    ("附件2.xlsx", "水电站", "水电", "hydro"),
    ("附件2.xlsx", "抽水蓄能电站", "抽蓄", "pumped"),
    ("附件3.xlsx", "风电场", "风电", "wind"),
    ("附件4.xlsx", "核电站", "核电", "nuclear"),
)
REGIONAL = (
    ("raw_coal_mt", "原煤", "Mt", "raw_coal_100million_tonnes", 100.),
    ("natural_gas_bcm", "天然气", "bcm", "gas_100million_m3", .1),
    ("hydro_generation_twh", "水电", "TWh", "hydro_generation_100million_kwh", .1),
    ("generation_twh", "总发电", "TWh", "generation_100million_kwh", .1),
)
BLUE, TEAL, ORANGE, PURPLE = "#2474AA", "#16877B", "#CE7D35", "#80639F"
INK, GRAY, LIGHT = "#26343D", "#82909A", "#E5E9EC"
COLORS = (GRAY, BLUE, TEAL, ORANGE, PURPLE)
STEM = "fig00e_observed_regions"
CAPTION = (
    "题面电站名录与2024年省级生产集中度。左上按附件原值绘制47条经纬度记录，"
    "点面积与装机成正比，坐标图不含行政边界；左下汇总五张表全部890条名录记录，"
    "火电、水电、核电843条未提供坐标。右侧四组沿用同版年鉴的全国分母和前三省原值。"
    "名录参考年份未注明，同名与重复记录保留，装机合计不代表2024年全国总量。"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_official(folder=OFFICIAL):
    """Keep source rows, duplicates, unstated years, and coordinates as supplied."""
    records, metadata, ancillary = [], [], []
    for filename, sheet, label, key in SHEETS:
        path = folder / filename
        book = openpyxl.load_workbook(path, data_only=True, read_only=True)
        ws = book[sheet]
        rows = list(ws.values)
        header = list(rows[0])
        assert header[:3] == ["name", "installedcapacity", "location"], (filename, sheet, header)
        coords = "经度" in header and "纬度" in header
        start = len(records)
        blank_rows = []
        for rownum, row in enumerate(rows[1:], 2):
            if all(v is None for v in row):
                blank_rows.append(rownum)
                continue
            assert isinstance(row[0], str) and row[0].strip(), (filename, sheet, rownum)
            assert isinstance(row[1], (int, float)) and np.isfinite(row[1]) and row[1] > 0
            assert isinstance(row[2], str) and row[2].strip(), (filename, sheet, rownum)
            province = str(row[3]).strip() if coords else ""
            longitude, latitude = (float(row[4]), float(row[5])) if coords else (None, None)
            if coords:
                assert -180 <= longitude <= 180 and -90 <= latitude <= 90
            kw = row[3] if filename == "附件1.xlsx" else None
            if kw is not None:
                assert kw == row[1] * 1000, (filename, sheet, rownum)
            for col, value in enumerate(row, 1):
                if value is not None and (col > len(header) or header[col - 1] is None):
                    ancillary.append(dict(file=filename, sheet=sheet, row=rownum, column=col, value=value))
            records.append(dict(
                record_id=f"{key}_{rownum:04d}", source_file=filename, source_sheet=sheet,
                source_row=rownum, category=label, category_key=key,
                name_raw=row[0], name=row[0].strip(), installed_capacity_mw=float(row[1]),
                location_raw=row[2], location=row[2].strip(), province_as_supplied=province,
                longitude_as_supplied=longitude, latitude_as_supplied=latitude,
                coordinate_provided=coords, capacity_kw_check=kw, reference_year=None,
                source_cells_json=json.dumps(list(row), ensure_ascii=False),
            ))
        subset = records[start:]
        capacity = np.array([r["installed_capacity_mw"] for r in subset], dtype=float)
        names = Counter(r["name"] for r in subset)
        identities = Counter((r["name"], r["installed_capacity_mw"], r["location"]) for r in subset)
        # Sensitivity totals identify repeated tuples, not verified duplicate facilities.
        first = {identity: identity[1] for identity in identities}
        metadata.append(dict(
            source_file=filename, source_sheet=sheet, category=label, category_key=key,
            worksheet_max_row=ws.max_row, worksheet_max_column=ws.max_column, header=header,
            blank_row_numbers=blank_rows, record_count=len(subset),
            capacity_sum_mw=float(capacity.sum()), capacity_sum_gw=float(capacity.sum() / 1000),
            mean_mw=float(capacity.mean()), std_sample_mw=float(capacity.std(ddof=1)),
            min_mw=float(capacity.min()), q25_mw=float(np.quantile(capacity, .25)),
            median_mw=float(np.median(capacity)), q75_mw=float(np.quantile(capacity, .75)),
            max_mw=float(capacity.max()), maximum_records=[r["record_id"] for r in subset if r["installed_capacity_mw"] == capacity.max()],
            coordinate_count=sum(r["coordinate_provided"] for r in subset),
            no_coordinate_count=sum(not r["coordinate_provided"] for r in subset),
            missing_name_count=0, missing_capacity_count=0, missing_location_count=0,
            repeated_name_group_count=sum(c > 1 for c in names.values()),
            repeated_name_excess_rows=sum(c - 1 for c in names.values()),
            repeated_tuple_group_count=sum(c > 1 for c in identities.values()),
            repeated_tuple_excess_rows=sum(c - 1 for c in identities.values()),
            tuple_unique_count=len(identities), tuple_unique_capacity_sum_mw=float(sum(first.values())),
            reference_year=None, explicit_capacity_unit_in_header="kW cross-check only" if filename == "附件1.xlsx" else None,
        ))
        book.close()
    names = Counter((r["category_key"], r["name"]) for r in records)
    identities = Counter((r["category_key"], r["name"], r["installed_capacity_mw"], r["location"]) for r in records)
    for record in records:
        record["same_name_occurrences"] = names[record["category_key"], record["name"]]
        record["same_name_capacity_location_occurrences"] = identities[
            record["category_key"], record["name"], record["installed_capacity_mw"], record["location"]]
    return pd.DataFrame(records), metadata, ancillary


def regional_values():
    d = pd.read_csv(ROOT / "data/regional_energy_2024.csv", float_precision="round_trip")
    n = pd.read_csv(ROOT / "data/raw/nbs_national_products_2010_2024_transcribed.csv", float_precision="round_trip")
    national = n.loc[n.year.eq(2024)].iloc[0]
    result = []
    for metric, label, unit, column, factor in REGIONAL:
        top = d.dropna(subset=[metric]).nlargest(3, metric)
        denominator = float(national[column] * 100 if factor == 100 else national[column] / 10)
        cr3 = float(top[metric].sum() / denominator * 100)
        for rank, row in enumerate(top.to_dict("records"), 1):
            result.append(dict(metric=metric, label=label, unit=unit, year=2024, rank=rank,
                province=row["province"], value=row[metric], national_denominator=denominator,
                top3_percent=cr3, source_id=row["source_id"], national_column=column,
                scope="2024 provincial production; denominator is same-vintage national total, not sum of nonmissing provinces"))
    return pd.DataFrame(result)


def four_dimensions(regions):
    annual = pd.read_csv(ROOT / "data/annual_china.csv", float_precision="round_trip").set_index("year")
    fuel = pd.read_csv(ROOT / "data/fuel_supply_trade.csv", float_precision="round_trip").query("year==2024").set_index("fuel_class")
    prices = pd.read_csv(ROOT / "data/energy_prices_snapshot.csv", float_precision="round_trip")
    power = pd.read_csv(ROOT / "data/ember_power_china.csv", float_precision="round_trip").query("year==2024").set_index("technology")
    wind_solar_share = (power.loc["wind", "generation_twh"] + power.loc["solar", "generation_twh"]) / power.loc["total", "generation_twh"] * 100
    coal = fuel.loc["coal_and_lignite_trade", "imports"] / (fuel.loc["raw_coal_output", "production"] + fuel.loc["coal_and_lignite_trade", "imports"]) * 100
    oil = fuel.loc["crude_oil", "imports"] / (fuel.loc["crude_oil", "production"] + fuel.loc["crude_oil", "imports"]) * 100
    gas = (fuel.loc["natural_gas_volume", "consumption"] - fuel.loc["natural_gas_volume", "production"]) / fuel.loc["natural_gas_volume", "consumption"] * 100
    price_coal = prices.loc[prices["product"].str.contains("5500"), "price_cny_per_unit"].item()
    price_lng = prices.loc[prices["product"].str.contains("LNG"), "price_cny_per_unit"].item()
    return [
        dict(dimension="供给保障能力", year="2024", indicator="煤进口构成；原油进口构成；气表观缺口", values=[float(coal), float(oil), float(gas)], unit="%",
             source="data/fuel_supply_trade.csv", interpretation="煤、油分母为国内生产＋进口；气分母为表观消费，三者不混作统一进口依赖率。"),
        dict(dimension="清洁低碳水平", year="2010→2024；2024", indicator="煤炭一次能源占比2010/2024；2024风光发电占比", values=[float(annual.loc[2010, "coal_share"] * 100), float(annual.loc[2024, "coal_share"] * 100), float(wind_solar_share)], unit="%",
             source="data/annual_china.csv;data/ember_power_china.csv", interpretation="同版年鉴煤炭份额下降16.0个百分点；风光采用Ember同版发电分母，一次电力及其他口径不直接重命名为非化石。"),
        dict(dimension="经济性", year="2024-12-late", indicator="山西优混5500大卡；LNG流通价格", values=[float(price_coal), float(price_lng)], unit="元/吨",
             source="data/energy_prices_snapshot.csv", interpretation="2024年12月下旬流通价格快照；非年均、到厂成本或单位热值成本排名。"),
        dict(dimension="协调性", year="2024；名录年未注明", indicator="原煤/气/水电产区CR3；附件有坐标记录", values=[float(regions.loc[regions.metric.eq(k), "top3_percent"].iloc[0]) for k in ["raw_coal_mt", "natural_gas_bcm", "hydro_generation_twh"]] + [47], unit="%；条",
             source="data/regional_energy_2024.csv;data/raw/nbs_national_products_2010_2024_transcribed.csv;附件2.xlsx;附件3.xlsx",
             interpretation="产出集中度与名录坐标分布分别描述；缺少同口径省级消费与输电数据，不计算产消错配率。"),
    ]


def setup():
    font = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": [font_manager.FontProperties(fname=font).get_name(), "DejaVu Sans"],
        "font.size": 8.8, "axes.labelsize": 8.8, "axes.titlesize": 9.4,
        "xtick.labelsize": 8.8, "ytick.labelsize": 8.8, "legend.fontsize": 8.8,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRAY,
        "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False,
        "grid.color": LIGHT, "grid.linewidth": .5, "xtick.major.size": 2.5, "ytick.major.size": 0,
        "legend.frameon": False, "savefig.facecolor": "white"})


def draw(records, summary, regions):
    setup()
    fig = plt.figure(figsize=(160 / 25.4, 170 / 25.4))
    a = fig.add_axes((.095, .46, .39, .40))
    a.set_title("a  原表经纬度 · 47条", loc="left", pad=36, fontweight="bold")
    for category, marker, color in [("pumped", "o", TEAL), ("wind", "^", ORANGE)]:
        q = records.loc[records.category_key.eq(category)]
        a.scatter(q.longitude_as_supplied, q.latitude_as_supplied,
                  s=q.installed_capacity_mw.to_numpy() * .025, marker=marker,
                  color=color, edgecolor="white", linewidth=.45, alpha=.85, zorder=3)
    a.set(xlim=(84, 130), ylim=(20, 48), xticks=[90, 110, 130], yticks=[20, 30, 40],
          xlabel="经度（°E）", ylabel="纬度（°N）")
    a.grid(); a.set_axisbelow(True)
    a.legend(handles=[Line2D([], [], linestyle="", marker="o", color=TEAL, label="抽蓄31条", markersize=5),
                      Line2D([], [], linestyle="", marker="^", color=ORANGE, label="风电16条", markersize=5)],
             loc="lower left", bbox_to_anchor=(-.11, 1.04), ncol=2, handletextpad=.3, columnspacing=.8,
             borderaxespad=0)
    for name, label, dx, dy in [("酒泉风电场", "酒泉 7.965 GW", -8, -24),
                               ("丰宁抽水蓄能电站", "丰宁 3.600 GW", -57, 18)]:
        r = records.loc[records.name.eq(name)].iloc[0]
        a.annotate(label, (r.longitude_as_supplied, r.latitude_as_supplied), xytext=(dx, dy),
                   textcoords="offset points", fontsize=8.8,
                   arrowprops={"arrowstyle": "-", "lw": .6, "color": GRAY},
                   bbox={"facecolor": "white", "edgecolor": "none", "pad": .7})
    b = fig.add_axes((.18, .095, .305, .245))
    b.set_title("b  五表名录装机（GW）", loc="left", pad=11, fontweight="bold", x=-.28)
    for i, (s, color) in enumerate(zip(summary, COLORS)):
        v = s["capacity_sum_gw"]
        b.plot([1, v], [i, i], color=LIGHT, lw=2.4, zorder=1)
        b.scatter([v], [i], color=color, s=25, zorder=3)
        b.annotate(f"{v:.3f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.8)
    b.set(xscale="log", xlim=(1, 12000), ylim=(4.65, -.65), yticks=range(5),
          yticklabels=[f'{s["category"]} {s["record_count"]}' for s in summary],
          xticks=[1, 100, 10000], xticklabels=["1", "100", "10⁴"])
    b.minorticks_off(); b.grid(axis="x"); b.set_axisbelow(True)
    fig.text(.095, .035, "计数单位：名录记录；参考年份未注明", fontsize=8.8, color=GRAY)
    for i, (metric, label, unit, _, _) in enumerate(REGIONAL):
        ax = fig.add_axes((.665, .735 - i * .207, .30, .13))
        q = regions.loc[regions.metric.eq(metric)]
        ax.barh(range(3), q.value, height=.44, color=[TEAL, "#95A9B0", "#C9D3D7"])
        for y, v in enumerate(q.value):
            ax.annotate(f"{v:,.1f}", (v, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.8)
        ax.set(yticks=range(3), yticklabels=q.province.tolist(), ylim=(2.55, -.55),
               xlim=(0, q.value.max() * 1.65), xticks=[0, q.value.max()],
               xticklabels=["0", f"{q.value.max():,.0f}"])
        ax.set_title(f'{"cdef"[i]}  {label} · CR3 {q.top3_percent.iloc[0]:.2f}%',
                     loc="left", pad=8, fontweight="bold", fontsize=9.0, x=-.22)
        ax.set_xlabel(unit, labelpad=0)
        ax.grid(axis="x"); ax.set_axisbelow(True)
    fig.text(.095, .976, "题面场站名录 × 同版省级生产统计", fontsize=11.2, fontweight="bold")
    fig.text(.665, .931, "2024年 · 同版全国分母", fontsize=8.8, color=GRAY, ha="center")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [o for o in fig.findobj(Text) if o.get_visible() and o.get_text()]
    assert min(t.get_fontsize() for t in texts) >= 8.8
    clipped = []
    for t in texts:
        e = t.get_window_extent(renderer)
        if e.x0 < -.5 or e.y0 < -.5 or e.x1 > fig.bbox.width + .5 or e.y1 > fig.bbox.height + .5:
            clipped.append(t.get_text())
    assert not clipped, clipped
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{STEM}.{ext}", dpi=240)
    plt.close(fig)
    doc = fitz.open(FIG / f"{STEM}.pdf")
    p = doc[0]
    spans = [s for b in p.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] for s in l["spans"]]
    p.get_pixmap(dpi=200).save(OUT / "figure_pdf_200dpi.png")
    layout = dict(width_mm=p.rect.width / 72 * 25.4, height_mm=p.rect.height / 72 * 25.4,
                  minimum_visible_font_pt=min(t.get_fontsize() for t in texts),
                  minimum_decoded_pdf_font_pt=min(s["size"] for s in spans), text_outside_figure=clipped,
                  actual_visual_inspection="PENDING", map_boundary="No basemap; original-coordinate scatter only")
    assert abs(layout["width_mm"] - 160) < .001 and layout["minimum_decoded_pdf_font_pt"] >= 8.79
    doc.close()
    return layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mark-visual-inspection", action="store_true",
                        help="Record the operator's inspection of the exported one-page PDF rendering")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    records, summary, ancillary = read_official()
    regions = regional_values()
    for name, frame in [("station_records.csv", records), ("summary.csv", pd.DataFrame(summary).drop(columns=["header", "blank_row_numbers", "maximum_records"])),
                        ("duplicate_name_records.csv", records.loc[records.same_name_occurrences.gt(1)]),
                        ("original_coordinate_records.csv", records.loc[records.coordinate_provided]),
                        ("regional_top3_original_values.csv", regions)]:
        frame.to_csv(OUT / name, index=False, float_format="%.17g")
    provinces = records.loc[records.coordinate_provided].groupby(["category", "province_as_supplied"], sort=False).agg(
        record_count=("record_id", "size"), capacity_sum_mw=("installed_capacity_mw", "sum")).reset_index()
    provinces.to_csv(OUT / "as_supplied_province_summary.csv", index=False, float_format="%.17g")
    dimensions = four_dimensions(regions)
    write_json(OUT / "four_dimensions.json", dimensions)
    pd.DataFrame([{**r, "values": json.dumps(r["values"], ensure_ascii=False)} for r in dimensions]).to_csv(OUT / "four_dimensions.csv", index=False)
    write_json(OUT / "summary.json", dict(sheets=summary, total_records=len(records),
        coordinate_records=int(records.coordinate_provided.sum()),
        reference_year="Not stated in workbook data; workbook file timestamps are not reference years",
        capacity_unit="MW working convention: all 633 thermal rows have independent KW=1000*installedcapacity; the other sheets reuse installedcapacity without an explicit unit header; hydro D2=203235500 equals 1000*its listed sum",
        unheaded_ancillary_cells=ancillary,
        duplicate_policy="All supplied rows retained; unique name/capacity/location totals are sensitivity checks, not corrected unique-plant totals",
        geography_policy="Only 47 supplied coordinate pairs plotted, without geocoding or correction; province fields remain as supplied and are not used to infer coordinates or boundaries",
        source_identity="Official problem station lists, not a documented random sample, census, 2024 inventory, or empirical hourly profile"))
    layout = draw(records, summary, regions)
    if args.mark_visual_inspection:
        layout["actual_visual_inspection"] = "PASS_OPERATOR_INSPECTED_200_DPI_PDF_RENDER_NO_CLIPPING_OR_OVERLAP"
    write_json(OUT / "figure_index.json", {STEM: dict(pdf=(FIG / f"{STEM}.pdf").relative_to(ROOT).as_posix(),
        svg=(FIG / f"{STEM}.svg").relative_to(ROOT).as_posix(), png=(FIG / f"{STEM}.png").relative_to(ROOT).as_posix(),
        caption_notes=CAPTION)})
    sources = [ROOT / "src/v17_station_evidence.py", ROOT / "data/regional_energy_2024.csv",
        ROOT / "data/raw/nbs_national_products_2010_2024_transcribed.csv", ROOT / "data/annual_china.csv",
        ROOT / "data/fuel_supply_trade.csv", ROOT / "data/energy_prices_snapshot.csv", ROOT / "data/ember_power_china.csv"]
    audit = dict(status=("PASS_OFFICIAL_ROWS_REGIONAL_ORIGINALS_ACTUAL_PDF_VISUAL" if args.mark_visual_inspection else "PASS_RAW_TABLES_NUMERICS_LAYOUT_PENDING_ACTUAL_VISUAL_REVIEW"), completed_utc=datetime.now(timezone.utc).isoformat(),
        official_files={p.name: {"sha256": sha(p), "bytes": p.stat().st_size, "path": str(p)} for p in sorted(OFFICIAL.glob("附件*.xlsx"))},
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
        record_count=len(records), coordinate_count=int(records.coordinate_provided.sum()), all_original_rows_retained=True,
        plotted_coordinate_ids=records.loc[records.coordinate_provided, "record_id"].tolist(),
        no_synthetic_coordinates=True, no_provincial_consumption_or_flow_claim=True,
        layout=layout, scientific_lp_calls=0, new_annual_paths=0, new_physical_hours=0,
        caption=CAPTION,
        output_sha256={p.relative_to(ROOT).as_posix(): sha(p) for folder in (OUT, FIG) for p in sorted(folder.iterdir()) if p.is_file() and p.name not in {"audit.json", "code_review.json"}})
    write_json(OUT / "audit.json", audit)
    print(json.dumps(dict(status=audit["status"], records=len(records), coordinates=int(records.coordinate_provided.sum()),
                         capacity_sum_mw={s["category"]: s["capacity_sum_mw"] for s in summary}, layout=layout), ensure_ascii=False))


if __name__ == "__main__":
    main()
