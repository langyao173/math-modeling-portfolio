"""Build analysis tables from preserved, cited source snapshots (no network).
AI-assisted: OpenAI Codex GPT-6, 2026-10-16; model release date unverified.
"""

from pathlib import Path
from collections import Counter
import csv
import html
import json
import re


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return float(value) if value not in (None, "") else None


def build_annual():
    nbs = {int(row["year"]): row for row in read_csv(RAW / "nbs_energy_2010_2024_transcribed.csv")}
    carbon = {int(row["year"]): row for row in read_csv(RAW / "owid_co2_china_2010_2025.csv")}
    energy = {int(row["year"]): row for row in read_csv(RAW / "owid_energy_china_2010_2025.csv")}
    wb = json.loads((RAW / "worldbank_gdp_china.json").read_text(encoding="utf-8"))
    gdp = {int(row["date"]): row["value"] for row in wb[1]}
    benchmarks = {(int(row["year"]), row["indicator"]): float(row["value"]) for row in read_csv(RAW / "nbs_nea_benchmark_2024_2025.csv")}
    annual = []
    for year in range(2010, 2026):
        source = nbs.get(year)
        co2 = carbon.get(year, {})
        coal, oil, gas, flaring = [number(co2.get(f"{fuel}_co2")) for fuel in ("coal", "oil", "gas", "flaring")]
        components = (coal, oil, gas, flaring)
        fuel_co2 = round(sum(components), 6) if all(value is not None for value in components) else None
        shares = [round(float(source[key]) / 100, 3) for key in ("coal_pct", "oil_pct", "gas_pct", "primary_electricity_other_pct")] if source else [0.514, None, None, None]
        if source:
            assert abs(sum(shares) - 1) <= 0.002, f"NBS rounding/entry error: {year}"
        value = gdp.get(year)
        annual.append({
            "year": year,
            "energy_total_mtce": float(source["energy_total_10000tce"]) / 100 if source else 6170.0,
            "coal_share": shares[0], "oil_share": shares[1], "gas_share": shares[2],
            "nonfossil_share": shares[3], "nbs_primary_electricity_other_share": shares[3],
            "clean_energy_share": 0.304 if year == 2025 else (0.286 if year == 2024 else None),
            "energy_co2_mt": fuel_co2,
            "coal_co2_mt": coal, "oil_co2_mt": oil, "gas_co2_mt": gas, "flaring_co2_mt": flaring,
            "cement_co2_mt": number(co2.get("cement_co2")),
            "other_industry_co2_mt": number(co2.get("other_industry_co2")),
            "fossil_industry_co2_mt": number(co2.get("co2")),
            "gdp_real_usd2015": value,
            "gdp_real_index": round(100 * value / gdp[2010], 6) if value is not None else None,
            "electricity_demand_twh": number(energy.get(year, {}).get("electricity_demand_twh")),
            "electricity_generation_twh": number(energy.get(year, {}).get("electricity_generation_twh")),
            "nbs_electricity_consumption_twh": benchmarks.get((year, "electricity_consumption_twh")),
            "nbs_energy_source_id": "nbs_yearbook_2025_9_2" if source else "nbs_communique_2025",
            "energy_status": "yearbook_2025_vintage" if source else "preliminary_partial_do_not_use_for_joint_calibration",
            "carbon_source_id": "owid_gcb_2025" if co2 else "",
            "energy_co2_scope": "coal+oil+gas+flaring; excludes cement and other industry; fuel-related proxy, not an energy-sector inventory",
            "gdp_source_id": "worldbank_NY.GDP.MKTP.KD",
            "electricity_source_id": "owid_energy_2026_ember_ei",
        })
    write_csv(DATA / "annual_china.csv", annual)
    return annual


def build_power():
    capacity = {int(row["year"]): row for row in read_csv(RAW / "nbs_capacity_2010_2024_transcribed.csv")}
    generation = {(int(row["year"]), row["technology"]): row for row in read_csv(RAW / "nbs_generation_transcribed.csv")}
    for row in read_csv(RAW / "nbs_national_products_2010_2024_transcribed.csv"):
        for technology, column in (("total", "generation_100million_kwh"), ("hydro", "hydro_generation_100million_kwh")):
            generation[int(row["year"]), technology] = {"generation_100million_kwh": row[column], "source_id": "nbs_yearbook_2025_9_19"}
    benchmark = {(int(row["year"]), row["indicator"]): float(row["value"]) for row in read_csv(RAW / "nbs_nea_benchmark_2024_2025.csv")}
    power = []
    for year in range(2010, 2026):
        for technology in ("total", "thermal", "hydro", "nuclear", "wind", "solar", "other"):
            cap = float(capacity[year][f"{technology}_10000kw"]) / 100 if year in capacity else benchmark.get((year, f"{technology}_capacity_gw"))
            gen = generation.get((year, technology))
            power.append({
                "year": year, "technology": technology, "capacity_gw": cap,
                "generation_twh": float(gen["generation_100million_kwh"]) / 10 if gen else None,
                "capacity_source_id": "nbs_yearbook_2025_9_15" if year in capacity else "nbs_communique_2025",
                "generation_source_id": gen["source_id"] if gen else "",
                "status": "yearbook_2025_vintage" if year in capacity else "preliminary_2025_supplement",
                "scope": "NBS/CEC; year-end capacity; thermal includes fossil/bioenergy/waste; hydro includes pumped storage",
            })
    write_csv(DATA / "power_china.csv", power)
    ember = []
    for row in read_csv(RAW / "ember_china_latest_2010_2025.csv"):
        if row["Is aggregated source"] == "True" and row["Electricity source"] != "Total generation":
            continue
        if row["Electricity source"] in ("Demand", "Net imports"):
            continue
        ember.append({
            "year": int(row["Year"]),
            "technology": "total" if row["Electricity source"] == "Total generation" else row["Electricity source"].lower().replace(" ", "_"),
            "capacity_gw": number(row["Capacity (GW)"]), "generation_twh": number(row["Generation (TWh)"]),
            "source_id": "ember_latest_2026_10_16",
            "scope": "Ember fuel taxonomy; capacity primarily GEM/IRENA; hydro excludes pumped storage where possible; see methodology",
        })
    ember.sort(key=lambda row: (row["year"], row["technology"]))
    write_csv(DATA / "ember_power_china.csv", ember)
    comparisons = []
    lookup = {(row["year"], row["technology"]): row for row in power}
    for row in read_csv(RAW / "nbs_nea_benchmark_2024_2025.csv"):
        year, indicator = int(row["year"]), row["indicator"]
        if year != 2024 or row["source_id"] != "nbs_communique_2024" or not indicator.endswith("_capacity_gw"):
            continue
        technology = indicator.removesuffix("_capacity_gw")
        preferred = lookup[year, technology]["capacity_gw"]
        published = float(row["value"])
        comparisons.append({"year": year, "indicator": indicator, "yearbook_value": preferred,
                            "communique_preliminary_value": published, "difference_gw": round(preferred - published, 6),
                            "result": "retained_publication_vintage_difference; not silently reconciled"})
    write_csv(DATA / "publication_vintage_checks.csv", comparisons)
    return power, ember


def build_asset_audit():
    """Preserve supplied rows and flag identity/classification risks; never auto-deduplicate."""
    from openpyxl import load_workbook

    attachments = DATA / "v17/official_attachments"
    records, summaries = [], []
    for path in sorted(attachments.glob("附件*.xlsx")):
        workbook = load_workbook(path, read_only=True, data_only=True)
        for sheet in workbook:
            rows = [(index, tuple(row)) for index, row in enumerate(sheet.values, 1) if index > 1 and row[0] is not None and str(row[0]).strip()]
            names = Counter(str(row[0]).strip() for _, row in rows)
            summaries.append({"attachment": path.name, "sheet": sheet.title, "valid_asset_rows": len(rows),
                              "worksheet_max_rows": sheet.max_row, "raw_capacity_sum_assumed_mw": sum(float(row[1]) for _, row in rows),
                              "repeated_name_groups": sum(count > 1 for count in names.values()),
                              "unit_status": "MW inferred from attachment1 KW=capacity*1000; not explicitly marked elsewhere",
                              "time_scope": "no observation year or time series", "use": "asset initialization only; not national capacity denominator"})
            for index, row in rows:
                flags = []
                if names[str(row[0]).strip()] > 1:
                    flags.append("repeated_name_needs_identity_check")
                if sheet.title == "水电站" and any(token in str(row[0]) for token in ("抽水", "潮汐")):
                    flags.append("mixed_generation_class")
                if sheet.title == "抽水蓄能电站" and str(row[0]).startswith(("张河湾", "潘家口", "荒沟")):
                    flags.append("province_coordinate_requires_verification")
                coordinates = len(row) >= 6
                records.append({"attachment": path.name, "sheet": sheet.title, "excel_row": index, "name": row[0],
                                "capacity_raw": row[1], "capacity_unit_assumption": "MW", "location": row[2],
                                "province_raw": row[3] if coordinates else "", "longitude_raw": row[4] if coordinates else "",
                                "latitude_raw": row[5] if coordinates else "", "flags": ";".join(flags)})
        workbook.close()
    if not records:
        raise FileNotFoundError(f"No supplied attachment assets found at {attachments}")
    write_csv(DATA / "attachment_assets.csv", records)
    write_csv(DATA / "asset_audit.csv", summaries)
    return summaries


def build_2005_benchmark():
    source = read_csv(RAW / "nbs_energy_2005_transcribed.csv")[0]
    carbon = read_csv(RAW / "owid_co2_china_2005.csv")[0]
    gdp_2005 = json.loads((RAW / "worldbank_gdp_china_2005.json").read_text(encoding="utf-8"))[1][0]["value"]
    gdp_2010 = next(row["value"] for row in json.loads((RAW / "worldbank_gdp_china.json").read_text(encoding="utf-8"))[1] if row["date"] == "2010")
    index = 100 * gdp_2005 / gdp_2010
    benchmark = {
        "year": 2005, "role": "isolated policy base-year benchmark, excluded from training panel",
        "energy_total_mtce": float(source["energy_total_10000tce"]) / 100,
        "coal_share": round(float(source["coal_pct"]) / 100, 3), "oil_share": round(float(source["oil_pct"]) / 100, 3),
        "gas_share": round(float(source["gas_pct"]) / 100, 3), "nonfossil_share": round(float(source["primary_electricity_other_pct"]) / 100, 3),
        "gdp_real_usd2015": gdp_2005, "gdp_real_index": index,
        "coal_co2_mt": float(carbon["coal_co2"]), "oil_co2_mt": float(carbon["oil_co2"]),
        "gas_co2_mt": float(carbon["gas_co2"]), "flaring_co2_mt": float(carbon["flaring_co2"]),
        "energy_co2_mt": round(sum(float(carbon[f"{fuel}_co2"]) for fuel in ("coal", "oil", "gas", "flaring")), 6),
        "fossil_industry_co2_mt": float(carbon["co2"]),
        "source_ids": ["nbs_yearbook_2025_9_2", "owid_gcb_2025", "worldbank_NY.GDP.MKTP.KD_2005"],
        "access_date": "2026-10-16",
        "intensity_change_formula": "(year_value/year_gdp_real_index)/(benchmark_value/benchmark_gdp_real_index)-1",
        "carbon_scope": "Fuel-related coal+oil+gas+flaring proxy; not the inventory metric used to assess official carbon-intensity target compliance",
    }
    (DATA / "benchmarks_2005.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")


def build_2025_supplement():
    observations = {row["indicator"]: float(row["value"]) for row in read_csv(RAW / "nbs_nea_benchmark_2024_2025.csv") if row["year"] == "2025"}
    supplement = {"year": 2025, "role": "Additional validation only; separate from 2010-2024 yearbook training vintage",
                  **observations, "energy_co2_mt": None,
                  "source_ids": ["nbs_communique_2025", "nbs_energy_mix_2025_supplement", "nea_electricity_2025"],
                  "scope": "Energy mix supplement published June 2026; GCB observed 2025 fuel-related CO2 not available in preserved snapshot; no carbon-intensity validation for 2025"}
    assert abs(sum(supplement[key] for key in ("coal_share", "oil_share", "gas_share", "nonfossil_share")) - 1) < 1e-9
    (DATA / "benchmarks_2025.json").write_text(json.dumps(supplement, ensure_ascii=False, indent=2), encoding="utf-8")


def build_supply_and_regions(annual):
    consumption = {row["year"]: row for row in annual}
    production = []
    for row in read_csv(RAW / "nbs_production_2010_2024_transcribed.csv"):
        total = float(row["production_total_10000tce"]) / 100
        item = {"year": int(row["year"]), "energy_production_mtce": total,
                "coal_share": float(row["coal_pct"]) / 100, "oil_share": float(row["oil_pct"]) / 100,
                "gas_share": float(row["gas_pct"]) / 100,
                "primary_electricity_other_share": float(row["primary_electricity_other_pct"]) / 100,
                "source_id": "nbs_yearbook_2025_9_1"}
        assert abs(sum(item[key] for key in ("coal_share", "oil_share", "gas_share", "primary_electricity_other_share")) - 1) <= 0.002
        for fuel in ("coal", "oil", "gas"):
            item[f"{fuel}_production_mtce_derived"] = round(total * item[f"{fuel}_share"], 6)
            item[f"{fuel}_consumption_mtce_derived"] = round(consumption[item["year"]]["energy_total_mtce"] * consumption[item["year"]][f"{fuel}_share"], 6)
        item["scope"] = "Same NBS energy-balance vintage; product values derived from rounded shares; production-consumption gap is not observed net imports"
        production.append(item)
    write_csv(DATA / "energy_production_china.csv", production)
    supply = []
    for row in read_csv(RAW / "nbs_selected_fuel_balances_transcribed.csv"):
        supply.append({"year": int(row["year"]), "fuel_class": row["fuel_class"], "unit": "million_tonnes",
                       **{key: float(row[key + "_10000tonnes"]) / 100 for key in ("production", "imports", "exports", "stock_change", "consumption")},
                       "source_id": row["source_id"], "scope": "Energy balance; petroleum production is crude oil but petroleum trade includes products; keep statistical discrepancy"})
    for row in read_csv(RAW / "nbs_national_products_2010_2024_transcribed.csv"):
        year = int(row["year"])
        for fuel, column, factor, unit in (("raw_coal_output", "raw_coal_100million_tonnes", 100, "million_tonnes"),
                                            ("natural_gas_volume", "gas_100million_m3", 0.1, "billion_m3")):
            supply.append({"year": year, "fuel_class": fuel, "unit": unit, "production": float(row[column]) * factor,
                           "imports": None, "exports": None, "stock_change": None,
                           "consumption": 426.05 if year == 2024 and fuel == "natural_gas_volume" else None,
                           "source_id": "nbs_yearbook_2025_9_19" + (";ndrc_gas_consumption_2024" if year == 2024 and fuel == "natural_gas_volume" else ""),
                           "scope": "Raw output, not standard coal; natural gas 2024 consumption is NDRC apparent consumption; no mass/volume mixing"})
    for fuel, production_value, import_value, scope in (
        ("crude_oil", 212.8909, 553.42, "Production yearbook 9-17; crude imports NBS 2024 communique; exports/stocks/consumption unobserved here"),
        ("coal_and_lignite_trade", None, 542.70, "NBS communique customs coal and lignite imports; not a complete raw-coal balance"),
        ("petroleum_products_trade", None, 48.23, "NBS communique customs petroleum-product imports; separate from crude oil"),
        ("natural_gas_import_mass", None, 131.69, "NBS communique customs LNG+pipeline gas import mass; do not divide by domestic bcm output"),
    ):
        supply.append({"year": 2024, "fuel_class": fuel, "unit": "million_tonnes", "production": production_value,
                       "imports": import_value, "exports": None, "stock_change": None, "consumption": None,
                       "source_id": "nbs_communique_2024" + (";nbs_yearbook_2025_9_17" if production_value else ""), "scope": scope})
    write_csv(DATA / "fuel_supply_trade.csv", supply)
    regional = []
    for row in read_csv(RAW / "nbs_regional_products_2024_transcribed.csv"):
        regional.append({"year": 2024, "province": row["province"],
                         "raw_coal_mt": float(row["raw_coal_100million_tonnes"]) * 100 if row["raw_coal_100million_tonnes"] else None,
                         "natural_gas_bcm": float(row["gas_100million_m3"]) / 10 if row["gas_100million_m3"] else None,
                         "generation_twh": float(row["generation_100million_kwh"]) / 10,
                         "hydro_generation_twh": float(row["hydro_generation_100million_kwh"]) / 10 if row["hydro_generation_100million_kwh"] else None,
                         "source_id": "nbs_yearbook_2025_9_19", "scope": "2024 production by province; blank source cells remain missing; not regional consumption or hourly dispatch"})
    assert len(regional) == 31 and len({row["province"] for row in regional}) == 31
    write_csv(DATA / "regional_energy_2024.csv", regional)
    benchmark = {
        "year": 2024, "pumped_hydro_capacity_gw": next(float(row["value"]) for row in read_csv(RAW / "nbs_nea_benchmark_2024_2025.csv") if row["year"] == "2024" and row["indicator"] == "pumped_hydro_capacity_gw"), "source_id": "nea_renewable_2024",
        "crude_oil_import_share_of_output_plus_imports": 553.42 / (212.8909 + 553.42),
        "natural_gas_apparent_consumption_gap_share": (426.05 - 246.451) / 426.05,
        "coal_import_share_of_raw_output_plus_imports": 542.70 / (4781.0 + 542.70),
        "scope": "Security proxies, not published external-dependence statistics. Crude and coal shares omit exports/stocks; gas uses apparent consumption minus production, not measured net imports. Coal/lignite trade and raw-coal output are related but not identical product baskets.",
        "sources": ["nbs_yearbook_2025_9_17", "nbs_yearbook_2025_9_19", "nbs_communique_2024", "ndrc_gas_consumption_2024", "nea_renewable_2024"],
    }
    (DATA / "security_supply_proxies_2024.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")


def build_price_snapshot():
    prices = []
    source = (RAW / "nbs_prices_2024_dec_late.html").read_text(encoding="utf-8")
    keywords = ("液化天然气", "液化石油气", "汽油", "柴油", "无烟煤", "普通混煤", "山西大混", "山西优混", "大同混煤", "焦煤", "焦炭")
    for table_row in re.findall(r"<tr[^>]*>(.*?)</tr>", source, flags=re.S | re.I):
        cells = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", cell))).strip()
                 for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", table_row, flags=re.S | re.I)]
        if len(cells) != 5 or not any(word in cells[0] for word in keywords):
            continue
        prices.append({"period": "2024-12-late", "product": cells[0], "physical_unit": cells[1],
                       "price_cny_per_unit": float(cells[2]), "change_cny_per_unit": float(cells[3]), "change_pct": float(cells[4]),
                       "source_id": "nbs_prices_2024_dec_late",
                       "scope": "National circulation wholesale/distribution price; ten-day snapshot, not annual average, plant delivered cost, or regulated tariff"})
    if len(prices) != 12:
        raise ValueError(f"Unexpected price table extraction: {len(prices)} rows")
    write_csv(DATA / "energy_prices_snapshot.csv", prices)


def build_carrier_supplement():
    conversions = {"10000_tonnes": (0.01, "million_tonnes"),
                   "10000_tonnes_per_year": (0.01, "million_tonnes_per_year"),
                   "10000_kW": (0.01, "GW"), "100million_kWh": (0.1, "TWh")}
    rows = read_csv(RAW / "supplement_carriers_2024_transcribed.csv")
    for row in rows:
        factor, unit = conversions[row["unit"]]
        row["value"] = round(float(row["value"]) * factor, 6)
        row["unit"] = unit
    write_csv(DATA / "supplement_carriers_2024.csv", rows)


def build_quality_report(annual, power, ember, assets):
    assert len(annual) == 16 and len({row["year"] for row in annual}) == 16
    assert len({(row["year"], row["technology"]) for row in power}) == len(power)
    assert len({(row["year"], row["technology"]) for row in ember}) == len(ember)
    quality = {
        "data_access_date": "2026-10-16", "calibration_years": list(range(2010, 2025)), "primary_baseline_year": 2024,
        "annual_rows": len(annual), "nbs_power_rows": len(power), "ember_power_rows": len(ember),
        "annual_missing_by_column": {key: sum(row[key] in (None, "") for row in annual) for key in annual[0]},
        "power_missing_generation_rows": sum(row["generation_twh"] is None for row in power),
        "supplied_asset_count": sum(row["valid_asset_rows"] for row in assets),
        "regional_rounding_check": {},
        "boundaries": ["2025 main-panel snapshot is partial; later mix supplement stays in benchmarks_2025.json and outside joint calibration",
                       "energy_co2_mt is coal+oil+gas+flaring fuel-related proxy, not a sector inventory",
                       "NBS yearbook and Ember power panels must not be silently mixed",
                       "electricity demand from OWID equals gross generation plus net imports, not NEA all-society consumption",
                       "all annual and attachment data are national/static, not observed hourly regional dispatch"],
    }
    regional = read_csv(DATA / "regional_energy_2024.csv")
    for column, national in (("raw_coal_mt", 4781), ("natural_gas_bcm", 246.451), ("generation_twh", 10086.881), ("hydro_generation_twh", 1425.682)):
        values = [float(row[column]) for row in regional if row[column]]
        quality["regional_rounding_check"][column] = {"observed_provinces": len(values), "province_sum": round(sum(values), 6), "national": national, "difference": round(sum(values) - national, 6), "reconciliation": "original rounding/blank cells retained, no rescaling"}
    (DATA / "data_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    rows = build_annual()
    power_rows, ember_rows = build_power()
    asset_rows = build_asset_audit()
    build_2005_benchmark()
    build_2025_supplement()
    build_supply_and_regions(rows)
    build_price_snapshot()
    build_carrier_supplement()
    build_quality_report(rows, power_rows, ember_rows, asset_rows)
    print(f"annual={len(rows)} years; NBS power={len(power_rows)} rows; Ember power={len(ember_rows)} rows; baseline=2024")
