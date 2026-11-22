#!/usr/bin/env python3
"""Reproduce the five screenshot data-encoding claims without changing old files."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter
from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
AWORK = HERE.parents[1]
DATA_CANDIDATES = [AWORK.parent / "data/A", AWORK / "package/MathorCup_A/data/A"]
DATA = Path(os.environ["MATHORCUP_DATA_DIR"]).expanduser() if os.environ.get("MATHORCUP_DATA_DIR") else next((p for p in DATA_CANDIDATES if (p / "1.csv").exists()),AWORK.parent / "data/A")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def norm(text, remove_r_prefix=False):
    text = str(text).casefold()
    if remove_r_prefix:
        text = re.sub(r"^x(?=[.\d])", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def main():
    global DATA
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",type=Path,default=DATA,help="Attachment directory; defaults to MATHORCUP_DATA_DIR, then project data/A.")
    DATA=ap.parse_args().data
    HERE.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(DATA / "1.csv")
    corr = pd.read_csv(DATA / "2.csv", index_col=0)
    thresholds = pd.read_excel(DATA / "8.xlsx", sheet_name="Sensory thresholds")
    calibrated = pd.read_excel(DATA / "8.xlsx", sheet_name="Calibrated chemical dataset")
    fields = list(a.columns[3:])
    assert len(fields) == 231 and len(a) == 250
    assert np.array_equal(a["beer_id"].astype(str), calibrated["beer_id"].astype(str))
    md = pd.read_csv(AWORK / "audit/chemical_metadata.csv").set_index("field")
    common = [x for x in fields if x in calibrated.columns]
    assert len(common) == 70
    tr = thresholds.set_index("field_name_R", drop=False)
    transform_rows, example_rows = [], []
    for field in common:
        stored = pd.to_numeric(a[field], errors="coerce").to_numpy(float)
        raw = pd.to_numeric(calibrated[field], errors="coerce").to_numpy(float)
        paired = np.isfinite(stored) & np.isfinite(raw)
        positive = paired & (raw > 0)
        raw_error = np.abs(stored[paired] - raw[paired])
        log_error = np.abs(stored[positive] - np.log10(raw[positive]))
        ln_error = np.abs(stored[positive] - np.log(raw[positive]))
        raw_matches = np.isclose(stored[paired], raw[paired], rtol=1e-8, atol=1e-8)
        log_matches = np.isclose(stored[positive], np.log10(raw[positive]), rtol=1e-8, atol=1e-8)
        ln_matches = np.isclose(stored[positive], np.log(raw[positive]), rtol=1e-8, atol=1e-8)
        raw_fraction = float(raw_matches.mean()) if len(raw_matches) else np.nan
        log_fraction = float(log_matches.mean()) if len(log_matches) else np.nan
        kind = "log10" if log_fraction > .98 else "raw_identity" if raw_fraction > .8 else "unresolved"
        raw_positive = raw[np.isfinite(raw) & (raw > 0)]
        floor = float(raw_positive.min() / 2)
        floored_log = np.log10(np.where(np.isfinite(raw) & (raw > 0), raw, floor))
        all_stored = np.isfinite(stored)
        threshold_idx = list(thresholds.field_name_R).index(field) if field in tr.index else None
        unit = tr.loc[field, "unit"] if field in tr.index else None
        row = dict(field=field, stored_column_number=fields.index(field) + 4,
                   calibrated_column=get_column_letter(calibrated.columns.get_loc(field) + 1),
                   dictionary_row=threshold_idx + 2 if threshold_idx is not None else None,
                   unit=unit, paired_n=int(paired.sum()), positive_pair_n=int(positive.sum()),
                   raw_match_n=int(raw_matches.sum()), raw_match_fraction=raw_fraction,
                   log10_match_n=int(log_matches.sum()), log10_match_fraction=log_fraction,
                   ln_match_n=int(ln_matches.sum()), raw_median_abs_error=float(np.median(raw_error)),
                   raw_max_abs_error=float(np.max(raw_error)), log10_median_abs_error=float(np.median(log_error)),
                   log10_max_abs_error=float(np.max(log_error)), ln_median_abs_error=float(np.median(ln_error)),
                   ln_max_abs_error=float(np.max(ln_error)), half_minimum_floor=floor,
                   all_row_floored_log10_max_abs_error=float(np.max(np.abs(stored[all_stored] - floored_log[all_stored]))),
                   empirical_encoding=kind)
        transform_rows.append(row)
        eligible = np.flatnonzero(positive & (np.abs(stored - np.log10(np.maximum(raw, 1e-300))) > 1e-4)) if kind == "raw_identity" else np.flatnonzero(positive)
        if kind == "raw_identity":
            eligible = [i for i in eligible if np.isclose(stored[i], raw[i], rtol=1e-8, atol=1e-8)]
        if len(eligible):
            i = int(eligible[0])
            example_rows.append(dict(field=field, encoding=kind, beer_id=str(a.loc[i,"beer_id"]),
                csv_record_number=i+2, csv_column=get_column_letter(a.columns.get_loc(field)+1),
                calibrated_cell=f"Calibrated chemical dataset!{get_column_letter(calibrated.columns.get_loc(field)+1)}{i+2}",
                stored_value=float(stored[i]), calibrated_raw_value=float(raw[i]),
                log10_of_calibrated=float(np.log10(raw[i])), ln_of_calibrated=float(np.log(raw[i])), unit=unit))
    transforms = pd.DataFrame(transform_rows)
    transforms.to_csv(HERE / "paired_70_field_encodings.csv", index=False)
    examples = pd.DataFrame(example_rows)
    examples.to_csv(HERE / "encoding_cell_examples.csv", index=False)

    # The correlation matrix order is independently checked against its entries.
    recalc = a[fields].corr(method="spearman").to_numpy()
    corr_numeric = corr.to_numpy(float)
    assert corr_numeric.shape == (231, 231)
    corr_error = float(np.nanmax(np.abs(recalc-corr_numeric)))
    assert corr_error < 1e-12
    correlation_identity_checks = {
        "matrix_row_labels_equal_column_labels": list(corr.index.astype(str)) == list(corr.columns.astype(str)),
        "correlation_profile_duplicate_pairs_at_1e_minus_12": [],
        "non_diagonal_perfect_absolute_correlations_at_1e_minus_12": [],
        "scope": "Numerical consistency of the provided positional mapping, not a proof of unique simultaneous row/column permutation or chemical identity."
    }
    for i in range(len(fields)):
        for j in range(i):
            if np.allclose(recalc[i],recalc[j],rtol=0,atol=1e-12,equal_nan=True):
                correlation_identity_checks["correlation_profile_duplicate_pairs_at_1e_minus_12"].append([fields[j],fields[i]])
            if np.isfinite(recalc[i,j]) and abs(recalc[i,j]) >= 1-1e-12:
                correlation_identity_checks["non_diagonal_perfect_absolute_correlations_at_1e_minus_12"].append([fields[j],fields[i],float(recalc[i,j])])
    labels = list(corr.columns)
    counts = {}
    names = []
    methods = {"literal": str, "casefold": lambda s: str(s).casefold(),
               "alphanumeric": norm, "alphanumeric_remove_R_prefix": lambda s: norm(s, True)}
    methods["alphanumeric_remove_R_numeric_prefix"] = lambda s: norm(re.sub(r"^[Xx](?=\d)", "", str(s)))
    for key, fn in methods.items():
        target_set = {fn(s) for s in labels}
        counts[key] = sum(fn(s) in target_set for s in fields)
        counts[key + "_same_position"] = sum(fn(s) == fn(t) for s,t in zip(fields,labels))
    for i,(field,label) in enumerate(zip(fields,labels)):
        row = dict(numeric_index=i+1, field_1_csv=field, label_2_csv=label,
                   exact_equal=field==label, casefold_equal=field.casefold()==label.casefold(),
                   alphanumeric_equal=norm(field)==norm(label),
                   alphanumeric_R_prefix_equal=norm(field,True)==norm(label,True))
        if field in md.index:
            row["dictionary_chemical_name"] = md.loc[field,"chemical_name"]
            row["old_audit_matrix2_name"] = md.loc[field,"matrix2_name"]
        names.append(row)
    pd.DataFrame(names).to_csv(HERE / "all_231_name_mapping.csv",index=False)

    unitrows = thresholds.copy()
    unitrows.insert(0,"excel_row",np.arange(2,len(unitrows)+2))
    unitrows.insert(1,"present_in_1_csv",unitrows.field_name_R.isin(fields))
    unitrows.insert(2,"calibrated_in_8_xlsx",unitrows.field_name_R.isin(common))
    unitrows.to_csv(HERE / "dictionary_units_and_thresholds.csv",index=False)
    unit_counts = {"all_261_dictionary_rows": thresholds.unit.fillna("missing").value_counts().to_dict(),
                   "numeric_231_fields": thresholds.loc[thresholds.field_name_R.isin(fields),"unit"].fillna("missing").value_counts().to_dict(),
                   "paired_70_fields": thresholds.loc[thresholds.field_name_R.isin(common),"unit"].fillna("missing").value_counts().to_dict()}
    original_path = AWORK / "audit/external/original_supplement.xlsx"
    original_unit_check = {}
    known_source_unit_caveat = {}
    if original_path.exists():
        xls = pd.ExcelFile(original_path)
        candidate = [s for s in xls.sheet_names if "threshold" in s.casefold()]
        original_unit_check["matching_sheet_names"] = candidate
        for sheet in candidate:
            od = pd.read_excel(xls,sheet_name=sheet)
            original_unit_check[sheet] = {"shape":list(od.shape),"columns":list(od.columns)}
            if "unit" in od.columns:
                original_unit_check[sheet]["unit_counts"] = od.unit.fillna("missing").value_counts().to_dict()
                original_unit_check[sheet]["unit_column_equals_attachment_8"] = od.unit.fillna("<NA>").equals(thresholds.unit.fillna("<NA>"))
        val = pd.read_excel(xls,sheet_name="Validation Concentrations")
        lact_col = next(c for c in val.columns if "lactic" in c.casefold())
        known_source_unit_caveat = {
            "source_header_cell": "Validation Concentrations!I1", "source_header":lact_col,
            "validation_base_blond_cell":"Validation Concentrations!I2", "validation_base_blond_value":float(val.iloc[0][lact_col]),
            "validation_base_NABLAB_cell":"Validation Concentrations!I5", "validation_base_NABLAB_value":float(val.iloc[3][lact_col]),
            "paired_calibrated_base_beers":[],
            "inference":"The validation header says g/l, but base-beer values exactly match the separately identified mg/L field. Treat this as a source unit-label inconsistency; do not multiply the values by 1000 on the strength of that header alone."
        }
        for beer_number in [64,223]:
            i=int(calibrated.index[calibrated.beer==beer_number][0])
            known_source_unit_caveat["paired_calibrated_base_beers"].append({
                "beer":beer_number,"beer_id":str(calibrated.loc[i,"beer_id"]),
                "calibrated_field":"lactic_acid.mg.L.","value":float(calibrated.loc[i,"lactic_acid.mg.L."]),
                "cell":f"Calibrated chemical dataset!{get_column_letter(calibrated.columns.get_loc('lactic_acid.mg.L.')+1)}{i+2}"})

    defect_fields = ["DMS", "X2.3.butanedione", "X4VG", "X.E..2.nonenal"]
    defect_rows = []
    for field in defect_fields:
        row = tr.loc[field]
        excel_row = int(thresholds.index[thresholds.field_name_R==field][0])+2
        mean = pd.to_numeric(row["threshold_mean"], errors="coerce")
        median = pd.to_numeric(row["threshold_median"], errors="coerce")
        defect_rows.append(dict(field=field,chemical_name=row["chemical_name"],unit=row["unit"],
            unit_cell=f"Sensory thresholds!G{excel_row}",threshold_mean=mean,
            mean_cell=f"Sensory thresholds!J{excel_row}",threshold_median=median,
            median_cell=f"Sensory thresholds!K{excel_row}",mean_over_median=mean/median if median>0 else np.nan))
    pd.DataFrame(defect_rows).to_csv(HERE / "defect_threshold_cells.csv",index=False)
    missing = a[fields].isna()
    pd.DataFrame({"field":fields,"missing_n":missing.sum().values}).to_csv(HERE / "missingness_by_field.csv",index=False)
    derived = md.index[md.is_derived.astype(str).str.casefold().isin(["true","1"])].tolist()
    card_formulas = load_workbook(DATA/"3.xlsx",data_only=False)
    card_cached = load_workbook(DATA/"3.xlsx",data_only=True)
    card_summary = pd.read_excel(DATA/"3.xlsx",sheet_name="summary")
    scorecard_checks = {"summary_data_shape":list(card_summary.shape),
        "summary_range":card_formulas["summary"].calculate_dimension(),
        "summary_metadata_columns":["Beer","comments"],"scored_dimensions":50,
        "score_cache_zero_count":int((card_summary.iloc[:,1:-1]==0).sum().sum()),
        "comments_cache_zero_count":int((card_summary.comments==0).sum()),
        "example_summary_formula":card_formulas["summary"]["B2"].value,
        "example_source_formula":card_formulas["beer_1"]["Q7"].value,"cards":[]}
    for sheet in [f"beer_{i}" for i in range(1,13)]:
        formulas=[cell for row in card_formulas[sheet] for cell in row if cell.data_type=="f"]
        input_refs=set(ref.replace("$","") for cell in formulas for ref in re.findall(r"ISBLANK\((\$?[A-Z]+\$?\d+)\)",cell.value))
        scorecard_checks["cards"].append({"sheet":sheet,"formula_count":len(formulas),
            "cached_zero_count":sum(card_cached[sheet][cell.coordinate].value==0 for cell in formulas),
            "distinct_ISBLANK_input_cells":len(input_refs),
            "nonempty_ISBLANK_input_cells":sum(card_formulas[sheet][ref].value is not None for ref in input_refs)})
    (HERE/"scorecard_evidence.json").write_text(json.dumps(scorecard_checks,ensure_ascii=False,indent=2),encoding="utf-8")
    result = dict(data_shape=list(a.shape),metadata_columns=list(a.columns[:3]),numeric_fields=len(fields),
        direct_base_properties=226,derived_fields=derived,complete_cases=int((~missing.any(axis=1)).sum()),
        incomplete_rows=int(missing.any(axis=1).sum()),explicit_missing_cells=int(missing.sum().sum()),
        fields_with_missing=int((missing.sum()>0).sum()),encoding_counts=transforms.empirical_encoding.value_counts().to_dict(),
        scorecard_evidence=scorecard_checks,
        log10_fields=transforms.loc[transforms.empirical_encoding=="log10","field"].tolist(),
        raw_identity_fields=transforms.loc[transforms.empirical_encoding=="raw_identity","field"].tolist(),
        paired_fields=len(common),unpaired_fields=len(fields)-len(common),
        name_match_counts=counts,spearman_matrix_max_abs_error=corr_error,
        positional_mapping_evidence=correlation_identity_checks,
        unit_counts=unit_counts,original_supplement_unit_check=original_unit_check,
        known_source_unit_label_caveat=known_source_unit_caveat,
        microgram_fields=thresholds.loc[thresholds.unit.astype(str).str.casefold().isin(["ug/l","µg/l","μg/l"]),"field_name_R"].tolist(),
        defects=defect_rows,stale_hops_is_chemical_field=any("stale" in f.casefold() for f in fields),
        note="Encoding inference applies to the 70 paired calibration fields; the other 161 fields are not assigned an unsupported universal scale.")
    (HERE/"summary.json").write_text(json.dumps(clean(result),ensure_ascii=False,indent=2),encoding="utf-8")

    ex = examples.set_index("field")
    ex_fields = [f for f in ["pH","ethanol..v.v.","DMS","X2.3.butanedione","linalool","X.E..2.nonenal","Z.nerolidol"] if f in ex.index]
    lines = ["# 截图中五项数据口径的逐项实证核验", "", "本报告由同目录 audit_claims.py 从原始附件重算；不修改已有模型或数据。所有单元格均以附件 8.xlsx 的工作表名和 Excel 坐标定位，CSV 行号含表头。", "",
        "## 1. 并非 231 列全为 log10", "",
        f"附件 1.csv 有 231 个数值字段，其中 70 个可与附件 8.xlsx 的 Calibrated chemical dataset 逐啤酒配对。逐列比较原值、log10 和自然对数后，得到 {result['encoding_counts']}。因此 pH 确实是 log10，不能据此推导全部 231 字段均为 log10。其余 161 个无配对标定字段不因本结果而被擅自指定为原值或对数。", "",
        "| 字段 | 1.csv 记录行/列 | 1.csv 值 | 8.xlsx 标定单元格 | 标定值 | log10(标定值) | 判定 |", "|---|---:|---:|---|---:|---:|---|"]
    for f in ex_fields:
        r=ex.loc[f]
        lines.append(f"| {f} | {r.csv_record_number}/{r.csv_column} | {r.stored_value:.12g} | {r.calibrated_cell} | {r.calibrated_raw_value:.12g} | {r.log10_of_calibrated:.12g} | {r.encoding} |")
    lines += ["", "70 字段完整逐列误差和匹配数见 paired_70_field_encodings.csv；每字段一个可追溯示例见 encoding_cell_examples.csv。原值组中的少量跨表差异单独保留，不将一张表覆盖另一张表。", "", "## 2. 226、231、250、187 各自回答不同问题", "",
        f"原论文报告 226 项测量的化学性质；附件 1.csv 另有 5 个派生汇总字段，形成 231 数值特征，加 3 元数据列共 234 列。附件含 250 个啤酒样本；若对 231 数值字段执行整行 dropna，仅剩 {result['complete_cases']} 个完整案例；其余 {result['incomplete_rows']} 行共有 {result['explicit_missing_cells']} 个显式缺失单元，分布在 {result['fields_with_missing']} 个字段。187 不是原始样本总数，也不是所有模型都必须采用的样本数。", "",
        "派生字段：" + ", ".join(derived) + "。缺失统计见 missingness_by_field.csv。", "", "## 3. 名称匹配必须指定规范化规则", "",
        "附件 1.csv 使用 R 字段名，附件 2.csv 使用人读化学名。以下分别列出集合交集与同位置比较，不能把规范化后匹配称为原始字符串精确匹配。", "", "```json", json.dumps(counts,ensure_ascii=False,indent=2), "```", "",
        f"重新从附件 1.csv 计算 Spearman 相关矩阵，按现有列序与附件 2.csv 比较，最大绝对误差为 {corr_error:.3g}。这为原始顺序下的 231 项位置映射提供了全矩阵数值一致性证据，不必把未通过简单字符串规则的字段视为需要凭经验手工猜名。完整位置映射见 all_231_name_mapping.csv。", "",
        "**证据边界：**顺序和 Spearman 一致足以支持采用附件给定的位置映射，但不能单凭相关矩阵宣称化学身份已经唯一确定。相同或对称相关结构可能造成不可辨识；本审计检查了重复相关谱和非对角完美相关，结果在 summary.json 中，但没有穷举证明所有同步行列置换的唯一性。更不能由数值一致性消除源字典的语义冲突，例如 R 字段 X2.ethylhexylsalicylate 与字典/矩阵标签 2-ethylhexyl acrylate 并非同一化合物，仍须标记冲突而不能据此用于实物加料。", "", "## 4. 微克/毫克单位混用确实存在，但“18 个、其余全为毫克”不符本附件字典", "",
        "附件 8.xlsx / Sensory thresholds / unit 列的完整计数如下。此处只按源表明确标注计数；不把未标单位的峰强度/汇总字段强行赋予质量浓度单位。", "", "```json", json.dumps(unit_counts,ensure_ascii=False,indent=2), "```", "",
        "明确标注 ug/L 的字段为："+", ".join(result["microgram_fields"])+"。原始补充材料工作簿的同名 unit 列另行比对，结果见 summary.json 的 original_supplement_unit_check。", "",
        "质量浓度换算关系 1 mg/L = 1000 ug/L 当然成立；但数据也含 g/L、A.U.、v/v、pH、IBU、EBC、PSI、kcal% 以及未标注单位，不能把全部其他字段都视为 mg/L。模型输入尺度（log10/原值）与物理单位是两个独立属性。", "", "## 5. 老化约束：阈值缺失和均值/中位数必须显式处理", "",
        "| 化合物 | 单位及单元格 | 均值阈值及单元格 | 中位阈值及单元格 | 均值/中位数 |", "|---|---|---|---|---:|"]
    for r in defect_rows:
        mean_text=f"{r['threshold_mean']:.12g}" if np.isfinite(r['threshold_mean']) else "缺失"
        median_text=f"{r['threshold_median']:.12g}" if np.isfinite(r['threshold_median']) else "缺失"
        ratio_text=f"{r['mean_over_median']:.6g}" if np.isfinite(r['mean_over_median']) else "不适用"
        lines.append(f"| {r['field']} | {r['unit']} ({r['unit_cell']}) | {mean_text} ({r['mean_cell']}) | {median_text} ({r['median_cell']}) | {ratio_text} |")
    lines += ["", "trans-2-nonenal 的阈值在本字典为空；stale hops 是感官维度，不对应一项同名的化学浓度阈值。可用源啤酒相对非恶化约束与感官预测代理进行共同验收；不得填造所谓原表阈值。Diacetyl 的均值/中位数约 3.325 倍，阈值策略必须注明采用哪个统计量。源点已经超阈值时，仅满足“不恶化”并不等于已经低于感官阈值。", "", "## 证据文件", "",
        "- 输入：data/A/1.csv、2.csv、8.xlsx；原始来源对照：A_work/audit/external/original_supplement.xlsx。", "- 字段字典的既有提取：A_work/audit/chemical_metadata.csv；本次重新计算得到的完整误差与名称表均在当前目录。", "- 原论文正文：A_work/nature_upgrade/source/paper_fulltext.txt；226 项测量、250 个啤酒、派生字段的论述应与具体表口径同时引用。", ""]
    caveat_paragraph = "**原始来源也有已知单位疑点，不能盲信表头。**原始补充材料 Validation Concentrations!I1 写作 Lactic acid (g/l)，但 Blond 基酒 I2=148、无醇基酒 I5=88，分别恰好等于标定表字段 lactic_acid.mg.L. 的 AD23=148（Beer 64）和 AD33=88（Beer 223）。这个跨表对照支持把验证表的 g/l 标头视为单位标注不一致；不能因此将 148 或 88 再乘以 1000。这里关于五项 ug/L 的结论是源表 unit 列的计数结论，不是声称所有源表物理单位都经过独立实验认证。\n"
    insert_at=lines.index("## 5. 老化约束：阈值缺失和均值/中位数必须显式处理")
    lines[insert_at:insert_at]=[caveat_paragraph,""]
    (HERE/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(clean({k:result[k] for k in ["data_shape","complete_cases","encoding_counts","name_match_counts","microgram_fields","original_supplement_unit_check"]}),ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
