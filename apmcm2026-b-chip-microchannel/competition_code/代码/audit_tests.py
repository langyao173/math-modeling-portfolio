# -*- coding: utf-8 -*-
"""Audit checks for the revised reproducibility chain."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pandas as pd

from config import CODE_DIR, BASE_DIR, FIG_DIR, gamma_max, gamma_min


def require(path: Path):
    if not path.exists():
        raise AssertionError(f"missing required file: {path}")
    return path


def docx_text(path: Path) -> str:
    if not path.exists():
        return ""
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    parts = []
    for t in root.iter("{%s}t" % ns["w"]):
        parts.append(t.text or "")
    return "".join(parts)


def docx_xml_text(path: Path) -> str:
    if not path.exists():
        return ""
    chunks = []
    with ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith(".xml"):
                chunks.append(z.read(name).decode("utf-8", errors="ignore"))
    return "\n".join(chunks)


def contains_banned_phrase(text: str, phrase: str, allowed_contexts=None) -> bool:
    if phrase not in text:
        return False
    allowed_contexts = allowed_contexts or []
    allowed_spans = []
    for context in allowed_contexts:
        for match in re.finditer(re.escape(context), text):
            allowed_spans.append((match.start(), match.end()))
    for match in re.finditer(re.escape(phrase), text):
        if not any(start <= match.start() and match.end() <= end for start, end in allowed_spans):
            return True
    return False


def main():
    require(CODE_DIR / "p2_nested_cv_results.json")
    require(CODE_DIR / "p2_nested_loocv_all_models.csv")
    require(CODE_DIR / "p2_cv_protocol_audit.json")
    with open(CODE_DIR / "p2_cv_protocol_audit.json", encoding="utf-8") as f:
        audit = json.load(f)
    assert audit["pre_fit_before_outer_loop"] is False
    assert all(f["train_count"] == 83 and f["test_count"] == 1 for f in audit["outer_folds"])

    require(CODE_DIR / "p5_mc_domain_audit.json")
    with open(CODE_DIR / "p5_mc_domain_audit.json", encoding="utf-8") as f:
        mc = json.load(f)
    assert mc["outside_domain_count"] == 0
    assert mc["gamma_min_sample"] >= gamma_min - 1e-12
    assert mc["gamma_max_sample"] <= gamma_max + 1e-12
    assert mc["distribution_type"] == "conditional_truncated_normal"

    require(CODE_DIR / "p5_sobol_table_for_paper.csv")
    sob = pd.read_csv(CODE_DIR / "p5_sobol_table_for_paper.csv", encoding="utf-8-sig")
    bad = sob[sob["ST_display"] + 1e-12 < sob["S1_mean"]]
    assert bad.empty, "Sobol paper table has unadjusted ST_display < S1_mean rows"

    for name in [
        "p1_error_audit.csv",
        "p1_error_summary.json",
        "p1_correlation_audit.json",
        "p1_property_assumption.json",
        "p1_ablation_identifiability.csv",
        "p3_grid_convergence.csv",
        "p3_local_refinement.csv",
        "p4_robust_tradeoff.csv",
        "p5_boundary_risk_compare.csv",
        "dual_proxy_check.csv",
    ]:
        require(CODE_DIR / name)

    key_figure_stems = [
        "p1_fig1_主效应_数据vs模型",
        "p1_fig2_模型校验parity",
        "p2_fig1_模型横评",
        "p2_fig2_GPR_parity",
        "p3_fig1_Pareto前沿",
        "p3_fig4_双代理互验",
        "p4_fig1_权重三角图",
        "p4_fig2_鲁棒判定",
        "p5_fig1_敏感性总览",
        "p5_fig2_公差MC分布",
        "p5_fig4_工况波动",
    ]
    for stem in key_figure_stems:
        for suffix in [".png", ".pdf", ".svg"]:
            path = require(FIG_DIR / f"{stem}{suffix}")
            if path.stat().st_size < 5000:
                raise AssertionError(f"figure export appears too small: {path}")

    manuscript = BASE_DIR.parent / "Bapmcm26202913.docx"
    text = docx_text(manuscript)
    xml_text = docx_xml_text(manuscript)
    banned = [
        "全部落于 ±3%",
        "全部落于±3%",
        "等价穷举",
        "枚举等价于穷举",
        "全局最优",
        "全局覆盖保证",
        "不存在收敛性问题",
        "主模型：高斯过程",
        "高斯过程回归为主模型",
        "为主代理模型",
        "主代理",
        "GPR 主代理",
        "GPR 主模型",
        "GPR-Matérn 主代理",
        "GPR-Matern 主代理",
        "先经 5 折交叉验证网格搜索整定，再进入统一的 LOOCV",
        "截断于 ±3σ",
        "n=8192",
        "0.35 / 0.34",
        "0.35/0.34",
        "三视角严格同解",
        "三个独立视角收敛",
        "三视角同解",
        "覆盖率主导鲁棒方案与综合折中方案一致",
        "水的物性变化小于 2%",
        "水的物性变化小于2%",
    ]
    allowed_banned_contexts = {
        "全局最优": ["避免将离散网格误表述为连续域严格全局最优"],
    }
    present = []
    for phrase in banned:
        if phrase == "全局最优":
            if contains_banned_phrase(text, phrase, allowed_banned_contexts[phrase]):
                present.append(phrase)
        elif phrase in text or phrase in xml_text:
            present.append(phrase)
    if present:
        raise AssertionError("banned manuscript phrases still present: " + ", ".join(present))

    if os.environ.get("STRICT_MANUSCRIPT_AUDIT") == "1":
        required_doc_phrases = [
            "strict nested LOOCV",
            "p2_cv_protocol_audit.json",
            "RSM-3 作为解析精度代理",
            "GPR-Matérn 作为不确定性代理",
            "p3_grid_convergence.csv",
            "p3_local_refinement.csv",
            "γ∈[3.0,4.5]",
            "outside_domain_count=0",
            "n=32768",
            "p5_sobol_table_for_paper.csv",
            "p5_mc_domain_audit.json",
            "p1_ablation_identifiability.csv",
        ]
        missing = [w for w in required_doc_phrases if w not in text and w not in xml_text]
        if missing:
            raise AssertionError("required revised manuscript evidence missing: " + ", ".join(missing))

    print("audit tests passed")


if __name__ == "__main__":
    main()
