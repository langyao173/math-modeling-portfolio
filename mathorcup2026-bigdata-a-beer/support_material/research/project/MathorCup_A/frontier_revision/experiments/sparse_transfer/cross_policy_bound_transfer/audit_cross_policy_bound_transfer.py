"""Read-only propagation of compatible lower bounds across nested policies.

No optimizer, fit, candidate selection, or mutation of frozen experiment files.
The output is an additional set-inclusion corollary, not a rewritten solver log.
"""
from pathlib import Path
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import sys
import json
import math
import hashlib
import time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPARSE = HERE.parent
UPGRADE = SPARSE.parent.parent
FINAL = SPARSE / "final_summary"
INTEGRITY = SPARSE / "post_aggregation_integrity/verification.json"
FREEZE = UPGRADE / "experiments/transfer_robustness/candidate_freeze.json"
sys.path.insert(0, str(SPARSE))
from control_problem import FixedProblem, PROJECT, AW


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    files = sorted(p for p in FINAL.iterdir() if p.is_file()) + [INTEGRITY, FREEZE]
    return {str(p.relative_to(UPGRADE)): sha(p) for p in files}


def main():
    pool = json.loads((FINAL / "COMPLETE.json").read_text())
    assert all(pool[k] for k in ("complete", "all_perspective", "sdp_finished"))
    integrity = json.loads(INTEGRITY.read_text())
    assert integrity["complete"] and integrity["passed"]
    assert integrity["selection_cases_sha256"] == sha(FINAL / "selection_cases.csv")
    assert integrity["final_summary_sha256"] == sha(FINAL / "COMPLETE.json")
    frozen = json.loads(FREEZE.read_text())
    assert frozen["post_aggregation_integrity_sha256"] == sha(INTEGRITY)
    before = snapshot()
    models_before = {str(p.relative_to(PROJECT)): sha(p) for p in [
        AW / "q1/sensory_ensemble.joblib",
        AW / "q3/transfer_and_augmentation_models.joblib",
        AW / "competition_revision/style_feasibility/frozen_style_classifier.joblib"]}
    strict = FixedProblem("strict", style_epsilon=1e-5)
    relaxed = FixedProblem("exceedance", style_epsilon=1e-5)
    assert strict.features == relaxed.features and strict.p == relaxed.p == 225
    assert set(strict.contexts) == set(relaxed.contexts) and len(strict.contexts) == 11
    common = {}
    for name in ("P", "H", "mu", "a", "w", "b", "z"):
        x, y = np.asarray(getattr(strict, name)), np.asarray(getattr(relaxed, name))
        common[name] = float(np.max(np.abs(x-y)))
        assert common[name] == 0
    assert np.array_equal(strict.sensory0.to_numpy(), relaxed.sensory0.to_numpy())
    assert strict.clf.classes_.tolist() == relaxed.clf.classes_.tolist()
    sel = pd.read_csv(FINAL / "selection_cases.csv")
    selected = sel[np.isclose(sel.rho, .1)].set_index(["policy", "beer"])
    assert len(selected) == 22
    rows, boxes, caps, certificate_sources = [], [], [], {}
    for beer in sorted(strict.contexts):
        s, e = strict.contexts[beer], relaxed.contexts[beer]
        As, hs = strict.linear_constraints(beer, margin=0.)
        Ae, he = relaxed.linear_constraints(beer, margin=0.)
        assert np.array_equal(As, Ae) and np.array_equal(hs, he)
        for name in ("z0", "gradient"):
            assert np.array_equal(s[name], e[name])
        assert np.array_equal(s["x0"], e["x0"], equal_nan=True)
        assert s["mandatory"] == e["mandatory"]
        assert s["F0"] == e["F0"] and s["d0"] == e["d0"] and s["stale0"] == e["stale0"]
        assert np.all(s["lo"] >= e["lo"]) and np.all(s["hi"] <= e["hi"])
        assert set(s["caps"]) == set(e["caps"])
        for j, feature in enumerate(strict.features):
            boxes.append(dict(beer=beer, feature=feature, strict_lower=float(s["lo"][j]),
                relaxed_lower=float(e["lo"][j]), strict_upper=float(s["hi"][j]),
                relaxed_upper=float(e["hi"][j]),
                included=bool(s["lo"][j] >= e["lo"][j] and s["hi"][j] <= e["hi"][j])))
        for j in s["caps"]:
            assert s["caps"][j] <= e["caps"][j]
            caps.append(dict(beer=beer, feature=strict.features[j],
                strict_cap=float(s["caps"][j]), relaxed_cap=float(e["caps"][j]),
                scale="Original supplied field scale; log10 where documented.",
                included=True))
        evidence = {}
        for policy in ("strict", "exceedance"):
            record = selected.loc[(policy, beer)]
            path = SPARSE / record.lower_bound_source
            certificate = json.loads(path.read_text())
            assert int(certificate["beer"]) == beer and certificate["policy"] == policy
            assert certificate["rho"] == .1 and certificate["margin"] == 0
            assert certificate["style_epsilon"] == 1e-5
            raw = float(certificate["dual_bound"])
            assert abs(raw-float(record.raw_solver_lower_bound)) < 1e-12
            integer = max(0, math.ceil(raw-1e-6))
            assert integer == int(record.integer_lower_bound)
            evidence[policy] = dict(raw=raw, integer=integer,
                certificate=str(path.relative_to(SPARSE)), sha256=sha(path))
            certificate_sources[str(path.relative_to(UPGRADE))] = sha(path)
        # A lower bound on the larger feasible set also bounds the subset.
        best_policy = ("exceedance" if evidence["exceedance"]["raw"] > evidence["strict"]["raw"]
                       else "strict")
        best = evidence[best_policy]
        rows.append(dict(beer=beer, evaluation_policy="strict", rho=.1,
            original_strict_lower=evidence["strict"]["integer"],
            original_strict_raw=evidence["strict"]["raw"],
            original_strict_certificate=evidence["strict"]["certificate"],
            original_strict_certificate_sha256=evidence["strict"]["sha256"],
            relaxed_lower=evidence["exceedance"]["integer"],
            relaxed_raw=evidence["exceedance"]["raw"],
            relaxed_certificate=evidence["exceedance"]["certificate"],
            relaxed_certificate_sha256=evidence["exceedance"]["sha256"],
            strongest_propagated_lower=best["integer"],
            strongest_propagated_raw=best["raw"], originating_policy=best_policy,
            originating_certificate=best["certificate"], originating_certificate_sha256=best["sha256"],
            integer_improvement=best["integer"]-evidence["strict"]["integer"],
            strict_subset_relaxed=True, classification_matrix_error=0., classification_rhs_error=0.,
            box_inclusion_violation=float(max(0.,np.max(e["lo"]-s["lo"]),np.max(s["hi"]-e["hi"]))),
            nonlinear_guard="The identical frozen 50-output ensemble and the identical source stale prediction.",
            transfer_reason="F_strict is a subset of F_exceedance at the same rho, class epsilon and linear/nonlinear guards."))
    comparison = pd.DataFrame(rows)
    assert comparison.loc[comparison.integer_improvement.gt(0), "beer"].tolist() == [81]
    assert comparison.integer_improvement.sum() == 1
    pd.DataFrame(boxes).to_csv(HERE / "coordinate_box_inclusion.csv", index=False)
    pd.DataFrame(caps).to_csv(HERE / "raw_defect_cap_inclusion.csv", index=False)
    comparison.to_csv(HERE / "propagated_lower_bounds.csv", index=False)
    after = snapshot()
    models_after = {name: sha(PROJECT / name) for name in models_before}
    assert before == after and models_before == models_after
    result = dict(completed=True, passed=True,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_cases=11, rho=.1, style_epsilon=1e-5, linear_stale_margin=0.,
        coordinate_boxes_checked=len(boxes), raw_defect_caps_checked=len(caps),
        common_array_max_errors=common, integerization_tolerance=1e-6,
        stronger_integer_bound_cases=1, stronger_integer_bound_beers=[81],
        total_integer_bound_gain=1, frozen_files_unchanged=True, frozen_models_unchanged=True,
        source_sha256_before=before, source_sha256_after=after,
        frozen_model_sha256=models_before, originating_certificate_sha256=certificate_sources,
        no_fit_no_optimization_no_candidate_reselection=True,
        mathematical_scope="Complete guarded fixed-model feasible sets. F_strict subset F_exceedance; any valid lower bound for the latter also bounds the former. The original one-policy numerical solver records are left unchanged.",
        numerical_scope="Inherits the original floating-point SCIP certificate scope, not an exact-arithmetic certificate.",
        implementation_sha256=sha(__file__),
        output_sha256={name:sha(HERE/name) for name in ["coordinate_box_inclusion.csv", "raw_defect_cap_inclusion.csv", "propagated_lower_bounds.csv"]})
    (HERE / "independent_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    source81 = comparison.loc[comparison.beer.eq(81)].iloc[0]
    short = ("利用两政策的可行集包含关系，还可得到一条独立加强：Beer 81的严格政策登记下界为10，"
        f"宽松政策的有效原始下界为{source81.relaxed_raw:.9f}（取整为11），后者同样约束严格政策，"
        "故其严格模型下界可加强为11。主表保留已冻结的同政策求解汇总，候选和全部挑战输入不变。")
    (HERE / "manuscript_results.md").write_text(short+"\n")
    lines = ["# 跨政策下界传播：冻结之后的只读集合推论", "", short, "",
        "## 核验对象", "",
        "对11款源啤酒，逐项重建两政策的225坐标盒界及4项原字段缺陷上界。分类矩阵/右端、距离中心和精度矩阵、源坐标、线性守卫、强制缺陷补全与完整冻结感官模型均相同；严格盒界与原缺陷上界均包含于宽松集合。共核验2475个坐标及44个缺陷上界，误差和记录见逐项CSV。", "",
        "对任一完整严格可行的原字段谱，同一谱满足宽松政策的全部条件。因此 $\\mathcal F_{\\rm strict}\\subseteq\\mathcal F_{\\rm exceedance}$，最少编辑值满足 $k^\\star_{\\rm strict}\\ge k^\\star_{\\rm exceedance}$。宽松问题的任何有效下界可传给严格问题；反向传播一般无效。这里保留两种模型中的同一线性stale额外守卫，不能把这条关系扩展到另一个删守卫问题。", "",
        "## 逐源结果", "", "| Beer | 原严格LB | 宽松LB | 可传播的严格LB | 整数增量 |",
        "|---|---:|---:|---:|---:|"]
    for r in comparison.itertuples():
        lines.append(f"| {r.beer} | {r.original_strict_lower} | {r.relaxed_lower} | {r.strongest_propagated_lower} | {r.integer_improvement} |")
    lines += ["", "来源证书路径、原始浮点界、来源政策和SHA256逐行保存在`propagated_lower_bounds.csv`。整数化与主登记表一致，采用ceil(raw−1e−6)。这是继承原数值证书的集合推论，没有启动新的求解、拟合或候选筛选，也没有把原日志的10改写为11。", "",
        "全部final_summary文件、聚合后verification和bootstrap候选freeze在操作前后逐字节SHA256一致，三个冻结模型文件亦不变。只有本目录新增派生审计产物；最终完整性门、候选UID及200次模型挑战的输入身份保持不变。", ""]
    (HERE / "independent_review.md").write_text("\n".join(lines))
    print(json.dumps({k:result[k] for k in ["completed", "passed", "source_cases", "coordinate_boxes_checked", "stronger_integer_bound_beers", "frozen_files_unchanged"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
