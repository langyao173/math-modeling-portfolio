"""Replay concentration-subspace certificates and witnesses without fitting.

The restricted set retains the original linear stale guard. Its negative
certificates do not establish impossibility after that guard is removed.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "experiments/sparse_transfer"
OUT = EXP / "concentration_space"
sys.path.insert(0, str(EXP))
from control_problem import FixedProblem, AW, file_sha


def maximum_error(a, b):
    assert a.shape == b.shape
    return float(np.max(np.abs(a-b))) if a.size else 0.


def main():
    marker = json.loads((OUT / "COMPLETE.json").read_text())
    assert marker["complete"] and marker["margin_cases"] == 132
    protocol = json.loads((OUT / "protocol.json").read_text())
    source = AW / "nature_upgrade/data_claim_audit/paired_70_field_encodings.csv"
    assert protocol["evidence_sha256"] == file_sha(source)
    assert protocol["code_sha256"] == file_sha(EXP / "concentration_space_audit.py")
    units = pd.read_csv(source)
    allowed_names = set(units.loc[units.unit.isin(["mg/L", "ug/L", "g/L", "v/v"]), "field"])
    rows, witness_frames, witness_keys = [], [], []
    worst_matrix_error, worst_rhs_error = 0., 0.
    worst_dual_error, worst_distance_error = 0., 0.
    for policy in ["strict", "exceedance"]:
        problem = FixedProblem(policy, style_epsilon=1e-5)
        support = np.array([j for j, name in enumerate(problem.features) if name in allowed_names], dtype=int)
        assert [problem.features[j] for j in support] == protocol["allowed_fields"]
        assert marker["allowed_fields"] == len(support)
        frozen = np.setdiff1d(np.arange(problem.p), support)
        for beer in sorted(problem.contexts):
            context = problem.contexts[beer]
            assert set(context["mandatory"]) <= set(support)
            for margin in protocol["linear_stale_margins"]:
                folder = OUT / policy / f"beer_{beer}_margin_{int(round(margin*1000))}"
                reported = json.loads((folder / "result.json").read_text())
                saved = np.load(folder / "certificate.npz", allow_pickle=False)
                np.testing.assert_array_equal(saved["support"], support)
                # Rebuild the entire restricted feasible set from the frozen
                # control model, rather than trusting the stored certificate.
                A, rhs = problem.linear_constraints(beer, margin=margin)
                lower, upper = context["lo"][support], context["hi"][support]
                G = np.concatenate([A[:, support], np.eye(len(support)), -np.eye(len(support))])
                h = np.concatenate([rhs, upper, -lower])
                P = problem.P[np.ix_(support, support)]
                P = (P+P.T)/2
                g, F0 = context["gradient"][support], context["F0"]
                errors = {"P":maximum_error(P, saved["P"]),
                          "g":maximum_error(g, saved["g"]),
                          "G":maximum_error(G, saved["G"]),
                          "h":maximum_error(h, saved["h"]),
                          "lower":maximum_error(lower, saved["lo"]),
                          "upper":maximum_error(upper, saved["hi"])}
                assert max(errors.values()) < 1e-12
                assert abs(float(saved["F0"][0])-F0) < 1e-12
                assert np.isfinite(lower).all() and np.isfinite(upper).all()
                worst_matrix_error = max(worst_matrix_error, errors["P"], errors["G"])
                worst_rhs_error = max(worst_rhs_error, errors["g"], errors["h"])
                row = dict(policy=policy, beer=beer, linear_margin=margin,
                           certificate_sha256=file_sha(folder / "certificate.npz"),
                           allowed_concentration_fields=len(support),
                           linear_feasible=bool(reported["linear_feasible"]),
                           geometry_impossible=bool(reported["geometry_impossible"]),
                           full_model_accepted=bool(reported["full_model_accepted"]))
                if not reported["linear_feasible"]:
                    nu = saved["farkas_multiplier"]
                    assert nu.shape == h.shape and np.min(nu) >= 0.
                    # Any feasible d must have nu^T G d <= nu^T h. This
                    # exact finite-box lower bound tolerates G^T nu != 0.
                    residual = G.T @ nu
                    box_minimum = np.minimum(residual*lower, residual*upper).sum()
                    separation = float(box_minimum-nu@h)
                    assert separation > 1e-8
                    assert abs(separation-reported["farkas_separation"]) < 1e-10
                    assert row["geometry_impossible"] and not row["full_model_accepted"]
                    eps = 1e-8
                    expanded_box_minimum = np.minimum(residual*(lower-eps), residual*(upper+eps)).sum()
                    tolerant_separation = float(expanded_box_minimum-nu@(h+eps))
                    row.update(certificate_type="box_robust_farkas", separation=separation,
                               check_tolerance_adjusted_separation=tolerant_separation,
                               check_tolerance_excluded=bool(tolerant_separation > 1e-8),
                               multiplier_minimum=float(nu.min()),
                               stationarity_residual=float(np.max(abs(residual))))
                else:
                    nu, d = saved["dual_multiplier"], saved["primal"]
                    assert nu.shape == h.shape and np.min(nu) >= 0.
                    residual = g+G.T@nu
                    solved = np.linalg.solve(P, residual)
                    # Independent dense solve instead of the experiment's
                    # Cholesky helper reconstructs the dual lower bound.
                    dual = float(F0-nu@h-.5*residual@solved)
                    primal = float(F0+g@d+.5*d@P@d)
                    violation = float(np.max(G@d-h))
                    assert violation <= 2e-7 and primal-dual >= -1e-7
                    assert np.linalg.eigvalsh(P)[0] > 0
                    dual_error = abs(dual-reported["distance_half_square_dual"])
                    assert dual_error < 1e-8
                    assert abs(primal-reported["distance_half_square_primal"]) < 1e-9
                    impossibility = dual > .9**2*F0+1e-7
                    assert impossibility == row["geometry_impossible"]
                    upper_reduction = 1-np.sqrt(max(dual, 0.)/F0)
                    distance_error = abs(upper_reduction-reported["best_distance_reduction_upper_bound"])
                    assert distance_error < 1e-9
                    worst_dual_error = max(worst_dual_error, dual_error)
                    worst_distance_error = max(worst_distance_error, distance_error)
                    # The raw-row checker allows 1e-8 on each linear/box
                    # inequality and on rho. Expand the relaxation to include
                    # this tolerance before drawing an impossibility corollary.
                    eps = 1e-8
                    tolerant_dual = dual-eps*nu.sum()
                    tolerant_distance_threshold = (.9+eps)**2*F0
                    tolerant_exclusion_gap = float(tolerant_dual-tolerant_distance_threshold)
                    delta = saved["full_delta"]
                    np.testing.assert_array_equal(delta[frozen], np.zeros(len(frozen)))
                    np.testing.assert_array_equal(delta[support], d)
                    profile = pd.read_csv(folder / "profile.csv", float_precision="round_trip")
                    values = profile[problem.features].to_numpy(float)[0]
                    original = context["x0"]
                    same = (values == original) | (np.isnan(values) & np.isnan(original))
                    assert same[frozen].all(), "A frozen raw coordinate changed"
                    decoded, _ = problem.decode(beer, delta, support)
                    expected = np.array([decoded[field] for field in problem.features])
                    np.testing.assert_allclose(values, expected, rtol=0, atol=0, equal_nan=True)
                    profile["_audit_margin"] = margin
                    witness_frames.append(profile)
                    witness_keys.append((policy, beer, margin, folder))
                    row.update(certificate_type="convex_quadratic_dual", dual=dual, primal=primal,
                               primal_dual_gap=primal-dual, primal_violation=violation,
                               best_distance_reduction_upper_bound=float(upper_reduction),
                               minimum_restricted_eigenvalue=float(np.linalg.eigvalsh(P)[0]),
                               check_tolerance_adjusted_dual=float(tolerant_dual),
                               check_tolerance_adjusted_distance_threshold=float(tolerant_distance_threshold),
                               check_tolerance_adjusted_exclusion_gap=tolerant_exclusion_gap,
                               check_tolerance_excluded=bool(tolerant_exclusion_gap > 1e-7),
                               multiplier_minimum=float(nu.min()),
                               actual_fields_changed=int(np.sum(~same)),
                               frozen_raw_fields_preserved=True)
                rows.append(row)
        # Full sensory evaluation is batched by policy, but still uses the
        # original six-family all-50-target frozen model for every witness.
        selected = [(frame, key) for frame, key in zip(witness_frames, witness_keys) if key[0] == policy]
        if selected:
            profiles = pd.concat([frame for frame, _ in selected], ignore_index=True)
            replay, sensory = problem.check_rows(profiles.to_dict("records"))
            for k, (_, key) in enumerate(selected):
                _, beer, margin, folder = key
                stored = pd.read_csv(folder / "full_check.csv", float_precision="round_trip").iloc[0]
                row = replay.iloc[k]
                for name in replay.select_dtypes(include=[np.number]).columns:
                    assert abs(float(row[name])-float(stored[name])) < 1e-10
                assert bool(row.accepted) == bool(stored.accepted)
                stored_sensory = pd.read_csv(folder / "sensory.csv", float_precision="round_trip")
                targets = list(problem.bundle["targets"])
                np.testing.assert_allclose(sensory.loc[k, targets].to_numpy(float),
                                           stored_sensory.loc[0, targets].to_numpy(float), atol=1e-10, rtol=0)
                record = next(v for v in rows if v["policy"] == policy and v["beer"] == beer and v["linear_margin"] == margin)
                assert bool(row.accepted) == record["full_model_accepted"]
                record.update(full_model_replayed=True,
                              actual_distance_reduction=float(row.target_distance_reduction),
                              full_ensemble_stale_delta=float(row.ensemble_stale_delta))
    checked = pd.DataFrame(rows)
    assert len(checked) == 132 and not checked.duplicated(["policy", "beer", "linear_margin"]).any()
    summary = pd.read_csv(OUT / "summary.csv", float_precision="round_trip")
    assessments = []
    for (policy, beer), group in checked.groupby(["policy", "beer"]):
        zero = group[group.linear_margin.eq(0.)].iloc[0]
        has_witness = bool(group.full_model_accepted.any())
        excluded = bool(zero.check_tolerance_excluded)
        assert excluded == bool(zero.geometry_impossible)
        assert not (has_witness and excluded)
        prior = summary[summary.policy.eq(policy) & summary.beer.eq(beer)].iloc[0]
        assert bool(prior.any_complete_witness) == has_witness
        assert bool(prior.zero_margin_geometry_impossible) == excluded
        if has_witness:
            status = "complete_witness_in_calibrated_subspace"
        elif excluded:
            status = "entire_guarded_subspace_excluded"
        else:
            status = "nonlinear_feasibility_unresolved"
        assessments.append(dict(policy=policy, beer=int(beer), status=status,
                                full_feasible_margin_witnesses=int(group.full_model_accepted.sum()),
                                zero_margin_linear_feasible=bool(zero.linear_feasible),
                                zero_margin_geometry_impossible=excluded))
    assessed = pd.DataFrame(assessments)
    checked.to_csv(HERE / "concentration_space_certificate_checks.csv", index=False)
    assessed.to_csv(HERE / "concentration_space_set_assessment.csv", index=False)
    result = dict(completed=True, no_new_fitting_or_search=True,
                  margin_cases=len(checked), source_policy_cases=len(assessed),
                  allowed_concentration_fields=marker["allowed_fields"],
                  maximum_rebuilt_matrix_error=worst_matrix_error,
                  maximum_rebuilt_rhs_error=worst_rhs_error,
                  maximum_independent_dual_error=worst_dual_error,
                  maximum_distance_upper_bound_error=worst_distance_error,
                  counts_by_policy={policy:group.status.value_counts().to_dict() for policy, group in assessed.groupby("policy")},
                  source_sha256=file_sha(source), experiment_code_sha256=file_sha(EXP / "concentration_space_audit.py"),
                  audit_code_sha256=file_sha(__file__),
                  scope="The coordinate set has verified calibration and concentration units. Negative certificates retain the original linear stale guard; complete witnesses also pass frozen nonlinear f*. Neither establishes physical process reachability. No L0 minimum is claimed.",
                  numerical_scope="Double-precision certificate replay with saved residuals and explicit tolerances; not interval-arithmetic or rational exact proof.")
    (HERE / "concentration_space_independent_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    lines = ["# 有标定浓度子空间：独立证书与可行集合审计", "",
             f"逐份重建132个凸问题，允许字段数为{marker['allowed_fields']}；重新计算所有保存的QP对偶或盒界残差容忍Farkas分离式，并对所有解码见证重放完整50目标感官模型。没有重新训练或搜索。", "",
             "| 政策 | 完整见证 | 整个含守卫子空间被排除 | 非线性可行性未决 |", "|---|---:|---:|---:|"]
    for policy in ["strict", "exceedance"]:
        count = result["counts_by_policy"][policy]
        lines.append(f"| {policy} | {count.get('complete_witness_in_calibrated_subspace',0)} | {count.get('entire_guarded_subspace_excluded',0)} | {count.get('nonlinear_feasibility_unresolved',0)} |")
    lines += ["", f"独立线性系统重建最大误差{worst_matrix_error:.3g}，右端/梯度最大误差{worst_rhs_error:.3g}，用独立稠密求解重算对偶的最大误差{worst_dual_error:.3g}。", "",
              "## 能得出什么", "",
              "零余量问题包含所有更大非负线性余量的可行域，因此其合法几何下界或线性不可行分离可排除整个含守卫浓度子空间。通过原字段冻结检查和完整f★的候选给出存在性见证；多个余量返回点都未通过f★、但几何证书不足以排除时，结论仍是未决。", "",
              "这里固定的是225个直接字段中的所有非允许坐标；5项派生量按原规则重建，因此不能将“其他直接字段冻结”写成“全部其他附件字段不变”。校准与单位已知使变更具有物理量纲，仍未证明真实工艺能独立实现这些浓度变化。", "",
              "进一步统一原字段检查器的ε=10⁻⁸容差：把Gd≤h放宽为Gd≤h+ε1，下界降为ℓ−ε‖ν‖₁；把ρ=.1要求放宽为ρ≥.1−ε，对应距离阈值(.9+ε)²F₀。四个零余量排除证书在这一更大集合上仍保持正间隙，详细值见证书检查CSV。因任何C外原字段均未改的完整方案满足d_C外=0，故Beer96、124的任何完整含守卫10%改善方案必至少改一个C外直接字段。C外还包含已知量纲的理化指数，不能把这一结论改写成“必须改变未标定化合物”。", "",
              "这些证书对应附加线性陈旧味守卫的决策模型，不能用于排除删除该守卫后的更大集合。该实验不最小化编辑数，不能与主基数最优区间混称。所有计算证书为显式容差下的双精度复算。"]
    (HERE / "concentration_space_independent_review.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
