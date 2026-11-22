"""Audit saved broader-model certificates without additional optimization.

Checks constraint deletion, source identity, SDP decomposition, solver logs,
registered budgets and every decoded complete witness. The global lower bound
remains SCIP's floating-point branch-and-bound result, not an exact rational
proof and not a bound recomputed by this no-solve audit.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import json
import math
import re
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "experiments/sparse_transfer"
OUT = EXP / "certificate_without_proxy"
sys.path.insert(0, str(EXP))
from control_problem import FixedProblem, PROJECT, AW, file_sha


def number_from_log(text, field):
    matches = re.findall(r"^"+re.escape(field)+r"\s*:\s*([+\-0-9.eE]+)", text, flags=re.MULTILINE)
    assert matches, field
    return float(matches[-1])


def main():
    marker = json.loads((OUT / "COMPLETE.json").read_text())
    assert marker["complete"] and marker["cases"] == 22
    assert marker["seconds_per_case"] == 30. and marker["primary_candidate_pool_unchanged"]
    assert marker["comparison_sha256"] == file_sha(OUT / "comparison.csv")
    comparison = pd.read_csv(OUT / "comparison.csv", float_precision="round_trip")
    assert len(comparison) == 22 and not comparison.duplicated(["policy", "beer"]).any()
    historical = json.loads((HERE / "sparse_independent_review.json").read_text())["snapshot_sha256"]
    chemistry_sha = next(value for name, value in historical.items() if name.endswith("data/A/1.csv"))
    assert file_sha(PROJECT / "data/A/1.csv") == chemistry_sha
    integration = json.loads((EXP / "perspective/INTEGRITY_DONE.json").read_text())
    assert integration["state"] == "complete_independently_verified" and integration["completed_cases"] == 22
    expected_hashes = {
        "wrapper_sha256":file_sha(EXP / "certificate_without_proxy.py"),
        "solver_core_sha256":file_sha(EXP / "perspective_solver.py"),
        "control_sha256":file_sha(EXP / "control_problem.py"),
        "diagonal_split_sha256":file_sha(EXP / "perspective_sdp/diagonal_sdp.npz"),
        "predictor_sha256":file_sha(AW / "q1/sensory_ensemble.joblib"),
        "model_sha256":file_sha(AW / "q3/transfer_and_augmentation_models.joblib"),
        "classifier_sha256":file_sha(AW / "competition_revision/style_feasibility/frozen_style_classifier.joblib"),
    }
    allcases, pool_diagnostics, witness_diagnostics = [], [], []
    first_P = None
    for policy in ["strict", "exceedance"]:
        problem = FixedProblem(policy, style_epsilon=1e-5)
        assert set(comparison.loc[comparison.policy.eq(policy), "beer"]) == set(problem.contexts)
        for beer in sorted(problem.contexts):
            c = problem.contexts[beer]
            folder = OUT / policy / f"beer_{beer}"
            record = json.loads((folder / "DONE.json").read_text())
            protocol = json.loads((folder / "protocol.json").read_text())
            bound = json.loads((folder / "solver_bound.json").read_text())
            parameters = json.loads((folder / "solver_parameters.json").read_text())
            for key, digest in expected_hashes.items():assert protocol[key] == digest
            for name, digest in record["source_files"].items():assert file_sha(folder / name) == digest
            assert protocol["beer"] == beer and protocol["policy"] == policy
            assert protocol["rho"] == .1 and protocol["style_epsilon"] == 1e-5
            assert protocol["omit_only_linear_stale_guard"]
            assert protocol["nonlinear_stale_relaxed_for_lower_bound"]
            assert protocol["complete_upper_checks_keep_both_stale_conditions"]
            assert not protocol["candidate_pool_enters_primary_selection"]
            assert protocol["threads"] == 1 and protocol["seconds_limit"] == 30.
            assert parameters["limits/time"] == bound["time_limit"] == record["seconds_limit"] == 30.
            assert parameters["parallel/maxnthreads"] == 1
            assert parameters["numerics/feastol"] == 1e-8 and parameters["limits/gap"] == 0.
            assert parameters["randomization/randomseedshift"] == 20261024
            assert not parameters["constraints/nonlinear/assumeconvex"]
            assert parameters["nlhdlr/soc/enabled"] and parameters["nlhdlr/perspective/enabled"]
            retained = np.load(folder / "retained_linear_constraints.npz", allow_pickle=False)
            A, rhs = problem.linear_constraints(beer, margin=0.)
            np.testing.assert_array_equal(retained["A"], A[1:])
            np.testing.assert_array_equal(retained["rhs"], rhs[1:])
            np.testing.assert_array_equal(retained["omitted_guard"], A[0])
            np.testing.assert_array_equal(retained["omitted_rhs"], rhs[:1])
            assert retained["A"].shape == (21, 225)
            split = np.load(folder / "diagonal_split.npz", allow_pickle=False)
            common = np.load(EXP / "perspective_sdp/diagonal_sdp.npz", allow_pickle=False)
            for key in ["P", "diagonal", "R", "factor"]:np.testing.assert_array_equal(split[key], common[key])
            P, diag, R, L = split["P"], split["diagonal"], split["R"], split["factor"]
            if first_P is None:first_P = P.copy()
            else:np.testing.assert_array_equal(P, first_P)
            precision_error = float(np.max(abs(P-(problem.P+problem.P.T)/2)))
            split_error = float(np.max(abs(P-R-np.diag(diag))))
            factor_error = float(np.max(abs(R-L.T@L)))
            assert precision_error < 1e-12 and split_error < 1e-10 and factor_error < 1e-10
            assert diag.min() >= 0 and np.linalg.eigvalsh(R)[0] > 0
            log = (folder / "scip.log").read_text()
            dual_log = number_from_log(log, "Dual Bound")
            primal_log = number_from_log(log, "Primal Bound")
            assert abs(dual_log-bound["dual_bound"]) < 1e-10
            assert abs(primal_log-bound["primal_bound"]) < 1e-10
            assert int(number_from_log(log, "Solving Nodes")) == bound["nodes"]
            if bound["status"] == "timelimit":assert "time limit reached" in log
            elif bound["status"] == "optimal":assert "optimal solution found" in log
            else:raise AssertionError("Unexpected solver termination: "+bound["status"])
            assert bound["dual_bound"] <= bound["primal_bound"]+1e-7
            lower = max(len(c["mandatory"]), math.ceil(bound["dual_bound"]-1e-6))
            assert lower == record["broader_integer_lower_bound"]
            warm = np.load(folder / "warm_start.npz", allow_pickle=False)
            warm_info = json.loads((folder / "warm_start.json").read_text())
            source_path = EXP / policy / f"beer_{beer}/profiles.csv"
            assert file_sha(source_path) == warm_info["source_profiles_sha256"]
            old_profiles = pd.read_csv(source_path)
            old = old_profiles[old_profiles.candidate_id.eq(warm_info["candidate_id"])].iloc[0]
            source_values = old[problem.features].to_numpy(float)
            same = (source_values == c["x0"]) | (np.isnan(source_values) & np.isnan(c["x0"]))
            expected_support = np.union1d(np.flatnonzero(~same), c["mandatory"]).astype(int)
            np.testing.assert_array_equal(warm["support"], expected_support)
            expected_delta = problem.control.map_.transform(source_values[None])[0]-c["z0"]
            expected_delta[np.setdiff1d(np.arange(problem.p), expected_support)] = 0.
            np.testing.assert_allclose(warm["delta"], expected_delta, rtol=0, atol=1e-13)
            pool = np.load(folder / "raw_pool.npz", allow_pickle=False)
            assert pool["delta"].shape == pool["binary"].shape == pool["epigraph"].shape
            assert len(pool["delta"]) == bound["solutions"] == record["stored_pool_points"]
            profiles = pd.read_csv(folder / "raw_profiles.csv", float_precision="round_trip")
            assert len(profiles) == len(pool["delta"])+1 and profiles.candidate_id.is_unique
            np.testing.assert_allclose(profiles.iloc[0][problem.features].to_numpy(float),
                                       source_values, rtol=0, atol=0, equal_nan=True)
            for k, (d, b, t) in enumerate(zip(pool["delta"], pool["binary"], pool["epigraph"])):
                np.testing.assert_allclose(pool["objective"][k], b.sum(), rtol=0, atol=1e-7)
                support = np.flatnonzero(b > .5)
                decoded, _ = problem.decode(beer, d, support)
                np.testing.assert_allclose(profiles.iloc[k+1][problem.features].to_numpy(float),
                    np.array([decoded[name] for name in problem.features]), rtol=0, atol=0, equal_nan=True)
                maximum_abs = np.maximum(abs(c["lo"]), abs(c["hi"]))
                offset = c["z0"]-problem.mu
                radius2 = .9**2*(offset@P@offset)
                perspective_value = np.linalg.norm(L@d)**2+diag@t+2*(P@offset)@d+offset@P@offset
                diagnostic = dict(policy=policy, beer=beer, pool_id=k,
                    objective=float(b.sum()), binary_max_error=float(np.max(abs(b-np.rint(b)))),
                    link_violation=float(max(np.max(d-c["hi"]*b), np.max(c["lo"]*b-d))),
                    perspective_violation=float(np.max(d*d-t*b)),
                    epigraph_upper_violation=float(np.max(t-maximum_abs**2*b)),
                    secant_violation=float(np.max(t-(c["lo"]+c["hi"])*d+c["lo"]*c["hi"]*b)),
                    classification_violation=float(np.max(retained["A"]@d-retained["rhs"])),
                    distance_ball_violation=float(perspective_value-radius2),
                    omitted_linear_stale_delta=float(problem.a@d))
                pool_diagnostics.append(diagnostic)
            checks, sensory = problem.check_rows(profiles.to_dict("records"))
            saved_checks = pd.read_csv(folder / "full_checks.csv", float_precision="round_trip")
            assert np.array_equal(checks.accepted, saved_checks.accepted)
            for name in checks.select_dtypes(include=[np.number]).columns:
                np.testing.assert_allclose(checks[name], saved_checks[name], rtol=0, atol=1e-10)
            saved_sensory = pd.read_csv(folder / "predicted_sensory.csv", float_precision="round_trip")
            np.testing.assert_allclose(sensory[problem.bundle["targets"]], saved_sensory[problem.bundle["targets"]], rtol=0, atol=1e-10)
            assert checks.iloc[0].accepted and checks.accepted.any()
            valid = checks[checks.accepted]
            upper = int(valid.raw_fields_changed.min())
            assert upper == record["stronger_guarded_full_upper_bound"] and lower <= upper
            assert int(checks.accepted.sum()) == record["fully_guarded_accepted"]
            for k in np.flatnonzero(checks.accepted):
                witness_diagnostics.append(dict(policy=policy, beer=beer, candidate_id=int(profiles.iloc[k].candidate_id),
                    kind=profiles.iloc[k].method, raw_fields_changed=int(checks.iloc[k].raw_fields_changed),
                    distance_reduction=float(checks.iloc[k].target_distance_reduction),
                    linear_stale_delta=float(checks.iloc[k].linear_stale_delta),
                    nonlinear_stale_delta=float(checks.iloc[k].ensemble_stale_delta),
                    both_stale_conditions_pass=True))
            prior = comparison[comparison.policy.eq(policy) & comparison.beer.eq(beer)].iloc[0]
            assert prior.broader_integer_lower_bound == lower and prior.stronger_guarded_full_upper_bound == upper
            allcases.append(dict(policy=policy, beer=beer, omitted_constraints=1, retained_classification_rows=21,
                declared_seconds=30., solver_seconds=bound["seconds"], status=bound["status"],
                dual_bound=bound["dual_bound"], integer_lower=lower, complete_upper=upper,
                pool_points=len(pool["delta"]), fully_guarded_witnesses=len(valid),
                warm_complete_check=True, split_max_error=split_error, factor_max_error=factor_error,
                protocol_sha256=file_sha(folder / "protocol.json"), scip_log_sha256=file_sha(folder / "scip.log")))
    case_frame = pd.DataFrame(allcases)
    pool_frame = pd.DataFrame(pool_diagnostics)
    witness_frame = pd.DataFrame(witness_diagnostics)
    case_frame.to_csv(HERE / "certificate_without_proxy_case_checks.csv", index=False)
    pool_frame.to_csv(HERE / "certificate_without_proxy_pool_residuals.csv", index=False)
    witness_frame.to_csv(HERE / "certificate_without_proxy_witness_checks.csv", index=False)
    answer = dict(completed=True, no_new_optimization_or_fitting=True, cases=len(case_frame),
        total_saved_pool_points=len(pool_frame), complete_guarded_witnesses=len(witness_frame),
        all_22_warm_witnesses_pass=True, only_linear_stale_row_omitted=True,
        retained_classification_rows_per_case=21, declared_seconds_per_case=30.,
        solver_seconds_min=float(case_frame.solver_seconds.min()), solver_seconds_max=float(case_frame.solver_seconds.max()),
        lower_bound_ranges_by_policy={policy:[int(group.integer_lower.min()),int(group.integer_lower.max())]
                                     for policy, group in case_frame.groupby("policy")},
        chemistry_sha256=chemistry_sha, source_hashes=expected_hashes,
        maximum_split_error=float(case_frame.split_max_error.max()), maximum_factor_error=float(case_frame.factor_max_error.max()),
        raw_pool_maximum_violations={name:float(pool_frame[name].max()) for name in pool_frame.columns if name.endswith("violation") or name == "binary_max_error"},
        audit_code_sha256=file_sha(__file__),
        global_bound_scope="SCIP floating-point branch-and-bound bounds checked against saved definition, parameters and logs; not independently solved or exact-arithmetic verified.",
        set_scope="F_guard_full subset F_broad_full subset R_broad. Removing the linear guard from the relaxation yields a valid bound for the broader complete-nonlinear-stale problem. Guarded complete witnesses remain valid upper bounds for it.",
        primary_selection_unchanged=True)
    (HERE / "certificate_without_proxy_independent_audit.json").write_text(json.dumps(answer, ensure_ascii=False, indent=2)+"\n")
    lines = ["# 删除线性守卫后的更广问题：独立证书审计", "",
        f"22个已完成案例逐份核验代码、数据、模型、SDP分解和结果哈希；全部仅删除线性陈旧味守卫一行，保留21个风格分类约束、同样的化学盒界、补全成本与10%距离要求。每例登记30秒、1线程，实际求解时间为{case_frame.solver_seconds.min():.3f}–{case_frame.solver_seconds.max():.3f}秒。没有另行求解。", "",
        "## 集合与证书适用范围", "",
        "设F₀包含盒界、化学政策、类别、距离及必须补全字段；令g为线性陈旧味守卫，f★为完整冻结预测器。则F_guard_full=F₀∩{g≤0}∩{Δf★≤0}，F_broad_full=F₀∩{Δf★≤0}，而本次求解松弛为R_broad=F₀。因此F_guard_full⊆F_broad_full⊆R_broad。对R_broad的合法下界可以约束更广完整问题，原含守卫完整合格见证仍给它提供上界。原含守卫松弛的下界则不能反向当作本问题的下界。", "",
        "| 政策 | 逐款整数下界范围 | 完整见证上界范围 |", "|---|---:|---:|"]
    for policy, group in case_frame.groupby("policy"):
        lines.append(f"| {policy} | {group.integer_lower.min()}–{group.integer_lower.max()} | {group.complete_upper.min()}–{group.complete_upper.max()} |")
    lines += ["", f"全部22个原始warm见证重新通过完整模型；保存池共{len(pool_frame)}个点，其中逐点解码后共有{len(witness_frame)}个含warm的完整合格行。所有上界按实际原字段变化计数。失败的池点不提供完整问题上界；解池的分类/几何可行性也不代替完整陈旧味约束。", "",
              "## 数值与复现", "",
              "SDP对角分解逐值匹配冻结精度矩阵，残差正定、对角非负、因子与矩阵恒等式成立。日志末行对偶/原始界与JSON一致；整数下界采用ceil(dual−10⁻⁶)并计入必须补全字段。SCIP对非线性式没有全局assumeconvex错误声明，SOC和perspective处理器启用。", "",
              "该审计核查已保存数值证书的定义、记录和完整见证，不重放全部分支定界树，也不是精确算术下界证明。30秒预算是补充求解器阶段预算，不能据其与主180秒结果直接作算法速度排名。新池只服务于独立证书检查，没有进入主候选选择或bootstrap反馈。", "",
              "逐例审计、原始池约束残差和完整见证分别保存为certificate_without_proxy_case_checks.csv、certificate_without_proxy_pool_residuals.csv和certificate_without_proxy_witness_checks.csv。"]
    (HERE / "certificate_without_proxy_independent_review.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
