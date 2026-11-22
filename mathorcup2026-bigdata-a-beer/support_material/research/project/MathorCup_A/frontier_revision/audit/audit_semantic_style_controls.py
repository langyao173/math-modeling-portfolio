"""Recompute style-preserving semantic controls from their saved evidence.

No tuning or fitting is performed. Predictions, inner selections, closed-form
blend weights, all local permutations and conditional paired intervals are
independently replayed after all six registered variants have completed.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "experiments/semantic_auxiliary"
OUT = EXP / "style_controls"
sys.path.insert(0, str(EXP))
# Original aligned checkpoints were emitted when run_semantic_rrr.py was the
# __main__ script. Exposing its class here resolves that historical pickle name
# without rewriting the checkpoint or changing the model implementation.
from run_semantic_rrr import PROJECT, AW, SEED, SPECS, SemanticRRR


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def direct_r2(y, p):
    return 1-np.sum((y-p)**2, axis=0)/np.sum((y-y.mean(0))**2, axis=0)


def local_auxiliary(S, labels, kind, seed):
    """Rebuild local labels without calling the experiment's helper."""
    levels = sorted(set(labels))
    if kind == "style_only":
        values = np.column_stack([labels == name for name in levels]).astype(float)
        permutation = np.arange(len(labels))
        singletons = 0
    else:
        generator = np.random.default_rng(seed)
        permutation = np.arange(len(labels))
        singletons = 0
        for name in levels:
            rows = np.flatnonzero(labels == name)
            permutation[rows] = generator.permutation(rows)
            singletons += int(len(rows) == 1)
        values = S[permutation]
    assert np.array_equal(np.sort(permutation), np.arange(len(labels)))
    assert np.array_equal(labels[permutation], labels)
    return values, permutation, singletons, len(levels)


def active_columns(values):
    count = np.isfinite(values).sum(0)
    means = np.divide(np.nansum(values, axis=0), count,
                      out=np.zeros(values.shape[1]), where=count > 0)
    filled = np.where(np.isfinite(values), values, means)
    return np.std(filled, axis=0) > 1e-10


def main():
    complete = json.loads((OUT / "COMPLETE.json").read_text())
    assert complete["complete"] and complete["fitted_outer_models"] == 60
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert complete["protocol_sha256"] == sha(OUT / "protocol.json")
    assert complete["prediction_sha256"] == sha(OUT / "oof_predictions.npz")
    assert protocol["code_sha256"] == sha(EXP / "style_controls.py")
    assert protocol["reference_code_sha256"] == sha(EXP / "run_semantic_rrr.py")
    assert protocol["specifications"] == SPECS and len(SPECS) == 190
    for index in (1, 4, 7):
        assert protocol["source_hashes"][str(index)] == sha(PROJECT / f"data/A/{index}.csv")
    xframe = pd.read_csv(PROJECT / "data/A/1.csv").sort_values("beer").reset_index(drop=True)
    yframe = pd.read_csv(PROJECT / "data/A/4.csv").set_index("beer").loc[xframe.beer]
    sframe = pd.read_csv(PROJECT / "data/A/7.csv").set_index("beer").loc[xframe.beer]
    targets = list(yframe.columns[2:])
    X, Y, S = xframe.iloc[:, 3:].to_numpy(float), yframe[targets].to_numpy(float), sframe.iloc[:, 2:].to_numpy(float)
    labels = xframe.tasting_category_fine.to_numpy(str)
    core = np.flatnonzero([name.startswith(("A_", "F_")) for name in targets])
    assert X.shape == (250, 231) and Y.shape == (250, 50) and len(core) == 30
    prediction = np.load(OUT / "oof_predictions.npz", allow_pickle=False)
    blended = np.load(OUT / "blend_oof.npz", allow_pickle=False)
    for saved in (prediction, blended):
        np.testing.assert_array_equal(saved["beer"], xframe.beer)
        np.testing.assert_array_equal(saved["Y"], Y)
    reference = np.load(EXP / "oof_predictions.npz", allow_pickle=False)
    reference_blend = np.load(EXP / "blend_oof.npz", allow_pickle=False)
    old_oof = np.load(AW / "q1/oof_arrays.npz", allow_pickle=False)["nested_ensemble"]
    selections = json.loads((OUT / "nested_selection.json").read_text())
    grid = pd.read_csv(OUT / "inner_scores.csv", float_precision="round_trip")
    recorded_weights = pd.read_csv(OUT / "blend_weights.csv", float_precision="round_trip")
    permutations = pd.read_csv(OUT / "permutation_diagnostics.csv", float_precision="round_trip")
    assert len(selections) == 60 and len(grid) == 60*190 and len(permutations) == 120
    assert not recorded_weights.duplicated(["fold", "model"]).any()
    assert not permutations.duplicated(["variant", "fold", "inner", "stage"]).any()
    allchecks, permutation_checks, reference_capacity = [], [], []
    folds = np.full(250, -1, dtype=int)
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(X)):
        folds[te] = fold
        old = np.load(AW / f"nature_upgrade/adaptive/outer_{fold}_old_cache.npz", allow_pickle=False)
        np.testing.assert_array_equal(old["train_indices"], tr)
        np.testing.assert_array_equal(old["test_indices"], te)
        np.testing.assert_allclose(old["test_predictions"], old_oof[te], rtol=0, atol=1e-13)
        sd = np.maximum(Y[tr].std(0, ddof=1), 1e-10)
        for metric in ["all50", "core30"]:
            aligned_model = joblib.load(EXP / f"fold_{fold}_aligned_aux_{metric}.joblib")
            reference_active = int(aligned_model.semantic_active.sum())
            reference_rank = int(aligned_model.spec["rank"])
            reference_capacity.append(dict(fold=fold, variant="aligned_aux", selection=metric,
                selected_rank=reference_rank, active_auxiliary_columns=reference_active,
                joint_target_columns=50+reference_active,
                nominal_rank_covers_all_joint_columns=reference_rank >= 50+reference_active,
                coefficient_numerical_rank=int(np.linalg.matrix_rank(aligned_model.coef))))
        for name, seed in protocol["variants"]:
            kind = "style_only" if name == "style_only" else "within_style"
            splits = [("inner", inner, tr[itr], tr[ite], seed+1009*fold+137*inner)
                      for inner, (itr, ite) in enumerate(KFold(3, shuffle=True, random_state=SEED+fold).split(tr))]
            splits.append(("outer", -1, tr, te, seed+100003+fold))
            outer_aux = None
            for stage, inner, fitting, validation, local_seed in splits:
                assert set(fitting).isdisjoint(validation)
                assert set(fitting).isdisjoint(te) if stage == "inner" else True
                aux, perm, singletons, levels = local_auxiliary(S[fitting], labels[fitting], kind, local_seed)
                record = permutations[(permutations.variant == name) & (permutations.fold == fold)
                                      & (permutations.inner == inner) & (permutations.stage == stage)].iloc[0]
                assert record.n == len(fitting) and record.active_styles == levels
                assert record.singleton_rows == singletons
                assert record.moved_rows == np.sum(perm != np.arange(len(perm)))
                if kind == "within_style":
                    assert record.permutation_sha256 == hashlib.sha256(perm.tobytes()).hexdigest()
                    # Absolute row identities stay in this fit partition;
                    # label exchange also preserves each source style.
                    assert set(fitting[perm]) == set(fitting)
                    assert set(fitting[perm]).isdisjoint(validation)
                permutation_checks.append(dict(variant=name, fold=fold, stage=stage, inner=inner,
                    training_rows=len(fitting), active_styles=levels, singleton_rows=singletons,
                    moved_rows=int(np.sum(perm != np.arange(len(perm)))),
                    validation_rows_excluded=True, style_membership_preserved=True,
                    fitting_row_sha256=hashlib.sha256(fitting.tobytes()).hexdigest(),
                    local_seed=local_seed))
                if stage == "outer":outer_aux = aux
            inner_saved = np.load(OUT / f"fold_{fold}_{name}_selected_inner.npz", allow_pickle=False)
            np.testing.assert_array_equal(inner_saved["train_indices"], tr)
            np.testing.assert_array_equal(inner_saved["test_indices"], te)
            for metric, idx in [("all50", np.arange(50)), ("core30", core)]:
                family = f"{name}_{metric}"
                selected = next(record for record in selections if record["fold"] == fold and record["family"] == family)
                scores = grid[(grid.fold == fold) & (grid.variant == name) & (grid.selection == metric)].sort_values("spec_id")
                assert np.array_equal(scores.spec_id, np.arange(190))
                eligible = np.array([spec["tau"] > 0 for spec in SPECS])
                assert np.array_equal(scores.eligible.to_numpy(), eligible)
                candidate_ids = np.flatnonzero(eligible)
                best = int(candidate_ids[np.argmin(scores.inner_nmse.to_numpy()[candidate_ids])])
                assert selected["spec"] == SPECS[best]
                pp = inner_saved[metric]
                nmse = float(np.mean(((pp[:, idx]-Y[tr][:, idx])/sd[idx])**2))
                assert abs(nmse-selected["inner_nmse"]) < 1e-12
                assert abs(nmse-scores.iloc[best].inner_nmse) < 1e-12
                model = joblib.load(OUT / f"fold_{fold}_{family}.joblib")
                assert model.spec == SPECS[best] and not model.permuted
                expected_active = active_columns(outer_aux)
                np.testing.assert_array_equal(model.semantic_active, expected_active)
                replay = model.predict(X[te])
                prediction_error = float(np.max(abs(replay-prediction[family][te])))
                assert prediction_error < 1e-12
                residual = (old["inner_predictions"][:, idx]-Y[tr][:, idx])/sd[idx]
                direction = (pp[:, idx]-old["inner_predictions"][:, idx])/sd[idx]
                raw_weight = -float(np.mean(residual*direction))/(float(np.mean(direction**2))+.02)
                weight = float(np.clip(raw_weight, 0, 1))
                stored_weight = float(recorded_weights[(recorded_weights.fold == fold)
                    & (recorded_weights.model == family)].weight.iloc[0])
                assert abs(weight-stored_weight) < 1e-13
                replay_blend = (1-weight)*old_oof[te]+weight*replay
                blend_error = float(np.max(abs(replay_blend-blended[family][te])))
                assert blend_error < 1e-12
                active = int(expected_active.sum())
                rank = int(model.spec["rank"])
                allchecks.append(dict(fold=fold, model=family, variant=name, selection=metric,
                    selected_spec_id=best, selected_rank=rank, active_auxiliary_columns=active,
                    joint_target_columns=50+active,
                    nominal_rank_covers_all_joint_columns=rank >= 50+active,
                    coefficient_numerical_rank=int(np.linalg.matrix_rank(model.coef)),
                    alpha=model.spec["alpha"], tau=model.spec["tau"], mode=model.spec["mode"],
                    selected_inner_nmse=nmse, blend_weight=weight,
                    prediction_replay_max_error=prediction_error, blend_replay_max_error=blend_error))
    for saved in (prediction, blended, reference, reference_blend):
        np.testing.assert_array_equal(saved["fold"], folds)
    checks = pd.DataFrame(allchecks)
    # Reconstruct all paired bootstrap replicates with count-weighted sufficient
    # statistics. This is algebraically independent of row-slice paired_ci().
    generator = np.random.default_rng(10041)
    counts = np.array([np.bincount(generator.integers(0, 250, 250), minlength=250) for _ in range(2000)])
    denominator = counts@(Y**2)-(counts@Y)**2/250
    denominator = np.maximum(denominator, 1e-20)
    reported = pd.read_csv(OUT / "model_comparison.csv", float_precision="round_trip")
    contrasts = json.loads((OUT / "contrasts.json").read_text())
    result_rows, dimension_rows = [], []
    for stage, saved, aligned in [("rrr", prediction, reference), ("blend", blended, reference_blend)]:
        for name, _ in protocol["variants"]:
            for metric in ["all50", "core30"]:
                family = f"{name}_{metric}"
                p = saved[family]
                q = aligned[f"aligned_aux_{metric}"]
                r, ar = direct_r2(Y, p), direct_r2(Y, q)
                row = reported[(reported.stage == stage) & (reported.model == family)].iloc[0]
                assert abs(r.mean()-row.all50_r2) < 1e-13 and abs(r[core].mean()-row.core30_r2) < 1e-13
                record = next(item for item in contrasts if item["stage"] == stage and item["control"] == family)
                errors = (p-Y)**2-(q-Y)**2
                draw_deltas = (counts@errors)/denominator
                allci = np.quantile(draw_deltas.mean(1), [.025, .975])
                coreci = np.quantile(draw_deltas[:, core].mean(1), [.025, .975])
                np.testing.assert_allclose(allci, record["all50_ci"], atol=1e-12, rtol=0)
                np.testing.assert_allclose(coreci, record["core30_ci"], atol=1e-12, rtol=0)
                assert abs((ar-r).mean()-record["all50_delta"]) < 1e-13
                assert abs((ar-r)[core].mean()-record["core30_delta"]) < 1e-13
                result_rows.append(dict(stage=stage, model=family, variant=name, selection=metric,
                    all50_r2=float(r.mean()), core30_r2=float(r[core].mean()),
                    aligned_minus_control_all50=float((ar-r).mean()), all50_ci_lower=allci[0], all50_ci_upper=allci[1],
                    aligned_minus_control_core30=float((ar-r)[core].mean()), core30_ci_lower=coreci[0], core30_ci_upper=coreci[1]))
                dimension_rows.extend(dict(stage=stage, model=family, target=target, r2=float(r[j]),
                    aligned_r2=float(ar[j]), aligned_minus_control=float(ar[j]-r[j])) for j, target in enumerate(targets))
    results = pd.DataFrame(result_rows)
    permutation_frame = pd.DataFrame(permutation_checks)
    checks.to_csv(HERE / "semantic_style_control_model_checks.csv", index=False)
    capacity = pd.concat([checks, pd.DataFrame(reference_capacity)], ignore_index=True)
    capacity.to_csv(HERE / "semantic_style_control_effective_capacity.csv", index=False)
    permutation_frame.to_csv(HERE / "semantic_style_control_permutation_checks.csv", index=False)
    results.to_csv(HERE / "semantic_style_control_metric_checks.csv", index=False)
    pd.DataFrame(dimension_rows).to_csv(HERE / "semantic_style_control_dimension_checks.csv", index=False)
    outcome = dict(completed=True, no_new_fitting_or_selection=True, outer_models_replayed=len(checks),
        training_partition_auxiliaries_checked=len(permutation_checks),
        within_style_permutations_checked=int(permutation_frame.variant.ne("style_only").sum()),
        style_only_auxiliaries_checked=int(permutation_frame.variant.eq("style_only").sum()), specifications=190,
        compared_prediction_variants=len(results), paired_bootstrap_replicates=2000,
        maximum_outer_prediction_error=float(checks.prediction_replay_max_error.max()),
        maximum_blend_replay_error=float(checks.blend_replay_max_error.max()),
        local_style_membership_and_validation_exclusion=True,
        paired_CI_independent_count_weighted_reconstruction=True,
        source_protocol_sha256=sha(OUT / "protocol.json"), audit_code_sha256=sha(__file__),
        effective_capacity_scope="True S versus within-style-permuted S share dimensions and nominal rank grid. Style-only replaces the auxiliary task and may cover its complete joint output span at rank 80; it is not a capacity-identical causal ablation.",
        interval_scope="Paired intervals condition on saved internal OOF predictions; five permutation controls are not ensembled and do not yield familywise or external-validation significance.")
    (HERE / "semantic_style_controls_independent_audit.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2)+"\n")
    lines = ["# 语义风格控制：独立复算与解释边界", "",
        "60个外层模型均以保存模型重放；120个训练分区的辅助构造均由原行身份和固定种子重建，验证行始终未进入置换。内层融合权重由保存OOF和原基线独立计算，所有24组输出的指标及2000次成对区间由计数加权公式复算。没有重新训练、选择或平均五个控制模型。", "",
        "## 可比较的性能", "",
        "下表all50和core30分别使用对应目标在内层选出的支路；不是在外层结果中再挑最优支路。", "",
        "| 辅助任务 | RRR all50 | 融合 all50 | RRR core30 | 融合 core30 |",
        "|---|---:|---:|---:|---:|"]
    def value(stage, variant, selection, target):
        return float(results[(results.stage == stage) & (results.variant == variant) & (results.selection == selection)][target].iloc[0])
    for name, _ in protocol["variants"]:
        lines.append(f"| {name} | {value('rrr',name,'all50','all50_r2'):.6f} | {value('blend',name,'all50','all50_r2'):.6f} | {value('rrr',name,'core30','core30_r2'):.6f} | {value('blend',name,'core30','core30_r2'):.6f} |")
    for stage, aligned in [("rrr", reference), ("blend", reference_blend)]:
        allvalue = direct_r2(Y, aligned["aligned_aux_all50"]).mean()
        corevalue = direct_r2(Y, aligned["aligned_aux_core30"])[core].mean()
        lines += ["", f"真实对齐S的{stage}参照：all50={allvalue:.6f}，core30={corevalue:.6f}。"]
    lines += ["", "## 实际选择与容量", "",
              "| 辅助类型 | 选择目标 | 选中rank范围 | 有效辅助列数范围 | rank覆盖全部联合列的外折数/拟合数 |", "|---|---|---:|---:|---:|"]
    for (name, metric), group in capacity.groupby(["variant", "selection"]):
        lines.append(f"| {name} | {metric} | {group.selected_rank.min()}–{group.selected_rank.max()} | {group.active_auxiliary_columns.min()}–{group.active_auxiliary_columns.max()} | {int(group.nominal_rank_covers_all_joint_columns.sum())}/{len(group)} |")
    style_selected = checks[checks.variant.eq("style_only")]
    full_count = int(style_selected.nominal_rank_covers_all_joint_columns.sum())
    actual_capacity = (f"本次实际style-only选择的rank为{style_selected.selected_rank.min()}–{style_selected.selected_rank.max()}，"
                      f"覆盖全部联合列的配置为{full_count}/{len(style_selected)}；")
    actual_capacity += "全秩特例没有出现在实际选中配置中。" if full_count == 0 else "其具体外折已逐项保留。"
    lines += ["", "真实S与风格内置换S保持辅助维数、缩放和名义秩网格，是本项控制中更直接的比较。style-only用独热风格替换辅助任务，联合输出列数约为50加有效风格列数；rank=80可覆盖其完整联合输出空间，此时退化为全秩岭的Y分支。故相同190规格不代表两类辅助的有效容量严格相同，也不能将style-only差异解释为风格的因果贡献。"+actual_capacity, "",
              "内层融合精确最小化平均标准化平方误差加0.02w²，其解为w=clip(−〈r,h〉/(〈h,h〉+0.02),0,1)。所有量仅使用相应外层训练集的inner-OOF预测。外层是本轮持续方法探索共用的内部评价，不构成新外部验证。", "",
              "每个固定种子单独报告其相对真实S的成对区间。区间的分母重新计算重抽样目标方差；它只条件于已有OOF预测，未重做完整训练，也没有多重比较家族覆盖保证。五次随机置换不能据此包装为高精度随机化显著性检验。"]
    (HERE / "semantic_style_controls_independent_review.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(outcome, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
