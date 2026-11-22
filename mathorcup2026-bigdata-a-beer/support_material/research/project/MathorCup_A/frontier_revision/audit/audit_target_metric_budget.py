"""Independent read-only recomputation of the weighted OOF error identity.

No prediction model is imported, fitted, selected, or run. The training-fold
PCA projector is checked from a direct eigen-decomposition of saved Y solely
to verify the provenance of the saved orthogonal basis.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MOD = HERE.parent / "experiments/modern_prediction"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    a = np.load(MOD / "oof_arrays.npz", allow_pickle=False)
    d = np.load(MOD / "tabdpt/oof_arrays.npz", allow_pickle=False)
    y, fold = a["Y"], a["fold"]
    assert np.array_equal(d["Y"], y) and np.array_equal(d["fold"], fold)
    assert len(set(a["beer"].tolist())) == 250
    assert all(t.startswith(("A_", "F_")) for t in a["targets"][:30])
    predictions = {k: a[k] for k in a.files if k not in ("Y", "fold", "beer", "targets")}
    predictions["tabdpt13_raw231_pca128_y50"] = d["prediction"]
    protocol = json.loads((MOD / "diagnostics/target_metric_budget_protocol.json").read_text())
    for name, value in protocol["sources_sha256"].items():
        assert sha(MOD / name) == value
    assert sha(MOD / "diagnose_target_metric_budget.py") == protocol["implementation_sha256"]
    sst = np.sum((y - y.mean(0)) ** 2, axis=0)
    fold_checks, results = [], []
    for f in range(5):
        c = np.load(MOD / "checkpoints" / f"raw231_y16_fold{f}_complete.npz", allow_pickle=False)
        test = np.flatnonzero(fold == f)
        train = np.flatnonzero(fold != f)
        assert np.array_equal(c["test_indices"], test) and np.array_equal(c["train_indices"], train)
        assert len(train) == 200 and len(test) == 50 and len(np.intersect1d(train, test)) == 0
        mu, sigma = y[train].mean(0), y[train].std(0, ddof=1)
        mean_error = np.max(np.abs(mu - c["target_mean"]))
        scale_error = np.max(np.abs(sigma - c["target_scale"]))
        z = (y[train] - mu) / sigma
        _, eigenvectors = np.linalg.eigh(z.T @ z)
        independent_p = eigenvectors[:, -16:] @ eigenvectors[:, -16:].T
        v = c["target_components"].T
        p = v @ v.T
        projector_error = np.max(np.abs(p - independent_p))
        assert mean_error < 1e-12 and scale_error < 1e-12 and projector_error < 1e-10
        fold_checks.append(dict(fold=f, train_rows=len(train), test_rows=len(test),
            mean_error=float(mean_error), scale_error=float(scale_error),
            independently_reconstructed_projector_error=float(projector_error),
            idempotence_error=float(np.max(np.abs(p @ p - p)))))
        for name, pred in predictions.items():
            e = (pred[test] - y[test]) / sigma
            # Reconstruct both error components in raw target units, then
            # score directly. This avoids borrowing the original code's
            # coordinate-weight expression for the independent calculation.
            u = (e @ p) * sigma
            w = (e @ (np.eye(50) - p)) * sigma
            for group, cols in {"all50": np.arange(50), "core30": np.arange(30)}.items():
                first = float(np.mean((u[:, cols] ** 2).sum(0) / sst[cols]))
                second = float(np.mean((w[:, cols] ** 2).sum(0) / sst[cols]))
                cross = float(np.mean((2 * u[:, cols] * w[:, cols]).sum(0) / sst[cols]))
                direct = float(np.mean(((pred[test][:, cols] - y[test][:, cols]) ** 2).sum(0) / sst[cols]))
                results.append(dict(model=name, targets=group, fold=f,
                    projected_error_term=first, orthogonal_error_term=second,
                    weighted_cross_term=cross, r2_loss_contribution=direct,
                    identity_error=abs(first + second + cross - direct)))
    rows = pd.DataFrame(results)
    expected = pd.read_csv(MOD / "diagnostics/target_metric_budget_by_fold.csv")
    keys = ["model", "targets", "fold"]
    left, right = rows.set_index(keys), expected.set_index(keys)
    columns = ["projected_error_term", "orthogonal_error_term", "weighted_cross_term", "r2_loss_contribution"]
    saved_error = float(np.max(np.abs(left[columns].sort_index().to_numpy() - right[columns].sort_index().to_numpy())))
    total = rows.groupby(["model", "targets"])[columns].sum()
    total["macro_r2"] = 1 - total.r2_loss_contribution
    direct_errors = []
    for (name, group), row in total.iterrows():
        cols = np.arange(50 if group == "all50" else 30)
        direct = np.mean((1 - ((predictions[name] - y) ** 2).sum(0) / sst)[cols])
        direct_errors.append(abs(direct - row.macro_r2))
    contrasts = pd.read_csv(MOD / "diagnostics/target_metric_budget_contrasts.csv")
    delta_errors = []
    for row in contrasts.itertuples():
        difference = total.loc[(row.model, row.targets)] - total.loc[(row.reference, row.targets)]
        for original, derived in [("delta_projected_error", "projected_error_term"),
                                  ("delta_orthogonal_error", "orthogonal_error_term"),
                                  ("delta_weighted_cross", "weighted_cross_term"),
                                  ("delta_macro_r2", "macro_r2")]:
            delta_errors.append(abs(float(getattr(row, original)) - float(difference[derived])))
    assert saved_error < 1e-12 and max(direct_errors) < 1e-12 and max(delta_errors) < 1e-12
    rows.to_csv(HERE / "target_metric_budget_independent_checks.csv", index=False)
    pd.DataFrame(fold_checks).to_csv(HERE / "target_metric_budget_projector_checks.csv", index=False)
    result = dict(completed=True, no_new_model_fitting_or_inference=True,
        models=len(predictions), target_groups=2, fold_target_model_cells=len(rows),
        training_fold_projectors_reconstructed=5,
        maximum_projector_error=max(x["independently_reconstructed_projector_error"] for x in fold_checks),
        maximum_saved_component_error=saved_error,
        maximum_direct_macro_r2_error=max(direct_errors),
        maximum_contrast_error=max(delta_errors),
        maximum_independent_identity_error=float(rows.identity_error.max()),
        source_hashes_match=True,
        protocol_sha256=sha(MOD / "diagnostics/target_metric_budget_protocol.json"),
        audit_code_sha256=sha(__file__),
        interpretation="Exact accounting for fixed saved predictions. The cross term need not vanish under unequal macro-R2 weights or a coordinate subset; no causal, irreducible-noise, or new-validation claim.")
    (HERE / "target_metric_budget_independent_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 任务加权误差恒等式的独立复核", "",
        "结论：附录C.5的定义、符号和数值均通过独立复算。没有重新训练或运行任何预测模型。", "",
        f"从保存的真实Y与折号分别重建5个训练均值、样本标准差及PCA16投影；每折200训练/50评价，投影矩阵与独立特征分解的最大误差为{result['maximum_projector_error']:.3g}。随后将两个正交误差分量重新变回原标签单位，再直接评分，独立于原实现的坐标权重写法。", "",
        f"14个预测器、2个目标组和5折的140项分解均核对；保存分量最大误差{saved_error:.3g}，直接宏R²最大误差{max(direct_errors):.3g}，6个对比的最大误差{max(delta_errors):.3g}。输入数据及检查点哈希均与原协议一致。", "",
        "正交性只在完整50维、训练标准化、无权欧氏内积中消去交叉项。宏R²逐目标以原尺度SST归一化，且核心30维再截取坐标，因此保留加权交叉项是必要的。Native50相对Y-PCA16在全50维提高而核心30维降低，与分解数值一致。", "",
        "该恒等式说明同一组固定预测的评分预算，不把三个项变为可独立干预的机制，也不证明其中一项是不可约噪声。原分解不产生新的外部验证或独立选型证据。", "",
        "证据：target_metric_budget_independent_checks.csv、target_metric_budget_projector_checks.csv与target_metric_budget_independent_audit.json。"]
    (HERE / "target_metric_budget_independent_review.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
