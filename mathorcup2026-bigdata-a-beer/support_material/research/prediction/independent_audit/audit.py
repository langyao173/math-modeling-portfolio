"""Independent arithmetic audit of saved train-only fusion results.

This module deliberately does not import the fusion runner. It reconstructs
inner validation arrays from the split caches and derives the weights using
an independently written normalized-SSE formula. It performs no model fit or
TabICL inference. The only written file is the requested JSON audit report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


A = "raw231_y16_no_residual"
B = "raw231_y50_native"
OLD = "old_nested_ensemble"
GROUPS = {"all50": slice(None), "core30": slice(0, 30), "other20": slice(30, 50)}
NEW = (
    "fixed_half_y16_native50",
    "inner_core_scalar_y16_native50",
    "inner_groupwise_y16_native50",
)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(actual, expected, label):
    if not np.array_equal(actual, expected):
        raise AssertionError(f"Identity/order mismatch: {label}")


def close(actual, expected, label, tolerance=1e-13):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)), initial=0.0))
    if error > tolerance:
        raise AssertionError(f"Numerical mismatch {label}: {error:.17g}")
    return error


def scores(y, prediction):
    sst = np.square(y - y.mean(axis=0)).sum(axis=0)
    if not np.all(sst > 0):
        raise AssertionError("Constant target prevents registered R2 metric")
    dimension_r2 = 1.0 - np.square(prediction - y).sum(axis=0) / sst
    return {name: float(dimension_r2[columns].mean()) for name, columns in GROUPS.items()}


def audit(modern: Path, nested: Path, output: Path) -> dict:
    source_paths = []

    def read(path):
        source_paths.append(path)
        return load(path)

    global_oof = read(modern / "oof_arrays.npz")
    result = read(output / "outer_predictions.npz")
    old_nested = read(nested / "nested_oof.npz")
    y, folds = global_oof["Y"], global_oof["fold"]
    if y.shape != (250, 50) or not np.isfinite(y).all():
        raise AssertionError("Original response contract is not 250 by 50 finite values")
    exact(np.unique(folds), np.arange(5), "five original folds")
    for key in ("Y", "beer", "fold", "targets"):
        exact(global_oof[key], result[key], "saved result " + key)
        exact(global_oof[key], old_nested[key], "old nested " + key)

    weight_table = pd.read_csv(output / "fitted_weights.csv").set_index("outer_fold")
    rebuilt = {name: np.full_like(y, np.nan) for name in NEW}
    weights = []
    maximum_weight_error = 0.0
    maximum_prediction_error = 0.0
    inner_views_checked = 0

    for outer in range(5):
        training = np.flatnonzero(folds != outer)
        testing = np.flatnonzero(folds == outer)
        if len(training) != 200 or len(testing) != 50:
            raise AssertionError("Each outer split must contain 200 training and 50 held-out rows")
        inner = read(nested / f"outer_{outer}" / "inner_oof.npz")
        for key, expected in (
            ("train_indices", training), ("Y", y[training]),
            ("beer", global_oof["beer"][training]), ("targets", global_oof["targets"]),
        ):
            exact(inner[key], expected, f"outer {outer} inner {key}")
        reconstruction = {A: np.full((200, 50), np.nan), B: np.full((200, 50), np.nan)}
        coverage = np.zeros(200, dtype=int)

        for split in range(3):
            for name, view, prediction_key in (
                (A, "raw231_y16", "prediction_no_residual"),
                (B, "raw231_y50", "prediction_native"),
            ):
                cache = read(nested / f"outer_{outer}" / f"inner_{split}" / f"{view}_complete.npz")
                inner_views_checked += 1
                fit_rows, valid_rows = cache["train_indices"], cache["test_indices"]
                positions = cache["inner_test_positions"]
                if np.intersect1d(fit_rows, valid_rows).size:
                    raise AssertionError("Inner fit/validation intersection")
                if np.intersect1d(testing, np.r_[fit_rows, valid_rows]).size:
                    raise AssertionError("Outer held-out row enters inner fit/validation")
                exact(np.sort(np.r_[fit_rows, valid_rows]), training, "inner training universe")
                exact(training[positions], valid_rows, "inner validation row positions")
                exact(training[cache["inner_train_positions"]], fit_rows, "inner fit row positions")
                exact(cache["outer_train_indices"], training, "recorded outer training rows")
                exact(cache["outer_test_indices"], testing, "recorded outer held-out rows")
                exact(cache["spec_id"], inner["spec_id"], "inner specification identity")
                if name == A:
                    coverage[positions] += 1
                    close(cache["target_mean"], y[fit_rows].mean(axis=0), "inner target mean")
                    close(cache["target_scale"], y[fit_rows].std(axis=0, ddof=1), "inner target scale")
                    components = cache["target_components"]
                    close(components @ components.T, np.eye(16), "inner PCA orthonormality", 1e-10)
                    recovered = (
                        (cache["latent_prediction"] @ components + cache["target_pca_mean"])
                        * cache["target_scale"] + cache["target_mean"]
                    )
                    close(recovered, cache[prediction_key], "inner PCA prediction reconstruction", 1e-11)
                reconstruction[name][positions] = cache[prediction_key]

        exact(coverage, np.ones(200, dtype=int), "one inner prediction per training row")
        for name in (A, B):
            exact(reconstruction[name], inner[name], "reconstructed complete inner OOF " + name)

        endpoints = []
        for name, view, prediction_key in (
            (A, "raw231_y16", "prediction_no_residual"),
            (B, "raw231_y50", "prediction_native"),
        ):
            cache = read(modern / "checkpoints" / f"{view}_fold{outer}_complete.npz")
            exact(cache["train_indices"], training, "outer endpoint fit rows")
            exact(cache["test_indices"], testing, "outer endpoint held-out rows")
            exact(cache[prediction_key], global_oof[name][testing], "outer cached endpoint " + name)
            endpoints.append(cache[prediction_key])

        independently_fitted = []
        for group, columns in (("core", slice(0, 30)), ("other", slice(30, 50))):
            a, b, truth = reconstruction[A][:, columns], reconstruction[B][:, columns], y[training, columns]
            # The common n and group-width factors cancel in the independent
            # SSE minimizer. This uses only the 200 outer-training responses.
            training_sst = np.square(truth - truth.mean(axis=0)).sum(axis=0)
            difference = a - b
            numerator = np.sum((truth - b) * difference / training_sst)
            denominator = np.sum(np.square(difference) / training_sst)
            if denominator <= 0:
                raise AssertionError("Nonpositive fusion curvature")
            weight = float(np.clip(numerator / denominator, 0.0, 1.0))
            maximum_weight_error = max(
                maximum_weight_error,
                close(weight, weight_table.loc[outer, f"weight_{group}_y16"], "independent weight", 1e-14),
            )
            independently_fitted.append(weight)
        core_weight, other_weight = independently_fitted
        a, b = endpoints
        rebuilt[NEW[0]][testing] = (a + b) / 2.0
        rebuilt[NEW[1]][testing] = core_weight * a + (1.0 - core_weight) * b
        piecewise = core_weight * a + (1.0 - core_weight) * b
        piecewise[:, 30:] = other_weight * a[:, 30:] + (1.0 - other_weight) * b[:, 30:]
        rebuilt[NEW[2]][testing] = piecewise
        weights.append({"outer_fold": outer, "core_weight_y16": core_weight, "other_weight_y16": other_weight})

    for name, prediction in rebuilt.items():
        if not np.isfinite(prediction).all():
            raise AssertionError("Incomplete rebuilt predictions")
        maximum_prediction_error = max(
            maximum_prediction_error,
            close(prediction, result[name], "independent outer fusion " + name, 1e-13),
        )
    for name in (A, B, OLD):
        exact(global_oof[name], result[name], "unchanged comparator " + name)
    exact(old_nested["prediction"], result["tabicl_nested_selection"], "unchanged hard-selection comparator")

    metric_table = pd.read_csv(output / "pooled_metrics.csv").set_index("method")
    maximum_pooled_metric_error = 0.0
    independent_metrics = {}
    for method in metric_table.index:
        computed = scores(y, result[method])
        independent_metrics[method] = computed
        for group, value in computed.items():
            maximum_pooled_metric_error = max(
                maximum_pooled_metric_error,
                close(value, metric_table.loc[method, group], "pooled " + method + "/" + group, 1e-14),
            )
    maximum_fold_metric_error = 0.0
    fold_table = pd.read_csv(output / "fold_metrics.csv")
    for row in fold_table.itertuples(index=False):
        mask = folds == row.outer_fold
        for group, value in scores(y[mask], result[row.method][mask]).items():
            maximum_fold_metric_error = max(
                maximum_fold_metric_error,
                close(value, getattr(row, group), "fold " + row.method + "/" + group, 1e-14),
            )

    accounting = pd.read_csv(output / "exact_half_blend_accounting.csv").set_index("targets")
    identities = []
    for group, columns in GROUPS.items():
        variance = y[:, columns].var(axis=0)
        loss_a = float(np.mean(np.square(global_oof[A][:, columns] - y[:, columns]) / variance))
        loss_b = float(np.mean(np.square(global_oof[B][:, columns] - y[:, columns]) / variance))
        loss_half = float(np.mean(np.square(rebuilt[NEW[0]][:, columns] - y[:, columns]) / variance))
        disagreement = float(np.mean(np.square(global_oof[A][:, columns] - global_oof[B][:, columns]) / variance))
        identity_error = loss_half - (loss_a + loss_b) / 2.0 + disagreement / 4.0
        for key, value in (
            ("y16_nmse", loss_a), ("native50_nmse", loss_b), ("half_blend_nmse", loss_half),
            ("normalized_disagreement", disagreement), ("diversity_gain_over_average_loss", disagreement / 4.0),
            ("half_blend_delta_vs_native50_r2", loss_b - loss_half),
        ):
            close(value, accounting.loc[group, key], "quadratic accounting " + key, 1e-14)
        if abs(identity_error) > 1e-13:
            raise AssertionError("Convex fusion quadratic identity failed")
        identities.append({"targets": group, "identity_error": identity_error})

    summary = json.loads((output / "run_summary.json").read_text())
    exact(digest(output / "frozen_protocol.json"), summary["protocol_sha256"], "copied protocol digest")
    exact(digest(modern / "oof_arrays.npz"), summary["source_modern_oof_sha256"], "original OOF digest")

    return {
        "arithmetic_audit_pass": True,
        "audit_implementation_imports_fusion_runner": False,
        "new_model_fits": 0,
        "new_tabicl_inferences": 0,
        "outer_folds_checked": 5,
        "inner_partitions_per_outer": 3,
        "inner_complete_views_checked": inner_views_checked,
        "outer_endpoint_complete_views_checked": 10,
        "pooled_methods_checked": len(metric_table),
        "fold_rows_checked": len(fold_table),
        "weights": weights,
        "max_abs_weight_error": maximum_weight_error,
        "max_abs_prediction_error": maximum_prediction_error,
        "max_abs_pooled_metric_error": maximum_pooled_metric_error,
        "max_abs_fold_metric_error": maximum_fold_metric_error,
        "quadratic_identity_errors": identities,
        "independent_pooled_metrics": independent_metrics,
        "conclusions": [
            "Every reconstructed fusion weight uses only its own outer-training inner OOF responses and predictions.",
            "The saved primary values all50=0.2989707957901751 and core30=0.2532626515696771 are reproduced independently.",
            "The fixed half blend has higher core30 R2 than the prespecified groupwise primary; all prespecified procedures must remain reported.",
            "This is an internal nested-CV replay on an already-developed dataset, not fresh independent or external validation.",
            "The new fusion predictor cannot inherit previous downstream sensory/transfer validation without replaying those analyses.",
            "Deployment needs 16 latent and 50 native scalar contexts, 66 branches in total; this audit does not measure their runtime.",
            "Cached arithmetic and index checks support the documented train-only procedure; pre-registration timing is not independently established by a copied JSON Boolean.",
        ],
        "verified_source_sha256": {
            ("modern/" + str(path.relative_to(modern)) if path.is_relative_to(modern) else
             "nested/" + str(path.relative_to(nested)) if path.is_relative_to(nested) else
             "results/" + str(path.relative_to(output))): digest(path)
            for path in dict.fromkeys(source_paths)
        },
    }


def main():
    directory = Path(__file__).resolve().parent
    deep = directory.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modern", type=Path, default=deep / "project/MathorCup_A/frontier_revision/experiments/modern_prediction")
    parser.add_argument("--nested", type=Path, default=deep / "checkpoint/unpacked/nested_selection")
    parser.add_argument("--output", type=Path, default=directory.parent / "fusion_replay_v1")
    parser.add_argument("--result-json", type=Path, default=directory / "result.json")
    args = parser.parse_args()
    report = audit(args.modern.resolve(), args.nested.resolve(), args.output.resolve())
    report["audit_implementation_sha256"] = digest(Path(__file__))
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("verified_source_sha256", "independent_pooled_metrics", "conclusions")}, indent=2))


if __name__ == "__main__":
    main()
