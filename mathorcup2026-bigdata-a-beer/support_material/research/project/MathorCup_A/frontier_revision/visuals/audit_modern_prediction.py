#!/usr/bin/env python3
"""Independently recompute saved modern-model metrics and paired uncertainty.

No experiment scoring, fitting, or resampling function is imported. This audit
uses weighted sufficient statistics for the bootstrap and direct training-fold
quantiles for relative grades. Incomplete registered runs are explicitly marked.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
for thread_option in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_option] = "1"
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
UP = HERE.parent
EXPECTED = {
    "rank_pca64_y16_no_residual", "rank_pca64_y16_ridge_residual",
    "raw231_y16_no_residual", "raw231_y16_ridge_residual", "raw231_y50_native",
}
TABDPT = "tabdpt13_raw231_pca128_y50"
SEED = 20261023


def main(args):
    project = args.project_dir
    if project is None:
        candidate = UP / "project/MathorCup_A"
        project = candidate if (candidate / "data/A/1.csv").is_file() else next(
            (parent for parent in HERE.parents if (parent / "data/A/1.csv").is_file()), None)
    if project is None:
        raise FileNotFoundError("Supply --project-dir with the original competition attachments.")
    experiment = args.modern_dir
    sources = {}

    def record(path):
        path = Path(path)
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def arrays(path):
        with np.load(record(path), allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    raw_x = pd.read_csv(record(project / "data/A/1.csv")).sort_values("beer")
    raw_y = pd.read_csv(record(project / "data/A/4.csv")).set_index("beer").loc[raw_x.beer]
    targets = raw_y.columns[2:].to_numpy(str)
    Y = raw_y[targets].to_numpy(float)
    old = arrays(project / "A_work/q1/oof_arrays.npz")
    fold_table = pd.read_csv(record(project / "A_work/q1/fold_assignments.csv")).set_index("beer")
    folds = fold_table.loc[raw_x.beer, "fold"].to_numpy()
    data = arrays(experiment / "oof_arrays.npz")
    np.testing.assert_array_equal(data["Y"], Y)
    np.testing.assert_array_equal(data["Y"], old["Y"])
    np.testing.assert_array_equal(data["beer"], raw_x.beer.to_numpy())
    np.testing.assert_array_equal(data["targets"], targets)
    np.testing.assert_array_equal(data["fold"], folds)
    np.testing.assert_array_equal(data["old_nested_ensemble"], old["nested_ensemble"])
    assert Y.shape == (250, 50) and all(np.sum(folds == k) == 50 for k in range(5))
    core = np.array([target.startswith(("A_", "F_")) for target in targets])
    np.testing.assert_array_equal(np.flatnonzero(core), np.arange(30))
    metadata = {"Y", "beer", "targets", "fold"}
    predictions = {key: value for key, value in data.items() if key not in metadata}
    metrics = pd.read_csv(record(experiment / "model_comparison.csv")).set_index("model")
    dimension = pd.read_csv(record(experiment / "dimension_metrics.csv")).set_index(["model", "target"])
    den = ((Y - Y.mean(0)) ** 2).sum(0)
    ref = old["nested_ensemble"]
    ref_sse = (Y - ref) ** 2
    ref_r2 = 1 - ref_sse.sum(0) / den
    metric_error = 0.
    scores = {}

    def check_score(name, prediction, row):
        nonlocal metric_error
        assert prediction.shape == Y.shape and np.isfinite(prediction).all()
        squared = (Y - prediction) ** 2
        r2 = 1 - squared.sum(0) / den
        observed = np.array([r2.mean(), r2[core].mean(), np.sqrt(squared.mean(0)).mean()])
        reported = row[["macro_r2", "core30_r2", "macro_rmse"]].to_numpy(float)
        error = float(abs(observed - reported).max())
        assert error < 1e-12
        metric_error = max(metric_error, error)
        scores[name] = dict(all50_r2=float(r2.mean()), core30_r2=float(r2[core].mean()),
                            macro_rmse=float(observed[2]))
        return r2

    for name, prediction in predictions.items():
        r2 = check_score(name, prediction, metrics.loc[name])
        stated = dimension.loc[name].loc[targets, "r2"].to_numpy(float)
        error = float(abs(r2 - stated).max())
        assert error < 1e-12
        metric_error = max(metric_error, error)

    ci_path = experiment / "paired_bootstrap.csv"
    canonical_ci = ci_path.is_file()
    if not canonical_ci:
        ci_path = experiment / "diagnostics/paired_bootstrap.csv"
    ci_table = pd.read_csv(record(ci_path)).set_index(["model", "targets"])
    ci_error = 0.
    ci_records = []

    def bootstrap_check(name, prediction, rng, table):
        nonlocal ci_error
        sampled = rng.integers(0, len(Y), (2000, len(Y)))
        weights = np.zeros((len(sampled), len(Y)), dtype=float)
        np.add.at(weights, (np.arange(len(sampled))[:, None], sampled), 1)
        total = weights @ Y
        variance_sum = weights @ (Y * Y) - total * total / len(Y)
        assert (variance_sum > 0).all()
        per_target = weights @ (ref_sse - (Y - prediction) ** 2) / variance_sum
        samples = np.column_stack([per_target.mean(1), per_target[:, core].mean(1)])
        intervals = np.quantile(samples, [.025, .975], axis=0)
        point = ref_sse.sum(0) / den - ((Y - prediction) ** 2).sum(0) / den
        point = np.array([point.mean(), point[core].mean()])
        for i, group in enumerate(["all50", "core30"]):
            row = table.loc[(name, group)]
            assert row.reference == "old_nested_ensemble" and row.repetitions == 2000
            observed = np.r_[point[i], intervals[:, i]]
            reported = row[["delta_r2", "ci025", "ci975"]].to_numpy(float)
            error = float(abs(observed - reported).max())
            assert error < 1e-12
            ci_error = max(ci_error, error)
            ci_records.append(dict(model=name, group=group, delta=float(point[i]),
                                   ci95=intervals[:, i].tolist(), max_error=error))

    rng = np.random.default_rng(SEED + 2026)
    for name, prediction in predictions.items():
        if not name.startswith("old_"):
            bootstrap_check(name, prediction, rng, ci_table)

    dpt_dir = experiment / "tabdpt"
    dpt_complete = all((dpt_dir / name).is_file() for name in
                       ["model_comparison.csv", "oof_arrays.npz", "paired_bootstrap.csv"])
    if dpt_complete:
        dpt = arrays(dpt_dir / "oof_arrays.npz")
        for key in metadata:
            np.testing.assert_array_equal(dpt[key], data[key])
        dpt_metrics = pd.read_csv(record(dpt_dir / "model_comparison.csv")).set_index("model")
        assert dpt_metrics.index.tolist() == [TABDPT]
        check_score(TABDPT, dpt["prediction"], dpt_metrics.loc[TABDPT])
        dpt_ci = pd.read_csv(record(dpt_dir / "paired_bootstrap.csv")).set_index(["model", "targets"])
        bootstrap_check(TABDPT, dpt["prediction"], np.random.default_rng(SEED + 2026), dpt_ci)
        predictions[TABDPT] = dpt["prediction"]

    # Independently reconstruct training-fold relative grades. Each test beer
    # is assigned using only its own outer training fold's quantile boundaries.
    grade_dir = experiment / "diagnostics"
    grade_metric_error = 0.
    grade_rows = 0
    graded_models = set()
    if (grade_dir / "relative_grade_metrics.csv").is_file():
        cuts = np.stack([np.quantile(Y[folds != f], np.arange(1, 7) / 7, axis=0)
                         for f in range(5)])

        def grades(values):
            out = np.empty(values.shape, int)
            for f in range(5):
                rows = folds == f
                for j in range(values.shape[1]):
                    out[rows, j] = np.sum(values[rows, j, None] >= cuts[f, :, j], axis=1)
            return out

        truth = grades(Y)
        stated = pd.read_csv(record(grade_dir / "oof_truth_relative_grades_0_6.csv"))
        np.testing.assert_array_equal(stated[targets].to_numpy(), truth)
        grade_metrics = pd.read_csv(record(grade_dir / "relative_grade_metrics.csv"))
        for name, prediction in predictions.items():
            available = grade_metrics[grade_metrics.model == name].set_index("targets")
            if available.empty:
                continue
            graded_models.add(name)
            grade = grades(prediction)
            for group, mask in [("all50", np.ones(50, bool)), ("core30", core)]:
                residual = abs(grade[:, mask] - truth[:, mask])
                observed = np.array([residual.mean(), (residual == 0).mean(), (residual <= 1).mean()])
                reported = available.loc[group, ["grade_mae", "grade_exact", "grade_within_one"]].to_numpy(float)
                error = float(abs(observed - reported).max())
                assert error < 1e-12
                grade_metric_error = max(grade_metric_error, error)
                grade_rows += 1

    missing = sorted(EXPECTED - set(predictions)) + ([] if dpt_complete else [TABDPT])
    missing_grades = sorted((EXPECTED | {TABDPT}) - graded_models)
    report = dict(all_registered_complete=not missing and canonical_ci and not missing_grades,
                  missing_models=missing, missing_relative_grade_models=missing_grades,
                  canonical_tabicl_intervals=canonical_ci, same_raw_targets_and_outer_folds=True,
                  frozen_reference_identical=True, n_beers=250, n_targets=50, n_core=30,
                  max_metric_error=metric_error, max_paired_interval_error=ci_error,
                  max_relative_grade_metric_error=grade_metric_error, relative_grade_rows_checked=grade_rows,
                  scores=scores, paired_intervals=ci_records,
                  bootstrap_repetitions=2000, bootstrap_seed=SEED + 2026,
                  uncertainty_scope="paired fixed-OOF beer resampling with resampled Y variance; no refit or multiplicity correction",
                  source_sha256=sources)
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ["scores", "paired_intervals", "source_sha256"]}, ensure_ascii=False, indent=2))
    if args.require_complete and not report["all_registered_complete"]:
        raise SystemExit("Registered modern-model comparison is not yet complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path)
    parser.add_argument("--modern-dir", type=Path, default=UP / "experiments/modern_prediction")
    parser.add_argument("--output", type=Path, default=HERE / "qa/modern_prediction_numeric_audit.json")
    parser.add_argument("--require-complete", action="store_true")
    main(parser.parse_args())
