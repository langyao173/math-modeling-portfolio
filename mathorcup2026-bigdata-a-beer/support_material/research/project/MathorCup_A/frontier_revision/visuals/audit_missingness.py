#!/usr/bin/env python3
"""Independent matched-mask, metric and paired-interval audit for figure 05."""
from pathlib import Path
import argparse
import hashlib
import json
import os
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
UP = HERE.parent
SEED = 20261023
SCENARIOS = ["original", "derived_only_missing", "MCAR_10_strict", "MCAR_30_strict", "MCAR_50_strict"]
MODELS = ["old_median", "old_knn", "tabicl_raw231_y16"]


def main(args):
    experiment = args.experiment_dir
    status = experiment / "status.json"
    state = json.loads(status.read_text()) if status.is_file() else {}
    if not state.get("complete", False) or state.get("completed_folds") != 5:
        report = dict(complete=False, completed_folds=state.get("completed_folds", 0),
                      interpretation="Matched-mask experiment is incomplete; no partial score or interval is approved.")
        args.output.parent.mkdir(exist_ok=True, parents=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
        if args.require_complete:
            raise SystemExit("Matched-mask experiment must complete before the final visual audit.")
        return
    project = args.project_dir
    if project is None:
        default = UP / "project/MathorCup_A"
        project = default if (default / "data/A/1.csv").is_file() else next(
            (p for p in HERE.parents if (p / "data/A/1.csv").is_file()), None)
    if project is None:
        raise FileNotFoundError("Supply --project-dir with original competition attachments.")
    sources = {}

    def record(path):
        path = Path(path)
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def load(path):
        with np.load(record(path), allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    record(status)
    raw_x = pd.read_csv(record(project / "data/A/1.csv")).sort_values("beer")
    raw_y = pd.read_csv(record(project / "data/A/4.csv")).set_index("beer").loc[raw_x.beer]
    fields = raw_x.columns[3:].tolist();targets = raw_y.columns[2:].tolist()
    X = raw_x[fields].to_numpy(float);Y = raw_y[targets].to_numpy(float)
    fold_table = pd.read_csv(record(project / "A_work/q1/fold_assignments.csv")).set_index("beer")
    fold = fold_table.loc[raw_x.beer, "fold"].to_numpy()
    data = load(experiment / "oof_predictions.npz")
    for key, value in [("Y", Y), ("beer", raw_x.beer.to_numpy()), ("targets", targets), ("fold", fold)]:
        np.testing.assert_array_equal(data[key], value)
    assert X.shape == (250, 231) and Y.shape == (250, 50)
    core = np.array([t.startswith(("A_", "F_")) for t in targets])
    np.testing.assert_array_equal(np.flatnonzero(core), np.arange(30))
    derived = [fields.index(name) for name in ["kcalperc", "esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum"]]
    masks_checked = 0
    reconstructed_error = 0.
    for f in range(5):
        test = np.flatnonzero(fold == f);train = np.flatnonzero(fold != f)
        masks = load(experiment / "checkpoints" / f"fold{f}_masks.npz")
        np.testing.assert_array_equal(masks["test_indices"], test)
        np.testing.assert_array_equal(masks["train_indices"], train)
        for scenario in SCENARIOS[1:]:
            expected = np.isnan(X[test])
            if scenario.startswith("MCAR_"):
                level = int(scenario.split("_")[1])
                expected |= np.random.default_rng(SEED + 1000*f + level).random(expected.shape) < level/100
            expected[:, derived] = True
            np.testing.assert_array_equal(masks[scenario], expected)
            masks_checked += 1
        checkpoint = load(experiment / "checkpoints" / f"fold{f}_complete.npz")
        np.testing.assert_array_equal(checkpoint["test_indices"], test)
        np.testing.assert_array_equal(checkpoint["train_indices"], train)
        np.testing.assert_allclose(checkpoint["target_mean"], Y[train].mean(0), atol=1e-12, rtol=0)
        np.testing.assert_allclose(checkpoint["target_scale"], Y[train].std(0, ddof=1), atol=1e-12, rtol=0)
        for i, scenario in enumerate(SCENARIOS[1:]):
            rebuilt = (checkpoint["latent_prediction"][i] @ checkpoint["target_components"]
                       + checkpoint["target_pca_mean"]) * checkpoint["target_scale"] + checkpoint["target_mean"]
            error = float(abs(rebuilt - data[f"tabicl_raw231_y16__{scenario}"][test]).max())
            assert error < 1e-12
            reconstructed_error = max(reconstructed_error, error)
    den = ((Y - Y.mean(0))**2).sum(0)
    cut = np.stack([np.quantile(Y[fold != f], np.arange(1, 7)/7, axis=0) for f in range(5)])

    def grade(values):
        out = np.empty(values.shape, int)
        for f in range(5):
            mask = fold == f
            for j in range(50):
                out[mask, j] = np.sum(values[mask, j, None] >= cut[f, :, j], axis=1)
        return out

    truth = grade(Y)
    metrics = pd.read_csv(record(experiment / "comparison.csv")).set_index(["model", "scenario", "targets"])
    intervals = pd.read_csv(record(experiment / "paired_ci_vs_knn.csv")).set_index(["scenario", "targets", "metric"])
    sampled = np.random.default_rng(SEED + 79011).integers(0, len(Y), (2000, len(Y)))
    weights = np.zeros((len(sampled), len(Y)), float)
    np.add.at(weights, (np.arange(len(sampled))[:, None], sampled), 1)
    total = weights @ Y
    bootstrap_den = weights @ (Y * Y) - total * total / len(Y)
    assert (bootstrap_den > 0).all()
    score_error = 0.;interval_error = 0.;rows_checked = 0
    for scenario in SCENARIOS:
        residuals = {};r2 = {};sse = {}
        for model in MODELS:
            prediction = data[f"{model}__{scenario}"]
            assert prediction.shape == Y.shape and np.isfinite(prediction).all()
            sse[model] = (Y - prediction)**2
            r2[model] = 1 - sse[model].sum(0)/den
            residuals[model] = abs(grade(prediction) - truth)
            for group, mask in [("all50", np.ones(50, bool)), ("core30", core)]:
                e = residuals[model][:, mask]
                values = np.array([r2[model][mask].mean(), e.mean(), (e == 0).mean(), (e <= 1).mean()])
                stated = metrics.loc[(model, scenario, group), ["r2", "grade_mae", "grade_exact", "grade_within_one"]].to_numpy(float)
                error = float(abs(values - stated).max());assert error < 1e-12
                score_error = max(score_error, error);rows_checked += 1
        per_target = weights @ (sse["old_knn"] - sse["tabicl_raw231_y16"]) / bootstrap_den
        for group, mask in [("all50", np.ones(50, bool)), ("core30", core)]:
            difference = r2["tabicl_raw231_y16"][mask] - r2["old_knn"][mask]
            statistics = {"r2": (float(difference.mean()), per_target[:, mask].mean(1))}
            a = residuals["tabicl_raw231_y16"][:, mask];b = residuals["old_knn"][:, mask]
            for metric, delta in [("grade_mae", a.mean(1)-b.mean(1)),
                                  ("grade_exact", (a == 0).mean(1)-(b == 0).mean(1)),
                                  ("grade_within_one", (a <= 1).mean(1)-(b <= 1).mean(1))]:
                statistics[metric] = (float(delta.mean()), weights @ delta / len(Y))
            for metric, (point, samples) in statistics.items():
                row = intervals.loc[(scenario, group, metric)]
                assert row.reference == "old_knn"
                values = np.r_[point, np.quantile(samples, [.025, .975])]
                stated = row[["delta", "ci025", "ci975"]].to_numpy(float)
                error = float(abs(values - stated).max());assert error < 1e-12
                interval_error = max(interval_error, error)
    report = dict(complete=True, masks_checked=masks_checked, metric_rows_checked=rows_checked,
                  paired_intervals_checked=len(intervals), n_beers=250, target_dimensions=50,
                  max_prediction_reconstruction_error=reconstructed_error,
                  max_metric_error=score_error, max_paired_interval_error=interval_error,
                  identical_registered_masks=True, same_original_targets_and_folds=True,
                  bootstrap_repetitions=2000, bootstrap_seed=SEED+79011,
                  uncertainty_scope="fixed-OOF paired beer bootstrap; R2 denominator recomputed, grade cutpoints kept fixed",
                  source_sha256=sources)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key:value for key,value in report.items() if key != "source_sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path)
    parser.add_argument("--experiment-dir", type=Path, default=UP/"experiments/modern_prediction/missingness")
    parser.add_argument("--output", type=Path, default=HERE/"qa/missingness_numeric_audit.json")
    parser.add_argument("--require-complete", action="store_true")
    main(parser.parse_args())
