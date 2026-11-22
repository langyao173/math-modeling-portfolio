# Team review and independent work must be documented truthfully.
"""Four predeclared CatBoost MultiRMSE candidates, nested and author-split.

Selection uses one shared hyperparameter setting across all 50 outputs.
The complete candidate OOF tensors are retained for cross-family ensembles.
No held-out label is used for fitting, imputation, scaling, or selection.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import hashlib
import json
import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA_HELP, resolve_data_dir

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from catboost_model import CatboostMultiTarget, MODEL_VERSION, FIXED, SEED
import catboost

CHECKPOINTS = [100, 300]


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(Y, P):
    r = r2_score(Y, P, multioutput="raw_values")
    s = [spearmanr(Y[:, j], P[:, j]).statistic for j in range(Y.shape[1])]
    return dict(macro_r2=float(r.mean()), core30_r2=float(r[:30].mean()),
                mean_rmse=float(np.sqrt(np.mean((Y-P)**2, axis=0)).mean()),
                macro_spearman=float(np.nanmean(s)), positive_r2=int((r > 0).sum()))


def validate_cache(a, fingerprint, tr, te, fold_id):
    assert str(a["fingerprint"]) == fingerprint, "Stale experiment cache"
    assert np.array_equal(a["train_indices"], tr), "Training order mismatch"
    assert np.array_equal(a["test_indices"], te), "Test order mismatch"
    assert np.array_equal(a["inner_fold"], fold_id), "Inner fold mismatch"


def run_partition(tag, tr, te, cv, X, Y, meta, specs, depths, out,
                  fingerprint, features, targets, save_models=False):
    start = time.perf_counter()
    fold_id = np.empty(len(tr), dtype=int)
    for fi, (_, va) in enumerate(cv):
        fold_id[va] = fi
    common = dict(algorithm_version=MODEL_VERSION, fingerprint=fingerprint,
                  train_indices=tr, test_indices=te, inner_fold=fold_id,
                  features=np.asarray(features), targets=np.asarray(targets),
                  train_beer=meta.iloc[tr].beer.to_numpy(),
                  test_beer=meta.iloc[te].beer.to_numpy())
    dest = out / f"{tag}_candidates.npz"
    if dest.exists():
        a = np.load(dest)
        validate_cache(a, fingerprint, tr, te, fold_id)
        inner, test = a["inner_predictions"], a["test_predictions"]
        print(tag, "loaded completed candidate checkpoint", flush=True)
    else:
        inner = np.empty((len(specs), len(tr), Y.shape[1]), dtype=float)
        test = np.empty((len(specs), len(te), Y.shape[1]), dtype=float)
        for d, depth in enumerate(depths):
            checkpoint = out / f"{tag}_depth{depth}_partial.npz"
            if checkpoint.exists():
                c = np.load(checkpoint)
                validate_cache(c, fingerprint, tr, te, fold_id)
                ip, tp, done = c["inner_predictions"], c["test_predictions"], c["done"]
            else:
                ip = np.full((2, len(tr), Y.shape[1]), np.nan)
                tp = np.full((2, len(te), Y.shape[1]), np.nan)
                done = np.zeros(len(cv)+1, dtype=bool)
            for fi, (a, b) in enumerate(cv):
                if done[fi]:
                    continue
                t0 = time.perf_counter()
                model = CatboostMultiTarget(depth, max(CHECKPOINTS)).fit(X[tr[a]], Y[tr[a]])
                for ci, trees in enumerate(CHECKPOINTS):
                    ip[ci, b] = model.predict(X[tr[b]], trees)
                done[fi] = True
                write_npz(checkpoint, **common, inner_predictions=ip,
                          test_predictions=tp, done=done)
                print(tag, "depth", depth, "inner", fi+1, "/", len(cv),
                      "seconds", round(time.perf_counter()-t0, 2), flush=True)
            model_path = out / f"full_depth{depth}.cbm"
            if not done[-1] or (save_models and not model_path.exists()):
                t0 = time.perf_counter()
                model = CatboostMultiTarget(depth, max(CHECKPOINTS)).fit(X[tr], Y[tr])
                for ci, trees in enumerate(CHECKPOINTS):
                    tp[ci] = model.predict(X[te], trees)
                if save_models:
                    model.save_native(out, f"full_depth{depth}", features, targets)
                    loaded = CatboostMultiTarget.load_native(out, f"full_depth{depth}")
                    expected = model.predict(X[tr[:7]], 100)
                    actual = loaded.predict(X[tr[:7]], 100)
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
                done[-1] = True
                write_npz(checkpoint, **common, inner_predictions=ip,
                          test_predictions=tp, done=done)
                print(tag, "depth", depth, "full refit seconds",
                      round(time.perf_counter()-t0, 2), flush=True)
            assert np.isfinite(ip).all() and np.isfinite(tp).all()
            inner[2*d:2*d+2], test[2*d:2*d+2] = ip, tp
        write_npz(dest, **common, inner_predictions=inner,
                  test_predictions=test, y_train=Y[tr], y_test=Y[te])
    assert inner.shape == (4, len(tr), 50) and test.shape == (4, len(te), 50)
    assert np.isfinite(inner).all() and np.isfinite(test).all()
    # All candidate selection and the target normalization used by this score
    # see only the partition's training labels, never its held-out labels.
    scale = np.maximum(Y[tr].std(axis=0, ddof=1), 1e-8)
    loss = np.mean(((inner-Y[tr][None, :, :])/scale[None, None, :])**2, axis=1)
    global_index = int(np.argmin(loss.mean(axis=1)))
    P, I = test[global_index], inner[global_index]
    selection = dict(tag=tag, global_index=global_index,
                     selected_spec=specs[global_index], candidate_specs=specs,
                     candidate_normalized_mse=loss.tolist(),
                     candidate_macro_normalized_mse=loss.mean(axis=1).tolist(),
                     elapsed_seconds=time.perf_counter()-start)
    # The final full-data branch's "test" tensor is deployment predictions on
    # training rows, not an unbiased performance estimate. Do not score it.
    if not save_models:
        selection["test_metrics"] = assess(Y[te], P)
    else:
        selected = specs[global_index]
        model = CatboostMultiTarget.load_native(out, f"full_depth{selected['depth']}")
        model.save_native(out, "final_global_model", features, targets,
                          selected_trees=selected["n_estimators"])
    write_json(out / f"{tag}_selection.json", selection)
    write_npz(out / f"{tag}_selected.npz", **common,
              catboost_global=P, inner_catboost_global=I)
    if not save_models:
        pd.concat([meta.iloc[te].reset_index(drop=True),
                   pd.DataFrame(P, columns=targets)], axis=1).to_csv(
                       out / f"{tag}_catboost_global_predictions.csv", index=False)
    print(tag, "completed", round(time.perf_counter()-start, 1), "seconds", flush=True)
    return P


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(resolve_data_dir()), help=DATA_HELP)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--partitions", default="0,1,2,3,4,author,final")
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    timing = json.loads((out / "reduced_timing_result.json").read_text())
    depths = timing["selected_depths"]
    assert depths in ([3, 5], [2, 4])
    specs = [dict(family="catboost_multirmse", depth=d, n_estimators=n,
                  learning_rate=FIXED["learning_rate"], l2_leaf_reg=FIXED["l2_leaf_reg"],
                  border_count=FIXED["border_count"]) for d in depths for n in CHECKPOINTS]
    original = pd.read_csv(data / "1.csv")
    x = original.sort_values("beer").reset_index(drop=True)
    y = pd.read_csv(data / "4.csv").set_index("beer").loc[x.beer]
    features, targets = x.columns[3:].tolist(), y.columns[2:].tolist()
    X, Y = x[features].to_numpy(float), y[targets].to_numpy(float)
    assert X.shape == (250, 231) and Y.shape == (250, 50)
    assert x.beer.is_unique and np.isfinite(Y).all()
    meta = x.iloc[:, :3].copy()
    design = dict(algorithm_version=MODEL_VERSION, package_version=catboost.__version__,
                  seed=SEED, specs=specs, fixed_parameters=FIXED,
                  features=features, targets=targets,
                  input_sha256={f: sha_file(data/f) for f in ["1.csv", "4.csv"]},
                  runtime_grid_decision=timing,
                  imputation="Median fit on current training fold; keep all 231 columns.",
                  output_scaling="Per-output mean and sample standard deviation fit only on current training fold.",
                  selection="One shared candidate by mean pooled inner-OOF normalized MSE across 50 outputs.",
                  outer="Beer-sorted KFold(5, shuffle=True, random_state=20261023); inner KFold(3, shuffle=True, random_state=SEED+fold).",
                  author="Original-row-order 175/75 split, stratify 22 style, random_state=0; inner KFold(5, shuffle=False).",
                  final="All 250 rows, inner KFold(3, shuffle=True, random_state=SEED+99).",
                  treatment_of_test="No evaluation set/early stopping; held-out labels used only after training and inner selection.",
                  interpretation="Retrospective methodological comparison on an already explored dataset, not a new blind external test.",
                  official_sources=[
                      "https://catboost.ai/docs/en/concepts/loss-functions-multiregression",
                      "https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_staged_predict",
                      "https://catboost.ai/docs/en/references/training-parameters/common"])
    fingerprint = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    design["fingerprint"] = fingerprint
    design_path = out / "experiment_spec.json"
    if design_path.exists():
        assert json.loads(design_path.read_text())["fingerprint"] == fingerprint, "Experiment definition changed"
    else:
        write_json(design_path, design)
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(X))
    position = pd.Series(np.arange(len(x)), index=x.beer)
    tr0, te0 = train_test_split(np.arange(len(original)), test_size=.3,
                               random_state=0, shuffle=True,
                               stratify=original.tasting_category_fine)
    author_train = position.loc[original.iloc[tr0].beer].to_numpy()
    author_test = position.loc[original.iloc[te0].beer].to_numpy()
    started = time.perf_counter()
    for part in args.partitions.split(","):
        if part == "author":
            tr, te = author_train, author_test
            cv = list(KFold(5, shuffle=False).split(tr))
            tag = "author_holdout"
        elif part == "final":
            tr = te = np.arange(len(X))
            cv = list(KFold(3, shuffle=True, random_state=SEED+99).split(tr))
            tag = "full_data"
        else:
            k = int(part); tr, te = folds[k]
            cv = list(KFold(3, shuffle=True, random_state=SEED+k).split(tr))
            tag = f"outer_{k}"
        run_partition(tag, tr, te, cv, X, Y, meta, specs, depths, out,
                      fingerprint, features, targets, save_models=part == "final")
    if all((out / f"outer_{k}_selected.npz").exists() for k in range(5)):
        oof = np.full_like(Y, np.nan)
        outer_fold = np.empty(len(Y), dtype=int)
        for k in range(5):
            a = np.load(out / f"outer_{k}_selected.npz")
            oof[a["test_indices"]] = a["catboost_global"]
            outer_fold[a["test_indices"]] = k
        assert np.isfinite(oof).all()
        write_npz(out / "oof_arrays.npz", Y=Y, catboost_global=oof,
                  beer=meta.beer.to_numpy(), outer_fold=outer_fold, targets=np.asarray(targets))
        pd.concat([meta, pd.DataFrame(oof, columns=targets)], axis=1).to_csv(
            out / "catboost_global_oof.csv", index=False)
        scores = assess(Y, oof)
        write_json(out / "summary.json", dict(model="catboost_global", **scores,
                   elapsed_this_run_seconds=time.perf_counter()-started))
        pd.DataFrame([dict(model="catboost_global", **scores)]).to_csv(
            out / "model_comparison.csv", index=False)
        pd.DataFrame(dict(target=targets,
                         r2=r2_score(Y, oof, multioutput="raw_values"))).to_csv(
                             out / "dimension_metrics.csv", index=False)
        print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
