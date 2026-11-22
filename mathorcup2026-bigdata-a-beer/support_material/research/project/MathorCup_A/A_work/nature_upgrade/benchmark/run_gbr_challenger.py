# Team review and independent work must be documented truthfully.
"""Nested classical gradient-boosting benchmark for MathorCup A.

The published 648-setting grid is NOT claimed to have been rerun. This study
uses 6 prespecified structures and 4 shared boosting-iteration checkpoints.
All imputers are fitted only on the respective training partition. Both the
original project outer folds and the author's 175/75 stratified split are kept.
"""
import os
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import DATA_HELP, resolve_data_dir

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

SEED = 20261023
CHECKPOINTS = [50, 100, 200, 300]
STRUCTURES = [
    dict(name="stumps", learning_rate=.04, max_depth=1, subsample=1., max_features=1., min_samples_leaf=5),
    dict(name="shallow", learning_rate=.04, max_depth=2, subsample=.8, max_features=.7, min_samples_leaf=5),
    dict(name="depth3", learning_rate=.04, max_depth=3, subsample=.8, max_features=.7, min_samples_leaf=5),
    dict(name="author_depth6", learning_rate=.04, max_depth=6, subsample=.6, max_features=1., min_samples_leaf=1),
    dict(name="author_depth9", learning_rate=.08, max_depth=9, subsample=.4, max_features=1., min_samples_leaf=1),
    dict(name="feature_subsample", learning_rate=.04, max_depth=2, subsample=.8, max_features=.35, min_samples_leaf=5),
]


def specifications():
    return [dict(s, n_estimators=n) for s in STRUCTURES for n in CHECKPOINTS]


def staged_fit_predict(X, y, xt, structure):
    pars = {k: v for k, v in structure.items() if k != "name"}
    model = GradientBoostingRegressor(**pars, n_estimators=max(CHECKPOINTS), random_state=SEED)
    model.fit(X, y)
    checkpoint_set = set(CHECKPOINTS)
    return np.stack([p for n, p in enumerate(model.staged_predict(xt), 1) if n in checkpoint_set])


def task_for_target_and_structure(j, sidx, datasets, full, Ytrain):
    structure = STRUCTURES[sidx]
    inner = np.empty((len(CHECKPOINTS), len(Ytrain)))
    for train_idx, val_idx, xt, xv in datasets:
        inner[:, val_idx] = staged_fit_predict(xt, Ytrain[train_idx, j], xv, structure)
    outer = staged_fit_predict(full[0], Ytrain[:, j], full[1], structure)
    return j, sidx, inner, outer


def prepare_datasets(X, Xt, inner_splits):
    datasets = []
    for tr, va in inner_splits:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        a = imputer.fit_transform(X[tr])
        b = imputer.transform(X[va])
        datasets.append((tr, va, a, b))
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    full = (imputer.fit_transform(X), imputer.transform(Xt))
    return datasets, full


def selected_predictions(inner, test, Y, inner_splits):
    # For original nested folds, this is pooled inner normalized MSE. No outer
    # labels enter any model selection or weighting decision.
    sq = (inner - Y[None, :, :]) ** 2
    scale = np.maximum(np.std(Y, axis=0, ddof=1), 1e-10)
    nmse = sq.mean(axis=1) / scale[None, :] ** 2
    avg = nmse.mean(axis=1)
    global_idx = int(np.argmin(avg))
    individual_idx = np.argmin(nmse, axis=0)
    shrink_idx = np.argmin(.5 * nmse + .5 * avg[:, None], axis=0)
    # Author's sklearn GridSearchCV(scoring='r2', cv=5) averages the fold-level
    # uniform-output R2 values, rather than scoring pooled OOF predictions.
    cv_r2 = np.array([np.mean([r2_score(Y[v], p[v], multioutput="uniform_average")
                              for _, v in inner_splits]) for p in inner])
    author_idx = int(np.argmax(cv_r2))
    jj = np.arange(Y.shape[1])
    preds = dict(
        gbr_global=test[global_idx],
        gbr_targetwise=test[individual_idx, :, jj].T,
        gbr_shrunk_selection=test[shrink_idx, :, jj].T,
        gbr_cv_fold_r2=test[author_idx],
    )
    inner_preds = dict(
        gbr_global=inner[global_idx],
        gbr_targetwise=inner[individual_idx, :, jj].T,
        gbr_shrunk_selection=inner[shrink_idx, :, jj].T,
        gbr_cv_fold_r2=inner[author_idx],
    )
    selection = dict(global_index=global_idx, targetwise_indices=individual_idx.tolist(),
                     shrink_indices=shrink_idx.tolist(), fold_r2_index=author_idx,
                     candidate_normalized_mse=nmse.tolist(), candidate_fold_r2=cv_r2.tolist())
    return preds, inner_preds, selection


def assess(Y, P, targets):
    r = r2_score(Y, P, multioutput="raw_values")
    return dict(macro_r2=float(r.mean()), core30_r2=float(r[:30].mean()),
                macro_rmse=float(np.sqrt(np.mean((Y-P)**2, axis=0)).mean()),
                macro_spearman=float(np.nanmean([spearmanr(Y[:, j], P[:, j]).statistic
                                                for j in range(Y.shape[1])])),
                positive_r2=int((r > 0).sum()), target_r2=dict(zip(targets, r.tolist())))


def run_partition(tag, tr, te, X, Y, meta, inner_splits, out, jobs):
    path = out / f"{tag}_candidates.npz"
    selection_path = out / f"{tag}_selection.json"
    start = time.perf_counter()
    prior_fit_seconds = None
    if path.exists():
        if selection_path.exists():
            old_selection = json.loads(selection_path.read_text())
            prior_fit_seconds = old_selection.get("fit_elapsed_seconds", old_selection.get("elapsed_seconds"))
        stored = np.load(path)
        assert np.array_equal(tr, stored["train_indices"])
        assert np.array_equal(te, stored["test_indices"])
        inner, test = stored["inner_predictions"], stored["test_predictions"]
        print(f"{tag}: loaded candidate checkpoint", flush=True)
    else:
        datasets, full = prepare_datasets(X[tr], X[te], inner_splits)
        inner = np.empty((len(specifications()), len(tr), Y.shape[1]))
        test = np.empty((len(specifications()), len(te), Y.shape[1]))
        task_list = [delayed(task_for_target_and_structure)(j, s, datasets, full, Y[tr])
                     for s in range(len(STRUCTURES)) for j in range(Y.shape[1])]
        iterator = Parallel(n_jobs=jobs, return_as="generator_unordered", verbose=0)(task_list)
        for count, (j, s, ip, tp) in enumerate(iterator, 1):
            ii = slice(s * len(CHECKPOINTS), (s+1) * len(CHECKPOINTS))
            inner[ii, :, j], test[ii, :, j] = ip, tp
            if count % 25 == 0:
                print(f"{tag}: {count}/{len(task_list)} target-structure tasks, "
                      f"{time.perf_counter()-start:.1f}s", flush=True)
        inner_fold = np.empty(len(tr), dtype=int)
        for fi, (_, va) in enumerate(inner_splits):
            inner_fold[va] = fi
        np.savez_compressed(path, train_indices=tr, test_indices=te,
                            inner_predictions=inner, test_predictions=test,
                            inner_fold=inner_fold, y_train=Y[tr], y_test=Y[te])
    pred, inner_pred, selection = selected_predictions(inner, test, Y[tr], inner_splits)
    selection["tag"] = tag
    selection["elapsed_seconds"] = time.perf_counter()-start
    selection["fit_elapsed_seconds"] = prior_fit_seconds if prior_fit_seconds is not None else selection["elapsed_seconds"]
    selection["candidate_specs"] = specifications()
    selection_path.write_text(json.dumps(selection, indent=2))
    np.savez_compressed(out / f"{tag}_selected.npz", train_indices=tr, test_indices=te,
                        **pred, **{f"inner_{k}": v for k, v in inner_pred.items()})
    for key, pp in pred.items():
        frame = pd.concat([meta.iloc[te].reset_index(drop=True),
                           pd.DataFrame(pp, columns=meta.attrs["targets"])], axis=1)
        frame.to_csv(out / f"{tag}_{key}_predictions.csv", index=False)
    print(f"{tag}: completed {time.perf_counter()-start:.1f}s", flush=True)
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(resolve_data_dir()), help=DATA_HELP)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--partitions", default="0,1,2,3,4,author")
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    unsorted = pd.read_csv(base / "1.csv")
    x = unsorted.sort_values("beer").reset_index(drop=True)
    y = pd.read_csv(base / "4.csv").set_index("beer").loc[x.beer].reset_index()
    features, targets = x.columns[3:].tolist(), y.columns[3:].tolist()
    X, Y = x[features].to_numpy(float), y[targets].to_numpy(float)
    meta = x.iloc[:, :3].copy(); meta.attrs["targets"] = targets
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(X))
    original_to_sorted = pd.Series(np.arange(len(x)), index=x.beer)
    at, av = train_test_split(np.arange(len(unsorted)), test_size=.30, random_state=0,
                              shuffle=True, stratify=unsorted.tasting_category_fine)
    author_train = original_to_sorted.loc[unsorted.iloc[at].beer].to_numpy()
    author_test = original_to_sorted.loc[unsorted.iloc[av].beer].to_numpy()
    design = dict(seed=SEED, n=len(X), p=len(features), targets=targets, features=features,
                  candidate_specs=specifications(), jobs=args.jobs,
                  original_author_grid_size=648, actual_candidate_count=len(specifications()),
                  imputation="Training-partition median only; author notebook global mean imputation not used.",
                  author_split="Original input row order, stratified test_size=.30, random_state=0; inner KFold(5, shuffle=False)",
                  outer_split="Beer-sorted input, KFold(5, shuffle=True, random_state=20261023); inner KFold(3, shuffle=True, random_state=SEED+fold)")
    (out / "design.json").write_text(json.dumps(design, indent=2))
    results = {}
    for part in args.partitions.split(","):
        if part == "author":
            tr, te = author_train, author_test
            inner = list(KFold(5, shuffle=False).split(tr))
            tag = "author_holdout"
        else:
            k = int(part); tr, te = folds[k]
            inner = list(KFold(3, shuffle=True, random_state=SEED+k).split(tr))
            tag = f"outer_{k}"
        pred = run_partition(tag, tr, te, X, Y, meta, inner, out, args.jobs)
        results[tag] = {key: assess(Y[te], pp, targets) for key, pp in pred.items()}
        (out / "partition_metrics.json").write_text(json.dumps(results, indent=2))
    if all((out / f"outer_{k}_selected.npz").exists() for k in range(5)):
        predictions = {key: np.empty_like(Y) for key in ["gbr_global", "gbr_targetwise", "gbr_shrunk_selection", "gbr_cv_fold_r2"]}
        for k in range(5):
            a = np.load(out / f"outer_{k}_selected.npz")
            for key in predictions:
                predictions[key][a["test_indices"]] = a[key]
        scores = {key: assess(Y, p, targets) for key, p in predictions.items()}
        np.savez_compressed(out / "gbr_oof_predictions.npz", Y=Y, **predictions)
        for key, pp in predictions.items():
            pd.concat([meta, pd.DataFrame(pp, columns=targets)], axis=1).to_csv(out / f"{key}_oof.csv", index=False)
        (out / "oof_metrics.json").write_text(json.dumps(scores, indent=2))
        print(json.dumps({key: {k: v for k, v in val.items() if k != "target_r2"} for key, val in scores.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
