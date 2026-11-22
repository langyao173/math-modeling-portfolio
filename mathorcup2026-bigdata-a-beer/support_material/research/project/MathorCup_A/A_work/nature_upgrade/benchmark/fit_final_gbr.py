# Team review and independent work must be documented truthfully.
"""Full-data, train-fold-only 3-fold selection and deployable GBR bundles."""
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

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

from gbr_model import GBRTargetModel
from run_gbr_challenger import CHECKPOINTS, SEED, STRUCTURES, specifications, staged_fit_predict


def fit_inner(j, si, datasets, Y):
    pp = np.empty((len(CHECKPOINTS), len(Y)))
    for tr, te, a, b in datasets:
        pp[:, te] = staged_fit_predict(a, Y[tr, j], b, STRUCTURES[si])
    return j, si, pp


def fit_selected_target(j, global_idx, target_idx, specs, X, Y):
    fitted = {}
    for i in set([global_idx, target_idx]):
        pars = {k: v for k, v in specs[i].items() if k != "name"}
        fitted[i] = GradientBoostingRegressor(**pars, random_state=SEED).fit(X, Y[:, j])
    return j, fitted[global_idx], fitted[target_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(resolve_data_dir()), help=DATA_HELP)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    x = pd.read_csv(Path(args.data) / "1.csv").sort_values("beer").reset_index(drop=True)
    y = pd.read_csv(Path(args.data) / "4.csv").set_index("beer").loc[x.beer].reset_index()
    features, targets = x.columns[3:].tolist(), y.columns[3:].tolist()
    X, Y = x[features].to_numpy(float), y[targets].to_numpy(float)
    splits = list(KFold(3, shuffle=True, random_state=SEED+99).split(X))
    checkpoint = out / "full_data_candidates.npz"
    t0 = time.perf_counter()
    if checkpoint.exists():
        saved = np.load(checkpoint)
        assert np.array_equal(saved["Y"], Y)
        pp = saved["inner_predictions"]
    else:
        data = []
        inner_fold = np.empty(len(Y), dtype=int)
        for f, (tr, te) in enumerate(splits):
            imp = SimpleImputer(strategy="median", keep_empty_features=True)
            data.append((tr, te, imp.fit_transform(X[tr]), imp.transform(X[te])))
            inner_fold[te] = f
        pp = np.empty((len(specifications()), len(Y), Y.shape[1]))
        tasks = [delayed(fit_inner)(j, si, data, Y) for si in range(len(STRUCTURES)) for j in range(Y.shape[1])]
        iterator = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(tasks)
        for count, (j, si, pred) in enumerate(iterator, 1):
            pp[si*len(CHECKPOINTS):(si+1)*len(CHECKPOINTS), :, j] = pred
            if count % 25 == 0:
                print(f"full: {count}/{len(tasks)} tasks, {time.perf_counter()-t0:.1f}s", flush=True)
        np.savez_compressed(checkpoint, inner_predictions=pp, inner_fold=inner_fold, Y=Y)
    scale = np.maximum(np.std(Y, axis=0, ddof=1), 1e-10)
    nmse = np.mean((pp-Y[None, :, :])**2, axis=1) / scale[None, :]**2
    global_idx = int(np.argmin(nmse.mean(axis=1)))
    target_idx = np.argmin(nmse, axis=0)
    specs = specifications()
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    Z = imp.fit_transform(X)
    result = Parallel(n_jobs=args.jobs)(delayed(fit_selected_target)(j, global_idx, int(target_idx[j]), specs, Z, Y)
                                       for j in range(Y.shape[1]))
    global_estimators, target_estimators = [None]*Y.shape[1], [None]*Y.shape[1]
    for j, g, t in result:
        global_estimators[j], target_estimators[j] = g, t
    bundle = dict(features=features, targets=targets,
                  global_model=GBRTargetModel(imp, global_estimators, [specs[global_idx]]*Y.shape[1]),
                  targetwise_model=GBRTargetModel(imp, target_estimators, [specs[i] for i in target_idx]),
                  reference_y=Y, seed=SEED,
                  selection_protocol="Full-data three-fold CV, seed 20261023+99; train-fold-only median imputation.")
    joblib.dump(bundle, out / "final_gbr_models.joblib", compress=3)
    jj = np.arange(Y.shape[1])
    np.savez_compressed(out / "full_data_selected.npz", gbr_global=pp[global_idx],
                        gbr_targetwise=pp[target_idx, :, jj].T, Y=Y)
    selection = dict(global_index=global_idx, global_spec=specs[global_idx],
                     targetwise_indices=target_idx.tolist(),
                     targetwise_specs={target: specs[i] for target, i in zip(targets, target_idx)},
                     candidate_normalized_mse=nmse.tolist(), elapsed_seconds=time.perf_counter()-t0,
                     selected_single_output_model_count=len(targets)*2-int(np.sum(target_idx==global_idx)))
    (out / "full_data_selection.json").write_text(json.dumps(selection, indent=2))
    # A missing-value prediction check exercises the saved interface, not an
    # estimate of prediction quality; the latter remains in the OOF reports.
    loaded = joblib.load(out / "final_gbr_models.joblib")
    test_X = X[:3].copy(); test_X[:, ::2] = np.nan
    checks = {}
    for key in ["global_model", "targetwise_model"]:
        p = loaded[key].predict(test_X)
        assert p.shape == (3, 50) and np.isfinite(p).all()
        checks[key] = {"shape": list(p.shape), "all_finite": True}
    (out / "final_model_interface_check.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps({"global_spec": specs[global_idx], "elapsed_seconds": selection["elapsed_seconds"],
                      "model_bytes": (out / "final_gbr_models.joblib").stat().st_size}, indent=2), flush=True)


if __name__ == "__main__":
    main()
