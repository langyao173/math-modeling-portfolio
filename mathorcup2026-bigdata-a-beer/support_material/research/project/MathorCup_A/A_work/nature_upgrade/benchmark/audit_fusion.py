# Team review and independent work must be documented truthfully.
"""Independent row/selection/weight audit of the research fusion artifacts.

This checks data alignment, saved predictions and constrained-weight arithmetic.
Source inspection is additionally needed for the training-only preprocessing
statement; binary prediction arrays alone cannot prove how a model was fitted.
"""
import os
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

HERE = Path(__file__).resolve().parent
UP = HERE.parent
WORK = UP.parent
sys.path.insert(0, str(UP))
sys.path.insert(0, str(UP / "adaptive"))
from project_paths import resolve_data_dir
from adaptive_models import MODEL_VERSION, specifications


def close(a, b, tol=1e-12):
    np.testing.assert_allclose(a, b, rtol=0, atol=tol)


def independent_select(inner, test, y, groups=None):
    loss = np.mean((inner-y[None])**2, axis=1) / np.std(y, axis=0, ddof=1)[None]**2
    if groups is None:
        gi = int(np.argmin(loss.mean(axis=1)))
        ti = np.argmin(loss, axis=0)
        return [inner[gi], np.column_stack([inner[c, :, j] for j, c in enumerate(ti)])], [test[gi], np.column_stack([test[c, :, j] for j, c in enumerate(ti)])], dict(global_index=gi, target_indices=ti.tolist())
    pi, pt, selected = [], [], {}
    for name, ix in groups.items():
        ti = np.asarray(ix)[np.argmin(loss[ix], axis=0)]
        selected[name] = ti.tolist()
        pi.append(np.column_stack([inner[c, :, j] for j, c in enumerate(ti)]))
        pt.append(np.column_stack([test[c, :, j] for j, c in enumerate(ti)]))
    return pi, pt, selected


def check_weights(saved, Y):
    P, T = saved["inner_predictions"], saved["test_predictions"]
    gw, tw = saved["global_weights"], saved["target_weights"]
    close(gw.sum(), 1, 1e-9)
    close(tw.sum(axis=0), 1, 1e-9)
    assert gw.min() >= -1e-9 and tw.min() >= -1e-9
    close(saved["global_prediction"], np.sum(T*gw[:, None, None], axis=0), 1e-12)
    close(saved["target_prediction"], np.sum(T*tw[:, None, :], axis=0), 1e-12)
    # Check the simplex KKT conditions of the stated per-target convex quadratic.
    errors = []
    scale = Y.std(axis=0, ddof=1)
    for j in range(Y.shape[1]):
        Z = P[:, :, j].T / scale[j]
        yy = Y[:, j] / scale[j]
        w = tw[:, j]
        grad = 2*Z.T@(Z@w-yy)/len(Y) + .4*(w-gw)
        active = w > 1e-6
        level = np.mean(grad[active])
        err = np.max(np.abs(grad[active]-level))
        if np.any(~active):
            err = max(err, max(0., float(level-np.min(grad[~active]))))
        errors.append(float(err))
    assert max(errors) < 2e-4
    return dict(max_target_weight_kkt_residual=max(errors),
                min_weight=float(min(gw.min(), tw.min())),
                max_simplex_sum_error=float(np.max(np.abs(tw.sum(axis=0)-1))))


def main():
    data = resolve_data_dir()
    original = pd.read_csv(data / "1.csv")
    x = original.sort_values("beer").reset_index(drop=True)
    y = pd.read_csv(data / "4.csv").set_index("beer").loc[x.beer]
    Y = y.iloc[:, 2:].to_numpy(float)
    old = np.load(WORK / "q1/oof_arrays.npz")
    close(old["Y"], Y)
    baseline_manifest = json.loads((WORK / "q1/nested_selection.json").read_text())
    specs = specifications()
    families = ["sparse_ridge", "sparse_rbf", "sparse_svr", "auxiliary_rrr"]
    groups = {f: [i for i, s in enumerate(specs) if s["family"] == f] for f in families}
    splits = [(f"outer_{i}", tr, te) for i, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=20261023).split(Y))]
    tr0, te0 = train_test_split(np.arange(len(original)), test_size=.3, random_state=0,
                              shuffle=True, stratify=original.tasting_category_fine)
    lookup = pd.Series(np.arange(len(x)), index=x.beer)
    splits += [("author_holdout", lookup.loc[original.iloc[tr0].beer].to_numpy(), lookup.loc[original.iloc[te0].beer].to_numpy())]
    records = []
    for tag, tr, te in splits:
        assert len(set(tr) & set(te)) == 0
        assert sorted(np.r_[tr, te].tolist()) == list(range(250))
        aa = np.load(UP / "adaptive" / f"{tag}_candidates.npz")
        gb = np.load(HERE / f"{tag}_candidates.npz")
        assert str(aa["algorithm_version"]) == MODEL_VERSION
        for arr in [aa, gb]:
            assert np.array_equal(arr["train_indices"], tr)
            assert np.array_equal(arr["test_indices"], te)
            assert np.isfinite(arr["inner_predictions"]).all()
            assert np.isfinite(arr["test_predictions"]).all()
        close(gb["y_train"], Y[tr]); close(gb["y_test"], Y[te])
        if tag.startswith("outer_"):
            k = int(tag.split("_")[1])
            old_cache = np.load(UP / "adaptive" / f"{tag}_old_cache.npz")
            assert baseline_manifest[k]["train_beer"] == x.iloc[tr].beer.tolist()
            assert baseline_manifest[k]["test_beer"] == x.iloc[te].beer.tolist()
            assert np.array_equal(old_cache["train_indices"], tr)
            assert np.array_equal(old_cache["test_indices"], te)
            old_inner, old_test = old_cache["inner_predictions"], old_cache["test_predictions"]
            close(old_test, old["nested_ensemble"][te])
            expected_inner_fold = np.empty(len(tr), int)
            for f, (_, v) in enumerate(KFold(3, shuffle=True, random_state=20261023+k).split(tr)):
                expected_inner_fold[v] = f
            assert np.array_equal(expected_inner_fold, gb["inner_fold"])
        else:
            oi = np.load(UP / "author_protocol_baseline/inner_selection_arrays.npz")
            ot = np.load(UP / "author_protocol_baseline/holdout_arrays.npz")
            ii = pd.Series(np.arange(len(oi["beer"])), index=oi["beer"]).loc[x.iloc[tr].beer].to_numpy()
            ti = pd.Series(np.arange(len(ot["beer"])), index=ot["beer"]).loc[x.iloc[te].beer].to_numpy()
            close(oi["Y"][ii], Y[tr]); close(ot["Y"][ti], Y[te])
            old_inner, old_test = oi["ensemble"][ii], ot["ensemble"][ti]
        api, apt, _ = independent_select(aa["inner_predictions"], aa["test_predictions"], Y[tr], groups)
        gpi, gpt, gs = independent_select(gb["inner_predictions"], gb["test_predictions"], Y[tr])
        saved_gs = json.loads((HERE / f"{tag}_selection.json").read_text())
        assert gs["global_index"] == saved_gs["global_index"]
        assert gs["target_indices"] == saved_gs["targetwise_indices"]
        record = dict(partition=tag, training_rows=len(tr), test_rows=len(te),
                      row_mapping_correct=True, preprocessing_version=MODEL_VERSION,
                      gbr_selection_matches=True, fusion_variants_checked={})
        for nbase in [5, 7, 8]:
            p = UP / "fusion" / f"{tag}_{nbase}base.npz"
            if not p.exists():
                continue
            fi = np.load(p)
            expected_i = [old_inner] + api
            expected_t = [old_test] + apt
            if nbase >= 7:
                expected_i += gpi
                expected_t += gpt
            if nbase == 8:
                cb = np.load(UP / "catboost" / f"{tag}_candidates.npz")
                assert np.array_equal(cb["train_indices"], tr)
                assert np.array_equal(cb["test_indices"], te)
                ci, ct, _ = independent_select(cb["inner_predictions"], cb["test_predictions"], Y[tr])
                expected_i += [ci[0]]
                expected_t += [ct[0]]
            close(fi["inner_predictions"], np.stack(expected_i), 1e-12)
            close(fi["test_predictions"], np.stack(expected_t), 1e-12)
            record["fusion_variants_checked"][str(nbase)] = check_weights(fi, Y[tr])
        records.append(record)
    production_selection = json.loads((HERE / "full_data_selection.json").read_text())
    full = np.load(HERE / "full_data_candidates.npz")
    close(full["Y"], Y)
    gi, _, gs = independent_select(full["inner_predictions"], full["inner_predictions"], Y)
    assert production_selection["global_index"] == gs["global_index"]
    assert production_selection["targetwise_indices"] == gs["target_indices"]
    source_paths = [UP / "fusion/run_fusion.py", UP / "fusion/fusion_model.py",
                    UP / "adaptive/prepare_fusion.py", UP / "adaptive/run_adaptive.py",
                    UP / "adaptive/adaptive_models.py", UP / "project_paths.py",
                    HERE / "run_gbr_challenger.py"]
    report = dict(status="passed", partitions=records, full_gbr_selection_verified=True,
                  source_sha256={str(p.relative_to(UP)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                  source_review=[
                      "Feature imputation, rank transforms, target scaling, feature selection and auxiliary-output fitting use the enclosing training partition.",
                      "Adaptive and tree hyperparameters and fusion weights use training-partition inner predictions and Y[train]; Y[test] only enters reported metrics.",
                      "The author holdout is realigned by unique beer names, and all 50 target values are checked after alignment.",
                      "Selected inner predictions are reused to fit the meta learner: they are development/training predictions, not an independent performance estimate. The complete enclosing outer test remains untouched.",
                      "The provided TF-IDF table is used only via training rows. Raw-text vocabulary/IDF fitting cannot be reconstructed from the supplied table and is not independently certified by this audit."
                  ])
    (HERE / "fusion_independent_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"status": report["status"], "partitions": records}, indent=2))


if __name__ == "__main__":
    main()
