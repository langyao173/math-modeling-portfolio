# Team review and independent work must be documented truthfully.
"""Read-only checks of splits, selected predictions, and native deployment files."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import hashlib
import json
import platform
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import resolve_data_dir

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import KFold, train_test_split
from catboost_model import CatboostMultiTarget, MODEL_VERSION, SEED
import catboost


def main():
    out = Path(__file__).resolve().parent
    root = out.parents[1]
    data = resolve_data_dir()
    spec = json.loads((out / "experiment_spec.json").read_text())
    assert spec["algorithm_version"] == MODEL_VERSION
    assert spec["package_version"] == catboost.__version__
    for name, expected in spec["input_sha256"].items():
        assert hashlib.sha256((data/name).read_bytes()).hexdigest() == expected
    original = pd.read_csv(data / "1.csv")
    x = original.sort_values("beer").reset_index(drop=True)
    y = pd.read_csv(data / "4.csv").set_index("beer").loc[x.beer]
    features, targets = x.columns[3:].tolist(), y.columns[2:].tolist()
    X, Y = x[features].to_numpy(float), y[targets].to_numpy(float)
    assert spec["features"] == features and spec["targets"] == targets
    partitions = []
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(X)):
        cv = list(KFold(3, shuffle=True, random_state=SEED+k).split(tr))
        partitions.append((f"outer_{k}", tr, te, cv))
    positions = pd.Series(np.arange(len(x)), index=x.beer)
    a, b = train_test_split(np.arange(len(original)), test_size=.3, shuffle=True,
                           random_state=0, stratify=original.tasting_category_fine)
    tr = positions.loc[original.iloc[a].beer].to_numpy()
    te = positions.loc[original.iloc[b].beer].to_numpy()
    partitions.append(("author_holdout", tr, te, list(KFold(5, shuffle=False).split(tr))))
    rows = np.arange(len(X))
    partitions.append(("full_data", rows, rows,
                       list(KFold(3, shuffle=True, random_state=SEED+99).split(rows))))
    report = {"checks": [], "native_max_abs_errors": {},
              "legacy_full_data_metadata_limits": [], "all_passed": False}
    coverage = np.zeros(len(X), dtype=int)
    for tag, tr, te, cv in partitions:
        a = np.load(out / f"{tag}_candidates.npz")
        s = np.load(out / f"{tag}_selected.npz")
        choice = json.loads((out / f"{tag}_selection.json").read_text())
        fold_id = np.empty(len(tr), dtype=int)
        for fi, (ct, cvv) in enumerate(cv):
            assert not np.intersect1d(ct, cvv).size
            assert len(ct)+len(cvv) == len(tr)
            fold_id[cvv] = fi
        for cache in [a, s]:
            assert str(cache["algorithm_version"]) == MODEL_VERSION
            assert str(cache["fingerprint"]) == spec["fingerprint"]
            np.testing.assert_array_equal(cache["train_indices"], tr)
            np.testing.assert_array_equal(cache["test_indices"], te)
            np.testing.assert_array_equal(cache["inner_fold"], fold_id)
            np.testing.assert_array_equal(cache["train_beer"], x.iloc[tr].beer)
            np.testing.assert_array_equal(cache["test_beer"], x.iloc[te].beer)
            assert cache["features"].tolist() == features
            assert cache["targets"].tolist() == targets
        if tag != "full_data":
            assert not np.intersect1d(tr, te).size
        if tag.startswith("outer_"):
            coverage[te] += 1
        assert a["inner_predictions"].shape == (4, len(tr), 50)
        assert a["test_predictions"].shape == (4, len(te), 50)
        assert np.isfinite(a["inner_predictions"]).all()
        assert np.isfinite(a["test_predictions"]).all()
        np.testing.assert_array_equal(a["y_train"], Y[tr])
        np.testing.assert_array_equal(a["y_test"], Y[te])
        scale = np.maximum(Y[tr].std(axis=0, ddof=1), 1e-8)
        loss = np.mean(((a["inner_predictions"]-Y[tr][None, :, :]) /
                        scale[None, None, :])**2, axis=1)
        idx = int(loss.mean(axis=1).argmin())
        assert idx == choice["global_index"]
        np.testing.assert_array_equal(s["catboost_global"], a["test_predictions"][idx])
        np.testing.assert_array_equal(s["inner_catboost_global"], a["inner_predictions"][idx])
        for family_dir in ["adaptive", "benchmark"]:
            other = out.parent / family_dir / f"{tag}_candidates.npz"
            if other.exists():
                old = np.load(other)
                if "train_indices" in old.files and "test_indices" in old.files:
                    np.testing.assert_array_equal(old["train_indices"], tr)
                    np.testing.assert_array_equal(old["test_indices"], te)
                else:
                    assert tag == "full_data", "Held-out comparison must carry explicit indices"
                    assert old["inner_predictions"].shape[1:] == (250, 50)
                    report["legacy_full_data_metadata_limits"].append(
                        str(other.relative_to(out.parent)))
                if "inner_fold" in old.files:
                    np.testing.assert_array_equal(old["inner_fold"], fold_id)
        report["checks"].append(dict(partition=tag, train_n=len(tr), test_n=len(te),
                                     selected_index=idx, selected_spec=choice["selected_spec"]))
    np.testing.assert_array_equal(coverage, np.ones(len(X), dtype=int))
    old = np.load(out.parent / "author_protocol_baseline/holdout_arrays.npz")
    author = np.load(out / "author_holdout_candidates.npz")
    np.testing.assert_array_equal(author["test_beer"], old["beer"])
    np.testing.assert_array_equal(author["y_test"], old["Y"])
    full = np.load(out / "full_data_candidates.npz")
    for i, candidate in enumerate(spec["specs"]):
        model = CatboostMultiTarget.load_native(out, f"full_depth{candidate['depth']}")
        predicted = model.predict(X, candidate["n_estimators"])
        expected = full["test_predictions"][i]
        error = float(np.max(np.abs(predicted-expected)))
        assert error <= 1e-12
        report["native_max_abs_errors"][str(i)] = error
    final_choice = json.loads((out / "full_data_selection.json").read_text())["global_index"]
    final_model = CatboostMultiTarget.load_native(out, "final_global_model")
    final_error = float(np.max(np.abs(final_model.predict(X)-full["test_predictions"][final_choice])))
    assert final_error <= 1e-12
    report["native_max_abs_errors"]["final_global"] = final_error
    report["all_passed"] = True
    report["fingerprint"] = spec["fingerprint"]
    report["environment"] = dict(python=platform.python_version(), numpy=np.__version__,
                                  pandas=pd.__version__, scipy=scipy.__version__,
                                  sklearn=sklearn.__version__, catboost=catboost.__version__)
    (out / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
