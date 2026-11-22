# Team review and independent work must be documented truthfully.
"""Verify the saved GBR bundle in an independent temporary working directory."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import resolve_data_dir

import joblib
import numpy as np
import pandas as pd


def main():
    here = Path(__file__).resolve().parent
    data = resolve_data_dir() / "1.csv"
    x = pd.read_csv(data).sort_values("beer").reset_index(drop=True)
    b = joblib.load(here / "final_gbr_models.joblib")
    X = x[b["features"]].to_numpy(float)[:5].copy()
    rng = np.random.default_rng(20261023)
    X[rng.random(X.shape) < .5] = np.nan
    expected = {key: b[key].predict(X) for key in ["global_model", "targetwise_model"]}
    with tempfile.TemporaryDirectory(prefix="mathorcup_gbr_portability_") as td:
        folder = Path(td)
        shutil.copyfile(here / "gbr_model.py", folder / "gbr_model.py")
        shutil.copyfile(here / "final_gbr_models.joblib", folder / "model.joblib")
        np.save(folder / "input.npy", X)
        child = """import joblib, numpy as np
b = joblib.load('model.joblib')
X = np.load('input.npy')
for key in ['global_model', 'targetwise_model']:
    np.save(key+'.npy', b[key].predict(X))
"""
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        subprocess.run([sys.executable, "-c", child], cwd=folder, env=env, check=True)
        checks = {}
        for key, p in expected.items():
            actual = np.load(folder / f"{key}.npy")
            np.testing.assert_array_equal(actual, p)
            assert actual.shape == (5, 50) and np.isfinite(actual).all()
            checks[key] = dict(shape=list(actual.shape), all_finite=True,
                               maximum_roundtrip_difference=float(np.max(np.abs(actual-p))))
    report = dict(status="passed", rows=5, input_features=231,
                  missing_input_fraction=float(np.isnan(X).mean()),
                  copied_files=["gbr_model.py", "final_gbr_models.joblib"],
                  independent_working_directory=True, pythonpath_removed=True, results=checks,
                  scope="Interface and serialization verification only; not a prediction-accuracy estimate.")
    (here / "roundtrip_portability_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
