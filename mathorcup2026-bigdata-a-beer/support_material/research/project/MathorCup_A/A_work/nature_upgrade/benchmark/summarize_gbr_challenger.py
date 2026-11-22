# Team review and independent work must be documented truthfully.
"""Compare completed classical GBR OOF predictions with the locked old OOF.

Bootstrap intervals resample fixed paired beer-level predictions; they do not
include model refitting, model search, or post-development selection uncertainty.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def main():
    out = Path(__file__).resolve().parent
    old_path = out.parents[1] / "q1" / "oof_arrays.npz"
    old = np.load(old_path)
    new = np.load(out / "gbr_oof_predictions.npz")
    assert np.allclose(old["Y"], new["Y"], rtol=0, atol=0)
    Y = old["Y"]
    design = json.loads((out / "design.json").read_text())
    targets = design["targets"]
    predictions = {"previous_nested_ensemble": old["nested_ensemble"],
                   "previous_extra_trees": old["extra_trees"],
                   **{k: new[k] for k in new.files if k != "Y"}}
    per_target = []
    summary = []
    for key, P in predictions.items():
        r = r2_score(Y, P, multioutput="raw_values")
        rmse = np.sqrt(np.mean((Y-P)**2, axis=0))
        summary.append(dict(model=key, macro_r2=float(r.mean()),
                            core30_r2=float(r[:30].mean()), aroma15_r2=float(r[:15].mean()),
                            flavour15_r2=float(r[15:30].mean()), other20_r2=float(r[30:].mean()),
                            macro_rmse=float(rmse.mean()), positive_r2=int((r > 0).sum())))
        per_target += [dict(model=key, target=t, r2=float(a), rmse=float(b))
                       for t, a, b in zip(targets, r, rmse)]
    pd.DataFrame(summary).to_csv(out / "model_comparison.csv", index=False)
    pd.DataFrame(per_target).to_csv(out / "dimension_metrics.csv", index=False)
    rng = np.random.default_rng(20261023)
    stats = {k: [] for k in predictions}
    for _ in range(2000):
        ii = rng.integers(len(Y), size=len(Y))
        for key, P in predictions.items():
            stats[key].append(float(r2_score(Y[ii], P[ii], multioutput="uniform_average")))
    base = np.array(stats["previous_nested_ensemble"])
    paired = {k: dict(delta=float(np.mean(r2_score(Y, P, multioutput="raw_values")-
                                          r2_score(Y, predictions["previous_nested_ensemble"], multioutput="raw_values"))),
                      lower=float(np.quantile(np.array(stats[k])-base, .025)),
                      upper=float(np.quantile(np.array(stats[k])-base, .975)))
              for k, P in predictions.items() if k != "previous_nested_ensemble"}
    (out / "paired_bootstrap.json").write_text(json.dumps(dict(
        repetitions=2000,
        interpretation="Descriptive paired bootstrap of fixed OOF predictions; no model refitting or correction for development selection.",
        comparisons_to_previous_ensemble=paired), indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
