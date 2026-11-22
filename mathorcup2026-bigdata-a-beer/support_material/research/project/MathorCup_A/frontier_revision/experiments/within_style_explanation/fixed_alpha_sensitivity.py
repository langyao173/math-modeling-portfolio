"""Matched-penalty bootstrap and target-scale sensitivity of style adjustment.

Uses exactly the 400 resampling indices from the main nested experiment. The
four full-data fits all select alpha=3; keeping that penalty fixed separates
style centering from bootstrap hyperparameter selection. This is descriptive
conditional stability, complementary to the retuned-bootstrap experiment.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from within_style_model import prepare, ridge_fit, top_indices

HERE = Path(__file__).resolve().parent
from project_location import PROJECT


def correlation(a, b):
    a, b = a-a.mean(), b-b.mean()
    den = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den > 1e-12 else np.nan


def main():
    t0 = time.perf_counter()
    xframe = pd.read_csv(PROJECT / "data/A/1.csv").sort_values("beer").reset_index(drop=True)
    yframe = pd.read_csv(PROJECT / "data/A/4.csv").set_index("beer").loc[xframe.beer].reset_index()
    features, targets = xframe.columns[3:].to_list(), yframe.columns[3:].to_list()
    x, y = xframe[features].to_numpy(float), yframe[targets].to_numpy(float)
    styles = xframe.tasting_category_fine.to_numpy()
    dictionary = pd.read_csv(PROJECT / "A_work/q2/chemical_dictionary.csv").set_index("field").loc[features]
    eligible = np.flatnonzero(dictionary.single_compound_candidate.to_numpy(bool))
    primary = np.load(HERE / "bootstrap_coefficients.npz", allow_pickle=False)
    ixall = primary["sample_indices"]
    reps, p, m = len(ixall), len(features), len(targets)
    parameters = pd.read_csv(HERE / "nested_cv_parameters.csv")
    chosen = parameters[parameters.fold.astype(str).eq("full")].set_index("model").alpha
    assert set(chosen.to_numpy()) == {3.}, "Inspect and explicitly revise matched penalty if the full fit changes"
    alpha = 3.
    models = ["total_rank_fixed", "within_rank_fixed", "total_linear_fixed", "within_linear_fixed"]
    coefficients = np.empty((4, reps, p, m), dtype=np.float32)
    tops = np.empty((4, reps, 3, m), dtype=np.int32)
    full = {mode:prepare(x, y, styles, mode) for mode in ["rank", "linear"]}
    full_coef = np.stack([ridge_fit(full[mode], within, alpha)
        for mode in ["rank", "linear"] for within in [False, True]])
    for b, ix in enumerate(ixall):
        for mode_index, mode in enumerate(["rank", "linear"]):
            prep = prepare(x[ix], y[ix], styles[ix], mode)
            for wi, within in enumerate([False, True]):
                mi = mode_index*2+wi
                coef = ridge_fit(prep, within, alpha)
                coefficients[mi, b] = coef
                tops[mi, b] = top_indices(coef, eligible)
        if (b+1) % 100 == 0:
            print(json.dumps(dict(stage="matched_penalty_bootstrap", completed=b+1, seconds=time.perf_counter()-t0)), flush=True)
    np.savez_compressed(HERE / "bootstrap_fixed_alpha_coefficients.npz", coefficient=coefficients,
        top3=tops, models=np.array(models), features=np.array(features), targets=np.array(targets), alpha=alpha)
    evidence = pd.DataFrame(dict(target=np.repeat(targets, p), field=np.tile(features, m)))
    for model, label in enumerate(models):
        coefficient = coefficients[model].astype(float)
        freq = np.zeros((p, m))
        for b in range(reps):
            np.add.at(freq, (tops[model, b].ravel(), np.tile(np.arange(m), 3)), 1.)
        freq /= reps
        lo, hi = np.quantile(coefficient, [.025, .975], axis=0)
        median_abs = np.median(np.abs(coefficient), axis=0)
        stable_rank = np.full((p, m), np.nan)
        for j in range(m):
            order = np.lexsort((-median_abs[eligible, j], -freq[eligible, j]))
            stable_rank[eligible[order], j] = np.arange(1, len(eligible)+1)
        metrics = dict(coefficient=full_coef[model], top3_frequency=freq,
            sign_agreement=np.mean(np.sign(coefficient) == np.sign(full_coef[model])[None], axis=0),
            coefficient_025=lo, coefficient_975=hi, stability_rank=stable_rank)
        for name, values in metrics.items():
            evidence[label+"_"+name] = values.ravel(order="F")
    linear = full["linear"]
    den = np.sqrt(np.mean(linear["ur"]**2, axis=0)[:, None]*np.mean(linear["vr"]**2, axis=0)[None, :])
    evidence["correlation_total_linearY"] = (linear["u"].T@linear["v"]/len(x)).ravel(order="F")
    evidence["correlation_within_linearY"] = np.divide(linear["ur"].T@linear["vr"]/len(x), den,
        out=np.zeros((p,m)), where=den > 1e-12).ravel(order="F")
    evidence.to_csv(HERE / "fixed_alpha_pair_evidence.csv", index=False)
    old = pd.read_csv(HERE / "old_top3_alignment.csv")
    old = old.merge(evidence, on=["target", "field"], validate="one_to_one")
    old.to_csv(HERE / "old_top3_fixed_alpha_alignment.csv", index=False)

    # Influence of single styles on the fitted common within-style association.
    keys = set(zip(old.target, old.field))
    anchors = {"lactic":"lactic_acid.mg.L.", "acidity":"lactic_acid.mg.L.",
        "acetic":"acetic_acid.g.L.", "alcohol":"ethanol..v.v.", "X4vg":"X4VG",
        "diacetyl":"X2.3.butanedione", "dms":"DMS", "t2n":"X.E..2.nonenal"}
    keys |= set(anchors.items())
    keys = sorted(keys)
    fj, tj = np.array([features.index(f) for _,f in keys]), np.array([targets.index(t) for t,_ in keys])
    leave_rows = []
    for style in np.unique(styles):
        mask = styles != style
        for mode in ["rank", "linear"]:
            prep = prepare(x[mask], y[mask], styles[mask], mode)
            coef = ridge_fit(prep, True, alpha)
            for (target, field), v in zip(keys, coef[fj,tj]):
                leave_rows.append(dict(excluded_style=style, model="within_"+mode,
                    target=target, field=field, coefficient=float(v), training_n=int(mask.sum()), alpha=alpha))
    leave = pd.DataFrame(leave_rows)
    leave.to_csv(HERE / "leave_one_style_influence.csv", index=False)
    influence = leave.groupby(["model", "target", "field"]).coefficient.agg(["min", "max", "median"]).reset_index()
    sign = leave.assign(positive=leave.coefficient > 0).groupby(["model", "target", "field"]).positive.mean().reset_index()
    influence.merge(sign, on=["model", "target", "field"]).to_csv(HERE / "leave_one_style_summary.csv", index=False)
    style_rows = []
    for style in np.unique(styles):
        mask = styles == style
        for mode in ["rank", "linear"]:
            prep = full[mode]
            for target, field in anchors.items():
                j, k = features.index(field), targets.index(target)
                style_rows.append(dict(style=style, n=int(mask.sum()), target_mode=mode,
                    target=target, field=field, correlation=correlation(prep["u"][mask,j], prep["v"][mask,k]),
                    within_covariance_contribution=float(prep["ur"][mask,j]@prep["vr"][mask,k]/len(x))))
    pd.DataFrame(style_rows).to_csv(HERE / "stylewise_anchor_associations.csv", index=False)
    summary = dict(alpha=alpha, replicates=reps,
        source_bootstrap="Exactly the same 400 style-stratified indices as the retuned experiment",
        reason="Separate style adjustment from bootstrap hyperparameter selection and from Y rank transformation",
        conditional_inference="Fixed-alpha descriptive stability; percentile bands are not unbiased causal-effect confidence intervals",
        leave_one_style="Influence diagnostics only, full refit of maps/means at alpha3; not held-out prediction of unseen styles",
        old_top3_retained={label:int((old[label+"_stability_rank"] <= 3).sum()) for label in models},
        old_top3_frequency_ge_half={label:int((old[label+"_top3_frequency"] >= .5).sum()) for label in models},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), wall_seconds=time.perf_counter()-t0)
    (HERE / "fixed_alpha_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
